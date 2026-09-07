import hashlib
import os
import time

from corpus_metadata import (
    chroma_metadata_fields,
    load_json_file,
    metadata_sha256,
    normalize_faculty,
    write_json_file,
)
from project_config import (
    build_index_config_hash,
    faculty_index_paths,
    get_chunking_profile,
    index_config_payload,
    load_project_config,
    resolve_project_path,
    validate_index_config_hash,
)
from pdf_extract import EXTRACTION_VERSION


class VectorIndexer:
    def __init__(self, config_path):
        self.config = load_project_config(config_path)
        self.model_config = self.config["model"]
        self.indexing_config = self.config["indexing"]
        self.chunk_config = get_chunking_profile(config=self.config)
        configured_extraction_version = str(
            self.indexing_config.get("extraction_version")
            or EXTRACTION_VERSION
        )
        if configured_extraction_version != EXTRACTION_VERSION:
            raise RuntimeError(
                "Configured extraction_version does not match pdf_extract: "
                f"{configured_extraction_version} != {EXTRACTION_VERSION}"
            )
        self.index_config_hash = build_index_config_hash(
            self.config,
            extraction_version=EXTRACTION_VERSION,
        )
        self.index_config_payload = index_config_payload(
            self.config,
            extraction_version=EXTRACTION_VERSION,
        )
        self.model_cache = resolve_project_path(
            self.model_config["cache_dir"])
        os.makedirs(self.model_cache, exist_ok=True)
        os.environ["HF_HOME"] = self.model_cache
        os.environ["HF_HUB_CACHE"] = os.path.join(
            self.model_cache, "hub")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.model_cache
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        import chromadb
        import torch
        from chromadb.config import Settings
        from chromadb.errors import NotFoundError
        from FlagEmbedding import BGEM3FlagModel
        from chunking_config import (
            chunk_text,
            chunk_text_with_native_spans,
            create_chunker_tools,
        )

        self.chromadb = chromadb
        self.torch = torch
        self.Settings = Settings
        self.NotFoundError = NotFoundError
        self.chunk_text = chunk_text
        self.chunk_text_with_native_spans = chunk_text_with_native_spans
        self.chunker, self.refinery = create_chunker_tools(
            self.chunk_config)
        model_options = {
            "cache_dir": self.model_cache,
            "use_fp16": (
                torch.cuda.is_available()
                and bool(self.model_config["use_fp16_on_cuda"])
            ),
        }
        if self.model_config.get("revision"):
            model_options["revision"] = self.model_config["revision"]
        self.model = BGEM3FlagModel(
            self.model_config["name"], **model_options)
        if torch.cuda.is_available():
            self.model.model.to("cuda")
        self.model.model.eval()
        self.collections = {}
        self.paths = {}

    def paths_for(self, collection_folder):
        if collection_folder not in self.paths:
            paths = faculty_index_paths(
                collection_folder,
                self.chunk_config["index_version"],
                self.config,
            )
            os.makedirs(paths["normalized"], exist_ok=True)
            self.paths[collection_folder] = paths
        return self.paths[collection_folder]

    def collection_for(self, collection_folder):
        if collection_folder not in self.collections:
            paths = self.paths_for(collection_folder)
            client = self.chromadb.PersistentClient(
                path=paths["vector_db"],
                settings=self.Settings(anonymized_telemetry=False),
            )
            try:
                collection = client.get_collection(
                    name=self.indexing_config["collection_name"])
            except self.NotFoundError:
                collection = client.create_collection(
                    name=self.indexing_config["collection_name"],
                    metadata=self.collection_metadata(collection_folder),
                )
            else:
                self.validate_collection(
                    collection,
                    paths["marker"],
                    collection_folder,
                )
            self.collections[collection_folder] = collection
        return self.collections[collection_folder]

    def collection_metadata(self, collection_folder):
        return {
            "index_version": self.chunk_config["index_version"],
            "index_config_hash": self.index_config_hash,
            "model": self.model_config["name"],
            "model_revision": str(
                self.model_config.get("revision") or "default"),
            "chunking_strategy": self.chunk_config["strategy"],
            "chunk_size": self.chunk_config["chunk_size"],
            "chunk_size_unit": self.chunk_config["chunk_size_unit"],
            "chunk_overlap": self.chunk_config["chunk_overlap"],
            "encode_max_length": int(
                self.model_config["encode_max_length"]),
            "extraction_version": EXTRACTION_VERSION,
            "faculty": collection_folder,
        }

    def validate_collection(
        self, collection, marker_path, expected_collection_folder
    ):
        metadata = collection.metadata or {}
        stored_version = metadata.get("index_version")
        if (
            stored_version
            and stored_version != self.chunk_config["index_version"]
        ):
            raise RuntimeError(
                "Index version mismatch: expected "
                f"{self.chunk_config['index_version']}, "
                f"found {stored_version}. Existing index data was not modified."
            )
        validate_index_config_hash(
            metadata.get("index_config_hash"),
            self.index_config_hash,
            location="Chroma collection",
        )
        if marker_path and (
            os.path.isfile(marker_path) or collection.count() > 0
        ):
            marker = load_json_file(marker_path, default={})
            validate_index_config_hash(
                marker.get("index_config_hash"),
                self.index_config_hash,
                location=f"Index marker {marker_path}",
            )
        stored_faculty = metadata.get("faculty")
        if (
            stored_faculty
            and normalize_faculty(stored_faculty)
            != normalize_faculty(expected_collection_folder)
        ):
            raise RuntimeError(
                "Index faculty mismatch: expected "
                f"{expected_collection_folder}, found {stored_faculty}. "
                "Existing index data was not modified."
            )

    def remove_stale_sources(self, collection_folder, current_sources):
        paths = self.paths_for(collection_folder)
        marker = load_json_file(paths["marker"], default={})
        old_sources = set(marker.get("source_keys") or ())
        collection = self.collection_for(collection_folder)
        for source in old_sources - set(current_sources):
            collection.delete(where={"source": source})

    def is_current(self, document):
        collection = self.collection_for(document.collection_folder)
        result = collection.get(
            where={"source": document.source_key},
            limit=1,
            include=["metadatas"],
        )
        if not result.get("ids"):
            return False
        metadata = (result.get("metadatas") or [{}])[0] or {}
        if metadata.get("file_sha256") != document.file_sha256:
            return False
        if metadata.get("source_metadata_sha256") != metadata_sha256(
                document.metadata):
            return False
        expected_count = metadata.get("chunk_count")
        if expected_count is None:
            return False
        ids = collection.get(
            where={"source": document.source_key}, include=[]).get("ids", ())
        return len(ids) == int(expected_count)

    def find_spans(self, text, chunks):
        spans = []
        cursor = 0
        search_back = max(int(self.chunk_config.get(
            "chunk_overlap_search_characters", 1)), 1)
        for chunk in chunks:
            start = text.find(chunk, max(0, cursor - search_back))
            if start < 0:
                start = text.find(chunk)
            end = start + len(chunk) if start >= 0 else -1
            spans.append((start, end))
            if end >= 0:
                cursor = max(cursor, end)
        return spans

    def make_chunks(self, text):
        if self.chunk_config["strategy"] == "token":
            return self.chunk_text_with_native_spans(
                text, self.chunker, self.refinery)
        chunks = self.chunk_text(text, self.chunker, self.refinery)
        return chunks, self.find_spans(text, chunks)

    def encode(self, chunks):
        with self.torch.inference_mode():
            result = self.model.encode(
                chunks,
                batch_size=int(self.indexing_config["encode_batch_size"]),
                max_length=int(self.model_config["encode_max_length"]),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        return result["dense_vecs"].tolist()

    def index(self, extracted_document):
        document = extracted_document.source
        chunks, spans = self.make_chunks(extracted_document.text)
        if not chunks:
            raise ValueError("Chunker returned no chunks")
        embeddings = self.encode(chunks)
        metadata_hash = metadata_sha256(document.metadata)
        source_hash = hashlib.sha256(
            document.source_key.encode("utf-8")).hexdigest()[:16]
        ids = [
            f"{source_hash}_{extracted_document.content_sha256[:12]}_{index}"
            for index in range(len(chunks))
        ]
        provenance = chroma_metadata_fields(document.metadata)
        metadatas = []
        for index, span in enumerate(spans):
            metadata = {
                "source": document.source_key,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "start": span[0],
                "end": span[1],
                "content_sha256": extracted_document.content_sha256,
                "file_sha256": document.file_sha256,
                "source_metadata_sha256": metadata_hash,
                "index_version": self.chunk_config["index_version"],
                "index_config_hash": self.index_config_hash,
                "model": self.model_config["name"],
                "chunking_strategy": self.chunk_config["strategy"],
                "chunk_size": self.chunk_config["chunk_size"],
                "chunk_size_unit": self.chunk_config["chunk_size_unit"],
                "chunk_overlap": self.chunk_config["chunk_overlap"],
            }
            metadata.update(provenance)
            metadatas.append(metadata)

        collection = self.collection_for(document.collection_folder)
        collection.delete(where={"source": document.source_key})
        batch_size = int(self.indexing_config["max_chroma_batch"])
        try:
            for start in range(0, len(chunks), batch_size):
                end = start + batch_size
                collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=chunks[start:end],
                    metadatas=metadatas[start:end],
                )
        except Exception:
            collection.delete(where={"source": document.source_key})
            raise

        indexed_ids = collection.get(
            where={"source": document.source_key}, include=[]).get("ids", ())
        if len(indexed_ids) != len(chunks):
            collection.delete(where={"source": document.source_key})
            raise RuntimeError("ChromaDB did not store every chunk")
        return len(chunks)

    def finish_collection(self, collection_folder, source_keys):
        paths = self.paths_for(collection_folder)
        collection = self.collection_for(collection_folder)
        write_json_file(paths["marker"], {
            "index_version": self.chunk_config["index_version"],
            "index_config_hash": self.index_config_hash,
            "index_config": self.index_config_payload,
            "faculty": collection_folder,
            "updated_at_unix": time.time(),
            "source_files": len(source_keys),
            "collection_chunks": collection.count(),
            "source_keys": sorted(source_keys),
        })
