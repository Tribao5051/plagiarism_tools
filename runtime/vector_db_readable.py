import argparse
import hashlib
import os

from project_config import (
    PROJECT_ROOT,
    build_index_config_hash,
    faculty_index_paths,
    get_chunking_profile,
    index_config_payload,
    load_project_config,
    resolve_project_path,
    validate_index_config_hash,
)


CONFIG = load_project_config()
MODEL_CONFIG = CONFIG["model"]
INDEXING_CONFIG = CONFIG["indexing"]
CHUNK_CONFIG = get_chunking_profile(config=CONFIG)
MODEL_CACHE = resolve_project_path(MODEL_CONFIG["cache_dir"])

os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(MODEL_CACHE, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "mpl_temp")
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
from corpus_metadata import (
    chroma_metadata_fields,
    discover_corpus_files,
    file_sha256,
    load_json_file,
    load_source_metadata,
    metadata_sha256,
    normalize_faculty,
    normalize_source_key,
    write_json_file,
    write_normalized_document,
)
from pdf_extract import EXTRACTION_VERSION, extract_university_content


CONFIG_EXTRACTION_VERSION = str(
    INDEXING_CONFIG.get("extraction_version") or EXTRACTION_VERSION)
if CONFIG_EXTRACTION_VERSION != EXTRACTION_VERSION:
    raise RuntimeError(
        "Configured extraction_version does not match pdf_extract: "
        f"{CONFIG_EXTRACTION_VERSION} != {EXTRACTION_VERSION}"
    )
INDEX_CONFIG_HASH = build_index_config_hash(
    CONFIG,
    extraction_version=EXTRACTION_VERSION,
)
INDEX_CONFIG_PAYLOAD = index_config_payload(
    CONFIG,
    extraction_version=EXTRACTION_VERSION,
)


def find_spans(content, chunks):
    spans = []
    cursor = 0
    search_back = max(
        CHUNK_CONFIG.get("chunk_overlap_search_characters", 1), 1)
    for chunk in chunks:
        start = content.find(chunk, max(0, cursor - search_back))
        if start < 0:
            start = content.find(chunk)
        end = start + len(chunk) if start >= 0 else -1
        spans.append((start, end))
        if end >= 0:
            cursor = max(cursor, end)
    return spans


def make_chunks(content, chunker, refinery):
    if CHUNK_CONFIG["strategy"] == "token":
        return chunk_text_with_native_spans(content, chunker, refinery)
    chunks = chunk_text(content, chunker, refinery)
    return chunks, find_spans(content, chunks)


def source_path(corpus_directory, source):
    return os.path.join(corpus_directory, *source.split("/"))


def source_is_current(collection, source, file_hash, source_metadata_hash):
    result = collection.get(
        where={"source": source}, limit=1, include=["metadatas"])
    if not result.get("ids"):
        return False
    metadata = (result.get("metadatas") or [{}])[0] or {}
    if metadata.get("file_sha256") != file_hash:
        return False
    if metadata.get("source_metadata_sha256") != source_metadata_hash:
        return False
    expected = metadata.get("chunk_count")
    if expected is None:
        return False
    indexed = collection.get(
        where={"source": source}, include=[]).get("ids", ())
    return len(indexed) == int(expected)


def remove_stale_sources(collection, files, marker_path):
    marker = load_json_file(marker_path, default={})
    old_sources = set(marker.get("source_keys") or ())
    for source in old_sources - set(files):
        collection.delete(where={"source": source})


def encode_chunks(model, chunks):
    with torch.inference_mode():
        result = model.encode(
            chunks,
            batch_size=int(INDEXING_CONFIG["encode_batch_size"]),
            max_length=int(MODEL_CONFIG["encode_max_length"]),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    return result["dense_vecs"].tolist()


def chunk_ids(source, content_hash, count):
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return [
        f"{source_hash}_{content_hash[:12]}_{index}"
        for index in range(count)
    ]


def chunk_metadata(source, spans, content_hash, file_hash, metadata_hash,
                   source_metadata):
    provenance = chroma_metadata_fields(source_metadata)
    items = []
    for index, span in enumerate(spans):
        item = {
            "source": source,
            "chunk_index": index,
            "chunk_count": len(spans),
            "start": span[0],
            "end": span[1],
            "content_sha256": content_hash,
            "file_sha256": file_hash,
            "source_metadata_sha256": metadata_hash,
            "index_version": CHUNK_CONFIG["index_version"],
            "index_config_hash": INDEX_CONFIG_HASH,
            "model": MODEL_CONFIG["name"],
            "chunking_strategy": CHUNK_CONFIG["strategy"],
            "chunk_size": CHUNK_CONFIG["chunk_size"],
            "chunk_size_unit": CHUNK_CONFIG["chunk_size_unit"],
            "chunk_overlap": CHUNK_CONFIG["chunk_overlap"],
        }
        item.update(provenance)
        items.append(item)
    return items


def add_document(collection, model, corpus_directory, normalized_directory,
                 faculty, source, chunker, refinery):
    source = normalize_source_key(source)
    path = source_path(corpus_directory, source)
    file_hash = file_sha256(path)
    source_metadata = load_source_metadata(
        corpus_directory, source, default_faculty=faculty)
    source_metadata_hash = metadata_sha256(source_metadata)
    if source_is_current(
            collection, source, file_hash, source_metadata_hash):
        return

    collection.delete(where={"source": source})
    content = extract_university_content(path, save_debug=False)
    if not content:
        return
    write_normalized_document(
        normalized_directory, source, content, source_metadata)
    chunks, spans = make_chunks(content, chunker, refinery)
    if not chunks:
        return

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    embeddings = encode_chunks(model, chunks)
    ids = chunk_ids(source, content_hash, len(chunks))
    metadatas = chunk_metadata(
        source,
        spans,
        content_hash,
        file_hash,
        source_metadata_hash,
        source_metadata,
    )

    batch_size = int(INDEXING_CONFIG["max_chroma_batch"])
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
        collection.delete(where={"source": source})
        raise


def create_model():
    model_options = {
        "cache_dir": MODEL_CACHE,
        "use_fp16": (
            torch.cuda.is_available()
            and bool(MODEL_CONFIG["use_fp16_on_cuda"])
        ),
    }
    if MODEL_CONFIG.get("revision"):
        model_options["revision"] = MODEL_CONFIG["revision"]
    model = BGEM3FlagModel(MODEL_CONFIG["name"], **model_options)
    if torch.cuda.is_available():
        model.model.to("cuda")
    model.model.eval()
    return model


def collection_index_metadata(faculty):
    return {
        "index_version": CHUNK_CONFIG["index_version"],
        "index_config_hash": INDEX_CONFIG_HASH,
        "model": MODEL_CONFIG["name"],
        "model_revision": str(MODEL_CONFIG.get("revision") or "default"),
        "chunking_strategy": CHUNK_CONFIG["strategy"],
        "chunk_size": CHUNK_CONFIG["chunk_size"],
        "chunk_size_unit": CHUNK_CONFIG["chunk_size_unit"],
        "chunk_overlap": CHUNK_CONFIG["chunk_overlap"],
        "encode_max_length": int(MODEL_CONFIG["encode_max_length"]),
        "extraction_version": EXTRACTION_VERSION,
        "faculty": faculty,
    }


def validate_reusable_collection(collection, marker_path, expected_faculty):
    metadata = collection.metadata or {}
    stored_version = metadata.get("index_version")
    if stored_version and stored_version != CHUNK_CONFIG["index_version"]:
        raise RuntimeError(
            "Index version mismatch: expected "
            f"{CHUNK_CONFIG['index_version']}, found {stored_version}. "
            "Existing index data was not modified."
        )
    validate_index_config_hash(
        metadata.get("index_config_hash"),
        INDEX_CONFIG_HASH,
        location="Chroma collection",
    )
    if marker_path and (
        os.path.isfile(marker_path) or collection.count() > 0
    ):
        marker = load_json_file(marker_path, default={})
        validate_index_config_hash(
            marker.get("index_config_hash"),
            INDEX_CONFIG_HASH,
            location=f"Index marker {marker_path}",
        )
    stored_faculty = metadata.get("faculty")
    if (
        stored_faculty
        and normalize_faculty(stored_faculty) != normalize_faculty(
            expected_faculty)
    ):
        raise RuntimeError(
            f"Index faculty mismatch: expected {expected_faculty}, "
            f"found {stored_faculty}. Existing index data was not modified."
        )


def create_collection(database_path, faculty, marker_path=None):
    client = chromadb.PersistentClient(
        path=database_path,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(
            name=INDEXING_CONFIG["collection_name"])
    except NotFoundError:
        return client.create_collection(
            name=INDEXING_CONFIG["collection_name"],
            metadata=collection_index_metadata(faculty),
        )
    validate_reusable_collection(collection, marker_path, faculty)
    return collection


def build_vector_database(corpus_directory, faculty):
    faculty = normalize_faculty(faculty)
    paths = faculty_index_paths(
        faculty, CHUNK_CONFIG["index_version"], CONFIG)
    os.makedirs(paths["normalized"], exist_ok=True)
    files = discover_corpus_files(corpus_directory)
    collection = create_collection(
        paths["vector_db"], faculty, marker_path=paths["marker"])
    remove_stale_sources(collection, files, paths["marker"])
    model = create_model()
    chunker, refinery = create_chunker_tools(CHUNK_CONFIG)

    for source in files:
        add_document(
            collection,
            model,
            corpus_directory,
            paths["normalized"],
            faculty,
            source,
            chunker,
            refinery,
        )

    write_json_file(paths["marker"], {
        "index_version": CHUNK_CONFIG["index_version"],
        "index_config_hash": INDEX_CONFIG_HASH,
        "index_config": INDEX_CONFIG_PAYLOAD,
        "faculty": faculty,
        "target_directory": corpus_directory,
        "source_files": len(files),
        "collection_chunks": collection.count(),
        "source_keys": files,
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_directory")
    parser.add_argument("--faculty")
    return parser.parse_args()


def main():
    args = parse_args()
    corpus_directory = os.path.abspath(args.corpus_directory)
    faculty = args.faculty or os.path.basename(
        os.path.normpath(corpus_directory))
    build_vector_database(corpus_directory, faculty)


if __name__ == "__main__":
    main()
