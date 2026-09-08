import argparse
import os
import time
import queue
import threading
import subprocess
import hashlib

from project_config import (
    PROJECT_ROOT,
    build_index_config_hash,
    faculty_index_paths,
    index_config_payload,
    load_project_config,
    resolve_project_path,
    validate_index_config_hash,
)

# Set cache and telemetry environment before importing model/database libraries.
PROJECT_CONFIG = load_project_config()
MODEL_CONFIG = PROJECT_CONFIG["model"]
INDEXING_CONFIG = PROJECT_CONFIG["indexing"]
PROJECT_CACHE = resolve_project_path(MODEL_CONFIG["cache_dir"])
MPL_CACHE = os.path.join(PROJECT_ROOT, "mpl_temp")
os.makedirs(PROJECT_CACHE, exist_ok=True)
os.makedirs(MPL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = PROJECT_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(PROJECT_CACHE, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = PROJECT_CACHE
os.environ["MPLCONFIGDIR"] = MPL_CACHE
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import torch
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from FlagEmbedding import BGEM3FlagModel
from chunking_config import (
    chunk_text,
    chunk_text_with_native_spans,
    create_chunker_tools,
    load_chunking_config,
)
from corpus_metadata import (
    chroma_metadata_fields,
    discover_corpus_files,
    file_sha256,
    load_source_metadata,
    managed_normalized_document_paths,
    metadata_sha256,
    normalize_faculty,
    normalize_source_key,
    load_json_file,
    discard_staged_normalized_document,
    publish_staged_normalized_document,
    remove_normalized_document,
    stage_normalized_document,
    write_json_file,
)
from pdf_extract import EXTRACTION_VERSION, extract_university_content
from chromadb.config import Settings
from chromadb.errors import NotFoundError
# --- CONFIGURATION ---
MODEL_NAME = MODEL_CONFIG["name"]
CHUNK_CONFIG = load_chunking_config()
INDEX_VERSION = CHUNK_CONFIG["index_version"]
if INDEXING_CONFIG.get("extraction_version") != EXTRACTION_VERSION:
    raise RuntimeError(
        "Configured extraction_version does not match pdf_extract.py")
INDEX_CONFIG_HASH = build_index_config_hash(
    PROJECT_CONFIG,
    strategy=CHUNK_CONFIG["strategy"],
    extraction_version=EXTRACTION_VERSION,
)
INDEX_CONFIG_PAYLOAD = index_config_payload(
    PROJECT_CONFIG,
    strategy=CHUNK_CONFIG["strategy"],
    extraction_version=EXTRACTION_VERSION,
)
COLLECTION_NAME = INDEXING_CONFIG["collection_name"]
ACTIVE_FACULTY = "default"
DB_PATH = ""
NORMALIZED_CHECK_DIR = ""
REPORT_PATH = ""
CHANGES_PATH = ""
VERSIONED_REPORT_PATH = ""
VERSIONED_CHANGES_PATH = ""
REFRESH_MARKER_PATH = ""

CHUNKING_STRATEGY = CHUNK_CONFIG["strategy"]
CHUNK_SIZE = CHUNK_CONFIG["chunk_size"]
CHUNK_SIZE_UNIT = CHUNK_CONFIG["chunk_size_unit"]
CHUNK_OVERLAP = CHUNK_CONFIG["chunk_overlap"]
CHUNK_OVERLAP_SEARCH_CHARACTERS = CHUNK_CONFIG.get(
    "chunk_overlap_search_characters", CHUNK_OVERLAP * 2)
MIN_CHUNK_CHARACTERS = CHUNK_CONFIG["min_chunk_characters"]

# CPU-bound extraction/chunking can stay parallel — this is genuinely fast
EXTRACT_WORKERS = int(
    INDEXING_CONFIG["semantic_extract_workers"]
    if CHUNKING_STRATEGY == "semantic"
    else INDEXING_CONFIG["default_extract_workers"]
)

# GPU encoding uses one dedicated consumer. A dedicated 1,024-chunk indexing
# benchmark found batch 32 was 3.77% faster than batch 48 on this 4 GB GPU.
if torch.cuda.is_available():
    GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
else:
    GPU_MEMORY_GB = 0.0

ENCODE_BATCH_SIZE = int(INDEXING_CONFIG["encode_batch_size"])
ENCODE_MAX_LENGTH = int(MODEL_CONFIG["encode_max_length"])

# How many chunks to accumulate in RAM before encoding.
# FlagEmbedding still processes this input internally in batches of 32 on the GPU.
GPU_QUEUE_TARGET_SIZE = int(
    INDEXING_CONFIG["gpu_queue_target_chunks"])
MAX_CHROMA_BATCH = int(INDEXING_CONFIG["max_chroma_batch"])
WORK_QUEUE_MAX_FILES = int(INDEXING_CONFIG["work_queue_max_files"])
WRITE_QUEUE_MAX_BATCHES = int(
    INDEXING_CONFIG["write_queue_max_batches"])
THREAD_LOCAL = threading.local()


def collection_index_metadata(faculty):
    return {
        "index_version": INDEX_VERSION,
        "index_config_hash": INDEX_CONFIG_HASH,
        "model": MODEL_NAME,
        "model_revision": str(MODEL_CONFIG.get("revision") or "default"),
        "chunking_strategy": CHUNKING_STRATEGY,
        "chunk_size": CHUNK_SIZE,
        "chunk_size_unit": CHUNK_SIZE_UNIT,
        "chunk_overlap": CHUNK_OVERLAP,
        "semantic_embedding_model": CHUNK_CONFIG.get(
            "semantic_embedding_model", "none"),
        "semantic_threshold": CHUNK_CONFIG.get("semantic_threshold", 0.0),
        "chunk_tokenizer": CHUNK_CONFIG.get(
            "tokenizer", CHUNK_CONFIG.get(
                "semantic_embedding_model", "character")),
        "encode_max_length": ENCODE_MAX_LENGTH,
        "extraction_version": EXTRACTION_VERSION,
        "faculty": faculty,
    }


def validate_reusable_index(collection, marker_path, expected_faculty=None):
    metadata = collection.metadata or {}
    stored_version = metadata.get("index_version")
    if stored_version and stored_version != INDEX_VERSION:
        raise RuntimeError(
            f"Index version mismatch: expected {INDEX_VERSION}, "
            f"found {stored_version}. Existing index data was not modified."
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
        expected_faculty
        and stored_faculty
        and normalize_faculty(stored_faculty)
        != normalize_faculty(expected_faculty)
    ):
        raise RuntimeError(
            f"Index faculty mismatch: expected {expected_faculty}, "
            f"found {stored_faculty}. Existing index data was not modified."
        )
    return True


def open_compatible_collection(client, faculty):
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata=collection_index_metadata(faculty),
        )

    validate_reusable_index(
        collection,
        REFRESH_MARKER_PATH,
        expected_faculty=faculty,
    )
    return collection


def configure_runtime_paths(faculty):
    global ACTIVE_FACULTY
    global DB_PATH
    global NORMALIZED_CHECK_DIR
    global REPORT_PATH
    global CHANGES_PATH
    global VERSIONED_REPORT_PATH
    global VERSIONED_CHANGES_PATH
    global REFRESH_MARKER_PATH

    ACTIVE_FACULTY = normalize_faculty(faculty)
    paths = faculty_index_paths(
        ACTIVE_FACULTY, INDEX_VERSION, PROJECT_CONFIG)
    DB_PATH = paths["vector_db"]
    NORMALIZED_CHECK_DIR = paths["normalized"]
    REPORT_PATH = paths["report"]
    CHANGES_PATH = paths["changes"]
    VERSIONED_REPORT_PATH = REPORT_PATH
    VERSIONED_CHANGES_PATH = CHANGES_PATH
    REFRESH_MARKER_PATH = paths["marker"]
    os.makedirs(NORMALIZED_CHECK_DIR, exist_ok=True)
    return paths


def get_thread_chunk_tools():
    if not hasattr(THREAD_LOCAL, "chunker"):
        THREAD_LOCAL.chunker, THREAD_LOCAL.refinery = create_chunker_tools(
            CHUNK_CONFIG
        )
    return THREAD_LOCAL.chunker, THREAD_LOCAL.refinery


def get_gpu_snapshot():
    snapshot = {
        "cuda_available": torch.cuda.is_available(),
        "torch_allocated_mb": 0.0,
        "torch_reserved_mb": 0.0,
        "nvidia_util_percent": None,
        "nvidia_memory_used_mb": None,
        "nvidia_memory_total_mb": None,
    }

    if torch.cuda.is_available():
        snapshot["torch_allocated_mb"] = round(
            torch.cuda.memory_allocated() / (1024 ** 2), 1)
        snapshot["torch_reserved_mb"] = round(
            torch.cuda.memory_reserved() / (1024 ** 2), 1)

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip().splitlines()
        if output:
            util, used, total = [part.strip() for part in output[0].split(",")]
            snapshot["nvidia_util_percent"] = int(util)
            snapshot["nvidia_memory_used_mb"] = int(used)
            snapshot["nvidia_memory_total_mb"] = int(total)
    except Exception:
        pass

    return snapshot


def find_chunk_spans(content, chunk_texts):
    spans = []
    cursor = 0
    search_back = max(CHUNK_OVERLAP_SEARCH_CHARACTERS, 1)
    for chunk_text in chunk_texts:
        search_start = max(0, cursor - search_back)
        start = content.find(chunk_text, search_start)
        if start < 0:
            start = content.find(chunk_text)
        if start < 0:
            spans.append((-1, -1))
            continue
        end = start + len(chunk_text)
        spans.append((start, end))
        cursor = max(cursor, end)
    return spans


def sample_gpu_until_stopped(stop_event, samples):
    while not stop_event.wait(0.5):
        samples.append(get_gpu_snapshot())


def extract_and_chunk(fname, target_dir, existing_sources):
    """
    CPU-bound stage: runs in a thread pool, genuinely parallel.
    Returns (fname, list_of_chunk_texts) or None if skipped/failed.
    Does NOT touch the GPU.
    """
    fname = normalize_source_key(fname)
    fpath = os.path.join(target_dir, *fname.split("/"))
    started_at = time.time()
    source_metadata = load_source_metadata(
        target_dir, fname, default_faculty=ACTIVE_FACULTY)

    # Skip if already exists in ChromaDB
    if fname in existing_sources:
        return {
            "fname": fname,
            "chunks": None,
            "status": "skipped",
            "message": "already indexed",
            "source_metadata": source_metadata,
            "extract_seconds": round(time.time() - started_at, 3),
            "chunk_count": 0,
        }

    try:
        raw_file_hash = file_sha256(fpath)
        source_metadata_hash = metadata_sha256(source_metadata)
        content = extract_university_content(fpath, save_debug=False)
        if not content:
            return {
                "fname": fname,
                "chunks": None,
                "status": "unindexable",
                "message": "no content extracted",
                "source_metadata": source_metadata,
                "extract_seconds": round(time.time() - started_at, 3),
                "chunk_count": 0,
            }

        chunker, refinery = get_thread_chunk_tools()
        if CHUNKING_STRATEGY == "token":
            chunk_texts, chunk_spans = chunk_text_with_native_spans(
                content, chunker, refinery)
        else:
            chunk_texts = chunk_text(content, chunker, refinery)
            chunk_spans = find_chunk_spans(content, chunk_texts)
        content_hash = hashlib.sha256(
            content.encode("utf-8", errors="ignore")).hexdigest()

        if not chunk_texts:
            return {
                "fname": fname,
                "chunks": None,
                "status": "error",
                "message": "no chunks produced",
                "source_metadata": source_metadata,
                "extract_seconds": round(time.time() - started_at, 3),
                "chunk_count": 0,
            }

        return {
            "fname": fname,
            "content": content,
            "chunks": chunk_texts,
            "chunk_spans": chunk_spans,
            "content_hash": content_hash,
            "file_sha256": raw_file_hash,
            "source_metadata": source_metadata,
            "source_metadata_sha256": source_metadata_hash,
            "status": "ready",
            "message": f"{len(chunk_texts)} chunks ready",
            "extract_seconds": round(time.time() - started_at, 3),
            "chunk_count": len(chunk_texts),
        }
    except Exception as exc:
        return {
            "fname": fname,
            "chunks": None,
            "status": "error",
            "message": str(exc),
            "source_metadata": source_metadata,
            "extract_seconds": round(time.time() - started_at, 3),
            "chunk_count": 0,
        }


def source_vector_snapshot(collection, source):
    result = collection.get(
        where={"source": source},
        include=["embeddings", "documents", "metadatas"],
    )
    ids = list(result.get("ids") or ())

    def values(name):
        value = result.get(name)
        return [] if value is None else list(value)

    embeddings = values("embeddings")
    documents = values("documents")
    metadatas = values("metadatas")
    lengths = {len(ids), len(embeddings), len(documents), len(metadatas)}
    if len(lengths) != 1:
        raise RuntimeError(
            f"Cannot snapshot {source}: inconsistent Chroma record lengths")
    return {
        "ids": ids,
        "embeddings": embeddings,
        "documents": documents,
        "metadatas": metadatas,
    }


def restore_source_vectors(
    collection, snapshot, replacement_ids, batch_size
):
    ids_to_remove = list(dict.fromkeys(
        list(replacement_ids) + list(snapshot["ids"])))
    if ids_to_remove:
        collection.delete(ids=ids_to_remove)
    for start in range(0, len(snapshot["ids"]), batch_size):
        end = start + batch_size
        collection.add(
            ids=snapshot["ids"][start:end],
            embeddings=snapshot["embeddings"][start:end],
            documents=snapshot["documents"][start:end],
            metadatas=snapshot["metadatas"][start:end],
        )
    if snapshot["ids"]:
        restored = collection.get(ids=snapshot["ids"], include=[])
        if set(restored.get("ids") or ()) != set(snapshot["ids"]):
            raise RuntimeError("Old Chroma source could not be fully restored")


def publish_source_update(
    collection,
    normalized_root,
    source,
    text,
    source_metadata,
    ids,
    embeddings,
    documents,
    metadatas,
    batch_size,
):
    ids = list(ids)
    embeddings = list(embeddings)
    documents = list(documents)
    metadatas = list(metadatas)
    lengths = {len(ids), len(embeddings), len(documents), len(metadatas)}
    if len(lengths) != 1 or not ids or batch_size <= 0:
        raise ValueError("A source update requires equal, non-empty vector data")

    snapshot = source_vector_snapshot(collection, source)
    staged = stage_normalized_document(
        normalized_root, source, text, source_metadata)
    vector_mutation_started = False
    try:
        vector_mutation_started = True
        if snapshot["ids"]:
            collection.delete(ids=snapshot["ids"])
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        indexed = collection.get(ids=ids, include=[])
        if set(indexed.get("ids") or ()) != set(ids):
            raise RuntimeError(
                f"post-write verification failed: "
                f"{len(indexed.get('ids') or ())}/{len(ids)} chunks")
        publish_staged_normalized_document(staged)
    except Exception as exc:
        rollback_error = None
        if vector_mutation_started:
            try:
                restore_source_vectors(
                    collection, snapshot, ids, batch_size)
            except Exception as rollback_exc:
                rollback_error = rollback_exc
        discard_staged_normalized_document(staged)
        if rollback_error is not None:
            raise RuntimeError(
                f"{exc}; source rollback failed: {rollback_error}"
            ) from exc
        raise
    return len(ids)


def chroma_write_worker(write_queue, collection, result_log, stats):
    """Write CPU-resident embeddings while the GPU encodes the next batch."""
    while True:
        payload = write_queue.get()
        try:
            if payload is None:
                return

            embeddings = payload["embeddings"]
            cursor = 0
            write_started_at = time.time()
            for (
                fname,
                content,
                texts,
                spans,
                content_hash,
                file_hash,
                source_metadata,
                source_metadata_hash,
                extract_seconds,
            ) in payload["files"]:
                chunk_count = len(texts)
                file_embeddings = embeddings[cursor:cursor + chunk_count]
                cursor += chunk_count
                hash_prefix = content_hash[:12]
                source_prefix = hashlib.sha256(
                    fname.encode("utf-8", errors="ignore")
                ).hexdigest()[:16]
                file_ids = [
                    f"{source_prefix}_{hash_prefix}_{index}"
                    for index in range(chunk_count)
                ]
                provenance = chroma_metadata_fields(source_metadata)
                file_metas = []
                for index in range(chunk_count):
                    metadata = {
                        "source": fname,
                        "chunk_index": index,
                        "chunk_count": chunk_count,
                        "start": spans[index][0],
                        "end": spans[index][1],
                        "content_sha256": content_hash,
                        "file_sha256": file_hash,
                        "source_metadata_sha256": source_metadata_hash,
                        "index_version": INDEX_VERSION,
                        "index_config_hash": INDEX_CONFIG_HASH,
                        "model": MODEL_NAME,
                        "chunking_strategy": CHUNKING_STRATEGY,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_size_unit": CHUNK_SIZE_UNIT,
                        "chunk_overlap": CHUNK_OVERLAP,
                    }
                    metadata.update(provenance)
                    file_metas.append(metadata)

                try:
                    file_embeds_list = file_embeddings.tolist()
                    normalized_metadata = dict(source_metadata)
                    normalized_metadata.update({
                        "content_sha256": content_hash,
                        "file_sha256": file_hash,
                        "source_metadata_sha256": source_metadata_hash,
                        "index_version": INDEX_VERSION,
                        "index_config_hash": INDEX_CONFIG_HASH,
                    })
                    publish_source_update(
                        collection=collection,
                        normalized_root=NORMALIZED_CHECK_DIR,
                        source=fname,
                        text=content,
                        source_metadata=normalized_metadata,
                        ids=file_ids,
                        embeddings=file_embeds_list,
                        documents=texts,
                        metadatas=file_metas,
                        batch_size=MAX_CHROMA_BATCH,
                    )
                    allocated_encode_seconds = (
                        payload["encode_seconds"]
                        * chunk_count
                        / max(1, len(embeddings))
                    )
                    result_log.append({
                        "fname": fname,
                        "status": "success",
                        "message": "indexed",
                        "chunk_count": chunk_count,
                        "faculty": provenance["faculty"],
                        "collection_uuid": provenance.get(
                            "dspace_collection_uuid"),
                        "collection_name": provenance.get(
                            "dspace_collection_name"),
                        "community_name": provenance.get(
                            "dspace_community_name"),
                        "extract_seconds": extract_seconds,
                        "encode_batch_seconds": round(
                            allocated_encode_seconds, 3),
                    })
                except Exception as exc:
                    result_log.append({
                        "fname": fname,
                        "status": "error",
                        "message": f"Source publish failed: {exc}",
                        "chunk_count": chunk_count,
                        "faculty": provenance["faculty"],
                        "collection_uuid": provenance.get(
                            "dspace_collection_uuid"),
                        "collection_name": provenance.get(
                            "dspace_collection_name"),
                        "community_name": provenance.get(
                            "dspace_community_name"),
                        "extract_seconds": extract_seconds,
                        "encode_batch_seconds": payload["encode_seconds"],
                    })
            stats["chroma_write_seconds"] += time.time() - write_started_at
        finally:
            write_queue.task_done()


def gpu_encode_worker(work_queue, write_queue, model, result_log, stats, stop_event):
    """
    Single dedicated consumer. Pulls (fname, chunk_texts) items from the queue,
    accumulates them until GPU_QUEUE_TARGET_SIZE chunks are buffered (or the
    queue is drained), then encodes everything in ONE large batch call.
    This replaces per-file small-batch encode() calls and removes the need
    for a lock, since only this single thread ever touches the GPU.
    """
    pending_fname_chunks = []  # list of (fname, [texts])
    pending_texts = []         # flat list of all texts across files in this batch
    active_batch_size = ENCODE_BATCH_SIZE

    def flush():
        nonlocal active_batch_size
        if not pending_texts:
            return
        before_gpu = get_gpu_snapshot()
        monitor_stop = threading.Event()
        monitor_samples = [before_gpu]
        monitor_thread = threading.Thread(
            target=sample_gpu_until_stopped,
            args=(monitor_stop, monitor_samples),
            daemon=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor_thread.start()
        t0 = time.time()
        try:
            while True:
                try:
                    with torch.inference_mode():
                        results = model.encode(
                            pending_texts,
                            batch_size=active_batch_size,
                            max_length=ENCODE_MAX_LENGTH,
                            return_dense=True,
                            return_sparse=False,
                            return_colbert_vecs=False)
                    break
                except torch.cuda.OutOfMemoryError:
                    if active_batch_size <= 4:
                        raise
                    active_batch_size = max(4, active_batch_size // 2)
                    torch.cuda.empty_cache()
                    print(f"    [GPU] VRAM limit reached; retrying with "
                          f"batch={active_batch_size}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            embeddings = results['dense_vecs']
        except Exception as exc:
            elapsed = time.time() - t0
            for (
                fname,
                _content,
                texts,
                _spans,
                _content_hash,
                _file_hash,
                source_metadata,
                _source_metadata_hash,
                extract_seconds,
            ) in pending_fname_chunks:
                result_log.append({
                    "fname": fname,
                    "status": "error",
                    "message": f"GPU encode failed: {exc}",
                    "chunk_count": len(texts),
                    "faculty": source_metadata.get("faculty", "default"),
                    "collection_uuid": source_metadata.get(
                        "dspace_collection_uuid"),
                    "collection_name": source_metadata.get(
                        "dspace_collection_name"),
                    "community_name": source_metadata.get(
                        "dspace_community_name"),
                    "extract_seconds": extract_seconds,
                    "encode_batch_seconds": round(elapsed, 3),
                })
            print(f"    [GPU] Encode failed: {exc}")
            pending_fname_chunks.clear()
            pending_texts.clear()
            return
        finally:
            monitor_stop.set()
            monitor_thread.join()
        elapsed = time.time() - t0
        after_gpu = get_gpu_snapshot()
        monitor_samples.append(after_gpu)

        write_queue.put({
            "embeddings": embeddings,
            "files": pending_fname_chunks.copy(),
            "encode_seconds": round(elapsed, 3),
        })
        stats["encoded_chunks"] += len(pending_texts)
        stats["encode_batches"] += 1
        stats["encode_seconds"] += elapsed
        stats["active_encode_batch_size"] = active_batch_size
        if torch.cuda.is_available():
            stats["torch_peak_allocated_mb"] = max(
                stats["torch_peak_allocated_mb"],
                torch.cuda.max_memory_allocated() / (1024 ** 2))
            stats["torch_peak_reserved_mb"] = max(
                stats["torch_peak_reserved_mb"],
                torch.cuda.max_memory_reserved() / (1024 ** 2))
        stats["gpu_snapshots"].append({
            "before": before_gpu,
            "after": after_gpu,
            "chunks": len(pending_texts),
            "files": len(pending_fname_chunks),
            "seconds": round(elapsed, 3),
        })

        print(f"    [GPU] Encoded {len(pending_texts)} chunks across "
              f"{len(pending_fname_chunks)} file(s) in {elapsed:.1f}s "
              f"| util={after_gpu['nvidia_util_percent']}% "
              f"| mem={after_gpu['nvidia_memory_used_mb']}MB")

        pending_fname_chunks.clear()
        pending_texts.clear()

    while not stop_event.is_set() or not work_queue.empty():
        try:
            item = work_queue.get(timeout=0.2)
        except queue.Empty:
            flush()
            continue

        try:
            if item["status"] == "ready":
                pending_fname_chunks.append(
                    (item["fname"], item["content"], item["chunks"],
                     item["chunk_spans"],
                     item["content_hash"], item["file_sha256"],
                     item["source_metadata"],
                     item["source_metadata_sha256"],
                     item["extract_seconds"]))
                pending_texts.extend(item["chunks"])
                if len(pending_texts) >= GPU_QUEUE_TARGET_SIZE:
                    flush()
            else:
                result_log.append({
                    "fname": item["fname"],
                    "status": item["status"],
                    "message": item["message"],
                    "chunk_count": item["chunk_count"],
                    "faculty": item.get(
                        "source_metadata", {}).get("faculty", "default"),
                    "collection_uuid": item.get(
                        "source_metadata", {}).get("dspace_collection_uuid"),
                    "collection_name": item.get(
                        "source_metadata", {}).get("dspace_collection_name"),
                    "community_name": item.get(
                        "source_metadata", {}).get("dspace_community_name"),
                    "extract_seconds": item["extract_seconds"],
                    "encode_batch_seconds": 0.0,
                })
        finally:
            work_queue.task_done()

    flush()


def load_existing_sources(
    collection, files, target_dir=None, normalized_root=None
):
    existing_sources = set()
    for fname in files:
        fname = normalize_source_key(fname)
        try:
            existing = collection.get(
                where={"source": fname},
                limit=1,
                include=["metadatas"],
            )
            ids = existing.get("ids", ())
            if not ids:
                continue
            metadata = (existing.get("metadatas") or [{}])[0] or {}
            if target_dir is not None:
                source_path = os.path.join(
                    target_dir, *fname.split("/"))
                current_file_hash = file_sha256(source_path)
                current_source_metadata = load_source_metadata(
                    target_dir, fname, default_faculty=ACTIVE_FACULTY)
                indexed_file_hash = metadata.get("file_sha256")
                indexed_faculty = metadata.get("faculty")
                indexed_metadata_hash = metadata.get(
                    "source_metadata_sha256")
                current_metadata_hash = metadata_sha256(
                    current_source_metadata)
                if (
                    indexed_file_hash
                    and indexed_file_hash != current_file_hash
                ) or (
                    indexed_faculty
                    and indexed_faculty
                    != current_source_metadata["faculty"]
                ) or (
                    indexed_metadata_hash
                    and indexed_metadata_hash != current_metadata_hash
                ):
                    print(
                        f" > Changed source {fname}; staging replacement "
                        "while keeping the published version.")
                    continue
            if normalized_root is not None:
                text_path, sidecar_path = managed_normalized_document_paths(
                    normalized_root, fname)
                if not (
                    os.path.isfile(text_path)
                    and os.path.isfile(sidecar_path)
                ):
                    print(
                        f" > Missing normalized artifacts for {fname}; "
                        "staging replacement.")
                    continue
            expected_count = metadata.get("chunk_count")
            if expected_count is None:
                # Backward-compatible trust for indexes built before the
                # chunk_count metadata and transactional verification existed.
                existing_sources.add(fname)
                continue
            all_ids = collection.get(
                where={"source": fname},
                include=[],
            ).get("ids", ())
            if len(all_ids) == int(expected_count):
                existing_sources.add(fname)
            else:
                print(
                    f" > Incomplete source {fname}: "
                    f"{len(all_ids)}/{expected_count}; staging replacement "
                    "without deleting the current records.")
        except Exception as exc:
            print(f" > Warning: Could not check existing source {fname}: {exc}")
    return existing_sources


def remove_stale_sources(
    collection, current_files, marker_path, normalized_root=None
):
    marker = load_json_file(marker_path, default={})
    indexed_sources = set(marker.get("source_keys") or ())
    current_sources = {
        normalize_source_key(source) for source in current_files
    }
    stale_sources = sorted(indexed_sources - current_sources)

    if normalized_root is not None:
        for source in stale_sources:
            managed_normalized_document_paths(normalized_root, source)

    for source in stale_sources:
        if normalized_root is not None:
            remove_normalized_document(normalized_root, source)
        collection.delete(where={"source": source})

    if stale_sources:
        marker["source_keys"] = sorted(indexed_sources - set(stale_sources))
        marker["active_source_count"] = len(marker["source_keys"])
        marker["removed_sources"] = (
            int(marker.get("removed_sources") or 0) + len(stale_sources))
        write_json_file(marker_path, marker)
    return stale_sources


def write_reports(
    target_dir,
    files,
    result_log,
    stats,
    started_at,
    initial_gpu,
    final_gpu,
    final_collection_chunks,
):
    elapsed = time.time() - started_at
    success = [r for r in result_log if r["status"] == "success"]
    skipped = [r for r in result_log if r["status"] == "skipped"]
    unindexable = [
        r for r in result_log if r["status"] == "unindexable"
    ]
    errors = [r for r in result_log if r["status"] == "error"]
    processed = len(success) + len(skipped) + len(unindexable) + len(errors)
    chunks = sum(r.get("chunk_count", 0) for r in result_log)
    avg_file_seconds = elapsed / processed if processed else 0.0
    avg_extract_seconds = (
        sum(r.get("extract_seconds", 0.0) for r in result_log) / processed
        if processed else 0.0
    )
    avg_encode_seconds_per_chunk = (
        stats["encode_seconds"] / stats["encoded_chunks"]
        if stats["encoded_chunks"] else 0.0
    )
    faculty_counts = {}
    collection_counts = {}
    for item in result_log:
        faculty = item.get("faculty") or "default"
        faculty_counts[faculty] = faculty_counts.get(faculty, 0) + 1
        collection_name = item.get("collection_name") or "local-or-unknown"
        collection_counts[collection_name] = (
            collection_counts.get(collection_name, 0) + 1)

    peak_gpu_util = None
    peak_gpu_mem = None
    util_values = []
    mem_values = []
    for sample in stats["gpu_samples"]:
        if sample["nvidia_util_percent"] is not None:
            util_values.append(sample["nvidia_util_percent"])
        if sample["nvidia_memory_used_mb"] is not None:
            mem_values.append(sample["nvidia_memory_used_mb"])
    if util_values:
        peak_gpu_util = max(util_values)
    if mem_values:
        peak_gpu_mem = max(mem_values)

    lines = [
        "Vector DB indexing report",
        f"Faculty: {ACTIVE_FACULTY}",
        f"Index version: {INDEX_VERSION}",
        f"Index config hash: {INDEX_CONFIG_HASH}",
        f"Database path: {DB_PATH}",
        f"Collection: {COLLECTION_NAME}",
        f"Model: {MODEL_NAME}",
        f"Chunking strategy: {CHUNKING_STRATEGY}",
        f"Chunk size: {CHUNK_SIZE} {CHUNK_SIZE_UNIT}",
        f"Chunk overlap: {CHUNK_OVERLAP}",
        f"Chunk tokenizer: {CHUNK_CONFIG.get('tokenizer', 'not applicable')}",
        f"GPU encode batch size: {ENCODE_BATCH_SIZE}",
        f"GPU encode max token length: {ENCODE_MAX_LENGTH}",
        f"Final active GPU encode batch size: {stats['active_encode_batch_size']}",
        f"GPU queue target chunks: {GPU_QUEUE_TARGET_SIZE}",
        f"Chroma writer queue batches: {WRITE_QUEUE_MAX_BATCHES}",
        f"Target directory: {target_dir}",
        f"Files discovered: {len(files)}",
        f"Faculties discovered: {len(faculty_counts)}",
        f"Files per faculty: {faculty_counts}",
        f"DSpace collections represented: {len(collection_counts)}",
        f"Files per DSpace collection: {collection_counts}",
        f"Files completed: {processed}",
        f"Success: {len(success)}",
        f"Skipped: {len(skipped)}",
        f"Unindexable source files: {len(unindexable)}",
        f"Errors: {len(errors)}",
        f"Chunks encoded this run: {chunks}",
        f"Final collection chunks: {final_collection_chunks}",
        f"Total runtime seconds: {elapsed:.2f}",
        f"Average wall time per file seconds: {avg_file_seconds:.3f}",
        f"Average extraction/chunking seconds per file: {avg_extract_seconds:.3f}",
        f"GPU encode batches: {stats['encode_batches']}",
        f"GPU encoded chunks: {stats['encoded_chunks']}",
        f"GPU encode seconds: {stats['encode_seconds']:.2f}",
        f"Average GPU encode seconds per chunk: {avg_encode_seconds_per_chunk:.5f}",
        f"Chroma write seconds: {stats['chroma_write_seconds']:.2f}",
        f"Workers: {EXTRACT_WORKERS} extraction workers + 1 GPU encoder worker",
        f"CUDA available: {initial_gpu['cuda_available']}",
        f"Initial GPU snapshot: {initial_gpu}",
        f"Final GPU snapshot: {final_gpu}",
        f"Peak nvidia-smi GPU utilization percent: {peak_gpu_util}",
        f"Average sampled GPU utilization percent: "
        f"{(sum(util_values) / len(util_values)) if util_values else None}",
        f"Peak nvidia-smi GPU memory used MB: {peak_gpu_mem}",
        f"Peak torch allocated MB: {stats['torch_peak_allocated_mb']:.1f}",
        f"Peak torch reserved MB: {stats['torch_peak_reserved_mb']:.1f}",
        "",
        "Per-file results:",
    ]
    for item in result_log:
        lines.append(
            f"{item['status'].upper()}: {item['fname']} | "
            f"faculty={item.get('faculty', 'default')} | "
            f"collection={item.get('collection_name') or 'local-or-unknown'} | "
            f"community={item.get('community_name') or 'local-or-unknown'} | "
            f"chunks={item.get('chunk_count', 0)} | "
            f"extract_s={item.get('extract_seconds', 0.0)} | "
            f"encode_batch_s={item.get('encode_batch_seconds', 0.0)} | "
            f"{item.get('message', '')}"
        )

    report_text = "\n".join(lines)
    for report_path in dict.fromkeys((REPORT_PATH, VERSIONED_REPORT_PATH)):
        with open(report_path, "w", encoding="utf-8") as report:
            report.write(report_text)

    changes = [
        "Short change summary",
        "- Fixed the GPU worker so it actually consumes queued files and exits cleanly.",
        "- Added parallel extraction/chunking workers feeding one GPU encoder worker.",
        "- Added batched GPU encoding and batched ChromaDB writes.",
        "- Added nvidia-smi / torch GPU snapshots and per-file timing report.",
        "- Added skip/error handling so bad files do not crash the whole run.",
        f"- Built versioned index {INDEX_VERSION} using {CHUNKING_STRATEGY} chunking.",
        "- Added source offsets, chunk index, content hash, model, and index version metadata.",
        "- Added configurable batch/queue/path settings and continuous GPU sampling during encoding.",
        "- Overlapped Chroma writes in RAM with the next GPU encode batch.",
        "- Added inference mode and automatic CUDA out-of-memory batch backoff.",
        "- Revalidated batch 32 as the indexing default; batch 48 was 3.77% slower on 1,024 chunks.",
        "- Limited 400-character chunks to 512 model tokens instead of 8192.",
        "- Added per-source chunk-count verification and rollback for partial writes.",
        "- Added chunk_count metadata so interrupted sources can be rebuilt safely.",
        "- Allocated batch encode time proportionally per file for accurate reporting.",
    ]
    changes_text = "\n".join(changes)
    for changes_path in dict.fromkeys((CHANGES_PATH, VERSIONED_CHANGES_PATH)):
        with open(changes_path, "w", encoding="utf-8") as change_file:
            change_file.write(changes_text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build one isolated vector database for one faculty.")
    parser.add_argument("corpus_directory")
    parser.add_argument(
        "--faculty",
        default=None,
        help="Faculty key. Defaults to the corpus folder name.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    target_dir = os.path.abspath(args.corpus_directory)
    if not os.path.isdir(target_dir):
        print(f"Error: corpus directory not found: {target_dir}")
        return 1
    faculty = normalize_faculty(
        args.faculty or os.path.basename(os.path.normpath(target_dir)))
    configure_runtime_paths(faculty)
    started_at = time.time()

    # 1. Validate an existing index before loading the model or mutating data.
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = open_compatible_collection(client, faculty)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2

    # 2. Initialize BGE-M3
    print("--- [STARTUP] Initializing BGE-M3 Model ---")
    model_options = {}
    if MODEL_CONFIG.get("revision"):
        model_options["revision"] = MODEL_CONFIG["revision"]
    model = BGEM3FlagModel(
        MODEL_NAME,
        cache_dir=PROJECT_CACHE,
        use_fp16=(
            torch.cuda.is_available()
            and bool(MODEL_CONFIG.get("use_fp16_on_cuda", True))
        ),
        **model_options,
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        model.model.to("cuda")
        model.model.eval()
        print(" > GPU Detected.")
    else:
        print(" > No GPU detected, running on CPU (will be slow).")

    files = discover_corpus_files(target_dir)
    stale_sources = remove_stale_sources(
        collection,
        files,
        REFRESH_MARKER_PATH,
        normalized_root=NORMALIZED_CHECK_DIR,
    )
    if stale_sources:
        print(f" > Removed {len(stale_sources)} stale source(s).")
    published_sources = set(
        load_json_file(REFRESH_MARKER_PATH, default={}).get(
            "source_keys") or ()
    )
    # Production run: always process the entire supplied corpus.
    existing_sources = load_existing_sources(
        collection,
        files,
        target_dir=target_dir,
        normalized_root=NORMALIZED_CHECK_DIR,
    )
    print(f" > Existing indexed sources in this run: {len(existing_sources)}")
    total_files = len(files)
    print(f"--- Processing {total_files} files "
          f"({EXTRACT_WORKERS} extraction threads -> 1 GPU encoder) ---")
    print(
        f"--- Index: {INDEX_VERSION} | strategy={CHUNKING_STRATEGY} | "
        f"max={CHUNK_SIZE} {CHUNK_SIZE_UNIT} | overlap={CHUNK_OVERLAP} ---"
    )
    print(f"--- GPU: {GPU_MEMORY_GB:.1f} GB VRAM | encode batch={ENCODE_BATCH_SIZE} ---")
    print(f"--- Database: {DB_PATH} ---")
    print(f"--- Faculty: {faculty} ---")
    print(f"--- Report will be written to: {REPORT_PATH} ---")

    work_queue = queue.Queue(maxsize=WORK_QUEUE_MAX_FILES)
    write_queue = queue.Queue(maxsize=WRITE_QUEUE_MAX_BATCHES)
    result_log = []
    stop_event = threading.Event()
    stats = {
        "encoded_chunks": 0,
        "encode_batches": 0,
        "encode_seconds": 0.0,
        "chroma_write_seconds": 0.0,
        "gpu_snapshots": [],
        "gpu_samples": [],
        "torch_peak_allocated_mb": 0.0,
        "torch_peak_reserved_mb": 0.0,
        "active_encode_batch_size": ENCODE_BATCH_SIZE,
    }
    initial_gpu = get_gpu_snapshot()
    stats["gpu_samples"].append(initial_gpu)
    global_monitor_stop = threading.Event()
    global_monitor_thread = threading.Thread(
        target=sample_gpu_until_stopped,
        args=(global_monitor_stop, stats["gpu_samples"]),
        daemon=True)
    global_monitor_thread.start()

    writer_thread = threading.Thread(
        target=chroma_write_worker,
        args=(write_queue, collection, result_log, stats),
        daemon=True)
    writer_thread.start()

    # Start the single GPU consumer thread
    gpu_thread = threading.Thread(
        target=gpu_encode_worker,
        args=(work_queue, write_queue, model, result_log, stats, stop_event),
        daemon=True
    )
    gpu_thread.start()

    # CPU-bound extraction runs in parallel, feeds the GPU queue as results come in
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
        futures = [executor.submit(
            extract_and_chunk, f, target_dir, existing_sources) for f in files]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{i}/{total_files}] {result['fname']}: {result['message']}")
            work_queue.put(result)

    # Signal the GPU worker that no more items are coming, then wait for it to drain
    work_queue.join()
    stop_event.set()
    gpu_thread.join()
    write_queue.put(None)
    write_queue.join()
    writer_thread.join()
    global_monitor_stop.set()
    global_monitor_thread.join()

    print("\n--- Batch Indexing Complete ---")
    final_gpu = get_gpu_snapshot()
    write_reports(
        target_dir,
        files,
        result_log,
        stats,
        started_at,
        initial_gpu,
        final_gpu,
        collection.count(),
    )
    skipped = sum(1 for r in result_log if r["status"] == "skipped")
    unindexable = sum(
        1 for r in result_log if r["status"] == "unindexable")
    errors = sum(1 for r in result_log if r["status"] == "error")
    success = sum(1 for r in result_log if r["status"] == "success")
    published_sources.update(
        result["fname"]
        for result in result_log
        if result["status"] in {"success", "skipped"}
    )
    write_json_file(REFRESH_MARKER_PATH, {
        "index_version": INDEX_VERSION,
        "index_config_hash": INDEX_CONFIG_HASH,
        "index_config": INDEX_CONFIG_PAYLOAD,
        "faculty": faculty,
        "target_directory": target_dir,
        "updated_at_unix": time.time(),
        "source_files": len(files),
        "active_source_count": len(published_sources),
        "success": success,
        "skipped": skipped,
        "unindexable": unindexable,
        "errors": errors,
        "collection_chunks": collection.count(),
        "source_keys": sorted(published_sources),
        "removed_sources": len(stale_sources),
    })
    print(
        f"  Success: {success} | Skipped: {skipped} | "
        f"Unindexable: {unindexable} | Errors: {errors}")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Versioned report: {VERSIONED_REPORT_PATH}")
    print(f"  Changes: {CHANGES_PATH}")
    print(f"  Versioned changes: {VERSIONED_CHANGES_PATH}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
