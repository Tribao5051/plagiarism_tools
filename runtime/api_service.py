import asyncio
import hashlib
import os
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Annotated

from project_config import (
    PROJECT_ROOT,
    build_index_config_hash,
    load_project_config,
    resolve_project_path,
    validate_index_config_hash,
)

# Cache configuration must be set before importing model libraries.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_CONFIG = load_project_config()
MODEL_CONFIG = PROJECT_CONFIG["model"]
INDEXING_CONFIG = PROJECT_CONFIG["indexing"]
API_CONFIG = PROJECT_CONFIG["api"]
CACHE_DIR = resolve_project_path(MODEL_CONFIG["cache_dir"])
MPL_CACHE_DIR = os.path.join(PROJECT_ROOT, "mpl_temp")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MPL_CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = os.path.join(CACHE_DIR, "hub")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR
os.environ['MPLCONFIGDIR'] = MPL_CACHE_DIR
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import chromadb
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer

from chunking_config import create_chunker_tools, load_chunking_config
from corpus_metadata import (
    DEFAULT_FACULTY,
    discover_normalized_documents,
    load_json_file,
    normalize_faculty,
    normalize_source_key,
)
from detection_module_diff_p4 import (
    DEFAULT_EVIDENCE_PROFILE as DETECTOR_DEFAULT_EVIDENCE_PROFILE,
    EVIDENCE_PROFILES,
    perform_plagiarism_scan,
)
from model_module import model_manager
from pdf_extract import EXTRACTION_VERSION, extract_university_content
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
FACULTY_INDEX_ROOT = os.path.join(
    resolve_project_path(INDEXING_CONFIG["faculty_index_root"]),
    INDEX_VERSION,
)
LEGACY_DB_PATH = os.path.join(
    PROJECT_ROOT, f"corpus_vector_db_{INDEX_VERSION}")
LEGACY_CORPUS_DIR = os.path.join(
    PROJECT_ROOT, f"corpus_normalized_{INDEX_VERSION}")
LEGACY_MARKER_PATH = os.path.join(
    PROJECT_ROOT, f"corpus_index_{INDEX_VERSION}.json")
MAX_UPLOAD_BYTES = int(
    os.getenv(
        "PLAGIARISM_MAX_UPLOAD_MB", API_CONFIG["max_upload_mb"]
    )) * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
RETRIEVAL_STRATEGIES = {"shortlist", "exhaustive"}
DEFAULT_RETRIEVAL_STRATEGY = os.getenv(
    "PLAGIARISM_RETRIEVAL_STRATEGY",
    API_CONFIG["default_retrieval_strategy"],
).strip().lower()
DEFAULT_EVIDENCE_PROFILE = os.getenv(
    "PLAGIARISM_EVIDENCE_PROFILE",
    API_CONFIG.get(
        "default_evidence_profile", DETECTOR_DEFAULT_EVIDENCE_PROFILE),
).strip().lower()


def resolve_candidate_sources(candidate_sources, available_sources):
    if candidate_sources is None:
        return None, None
    if not bool(API_CONFIG.get("allow_oracle_candidate_sources", False)):
        return None, JSONResponse(
            status_code=403,
            content={"error": "Oracle candidate sources are disabled"},
        )

    requested_sources = list(dict.fromkeys(
        source.strip()
        for source in candidate_sources.split(",")
        if source.strip()
    ))
    if not requested_sources:
        return None, JSONResponse(
            status_code=400,
            content={"error": "Candidate sources cannot be empty"},
        )

    available_sources = set(available_sources or ())
    invalid_sources = [
        source for source in requested_sources
        if source not in available_sources
    ]
    if invalid_sources:
        return None, JSONResponse(
            status_code=400,
            content={
                "error": "Unknown candidate source",
                "invalid_candidate_sources": invalid_sources,
            },
        )
    return requested_sources, None


ml_state = {}
scan_lock = asyncio.Lock()


def discover_faculty_index_specs():
    specs = {}
    if os.path.isdir(FACULTY_INDEX_ROOT):
        for name in sorted(os.listdir(FACULTY_INDEX_ROOT)):
            faculty_root = os.path.join(FACULTY_INDEX_ROOT, name)
            db_path = os.path.join(faculty_root, "vector_db")
            normalized_dir = os.path.join(faculty_root, "normalized")
            marker_path = os.path.join(faculty_root, "corpus_index.json")
            if not (
                os.path.isdir(faculty_root)
                and os.path.isdir(db_path)
                and os.path.isdir(normalized_dir)
                and os.path.isfile(marker_path)
            ):
                continue
            faculty = normalize_faculty(name)
            specs[faculty] = {
                "faculty": faculty,
                "db_path": db_path,
                "normalized_dir": normalized_dir,
                "marker_path": marker_path,
                "legacy": False,
            }
    return specs


def corpus_refresh_signature(specs=None):
    specs = specs or discover_faculty_index_specs()
    if specs:
        return tuple(
            (faculty, os.stat(spec["marker_path"]).st_mtime_ns
             if os.path.isfile(spec["marker_path"]) else 0.0)
            for faculty, spec in sorted(specs.items())
        )
    return ((
        "__legacy__",
        os.stat(LEGACY_MARKER_PATH).st_mtime_ns
        if os.path.isfile(LEGACY_MARKER_PATH) else 0.0,
    ),)


def build_text_index(
    documents,
    faculty,
    collection,
    chroma_client,
    where=None,
    cache_namespace=None,
    active_sources=None,
):
    active_sources = (
        {normalize_source_key(source) for source in active_sources}
        if active_sources is not None
        else None
    )
    filenames = []
    contents = []
    corpus_texts = {}
    source_versions = {}
    for document in documents:
        source = normalize_source_key(document["source"])
        if active_sources is not None and source not in active_sources:
            continue
        with open(
            document["path"], "r", encoding="utf-8", errors="ignore"
        ) as handle:
            text = handle.read()
        if not text.strip():
            continue
        filenames.append(source)
        contents.append(text)
        corpus_texts[source] = text
        source_versions[source] = hashlib.sha256(
            text.encode("utf-8", errors="ignore")
        ).hexdigest()
    if not contents:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(contents)
    collection_metadata = collection.metadata or {}
    if cache_namespace is None:
        cache_namespace = (
            faculty,
            str(collection_metadata.get("index_version") or INDEX_VERSION),
            str(collection_metadata.get("index_config_hash") or "unverified"),
            id(collection),
        )
    return {
        "faculty": faculty,
        "filenames": filenames,
        "filename_set": set(filenames),
        "corpus_texts": corpus_texts,
        "source_versions": source_versions,
        "source_faculties": {source: faculty for source in filenames},
        "tfidf_vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "collection": collection,
        "chroma": chroma_client,
        "collection_where": where,
        "cache_namespace": cache_namespace,
        "semantic_source_cache": OrderedDict(),
        "fingerprint_cache": OrderedDict(),
        "vector_db_chunks": collection.count(),
    }


def validate_collection(collection, expected_faculty=None, marker_path=None):
    metadata = collection.metadata or {}
    stored_version = metadata.get("index_version")
    if stored_version and stored_version != INDEX_VERSION:
        raise RuntimeError(
            f"Index mismatch: API requested {INDEX_VERSION}, "
            f"database contains {stored_version}")
    validate_index_config_hash(
        metadata.get("index_config_hash"),
        INDEX_CONFIG_HASH,
        location="Chroma collection",
    )
    if marker_path is not None:
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
        and normalize_faculty(stored_faculty) != expected_faculty
    ):
        raise RuntimeError(
            f"Faculty mismatch: folder={expected_faculty}, "
            f"database={stored_faculty}")


def load_physical_faculty_indexes(specs):
    result = {}
    for faculty, spec in specs.items():
        chroma_client = chromadb.PersistentClient(path=spec["db_path"])
        collection = chroma_client.get_collection(
            name=INDEXING_CONFIG["collection_name"])
        validate_collection(
            collection,
            expected_faculty=faculty,
            marker_path=spec["marker_path"],
        )
        documents = discover_normalized_documents(spec["normalized_dir"])
        active_sources = set(
            load_json_file(spec["marker_path"], default={}).get(
                "source_keys") or ()
        )
        state = build_text_index(
            documents,
            faculty,
            collection,
            chroma_client,
            active_sources=active_sources,
            cache_namespace=(
                faculty,
                os.path.normcase(os.path.abspath(spec["db_path"])),
                str(collection.metadata.get("index_version") or INDEX_VERSION),
                INDEX_CONFIG_HASH,
            ),
        )
        if state is not None:
            state.update(spec)
            result[faculty] = state
    return result


def load_legacy_faculty_indexes():
    if not (
        os.path.isdir(LEGACY_DB_PATH)
        and os.path.isdir(LEGACY_CORPUS_DIR)
    ):
        return {}
    chroma_client = chromadb.PersistentClient(path=LEGACY_DB_PATH)
    collection = chroma_client.get_collection(
        name=INDEXING_CONFIG["collection_name"])
    validate_collection(collection, marker_path=LEGACY_MARKER_PATH)
    active_sources = set(
        load_json_file(LEGACY_MARKER_PATH, default={}).get(
            "source_keys") or ()
    )
    grouped = {}
    for document in discover_normalized_documents(LEGACY_CORPUS_DIR):
        if normalize_source_key(document["source"]) not in active_sources:
            continue
        grouped.setdefault(document["faculty"], []).append(document)
    result = {}
    for faculty, documents in grouped.items():
        where = {"faculty": faculty} if faculty != DEFAULT_FACULTY else None
        state = build_text_index(
            documents,
            faculty,
            collection,
            chroma_client,
            where=where,
            active_sources=active_sources,
            cache_namespace=(
                faculty,
                os.path.normcase(os.path.abspath(LEGACY_DB_PATH)),
                str(collection.metadata.get("index_version") or INDEX_VERSION),
                INDEX_CONFIG_HASH,
            ),
        )
        if state is not None:
            state.update({
                "db_path": LEGACY_DB_PATH,
                "normalized_dir": LEGACY_CORPUS_DIR,
                "marker_path": LEGACY_MARKER_PATH,
                "legacy": True,
            })
            result[faculty] = state
    return result


def refresh_corpus_state(force=False):
    specs = discover_faculty_index_specs()
    signature = corpus_refresh_signature(specs)
    if (
        not force
        and ml_state.get("corpus_refresh_signature") == signature
        and ml_state.get("faculty_indexes")
    ):
        return False

    faculty_indexes = (
        load_physical_faculty_indexes(specs)
        if specs else load_legacy_faculty_indexes()
    )
    if not faculty_indexes:
        raise RuntimeError(
            f"No usable faculty index found in {FACULTY_INDEX_ROOT} "
            f"or legacy path {LEGACY_DB_PATH}")

    corpus_texts = {}
    source_faculties = {}
    total_chunks = 0
    for faculty, state in faculty_indexes.items():
        corpus_texts.update(state["corpus_texts"])
        source_faculties.update(state["source_faculties"])
        total_chunks += state["vector_db_chunks"]
    ml_state["faculty_indexes"] = faculty_indexes
    ml_state["faculties"] = sorted(faculty_indexes)
    ml_state["filenames"] = sorted(corpus_texts)
    ml_state["filename_set"] = set(corpus_texts)
    ml_state["corpus_texts"] = corpus_texts
    ml_state["source_faculties"] = source_faculties
    ml_state["vector_db_chunks"] = total_chunks
    ml_state["corpus_refresh_signature"] = signature
    ml_state.pop("semantic_source_cache", None)
    ml_state.pop("fingerprint_cache", None)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Loads models and 10k files into RAM ONCE """
    print("\n" + "="*50)
    print("--- [STARTUP] INITIALIZING SINGLETON ---")

    # 1. Validate indexes before loading the GPU model.
    refresh_corpus_state(force=True)

    # 2. SBERT from Cache
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" > Loading Model to {device.upper()} from {CACHE_DIR}...")
    ml_state["sbert"] = model_manager.get_model(
        model_name=MODEL_CONFIG["name"],
        cache_dir=CACHE_DIR)

    # 3. Configure the target chunker once.
    ml_state["chunking_config"] = dict(CHUNK_CONFIG)
    (
        ml_state["target_chunker"],
        ml_state["target_chunk_refinery"],
    ) = create_chunker_tools(CHUNK_CONFIG, include_overlap=False)

    print(f" > Faculty index root: {FACULTY_INDEX_ROOT}")
    print(
        f" > Chunking: {CHUNK_CONFIG['strategy']} | "
        f"max={CHUNK_CONFIG['chunk_size']} "
        f"{CHUNK_CONFIG['chunk_size_unit']}"
    )
    print(f" > Faculties: {', '.join(ml_state['faculties'])}")
    print(f" > Sources: {len(ml_state['filenames'])} | Chunks: "
          f"{ml_state['vector_db_chunks']}")
    print("--- [STARTUP] READY ---")
    print("="*50 + "\n")
    yield
    ml_state.clear()

app = FastAPI(lifespan=lifespan)

# --- ENABLE CORS ---
configured_origins = [
    origin.strip()
    for origin in os.getenv("PLAGIARISM_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_credentials=configured_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ready" if ml_state.get("faculty_indexes") else "starting",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sources": len(ml_state.get("filenames", ())),
        "vector_db_chunks": ml_state.get("vector_db_chunks", 0),
        "index_version": INDEX_VERSION,
        "chunking_strategy": CHUNK_CONFIG["strategy"],
        "faculties": {
            faculty: len(state["filenames"])
            for faculty, state in ml_state.get(
                "faculty_indexes", {}).items()
        },
        "default_retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
        "default_evidence_profile": DEFAULT_EVIDENCE_PROFILE,
        "evidence_profiles": sorted(EVIDENCE_PROFILES),
        "scan_serialization": "single_gpu_worker",
    }


@app.get("/faculties")
async def faculties():
    return {
        "faculties": [
            {
                "id": faculty,
                "source_count": len(state["filenames"]),
            }
            for faculty, state in sorted(
                ml_state.get("faculty_indexes", {}).items())
        ]
    }


@app.post("/scan")
async def scan(
    file: UploadFile = File(...),
    faculty: str | None = None,
    faculty_form: Annotated[
        str | None, Form(alias="faculty")
    ] = None,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    evidence_profile: str = DEFAULT_EVIDENCE_PROFILE,
    candidate_sources: str | None = None,
    encode_batch_size: int | None = None,
    include_raw_evidence: bool = False,
):
    if ml_state.get("faculty_indexes"):
        await run_in_threadpool(refresh_corpus_state)
    requested_faculty = normalize_faculty(
        faculty_form or faculty or DEFAULT_FACULTY)
    requested_strategy = retrieval_strategy.strip().lower()
    if requested_strategy not in RETRIEVAL_STRATEGIES:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported retrieval strategy",
                "allowed_retrieval_strategies": sorted(
                    RETRIEVAL_STRATEGIES),
            },
        )
    requested_evidence_profile = evidence_profile.strip().lower()
    if requested_evidence_profile not in EVIDENCE_PROFILES:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported evidence profile",
                "allowed_evidence_profiles": sorted(EVIDENCE_PROFILES),
            },
        )
    faculty_indexes = ml_state.get("faculty_indexes", {})
    if faculty_indexes and requested_faculty not in faculty_indexes:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unknown faculty",
                "requested_faculty": requested_faculty,
                "available_faculties": sorted(faculty_indexes),
            },
        )

    candidate_state = faculty_indexes.get(requested_faculty, ml_state)
    resolved_candidate_sources, candidate_error = resolve_candidate_sources(
        candidate_sources,
        candidate_state.get("filename_set", set()),
    )
    if candidate_error is not None:
        return candidate_error

    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported file type",
                "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            },
        )

    await file.seek(0)
    pdf_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": "File is too large",
                "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            },
        )
    if not pdf_bytes:
        return JSONResponse(
            status_code=400,
            content={"error": "Uploaded file is empty"},
        )

    def run_pipeline():
        request_started_at = time.perf_counter()
        extraction_started_at = time.perf_counter()
        content = extract_university_content(
            pdf_bytes, filename=file.filename, save_debug=False)
        extraction_seconds = time.perf_counter() - extraction_started_at
        if not content:
            return None

        print(f"DEBUG: Content đã trích xuất sạch (Dài: {len(content)} ký tự)")
        scan_state = ml_state.copy()
        current_faculty_indexes = ml_state.get("faculty_indexes", {})
        faculty_state = current_faculty_indexes.get(requested_faculty)
        if faculty_state is not None:
            scan_state.update(faculty_state)
        scan_state["faculty"] = requested_faculty
        scan_state["retrieval_strategy"] = requested_strategy
        scan_state["evidence_profile"] = requested_evidence_profile
        if resolved_candidate_sources is not None:
            scan_state["forced_candidate_sources"] = (
                resolved_candidate_sources
            )
        if encode_batch_size is not None:
            scan_state["encode_batch_size"] = max(
                4, min(int(encode_batch_size), 128))

        report = perform_plagiarism_scan(content, scan_state)
        if not include_raw_evidence:
            for result in report.get("results", ()):
                result.pop("raw_evidence", None)
                result.pop("consolidated_evidence", None)
        performance = report.setdefault("performance", {})
        performance["extraction_seconds"] = round(extraction_seconds, 4)
        performance["request_pipeline_seconds"] = round(
            time.perf_counter() - request_started_at, 4)
        performance["input_characters"] = len(content)
        performance["faculty"] = requested_faculty
        performance["requested_retrieval_strategy"] = requested_strategy
        performance["evidence_profile"] = requested_evidence_profile
        report["faculty"] = requested_faculty
        report["evidence_profile"] = requested_evidence_profile
        return report

    queue_started_at = time.perf_counter()
    try:
        # BGE-M3 is a singleton on one GPU. Serializing scans prevents
        # concurrent requests from racing or exhausting the 4 GB VRAM.
        async with scan_lock:
            queue_wait_seconds = time.perf_counter() - queue_started_at
            if ml_state.get("faculty_indexes"):
                await run_in_threadpool(refresh_corpus_state)
            report = await run_in_threadpool(run_pipeline)
    except Exception as exc:
        print(f"Scan Error: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Plagiarism scan failed",
                "detail": str(exc),
            },
        )

    if report is None:
        return JSONResponse(
            status_code=422,
            content={
                "error": (
                    "Extraction produced no text. "
                    "The PDF may be scanned and require OCR."
                )
            },
        )

    report.setdefault("performance", {})["queue_wait_seconds"] = round(
        queue_wait_seconds, 4)
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
