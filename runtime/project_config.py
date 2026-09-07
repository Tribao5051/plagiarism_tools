import copy
import hashlib
import json
import os
import tomllib
from functools import lru_cache


RUNTIME_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RUNTIME_ROOT)
DEFAULT_CONFIG_PATH = os.path.join(RUNTIME_ROOT, "plagiarism_config.toml")
INDEX_CONFIG_HASH_SCHEMA_VERSION = 1
INDEX_CHUNKING_KEYS = (
    "strategy",
    "chunk_size",
    "chunk_size_unit",
    "chunk_overlap",
    "chunk_overlap_search_characters",
    "min_chunk_characters",
    "tokenizer",
    "semantic_embedding_model",
    "semantic_threshold",
    "semantic_similarity_window",
    "semantic_min_sentences",
)


def resolve_project_path(value, project_root=PROJECT_ROOT):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(project_root, value))


@lru_cache(maxsize=4)
def _load_project_config(config_path):
    if os.path.splitext(config_path)[1].lower() == ".toml":
        with open(config_path, "rb") as handle:
            config = tomllib.load(handle)
    else:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    required_sections = {
        "model", "chunking", "indexing", "detection", "api", "scheduler"
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(
            "plagiarism config is missing sections: " + ", ".join(missing))
    profiles = config["chunking"].get("profiles") or {}
    default_strategy = config["chunking"].get("default_strategy")
    if default_strategy not in profiles:
        raise ValueError(
            f"Unknown default chunking strategy: {default_strategy}")
    return config


def load_project_config(config_path=None):
    selected_path = os.path.abspath(
        config_path
        or os.getenv("PLAGIARISM_CONFIG", DEFAULT_CONFIG_PATH)
    )
    return copy.deepcopy(_load_project_config(selected_path))


def get_chunking_profile(strategy=None, config=None):
    config = config or load_project_config()
    selected = (
        strategy
        or os.getenv("PLAGIARISM_CHUNKING_STRATEGY")
        or config["chunking"]["default_strategy"]
    ).strip().lower()
    aliases = {
        "semantic_chunker": "semantic",
        "token_chunker": "token",
        "recursive": "recursive_character",
        "character": "recursive_character",
        "recursive_token": "token",
    }
    selected = aliases.get(selected, selected)
    profiles = config["chunking"]["profiles"]
    if selected not in profiles:
        raise ValueError(f"Unsupported chunking strategy: {selected}")
    return copy.deepcopy(profiles[selected])


def index_config_payload(config=None, strategy=None, extraction_version=None):
    config = config or load_project_config()
    profile = get_chunking_profile(strategy=strategy, config=config)
    model_config = config["model"]
    extraction_version = (
        extraction_version
        or config["indexing"].get("extraction_version")
    )
    if not extraction_version:
        raise ValueError("extraction_version is required for index hashing")

    chunk_tokenizer = profile.get("tokenizer")
    if not chunk_tokenizer:
        chunk_tokenizer = (
            profile.get("semantic_embedding_model")
            if profile["strategy"] == "semantic"
            else "character"
        )

    return {
        "schema_version": INDEX_CONFIG_HASH_SCHEMA_VERSION,
        "model": {
            "name": model_config["name"],
            "revision": model_config.get("revision") or "default",
            "use_fp16_on_cuda": bool(
                model_config.get("use_fp16_on_cuda", True)),
        },
        "tokenizer": {
            "embedding": model_config.get("tokenizer")
            or model_config["name"],
            "chunking": chunk_tokenizer,
        },
        "chunking": {
            key: profile[key]
            for key in INDEX_CHUNKING_KEYS
            if key in profile
        },
        "encoding": {
            "max_length": int(model_config["encode_max_length"]),
        },
        "extraction": {
            "version": str(extraction_version),
        },
    }


def canonical_json_sha256(value):
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_index_config_hash(config=None, strategy=None,
                            extraction_version=None):
    return canonical_json_sha256(index_config_payload(
        config=config,
        strategy=strategy,
        extraction_version=extraction_version,
    ))


def validate_index_config_hash(stored_hash, expected_hash, location="Index"):
    if not stored_hash:
        raise RuntimeError(
            f"{location} has no index_config_hash; compatibility is "
            "unverified; verify or rebuild manually. Existing index data "
            "was not modified."
        )
    if stored_hash != expected_hash:
        raise RuntimeError(
            f"{location} configuration mismatch: expected "
            f"{expected_hash}, found {stored_hash}; verify or rebuild "
            "manually. Existing index data was not modified."
        )
    return True


def faculty_index_paths(faculty, index_version, config=None):
    config = config or load_project_config()
    root = resolve_project_path(
        config["indexing"]["faculty_index_root"])
    faculty_root = os.path.join(root, index_version, faculty)
    return {
        "root": faculty_root,
        "vector_db": os.path.join(faculty_root, "vector_db"),
        "normalized": os.path.join(faculty_root, "normalized"),
        "marker": os.path.join(faculty_root, "corpus_index.json"),
        "report": os.path.join(faculty_root, "vector_db_run_report.txt"),
        "changes": os.path.join(faculty_root, "vector_db_changes.txt"),
    }
