import copy
import json
import os
import tomllib
from functools import lru_cache


RUNTIME_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RUNTIME_ROOT)
DEFAULT_CONFIG_PATH = os.path.join(RUNTIME_ROOT, "plagiarism_config.toml")


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
