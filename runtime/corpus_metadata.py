import hashlib
import json
import os
import re
import unicodedata
import uuid


SUPPORTED_CORPUS_EXTENSIONS = {".pdf", ".txt"}
METADATA_SUFFIX = ".metadata.json"
DEFAULT_FACULTY = "default"


def normalize_faculty(value):
    """Return a stable ASCII faculty key suitable for paths and API values."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return slug or DEFAULT_FACULTY


def normalize_source_key(value):
    return str(value).replace("\\", "/").lstrip("./")


def metadata_sidecar_path(file_path):
    return str(file_path) + METADATA_SUFFIX


def load_json_file(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {} if default is None else default


def write_json_file(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = str(path) + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def file_sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def discover_corpus_files(root_dir):
    files = []
    for current_root, directories, filenames in os.walk(root_dir):
        directories[:] = sorted(
            directory
            for directory in directories
            if not directory.startswith(".")
        )
        for filename in sorted(filenames):
            extension = os.path.splitext(filename)[1].lower()
            if extension not in SUPPORTED_CORPUS_EXTENSIONS:
                continue
            absolute_path = os.path.join(current_root, filename)
            relative_path = os.path.relpath(absolute_path, root_dir)
            files.append(normalize_source_key(relative_path))
    return sorted(files)


def infer_faculty(source_key, metadata=None):
    metadata = metadata or {}
    explicit = metadata.get("faculty") or metadata.get("department")
    if explicit:
        return normalize_faculty(explicit)
    parts = normalize_source_key(source_key).split("/")
    if len(parts) > 1:
        return normalize_faculty(parts[0])
    return DEFAULT_FACULTY


def load_source_metadata(root_dir, source_key, default_faculty=None):
    source_key = normalize_source_key(source_key)
    source_path = os.path.join(root_dir, *source_key.split("/"))
    metadata = load_json_file(metadata_sidecar_path(source_path), default={})
    result = dict(metadata) if isinstance(metadata, dict) else {}
    result["source"] = source_key
    if not (result.get("faculty") or result.get("department")):
        result["faculty"] = normalize_faculty(
            default_faculty) if default_faculty else infer_faculty(
                source_key, result)
    else:
        result["faculty"] = infer_faculty(source_key, result)
    return result


def normalized_text_path(normalized_root, source_key):
    source_key = normalize_source_key(source_key)
    return os.path.join(normalized_root, *source_key.split("/")) + ".txt"


def managed_normalized_document_paths(normalized_root, source_key):
    raw_source = str(source_key).replace("\\", "/")
    drive, _tail = os.path.splitdrive(raw_source)
    parts = raw_source.split("/")
    if (
        not raw_source
        or drive
        or os.path.isabs(raw_source)
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"Unsafe normalized source path: {source_key}")

    managed_root = os.path.realpath(os.path.abspath(normalized_root))
    text_path = os.path.realpath(
        os.path.abspath(os.path.join(managed_root, *parts) + ".txt"))
    if os.path.commonpath([managed_root, text_path]) != managed_root:
        raise ValueError(
            f"Normalized source is outside managed root: {source_key}")
    return text_path, metadata_sidecar_path(text_path)


def remove_normalized_document(normalized_root, source_key):
    text_path, sidecar_path = managed_normalized_document_paths(
        normalized_root, source_key)
    removed = []
    for path in (text_path, sidecar_path):
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)

    managed_root = os.path.realpath(os.path.abspath(normalized_root))
    current = os.path.dirname(text_path)
    while current != managed_root:
        try:
            os.rmdir(current)
        except OSError:
            break
        current = os.path.dirname(current)
    return removed


def stage_normalized_document_removal(normalized_root, source_key):
    text_path, sidecar_path = managed_normalized_document_paths(
        normalized_root, source_key)
    token = uuid.uuid4().hex
    staged_paths = {}
    staged = {
        "managed_root": os.path.realpath(os.path.abspath(normalized_root)),
        "text_path": text_path,
        "staged_paths": staged_paths,
    }
    try:
        for target in (text_path, sidecar_path):
            if not os.path.isfile(target):
                continue
            staged_path = f"{target}.delete-{token}"
            os.replace(target, staged_path)
            staged_paths[target] = staged_path
    except Exception:
        rollback_normalized_document_removal(staged)
        raise
    return staged


def rollback_normalized_document_removal(staged):
    errors = []
    for target, staged_path in reversed(list(
            staged.get("staged_paths", {}).items())):
        if not os.path.isfile(staged_path):
            continue
        try:
            if os.path.exists(target):
                raise RuntimeError(
                    f"Cannot restore normalized artifact over {target}")
            os.replace(staged_path, target)
        except Exception as exc:
            errors.append(f"{target}: {exc}")
    if errors:
        raise RuntimeError(
            "Could not restore normalized artifacts: " + "; ".join(errors))


def finalize_normalized_document_removal(staged):
    cleanup_errors = []
    for staged_path in staged.get("staged_paths", {}).values():
        if not os.path.isfile(staged_path):
            continue
        try:
            os.remove(staged_path)
        except OSError as exc:
            cleanup_errors.append(f"{staged_path}: {exc}")

    managed_root = staged.get("managed_root")
    current = os.path.dirname(staged.get("text_path", ""))
    while managed_root and current and current != managed_root:
        try:
            os.rmdir(current)
        except OSError:
            break
        current = os.path.dirname(current)
    return cleanup_errors


def stage_normalized_document(normalized_root, source_key, text, metadata):
    text_path, sidecar_path = managed_normalized_document_paths(
        normalized_root, source_key)
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    token = uuid.uuid4().hex
    staged_text_path = f"{text_path}.stage-{token}"
    staged_sidecar_path = f"{sidecar_path}.stage-{token}"
    normalized_metadata = dict(metadata or {})
    normalized_metadata["source"] = normalize_source_key(source_key)
    normalized_metadata["faculty"] = infer_faculty(
        source_key, normalized_metadata)
    staged = {
        "text_path": text_path,
        "sidecar_path": sidecar_path,
        "staged_text_path": staged_text_path,
        "staged_sidecar_path": staged_sidecar_path,
        "token": token,
    }
    try:
        with open(staged_text_path, "x", encoding="utf-8") as handle:
            handle.write(text)
        write_json_file(staged_sidecar_path, normalized_metadata)
    except Exception:
        discard_staged_normalized_document(staged)
        raise
    return staged


def discard_staged_normalized_document(staged):
    for key in ("staged_text_path", "staged_sidecar_path"):
        path = staged.get(key)
        if path and os.path.isfile(path):
            os.remove(path)


def publish_staged_normalized_document(staged):
    text_path = staged["text_path"]
    sidecar_path = staged["sidecar_path"]
    staged_text_path = staged["staged_text_path"]
    staged_sidecar_path = staged["staged_sidecar_path"]
    token = staged["token"]
    backups = {
        text_path: f"{text_path}.backup-{token}",
        sidecar_path: f"{sidecar_path}.backup-{token}",
    }
    moved_targets = set()
    try:
        for target, backup in backups.items():
            if os.path.isfile(target):
                os.replace(target, backup)
        os.replace(staged_text_path, text_path)
        moved_targets.add(text_path)
        os.replace(staged_sidecar_path, sidecar_path)
        moved_targets.add(sidecar_path)
    except Exception:
        for target in moved_targets:
            if os.path.isfile(target):
                os.remove(target)
        for target, backup in backups.items():
            if os.path.isfile(backup):
                os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            if os.path.isfile(backup):
                try:
                    os.remove(backup)
                except OSError:
                    # The new pair is already committed. A backup cleanup
                    # failure must not make the vector layer roll back alone.
                    pass
        return text_path
    finally:
        discard_staged_normalized_document(staged)


def write_normalized_document(normalized_root, source_key, text, metadata):
    output_path = normalized_text_path(normalized_root, source_key)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    normalized_metadata = dict(metadata or {})
    normalized_metadata["source"] = normalize_source_key(source_key)
    normalized_metadata["faculty"] = infer_faculty(
        source_key, normalized_metadata)
    write_json_file(metadata_sidecar_path(output_path), normalized_metadata)
    return output_path


def discover_normalized_documents(normalized_root):
    documents = []
    if not os.path.isdir(normalized_root):
        return documents

    for current_root, directories, filenames in os.walk(normalized_root):
        directories[:] = sorted(
            directory
            for directory in directories
            if not directory.startswith(".")
        )
        for filename in sorted(filenames):
            if not filename.lower().endswith(".txt"):
                continue
            path = os.path.join(current_root, filename)
            relative_path = normalize_source_key(
                os.path.relpath(path, normalized_root))
            metadata = load_json_file(metadata_sidecar_path(path), default={})
            if not isinstance(metadata, dict):
                metadata = {}
            source = normalize_source_key(
                metadata.get("source") or relative_path)
            faculty = infer_faculty(source, metadata)
            documents.append({
                "path": path,
                "source": source,
                "faculty": faculty,
                "metadata": metadata,
            })
    documents.sort(key=lambda item: (item["faculty"], item["source"]))
    return documents


def chroma_metadata_fields(metadata):
    """Keep only scalar provenance fields accepted by Chroma metadata."""
    selected = {
        "faculty": infer_faculty(metadata.get("source", ""), metadata),
    }
    field_names = (
        "title",
        "dspace_item_uuid",
        "dspace_bitstream_uuid",
        "dspace_handle",
        "dspace_collection_uuid",
        "dspace_collection_name",
        "dspace_collection_folder",
        "dspace_community_uuid",
        "dspace_community_name",
        "dspace_community_path",
        "dspace_last_modified",
        "dspace_checksum",
        "dspace_file_sha256",
    )
    for field_name in field_names:
        value = metadata.get(field_name)
        if isinstance(value, (str, int, float, bool)) and value != "":
            selected[field_name] = value
    return selected


def metadata_sha256(metadata):
    payload = json.dumps(
        chroma_metadata_fields(metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
