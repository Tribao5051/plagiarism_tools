import os

from corpus_metadata import (
    METADATA_SUFFIX,
    file_sha256,
    load_json_file,
    normalize_faculty,
    normalize_source_key,
)
from dspace_sync import (
    DEFAULT_MANIFEST_NAME,
    load_config,
    resolve_output_dir,
    sync_dspace,
)

from .contracts import SourceDocument


def load_dspace_documents(config_path, synchronize=True, max_downloads=None):
    dspace_config = load_config(config_path)
    output_directory = resolve_output_dir(dspace_config, config_path)
    sync_report = None

    if synchronize:
        sync_report = sync_dspace(
            config_path=config_path,
            max_downloads=max_downloads,
        )
        if sync_report["errors"]:
            raise RuntimeError(
                f"DSpace sync failed with {len(sync_report['errors'])} error(s)")

    manifest_path = os.path.join(
        output_directory, DEFAULT_MANIFEST_NAME)
    manifest = load_json_file(manifest_path, default={})
    documents = []
    output_root = os.path.abspath(output_directory)

    for record in manifest.get("bitstreams", {}).values():
        relative_path = normalize_source_key(record.get("path") or "")
        if not relative_path:
            continue
        local_path = os.path.abspath(os.path.join(
            output_root, *relative_path.split("/")))
        if (
            os.path.commonpath([output_root, local_path]) != output_root
            or not os.path.isfile(local_path)
        ):
            continue

        metadata = load_json_file(
            local_path + METADATA_SUFFIX, default={})
        path_parts = relative_path.split("/")
        collection_folder = normalize_faculty(
            record.get("collection_folder")
            or metadata.get("dspace_collection_folder")
            or path_parts[0]
        )
        collection_root = os.path.join(output_root, collection_folder)
        source_key = normalize_source_key(
            os.path.relpath(local_path, collection_root))
        documents.append(SourceDocument(
            collection_folder=collection_folder,
            collection_name=str(
                record.get("collection_name")
                or metadata.get("dspace_collection_name")
                or collection_folder
            ),
            collection_uuid=str(
                record.get("collection_uuid")
                or metadata.get("dspace_collection_uuid")
                or ""
            ),
            source_key=source_key,
            local_path=local_path,
            file_sha256=(
                record.get("local_sha256") or file_sha256(local_path)
            ),
            metadata=metadata,
        ))

    documents.sort(key=lambda item: (
        item.collection_folder, item.source_key))
    return documents, sync_report
