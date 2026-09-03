from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    collection_folder: str
    collection_name: str
    collection_uuid: str
    source_key: str
    local_path: str
    file_sha256: str
    metadata: dict


@dataclass(frozen=True)
class ExtractedDocument:
    source: SourceDocument
    text: str
    text_path: str
    content_sha256: str
