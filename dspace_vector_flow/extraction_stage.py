import hashlib

from corpus_metadata import write_normalized_document
from pdf_extract import extract_university_content

from .contracts import ExtractedDocument, SourceDocument


def extract_and_store_text(document: SourceDocument, normalized_directory):
    text = extract_university_content(
        document.local_path, save_debug=False)
    if not text:
        raise ValueError("No text could be extracted from the document")

    text_path = write_normalized_document(
        normalized_directory,
        document.source_key,
        text,
        document.metadata,
    )
    content_sha256 = hashlib.sha256(
        text.encode("utf-8", errors="ignore")).hexdigest()
    return ExtractedDocument(
        source=document,
        text=text,
        text_path=text_path,
        content_sha256=content_sha256,
    )
