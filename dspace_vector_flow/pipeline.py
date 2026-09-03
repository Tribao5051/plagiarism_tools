import argparse
import json
import os
import sys

from project_config import DEFAULT_CONFIG_PATH

from .dspace_stage import load_dspace_documents
from .extraction_stage import extract_and_store_text
from .vector_stage import VectorIndexer


# DAY LA FILE BAN NEN DOC DAU TIEN.
# File nay chi mo ta thu tu du lieu di qua he thong.
# Chi tiet HTTP nam trong dspace_stage.py va dspace_sync.py.
# Chi tiet PDF extraction nam trong extraction_stage.py va pdf_extract.py.
# Chi tiet chunk, embedding va ChromaDB nam trong vector_stage.py.


def group_documents_by_collection(documents):
    # Moi collection folder co mot normalized-text folder va mot vector DB rieng.
    grouped = {}
    for document in documents:
        grouped.setdefault(document.collection_folder, []).append(document)
    return grouped


def run_pipeline(config_path, skip_dspace_sync=False, limit=None):
    # BUOC 1: Doc DSpace va dua PDF/TXT ve local.
    # Ket qua tra ve la danh sach SourceDocument. Moi object chua duong dan file,
    # Collection UUID, Item/Bitstream metadata va SHA-256 cua file goc.
    # Item khong co ORIGINAL PDF/TXT se khong the tao vector va khong nam trong list.
    documents, sync_report = load_dspace_documents(
        config_path,
        synchronize=not skip_dspace_sync,
        max_downloads=limit,
    )

    # --limit chi dung de hoc/thu nhanh. Mac dinh None nghia la xu ly tat ca file.
    if limit is not None:
        documents = documents[:limit]

    # BUOC 2: Gom file theo collection de moi collection co index rieng.
    collection_documents = group_documents_by_collection(documents)

    if not documents:
        return {
            "collections": {},
            "files_received": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "errors": [],
            "sync": sync_report,
            "success": True,
        }

    # BUOC 3: Load Chonkie, BGE-M3 va ket noi ChromaDB mot lan.
    # Model se duoc tai vao GPU neu CUDA kha dung.
    indexer = VectorIndexer(config_path)
    result = {
        "collections": {},
        "files_received": len(documents),
        "files_indexed": 0,
        "files_skipped": 0,
        "errors": [],
        "sync": sync_report,
    }

    for collection_folder, collection_files in collection_documents.items():
        source_keys = [item.source_key for item in collection_files]

        # BUOC 4: Chi xoa vector cua file da bien mat khi dang chay full corpus.
        # Khong cleanup trong che do --limit de tranh xoa nham cac vector con lai.
        if limit is None:
            indexer.remove_stale_sources(collection_folder, source_keys)

        collection_result = {
            "files": len(collection_files),
            "indexed": 0,
            "skipped": 0,
            "chunks": 0,
        }

        for document in collection_files:
            try:
                # BUOC 5: So sanh SHA-256 file va hash metadata voi ChromaDB.
                # Neu ca hash va so chunk deu dung, file khong can xu ly lai.
                if indexer.is_current(document):
                    collection_result["skipped"] += 1
                    result["files_skipped"] += 1
                    continue

                # BUOC 6: Truyen PDF/TXT local vao pdf_extract.py.
                # Text duoc tra ve va luu trong normalized/<item>/<file>.txt.
                paths = indexer.paths_for(collection_folder)
                extracted_document = extract_and_store_text(
                    document, paths["normalized"])

                # BUOC 7: Truyen text sang vector_stage.py.
                # Stage nay chunk text bang Chonkie, embed bang BGE-M3 tren GPU,
                # roi ghi chunk text + vector + offset + hash vao ChromaDB.
                chunk_count = indexer.index(extracted_document)
                collection_result["indexed"] += 1
                collection_result["chunks"] += chunk_count
                result["files_indexed"] += 1
            except Exception as exc:
                result["errors"].append({
                    "source": document.source_key,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        # BUOC 8: Sau khi collection thanh cong, cap nhat corpus_index.json.
        # Marker nay giup lan sau biet file nao con ton tai hoac da bi xoa.
        collection_failed = any(
            error["source"] in source_keys for error in result["errors"])
        if limit is None and not collection_failed:
            indexer.finish_collection(collection_folder, source_keys)
        result["collections"][collection_folder] = collection_result

    # BUOC 9: Tra summary cho CMD hoac script ben ngoai.
    # Du lieu chi duoc xem la thanh cong khi khong co file nao loi.
    result["success"] = not result["errors"]
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--skip-dspace-sync", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    result = run_pipeline(
        os.path.abspath(args.config),
        skip_dspace_sync=args.skip_dspace_sync,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
