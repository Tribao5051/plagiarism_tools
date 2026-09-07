import argparse
import json
import os
import sys

from pdf_extract import extract_university_content
from project_config import DEFAULT_CONFIG_PATH

from .dspace_pdf_source import DSpacePdfSource


def run_pipeline(item_uuid, config_path):
    # BUOC 1: Tao ket noi DSpace tu config.
    dspace_source = DSpacePdfSource(config_path)

    # BUOC 2: Lay mot PDF tu Item UUID.
    # pdf_input.content la bytes ma pdf_extract co the doc truc tiep.
    pdf_input = dspace_source.fetch_item_pdf(item_uuid)

    # BUOC 3: Truyen bytes va filename sang module PDF extraction hien tai.
    extracted_text = extract_university_content(
        pdf_input.content,
        filename=pdf_input.filename,
        save_debug=False,
    )
    if not extracted_text:
        raise RuntimeError("pdf_extract returned empty text")

    # BUOC TIEP THEO SE DUOC THEM SAU:
    # extracted_text -> chunking -> embedding -> vector database.

    return {
        "item_uuid": pdf_input.item_uuid,
        "bitstream_uuid": pdf_input.bitstream_uuid,
        "filename": pdf_input.filename,
        "pdf_bytes": len(pdf_input.content),
        "text_characters": len(extracted_text),
        "metadata_fields": sorted(pdf_input.metadata),
        "text": extracted_text,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("item_uuid")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError):
        pass

    args = parse_args(argv)
    result = run_pipeline(args.item_uuid, os.path.abspath(args.config))

    # Khong in toan bo text ra CMD. Text van nam trong result de buoc sau su dung.
    summary = {key: value for key, value in result.items() if key != "text"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
