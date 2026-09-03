import argparse
import os
import re
import sys
import tomllib
from urllib.parse import urljoin

import requests


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_DIRECTORY, "plagiarism_config.toml")
DEFAULT_ITEM_OUTPUT_DIRECTORY = os.path.join(
    PROJECT_DIRECTORY, "dspace_item_downloads")

if PROJECT_DIRECTORY not in sys.path:
    sys.path.insert(0, PROJECT_DIRECTORY)

from pdf_extract import extract_university_content


# Đọc riêng phần [dspace] từ file cấu hình chung của project.
def load_dspace_config(config_path):
    with open(config_path, "rb") as config_file:
        project_config = tomllib.load(config_file)
    dspace_config = project_config.get("dspace")
    if not isinstance(dspace_config, dict):
        raise ValueError(
            f"Missing [dspace] section in config: {config_path}")
    return dspace_config


# Chuyển tên bitstream thành tên file hợp lệ trên Windows.
def safe_filename(value):
    filename = os.path.basename(str(value or "document.pdf"))
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).strip(" .")
    return filename or "document.pdf"


class DSpaceReader:
    # Khởi tạo HTTP session và các thông số kết nối DSpace.
    def __init__(self, config_path):
        config = load_dspace_config(config_path)
        self.base_url = config["base_url"].rstrip("/") + "/"
        self.timeout = int(config["request_timeout_seconds"])
        self.page_size = int(config["page_size"])
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/hal+json"

        token_environment = config.get(
            "bearer_token_env", "DSPACE_BEARER_TOKEN")
        bearer_token = os.getenv(token_environment)
        if bearer_token:
            self.session.headers["Authorization"] = (
                f"Bearer {bearer_token}")

    # Chuyển đường dẫn tương đối của DSpace thành URL đầy đủ.
    def absolute_url(self, path_or_url):
        if str(path_or_url).startswith(("http://", "https://")):
            return str(path_or_url)
        return urljoin(self.base_url, str(path_or_url).lstrip("/"))

    # Gửi GET request và trả về response JSON hoặc phát sinh HTTPError.
    def get_json(self, path_or_url, params=None):
        response = self.session.get(
            self.absolute_url(path_or_url),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # Đọc tuần tự tất cả trang của một HAL collection trong DSpace.
    def iter_embedded(self, path_or_url, embedded_path):
        page = 0
        while True:
            payload = self.get_json(
                path_or_url,
                params={"page": page, "size": self.page_size},
            )
            current = payload
            for key in embedded_path:
                current = current.get(key, {})
            records = current if isinstance(current, list) else []
            yield from records

            page_data = payload.get("page") or {}
            total_pages = int(page_data.get("totalPages") or 0)
            if not records or page + 1 >= total_pages:
                break
            page += 1

    # Lấy tất cả community mà tài khoản hiện tại có quyền đọc.
    def list_communities(self):
        return list(self.iter_embedded(
            "/api/core/communities",
            ("_embedded", "communities"),
        ))

    # Lấy tất cả collection trực tiếp thuộc một community UUID.
    def list_collections(self, community_uuid):
        return list(self.iter_embedded(
            f"/api/core/communities/{community_uuid}/collections",
            ("_embedded", "collections"),
        ))

    # Dùng Discovery API để duyệt các item thuộc một collection UUID.
    def iter_collection_items(self, collection_uuid):
        page = 0
        while True:
            payload = self.get_json(
                "/api/discover/search/objects",
                params={
                    "scope": collection_uuid,
                    "page": page,
                    "size": self.page_size,
                },
            )
            search_result = payload.get("_embedded", {}).get(
                "searchResult", {})
            objects = search_result.get("_embedded", {}).get("objects", [])
            for result in objects:
                item = result.get("_embedded", {}).get("indexableObject", {})
                if (
                    item.get("type") in (None, "item")
                    and (item.get("uuid") or item.get("id"))
                ):
                    yield item

            page_data = search_result.get("page") or payload.get("page") or {}
            total_pages = int(page_data.get("totalPages") or 0)
            if not objects or page + 1 >= total_pages:
                break
            page += 1

    # Ghép quan hệ Community -> Collection -> Item cho toàn repository.
    def iter_item_memberships(self):
        for community in self.list_communities():
            community_uuid = community.get("uuid") or community.get("id")
            community_name = community.get("name") or community_uuid
            for collection in self.list_collections(community_uuid):
                collection_uuid = collection.get("uuid") or collection.get("id")
                collection_name = collection.get("name") or collection_uuid
                for item in self.iter_collection_items(collection_uuid):
                    yield {
                        "community_uuid": community_uuid,
                        "community_name": community_name,
                        "collection_uuid": collection_uuid,
                        "collection_name": collection_name,
                        "item_uuid": item.get("uuid") or item.get("id"),
                        "item_name": item.get("name") or "(no name)",
                    }

    # Lấy các PDF bitstream trong ORIGINAL bundle của đúng một item UUID.
    def list_item_pdf_bitstreams(self, item_uuid):
        item = self.get_json(f"/api/core/items/{item_uuid}")
        item_uuid = str(item.get("uuid") or item_uuid)
        item_subjects = [
            str(entry.get("value") or "").strip()
            for entry in item.get("metadata", {}).get("dc.subject", [])
            if isinstance(entry, dict) and str(entry.get("value") or "").strip()
        ]

        bundles_url = (
            item.get("_links", {}).get("bundles", {}).get("href")
            or f"/api/core/items/{item_uuid}/bundles"
        )
        bundles = list(self.iter_embedded(
            bundles_url,
            ("_embedded", "bundles"),
        ))
        original_bundles = [
            bundle for bundle in bundles
            if str(bundle.get("name", "")).upper() == "ORIGINAL"
        ]

        pdf_records = []
        for original_bundle in original_bundles:
            bundle_uuid = str(original_bundle.get("uuid") or "")
            bitstreams_url = (
                original_bundle.get("_links", {})
                .get("bitstreams", {})
                .get("href")
                or f"/api/core/bundles/{bundle_uuid}/bitstreams"
            )
            bitstreams = list(self.iter_embedded(
                bitstreams_url,
                ("_embedded", "bitstreams"),
            ))
            for bitstream in bitstreams:
                bitstream_name = str(bitstream.get("name") or "")
                if not bitstream_name.lower().endswith(".pdf"):
                    continue
                bitstream_uuid = str(bitstream.get("uuid") or "")
                content_url = (
                    bitstream.get("_links", {})
                    .get("content", {})
                    .get("href")
                    or f"/api/core/bitstreams/{bitstream_uuid}/content"
                )
                checksum = bitstream.get("checkSum") or {}
                pdf_records.append({
                    "item_uuid": item_uuid,
                    "item_name": item.get("name") or "(no name)",
                    "subjects": item_subjects,
                    "bundle_uuid": bundle_uuid,
                    "bitstream_uuid": bitstream_uuid,
                    "bitstream_name": bitstream_name,
                    "size_bytes": int(bitstream.get("sizeBytes") or 0),
                    "checksum_algorithm": checksum.get("checkSumAlgorithm"),
                    "checksum_value": checksum.get("value"),
                    "content_url": content_url,
                })

        return sorted(pdf_records, key=lambda record: (
            record["bitstream_name"].casefold(),
            record["bitstream_uuid"],
        ))

    # Duyệt toàn repository và trả từng PDF bitstream mà không tải nội dung.
    def iter_all_pdf_bitstreams(self, item_limit=None):
        inspected_item_uuids = set()
        yielded_bitstream_uuids = set()

        for membership in self.iter_item_memberships():
            item_uuid = membership["item_uuid"]
            if item_uuid in inspected_item_uuids:
                continue
            if (
                item_limit is not None
                and len(inspected_item_uuids) >= item_limit
            ):
                break
            inspected_item_uuids.add(item_uuid)

            for pdf_record in self.list_item_pdf_bitstreams(item_uuid):
                bitstream_uuid = pdf_record["bitstream_uuid"]
                if bitstream_uuid in yielded_bitstream_uuids:
                    continue
                yielded_bitstream_uuids.add(bitstream_uuid)
                yield {**membership, **pdf_record}

    # Tải PDF đầu tiên của một item, extract text và ghi cả hai ra đĩa.
    def download_item_pdf_and_extract(self, item_uuid, output_directory):
        pdf_records = self.list_item_pdf_bitstreams(item_uuid)
        if not pdf_records:
            raise LookupError(
                f"Item {item_uuid} has no PDF in its ORIGINAL bundle")

        pdf_record = pdf_records[0]
        response = self.session.get(
            self.absolute_url(pdf_record["content_url"]),
            headers={"Accept": "application/pdf"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        pdf_bytes = response.content
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError(
                f"Bitstream {pdf_record['bitstream_uuid']} is not a PDF response")

        item_directory = os.path.join(
            os.path.abspath(output_directory), str(item_uuid))
        os.makedirs(item_directory, exist_ok=True)
        pdf_filename = safe_filename(pdf_record["bitstream_name"])
        pdf_path = os.path.join(item_directory, pdf_filename)
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)

        extracted_text = extract_university_content(
            pdf_bytes,
            filename=pdf_filename,
            save_debug=False,
        )
        if not extracted_text:
            raise RuntimeError(
                f"pdf_extract returned empty text for item {item_uuid}")

        text_filename = os.path.splitext(pdf_filename)[0] + ".txt"
        text_path = os.path.join(item_directory, text_filename)
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)

        return {
            "item_uuid": pdf_record["item_uuid"],
            "item_name": pdf_record["item_name"],
            "subjects": pdf_record["subjects"],
            "bundle_uuid": pdf_record["bundle_uuid"],
            "bitstream_uuid": pdf_record["bitstream_uuid"],
            "pdf_path": pdf_path,
            "pdf_bytes": len(pdf_bytes),
            "text_path": text_path,
            "text_characters": len(extracted_text),
            "pdf_bitstreams_found": len(pdf_records),
        }


# In Community, Collection và Item để kiểm tra flow Discovery API.
def print_all_items(config_path, limit=None):
    reader = DSpaceReader(config_path)
    relationship_count = 0
    unique_items = set()
    unique_collections = set()

    for membership in reader.iter_item_memberships():
        if limit is not None and relationship_count >= limit:
            break
        relationship_count += 1
        unique_items.add(membership["item_uuid"])
        unique_collections.add(membership["collection_uuid"])
        print(
            f"[{relationship_count}] "
            f"Community: {membership['community_name']} "
            f"({membership['community_uuid']}) | "
            f"Collection: {membership['collection_name']} "
            f"({membership['collection_uuid']}) | "
            f"Item: {membership['item_name']} "
            f"({membership['item_uuid']})"
        )

    print()
    print(f"Relationships printed: {relationship_count}")
    print(f"Unique collections: {len(unique_collections)}")
    print(f"Unique items: {len(unique_items)}")


# In các PDF bitstream tìm được sau khi duyệt toàn bộ flow DSpace.
def print_all_pdf_bitstreams(config_path, limit=None):
    reader = DSpaceReader(config_path)
    bitstream_count = 0
    item_uuids = set()
    collection_uuids = set()

    for bitstream in reader.iter_all_pdf_bitstreams(item_limit=limit):
        bitstream_count += 1
        item_uuids.add(bitstream["item_uuid"])
        collection_uuids.add(bitstream["collection_uuid"])
        print(
            f"[{bitstream_count}] "
            f"Collection: {bitstream['collection_name']} "
            f"({bitstream['collection_uuid']}) | "
            f"Item: {bitstream['item_name']} "
            f"({bitstream['item_uuid']}) | "
            f"PDF: {bitstream['bitstream_name']} "
            f"({bitstream['bitstream_uuid']}) | "
            f"Subjects: {bitstream['subjects']}"
        )

    print()
    print(f"PDF bitstreams printed: {bitstream_count}")
    print(f"Collections with PDFs: {len(collection_uuids)}")
    print(f"Unique items with PDFs: {len(item_uuids)}")


# Khai báo các tham số dòng lệnh của script.
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--item-uuid")
    parser.add_argument("--list-pdf-bitstreams", action="store_true")
    parser.add_argument(
        "--output-dir", default=DEFAULT_ITEM_OUTPUT_DIRECTORY)
    return parser.parse_args(argv)


# Chọn chế độ list item, list PDF bitstream hoặc tải một item.
def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    if args.item_uuid:
        reader = DSpaceReader(os.path.abspath(args.config))
        result = reader.download_item_pdf_and_extract(
            args.item_uuid,
            os.path.abspath(args.output_dir),
        )
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0
    if args.list_pdf_bitstreams:
        print_all_pdf_bitstreams(os.path.abspath(args.config), args.limit)
        return 0
    print_all_items(os.path.abspath(args.config), args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
