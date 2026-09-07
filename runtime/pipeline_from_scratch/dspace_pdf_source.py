import os
from dataclasses import dataclass
from urllib.parse import urljoin

import requests

from project_config import load_project_config


@dataclass(frozen=True)
class PdfInput:
    content: bytes
    filename: str
    item_uuid: str
    bitstream_uuid: str
    metadata: dict


class DSpacePdfSource:
    def __init__(self, config_path):
        # Doc rieng nhom [dspace] tu plagiarism_config.toml.
        project_config = load_project_config(config_path)
        dspace_config = project_config["dspace"]

        self.base_url = dspace_config["base_url"].rstrip("/") + "/"
        self.timeout = int(dspace_config["request_timeout_seconds"])
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/hal+json"

        # Neu co token trong environment thi gui token cho moi request.
        token_environment = dspace_config.get(
            "bearer_token_env", "DSPACE_BEARER_TOKEN")
        bearer_token = os.getenv(token_environment)
        if bearer_token:
            self.session.headers["Authorization"] = (
                f"Bearer {bearer_token}")

    def absolute_url(self, path_or_url):
        # DSpace co the tra link day du hoac chi tra path bat dau bang /api/.
        if str(path_or_url).startswith(("http://")):
            return str(path_or_url)
        return urljoin(self.base_url, str(path_or_url).lstrip("/"))

    def get_json(self, path_or_url, params=None):
        # Tat ca request JSON di qua mot function de xu ly timeout va HTTP error.
        response = self.session.get(
            self.absolute_url(path_or_url),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_item(self, item_uuid):
        # Lay item cu the de co full metadata va link den bundles.
        return self.get_json(f"/api/core/items/{item_uuid}")

    def get_original_bundle(self, item):
        # Mot item co the co ORIGINAL, THUMBNAIL, LICENSE va nhieu bundle khac.
        # File luan van that nam trong bundle ORIGINAL.
        bundles_url = (
            item.get("_links", {}).get("bundles", {}).get("href")
            or f"/api/core/items/{item['uuid']}/bundles"
        )
        payload = self.get_json(bundles_url, params={"size": 100})
        bundles = payload.get("_embedded", {}).get("bundles", [])
        for bundle in bundles:
            if str(bundle.get("name", "")).upper() == "ORIGINAL":
                return bundle
        raise LookupError(f"Item {item['uuid']} has no ORIGINAL bundle")

    def get_pdf_bitstream(self, bundle):
        # ORIGINAL co the chua PDF, TXT hoac file phu. Buoc dau chi lay mot PDF.
        bitstreams_url = (
            bundle.get("_links", {}).get("bitstreams", {}).get("href")
            or f"/api/core/bundles/{bundle['uuid']}/bitstreams"
        )
        payload = self.get_json(bitstreams_url, params={"size": 100})
        bitstreams = payload.get("_embedded", {}).get("bitstreams", [])
        pdf_bitstreams = [
            bitstream for bitstream in bitstreams
            if str(bitstream.get("name", "")).lower().endswith(".pdf")
        ]
        if not pdf_bitstreams:
            raise LookupError(
                f"ORIGINAL bundle {bundle['uuid']} has no PDF bitstream")
        return sorted(
            pdf_bitstreams,
            key=lambda item: (
                str(item.get("name", "")).casefold(),
                str(item.get("uuid", "")),
            ),
        )[0]

    def download_pdf_bytes(self, bitstream):
        # Tai PDF vao RAM. pdf_extract co the doc truc tiep kieu bytes nay.
        content_url = (
            bitstream.get("_links", {}).get("content", {}).get("href")
            or f"/api/core/bitstreams/{bitstream['uuid']}/content"
        )
        response = self.session.get(
            self.absolute_url(content_url),
            timeout=self.timeout,
            headers={"Accept": "application/pdf"},
        )
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            raise ValueError(
                f"Bitstream {bitstream['uuid']} is not a valid PDF response")
        return content

    def fetch_item_pdf(self, item_uuid):
        # Function public duy nhat ma main pipeline can goi.
        item = self.get_item(item_uuid)
        original_bundle = self.get_original_bundle(item)
        pdf_bitstream = self.get_pdf_bitstream(original_bundle)
        pdf_bytes = self.download_pdf_bytes(pdf_bitstream)

        return PdfInput(
            content=pdf_bytes,
            filename=str(pdf_bitstream.get("name") or "document.pdf"),
            item_uuid=str(item.get("uuid") or item_uuid),
            bitstream_uuid=str(pdf_bitstream.get("uuid") or ""),
            metadata=item.get("metadata") or {},
        )
