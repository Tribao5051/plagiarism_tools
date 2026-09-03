import asyncio
import hashlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from fastapi import UploadFile

import api_service
from corpus_metadata import (
    chroma_metadata_fields,
    discover_normalized_documents,
    normalize_faculty,
    write_normalized_document,
)
from corpus_scheduler import corpus_snapshot, discover_faculty_corpora
import dspace_sync
from dspace_sync import DSpaceClient, checksum_matches, resolve_faculty
from dspace_vector_pipeline import collection_corpora
from project_config import get_chunking_profile, load_project_config
from vector_db import remove_stale_sources


class FakeCollection:
    def __init__(self):
        self.deleted = []

    def delete(self, where):
        self.deleted.append(where["source"])


class RuntimeSelfTests(unittest.TestCase):
    def test_selected_config_is_semantic_batch_32(self):
        config = load_project_config()
        profile = get_chunking_profile(config=config)
        self.assertEqual(profile["strategy"], "semantic")
        self.assertEqual(profile["chunk_size"], 384)
        self.assertEqual(config["indexing"]["encode_batch_size"], 32)
        self.assertEqual(config["model"]["encode_max_length"], 512)

    def test_cuda_is_available_on_this_machine(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(
            torch.cuda.get_device_properties(0).total_memory, 3 * 1024**3)

    def test_faculty_metadata_mapping_has_priority(self):
        item = {
            "metadata": {
                "local.department": [{"value": "Computer Science"}]
            }
        }
        faculty = resolve_faculty(
            item,
            ["local.department"],
            {"Computer Science": "computer-science"},
            "wrong-scope",
        )
        self.assertEqual(faculty, "computer-science")

    def test_dspace_collection_discovery_preserves_community_path(self):
        client = DSpaceClient("https://example.invalid/server")
        top = {"uuid": "community-1", "name": "Year 2022"}
        collection = {"uuid": "collection-1", "name": "Computer Science"}
        with (
            patch.object(client, "list_communities", return_value=[top]),
            patch.object(client, "list_collections", return_value=[collection]),
        ):
            scopes = client.discover_collection_scopes()
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0]["faculty"], "computer-science")
        self.assertEqual(scopes[0]["community_path"], "Year 2022")

    def test_dspace_hash_and_collection_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "document.pdf")
            payload = b"same DSpace content"
            with open(path, "wb") as handle:
                handle.write(payload)
            checksum = "MD5:" + hashlib.md5(payload).hexdigest()
            self.assertTrue(checksum_matches(path, checksum))
            self.assertFalse(checksum_matches(path, "MD5:" + "0" * 32))

        metadata = chroma_metadata_fields({
            "source": "item/document.pdf",
            "faculty": "Computer Science",
            "dspace_collection_uuid": "collection-1",
            "dspace_collection_name": "Computer Science",
            "dspace_community_uuid": "community-1",
            "dspace_community_name": "Year 2022",
            "dspace_community_path": "Repository / Year 2022",
            "dspace_file_sha256": "a" * 64,
        })
        self.assertEqual(metadata["dspace_collection_uuid"], "collection-1")
        self.assertEqual(metadata["dspace_collection_name"], "Computer Science")
        self.assertEqual(metadata["dspace_community_name"], "Year 2022")

    def test_full_dspace_sync_hashes_new_uuid_and_removes_stale_uuid(self):
        payload = b"new bitstream content"
        checksum = "MD5:" + hashlib.md5(payload).hexdigest()

        class FakeDSpaceClient:
            def __init__(self, *args, **kwargs):
                pass

            def login(self, username, password):
                pass

            def discover_collection_scopes(self, page_size):
                return [{
                    "uuid": "collection-new",
                    "faculty": "computer-science",
                    "collection_name": "Computer Science",
                    "community_uuid": "community-2022",
                    "community_name": "Year 2022",
                    "community_path": "Year 2022",
                }]

            def iter_discovery_items(self, scope=None, page_size=100):
                yield {"uuid": "item-new"}

            def get_item(self, item_uuid):
                return {
                    "uuid": item_uuid,
                    "handle": "123/new",
                    "lastModified": "2026-08-21T00:00:00Z",
                    "metadata": {"dc.title": [{"value": "New document"}]},
                }

            def list_bundles(self, item):
                return [{"uuid": "bundle-new", "name": "ORIGINAL"}]

            def list_bitstreams(self, bundle):
                return [{
                    "uuid": "bitstream-new",
                    "name": "new.pdf",
                    "checkSum": {
                        "value": checksum.split(":", 1)[1],
                        "checkSumAlgorithm": "MD5",
                    },
                }]

            def download(self, path_or_url, destination):
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with open(destination, "wb") as handle:
                    handle.write(payload)

        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "corpus")
            old_relative = "old-faculty/old-item/old-bitstream_old.pdf"
            old_path = os.path.join(output, *old_relative.split("/"))
            os.makedirs(os.path.dirname(old_path), exist_ok=True)
            with open(old_path, "wb") as handle:
                handle.write(b"old")
            with open(
                os.path.join(output, "dspace_sync_manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump({
                    "bitstreams": {
                        "bitstream-old": {
                            "path": old_relative,
                            "collection_uuid": "collection-old",
                        }
                    }
                }, handle)
            config_path = os.path.join(root, "dspace.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "base_url": "https://example.invalid/server",
                    "output_dir": output,
                    "discover_all_collections": True,
                    "group_by_collection_name": True,
                    "allowed_extensions": [".pdf"],
                }, handle)

            with patch.object(dspace_sync, "DSpaceClient", FakeDSpaceClient):
                report = dspace_sync.sync_dspace(config_path)

            self.assertEqual(report["downloaded"], 1)
            self.assertEqual(report["removed_stale"], 1)
            self.assertFalse(os.path.exists(old_path))
            manifest_path = os.path.join(output, "dspace_sync_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            record = manifest["bitstreams"]["bitstream-new"]
            self.assertEqual(record["local_sha256"], hashlib.sha256(
                payload).hexdigest())
            self.assertEqual(record["collection_name"], "Computer Science")
            self.assertEqual(record["collection_folder"], "computer-science")
            self.assertTrue(os.path.isfile(os.path.join(
                output,
                "computer-science",
                "item-new",
                "bitstream-new_new.pdf",
            )))

    def test_normalized_text_preserves_source_and_faculty(self):
        with tempfile.TemporaryDirectory() as root:
            write_normalized_document(
                root,
                "item/paper.pdf",
                "normalized content",
                {"faculty": "Computer Science"},
            )
            documents = discover_normalized_documents(root)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["source"], "item/paper.pdf")
            self.assertEqual(documents[0]["faculty"], "computer-science")

    def test_same_named_collections_share_one_vector_corpus(self):
        with tempfile.TemporaryDirectory() as root:
            folder = "computer-science"
            os.makedirs(os.path.join(root, folder))
            corpora = collection_corpora({
                "output_dir": root,
                "collections": {
                    "collection-2021": {
                        "folder": folder,
                        "name": "Computer Science",
                        "selected_files": 10,
                    },
                    "collection-2022": {
                        "folder": folder,
                        "name": "Computer Science",
                        "selected_files": 20,
                    },
                },
            })
            self.assertEqual(len(corpora), 1)
            self.assertEqual(corpora[0]["collection_uuids"], [
                "collection-2021", "collection-2022"])

    def test_scheduler_detects_each_faculty_and_file_change(self):
        with tempfile.TemporaryDirectory() as root:
            faculty_dir = os.path.join(root, "Computer Science")
            os.makedirs(faculty_dir)
            source = os.path.join(faculty_dir, "paper.txt")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("first")
            corpora = discover_faculty_corpora(root)
            before = corpus_snapshot(faculty_dir)
            with open(source, "a", encoding="utf-8") as handle:
                handle.write(" second")
            after = corpus_snapshot(faculty_dir)
            self.assertEqual(set(corpora), {"computer-science"})
            self.assertNotEqual(before, after)

    def test_vector_index_removes_stale_source(self):
        with tempfile.TemporaryDirectory() as root:
            marker = os.path.join(root, "corpus_index.json")
            with open(marker, "w", encoding="utf-8") as handle:
                json.dump({"source_keys": ["old.pdf", "current.pdf"]}, handle)
            collection = FakeCollection()
            stale = remove_stale_sources(
                collection, ["current.pdf"], marker)
            self.assertEqual(stale, ["old.pdf"])
            self.assertEqual(collection.deleted, ["old.pdf"])

    def test_openapi_accepts_faculty_in_query_and_form(self):
        specification = api_service.app.openapi()
        operation = specification["paths"]["/scan"]["post"]
        query_names = {item["name"] for item in operation["parameters"]}
        body_schema = operation["requestBody"]["content"][
            "multipart/form-data"]["schema"]
        schema_name = body_schema["$ref"].split("/")[-1]
        form_names = set(
            specification["components"]["schemas"][schema_name][
                "properties"])
        self.assertIn("faculty", query_names)
        self.assertEqual(form_names, {"faculty", "file"})

    def test_unknown_faculty_is_rejected_before_scan(self):
        upload = UploadFile(
            filename="submission.txt",
            file=io.BytesIO(b"content"),
        )
        with (
            patch.dict(
                api_service.ml_state,
                {"faculty_indexes": {
                    "computer-science": {"filenames": []}
                }},
                clear=True,
            ),
            patch.object(api_service, "refresh_corpus_state"),
        ):
            response = asyncio.run(api_service.scan(
                upload, faculty="unknown"))
        self.assertEqual(response.status_code, 400)

    def test_vietnamese_faculty_key_is_stable_ascii(self):
        self.assertEqual(
            normalize_faculty("Khoa Cong nghe Thong tin"),
            "khoa-cong-nghe-thong-tin",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
