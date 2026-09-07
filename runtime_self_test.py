import asyncio
import hashlib
import io
import json
import os
import tempfile
import unittest
from collections import OrderedDict
from unittest.mock import patch

import fitz
import torch
from fastapi import UploadFile

import api_service
import detection_module_diff_p4 as detection_module
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
from pdf_extract import extract_university_content
from project_config import get_chunking_profile, load_project_config
from vector_db import remove_stale_sources


class FakeCollection:
    def __init__(self):
        self.deleted = []

    def delete(self, where):
        self.deleted.append(where["source"])


class FakeSemanticCollection:
    def __init__(self, document, index_version="semantic-v1"):
        self.document = document
        self.metadata = {"index_version": index_version}
        self.get_calls = []

    def count(self):
        return 1

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return {
            "embeddings": [[1.0, 0.0]],
            "documents": [self.document],
            "metadatas": [{"source": "paper.pdf"}],
        }


class RuntimeSelfTests(unittest.TestCase):
    @staticmethod
    def build_pdf_bytes(pages):
        document = fitz.open()
        for lines in pages:
            page = document.new_page()
            y_position = 72
            for line in lines:
                page.insert_text((72, y_position), line)
                y_position += 22
        payload = document.tobytes()
        document.close()
        return payload

    @staticmethod
    def scan_upload():
        return UploadFile(
            filename="submission.txt",
            file=io.BytesIO(b"submission content"),
        )

    @staticmethod
    def build_cache_test_state(root, faculty, text, collection):
        faculty_directory = os.path.join(root, faculty)
        os.makedirs(faculty_directory, exist_ok=True)
        text_path = os.path.join(faculty_directory, "paper.txt")
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return api_service.build_text_index(
            [{"path": text_path, "source": "paper.pdf"}],
            faculty,
            collection,
            object(),
        )

    def test_pdf_keeps_content_before_references_on_same_page(self):
        payload = self.build_pdf_bytes([[
            "A meaningful conclusion remains part of the submission.",
            "References",
            "Smith, A. Example source.",
        ]])

        extracted = extract_university_content(
            payload, filename="same-page.pdf", save_debug=False)

        self.assertIn("meaningful conclusion", extracted)
        self.assertNotIn("Example source", extracted)

    def test_pdf_removes_page_that_only_contains_references(self):
        payload = self.build_pdf_bytes([
            ["The main discussion must remain available."],
            ["References", "Smith, A. Example source."],
        ])

        extracted = extract_university_content(
            payload, filename="reference-page.pdf", save_debug=False)

        self.assertIn("main discussion", extracted)
        self.assertNotIn("References", extracted)
        self.assertNotIn("Example source", extracted)

    def test_pdf_does_not_cut_references_used_inside_a_sentence(self):
        sentence = "This discussion references earlier studies in the field."
        expected = (
            "The introduction must remain available. "
            f"{sentence} "
            "The conclusion must also remain available."
        )
        payload = self.build_pdf_bytes([[
            "The introduction must remain available.",
            sentence,
            "The conclusion must also remain available.",
        ]])

        extracted = extract_university_content(
            payload, filename="no-reference-heading.pdf", save_debug=False)

        self.assertEqual(extracted, expected)

    @staticmethod
    def verified_fingerprint_anchors(student_text, source_text):
        student_hashes, student_compact = (
            detection_module.get_fingerprints(student_text)
        )
        source_hashes, source_compact = (
            detection_module.get_fingerprints(source_text)
        )
        return detection_module.build_verified_fingerprint_anchors(
            student_hashes,
            student_compact,
            source_hashes,
            source_compact,
        )

    def test_fingerprints_detect_copy_after_one_to_four_prefix_chars(self):
        source = (
            "Distinct copied material remains detectable when its alignment "
            "moves by a few characters at the beginning of the submission."
        )

        for prefix_length in range(1, 5):
            student = ("x" * prefix_length) + source
            anchors, matched_positions = self.verified_fingerprint_anchors(
                student, source)
            groups = detection_module.consolidate_verbatim_anchors(anchors)
            qualifying_groups = [
                group for group in groups
                if max(item["student_end"] for item in group)
                - group[0]["student_start"]
                >= detection_module.VERBATIM_MIN_MATCH_CHARS
            ]

            self.assertTrue(matched_positions, prefix_length)
            self.assertTrue(qualifying_groups, prefix_length)
            self.assertEqual(
                min(item["student_start"] for item in anchors),
                prefix_length,
            )

    def test_fingerprints_default_to_every_window(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        fingerprints, compact_text = detection_module.get_fingerprints(text)
        positions = sorted(
            occurrence["pos"]
            for occurrences in fingerprints.values()
            for occurrence in occurrences
        )

        self.assertEqual(
            positions,
            list(range(len(compact_text) - 25 + 1)),
        )

    def test_fingerprint_highlight_maps_whitespace_to_original_offsets(self):
        source = (
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu."
        )
        student = (
            " \n\tAlpha   beta\tgamma delta  epsilon zeta eta theta "
            "iota kappa lambda mu.\n"
        )

        anchors, _matched_positions = self.verified_fingerprint_anchors(
            student, source)
        group = max(
            detection_module.consolidate_verbatim_anchors(anchors),
            key=len,
        )
        student_start = group[0]["student_start"]
        student_end = max(item["student_end"] for item in group)
        highlighted = student[student_start:student_end]
        highlighted_compact, _ = detection_module.compact_with_index_map(
            highlighted.lower())
        source_compact, _ = detection_module.compact_with_index_map(
            source.lower())

        self.assertEqual(student_start, student.index("Alpha"))
        self.assertEqual(highlighted_compact, source_compact)

    def test_fingerprints_keep_all_repeated_source_positions(self):
        passage = (
            "Repeated copied material must retain every valid source "
            "location for correct evidence highlighting."
        )
        source = f"First copy: {passage}\nBetween copies.\nSecond copy: {passage}"

        anchors, _matched_positions = self.verified_fingerprint_anchors(
            passage, source)
        first_window_sources = {
            item["source_start"]
            for item in anchors
            if item["student_pos"] == 0
        }
        groups = detection_module.consolidate_verbatim_anchors(anchors)
        qualifying_source_starts = {
            group[0]["source_start"]
            for group in groups
            if max(item["student_end"] for item in group)
            - group[0]["student_start"]
            >= detection_module.VERBATIM_MIN_MATCH_CHARS
        }
        expected_source_starts = {
            source.index(passage),
            source.rindex(passage),
        }

        self.assertEqual(first_window_sources, expected_source_starts)
        self.assertTrue(
            expected_source_starts.issubset(qualifying_source_starts)
        )

    def test_fingerprint_hash_collision_requires_content_verification(self):
        occurrence = {
            "pos": 0,
            "student_start": 0,
            "student_end": 25,
            "length": 25,
        }

        anchors, matched_positions = (
            detection_module.build_verified_fingerprint_anchors(
                {"forced-collision": [dict(occurrence)]},
                "a" * 25,
                {"forced-collision": [dict(occurrence)]},
                "b" * 25,
            )
        )

        self.assertEqual(anchors, [])
        self.assertEqual(matched_positions, set())

    @staticmethod
    def merge_test_verbatim(**overrides):
        evidence = {
            "source_file": "paper.pdf",
            "student_start": 100,
            "student_end": 150,
            "source_start": 100,
            "source_end": 150,
            "student_text": "Student context around copied material.",
            "source_text": "Source context around copied material.",
            "match_text": "copied material",
            "student_match_text": "copied material",
            "source_match_text": "copied material",
        }
        evidence.update(overrides)
        return evidence

    @staticmethod
    def merge_test_semantic(**overrides):
        evidence = {
            "source_file": "paper.pdf",
            "student_start": 90,
            "student_end": 170,
            "source_start": 90,
            "source_end": 170,
            "student_text": "Student context around copied material.",
            "source_text": "Source context around copied material.",
            "student": "semantic student span",
            "source": "semantic source span",
            "score": 0.91,
            "source_chunk_index": 2,
        }
        evidence.update(overrides)
        return evidence

    def test_merge_evidence_keeps_distant_source_spans_separate(self):
        verbatim = self.merge_test_verbatim(
            source_start=100,
            source_end=150,
        )
        semantic = self.merge_test_semantic(
            student_start=100,
            student_end=150,
            source_start=9000,
            source_end=9050,
        )

        merged = detection_module.merge_evidence(
            [verbatim], [semantic], "paper.pdf")

        self.assertEqual(
            [item["type"] for item in merged],
            ["verbatim", "semantic"],
        )

    def test_merge_evidence_combines_spans_overlapping_in_both_documents(self):
        verbatim = self.merge_test_verbatim(
            student_start=110,
            student_end=150,
            source_start=120,
            source_end=160,
        )
        semantic = self.merge_test_semantic(
            student_start=90,
            student_end=170,
            source_start=100,
            source_end=180,
        )

        merged = detection_module.merge_evidence(
            [verbatim], [semantic], "paper.pdf")

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["type"], "verbatim_semantic")

    def test_merge_evidence_keeps_different_sources_separate(self):
        verbatim = self.merge_test_verbatim(source_file="paper-a.pdf")
        semantic = self.merge_test_semantic(source_file="paper-b.pdf")

        merged = detection_module.merge_evidence(
            [verbatim], [semantic], "paper-a.pdf")

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            {item["source_file"] for item in merged},
            {"paper-a.pdf", "paper-b.pdf"},
        )

    def test_merged_match_text_matches_its_reported_offsets(self):
        student_document = "prefix STUDENT EXACT SLICE suffix"
        source_document = "header SOURCE EXACT SLICE footer"
        student_start = student_document.index("STUDENT")
        student_end = student_document.index(" suffix")
        source_start = source_document.index("SOURCE")
        source_end = source_document.index(" footer")
        verbatim = self.merge_test_verbatim(
            student_start=student_start,
            student_end=student_end,
            source_start=source_start,
            source_end=source_end,
            match_text=student_document[student_start:student_end],
            student_match_text=student_document[student_start:student_end],
            source_match_text=source_document[source_start:source_end],
        )
        semantic = self.merge_test_semantic(
            student_start=max(0, student_start - 2),
            student_end=min(len(student_document), student_end + 2),
            source_start=max(0, source_start - 2),
            source_end=min(len(source_document), source_end + 2),
            student=student_document[
                max(0, student_start - 2):min(
                    len(student_document), student_end + 2)
            ],
            source=source_document[
                max(0, source_start - 2):min(
                    len(source_document), source_end + 2)
            ],
        )

        merged = detection_module.merge_evidence(
            [verbatim], [semantic], "paper.pdf")[0]

        self.assertEqual(
            merged["student_match_text"],
            student_document[
                merged["student_start"]:merged["student_end"]
            ],
        )
        self.assertEqual(
            merged["source_match_text"],
            source_document[merged["source_start"]:merged["source_end"]],
        )

    def test_selected_config_is_semantic_batch_32(self):
        config = load_project_config()
        profile = get_chunking_profile(config=config)
        self.assertEqual(profile["strategy"], "semantic")
        self.assertEqual(profile["chunk_size"], 384)
        self.assertEqual(config["indexing"]["encode_batch_size"], 32)
        self.assertEqual(config["model"]["encode_max_length"], 512)
        self.assertFalse(config["api"]["allow_oracle_candidate_sources"])

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

    def test_oracle_candidate_sources_are_disabled_by_default(self):
        with (
            patch.dict(
                api_service.ml_state,
                {"faculty_indexes": {}, "filename_set": {"known.pdf"}},
                clear=True,
            ),
            patch.dict(
                api_service.API_CONFIG,
                {"allow_oracle_candidate_sources": False},
            ),
            patch.object(api_service, "perform_plagiarism_scan") as detector,
        ):
            response = asyncio.run(api_service.scan(
                self.scan_upload(), candidate_sources="known.pdf"))
        self.assertEqual(response.status_code, 403)
        detector.assert_not_called()

    def test_empty_oracle_candidate_sources_are_rejected(self):
        with (
            patch.dict(
                api_service.ml_state,
                {"faculty_indexes": {}, "filename_set": {"known.pdf"}},
                clear=True,
            ),
            patch.dict(
                api_service.API_CONFIG,
                {"allow_oracle_candidate_sources": True},
            ),
            patch.object(api_service, "perform_plagiarism_scan") as detector,
        ):
            response = asyncio.run(api_service.scan(
                self.scan_upload(), candidate_sources=" , "))
        self.assertEqual(response.status_code, 400)
        detector.assert_not_called()

    def test_unknown_oracle_candidate_source_is_rejected(self):
        with (
            patch.dict(
                api_service.ml_state,
                {"faculty_indexes": {}, "filename_set": {"known.pdf"}},
                clear=True,
            ),
            patch.dict(
                api_service.API_CONFIG,
                {"allow_oracle_candidate_sources": True},
            ),
            patch.object(api_service, "perform_plagiarism_scan") as detector,
        ):
            response = asyncio.run(api_service.scan(
                self.scan_upload(),
                candidate_sources="known.pdf,missing.pdf",
            ))
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"missing.pdf", response.body)
        detector.assert_not_called()

    def test_valid_oracle_source_is_forwarded_when_enabled(self):
        captured_state = {}

        def fake_detector(_content, state):
            captured_state.update(state)
            return {"results": [], "performance": {}}

        with (
            patch.dict(
                api_service.ml_state,
                {"faculty_indexes": {}, "filename_set": {"known.pdf"}},
                clear=True,
            ),
            patch.dict(
                api_service.API_CONFIG,
                {"allow_oracle_candidate_sources": True},
            ),
            patch.object(
                api_service,
                "extract_university_content",
                return_value="submission content",
            ),
            patch.object(
                api_service,
                "perform_plagiarism_scan",
                side_effect=fake_detector,
            ),
        ):
            response = asyncio.run(api_service.scan(
                self.scan_upload(), candidate_sources="known.pdf"))
        self.assertIsInstance(response, dict)
        self.assertEqual(
            captured_state["forced_candidate_sources"], ["known.pdf"])

    def test_normal_scan_does_not_force_candidate_sources(self):
        captured_state = {}

        def fake_detector(_content, state):
            captured_state.update(state)
            return {"results": [], "performance": {}}

        with (
            patch.dict(
                api_service.ml_state,
                {"faculty_indexes": {}, "filename_set": {"known.pdf"}},
                clear=True,
            ),
            patch.object(
                api_service,
                "extract_university_content",
                return_value="submission content",
            ),
            patch.object(
                api_service,
                "perform_plagiarism_scan",
                side_effect=fake_detector,
            ),
        ):
            response = asyncio.run(api_service.scan(self.scan_upload()))
        self.assertIsInstance(response, dict)
        self.assertNotIn("forced_candidate_sources", captured_state)

    def test_faculty_caches_are_isolated_when_source_names_match(self):
        with tempfile.TemporaryDirectory() as root:
            collection_a = FakeSemanticCollection("faculty A vector text")
            collection_b = FakeSemanticCollection("faculty B vector text")
            text_a = (
                "faculty A contains unique computer vision research evidence")
            text_b = (
                "faculty B contains separate network security research evidence")
            state_a = self.build_cache_test_state(
                root, "faculty-a", text_a, collection_a)
            state_b = self.build_cache_test_state(
                root, "faculty-b", text_b, collection_b)

            shared = {
                "semantic_source_cache": OrderedDict(),
                "fingerprint_cache": OrderedDict(),
            }
            scan_a = shared.copy()
            scan_a.update(state_a)
            scan_b = shared.copy()
            scan_b.update(state_b)

            _, documents_a, _, hit_a = (
                detection_module.get_cached_semantic_source(
                    scan_a, "paper.pdf"))
            _, documents_b, _, hit_b = (
                detection_module.get_cached_semantic_source(
                    scan_b, "paper.pdf"))
            fingerprints_a, _compact_a, fingerprint_hit_a = (
                detection_module.get_cached_fingerprints(
                    scan_a, "paper.pdf", text_a))
            fingerprints_b, _compact_b, fingerprint_hit_b = (
                detection_module.get_cached_fingerprints(
                    scan_b, "paper.pdf", text_b))

        self.assertEqual(documents_a, ["faculty A vector text"])
        self.assertEqual(documents_b, ["faculty B vector text"])
        self.assertFalse(hit_a)
        self.assertFalse(hit_b)
        self.assertFalse(fingerprint_hit_a)
        self.assertFalse(fingerprint_hit_b)
        self.assertNotEqual(set(fingerprints_a), set(fingerprints_b))
        self.assertEqual(len(collection_a.get_calls), 1)
        self.assertEqual(len(collection_b.get_calls), 1)
        self.assertIsNot(
            state_a["semantic_source_cache"],
            state_b["semantic_source_cache"],
        )
        self.assertIsNot(
            state_a["fingerprint_cache"],
            state_b["fingerprint_cache"],
        )

    def test_source_version_change_invalidates_both_caches(self):
        with tempfile.TemporaryDirectory() as root:
            collection = FakeSemanticCollection("old vector text")
            state = self.build_cache_test_state(
                root, "faculty-a", "old full text for fingerprints", collection)
            detection_module.get_cached_semantic_source(state, "paper.pdf")
            old_fingerprints, _old_compact, _ = (
                detection_module.get_cached_fingerprints(
                    state, "paper.pdf", "old full text for fingerprints")
            )

            new_text = "new and different full text for fingerprints"
            state["source_versions"]["paper.pdf"] = hashlib.sha256(
                new_text.encode("utf-8")).hexdigest()
            collection.document = "new vector text"
            _, documents, _, semantic_hit = (
                detection_module.get_cached_semantic_source(
                    state, "paper.pdf"))
            new_fingerprints, _new_compact, fingerprint_hit = (
                detection_module.get_cached_fingerprints(
                    state, "paper.pdf", new_text))

        self.assertEqual(documents, ["new vector text"])
        self.assertFalse(semantic_hit)
        self.assertFalse(fingerprint_hit)
        self.assertNotEqual(set(old_fingerprints), set(new_fingerprints))
        self.assertEqual(len(collection.get_calls), 2)

    def test_unchanged_source_reuses_both_faculty_caches(self):
        with tempfile.TemporaryDirectory() as root:
            text = "stable source text long enough for fingerprint caching"
            collection = FakeSemanticCollection("stable vector text")
            state = self.build_cache_test_state(
                root, "faculty-a", text, collection)

            detection_module.get_cached_semantic_source(state, "paper.pdf")
            _, documents, _, semantic_hit = (
                detection_module.get_cached_semantic_source(
                    state, "paper.pdf"))
            detection_module.get_cached_fingerprints(
                state, "paper.pdf", text)
            _, _compact, fingerprint_hit = (
                detection_module.get_cached_fingerprints(
                    state, "paper.pdf", text)
            )

        self.assertEqual(documents, ["stable vector text"])
        self.assertTrue(semantic_hit)
        self.assertTrue(fingerprint_hit)
        self.assertEqual(len(collection.get_calls), 1)

    def test_index_namespace_change_invalidates_semantic_cache(self):
        with tempfile.TemporaryDirectory() as root:
            collection = FakeSemanticCollection("version one vector")
            state = self.build_cache_test_state(
                root, "faculty-a", "unchanged full text", collection)
            detection_module.get_cached_semantic_source(state, "paper.pdf")

            state["cache_namespace"] = ("faculty-a", "semantic-v2")
            collection.document = "version two vector"
            _, documents, _, cache_hit = (
                detection_module.get_cached_semantic_source(
                    state, "paper.pdf"))

        self.assertEqual(documents, ["version two vector"])
        self.assertFalse(cache_hit)
        self.assertEqual(len(collection.get_calls), 2)

    def test_semantic_cache_query_keeps_faculty_filter(self):
        with tempfile.TemporaryDirectory() as root:
            collection = FakeSemanticCollection("faculty vector")
            state = self.build_cache_test_state(
                root, "faculty-a", "faculty full text", collection)
            state["collection_where"] = {"faculty": "faculty-a"}
            detection_module.get_cached_semantic_source(state, "paper.pdf")

        self.assertEqual(collection.get_calls[0]["where"], {
            "$and": [
                {"source": "paper.pdf"},
                {"faculty": "faculty-a"},
            ]
        })

    def test_vietnamese_faculty_key_is_stable_ascii(self):
        self.assertEqual(
            normalize_faculty("Khoa Cong nghe Thong tin"),
            "khoa-cong-nghe-thong-tin",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
