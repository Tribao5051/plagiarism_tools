import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests import _project_paths

import run_oracle_noise as oracle


def truth(source_file="correct-source.txt", suffix=""):
    return {
        "source_file": source_file,
        "student_text": "copied submission passage " + suffix,
        "source_text": "matching source passage " + suffix,
        "obfuscation": "none",
        "translation": False,
    }


def evidence(source_file="correct-source.txt", student_suffix="",
             source_suffix="", evidence_type="semantic"):
    return {
        "source_file": source_file,
        "type": evidence_type,
        "student_match_text": "copied submission passage " + student_suffix,
        "source_match_text": "matching source passage " + source_suffix,
    }


class OfflineScoringTests(unittest.TestCase):
    def setUp(self):
        self.scoring = oracle.load_scoring()

    def test_scorer_is_local_and_does_not_need_external_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_project = pathlib.Path(directory) / "missing-project"
            with mock.patch.object(oracle, "PROJECT", missing_project):
                scoring = oracle.load_scoring()

        self.assertEqual(
            pathlib.Path(scoring.__file__).resolve().parent,
            pathlib.Path(oracle.__file__).resolve().parent,
        )

    def test_two_sided_hit_requires_source_file_and_source_passage(self):
        truths = [truth()]
        predictions = [
            evidence(),
            evidence(source_file="wrong-source.txt"),
            dict(evidence(), source_match_text=(
                "unrelated words from another section")),
        ]

        records = self.scoring.records_for_predictions(predictions, truths)

        self.assertEqual(
            [record["truth_ids"] for record in records], [[0], [], []])

    def test_duplicate_evidence_hits_do_not_duplicate_truth_recall(self):
        records = [{"truth_ids": [0]}, {"truth_ids": [0]}]

        score = self.scoring.evidence_metrics(records, truth_count=2)

        self.assertEqual(score["true_positive_evidence"], 2)
        self.assertEqual(score["evidence_hit_precision"], 1.0)
        self.assertEqual(score["caught_truth_segments"], 1)
        self.assertEqual(score["segment_recall"], 0.5)

    def test_empty_truth_has_na_recall_and_negative_evidence_warning(self):
        score = self.scoring.evidence_metrics(
            [{"truth_ids": []}], truth_count=0)

        self.assertIsNone(score["segment_recall"])
        self.assertTrue(score["negative_file_has_evidence"])
        self.assertEqual(score["false_positive_evidence"], 1)
        self.assertIsNone(score["character_precision"])

    def test_response_reports_same_metric_schema_for_stages_and_families(self):
        response = {
            "results": [{
                "filename": "correct-source.txt",
                "raw_evidence": [evidence(evidence_type="semantic")],
                "consolidated_evidence": [
                    evidence(evidence_type="verbatim_semantic")],
                "evidence": [evidence(evidence_type="verbatim_semantic")],
            }],
        }

        evaluation, display = self.scoring.score_response(
            response, [truth()], ["correct-source.txt"])

        self.assertEqual(
            set(evaluation["stages"]),
            {"raw_evidence", "consolidated_evidence", "evidence"},
        )
        self.assertEqual(
            set(evaluation["families"]),
            {"semantic", "verbatim", "mixed"},
        )
        for metric in [
                *evaluation["stages"].values(),
                *evaluation["families"].values()]:
            self.assertIn("evidence_hit_precision", metric)
            self.assertIn("segment_recall", metric)
            self.assertIsNone(metric["character_precision"])
        self.assertEqual(display[0]["truth_ids"], [0])

    def test_scoring_metadata_contains_version_and_code_hash(self):
        metadata = self.scoring.scoring_metadata()

        self.assertRegex(metadata["scoring_schema_version"], r"^two_sided_")
        self.assertRegex(metadata["scoring_code_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNone(metadata["character_precision"])

    def test_offline_scorer_does_not_import_model_stack(self):
        performance_dir = pathlib.Path(oracle.__file__).resolve().parent
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(performance_dir)!r}); "
            "import run_oracle_noise as oracle; "
            "oracle.load_scoring(); "
            "blocked=('torch','transformers','FlagEmbedding','chromadb'); "
            "print(','.join(name for name in blocked if name in sys.modules))"
        )

        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
