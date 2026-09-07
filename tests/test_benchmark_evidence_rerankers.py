import pathlib
import tempfile
import unittest

from tests import _project_paths

import benchmark_evidence_rerankers as benchmark


def normalized(text):
    return " ".join((text or "").lower().split())


def text_profile(text):
    value = normalized(text)
    return value, set(value.split())


def containment(candidate_profile, reference_profile):
    candidate = candidate_profile[0]
    reference = reference_profile[0]
    return 1.0 if candidate and candidate == reference else 0.0


def token_containment(candidate, reference):
    return 1.0 if normalized(candidate) == normalized(reference) else 0.0


FAKE_EVALUATOR = {
    "threshold": 0.5,
    "text_profile": text_profile,
    "containment": containment,
    "token_containment": token_containment,
}


def truth():
    return {
        "source_file": "correct-source.txt",
        "student_text": "the copied submission passage",
        "source_text": "the matching ground truth source passage",
        "obfuscation": "none",
    }


def record(source_file="correct-source.txt", student_text=None,
           source_text=None):
    return {
        "record_id": "record-1",
        "source_file": source_file,
        "evidence": {
            "student_match_text": (
                student_text or "the copied submission passage"),
            "source_match_text": (
                source_text or "the matching ground truth source passage"),
            "confidence_score": 0.9,
            "evidence_count": 1,
        },
    }


class TruthAnnotationTests(unittest.TestCase):
    def annotate(self, item):
        benchmark.annotate_truth_matches(item, [truth()], FAKE_EVALUATOR)
        benchmark.assign_best_candidate_labels(item)
        return item[0]

    def test_correct_submission_with_wrong_source_file_is_negative(self):
        item = self.annotate([record(source_file="wrong-source.txt")])

        self.assertEqual(item["truth_scores"], {})
        self.assertEqual(item["label"], 0)

    def test_correct_submission_with_wrong_source_passage_is_negative(self):
        item = self.annotate([record(
            source_text="an unrelated passage from the correct file")])

        self.assertEqual(item["truth_scores"], {})
        self.assertEqual(item["label"], 0)

    def test_matching_submission_and_source_passages_are_positive(self):
        item = self.annotate([record()])

        self.assertIn(0, item["truth_scores"])
        self.assertEqual(item["label"], 1)


class BenchmarkMetricTests(unittest.TestCase):
    def test_prediction_metrics_require_both_text_sides(self):
        predictions = [{
            "prediction_id": "prediction-1",
            "source_file": "correct-source.txt",
            "student_match_text": "the copied submission passage",
            "source_match_text": "an unrelated source passage",
        }]

        metrics = benchmark.evidence_hit_metrics_from_predictions(
            [truth()], predictions, FAKE_EVALUATOR)

        self.assertEqual(metrics["evidence_true_positive"], 0)
        self.assertEqual(metrics["evidence_hit_precision"], 0.0)

    def test_metrics_name_evidence_hits_and_do_not_fake_character_precision(self):
        selected_record = {
            "record_id": "record-1",
            "source_file": "correct-source.txt",
            "evidence": {
                "student_start": 10,
                "student_end": 50,
                "source_start": 100,
                "source_end": 140,
                "evidence_count": 1,
            },
            "truth_scores": {0: 1.0},
        }
        cases = {
            "case-1": {
                "records": [selected_record],
                "truths": [truth()],
            }
        }

        metrics = benchmark.fast_metrics(
            cases,
            ["case-1"],
            {"case-1": [0.9]},
            {"threshold": 0.5},
        )

        self.assertEqual(metrics["evidence_hit_precision"], 1.0)
        self.assertIsNone(metrics["character_precision"])
        self.assertIn("raw", metrics["character_precision_note"].lower())

    def test_split_validation_rejects_test_leakage(self):
        split = {
            "train": ["case-1"],
            "validation": ["case-2"],
            "test": ["case-1"],
        }

        with self.assertRaisesRegex(ValueError, "overlap"):
            benchmark.validate_experiment_split(split)

    def test_report_distinguishes_evidence_and_character_precision(self):
        metrics = {
            **benchmark.precision_metric_fields(1.0),
            "recall": 1.0,
            "f1": 1.0,
            "evidence_count": 1,
        }
        payload = {
            "split": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            "dataset": {"candidate_count": 1, "positive_candidates": 1},
            "strategies": {
                "rules": {
                    "balanced": {
                        "development_cv": metrics,
                        "test": metrics,
                    },
                    "high_precision": {
                        "development_cv": metrics,
                        "test": metrics,
                    },
                }
            },
            "recommendation": {
                "strategy": "rules",
                "development_cv": metrics,
                "test": metrics,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            report_path = pathlib.Path(directory) / "report.md"
            benchmark.write_report(payload, report_path)
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("Evidence-hit precision", report)
        self.assertIn("Character precision: not computed", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
