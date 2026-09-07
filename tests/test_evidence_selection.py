import copy
import unittest

from tests import _project_paths

from detection_module_diff_p4 import select_display_evidence


def semantic_evidence(
        student_start,
        student_end,
        source_start,
        source_end,
        score=0.92,
        source_file="source-a.txt"):
    return {
        "type": "semantic",
        "source_file": source_file,
        "student_start": student_start,
        "student_end": student_end,
        "source_start": source_start,
        "source_end": source_end,
        "student": "student evidence",
        "source": "source evidence",
        "student_match_text": "student evidence",
        "source_match_text": "source evidence",
        "semantic_score": score,
        "match_method": "dense_embedding",
        "match_methods": ["dense_embedding"],
        "evidence_count": 2,
    }


def result_with(*evidence):
    return {
        "filename": "source-a.txt",
        "semantic_coverage": 20.0,
        "verbatim_score": 0.0,
        "consolidated_evidence": list(evidence),
    }


class EvidenceSelectionTests(unittest.TestCase):
    def select(self, evidence, profile="balanced"):
        results = [result_with(*copy.deepcopy(evidence))]
        summary = select_display_evidence(results, profile=profile)
        return results[0]["evidence"], summary

    def test_overlapping_student_and_source_spans_are_deduplicated(self):
        evidence = [
            semantic_evidence(100, 300, 1000, 1200, score=0.94),
            semantic_evidence(120, 290, 1020, 1190, score=0.91),
        ]

        selected, summary = self.select(evidence)

        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["rejections"]["same_source_duplicate"], 1)
        self.assertEqual(
            summary["hidden_count"], sum(summary["rejections"].values()))

    def test_separate_student_regions_from_same_source_are_preserved(self):
        evidence = [
            semantic_evidence(100, 300, 1000, 1200),
            semantic_evidence(700, 900, 3000, 3200, score=0.91),
        ]

        selected, summary = self.select(evidence)

        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["rejections"]["same_source_duplicate"], 0)

    def test_high_precision_keeps_one_source_per_student_region(self):
        evidence = [
            semantic_evidence(100, 300, 1000, 1200, source_file="a.txt"),
            semantic_evidence(
                110, 290, 4000, 4180, score=0.91, source_file="b.txt"),
        ]

        selected, summary = self.select(evidence, profile="high_precision")

        self.assertEqual(len(selected), 1)
        self.assertEqual(
            summary["rejections"]["cross_source_competition"], 1)

    def test_all_profile_preserves_every_candidate(self):
        evidence = [
            semantic_evidence(100, 300, 1000, 1200),
            semantic_evidence(120, 290, 1020, 1190, score=0.91),
        ]

        selected, summary = self.select(evidence, profile="all")

        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["hidden_count"], 0)

    def test_agreement_rejects_weak_semantic_only_evidence(self):
        weak = semantic_evidence(100, 300, 1000, 1200, score=0.73)
        weak["student_match_text"] = "unrelated student passage"
        weak["source_match_text"] = "different source passage"
        weak.pop("lexical_score", None)
        weak["evidence_count"] = 1
        results = [result_with(weak)]

        summary = select_display_evidence(results, profile="agreement")

        self.assertEqual(results[0]["evidence"], [])
        self.assertEqual(summary["rejections"]["agreement_score"], 1)

    def test_agreement_keeps_strong_multi_signal_evidence(self):
        strong = semantic_evidence(100, 300, 1000, 1200, score=0.91)
        strong.update({
            "type": "verbatim_semantic",
            "lexical_score": 0.88,
            "verbatim_characters": 240,
            "match_methods": [
                "dense_embedding",
                "lexical_window",
                "verbatim_fingerprint",
            ],
        })
        results = [result_with(strong)]

        summary = select_display_evidence(results, profile="agreement")

        self.assertEqual(len(results[0]["evidence"]), 1)
        shown = results[0]["evidence"][0]
        self.assertGreaterEqual(shown["agreement_score"], 0.40)
        self.assertEqual(summary["shown_count"], 1)


if __name__ == "__main__":
    unittest.main()
