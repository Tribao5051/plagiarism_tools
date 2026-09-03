import copy
import math
import unittest

from evidence_reranker import (
    FEATURE_NAMES,
    agreement_score,
    build_candidate_records,
    select_ranked_records,
)


def evidence(student_start, student_end, source_start, source_end,
             source_file="source-a.txt", semantic_score=0.9):
    return {
        "type": "semantic",
        "source_file": source_file,
        "student_start": student_start,
        "student_end": student_end,
        "source_start": source_start,
        "source_end": source_end,
        "student_match_text": "a shared plagiarism detection passage",
        "source_match_text": "a shared plagiarism detection passage",
        "semantic_score": semantic_score,
        "lexical_score": 0.8,
        "confidence_score": semantic_score,
        "match_methods": ["dense_embedding", "lexical_window"],
        "evidence_count": 2,
    }


def result(*items):
    return {
        "filename": "source-a.txt",
        "semantic_coverage": 20.0,
        "verbatim_score": 0.0,
        "final_score": 15.0,
        "consolidated_evidence": list(items),
    }


class EvidenceRerankerTests(unittest.TestCase):
    def test_features_are_complete_finite_and_do_not_mutate_input(self):
        results = [result(evidence(100, 300, 1000, 1200))]
        original = copy.deepcopy(results)

        records = build_candidate_records(results)

        self.assertEqual(results, original)
        self.assertEqual(len(records[0]["features"]), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value)
                            for value in records[0]["features"]))
        self.assertGreaterEqual(agreement_score(records[0]), 0.0)
        self.assertLessEqual(agreement_score(records[0]), 1.0)

    def test_higher_score_wins_for_overlapping_same_source_evidence(self):
        records = build_candidate_records([result(
            evidence(100, 300, 1000, 1200),
            evidence(120, 290, 1020, 1190),
        )])

        selected = select_ranked_records(
            records, [0.7, 0.9], threshold=0.5,
            same_source_gap=0,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["record_id"], "0:1")
        self.assertEqual(selected[0]["reranker_score"], 0.9)

    def test_separate_regions_from_same_source_are_preserved(self):
        records = build_candidate_records([result(
            evidence(100, 300, 1000, 1200),
            evidence(700, 900, 3000, 3200),
        )])

        selected = select_ranked_records(
            records, [0.8, 0.7], threshold=0.5,
        )

        self.assertEqual(len(selected), 2)

    def test_source_cap_is_applied_after_ranking(self):
        records = build_candidate_records([result(
            evidence(100, 200, 1000, 1100),
            evidence(700, 800, 3000, 3100),
        )])

        selected = select_ranked_records(
            records, [0.6, 0.9], threshold=0.5, max_per_source=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["record_id"], "0:1")

    def test_threshold_rejects_low_scoring_candidates(self):
        records = build_candidate_records([
            result(evidence(100, 300, 1000, 1200))
        ])

        selected = select_ranked_records(
            records, [0.49], threshold=0.5,
        )

        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
