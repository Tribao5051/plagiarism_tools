import copy
import unittest
from collections import Counter
from unittest import mock

from tests import _project_paths

import detection_module_diff_p4 as detector


def strong_evidence(student_start, source_start, source_file="source-a.txt"):
    return {
        "type": "verbatim_semantic",
        "source_file": source_file,
        "student_start": student_start,
        "student_end": student_start + 200,
        "source_start": source_start,
        "source_end": source_start + 200,
        "student": "shared student passage",
        "source": "shared source passage",
        "student_match_text": "shared student passage",
        "source_match_text": "shared source passage",
        "semantic_score": 0.94,
        "lexical_score": 0.88,
        "verbatim_characters": 240,
        "match_methods": [
            "dense_embedding", "lexical_window", "verbatim_fingerprint"],
        "evidence_count": 3,
    }


def result_with(*evidence):
    return {
        "filename": "source-a.txt",
        "semantic_coverage": 20.0,
        "verbatim_score": 8.0,
        "consolidated_evidence": list(evidence),
    }


class Phase7TraceTests(unittest.TestCase):
    def replay(self, evidence):
        results = [result_with(*copy.deepcopy(evidence))]
        trace = []
        summary = detector.select_display_evidence(
            results,
            profile="high_precision",
            trace=trace,
            trace_namespace="pan-evaluation/suspicious-document00320",
        )
        return results, summary, trace

    def test_trace_does_not_change_selection_or_summary(self):
        evidence = [
            strong_evidence(100, 1000),
            strong_evidence(120, 1020),
            strong_evidence(2000, 5000),
        ]
        plain_results = [result_with(*copy.deepcopy(evidence))]
        plain_summary = detector.select_display_evidence(
            plain_results, profile="high_precision")

        traced_results, traced_summary, trace = self.replay(evidence)

        self.assertEqual(traced_results, plain_results)
        self.assertEqual(traced_summary, plain_summary)
        self.assertEqual(len(trace), len(evidence))
        self.assertTrue(all(
            row["status"] in {"selected", "rejected"} for row in trace))

    def test_candidate_ids_are_stable_and_distinguish_repeats(self):
        repeated = strong_evidence(100, 1000)
        evidence = [copy.deepcopy(repeated), copy.deepcopy(repeated)]

        _results, _summary, first_trace = self.replay(evidence)
        _results, _summary, second_trace = self.replay(evidence)

        first_ids = [row["candidate_id"] for row in first_trace]
        self.assertEqual(
            first_ids, [row["candidate_id"] for row in second_trace])
        self.assertEqual(len(set(first_ids)), 2)

    def test_trace_reasons_equal_existing_counters(self):
        evidence = [
            strong_evidence(0, 0),
            strong_evidence(1500, 1500),
            strong_evidence(3000, 3000),
        ]

        _results, summary, trace = self.replay(evidence)

        reasons = Counter(
            row["rejection_reason"] for row in trace
            if row["status"] == "rejected")
        self.assertEqual(
            dict(reasons),
            {key: value for key, value in summary["rejections"].items()
             if value},
        )
        rejected = [row for row in trace if row["status"] == "rejected"]
        self.assertEqual(rejected[0]["rejection_reason"], "source_cap")
        self.assertEqual(rejected[0]["source_selected_count_at_decision"], 2)
        self.assertEqual(sorted(row["rank"] for row in trace), [1, 2, 3])

    def test_competition_records_selected_blocker_id(self):
        evidence = [
            strong_evidence(100, 1000, source_file="source-a.txt"),
            strong_evidence(110, 9000, source_file="source-b.txt"),
        ]

        _results, summary, trace = self.replay(evidence)

        self.assertEqual(
            summary["rejections"]["cross_source_competition"], 1)
        selected_id = next(
            row["candidate_id"] for row in trace
            if row["status"] == "selected")
        rejected = next(
            row for row in trace if row["status"] == "rejected")
        self.assertEqual(rejected["blocking_candidate_ids"], [selected_id])

    def test_disabled_trace_does_not_create_candidate_keys(self):
        results = [result_with(strong_evidence(100, 1000))]
        with mock.patch.object(
                detector,
                "_phase7_candidate_key",
                side_effect=AssertionError("trace path was entered")):
            detector.select_display_evidence(
                results, profile="high_precision")


if __name__ == "__main__":
    unittest.main()
