import unittest

from tests import _project_paths

import trace_phase7_selection as report


def trace_candidate(candidate_id, reason):
    return {
        "candidate_id": candidate_id,
        "source_file": "source-a.txt",
        "evidence_type": "semantic",
        "student_span": [10, 100],
        "source_span": [20, 110],
        "confidence_score": 0.8,
        "agreement_score": 0.6,
        "selection_score": 0.6,
        "rank": 1,
        "status": "rejected",
        "rejection_reason": reason,
        "blocking_candidate_ids": [],
        "selected_count_at_decision": 4,
        "source_selected_count_at_decision": 4,
        "truth_ids": [0],
    }


class Phase7TraceReportTests(unittest.TestCase):
    def test_one_xml_can_have_multiple_candidates_and_reasons(self):
        truths = [{
            "source_file": "source-a.txt",
            "obfuscation": "high",
        }]
        trace = [
            trace_candidate("candidate-1", "source_cap"),
            trace_candidate("candidate-2", "same_source_duplicate"),
        ]

        rows = report.build_xml_rows(
            "submission-a",
            truths,
            raw_ids={0},
            consolidated_ids={0},
            display_ids=set(),
            trace=trace,
        )
        summary = report.summarize(trace, rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "lost_phase7")
        self.assertEqual(rows[0]["supporting_candidate_count"], 2)
        self.assertEqual(
            rows[0]["rejection_reasons"],
            ["same_source_duplicate", "source_cap"],
        )
        self.assertEqual(summary["candidate_counts"]["total"], 2)
        self.assertEqual(summary["xml_counts"]["lost_phase7"], 1)
        self.assertEqual(
            summary["xml_counts"]["lost_phase7_reason_incidence"],
            {"same_source_duplicate": 1, "source_cap": 1},
        )

    def test_detector_trace_has_no_ground_truth_before_evaluator(self):
        response = {
            "results": [{
                "filename": "source-a.txt",
                "semantic_coverage": 20.0,
                "verbatim_score": 0.0,
                "consolidated_evidence": [{
                    "type": "semantic",
                    "source_file": "source-a.txt",
                    "student_start": 10,
                    "student_end": 200,
                    "source_start": 20,
                    "source_end": 210,
                    "student_match_text": "student passage",
                    "source_match_text": "source passage",
                    "semantic_score": 0.94,
                    "match_methods": ["dense_embedding"],
                    "evidence_count": 2,
                }],
                "evidence": [],
                "evidence_summary": {},
            }]
        }

        replayed, _summary, trace = report.replay_response(
            response, "balanced", "faculty/submission")

        self.assertEqual(len(trace), 1)
        self.assertNotIn("truth_ids", trace[0])
        self.assertNotIn("truth_scores", trace[0])
        self.assertEqual(report.replay_matches_saved(replayed, replayed), [])


if __name__ == "__main__":
    unittest.main()
