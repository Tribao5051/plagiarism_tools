"""Replay Phase 7 from saved JSON and diagnose XML loss offline."""

import argparse
import copy
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

from _runtime_path import PROJECT_ROOT

from detection_module_diff_p4 import select_display_evidence
import evidence_metrics


DEFAULT_BASELINE = Path(
    r"D:\hochanh\hk252\plagiarism_performance"
    r"\oracle_noise_pan18_20260909\results"
)
DEFAULT_EVALUATION = Path(
    r"D:\hochanh\hk252\doan\project\evaluation_folder"
)
DEFAULT_SOURCES = Path(
    r"D:\hochanh\hk252\doan\project\combined_test_corpus"
)
TRACE_SCHEMA = "phase7_selection_trace_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay_response(response, profile, namespace):
    replayed = copy.deepcopy(response)
    trace = []
    summary = select_display_evidence(
        replayed["results"],
        profile=profile,
        trace=trace,
        trace_namespace=namespace,
    )
    return replayed, summary, trace


def replay_matches_saved(saved, replayed):
    mismatches = []
    if len(saved.get("results", ())) != len(replayed.get("results", ())):
        return ["result_count"]
    for result_index, (before, after) in enumerate(zip(
            saved["results"], replayed["results"])):
        for field in ("evidence", "evidence_summary"):
            if before.get(field) != after.get(field):
                mismatches.append(f"result[{result_index}].{field}")
    return mismatches


def attach_truth_ids(response, trace, truths):
    records = []
    for row in trace:
        result = response["results"][row["result_index"]]
        evidence = result["consolidated_evidence"][row["evidence_index"]]
        records.append({
            "record_id": row["candidate_id"],
            "source_file": (
                evidence.get("source_file") or result.get("filename", "")),
            "evidence": evidence,
        })
    evidence_metrics.annotate_truth_matches(records, truths)
    for row, record in zip(trace, records):
        row["truth_ids"] = sorted(record["truth_scores"])
        row["truth_scores"] = {
            str(truth_id): round(score, 6)
            for truth_id, score in sorted(record["truth_scores"].items())
        }
    return records


def caught_ids(response, field, truths):
    records = evidence_metrics.records_for_response(
        response, field, truths)
    return {
        truth_id
        for record in records
        for truth_id in record.get("truth_ids", ())
    }


def candidate_summary(row):
    return {
        key: row.get(key)
        for key in (
            "candidate_id",
            "source_file",
            "evidence_type",
            "student_span",
            "source_span",
            "confidence_score",
            "agreement_score",
            "selection_score",
            "rank",
            "status",
            "rejection_reason",
            "blocking_candidate_ids",
            "selected_count_at_decision",
            "source_selected_count_at_decision",
        )
    }


def build_xml_rows(document, truths, raw_ids, consolidated_ids, display_ids,
                   trace):
    rows = []
    for truth_id, truth in enumerate(truths):
        if truth_id not in raw_ids:
            stage = "not_detected_raw"
        elif truth_id not in consolidated_ids:
            stage = "lost_consolidation"
        elif truth_id not in display_ids:
            stage = "lost_phase7"
        else:
            stage = "selected"
        supporters = [
            candidate_summary(candidate)
            for candidate in trace
            if truth_id in candidate.get("truth_ids", ())
        ]
        reasons = sorted({
            candidate["rejection_reason"]
            for candidate in supporters
            if candidate["rejection_reason"]
        })
        rows.append({
            "xml_key": f"{document}#xml-{truth_id}",
            "document": document,
            "xml_id": truth_id,
            "source_file": truth["source_file"],
            "obfuscation": truth.get("obfuscation"),
            "stage": stage,
            "supporting_candidate_count": len(supporters),
            "rejection_reasons": reasons,
            "supporting_candidates": supporters,
        })
    return rows


def summarize(candidate_rows, xml_rows):
    candidate_rejections = Counter(
        row["rejection_reason"]
        for row in candidate_rows
        if row["status"] == "rejected"
    )
    lost_phase7 = [
        row for row in xml_rows if row["stage"] == "lost_phase7"]
    supporting_lost_candidates = {
        candidate["candidate_id"]
        for row in lost_phase7
        for candidate in row["supporting_candidates"]
    }
    xml_reason_incidence = Counter(
        reason
        for row in lost_phase7
        for reason in set(row["rejection_reasons"])
    )
    return {
        "candidate_counts": {
            "total": len(candidate_rows),
            "selected": sum(
                row["status"] == "selected" for row in candidate_rows),
            "rejected": sum(
                row["status"] == "rejected" for row in candidate_rows),
            "supporting_lost_phase7_xml": len(
                supporting_lost_candidates),
            "rejections": dict(sorted(candidate_rejections.items())),
        },
        "xml_counts": {
            "total": len(xml_rows),
            "selected": sum(
                row["stage"] == "selected" for row in xml_rows),
            "not_detected_raw": sum(
                row["stage"] == "not_detected_raw" for row in xml_rows),
            "lost_consolidation": sum(
                row["stage"] == "lost_consolidation" for row in xml_rows),
            "lost_phase7": len(lost_phase7),
            "lost_phase7_without_supporting_candidate": sum(
                not row["supporting_candidates"] for row in lost_phase7),
            "lost_phase7_reason_incidence": dict(
                sorted(xml_reason_incidence.items())),
        },
    }


def write_xml_csv(path, xml_rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "xml_key", "document", "xml_id", "source_file",
            "obfuscation", "stage", "supporting_candidate_count",
            "rejection_reasons",
        ])
        writer.writeheader()
        for row in xml_rows:
            output = {
                key: row[key] for key in writer.fieldnames
                if key != "rejection_reasons"
            }
            output["rejection_reasons"] = "|".join(
                row["rejection_reasons"])
            writer.writerow(output)


def write_report(path, summary, replay):
    candidates = summary["candidate_counts"]
    xml = summary["xml_counts"]
    lines = [
        "N01 PHASE 7 OFFLINE TRACE",
        "",
        f"Saved JSON files replayed: {replay['files']}",
        f"Replay mismatches: {replay['mismatch_count']}",
        "",
        "Candidate counts (not XML segments):",
        f"- total: {candidates['total']}",
        f"- selected: {candidates['selected']}",
        f"- rejected: {candidates['rejected']}",
        "- rejections: " + json.dumps(
            candidates["rejections"], sort_keys=True),
        "",
        "XML segment counts:",
        f"- total: {xml['total']}",
        f"- selected by Phase 7: {xml['selected']}",
        f"- not detected in raw evidence: {xml['not_detected_raw']}",
        f"- lost during consolidation: {xml['lost_consolidation']}",
        f"- lost during Phase 7: {xml['lost_phase7']}",
        "- Phase 7 reason incidence: " + json.dumps(
            xml["lost_phase7_reason_incidence"], sort_keys=True),
        "",
        "One XML segment may have many supporting candidates and many",
        "rejection reasons. Candidate counts must not be reported as XML",
        "segment counts. Ground truth is attached only by this evaluator.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    baseline = Path(args.baseline)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Output must be a new directory: {output}")
    raw_dir = baseline / "raw_json" / "oracle_10"
    manifest = read_json(baseline / "manifest.json")
    config_path = PROJECT_ROOT / "runtime" / "plagiarism_config.toml"
    current_config_hash = digest(config_path)
    if current_config_hash != manifest["config_sha256"]:
        raise ValueError(
            "Current Phase 7 config does not match the saved baseline")
    documents = {doc["file"]: doc for doc in manifest["documents"]}
    profile = manifest["evidence_profile"]
    faculty = manifest["faculty"]
    output.mkdir(parents=True)

    candidate_rows = []
    xml_rows = []
    replay_rows = []
    for json_path in sorted(raw_dir.glob("*.json")):
        document = json_path.stem
        saved = read_json(json_path)
        replayed, selection_summary, trace = replay_response(
            saved, profile, f"{faculty}/{document}")
        mismatches = replay_matches_saved(saved, replayed)
        doc = documents[document]
        truths = evidence_metrics.parse_ground_truth(
            Path(args.evaluation) / f"{document}.xml",
            Path(args.evaluation) / f"{document}.txt",
            Path(args.sources),
        )
        attach_truth_ids(replayed, trace, truths)
        raw_ids = caught_ids(replayed, "raw_evidence", truths)
        consolidated_ids = caught_ids(
            replayed, "consolidated_evidence", truths)
        display_ids = caught_ids(replayed, "evidence", truths)
        for row in trace:
            row["document"] = document
            row["faculty"] = faculty
        candidate_rows.extend(trace)
        xml_rows.extend(build_xml_rows(
            document, truths, raw_ids, consolidated_ids, display_ids, trace))
        traced_rejections = Counter(
            row["rejection_reason"] for row in trace
            if row["status"] == "rejected")
        expected_rejections = {
            key: value
            for key, value in selection_summary["rejections"].items()
            if value
        }
        replay_rows.append({
            "document": document,
            "truth_segments": len(truths),
            "candidate_count": len(trace),
            "selected_count": selection_summary["shown_count"],
            "mismatches": mismatches,
            "trace_rejections_match_summary": (
                dict(traced_rejections) == expected_rejections),
            "manifest_truth_segments_match": (
                len(truths) == doc["truth_segments"]),
        })

    if not replay_rows:
        raise ValueError(f"No saved JSON files found in {raw_dir}")
    mismatch_rows = [
        row for row in replay_rows
        if row["mismatches"]
        or not row["trace_rejections_match_summary"]
        or not row["manifest_truth_segments_match"]
    ]
    if mismatch_rows:
        raise AssertionError(
            "Replay validation failed: "
            + json.dumps(mismatch_rows, ensure_ascii=False))

    summary = summarize(candidate_rows, xml_rows)
    if (
        args.expected_phase7_lost is not None
        and summary["xml_counts"]["lost_phase7"]
        != args.expected_phase7_lost
    ):
        raise AssertionError(
            "Unexpected Phase 7 XML loss: "
            f"{summary['xml_counts']['lost_phase7']} != "
            f"{args.expected_phase7_lost}")
    replay_summary = {
        "files": len(replay_rows),
        "mismatch_count": len(mismatch_rows),
        "rows": replay_rows,
    }
    output_summary = {
        "schema": TRACE_SCHEMA,
        "source_baseline": str(baseline),
        "condition": "oracle_10",
        "profile": profile,
        "faculty": faculty,
        "provenance": {
            "config_sha256": current_config_hash,
            "detector_sha256": digest(
                PROJECT_ROOT / "runtime"
                / "detection_module_diff_p4.py"),
        },
        "scoring": evidence_metrics.scoring_metadata(),
        **summary,
        "replay": replay_summary,
    }
    write_json(output / "summary.json", output_summary)
    write_json(output / "xml_stage_table.json", xml_rows)
    write_json(output / "replay_validation.json", replay_summary)
    with (output / "candidate_trace.jsonl").open(
            "w", encoding="utf-8") as stream:
        for row in candidate_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_xml_csv(output / "xml_stage_table.csv", xml_rows)
    write_report(output / "REPORT.txt", summary, replay_summary)
    return output_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-phase7-lost", type=int, default=45)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
