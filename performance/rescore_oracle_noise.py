"""Re-score saved oracle JSON without starting the API, GPU, or detector."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

import evidence_metrics


RUNTIME = Path(__file__).resolve().parents[1]
OLD_PROJECT = Path(r"D:\hochanh\hk252\doan\project")
DEFAULT_INPUT = Path(
    r"D:\hochanh\hk252\plagiarism_performance"
    r"\oracle_noise_pan18_20260909\results"
)
DEFAULT_BASELINE = (
    RUNTIME / "test_data" / "oracle_noise_baseline" / "summary.json")
DEFAULT_EVALUATION = OLD_PROJECT / "evaluation_folder"
DEFAULT_SOURCES = OLD_PROJECT / "combined_test_corpus"
CONDITIONS = ("oracle_0", "oracle_5", "oracle_10")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_new_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            "Output directory must be new and empty: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def aggregate(rows):
    return evidence_metrics.aggregate_metrics(rows)


def summarize(cases):
    groups = {}
    for condition in CONDITIONS:
        rows = [case for case in cases if case["condition"] == condition]
        if not rows:
            continue
        positive = [case for case in rows if case["truth_segments"]]
        negative = [case for case in rows if not case["truth_segments"]]
        times = [
            case["wall_seconds"]
            for case in rows
            if case.get("wall_seconds") is not None
        ]
        groups[condition] = {
            "files": len(rows),
            "positive_files": len(positive),
            "negative_files": len(negative),
            "positive_stages": {
                stage: aggregate([
                    case["evaluation"]["stages"][stage]
                    for case in positive
                ])
                for stage in (
                    "raw_evidence", "consolidated_evidence", "evidence")
            },
            "positive_families": {
                family: aggregate([
                    case["evaluation"]["families"][family]
                    for case in positive
                ])
                for family in evidence_metrics.EVIDENCE_FAMILIES
            },
            "all_display": aggregate([
                case["evaluation"]["stages"]["evidence"]
                for case in rows
            ]),
            "negative_files_with_evidence": sum(
                case["evaluation"]["stages"]["evidence"][
                    "evidence_count"] > 0
                for case in negative
            ),
            "evidence_on_noise_sources": sum(
                case["evaluation"]["diagnostics"][
                    "display_on_non_truth_sources"]
                for case in rows
            ),
            "mean_wall_seconds": statistics.mean(times) if times else None,
            "phase_mean_seconds": {
                f"phase_{phase}": statistics.mean(
                    case["performance"].get(
                        f"phase_{phase}_seconds", 0)
                    for case in rows
                )
                for phase in range(2, 8)
            },
        }
    return groups


def compare_values(expected, actual, path=""):
    differences = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{
                "path": path,
                "expected": expected,
                "actual": actual,
            }]
        for key, expected_value in expected.items():
            child_path = f"{path}.{key}" if path else key
            if key not in actual:
                differences.append({
                    "path": child_path,
                    "expected": expected_value,
                    "actual": "<missing>",
                })
                continue
            differences.extend(compare_values(
                expected_value, actual[key], child_path))
        return differences
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [{
                "path": path,
                "expected": expected,
                "actual": actual,
            }]
        for index, expected_value in enumerate(expected):
            differences.extend(compare_values(
                expected_value, actual[index], f"{path}[{index}]"))
        return differences
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        equal = math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)
    else:
        equal = expected == actual
    if not equal:
        differences.append({
            "path": path,
            "expected": expected,
            "actual": actual,
        })
    return differences


def compare_with_baseline(groups, baseline):
    expected_groups = baseline["groups"]
    differences = []
    for condition in CONDITIONS:
        if condition not in expected_groups:
            differences.append({
                "path": f"groups.{condition}",
                "expected": "<condition present>",
                "actual": "<missing baseline condition>",
            })
            continue
        differences.extend(compare_values(
            expected_groups[condition], groups.get(condition),
            f"groups.{condition}",
        ))
    return {
        "matches": not differences,
        "conditions": list(CONDITIONS),
        "differences": differences,
    }


def load_cases(input_root, manifest, evaluation_dir, source_dir):
    cases = []
    expected_files = []
    for document in manifest["documents"]:
        name = document["file"]
        truths = evidence_metrics.parse_ground_truth(
            Path(evaluation_dir) / (name + ".xml"),
            Path(evaluation_dir) / (name + ".txt"),
            source_dir,
        )
        if len(truths) != document["truth_segments"]:
            raise RuntimeError(
                f"Truth count changed for {name}: "
                f"{len(truths)} != {document['truth_segments']}")
        for noise_count in document["condition_order"]:
            condition = "oracle_" + str(noise_count)
            response_path = (
                input_root / "raw_json" / condition / (name + ".json"))
            request_path = (
                input_root / "requests" / condition / (name + ".json"))
            if not response_path.is_file() or not request_path.is_file():
                raise FileNotFoundError(
                    "Saved response/request pair is incomplete: "
                    + str(response_path))
            expected_files.append(response_path.resolve())
            response = read_json(response_path)
            request = read_json(request_path)
            evaluation, _display = evidence_metrics.score_response(
                response, truths, document["truth_sources"])
            cases.append({
                "condition": condition,
                "file": name,
                "partition": document["partition"],
                "truth_segments": len(truths),
                "evaluation": evaluation,
                "candidate_count": len(request.get("candidates", ())),
                "performance": response.get("performance", {}),
                "wall_seconds": request.get("wall_seconds"),
            })
    discovered = sorted(
        path.resolve()
        for path in (input_root / "raw_json").glob("oracle_*/*.json")
    )
    if sorted(expected_files) != discovered:
        raise RuntimeError(
            "Saved JSON set differs from the 50 manifest requests")
    return cases


def percentage(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def run(args):
    input_root = Path(args.input_results).resolve()
    output = prepare_new_output(args.output)
    manifest_path = input_root / "manifest.json"
    manifest = read_json(manifest_path)
    cases = load_cases(
        input_root,
        manifest,
        Path(args.evaluation_dir).resolve(),
        Path(args.source_dir).resolve(),
    )
    if len(cases) != 50:
        raise RuntimeError(f"Expected 50 saved JSON cases, got {len(cases)}")
    groups = summarize(cases)
    baseline_path = Path(args.baseline_summary).resolve()
    baseline = read_json(baseline_path)
    comparison = compare_with_baseline(groups, baseline)
    scoring = evidence_metrics.scoring_metadata()
    summary = {
        "schema": "n00_offline_rescore_v1",
        "scoring": scoring,
        "groups": groups,
        "completed_cases": len(cases),
        "expected_saved_json": 50,
        "metric": (
            "Both submission and source text must match the same XML pair; "
            "every evidence with a valid pair is a hit; recall is the union "
            "of XML IDs. Character precision is unavailable."),
        "stage_fields": {
            "raw": "raw_evidence",
            "consolidated": "consolidated_evidence",
            "display": "evidence",
        },
        "families": list(evidence_metrics.EVIDENCE_FAMILIES),
        "baseline_comparison": comparison,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", summary)
    write_json(output / "per_case.json", cases)
    write_json(output / "baseline_comparison.json", comparison)
    write_json(output / "run_environment.json", {
        "mode": "offline_saved_json_rescore",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "input_results": str(input_root),
        "input_manifest_sha256": evidence_metrics.canonical_json_hash(
            manifest),
        "baseline_summary": str(baseline_path),
        "baseline_summary_sha256": evidence_metrics.canonical_json_hash(
            baseline),
        "scoring": scoring,
        "api_started": False,
        "gpu_used": False,
        "model_modules_loaded": sorted(
            name for name in (
                "torch", "transformers", "FlagEmbedding", "chromadb")
            if name in sys.modules
        ),
    })
    lines = [
        "N00 OFFLINE ORACLE RE-SCORE",
        "",
        f"Saved JSON scored: {len(cases)}",
        f"Baseline match: {comparison['matches']}",
        f"Scoring schema: {scoring['scoring_schema_version']}",
        f"Scoring code SHA-256: {scoring['scoring_code_sha256']}",
        "Character precision: N/A until original-offset mapping exists.",
        "",
    ]
    for condition in CONDITIONS:
        metric = groups[condition]["positive_stages"]["evidence"]
        lines.append(
            f"{condition}: P={percentage(metric['evidence_hit_precision'])} "
            f"R={percentage(metric['segment_recall'])} "
            f"hits={metric['true_positive_evidence']}/"
            f"{metric['evidence_count']}")
    if comparison["differences"]:
        lines.extend([
            "",
            "Differences:",
            *[
                f"- {item['path']}: expected={item['expected']!r}, "
                f"actual={item['actual']!r}"
                for item in comparison["differences"]
            ],
        ])
    (output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return comparison["matches"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-results", default=str(DEFAULT_INPUT))
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE))
    parser.add_argument("--evaluation-dir", default=str(DEFAULT_EVALUATION))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCES))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    return 0 if run(parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
