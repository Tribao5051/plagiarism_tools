"""Paired PAN oracle benchmark. Writes only beneath --output; never rebuilds an index."""

import argparse
import ast
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import threading
import time
import unittest
import xml.etree.ElementTree as ET

import psutil
import requests


RUNTIME = Path(r"D:\hochanh\hk252\plagiarism_runtime")
PROJECT = Path(r"D:\hochanh\hk252\doan\project")
PREVIOUS = Path(r"D:\hochanh\hk252\plagiarism_performance\post_task9_pan18_20260909\raw_json")
EVALUATION = PROJECT / "evaluation_folder"
SOURCES = PROJECT / "combined_test_corpus"
SPLIT = RUNTIME / "test_data/reranker_experiment_split.json"
SEED = 20260909
PROFILES = {"semantic": {"semantic", "verbatim_semantic"},
            "verbatim": {"verbatim", "verbatim_semantic"},
            "mixed": {"verbatim_semantic"}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_scoring():
    path = PROJECT / "evaluate_pan_side_by_side.py"
    spec = importlib.util.spec_from_file_location("pan_oracle_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    # Load the exact pure Task 6 functions without importing the detector/model in the client.
    reranker_path = RUNTIME / "performance/benchmark_evidence_rerankers.py"
    tree = ast.parse(reranker_path.read_text(encoding="utf-8"))
    names = {"prediction_text", "annotate_truth_matches"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("Task 6 scoring functions are missing")
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reranker_path), "exec"), namespace)
    helpers = {"threshold": evaluator.TEXT_MATCH_THRESHOLD,
               "text_profile": evaluator.text_profile,
               "containment": evaluator.containment_from_profiles,
               "token_containment": evaluator.token_containment}
    return evaluator, namespace["annotate_truth_matches"], helpers


def choose_noise(available, truth_sources, ranked, seed):
    allowed = set(available) - set(truth_sources)
    hard = list(dict.fromkeys(name for name in ranked if name in allowed))[:6]
    pool = sorted(allowed - set(hard))
    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = []
    # Both prefixes contain difficult and random distractors: 3+2 and 6+4.
    for hard_slot in (True, True, False, True, False) * 2:
        if hard_slot and hard:
            selected.append({"source": hard.pop(0), "kind": "prior_shortlist_non_truth"})
        elif pool:
            selected.append({"source": pool.pop(), "kind": "seeded_random"})
        elif hard:
            selected.append({"source": hard.pop(0), "kind": "prior_shortlist_non_truth"})
        else:
            raise ValueError("At least ten distinct non-truth sources are required")
    return selected


def build_manifest():
    split = read_json(SPLIT)
    entries = [(name, partition) for partition, names in split.items() for name in names]
    if len(entries) != len({name for name, _ in entries}):
        raise ValueError("Duplicate submission across partitions")
    available = sorted(path.name for path in SOURCES.glob("*.txt"))
    documents = []
    for index, (name, partition) in enumerate(entries):
        xml = EVALUATION / (name + ".xml")
        features = [x for x in ET.parse(xml).getroot().findall("feature")
                    if x.get("name") == "artificial-plagiarism"]
        truth_sources = sorted({x.get("source_reference") for x in features})
        if not set(truth_sources) <= set(available):
            raise ValueError("Missing ground-truth source for " + name)
        previous = read_json(PREVIOUS / (name + ".json"))
        noise = choose_noise(available, truth_sources,
                             [x["filename"] for x in previous["results"]], SEED + index)
        order = [0, 5, 10]
        order = order[index % 3:] + order[:index % 3]
        if not features:
            order = [n for n in order if n]
        documents.append({"file": name, "partition": partition,
                          "truth_segments": len(features), "truth_sources": truth_sources,
                          "noise": noise, "condition_order": order,
                          "submission_sha256": digest(EVALUATION / (name + ".txt")),
                          "xml_sha256": digest(xml),
                          "previous_response_sha256": digest(PREVIOUS / (name + ".json"))})
    used_sources = sorted({source for doc in documents for source in
                           doc["truth_sources"] + [x["source"] for x in doc["noise"]]})
    return {"schema": "oracle_noise_task6_v1", "seed": SEED,
            "conditions": [0, 5, 10], "faculty": "pan-evaluation",
            "evidence_profile": "agreement", "encode_batch_size": 48,
            "documents": documents,
            "source_sha256": {name: digest(SOURCES / name) for name in used_sources},
            "evaluator_sha256": digest(PROJECT / "evaluate_pan_side_by_side.py"),
            "task6_scorer_sha256": digest(RUNTIME / "performance/benchmark_evidence_rerankers.py"),
            "config_sha256": digest(RUNTIME / "runtime/plagiarism_config.toml"),
            "notes": ["Oracle uses whole source files, never XML text spans as input.",
                      "All forced candidates are compared; faculty/hybrid retrieval is bypassed.",
                      "Noise means not annotated as a source in this submission XML.",
                      "Hard distractors come from this submission's previous top-20 results.",
                      "Ten-noise condition includes the same five-noise sources.",
                      "Pure oracle is not applicable to negative documents: no sources to compare.",
                      "One sequential request per condition; rotated order, mixed natural cache.",
                      "Source count and source text lengths differ, so latency is descriptive.",
                      "Fixed 18 documents reused for diagnosis; this is not a new held-out test.",
                      "No model, detection thresholds, or evidence selection tuning.",
                      "Character precision not measured: original-offset mapping is required."]}


def records_for(response, field, truths, annotator, helpers):
    records = []
    for result in response["results"]:
        if field not in result:
            raise ValueError("Response missing " + field)
        for evidence in result[field]:
            records.append({"source_file": evidence.get("source_file") or result["filename"],
                            "evidence": evidence})
    annotator(records, truths, helpers)
    for record in records:
        record["truth_ids"] = sorted(record.pop("truth_scores"))
    return records


def metrics(records, truth_count):
    caught = sorted({i for record in records for i in record["truth_ids"]})
    hits = sum(bool(record["truth_ids"]) for record in records)
    precision = hits / len(records) if records else None
    recall = len(caught) / truth_count if truth_count else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.0)
    return {"truth_segments": truth_count, "caught_truth_segments": len(caught),
            "caught_truth_ids": caught, "evidence_count": len(records),
            "true_positive_evidence": hits, "false_positive_evidence": len(records) - hits,
            "evidence_hit_precision": precision, "segment_recall": recall, "f1": f1}


def evaluate_response(response, doc, scoring):
    evaluator, annotator, helpers = scoring
    name = doc["file"]
    truths = evaluator.parse_ground_truth(str(EVALUATION / (name + ".xml")),
                                         str(EVALUATION / (name + ".txt")), str(SOURCES))
    stages = {}
    display = []
    for field in ("raw_evidence", "consolidated_evidence", "evidence"):
        records = records_for(response, field, truths, annotator, helpers)
        stages[field] = metrics(records, len(truths))
        if field == "evidence":
            display = records
    families = {name: metrics([x for x in display if x["evidence"].get("type") in kinds], len(truths))
                for name, kinds in PROFILES.items()}
    raw = set(stages["raw_evidence"]["caught_truth_ids"])
    consolidated = set(stages["consolidated_evidence"]["caught_truth_ids"])
    shown = set(stages["evidence"]["caught_truth_ids"])
    available = set(range(len(truths)))
    diagnostics = {"missing_in_raw": sorted(available - raw),
                   "lost_by_consolidation": sorted(raw - consolidated),
                   "lost_by_selection": sorted(consolidated - shown),
                   "display_on_non_truth_sources": sum(x["source_file"] not in doc["truth_sources"] for x in display)}
    breakdown = {}
    for label in ("none", "low", "high", "translation"):
        ids = {i for i, truth in enumerate(truths) if
               (truth.get("translation") if label == "translation" else truth.get("obfuscation") == label)}
        breakdown[label] = {"total": len(ids), "caught": len(ids & shown)}
    return {"stages": stages, "families": families, "diagnostics": diagnostics,
            "breakdown": breakdown}, truths, display


class Resources:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        psutil.cpu_percent()
        while not self.stop.is_set():
            self.samples.append({"system_cpu_percent": psutil.cpu_percent(),
                                 "system_ram_percent": psutil.virtual_memory().percent,
                                 "client_rss_mb": psutil.Process().memory_info().rss / 1024**2,
                                 "gpu": self.evaluator.query_gpu()})
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return self.samples


def aggregate(rows):
    count = sum(x["truth_segments"] for x in rows)
    caught = sum(x["caught_truth_segments"] for x in rows)
    evidence = sum(x["evidence_count"] for x in rows)
    hits = sum(x["true_positive_evidence"] for x in rows)
    p = hits / evidence if evidence else None
    r = caught / count if count else None
    return {"truth_segments": count, "caught_truth_segments": caught,
            "evidence_count": evidence, "true_positive_evidence": hits,
            "evidence_hit_precision": p, "segment_recall": r,
            "f1": 2*p*r/(p+r) if p is not None and r is not None and p+r else 0.0}


def summarize(cases):
    groups = {}
    for condition in ("prior_shortlist", "oracle_0", "oracle_5", "oracle_10"):
        rows = [x for x in cases if x["condition"] == condition]
        if not rows:
            continue
        positive = [x for x in rows if x["truth_segments"]]
        negative = [x for x in rows if not x["truth_segments"]]
        stages = {stage: aggregate([x["evaluation"]["stages"][stage] for x in positive])
                  for stage in ("raw_evidence", "consolidated_evidence", "evidence")}
        families = {family: aggregate([x["evaluation"]["families"][family] for x in positive])
                    for family in PROFILES}
        times = [x["wall_seconds"] for x in rows if x.get("wall_seconds") is not None]
        groups[condition] = {"files": len(rows), "positive_files": len(positive),
                             "negative_files": len(negative), "positive_stages": stages,
                             "positive_families": families,
                             "all_display": aggregate([x["evaluation"]["stages"]["evidence"] for x in rows]),
                             "negative_files_with_evidence": sum(x["evaluation"]["stages"]["evidence"]["evidence_count"] > 0 for x in negative),
                             "evidence_on_noise_sources": sum(x["evaluation"]["diagnostics"]["display_on_non_truth_sources"] for x in rows),
                             "mean_wall_seconds": statistics.mean(times) if times else None,
                             "phase_mean_seconds": {f"phase_{i}": statistics.mean(x["performance"].get(f"phase_{i}_seconds", 0) for x in rows) for i in range(2, 8)}}
    return groups


def percentage(value):
    return "N/A" if value is None else "{:.2f}%".format(100*value)


def render_case(case, truths, display):
    escape = html.escape
    snippets = []
    for record in display:
        evidence = record["evidence"]
        truth_id = record["truth_ids"][0] if record["truth_ids"] else None
        truth = truths[truth_id] if truth_id is not None else {}
        cells = [truth.get("student_text", "No matching XML passage"), truth.get("source_text", ""),
                 evidence.get("student_match_text") or evidence.get("student_text", ""),
                 evidence.get("source_match_text") or evidence.get("source_text", "")]
        label = "{} | {} | XML IDs {} | student {}:{} source {}:{}".format(
            record["source_file"], evidence.get("type"), record["truth_ids"],
            evidence.get("student_start"), evidence.get("student_end"),
            evidence.get("source_start"), evidence.get("source_end"))
        snippets.append("<tr><td colspan='4'><b>" + escape(label) + "</b></td></tr><tr>" +
                        "".join("<td>" + escape(str(x)[:1600]) + (" ..." if len(str(x)) > 1600 else "") + "</td>" for x in cells) + "</tr>")
    diag = case["evaluation"]["diagnostics"]
    missing = sorted(set(range(len(truths))) - set(case["evaluation"]["stages"]["evidence"]["caught_truth_ids"]))
    missing_rows = []
    for i in missing:
        truth = truths[i]
        missing_rows.append("<tr><td>XML {}: {}</td><td>{}</td></tr>".format(
            i, escape(truth["student_text"][:1600]), escape(truth["source_text"][:1600])))
    return ("<details><summary>" + escape(case["condition"] + " / " + case["file"]) + "</summary><pre>" +
            escape(json.dumps(diag, indent=2)) + "</pre><table><tr><th>XML submission</th><th>XML source</th>"
            "<th>Tool submission</th><th>Tool source</th></tr>" + "".join(snippets) +
            "</table><h3>Missed XML passages</h3><table>" + "".join(missing_rows) + "</table></details>")


def save_report(output, manifest, cases, sections):
    summary = summarize(cases)
    write_json(output / "summary.json", {"groups": summary, "completed_cases": len(cases),
               "expected_live_requests": sum(len(x["condition_order"]) for x in manifest["documents"]),
               "completed_live_requests": sum(x["condition"] != "prior_shortlist" for x in cases),
               "metric": "Task 6: both submission and source text must match the same XML pair; every evidence with a valid pair is a hit; recall is union of XML IDs. Character precision unavailable.",
               "notes": manifest["notes"]})
    write_json(output / "per_case.json", cases)
    lines = ["PAN ORACLE + NOISE BENCHMARK", "", *manifest["notes"], "",
             "Comparable positive submissions (same 14 documents, 117 XML passages):"]
    for name, group in summary.items():
        m = group["positive_stages"]["evidence"]
        lines.append("{} | P={} R={} F1={} | negative false alarms={}/{} | mean wall={}".format(
            name, percentage(m["evidence_hit_precision"]), percentage(m["segment_recall"]), percentage(m["f1"]),
            group["negative_files_with_evidence"], group["negative_files"], group["mean_wall_seconds"]))
    (output / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    with (output / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        rows = []
        for case in cases:
            metric = case["evaluation"]["stages"]["evidence"]
            rows.append({"condition": case["condition"], "file": case["file"],
                         "partition": case["partition"], "candidates": case.get("candidate_count"),
                         "wall_seconds": case.get("wall_seconds"),
                         **{k:v for k,v in metric.items() if k != "caught_truth_ids"},
                         **{f"phase_{i}_seconds": case["performance"].get(f"phase_{i}_seconds") for i in range(2,8)}})
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><title>PAN Oracle + Noise</title>
<style>body{font:15px Arial;margin:24px;color:#202124}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #ccc;padding:10px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#e8f0f2}summary{padding:14px;cursor:pointer;font-weight:bold}pre{white-space:pre-wrap}details{border-top:1px solid #ccc;margin-top:14px}</style>
<h1>PAN: Oracle + 5 / 10 Noise Sources</h1><p>XML passages are ground truth. Tool passages are predictions. XML IDs are zero-based list positions. Both sides must match. Excerpts are capped at 1,600 characters; full responses are in raw_json. Mixed evidence belongs to both semantic and verbatim families.</p><pre>"""
    document += html.escape("\n".join(lines)) + "</pre>" + "".join(sections) + "</html>"
    (output / "side_by_side.html").write_text(document, encoding="utf-8")


def run(output, manifest, api_url, scoring):
    session = requests.Session()
    health = session.get(api_url.rsplit("/", 1)[0] + "/health", timeout=10)
    health.raise_for_status()
    probe = session.post(api_url, params={"faculty":"pan-evaluation", "candidate_sources":""},
                         files={"file":("probe.txt", b"probe", "text/plain")}, timeout=30)
    if probe.status_code != 400 or probe.json().get("error") != "Candidate sources cannot be empty":
        raise RuntimeError("Oracle is not enabled/ready: {} {}".format(probe.status_code, probe.text[:300]))
    write_json(output / "run_environment.json", {"started":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "health":health.json(), "script_sha256":digest(__file__), "python":sys.version,
               "ram_bytes":psutil.virtual_memory().total, "cpu_logical":psutil.cpu_count(),
               "cpu_physical":psutil.cpu_count(logical=False), "gpu":scoring[0].query_gpu(),
               "cache_policy":"Natural API cache; conditions rotated by document; no resets."})
    cases, sections = [], []
    expected = sum(len(x["condition_order"]) for x in manifest["documents"])
    completed = 0
    for doc in manifest["documents"]:
        name = doc["file"]
        old = read_json(PREVIOUS / (name + ".json"))
        evaluation, truths, records = evaluate_response(old, doc, scoring)
        cases.append({"condition":"prior_shortlist", "file":name, "partition":doc["partition"],
                      "truth_segments":doc["truth_segments"], "evaluation":evaluation,
                      "performance":old.get("performance",{}), "wall_seconds":None})
        for noise_count in doc["condition_order"]:
            condition = "oracle_" + str(noise_count)
            candidates = sorted(doc["truth_sources"] + [x["source"] for x in doc["noise"][:noise_count]])
            result_path = output / "raw_json" / condition / (name + ".json")
            request_path = output / "requests" / condition / (name + ".json")
            if result_path.exists() and request_path.exists():
                response, request = read_json(result_path), read_json(request_path)
                if request["candidates"] != candidates:
                    raise RuntimeError("Saved candidate set differs; use a new output directory")
            else:
                sampler = Resources(scoring[0])
                sampler.thread.start()
                start = time.perf_counter()
                try:
                    with (EVALUATION / (name + ".txt")).open("rb") as stream:
                        result = session.post(api_url, params={"faculty":"pan-evaluation",
                            "candidate_sources":",".join(candidates), "evidence_profile":"agreement",
                            "include_raw_evidence":"true", "encode_batch_size":48},
                            files={"file":(name+".txt", stream, "text/plain")}, timeout=600)
                    result.raise_for_status()
                    response = result.json()
                    request = {"candidates":candidates, "wall_seconds":time.perf_counter()-start,
                               "started_unix":time.time()-(time.perf_counter()-start)}
                finally:
                    samples = sampler.finish()
                request["resource_samples"] = samples
                write_json(result_path, response)
                write_json(request_path, request)
            performance = response.get("performance", {})
            if performance.get("retrieval_mode") != "oracle_xml" or performance.get("candidate_sources") != len(candidates):
                raise RuntimeError("Server did not compare the requested oracle candidates")
            if {r["filename"] for r in response.get("results", [])} != set(candidates):
                raise RuntimeError("Returned source set differs from the forced candidate set")
            evaluation, truths, records = evaluate_response(response, doc, scoring)
            case = {"condition":condition, "file":name, "partition":doc["partition"],
                    "truth_segments":doc["truth_segments"], "evaluation":evaluation,
                    "candidate_count":len(candidates), "performance":performance,
                    "wall_seconds":request["wall_seconds"]}
            cases.append(case)
            sections.append(render_case(case, truths, records))
            completed += 1
            m = evaluation["stages"]["evidence"]
            print("[{}/{}] {} {} candidates={} P={} R={} wall={:.2f}s".format(
                completed,expected,name,condition,len(candidates),percentage(m["evidence_hit_precision"]),
                percentage(m["segment_recall"]),case["wall_seconds"]), flush=True)
            save_report(output, manifest, cases, sections)
    print("COMPLETE " + str(output), flush=True)


class BenchmarkTests(unittest.TestCase):
    def test_noise_is_distinct_and_excludes_truth(self):
        noise = choose_noise([str(i) for i in range(30)], ["0"], [str(i) for i in range(10)], 42)
        self.assertEqual(len({x["source"] for x in noise}), 10)
        self.assertNotIn("0", [x["source"] for x in noise])
        self.assertEqual(sum(x["kind"] == "prior_shortlist_non_truth" for x in noise[:5]), 3)
        self.assertEqual(noise, choose_noise([str(i) for i in range(30)], ["0"], [str(i) for i in range(10)], 42))

    def test_task6_rejects_wrong_source_and_wrong_source_text(self):
        _, annotate, helpers = load_scoring()
        passage = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        truth = {"source_file":"right.txt", "student_text":passage, "source_text":passage, "obfuscation":"none"}
        records = [{"source_file":name, "evidence":{"student_match_text":passage,"source_match_text":text}}
                   for name,text in [("right.txt",passage),("wrong.txt",passage),("right.txt","red blue green yellow pink black white")]]
        annotate(records, [truth], helpers)
        self.assertEqual([bool(r["truth_scores"]) for r in records], [True, False, False])

    def test_recall_union_and_negative_are_not_inflated(self):
        score = metrics([{"truth_ids":[0]}, {"truth_ids":[0]}], 2)
        self.assertEqual(score["segment_recall"], 0.5)
        self.assertEqual(score["evidence_hit_precision"], 1.0)
        score = metrics([{"truth_ids":[]}], 0)
        self.assertIsNone(score["segment_recall"])
        self.assertEqual(score["false_positive_evidence"], 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/scan")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BenchmarkTests))
        return 0 if result.wasSuccessful() else 1
    manifest = build_manifest()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("Input files/config/scorer changed. Use a new output directory.")
    write_json(manifest_path, manifest)
    print("Prepared {} files, {} requests, {} source hashes.".format(len(manifest["documents"]),
          sum(len(x["condition_order"]) for x in manifest["documents"]), len(manifest["source_sha256"])), flush=True)
    if args.run:
        run(output, manifest, args.api_url, load_scoring())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
