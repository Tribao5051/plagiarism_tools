import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from _runtime_path import PROJECT_ROOT

import fitz
import psutil

from pdf_extract import extract_university_content


ROOT = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_size_sample(files, sample_size):
    ordered = sorted(files, key=lambda path: (path.stat().st_size, str(path)))
    if sample_size <= 0 or sample_size >= len(ordered):
        return ordered
    if sample_size == 1:
        return [ordered[len(ordered) // 2]]
    positions = [
        round(index * (len(ordered) - 1) / (sample_size - 1))
        for index in range(sample_size)
    ]
    return [ordered[position] for position in positions]


def count_pages(path):
    with fitz.open(path) as document:
        return document.page_count


def mean(values):
    return statistics.mean(values) if values else 0.0


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def benchmark(corpus, sample_size, repeats):
    corpus = corpus.resolve()
    all_files = sorted(
        path for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not all_files:
        raise RuntimeError(f"No PDF files found under {corpus}")

    selected = stratified_size_sample(all_files, sample_size)
    records = []
    manifest_started = time.perf_counter()
    for path in selected:
        hash_started = time.perf_counter()
        digest = sha256_file(path)
        hash_seconds = time.perf_counter() - hash_started
        records.append({
            "path": path,
            "relative_path": path.relative_to(corpus).as_posix(),
            "pdf_bytes": path.stat().st_size,
            "pages": count_pages(path),
            "sha256": digest,
            "hash_seconds": hash_seconds,
            "runs": [],
        })
    manifest_seconds = time.perf_counter() - manifest_started

    process = psutil.Process()
    repeat_summaries = []
    for repeat in range(1, repeats + 1):
        repeat_started = time.perf_counter()
        successful_seconds = []
        successful_files = 0
        for record in records:
            started = time.perf_counter()
            text = extract_university_content(
                str(record["path"]), save_debug=False)
            elapsed = time.perf_counter() - started
            text_bytes = len(text.encode("utf-8")) if text else 0
            run = {
                "repeat": repeat,
                "seconds": elapsed,
                "characters": len(text),
                "text_utf8_bytes": text_bytes,
                "status": "ok" if text.strip() else "no_text",
            }
            record["runs"].append(run)
            if run["status"] == "ok":
                successful_seconds.append(elapsed)
                successful_files += 1
        repeat_summaries.append({
            "repeat": repeat,
            "wall_seconds": time.perf_counter() - repeat_started,
            "successful_files": successful_files,
            "failed_files": len(records) - successful_files,
            "mean_successful_file_seconds": mean(successful_seconds),
        })

    serializable_records = []
    for record in records:
        run_seconds = [
            run["seconds"] for run in record["runs"]
            if run["status"] == "ok"
        ]
        text_sizes = [
            run["text_utf8_bytes"] for run in record["runs"]
            if run["status"] == "ok"
        ]
        serializable_records.append({
            "relative_path": record["relative_path"],
            "pdf_bytes": record["pdf_bytes"],
            "pages": record["pages"],
            "sha256": record["sha256"],
            "hash_seconds": round(record["hash_seconds"], 6),
            "mean_extraction_seconds": round(mean(run_seconds), 6),
            "text_utf8_bytes": text_sizes[0] if text_sizes else 0,
            "runs": [
                {
                    **run,
                    "seconds": round(run["seconds"], 6),
                }
                for run in record["runs"]
            ],
        })

    first_runs = [record["runs"][0] for record in serializable_records]
    successful_first = [run for run in first_runs if run["status"] == "ok"]
    first_seconds = [run["seconds"] for run in successful_first]
    total_pdf_bytes = sum(record["pdf_bytes"] for record in serializable_records)
    total_text_bytes = sum(run["text_utf8_bytes"] for run in successful_first)
    average_seconds = mean(first_seconds)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": str(corpus),
        "selection": "deterministic size-stratified sample",
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_total_gib": round(psutil.virtual_memory().total / 1024**3, 2),
            "process_rss_mib_after": round(process.memory_info().rss / 1024**2, 2),
        },
        "corpus_pdf_count": len(all_files),
        "sample_size": len(selected),
        "repeats": repeats,
        "manifest_seconds": round(manifest_seconds, 6),
        "summary": {
            "successful_first_pass": len(successful_first),
            "failed_first_pass": len(selected) - len(successful_first),
            "total_pdf_bytes": total_pdf_bytes,
            "average_pdf_bytes": round(total_pdf_bytes / len(selected), 2),
            "total_text_utf8_bytes": total_text_bytes,
            "average_text_utf8_bytes": round(
                total_text_bytes / len(successful_first), 2
            ) if successful_first else 0,
            "text_to_pdf_percent": round(
                total_text_bytes * 100 / total_pdf_bytes, 4
            ) if total_pdf_bytes else 0,
            "first_pass_total_seconds": round(sum(first_seconds), 6),
            "first_pass_mean_seconds": round(average_seconds, 6),
            "first_pass_median_seconds": round(
                statistics.median(first_seconds), 6
            ) if first_seconds else 0,
            "first_pass_p95_seconds": round(percentile(first_seconds, 95), 6),
            "estimated_10_files_seconds": round(average_seconds * 10, 3),
            "estimated_15_files_seconds": round(average_seconds * 15, 3),
        },
        "repeat_summaries": [
            {
                **item,
                "wall_seconds": round(item["wall_seconds"], 6),
                "mean_successful_file_seconds": round(
                    item["mean_successful_file_seconds"], 6),
            }
            for item in repeat_summaries
        ],
        "files": serializable_records,
    }


def write_report(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pdf_extraction_results.json"
    text_path = output_dir / "PDF_EXTRACTION_REPORT.txt"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = result["summary"]
    lines = [
        "PDF EXTRACTION BENCHMARK",
        "",
        f"Generated: {result['generated_at']}",
        f"Corpus: {result['corpus']}",
        f"Corpus PDF count: {result['corpus_pdf_count']}",
        f"Sample: {result['sample_size']} size-stratified files",
        f"Repeats on the exact same files: {result['repeats']}",
        "",
        "SUMMARY",
        f"Successful/failed: {summary['successful_first_pass']}/"
        f"{summary['failed_first_pass']}",
        f"Selected PDF bytes: {summary['total_pdf_bytes']}",
        f"Average PDF bytes/file: {summary['average_pdf_bytes']}",
        f"Extracted UTF-8 bytes: {summary['total_text_utf8_bytes']}",
        f"Average text bytes/successful file: "
        f"{summary['average_text_utf8_bytes']}",
        f"Text/PDF storage ratio: {summary['text_to_pdf_percent']}%",
        f"First pass total extraction: {summary['first_pass_total_seconds']}s",
        f"First pass mean/file: {summary['first_pass_mean_seconds']}s",
        f"First pass median/file: {summary['first_pass_median_seconds']}s",
        f"First pass p95/file: {summary['first_pass_p95_seconds']}s",
        f"Estimated 10 files: {summary['estimated_10_files_seconds']}s",
        f"Estimated 15 files: {summary['estimated_15_files_seconds']}s",
        "",
        "REPEATS",
    ]
    for repeat in result["repeat_summaries"]:
        lines.append(
            f"repeat={repeat['repeat']} | wall={repeat['wall_seconds']}s | "
            f"mean/file={repeat['mean_successful_file_seconds']}s | "
            f"ok={repeat['successful_files']} | failed={repeat['failed_files']}"
        )
    lines.extend(["", "FILES"])
    for item in result["files"]:
        statuses = ",".join(run["status"] for run in item["runs"])
        lines.append(
            f"{item['relative_path']} | pdf_bytes={item['pdf_bytes']} | "
            f"pages={item['pages']} | text_bytes={item['text_utf8_bytes']} | "
            f"hash={item['hash_seconds']}s | "
            f"extract_mean={item['mean_extraction_seconds']}s | "
            f"status={statuses} | sha256={item['sha256']}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PDF extraction without persisting extracted text.")
    parser.add_argument("--corpus", default=str(PROJECT_ROOT / "dspace_corpus"))
    parser.add_argument("--sample-size", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "pdf_extraction_15"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_size < 1 or args.repeats < 1:
        raise SystemExit("sample-size and repeats must be positive")
    result = benchmark(Path(args.corpus), args.sample_size, args.repeats)
    json_path, text_path = write_report(result, Path(args.output).resolve())
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Text: {text_path}")


if __name__ == "__main__":
    main()
