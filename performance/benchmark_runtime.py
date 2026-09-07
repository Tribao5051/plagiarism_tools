import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from _runtime_path import PROJECT_ROOT, RUNTIME_ROOT

import psutil
import requests


ROOT = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_files(root):
    root = Path(root).resolve()
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


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


def manifest_for(files, corpus_root):
    corpus_root = Path(corpus_root).resolve()
    return [
        {
            "relative_path": path.relative_to(corpus_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def nvidia_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL).strip().split(", ")
        return {
            "name": values[0],
            "driver": values[1],
            "memory_total_mb": float(values[2]),
            "memory_used_mb": float(values[3]),
            "utilization_percent": float(values[4]),
            "temperature_c": float(values[5]),
            "power_w": None if values[6] == "[N/A]" else float(values[6]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def hardware_snapshot():
    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gb": round(memory.total / 1024**3, 2),
        "gpu": nvidia_snapshot(),
    }


def start_resource_sampler(interval=0.2):
    samples = []
    stop_event = threading.Event()
    process = psutil.Process()

    def sample():
        while not stop_event.wait(interval):
            gpu = nvidia_snapshot()
            samples.append({
                "time": time.time(),
                "process_rss_mb": process.memory_info().rss / 1024**2,
                "system_ram_percent": psutil.virtual_memory().percent,
                "gpu_utilization_percent": gpu.get("utilization_percent"),
                "gpu_memory_used_mb": gpu.get("memory_used_mb"),
                "gpu_temperature_c": gpu.get("temperature_c"),
            })

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return stop_event, thread, samples


def summarize_samples(samples):
    def values(field):
        return [sample[field] for sample in samples if sample.get(field) is not None]

    result = {"sample_count": len(samples)}
    for field in (
        "process_rss_mb",
        "system_ram_percent",
        "gpu_utilization_percent",
        "gpu_memory_used_mb",
        "gpu_temperature_c",
    ):
        field_values = values(field)
        result[f"average_{field}"] = (
            round(statistics.mean(field_values), 3) if field_values else None)
        result[f"peak_{field}"] = (
            round(max(field_values), 3) if field_values else None)
    return result


def evenly_limit(items, maximum):
    if maximum <= 0 or len(items) <= maximum:
        return list(items)
    positions = [
        round(index * (len(items) - 1) / (maximum - 1))
        for index in range(maximum)
    ]
    return [items[position] for position in positions]


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def prepare_environment(model_cache):
    cache = str(Path(model_cache).resolve())
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = str(Path(cache) / "hub")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache
    os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "mpl_temp")
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


def extract_selected_files(files):
    from pdf_extract import extract_university_content

    extracted = []
    for path in files:
        started = time.perf_counter()
        text = extract_university_content(str(path), save_debug=False)
        elapsed = time.perf_counter() - started
        if not text:
            raise RuntimeError(f"No text extracted from {path}")
        extracted.append({
            "path": path,
            "text": text,
            "characters": len(text),
            "seconds": elapsed,
        })
    return extracted


def benchmark_chunking(extracted, strategies):
    from chunking_config import chunk_text, create_chunker_tools
    from project_config import get_chunking_profile, load_project_config

    project_config = load_project_config()
    results = {}
    chunk_sets = {}
    for strategy in strategies:
        profile = get_chunking_profile(strategy, config=project_config)
        chunker, refinery = create_chunker_tools(profile)
        per_file = []
        all_chunks = []
        started = time.perf_counter()
        for item in extracted:
            file_started = time.perf_counter()
            chunks = chunk_text(item["text"], chunker, refinery)
            per_file.append({
                "file": item["path"].name,
                "characters": item["characters"],
                "chunks": len(chunks),
                "seconds": round(time.perf_counter() - file_started, 6),
            })
            all_chunks.extend(chunks)
        elapsed = time.perf_counter() - started
        chunk_sets[strategy] = all_chunks
        results[strategy] = {
            "profile": profile,
            "files": len(extracted),
            "chunks": len(all_chunks),
            "seconds": round(elapsed, 6),
            "files_per_second": round(len(extracted) / elapsed, 4),
            "chunks_per_second": round(len(all_chunks) / elapsed, 4),
            "average_chunk_characters": round(
                statistics.mean(len(chunk) for chunk in all_chunks), 2
            ) if all_chunks else 0,
            "per_file": per_file,
        }
    return results, chunk_sets


def encode_once(model, chunks, batch_size, max_length):
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    stop_event, thread, samples = start_resource_sampler()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            encoded = model.encode(
                chunks,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        dense = encoded["dense_vecs"]
        shape = list(dense.shape)
        del dense, encoded
        return {
            "status": "ok",
            "seconds": round(elapsed, 6),
            "chunks_per_second": round(len(chunks) / elapsed, 4),
            "embedding_shape": shape,
            "torch_peak_allocated_mb": round(
                torch.cuda.max_memory_allocated() / 1024**2, 3
            ) if torch.cuda.is_available() else 0,
            "torch_peak_reserved_mb": round(
                torch.cuda.max_memory_reserved() / 1024**2, 3
            ) if torch.cuda.is_available() else 0,
        }
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "oom", "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        stop_event.set()
        thread.join()
        gc.collect()
        if "samples" in locals():
            resource_summary = summarize_samples(samples)
            if "elapsed" in locals():
                resource_summary["wall_seconds"] = round(elapsed, 6)
            encode_once.last_resource_summary = resource_summary


def benchmark_encoding(chunks, batches, repeats, model_cache, max_length):
    import torch
    from FlagEmbedding import BGEM3FlagModel
    from project_config import load_project_config

    config = load_project_config()
    model_name = config["model"]["name"]
    load_started = time.perf_counter()
    model = BGEM3FlagModel(
        model_name,
        cache_dir=str(Path(model_cache).resolve()),
        use_fp16=torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        model.model.to("cuda")
    model.model.eval()
    model_load_seconds = time.perf_counter() - load_started

    warmup_chunks = chunks[:min(64, len(chunks))]
    warmup = encode_once(model, warmup_chunks, min(16, max(batches)), max_length)
    rows = []
    for repeat in range(repeats):
        order = batches if repeat % 2 == 0 else list(reversed(batches))
        for batch_size in order:
            result = encode_once(model, chunks, batch_size, max_length)
            result.update({
                "batch_size": batch_size,
                "repeat": repeat + 1,
                "chunks": len(chunks),
                "resources": getattr(
                    encode_once, "last_resource_summary", {}),
            })
            rows.append(result)

    summaries = []
    for batch_size in batches:
        matching = [
            row for row in rows
            if row["batch_size"] == batch_size and row["status"] == "ok"
        ]
        failures = [
            row for row in rows
            if row["batch_size"] == batch_size and row["status"] != "ok"
        ]
        throughputs = [row["chunks_per_second"] for row in matching]
        summaries.append({
            "batch_size": batch_size,
            "successful_repeats": len(matching),
            "failed_repeats": len(failures),
            "mean_chunks_per_second": round(
                statistics.mean(throughputs), 4) if throughputs else None,
            "median_chunks_per_second": round(
                statistics.median(throughputs), 4) if throughputs else None,
            "mean_seconds": round(statistics.mean(
                row["seconds"] for row in matching), 6) if matching else None,
            "peak_torch_reserved_mb": max(
                (row["torch_peak_reserved_mb"] for row in matching),
                default=None,
            ),
            "peak_nvidia_memory_used_mb": max(
                (row["resources"].get("peak_gpu_memory_used_mb") or 0
                 for row in matching),
                default=None,
            ),
            "average_gpu_utilization_percent": round(statistics.mean(
                row["resources"].get("average_gpu_utilization_percent") or 0
                for row in matching
            ), 3) if matching else None,
        })
    return {
        "model": model_name,
        "model_load_seconds": round(model_load_seconds, 4),
        "warmup": warmup,
        "chunks": len(chunks),
        "max_length": max_length,
        "runs": rows,
        "summary": summaries,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def copy_sample(files, corpus_root, destination):
    destination = Path(destination)
    corpus_root = Path(corpus_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        relative = path.relative_to(corpus_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sidecar = Path(str(path) + ".metadata.json")
        if sidecar.is_file():
            shutil.copy2(sidecar, Path(str(target) + ".metadata.json"))


def parse_index_report(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if ": " not in line or line.startswith(("SUCCESS:", "ERROR:", "SKIPPED:")):
            continue
        key, value = line.split(": ", 1)
        values[key] = value
    return values


def run_full_index_benchmarks(
    specifications, base_config, sample_corpus, output_dir, model_cache
):
    output_dir = Path(output_dir)
    configs_dir = output_dir / "configs"
    indexes_dir = output_dir / "indexes"
    logs_dir = output_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for order, specification in enumerate(specifications, 1):
        strategy, batch_text = specification.split(":", 1)
        batch_size = int(batch_text)
        config = copy.deepcopy(base_config)
        profile = config["chunking"]["profiles"][strategy]
        version = f"benchmark_{order}_{strategy}_b{batch_size}"
        profile["index_version"] = version
        config["chunking"]["default_strategy"] = strategy
        config["indexing"]["encode_batch_size"] = batch_size
        config["indexing"]["faculty_index_root"] = str(indexes_dir)
        config["model"]["cache_dir"] = str(Path(model_cache).resolve())
        config_path = configs_dir / f"{version}.json"
        write_json(config_path, config)
        log_path = logs_dir / f"{version}.log"
        environment = os.environ.copy()
        environment["PLAGIARISM_CONFIG"] = str(config_path)
        environment["PLAGIARISM_CHUNKING_STRATEGY"] = strategy
        command = [
            sys.executable,
            str(RUNTIME_ROOT / "vector_db.py"),
            str(sample_corpus),
            "--faculty",
            "benchmark",
        ]
        started = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - started
        report_path = (
            indexes_dir / version / "benchmark" / "vector_db_run_report.txt")
        results.append({
            "specification": specification,
            "strategy": strategy,
            "batch_size": batch_size,
            "return_code": completed.returncode,
            "subprocess_wall_seconds": round(elapsed, 4),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "report_path": str(report_path),
            "report": parse_index_report(report_path)
            if report_path.is_file() else {},
        })
    return results


def render_vector_report(result):
    lines = [
        "PLAGIARISM RUNTIME VECTOR BENCHMARK",
        "",
        f"Generated: {result['generated_at']}",
        f"Corpus: {result['corpus']}",
        f"Sample files: {len(result['manifest'])}",
        f"Hardware: {result['hardware']}",
        "",
        "FAIRNESS",
        "- Every strategy used the exact files and SHA-256 manifest below.",
        "- Every GPU batch encoded the exact same ordered chunk list.",
        "- Batch order was reversed on alternating repeats to reduce heat bias.",
        "- This benchmark measures speed/resources, not plagiarism accuracy.",
        "",
        "CHUNKING",
    ]
    for strategy, item in result["chunking"].items():
        lines.append(
            f"{strategy}: chunks={item['chunks']} | seconds={item['seconds']} | "
            f"avg_chars={item['average_chunk_characters']} | "
            f"chunks/s={item['chunks_per_second']}"
        )
    lines.extend(["", "GPU BATCH ENCODING"])
    for item in result["encoding"]["summary"]:
        lines.append(
            f"batch={item['batch_size']}: mean_chunks/s="
            f"{item['mean_chunks_per_second']} | mean_s={item['mean_seconds']} | "
            f"torch_reserved_mb={item['peak_torch_reserved_mb']} | "
            f"nvidia_mem_mb={item['peak_nvidia_memory_used_mb']} | "
            f"gpu_util={item['average_gpu_utilization_percent']} | "
            f"failures={item['failed_repeats']}"
        )
    if result.get("recommendation"):
        lines.extend(["", "RECOMMENDATION", result["recommendation"]])
    if result.get("full_index_runs"):
        lines.extend(["", "FULL VECTOR_DB RUNS"])
        for item in result["full_index_runs"]:
            report = item["report"]
            lines.append(
                f"{item['specification']}: return={item['return_code']} | "
                f"wall_s={item['subprocess_wall_seconds']} | "
                f"report_runtime_s={report.get('Total runtime seconds')} | "
                f"chunks={report.get('Chunks encoded this run')} | "
                f"gpu_peak_mb={report.get('Peak nvidia-smi GPU memory used MB')}"
            )
    lines.extend(["", "MANIFEST"])
    for item in result["manifest"]:
        lines.append(
            f"{item['relative_path']} | bytes={item['size_bytes']} | "
            f"sha256={item['sha256']}"
        )
    return "\n".join(lines) + "\n"


def recommend_batch(summary, configured_batch):
    valid = [
        item for item in summary
        if item["mean_chunks_per_second"] is not None
        and item["failed_repeats"] == 0
    ]
    if not valid:
        return "No batch completed successfully; keep the current config and inspect logs."
    fastest = max(valid, key=lambda item: item["mean_chunks_per_second"])
    configured = next(
        (item for item in valid if item["batch_size"] == configured_batch), None)
    if configured:
        improvement = (
            fastest["mean_chunks_per_second"]
            / configured["mean_chunks_per_second"] - 1
        ) * 100
        if fastest["batch_size"] != configured_batch and improvement < 5:
            return (
                f"Keep batch {configured_batch}: batch {fastest['batch_size']} was only "
                f"{improvement:.2f}% faster, below the 5% stability threshold."
            )
        if fastest["batch_size"] != configured_batch:
            return (
                f"Use batch {fastest['batch_size']}: it was {improvement:.2f}% faster "
                f"than configured batch {configured_batch} without failures."
            )
    return f"Keep batch {fastest['batch_size']}; it was the fastest stable option."


def run_vector_benchmark(args):
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_environment(args.model_cache)
    corpus = Path(args.corpus).resolve()
    files = discover_files(corpus)
    if not files:
        raise SystemExit(f"No PDF/TXT files found in {corpus}")
    selected = stratified_size_sample(files, args.sample_size)
    manifest = manifest_for(selected, corpus)
    write_json(output_dir / "benchmark_manifest.json", manifest)
    sample_corpus = output_dir / "sample_corpus"
    copy_sample(selected, corpus, sample_corpus)

    extraction_started = time.perf_counter()
    extracted = extract_selected_files(selected)
    extraction_seconds = time.perf_counter() - extraction_started
    chunking, chunk_sets = benchmark_chunking(extracted, args.strategies)
    if args.encode_strategy not in chunk_sets:
        raise SystemExit(
            f"--encode-strategy {args.encode_strategy} must be in --strategies")
    encode_chunks = evenly_limit(
        chunk_sets[args.encode_strategy], args.max_encode_chunks)
    encoding = benchmark_encoding(
        encode_chunks,
        args.batches,
        args.repeats,
        args.model_cache,
        args.max_length,
    )

    from project_config import load_project_config
    base_config = load_project_config()
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": str(corpus),
        "output": str(output_dir),
        "hardware": hardware_snapshot(),
        "manifest": manifest,
        "extraction": {
            "seconds": round(extraction_seconds, 6),
            "files": len(extracted),
            "characters": sum(item["characters"] for item in extracted),
            "per_file": [{
                "file": item["path"].name,
                "characters": item["characters"],
                "seconds": round(item["seconds"], 6),
            } for item in extracted],
        },
        "chunking": chunking,
        "encoding": encoding,
        "configured_batch": base_config["indexing"]["encode_batch_size"],
    }
    result["recommendation"] = recommend_batch(
        encoding["summary"], result["configured_batch"])
    if args.full_index:
        result["full_index_runs"] = run_full_index_benchmarks(
            args.full_index,
            base_config,
            sample_corpus,
            output_dir / "full_index",
            args.model_cache,
        )
    write_json(output_dir / "benchmark_results.json", result)
    (output_dir / "BENCHMARK_REPORT.txt").write_text(
        render_vector_report(result), encoding="utf-8")
    print(render_vector_report(result))
    print(f"Results: {output_dir}")


def run_api_benchmark(args):
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    submissions = Path(args.submissions).resolve()
    files = discover_files(submissions)
    if not files:
        raise SystemExit(f"No PDF/TXT files found in {submissions}")
    selected = stratified_size_sample(files, args.sample_size)
    manifest = manifest_for(selected, submissions)
    rows = []
    session = requests.Session()
    for repeat in range(1, args.repeats + 1):
        order = selected if repeat % 2 else list(reversed(selected))
        for path in order:
            stop_event, thread, samples = start_resource_sampler()
            started = time.perf_counter()
            try:
                with open(path, "rb") as handle:
                    response = session.post(
                        args.api_url,
                        params={
                            "faculty": args.faculty,
                            "retrieval_strategy": args.retrieval_strategy,
                            "evidence_profile": args.evidence_profile,
                        },
                        files={"file": (path.name, handle)},
                        timeout=args.timeout,
                    )
                elapsed = time.perf_counter() - started
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"error": response.text[:500]}
                performance = payload.get("performance") or {}
                rows.append({
                    "file": path.name,
                    "repeat": repeat,
                    "status_code": response.status_code,
                    "wall_seconds": round(elapsed, 6),
                    "result_count": len(payload.get("results") or []),
                    "performance": performance,
                    "error": payload.get("error"),
                    "resources": summarize_samples(samples),
                })
            finally:
                stop_event.set()
                thread.join()
    successful = [row for row in rows if row["status_code"] == 200]
    latencies = [row["wall_seconds"] for row in successful]
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "api_url": args.api_url,
        "faculty": args.faculty,
        "hardware": hardware_snapshot(),
        "manifest": manifest,
        "runs": rows,
        "summary": {
            "requests": len(rows),
            "successful": len(successful),
            "failed": len(rows) - len(successful),
            "mean_seconds": round(statistics.mean(latencies), 4)
            if latencies else None,
            "p50_seconds": round(percentile(latencies, 50), 4)
            if latencies else None,
            "p95_seconds": round(percentile(latencies, 95), 4)
            if latencies else None,
        },
        "note": "Latency/resource benchmark only; this does not calculate precision or recall.",
    }
    write_json(output_dir / "api_benchmark_results.json", result)
    lines = [
        "PLAGIARISM RUNTIME API BENCHMARK",
        json.dumps(result["summary"], ensure_ascii=False),
        result["note"],
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['file']} repeat={row['repeat']} status={row['status_code']} "
            f"wall_s={row['wall_seconds']} results={row['result_count']}"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "API_BENCHMARK_REPORT.txt").write_text(
        report, encoding="utf-8")
    print(report)
    print(f"Results: {output_dir}")


def default_output(prefix):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(ROOT / "results" / f"{prefix}_{stamp}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fair, reproducible vector/API performance benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    vector = subparsers.add_parser("vector")
    vector.add_argument("--corpus", required=True)
    vector.add_argument("--model-cache", required=True)
    vector.add_argument("--sample-size", type=int, default=8)
    vector.add_argument(
        "--strategies",
        nargs="+",
        default=["semantic", "token", "recursive_character"],
        choices=["semantic", "token", "recursive_character"],
    )
    vector.add_argument("--encode-strategy", default="semantic")
    vector.add_argument(
        "--batches", nargs="+", type=int, default=[16, 24, 32, 40, 48])
    vector.add_argument("--repeats", type=int, default=2)
    vector.add_argument("--max-encode-chunks", type=int, default=1024)
    vector.add_argument("--max-length", type=int, default=512)
    vector.add_argument(
        "--full-index",
        nargs="*",
        default=[],
        help="Full pipeline specs, e.g. semantic:32 semantic:48 token:32",
    )
    vector.add_argument("--output", default=default_output("vector"))

    api = subparsers.add_parser("api")
    api.add_argument("--api-url", default="http://127.0.0.1:8000/scan")
    api.add_argument("--submissions", required=True)
    api.add_argument("--faculty", required=True)
    api.add_argument("--sample-size", type=int, default=5)
    api.add_argument("--repeats", type=int, default=1)
    api.add_argument("--timeout", type=float, default=600)
    api.add_argument("--retrieval-strategy", default="shortlist")
    api.add_argument("--evidence-profile", default="balanced")
    api.add_argument("--output", default=default_output("api"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "vector":
        run_vector_benchmark(args)
    else:
        run_api_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
