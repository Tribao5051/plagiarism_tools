import argparse
import json
import os
import subprocess
import sys
import time

from corpus_metadata import normalize_faculty, write_json_file
from dspace_sync import load_config, resolve_output_dir, sync_dspace
from project_config import DEFAULT_CONFIG_PATH, load_project_config


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def configure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError):
        pass


def collection_corpora(sync_report):
    output_directory = sync_report["output_dir"]
    corpora = {}
    for collection_uuid, collection in sync_report["collections"].items():
        folder = collection.get("folder")
        if not folder or not collection.get("selected_files"):
            continue
        path = os.path.join(output_directory, folder)
        if not os.path.isdir(path):
            continue
        corpus = corpora.setdefault(folder, {
            "folder": folder,
            "path": path,
            "collection_uuids": [],
            "collection_names": [],
        })
        corpus["collection_uuids"].append(collection_uuid)
        name = collection.get("name")
        if name and name not in corpus["collection_names"]:
            corpus["collection_names"].append(name)
    return [corpora[key] for key in sorted(corpora)]


def run_vector_database(corpus, strategy):
    environment = os.environ.copy()
    environment["PLAGIARISM_CHUNKING_STRATEGY"] = strategy
    command = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "vector_db.py"),
        corpus["path"],
        "--faculty",
        normalize_faculty(corpus["folder"]),
    ]
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return {
        "folder": corpus["folder"],
        "path": corpus["path"],
        "collection_uuids": corpus["collection_uuids"],
        "collection_names": corpus["collection_names"],
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def write_pipeline_report(output_directory, sync_report, index_results):
    report = {
        "finished_at_unix": time.time(),
        "sync": sync_report,
        "indexes": index_results,
        "success": (
            not sync_report["errors"]
            and all(item["return_code"] == 0 for item in index_results)
        ),
    }
    path = os.path.join(
        output_directory, "dspace_vector_pipeline_report.json")
    write_json_file(path, report)
    return report


def run_pipeline(config_path, dry_run=False, max_downloads=None,
                 max_per_collection=None):
    project_config = load_project_config(config_path)
    dspace_config = load_config(config_path)
    output_directory = resolve_output_dir(dspace_config, config_path)
    sync_report = sync_dspace(
        config_path=config_path,
        dry_run=dry_run,
        max_downloads=max_downloads,
        max_per_collection=max_per_collection,
    )
    if dry_run:
        return {"sync": sync_report, "indexes": [], "success": True}
    if (
        sync_report["errors"]
        and dspace_config.get("stop_indexing_on_sync_errors", True)
    ):
        return write_pipeline_report(
            output_directory, sync_report, [])

    strategy = project_config["scheduler"]["strategy"]
    corpora = collection_corpora(sync_report)
    index_results = []
    for corpus in corpora:
        index_results.append(run_vector_database(corpus, strategy))
    return write_pipeline_report(
        output_directory, sync_report, index_results)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-downloads", type=int)
    parser.add_argument("--max-per-collection", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    configure_stdout()
    args = parse_args(argv)
    report = run_pipeline(
        os.path.abspath(args.config),
        dry_run=args.dry_run,
        max_downloads=args.max_downloads,
        max_per_collection=args.max_per_collection,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
