import argparse
import json
import os
import subprocess
import sys
import time

from corpus_metadata import METADATA_SUFFIX, normalize_faculty, write_json_file
from dspace_sync import sync_dspace
from project_config import load_project_config, resolve_project_path


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_CONFIG = load_project_config()
SCHEDULER_CONFIG = PROJECT_CONFIG["scheduler"]
DEFAULT_STATE_PATH = resolve_project_path(SCHEDULER_CONFIG["state_file"])
DEFAULT_DSPACE_CONFIG = resolve_project_path(
    SCHEDULER_CONFIG["dspace_config"])


def corpus_snapshot(corpus_dir):
    snapshot = {}
    if not os.path.isdir(corpus_dir):
        return snapshot
    for current_root, directories, filenames in os.walk(corpus_dir):
        directories[:] = sorted(
            directory
            for directory in directories
            if not directory.startswith(".")
        )
        for filename in sorted(filenames):
            lower_name = filename.lower()
            if not (
                lower_name.endswith((".pdf", ".txt"))
                or lower_name.endswith(METADATA_SUFFIX)
            ):
                continue
            path = os.path.join(current_root, filename)
            stat = os.stat(path)
            relative_path = os.path.relpath(
                path, corpus_dir).replace("\\", "/")
            snapshot[relative_path] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return snapshot


def discover_faculty_corpora(corpus_root):
    if not os.path.isdir(corpus_root):
        return {}
    result = {}
    for name in sorted(os.listdir(corpus_root)):
        path = os.path.join(corpus_root, name)
        if name.startswith(".") or not os.path.isdir(path):
            continue
        result[normalize_faculty(name)] = path
    return result


def load_state(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def run_vector_index(corpus_dir, faculty, strategy):
    environment = os.environ.copy()
    environment["PLAGIARISM_CHUNKING_STRATEGY"] = strategy
    command = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "vector_db.py"),
        corpus_dir,
        "--faculty",
        faculty,
    ]
    print("[scheduler] Running:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


def snapshot_change_counts(previous, current):
    before_files = set(previous or {})
    after_files = set(current)
    changed_files = {
        path for path in before_files & after_files
        if (previous or {}).get(path) != current.get(path)
    }
    return {
        "new": len(after_files - before_files),
        "updated": len(changed_files),
        "removed": len(before_files - after_files),
    }


def run_cycle(args):
    corpus_root = args.corpus_root
    dspace_had_errors = False
    if args.dspace_config:
        print("[scheduler] Synchronizing all DSpace items...")
        report = sync_dspace(args.dspace_config)
        corpus_root = report["output_dir"]
        print(
            "[scheduler] DSpace: "
            f"downloaded={report['downloaded']} | "
            f"unchanged={report['skipped_unchanged']} | "
            f"metadata_updated={report['metadata_updated']} | "
            f"errors={len(report['errors'])}"
        )
        if report["errors"]:
            for error in report["errors"][:10]:
                print(f"[scheduler] DSpace error: {error}")
            dspace_had_errors = True

    if not corpus_root:
        print("[scheduler] No corpus root. Configure DSpace or --corpus-root.")
        return False
    corpus_root = os.path.abspath(corpus_root)
    faculty_corpora = discover_faculty_corpora(corpus_root)
    if not faculty_corpora:
        print(f"[scheduler] No faculty folders found in {corpus_root}")
        return False

    state = load_state(args.state_file)
    faculty_states = state.setdefault("faculties", {})
    strategy_changed = state.get("strategy") != args.strategy
    all_successful = not dspace_had_errors
    indexed_count = 0

    for faculty, faculty_dir in faculty_corpora.items():
        current_snapshot = corpus_snapshot(faculty_dir)
        previous_snapshot = faculty_states.get(faculty, {}).get("snapshot")
        if (
            previous_snapshot == current_snapshot
            and not args.force
            and not strategy_changed
        ):
            print(f"[scheduler] {faculty}: no changes; index is current.")
            continue

        counts = snapshot_change_counts(previous_snapshot, current_snapshot)
        print(
            f"[scheduler] {faculty}: new={counts['new']} | "
            f"updated={counts['updated']} | removed={counts['removed']}"
        )
        return_code = run_vector_index(
            faculty_dir, faculty, args.strategy)
        if return_code != 0:
            all_successful = False
            print(
                f"[scheduler] {faculty}: vector_db.py failed with exit code "
                f"{return_code}; retrying next cycle."
            )
            continue

        indexed_count += 1
        faculty_states[faculty] = {
            "corpus_dir": faculty_dir,
            "snapshot": current_snapshot,
            "last_success_unix": time.time(),
        }
        state.update({
            "corpus_root": corpus_root,
            "strategy": args.strategy,
            "last_cycle_unix": time.time(),
        })
        write_json_file(args.state_file, state)

    print(
        f"[scheduler] Cycle complete: faculties={len(faculty_corpora)} | "
        f"indexes_updated={indexed_count} | success={all_successful}"
    )
    return all_successful


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Periodically sync DSpace and build one vector DB per faculty."
        ))
    parser.add_argument(
        "--dspace-config",
        default=(DEFAULT_DSPACE_CONFIG if os.path.isfile(
            DEFAULT_DSPACE_CONFIG) else None),
    )
    parser.add_argument(
        "--corpus-root",
        "--corpus-dir",
        dest="corpus_root",
        default=None,
        help="Local DSpace mirror root; useful for offline tests.",
    )
    parser.add_argument(
        "--strategy",
        choices=("recursive_character", "semantic", "token"),
        default=SCHEDULER_CONFIG["strategy"],
    )
    parser.add_argument(
        "--interval-days",
        type=float,
        default=float(SCHEDULER_CONFIG["interval_days"]),
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=None,
        help="Testing override; production default is --interval-days 30.",
    )
    parser.add_argument("--state-file", default=DEFAULT_STATE_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.state_file = os.path.abspath(args.state_file)
    if args.corpus_root:
        args.corpus_root = os.path.abspath(args.corpus_root)
    if args.dspace_config:
        args.dspace_config = os.path.abspath(args.dspace_config)

    while True:
        started = time.time()
        try:
            run_cycle(args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[scheduler] Cycle failed: {type(exc).__name__}: {exc}")
        if args.once:
            break
        elapsed = time.time() - started
        interval_seconds = (
            args.interval_minutes * 60
            if args.interval_minutes is not None
            else args.interval_days * 24 * 60 * 60
        )
        sleep_seconds = max(5.0, interval_seconds - elapsed)
        print(f"[scheduler] Next check in {sleep_seconds / 86400:.2f} day(s).")
        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\n[scheduler] Stopped by user.")
            break


if __name__ == "__main__":
    main()
