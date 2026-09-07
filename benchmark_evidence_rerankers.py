import argparse
import copy
import json
import math
import os
import pathlib
import sys
import time
import warnings
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from detection_module_diff_p4 import (
    _evidence_confidence,
    select_display_evidence,
)
from evidence_reranker import (
    FEATURE_NAMES,
    agreement_score,
    build_candidate_records,
    select_ranked_records,
)


RUNTIME_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_SPLIT = RUNTIME_ROOT / "reranker_experiment_split.json"
LABEL_SCHEMA_VERSION = "student_and_source_text_v2"
CHARACTER_PRECISION_NOTE = (
    "Not computed: PAN XML offsets refer to original raw text, so character "
    "precision requires an explicit mapping from normalized tool text back "
    "to the original submission and source text."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare evidence-only reranking strategies.")
    parser.add_argument(
        "--raw-dir",
        action="append",
        required=True,
        help="Directory containing saved API raw_json files; repeatable.",
    )
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high-min-recall", type=float, default=0.30)
    parser.add_argument("--cross-encoder-model", default=None)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=8)
    parser.add_argument(
        "--cross-encoder-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def validate_experiment_split(split):
    required = ("train", "validation", "test")
    missing = [name for name in required if name not in split]
    if missing:
        raise ValueError(
            "Missing split partition(s): " + ", ".join(missing))

    partitions = {name: list(split[name]) for name in required}
    for name, values in partitions.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate documents in {name} split")

    for left_index, left_name in enumerate(required):
        for right_name in required[left_index + 1:]:
            overlap = sorted(
                set(partitions[left_name]) & set(partitions[right_name])
            )
            if overlap:
                raise ValueError(
                    f"Split partitions overlap: {left_name} and "
                    f"{right_name}: {', '.join(overlap)}"
                )
    return partitions


def import_evaluator(evaluator_root):
    evaluator_root = os.path.abspath(evaluator_root)
    if evaluator_root not in sys.path:
        sys.path.append(evaluator_root)
    from evaluate_pan_side_by_side import (  # noqa: PLC0415
        TEXT_MATCH_THRESHOLD,
        containment_from_profiles,
        flatten_predictions,
        parse_ground_truth,
        text_profile,
        token_containment,
    )
    return {
        "threshold": TEXT_MATCH_THRESHOLD,
        "containment": containment_from_profiles,
        "flatten": flatten_predictions,
        "parse_truth": parse_ground_truth,
        "text_profile": text_profile,
        "token_containment": token_containment,
    }


def load_raw_cases(raw_directories):
    paths = {}
    for directory in raw_directories:
        for path in sorted(pathlib.Path(directory).glob("*.json")):
            paths.setdefault(path.stem, path)
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }


def candidate_confidence(evidence, result):
    candidate = dict(evidence)
    candidate["source_semantic_coverage"] = (
        float(result.get("semantic_coverage") or 0.0) / 100.0)
    candidate["source_verbatim_score"] = (
        float(result.get("verbatim_score") or 0.0) / 100.0)
    return _evidence_confidence(candidate)[0]


def prediction_text(evidence, prefix):
    if prefix == "student":
        return (
            evidence.get("student_match_text")
            or evidence.get("student_text")
            or evidence.get("student")
            or evidence.get("text")
            or ""
        )
    return (
        evidence.get("source_match_text")
        or evidence.get("source_text")
        or evidence.get("source")
        or ""
    )


def annotate_truth_matches(records, truths, evaluator):
    truth_profiles = []
    for truth in truths:
        truth_profiles.append((
            evaluator["text_profile"](truth["student_text"]),
            evaluator["text_profile"](truth["source_text"]),
        ))

    for record in records:
        evidence = record["evidence"]
        student_text = prediction_text(evidence, "student")
        source_text = prediction_text(evidence, "source")
        student_profile = evaluator["text_profile"](student_text)
        source_profile = evaluator["text_profile"](source_text)
        matches = {}
        for truth_index, truth in enumerate(truths):
            if record["source_file"] != truth["source_file"]:
                continue
            truth_student_profile, truth_source_profile = truth_profiles[
                truth_index]
            student_score = max(
                evaluator["containment"](
                    student_profile, truth_student_profile),
                evaluator["containment"](
                    truth_student_profile, student_profile),
            )
            source_score = max(
                evaluator["containment"](
                    source_profile, truth_source_profile),
                evaluator["containment"](
                    truth_source_profile, source_profile),
            )
            if truth.get("obfuscation") == "high":
                student_score = max(
                    student_score,
                    evaluator["token_containment"](
                        student_text, truth["student_text"]),
                )
                source_score = max(
                    source_score,
                    evaluator["token_containment"](
                        source_text, truth["source_text"]),
                )
            if (
                student_score >= evaluator["threshold"]
                and source_score >= evaluator["threshold"]
            ):
                matches[truth_index] = (
                    0.7 * student_score + 0.3 * source_score)
        record["truth_scores"] = matches


def assign_best_candidate_labels(records):
    best_by_truth = {}
    for record in records:
        for truth_index, match_score in record["truth_scores"].items():
            previous = best_by_truth.get(truth_index)
            candidate_key = (
                match_score,
                float(record["evidence"].get("confidence_score") or 0.0),
                int(record["evidence"].get("evidence_count") or 0),
            )
            if previous is None or candidate_key > previous[0]:
                best_by_truth[truth_index] = (
                    candidate_key, record["record_id"])
    positive_ids = {
        record_id for _key, record_id in best_by_truth.values()
    }
    for record in records:
        record["label"] = int(record["record_id"] in positive_ids)


def build_cases(raw_cases, split, evaluation_dir, source_dir, evaluator):
    requested = {
        name for partition in split.values() for name in partition
    }
    missing = sorted(requested - set(raw_cases))
    if missing:
        raise FileNotFoundError(
            "Missing raw JSON for: " + ", ".join(missing))

    cases = {}
    for case_name in sorted(requested):
        api_result = raw_cases[case_name]
        records = build_candidate_records(
            api_result.get("results", ()),
            confidence_scorer=candidate_confidence,
        )
        truths = evaluator["parse_truth"](
            os.path.join(evaluation_dir, case_name + ".xml"),
            os.path.join(evaluation_dir, case_name + ".txt"),
            source_dir,
        )
        annotate_truth_matches(records, truths, evaluator)
        assign_best_candidate_labels(records)
        cases[case_name] = {
            "api_result": api_result,
            "records": records,
            "truths": truths,
        }
    return cases


def training_matrix(cases, names, include_cross=False):
    records = [
        record
        for name in names
        for record in cases[name]["records"]
    ]
    features = []
    for record in records:
        row = list(record["features"])
        if include_cross:
            row.append(float(record.get("cross_encoder_score") or 0.0))
        features.append(row)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray([record["label"] for record in records], dtype=np.int64),
        records,
    )


def document_balanced_weights(records, labels):
    weights = compute_sample_weight(class_weight="balanced", y=labels)
    counts = Counter(record["case_name"] for record in records)
    mean_count = sum(counts.values()) / max(1, len(counts))
    for index, record in enumerate(records):
        weights[index] *= mean_count / counts[record["case_name"]]
    return weights / np.mean(weights)


def build_models(seed):
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=4000, C=1.0, random_state=seed)),
        ]),
        "linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                kernel="linear", C=0.5, probability=True,
                random_state=seed)),
        ]),
        "rbf_svm": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                kernel="rbf", C=2.0, gamma="scale", probability=True,
                random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=3,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=8,
            l2_regularization=0.5,
            random_state=seed,
        ),
        "gaussian_nb": Pipeline([
            ("scale", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "mlp": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                early_stopping=True,
                validation_fraction=0.2,
                max_iter=600,
                random_state=seed,
            )),
        ]),
    }


def fit_model(model, features, labels, sample_weights):
    if isinstance(model, Pipeline):
        model.fit(features, labels, model__sample_weight=sample_weights)
    else:
        model.fit(features, labels, sample_weight=sample_weights)


def model_scores(model, features):
    probabilities = model.predict_proba(features)
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    return probabilities[:, 1].astype(float)


def add_record_case_names(cases):
    for case_name, case in cases.items():
        for record in case["records"]:
            record["case_name"] = case_name


def score_by_case(cases, names, scorer):
    return {
        name: [float(scorer(record)) for record in cases[name]["records"]]
        for name in names
    }


def model_score_by_case(model, cases, names, include_cross=False):
    output = {}
    for name in names:
        features, _labels, _records = training_matrix(
            cases, [name], include_cross=include_cross)
        output[name] = model_scores(model, features).tolist()
    return output


def precision_metric_fields(evidence_hit_precision):
    return {
        "precision": evidence_hit_precision,
        "evidence_hit_precision": evidence_hit_precision,
        "character_precision": None,
        "character_precision_note": CHARACTER_PRECISION_NOTE,
    }


def evidence_hit_metrics_from_predictions(
        truths, predictions, evaluator, allow_prediction_reuse=True):
    records = [
        {
            "record_id": prediction.get(
                "prediction_id", f"prediction-{index}"),
            "source_file": prediction.get("source_file", ""),
            "evidence": prediction,
        }
        for index, prediction in enumerate(predictions)
    ]
    annotate_truth_matches(records, truths, evaluator)

    candidate_pairs = sorted(
        (
            match_score,
            truth_index,
            record["record_id"],
        )
        for record in records
        for truth_index, match_score in record["truth_scores"].items()
    )
    assigned_truths = set()
    hit_records = set()
    for _score, truth_index, record_id in reversed(candidate_pairs):
        if truth_index in assigned_truths:
            continue
        if not allow_prediction_reuse and record_id in hit_records:
            continue
        assigned_truths.add(truth_index)
        hit_records.add(record_id)

    evidence_count = len(predictions)
    truth_count = len(truths)
    evidence_hit_precision = (
        len(hit_records) / evidence_count
        if evidence_count else (1.0 if not truth_count else 0.0)
    )
    recall = len(assigned_truths) / truth_count if truth_count else 1.0
    return {
        "truth_segments": truth_count,
        "caught_truth_segments": len(assigned_truths),
        "evidence_count": evidence_count,
        "evidence_true_positive": len(hit_records),
        "evidence_false_positive": max(
            0, evidence_count - len(hit_records)),
        **precision_metric_fields(evidence_hit_precision),
        "recall": recall,
        "f1": (
            2 * evidence_hit_precision * recall
            / (evidence_hit_precision + recall)
            if evidence_hit_precision + recall else 0.0
        ),
    }


def fast_metrics(cases, names, scores_by_case, selection):
    truth_count = 0
    caught_count = 0
    evidence_count = 0
    evidence_hits = 0
    per_file = []
    for name in names:
        case = cases[name]
        selected = select_ranked_records(
            case["records"], scores_by_case[name], **selection)
        best_by_truth = {}
        for record in selected:
            for truth_index, match_score in record["truth_scores"].items():
                previous = best_by_truth.get(truth_index)
                if previous is None or match_score > previous[0]:
                    best_by_truth[truth_index] = (
                        match_score, record["record_id"])
        hit_records = {
            record_id for _score, record_id in best_by_truth.values()
        }
        file_truths = len(case["truths"])
        file_caught = len(best_by_truth)
        file_evidence = len(selected)
        file_hits = len(hit_records)
        truth_count += file_truths
        caught_count += file_caught
        evidence_count += file_evidence
        evidence_hits += file_hits
        file_precision = (
            file_hits / file_evidence if file_evidence
            else (1.0 if not file_truths else 0.0)
        )
        file_recall = (
            file_caught / file_truths if file_truths else 1.0)
        per_file.append({
            "file": name,
            "truth_segments": file_truths,
            "caught_truth_segments": file_caught,
            "evidence_count": file_evidence,
            "evidence_true_positive": file_hits,
            **precision_metric_fields(file_precision),
            "recall": file_recall,
        })
    precision = evidence_hits / evidence_count if evidence_count else 0.0
    recall = caught_count / truth_count if truth_count else 1.0
    return {
        "truth_segments": truth_count,
        "caught_truth_segments": caught_count,
        "evidence_count": evidence_count,
        "evidence_true_positive": evidence_hits,
        **precision_metric_fields(precision),
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
        "per_file": per_file,
    }


def threshold_candidates(scores_by_case, names):
    scores = np.asarray([
        score for name in names for score in scores_by_case[name]
    ], dtype=float)
    if not len(scores):
        return [0.5]
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 101))
    candidates = set(float(value) for value in quantiles)
    candidates.update(np.linspace(0.05, 0.95, 19).tolist())
    candidates.add(float(scores.max()) + 1e-8)
    return sorted(candidates)


def tune_selection(
        cases,
        names,
        scores_by_case,
        objective,
        high_min_recall):
    best = None
    for threshold in threshold_candidates(scores_by_case, names):
        for max_per_source in (None, 2, 4, 8, 12, 20):
            for max_sources_per_region in (1, 2):
                selection = {
                    "threshold": threshold,
                    "max_total": 220,
                    "max_per_source": max_per_source,
                    "same_source_overlap": 0.45,
                    "same_source_gap": 250,
                    "cross_source_overlap": 0.75,
                    "max_sources_per_region": max_sources_per_region,
                }
                metrics = fast_metrics(
                    cases, names, scores_by_case, selection)
                if objective == "high_precision":
                    if metrics["recall"] < high_min_recall:
                        continue
                    key = (
                        metrics["evidence_hit_precision"],
                        metrics["f1"],
                        metrics["recall"],
                        -metrics["evidence_count"],
                    )
                else:
                    key = (
                        metrics["f1"],
                        metrics["evidence_hit_precision"],
                        metrics["recall"],
                        -metrics["evidence_count"],
                    )
                if best is None or key > best[0]:
                    best = (key, selection, metrics)
    if best is None:
        raise RuntimeError(
            f"No {objective} operating point met the recall requirement")
    return {"selection": best[1], "metrics": best[2]}


def baseline_metrics(cases, names, profile, evaluator):
    totals = Counter()
    per_file = []
    for name in names:
        data = copy.deepcopy(cases[name]["api_result"])
        selection = select_display_evidence(data.get("results", ()), profile)
        predictions, _sources = evaluator["flatten"](data)
        metrics = evidence_hit_metrics_from_predictions(
            cases[name]["truths"],
            predictions,
            evaluator,
            allow_prediction_reuse=True,
        )
        for key in (
            "truth_segments", "caught_truth_segments",
            "evidence_count", "evidence_true_positive",
        ):
            totals[key] += metrics[key]
        per_file.append({
            "file": name,
            "evidence_count": selection["shown_count"],
            **precision_metric_fields(metrics["evidence_hit_precision"]),
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })
    precision = (
        totals["evidence_true_positive"] / totals["evidence_count"]
        if totals["evidence_count"] else 0.0)
    recall = (
        totals["caught_truth_segments"] / totals["truth_segments"]
        if totals["truth_segments"] else 1.0)
    return {
        **dict(totals),
        **precision_metric_fields(precision),
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0),
        "per_file": per_file,
    }


def classifier_diagnostics(labels, scores):
    if len(set(labels.tolist())) < 2:
        return {"average_precision": None, "roc_auc": None}
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def sigmoid(value):
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def add_cross_encoder_scores(cases, model_name, output_dir, batch_size, device):
    cache_path = output_dir / "cross_encoder_scores.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    missing = []
    pairs = []
    for case_name, case in cases.items():
        case_cache = cache.setdefault(case_name, {})
        for record in case["records"]:
            if record["record_id"] in case_cache:
                continue
            evidence = record["evidence"]
            pairs.append((
                prediction_text(evidence, "student")[:5000],
                prediction_text(evidence, "source")[:5000],
            ))
            missing.append((case_name, record["record_id"]))

    if missing:
        import torch  # noqa: PLC0415
        from FlagEmbedding import FlagReranker  # noqa: PLC0415

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        devices = ["cuda:0"] if device == "cuda" else ["cpu"]
        model = FlagReranker(
            model_name,
            use_fp16=device == "cuda",
            cache_dir=str(RUNTIME_ROOT / "project_cache"),
            devices=devices,
            batch_size=batch_size,
            max_length=512,
            normalize=True,
        )
        raw_scores = model.compute_score(
            pairs, batch_size=batch_size, normalize=True)
        if np.isscalar(raw_scores):
            raw_scores = [raw_scores]
        for (case_name, record_id), score in zip(missing, raw_scores):
            score = float(score)
            cache[case_name][record_id] = (
                score if 0.0 <= score <= 1.0 else sigmoid(score))
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for case_name, case in cases.items():
        for record in case["records"]:
            record["cross_encoder_score"] = float(
                cache[case_name][record["record_id"]])


def cross_validated_model_scores(
        cases,
        development_names,
        test_names,
        model_name,
        seed,
        include_cross=False):
    development_names = np.asarray(development_names)
    fold_scores = {}
    fit_seconds = 0.0
    splitter = KFold(
        n_splits=min(5, len(development_names)),
        shuffle=True,
        random_state=seed,
    )
    for fold_index, (train_indices, validation_indices) in enumerate(
            splitter.split(development_names)):
        train_names = development_names[train_indices].tolist()
        validation_names = development_names[validation_indices].tolist()
        features, labels, records = training_matrix(
            cases, train_names, include_cross=include_cross)
        weights = document_balanced_weights(records, labels)
        model = build_models(seed + fold_index)[model_name]
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_model(model, features, labels, weights)
        fit_seconds += time.perf_counter() - started
        fold_scores.update(model_score_by_case(
            model,
            cases,
            validation_names,
            include_cross=include_cross,
        ))

    features, labels, records = training_matrix(
        cases, development_names.tolist(), include_cross=include_cross)
    weights = document_balanced_weights(records, labels)
    final_model = build_models(seed)[model_name]
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_model(final_model, features, labels, weights)
    fit_seconds += time.perf_counter() - started
    fold_scores.update(model_score_by_case(
        final_model,
        cases,
        test_names,
        include_cross=include_cross,
    ))
    return fold_scores, {
        "model": final_model,
        "fit_seconds": fit_seconds,
        "include_cross": include_cross,
    }


def feature_importance(model, feature_names):
    estimator = model.named_steps.get("model") if isinstance(
        model, Pipeline) else model
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    if values is None:
        return []
    ranked = sorted(
        zip(feature_names, values), key=lambda item: item[1], reverse=True)
    return [
        {"feature": name, "importance": float(value)}
        for name, value in ranked[:12]
    ]


def percent(value):
    return f"{100.0 * value:.2f}%"


def write_report(payload, output_path):
    lines = [
        "# Evidence reranker benchmark",
        "",
        "Phase 1-6 are unchanged. Every strategy ranks the same saved "
        "`consolidated_evidence` candidates.",
        "",
        "## Dataset",
        "",
        f"- Train documents: {len(payload['split']['train'])}",
        f"- Validation documents: {len(payload['split']['validation'])}",
        f"- Test documents: {len(payload['split']['test'])}",
        f"- Total candidates: {payload['dataset']['candidate_count']}",
        f"- Positive candidates: {payload['dataset']['positive_candidates']}",
        "",
        "The train and validation documents form a development set. Model "
        "scores below are out-of-fold on that development set.",
        "",
        "## Metric definitions",
        "",
        "- Evidence-hit precision: correctly matched evidence passages divided "
        "by all selected evidence passages.",
        "- Character precision: not computed. It requires mapping normalized "
        "tool offsets back to the original raw PAN submission and source text.",
        "- `precision` in JSON is retained only as a compatibility alias for "
        "`evidence_hit_precision`.",
        "",
        "## Balanced operating point",
        "",
        "| Strategy | Dev-CV evidence-hit P | Dev-CV R | Dev-CV F1 | "
        "Test evidence-hit P | Test R | Test F1 | Test evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for name, result in payload["strategies"].items():
        val = result["balanced"]["development_cv"]
        test = result["balanced"]["test"]
        rows.append((val["f1"], name, val, test))
    for _score, name, val, test in sorted(rows, reverse=True):
        lines.append(
            f"| {name} | {percent(val['evidence_hit_precision'])} | "
            f"{percent(val['recall'])} | {percent(val['f1'])} | "
            f"{percent(test['evidence_hit_precision'])} | "
            f"{percent(test['recall'])} | "
            f"{percent(test['f1'])} | {test['evidence_count']} |")

    lines.extend([
        "",
        "## High evidence-hit precision operating point",
        "",
        "| Strategy | Dev-CV evidence-hit P | Dev-CV R | Dev-CV F1 | "
        "Test evidence-hit P | Test R | Test F1 | Test evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    rows = []
    for name, result in payload["strategies"].items():
        val = result["high_precision"]["development_cv"]
        test = result["high_precision"]["test"]
        rows.append((
            val["evidence_hit_precision"], val["f1"], name, val, test
        ))
    for _precision, _f1, name, val, test in sorted(rows, reverse=True):
        lines.append(
            f"| {name} | {percent(val['evidence_hit_precision'])} | "
            f"{percent(val['recall'])} | {percent(val['f1'])} | "
            f"{percent(test['evidence_hit_precision'])} | "
            f"{percent(test['recall'])} | "
            f"{percent(test['f1'])} | {test['evidence_count']} |")

    recommendation = payload["recommendation"]
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"Selected by development cross-validation F1: `{recommendation['strategy']}`.",
        f"Development cross-validation F1: "
        f"{percent(recommendation['development_cv']['f1'])}.",
        f"Held-out test evidence-hit precision: "
        f"{percent(recommendation['test']['evidence_hit_precision'])}.",
        f"Held-out test recall: {percent(recommendation['test']['recall'])}.",
        f"Held-out test F1: {percent(recommendation['test']['f1'])}.",
        "",
        "The test set was not used to choose models or thresholds. These "
        "results are "
        "still preliminary because the number of documents is small.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(RUNTIME_ROOT / "project_cache"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", str(RUNTIME_ROOT / "project_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "mpl_temp"))

    split = validate_experiment_split(json.loads(pathlib.Path(
        args.split_file).read_text(encoding="utf-8")))
    evaluator = import_evaluator(args.evaluator_root)
    raw_cases = load_raw_cases(args.raw_dir)
    cases = build_cases(
        raw_cases,
        split,
        os.path.abspath(args.evaluation_dir),
        os.path.abspath(args.source_dir),
        evaluator,
    )
    add_record_case_names(cases)

    if args.cross_encoder_model:
        add_cross_encoder_scores(
            cases,
            args.cross_encoder_model,
            output_dir,
            args.cross_encoder_batch_size,
            args.cross_encoder_device,
        )

    development_names = split["train"] + split["validation"]
    test_names = split["test"]
    all_names = development_names + test_names
    strategies = {
        "legacy_confidence": score_by_case(
            cases, all_names,
            lambda record: record["evidence"].get(
                "confidence_score", 0.0),
        ),
        "agreement_rules": score_by_case(
            cases, all_names, agreement_score),
    }
    if args.cross_encoder_model:
        strategies["cross_encoder"] = score_by_case(
            cases,
            all_names,
            lambda record: record.get("cross_encoder_score", 0.0),
        )

    model_lookup = {}
    for name in build_models(args.seed):
        scores, model_info = cross_validated_model_scores(
            cases,
            development_names,
            test_names,
            name,
            args.seed,
            include_cross=False,
        )
        strategies[name] = scores
        model_lookup[name] = model_info
    if args.cross_encoder_model:
        for name in build_models(args.seed):
            strategy_name = name + "_plus_cross"
            scores, model_info = cross_validated_model_scores(
                cases,
                development_names,
                test_names,
                name,
                args.seed,
                include_cross=True,
            )
            strategies[strategy_name] = scores
            model_lookup[strategy_name] = model_info

    _development_features, development_labels, _development_records = (
        training_matrix(cases, development_names))

    payload = {
        "split": split,
        "dataset": {
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "candidate_count": sum(
                len(cases[name]["records"]) for name in all_names),
            "positive_candidates": sum(
                record["label"] for name in all_names
                for record in cases[name]["records"]),
            "development_candidates": int(len(development_labels)),
            "development_positive_candidates": int(
                development_labels.sum()),
        },
        "metric_definitions": {
            "precision": "Compatibility alias for evidence_hit_precision",
            "evidence_hit_precision": (
                "Correctly matched selected evidence passages divided by "
                "all selected evidence passages"
            ),
            "character_precision": None,
            "character_precision_note": CHARACTER_PRECISION_NOTE,
        },
        "tuning_partitions": ["train", "validation"],
        "held_out_partition": "test",
        "baseline": {
            profile: {
                "development": baseline_metrics(
                    cases, development_names, profile, evaluator),
                "test": baseline_metrics(
                    cases, test_names, profile, evaluator),
            }
            for profile in ("balanced", "high_precision", "all")
        },
        "strategies": {},
        "cross_encoder_model": args.cross_encoder_model,
    }

    for strategy_name, scores_by_case in strategies.items():
        balanced = tune_selection(
            cases,
            development_names,
            scores_by_case,
            objective="balanced",
            high_min_recall=args.high_min_recall,
        )
        high_precision = tune_selection(
            cases,
            development_names,
            scores_by_case,
            objective="high_precision",
            high_min_recall=args.high_min_recall,
        )
        validation_labels = np.asarray([
            record["label"] for name in development_names
            for record in cases[name]["records"]
        ])
        validation_scores = np.asarray([
            score for name in development_names
            for score in scores_by_case[name]
        ])
        payload["strategies"][strategy_name] = {
            "classifier_diagnostics": classifier_diagnostics(
                validation_labels, validation_scores),
            "fit_seconds": model_lookup.get(
                strategy_name, {}).get("fit_seconds", 0.0),
            "balanced": {
                "selection": balanced["selection"],
                "development_cv": balanced["metrics"],
                "test": fast_metrics(
                    cases,
                    test_names,
                    scores_by_case,
                    balanced["selection"],
                ),
            },
            "high_precision": {
                "selection": high_precision["selection"],
                "development_cv": high_precision["metrics"],
                "test": fast_metrics(
                    cases,
                    test_names,
                    scores_by_case,
                    high_precision["selection"],
                ),
            },
        }

    recommended_name = max(
        payload["strategies"],
        key=lambda name: (
            payload["strategies"][name]["balanced"][
                "development_cv"]["f1"],
            payload["strategies"][name]["balanced"][
                "development_cv"]["evidence_hit_precision"],
        ),
    )
    recommended_result = payload["strategies"][recommended_name]["balanced"]
    payload["recommendation"] = {
        "strategy": recommended_name,
        "selection": recommended_result["selection"],
        "development_cv": recommended_result["development_cv"],
        "test": recommended_result["test"],
    }

    feature_names = list(FEATURE_NAMES)
    if recommended_name.endswith("_plus_cross"):
        feature_names.append("cross_encoder_score")
    if recommended_name in model_lookup:
        model_info = model_lookup[recommended_name]
        payload["recommendation"]["top_features"] = feature_importance(
            model_info["model"], feature_names)
        final_model = model_info["model"]
        joblib.dump(final_model, output_dir / "recommended_reranker.joblib")
        (output_dir / "recommended_reranker_metadata.json").write_text(
            json.dumps({
                "strategy": recommended_name,
                "label_schema_version": LABEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "selection": recommended_result["selection"],
                "trained_documents": development_names,
                "held_out_test_documents": test_names,
                "cross_encoder_model": args.cross_encoder_model,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    result_path = output_dir / "reranker_benchmark.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(payload, output_dir / "RERANKER_REPORT.md")
    print(json.dumps({
        "result": str(result_path),
        "recommendation": payload["recommendation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
