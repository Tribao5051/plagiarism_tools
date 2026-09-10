"""Pure PAN evidence scoring shared by offline benchmarks."""

import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TEXT_MATCH_THRESHOLD = 0.20
CONTEXT_CHARS = 300
SCORING_SCHEMA_VERSION = "two_sided_evidence_hit_v1"
CHARACTER_PRECISION_NOTE = (
    "Not computed: PAN XML offsets refer to original raw text, so character "
    "precision requires an explicit mapping from normalized tool text back "
    "to the original submission and source text."
)
EVIDENCE_FAMILIES = {
    "semantic": {"semantic", "verbatim_semantic"},
    "verbatim": {"verbatim", "verbatim_semantic"},
    "mixed": {"verbatim_semantic"},
}


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def shingles(text, size=4):
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < size:
        return {" ".join(words)}
    return {
        " ".join(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def text_profile(text):
    normalized = normalize_text(text)
    return normalized, shingles(normalized)


def token_profile(text):
    return {
        token
        for token in normalize_text(text).split()
        if len(token) >= 3 and token not in ENGLISH_STOP_WORDS
    }


def token_containment(candidate, reference):
    candidate_tokens = token_profile(candidate)
    reference_tokens = token_profile(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    shared = candidate_tokens & reference_tokens
    if len(shared) < 8:
        return 0.0
    return len(shared) / min(len(candidate_tokens), len(reference_tokens))


def containment_from_profiles(candidate_profile, reference_profile):
    candidate_norm, candidate_shingles = candidate_profile
    reference_norm, reference_shingles = reference_profile
    if not candidate_norm or not reference_norm:
        return 0.0
    if candidate_norm in reference_norm:
        return 1.0
    if not candidate_shingles:
        return 0.0
    return (
        len(candidate_shingles & reference_shingles)
        / len(candidate_shingles)
    )


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def parse_ground_truth(xml_path, suspicious_path, source_dir):
    suspicious_text = read_text(suspicious_path)
    root = ET.parse(xml_path).getroot()
    truths = []
    for index, feature in enumerate(root.findall("feature")):
        if feature.get("name") != "artificial-plagiarism":
            continue
        suspicious_start = int(feature.get("this_offset"))
        suspicious_length = int(feature.get("this_length"))
        source_start = int(feature.get("source_offset"))
        source_length = int(feature.get("source_length"))
        source_file = feature.get("source_reference")
        source_text = read_text(os.path.join(source_dir, source_file))
        suspicious_end = suspicious_start + suspicious_length
        source_end = source_start + source_length
        truths.append({
            "truth_id": index,
            "source_file": source_file,
            "obfuscation": feature.get("obfuscation", "unknown").lower(),
            "translation": (
                feature.get("translation", "false").lower() == "true"),
            "student_start_raw": suspicious_start,
            "student_end_raw": suspicious_end,
            "source_start_raw": source_start,
            "source_end_raw": source_end,
            "student_before": suspicious_text[
                max(0, suspicious_start - CONTEXT_CHARS):suspicious_start],
            "student_text": suspicious_text[
                suspicious_start:suspicious_end],
            "student_after": suspicious_text[
                suspicious_end:suspicious_end + CONTEXT_CHARS],
            "source_before": source_text[
                max(0, source_start - CONTEXT_CHARS):source_start],
            "source_text": source_text[source_start:source_end],
            "source_after": source_text[
                source_end:source_end + CONTEXT_CHARS],
        })
    return truths


def flatten_predictions(
        api_result, evidence_field="evidence", fallback_field=None):
    predictions = []
    retrieved_sources = set()
    for result in api_result.get("results", []):
        source_file = result.get("filename", "")
        retrieved_sources.add(source_file)
        evidence_items = result.get(evidence_field)
        if evidence_items is None and fallback_field:
            evidence_items = result.get(fallback_field)
        for evidence_index, evidence in enumerate(evidence_items or []):
            predictions.append({
                "prediction_id": (
                    f"{evidence_field}:{source_file}:{evidence_index}"),
                "source_file": evidence.get("source_file") or source_file,
                "type": evidence.get("type", "unknown"),
                "student_start_clean": evidence.get("student_start"),
                "student_end_clean": evidence.get("student_end"),
                "source_start_clean": evidence.get("source_start"),
                "source_end_clean": evidence.get("source_end"),
                "student_text": (
                    evidence.get("student_text")
                    or evidence.get("student")
                    or evidence.get("text")
                    or ""
                ),
                "source_text": (
                    evidence.get("source_text")
                    or evidence.get("source")
                    or ""
                ),
                "student_match_text": (
                    evidence.get("student_match_text")
                    or evidence.get("verbatim_text")
                    or evidence.get("student")
                    or ""
                ),
                "source_match_text": (
                    evidence.get("source_match_text")
                    or evidence.get("verbatim_text")
                    or evidence.get("source")
                    or ""
                ),
                "semantic_score": evidence.get("semantic_score"),
                "lexical_score": evidence.get("lexical_score"),
                "match_method": evidence.get("match_method"),
                "match_methods": evidence.get("match_methods", []),
                "text_similarity": evidence.get("text_similarity"),
                "evidence_count": int(evidence.get("evidence_count") or 1),
                "confidence_score": evidence.get("confidence_score"),
                "confidence_level": evidence.get("confidence_level"),
                "verbatim_characters": int(
                    evidence.get("verbatim_characters") or 0),
            })
    return predictions, retrieved_sources


def evaluator_helpers():
    return {
        "threshold": TEXT_MATCH_THRESHOLD,
        "containment": containment_from_profiles,
        "flatten": flatten_predictions,
        "parse_truth": parse_ground_truth,
        "text_profile": text_profile,
        "token_containment": token_containment,
    }


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


def annotate_truth_matches(records, truths, evaluator=None):
    evaluator = evaluator or evaluator_helpers()
    truth_profiles = [
        (
            evaluator["text_profile"](truth["student_text"]),
            evaluator["text_profile"](truth["source_text"]),
        )
        for truth in truths
    ]
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
    return records


def records_for_predictions(predictions, truths, evaluator=None):
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
    for record in records:
        record["truth_ids"] = sorted(record["truth_scores"])
    return records


def records_for_response(response, field, truths, evaluator=None):
    records = []
    for result in response.get("results", []):
        if field not in result:
            raise ValueError("Response missing " + field)
        for evidence in result[field]:
            records.append({
                "source_file": (
                    evidence.get("source_file") or result["filename"]),
                "evidence": evidence,
            })
    annotate_truth_matches(records, truths, evaluator)
    for record in records:
        record["truth_ids"] = sorted(record["truth_scores"])
    return records


def precision_metric_fields(evidence_hit_precision):
    return {
        "precision": evidence_hit_precision,
        "evidence_hit_precision": evidence_hit_precision,
        "character_precision": None,
        "character_precision_note": CHARACTER_PRECISION_NOTE,
    }


def evidence_metrics(records, truth_count):
    caught = sorted({
        truth_id
        for record in records
        for truth_id in record.get("truth_ids", ())
    })
    hits = sum(bool(record.get("truth_ids")) for record in records)
    evidence_count = len(records)
    precision = hits / evidence_count if evidence_count else None
    recall = len(caught) / truth_count if truth_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else 0.0
    )
    return {
        "truth_segments": truth_count,
        "caught_truth_segments": len(caught),
        "caught_truth_ids": caught,
        "evidence_count": evidence_count,
        "true_positive_evidence": hits,
        "false_positive_evidence": evidence_count - hits,
        **precision_metric_fields(precision),
        "segment_recall": recall,
        "recall": recall,
        "f1": f1,
        "negative_file_has_evidence": bool(
            not truth_count and evidence_count),
    }


def evidence_hit_metrics_from_predictions(
        truths, predictions, evaluator=None, allow_prediction_reuse=True):
    del allow_prediction_reuse
    score = evidence_metrics(
        records_for_predictions(predictions, truths, evaluator), len(truths))
    return {
        "truth_segments": score["truth_segments"],
        "caught_truth_segments": score["caught_truth_segments"],
        "evidence_count": score["evidence_count"],
        "evidence_true_positive": score["true_positive_evidence"],
        "evidence_false_positive": score["false_positive_evidence"],
        **precision_metric_fields(score["evidence_hit_precision"]),
        "recall": score["segment_recall"],
        "f1": score["f1"],
        "negative_file_has_evidence": score[
            "negative_file_has_evidence"],
    }


def score_response(response, truths, truth_sources, evaluator=None):
    stages = {}
    display = []
    for field in ("raw_evidence", "consolidated_evidence", "evidence"):
        records = records_for_response(response, field, truths, evaluator)
        stages[field] = evidence_metrics(records, len(truths))
        if field == "evidence":
            display = records
    families = {
        family: evidence_metrics([
            record for record in display
            if record["evidence"].get("type") in kinds
        ], len(truths))
        for family, kinds in EVIDENCE_FAMILIES.items()
    }
    raw = set(stages["raw_evidence"]["caught_truth_ids"])
    consolidated = set(stages[
        "consolidated_evidence"]["caught_truth_ids"])
    shown = set(stages["evidence"]["caught_truth_ids"])
    available = set(range(len(truths)))
    diagnostics = {
        "missing_in_raw": sorted(available - raw),
        "lost_by_consolidation": sorted(raw - consolidated),
        "lost_by_selection": sorted(consolidated - shown),
        "display_on_non_truth_sources": sum(
            record["source_file"] not in truth_sources
            for record in display
        ),
    }
    breakdown = {}
    for label in ("none", "low", "high", "translation"):
        truth_ids = {
            index
            for index, truth in enumerate(truths)
            if (
                truth.get("translation")
                if label == "translation"
                else truth.get("obfuscation") == label
            )
        }
        breakdown[label] = {
            "total": len(truth_ids),
            "caught": len(truth_ids & shown),
        }
    return {
        "stages": stages,
        "families": families,
        "diagnostics": diagnostics,
        "breakdown": breakdown,
    }, display


def aggregate_metrics(rows):
    truth_count = sum(row["truth_segments"] for row in rows)
    caught_count = sum(row["caught_truth_segments"] for row in rows)
    evidence_count = sum(row["evidence_count"] for row in rows)
    evidence_hits = sum(row["true_positive_evidence"] for row in rows)
    precision = evidence_hits / evidence_count if evidence_count else None
    recall = caught_count / truth_count if truth_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else 0.0
    )
    return {
        "truth_segments": truth_count,
        "caught_truth_segments": caught_count,
        "evidence_count": evidence_count,
        "true_positive_evidence": evidence_hits,
        "false_positive_evidence": evidence_count - evidence_hits,
        **precision_metric_fields(precision),
        "segment_recall": recall,
        "recall": recall,
        "f1": f1,
        "negative_file_has_evidence": any(
            row.get("negative_file_has_evidence", False) for row in rows),
    }


def scoring_code_hash():
    with Path(__file__).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scoring_metadata():
    return {
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "scoring_code_sha256": scoring_code_hash(),
        "text_match_threshold": TEXT_MATCH_THRESHOLD,
        "positive_match_rule": (
            "same source file and both submission/source text at threshold"),
        "evidence_hit_rule": (
            "each evidence matching at least one XML pair is a hit"),
        "segment_recall_rule": "union of matched XML pair IDs",
        "character_precision": None,
        "character_precision_note": CHARACTER_PRECISION_NOTE,
    }


def canonical_json_hash(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
