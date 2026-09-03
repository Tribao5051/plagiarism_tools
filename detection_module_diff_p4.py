import os
import shutil
import time
import numpy as np
import torch
import uuid
import psutil
import re
import hashlib
from collections import OrderedDict, defaultdict
from difflib import SequenceMatcher
from chunking_config import (
    chunk_text,
    chunk_text_with_native_spans,
    create_chunker_tools,
    load_chunking_config,
)
# from copydetect import CopyDetector
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdf_extract import extract_university_content
from model_module import model_manager, device
from project_config import load_project_config
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_CONFIG = load_project_config()
DETECTION_CONFIG = PROJECT_CONFIG["detection"]
corpus_dir = os.path.join(
    PROJECT_ROOT, "corpus_normalized_bge_m3_c400_o80_v1")
skip_front = 3
EVIDENCE_CONTEXT_CHARS = 420
EVIDENCE_OVERLAP_WINDOW = 300
SEMANTIC_ENCODE_BATCH_SIZE = int(
    DETECTION_CONFIG["semantic_encode_batch_size"])
SEMANTIC_MAX_LENGTH = int(DETECTION_CONFIG["semantic_max_length"])
SOURCE_CACHE_MAX_FILES = 64
SEMANTIC_STRONG_THRESHOLD = float(
    DETECTION_CONFIG.get("semantic_strong_threshold", 0.72))
SEMANTIC_FALLBACK_THRESHOLD = float(
    DETECTION_CONFIG.get("semantic_fallback_threshold", 0.68))
SEMANTIC_FALLBACK_MARGIN = float(
    DETECTION_CONFIG.get("semantic_fallback_margin", 0.035))
SEMANTIC_SOURCES_PER_STUDENT_CHUNK = int(
    DETECTION_CONFIG.get("semantic_sources_per_student_chunk", 2))
SEMANTIC_NMS_STUDENT_WINDOW = int(
    DETECTION_CONFIG.get("semantic_nms_student_window", 700))
SEMANTIC_NMS_SOURCE_WINDOW = int(
    DETECTION_CONFIG.get("semantic_nms_source_window", 900))
SEMANTIC_RETRIEVAL_QUERY_LIMIT = 512
SEMANTIC_RETRIEVAL_RESULTS_PER_CHUNK = int(
    DETECTION_CONFIG["semantic_retrieval_results_per_chunk"])
HYBRID_CANDIDATE_LIMIT = int(
    DETECTION_CONFIG["hybrid_candidate_limit"])
VERBATIM_ANCHOR_MAX_GAP = 240
VERBATIM_ALIGNMENT_TOLERANCE = 120
VERBATIM_MIN_MATCH_CHARS = 45
PASSAGE_STUDENT_MAX_GAP = 450
PASSAGE_SOURCE_MAX_GAP = 650
PASSAGE_ALIGNMENT_TOLERANCE = 700
PASSAGE_CONTEXT_CHARS = 1200
PASSAGE_MAX_STUDENT_CHARS = 3500
PASSAGE_MAX_SOURCE_CHARS = 4500
EVIDENCE_PROFILES = {"balanced", "high_precision", "all"}
DEFAULT_EVIDENCE_PROFILE = "balanced"
DEFAULT_EVIDENCE_PROFILE_CONFIGS = {
    "balanced": {
        "minimum_confidence": 0.50,
        "max_total": 220,
        "max_per_source": 12,
        "same_source_student_overlap": 0.45,
        "same_source_student_gap": 250,
        "cross_source_overlap": 0.75,
        "max_sources_per_region": 2,
    },
    "high_precision": {
        "minimum_confidence": 0.66,
        "max_total": 90,
        "max_per_source": 2,
        "same_source_student_overlap": 0.20,
        "same_source_student_gap": 1100,
        "cross_source_overlap": 0.55,
        "max_sources_per_region": 1,
    },
}
EVIDENCE_PROFILE_CONFIGS = {}
for _profile_name, _defaults in DEFAULT_EVIDENCE_PROFILE_CONFIGS.items():
    EVIDENCE_PROFILE_CONFIGS[_profile_name] = dict(_defaults)
    EVIDENCE_PROFILE_CONFIGS[_profile_name].update(
        DETECTION_CONFIG.get("evidence_profiles", {}).get(
            _profile_name, {}))
LEXICAL_WINDOW_SIZE = 2500
LEXICAL_WINDOW_STEP = 1250
LEXICAL_SCORE_THRESHOLD = 0.50
LEXICAL_MAX_SOURCES_PER_WINDOW = 2
LEXICAL_MAX_EVIDENCE_PER_SOURCE = 30

LEXICAL_VECTORIZER = HashingVectorizer(
    n_features=2 ** 15,
    alternate_sign=False,
    norm="l2",
    stop_words="english",
    ngram_range=(1, 1),
)


def compact_with_index_map(text):
    compact_chars = []
    index_map = []
    for idx, char in enumerate(text):
        if not char.isspace():
            compact_chars.append(char)
            index_map.append(idx)
    return "".join(compact_chars), index_map


def trim_to_word_boundary(text):
    text = text.strip()
    first_space = text.find(" ")
    last_space = text.rfind(" ")
    if first_space > 0:
        text = text[first_space + 1:]
    if last_space > 0:
        text = text[:last_space]
    return text.strip()


def expand_to_readable_context_span(
        text,
        start,
        end,
        max_chars=EVIDENCE_CONTEXT_CHARS,
        max_sentences=2):
    if not text:
        return "", 0, 0

    start = max(0, min(start or 0, len(text)))
    end = max(start, min(end or start, len(text)))
    boundaries = ".!?\n"

    previous_boundaries = [text.rfind(char, 0, start) for char in boundaries]
    left = max(previous_boundaries) + 1
    while left < start and text[left].isspace():
        left += 1

    right = end
    sentence_count = 0
    cursor = left
    while cursor < len(text) and sentence_count < max_sentences:
        positions = [
            text.find(char, cursor) for char in boundaries
            if text.find(char, cursor) >= 0
        ]
        if not positions:
            right = len(text)
            break
        boundary = min(positions)
        right = boundary + 1
        sentence_count += 1
        cursor = right
        if right >= end and sentence_count >= max_sentences:
            break

    if right < end:
        positions = [
            text.find(char, end) for char in boundaries
            if text.find(char, end) >= 0
        ]
        right = min(positions) + 1 if positions else len(text)

    while right < len(text) and text[right].isspace():
        right += 1
    snippet = text[left:right].strip()

    # Very long sentences or OCR text without punctuation still need a hard cap.
    if not snippet or len(snippet) > max_chars:
        center = (start + end) // 2
        half = max_chars // 2
        left = max(0, center - half)
        right = min(len(text), center + half)
        if left > 0:
            next_space = text.find(" ", left, min(start + 1, right))
            if next_space >= 0:
                left = next_space + 1
        if right < len(text):
            previous_space = text.rfind(" ", max(end, left), right)
            if previous_space > left:
                right = previous_space
        snippet = text[left:right].strip()

    return (snippet or text[start:end].strip()), left, right


def expand_to_readable_context(
        text, start, end, max_chars=EVIDENCE_CONTEXT_CHARS):
    snippet, _context_start, _context_end = expand_to_readable_context_span(
        text, start, end, max_chars=max_chars)
    return snippet


def find_chunk_spans(full_text, chunks):
    spans = []
    cursor = 0
    for chunk_text in chunks:
        start = full_text.find(chunk_text, cursor)
        if start == -1:
            start = full_text.find(chunk_text)
        if start == -1:
            start = cursor
        end = min(len(full_text), start + len(chunk_text))
        spans.append((start, end))
        cursor = end
    return spans


def evidence_text_similarity(left, right):
    left = " ".join((left or "").split()).lower()
    right = " ".join((right or "").split()).lower()
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _evidence_bucket(position):
    if position is None:
        return None
    return int(position) // EVIDENCE_OVERLAP_WINDOW


def merge_evidence(verbatim_evidence, semantic_evidence, source_file):
    merged = []
    used_semantic = set()
    semantic_buckets = defaultdict(list)
    for idx, sem in enumerate(semantic_evidence):
        bucket = _evidence_bucket(sem.get("student_start"))
        if bucket is not None:
            semantic_buckets[bucket].append(idx)

    for verb in verbatim_evidence:
        best_idx = None
        best_ratio = 0.0
        verb_bucket = _evidence_bucket(verb.get("student_start"))
        candidate_indices = []
        if verb_bucket is not None:
            for bucket in range(verb_bucket - 1, verb_bucket + 2):
                candidate_indices.extend(semantic_buckets.get(bucket, ()))

        for idx in candidate_indices:
            if idx in used_semantic:
                continue
            sem = semantic_evidence[idx]
            sem_start = sem.get("student_start")
            verb_start = verb.get("student_start")
            close_position = (
                sem_start is not None and
                verb_start is not None and
                abs(sem_start - verb_start) <= EVIDENCE_OVERLAP_WINDOW
            )
            ratio = evidence_text_similarity(
                verb.get("student_text"), sem.get("student_text"))
            if close_position or ratio >= 0.55:
                if ratio >= best_ratio:
                    best_idx = idx
                    best_ratio = ratio

        if best_idx is not None:
            sem = semantic_evidence[best_idx]
            used_semantic.add(best_idx)
            merged.append({
                "type": "verbatim_semantic",
                "label": "Exact copy also semantically matched",
                "source_file": source_file,
                "student_start": verb.get("student_start"),
                "student_end": verb.get("student_end"),
                "source_start": verb.get("source_start") if verb.get("source_start") is not None else sem.get("source_start"),
                "source_end": verb.get("source_end") if verb.get("source_end") is not None else sem.get("source_end"),
                "source_chunk_index": sem.get("source_chunk_index"),
                "student_text": verb.get("student_text"),
                "source_text": verb.get("source_text") or sem.get("source_text"),
                "verbatim_text": verb.get("match_text"),
                "student_match_text": sem.get("student") or verb.get("match_text"),
                "source_match_text": sem.get("source") or verb.get("match_text"),
                "semantic_score": sem.get("score"),
                "lexical_score": sem.get("lexical_score"),
                "match_method": sem.get("match_method", "dense_embedding"),
                "text_similarity": round(best_ratio, 3),
            })
        else:
            merged.append({
                "type": "verbatim",
                "label": "Exact copied text",
                "source_file": source_file,
                "student_start": verb.get("student_start"),
                "student_end": verb.get("student_end"),
                "source_start": verb.get("source_start"),
                "source_end": verb.get("source_end"),
                "student_text": verb.get("student_text"),
                "source_text": verb.get("source_text"),
                "verbatim_text": verb.get("match_text"),
                "student_match_text": verb.get("match_text"),
                "source_match_text": verb.get("match_text"),
            })

    for idx, sem in enumerate(semantic_evidence):
        if idx in used_semantic:
            continue
        merged.append({
            "type": "semantic",
            "label": "Semantic similarity / possible paraphrase",
            "source_file": source_file,
            "student_start": sem.get("student_start"),
            "student_end": sem.get("student_end"),
            "source_start": sem.get("source_start"),
            "source_end": sem.get("source_end"),
            "source_chunk_index": sem.get("source_chunk_index"),
            "student_text": sem.get("student_text"),
            "source_text": sem.get("source_text"),
            "student_match_text": sem.get("student"),
            "source_match_text": sem.get("source"),
            "semantic_score": sem.get("score"),
            "lexical_score": sem.get("lexical_score"),
            "match_method": sem.get("match_method", "dense_embedding"),
        })

    return merged


def _valid_evidence_span(evidence, prefix):
    start = evidence.get(f"{prefix}_start")
    end = evidence.get(f"{prefix}_end")
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end
    )


def _span_gap(left_start, left_end, right_start, right_end):
    return max(0, max(left_start, right_start) - min(left_end, right_end))


def _evidence_passages_align(previous, candidate):
    if not (
        _valid_evidence_span(previous, "student")
        and _valid_evidence_span(previous, "source")
        and _valid_evidence_span(candidate, "student")
        and _valid_evidence_span(candidate, "source")
    ):
        return False

    student_gap = _span_gap(
        previous["student_start"],
        previous["student_end"],
        candidate["student_start"],
        candidate["student_end"],
    )
    source_gap = _span_gap(
        previous["source_start"],
        previous["source_end"],
        candidate["source_start"],
        candidate["source_end"],
    )
    if (
        student_gap > PASSAGE_STUDENT_MAX_GAP
        or source_gap > PASSAGE_SOURCE_MAX_GAP
    ):
        return False

    student_delta = (
        candidate["student_start"] - previous["student_start"]
    )
    source_delta = candidate["source_start"] - previous["source_start"]
    both_overlap = student_gap == 0 and source_gap == 0
    if both_overlap:
        return True
    if student_delta < 0 or source_delta < 0:
        return False

    adaptive_tolerance = max(
        PASSAGE_ALIGNMENT_TOLERANCE,
        int(max(student_delta, source_delta) * 0.45),
    )
    return abs(student_delta - source_delta) <= adaptive_tolerance


def _passage_evidence(group, test_content, source_content, source_file):
    student_start = min(item["student_start"] for item in group)
    student_end = max(item["student_end"] for item in group)
    source_start = min(item["source_start"] for item in group)
    source_end = max(item["source_end"] for item in group)
    evidence_types = {item.get("type", "unknown") for item in group}
    has_verbatim = bool(
        evidence_types & {"verbatim", "verbatim_semantic"})
    has_semantic = bool(
        evidence_types & {"semantic", "verbatim_semantic"})
    if has_verbatim and has_semantic:
        evidence_type = "verbatim_semantic"
        label = "Consolidated passage with verbatim and semantic support"
    elif has_verbatim:
        evidence_type = "verbatim"
        label = "Consolidated exact copied passage"
    else:
        evidence_type = "semantic"
        label = "Consolidated semantic / paraphrased passage"

    methods = sorted({
        item.get("match_method") or "verbatim_fingerprint"
        for item in group
    })
    semantic_scores = [
        item.get("semantic_score")
        for item in group
        if item.get("semantic_score") is not None
    ]
    lexical_scores = [
        item.get("lexical_score")
        for item in group
        if item.get("lexical_score") is not None
    ]
    chunk_indices = sorted({
        item.get("source_chunk_index")
        for item in group
        if item.get("source_chunk_index") is not None
    })
    verbatim_parts = [
        item.get("verbatim_text")
        for item in group
        if item.get("verbatim_text")
    ]
    verbatim_spans = [
        (item["student_start"], item["student_end"])
        for item in group
        if item.get("type") in {"verbatim", "verbatim_semantic"}
        and _valid_evidence_span(item, "student")
    ]
    verbatim_characters = sum(
        max(0, end - start)
        for start, end in _merge_intervals(verbatim_spans)
    )
    semantic_support_count = sum(
        item.get("type") in {"semantic", "verbatim_semantic"}
        for item in group
    )
    verbatim_support_count = sum(
        item.get("type") in {"verbatim", "verbatim_semantic"}
        for item in group
    )
    lexical_support_count = sum(
        "lexical_window" in (item.get("match_method") or "")
        for item in group
    )
    student_context, student_context_start, student_context_end = (
        expand_to_readable_context_span(
            test_content,
            student_start,
            student_end,
            max_chars=PASSAGE_CONTEXT_CHARS,
        )
    )
    source_context, source_context_start, source_context_end = (
        expand_to_readable_context_span(
            source_content,
            source_start,
            source_end,
            max_chars=PASSAGE_CONTEXT_CHARS,
        )
    )

    return {
        "type": evidence_type,
        "label": label,
        "source_file": source_file,
        "student_start": student_start,
        "student_end": student_end,
        "source_start": source_start,
        "source_end": source_end,
        "source_chunk_index": (
            chunk_indices[0] if len(chunk_indices) == 1 else None),
        "source_chunk_indices": chunk_indices,
        "student_text": student_context,
        "student_context_start": student_context_start,
        "student_context_end": student_context_end,
        "source_text": source_context,
        "source_context_start": source_context_start,
        "source_context_end": source_context_end,
        "student_match_text": test_content[student_start:student_end],
        "source_match_text": source_content[source_start:source_end],
        "verbatim_text": " ".join(verbatim_parts),
        "semantic_score": max(semantic_scores) if semantic_scores else None,
        "lexical_score": max(lexical_scores) if lexical_scores else None,
        "match_method": "+".join(methods),
        "match_methods": methods,
        "evidence_count": len(group),
        "semantic_support_count": semantic_support_count,
        "verbatim_support_count": verbatim_support_count,
        "lexical_support_count": lexical_support_count,
        "verbatim_characters": verbatim_characters,
        "raw_types": sorted(evidence_types),
    }


def consolidate_evidence_passages(
        evidence, test_content, source_content, source_file):
    """Merge nearby aligned fragments into deterministic display passages."""
    positioned = []
    unpositioned = []
    for item in evidence:
        if (
            _valid_evidence_span(item, "student")
            and _valid_evidence_span(item, "source")
        ):
            positioned.append(item)
        else:
            unpositioned.append(item)
    positioned.sort(key=lambda item: (
        item["student_start"],
        item["source_start"],
        item["student_end"],
        item["source_end"],
        item.get("type", ""),
    ))

    groups = []
    current = []
    for item in positioned:
        if current:
            student_span = max(
                item["student_end"],
                max(entry["student_end"] for entry in current),
            ) - min(
                item["student_start"],
                min(entry["student_start"] for entry in current),
            )
            source_span = max(
                item["source_end"],
                max(entry["source_end"] for entry in current),
            ) - min(
                item["source_start"],
                min(entry["source_start"] for entry in current),
            )
            if (
                not _evidence_passages_align(current[-1], item)
                or student_span > PASSAGE_MAX_STUDENT_CHARS
                or source_span > PASSAGE_MAX_SOURCE_CHARS
            ):
                groups.append(current)
                current = []
        current.append(item)
    if current:
        groups.append(current)

    consolidated = [
        _passage_evidence(
            group, test_content, source_content, source_file)
        for group in groups
    ]
    for item in unpositioned:
        fallback = dict(item)
        fallback.setdefault("source_file", source_file)
        fallback.setdefault("evidence_count", 1)
        fallback.setdefault("raw_types", [item.get("type", "unknown")])
        consolidated.append(fallback)
    return consolidated


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _interval_overlap_ratio(left, right):
    left_start, left_end = left
    right_start, right_end = right
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    shorter = min(left_end - left_start, right_end - right_start)
    return overlap / shorter if shorter > 0 else 0.0


def _same_source_evidence_is_duplicate(previous, candidate, config):
    """Keep one display candidate per nearby student region and source."""
    previous_student = (
        previous["student_start"], previous["student_end"])
    candidate_student = (
        candidate["student_start"], candidate["student_end"])
    student_overlap = _interval_overlap_ratio(
        previous_student, candidate_student)
    student_gap = _span_gap(*previous_student, *candidate_student)
    return (
        student_overlap >= config["same_source_student_overlap"]
        or student_gap <= config["same_source_student_gap"]
    )


def _evidence_confidence(evidence):
    """Score independent detector support; no ground-truth data is used."""
    evidence_type = evidence.get("type", "semantic")
    score = {
        "semantic": 0.34,
        "verbatim": 0.62,
        "verbatim_semantic": 0.72,
    }.get(evidence_type, 0.25)
    reasons = []

    semantic_score = evidence.get("semantic_score")
    if semantic_score is not None:
        semantic_quality = max(
            0.0, min(1.0, (float(semantic_score) - 0.68) / 0.32))
        score += 0.22 * semantic_quality
        if semantic_score >= 0.82:
            reasons.append("strong semantic similarity")

    lexical_score = evidence.get("lexical_score")
    if lexical_score is not None:
        lexical_quality = max(
            0.0, min(1.0, (float(lexical_score) - 0.50) / 0.50))
        score += 0.08 * lexical_quality
        if lexical_score >= 0.65:
            reasons.append("lexical support")

    evidence_count = max(1, int(evidence.get("evidence_count") or 1))
    score += min(0.14, 0.035 * evidence_count)
    if evidence_count >= 2:
        reasons.append(f"{evidence_count} aligned fragments")

    methods = set(evidence.get("match_methods") or [])
    if not methods and evidence.get("match_method"):
        methods = set(evidence["match_method"].split("+"))
    score += min(0.06, 0.02 * max(0, len(methods) - 1))
    if len(methods) >= 2:
        reasons.append("multiple detectors agree")

    source_semantic_coverage = float(
        evidence.get("source_semantic_coverage") or 0.0)
    source_verbatim_score = float(
        evidence.get("source_verbatim_score") or 0.0)
    source_support = max(
        source_semantic_coverage, source_verbatim_score)
    if source_support:
        score += min(0.12, source_support * 0.45)
        if source_support >= 0.05:
            reasons.append("repeated support across source")

    verbatim_characters = int(evidence.get("verbatim_characters") or 0)
    if verbatim_characters:
        score += min(0.12, verbatim_characters / 1200.0)
        reasons.append(f"{verbatim_characters} exact characters")

    student_start = evidence.get("student_start")
    student_end = evidence.get("student_end")
    span_length = (
        student_end - student_start
        if isinstance(student_start, int) and isinstance(student_end, int)
        else 0
    )
    if span_length >= 160:
        score += min(0.05, span_length / 20000.0)
    elif span_length and span_length < VERBATIM_MIN_MATCH_CHARS:
        score -= 0.10

    score = round(max(0.0, min(1.0, score)), 4)
    level = "high" if score >= 0.78 else "medium" if score >= 0.60 else "low"
    return score, level, reasons


def _passes_evidence_profile(evidence, profile):
    if profile == "all":
        return True

    evidence_type = evidence.get("type", "semantic")
    semantic_score = float(evidence.get("semantic_score") or 0.0)
    lexical_score = float(evidence.get("lexical_score") or 0.0)
    evidence_count = max(1, int(evidence.get("evidence_count") or 1))
    verbatim_characters = int(evidence.get("verbatim_characters") or 0)
    methods = set(evidence.get("match_methods") or [])
    if not methods and evidence.get("match_method"):
        methods = set(evidence["match_method"].split("+"))
    has_dense = "dense_embedding" in methods
    has_lexical = "lexical_window" in methods

    if profile == "balanced":
        if evidence_type == "verbatim_semantic":
            return (
                semantic_score >= 0.72 or verbatim_characters >= 45
            )
        if evidence_type == "verbatim":
            return verbatim_characters >= 60
        if has_dense and has_lexical:
            return semantic_score >= 0.70 and lexical_score >= 0.53
        if has_dense:
            return semantic_score >= 0.70
        return has_lexical and lexical_score >= 0.58

    if evidence_type == "verbatim_semantic":
        return (
            (evidence_count >= 2 or verbatim_characters >= 200)
            and (semantic_score >= 0.76 or verbatim_characters >= 80)
        )
    if evidence_type == "verbatim":
        return verbatim_characters >= 800
    if has_dense and has_lexical:
        return (
            semantic_score >= 0.82
            and lexical_score >= 0.62
            and evidence_count >= 2
        )
    if has_dense:
        return (
            semantic_score >= 0.80
            and (evidence_count >= 2 or semantic_score >= 0.90)
        )
    return False


def select_display_evidence(results, profile=DEFAULT_EVIDENCE_PROFILE):
    """Phase 7: rank and de-duplicate passages for user-facing output."""
    if profile not in EVIDENCE_PROFILES:
        raise ValueError(f"Unsupported evidence profile: {profile}")

    candidates = []
    for result_index, result in enumerate(results):
        for evidence in result.get("consolidated_evidence", ()):
            candidate = dict(evidence)
            candidate["source_semantic_coverage"] = (
                float(result.get("semantic_coverage") or 0.0) / 100.0)
            candidate["source_verbatim_score"] = (
                float(result.get("verbatim_score") or 0.0) / 100.0)
            confidence, level, reasons = _evidence_confidence(candidate)
            candidate["confidence_score"] = confidence
            candidate["confidence_level"] = level
            candidate["confidence_reasons"] = reasons
            candidate["_result_index"] = result_index
            candidates.append(candidate)

    rejection_keys = (
        "profile_rule",
        "confidence",
        "invalid_span",
        "source_cap",
        "same_source_duplicate",
        "cross_source_competition",
        "total_cap",
    )
    rejection_counts = {key: 0 for key in rejection_keys}
    result_rejection_counts = [
        {key: 0 for key in rejection_keys} for _result in results
    ]

    def reject(candidate, reason):
        rejection_counts[reason] += 1
        result_rejection_counts[candidate["_result_index"]][reason] += 1

    if profile == "all":
        selected = candidates
    else:
        config = EVIDENCE_PROFILE_CONFIGS[profile]
        eligible = []
        for candidate in candidates:
            if not _passes_evidence_profile(candidate, profile):
                reject(candidate, "profile_rule")
                continue
            if candidate["confidence_score"] < config["minimum_confidence"]:
                reject(candidate, "confidence")
                continue
            eligible.append(candidate)

        ranked = sorted(
            eligible,
            key=lambda item: (
                item["confidence_score"],
                item.get("evidence_count", 1),
                item.get("verbatim_characters", 0),
                item.get("semantic_score") or 0.0,
                -(item.get("student_start") or 0),
                item.get("source_file", ""),
            ),
            reverse=True,
        )
        selected = []
        source_counts = defaultdict(int)
        for candidate in ranked:
            if len(selected) >= config["max_total"]:
                reject(candidate, "total_cap")
                continue
            source_file = candidate.get("source_file", "")
            if source_counts[source_file] >= config["max_per_source"]:
                reject(candidate, "source_cap")
                continue
            if not (
                _valid_evidence_span(candidate, "student")
                and _valid_evidence_span(candidate, "source")
            ):
                reject(candidate, "invalid_span")
                continue

            candidate_student = (
                candidate["student_start"], candidate["student_end"])
            duplicate = False
            competing_sources = set()
            for kept in selected:
                kept_student = (kept["student_start"], kept["student_end"])
                overlap = _interval_overlap_ratio(
                    candidate_student, kept_student)
                if kept.get("source_file") == source_file:
                    if _same_source_evidence_is_duplicate(
                            kept, candidate, config):
                        duplicate = True
                        break
                elif overlap >= config["cross_source_overlap"]:
                    competing_sources.add(kept.get("source_file"))

            if duplicate:
                reject(candidate, "same_source_duplicate")
                continue
            if len(competing_sources) >= config["max_sources_per_region"]:
                reject(candidate, "cross_source_competition")
                continue

            selected.append(candidate)
            source_counts[source_file] += 1

    selected_by_result = defaultdict(list)
    for candidate in selected:
        result_index = candidate.pop("_result_index")
        selected_by_result[result_index].append(candidate)

    for result_index, result in enumerate(results):
        shown = sorted(
            selected_by_result.get(result_index, ()),
            key=lambda item: (
                item.get("student_start") is None,
                item.get("student_start") or 0,
                item.get("source_start") or 0,
            ),
        )
        candidate_count = len(result.get("consolidated_evidence", ()))
        result["evidence"] = shown
        result["evidence_summary"] = {
            "profile": profile,
            "candidate_count": candidate_count,
            "shown_count": len(shown),
            "hidden_count": max(0, candidate_count - len(shown)),
            "rejections": result_rejection_counts[result_index],
        }

    return {
        "profile": profile,
        "candidate_count": len(candidates),
        "shown_count": len(selected),
        "hidden_count": max(0, len(candidates) - len(selected)),
        "rejections": rejection_counts,
    }


def consolidate_verbatim_anchors(anchors):
    """Combine aligned fingerprint anchors into one readable copied passage."""
    if not anchors:
        return []

    ordered = sorted(
        anchors,
        key=lambda item: (
            item["student_start"],
            item["source_start"],
        ),
    )
    groups = []
    current = [ordered[0]]

    for anchor in ordered[1:]:
        previous = current[-1]
        student_delta = anchor["student_start"] - previous["student_start"]
        source_delta = anchor["source_start"] - previous["source_start"]
        aligned = (
            0 <= student_delta <= VERBATIM_ANCHOR_MAX_GAP
            and source_delta >= 0
            and abs(student_delta - source_delta)
            <= VERBATIM_ALIGNMENT_TOLERANCE
        )
        if aligned:
            current.append(anchor)
        else:
            groups.append(current)
            current = [anchor]
    groups.append(current)
    return groups


def prune_semantic_evidence(semantic_results):
    """Keep strong matches and one well-separated fallback per student chunk."""
    evidence_by_student = defaultdict(list)
    for source_file, result in semantic_results.items():
        for evidence in result.get("evidence", ()):
            evidence_by_student[evidence.get("student_start")].append(
                (source_file, evidence)
            )

    keep_ids = set()
    for candidates in evidence_by_student.values():
        ranked = sorted(
            candidates,
            key=lambda item: item[1].get("score", 0.0),
            reverse=True,
        )
        strong = [
            item for item in ranked
            if item[1].get("score", 0.0) > SEMANTIC_STRONG_THRESHOLD
        ]
        for source_file, evidence in strong[
                :SEMANTIC_SOURCES_PER_STUDENT_CHUNK]:
            keep_ids.add((source_file, id(evidence)))

        if strong or not ranked:
            continue
        best_score = ranked[0][1].get("score", 0.0)
        second_score = (
            ranked[1][1].get("score", 0.0)
            if len(ranked) > 1 else 0.0
        )
        if (
            best_score >= SEMANTIC_FALLBACK_THRESHOLD
            and best_score - second_score >= SEMANTIC_FALLBACK_MARGIN
        ):
            source_file, evidence = ranked[0]
            keep_ids.add((source_file, id(evidence)))

    for source_file, result in semantic_results.items():
        pruned = [
            evidence
            for evidence in result.get("evidence", ())
            if (source_file, id(evidence)) in keep_ids
        ]
        selected = []
        for evidence in sorted(
                pruned,
                key=lambda item: item.get("score", 0.0),
                reverse=True):
            student_start = evidence.get("student_start")
            source_start = evidence.get("source_start")
            duplicate_region = any(
                student_start is not None
                and kept.get("student_start") is not None
                and abs(student_start - kept["student_start"])
                < SEMANTIC_NMS_STUDENT_WINDOW
                and source_start is not None
                and source_start >= 0
                and kept.get("source_start") is not None
                and kept["source_start"] >= 0
                and abs(source_start - kept["source_start"])
                < SEMANTIC_NMS_SOURCE_WINDOW
                for kept in selected
            )
            if not duplicate_region:
                selected.append(evidence)
        result["evidence"] = selected


def rank_semantic_sources(query_result):
    """Aggregate chunk-level vector hits while preventing one chunk dominating."""
    source_scores = {}
    metadatas = query_result.get("metadatas") or []
    distances = query_result.get("distances") or []

    for query_metadatas, query_distances in zip(metadatas, distances):
        best_for_query = {}
        for metadata, distance in zip(query_metadatas, query_distances):
            source = (metadata or {}).get("source")
            if not source:
                continue
            similarity = 1.0 - float(distance)
            best_for_query[source] = max(
                similarity,
                best_for_query.get(source, float("-inf")),
            )

        for source, similarity in best_for_query.items():
            stats = source_scores.setdefault(
                source,
                {"best": float("-inf"), "hits": 0, "strong_hits": 0},
            )
            stats["best"] = max(stats["best"], similarity)
            stats["hits"] += 1
            if similarity >= SEMANTIC_FALLBACK_THRESHOLD:
                stats["strong_hits"] += 1

    ranked = sorted(
        source_scores,
        key=lambda source: (
            source_scores[source]["best"],
            source_scores[source]["strong_hits"],
            source_scores[source]["hits"],
            source,
        ),
        reverse=True,
    )
    return ranked, source_scores


def retrieve_semantic_candidate_sources(
        collection,
        target_vecs,
        collection_where=None,
        allowed_sources=None):
    query_count = min(
        len(target_vecs),
        SEMANTIC_RETRIEVAL_QUERY_LIMIT,
    )
    if query_count <= 0:
        return [], {}

    if len(target_vecs) == query_count:
        query_indices = torch.arange(len(target_vecs), device=target_vecs.device)
    else:
        query_indices = torch.linspace(
            0,
            len(target_vecs) - 1,
            steps=query_count,
            device=target_vecs.device,
        ).long().unique()

    query_embeddings = (
        target_vecs.index_select(0, query_indices)
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    query_kwargs = {
        "query_embeddings": query_embeddings,
        "n_results": SEMANTIC_RETRIEVAL_RESULTS_PER_CHUNK,
        "include": ["metadatas", "distances"],
    }
    if collection_where:
        query_kwargs["where"] = collection_where
    result = collection.query(
        **query_kwargs,
    )
    ranked, scores = rank_semantic_sources(result)
    if allowed_sources is not None:
        allowed_sources = set(allowed_sources)
        ranked = [source for source in ranked if source in allowed_sources]
        scores = {
            source: score for source, score in scores.items()
            if source in allowed_sources
        }
    return ranked, scores


def sliding_text_windows(
        text,
        size=LEXICAL_WINDOW_SIZE,
        step=LEXICAL_WINDOW_STEP):
    if not text:
        return []
    if len(text) <= size:
        return [(0, len(text), text)]

    starts = list(range(0, len(text) - size + 1, step))
    final_start = len(text) - size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [
        (start, min(len(text), start + size), text[start:start + size])
        for start in starts
    ]


def add_lexical_window_evidence(
        test_content,
        top_suspects,
        ml_state,
        semantic_results):
    """Detect large shuffled/reordered passages with sparse bag-of-words windows."""
    target_windows = sliding_text_windows(test_content)
    if not target_windows:
        return 0
    target_matrix = LEXICAL_VECTORIZER.transform(
        [window[2] for window in target_windows])
    candidates_by_target = defaultdict(list)

    for source_file in top_suspects:
        source_text = ml_state.get(
            "corpus_texts", {}).get(source_file, "")
        if not source_text:
            continue
        source_windows = sliding_text_windows(source_text)
        source_matrix = LEXICAL_VECTORIZER.transform(
            [window[2] for window in source_windows])
        similarities = cosine_similarity(
            target_matrix,
            source_matrix,
            dense_output=False,
        ).tocsr()

        for target_index in range(similarities.shape[0]):
            row = similarities.getrow(target_index)
            if not row.nnz:
                continue
            local_index = int(np.argmax(row.data))
            score = float(row.data[local_index])
            if score < LEXICAL_SCORE_THRESHOLD:
                continue
            source_index = int(row.indices[local_index])
            student_start, student_end, student_window = target_windows[
                target_index]
            source_start, source_end, source_window = source_windows[
                source_index]
            candidates_by_target[student_start].append((
                score,
                source_file,
                {
                    "student": student_window,
                    "student_text": expand_to_readable_context(
                        test_content, student_start, student_end),
                    "student_start": student_start,
                    "student_end": student_end,
                    "source": source_window,
                    "source_text": expand_to_readable_context(
                        source_text, source_start, source_end),
                    "source_start": source_start,
                    "source_end": source_end,
                    "source_chunk_index": None,
                    "score": score,
                    "lexical_score": score,
                    "match_method": "lexical_window",
                    "type": "semantic",
                    "label": "Lexical window similarity / shuffled paraphrase",
                },
            ))

    kept_by_source = defaultdict(list)
    for candidates in candidates_by_target.values():
        for _score, source_file, evidence in sorted(
                candidates,
                key=lambda item: (item[0], item[1]),
                reverse=True)[
                    :LEXICAL_MAX_SOURCES_PER_WINDOW]:
            kept_by_source[source_file].append(evidence)

    added = 0
    for source_file, candidates in kept_by_source.items():
        selected = []
        for evidence in sorted(
                candidates,
                key=lambda item: item["score"],
                reverse=True):
            duplicate_region = any(
                abs(
                    evidence["student_start"]
                    - kept["student_start"]
                ) < LEXICAL_WINDOW_SIZE
                and abs(
                    evidence["source_start"]
                    - kept["source_start"]
                ) < LEXICAL_WINDOW_SIZE * 2
                for kept in selected
            )
            if duplicate_region:
                continue
            selected.append(evidence)
            if len(selected) >= LEXICAL_MAX_EVIDENCE_PER_SOURCE:
                break
        semantic_results[source_file]["evidence"].extend(selected)
        added += len(selected)
    return added


def get_fingerprints(text, k=25, step=5):
    # Thay vì xóa hết, ta chỉ cần đảm bảo không có ký tự lạ ẩn nấp
    # Nếu bạn đã tự tin về phần clean, chỉ cần lower và chuẩn hóa khoảng trắng
    clean = text.lower().strip()

    # Chúng ta vẫn nên giữ lại khoảng trắng đơn để phân tách từ
    # nhưng để hash khớp nhau tuyệt đối, ta có thể hash chuỗi không trắng
    hashable_str, index_map = compact_with_index_map(clean)

    fingerprints = {}
    if len(hashable_str) < k:
        return fingerprints, hashable_str

    for i in range(0, len(hashable_str) - k + 1, step):
        chunk = hashable_str[i:i+k]
        h = hashlib.md5(chunk.encode()).hexdigest()
        if h not in fingerprints:
            # Lưu lại đoạn text có dấu cách để hiển thị cho người dùng dễ đọc
            # Tìm cách ánh xạ vị trí i về text gốc (tương đối)
            readable_start = index_map[i]
            readable_end = index_map[i + k - 1] + 1
            fingerprints[h] = {
                "pos": i,
                "student_start": readable_start,
                "student_end": readable_end,
                "match_text": clean[readable_start:readable_end],
            }
    return fingerprints, hashable_str


def get_chunks(text, k=60, step=30):
    """
    Standardizes text and breaks it into hashes.
    K=60 (approx 12-15 words) is a strong 'anchor' for a sentence.
    """
    # Normalize spacing to handle PDF-to-TXT artifacts
    clean = " ".join(text.split()).lower()
    chunks = []
    for i in range(0, len(clean) - k, step):
        chunk_text = clean[i:i+k]
        chunks.append({
            "h": hashlib.md5(chunk_text.encode()).hexdigest(),
            "text": chunk_text,
            "pos": i
        })
    return chunks, clean


def get_cached_semantic_source(ml_state, source_file):
    cache = ml_state.setdefault("semantic_source_cache", OrderedDict())
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict(cache)
        ml_state["semantic_source_cache"] = cache

    if source_file in cache:
        cached = cache.pop(source_file)
        cache[source_file] = cached
        embeddings, documents, metadatas = cached
        return embeddings, documents, metadatas, True

    source_data = ml_state["collection"].get(
        where={"source": source_file},
        include=["embeddings", "documents", "metadatas"])
    embeddings = np.asarray(source_data["embeddings"], dtype=np.float32)
    documents = source_data["documents"]
    metadatas = source_data.get("metadatas") or [{} for _ in documents]
    cache[source_file] = (embeddings, documents, metadatas)
    while len(cache) > SOURCE_CACHE_MAX_FILES:
        cache.popitem(last=False)
    return embeddings, documents, metadatas, False


# def normalize_text(text):
#     if not text:
#         return ""
#     # Xóa ký tự BOM và các ký tự điều khiển lạ
#     text = text.replace('\ufeff', '').replace('\u200b', '')
#     text = text.lower()
#     # Giữ nguyên logic cũ của bạn bên dưới
#     text = re.sub(r'[^a-z0-0\s]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


def perform_plagiarism_scan(test_content, ml_state):
    """
    Executes the 4-Phase Hybrid Funnel using pre-loaded RAM objects.
    ml_state: Dictionary containing sbert, collection, tfidf_matrix, etc.
    """
    start_time = time.time()
    process = psutil.Process(os.getpid())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    encode_batch_size = int(ml_state.get(
        "encode_batch_size", SEMANTIC_ENCODE_BATCH_SIZE))
    evidence_profile = ml_state.get(
        "evidence_profile", DEFAULT_EVIDENCE_PROFILE)
    if evidence_profile not in EVIDENCE_PROFILES:
        raise ValueError(
            f"Unsupported evidence profile: {evidence_profile}")

    # --- PHASE 2: HYBRID CANDIDATE RETRIEVAL ---
    # with open("debug_extracted_text.txt", "w", encoding="utf-8") as debug_file:
    #     debug_file.write(test_content)
    # print(f"DEBUG: Extracted text saved to 'debug_extracted_text.txt' for verification.")
    def encode_texts(texts):
        # model_manager đã load BGE-M3, device được import từ model_module
        model = model_manager.get_model()
        # BGE-M3 trả về dict, ta lấy 'dense_vecs'
        with torch.inference_mode():
            vecs = model.encode(
                texts,
                batch_size=encode_batch_size,
                max_length=SEMANTIC_MAX_LENGTH,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False)['dense_vecs']
        return torch.tensor(vecs, device=device)

    phase_2_start = time.time()
    forced_candidate_sources = ml_state.get("forced_candidate_sources")
    requested_retrieval_strategy = ml_state.get(
        "retrieval_strategy", "shortlist")
    faculty = ml_state.get("faculty", "default")
    tfidf_suspects = []
    semantic_retrieval_scores = {}
    semantic_retrieval_seconds = 0.0
    if forced_candidate_sources is not None:
        retrieval_mode = "oracle_xml"
        top_suspects = list(dict.fromkeys(forced_candidate_sources))
        print(
            f"\nSkipping Phase 2 retrieval: using {len(top_suspects)} "
            "source reference(s) from XML.")
    elif requested_retrieval_strategy == "exhaustive":
        retrieval_mode = "faculty_exhaustive_all_sources"
        top_suspects = list(ml_state.get("filenames", ()))
        print(
            f"\nStarting exhaustive faculty scan: {len(top_suspects)} "
            f"source(s) in faculty '{faculty}'.")
    else:
        retrieval_mode = "faculty_hybrid_shortlist_top20"
        print(
            f"\nStarting Phase 2: Hybrid Candidate Retrieval "
            f"inside faculty '{faculty}'...")
        test_tfidf_vec = ml_state["tfidf_vectorizer"].transform([test_content])
        tfidf_scores = cosine_similarity(
            test_tfidf_vec, ml_state["tfidf_matrix"])[0]
        top_indices = np.argsort(tfidf_scores)[-HYBRID_CANDIDATE_LIMIT:][::-1]
        tfidf_suspects = [ml_state["filenames"][i] for i in top_indices]
        top_suspects = tfidf_suspects
    phase_2_seconds = time.time() - phase_2_start

    # --- PHASE 3: SEMANTIC ANALYSIS (BGE-M3 GPU) ---
    print("\nStarting Phase 3: Semantic Analysis...")
    phase_3_start = time.time()
    chunking_started = time.time()
    chunk_config = ml_state.get("chunking_config") or load_chunking_config()
    chunker = ml_state.get("target_chunker")
    refinery = ml_state.get("target_chunk_refinery")
    if chunker is None:
        chunker, refinery = create_chunker_tools(
            chunk_config, include_overlap=False)
    if chunk_config["strategy"] == "token":
        target_chunks, target_spans = chunk_text_with_native_spans(
            test_content, chunker, refinery)
    else:
        target_chunks = chunk_text(test_content, chunker, refinery)
        target_spans = find_chunk_spans(test_content, target_chunks)
    target_chunking_seconds = time.time() - chunking_started
    if not target_chunks:
        print(
            f"[!] Warning: No text chunks extracted from the file. Skipping semantic analysis.")
        return {
            "results": [],
            "execution_time": round(time.time() - start_time, 2),
            "performance": {
                "phase_2_seconds": round(phase_2_seconds, 4),
                "phase_3_seconds": round(time.time() - phase_3_start, 4),
                "phase_4_seconds": 0.0,
                "phase_5_seconds": 0.0,
                "phase_6_seconds": 0.0,
                "retrieval_mode": retrieval_mode,
                "retrieval_strategy": requested_retrieval_strategy,
                "faculty": faculty,
                "encode_batch_size": encode_batch_size,
                "target_chunking_seconds": round(
                    target_chunking_seconds, 4),
                "target_chunking_strategy": chunk_config["strategy"],
                "index_version": chunk_config["index_version"],
            },
            "error": "File content too short or empty after cleaning."
        }
    # 1. Encode trên GPU bằng model Singleton

    # Đã là Tensor 1024 chiều trên GPU
    target_encoding_started = time.time()
    target_vecs = encode_texts(target_chunks).to(dtype=torch.float32)
    target_vecs = torch.nn.functional.normalize(target_vecs, p=2, dim=1)
    target_encoding_seconds = time.time() - target_encoding_started

    if (
        forced_candidate_sources is None
        and requested_retrieval_strategy != "exhaustive"
    ):
        semantic_retrieval_started = time.time()
        semantic_suspects, semantic_retrieval_scores = (
            retrieve_semantic_candidate_sources(
                ml_state["collection"],
                target_vecs,
                collection_where=ml_state.get("collection_where"),
                allowed_sources=ml_state.get("filename_set"),
            )
        )
        semantic_retrieval_seconds = (
            time.time() - semantic_retrieval_started
        )
        phase_2_seconds += semantic_retrieval_seconds
        top_suspects = list(dict.fromkeys(
            semantic_suspects[:15] + tfidf_suspects
        ))[:HYBRID_CANDIDATE_LIMIT]
        print(
            f"Hybrid retrieval selected {len(top_suspects)} source(s): "
            f"{min(15, len(semantic_suspects))} vector-first + TF-IDF fallback."
        )

    semantic_results = {}
    semantic_cache_hits = 0
    semantic_cache_misses = 0
    semantic_source_load_seconds = 0.0
    semantic_similarity_seconds = 0.0
    semantic_evidence_build_seconds = 0.0
    source_vectors_examined = 0
    for suspect_fname in top_suspects:
        # Lấy từ ChromaDB (ChromaDB lưu vector dưới dạng list hoặc numpy)
        source_load_started = time.time()
        suspect_vecs_raw, suspect_docs, suspect_metadatas, cache_hit = (
            get_cached_semantic_source(ml_state, suspect_fname)
        )
        semantic_source_load_seconds += time.time() - source_load_started
        if cache_hit:
            semantic_cache_hits += 1
        else:
            semantic_cache_misses += 1

        # 2. CHUYỂN VỀ DẠNG NUMPY TRƯỚC (để xử lý an toàn)

        # Chuyển vector từ DB lên GPU và cast về float32
        similarity_started = time.time()
        suspect_vecs = torch.as_tensor(
            suspect_vecs_raw, device=device, dtype=torch.float32)
        source_vectors_examined += int(suspect_vecs.shape[0])
        if suspect_vecs.shape[0] == 0:
            semantic_similarity_seconds += time.time() - similarity_started
            semantic_results[suspect_fname] = {
                "peak": 0, "cov": 0, "evidence": []}
            continue

        # 2. Tính toán ma trận Cosine Similarity trên GPU (Cực nhanh)
        # Công thức: (A . B^T) / (||A|| * ||B||)
        # target_vecs (N, 1024), suspect_vecs (M, 1024)

        suspect_vecs = torch.nn.functional.normalize(
            suspect_vecs, p=2, dim=1)
        sim_matrix = torch.matmul(target_vecs, suspect_vecs.T)
        chunk_best_scores_tensor, chunk_best_indices_tensor = torch.max(
            sim_matrix, dim=1)
        chunk_best_scores = chunk_best_scores_tensor.cpu().numpy()
        chunk_best_indices = chunk_best_indices_tensor.cpu().numpy()
        semantic_similarity_seconds += time.time() - similarity_started

        # Tính coverage và Evidence
        coverage = (np.sum(chunk_best_scores > 0.7) / len(target_chunks)) * 100

        evidence_build_started = time.time()
        raw_evidence = []
        source_full_text = ml_state.get(
            "corpus_texts", {}).get(suspect_fname, "")

        # Only process chunks that pass your threshold
        valid_indices = np.where(
            chunk_best_scores >= SEMANTIC_FALLBACK_THRESHOLD)[0]

        for i in valid_indices:
            student_start, student_end = target_spans[i]
            source_index = int(chunk_best_indices[i])
            source_text = suspect_docs[source_index]
            source_metadata = suspect_metadatas[source_index] or {}
            source_start = source_metadata.get("start")
            source_end = source_metadata.get("end")
            if source_start is None:
                source_start = source_full_text.find(
                    source_text) if source_full_text else -1
            if source_end is None:
                source_end = (
                    source_start + len(source_text)
                    if source_start >= 0 else None
                )

            raw_evidence.append({
                "student_chunk": target_chunks[i],
                "student_start": student_start,
                "student_end": student_end,
                "source_text_raw": source_text,  # Just the raw matched string
                "source_start": source_start,
                "source_end": source_end,
                "source_chunk_index": source_index,
                "score": float(chunk_best_scores[i])
            })

        # 2. Sort by score to get the absolute best matches
        raw_evidence = sorted(
            raw_evidence, key=lambda x: x["score"], reverse=True)

        # 3. NOW extract the heavy text context ONLY for the top 3 (or top 10) results
        final_evidence = []
        # Change 3 to whatever max evidence limit you want
        for ev in raw_evidence[:200]:
            # Perform the heavy while-loop text expansion here
            student_context = expand_to_readable_context(
                test_content, ev["student_start"], ev["student_end"])

            if ev["source_start"] >= 0:
                source_context = expand_to_readable_context(
                    source_full_text, ev["source_start"], ev["source_end"])
            else:
                source_context = ev["source_text_raw"]

            final_evidence.append({
                "student": ev["student_chunk"],
                "student_text": student_context,
                "student_start": ev["student_start"],
                "student_end": ev["student_end"],
                "source": ev["source_text_raw"],
                "source_text": source_context,
                "source_start": ev["source_start"],
                "source_end": ev["source_end"],
                "source_chunk_index": ev["source_chunk_index"],
                "score": ev["score"],
                "type": "semantic",
                "label": "Semantic similarity / possible paraphrase"
            })

        semantic_results[suspect_fname] = {
            "peak": float(np.max(chunk_best_scores) * 100) if len(chunk_best_scores) > 0 else 0,
            "cov": float(coverage),
            "evidence": final_evidence  # Pass the lightweight array
        }
        semantic_evidence_build_seconds += (
            time.time() - evidence_build_started)
        del sim_matrix
        del suspect_vecs

    semantic_pruning_started = time.time()
    prune_semantic_evidence(semantic_results)
    semantic_pruning_seconds = time.time() - semantic_pruning_started
    lexical_started = time.time()
    lexical_evidence_count = add_lexical_window_evidence(
        test_content,
        top_suspects,
        ml_state,
        semantic_results,
    )
    lexical_seconds = time.time() - lexical_started

    # --- PHASE 4: INDUSTRIAL WINNOWING (BẢN CẬP NHẬT ĐỘ CHÍNH XÁC CAO) ---
    phase_3_seconds = (
        time.time() - phase_3_start - semantic_retrieval_seconds
    )
    print("\nStarting Phase 4: Robust Hashing...")
    phase_4_total_start = time.time()
    student_fingerprint_started = time.time()

    # 1. Lấy fingerprints của sinh viên
    student_hashes, student_clean_str = get_fingerprints(test_content)
    student_hash_set = set(student_hashes.keys())
    student_fingerprint_seconds = (
        time.time() - student_fingerprint_started)
    source_fingerprint_seconds = 0.0
    verbatim_matching_seconds = 0.0

    verbatim_results = {}

    for s in top_suspects:
        v_score = 0.0
        evidence = []
        # [FIX 1] Đặt mốc thời gian cho TỪNG file suspect
        file_start_time = time.time()

        try:
            source_fingerprint_started = time.time()
            # ƯU TIÊN 1: Lấy trực tiếp từ RAM
            source_raw = ml_state.get("corpus_texts", {}).get(s, "")

            # ƯU TIÊN 2: Đọc từ folder nếu RAM không có
            if not source_raw:
                # [FIX 2] Tách tên file an toàn bất kể là .pdf hay .txt
                base_name = os.path.splitext(s)[0]
                txt_filename = f"{base_name}.txt"
                potential_path = os.path.join(corpus_dir, txt_filename)

                if os.path.exists(potential_path):
                    with open(potential_path, 'r', encoding='utf-8', errors='ignore') as f:
                        source_raw = f.read()

            if not source_raw:
                print(f"Warning: Không tìm thấy nội dung sạch cho {s}")
                continue

            # Tiến hành so khớp hashing
            fingerprint_cache = ml_state.setdefault(
                "fingerprint_cache", OrderedDict())
            if not isinstance(fingerprint_cache, OrderedDict):
                fingerprint_cache = OrderedDict(fingerprint_cache)
                ml_state["fingerprint_cache"] = fingerprint_cache
            if s in fingerprint_cache:
                source_hashes_data = fingerprint_cache.pop(s)
                fingerprint_cache[s] = source_hashes_data
            else:
                source_hashes_data, _ = get_fingerprints(source_raw)
                fingerprint_cache[s] = source_hashes_data
                while len(fingerprint_cache) > SOURCE_CACHE_MAX_FILES:
                    fingerprint_cache.popitem(last=False)
            source_hash_set = set(source_hashes_data.keys())
            source_fingerprint_seconds += (
                time.time() - source_fingerprint_started)

            # 4. Giao thoa
            verbatim_matching_started = time.time()
            common_hashes = student_hash_set.intersection(source_hash_set)

            # 5. Tính toán Coverage
            if len(student_hash_set) > 0:
                v_score = (len(common_hashes) / len(student_hash_set)) * 100
            else:
                v_score = 0

            # 6. Thu thập bằng chứng
            sorted_common = sorted(list(common_hashes),
                                   key=lambda x: student_hashes[x]["pos"])

            anchors = []
            for h in sorted_common:
                info = student_hashes[h]
                source_info = source_hashes_data[h]
                anchors.append({
                    "student_start": info["student_start"],
                    "student_end": info["student_end"],
                    "source_start": source_info["student_start"],
                    "source_end": source_info["student_end"],
                })

            groups = [
                group
                for group in consolidate_verbatim_anchors(anchors)
                if (
                    max(anchor["student_end"] for anchor in group)
                    - group[0]["student_start"]
                    >= VERBATIM_MIN_MATCH_CHARS
                )
            ]
            for group in groups[:500]:
                student_start = group[0]["student_start"]
                student_end = max(
                    anchor["student_end"] for anchor in group)
                source_start = group[0]["source_start"]
                source_end = max(
                    anchor["source_end"] for anchor in group)
                student_context = expand_to_readable_context(
                    test_content, student_start, student_end)
                evidence.append({
                    "text": student_context,
                    "student_text": student_context,
                    "source_text": expand_to_readable_context(
                        source_raw, source_start, source_end),
                    "student_start": student_start,
                    "student_end": student_end,
                    "source_start": source_start,
                    "source_end": source_end,
                    "match_text": test_content[
                        student_start:student_end],
                    "type": "verbatim",
                    "label": "Exact copied text",
                    "match_type": "Chính xác tuyệt đối"
                })

            # [FIX 1] In ra thời gian xử lý chính xác của riêng file này
            verbatim_matching_seconds += (
                time.time() - verbatim_matching_started)
            print(
                f"Phase 4 Optimized Time for {s}: {time.time() - file_start_time:.2f}s | Matches: {len(common_hashes)}")

        except Exception as e:
            print(f"Error in Phase 4 for {s}: {e}")

        # [FIX 3] Gom việc gán kết quả ra ngoài cùng khối try-except để code gọn gàng hơn
        verbatim_results[s] = {
            "score": round(v_score, 1),
            "evidence": evidence
        }

    phase_4_seconds = time.time() - phase_4_total_start
    print(f"\n--- Total Phase 4 Time: {phase_4_seconds:.2f}s ---")
    phase_5_seconds = 0.0
    phase_6_seconds = 0.0
    phase_7_seconds = 0.0
    final_results = []
    raw_evidence_count = 0
    consolidated_evidence_count = 0

    for s in top_suspects:
        # 1. Fetch results from Phase 3 and Phase 4
        v_data = verbatim_results.get(s, {"score": 0.0, "evidence": []})
        v_rate = v_data["score"]
        c_data = semantic_results.get(
            s, {"peak": 0.0, "cov": 0.0, "evidence": []})
        c_rate = c_data['cov']

        # Final score is the higher of the two detection methods
        final_score = max(v_rate, c_rate)
        phase_5_started = time.time()
        raw_combined_evidence = merge_evidence(
            v_data["evidence"], c_data["evidence"], s)
        phase_5_seconds += time.time() - phase_5_started

        phase_6_started = time.time()
        combined_evidence = consolidate_evidence_passages(
            raw_combined_evidence,
            test_content,
            ml_state.get("corpus_texts", {}).get(s, ""),
            s,
        )
        phase_6_seconds += time.time() - phase_6_started
        raw_evidence_count += len(raw_combined_evidence)
        consolidated_evidence_count += len(combined_evidence)

        # 2. Print Summary Row
        print(
            f"{s[:35]:<35} | {v_rate:>8.1f}% | {c_rate:>8.1f}% | {final_score:>10.1f}%")

        # 3. Keep all Phase 6 passages for Phase 7 and evaluator/debug output.
        final_results.append({
            "filename": s,
            "verbatim_score": v_rate,
            "semantic_peak": round(c_data['peak'], 1),
            "semantic_coverage": round(c_rate, 1),
            "final_score": round(final_score, 1),
            "evidence": [],
            "consolidated_evidence": combined_evidence,
            "raw_evidence": raw_combined_evidence,
            "verbatim_evidence": v_data["evidence"],
            "semantic_evidence": c_data['evidence']
        })

    phase_7_started = time.time()
    evidence_selection = select_display_evidence(
        final_results, profile=evidence_profile)
    phase_7_seconds = time.time() - phase_7_started

    for result in final_results:
        selected_evidence = result["evidence"]
        if not selected_evidence:
            continue
        print(f"\n   >>> [!] EVIDENCE FOUND IN: {result['filename']}")
        counts = {
            evidence_type: sum(
                item.get("type") == evidence_type
                for item in selected_evidence
            )
            for evidence_type in (
                "semantic", "verbatim", "verbatim_semantic")
        }
        summary = result["evidence_summary"]
        print(
            "       Phase 7 display: "
            f"profile={summary['profile']} | shown={summary['shown_count']} | "
            f"hidden={summary['hidden_count']} | "
            f"semantic={counts['semantic']} | "
            f"verbatim={counts['verbatim']} | "
            f"mixed={counts['verbatim_semantic']}"
        )
        display_list = sorted(
            selected_evidence,
            key=lambda item: item.get("confidence_score", 0.0),
            reverse=True,
        )[:6]
        for idx, evidence in enumerate(display_list, 1):
            student_snippet = " ".join(
                (evidence.get("student_text") or "").split())
            source_snippet = " ".join(
                (evidence.get("source_text") or "").split())
            if len(student_snippet) > 160:
                student_snippet = student_snippet[:157] + "..."
            if len(source_snippet) > 160:
                source_snippet = source_snippet[:157] + "..."
            semantic_text = (
                f" | semantic={evidence['semantic_score']:.3f}"
                if evidence.get("semantic_score") is not None else ""
            )
            print(
                f"       {idx}. [{evidence['type'].upper()}] "
                f"confidence={evidence['confidence_score']:.3f} | "
                f"student={evidence.get('student_start')}:"
                f"{evidence.get('student_end')} | "
                f"source={evidence.get('source_start')}:"
                f"{evidence.get('source_end')}{semantic_text}"
            )
            print(f"          > Student: \"{student_snippet}\"")
            print(f"          > Source : \"{source_snippet}\"")

    print(
        "\nPhase 6 passage consolidation: "
        f"{raw_evidence_count} raw evidence -> "
        f"{consolidated_evidence_count} passages "
        f"in {phase_6_seconds:.2f}s"
    )
    print(
        "Phase 7 evidence selection: "
        f"profile={evidence_profile} | "
        f"{evidence_selection['candidate_count']} candidates -> "
        f"{evidence_selection['shown_count']} displayed "
        f"in {phase_7_seconds:.2f}s"
    )
    print("="*95)
    total_time = time.time() - start_time
    ram_usage = process.memory_info().rss / (1024 ** 2)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    print(
        f"HIỆU NĂNG: Tổng thời gian: {total_time:.2f}s | RAM sử dụng: {ram_usage:.1f} MB | Thiết bị: CPU/GPU")
    print("="*95 + "\n")

    return {
        "results": final_results,
        "execution_time": round(total_time, 2),
        "performance": {
            "phase_2_seconds": round(phase_2_seconds, 4),
            "phase_3_seconds": round(phase_3_seconds, 4),
            "semantic_target_encoding_seconds": round(
                target_encoding_seconds, 4),
            "semantic_source_load_seconds": round(
                semantic_source_load_seconds, 4),
            "semantic_similarity_seconds": round(
                semantic_similarity_seconds, 4),
            "semantic_evidence_build_seconds": round(
                semantic_evidence_build_seconds, 4),
            "semantic_pruning_seconds": round(
                semantic_pruning_seconds, 4),
            "semantic_lexical_seconds": round(lexical_seconds, 4),
            "phase_4_seconds": round(phase_4_seconds, 4),
            "verbatim_student_fingerprint_seconds": round(
                student_fingerprint_seconds, 4),
            "verbatim_source_fingerprint_seconds": round(
                source_fingerprint_seconds, 4),
            "verbatim_matching_seconds": round(
                verbatim_matching_seconds, 4),
            "phase_5_seconds": round(phase_5_seconds, 4),
            "phase_6_seconds": round(phase_6_seconds, 4),
            "phase_7_seconds": round(phase_7_seconds, 4),
            "ram_rss_mb": round(ram_usage, 1),
            "peak_torch_vram_mb": round(peak_vram_mb, 1),
            "target_chunks": len(target_chunks),
            "target_chunking_seconds": round(target_chunking_seconds, 4),
            "target_chunking_strategy": chunk_config["strategy"],
            "index_version": chunk_config["index_version"],
            "candidate_sources": len(top_suspects),
            "retrieval_mode": retrieval_mode,
            "retrieval_strategy": requested_retrieval_strategy,
            "faculty": faculty,
            "encode_batch_size": encode_batch_size,
            "hybrid_candidate_limit": HYBRID_CANDIDATE_LIMIT,
            "semantic_retrieval_sources": len(semantic_retrieval_scores),
            "source_vectors_examined": source_vectors_examined,
            "semantic_strong_threshold": SEMANTIC_STRONG_THRESHOLD,
            "semantic_fallback_threshold": SEMANTIC_FALLBACK_THRESHOLD,
            "semantic_fallback_margin": SEMANTIC_FALLBACK_MARGIN,
            "lexical_window_evidence": lexical_evidence_count,
            "lexical_score_threshold": LEXICAL_SCORE_THRESHOLD,
            "raw_evidence_count": raw_evidence_count,
            "consolidated_evidence_count": consolidated_evidence_count,
            "evidence_count": evidence_selection["shown_count"],
            "evidence_hidden_count": evidence_selection["hidden_count"],
            "evidence_profile": evidence_profile,
            "evidence_rejections": evidence_selection["rejections"],
            "evidence_reduction_percent": round(
                (
                    1.0 - evidence_selection["shown_count"]
                    / raw_evidence_count
                ) * 100,
                2,
            ) if raw_evidence_count else 0.0,
            "semantic_cache_hits": semantic_cache_hits,
            "semantic_cache_misses": semantic_cache_misses,
            "semantic_cache_sources": len(
                ml_state.get("semantic_source_cache", {})),
            "fingerprint_cache_sources": len(
                ml_state.get("fingerprint_cache", {})),
            "vector_db_chunks": ml_state.get("vector_db_chunks"),
            "device": device,
        }
    }
