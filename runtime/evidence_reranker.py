import math
import re
from collections import defaultdict


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
FEATURE_NAMES = (
    "semantic_score",
    "semantic_present",
    "lexical_score",
    "lexical_present",
    "text_similarity",
    "legacy_confidence",
    "source_semantic_coverage",
    "source_verbatim_score",
    "source_final_score",
    "evidence_count_log",
    "semantic_support_log",
    "verbatim_support_log",
    "lexical_support_log",
    "verbatim_characters_log",
    "student_span_log",
    "source_span_log",
    "span_length_ratio",
    "token_jaccard",
    "token_containment",
    "char_5gram_jaccard",
    "char_5gram_containment",
    "dense_method",
    "lexical_method",
    "fingerprint_method",
    "semantic_type",
    "verbatim_type",
    "mixed_type",
    "source_candidate_count_log",
    "nearby_same_source_count_log",
    "nearby_any_source_count_log",
    "source_rank_reciprocal",
    "candidate_rank_reciprocal",
    "student_position_ratio",
    "semantic_lexical_agreement",
    "semantic_above_fallback",
)


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _span_length(evidence, prefix):
    start = evidence.get(f"{prefix}_start")
    end = evidence.get(f"{prefix}_end")
    if not isinstance(start, int) or not isinstance(end, int):
        return 0
    return max(0, end - start)


def _tokens(text):
    return set(TOKEN_PATTERN.findall((text or "").casefold()))


def _character_shingles(text, size=5, limit=3000):
    compact = "".join((text or "").casefold().split())[:limit]
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index:index + size]
            for index in range(len(compact) - size + 1)}


def _set_scores(left, right):
    if not left or not right:
        return 0.0, 0.0
    overlap = len(left & right)
    union = len(left | right)
    shorter = min(len(left), len(right))
    return (
        overlap / union if union else 0.0,
        overlap / shorter if shorter else 0.0,
    )


def _valid_span(evidence, prefix="student"):
    start = evidence.get(f"{prefix}_start")
    end = evidence.get(f"{prefix}_end")
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end
    )


def _interval_overlap_ratio(left, right):
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    shorter = min(left[1] - left[0], right[1] - right[0])
    return overlap / shorter if shorter > 0 else 0.0


def _span_gap(left, right):
    return max(0, max(left[0], right[0]) - min(left[1], right[1]))


def evidence_feature_vector(evidence, result, context=None):
    context = context or {}
    semantic_score = _safe_float(evidence.get("semantic_score"))
    lexical_score = _safe_float(evidence.get("lexical_score"))
    text_similarity = _safe_float(evidence.get("text_similarity"))
    student_span = _span_length(evidence, "student")
    source_span = _span_length(evidence, "source")
    longer_span = max(student_span, source_span)
    length_ratio = (
        min(student_span, source_span) / longer_span
        if longer_span else 0.0
    )

    student_text = (
        evidence.get("student_match_text")
        or evidence.get("student_text")
        or evidence.get("student")
        or ""
    )
    source_text = (
        evidence.get("source_match_text")
        or evidence.get("source_text")
        or evidence.get("source")
        or ""
    )
    token_jaccard, token_containment = _set_scores(
        _tokens(student_text), _tokens(source_text))
    char_jaccard, char_containment = _set_scores(
        _character_shingles(student_text),
        _character_shingles(source_text),
    )

    methods = set(evidence.get("match_methods") or [])
    if not methods and evidence.get("match_method"):
        methods = set(str(evidence["match_method"]).split("+"))
    evidence_type = evidence.get("type", "semantic")
    document_end = max(1, int(context.get("document_end") or 1))
    student_start = evidence.get("student_start")
    student_position = (
        max(0.0, min(1.0, student_start / document_end))
        if isinstance(student_start, int) else 0.0
    )

    values = {
        "semantic_score": semantic_score,
        "semantic_present": float(evidence.get("semantic_score") is not None),
        "lexical_score": lexical_score,
        "lexical_present": float(evidence.get("lexical_score") is not None),
        "text_similarity": text_similarity,
        "legacy_confidence": _safe_float(
            evidence.get("confidence_score")),
        "source_semantic_coverage": (
            _safe_float(result.get("semantic_coverage")) / 100.0),
        "source_verbatim_score": (
            _safe_float(result.get("verbatim_score")) / 100.0),
        "source_final_score": (
            _safe_float(result.get("final_score")) / 100.0),
        "evidence_count_log": math.log1p(
            max(0, int(evidence.get("evidence_count") or 0))),
        "semantic_support_log": math.log1p(
            max(0, int(evidence.get("semantic_support_count") or 0))),
        "verbatim_support_log": math.log1p(
            max(0, int(evidence.get("verbatim_support_count") or 0))),
        "lexical_support_log": math.log1p(
            max(0, int(evidence.get("lexical_support_count") or 0))),
        "verbatim_characters_log": math.log1p(
            max(0, int(evidence.get("verbatim_characters") or 0))),
        "student_span_log": math.log1p(student_span),
        "source_span_log": math.log1p(source_span),
        "span_length_ratio": length_ratio,
        "token_jaccard": token_jaccard,
        "token_containment": token_containment,
        "char_5gram_jaccard": char_jaccard,
        "char_5gram_containment": char_containment,
        "dense_method": float("dense_embedding" in methods),
        "lexical_method": float("lexical_window" in methods),
        "fingerprint_method": float(
            "verbatim_fingerprint" in methods),
        "semantic_type": float(evidence_type == "semantic"),
        "verbatim_type": float(evidence_type == "verbatim"),
        "mixed_type": float(evidence_type == "verbatim_semantic"),
        "source_candidate_count_log": math.log1p(
            max(0, int(context.get("source_candidate_count") or 0))),
        "nearby_same_source_count_log": math.log1p(
            max(0, int(context.get("nearby_same_source_count") or 0))),
        "nearby_any_source_count_log": math.log1p(
            max(0, int(context.get("nearby_any_source_count") or 0))),
        "source_rank_reciprocal": 1.0 / (
            1.0 + max(0, int(context.get("source_rank") or 0))),
        "candidate_rank_reciprocal": 1.0 / (
            1.0 + max(0, int(context.get("candidate_rank") or 0))),
        "student_position_ratio": student_position,
        "semantic_lexical_agreement": semantic_score * lexical_score,
        "semantic_above_fallback": max(0.0, semantic_score - 0.68),
    }
    return [values[name] for name in FEATURE_NAMES]


def build_candidate_records(results, confidence_scorer=None):
    records = []
    document_end = 0
    for result_index, result in enumerate(results):
        for evidence_index, original in enumerate(
                result.get("consolidated_evidence", ())):
            evidence = dict(original)
            evidence.setdefault("source_file", result.get("filename", ""))
            if confidence_scorer is not None:
                evidence["confidence_score"] = confidence_scorer(
                    evidence, result)
            document_end = max(
                document_end,
                int(evidence.get("student_end") or 0),
            )
            records.append({
                "record_id": f"{result_index}:{evidence_index}",
                "result_index": result_index,
                "evidence_index": evidence_index,
                "source_file": evidence.get("source_file", ""),
                "evidence": evidence,
                "result": result,
            })

    records_by_source = defaultdict(list)
    for record in records:
        records_by_source[record["source_file"]].append(record)

    for source_records in records_by_source.values():
        ranked = sorted(
            source_records,
            key=lambda record: (
                _safe_float(record["evidence"].get("confidence_score")),
                _safe_float(record["evidence"].get("semantic_score")),
                _safe_float(record["evidence"].get("lexical_score")),
                int(record["evidence"].get("evidence_count") or 0),
            ),
            reverse=True,
        )
        for rank, record in enumerate(ranked):
            record["candidate_rank"] = rank

    for record in records:
        evidence = record["evidence"]
        if _valid_span(evidence):
            candidate_span = (
                evidence["student_start"], evidence["student_end"])
            nearby = []
            for other in records:
                if other is record or not _valid_span(other["evidence"]):
                    continue
                other_span = (
                    other["evidence"]["student_start"],
                    other["evidence"]["student_end"],
                )
                if (
                    _interval_overlap_ratio(candidate_span, other_span) > 0
                    or _span_gap(candidate_span, other_span) <= 1000
                ):
                    nearby.append(other)
        else:
            nearby = []
        context = {
            "document_end": document_end,
            "source_candidate_count": len(
                records_by_source[record["source_file"]]),
            "nearby_same_source_count": sum(
                other["source_file"] == record["source_file"]
                for other in nearby
            ),
            "nearby_any_source_count": len(nearby),
            "source_rank": record["result_index"],
            "candidate_rank": record.get("candidate_rank", 0),
        }
        record["features"] = evidence_feature_vector(
            evidence, record["result"], context)
    return records


def agreement_score(record):
    feature = dict(zip(FEATURE_NAMES, record["features"]))
    return max(0.0, min(1.0, (
        0.30 * feature["legacy_confidence"]
        + 0.20 * feature["semantic_score"]
        + 0.12 * feature["lexical_score"]
        + 0.10 * feature["token_containment"]
        + 0.10 * feature["char_5gram_containment"]
        + 0.08 * feature["mixed_type"]
        + 0.05 * feature["fingerprint_method"]
        + 0.05 * feature["candidate_rank_reciprocal"]
    )))


def select_ranked_records(
        records,
        scores,
        threshold,
        max_total=220,
        max_per_source=None,
        same_source_overlap=0.45,
        same_source_gap=250,
        cross_source_overlap=0.75,
        max_sources_per_region=2):
    ranked = sorted(
        zip(records, scores),
        key=lambda item: (
            _safe_float(item[1]),
            int(item[0]["evidence"].get("evidence_count") or 0),
            int(item[0]["evidence"].get("verbatim_characters") or 0),
            _safe_float(item[0]["evidence"].get("semantic_score")),
        ),
        reverse=True,
    )
    selected = []
    source_counts = defaultdict(int)
    for record, score in ranked:
        if score < threshold or len(selected) >= max_total:
            continue
        evidence = record["evidence"]
        if not (_valid_span(evidence, "student")
                and _valid_span(evidence, "source")):
            continue
        source_file = record["source_file"]
        if (
            max_per_source is not None
            and source_counts[source_file] >= max_per_source
        ):
            continue

        student_span = (
            evidence["student_start"], evidence["student_end"])
        duplicate = False
        competing_sources = set()
        for kept in selected:
            kept_evidence = kept["evidence"]
            kept_span = (
                kept_evidence["student_start"],
                kept_evidence["student_end"],
            )
            overlap = _interval_overlap_ratio(student_span, kept_span)
            if kept["source_file"] == source_file:
                if (
                    overlap >= same_source_overlap
                    or _span_gap(student_span, kept_span) <= same_source_gap
                ):
                    duplicate = True
                    break
            elif overlap >= cross_source_overlap:
                competing_sources.add(kept["source_file"])
        if duplicate:
            continue
        if len(competing_sources) >= max_sources_per_region:
            continue

        selected_record = dict(record)
        selected_record["reranker_score"] = _safe_float(score)
        selected.append(selected_record)
        source_counts[source_file] += 1
    return selected
