"""Diagnostic-only correction policy proposal helpers."""

from __future__ import annotations

import ast
from typing import Any


SUPPORTED_CORRECTION_POLICIES = ("conservative", "balanced", "liberal")

MODE_DYNAMIC_ISOSBESTIC = "dynamic_isosbestic"
MODE_BASELINE_LEGACY = "baseline_reference_candidate"
MODE_NO_ISOSBESTIC = "no_isosbestic_candidate"
MODE_NO_CLEAN_REFERENCE = "no_clean_reference_candidate"
MODE_REVIEW = "review_required"

POLICY_FIELD_STEMS = (
    "proposed_correction_mode",
    "proposal_confidence",
    "review_required",
    "review_queue_candidate",
    "review_priority",
    "warning_level",
    "proposal_reason",
    "proposal_flags",
)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_NONE = "none"

PRIORITY_NONE = "none"
PRIORITY_LOW = "low"
PRIORITY_MEDIUM = "medium"
PRIORITY_HIGH = "high"

WARNING_NONE = "none"
WARNING_CONTEXTUAL = "contextual"
WARNING_CAUTION = "caution"
WARNING_SEVERE = "severe"

FLAG_DYNAMIC_ACCEPTED = "DYNAMIC_ACCEPTED"
FLAG_LEGACY_BASELINE_DIAGNOSTIC_PRESENT = "LEGACY_REFERENCE_BASELINE_DIAGNOSTIC_PRESENT"
FLAG_LEGACY_BASELINE_VIABLE_DIAGNOSTIC = "LEGACY_REFERENCE_BASELINE_WAS_VIABLE_DIAGNOSTIC"
FLAG_SIGNAL_ONLY_F0_NOT_AVAILABLE = "SIGNAL_ONLY_F0_FALLBACK_NOT_YET_AVAILABLE"
FLAG_DYNAMIC_HARD = "DYNAMIC_HARD_INSPECT"
FLAG_DYNAMIC_CONTEXTUAL = "DYNAMIC_CONTEXTUAL"
FLAG_BASELINE_CONTEXTUAL = "BASELINE_CONTEXTUAL"
FLAG_BASELINE_HARD = "BASELINE_HARD_INSPECT"
FLAG_BASELINE_POSITIVE = "BASELINE_POSITIVE_REFERENCE_RELATIONSHIP"
FLAG_BASELINE_NEGATIVE = "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP"
FLAG_BASELINE_WEAK = "BASELINE_WEAK_REFERENCE_RELATIONSHIP"
FLAG_INVERTED = "INVERTED_REFERENCE_RELATIONSHIP"
FLAG_BOTH_CONTEXTUAL = "BOTH_CANDIDATES_CONTEXTUAL"
FLAG_BOTH_HARD_OR_UNAVAILABLE = "BOTH_CANDIDATES_HARD_OR_UNAVAILABLE"
FLAG_REVIEW_BY_POLICY = "REVIEW_REQUIRED_BY_POLICY"
FLAG_NO_CLEAN_REFERENCE = "NO_CLEAN_REFERENCE_CANDIDATE"


def policy_field_names() -> list[str]:
    """Return all per-policy proposal field names in stable output order."""
    return [
        f"{stem}_{policy}"
        for policy in SUPPORTED_CORRECTION_POLICIES
        for stem in POLICY_FIELD_STEMS
    ]


def normalize_policy_flags(value: Any) -> list[str]:
    """Normalize proposal/comparison flag values from JSON, CSV, or repr strings."""
    if value is None:
        return []
    if isinstance(value, float) and value != value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(part).strip() for part in parsed if str(part).strip()]
        except Exception:
            pass
    return [part.strip() for part in text.split(";") if part.strip()]


def _baseline_relationship_class(record: dict[str, Any]) -> str:
    relationship = _text(record.get("baseline_fit_relationship_class"))
    if relationship in {
        "positive_reference_relationship",
        "negative_reference_relationship",
        "weak_reference_relationship",
        "mixed_or_unclear_reference_relationship",
        "unknown",
    }:
        return relationship
    return "unknown"


def _baseline_relationship_is_clean_positive(record: dict[str, Any]) -> bool:
    return _baseline_relationship_class(record) == "positive_reference_relationship"


def _as_flag_set(value: Any) -> set[str]:
    return set(normalize_policy_flags(value))


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else "").strip().lower()
    return text or default


def _base_flags(record: dict[str, Any], dynamic: str, baseline: str) -> list[str]:
    comparison_flags = _as_flag_set(record.get("reference_comparison_flags"))
    relationship = _baseline_relationship_class(record)
    out: list[str] = []
    if dynamic == "hard_inspect":
        out.append(FLAG_DYNAMIC_HARD)
    elif dynamic == "contextual":
        out.append(FLAG_DYNAMIC_CONTEXTUAL)
    if baseline == "hard_inspect":
        out.append(FLAG_BASELINE_HARD)
    elif baseline == "contextual":
        out.append(FLAG_BASELINE_CONTEXTUAL)
    elif baseline == "viable":
        out.append(FLAG_LEGACY_BASELINE_VIABLE_DIAGNOSTIC)
    if baseline in {"viable", "contextual", "hard_inspect"}:
        out.append(FLAG_LEGACY_BASELINE_DIAGNOSTIC_PRESENT)
    if (
        "BASELINE_POSITIVE_REFERENCE_RELATIONSHIP" in comparison_flags
        or relationship == "positive_reference_relationship"
    ):
        out.append(FLAG_BASELINE_POSITIVE)
    if (
        "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP" in comparison_flags
        or relationship == "negative_reference_relationship"
    ):
        out.extend([FLAG_BASELINE_NEGATIVE, FLAG_INVERTED])
    if (
        "BASELINE_WEAK_REFERENCE_RELATIONSHIP" in comparison_flags
        or relationship == "weak_reference_relationship"
    ):
        out.append(FLAG_BASELINE_WEAK)
    if dynamic == "contextual" and baseline == "contextual":
        out.append(FLAG_BOTH_CONTEXTUAL)
    if dynamic == "hard_inspect" and baseline in {"hard_inspect", "unavailable"}:
        out.append(FLAG_BOTH_HARD_OR_UNAVAILABLE)
    return out


def _proposal(
    *,
    policy: str,
    mode: str,
    confidence: str,
    review_required: bool,
    review_queue_candidate: bool,
    review_priority: str,
    warning_level: str,
    reason: str,
    flags: list[str],
) -> dict[str, Any]:
    return {
        "correction_policy": policy,
        "proposed_correction_mode": mode,
        "proposal_confidence": confidence,
        "review_required": bool(review_required),
        "review_queue_candidate": bool(review_queue_candidate),
        "review_priority": review_priority,
        "warning_level": warning_level,
        "proposal_reason": reason,
        "proposal_flags": list(dict.fromkeys(flags)),
    }


def _balanced(record: dict[str, Any], flags: list[str], dynamic: str, baseline: str) -> dict[str, Any]:
    baseline_clean_positive = _baseline_relationship_is_clean_positive(record)
    if dynamic == "viable":
        return _proposal(
            policy="balanced",
            mode=MODE_DYNAMIC_ISOSBESTIC,
            confidence=CONFIDENCE_HIGH,
            review_required=False,
            review_queue_candidate=False,
            review_priority=PRIORITY_NONE,
            warning_level=WARNING_NONE,
            reason="dynamic_isosbestic_viable",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "hard_inspect" and baseline == "viable" and baseline_clean_positive:
        return _proposal(
            policy="balanced",
            mode=MODE_NO_CLEAN_REFERENCE,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_MEDIUM,
            warning_level=WARNING_CAUTION,
            reason="dynamic_hard_inspect_legacy_reference_baseline_candidate_not_policy_fallback",
            flags=[*flags, FLAG_NO_CLEAN_REFERENCE, FLAG_SIGNAL_ONLY_F0_NOT_AVAILABLE],
        )
    if dynamic == "hard_inspect" and baseline == "viable":
        return _proposal(
            policy="balanced",
            mode=MODE_NO_CLEAN_REFERENCE,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_MEDIUM,
            warning_level=WARNING_CAUTION,
            reason="dynamic_hard_inspect_baseline_viable_without_clean_positive_relationship",
            flags=[*flags, FLAG_NO_CLEAN_REFERENCE, FLAG_SIGNAL_ONLY_F0_NOT_AVAILABLE],
        )
    if dynamic == "contextual" and baseline == "viable":
        return _proposal(
            policy="balanced",
            mode=MODE_NO_CLEAN_REFERENCE,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_MEDIUM,
            warning_level=WARNING_CAUTION,
            reason="contextual_dynamic_with_legacy_reference_baseline_candidate",
            flags=[*flags, FLAG_NO_CLEAN_REFERENCE, FLAG_SIGNAL_ONLY_F0_NOT_AVAILABLE],
        )
    if dynamic == "contextual" and baseline == "contextual":
        return _proposal(
            policy="balanced",
            mode=MODE_NO_CLEAN_REFERENCE,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_LOW,
            warning_level=WARNING_CONTEXTUAL,
            reason="contextual_reference_evidence_no_clean_reference_candidate",
            flags=[*flags, FLAG_NO_CLEAN_REFERENCE],
        )
    if dynamic == "hard_inspect" and baseline == "contextual":
        return _proposal(
            policy="balanced",
            mode=MODE_REVIEW,
            confidence=CONFIDENCE_LOW,
            review_required=True,
            review_queue_candidate=True,
            review_priority=PRIORITY_HIGH,
            warning_level=WARNING_SEVERE,
            reason="dynamic_hard_inspect_no_clean_reference_candidate",
            flags=[*flags, FLAG_REVIEW_BY_POLICY, FLAG_NO_CLEAN_REFERENCE],
        )
    if dynamic == "hard_inspect" and baseline in {"hard_inspect", "unavailable"}:
        return _proposal(
            policy="balanced",
            mode=MODE_REVIEW,
            confidence=CONFIDENCE_NONE,
            review_required=True,
            review_queue_candidate=True,
            review_priority=PRIORITY_HIGH,
            warning_level=WARNING_SEVERE,
            reason="both_reference_candidates_hard_or_unavailable",
            flags=[*flags, FLAG_REVIEW_BY_POLICY, FLAG_NO_CLEAN_REFERENCE],
        )
    return _proposal(
        policy="balanced",
        mode=MODE_REVIEW,
        confidence=CONFIDENCE_NONE,
        review_required=True,
        review_queue_candidate=True,
        review_priority=PRIORITY_HIGH,
        warning_level=WARNING_SEVERE,
        reason="unknown_reference_policy_state",
        flags=[*flags, FLAG_REVIEW_BY_POLICY, FLAG_NO_CLEAN_REFERENCE],
    )


def _conservative(record: dict[str, Any], flags: list[str], dynamic: str, baseline: str) -> dict[str, Any]:
    baseline_clean_positive = _baseline_relationship_is_clean_positive(record)
    if dynamic == "viable":
        return _proposal(
            policy="conservative",
            mode=MODE_DYNAMIC_ISOSBESTIC,
            confidence=CONFIDENCE_HIGH,
            review_required=False,
            review_queue_candidate=False,
            review_priority=PRIORITY_NONE,
            warning_level=WARNING_NONE,
            reason="dynamic_isosbestic_viable_under_conservative_policy",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "hard_inspect" and baseline == "viable" and baseline_clean_positive:
        return _proposal(
            policy="conservative",
            mode=MODE_NO_CLEAN_REFERENCE,
            confidence=CONFIDENCE_LOW,
            review_required=True,
            review_queue_candidate=True,
            review_priority=PRIORITY_HIGH,
            warning_level=WARNING_CAUTION,
            reason="legacy_reference_baseline_candidate_not_policy_fallback",
            flags=[
                *flags,
                FLAG_REVIEW_BY_POLICY,
                FLAG_NO_CLEAN_REFERENCE,
                FLAG_SIGNAL_ONLY_F0_NOT_AVAILABLE,
            ],
        )
    priority = PRIORITY_HIGH if dynamic == "hard_inspect" or baseline != "viable" else PRIORITY_MEDIUM
    return _proposal(
        policy="conservative",
        mode=MODE_REVIEW,
        confidence=CONFIDENCE_LOW if dynamic == "contextual" else CONFIDENCE_NONE,
        review_required=True,
        review_queue_candidate=True,
        review_priority=priority,
        warning_level=WARNING_CAUTION if dynamic == "contextual" else WARNING_SEVERE,
        reason="conservative_policy_requires_review_for_contextual_or_unclean_evidence",
        flags=[*flags, FLAG_REVIEW_BY_POLICY],
    )


def _liberal(record: dict[str, Any], flags: list[str], dynamic: str, baseline: str) -> dict[str, Any]:
    baseline_clean_positive = _baseline_relationship_is_clean_positive(record)
    if dynamic == "viable":
        return _proposal(
            policy="liberal",
            mode=MODE_DYNAMIC_ISOSBESTIC,
            confidence=CONFIDENCE_HIGH,
            review_required=False,
            review_queue_candidate=False,
            review_priority=PRIORITY_NONE,
            warning_level=WARNING_NONE,
            reason="dynamic_isosbestic_viable",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "contextual" and baseline in {"viable", "contextual"}:
        return _proposal(
            policy="liberal",
            mode=MODE_NO_CLEAN_REFERENCE,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_LOW,
            warning_level=WARNING_CONTEXTUAL,
            reason="liberal_policy_logs_contextual_reference_conflict_for_screening",
            flags=[*flags, FLAG_NO_CLEAN_REFERENCE],
        )
    if dynamic == "hard_inspect" and baseline == "viable" and baseline_clean_positive:
        return _proposal(
            policy="liberal",
            mode=MODE_NO_CLEAN_REFERENCE,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_MEDIUM,
            warning_level=WARNING_CAUTION,
            reason="liberal_policy_legacy_reference_baseline_candidate_not_policy_fallback",
            flags=[*flags, FLAG_NO_CLEAN_REFERENCE, FLAG_SIGNAL_ONLY_F0_NOT_AVAILABLE],
        )
    if dynamic == "hard_inspect" and baseline == "contextual":
        return _proposal(
            policy="liberal",
            mode=MODE_REVIEW,
            confidence=CONFIDENCE_LOW,
            review_required=True,
            review_queue_candidate=True,
            review_priority=PRIORITY_HIGH,
            warning_level=WARNING_SEVERE,
            reason="dynamic_hard_inspect_baseline_contextual",
            flags=[*flags, FLAG_REVIEW_BY_POLICY, FLAG_NO_CLEAN_REFERENCE],
        )
    return _proposal(
        policy="liberal",
        mode=MODE_REVIEW,
        confidence=CONFIDENCE_NONE,
        review_required=True,
        review_queue_candidate=True,
        review_priority=PRIORITY_HIGH,
        warning_level=WARNING_SEVERE,
        reason="no_clean_reference_candidate",
        flags=[*flags, FLAG_REVIEW_BY_POLICY, FLAG_NO_CLEAN_REFERENCE],
    )


def propose_correction_policy(
    *,
    comparison_record: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    """Return diagnostic-only policy proposal fields for one ROI/chunk record."""
    policy_norm = str(policy or "").strip().lower()
    if policy_norm not in SUPPORTED_CORRECTION_POLICIES:
        raise ValueError(f"Unsupported correction policy: {policy}")
    record = comparison_record if isinstance(comparison_record, dict) else {}
    dynamic = _text(record.get("dynamic_reference_viability"))
    baseline = _text(record.get("baseline_reference_viability"))
    flags = _base_flags(record, dynamic, baseline)
    if policy_norm == "conservative":
        return _conservative(record, flags, dynamic, baseline)
    if policy_norm == "liberal":
        return _liberal(record, flags, dynamic, baseline)
    return _balanced(record, flags, dynamic, baseline)


def apply_correction_policy_proposals(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one record with refreshed per-policy proposal fields."""
    out = dict(record if isinstance(record, dict) else {})
    for policy in SUPPORTED_CORRECTION_POLICIES:
        proposal = propose_correction_policy(comparison_record=out, policy=policy)
        suffix = str(policy)
        out[f"proposed_correction_mode_{suffix}"] = proposal["proposed_correction_mode"]
        out[f"proposal_confidence_{suffix}"] = proposal["proposal_confidence"]
        out[f"review_required_{suffix}"] = proposal["review_required"]
        out[f"review_queue_candidate_{suffix}"] = proposal["review_queue_candidate"]
        out[f"review_priority_{suffix}"] = proposal["review_priority"]
        out[f"warning_level_{suffix}"] = proposal["warning_level"]
        out[f"proposal_reason_{suffix}"] = proposal["proposal_reason"]
        out[f"proposal_flags_{suffix}"] = proposal["proposal_flags"]
    return out


def _count_values(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        val = rec.get(key)
        if isinstance(val, bool):
            text = str(bool(val)).lower()
        else:
            text = str(val or "").strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return {k: int(v) for k, v in sorted(counts.items())}


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def summarize_correction_policy_proposals(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize refreshed correction-policy proposal fields for qc_summary.json."""
    records = list(records or [])
    out: dict[str, Any] = {}
    for policy in SUPPORTED_CORRECTION_POLICIES:
        flag_counts: dict[str, int] = {}
        flag_key = f"proposal_flags_{policy}"
        for rec in records:
            for flag in normalize_policy_flags(rec.get(flag_key, [])):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        n_records = len(records)
        n_required = sum(
            1 for rec in records if _boolish(rec.get(f"review_required_{policy}", False))
        )
        n_queue = sum(
            1
            for rec in records
            if _boolish(rec.get(f"review_queue_candidate_{policy}", False))
        )
        out[policy] = {
            "roi_chunk_proposal_count": int(n_records),
            "mandatory_review_fraction": (
                float(n_required) / float(n_records) if n_records else 0.0
            ),
            "review_queue_candidate_fraction": (
                float(n_queue) / float(n_records) if n_records else 0.0
            ),
            "proposed_correction_mode_counts": _count_values(
                records, f"proposed_correction_mode_{policy}"
            ),
            "proposal_confidence_counts": _count_values(
                records, f"proposal_confidence_{policy}"
            ),
            "review_required_counts": _count_values(records, f"review_required_{policy}"),
            "review_queue_candidate_counts": _count_values(
                records, f"review_queue_candidate_{policy}"
            ),
            "review_priority_counts": _count_values(records, f"review_priority_{policy}"),
            "warning_level_counts": _count_values(records, f"warning_level_{policy}"),
            "proposal_flag_counts": {k: int(v) for k, v in sorted(flag_counts.items())},
        }
    return out
