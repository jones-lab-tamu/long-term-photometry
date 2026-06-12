"""Diagnostic-only correction policy proposal helpers."""

from __future__ import annotations

from typing import Any


SUPPORTED_CORRECTION_POLICIES = ("conservative", "balanced", "liberal")

MODE_DYNAMIC = "dynamic"
MODE_BASELINE = "baseline_reference_candidate"
MODE_NO_ISOSBESTIC = "no_isosbestic_candidate"
MODE_REVIEW = "review_required"

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
FLAG_BASELINE_ACCEPTED = "BASELINE_REFERENCE_CANDIDATE_ACCEPTED"
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
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(";") if part.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(part).strip() for part in value if str(part).strip()}
    text = str(value).strip()
    return {text} if text else set()


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
            mode=MODE_DYNAMIC,
            confidence=CONFIDENCE_HIGH,
            review_required=False,
            review_queue_candidate=False,
            review_priority=PRIORITY_NONE,
            warning_level=WARNING_NONE,
            reason="dynamic_reference_viable",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "hard_inspect" and baseline == "viable" and baseline_clean_positive:
        return _proposal(
            policy="balanced",
            mode=MODE_BASELINE,
            confidence=CONFIDENCE_MEDIUM,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_MEDIUM,
            warning_level=WARNING_CAUTION,
            reason="dynamic_hard_inspect_baseline_viable_positive_relationship",
            flags=[*flags, FLAG_BASELINE_ACCEPTED],
        )
    if dynamic == "hard_inspect" and baseline == "viable":
        return _proposal(
            policy="balanced",
            mode=MODE_REVIEW,
            confidence=CONFIDENCE_LOW,
            review_required=True,
            review_queue_candidate=True,
            review_priority=PRIORITY_HIGH,
            warning_level=WARNING_SEVERE,
            reason="dynamic_hard_inspect_baseline_viable_without_clean_positive_relationship",
            flags=[*flags, FLAG_REVIEW_BY_POLICY, FLAG_NO_CLEAN_REFERENCE],
        )
    if dynamic == "contextual" and baseline == "viable":
        return _proposal(
            policy="balanced",
            mode=MODE_DYNAMIC,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_MEDIUM,
            warning_level=WARNING_CAUTION,
            reason="contextual_dynamic_with_alternative_baseline_candidate",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "contextual" and baseline == "contextual":
        return _proposal(
            policy="balanced",
            mode=MODE_DYNAMIC,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_LOW,
            warning_level=WARNING_CONTEXTUAL,
            reason="contextual_candidate_evidence_logged_for_audit",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
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
            reason="dynamic_hard_inspect_baseline_contextual",
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
            reason="both_candidates_hard_inspect_or_unavailable",
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
        reason="unknown_dynamic_or_baseline_viability",
        flags=[*flags, FLAG_REVIEW_BY_POLICY, FLAG_NO_CLEAN_REFERENCE],
    )


def _conservative(record: dict[str, Any], flags: list[str], dynamic: str, baseline: str) -> dict[str, Any]:
    baseline_clean_positive = _baseline_relationship_is_clean_positive(record)
    if dynamic == "viable":
        return _proposal(
            policy="conservative",
            mode=MODE_DYNAMIC,
            confidence=CONFIDENCE_HIGH,
            review_required=False,
            review_queue_candidate=False,
            review_priority=PRIORITY_NONE,
            warning_level=WARNING_NONE,
            reason="dynamic_reference_viable_under_conservative_policy",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "hard_inspect" and baseline == "viable" and baseline_clean_positive:
        return _proposal(
            policy="conservative",
            mode=MODE_BASELINE,
            confidence=CONFIDENCE_MEDIUM,
            review_required=True,
            review_queue_candidate=True,
            review_priority=PRIORITY_HIGH,
            warning_level=WARNING_CAUTION,
            reason="baseline_rescue_candidate_requires_high_priority_review",
            flags=[*flags, FLAG_BASELINE_ACCEPTED, FLAG_REVIEW_BY_POLICY],
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
            mode=MODE_DYNAMIC,
            confidence=CONFIDENCE_HIGH,
            review_required=False,
            review_queue_candidate=False,
            review_priority=PRIORITY_NONE,
            warning_level=WARNING_NONE,
            reason="dynamic_reference_viable",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "contextual" and baseline in {"viable", "contextual"}:
        return _proposal(
            policy="liberal",
            mode=MODE_DYNAMIC,
            confidence=CONFIDENCE_LOW,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_LOW,
            warning_level=WARNING_CONTEXTUAL,
            reason="liberal_policy_allows_contextual_dynamic_for_screening",
            flags=[*flags, FLAG_DYNAMIC_ACCEPTED],
        )
    if dynamic == "hard_inspect" and baseline == "viable" and baseline_clean_positive:
        return _proposal(
            policy="liberal",
            mode=MODE_BASELINE,
            confidence=CONFIDENCE_MEDIUM,
            review_required=False,
            review_queue_candidate=True,
            review_priority=PRIORITY_LOW,
            warning_level=WARNING_CAUTION,
            reason="liberal_policy_allows_positive_baseline_rescue_for_screening",
            flags=[*flags, FLAG_BASELINE_ACCEPTED],
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
