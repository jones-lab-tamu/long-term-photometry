"""Diagnostic-only correction policy proposal helpers."""

from __future__ import annotations

import ast
from typing import Any


SUPPORTED_CORRECTION_POLICIES = ("conservative", "balanced", "liberal")

MODE_DYNAMIC_ISOSBESTIC = "dynamic_isosbestic"
MODE_BASELINE_LEGACY = "baseline_reference_candidate"
MODE_SIGNAL_ONLY_F0 = "signal_only_f0_candidate"
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
FLAG_SIGNAL_ONLY_F0_AVAILABLE = "SIGNAL_ONLY_F0_CANDIDATE_AVAILABLE"
FLAG_SIGNAL_ONLY_F0_VIABLE = "SIGNAL_ONLY_F0_CANDIDATE_VIABLE"
FLAG_SIGNAL_ONLY_F0_CONTEXTUAL = "SIGNAL_ONLY_F0_CANDIDATE_CONTEXTUAL"
FLAG_SIGNAL_ONLY_F0_CONFIDENCE_HIGH = "SIGNAL_ONLY_F0_CANDIDATE_CONFIDENCE_HIGH"
FLAG_SIGNAL_ONLY_F0_CONFIDENCE_MEDIUM = "SIGNAL_ONLY_F0_CANDIDATE_CONFIDENCE_MEDIUM"
FLAG_SIGNAL_ONLY_F0_CONFIDENCE_LOW = "SIGNAL_ONLY_F0_CANDIDATE_CONFIDENCE_LOW"
FLAG_SIGNAL_ONLY_F0_STATE_AWARE_ANCHORED = "SIGNAL_ONLY_F0_STATE_AWARE_ANCHORED"
FLAG_SIGNAL_ONLY_F0_SUFFICIENT_ANCHORS = "SIGNAL_ONLY_F0_SUFFICIENT_ANCHORS"
FLAG_SIGNAL_ONLY_F0_LOW_CONFIDENCE = "SIGNAL_ONLY_F0_LOW_CONFIDENCE"
FLAG_SIGNAL_ONLY_F0_HARD = "SIGNAL_ONLY_F0_HARD_INSPECT"
FLAG_SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS = "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS"
FLAG_SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT = "SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT"
FLAG_SIGNAL_ONLY_F0_POLICY_CANDIDATE = "SIGNAL_ONLY_F0_POLICY_CANDIDATE"
FLAG_SIGNAL_ONLY_F0_POLICY_REJECTED = "SIGNAL_ONLY_F0_POLICY_REJECTED"
FLAG_SIGNAL_ONLY_F0_POLICY_REQUIRES_REVIEW = "SIGNAL_ONLY_F0_POLICY_REQUIRES_REVIEW"
FLAG_DYNAMIC_CONTEXT_SUPPORTS_SIGNAL_ONLY_F0 = (
    "DYNAMIC_CONTEXT_SUPPORTS_SIGNAL_ONLY_F0_FALLBACK"
)
FLAG_DYNAMIC_CONTEXT_DOES_NOT_SUPPORT_SIGNAL_ONLY_F0 = (
    "DYNAMIC_CONTEXT_DOES_NOT_SUPPORT_SIGNAL_ONLY_F0_FALLBACK"
)
FLAG_BOTH_CONTEXTUAL = "BOTH_CANDIDATES_CONTEXTUAL"
FLAG_BOTH_HARD_OR_UNAVAILABLE = "BOTH_CANDIDATES_HARD_OR_UNAVAILABLE"
FLAG_REVIEW_BY_POLICY = "REVIEW_REQUIRED_BY_POLICY"
FLAG_NO_CLEAN_REFERENCE = "NO_CLEAN_REFERENCE_CANDIDATE"

SIGNAL_ONLY_F0_REJECT_FLAGS = {
    "SIGNAL_ONLY_F0_HARD_INSPECT",
    "SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT",
    "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS",
    "SIGNAL_ONLY_F0_ABOVE_SIGNAL_EXCESSIVE",
}

SIGNAL_ONLY_F0_CAP_FLAGS = {
    "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION",
    "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_LOW_ANCHOR_SUPPORT",
    "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS",
    "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_LARGE_GAP",
    "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP",
}

DYNAMIC_CONTEXT_SIGNAL_ONLY_F0_TRIGGER_FLAGS = {
    "DYNAMIC_NEGATIVE_OR_MIXED_COUPLING",
    "NEGATIVE_OR_MIXED_REFERENCE_COUPLING",
    "FITTED_REFERENCE_LOW_RANGE",
    "FITTED_REFERENCE_FLAT_OR_UNINFORMATIVE",
    "DYNAMIC_LOW_OR_FLAT_REFERENCE",
    "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP",
    "BASELINE_MIXED_OR_UNCLEAR_REFERENCE_RELATIONSHIP",
    "INVERTED_REFERENCE_RELATIONSHIP",
}


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


def _has_high_state_context(record: dict[str, Any]) -> bool:
    f0_flags = _as_flag_set(record.get("signal_only_f0_flags"))
    signal_flags = _as_flag_set(record.get("signal_state_flags"))
    return bool(
        {
            "SIGNAL_ONLY_F0_HIGH_STATE_PRESENT",
            "SIGNAL_ONLY_F0_PARTIAL_HIGH_STATE_PRESENT",
            "SIGNAL_ONLY_F0_EDGE_HIGH_STATE_PRESENT",
        }
        & f0_flags
        or {
            "SIGNAL_HIGH_STATE_CANDIDATE",
            "SIGNAL_PARTIAL_HIGH_STATE_CANDIDATE",
            "SIGNAL_EDGE_HIGH_STATE_CANDIDATE",
            "SIGNAL_STARTS_HIGH",
            "SIGNAL_ENDS_HIGH",
        }
        & signal_flags
    )


def _dynamic_context_supports_signal_only_f0_fallback(record: dict[str, Any]) -> bool:
    flags: set[str] = set()
    for key in (
        "dynamic_fit_qc_flags",
        "dynamic_fit_qc_hard_flags",
        "dynamic_fit_qc_soft_flags",
        "reference_comparison_flags",
    ):
        flags.update(_as_flag_set(record.get(key)))
    if flags & DYNAMIC_CONTEXT_SIGNAL_ONLY_F0_TRIGGER_FLAGS:
        return True
    relationship = _baseline_relationship_class(record)
    if relationship in {
        "negative_reference_relationship",
        "mixed_or_unclear_reference_relationship",
    }:
        return True
    for key in (
        "dynamic_fit_negative_or_mixed_coupling",
        "dynamic_fit_reference_low_range",
        "dynamic_fit_reference_flat_or_uninformative",
        "dynamic_has_negative_or_mixed_coupling",
        "dynamic_has_low_or_flat_reference",
    ):
        if _boolish(record.get(key, False)):
            return True
    return False


def _signal_only_f0_evidence(record: dict[str, Any], *, allow_low: bool = False) -> dict[str, Any]:
    f0_flags = _as_flag_set(record.get("signal_only_f0_flags"))
    available = _boolish(record.get("signal_only_f0_candidate_available", False))
    viability = _text(record.get("signal_only_f0_candidate_viability"))
    confidence = _text(record.get("signal_only_f0_candidate_confidence"), CONFIDENCE_NONE)
    anchor_status = _text(record.get("signal_only_f0_anchor_status"))
    state_aware = _boolish(record.get("signal_only_f0_state_aware_used", False))
    high_state_context = _has_high_state_context(record)

    flags: list[str] = []
    if available:
        flags.append(FLAG_SIGNAL_ONLY_F0_AVAILABLE)
    if viability == "viable":
        flags.append(FLAG_SIGNAL_ONLY_F0_VIABLE)
    elif viability == "contextual":
        flags.append(FLAG_SIGNAL_ONLY_F0_CONTEXTUAL)
    elif viability == "hard_inspect":
        flags.append(FLAG_SIGNAL_ONLY_F0_HARD)
    if confidence == CONFIDENCE_HIGH:
        flags.append(FLAG_SIGNAL_ONLY_F0_CONFIDENCE_HIGH)
    elif confidence == CONFIDENCE_MEDIUM:
        flags.append(FLAG_SIGNAL_ONLY_F0_CONFIDENCE_MEDIUM)
    elif confidence == CONFIDENCE_LOW:
        flags.append(FLAG_SIGNAL_ONLY_F0_CONFIDENCE_LOW)
        flags.append(FLAG_SIGNAL_ONLY_F0_LOW_CONFIDENCE)
    if state_aware:
        flags.append(FLAG_SIGNAL_ONLY_F0_STATE_AWARE_ANCHORED)
    if anchor_status == "sufficient_anchors":
        flags.append(FLAG_SIGNAL_ONLY_F0_SUFFICIENT_ANCHORS)
    if "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS" in f0_flags:
        flags.append(FLAG_SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS)
    if "SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT" in f0_flags:
        flags.append(FLAG_SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT)

    reject_reasons = []
    if not available:
        reject_reasons.append("unavailable")
    if viability not in {"viable", "contextual"}:
        reject_reasons.append("unusable_viability")
    if confidence not in {"high", "medium"} and not (allow_low and confidence == "low"):
        reject_reasons.append("insufficient_confidence")
    if anchor_status != "sufficient_anchors":
        reject_reasons.append("insufficient_anchors")
    if not state_aware and str(record.get("signal_only_f0_anchor_status", "")) != "ordinary_dynamic_fallback":
        reject_reasons.append("state_aware_not_documented")
    if f0_flags & SIGNAL_ONLY_F0_REJECT_FLAGS:
        reject_reasons.append("hard_signal_only_f0_flags")

    if reject_reasons:
        flags.append(FLAG_SIGNAL_ONLY_F0_POLICY_REJECTED)
    else:
        flags.append(FLAG_SIGNAL_ONLY_F0_POLICY_CANDIDATE)

    confidence_caps = bool(f0_flags & SIGNAL_ONLY_F0_CAP_FLAGS)
    return {
        "candidate": not reject_reasons,
        "flags": flags,
        "reject_reasons": reject_reasons,
        "available": available,
        "viability": viability,
        "confidence": confidence,
        "anchor_status": anchor_status,
        "state_aware": state_aware,
        "high_state_context": high_state_context,
        "confidence_caps": confidence_caps,
        "f0_flags": f0_flags,
    }


def _signal_only_f0_proposal(
    *,
    policy: str,
    record: dict[str, Any],
    flags: list[str],
    dynamic: str,
    allow_low: bool = False,
) -> dict[str, Any] | None:
    evidence = _signal_only_f0_evidence(record, allow_low=allow_low)
    if not evidence["candidate"]:
        return None

    high_state_context = bool(evidence["high_state_context"])
    f0_viability = str(evidence["viability"])
    f0_confidence = str(evidence["confidence"])
    confidence_caps = bool(evidence["confidence_caps"])

    if (
        not high_state_context
        and f0_viability == "viable"
        and f0_confidence == CONFIDENCE_HIGH
        and not confidence_caps
    ):
        confidence = CONFIDENCE_HIGH
        warning = WARNING_NONE
    elif f0_confidence in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}:
        confidence = CONFIDENCE_MEDIUM
        warning = WARNING_CAUTION if high_state_context or dynamic == "hard_inspect" else WARNING_CONTEXTUAL
    else:
        confidence = CONFIDENCE_LOW
        warning = WARNING_CAUTION
    if dynamic == "hard_inspect":
        warning = WARNING_CAUTION
    elif dynamic == "contextual" and warning == WARNING_NONE:
        warning = WARNING_CONTEXTUAL

    review_required = False
    review_queue = True
    priority = PRIORITY_LOW
    proposal_flags = [*flags, *evidence["flags"]]
    if high_state_context:
        proposal_flags.append(FLAG_SIGNAL_ONLY_F0_POLICY_REQUIRES_REVIEW)
        priority = PRIORITY_MEDIUM
    if dynamic == "hard_inspect":
        priority = PRIORITY_MEDIUM if priority == PRIORITY_LOW else priority
    if policy == "conservative":
        review_required = bool(high_state_context or confidence != CONFIDENCE_HIGH)
        priority = PRIORITY_HIGH if review_required else PRIORITY_MEDIUM
    elif policy == "balanced":
        review_required = bool(high_state_context or confidence == CONFIDENCE_LOW)
        priority = PRIORITY_MEDIUM if review_required else priority
    elif policy == "liberal":
        review_required = False

    if review_required:
        proposal_flags.append(FLAG_REVIEW_BY_POLICY)
    reason = (
        "signal_only_f0_candidate_contextual_fallback"
        if high_state_context
        else "signal_only_f0_candidate_supported_fallback"
    )
    return _proposal(
        policy=policy,
        mode=MODE_SIGNAL_ONLY_F0,
        confidence=confidence,
        review_required=review_required,
        review_queue_candidate=review_queue,
        review_priority=priority,
        warning_level=warning,
        reason=reason,
        flags=proposal_flags,
    )


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
    if dynamic == "hard_inspect":
        signal_only = _signal_only_f0_proposal(
            policy="balanced",
            record=record,
            flags=flags,
            dynamic=dynamic,
        )
        if signal_only is not None:
            return signal_only
    if dynamic == "contextual":
        if _dynamic_context_supports_signal_only_f0_fallback(record):
            signal_only = _signal_only_f0_proposal(
                policy="balanced",
                record=record,
                flags=[*flags, FLAG_DYNAMIC_CONTEXT_SUPPORTS_SIGNAL_ONLY_F0],
                dynamic=dynamic,
            )
            if signal_only is not None:
                return signal_only
        else:
            flags = [*flags, FLAG_DYNAMIC_CONTEXT_DOES_NOT_SUPPORT_SIGNAL_ONLY_F0]
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
    if dynamic in {"hard_inspect", "contextual"}:
        signal_only = _signal_only_f0_proposal(
            policy="conservative",
            record=record,
            flags=flags,
            dynamic=dynamic,
        )
        if signal_only is not None and (
            not _has_high_state_context(record)
            or _text(record.get("signal_only_f0_candidate_confidence")) == CONFIDENCE_HIGH
        ):
            return signal_only
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
    if dynamic in {"hard_inspect", "contextual"}:
        signal_only = _signal_only_f0_proposal(
            policy="liberal",
            record=record,
            flags=flags,
            dynamic=dynamic,
            allow_low=True,
        )
        if signal_only is not None:
            return signal_only
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
