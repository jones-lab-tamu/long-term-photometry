"""Diagnostic comparison of dynamic and baseline reference candidates."""

from __future__ import annotations

from typing import Any


DYNAMIC_NEGATIVE_OR_MIXED_COUPLING_FLAG = "NEGATIVE_OR_MIXED_REFERENCE_COUPLING"
DYNAMIC_RESPONSE_SCALE_RICH_FLAG = "FITTED_REFERENCE_RESPONSE_SCALE_RICH"
DYNAMIC_LOW_RANGE_FLAG = "FITTED_REFERENCE_LOW_RANGE"
DYNAMIC_FLAT_FLAG = "FITTED_REFERENCE_FLAT_OR_UNINFORMATIVE"

COMPARISON_FLAG_DYNAMIC_NEGATIVE = "DYNAMIC_NEGATIVE_OR_MIXED_COUPLING"
COMPARISON_FLAG_DYNAMIC_RESPONSE = "DYNAMIC_RESPONSE_SCALE_RICH"
COMPARISON_FLAG_DYNAMIC_LOW_OR_FLAT = "DYNAMIC_LOW_OR_FLAT_REFERENCE"
COMPARISON_FLAG_BASELINE_UNAVAILABLE = "BASELINE_CANDIDATE_UNAVAILABLE"
COMPARISON_FLAG_BASELINE_LOW_OR_FLAT = "BASELINE_CANDIDATE_LOW_OR_FLAT"
COMPARISON_FLAG_BASELINE_RESPONSE = "BASELINE_CANDIDATE_RESPONSE_SCALE_RICH"
COMPARISON_FLAG_BASELINE_WINDOW_LARGE = "BASELINE_WINDOW_LARGE_FRACTION_OF_CHUNK"
COMPARISON_FLAG_BASELINE_WINDOW_ADJUSTED = "BASELINE_WINDOW_ADJUSTED"


def _as_flag_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(";") if part.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(part).strip() for part in value if str(part).strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _dynamic_viability(dynamic_qc: dict[str, Any]) -> str:
    severity = str(dynamic_qc.get("dynamic_fit_qc_severity", "") or "").strip().lower()
    if severity == "ok":
        return "viable"
    if severity == "context":
        return "contextual"
    if severity == "inspect":
        return "hard_inspect"
    return "unknown"


def _baseline_viability(baseline_record: dict[str, Any]) -> str:
    if "baseline_ref_candidate_available" not in baseline_record:
        return "unknown"
    if not _as_bool(baseline_record.get("baseline_ref_candidate_available")):
        return "unavailable"
    if _as_bool(baseline_record.get("baseline_ref_low_range")) or _as_bool(
        baseline_record.get("baseline_ref_flat_or_uninformative")
    ):
        return "hard_inspect"
    if _as_bool(baseline_record.get("baseline_ref_response_scale_rich")):
        return "contextual"
    return "viable"


def _comparison_class(dynamic_viability: str, baseline_viability: str) -> str:
    if baseline_viability == "unavailable":
        return "baseline_unavailable"
    if dynamic_viability == "unknown" or baseline_viability == "unknown":
        return "unknown"

    dynamic_part = {
        "viable": "dynamic_viable",
        "contextual": "dynamic_context",
        "hard_inspect": "dynamic_hard_inspect",
    }.get(dynamic_viability)
    baseline_part = {
        "viable": "baseline_viable",
        "contextual": "baseline_contextual",
        "hard_inspect": "baseline_hard_inspect",
    }.get(baseline_viability)
    if dynamic_part is None or baseline_part is None:
        return "unknown"
    return f"{dynamic_part}_{baseline_part}"


def _review_level(dynamic_viability: str, baseline_viability: str, flags: list[str]) -> str:
    dynamic_major = dynamic_viability == "hard_inspect"
    baseline_major = baseline_viability in {"hard_inspect", "unavailable"}
    if dynamic_major and baseline_major:
        return "high"
    if (
        COMPARISON_FLAG_DYNAMIC_LOW_OR_FLAT in flags
        and COMPARISON_FLAG_BASELINE_LOW_OR_FLAT in flags
    ):
        return "high"
    if dynamic_viability == "viable" and baseline_viability in {
        "viable",
        "contextual",
        "hard_inspect",
        "unavailable",
    }:
        return "low"
    if dynamic_viability == "contextual" and baseline_viability in {"viable", "contextual"}:
        return "medium"
    if dynamic_viability in {"viable", "contextual"} and baseline_viability in {
        "viable",
        "contextual",
    }:
        return "medium"
    return "medium"


def classify_reference_candidates(
    *,
    dynamic_qc: dict[str, Any],
    baseline_record: dict[str, Any],
) -> dict[str, Any]:
    """Classify dynamic-vs-baseline reference candidate QC for triage only."""
    dynamic_qc = dynamic_qc if isinstance(dynamic_qc, dict) else {}
    baseline_record = baseline_record if isinstance(baseline_record, dict) else {}

    dynamic_flags = _as_flag_set(dynamic_qc.get("dynamic_fit_qc_flags"))
    dynamic_hard_flags = _as_flag_set(dynamic_qc.get("dynamic_fit_qc_hard_flags"))
    dynamic_soft_flags = _as_flag_set(dynamic_qc.get("dynamic_fit_qc_soft_flags"))
    all_dynamic_flags = dynamic_flags | dynamic_hard_flags | dynamic_soft_flags

    dynamic_has_negative = (
        DYNAMIC_NEGATIVE_OR_MIXED_COUPLING_FLAG in all_dynamic_flags
        or _as_bool(dynamic_qc.get("dynamic_fit_negative_or_mixed_coupling"))
    )
    dynamic_has_response = (
        DYNAMIC_RESPONSE_SCALE_RICH_FLAG in all_dynamic_flags
        or _as_bool(dynamic_qc.get("dynamic_fit_response_scale_rich"))
    )
    dynamic_has_low_or_flat = (
        DYNAMIC_LOW_RANGE_FLAG in all_dynamic_flags
        or DYNAMIC_FLAT_FLAG in all_dynamic_flags
        or _as_bool(dynamic_qc.get("dynamic_fit_reference_low_range"))
        or _as_bool(dynamic_qc.get("dynamic_fit_reference_flat_or_uninformative"))
    )
    baseline_available = _as_bool(baseline_record.get("baseline_ref_candidate_available"))
    baseline_low_or_flat = _as_bool(baseline_record.get("baseline_ref_low_range")) or _as_bool(
        baseline_record.get("baseline_ref_flat_or_uninformative")
    )
    baseline_response = _as_bool(baseline_record.get("baseline_ref_response_scale_rich"))
    baseline_window_adjusted = _as_bool(
        baseline_record.get("baseline_ref_smoothing_window_adjusted")
    )
    baseline_window_large = False
    try:
        fraction = float(baseline_record.get("baseline_ref_smoothing_window_fraction_of_chunk"))
        threshold = float(baseline_record.get("baseline_ref_large_window_fraction_warning"))
        baseline_window_large = fraction >= threshold
    except Exception:
        warning = str(baseline_record.get("baseline_ref_smoothing_window_warning") or "")
        baseline_window_large = "large_fraction" in warning

    flags: list[str] = []
    if dynamic_has_negative:
        flags.append(COMPARISON_FLAG_DYNAMIC_NEGATIVE)
    if dynamic_has_response:
        flags.append(COMPARISON_FLAG_DYNAMIC_RESPONSE)
    if dynamic_has_low_or_flat:
        flags.append(COMPARISON_FLAG_DYNAMIC_LOW_OR_FLAT)
    if not baseline_available:
        flags.append(COMPARISON_FLAG_BASELINE_UNAVAILABLE)
    if baseline_low_or_flat:
        flags.append(COMPARISON_FLAG_BASELINE_LOW_OR_FLAT)
    if baseline_response:
        flags.append(COMPARISON_FLAG_BASELINE_RESPONSE)
    if baseline_window_large:
        flags.append(COMPARISON_FLAG_BASELINE_WINDOW_LARGE)
    if baseline_window_adjusted:
        flags.append(COMPARISON_FLAG_BASELINE_WINDOW_ADJUSTED)

    dynamic_viability = _dynamic_viability(dynamic_qc)
    baseline_viability = _baseline_viability(baseline_record)
    comparison_class = _comparison_class(dynamic_viability, baseline_viability)
    review_level = _review_level(dynamic_viability, baseline_viability, flags)
    notes = [
        f"dynamic={dynamic_viability}",
        f"baseline={baseline_viability}",
        "diagnostic_only",
    ]

    return {
        "reference_comparison_class": comparison_class,
        "dynamic_reference_viability": dynamic_viability,
        "baseline_reference_viability": baseline_viability,
        "reference_comparison_review_level": review_level,
        "reference_comparison_notes": ";".join(notes),
        "reference_comparison_flags": flags,
        "dynamic_has_negative_or_mixed_coupling": bool(dynamic_has_negative),
        "dynamic_has_response_scale_rich": bool(dynamic_has_response),
        "dynamic_has_low_or_flat_reference": bool(dynamic_has_low_or_flat),
        "baseline_is_available": bool(baseline_available),
        "baseline_has_low_or_flat_reference": bool(baseline_low_or_flat),
        "baseline_has_response_scale_rich": bool(baseline_response),
        "baseline_window_large_fraction_of_chunk": bool(baseline_window_large),
        "baseline_window_adjusted": bool(baseline_window_adjusted),
    }
