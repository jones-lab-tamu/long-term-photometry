"""Compact diagnostics for UV-to-signal fit slopes."""

from __future__ import annotations

from typing import Any

import numpy as np


def _negative_span_stats(mask: np.ndarray) -> tuple[int, int]:
    spans = 0
    longest = 0
    start = None
    for i, flag in enumerate(np.asarray(mask, dtype=bool)):
        if bool(flag) and start is None:
            start = i
        elif not bool(flag) and start is not None:
            spans += 1
            longest = max(longest, i - start)
            start = None
    if start is not None:
        spans += 1
        longest = max(longest, int(mask.size) - start)
    return int(spans), int(longest)


def slope_warning_level(negative_fraction: float) -> str:
    try:
        frac = float(negative_fraction)
    except Exception:
        frac = 0.0
    if not np.isfinite(frac) or frac <= 0.0:
        return "none"
    if frac <= 0.01:
        return "low"
    if frac <= 0.05:
        return "moderate"
    if frac <= 0.20:
        return "high"
    return "critical"


def summarize_slope(values: Any, sample_rate_hz: float | None = None) -> dict[str, Any]:
    """Summarize scalar or per-sample UV-to-signal fit slope values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    n_total = int(arr.size)
    finite = np.isfinite(arr)
    n_finite = int(np.sum(finite))
    n_nonfinite = int(n_total - n_finite)

    if n_total == 0:
        negative_mask = np.zeros(0, dtype=bool)
        n_negative = 0
        neg_fraction = 0.0
        nonfinite_fraction = 0.0
        slope_min = slope_max = slope_median = slope_mean = float("nan")
    else:
        finite_values = arr[finite]
        negative_mask = finite & (arr < 0.0)
        n_negative = int(np.sum(negative_mask))
        neg_fraction = float(n_negative / n_finite) if n_finite > 0 else 0.0
        nonfinite_fraction = float(n_nonfinite / n_total)
        if n_finite > 0:
            slope_min = float(np.min(finite_values))
            slope_max = float(np.max(finite_values))
            slope_median = float(np.median(finite_values))
            slope_mean = float(np.mean(finite_values))
        else:
            slope_min = slope_max = slope_median = slope_mean = float("nan")

    n_spans, longest_span = _negative_span_stats(negative_mask)
    span_sec = None
    try:
        fs = float(sample_rate_hz) if sample_rate_hz is not None else float("nan")
    except Exception:
        fs = float("nan")
    if np.isfinite(fs) and fs > 0.0:
        span_sec = float(longest_span / fs)

    return {
        "slope_min": slope_min,
        "slope_max": slope_max,
        "slope_median": slope_median,
        "slope_mean": slope_mean,
        "slope_negative_fraction": neg_fraction,
        "slope_nonfinite_fraction": nonfinite_fraction,
        "n_slope_samples": n_total,
        "n_negative_slope_samples": n_negative,
        "n_nonfinite_slope_samples": n_nonfinite,
        "n_negative_slope_spans": n_spans,
        "longest_negative_slope_span_samples": longest_span,
        "longest_negative_slope_span_sec": span_sec,
        "warning_level": slope_warning_level(neg_fraction),
    }


def apply_slope_constraint(
    slope_values: Any,
    *,
    constraint_mode: str,
    min_slope: float,
    sample_rate_hz: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply optional final-stage slope constraints and return compact provenance."""
    mode = str(constraint_mode or "unconstrained").strip().lower()
    if mode not in {"unconstrained", "nonnegative"}:
        raise ValueError(
            "dynamic_fit_slope_constraint must be one of {'unconstrained', 'nonnegative'}"
        )
    try:
        min_allowed = float(min_slope)
    except Exception as exc:
        raise ValueError("dynamic_fit_min_slope must be a finite float") from exc
    if not np.isfinite(min_allowed):
        raise ValueError("dynamic_fit_min_slope must be a finite float")
    if mode == "nonnegative" and min_allowed < 0.0:
        raise ValueError(
            "dynamic_fit_min_slope must be >= 0 when "
            "dynamic_fit_slope_constraint is 'nonnegative'"
        )

    original = np.asarray(slope_values, dtype=float)
    original_shape = original.shape
    arr = original.reshape(-1)
    unconstrained_summary = summarize_slope(arr, sample_rate_hz=sample_rate_hz)

    finite = np.isfinite(arr)
    clamped_mask = finite & (arr < min_allowed) if mode == "nonnegative" else np.zeros(arr.shape, dtype=bool)
    if mode == "nonnegative":
        constrained_arr = np.array(arr, dtype=float, copy=True)
        constrained_arr[clamped_mask] = min_allowed
    else:
        constrained_arr = np.array(arr, dtype=float, copy=True)

    constrained_summary = summarize_slope(constrained_arr, sample_rate_hz=sample_rate_hz)
    n_samples = int(arr.size)
    n_clamped = int(np.sum(clamped_mask))
    n_spans, longest_span = _negative_span_stats(clamped_mask)
    span_sec = None
    try:
        fs = float(sample_rate_hz) if sample_rate_hz is not None else float("nan")
    except Exception:
        fs = float("nan")
    if np.isfinite(fs) and fs > 0.0:
        span_sec = float(longest_span / fs)

    summary = {
        "slope_constraint_mode": mode,
        "slope_min_allowed": min_allowed,
        "slope_constraint_applied": bool(n_clamped > 0),
        "n_slope_samples": n_samples,
        "n_clamped_slope_samples": n_clamped,
        "slope_clamped_fraction": float(n_clamped / n_samples) if n_samples > 0 else 0.0,
        "n_clamped_slope_spans": int(n_spans),
        "longest_clamped_slope_span_samples": int(longest_span),
        "longest_clamped_slope_span_sec": span_sec,
        "unconstrained_slope_summary": unconstrained_summary,
        "constrained_slope_summary": constrained_summary,
    }

    constrained = constrained_arr.reshape(original_shape)
    if np.isscalar(slope_values) or original_shape == ():
        return float(constrained.reshape(-1)[0]) if constrained.size else float("nan"), summary
    return constrained, summary


SLOPE_SUMMARY_NUMERIC_FIELDS = (
    "slope_min",
    "slope_max",
    "slope_median",
    "slope_mean",
    "slope_negative_fraction",
    "slope_nonfinite_fraction",
    "n_slope_samples",
    "n_negative_slope_samples",
    "n_nonfinite_slope_samples",
    "n_negative_slope_spans",
    "longest_negative_slope_span_samples",
    "longest_negative_slope_span_sec",
)
