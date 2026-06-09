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
