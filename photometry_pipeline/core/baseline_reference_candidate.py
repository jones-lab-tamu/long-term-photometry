"""Diagnostic baseline-only reference candidate construction.

The candidate generated here is not an applied correction mode. It is a
configurable ultra-low-pass smoothing comparison trace used to evaluate whether
the reference channel supports slow baseline-scale correction without following
response-scale events.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dynamic_fit_qc import (
    BASELINE_SCALE_MAX_HZ,
    RESPONSE_SCALE_MAX_HZ,
    RESPONSE_SCALE_MIN_HZ,
    _range_p95_p05,
    _safe_corr,
    _safe_ratio,
    _variance_band_metrics,
)


DEFAULT_BASELINE_REFERENCE_SMOOTHING_WINDOW_SEC = 300.0
DEFAULT_BASELINE_REFERENCE_MIN_SMOOTHING_WINDOW_SEC = 60.0
DEFAULT_BASELINE_REFERENCE_MAX_WINDOW_FRACTION_OF_CHUNK = 0.75
DEFAULT_BASELINE_REFERENCE_LARGE_WINDOW_FRACTION_WARNING = 0.50


def _json_float_or_none(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    return val if np.isfinite(val) else None


def _window_sum(padded: np.ndarray, window: int) -> np.ndarray:
    """O(len(padded)) sliding-window sum, equivalent to
    ``np.convolve(padded, np.ones(window), mode="valid")`` up to floating-point
    summation order (a single running cumulative sum plus subtraction, instead
    of a fresh per-window direct sum)."""
    csum = np.concatenate(([0.0], np.cumsum(padded, dtype=float)))
    return csum[window:] - csum[:-window]


def _nan_aware_moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    window = int(max(1, window_samples))
    window = min(window, int(arr.size))
    if window % 2 == 0 and window > 1:
        window -= 1
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    pad = window // 2
    if pad > 0:
        filled = np.pad(filled, pad_width=pad, mode="reflect")
        finite_weights = np.pad(finite.astype(float), pad_width=pad, mode="reflect")
    else:
        finite_weights = finite.astype(float)
    sums = _window_sum(filled, window)
    counts = _window_sum(finite_weights, window)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = counts > 0.0
    out[valid] = sums[valid] / counts[valid]
    return out


def _robust_linear_fit(x: np.ndarray, y: np.ndarray, robust: bool) -> tuple[float, float, int, str]:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 3:
        return float("nan"), float("nan"), int(np.sum(finite)), "insufficient_finite_pairs"
    xx = np.asarray(x[finite], dtype=float)
    yy = np.asarray(y[finite], dtype=float)
    if float(np.nanstd(xx)) <= 1e-12:
        return float("nan"), float("nan"), int(xx.size), "reference_slow_variance_too_small"

    slope, intercept = np.polyfit(xx, yy, 1)
    if robust and xx.size >= 10:
        resid = yy - (float(slope) * xx + float(intercept))
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        if np.isfinite(mad) and mad > 1e-12:
            keep = np.abs(resid - med) <= 3.5 * 1.4826 * mad
            if int(np.sum(keep)) >= 3 and float(np.nanstd(xx[keep])) > 1e-12:
                slope, intercept = np.polyfit(xx[keep], yy[keep], 1)
                return float(slope), float(intercept), int(np.sum(keep)), "ok_robust_trimmed"
    return float(slope), float(intercept), int(xx.size), "ok"


def _fit_with_residual_exclusion(
    x: np.ndarray,
    y: np.ndarray,
    *,
    robust: bool,
    min_samples: int,
    mad_z: float = 3.5,
    return_diagnostics: bool = False,
) -> tuple[float, float, int, dict[str, Any]]:
    initial_slope, initial_intercept, initial_n, initial_status = _robust_linear_fit(
        x, y, robust
    )
    finite = np.isfinite(x) & np.isfinite(y)
    meta: dict[str, Any] = {
        "baseline_ref_residual_exclusion_fraction": 0.0,
        "baseline_ref_residual_exclusion_threshold": None,
        "baseline_ref_initial_fit_status": initial_status,
        "baseline_ref_final_fit_status": initial_status,
        "baseline_ref_fit_stage": "initial_only",
        "baseline_ref_initial_slope": (
            float(initial_slope) if np.isfinite(initial_slope) else None
        ),
        "baseline_ref_initial_intercept": (
            float(initial_intercept) if np.isfinite(initial_intercept) else None
        ),
        "baseline_ref_final_slope": (
            float(initial_slope) if np.isfinite(initial_slope) else None
        ),
        "baseline_ref_final_intercept": (
            float(initial_intercept) if np.isfinite(initial_intercept) else None
        ),
    }
    if return_diagnostics:
        meta["baseline_ref_fit_included_mask"] = finite.copy()
        meta["baseline_ref_fit_finite_mask"] = finite.copy()
    if not np.isfinite(initial_slope) or not np.isfinite(initial_intercept):
        return initial_slope, initial_intercept, initial_n, meta

    if int(np.sum(finite)) < int(min_samples):
        meta["baseline_ref_fit_stage"] = "residual_refit_insufficient_samples"
        return initial_slope, initial_intercept, initial_n, meta

    xx = np.asarray(x[finite], dtype=float)
    yy = np.asarray(y[finite], dtype=float)
    residual = yy - (float(initial_slope) * xx + float(initial_intercept))
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    if not np.isfinite(mad) or mad <= 1e-12:
        meta["baseline_ref_final_fit_status"] = "initial_fit_kept_residual_mad_too_small"
        return initial_slope, initial_intercept, initial_n, meta

    threshold = float(mad_z * 1.4826 * mad)
    keep = np.abs(residual - med) <= threshold
    full_keep = np.zeros_like(finite, dtype=bool)
    full_keep[np.where(finite)[0]] = keep
    if return_diagnostics:
        meta["baseline_ref_fit_included_mask"] = full_keep
    meta["baseline_ref_residual_exclusion_threshold"] = threshold
    meta["baseline_ref_residual_exclusion_fraction"] = float(1.0 - (np.sum(keep) / keep.size))
    if int(np.sum(keep)) < int(min_samples) or float(np.nanstd(xx[keep])) <= 1e-12:
        meta["baseline_ref_fit_stage"] = "residual_refit_insufficient_samples"
        meta["baseline_ref_final_fit_status"] = initial_status
        return initial_slope, initial_intercept, initial_n, meta

    final_slope, final_intercept, final_n, final_status = _robust_linear_fit(
        xx[keep], yy[keep], robust
    )
    if not np.isfinite(final_slope) or not np.isfinite(final_intercept):
        meta["baseline_ref_fit_stage"] = "residual_refit_insufficient_samples"
        meta["baseline_ref_final_fit_status"] = final_status
        return initial_slope, initial_intercept, initial_n, meta

    meta["baseline_ref_fit_stage"] = "residual_refit"
    meta["baseline_ref_final_fit_status"] = final_status
    meta["baseline_ref_final_slope"] = float(final_slope)
    meta["baseline_ref_final_intercept"] = float(final_intercept)
    return final_slope, final_intercept, final_n, meta


def classify_baseline_fit_relationship(
    *,
    slope: Any,
    corr: Any,
    slope_near_zero: float = 1e-9,
    corr_threshold: float = 0.25,
) -> str:
    """Classify the baseline-candidate fit relationship for diagnostic plotting."""
    try:
        slope_f = float(slope)
        corr_f = float(corr)
    except Exception:
        return "unknown"
    if not np.isfinite(slope_f) or not np.isfinite(corr_f):
        return "unknown"
    if abs(slope_f) <= float(slope_near_zero) or abs(corr_f) < float(corr_threshold):
        return "weak_reference_relationship"
    if slope_f < 0.0:
        return "negative_reference_relationship"
    if slope_f > 0.0:
        return "positive_reference_relationship"
    return "mixed_or_unclear_reference_relationship"


def _resolve_smoothing_window(
    requested: Any,
    *,
    n_samples: int,
    fs: float,
    default_sec: float,
    min_sec: float,
    max_fraction: float,
    large_fraction_warning: float,
) -> dict[str, Any]:
    chunk_duration_sec = (
        float(n_samples) / float(fs) if np.isfinite(fs) and fs > 0.0 else float("nan")
    )
    warnings: list[str] = []
    adjusted = False
    try:
        requested_sec = float(requested)
    except Exception:
        requested_sec = float("nan")
    if not np.isfinite(requested_sec) or requested_sec <= 0.0:
        requested_sec = float(default_sec)
        warnings.append("invalid_requested_smoothing_window_used_default")

    actual_sec = float(requested_sec)
    if np.isfinite(chunk_duration_sec) and chunk_duration_sec > 0.0:
        max_allowed = float(max_fraction) * chunk_duration_sec
        if max_allowed < float(min_sec):
            if actual_sec > max_allowed:
                warnings.append("requested_smoothing_window_too_large_for_chunk_adjusted")
            if actual_sec != max_allowed:
                actual_sec = max_allowed
                adjusted = True
            warnings.append("max_fraction_smaller_than_min_window_used_max_allowed")
        else:
            if actual_sec > max_allowed:
                actual_sec = max_allowed
                adjusted = True
                warnings.append("requested_smoothing_window_too_large_for_chunk_adjusted")
            if actual_sec < float(min_sec):
                actual_sec = float(min_sec)
                adjusted = True
                warnings.append("requested_smoothing_window_below_minimum_adjusted")
        fraction = float(actual_sec / chunk_duration_sec)
        if fraction >= float(large_fraction_warning):
            warnings.append("smoothing_window_large_fraction_of_chunk_interpret_cautiously")
    else:
        fraction = float("nan")
        warnings.append("invalid_chunk_duration")

    return {
        "baseline_ref_requested_smoothing_window_sec": float(requested_sec),
        "baseline_ref_actual_smoothing_window_sec": float(actual_sec),
        "baseline_ref_default_smoothing_window_sec": float(default_sec),
        "baseline_ref_min_smoothing_window_sec": float(min_sec),
        "baseline_ref_chunk_duration_sec": chunk_duration_sec,
        "baseline_ref_smoothing_window_fraction_of_chunk": fraction,
        "baseline_ref_max_window_fraction_of_chunk": float(max_fraction),
        "baseline_ref_large_window_fraction_warning": float(large_fraction_warning),
        "baseline_ref_smoothing_window_adjusted": bool(adjusted),
        "baseline_ref_smoothing_window_warning": ";".join(warnings) if warnings else None,
    }


def compute_baseline_reference_candidate(
    signal: np.ndarray,
    reference: np.ndarray,
    fs: float,
    *,
    lowpass_cutoff_hz: float | None = None,
    smoothing_window_sec: float | None = None,
    default_smoothing_window_sec: float = DEFAULT_BASELINE_REFERENCE_SMOOTHING_WINDOW_SEC,
    min_smoothing_window_sec: float = DEFAULT_BASELINE_REFERENCE_MIN_SMOOTHING_WINDOW_SEC,
    max_window_fraction_of_chunk: float = DEFAULT_BASELINE_REFERENCE_MAX_WINDOW_FRACTION_OF_CHUNK,
    large_window_fraction_warning: float = DEFAULT_BASELINE_REFERENCE_LARGE_WINDOW_FRACTION_WARNING,
    robust: bool = True,
    min_samples: int = 100,
    return_diagnostics: bool = False,
) -> dict[str, Any]:
    """Construct an ultra-low-pass reference candidate in signal units."""
    sig = np.asarray(signal, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cutoff = float(BASELINE_SCALE_MAX_HZ if lowpass_cutoff_hz is None else lowpass_cutoff_hz)
    if smoothing_window_sec is None:
        smoothing_window_sec = float(default_smoothing_window_sec)
    try:
        sample_rate = float(fs)
    except Exception:
        sample_rate = float("nan")

    meta: dict[str, Any] = {
        "baseline_ref_candidate_available": False,
        "baseline_ref_method": "configurable_ultra_lowpass_smoothing_reference_linear_fit",
        "baseline_ref_lowpass_cutoff_hz": _json_float_or_none(cutoff),
        "baseline_ref_smoothing_window_sec": None,
        "baseline_ref_requested_smoothing_window_sec": None,
        "baseline_ref_actual_smoothing_window_sec": None,
        "baseline_ref_default_smoothing_window_sec": _json_float_or_none(default_smoothing_window_sec),
        "baseline_ref_min_smoothing_window_sec": _json_float_or_none(min_smoothing_window_sec),
        "baseline_ref_chunk_duration_sec": None,
        "baseline_ref_smoothing_window_fraction_of_chunk": None,
        "baseline_ref_max_window_fraction_of_chunk": _json_float_or_none(max_window_fraction_of_chunk),
        "baseline_ref_large_window_fraction_warning": _json_float_or_none(large_window_fraction_warning),
        "baseline_ref_smoothing_window_adjusted": False,
        "baseline_ref_smoothing_window_warning": None,
        "baseline_ref_slope": None,
        "baseline_ref_intercept": None,
        "baseline_ref_n_samples": 0,
        "baseline_ref_status": "unavailable",
        "baseline_ref_warning": None,
        "baseline_ref_residual_exclusion_fraction": None,
        "baseline_ref_residual_exclusion_threshold": None,
        "baseline_ref_initial_fit_status": None,
        "baseline_ref_final_fit_status": None,
        "baseline_ref_fit_stage": None,
        "baseline_ref_candidate": None,
    }

    if sig.shape != ref.shape:
        meta["baseline_ref_warning"] = f"shape_mismatch:{sig.shape}!={ref.shape}"
        return meta
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        meta["baseline_ref_warning"] = "invalid_sample_rate"
        return meta
    window_meta = _resolve_smoothing_window(
        smoothing_window_sec,
        n_samples=int(sig.size),
        fs=sample_rate,
        default_sec=float(default_smoothing_window_sec),
        min_sec=float(min_smoothing_window_sec),
        max_fraction=float(max_window_fraction_of_chunk),
        large_fraction_warning=float(large_window_fraction_warning),
    )
    meta.update(window_meta)
    meta["baseline_ref_smoothing_window_sec"] = meta["baseline_ref_actual_smoothing_window_sec"]
    finite_pairs = np.isfinite(sig) & np.isfinite(ref)
    if int(np.sum(finite_pairs)) < int(min_samples):
        meta["baseline_ref_n_samples"] = int(np.sum(finite_pairs))
        meta["baseline_ref_warning"] = "insufficient_finite_samples"
        return meta

    window_samples = max(
        3,
        int(round(float(meta["baseline_ref_actual_smoothing_window_sec"]) * sample_rate)),
    )
    sig_slow = _nan_aware_moving_average(sig, window_samples)
    ref_slow = _nan_aware_moving_average(ref, window_samples)
    slope, intercept, n_used, fit_meta = _fit_with_residual_exclusion(
        ref_slow,
        sig_slow,
        robust=robust,
        min_samples=int(min_samples),
        return_diagnostics=return_diagnostics,
    )
    meta["baseline_ref_n_samples"] = int(n_used)
    meta.update(fit_meta)
    meta["baseline_ref_status"] = str(fit_meta.get("baseline_ref_final_fit_status") or "unavailable")
    corr, corr_reason = _safe_corr(sig_slow, ref_slow)
    meta["baseline_ref_smoothed_signal_reference_corr"] = corr
    meta["baseline_ref_smoothed_signal_reference_corr_reason"] = corr_reason
    if not np.isfinite(slope) or not np.isfinite(intercept):
        meta["baseline_ref_warning"] = meta["baseline_ref_status"]
        if return_diagnostics:
            meta["baseline_ref_smoothed_signal"] = sig_slow
            meta["baseline_ref_smoothed_reference"] = ref_slow
            meta["baseline_fit_relationship_class"] = "unknown"
        return meta

    candidate = float(intercept) + float(slope) * ref_slow
    relationship_class = classify_baseline_fit_relationship(
        slope=slope,
        corr=corr,
    )
    meta.update(
        {
            "baseline_ref_candidate_available": True,
            "baseline_ref_slope": float(slope),
            "baseline_ref_intercept": float(intercept),
            "baseline_ref_candidate": candidate,
            "baseline_fit_relationship_class": relationship_class,
        }
    )
    if return_diagnostics:
        initial_slope = fit_meta.get("baseline_ref_initial_slope")
        initial_intercept = fit_meta.get("baseline_ref_initial_intercept")
        if initial_slope is not None and initial_intercept is not None:
            initial_candidate = float(initial_intercept) + float(initial_slope) * ref_slow
        else:
            initial_candidate = np.full_like(ref_slow, np.nan, dtype=float)
        meta.update(
            {
                "baseline_ref_smoothed_signal": sig_slow,
                "baseline_ref_smoothed_reference": ref_slow,
                "baseline_ref_initial_candidate": initial_candidate,
            }
        )
    return meta


def compute_baseline_reference_candidate_metrics(
    signal: np.ndarray,
    reference: np.ndarray,
    dynamic_fitted_ref: np.ndarray,
    baseline_candidate: np.ndarray,
    fs: float,
) -> dict[str, Any]:
    """Return comparison metrics for dynamic and baseline-only reference candidates."""
    sig = np.asarray(signal, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    dyn = np.asarray(dynamic_fitted_ref, dtype=float).reshape(-1)
    base = np.asarray(baseline_candidate, dtype=float).reshape(-1)
    if sig.shape != ref.shape or sig.shape != dyn.shape or sig.shape != base.shape:
        raise ValueError("Baseline reference metric inputs must have matching shapes")

    sig_p05, sig_p95, sig_range = _range_p95_p05(sig)
    ref_p05, ref_p95, ref_range = _range_p95_p05(ref)
    base_p05, base_p95, base_range = _range_p95_p05(base)
    dyn_p05, dyn_p95, dyn_range = _range_p95_p05(dyn)
    base_sig_ratio, base_sig_reason = _safe_ratio(base_range, sig_range)
    base_ref_ratio, base_ref_reason = _safe_ratio(base_range, ref_range)
    dyn_sig_ratio, dyn_sig_reason = _safe_ratio(dyn_range, sig_range)
    sig_base_corr, sig_base_corr_reason = _safe_corr(sig, base)
    ref_base_corr, ref_base_corr_reason = _safe_corr(ref, base)
    dyn_base_corr, dyn_base_corr_reason = _safe_corr(dyn, base)

    diff = dyn - base
    finite_diff = diff[np.isfinite(diff)]
    diff_rms = float(np.sqrt(np.mean(finite_diff**2))) if finite_diff.size else float("nan")
    diff_p05, diff_p95, diff_range = _range_p95_p05(diff)

    base_power = _variance_band_metrics(
        base,
        fs,
        baseline_cutoff_hz=BASELINE_SCALE_MAX_HZ,
        response_band_min_hz=RESPONSE_SCALE_MIN_HZ,
        response_band_max_hz=RESPONSE_SCALE_MAX_HZ,
    )
    dyn_power = _variance_band_metrics(
        dyn,
        fs,
        baseline_cutoff_hz=BASELINE_SCALE_MAX_HZ,
        response_band_min_hz=RESPONSE_SCALE_MIN_HZ,
        response_band_max_hz=RESPONSE_SCALE_MAX_HZ,
    )
    base_response_frac = float(base_power.get("fitted_ref_response_scale_fraction", float("nan")))
    dyn_response_frac = float(dyn_power.get("fitted_ref_response_scale_fraction", float("nan")))
    base_baseline_frac = float(base_power.get("fitted_ref_baseline_scale_fraction", float("nan")))

    base_low_range = (
        (np.isfinite(base_sig_ratio) and base_sig_ratio < 0.05)
        or (np.isfinite(base_ref_ratio) and base_ref_ratio < 0.05)
    )
    base_flat = bool(base_low_range)
    base_response_rich = bool(np.isfinite(base_response_frac) and base_response_frac > 0.35)
    return {
        "baseline_ref_to_signal_range_ratio": base_sig_ratio,
        "baseline_ref_to_signal_range_ratio_reason": base_sig_reason,
        "baseline_ref_to_iso_range_ratio": base_ref_ratio,
        "baseline_ref_to_iso_range_ratio_reason": base_ref_reason,
        "signal_baseline_ref_corr": sig_base_corr,
        "signal_baseline_ref_corr_reason": sig_base_corr_reason,
        "iso_baseline_ref_corr": ref_base_corr,
        "iso_baseline_ref_corr_reason": ref_base_corr_reason,
        "dynamic_ref_to_baseline_ref_corr": dyn_base_corr,
        "dynamic_ref_to_baseline_ref_corr_reason": dyn_base_corr_reason,
        "baseline_ref_baseline_scale_fraction": base_baseline_frac,
        "baseline_ref_response_scale_fraction": base_response_frac,
        "baseline_ref_response_scale_rich": base_response_rich,
        "baseline_ref_low_range": bool(base_low_range),
        "baseline_ref_flat_or_uninformative": bool(base_flat),
        "dynamic_minus_baseline_ref_rms": diff_rms,
        "dynamic_minus_baseline_ref_range": diff_range,
        "dynamic_ref_response_scale_fraction": dyn_response_frac,
        "response_scale_fraction_delta_dynamic_minus_baseline": (
            float(dyn_response_frac - base_response_frac)
            if np.isfinite(dyn_response_frac) and np.isfinite(base_response_frac)
            else float("nan")
        ),
        "dynamic_ref_to_signal_range_ratio": dyn_sig_ratio,
        "dynamic_ref_to_signal_range_ratio_reason": dyn_sig_reason,
        "baseline_ref_p05": base_p05,
        "baseline_ref_p95": base_p95,
        "baseline_ref_range_p95_p05": base_range,
        "dynamic_ref_p05": dyn_p05,
        "dynamic_ref_p95": dyn_p95,
        "dynamic_ref_range_p95_p05": dyn_range,
        "signal_p05": sig_p05,
        "signal_p95": sig_p95,
        "signal_range_p95_p05": sig_range,
        "iso_p05": ref_p05,
        "iso_p95": ref_p95,
        "iso_range_p95_p05": ref_range,
        "dynamic_minus_baseline_ref_p05": diff_p05,
        "dynamic_minus_baseline_ref_p95": diff_p95,
    }
