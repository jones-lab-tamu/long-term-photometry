"""Diagnostic-only signal-derived F0 candidate metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULTS = {
    "signal_only_f0_window_fraction": 0.20,
    "signal_only_f0_window_sec": None,
    "signal_only_f0_low_quantile": 0.10,
    "signal_only_f0_smoothing_window_fraction": 0.10,
    "signal_only_f0_smoothing_window_sec": None,
    "signal_only_f0_min_window_samples": 21,
    "signal_only_f0_max_window_fraction": 0.50,
    "signal_only_f0_min_robust_range": 1e-6,
    "signal_only_f0_max_above_signal_fraction": 0.20,
    "signal_only_f0_max_tracking_fraction": 0.85,
    "signal_only_f0_min_coverage_fraction": 0.80,
    "signal_only_f0_high_state_context_mode": "contextual_cap",
    "signal_only_f0_state_aware_enabled": True,
    "signal_only_f0_low_support_quantile": 0.35,
    "signal_only_f0_low_support_buffer_fraction": 0.02,
    "signal_only_f0_low_support_buffer_sec": None,
    "signal_only_f0_min_low_support_fraction": 0.10,
    "signal_only_f0_min_anchor_count": 3,
    "signal_only_f0_max_anchor_gap_fraction": 0.50,
    "signal_only_f0_max_anchor_gap_sec": None,
    "signal_only_f0_edge_extrapolation_mode": "hold_nearest_anchor",
    "signal_only_f0_max_edge_extrapolation_fraction": 0.50,
    "signal_only_f0_max_edge_extrapolation_sec": None,
    "signal_only_f0_medium_extrapolation_fraction": 0.25,
    "signal_only_f0_high_extrapolation_fraction": 0.50,
    "signal_only_f0_low_anchor_support_fraction": 0.10,
    "signal_only_f0_low_anchor_count": 5,
    "signal_only_f0_confidence_cap_on_large_gap": True,
}

VIABILITY_VIABLE = "viable"
VIABILITY_CONTEXTUAL = "contextual"
VIABILITY_HARD_INSPECT = "hard_inspect"
VIABILITY_UNAVAILABLE = "unavailable"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_NONE = "none"

FLAG_AVAILABLE = "SIGNAL_ONLY_F0_AVAILABLE"
FLAG_VIABLE = "SIGNAL_ONLY_F0_VIABLE"
FLAG_CONTEXTUAL = "SIGNAL_ONLY_F0_CONTEXTUAL"
FLAG_HARD_INSPECT = "SIGNAL_ONLY_F0_HARD_INSPECT"
FLAG_INSUFFICIENT_SAMPLES = "SIGNAL_ONLY_F0_INSUFFICIENT_SAMPLES"
FLAG_INSUFFICIENT_RANGE = "SIGNAL_ONLY_F0_INSUFFICIENT_RANGE"
FLAG_LOW_SUPPORT = "SIGNAL_ONLY_F0_LOW_SUPPORT"
FLAG_EXCESSIVE_TRACKING = "SIGNAL_ONLY_F0_EXCESSIVE_TRACKING"
FLAG_ABOVE_SIGNAL_EXCESSIVE = "SIGNAL_ONLY_F0_ABOVE_SIGNAL_EXCESSIVE"
FLAG_HIGH_STATE = "SIGNAL_ONLY_F0_HIGH_STATE_PRESENT"
FLAG_PARTIAL_HIGH_STATE = "SIGNAL_ONLY_F0_PARTIAL_HIGH_STATE_PRESENT"
FLAG_EDGE_HIGH_STATE = "SIGNAL_ONLY_F0_EDGE_HIGH_STATE_PRESENT"
FLAG_STATE_AWARE_USED = "SIGNAL_ONLY_F0_STATE_AWARE_USED"
FLAG_LOW_SUPPORT_ANCHORED = "SIGNAL_ONLY_F0_LOW_SUPPORT_ANCHORED"
FLAG_EDGE_EXTRAPOLATED = "SIGNAL_ONLY_F0_EDGE_EXTRAPOLATED"
FLAG_INTERPOLATED_HIGH = "SIGNAL_ONLY_F0_INTERPOLATED_OVER_HIGH_STATE"
FLAG_INSUFFICIENT_LOW_SUPPORT = "SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT"
FLAG_INSUFFICIENT_ANCHORS = "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS"
FLAG_LARGE_ANCHOR_GAP = "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP"
FLAG_ROLLING_FALLBACK = "SIGNAL_ONLY_F0_ROLLING_FALLBACK_USED"
FLAG_CONFIDENCE_CAPPED_EXTRAPOLATION = "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION"
FLAG_CONFIDENCE_CAPPED_LOW_ANCHOR_SUPPORT = (
    "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_LOW_ANCHOR_SUPPORT"
)
FLAG_CONFIDENCE_CAPPED_FEW_ANCHORS = "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS"
FLAG_CONFIDENCE_CAPPED_LARGE_GAP = "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_LARGE_GAP"


def _cfg(config: Mapping[str, Any] | None, key: str) -> Any:
    if isinstance(config, Mapping) and key in config:
        return config[key]
    if key == "signal_only_f0_high_state_context_mode" and isinstance(config, Mapping):
        legacy = config.get("signal_only_f0_high_state_exclusion_mode")
        if legacy in {"exclude_high_state_candidates", "downweight_high_state_candidates"}:
            return "contextual_cap"
        if legacy == "none":
            return "none"
    return DEFAULTS[key]


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        out = int(round(float(value)))
    except Exception:
        return int(default)
    return max(1, out)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    if value is None:
        return bool(default)
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _window_samples(
    *,
    n: int,
    duration_sec: float | None,
    sample_interval_sec: float | None,
    fraction: float,
    requested_sec: Any,
    min_samples: int,
    max_fraction: float,
) -> tuple[int, float | None]:
    if n <= 1:
        return 1, None
    sec = None
    try:
        if requested_sec is not None:
            sec_val = float(requested_sec)
            if np.isfinite(sec_val) and sec_val > 0:
                sec = sec_val
    except Exception:
        sec = None
    if sec is None and duration_sec is not None and np.isfinite(duration_sec) and duration_sec > 0:
        sec = max(float(duration_sec) * float(fraction), 0.0)
    if sec is not None and sample_interval_sec is not None and sample_interval_sec > 0:
        samples = int(round(sec / sample_interval_sec))
    else:
        samples = int(round(max(1.0, float(n) * float(fraction))))
    max_samples = max(1, int(round(float(n) * max(0.01, min(float(max_fraction), 1.0)))))
    samples = max(int(min_samples), samples)
    samples = max(1, min(int(samples), int(n), max_samples))
    if samples % 2 == 0 and samples < n:
        samples += 1
        if samples > max_samples and samples > 1:
            samples -= 2
    samples = max(1, min(samples, n))
    actual_sec = (
        float(samples) * float(sample_interval_sec)
        if sample_interval_sec is not None and sample_interval_sec > 0
        else None
    )
    return samples, actual_sec


def _moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    n = arr.size
    if n == 0:
        return arr.copy()
    window = max(1, min(int(window), n))
    if window <= 1:
        return arr.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    values = np.where(np.isfinite(arr), arr, 0.0)
    weights = np.isfinite(arr).astype(float)
    if n == 1:
        return arr.copy()
    values_pad = np.pad(values, (pad_left, pad_right), mode="reflect")
    weights_pad = np.pad(weights, (pad_left, pad_right), mode="reflect")
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(values_pad, kernel, mode="valid")
    denominator = np.convolve(weights_pad, kernel, mode="valid")
    out = np.full(n, np.nan, dtype=float)
    ok = denominator > 0
    out[ok] = numerator[ok] / denominator[ok]
    return out


def _base_result(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "signal_only_f0_candidate_available": False,
        "signal_only_f0_status": "unavailable",
        "signal_only_f0_warning": "",
        "signal_only_f0_method": "rolling_lower_quantile_envelope",
        "signal_only_f0_window_fraction": _as_float(
            _cfg(config, "signal_only_f0_window_fraction"), 0.20
        ),
        "signal_only_f0_window_sec_requested": _cfg(config, "signal_only_f0_window_sec"),
        "signal_only_f0_window_samples": 0,
        "signal_only_f0_window_sec_actual": None,
        "signal_only_f0_low_quantile": _as_float(
            _cfg(config, "signal_only_f0_low_quantile"), 0.10
        ),
        "signal_only_f0_smoothing_window_fraction": _as_float(
            _cfg(config, "signal_only_f0_smoothing_window_fraction"), 0.10
        ),
        "signal_only_f0_smoothing_window_sec_requested": _cfg(
            config, "signal_only_f0_smoothing_window_sec"
        ),
        "signal_only_f0_smoothing_window_samples": 0,
        "signal_only_f0_smoothing_window_sec_actual": None,
        "signal_only_f0_min_window_samples": _as_int(
            _cfg(config, "signal_only_f0_min_window_samples"), 21
        ),
        "signal_only_f0_max_window_fraction": _as_float(
            _cfg(config, "signal_only_f0_max_window_fraction"), 0.50
        ),
        "signal_only_f0_min_robust_range": _as_float(
            _cfg(config, "signal_only_f0_min_robust_range"), 1e-6
        ),
        "signal_only_f0_max_above_signal_fraction": _as_float(
            _cfg(config, "signal_only_f0_max_above_signal_fraction"), 0.20
        ),
        "signal_only_f0_max_tracking_fraction": _as_float(
            _cfg(config, "signal_only_f0_max_tracking_fraction"), 0.85
        ),
        "signal_only_f0_min_coverage_fraction": _as_float(
            _cfg(config, "signal_only_f0_min_coverage_fraction"), 0.80
        ),
        "signal_only_f0_high_state_context_mode": str(
            _cfg(config, "signal_only_f0_high_state_context_mode")
        ),
        "signal_only_f0_high_state_context_cap": None,
        "signal_only_f0_high_state_context_applied": False,
        "signal_only_f0_state_aware_enabled": _as_bool(
            _cfg(config, "signal_only_f0_state_aware_enabled"), True
        ),
        "signal_only_f0_state_aware_used": False,
        "signal_only_f0_low_support_quantile": _as_float(
            _cfg(config, "signal_only_f0_low_support_quantile"), 0.35
        ),
        "signal_only_f0_low_support_buffer_fraction": _as_float(
            _cfg(config, "signal_only_f0_low_support_buffer_fraction"), 0.02
        ),
        "signal_only_f0_low_support_buffer_sec": _cfg(
            config, "signal_only_f0_low_support_buffer_sec"
        ),
        "signal_only_f0_min_low_support_fraction": _as_float(
            _cfg(config, "signal_only_f0_min_low_support_fraction"), 0.10
        ),
        "signal_only_f0_min_anchor_count": _as_int(
            _cfg(config, "signal_only_f0_min_anchor_count"), 3
        ),
        "signal_only_f0_max_anchor_gap_fraction": _as_float(
            _cfg(config, "signal_only_f0_max_anchor_gap_fraction"), 0.50
        ),
        "signal_only_f0_max_anchor_gap_sec": _cfg(config, "signal_only_f0_max_anchor_gap_sec"),
        "signal_only_f0_edge_extrapolation_mode": str(
            _cfg(config, "signal_only_f0_edge_extrapolation_mode")
        ),
        "signal_only_f0_max_edge_extrapolation_fraction": _as_float(
            _cfg(config, "signal_only_f0_max_edge_extrapolation_fraction"), 0.50
        ),
        "signal_only_f0_max_edge_extrapolation_sec": _cfg(
            config, "signal_only_f0_max_edge_extrapolation_sec"
        ),
        "signal_only_f0_medium_extrapolation_fraction": _as_float(
            _cfg(config, "signal_only_f0_medium_extrapolation_fraction"), 0.25
        ),
        "signal_only_f0_high_extrapolation_fraction": _as_float(
            _cfg(config, "signal_only_f0_high_extrapolation_fraction"), 0.50
        ),
        "signal_only_f0_low_anchor_support_fraction": _as_float(
            _cfg(config, "signal_only_f0_low_anchor_support_fraction"), 0.10
        ),
        "signal_only_f0_low_anchor_count": _as_int(
            _cfg(config, "signal_only_f0_low_anchor_count"), 5
        ),
        "signal_only_f0_confidence_cap_on_large_gap": _as_bool(
            _cfg(config, "signal_only_f0_confidence_cap_on_large_gap"), True
        ),
        "signal_only_f0_anchor_count": 0,
        "signal_only_f0_low_support_fraction": 0.0,
        "signal_only_f0_anchor_support_fraction": 0.0,
        "signal_only_f0_direct_support_fraction": 0.0,
        "signal_only_f0_interpolated_fraction": 0.0,
        "signal_only_f0_extrapolated_fraction": 0.0,
        "signal_only_f0_edge_extrapolation_fraction": 0.0,
        "signal_only_f0_max_anchor_gap_fraction_observed": None,
        "signal_only_f0_max_anchor_gap_sec_observed": None,
        "signal_only_f0_anchor_status": "unavailable",
        "signal_only_f0_candidate_viability": VIABILITY_UNAVAILABLE,
        "signal_only_f0_candidate_confidence": CONFIDENCE_NONE,
        "signal_only_f0_support_fraction": 0.0,
        "signal_only_f0_low_state_support_fraction": 0.0,
        "signal_only_f0_to_signal_corr": None,
        "signal_only_f0_to_signal_range_ratio": None,
        "signal_only_f0_below_signal_fraction": 0.0,
        "signal_only_f0_above_signal_fraction_pre_cap": 0.0,
        "signal_only_f0_above_signal_fraction": 0.0,
        "signal_only_f0_tracking_score": None,
        "signal_only_f0_residual_p05": None,
        "signal_only_f0_residual_p50": None,
        "signal_only_f0_residual_p95": None,
        "signal_only_f0_p05": None,
        "signal_only_f0_p50": None,
        "signal_only_f0_p95": None,
        "signal_only_f0_range_p95_p05": None,
        "signal_only_f0_flags": [],
        "signal_only_f0_candidate": None,
    }


def _duration_and_dt(signal: np.ndarray, time: np.ndarray | None) -> tuple[float | None, float | None]:
    if time is None:
        return None, None
    t = np.asarray(time, dtype=float).reshape(-1)
    if t.shape != signal.shape:
        return None, None
    finite = np.isfinite(t) & np.isfinite(signal)
    finite_t = t[finite]
    if finite_t.size < 2:
        return None, None
    diffs = np.diff(finite_t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt = float(np.median(diffs)) if diffs.size else None
    duration = float(finite_t[-1] - finite_t[0] + (dt or 0.0))
    return duration, dt


def _lower_quantile_envelope(signal: np.ndarray, window: int, quantile: float) -> np.ndarray:
    sig = np.asarray(signal, dtype=float).reshape(-1)
    n = sig.size
    if n == 0:
        return sig.copy()
    window = max(1, min(int(window), n))
    step = max(1, window // 4)
    half = window // 2
    centers = list(range(0, n, step))
    if centers[-1] != n - 1:
        centers.append(n - 1)
    values = []
    valid_centers = []
    q = min(max(float(quantile), 0.0), 0.5)
    for center in centers:
        start = max(0, int(center) - half)
        end = min(n, int(center) + half + 1)
        local = sig[start:end]
        finite = local[np.isfinite(local)]
        if finite.size == 0:
            continue
        valid_centers.append(float(center))
        values.append(float(np.quantile(finite, q)))
    if not values:
        return np.full(n, np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    out = np.interp(x, np.asarray(valid_centers), np.asarray(values))
    return out


def _state_flag_present(signal_state: Mapping[str, Any] | None, flag: str) -> bool:
    if not isinstance(signal_state, Mapping):
        return False
    flags = signal_state.get("signal_state_flags", [])
    if isinstance(flags, str):
        flags = [x for x in flags.split(";") if x]
    return isinstance(flags, (list, tuple)) and flag in {str(x) for x in flags}


def _buffer_samples(
    *,
    n: int,
    fraction: float,
    requested_sec: Any,
    sample_interval_sec: float | None,
) -> int:
    samples = int(round(max(0.0, float(fraction)) * max(1, int(n))))
    try:
        if requested_sec is not None and sample_interval_sec is not None and sample_interval_sec > 0:
            sec_val = float(requested_sec)
            if np.isfinite(sec_val) and sec_val > 0:
                samples = int(round(sec_val / sample_interval_sec))
    except Exception:
        pass
    return max(0, min(samples, max(0, int(n) - 1)))


def _expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if radius <= 0 or arr.size == 0 or not np.any(arr):
        return arr.copy()
    kernel = np.ones(2 * int(radius) + 1, dtype=int)
    return np.convolve(arr.astype(int), kernel, mode="same") > 0


def _state_high_threshold(
    *,
    signal_state: Mapping[str, Any] | None,
    p05: float,
    robust_range: float,
) -> float:
    if isinstance(signal_state, Mapping):
        try:
            value = float(signal_state.get("signal_state_high_threshold"))
            if np.isfinite(value):
                return value
        except Exception:
            pass
    return float(p05 + 0.80 * robust_range)


def _build_state_aware_candidate(
    *,
    signal: np.ndarray,
    raw_candidate: np.ndarray,
    smoothed_signal: np.ndarray,
    p05: float,
    p50: float,
    robust_range: float,
    window: int,
    low_quantile: float,
    signal_state: Mapping[str, Any] | None,
    high_state_present: bool,
    partial_high_state_present: bool,
    edge_high_state_present: bool,
    result: dict[str, Any],
    sample_interval_sec: float | None,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    n = int(signal.size)
    flags: list[str] = []
    meta: dict[str, Any] = {
        "signal_only_f0_state_aware_used": False,
        "signal_only_f0_anchor_count": 0,
        "signal_only_f0_low_support_fraction": 0.0,
        "signal_only_f0_anchor_support_fraction": 0.0,
        "signal_only_f0_direct_support_fraction": 0.0,
        "signal_only_f0_interpolated_fraction": 0.0,
        "signal_only_f0_extrapolated_fraction": 0.0,
        "signal_only_f0_edge_extrapolation_fraction": 0.0,
        "signal_only_f0_max_anchor_gap_fraction_observed": None,
        "signal_only_f0_max_anchor_gap_sec_observed": None,
        "signal_only_f0_anchor_status": "fallback_rolling_candidate",
    }
    if not bool(result["signal_only_f0_state_aware_enabled"]):
        return raw_candidate, meta, flags

    finite = np.isfinite(signal)
    low_threshold = min(
        float(p50),
        float(p05 + float(result["signal_only_f0_low_support_quantile"]) * robust_range),
    )
    low_candidate_mask = finite & np.isfinite(smoothed_signal) & (smoothed_signal <= low_threshold)
    has_state_context = bool(high_state_present or partial_high_state_present or edge_high_state_present)
    high_like = np.zeros(n, dtype=bool)
    if has_state_context:
        high_threshold = _state_high_threshold(
            signal_state=signal_state,
            p05=p05,
            robust_range=robust_range,
        )
        high_like = np.isfinite(smoothed_signal) & (smoothed_signal >= high_threshold)
        buffer_n = _buffer_samples(
            n=n,
            fraction=float(result["signal_only_f0_low_support_buffer_fraction"]),
            requested_sec=result["signal_only_f0_low_support_buffer_sec"],
            sample_interval_sec=sample_interval_sec,
        )
        high_like = _expand_mask(high_like, buffer_n)
    low_support_mask = low_candidate_mask & ~high_like
    low_support_fraction = float(np.sum(low_support_mask) / max(1, n))
    meta["signal_only_f0_low_support_fraction"] = low_support_fraction
    meta["signal_only_f0_direct_support_fraction"] = low_support_fraction

    if not np.any(low_support_mask):
        meta["signal_only_f0_anchor_status"] = "no_low_support"
        flags.extend([FLAG_INSUFFICIENT_LOW_SUPPORT, FLAG_ROLLING_FALLBACK])
        return raw_candidate, meta, flags

    min_anchor_count = int(result["signal_only_f0_min_anchor_count"])
    step = max(1, int(window) // 4)
    half = max(1, int(window) // 2)
    anchor_x: list[float] = []
    anchor_y: list[float] = []
    anchor_support_samples = 0
    min_local_support = max(3, min(half, int(round(0.05 * max(1, int(window))))))
    for center in list(range(0, n, step)) + [n - 1]:
        start = max(0, int(center) - half)
        end = min(n, int(center) + half + 1)
        local_mask = low_support_mask[start:end]
        if int(np.sum(local_mask)) < min_local_support:
            continue
        local_idx = np.flatnonzero(local_mask) + start
        local_vals = signal[local_idx]
        finite_vals = local_vals[np.isfinite(local_vals)]
        if finite_vals.size < min_local_support:
            continue
        anchor_x.append(float(np.median(local_idx)))
        anchor_y.append(float(np.quantile(finite_vals, min(max(float(low_quantile), 0.0), 0.5))))
        anchor_support_samples += int(finite_vals.size)
    if anchor_x:
        order = np.argsort(anchor_x)
        anchor_x = [anchor_x[i] for i in order]
        anchor_y = [anchor_y[i] for i in order]
        dedup_x: list[float] = []
        dedup_y: list[float] = []
        for x_val, y_val in zip(anchor_x, anchor_y):
            if dedup_x and abs(x_val - dedup_x[-1]) < 1e-9:
                dedup_y[-1] = min(dedup_y[-1], y_val)
            else:
                dedup_x.append(x_val)
                dedup_y.append(y_val)
        anchor_x, anchor_y = dedup_x, dedup_y

    anchor_count = len(anchor_x)
    meta["signal_only_f0_anchor_count"] = int(anchor_count)
    meta["signal_only_f0_anchor_support_fraction"] = float(anchor_support_samples / max(1, n))
    if anchor_count < min_anchor_count:
        meta["signal_only_f0_anchor_status"] = (
            "no_low_support" if low_support_fraction <= 0.0 else "insufficient_anchors"
        )
        flags.append(FLAG_INSUFFICIENT_ANCHORS)
        if low_support_fraction < float(result["signal_only_f0_min_low_support_fraction"]):
            flags.append(FLAG_INSUFFICIENT_LOW_SUPPORT)
        flags.append(FLAG_ROLLING_FALLBACK)
        return raw_candidate, meta, flags

    x_all = np.arange(n, dtype=float)
    anchors_x_arr = np.asarray(anchor_x, dtype=float)
    anchors_y_arr = np.asarray(anchor_y, dtype=float)
    edge_mode = str(result["signal_only_f0_edge_extrapolation_mode"]).strip().lower()
    if edge_mode not in {"hold_nearest_anchor", "interpolate_only"}:
        edge_mode = "hold_nearest_anchor"
        meta["signal_only_f0_edge_extrapolation_mode"] = edge_mode
    candidate = np.interp(x_all, anchors_x_arr, anchors_y_arr)
    if edge_mode == "interpolate_only":
        candidate[x_all < anchors_x_arr[0]] = np.nan
        candidate[x_all > anchors_x_arr[-1]] = np.nan

    before_first = x_all < anchors_x_arr[0]
    after_last = x_all > anchors_x_arr[-1]
    extrapolated = (before_first | after_last) & np.isfinite(candidate)
    direct = low_support_mask & np.isfinite(candidate)
    interpolated = np.isfinite(candidate) & ~(direct | extrapolated)
    meta["signal_only_f0_state_aware_used"] = True
    meta["signal_only_f0_anchor_status"] = "sufficient_anchors"
    meta["signal_only_f0_direct_support_fraction"] = float(np.sum(direct) / max(1, n))
    meta["signal_only_f0_interpolated_fraction"] = float(np.sum(interpolated) / max(1, n))
    meta["signal_only_f0_extrapolated_fraction"] = float(np.sum(extrapolated) / max(1, n))
    meta["signal_only_f0_edge_extrapolation_fraction"] = meta[
        "signal_only_f0_extrapolated_fraction"
    ]
    gaps = np.diff(anchors_x_arr)
    if gaps.size:
        max_gap = float(np.max(gaps))
        meta["signal_only_f0_max_anchor_gap_fraction_observed"] = float(max_gap / max(1, n))
        meta["signal_only_f0_max_anchor_gap_sec_observed"] = (
            float(max_gap * sample_interval_sec)
            if sample_interval_sec is not None and sample_interval_sec > 0
            else None
        )
    flags.extend([FLAG_STATE_AWARE_USED, FLAG_LOW_SUPPORT_ANCHORED])
    if np.any(extrapolated):
        flags.append(FLAG_EDGE_EXTRAPOLATED)
    if has_state_context and np.any(interpolated):
        flags.append(FLAG_INTERPOLATED_HIGH)
    max_gap_fraction = meta.get("signal_only_f0_max_anchor_gap_fraction_observed")
    large_gap = bool(
        max_gap_fraction is not None
        and float(max_gap_fraction) > float(result["signal_only_f0_max_anchor_gap_fraction"])
    )
    try:
        max_gap_sec_allowed = result["signal_only_f0_max_anchor_gap_sec"]
        max_gap_sec_observed = meta.get("signal_only_f0_max_anchor_gap_sec_observed")
        if max_gap_sec_allowed is not None and max_gap_sec_observed is not None:
            large_gap = large_gap or float(max_gap_sec_observed) > float(max_gap_sec_allowed)
    except Exception:
        pass
    if large_gap:
        flags.append(FLAG_LARGE_ANCHOR_GAP)
    return candidate, meta, flags


def _confidence_rank(confidence: str) -> int:
    return {
        CONFIDENCE_NONE: 0,
        CONFIDENCE_LOW: 1,
        CONFIDENCE_MEDIUM: 2,
        CONFIDENCE_HIGH: 3,
    }.get(str(confidence), 0)


def _cap_confidence(confidence: str, cap: str) -> str:
    return confidence if _confidence_rank(confidence) <= _confidence_rank(cap) else cap


def _calibrate_candidate_confidence(
    *,
    result: Mapping[str, Any],
    flags: list[str],
    viability: str,
    confidence: str,
    has_state_context: bool,
    large_anchor_gap: bool,
) -> tuple[str, str, list[str]]:
    if viability == VIABILITY_HARD_INSPECT:
        return viability, CONFIDENCE_LOW, flags

    calibrated_flags = list(flags)
    extrapolated_fraction = float(result.get("signal_only_f0_extrapolated_fraction", 0.0) or 0.0)
    medium_extrapolation = float(
        result.get("signal_only_f0_medium_extrapolation_fraction", 0.25) or 0.25
    )
    high_extrapolation = float(
        result.get("signal_only_f0_high_extrapolation_fraction", 0.50) or 0.50
    )
    anchor_support = float(result.get("signal_only_f0_anchor_support_fraction", 0.0) or 0.0)
    low_anchor_support = float(
        result.get("signal_only_f0_low_anchor_support_fraction", 0.10) or 0.10
    )
    anchor_count = int(result.get("signal_only_f0_anchor_count", 0) or 0)
    low_anchor_count = int(result.get("signal_only_f0_low_anchor_count", 5) or 5)

    if has_state_context:
        confidence = _cap_confidence(confidence, CONFIDENCE_MEDIUM)
        viability = VIABILITY_CONTEXTUAL
        if extrapolated_fraction >= medium_extrapolation:
            confidence = CONFIDENCE_LOW
            calibrated_flags.append(FLAG_CONFIDENCE_CAPPED_EXTRAPOLATION)
    else:
        if extrapolated_fraction >= high_extrapolation:
            viability = VIABILITY_CONTEXTUAL
            confidence = CONFIDENCE_LOW
            calibrated_flags.append(FLAG_CONFIDENCE_CAPPED_EXTRAPOLATION)
        elif extrapolated_fraction >= medium_extrapolation:
            confidence = _cap_confidence(confidence, CONFIDENCE_MEDIUM)
            calibrated_flags.append(FLAG_CONFIDENCE_CAPPED_EXTRAPOLATION)

    if anchor_count > 0 and anchor_support < low_anchor_support:
        confidence = _cap_confidence(confidence, CONFIDENCE_MEDIUM)
        calibrated_flags.append(FLAG_CONFIDENCE_CAPPED_LOW_ANCHOR_SUPPORT)
    if anchor_count > 0 and anchor_count < low_anchor_count:
        confidence = _cap_confidence(confidence, CONFIDENCE_MEDIUM)
        calibrated_flags.append(FLAG_CONFIDENCE_CAPPED_FEW_ANCHORS)
    if large_anchor_gap and bool(result.get("signal_only_f0_confidence_cap_on_large_gap", True)):
        confidence = _cap_confidence(
            confidence,
            CONFIDENCE_LOW if has_state_context else CONFIDENCE_MEDIUM,
        )
        calibrated_flags.append(FLAG_CONFIDENCE_CAPPED_LARGE_GAP)
        if not has_state_context and viability == VIABILITY_VIABLE:
            viability = VIABILITY_CONTEXTUAL

    return viability, confidence, calibrated_flags


def compute_signal_only_f0_candidate(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    signal_state: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute diagnostic-only signal-derived lower-envelope F0 candidate metrics."""
    result = _base_result(config)
    sig = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(sig)
    if sig.size < 10 or int(np.sum(finite)) < 10:
        result.update(
            {
                "signal_only_f0_status": "insufficient",
                "signal_only_f0_warning": "insufficient_finite_signal_samples",
                "signal_only_f0_flags": [FLAG_INSUFFICIENT_SAMPLES],
            }
        )
        return result

    finite_sig = sig[finite]
    p05, p50, p95 = [float(x) for x in np.percentile(finite_sig, [5.0, 50.0, 95.0])]
    robust_range = float(p95 - p05)
    if robust_range < float(result["signal_only_f0_min_robust_range"]):
        result.update(
            {
                "signal_only_f0_status": "insufficient",
                "signal_only_f0_warning": "insufficient_robust_signal_range",
                "signal_only_f0_flags": [FLAG_INSUFFICIENT_RANGE],
                "signal_only_f0_p05": p05,
                "signal_only_f0_p50": p50,
                "signal_only_f0_p95": p95,
                "signal_only_f0_range_p95_p05": robust_range,
            }
        )
        return result

    duration, dt = _duration_and_dt(sig, time)
    min_samples = int(result["signal_only_f0_min_window_samples"])
    max_fraction = float(result["signal_only_f0_max_window_fraction"])
    window, window_sec = _window_samples(
        n=sig.size,
        duration_sec=duration,
        sample_interval_sec=dt,
        fraction=float(result["signal_only_f0_window_fraction"]),
        requested_sec=result["signal_only_f0_window_sec_requested"],
        min_samples=min_samples,
        max_fraction=max_fraction,
    )
    smooth_window, smooth_sec = _window_samples(
        n=sig.size,
        duration_sec=duration,
        sample_interval_sec=dt,
        fraction=float(result["signal_only_f0_smoothing_window_fraction"]),
        requested_sec=result["signal_only_f0_smoothing_window_sec_requested"],
        min_samples=max(3, min_samples // 2),
        max_fraction=max_fraction,
    )
    result.update(
        {
            "signal_only_f0_window_samples": int(window),
            "signal_only_f0_window_sec_actual": window_sec,
            "signal_only_f0_smoothing_window_samples": int(smooth_window),
            "signal_only_f0_smoothing_window_sec_actual": smooth_sec,
        }
    )
    high_state_present = _state_flag_present(signal_state, "SIGNAL_HIGH_STATE_CANDIDATE")
    partial_high_state_present = _state_flag_present(signal_state, "SIGNAL_PARTIAL_HIGH_STATE_CANDIDATE")
    edge_high_state_present = _state_flag_present(signal_state, "SIGNAL_EDGE_HIGH_STATE_CANDIDATE")

    rolling_candidate = _lower_quantile_envelope(
        sig,
        window=window,
        quantile=float(result["signal_only_f0_low_quantile"]),
    )
    rolling_candidate = _moving_average_reflect(rolling_candidate, smooth_window)
    smoothed_signal = _moving_average_reflect(sig, smooth_window)
    candidate, state_meta, state_flags = _build_state_aware_candidate(
        signal=sig,
        raw_candidate=rolling_candidate,
        smoothed_signal=smoothed_signal,
        p05=p05,
        p50=p50,
        robust_range=robust_range,
        window=window,
        low_quantile=float(result["signal_only_f0_low_quantile"]),
        signal_state=signal_state,
        high_state_present=high_state_present,
        partial_high_state_present=partial_high_state_present,
        edge_high_state_present=edge_high_state_present,
        result=result,
        sample_interval_sec=dt,
    )
    result.update(state_meta)
    if result["signal_only_f0_state_aware_used"]:
        result["signal_only_f0_method"] = "state_aware_lower_envelope"
    pre_cap_finite = np.isfinite(candidate) & finite
    above_fraction_pre_cap = (
        float(
            np.sum(candidate[pre_cap_finite] > sig[pre_cap_finite])
            / max(1, np.sum(pre_cap_finite))
        )
        if np.any(pre_cap_finite)
        else 0.0
    )
    # Conservative lower-envelope diagnostic: do not allow the candidate to sit
    # above the observed signal at finite samples.
    candidate = np.where(np.isfinite(sig) & np.isfinite(candidate), np.minimum(candidate, sig), candidate)
    context_mode = str(result["signal_only_f0_high_state_context_mode"]).strip().lower()
    if context_mode not in {"none", "contextual_cap"}:
        context_mode = "contextual_cap"
        result["signal_only_f0_high_state_context_mode"] = context_mode
    context_cap = None
    context_applied = False
    if context_mode == "contextual_cap" and (
        high_state_present or partial_high_state_present or edge_high_state_present
    ):
        # Without epoch masks, use scalar state diagnostics only as context and
        # avoid allowing the diagnostic F0 candidate to simply chase high states.
        context_cap = float(p05 + 0.50 * robust_range)
        before_cap = candidate.copy()
        candidate = np.where(np.isfinite(candidate), np.minimum(candidate, context_cap), candidate)
        context_applied = bool(
            np.any(np.isfinite(before_cap) & np.isfinite(candidate) & (candidate < before_cap))
        )

    cand_finite = np.isfinite(candidate) & finite
    support_fraction = float(np.sum(cand_finite) / max(1, sig.size))
    residual = sig[cand_finite] - candidate[cand_finite]
    cand_vals = candidate[cand_finite]
    sig_vals = sig[cand_finite]
    low_state_support_fraction = float(np.sum(sig_vals <= p50) / max(1, sig.size)) if sig_vals.size else 0.0
    above_fraction = float(np.sum(candidate[cand_finite] > sig[cand_finite]) / max(1, np.sum(cand_finite))) if np.any(cand_finite) else 0.0
    below_fraction = float(np.sum(candidate[cand_finite] <= sig[cand_finite]) / max(1, np.sum(cand_finite))) if np.any(cand_finite) else 0.0
    cand_p05, cand_p50, cand_p95 = [float(x) for x in np.percentile(cand_vals, [5.0, 50.0, 95.0])] if cand_vals.size else (None, None, None)
    cand_range = float(cand_p95 - cand_p05) if cand_p05 is not None and cand_p95 is not None else None
    range_ratio = float(cand_range / robust_range) if cand_range is not None and robust_range > 0 else None
    corr = None
    if cand_vals.size >= 3 and np.nanstd(cand_vals) > 0 and np.nanstd(sig_vals) > 0:
        corr = float(np.corrcoef(cand_vals, sig_vals)[0, 1])
    tracking_score = range_ratio
    res_p05, res_p50, res_p95 = [float(x) for x in np.percentile(residual, [5.0, 50.0, 95.0])] if residual.size else (None, None, None)

    flags = [FLAG_AVAILABLE]
    flags.extend(state_flags)
    if high_state_present:
        flags.append(FLAG_HIGH_STATE)
    if partial_high_state_present:
        flags.append(FLAG_PARTIAL_HIGH_STATE)
    if edge_high_state_present:
        flags.append(FLAG_EDGE_HIGH_STATE)

    low_support = support_fraction < float(result["signal_only_f0_min_coverage_fraction"])
    excessive_tracking = bool(
        tracking_score is not None
        and tracking_score > float(result["signal_only_f0_max_tracking_fraction"])
    )
    excessive_above = above_fraction_pre_cap > float(result["signal_only_f0_max_above_signal_fraction"])
    large_anchor_gap = FLAG_LARGE_ANCHOR_GAP in flags
    has_state_context = bool(
        high_state_present or partial_high_state_present or edge_high_state_present
    )
    insufficient_anchoring = (
        FLAG_INSUFFICIENT_LOW_SUPPORT in flags or FLAG_INSUFFICIENT_ANCHORS in flags
    )
    excessive_edge_extrapolation = bool(
        result.get("signal_only_f0_extrapolated_fraction", 0.0)
        > float(result["signal_only_f0_max_edge_extrapolation_fraction"])
    )
    try:
        max_edge_sec = result.get("signal_only_f0_max_edge_extrapolation_sec")
        if max_edge_sec is not None and dt is not None and dt > 0:
            max_edge_samples = float(max_edge_sec) / float(dt)
            edge_fraction_limit = max_edge_samples / max(1, sig.size)
            excessive_edge_extrapolation = excessive_edge_extrapolation or (
                result.get("signal_only_f0_extrapolated_fraction", 0.0) > edge_fraction_limit
            )
    except Exception:
        pass
    if low_support:
        flags.append(FLAG_LOW_SUPPORT)
    if excessive_tracking:
        flags.append(FLAG_EXCESSIVE_TRACKING)
    if excessive_above:
        flags.append(FLAG_ABOVE_SIGNAL_EXCESSIVE)

    if insufficient_anchoring and (high_state_present or partial_high_state_present or edge_high_state_present):
        viability = VIABILITY_HARD_INSPECT
        confidence = CONFIDENCE_LOW
    elif low_support:
        viability = VIABILITY_HARD_INSPECT
        confidence = CONFIDENCE_LOW
    elif excessive_above:
        viability = VIABILITY_HARD_INSPECT
        confidence = CONFIDENCE_LOW
    elif excessive_tracking:
        viability = VIABILITY_CONTEXTUAL
        confidence = CONFIDENCE_LOW
    elif large_anchor_gap or excessive_edge_extrapolation:
        viability = VIABILITY_CONTEXTUAL
        confidence = CONFIDENCE_LOW
    elif high_state_present or partial_high_state_present or edge_high_state_present:
        viability = VIABILITY_CONTEXTUAL
        confidence = CONFIDENCE_MEDIUM if result["signal_only_f0_state_aware_used"] else CONFIDENCE_LOW
    else:
        viability = VIABILITY_VIABLE
        confidence = CONFIDENCE_HIGH if support_fraction > 0.95 and not excessive_tracking else CONFIDENCE_MEDIUM

    viability, confidence, flags = _calibrate_candidate_confidence(
        result=result,
        flags=flags,
        viability=viability,
        confidence=confidence,
        has_state_context=has_state_context,
        large_anchor_gap=large_anchor_gap,
    )
    if viability == VIABILITY_HARD_INSPECT:
        flags.append(FLAG_HARD_INSPECT)
    elif viability == VIABILITY_CONTEXTUAL:
        flags.append(FLAG_CONTEXTUAL)
    elif viability == VIABILITY_VIABLE:
        flags.append(FLAG_VIABLE)

    result.update(
        {
            "signal_only_f0_candidate_available": True,
            "signal_only_f0_status": "ok",
            "signal_only_f0_warning": "",
            "signal_only_f0_candidate_viability": viability,
            "signal_only_f0_candidate_confidence": confidence,
            "signal_only_f0_support_fraction": support_fraction,
            "signal_only_f0_low_state_support_fraction": low_state_support_fraction,
            "signal_only_f0_high_state_context_cap": context_cap,
            "signal_only_f0_high_state_context_applied": bool(context_applied),
            "signal_only_f0_to_signal_corr": corr,
            "signal_only_f0_to_signal_range_ratio": range_ratio,
            "signal_only_f0_below_signal_fraction": below_fraction,
            "signal_only_f0_above_signal_fraction_pre_cap": above_fraction_pre_cap,
            "signal_only_f0_above_signal_fraction": above_fraction,
            "signal_only_f0_tracking_score": tracking_score,
            "signal_only_f0_residual_p05": res_p05,
            "signal_only_f0_residual_p50": res_p50,
            "signal_only_f0_residual_p95": res_p95,
            "signal_only_f0_p05": cand_p05,
            "signal_only_f0_p50": cand_p50,
            "signal_only_f0_p95": cand_p95,
            "signal_only_f0_range_p95_p05": cand_range,
            "signal_only_f0_flags": list(dict.fromkeys(flags)),
            "signal_only_f0_candidate": candidate,
        }
    )
    return result


def summarize_signal_only_f0_candidates(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize signal-only F0 candidate diagnostics for qc_summary.json."""

    def _count_values(key: str) -> dict[str, int]:
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

    def _flag_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in records:
            flags = rec.get("signal_only_f0_flags", [])
            if isinstance(flags, str):
                flags = [x for x in flags.split(";") if x]
            if isinstance(flags, (list, tuple)):
                for flag in flags:
                    flag_s = str(flag).strip()
                    if flag_s:
                        counts[flag_s] = counts.get(flag_s, 0) + 1
        return {k: int(v) for k, v in sorted(counts.items())}

    def _numeric_summary(key: str) -> dict[str, float | None]:
        vals = []
        for rec in records:
            try:
                val = float(rec.get(key, float("nan")))
            except Exception:
                continue
            if np.isfinite(val):
                vals.append(val)
        if not vals:
            return {"median": None, "p25": None, "p75": None}
        arr = np.asarray(vals, dtype=float)
        return {
            "median": float(np.percentile(arr, 50.0)),
            "p25": float(np.percentile(arr, 25.0)),
            "p75": float(np.percentile(arr, 75.0)),
        }

    return {
        "roi_chunk_signal_only_f0_count": int(len(records)),
        "signal_only_f0_candidate_available_counts": _count_values(
            "signal_only_f0_candidate_available"
        ),
        "signal_only_f0_candidate_viability_counts": _count_values(
            "signal_only_f0_candidate_viability"
        ),
        "signal_only_f0_candidate_confidence_counts": _count_values(
            "signal_only_f0_candidate_confidence"
        ),
        "signal_only_f0_flag_counts": _flag_counts(),
        "signal_only_f0_state_aware_used_counts": _count_values(
            "signal_only_f0_state_aware_used"
        ),
        "signal_only_f0_anchor_status_counts": _count_values(
            "signal_only_f0_anchor_status"
        ),
        "signal_only_f0_edge_extrapolation_mode_counts": _count_values(
            "signal_only_f0_edge_extrapolation_mode"
        ),
        "signal_only_f0_support_fraction": _numeric_summary("signal_only_f0_support_fraction"),
        "signal_only_f0_low_state_support_fraction": _numeric_summary(
            "signal_only_f0_low_state_support_fraction"
        ),
        "signal_only_f0_to_signal_range_ratio": _numeric_summary(
            "signal_only_f0_to_signal_range_ratio"
        ),
        "signal_only_f0_above_signal_fraction": _numeric_summary(
            "signal_only_f0_above_signal_fraction"
        ),
        "signal_only_f0_tracking_score": _numeric_summary("signal_only_f0_tracking_score"),
        "signal_only_f0_anchor_count": _numeric_summary("signal_only_f0_anchor_count"),
        "signal_only_f0_low_support_fraction": _numeric_summary(
            "signal_only_f0_low_support_fraction"
        ),
        "signal_only_f0_direct_support_fraction": _numeric_summary(
            "signal_only_f0_direct_support_fraction"
        ),
        "signal_only_f0_interpolated_fraction": _numeric_summary(
            "signal_only_f0_interpolated_fraction"
        ),
        "signal_only_f0_extrapolated_fraction": _numeric_summary(
            "signal_only_f0_extrapolated_fraction"
        ),
        "signal_only_f0_max_anchor_gap_fraction_observed": _numeric_summary(
            "signal_only_f0_max_anchor_gap_fraction_observed"
        ),
    }
