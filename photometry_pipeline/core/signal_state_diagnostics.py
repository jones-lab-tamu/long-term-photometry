"""Diagnostic-only signal-state candidate metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULTS = {
    "signal_state_smoothing_window_fraction": 0.05,
    "signal_state_smoothing_window_sec": None,
    "signal_state_high_quantile": 0.80,
    "signal_state_low_quantile": 0.20,
    "signal_state_min_episode_fraction": 0.20,
    "signal_state_min_episode_sec": 0.0,
    "signal_state_edge_fraction": 0.10,
    "signal_state_variability_window_fraction": 0.05,
    "signal_state_variability_window_sec": None,
    "signal_state_low_variability_quantile": 0.35,
    "signal_state_low_variability_ratio_threshold": 0.75,
    "signal_state_partial_min_high_fraction": 0.10,
    "signal_state_partial_min_longest_fraction": 0.075,
    "signal_state_partial_max_variability_ratio": 0.60,
    "signal_state_partial_min_variability_suppression": 0.35,
    "signal_state_partial_requires_low_variability": True,
    "signal_state_step_window_fraction": 0.03,
    "signal_state_step_window_sec": None,
    "signal_state_step_threshold_robust_z": 3.5,
    "signal_state_min_robust_range": 1e-6,
}

CLASS_ORDINARY = "ordinary_dynamic_candidate"
CLASS_SUSTAINED = "candidate_sustained_high_state"
CLASS_MIXED = "candidate_mixed_dynamic_high_state"
CLASS_EDGE = "candidate_edge_high_state"
CLASS_UNCERTAIN = "uncertain_signal_state"
CLASS_INSUFFICIENT = "insufficient_signal_state_information"

FLAG_HIGH = "SIGNAL_HIGH_STATE_CANDIDATE"
FLAG_PARTIAL_HIGH = "SIGNAL_PARTIAL_HIGH_STATE_CANDIDATE"
FLAG_MIXED = "SIGNAL_MIXED_DYNAMIC_HIGH_STATE_CANDIDATE"
FLAG_EDGE = "SIGNAL_EDGE_HIGH_STATE_CANDIDATE"
FLAG_STARTS_HIGH = "SIGNAL_STARTS_HIGH"
FLAG_ENDS_HIGH = "SIGNAL_ENDS_HIGH"
FLAG_STEP_UP = "SIGNAL_STEP_UP_CANDIDATE"
FLAG_STEP_DOWN = "SIGNAL_STEP_DOWN_CANDIDATE"
FLAG_LOW_VAR = "SIGNAL_LOW_VARIABILITY_HIGH_STATE"
FLAG_INSUFFICIENT_RANGE = "SIGNAL_INSUFFICIENT_RANGE"
FLAG_INSUFFICIENT_SAMPLES = "SIGNAL_INSUFFICIENT_SAMPLES"
FLAG_UNCERTAIN = "SIGNAL_STATE_UNCERTAIN"


def _cfg(config: Mapping[str, Any] | None, key: str) -> Any:
    if isinstance(config, Mapping) and key in config:
        return config[key]
    return DEFAULTS[key]


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


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
    samples = max(1, min(int(samples), int(n)))
    if samples % 2 == 0 and samples < n:
        samples += 1
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


def _true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(np.asarray(mask, dtype=bool)):
        if val and start is None:
            start = idx
        elif (not val) and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, int(mask.size)))
    return spans


def _base_result(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "signal_state_diagnostics_available": False,
        "signal_state_status": "unavailable",
        "signal_state_warning": "",
        "signal_state_candidate_class": CLASS_INSUFFICIENT,
        "signal_state_smoothing_window_fraction": _as_float(
            _cfg(config, "signal_state_smoothing_window_fraction"), 0.05
        ),
        "signal_state_smoothing_window_sec_requested": _cfg(
            config, "signal_state_smoothing_window_sec"
        ),
        "signal_state_smoothing_window_samples": 0,
        "signal_state_smoothing_window_sec_actual": None,
        "signal_state_variability_window_fraction": _as_float(
            _cfg(config, "signal_state_variability_window_fraction"), 0.05
        ),
        "signal_state_variability_window_sec_requested": _cfg(
            config, "signal_state_variability_window_sec"
        ),
        "signal_state_variability_window_samples": 0,
        "signal_state_variability_window_sec_actual": None,
        "signal_state_step_window_fraction": _as_float(
            _cfg(config, "signal_state_step_window_fraction"), 0.03
        ),
        "signal_state_step_window_sec_requested": _cfg(config, "signal_state_step_window_sec"),
        "signal_state_step_window_samples": 0,
        "signal_state_step_window_sec_actual": None,
        "signal_state_high_quantile": _as_float(_cfg(config, "signal_state_high_quantile"), 0.80),
        "signal_state_low_quantile": _as_float(_cfg(config, "signal_state_low_quantile"), 0.20),
        "signal_state_min_episode_fraction": _as_float(
            _cfg(config, "signal_state_min_episode_fraction"), 0.20
        ),
        "signal_state_min_episode_sec": _as_float(_cfg(config, "signal_state_min_episode_sec"), 0.0),
        "signal_state_edge_fraction": _as_float(_cfg(config, "signal_state_edge_fraction"), 0.10),
        "signal_state_low_variability_quantile": _as_float(
            _cfg(config, "signal_state_low_variability_quantile"), 0.35
        ),
        "signal_state_low_variability_ratio_threshold": _as_float(
            _cfg(config, "signal_state_low_variability_ratio_threshold"),
            _as_float(_cfg(config, "signal_state_low_variability_quantile"), 0.75),
        ),
        "signal_state_partial_min_high_fraction": _as_float(
            _cfg(config, "signal_state_partial_min_high_fraction"), 0.10
        ),
        "signal_state_partial_min_longest_fraction": _as_float(
            _cfg(config, "signal_state_partial_min_longest_fraction"), 0.075
        ),
        "signal_state_partial_max_variability_ratio": _as_float(
            _cfg(config, "signal_state_partial_max_variability_ratio"), 0.60
        ),
        "signal_state_partial_min_variability_suppression": _as_float(
            _cfg(config, "signal_state_partial_min_variability_suppression"), 0.35
        ),
        "signal_state_partial_requires_low_variability": _as_bool(
            _cfg(config, "signal_state_partial_requires_low_variability"), True
        ),
        "signal_state_step_threshold_robust_z": _as_float(
            _cfg(config, "signal_state_step_threshold_robust_z"), 3.5
        ),
        "signal_state_min_robust_range": _as_float(
            _cfg(config, "signal_state_min_robust_range"), 1e-6
        ),
        "signal_high_state_candidate_present": False,
        "signal_high_state_fraction": 0.0,
        "signal_longest_high_state_fraction": 0.0,
        "signal_longest_high_state_duration_sec": 0.0,
        "signal_high_state_episode_count": 0,
        "signal_partial_high_state_candidate_present": False,
        "signal_edge_high_state_present": False,
        "signal_start_high_state_candidate": False,
        "signal_end_high_state_candidate": False,
        "signal_step_transition_count": 0,
        "signal_step_up_count": 0,
        "signal_step_down_count": 0,
        "signal_max_step_robust_z": 0.0,
        "signal_step_like_transition_present": False,
        "signal_variability_suppression_score": 0.0,
        "signal_high_state_local_variability_median": None,
        "signal_low_state_local_variability_median": None,
        "signal_high_to_low_variability_ratio": None,
        "signal_state_p05": None,
        "signal_state_p50": None,
        "signal_state_p95": None,
        "signal_state_robust_range": None,
        "signal_state_high_threshold": None,
        "signal_state_low_threshold": None,
        "signal_state_flags": [],
    }


def compute_signal_state_diagnostics(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute scalar signal-only diagnostics for sustained high-state candidates."""
    result = _base_result(config)
    sig = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(sig)
    if sig.size < 10 or int(np.sum(finite)) < 10:
        result.update(
            {
                "signal_state_status": "insufficient",
                "signal_state_warning": "insufficient_finite_signal_samples",
                "signal_state_flags": [FLAG_INSUFFICIENT_SAMPLES],
            }
        )
        return result

    finite_sig = sig[finite]
    idx = np.flatnonzero(finite)
    if time is not None:
        t = np.asarray(time, dtype=float).reshape(-1)
        finite_time = t[finite] if t.shape == sig.shape else None
    else:
        finite_time = None
    if finite_time is not None and finite_time.size >= 2 and np.all(np.isfinite(finite_time)):
        diffs = np.diff(finite_time)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        dt = float(np.median(diffs)) if diffs.size else None
        duration = float(finite_time[-1] - finite_time[0] + (dt or 0.0))
    else:
        dt = None
        duration = None

    p05, p50, p95 = [float(x) for x in np.percentile(finite_sig, [5.0, 50.0, 95.0])]
    robust_range = float(p95 - p05)
    result.update(
        {
            "signal_state_p05": p05,
            "signal_state_p50": p50,
            "signal_state_p95": p95,
            "signal_state_robust_range": robust_range,
        }
    )
    if robust_range < result["signal_state_min_robust_range"]:
        result.update(
            {
                "signal_state_status": "insufficient",
                "signal_state_warning": "insufficient_robust_signal_range",
                "signal_state_flags": [FLAG_INSUFFICIENT_RANGE],
            }
        )
        return result

    n = int(sig.size)
    smooth_w, smooth_sec = _window_samples(
        n=n,
        duration_sec=duration,
        sample_interval_sec=dt,
        fraction=result["signal_state_smoothing_window_fraction"],
        requested_sec=result["signal_state_smoothing_window_sec_requested"],
    )
    var_w, var_sec = _window_samples(
        n=n,
        duration_sec=duration,
        sample_interval_sec=dt,
        fraction=result["signal_state_variability_window_fraction"],
        requested_sec=result["signal_state_variability_window_sec_requested"],
    )
    step_w, step_sec = _window_samples(
        n=n,
        duration_sec=duration,
        sample_interval_sec=dt,
        fraction=result["signal_state_step_window_fraction"],
        requested_sec=result["signal_state_step_window_sec_requested"],
    )
    result.update(
        {
            "signal_state_smoothing_window_samples": int(smooth_w),
            "signal_state_smoothing_window_sec_actual": smooth_sec,
            "signal_state_variability_window_samples": int(var_w),
            "signal_state_variability_window_sec_actual": var_sec,
            "signal_state_step_window_samples": int(step_w),
            "signal_state_step_window_sec_actual": step_sec,
        }
    )

    smooth = _moving_average_reflect(sig, smooth_w)
    smooth_finite = smooth[np.isfinite(smooth)]
    high_q = min(max(float(result["signal_state_high_quantile"]), 0.5), 0.99)
    low_q = min(max(float(result["signal_state_low_quantile"]), 0.01), 0.5)
    high_thr = float(p05 + high_q * robust_range)
    low_thr = float(p05 + low_q * robust_range)
    result["signal_state_high_threshold"] = high_thr
    result["signal_state_low_threshold"] = low_thr
    high_mask = np.isfinite(smooth) & (smooth >= high_thr)
    spans = _true_spans(high_mask)
    high_count = int(np.sum(high_mask))
    high_fraction = float(high_count / max(1, n))
    longest = max((end - start for start, end in spans), default=0)
    longest_fraction = float(longest / max(1, n))
    longest_duration = float(longest * dt) if dt is not None else float(longest)
    min_episode_fraction = float(result["signal_state_min_episode_fraction"])
    min_episode_sec = float(result["signal_state_min_episode_sec"])
    duration_ok = True if min_episode_sec <= 0 else longest_duration >= min_episode_sec
    high_present = bool(longest_fraction >= min_episode_fraction and duration_ok)

    edge_n = max(1, int(round(n * float(result["signal_state_edge_fraction"]))))
    raw_start_high = any(start < edge_n for start, end in spans if (end - start) > 0)
    raw_end_high = any(end > (n - edge_n) for start, end in spans if (end - start) > 0)
    start_high = bool(high_present and raw_start_high)
    end_high = bool(high_present and raw_end_high)
    edge_present = bool(high_present and (start_high or end_high))

    local_var = _moving_average_reflect(np.abs(sig - smooth), var_w)
    high_var = local_var[high_mask & np.isfinite(local_var)]
    low_mask = np.isfinite(smooth) & (smooth <= low_thr)
    low_var = local_var[low_mask & np.isfinite(local_var)]
    high_var_med = float(np.median(high_var)) if high_var.size else None
    low_var_med = float(np.median(low_var)) if low_var.size else None
    if high_var_med is not None and low_var_med is not None and low_var_med > 0:
        var_ratio = float(high_var_med / low_var_med)
        suppression = float(max(0.0, 1.0 - min(var_ratio, 1.0)))
    else:
        var_ratio = None
        suppression = 0.0
    low_var_threshold = float(result["signal_state_low_variability_ratio_threshold"])
    low_var_candidate = bool(var_ratio is not None and var_ratio <= low_var_threshold)
    partial_requires_low_var = bool(result["signal_state_partial_requires_low_variability"])
    partial_low_var_ok = (not partial_requires_low_var) or low_var_candidate
    partial_high_present = bool(
        (not high_present)
        and high_fraction >= float(result["signal_state_partial_min_high_fraction"])
        and longest_fraction >= float(result["signal_state_partial_min_longest_fraction"])
        and var_ratio is not None
        and var_ratio <= float(result["signal_state_partial_max_variability_ratio"])
        and suppression >= float(result["signal_state_partial_min_variability_suppression"])
        and partial_low_var_ok
    )

    if step_w >= 1 and smooth.size > step_w:
        delta = smooth[step_w:] - smooth[:-step_w]
        delta = delta[np.isfinite(delta)]
    else:
        delta = np.array([], dtype=float)
    if delta.size:
        med_delta = float(np.median(delta))
        mad_delta = float(np.median(np.abs(delta - med_delta)))
        scale = max(1.4826 * mad_delta, robust_range * 0.05, 1e-12)
        z = delta / scale
        abs_z = np.abs(z)
        threshold = float(result["signal_state_step_threshold_robust_z"])
        step_up_mask = z >= threshold
        step_down_mask = z <= -threshold
        step_up = len(_true_spans(step_up_mask))
        step_down = len(_true_spans(step_down_mask))
        max_step = float(np.max(abs_z)) if abs_z.size else 0.0
    else:
        step_up = 0
        step_down = 0
        max_step = 0.0
    step_count = int(step_up + step_down)

    flags: list[str] = []
    if high_present:
        flags.append(FLAG_HIGH)
    if edge_present:
        flags.append(FLAG_EDGE)
    if start_high:
        flags.append(FLAG_STARTS_HIGH)
    if end_high:
        flags.append(FLAG_ENDS_HIGH)
    if step_up > 0:
        flags.append(FLAG_STEP_UP)
    if step_down > 0:
        flags.append(FLAG_STEP_DOWN)
    if low_var_candidate:
        flags.append(FLAG_LOW_VAR)
    if partial_high_present:
        flags.append(FLAG_PARTIAL_HIGH)

    dynamic_portion_present = bool(np.sum(low_mask) / max(1, n) >= 0.10)
    if edge_present:
        cls = CLASS_EDGE
    elif high_present and low_var_candidate and dynamic_portion_present:
        cls = CLASS_MIXED
        flags.append(FLAG_MIXED)
    elif high_present and low_var_candidate:
        cls = CLASS_SUSTAINED
    elif partial_high_present and dynamic_portion_present:
        cls = CLASS_MIXED
        flags.append(FLAG_MIXED)
    elif partial_high_present:
        cls = CLASS_UNCERTAIN
        flags.append(FLAG_UNCERTAIN)
    elif high_present:
        cls = CLASS_UNCERTAIN
        flags.append(FLAG_UNCERTAIN)
    else:
        cls = CLASS_ORDINARY

    result.update(
        {
            "signal_state_diagnostics_available": True,
            "signal_state_status": "ok",
            "signal_state_warning": "",
            "signal_state_candidate_class": cls,
            "signal_high_state_candidate_present": bool(high_present),
            "signal_high_state_fraction": high_fraction,
            "signal_longest_high_state_fraction": longest_fraction,
            "signal_longest_high_state_duration_sec": longest_duration,
            "signal_high_state_episode_count": int(len(spans)),
            "signal_partial_high_state_candidate_present": bool(partial_high_present),
            "signal_edge_high_state_present": bool(edge_present),
            "signal_start_high_state_candidate": bool(start_high),
            "signal_end_high_state_candidate": bool(end_high),
            "signal_step_transition_count": int(step_count),
            "signal_step_up_count": int(step_up),
            "signal_step_down_count": int(step_down),
            "signal_max_step_robust_z": max_step,
            "signal_step_like_transition_present": bool(step_count > 0),
            "signal_variability_suppression_score": suppression,
            "signal_high_state_local_variability_median": high_var_med,
            "signal_low_state_local_variability_median": low_var_med,
            "signal_high_to_low_variability_ratio": var_ratio,
            "signal_state_flags": list(dict.fromkeys(flags)),
        }
    )
    return result


def summarize_signal_state_diagnostics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize signal-state diagnostic records for qc_summary.json."""

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
            flags = rec.get("signal_state_flags", [])
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
        "roi_chunk_signal_state_count": int(len(records)),
        "signal_state_candidate_class_counts": _count_values(
            "signal_state_candidate_class"
        ),
        "signal_high_state_candidate_present_counts": _count_values(
            "signal_high_state_candidate_present"
        ),
        "signal_edge_high_state_present_counts": _count_values(
            "signal_edge_high_state_present"
        ),
        "signal_step_like_transition_present_counts": _count_values(
            "signal_step_like_transition_present"
        ),
        "signal_state_flag_counts": _flag_counts(),
        "signal_high_state_fraction": _numeric_summary("signal_high_state_fraction"),
        "signal_longest_high_state_fraction": _numeric_summary(
            "signal_longest_high_state_fraction"
        ),
        "signal_step_transition_count": _numeric_summary(
            "signal_step_transition_count"
        ),
        "signal_variability_suppression_score": _numeric_summary(
            "signal_variability_suppression_score"
        ),
    }
