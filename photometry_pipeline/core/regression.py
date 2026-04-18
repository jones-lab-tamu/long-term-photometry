import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple
from ..config import Config
from .types import Chunk

def _get_window_indices(center: int, window_samples: int, n_samples: int) -> Optional[Tuple[int, int]]:
    """
    Returns (start, end) such that end - start == window_samples.
    Centered at center. Returns None if window exceeds boundaries [0, n_samples].
    """
    half = window_samples // 2
    start = center - half
    end = start + window_samples
    
    if start < 0 or end > n_samples:
        return None
        
    return start, end

# In fit generation loop:
# for c in centers:
#    indices = _get_window_indices(c, window_samples, n_samples)
#    if indices is None: continue
#    start, end = indices


def _assemble_delta_f_from_fit(sig_raw: np.ndarray, uv_fit: np.ndarray) -> np.ndarray:
    """
    Canonical numerator assembly for phasic mode.

    Contract:
      delta_f = sig_raw - uv_fit
    """
    sig_arr = np.asarray(sig_raw, dtype=float)
    fit_arr = np.asarray(uv_fit, dtype=float)
    if sig_arr.shape != fit_arr.shape:
        raise ValueError(
            "delta_f assembly shape mismatch: "
            f"sig_raw={sig_arr.shape}, uv_fit={fit_arr.shape}"
        )
    return sig_arr - fit_arr


def _rolling_sum_centered(values: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Centered rolling sum using O(N) cumulative-sum indexing.
    """
    n = int(values.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)

    half_left = int(window_samples // 2)
    half_right = int(window_samples - half_left)

    idx = np.arange(n, dtype=int)
    starts = np.maximum(0, idx - half_left)
    ends = np.minimum(n, idx + half_right)

    csum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return csum[ends] - csum[starts]


def _interp_fill_nearest_finite(values: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by linear interpolation with nearest-value edge extension.
    """
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    x = np.arange(arr.size, dtype=float)
    out = arr.copy()
    out[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    return out


def _freeze_values_over_gated_mask(
    values: np.ndarray,
    gated_mask: np.ndarray,
    trusted_anchor_mask: np.ndarray,
) -> np.ndarray:
    """
    Freeze values through gated spans using nearest trusted anchors.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    gated = np.asarray(gated_mask, dtype=bool).reshape(-1)
    anchors = np.asarray(trusted_anchor_mask, dtype=bool).reshape(-1) & np.isfinite(arr)
    if arr.size == 0 or not np.any(gated) or not np.any(anchors):
        return arr.copy()

    idx = np.arange(arr.size, dtype=int)
    prev_idx = np.where(anchors, idx, -1)
    prev_idx = np.maximum.accumulate(prev_idx)
    next_idx = np.where(anchors, idx, arr.size)
    next_idx = np.minimum.accumulate(next_idx[::-1])[::-1]

    out = arr.copy()
    gated_idx = np.where(gated)[0]
    for i in gated_idx:
        left = int(prev_idx[i])
        right = int(next_idx[i])
        if left >= 0:
            out[i] = arr[left]
        elif right < arr.size:
            out[i] = arr[right]
    return out


def _global_fit_params(u_f: np.ndarray, s_f: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[str], float]:
    """
    Returns (slope, intercept) from finite filtered samples, with DD code on failure.
    """
    m = np.isfinite(u_f) & np.isfinite(s_f)
    u_w = u_f[m]
    s_w = s_f[m]
    if len(u_w) < 2 or len(s_w) < 2:
        return None, "DD1", float("nan")

    cov = np.cov(u_w, s_w, bias=True)
    var_u = float(cov[0, 0])
    cov_us = float(cov[0, 1])
    if (not np.isfinite(var_u)) or (var_u <= 1e-12):
        return None, "DD2", var_u

    slope = cov_us / var_u
    intercept = float(np.mean(s_w) - slope * np.mean(u_w))
    return (float(slope), intercept), None, var_u


_DYNAMIC_FIT_MODES = {
    "rolling_filtered_to_raw",
    "rolling_filtered_to_filtered",
    "global_linear_regression",
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
}

_DYNAMIC_FIT_MODE_ALIASES = {
    # Backward-compatible alias retained to avoid breaking older configs/artifacts.
    "rolling_local_regression": "rolling_filtered_to_raw",
}
_BLEACH_CORRECTION_MODES = {
    "none",
    "single_exponential",
    "double_exponential",
}
_DOUBLE_BLEACH_MIN_TAU_RATIO = 1.8


def _normalize_signal_excursion_polarity(mode_raw: str) -> str:
    mode = str(mode_raw or "positive").strip().lower()
    if mode not in {"positive", "negative", "both"}:
        return "positive"
    return mode


def _normalize_bleach_correction_mode(mode_raw: Any) -> str:
    mode = str(mode_raw or "none").strip().lower()
    if mode not in _BLEACH_CORRECTION_MODES:
        return "none"
    return mode


def _single_exponential_components(
    n_samples: int,
    sample_rate_hz: float,
    *,
    amplitude: float,
    tau_sec: float,
    offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(n_samples), dtype=float) / max(float(sample_rate_hz), 1e-9)
    decay_component = float(amplitude) * np.exp(-t / float(tau_sec))
    full_fit = decay_component + float(offset)
    return full_fit, decay_component


def _double_exponential_components(
    n_samples: int,
    sample_rate_hz: float,
    *,
    amplitude_fast: float,
    tau_fast_sec: float,
    amplitude_slow: float,
    tau_slow_sec: float,
    offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    tau_fast = float(tau_fast_sec)
    tau_slow = float(tau_slow_sec)
    amp_fast = float(amplitude_fast)
    amp_slow = float(amplitude_slow)
    if tau_fast > tau_slow:
        tau_fast, tau_slow = tau_slow, tau_fast
        amp_fast, amp_slow = amp_slow, amp_fast
    t = np.arange(int(n_samples), dtype=float) / max(float(sample_rate_hz), 1e-9)
    decay_component = amp_fast * np.exp(-t / tau_fast) + amp_slow * np.exp(-t / tau_slow)
    full_fit = decay_component + float(offset)
    return full_fit, decay_component


def _resolve_bleach_tau_bounds(
    n_samples: int,
    sample_rate_hz: float,
) -> tuple[tuple[float, float] | None, str]:
    fs_hz = float(sample_rate_hz)
    if (not np.isfinite(fs_hz)) or fs_hz <= 0.0:
        return None, "invalid_sample_rate"
    duration_sec = float((int(n_samples) - 1) / fs_hz) if int(n_samples) > 1 else 0.0
    if duration_sec <= 0.0:
        return None, "nonpositive_duration"
    tau_min = max(2.0 / fs_hz, min(2.0, 0.02 * duration_sec))
    tau_max = max(tau_min * 2.0, duration_sec * 8.0)
    if not (np.isfinite(tau_min) and np.isfinite(tau_max) and tau_max > tau_min):
        return None, "invalid_tau_range"
    return (float(tau_min), float(tau_max)), ""


def _fit_single_exponential_with_offset(
    trace: np.ndarray,
    sample_rate_hz: float,
) -> dict:
    """
    Fit y(t) ~= offset + amplitude*exp(-t/tau) using a tau grid + linear least squares.

    Returns a metadata dict. On failure, fit_succeeded=False and fit_failure_reason
    is populated; callers must fall back to no bleach correction for that trace.
    """
    y = np.asarray(trace, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    n_finite = int(np.sum(finite))
    out = {
        "fit_model": "single_exponential",
        "fit_succeeded": False,
        "fit_failure_reason": "",
        "n_finite_samples": n_finite,
        "amplitude": float("nan"),
        "tau_sec": float("nan"),
        "offset": float("nan"),
        "fit_rmse": float("nan"),
    }
    if y.size < 8 or n_finite < 8:
        out["fit_failure_reason"] = "insufficient_finite_samples"
        return out

    tau_bounds, tau_err = _resolve_bleach_tau_bounds(y.size, sample_rate_hz)
    if tau_bounds is None:
        out["fit_failure_reason"] = tau_err or "invalid_tau_range"
        return out
    fs_hz = float(sample_rate_hz)
    tau_min, tau_max = tau_bounds

    t = np.arange(y.size, dtype=float) / fs_hz
    t_fit = t[finite]
    y_fit = y[finite]
    tau_grid = np.geomspace(tau_min, tau_max, num=40)

    best = None
    for tau in tau_grid:
        e = np.exp(-t_fit / float(tau))
        X = np.column_stack((e, np.ones_like(e)))
        try:
            beta, residuals, rank, _singular = np.linalg.lstsq(X, y_fit, rcond=None)
        except Exception:
            continue
        if int(rank) < 2:
            continue
        amplitude = float(beta[0])
        offset = float(beta[1])
        y_hat = X @ beta
        rmse = float(np.sqrt(np.mean((y_fit - y_hat) ** 2)))
        if not (np.isfinite(amplitude) and np.isfinite(offset) and np.isfinite(rmse)):
            continue
        if best is None or rmse < best["fit_rmse"]:
            best = {
                "amplitude": amplitude,
                "offset": offset,
                "tau_sec": float(tau),
                "fit_rmse": rmse,
            }

    if best is None:
        out["fit_failure_reason"] = "lstsq_fit_failed"
        return out

    out.update(best)
    out["fit_succeeded"] = True
    out["fit_failure_reason"] = ""
    return out


def _fit_double_exponential_with_offset(
    trace: np.ndarray,
    sample_rate_hz: float,
) -> dict:
    """
    Fit y(t) ~= offset + a_fast*exp(-t/tau_fast) + a_slow*exp(-t/tau_slow)
    using constrained tau-pair grid search + linear least squares.

    Constraints:
      - tau_fast > 0, tau_slow > 0
      - tau_fast < tau_slow
      - tau_slow / tau_fast >= _DOUBLE_BLEACH_MIN_TAU_RATIO
      - amplitudes constrained to be non-negative
    """
    y = np.asarray(trace, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    n_finite = int(np.sum(finite))
    out = {
        "fit_model": "double_exponential",
        "fit_succeeded": False,
        "fit_failure_reason": "",
        "n_finite_samples": n_finite,
        "amplitude_fast": float("nan"),
        "tau_fast_sec": float("nan"),
        "amplitude_slow": float("nan"),
        "tau_slow_sec": float("nan"),
        "offset": float("nan"),
        "fit_rmse": float("nan"),
    }
    if y.size < 12 or n_finite < 12:
        out["fit_failure_reason"] = "insufficient_finite_samples"
        return out

    tau_bounds, tau_err = _resolve_bleach_tau_bounds(y.size, sample_rate_hz)
    if tau_bounds is None:
        out["fit_failure_reason"] = tau_err or "invalid_tau_range"
        return out
    fs_hz = float(sample_rate_hz)
    tau_min, tau_max = tau_bounds

    t = np.arange(y.size, dtype=float) / fs_hz
    t_fit = t[finite]
    y_fit = y[finite]
    tau_grid = np.geomspace(tau_min, tau_max, num=28)

    best = None
    for fast_idx, tau_fast in enumerate(tau_grid[:-1]):
        for tau_slow in tau_grid[fast_idx + 1 :]:
            tau_ratio = float(tau_slow) / float(tau_fast)
            if tau_ratio < _DOUBLE_BLEACH_MIN_TAU_RATIO:
                continue
            e_fast = np.exp(-t_fit / float(tau_fast))
            e_slow = np.exp(-t_fit / float(tau_slow))
            X = np.column_stack((e_fast, e_slow, np.ones_like(e_fast)))
            try:
                beta, _residuals, rank, _singular = np.linalg.lstsq(X, y_fit, rcond=None)
            except Exception:
                continue
            if int(rank) < 3:
                continue
            amp_fast = float(beta[0])
            amp_slow = float(beta[1])
            offset = float(beta[2])
            if (
                (not np.isfinite(amp_fast))
                or (not np.isfinite(amp_slow))
                or (not np.isfinite(offset))
                or amp_fast < 0.0
                or amp_slow < 0.0
            ):
                continue
            y_hat = X @ beta
            rmse = float(np.sqrt(np.mean((y_fit - y_hat) ** 2)))
            if not np.isfinite(rmse):
                continue
            if best is None or rmse < best["fit_rmse"]:
                best = {
                    "amplitude_fast": amp_fast,
                    "tau_fast_sec": float(tau_fast),
                    "amplitude_slow": amp_slow,
                    "tau_slow_sec": float(tau_slow),
                    "offset": offset,
                    "fit_rmse": rmse,
                    "tau_ratio": tau_ratio,
                }

    if best is None:
        out["fit_failure_reason"] = "lstsq_fit_failed_or_constrained_out"
        return out
    if float(best.get("tau_ratio", 0.0)) < _DOUBLE_BLEACH_MIN_TAU_RATIO:
        out["fit_failure_reason"] = "degenerate_tau_separation"
        return out

    out.update(
        {
            "amplitude_fast": float(best["amplitude_fast"]),
            "tau_fast_sec": float(best["tau_fast_sec"]),
            "amplitude_slow": float(best["amplitude_slow"]),
            "tau_slow_sec": float(best["tau_slow_sec"]),
            "offset": float(best["offset"]),
            "fit_rmse": float(best["fit_rmse"]),
        }
    )
    out["fit_succeeded"] = True
    out["fit_failure_reason"] = ""
    return out


def _bleach_fit_components_from_meta(
    n_samples: int,
    sample_rate_hz: float,
    fit_meta: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(fit_meta, dict) or not bool(fit_meta.get("fit_succeeded", False)):
        return None, None
    model = str(fit_meta.get("fit_model", "")).strip().lower()
    if not model:
        if "tau_fast_sec" in fit_meta or "tau_slow_sec" in fit_meta:
            model = "double_exponential"
        else:
            model = "single_exponential"
    try:
        if model == "double_exponential":
            return _double_exponential_components(
                n_samples,
                sample_rate_hz,
                amplitude_fast=float(fit_meta.get("amplitude_fast")),
                tau_fast_sec=float(fit_meta.get("tau_fast_sec")),
                amplitude_slow=float(fit_meta.get("amplitude_slow")),
                tau_slow_sec=float(fit_meta.get("tau_slow_sec")),
                offset=float(fit_meta.get("offset")),
            )
        return _single_exponential_components(
            n_samples,
            sample_rate_hz,
            amplitude=float(fit_meta.get("amplitude")),
            tau_sec=float(fit_meta.get("tau_sec")),
            offset=float(fit_meta.get("offset")),
        )
    except Exception:
        return None, None


def _apply_bleach_correction_to_chunk_inputs(chunk: Chunk, config: Config) -> dict:
    """
    Optional correction-stage bleach correction prep for dynamic-fit inputs.

    v1 scope:
      - mode: none | single_exponential | double_exponential
      - target: both signal and isosbestic traces, independently
      - correction removes only the fitted exponential component (preserves offset)
    """
    mode_requested = str(getattr(config, "bleach_correction_mode", "none"))
    mode_resolved = _normalize_bleach_correction_mode(mode_requested)

    sig_raw = np.asarray(chunk.sig_raw, dtype=float)
    uv_raw = np.asarray(chunk.uv_raw, dtype=float)
    sig_corr = np.asarray(sig_raw, dtype=float).copy()
    uv_corr = np.asarray(uv_raw, dtype=float).copy()
    sig_decay_removed = np.zeros_like(sig_corr, dtype=float)
    uv_decay_removed = np.zeros_like(uv_corr, dtype=float)
    per_roi: Dict[str, Dict[str, Any]] = {}
    applied_any = False

    if mode_resolved not in {"single_exponential", "double_exponential"}:
        return {
            "mode_requested": mode_requested,
            "mode_resolved": mode_resolved,
            "target": "signal_and_isosbestic_independent",
            "applied_any": False,
            "sig_raw_corrected": sig_corr,
            "uv_raw_corrected": uv_corr,
            "sig_filt_corrected": (
                None
                if chunk.sig_filt is None
                else np.asarray(chunk.sig_filt, dtype=float).copy()
            ),
            "uv_filt_corrected": (
                None
                if chunk.uv_filt is None
                else np.asarray(chunk.uv_filt, dtype=float).copy()
            ),
            "sig_decay_removed": sig_decay_removed,
            "uv_decay_removed": uv_decay_removed,
            "per_roi": per_roi,
        }

    fs_hz = float(getattr(chunk, "fs_hz", np.nan))
    n_samples, n_rois = int(sig_raw.shape[0]), int(sig_raw.shape[1])
    fit_fn = (
        _fit_double_exponential_with_offset
        if mode_resolved == "double_exponential"
        else _fit_single_exponential_with_offset
    )
    for r_idx in range(n_rois):
        roi_name = (
            str(chunk.channel_names[r_idx])
            if r_idx < len(getattr(chunk, "channel_names", []))
            else f"roi_{r_idx}"
        )
        sig_fit_meta = dict(fit_fn(sig_raw[:, r_idx], fs_hz))
        uv_fit_meta = dict(fit_fn(uv_raw[:, r_idx], fs_hz))

        sig_fit, sig_decay = _bleach_fit_components_from_meta(n_samples, fs_hz, sig_fit_meta)
        uv_fit, uv_decay = _bleach_fit_components_from_meta(n_samples, fs_hz, uv_fit_meta)
        sig_applied = sig_fit is not None and sig_decay is not None
        uv_applied = uv_fit is not None and uv_decay is not None

        if bool(sig_fit_meta.get("fit_succeeded", False)) and (not sig_applied):
            sig_fit_meta["fit_succeeded"] = False
            sig_fit_meta["fit_failure_reason"] = str(
                sig_fit_meta.get("fit_failure_reason", "") or "fit_components_invalid"
            )
        if bool(uv_fit_meta.get("fit_succeeded", False)) and (not uv_applied):
            uv_fit_meta["fit_succeeded"] = False
            uv_fit_meta["fit_failure_reason"] = str(
                uv_fit_meta.get("fit_failure_reason", "") or "fit_components_invalid"
            )

        if sig_applied:
            sig_decay = np.asarray(sig_decay, dtype=float)
            sig_decay_removed[:, r_idx] = sig_decay
            sig_corr[:, r_idx] = sig_raw[:, r_idx] - sig_decay
            applied_any = True

        if uv_applied:
            uv_decay = np.asarray(uv_decay, dtype=float)
            uv_decay_removed[:, r_idx] = uv_decay
            uv_corr[:, r_idx] = uv_raw[:, r_idx] - uv_decay
            applied_any = True

        per_roi[roi_name] = {
            "mode": mode_resolved,
            "target": "signal_and_isosbestic_independent",
            "signal": dict(sig_fit_meta),
            "isosbestic": dict(uv_fit_meta),
            "signal_applied": sig_applied,
            "isosbestic_applied": uv_applied,
        }

    sig_filt_corr = None
    if chunk.sig_filt is not None:
        sig_filt_corr = np.asarray(chunk.sig_filt, dtype=float).copy() - sig_decay_removed
    uv_filt_corr = None
    if chunk.uv_filt is not None:
        uv_filt_corr = np.asarray(chunk.uv_filt, dtype=float).copy() - uv_decay_removed

    return {
        "mode_requested": mode_requested,
        "mode_resolved": mode_resolved,
        "target": "signal_and_isosbestic_independent",
        "applied_any": bool(applied_any),
        "sig_raw_corrected": sig_corr,
        "uv_raw_corrected": uv_corr,
        "sig_filt_corrected": sig_filt_corr,
        "uv_filt_corrected": uv_filt_corr,
        "sig_decay_removed": sig_decay_removed,
        "uv_decay_removed": uv_decay_removed,
        "per_roi": per_roi,
    }


def _residual_excursion_candidates(
    residual: np.ndarray,
    finite_mask: np.ndarray,
    *,
    center: float,
    robust_scale: float,
    z_thresh: float,
    signal_excursion_polarity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build polarity-aware residual excursion candidates.

    Returns (candidate_union, candidate_upper_tail, candidate_lower_tail) masks.
    """
    finite = np.asarray(finite_mask, dtype=bool).reshape(-1)
    resid = np.asarray(residual, dtype=float).reshape(-1)
    union = np.zeros_like(finite, dtype=bool)
    upper = np.zeros_like(finite, dtype=bool)
    lower = np.zeros_like(finite, dtype=bool)
    if (
        resid.shape != finite.shape
        or (not np.isfinite(robust_scale))
        or robust_scale <= 1e-12
        or (not np.isfinite(z_thresh))
        or z_thresh <= 0.0
    ):
        return union, upper, lower

    polarity = _normalize_signal_excursion_polarity(signal_excursion_polarity)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_signed = (resid - float(center)) / float(robust_scale)
    upper = finite & (z_signed > float(z_thresh))
    lower = finite & (z_signed < -float(z_thresh))
    if polarity == "positive":
        union = upper
    elif polarity == "negative":
        union = lower
    else:
        union = upper | lower
    return union, upper, lower


def _ensure_chunk_metadata(chunk: Chunk) -> None:
    if not hasattr(chunk, "metadata") or chunk.metadata is None:
        chunk.metadata = {}


def _resolve_dynamic_fit_mode(config: Config) -> str:
    requested = getattr(config, "dynamic_fit_mode", "rolling_local_regression")
    mode = str(requested).strip().lower() if requested is not None else "rolling_local_regression"
    if not mode:
        mode = "rolling_local_regression"
    mode = _DYNAMIC_FIT_MODE_ALIASES.get(mode, mode)
    if mode not in _DYNAMIC_FIT_MODES:
        allowed = ", ".join(sorted(_DYNAMIC_FIT_MODES))
        raise ValueError(f"Invalid dynamic_fit_mode: {requested}. Allowed: {allowed}")
    return mode


def _legacy_knobs_not_used_in_engine() -> list[str]:
    return [
        "step_sec",
        "min_valid_windows",
        "r_low",
        "r_high",
        "g_min",
    ]


def _centered_rolling_mean(values: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Finite-aware centered rolling mean for 1D arrays.
    """
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    count = _rolling_sum_centered(mask.astype(float), window_samples)
    total = _rolling_sum_centered(np.where(mask, arr, 0.0), window_samples)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = count > 0.0
    out[valid] = total[valid] / count[valid]
    return out


def _compute_fit_input_baseline(arr2d: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Compute centered moving baseline for each ROI column.
    """
    arr = np.asarray(arr2d, dtype=float)
    baseline = np.full_like(arr, np.nan, dtype=float)
    for idx in range(arr.shape[1]):
        baseline[:, idx] = _centered_rolling_mean(arr[:, idx], window_samples)
    return baseline


def _subtract_fit_input_baseline(arr2d: np.ndarray, window_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract centered moving baseline from each ROI column for fit-input preparation.
    Returns (centered_values, baseline_values).
    """
    arr = np.asarray(arr2d, dtype=float)
    baseline = _compute_fit_input_baseline(arr, window_samples)
    return arr - baseline, baseline


def _prepare_rolling_fit_inputs(
    chunk: Chunk,
    config: Config,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Prepare fit-input traces for rolling modes.
    """
    if chunk.uv_filt is None or chunk.sig_filt is None:
        raise RuntimeError("Filtered traces are required for rolling dynamic fit modes.")

    uv_fit_input = np.asarray(chunk.uv_filt, dtype=float)
    sig_fit_input = np.asarray(chunk.sig_filt, dtype=float)
    apply_baseline = bool(getattr(config, "baseline_subtract_before_fit", False))

    prep_info = {
        "fit_input_domain": "filtered",
        "baseline_subtract_before_fit": apply_baseline,
        "baseline_subtract_applied": False,
        "baseline_subtract_method": "none",
        "baseline_subtract_window_samples": 0,
        "_uv_fit_input_baseline": None,
        "_sig_fit_input_baseline": None,
    }
    if apply_baseline:
        sig_fit_input, sig_fit_baseline = _subtract_fit_input_baseline(sig_fit_input, window_samples)
        uv_fit_input, uv_fit_baseline = _subtract_fit_input_baseline(uv_fit_input, window_samples)
        prep_info.update(
            {
                "baseline_subtract_applied": True,
                "baseline_subtract_method": "centered_rolling_mean",
                "baseline_subtract_window_samples": int(window_samples),
                "_uv_fit_input_baseline": uv_fit_baseline,
                "_sig_fit_input_baseline": sig_fit_baseline,
            }
        )

    return uv_fit_input, sig_fit_input, prep_info


def _centered_rolling_variance(values: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Finite-aware centered rolling population variance for 1D arrays.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(arr)
    count = _rolling_sum_centered(mask.astype(float), window_samples)
    total = _rolling_sum_centered(np.where(mask, arr, 0.0), window_samples)
    total_sq = _rolling_sum_centered(np.where(mask, arr * arr, 0.0), window_samples)

    out = np.full(arr.shape, np.nan, dtype=float)
    valid = count >= 2.0
    if np.any(valid):
        mean = total[valid] / count[valid]
        var = (total_sq[valid] / count[valid]) - (mean * mean)
        out[valid] = np.maximum(var, 0.0)
    return out


def _fit_robust_linear(
    x: np.ndarray,
    y: np.ndarray,
    *,
    use_intercept: bool = True,
) -> tuple[Optional[tuple[float, float]], Optional[str], str]:
    """
    Fit robust global linear model y ~ a*x + b.

    Preferred backend order:
      1) sklearn HuberRegressor
      2) statsmodels RLM (HuberT)
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(finite)) < 3:
        return None, "insufficient_finite_samples", "none"
    x_fit = x_arr[finite]
    y_fit = y_arr[finite]
    if float(np.nanvar(x_fit)) <= 1e-12:
        return None, "near_zero_iso_variance", "none"

    try:
        from sklearn.linear_model import HuberRegressor  # type: ignore

        model = HuberRegressor(fit_intercept=bool(use_intercept), max_iter=200)
        model.fit(x_fit.reshape(-1, 1), y_fit)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_) if bool(use_intercept) else 0.0
        if np.isfinite(slope) and np.isfinite(intercept):
            return (slope, intercept), None, "sklearn_huber"
    except Exception:
        pass

    try:
        import statsmodels.api as sm  # type: ignore

        X = x_fit.reshape(-1, 1)
        if bool(use_intercept):
            X = sm.add_constant(X, has_constant="add")
        model = sm.RLM(y_fit, X, M=sm.robust.norms.HuberT())
        res = model.fit(maxiter=100)
        params = np.asarray(res.params, dtype=float).reshape(-1)
        if bool(use_intercept):
            if params.size < 2:
                return None, "statsmodels_bad_params", "statsmodels_rlm"
            intercept = float(params[0])
            slope = float(params[1])
        else:
            if params.size < 1:
                return None, "statsmodels_bad_params", "statsmodels_rlm"
            slope = float(params[0])
            intercept = 0.0
        if np.isfinite(slope) and np.isfinite(intercept):
            return (slope, intercept), None, "statsmodels_rlm"
        return None, "statsmodels_nonfinite_params", "statsmodels_rlm"
    except Exception:
        return None, "robust_regression_backend_unavailable", "none"


def _unpack_robust_fit_result(
    fit_result: Any,
) -> tuple[Optional[tuple[float, float]], Optional[str], str]:
    """
    Backward-compatible unpacking for robust fit results.
    Accepts:
      - (params, fail_reason) from legacy monkeypatch/tests
      - (params, fail_reason, backend_used) from current implementation
    """
    if not isinstance(fit_result, tuple):
        return None, "invalid_robust_fit_result", "unknown"
    if len(fit_result) >= 3:
        params, fail_reason, backend = fit_result[0], fit_result[1], fit_result[2]
        return params, fail_reason, str(backend)
    if len(fit_result) == 2:
        params, fail_reason = fit_result
        return params, fail_reason, "unknown"
    return None, "invalid_robust_fit_result", "unknown"


def fit_robust_global_event_reject(
    signal_raw: np.ndarray,
    iso_raw: np.ndarray,
    *,
    max_iters: int = 3,
    residual_z_thresh: float = 3.5,
    local_var_window_sec: float | None = None,
    local_var_ratio_thresh: float | None = None,
    min_keep_fraction: float = 0.5,
    sample_rate_hz: float,
    use_intercept: bool = True,
    signal_excursion_polarity: str = "positive",
) -> dict:
    """
    Robust global fit with iterative event-dominated sample rejection.
    """
    sig = np.asarray(signal_raw, dtype=float).reshape(-1)
    iso = np.asarray(iso_raw, dtype=float).reshape(-1)
    if sig.shape != iso.shape:
        raise ValueError(
            "robust_global_event_reject shape mismatch: "
            f"signal={sig.shape}, iso={iso.shape}"
        )
    if sig.size == 0:
        raise RuntimeError("robust_global_event_reject received empty input")

    finite = np.isfinite(sig) & np.isfinite(iso)
    n_finite = int(np.sum(finite))
    if n_finite < 3:
        raise RuntimeError("robust_global_event_reject requires at least 3 finite samples")

    max_iters_i = max(1, int(max_iters))
    z_thresh = float(residual_z_thresh)
    min_keep = float(min_keep_fraction)
    if z_thresh <= 0.0:
        raise ValueError("residual_z_thresh must be > 0")
    if not (0.0 < min_keep <= 1.0):
        raise ValueError("min_keep_fraction must be in (0, 1]")
    polarity = _normalize_signal_excursion_polarity(signal_excursion_polarity)

    use_var_rule = (
        local_var_window_sec is not None
        and local_var_ratio_thresh is not None
        and float(local_var_ratio_thresh) > 0.0
    )
    fs_hz = float(sample_rate_hz)
    if (not np.isfinite(fs_hz)) or fs_hz <= 0.0:
        fs_hz = 1.0
    local_var_window_samples = 0
    var_ratio = None
    if use_var_rule:
        local_var_window_samples = max(3, int(round(float(local_var_window_sec) * fs_hz)))
        if (local_var_window_samples % 2) == 0:
            local_var_window_samples += 1
        var_sig = _centered_rolling_variance(sig, local_var_window_samples)
        var_iso = _centered_rolling_variance(iso, local_var_window_samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_ratio = var_sig / np.maximum(var_iso, 1e-12)

    keep_mask = finite.copy()
    iteration_summaries: List[Dict[str, Any]] = []
    final_slope = np.nan
    final_intercept = np.nan
    final_fit = np.full_like(sig, np.nan, dtype=float)
    robust_backend_used = "unknown"
    stop_reason_final = "max_iters_reached"

    for iter_idx in range(max_iters_i):
        n_keep_before = int(np.sum(keep_mask))
        params, fail_reason, backend_used = _unpack_robust_fit_result(
            _fit_robust_linear(
                iso[keep_mask],
                sig[keep_mask],
                use_intercept=bool(use_intercept),
            )
        )
        robust_backend_used = str(backend_used)
        if params is None:
            raise RuntimeError(
                "robust_global_event_reject robust fit failed: "
                f"{fail_reason or 'unknown'}"
            )

        slope, intercept = params
        fit_all = (slope * iso) + intercept
        residual = sig - fit_all

        resid_finite = residual[finite]
        med = float(np.median(resid_finite))
        mad = float(np.median(np.abs(resid_finite - med)))
        robust_scale = float(1.4826 * mad)

        stop_reason = ""
        candidate_resid = np.zeros_like(finite, dtype=bool)
        candidate_resid_upper_tail = np.zeros_like(finite, dtype=bool)
        candidate_resid_lower_tail = np.zeros_like(finite, dtype=bool)
        candidate_var = np.zeros_like(finite, dtype=bool)
        if robust_scale <= 1e-12 or not np.isfinite(robust_scale):
            stop_reason = "mad_zero_or_nonfinite"
            new_keep = keep_mask.copy()
        else:
            (
                candidate_resid,
                candidate_resid_upper_tail,
                candidate_resid_lower_tail,
            ) = _residual_excursion_candidates(
                residual,
                finite,
                center=med,
                robust_scale=robust_scale,
                z_thresh=z_thresh,
                signal_excursion_polarity=polarity,
            )
            if use_var_rule and var_ratio is not None:
                candidate_var = (
                    finite
                    & np.isfinite(var_ratio)
                    & (var_ratio > float(local_var_ratio_thresh))
                )
            candidate_union = candidate_resid | candidate_var
            new_keep = finite & (~candidate_union)
            keep_fraction_after = float(np.sum(new_keep)) / float(n_finite)
            if keep_fraction_after < min_keep:
                stop_reason = "min_keep_fraction_guard"
                new_keep = keep_mask.copy()
            elif np.array_equal(new_keep, keep_mask):
                stop_reason = "converged_keep_mask"

        changed_count = int(np.sum(new_keep != keep_mask))
        keep_fraction = float(np.sum(new_keep)) / float(n_finite)
        iteration_summaries.append(
            {
                "iter_index": int(iter_idx + 1),
                "n_finite": int(n_finite),
                "n_keep_before": int(n_keep_before),
                "n_keep_after": int(np.sum(new_keep)),
                "keep_fraction_after": float(keep_fraction),
                "residual_median": float(med),
                "residual_mad": float(mad),
                "residual_robust_scale": float(robust_scale),
                "n_candidate_excluded_residual": int(np.sum(candidate_resid)),
                "n_candidate_excluded_residual_upper_tail": int(np.sum(candidate_resid_upper_tail)),
                "n_candidate_excluded_residual_lower_tail": int(np.sum(candidate_resid_lower_tail)),
                "n_candidate_excluded_local_var": int(np.sum(candidate_var)),
                "changed_count": int(changed_count),
                "slope": float(slope),
                "intercept": float(intercept),
                "signal_excursion_polarity_applied": str(polarity),
                "robust_backend_used": str(backend_used),
                "stop_reason": stop_reason,
            }
        )

        final_slope = float(slope)
        final_intercept = float(intercept)
        final_fit = fit_all
        keep_mask = new_keep
        stop_reason_final = stop_reason or "max_iters_reached"

        if stop_reason in {"mad_zero_or_nonfinite", "min_keep_fraction_guard", "converged_keep_mask"}:
            break

    params, fail_reason, backend_used = _unpack_robust_fit_result(
        _fit_robust_linear(
            iso[keep_mask],
            sig[keep_mask],
            use_intercept=bool(use_intercept),
        )
    )
    if params is not None:
        final_slope, final_intercept = float(params[0]), float(params[1])
        final_fit = (final_slope * iso) + final_intercept
        robust_backend_used = str(backend_used)
    elif not np.isfinite(final_slope):
        raise RuntimeError(
            "robust_global_event_reject final fit failed: "
            f"{fail_reason or 'unknown'}"
        )

    return {
        "iso_fit_signal_units": np.asarray(final_fit, dtype=float),
        "keep_mask": np.asarray(keep_mask, dtype=bool),
        "excluded_mask": np.asarray(finite & (~keep_mask), dtype=bool),
        "final_coef": {
            "slope": float(final_slope),
            "intercept": float(final_intercept),
            "use_intercept": bool(use_intercept),
            "n_kept": int(np.sum(keep_mask)),
            "n_finite": int(n_finite),
            "keep_fraction": float(np.sum(keep_mask) / float(max(1, n_finite))),
        },
        "iteration_summaries": iteration_summaries,
        "local_var_rule_enabled": bool(use_var_rule),
        "local_var_window_samples": int(local_var_window_samples),
        "n_iterations_completed": int(len(iteration_summaries)),
        "final_keep_fraction": float(np.sum(keep_mask) / float(max(1, n_finite))),
        "stop_reason": str(stop_reason_final),
        "robust_fit_backend_used": str(robust_backend_used),
        "signal_excursion_polarity_applied": str(polarity),
    }


def fit_adaptive_event_gated_regression(
    signal_raw: np.ndarray,
    iso_raw: np.ndarray,
    *,
    signal_fit_input: Optional[np.ndarray] = None,
    iso_fit_input: Optional[np.ndarray] = None,
    sample_rate_hz: float,
    residual_z_thresh: float = 3.5,
    local_var_window_sec: float | None = 10.0,
    local_var_ratio_thresh: float | None = None,
    smooth_window_sec: float = 60.0,
    min_trust_fraction: float = 0.5,
    freeze_interp_method: str = "linear_hold",
    use_intercept: bool = True,
    signal_excursion_polarity: str = "positive",
) -> dict:
    """
    Slow adaptive fit with event gating and coefficient freezing.
    """
    sig_raw_arr = np.asarray(signal_raw, dtype=float).reshape(-1)
    iso_raw_arr = np.asarray(iso_raw, dtype=float).reshape(-1)
    if sig_raw_arr.shape != iso_raw_arr.shape:
        raise ValueError(
            "adaptive_event_gated_regression shape mismatch: "
            f"signal={sig_raw_arr.shape}, iso={iso_raw_arr.shape}"
        )
    if sig_raw_arr.size == 0:
        raise RuntimeError("adaptive_event_gated_regression received empty input")
    if freeze_interp_method not in {"linear_hold"}:
        raise ValueError("adaptive_event_gate_freeze_interp_method must be 'linear_hold'")

    sig_fit = (
        np.asarray(signal_fit_input, dtype=float).reshape(-1)
        if signal_fit_input is not None
        else sig_raw_arr
    )
    iso_fit = (
        np.asarray(iso_fit_input, dtype=float).reshape(-1)
        if iso_fit_input is not None
        else iso_raw_arr
    )
    if sig_fit.shape != sig_raw_arr.shape or iso_fit.shape != iso_raw_arr.shape:
        raise ValueError("adaptive_event_gated_regression fit-input shapes must match raw arrays")

    finite_fit = np.isfinite(sig_fit) & np.isfinite(iso_fit)
    n_finite = int(np.sum(finite_fit))
    if n_finite < 3:
        raise RuntimeError("adaptive_event_gated_regression requires at least 3 finite samples")

    z_thresh = float(residual_z_thresh)
    if z_thresh <= 0.0:
        raise ValueError("adaptive_event_gate_residual_z_thresh must be > 0")
    smooth_sec = float(smooth_window_sec)
    if smooth_sec <= 0.0:
        raise ValueError("adaptive_event_gate_smooth_window_sec must be > 0")
    min_trust = float(min_trust_fraction)
    if not (0.0 < min_trust <= 1.0):
        raise ValueError("adaptive_event_gate_min_trust_fraction must be in (0, 1]")
    polarity = _normalize_signal_excursion_polarity(signal_excursion_polarity)

    fs_hz = float(sample_rate_hz)
    if (not np.isfinite(fs_hz)) or fs_hz <= 0.0:
        fs_hz = 1.0

    global_params, fail_reason, robust_backend_used = _unpack_robust_fit_result(
        _fit_robust_linear(iso_fit[finite_fit], sig_fit[finite_fit], use_intercept=bool(use_intercept))
    )
    if global_params is None:
        raise RuntimeError(
            "adaptive_event_gated_regression robust initialization failed: "
            f"{fail_reason or 'unknown'}"
        )
    slope_global, intercept_global = float(global_params[0]), float(global_params[1])
    global_fit = (slope_global * iso_fit) + intercept_global
    residual = sig_fit - global_fit

    residual_candidate = np.zeros_like(finite_fit, dtype=bool)
    residual_median = float(np.nanmedian(residual[finite_fit]))
    mad = float(np.nanmedian(np.abs(residual[finite_fit] - residual_median)))
    robust_scale = float(1.4826 * mad)
    residual_candidate_upper_tail = np.zeros_like(finite_fit, dtype=bool)
    residual_candidate_lower_tail = np.zeros_like(finite_fit, dtype=bool)
    if np.isfinite(robust_scale) and robust_scale > 1e-12:
        (
            residual_candidate,
            residual_candidate_upper_tail,
            residual_candidate_lower_tail,
        ) = _residual_excursion_candidates(
            residual,
            finite_fit,
            center=residual_median,
            robust_scale=robust_scale,
            z_thresh=z_thresh,
            signal_excursion_polarity=polarity,
        )

    var_candidate = np.zeros_like(finite_fit, dtype=bool)
    use_var_rule = (
        local_var_window_sec is not None
        and local_var_ratio_thresh is not None
        and float(local_var_ratio_thresh) > 0.0
    )
    local_var_window_samples = 0
    if use_var_rule:
        local_var_window_samples = max(3, int(round(float(local_var_window_sec) * fs_hz)))
        if (local_var_window_samples % 2) == 0:
            local_var_window_samples += 1
        var_sig = _centered_rolling_variance(sig_fit, local_var_window_samples)
        var_iso = _centered_rolling_variance(iso_fit, local_var_window_samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_ratio = var_sig / np.maximum(var_iso, 1e-12)
        var_candidate = (
            finite_fit
            & np.isfinite(var_ratio)
            & (var_ratio > float(local_var_ratio_thresh))
        )

    gated_mask = finite_fit & (residual_candidate | var_candidate)
    trusted_mask = finite_fit & (~gated_mask)
    trust_fraction = float(np.sum(trusted_mask)) / float(max(1, n_finite))
    if trust_fraction < min_trust:
        raise RuntimeError(
            f"adaptive_event_gated_regression trust_fraction_below_min: "
            f"{trust_fraction:.4f} < {min_trust:.4f}"
        )

    smooth_window_samples = max(5, int(round(smooth_sec * fs_hz)))
    if (smooth_window_samples % 2) == 0:
        smooth_window_samples += 1
    min_trusted_samples = max(3, int(round(0.2 * smooth_window_samples)))

    trusted_float = trusted_mask.astype(float)
    iso_use = np.where(trusted_mask, iso_fit, 0.0)
    sig_use = np.where(trusted_mask, sig_fit, 0.0)
    n_valid = _rolling_sum_centered(trusted_float, smooth_window_samples)
    sum_u = _rolling_sum_centered(iso_use, smooth_window_samples)
    sum_s = _rolling_sum_centered(sig_use, smooth_window_samples)
    sum_uu = _rolling_sum_centered(iso_use * iso_use, smooth_window_samples)
    sum_us = _rolling_sum_centered(iso_use * sig_use, smooth_window_samples)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_us = sum_us - (sum_u * sum_s) / np.maximum(n_valid, 1.0)
        var_u = sum_uu - (sum_u * sum_u) / np.maximum(n_valid, 1.0)

    var_floor = max(1e-12, 1e-6 * float(np.nanmedian(np.abs(iso_fit[trusted_mask])) ** 2))
    valid_coef = (
        (n_valid >= float(min_trusted_samples))
        & np.isfinite(cov_us)
        & np.isfinite(var_u)
        & (var_u > var_floor)
    )
    if int(np.sum(valid_coef)) < 2:
        raise RuntimeError("adaptive_event_gated_regression insufficient_trusted_windows_for_local_fit")

    slope_local = np.full(sig_fit.shape, np.nan, dtype=float)
    intercept_local = np.full(sig_fit.shape, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope_local[valid_coef] = cov_us[valid_coef] / var_u[valid_coef]
        intercept_local[valid_coef] = (
            sum_s[valid_coef] - slope_local[valid_coef] * sum_u[valid_coef]
        ) / np.maximum(n_valid[valid_coef], 1.0)

    slope_interp = _interp_fill_nearest_finite(slope_local)
    intercept_interp = _interp_fill_nearest_finite(intercept_local)
    if not np.any(np.isfinite(slope_interp)) or not np.any(np.isfinite(intercept_interp)):
        raise RuntimeError("adaptive_event_gated_regression interpolation_failed")

    support_frac = np.clip(n_valid / float(max(1, smooth_window_samples)), 0.0, 1.0)
    slope_reg = slope_global + support_frac * (slope_interp - slope_global)
    intercept_reg = intercept_global + support_frac * (intercept_interp - intercept_global)

    trusted_anchor_mask = valid_coef & trusted_mask
    if freeze_interp_method == "linear_hold":
        slope_final = _freeze_values_over_gated_mask(slope_reg, gated_mask, trusted_anchor_mask)
        intercept_final = _freeze_values_over_gated_mask(intercept_reg, gated_mask, trusted_anchor_mask)
    else:
        slope_final = slope_reg
        intercept_final = intercept_reg

    fit_raw = (slope_final * iso_raw_arr) + intercept_final
    finite_raw = np.isfinite(sig_raw_arr) & np.isfinite(iso_raw_arr)
    fit_raw = np.asarray(fit_raw, dtype=float)
    fit_raw[~finite_raw] = np.nan

    return {
        "iso_fit_signal_units": fit_raw,
        "trusted_mask": np.asarray(trusted_mask, dtype=bool),
        "gated_mask": np.asarray(gated_mask, dtype=bool),
        "global_init_coef": {
            "slope": float(slope_global),
            "intercept": float(intercept_global),
            "robust_fit_backend_used": str(robust_backend_used),
        },
        "coef_slope": np.asarray(slope_final, dtype=float),
        "coef_intercept": np.asarray(intercept_final, dtype=float),
        "residual_median": float(residual_median),
        "residual_mad": float(mad),
        "residual_robust_scale": float(robust_scale),
        "n_finite": int(n_finite),
        "n_trusted": int(np.sum(trusted_mask)),
        "trust_fraction": float(trust_fraction),
        "gated_fraction": float(np.sum(gated_mask) / float(max(1, n_finite))),
        "n_gated_residual": int(np.sum(residual_candidate)),
        "n_gated_residual_upper_tail": int(np.sum(residual_candidate_upper_tail)),
        "n_gated_residual_lower_tail": int(np.sum(residual_candidate_lower_tail)),
        "n_gated_local_var": int(np.sum(var_candidate)),
        "local_var_rule_enabled": bool(use_var_rule),
        "local_var_window_samples": int(local_var_window_samples),
        "smooth_window_samples": int(smooth_window_samples),
        "min_trusted_samples": int(min_trusted_samples),
        "freeze_interp_method": str(freeze_interp_method),
        "signal_excursion_polarity_applied": str(polarity),
    }


def _compute_dynamic_fit_ref_global_linear(chunk: Chunk, config: Config, mode: str) -> Optional[np.ndarray]:
    """
    Global OLS dynamic-fit mode:
      - fit filtered traces once per ROI: sig_filt ~ a*uv_filt + b
      - reconstruct fitted reference on raw UV: uv_fit = a*uv_raw + b
    """
    if mode == "tonic":
        raise RuntimeError("Invariant violated: tonic mode must not run dynamic isosbestic fitting.")

    if chunk.uv_filt is None or chunk.sig_filt is None:
        raise RuntimeError("Filtered traces are required for dynamic fit mode 'global_linear_regression'.")

    n_rois = int(chunk.uv_filt.shape[1])
    uv_fit_all = np.full_like(chunk.uv_raw, np.nan, dtype=float)
    _ensure_chunk_metadata(chunk)
    chunk.metadata["dynamic_fit_engine"] = "global_linear_ols_v1"
    chunk.metadata["dynamic_fit_engine_info"] = {
        "fit_inputs": "sig_filt ~ a*uv_filt + b",
        "reconstruction_signal": "uv_raw",
        "fit_mode_resolved": "global_linear_regression",
        "fit_input_domain": "filtered",
        "signal_excursion_polarity": str(
            getattr(config, "signal_excursion_polarity", "positive")
        ),
        "baseline_subtract_before_fit": bool(getattr(config, "baseline_subtract_before_fit", False)),
        "baseline_subtract_applied": False,
        "n_rois": n_rois,
        "legacy_knobs_not_used_in_engine": _legacy_knobs_not_used_in_engine(),
    }

    for r_idx in range(n_rois):
        u_f = chunk.uv_filt[:, r_idx]
        s_f = chunk.sig_filt[:, r_idx]
        params, fail_code, var_u = _global_fit_params(u_f, s_f)
        if params is None:
            if fail_code == "DD1":
                msg = f"DEGENERATE[DD1] <2 finite filtered samples in ROI {r_idx} global fit"
            else:
                msg = (
                    "DEGENERATE[DD2] var_u non-finite or too small in ROI "
                    f"{r_idx} global fit (var_u={var_u})"
                )
            chunk.metadata.setdefault("qc_warnings", []).append(msg)
            continue

        slope, intercept = params
        uv_fit_all[:, r_idx] = intercept + slope * chunk.uv_raw[:, r_idx]

    return uv_fit_all


def _compute_dynamic_fit_ref_robust_global_event_reject(
    chunk: Chunk,
    config: Config,
    mode: str,
) -> Optional[np.ndarray]:
    """
    Robust global fit with iterative event-dominated sample rejection.
    Fits on raw traces and reconstructs on raw UV.
    """
    if mode == "tonic":
        raise RuntimeError("Invariant violated: tonic mode must not run dynamic isosbestic fitting.")

    n_rois = int(chunk.uv_raw.shape[1])
    uv_fit_all = np.full_like(chunk.uv_raw, np.nan, dtype=float)
    _ensure_chunk_metadata(chunk)
    chunk.metadata["dynamic_fit_engine"] = "robust_global_event_reject_v1"
    chunk.metadata["dynamic_fit_engine_info"] = {
        "fit_inputs": "sig_raw ~ a*uv_raw + b with iterative event-point rejection",
        "reconstruction_signal": "uv_raw",
        "fit_mode_resolved": "robust_global_event_reject",
        "fit_input_domain": "raw",
        "signal_excursion_polarity": str(
            getattr(config, "signal_excursion_polarity", "positive")
        ),
        "baseline_subtract_before_fit": bool(getattr(config, "baseline_subtract_before_fit", False)),
        "baseline_subtract_applied": False,
        "robust_event_reject_max_iters": int(getattr(config, "robust_event_reject_max_iters", 3)),
        "robust_event_reject_residual_z_thresh": float(
            getattr(config, "robust_event_reject_residual_z_thresh", 3.5)
        ),
        "robust_event_reject_local_var_window_sec": (
            None
            if getattr(config, "robust_event_reject_local_var_window_sec", None) is None
            else float(getattr(config, "robust_event_reject_local_var_window_sec", 10.0))
        ),
        "robust_event_reject_local_var_ratio_thresh": (
            None
            if getattr(config, "robust_event_reject_local_var_ratio_thresh", None) is None
            else float(getattr(config, "robust_event_reject_local_var_ratio_thresh", 0.0))
        ),
        "robust_event_reject_min_keep_fraction": float(
            getattr(config, "robust_event_reject_min_keep_fraction", 0.5)
        ),
        "robust_fit_backend_preference": "sklearn_huber_then_statsmodels_rlm",
        "n_rois": n_rois,
        "legacy_knobs_not_used_in_engine": _legacy_knobs_not_used_in_engine(),
    }
    chunk.metadata["dynamic_fit_event_reject"] = {}

    signal_excursion_polarity = _normalize_signal_excursion_polarity(
        str(getattr(config, "signal_excursion_polarity", "positive"))
    )
    max_iters = int(getattr(config, "robust_event_reject_max_iters", 3))
    residual_z_thresh = float(getattr(config, "robust_event_reject_residual_z_thresh", 3.5))
    local_var_window_sec = getattr(config, "robust_event_reject_local_var_window_sec", 10.0)
    local_var_ratio_thresh = getattr(config, "robust_event_reject_local_var_ratio_thresh", None)
    min_keep_fraction = float(getattr(config, "robust_event_reject_min_keep_fraction", 0.5))

    fallback_roi_count = 0
    robust_backend_used_counts: Dict[str, int] = {}
    for r_idx in range(n_rois):
        roi_name = str(chunk.channel_names[r_idx]) if r_idx < len(chunk.channel_names) else f"roi_{r_idx}"
        sig_raw = np.asarray(chunk.sig_raw[:, r_idx], dtype=float)
        uv_raw = np.asarray(chunk.uv_raw[:, r_idx], dtype=float)
        try:
            robust_result = fit_robust_global_event_reject(
                signal_raw=sig_raw,
                iso_raw=uv_raw,
                max_iters=max_iters,
                residual_z_thresh=residual_z_thresh,
                local_var_window_sec=local_var_window_sec,
                local_var_ratio_thresh=local_var_ratio_thresh,
                min_keep_fraction=min_keep_fraction,
                sample_rate_hz=float(chunk.fs_hz),
                use_intercept=True,
                signal_excursion_polarity=signal_excursion_polarity,
            )
            uv_fit_all[:, r_idx] = np.asarray(
                robust_result["iso_fit_signal_units"], dtype=float
            )
            chunk.metadata["dynamic_fit_event_reject"][roi_name] = {
                "keep_mask": np.asarray(robust_result.get("keep_mask", []), dtype=bool),
                "excluded_mask": np.asarray(robust_result.get("excluded_mask", []), dtype=bool),
                "final_coef": dict(robust_result.get("final_coef", {})),
                "iteration_summaries": list(robust_result.get("iteration_summaries", [])),
                "robust_fit_backend_used": str(robust_result.get("robust_fit_backend_used", "unknown")),
                "n_iterations_completed": int(robust_result.get("n_iterations_completed", 0)),
                "final_keep_fraction": float(robust_result.get("final_keep_fraction", np.nan)),
                "stop_reason": str(robust_result.get("stop_reason", "")),
                "signal_excursion_polarity_applied": str(
                    robust_result.get("signal_excursion_polarity_applied", signal_excursion_polarity)
                ),
                "fallback_to_global_linear": False,
            }
            backend_used = str(robust_result.get("robust_fit_backend_used", "unknown"))
            robust_backend_used_counts[backend_used] = robust_backend_used_counts.get(backend_used, 0) + 1
            continue
        except Exception as exc:
            chunk.metadata.setdefault("qc_warnings", []).append(
                "ROBUST_GLOBAL_EVENT_REJECT_FALLBACK "
                f"roi={roi_name} reason={exc}"
            )

        if chunk.uv_filt is None or chunk.sig_filt is None:
            u_fit = uv_raw
            s_fit = sig_raw
        else:
            u_fit = np.asarray(chunk.uv_filt[:, r_idx], dtype=float)
            s_fit = np.asarray(chunk.sig_filt[:, r_idx], dtype=float)

        params, fail_code, var_u = _global_fit_params(u_fit, s_fit)
        if params is None:
            fallback_roi_count += 1
            if fail_code == "DD1":
                msg = (
                    "ROBUST_GLOBAL_EVENT_REJECT_FALLBACK_DEGENERATE[DD1] "
                    f"roi={roi_name} <2 finite samples"
                )
            else:
                msg = (
                    "ROBUST_GLOBAL_EVENT_REJECT_FALLBACK_DEGENERATE[DD2] "
                    f"roi={roi_name} var_u={var_u}"
                )
            chunk.metadata.setdefault("qc_warnings", []).append(msg)
            chunk.metadata["dynamic_fit_event_reject"][roi_name] = {
                "keep_mask": np.zeros(sig_raw.shape, dtype=bool),
                "excluded_mask": np.zeros(sig_raw.shape, dtype=bool),
                "final_coef": {},
                "iteration_summaries": [],
                "robust_fit_backend_used": "none",
                "n_iterations_completed": 0,
                "final_keep_fraction": 0.0,
                "stop_reason": "fallback_degenerate",
                "fallback_to_global_linear": True,
                "fallback_failed": True,
            }
            continue

        fallback_roi_count += 1
        slope, intercept = params
        uv_fit_all[:, r_idx] = intercept + slope * uv_raw
        chunk.metadata["dynamic_fit_event_reject"][roi_name] = {
            "keep_mask": np.isfinite(sig_raw) & np.isfinite(uv_raw),
            "excluded_mask": np.zeros(sig_raw.shape, dtype=bool),
            "final_coef": {
                "slope": float(slope),
                "intercept": float(intercept),
            },
            "iteration_summaries": [],
            "robust_fit_backend_used": "global_linear_fallback",
            "n_iterations_completed": 0,
            "final_keep_fraction": 1.0,
            "stop_reason": "fallback_global_linear",
            "fallback_to_global_linear": True,
            "fallback_failed": False,
        }
        robust_backend_used_counts["global_linear_fallback"] = robust_backend_used_counts.get(
            "global_linear_fallback", 0
        ) + 1

    chunk.metadata["dynamic_fit_engine_info"]["fallback_roi_count"] = int(fallback_roi_count)
    chunk.metadata["dynamic_fit_engine_info"]["success_roi_count"] = int(n_rois - fallback_roi_count)
    chunk.metadata["dynamic_fit_engine_info"]["robust_backend_used_counts"] = dict(robust_backend_used_counts)
    return uv_fit_all


def _compute_dynamic_fit_ref_adaptive_event_gated_regression(
    chunk: Chunk,
    config: Config,
    mode: str,
) -> Optional[np.ndarray]:
    """
    Adaptive event-gated regression:
      - robust global initialization
      - trust/gating from polarity-aware residual robust-z (+ optional local variance asymmetry)
      - slow local coefficient adaptation from trusted windows
      - coefficient freezing through gated spans
      - reconstruction on raw UV
    """
    if mode == "tonic":
        raise RuntimeError("Invariant violated: tonic mode must not run dynamic isosbestic fitting.")

    n_rois = int(chunk.uv_raw.shape[1])
    uv_fit_all = np.full_like(chunk.uv_raw, np.nan, dtype=float)
    _ensure_chunk_metadata(chunk)
    chunk.metadata["dynamic_fit_engine"] = "adaptive_event_gated_regression_v1"
    chunk.metadata["dynamic_fit_engine_info"] = {
        "fit_inputs": "robust-global init + trusted-window local coefficients + gated-span freeze",
        "fit_mode_resolved": "adaptive_event_gated_regression",
        "fit_input_domain": "filtered_if_available_else_raw",
        "trust_scoring_domain": "filtered_if_available_else_raw",
        "reconstruction_signal": "uv_raw",
        "signal_excursion_polarity": str(
            getattr(config, "signal_excursion_polarity", "positive")
        ),
        "baseline_subtract_before_fit": bool(getattr(config, "baseline_subtract_before_fit", False)),
        "baseline_subtract_applied": False,
        "adaptive_event_gate_residual_z_thresh": float(
            getattr(config, "adaptive_event_gate_residual_z_thresh", 3.5)
        ),
        "adaptive_event_gate_local_var_window_sec": (
            None
            if getattr(config, "adaptive_event_gate_local_var_window_sec", None) is None
            else float(getattr(config, "adaptive_event_gate_local_var_window_sec", 10.0))
        ),
        "adaptive_event_gate_local_var_ratio_thresh": (
            None
            if getattr(config, "adaptive_event_gate_local_var_ratio_thresh", None) is None
            else float(getattr(config, "adaptive_event_gate_local_var_ratio_thresh", 0.0))
        ),
        "adaptive_event_gate_smooth_window_sec": float(
            getattr(config, "adaptive_event_gate_smooth_window_sec", 60.0)
        ),
        "adaptive_event_gate_min_trust_fraction": float(
            getattr(config, "adaptive_event_gate_min_trust_fraction", 0.5)
        ),
        "adaptive_event_gate_freeze_interp_method": str(
            getattr(config, "adaptive_event_gate_freeze_interp_method", "linear_hold")
        ),
        "fallback_hierarchy": [
            "adaptive_event_gated_regression",
            "robust_global_event_reject",
            "global_linear_regression",
        ],
        "n_rois": n_rois,
        "legacy_knobs_not_used_in_engine": _legacy_knobs_not_used_in_engine(),
    }
    chunk.metadata["dynamic_fit_adaptive_event_gated"] = {}

    signal_excursion_polarity = _normalize_signal_excursion_polarity(
        str(getattr(config, "signal_excursion_polarity", "positive"))
    )
    residual_z_thresh = float(getattr(config, "adaptive_event_gate_residual_z_thresh", 3.5))
    local_var_window_sec = getattr(config, "adaptive_event_gate_local_var_window_sec", 10.0)
    local_var_ratio_thresh = getattr(config, "adaptive_event_gate_local_var_ratio_thresh", None)
    smooth_window_sec = float(getattr(config, "adaptive_event_gate_smooth_window_sec", 60.0))
    min_trust_fraction = float(getattr(config, "adaptive_event_gate_min_trust_fraction", 0.5))
    freeze_interp_method = str(getattr(config, "adaptive_event_gate_freeze_interp_method", "linear_hold"))

    success_roi_count = 0
    fallback_robust_roi_count = 0
    fallback_global_roi_count = 0
    backend_counts: Dict[str, int] = {}
    for r_idx in range(n_rois):
        roi_name = str(chunk.channel_names[r_idx]) if r_idx < len(chunk.channel_names) else f"roi_{r_idx}"
        sig_raw = np.asarray(chunk.sig_raw[:, r_idx], dtype=float)
        uv_raw = np.asarray(chunk.uv_raw[:, r_idx], dtype=float)
        sig_fit_input = (
            np.asarray(chunk.sig_filt[:, r_idx], dtype=float)
            if chunk.sig_filt is not None
            else sig_raw
        )
        uv_fit_input = (
            np.asarray(chunk.uv_filt[:, r_idx], dtype=float)
            if chunk.uv_filt is not None
            else uv_raw
        )

        try:
            adaptive_result = fit_adaptive_event_gated_regression(
                signal_raw=sig_raw,
                iso_raw=uv_raw,
                signal_fit_input=sig_fit_input,
                iso_fit_input=uv_fit_input,
                sample_rate_hz=float(chunk.fs_hz),
                residual_z_thresh=residual_z_thresh,
                local_var_window_sec=local_var_window_sec,
                local_var_ratio_thresh=local_var_ratio_thresh,
                smooth_window_sec=smooth_window_sec,
                min_trust_fraction=min_trust_fraction,
                freeze_interp_method=freeze_interp_method,
                use_intercept=True,
                signal_excursion_polarity=signal_excursion_polarity,
            )
            uv_fit_all[:, r_idx] = np.asarray(adaptive_result["iso_fit_signal_units"], dtype=float)
            global_coef = dict(adaptive_result.get("global_init_coef", {}))
            backend_used = str(global_coef.get("robust_fit_backend_used", "unknown"))
            backend_counts[backend_used] = backend_counts.get(backend_used, 0) + 1
            chunk.metadata["dynamic_fit_adaptive_event_gated"][roi_name] = {
                "trusted_mask": np.asarray(adaptive_result.get("trusted_mask", []), dtype=bool),
                "gated_mask": np.asarray(adaptive_result.get("gated_mask", []), dtype=bool),
                "global_init_coef": global_coef,
                "coef_slope": np.asarray(adaptive_result.get("coef_slope", []), dtype=float),
                "coef_intercept": np.asarray(adaptive_result.get("coef_intercept", []), dtype=float),
                "trust_fraction": float(adaptive_result.get("trust_fraction", np.nan)),
                "gated_fraction": float(adaptive_result.get("gated_fraction", np.nan)),
                "n_trusted": int(adaptive_result.get("n_trusted", 0)),
                "n_finite": int(adaptive_result.get("n_finite", 0)),
                "n_gated_residual": int(adaptive_result.get("n_gated_residual", 0)),
                "n_gated_residual_upper_tail": int(adaptive_result.get("n_gated_residual_upper_tail", 0)),
                "n_gated_residual_lower_tail": int(adaptive_result.get("n_gated_residual_lower_tail", 0)),
                "n_gated_local_var": int(adaptive_result.get("n_gated_local_var", 0)),
                "local_var_rule_enabled": bool(adaptive_result.get("local_var_rule_enabled", False)),
                "smooth_window_samples": int(adaptive_result.get("smooth_window_samples", 0)),
                "freeze_interp_method": str(adaptive_result.get("freeze_interp_method", freeze_interp_method)),
                "signal_excursion_polarity_applied": str(
                    adaptive_result.get("signal_excursion_polarity_applied", signal_excursion_polarity)
                ),
                "fallback_mode": "none",
            }
            success_roi_count += 1
            continue
        except Exception as exc:
            chunk.metadata.setdefault("qc_warnings", []).append(
                "ADAPTIVE_EVENT_GATED_REGRESSION_FALLBACK "
                f"roi={roi_name} reason={exc}"
            )

        try:
            robust_result = fit_robust_global_event_reject(
                signal_raw=sig_raw,
                iso_raw=uv_raw,
                max_iters=int(getattr(config, "robust_event_reject_max_iters", 3)),
                residual_z_thresh=float(getattr(config, "robust_event_reject_residual_z_thresh", 3.5)),
                local_var_window_sec=getattr(config, "robust_event_reject_local_var_window_sec", 10.0),
                local_var_ratio_thresh=getattr(config, "robust_event_reject_local_var_ratio_thresh", None),
                min_keep_fraction=float(getattr(config, "robust_event_reject_min_keep_fraction", 0.5)),
                sample_rate_hz=float(chunk.fs_hz),
                use_intercept=True,
                signal_excursion_polarity=signal_excursion_polarity,
            )
            uv_fit_all[:, r_idx] = np.asarray(robust_result.get("iso_fit_signal_units", np.full_like(sig_raw, np.nan)), dtype=float)
            chunk.metadata["dynamic_fit_adaptive_event_gated"][roi_name] = {
                "trusted_mask": np.asarray(robust_result.get("keep_mask", []), dtype=bool),
                "gated_mask": np.asarray(robust_result.get("excluded_mask", []), dtype=bool),
                "global_init_coef": dict(robust_result.get("final_coef", {})),
                "coef_slope": np.full(sig_raw.shape, np.nan, dtype=float),
                "coef_intercept": np.full(sig_raw.shape, np.nan, dtype=float),
                "trust_fraction": float(robust_result.get("final_keep_fraction", np.nan)),
                "gated_fraction": float(np.mean(np.asarray(robust_result.get("excluded_mask", []), dtype=bool))),
                "n_trusted": int(np.sum(np.asarray(robust_result.get("keep_mask", []), dtype=bool))),
                "n_finite": int(np.sum(np.isfinite(sig_raw) & np.isfinite(uv_raw))),
                "n_gated_residual": int(np.sum(np.asarray(robust_result.get("excluded_mask", []), dtype=bool))),
                "n_gated_residual_upper_tail": int(
                    robust_result.get("iteration_summaries", [{}])[-1].get(
                        "n_candidate_excluded_residual_upper_tail",
                        0,
                    )
                    if robust_result.get("iteration_summaries")
                    else 0
                ),
                "n_gated_residual_lower_tail": int(
                    robust_result.get("iteration_summaries", [{}])[-1].get(
                        "n_candidate_excluded_residual_lower_tail",
                        0,
                    )
                    if robust_result.get("iteration_summaries")
                    else 0
                ),
                "n_gated_local_var": 0,
                "local_var_rule_enabled": bool(robust_result.get("local_var_rule_enabled", False)),
                "smooth_window_samples": 0,
                "freeze_interp_method": str(freeze_interp_method),
                "signal_excursion_polarity_applied": str(
                    robust_result.get("signal_excursion_polarity_applied", signal_excursion_polarity)
                ),
                "fallback_mode": "robust_global_event_reject",
            }
            fallback_robust_roi_count += 1
            backend_used = str(robust_result.get("robust_fit_backend_used", "unknown"))
            backend_counts[backend_used] = backend_counts.get(backend_used, 0) + 1
            continue
        except Exception as exc:
            chunk.metadata.setdefault("qc_warnings", []).append(
                "ADAPTIVE_EVENT_GATED_REGRESSION_FALLBACK_ROBUST_FAILED "
                f"roi={roi_name} reason={exc}"
            )

        if chunk.uv_filt is None or chunk.sig_filt is None:
            u_fit = uv_raw
            s_fit = sig_raw
        else:
            u_fit = np.asarray(chunk.uv_filt[:, r_idx], dtype=float)
            s_fit = np.asarray(chunk.sig_filt[:, r_idx], dtype=float)
        params, fail_code, var_u = _global_fit_params(u_fit, s_fit)
        if params is None:
            if fail_code == "DD1":
                msg = (
                    "ADAPTIVE_EVENT_GATED_REGRESSION_FALLBACK_DEGENERATE[DD1] "
                    f"roi={roi_name} <2 finite samples"
                )
            else:
                msg = (
                    "ADAPTIVE_EVENT_GATED_REGRESSION_FALLBACK_DEGENERATE[DD2] "
                    f"roi={roi_name} var_u={var_u}"
                )
            chunk.metadata.setdefault("qc_warnings", []).append(msg)
            chunk.metadata["dynamic_fit_adaptive_event_gated"][roi_name] = {
                "trusted_mask": np.zeros(sig_raw.shape, dtype=bool),
                "gated_mask": np.zeros(sig_raw.shape, dtype=bool),
                "global_init_coef": {},
                "coef_slope": np.full(sig_raw.shape, np.nan, dtype=float),
                "coef_intercept": np.full(sig_raw.shape, np.nan, dtype=float),
                "trust_fraction": 0.0,
                "gated_fraction": 0.0,
                "n_trusted": 0,
                "n_finite": int(np.sum(np.isfinite(sig_raw) & np.isfinite(uv_raw))),
                "n_gated_residual": 0,
                "n_gated_local_var": 0,
                "local_var_rule_enabled": False,
                "smooth_window_samples": 0,
                "freeze_interp_method": str(freeze_interp_method),
                "fallback_mode": "global_linear_regression_failed",
                "fallback_failed": True,
            }
            fallback_global_roi_count += 1
            continue
        slope, intercept = params
        uv_fit_all[:, r_idx] = intercept + slope * uv_raw
        chunk.metadata["dynamic_fit_adaptive_event_gated"][roi_name] = {
            "trusted_mask": np.isfinite(sig_raw) & np.isfinite(uv_raw),
            "gated_mask": np.zeros(sig_raw.shape, dtype=bool),
            "global_init_coef": {"slope": float(slope), "intercept": float(intercept)},
            "coef_slope": np.full(sig_raw.shape, float(slope), dtype=float),
            "coef_intercept": np.full(sig_raw.shape, float(intercept), dtype=float),
            "trust_fraction": 1.0,
            "gated_fraction": 0.0,
            "n_trusted": int(np.sum(np.isfinite(sig_raw) & np.isfinite(uv_raw))),
            "n_finite": int(np.sum(np.isfinite(sig_raw) & np.isfinite(uv_raw))),
            "n_gated_residual": 0,
            "n_gated_local_var": 0,
            "local_var_rule_enabled": False,
            "smooth_window_samples": 0,
            "freeze_interp_method": str(freeze_interp_method),
            "fallback_mode": "global_linear_regression",
            "fallback_failed": False,
        }
        fallback_global_roi_count += 1
        backend_counts["global_linear_fallback"] = backend_counts.get("global_linear_fallback", 0) + 1

    chunk.metadata["dynamic_fit_engine_info"]["success_roi_count"] = int(success_roi_count)
    chunk.metadata["dynamic_fit_engine_info"]["fallback_robust_roi_count"] = int(fallback_robust_roi_count)
    chunk.metadata["dynamic_fit_engine_info"]["fallback_global_roi_count"] = int(fallback_global_roi_count)
    chunk.metadata["dynamic_fit_engine_info"]["robust_backend_used_counts"] = dict(backend_counts)
    return uv_fit_all


def _compute_dynamic_fit_ref(
    chunk: Chunk,
    config: Config,
    mode: str,
    fit_mode: str,
) -> Optional[np.ndarray]:
    """
    Student-style rolling local linear regression:
      - fit on lowpass-filtered traces (chunk.uv_filt, chunk.sig_filt)
      - compute dense local a(t), b(t) over centered rolling windows
      - reconstruct on raw UV: uv_fit(t) = a(t)*uv_raw(t) + b(t)

    Returns:
      uv_fit (artifact reference estimate), or None on unrecoverable failure.
    """
    if mode == 'tonic':
        raise RuntimeError("Invariant violated: tonic mode must not run dynamic isosbestic fitting.")

    dyn_started = time.perf_counter()
    timing_buckets = {
        'setup': 0.0,
        'fallback_mask_filter': 0.0,
        'fallback_covariance_fit': 0.0,
        'fallback_apply_fit': 0.0,
        'roi_prep': 0.0,
        'window_extract_mask': 0.0,
        'window_covariance_fit': 0.0,
        'window_pearson_gating': 0.0,
        'window_pearson_gating.pearson_call': 0.0,
        'window_pearson_gating.finite_check': 0.0,
        'window_pearson_gating.gating_branch': 0.0,
        'window_pearson_gating.stats_append': 0.0,
        'rolling_window_moments': 0.0,
        'rolling_param_interpolation': 0.0,
        'rolling_apply_fit': 0.0,
        'postprocess_interp_apply': 0.0,
    }
    timing_metrics = {
        'regression_calls': 1,
        'roi_count': 0,
        'center_count': 0,
        'window_iterations_total': 0,
        'window_valid_total': 0,
        'fallback_roi_count': 0,
        'window_pearson_gating.calls_total': 0,
        'window_pearson_gating.calls_exception': 0,
        'window_pearson_gating.calls_nonfinite': 0,
        'window_pearson_gating.branch_low': 0,
        'window_pearson_gating.branch_mid': 0,
        'window_pearson_gating.branch_high': 0,
        'window_pearson_gating.stats_appended': 0,
    }

    def _ensure_metadata():
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}

    def _finalize_and_attach():
        total = time.perf_counter() - dyn_started
        timing_metrics['elapsed_total_sec'] = float(total)
        _ensure_metadata()
        chunk.metadata['dynamic_regression_timing'] = {
            'buckets': timing_buckets,
            'metrics': timing_metrics
        }
        
    if fit_mode not in {"rolling_filtered_to_raw", "rolling_filtered_to_filtered"}:
        raise ValueError(f"Unsupported rolling fit mode: {fit_mode}")

    t_setup = time.perf_counter()
    n_samples = len(chunk.time_sec)
    fs = chunk.fs_hz

    window_samples = int(round(config.window_sec * fs))
    if window_samples < 3:
        window_samples = 3
    # Ensure odd length for a stable centered window definition.
    if (window_samples % 2) == 0:
        window_samples += 1

    uv_fit_input, sig_fit_input, fit_prep_info = _prepare_rolling_fit_inputs(
        chunk,
        config,
        window_samples,
    )
    uv_fit_input_baseline = fit_prep_info.get("_uv_fit_input_baseline")
    sig_fit_input_baseline = fit_prep_info.get("_sig_fit_input_baseline")
    baseline_applied = bool(fit_prep_info.get("baseline_subtract_applied", False))
    n_rois = int(uv_fit_input.shape[1])
    uv_fit_all = np.zeros_like(chunk.uv_raw) * np.nan
    reconstruction_signal = "uv_raw" if fit_mode == "rolling_filtered_to_raw" else "uv_filt"

    timing_metrics['roi_count'] = int(n_rois)

    _ensure_metadata()
    chunk.metadata['dynamic_fit_engine'] = 'rolling_local_ols_v1'
    chunk.metadata['dynamic_fit_engine_info'] = {
        'window_samples': int(window_samples),
        'window_sec': float(config.window_sec),
        'fit_mode_resolved': str(fit_mode),
        'fit_input_domain': fit_prep_info.get('fit_input_domain', 'filtered'),
        'signal_excursion_polarity': str(
            getattr(config, 'signal_excursion_polarity', 'positive')
        ),
        'reconstruction_signal': reconstruction_signal,
        'reconstruction_domain_consistency': (
            "baseline_mapped"
            if baseline_applied
            else "direct"
        ),
        'reconstruction_formula': (
            "uv_fit = a*(u_recon - uv_fit_input_baseline) + b + sig_fit_input_baseline"
            if baseline_applied
            else "uv_fit = a*u_recon + b"
        ),
        'baseline_subtract_before_fit': bool(fit_prep_info.get('baseline_subtract_before_fit', False)),
        'baseline_subtract_applied': bool(fit_prep_info.get('baseline_subtract_applied', False)),
        'baseline_subtract_method': str(fit_prep_info.get('baseline_subtract_method', 'none')),
        'baseline_subtract_window_samples': int(fit_prep_info.get('baseline_subtract_window_samples', 0)),
        'legacy_knobs_not_used_in_engine': _legacy_knobs_not_used_in_engine(),
    }

    min_samples = int(config.min_samples_per_window)
    if min_samples <= 0:
        min_samples = int(round(window_samples * 0.8))
    min_samples = max(2, min(min_samples, window_samples))

    timing_metrics['center_count'] = int(n_samples)
    timing_buckets['setup'] += (time.perf_counter() - t_setup)

    if window_samples >= n_samples:
        # Fallback: perform exactly one regression on the entire chunk
        for i in range(n_rois):
            u_f = uv_fit_input[:, i]
            s_f = sig_fit_input[:, i]
            u_recon = chunk.uv_raw[:, i] if fit_mode == "rolling_filtered_to_raw" else chunk.uv_filt[:, i]
            u_fit_baseline = (
                uv_fit_input_baseline[:, i]
                if baseline_applied and uv_fit_input_baseline is not None
                else None
            )
            s_fit_baseline = (
                sig_fit_input_baseline[:, i]
                if baseline_applied and sig_fit_input_baseline is not None
                else None
            )

            t_fallback_mask = time.perf_counter()
            m = np.isfinite(u_f) & np.isfinite(s_f)
            timing_buckets['fallback_mask_filter'] += (time.perf_counter() - t_fallback_mask)

            if np.sum(m) < 2:
                _ensure_metadata()
                chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD1] <2 samples in ROI {i} win fallback (var_u=NaN)")
                continue

            t_fallback_cov = time.perf_counter()
            params, fail_code, var_u = _global_fit_params(u_f, s_f)
            timing_buckets['fallback_covariance_fit'] += (time.perf_counter() - t_fallback_cov)

            if params is None:
                _ensure_metadata()
                if fail_code == "DD1":
                    chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD1] <2 samples in ROI {i} win fallback (var_u=NaN)")
                else:
                    chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD2] var_u non-finite or too small in ROI {i} win fallback (var_u={var_u})")
                continue

            slope, intercept = params
            t_fallback_apply = time.perf_counter()
            if baseline_applied and u_fit_baseline is not None and s_fit_baseline is not None:
                uv_fit_all[:, i] = slope * (u_recon - u_fit_baseline) + intercept + s_fit_baseline
            else:
                uv_fit_all[:, i] = intercept + slope * u_recon
            timing_buckets['fallback_apply_fit'] += (time.perf_counter() - t_fallback_apply)
            timing_metrics['fallback_roi_count'] += 1

        _ensure_metadata()
        chunk.metadata['window_fallback_global'] = True
        _finalize_and_attach()
        return uv_fit_all

    for r_idx in range(n_rois):
        u_f = uv_fit_input[:, r_idx]
        s_f = sig_fit_input[:, r_idx]
        u_recon = chunk.uv_raw[:, r_idx] if fit_mode == "rolling_filtered_to_raw" else chunk.uv_filt[:, r_idx]
        u_fit_baseline = (
            uv_fit_input_baseline[:, r_idx]
            if baseline_applied and uv_fit_input_baseline is not None
            else None
        )
        s_fit_baseline = (
            sig_fit_input_baseline[:, r_idx]
            if baseline_applied and sig_fit_input_baseline is not None
            else None
        )

        # Calculate variance floor from finite filtered UV.
        t_roi_prep = time.perf_counter()
        u_f_finite = u_f[np.isfinite(u_f)]
        try:
            med_val = np.median(u_f_finite) if len(u_f_finite) > 0 else 0.0
            var_floor = 1e-6 * (med_val ** 2)
            if var_floor < 1e-9:
                var_floor = 1e-9
        except Exception:
            var_floor = 1e-9
        timing_buckets['roi_prep'] += (time.perf_counter() - t_roi_prep)

        t_roll = time.perf_counter()
        m_pair = np.isfinite(u_f) & np.isfinite(s_f)
        m_float = m_pair.astype(float)

        u_use = np.where(m_pair, u_f, 0.0)
        s_use = np.where(m_pair, s_f, 0.0)

        n_valid = _rolling_sum_centered(m_float, window_samples)
        sum_u = _rolling_sum_centered(u_use, window_samples)
        sum_s = _rolling_sum_centered(s_use, window_samples)
        sum_uu = _rolling_sum_centered(u_use * u_use, window_samples)
        sum_us = _rolling_sum_centered(u_use * s_use, window_samples)
        timing_buckets['rolling_window_moments'] += (time.perf_counter() - t_roll)
        timing_metrics['window_iterations_total'] += int(n_samples)

        if np.any((n_valid > 0.0) & (n_valid < 2.0)):
            _ensure_metadata()
            chunk.metadata.setdefault('qc_warnings', []).append(
                f"DEGENERATE[DD1] <2 samples in centered rolling windows for ROI {r_idx}"
            )

        with np.errstate(invalid='ignore', divide='ignore'):
            cov_us = sum_us - (sum_u * sum_s) / n_valid
            var_u = sum_uu - (sum_u * sum_u) / n_valid

        valid_n = n_valid >= float(min_samples)
        var_bad = valid_n & ((~np.isfinite(var_u)) | (var_u <= var_floor))
        if np.any(var_bad):
            _ensure_metadata()
            chunk.metadata.setdefault('qc_warnings', []).append(
                f"DEGENERATE[DD2] var_u below floor/non-finite in centered rolling windows for ROI {r_idx}"
            )

        valid_fit = valid_n & np.isfinite(cov_us) & np.isfinite(var_u) & (var_u > max(var_floor, 1e-12))
        timing_metrics['window_valid_total'] += int(np.sum(valid_fit))

        slope = np.full(n_samples, np.nan, dtype=float)
        intercept = np.full(n_samples, np.nan, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            slope[valid_fit] = cov_us[valid_fit] / var_u[valid_fit]
            intercept[valid_fit] = (sum_s[valid_fit] - slope[valid_fit] * sum_u[valid_fit]) / n_valid[valid_fit]

        if not np.any(np.isfinite(slope)):
            t_fallback_cov = time.perf_counter()
            params, fail_code, var_u_global = _global_fit_params(u_f, s_f)
            timing_buckets['fallback_covariance_fit'] += (time.perf_counter() - t_fallback_cov)
            if params is None:
                _ensure_metadata()
                if fail_code == "DD1":
                    chunk.metadata.setdefault('qc_warnings', []).append(
                        f"DEGENERATE[DD1] <2 samples in ROI {r_idx} rolling fallback"
                    )
                else:
                    chunk.metadata.setdefault('qc_warnings', []).append(
                        f"DEGENERATE[DD2] var_u non-finite or too small in ROI {r_idx} rolling fallback (var_u={var_u_global})"
                    )
                continue

            slope_val, intercept_val = params
            slope[:] = slope_val
            intercept[:] = intercept_val
            timing_metrics['fallback_roi_count'] += 1
        else:
            t_interp = time.perf_counter()
            slope = _interp_fill_nearest_finite(slope)
            intercept = _interp_fill_nearest_finite(intercept)
            timing_buckets['rolling_param_interpolation'] += (time.perf_counter() - t_interp)

        t_apply = time.perf_counter()
        if baseline_applied and u_fit_baseline is not None and s_fit_baseline is not None:
            uv_fit_all[:, r_idx] = slope * (u_recon - u_fit_baseline) + intercept + s_fit_baseline
        else:
            uv_fit_all[:, r_idx] = slope * u_recon + intercept
        timing_buckets['rolling_apply_fit'] += (time.perf_counter() - t_apply)

    _finalize_and_attach()
    return uv_fit_all


def fit_chunk_dynamic(chunk: Chunk, config: Config, mode: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Orchestrates dynamic fit generation and canonical numerator assembly.
    Returns: (uv_fit, delta_f)
    """
    fit_mode_requested = getattr(config, "dynamic_fit_mode", "rolling_local_regression")
    fit_mode = _resolve_dynamic_fit_mode(config)
    baseline_toggle = bool(getattr(config, "baseline_subtract_before_fit", False))
    bleach_info = _apply_bleach_correction_to_chunk_inputs(chunk, config)
    bleach_mode_resolved = str(bleach_info.get("mode_resolved", "none"))
    bleach_applied = bool(bleach_info.get("applied_any", False))

    orig_sig_raw = chunk.sig_raw
    orig_uv_raw = chunk.uv_raw
    orig_sig_filt = chunk.sig_filt
    orig_uv_filt = chunk.uv_filt
    if bleach_mode_resolved != "none":
        chunk.sig_raw = np.asarray(bleach_info["sig_raw_corrected"], dtype=float)
        chunk.uv_raw = np.asarray(bleach_info["uv_raw_corrected"], dtype=float)
        chunk.sig_filt = (
            None
            if bleach_info.get("sig_filt_corrected", None) is None
            else np.asarray(bleach_info["sig_filt_corrected"], dtype=float)
        )
        chunk.uv_filt = (
            None
            if bleach_info.get("uv_filt_corrected", None) is None
            else np.asarray(bleach_info["uv_filt_corrected"], dtype=float)
        )

    try:
        if fit_mode == "global_linear_regression":
            uv_fit = _compute_dynamic_fit_ref_global_linear(chunk, config, mode)
        elif fit_mode == "robust_global_event_reject":
            uv_fit = _compute_dynamic_fit_ref_robust_global_event_reject(chunk, config, mode)
        elif fit_mode == "adaptive_event_gated_regression":
            uv_fit = _compute_dynamic_fit_ref_adaptive_event_gated_regression(chunk, config, mode)
        else:
            uv_fit = _compute_dynamic_fit_ref(chunk, config, mode, fit_mode=fit_mode)
    finally:
        chunk.sig_raw = orig_sig_raw
        chunk.uv_raw = orig_uv_raw
        chunk.sig_filt = orig_sig_filt
        chunk.uv_filt = orig_uv_filt

    if uv_fit is None:
        return None, None

    if bleach_mode_resolved != "none":
        uv_fit = np.asarray(uv_fit, dtype=float) + np.asarray(
            bleach_info.get("sig_decay_removed", np.zeros_like(uv_fit)),
            dtype=float,
        )

    _ensure_chunk_metadata(chunk)
    chunk.metadata["dynamic_fit_mode_requested"] = (
        "rolling_local_regression" if fit_mode_requested is None else str(fit_mode_requested)
    )
    chunk.metadata["dynamic_fit_mode_resolved"] = str(fit_mode)
    chunk.metadata["dynamic_fit_mode_alias_applied"] = (
        str(fit_mode_requested).strip().lower() in _DYNAMIC_FIT_MODE_ALIASES
        if fit_mode_requested is not None
        else True
    )
    chunk.metadata["baseline_subtract_before_fit_requested"] = baseline_toggle
    chunk.metadata["baseline_subtract_before_fit_applied"] = (
        baseline_toggle and fit_mode in {"rolling_filtered_to_raw", "rolling_filtered_to_filtered"}
    )
    chunk.metadata["bleach_correction_mode_requested"] = str(
        bleach_info.get("mode_requested", getattr(config, "bleach_correction_mode", "none"))
    )
    chunk.metadata["bleach_correction_mode_resolved"] = bleach_mode_resolved
    chunk.metadata["bleach_correction_target"] = str(
        bleach_info.get("target", "signal_and_isosbestic_independent")
    )
    chunk.metadata["bleach_correction_applied"] = bleach_applied
    chunk.metadata["bleach_correction"] = dict(bleach_info.get("per_roi", {}))
    engine_info = chunk.metadata.get("dynamic_fit_engine_info", {})
    if isinstance(engine_info, dict):
        engine_info["bleach_correction_mode"] = bleach_mode_resolved
        engine_info["bleach_correction_applied"] = bleach_applied
        engine_info["bleach_correction_target"] = str(
            bleach_info.get("target", "signal_and_isosbestic_independent")
        )

    delta_f = _assemble_delta_f_from_fit(chunk.sig_raw, uv_fit)
    return uv_fit, delta_f
