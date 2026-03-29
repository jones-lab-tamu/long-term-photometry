from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.optimize import curve_fit

TONIC_OUTPUT_MODE_PRESERVE_RAW = "preserve_raw_session_shape"
TONIC_OUTPUT_MODE_FLATTEN_BLEACH = (
    "flatten_session_bleach_preserve_session_baseline"
)

_TONIC_OUTPUT_MODE_ALIASES = {
    "raw": TONIC_OUTPUT_MODE_PRESERVE_RAW,
    "preserve_raw": TONIC_OUTPUT_MODE_PRESERVE_RAW,
    "preserve_raw_session_shape": TONIC_OUTPUT_MODE_PRESERVE_RAW,
    "flatten_session_bleach": TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
    "flatten_session_bleach_preserve_session_baseline": TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
}

_TONIC_OUTPUT_MODE_LABELS = {
    TONIC_OUTPUT_MODE_PRESERVE_RAW: "Preserve raw session shape",
    TONIC_OUTPUT_MODE_FLATTEN_BLEACH: "Flatten session bleach, preserve session baseline",
}


def normalize_tonic_output_mode(mode_raw: str | None) -> str:
    if mode_raw is None:
        return TONIC_OUTPUT_MODE_PRESERVE_RAW
    key = str(mode_raw).strip()
    if not key:
        return TONIC_OUTPUT_MODE_PRESERVE_RAW
    if key in _TONIC_OUTPUT_MODE_ALIASES:
        return _TONIC_OUTPUT_MODE_ALIASES[key]
    raise ValueError(
        f"Unknown tonic_output_mode={mode_raw!r}. "
        "Allowed: {'preserve_raw_session_shape', "
        "'flatten_session_bleach_preserve_session_baseline'}"
    )


def tonic_output_mode_label(mode_raw: str | None) -> str:
    mode = normalize_tonic_output_mode(mode_raw)
    return _TONIC_OUTPUT_MODE_LABELS.get(mode, mode)


def _exp_decay_model(t: np.ndarray, c: float, amp: float, tau: float) -> np.ndarray:
    return c + amp * np.exp(-t / tau)


def _fit_exponential_bleach_trend(time_sec: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "model": "exp_decay_c_plus_a_exp_neg_t_over_tau",
        "fallback": False,
        "reason": "",
        "anchor": np.nan,
        "c": np.nan,
        "amp": np.nan,
        "tau": np.nan,
        "n_valid": 0,
    }
    trend = np.full_like(values, np.nan, dtype=float)

    valid = np.isfinite(time_sec) & np.isfinite(values)
    meta["n_valid"] = int(np.sum(valid))
    if meta["n_valid"] < 3:
        meta["fallback"] = True
        meta["reason"] = "insufficient_valid_samples"
        return trend, meta

    t = np.asarray(time_sec[valid], dtype=float)
    x = np.asarray(values[valid], dtype=float)
    if np.ptp(t) <= 1e-12:
        meta["fallback"] = True
        meta["reason"] = "degenerate_time_axis"
        return trend, meta

    t0 = t - float(np.min(t))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = max(x_max - x_min, 1e-6)
    tail_n = max(5, min(len(x), len(x) // 5))
    c0 = float(np.median(x[-tail_n:]))
    amp0 = float(np.median(x[:tail_n]) - c0)
    if abs(amp0) < 1e-9:
        amp0 = float(np.mean(x) - c0)
    tau0 = max(float(np.ptp(t0)) / 3.0, 1e-3)
    tau_lo = max(float(np.ptp(t0)) / 1000.0, 1e-3)
    tau_hi = max(float(np.ptp(t0)) * 100.0, tau_lo * 10.0)

    lower = np.array([x_min - 2.0 * span, -5.0 * span, tau_lo], dtype=float)
    upper = np.array([x_max + 2.0 * span, 5.0 * span, tau_hi], dtype=float)

    try:
        params, _ = curve_fit(
            _exp_decay_model,
            t0,
            x,
            p0=[c0, amp0, tau0],
            bounds=(lower, upper),
            maxfev=20000,
        )
    except Exception:
        meta["fallback"] = True
        meta["reason"] = "exp_fit_failed"
        return trend, meta

    c_hat, amp_hat, tau_hat = [float(v) for v in params]
    local_trend = _exp_decay_model(t0, c_hat, amp_hat, tau_hat)
    if not np.all(np.isfinite(local_trend)):
        meta["fallback"] = True
        meta["reason"] = "nonfinite_trend"
        return trend, meta

    trend[valid] = local_trend
    meta["c"] = c_hat
    meta["amp"] = amp_hat
    meta["tau"] = tau_hat
    meta["anchor"] = float(np.mean(local_trend))
    return trend, meta


def _flatten_single_session_trace(time_sec: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    trend, fit_meta = _fit_exponential_bleach_trend(time_sec, values)
    out = np.array(values, dtype=float, copy=True)

    if fit_meta["fallback"]:
        return out, fit_meta

    valid = np.isfinite(values) & np.isfinite(trend)
    if not np.any(valid):
        fit_meta["fallback"] = True
        fit_meta["reason"] = "no_valid_samples_after_fit"
        return out, fit_meta

    anchor = float(fit_meta["anchor"])
    out[valid] = out[valid] - trend[valid] + anchor
    return out, fit_meta


def _fit_affine_uv_to_iso_fit(uv_raw: np.ndarray, iso_fit_raw: np.ndarray) -> tuple[float, float, Dict[str, Any]]:
    meta: Dict[str, Any] = {"success": False, "reason": "", "slope": np.nan, "intercept": np.nan, "n_valid": 0}
    m = np.isfinite(uv_raw) & np.isfinite(iso_fit_raw)
    meta["n_valid"] = int(np.sum(m))
    if meta["n_valid"] < 3:
        meta["reason"] = "insufficient_valid_samples"
        return np.nan, np.nan, meta
    u = np.asarray(uv_raw[m], dtype=float)
    y = np.asarray(iso_fit_raw[m], dtype=float)
    if float(np.var(u)) <= 1e-12:
        meta["reason"] = "degenerate_uv_variance"
        return np.nan, np.nan, meta
    try:
        slope, intercept = np.polyfit(u, y, 1)
    except Exception:
        meta["reason"] = "affine_fit_failed"
        return np.nan, np.nan, meta
    if not np.isfinite(slope) or not np.isfinite(intercept):
        meta["reason"] = "nonfinite_affine_params"
        return np.nan, np.nan, meta
    meta["success"] = True
    meta["slope"] = float(slope)
    meta["intercept"] = float(intercept)
    return float(slope), float(intercept), meta


def apply_tonic_output_mode_to_session(
    *,
    time_sec: np.ndarray,
    sig_raw: np.ndarray,
    uv_raw: np.ndarray,
    deltaf_raw: np.ndarray,
    mode_raw: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    mode = normalize_tonic_output_mode(mode_raw)
    if mode == TONIC_OUTPUT_MODE_PRESERVE_RAW:
        return (
            np.array(sig_raw, dtype=float, copy=True),
            np.array(uv_raw, dtype=float, copy=True),
            np.array(deltaf_raw, dtype=float, copy=True),
            {"mode": mode, "fallback_count": 0, "channels": {}},
        )

    sig_out, sig_meta = _flatten_single_session_trace(time_sec, sig_raw)
    uv_out, uv_meta = _flatten_single_session_trace(time_sec, uv_raw)
    iso_fit_raw = np.asarray(sig_raw, dtype=float) - np.asarray(deltaf_raw, dtype=float)
    slope, intercept, affine_meta = _fit_affine_uv_to_iso_fit(np.asarray(uv_raw, dtype=float), iso_fit_raw)

    deltaf_out = np.array(deltaf_raw, dtype=float, copy=True)
    deltaf_meta: Dict[str, Any] = {
        "model": "derived_from_flattened_sig_uv_with_affine_iso_fit",
        "fallback": False,
        "reason": "",
        "method": "sig_flat - (slope*uv_flat + intercept)",
    }
    valid_recon = np.isfinite(sig_out) & np.isfinite(uv_out)
    if affine_meta.get("success", False) and np.any(valid_recon):
        iso_fit_flat = slope * uv_out + intercept
        valid = valid_recon & np.isfinite(iso_fit_flat)
        if np.any(valid):
            deltaf_out[valid] = sig_out[valid] - iso_fit_flat[valid]
        else:
            deltaf_meta["fallback"] = True
            deltaf_meta["reason"] = "no_valid_samples_after_affine_reconstruction"
    else:
        iso_fit_flat, iso_fit_meta = _flatten_single_session_trace(time_sec, iso_fit_raw)
        valid = np.isfinite(sig_out) & np.isfinite(iso_fit_flat)
        if np.any(valid):
            deltaf_out[valid] = sig_out[valid] - iso_fit_flat[valid]
            deltaf_meta["method"] = "sig_flat - exp_flat(iso_fit_raw)"
            deltaf_meta["fallback"] = True
            deltaf_meta["reason"] = f"affine_unavailable:{affine_meta.get('reason', 'unknown')}"
            deltaf_meta["iso_fit_flat_meta"] = iso_fit_meta
        else:
            deltaf_meta["fallback"] = True
            deltaf_meta["reason"] = f"deltaf_reconstruction_failed:{affine_meta.get('reason', 'unknown')}"

    deltaf_meta["affine_meta"] = affine_meta

    channels = {
        "sig_raw": sig_meta,
        "uv_raw": uv_meta,
        "deltaF": deltaf_meta,
    }
    fallback_count = int(
        sum(1 for m in channels.values() if bool(m.get("fallback", False)))
    )
    meta = {
        "mode": mode,
        "fallback_count": fallback_count,
        "channels": channels,
    }
    return sig_out, uv_out, deltaf_out, meta
