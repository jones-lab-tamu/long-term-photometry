"""Dynamic-fit validity diagnostics.

These helpers compute inspection metrics only. They do not alter correction
behavior or choose correction modes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


BASELINE_SCALE_MAX_HZ = 1.0 / 300.0
RESPONSE_SCALE_MIN_HZ = 1.0 / 120.0
RESPONSE_SCALE_MAX_HZ = 1.0 / 10.0
DEFAULT_NEAR_ZERO_SLOPE_TOL = 1e-9


def _json_float(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def _percentile(arr: np.ndarray, pct: float) -> float:
    finite = np.asarray(arr, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, pct))


def _range_p95_p05(arr: np.ndarray) -> tuple[float, float, float]:
    p05 = _percentile(arr, 5.0)
    p95 = _percentile(arr, 95.0)
    rng = float(p95 - p05) if np.isfinite(p05) and np.isfinite(p95) else float("nan")
    return p05, p95, rng


def _safe_ratio(num: float, den: float) -> tuple[float, str | None]:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan"), "nonfinite_range"
    if abs(float(den)) <= 1e-12:
        return float("nan"), "denominator_range_too_small"
    return float(num / den), None


def _safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, str | None]:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError(f"Correlation input length mismatch: {aa.shape} vs {bb.shape}")
    m = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(m)) < 3:
        return float("nan"), "fewer_than_3_finite_pairs"
    x = aa[m]
    y = bb[m]
    if float(np.nanstd(x)) <= 1e-12 or float(np.nanstd(y)) <= 1e-12:
        return float("nan"), "variance_too_small"
    return float(np.corrcoef(x, y)[0, 1]), None


def _slope_metrics(values: Any, threshold: float, prefix: str = "slope") -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            f"{prefix}_median": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_fraction_negative": float("nan"),
            f"{prefix}_fraction_near_zero": float("nan"),
            f"{prefix}_fraction_positive": float("nan"),
            f"{prefix}_n_finite": 0,
        }
    n = float(finite.size)
    return {
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p05": float(np.percentile(finite, 5.0)),
        f"{prefix}_p95": float(np.percentile(finite, 95.0)),
        f"{prefix}_fraction_negative": float(np.sum(finite < -threshold) / n),
        f"{prefix}_fraction_near_zero": float(np.sum(np.abs(finite) <= threshold) / n),
        f"{prefix}_fraction_positive": float(np.sum(finite > threshold) / n),
        f"{prefix}_n_finite": int(finite.size),
    }


def _variance_band_metrics(
    fitted_ref: np.ndarray,
    sample_rate_hz: float,
    *,
    baseline_cutoff_hz: float,
    response_band_min_hz: float,
    response_band_max_hz: float,
) -> dict[str, Any]:
    arr = np.asarray(fitted_ref, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out = {
        "fitted_ref_total_variance": float("nan"),
        "fitted_ref_baseline_scale_variance": float("nan"),
        "fitted_ref_response_scale_variance": float("nan"),
        "fitted_ref_baseline_scale_fraction": float("nan"),
        "fitted_ref_response_scale_fraction": float("nan"),
        "fitted_ref_power_metric_reason": None,
        "dynamic_fit_qc_baseline_cutoff_hz": float(baseline_cutoff_hz),
        "dynamic_fit_qc_response_band_min_hz": float(response_band_min_hz),
        "dynamic_fit_qc_response_band_max_hz": float(response_band_max_hz),
    }
    try:
        fs = float(sample_rate_hz)
    except Exception:
        fs = float("nan")
    if not np.isfinite(fs) or fs <= 0.0:
        out["fitted_ref_power_metric_reason"] = "invalid_sample_rate"
        return out
    if finite.size < 4:
        out["fitted_ref_power_metric_reason"] = "fewer_than_4_finite_samples"
        return out

    centered = finite - float(np.mean(finite))
    total_var = float(np.var(centered))
    out["fitted_ref_total_variance"] = total_var
    if not np.isfinite(total_var) or total_var <= 1e-24:
        out["fitted_ref_power_metric_reason"] = "variance_too_small"
        out["fitted_ref_baseline_scale_variance"] = 0.0
        out["fitted_ref_response_scale_variance"] = 0.0
        out["fitted_ref_baseline_scale_fraction"] = 0.0
        out["fitted_ref_response_scale_fraction"] = 0.0
        return out

    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
    coeff = np.fft.rfft(centered)
    recon_vars: dict[str, float] = {}
    for name, mask in {
        "baseline": (freqs > 0.0) & (freqs <= baseline_cutoff_hz),
        "response": (freqs >= response_band_min_hz) & (freqs <= response_band_max_hz),
    }.items():
        band_coeff = np.zeros_like(coeff)
        band_coeff[mask] = coeff[mask]
        recon = np.fft.irfft(band_coeff, n=centered.size)
        recon_vars[name] = float(np.var(recon))

    baseline_var = recon_vars["baseline"]
    response_var = recon_vars["response"]
    out["fitted_ref_baseline_scale_variance"] = baseline_var
    out["fitted_ref_response_scale_variance"] = response_var
    out["fitted_ref_baseline_scale_fraction"] = float(baseline_var / total_var)
    out["fitted_ref_response_scale_fraction"] = float(response_var / total_var)
    return out


def compute_dynamic_fit_validity_metrics(
    signal: Any,
    iso: Any,
    fitted_ref: Any,
    sample_rate_hz: float,
    slope: Any | None = None,
    local_slope_unconstrained: Any | None = None,
    local_slope_final: Any | None = None,
    fit_mode: str | None = None,
    slope_constraint: str | None = None,
    min_slope: float | None = None,
    baseline_cutoff_hz: float = BASELINE_SCALE_MAX_HZ,
    response_band_min_hz: float = RESPONSE_SCALE_MIN_HZ,
    response_band_max_hz: float = RESPONSE_SCALE_MAX_HZ,
) -> dict[str, Any]:
    """Return per-chunk dynamic-fit validity metrics and inspection flags."""
    sig = np.asarray(signal, dtype=float).reshape(-1)
    uv = np.asarray(iso, dtype=float).reshape(-1)
    fit = np.asarray(fitted_ref, dtype=float).reshape(-1)
    if sig.shape != uv.shape or sig.shape != fit.shape:
        raise ValueError(
            f"Input length mismatch: signal={sig.shape}, iso={uv.shape}, fitted_ref={fit.shape}"
        )

    sig_p05, sig_p95, sig_range = _range_p95_p05(sig)
    uv_p05, uv_p95, uv_range = _range_p95_p05(uv)
    fit_p05, fit_p95, fit_range = _range_p95_p05(fit)
    fit_sig_ratio, fit_sig_reason = _safe_ratio(fit_range, sig_range)
    fit_uv_ratio, fit_uv_reason = _safe_ratio(fit_range, uv_range)

    sig_uv_corr, sig_uv_corr_reason = _safe_corr(sig, uv)
    sig_fit_corr, sig_fit_corr_reason = _safe_corr(sig, fit)
    uv_fit_corr, uv_fit_corr_reason = _safe_corr(uv, fit)

    try:
        min_allowed = float(min_slope) if min_slope is not None else 0.0
    except Exception:
        min_allowed = 0.0
    threshold = max(abs(min_allowed), 0.0) + DEFAULT_NEAR_ZERO_SLOPE_TOL

    metrics: dict[str, Any] = {
        "n_samples": int(sig.size),
        "n_finite_signal": int(np.sum(np.isfinite(sig))),
        "n_finite_iso": int(np.sum(np.isfinite(uv))),
        "n_finite_fitted_ref": int(np.sum(np.isfinite(fit))),
        "signal_p05": sig_p05,
        "signal_p95": sig_p95,
        "signal_range_p95_p05": sig_range,
        "iso_p05": uv_p05,
        "iso_p95": uv_p95,
        "iso_range_p95_p05": uv_range,
        "fitted_ref_p05": fit_p05,
        "fitted_ref_p95": fit_p95,
        "fitted_ref_range_p95_p05": fit_range,
        "fitted_ref_to_signal_range_ratio": fit_sig_ratio,
        "fitted_ref_to_signal_range_ratio_reason": fit_sig_reason,
        "fitted_ref_to_iso_range_ratio": fit_uv_ratio,
        "fitted_ref_to_iso_range_ratio_reason": fit_uv_reason,
        "signal_iso_corr": sig_uv_corr,
        "signal_iso_corr_reason": sig_uv_corr_reason,
        "signal_fitted_ref_corr": sig_fit_corr,
        "signal_fitted_ref_corr_reason": sig_fit_corr_reason,
        "iso_fitted_ref_corr": uv_fit_corr,
        "iso_fitted_ref_corr_reason": uv_fit_corr_reason,
        "fit_mode": "" if fit_mode is None else str(fit_mode),
        "slope_constraint": "" if slope_constraint is None else str(slope_constraint),
        "slope_near_zero_threshold_used": float(threshold),
    }

    if slope is not None:
        metrics.update(_slope_metrics(slope, threshold, "slope"))
    else:
        metrics.update(_slope_metrics([], threshold, "slope"))
    if local_slope_unconstrained is not None:
        metrics.update(
            _slope_metrics(
                local_slope_unconstrained,
                threshold,
                "unconstrained_slope",
            )
        )
    if local_slope_final is not None:
        metrics.update(_slope_metrics(local_slope_final, threshold, "final_slope"))

    metrics.update(
        _variance_band_metrics(
            fit,
            sample_rate_hz,
            baseline_cutoff_hz=baseline_cutoff_hz,
            response_band_min_hz=response_band_min_hz,
            response_band_max_hz=response_band_max_hz,
        )
    )

    low_range = (
        (np.isfinite(fit_sig_ratio) and fit_sig_ratio < 0.05)
        or (np.isfinite(fit_uv_ratio) and fit_uv_ratio < 0.05)
    )
    flat_or_uninformative = bool(low_range)
    neg_frac = _json_float(metrics.get("slope_fraction_negative", float("nan")))
    uncon_neg_frac = _json_float(
        metrics.get("unconstrained_slope_fraction_negative", float("nan"))
    )
    negative_or_mixed = (
        (np.isfinite(neg_frac) and neg_frac > 0.25)
        or (np.isfinite(uncon_neg_frac) and uncon_neg_frac > 0.25)
    )
    response_frac = _json_float(metrics.get("fitted_ref_response_scale_fraction", float("nan")))
    response_rich = bool(np.isfinite(response_frac) and response_frac > 0.35)

    flags: list[str] = []
    if low_range:
        flags.append("FITTED_REFERENCE_LOW_RANGE")
    if flat_or_uninformative:
        flags.append("FITTED_REFERENCE_FLAT_OR_UNINFORMATIVE")
    if negative_or_mixed:
        flags.append("NEGATIVE_OR_MIXED_REFERENCE_COUPLING")
    if response_rich:
        flags.append("FITTED_REFERENCE_RESPONSE_SCALE_RICH")
    needs_inspection = bool(flags)
    if needs_inspection:
        flags.append("DYNAMIC_FIT_NEEDS_INSPECTION")

    metrics.update(
        {
            "dynamic_fit_reference_low_range": bool(low_range),
            "dynamic_fit_reference_flat_or_uninformative": bool(flat_or_uninformative),
            "dynamic_fit_negative_or_mixed_coupling": bool(negative_or_mixed),
            "dynamic_fit_response_scale_rich": bool(response_rich),
            "dynamic_fit_needs_inspection": bool(needs_inspection),
            "dynamic_fit_qc_flags": flags,
        }
    )
    return metrics
