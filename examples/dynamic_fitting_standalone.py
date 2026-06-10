"""Single-file standalone isosbestic dynamic fitting utilities.

These functions are faithful array-level extractions of the package phasic
correction algorithms. This file is intended to be copied as one standalone
module and does not require the photometry_pipeline package or slope_qc.py. It uses
only NumPy plus an optional robust-regression backend (scikit-learn preferred,
statsmodels fallback). They perform no file I/O and do not depend on Config,
Chunk, Pipeline, GUI, or cache types.

Inputs are one-dimensional signal/reference arrays in matching acquisition
units. ``signal_excursion_polarity`` controls which residual tail is treated as
putative biological activity. Returned ``iso_fit_signal_units`` is the fitted
reference in signal units. Compute the correction numerator as::

    delta_f = signal - result["iso_fit_signal_units"]

The production dF/F contract requires an independently selected baseline F0::

    dff = 100.0 * delta_f / f0

Example::

    from dynamic_fitting_standalone import (
        fit_adaptive_event_gated_regression,
    )

    result = fit_adaptive_event_gated_regression(
        signal_raw=signal,
        iso_raw=reference,
        sample_rate_hz=40.0,
        smooth_window_sec=60.0,
        slope_constraint="unconstrained",
        min_slope=0.0,
    )
    fit_reference = result["iso_fit_signal_units"]
    delta_f = signal - fit_reference

Set ``slope_constraint="nonnegative"`` to prevent finite negative
UV/reference slopes from being used in the fitted reference. This is an
explicit correction intervention; inspect ``result["slope_constraint_summary"]``
and report how often clamping occurred.
"""

from typing import Any, Dict, List, Optional

import numpy as np


# -----------------------------------------------------------------------------
# Inlined slope-QC helpers.
# These are copied here so this file can be used as a single-file standalone
# module without photometry_pipeline/core/slope_qc.py.
# -----------------------------------------------------------------------------

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


__all__ = [
    "fit_robust_global_event_reject",
    "fit_adaptive_event_gated_regression",
]


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


def _normalize_signal_excursion_polarity(mode_raw: str) -> str:
    mode = str(mode_raw or "positive").strip().lower()
    if mode not in {"positive", "negative", "both"}:
        return "positive"
    return mode


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
    slope_constraint: str = "unconstrained",
    min_slope: float = 0.0,
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
        robust_backend_used = str(backend_used)
    elif not np.isfinite(final_slope):
        raise RuntimeError(
            "robust_global_event_reject final fit failed: "
            f"{fail_reason or 'unknown'}"
        )
    constrained_slope, constraint_summary = apply_slope_constraint(
        float(final_slope),
        constraint_mode=slope_constraint,
        min_slope=min_slope,
        sample_rate_hz=sample_rate_hz,
    )
    final_slope_used = float(constrained_slope)

    intercept_recomputed = False
    global_negative_slope_constrained = False
    if slope_constraint == "nonnegative" and final_slope < min_slope:
        final_slope_used = min_slope
        final_intercept = float(np.median(sig[keep_mask] - final_slope_used * iso[keep_mask]))
        intercept_recomputed = True
        global_negative_slope_constrained = True

    constraint_summary.update({
        "unconstrained_slope": float(final_slope),
        "constrained_slope": float(final_slope_used),
        "intercept_recomputed": bool(intercept_recomputed),
        "global_negative_slope_constrained": bool(global_negative_slope_constrained),
        "warning_level": str(constraint_summary.get("constrained_slope_summary", {}).get("warning_level", "none")),
    })
    final_fit = (final_slope_used * iso) + final_intercept

    return {
        "iso_fit_signal_units": np.asarray(final_fit, dtype=float),
        "keep_mask": np.asarray(keep_mask, dtype=bool),
        "excluded_mask": np.asarray(finite & (~keep_mask), dtype=bool),
        "final_coef": {
            "slope": float(final_slope_used),
            "unconstrained_slope": float(final_slope),
            "intercept": float(final_intercept),
            "use_intercept": bool(use_intercept),
            "n_kept": int(np.sum(keep_mask)),
            "n_finite": int(n_finite),
            "keep_fraction": float(np.sum(keep_mask) / float(max(1, n_finite))),
            "intercept_recomputed": bool(intercept_recomputed),
            "global_negative_slope_constrained": bool(global_negative_slope_constrained),
        },
        "slope_summary": summarize_slope(float(final_slope_used), sample_rate_hz=sample_rate_hz),
        "unconstrained_slope_summary": constraint_summary["unconstrained_slope_summary"],
        "slope_constraint_summary": constraint_summary,
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
    slope_constraint: str = "unconstrained",
    min_slope: float = 0.0,
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
    valid_coef_original = valid_coef.copy()
    n_valid_unconstrained = int(np.sum(valid_coef_original))
    if n_valid_unconstrained < 2:
        raise RuntimeError("adaptive_event_gated_regression insufficient_trusted_windows_for_local_fit")

    slope_local = np.full(sig_fit.shape, np.nan, dtype=float)
    intercept_local = np.full(sig_fit.shape, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope_local[valid_coef] = cov_us[valid_coef] / var_u[valid_coef]
        intercept_local[valid_coef] = (
            sum_s[valid_coef] - slope_local[valid_coef] * sum_u[valid_coef]
        ) / np.maximum(n_valid[valid_coef], 1.0)

    n_neg_support = 0
    neg_frac = 0.0
    n_valid_nonnegative = n_valid_unconstrained
    valid_nonnegative_frac = 1.0
    n_neg_spans = 0
    longest_span = 0
    longest_neg_span_sec = 0.0
    insufficient = False
    fallback_used = False
    fallback_reason = None

    if slope_constraint == "nonnegative":
        neg_support = valid_coef_original & (slope_local < min_slope)
        n_neg_support = int(np.sum(neg_support))
        neg_frac = float(n_neg_support / n_valid_unconstrained) if n_valid_unconstrained > 0 else 0.0

        valid_coef = valid_coef_original & ~neg_support
        n_valid_nonnegative = int(np.sum(valid_coef))
        valid_nonnegative_frac = float(n_valid_nonnegative / n_valid_unconstrained) if n_valid_unconstrained > 0 else 0.0

        n_neg_spans, longest_span = _negative_span_stats(neg_support)
        longest_neg_span_sec = float(longest_span / fs_hz)

        if n_valid_nonnegative < 2 or valid_nonnegative_frac < 0.5:
            insufficient = True
            if slope_global >= min_slope:
                fallback_used = True
                fallback_reason = (
                    f"insufficient nonnegative support (retained {n_valid_nonnegative}/{n_valid_unconstrained} windows, "
                    f"fraction {valid_nonnegative_frac:.4f} < 0.5), fell back to global robust fit"
                )
            else:
                raise RuntimeError(
                    f"adaptive_event_gated_regression: insufficient nonnegative support "
                    f"(retained {n_valid_nonnegative}/{n_valid_unconstrained} windows, "
                    f"fraction {valid_nonnegative_frac:.4f} < 0.5) and global robust fit slope ({slope_global:.4f}) is also < min_slope ({min_slope})."
                )

        slope_local = np.array(slope_local, copy=True)
        intercept_local = np.array(intercept_local, copy=True)
        slope_local[neg_support] = np.nan
        intercept_local[neg_support] = np.nan

    # 1. Compute unconstrained final traces for diagnostics
    slope_local_unc = np.full(sig_fit.shape, np.nan, dtype=float)
    intercept_local_unc = np.full(sig_fit.shape, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope_local_unc[valid_coef_original] = cov_us[valid_coef_original] / var_u[valid_coef_original]
        intercept_local_unc[valid_coef_original] = (
            sum_s[valid_coef_original] - slope_local_unc[valid_coef_original] * sum_u[valid_coef_original]
        ) / np.maximum(n_valid[valid_coef_original], 1.0)

    slope_interp_unc = _interp_fill_nearest_finite(slope_local_unc)
    intercept_interp_unc = _interp_fill_nearest_finite(intercept_local_unc)
    if not np.any(np.isfinite(slope_interp_unc)) or not np.any(np.isfinite(intercept_interp_unc)):
        raise RuntimeError("adaptive_event_gated_regression interpolation_failed")

    support_frac = np.clip(n_valid / float(max(1, smooth_window_samples)), 0.0, 1.0)
    slope_reg_unc = slope_global + support_frac * (slope_interp_unc - slope_global)
    intercept_reg_unc = intercept_global + support_frac * (intercept_interp_unc - intercept_global)

    trusted_anchor_mask_unc = valid_coef_original & trusted_mask
    if freeze_interp_method == "linear_hold":
        slope_unconstrained = _freeze_values_over_gated_mask(slope_reg_unc, gated_mask, trusted_anchor_mask_unc)
        intercept_unconstrained = _freeze_values_over_gated_mask(intercept_reg_unc, gated_mask, trusted_anchor_mask_unc)
    else:
        slope_unconstrained = slope_reg_unc
        intercept_unconstrained = intercept_reg_unc

    # 2. Compute final constrained/reconstructed traces
    if fallback_used:
        slope_final = np.full(sig_fit.shape, slope_global, dtype=float)
        intercept_final = np.full(sig_fit.shape, intercept_global, dtype=float)
    else:
        if slope_constraint == "nonnegative":
            slope_interp = _interp_fill_nearest_finite(slope_local)
            intercept_interp = _interp_fill_nearest_finite(intercept_local)
            if not np.any(np.isfinite(slope_interp)) or not np.any(np.isfinite(intercept_interp)):
                raise RuntimeError("adaptive_event_gated_regression interpolation_failed")

            if slope_global < min_slope:
                slope_reg = slope_interp
                intercept_reg = intercept_interp
            else:
                slope_reg = slope_global + support_frac * (slope_interp - slope_global)
                intercept_reg = intercept_global + support_frac * (intercept_interp - intercept_global)

            trusted_anchor_mask = valid_coef & trusted_mask
            if freeze_interp_method == "linear_hold":
                slope_final = _freeze_values_over_gated_mask(slope_reg, gated_mask, trusted_anchor_mask)
                intercept_final = _freeze_values_over_gated_mask(intercept_reg, gated_mask, trusted_anchor_mask)
            else:
                slope_final = slope_reg
                intercept_final = intercept_reg
        else:
            slope_final = slope_unconstrained.copy()
            intercept_final = intercept_unconstrained.copy()

    # To obtain base constraint_summary structure without modifying slope_final,
    # we call apply_slope_constraint with "unconstrained" mode:
    _, constraint_summary = apply_slope_constraint(
        slope_final,
        constraint_mode="unconstrained",
        min_slope=min_slope,
        sample_rate_hz=fs_hz,
    )

    final_negative_mask = np.isfinite(slope_final) & (slope_final < min_slope - 1e-12)
    final_neg_samples = int(np.sum(final_negative_mask))
    final_neg_frac = float(final_neg_samples / final_negative_mask.size) if final_negative_mask.size > 0 else 0.0
    final_negative_check_failed = False

    if slope_constraint == "nonnegative" and final_neg_samples > 0:
        final_negative_check_failed = True
        if slope_global >= min_slope:
            slope_final = np.full(sig_fit.shape, slope_global, dtype=float)
            intercept_final = np.full(sig_fit.shape, intercept_global, dtype=float)
            fallback_used = True
            fallback_reason = (
                "final adaptive reconstruction produced negative slopes after support filtering "
                f"({final_neg_samples} samples, fraction {final_neg_frac:.6g}), fell back to global robust fit"
            )
        else:
            raise RuntimeError(
                f"adaptive_event_gated_regression nonnegative mode failed: final adaptive reconstruction still "
                f"contained negative slopes ({final_neg_samples} samples, fraction {final_neg_frac:.6g}) and no valid nonnegative global fallback was available"
            )

    constraint_summary.update({
        "slope_constraint_mode": slope_constraint,
        "slope_constraint_applied": bool(n_neg_support > 0 or fallback_used or final_negative_check_failed),
        "n_clamped_slope_samples": int(n_neg_support),
        "slope_clamped_fraction": float(neg_frac),
        "n_clamped_slope_spans": int(n_neg_spans),
        "longest_clamped_slope_span_samples": int(longest_span),
        "longest_clamped_slope_span_sec": float(longest_neg_span_sec) if longest_span > 0 else None,
        "n_negative_slope_support_windows": int(n_neg_support),
        "n_negative_slope_support_samples": int(n_neg_support),
        "negative_slope_support_fraction": float(neg_frac),
        "n_valid_nonnegative_support_windows": int(n_valid_nonnegative),
        "n_valid_nonnegative_support_samples": int(n_valid_nonnegative),
        "valid_nonnegative_support_fraction": float(valid_nonnegative_frac),
        "longest_negative_slope_span_sec": float(longest_neg_span_sec),
        "nonnegative_support_insufficient": bool(insufficient),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "n_final_negative_slope_samples": int(final_neg_samples),
        "final_negative_slope_fraction": float(final_neg_frac),
        "final_negative_slope_check_failed": bool(final_negative_check_failed),
        "unconstrained_slope_summary": summarize_slope(slope_unconstrained, sample_rate_hz=fs_hz),
        "constrained_slope_summary": summarize_slope(slope_final, sample_rate_hz=fs_hz),
    })

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
        "coef_slope_unconstrained": np.asarray(slope_unconstrained, dtype=float),
        "coef_intercept": np.asarray(intercept_final, dtype=float),
        "slope_summary": summarize_slope(slope_final, sample_rate_hz=sample_rate_hz),
        "unconstrained_slope_summary": constraint_summary["unconstrained_slope_summary"],
        "slope_constraint_summary": constraint_summary,
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

