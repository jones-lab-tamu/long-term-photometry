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
}

_DYNAMIC_FIT_MODE_ALIASES = {
    # Backward-compatible alias retained to avoid breaking older configs/artifacts.
    "rolling_local_regression": "rolling_filtered_to_raw",
}


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
        candidate_var = np.zeros_like(finite, dtype=bool)
        if robust_scale <= 1e-12 or not np.isfinite(robust_scale):
            stop_reason = "mad_zero_or_nonfinite"
            new_keep = keep_mask.copy()
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                z_pos = (residual - med) / robust_scale
            candidate_resid = finite & (z_pos > z_thresh)
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
                "n_candidate_excluded_local_var": int(np.sum(candidate_var)),
                "changed_count": int(changed_count),
                "slope": float(slope),
                "intercept": float(intercept),
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
    if fit_mode == "global_linear_regression":
        uv_fit = _compute_dynamic_fit_ref_global_linear(chunk, config, mode)
    elif fit_mode == "robust_global_event_reject":
        uv_fit = _compute_dynamic_fit_ref_robust_global_event_reject(chunk, config, mode)
    else:
        uv_fit = _compute_dynamic_fit_ref(chunk, config, mode, fit_mode=fit_mode)

    if uv_fit is None:
        return None, None

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

    delta_f = _assemble_delta_f_from_fit(chunk.sig_raw, uv_fit)
    return uv_fit, delta_f
