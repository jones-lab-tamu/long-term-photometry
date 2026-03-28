import numpy as np
import time
from typing import Tuple, Optional
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
