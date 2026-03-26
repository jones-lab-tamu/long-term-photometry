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


def _compute_dynamic_fit_ref(chunk: Chunk, config: Config, mode: str) -> Optional[np.ndarray]:
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
        
    t_setup = time.perf_counter()
    n_samples = len(chunk.time_sec)
    n_rois = chunk.uv_filt.shape[1]
    fs = chunk.fs_hz

    uv_fit_all = np.zeros_like(chunk.uv_filt) * np.nan

    window_samples = int(round(config.window_sec * fs))
    if window_samples < 3:
        window_samples = 3
    # Ensure odd length for a stable centered window definition.
    if (window_samples % 2) == 0:
        window_samples += 1

    timing_metrics['roi_count'] = int(n_rois)

    _ensure_metadata()
    chunk.metadata['dynamic_fit_engine'] = 'rolling_local_ols_v1'
    chunk.metadata['dynamic_fit_engine_info'] = {
        'window_samples': int(window_samples),
        'window_sec': float(config.window_sec),
        'legacy_knobs_not_used_in_engine': [
            'step_sec',
            'min_valid_windows',
            'r_low',
            'r_high',
            'g_min',
        ],
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
            u_f = chunk.uv_filt[:, i]
            s_f = chunk.sig_filt[:, i]

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
            uv_fit_all[:, i] = intercept + slope * chunk.uv_raw[:, i]
            timing_buckets['fallback_apply_fit'] += (time.perf_counter() - t_fallback_apply)
            timing_metrics['fallback_roi_count'] += 1

        _ensure_metadata()
        chunk.metadata['window_fallback_global'] = True
        _finalize_and_attach()
        return uv_fit_all

    for r_idx in range(n_rois):
        u_f = chunk.uv_filt[:, r_idx]
        s_f = chunk.sig_filt[:, r_idx]
        u_r = chunk.uv_raw[:, r_idx]

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
        uv_fit_all[:, r_idx] = slope * u_r + intercept
        timing_buckets['rolling_apply_fit'] += (time.perf_counter() - t_apply)

    _finalize_and_attach()
    return uv_fit_all


def fit_chunk_dynamic(chunk: Chunk, config: Config, mode: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Orchestrates dynamic fit generation and canonical numerator assembly.
    Returns: (uv_fit, delta_f)
    """
    uv_fit = _compute_dynamic_fit_ref(chunk, config, mode)
    if uv_fit is None:
        return None, None
    delta_f = _assemble_delta_f_from_fit(chunk.sig_raw, uv_fit)
    return uv_fit, delta_f
