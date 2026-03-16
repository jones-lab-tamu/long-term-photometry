import numpy as np
import traceback
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

# In fit_chunk_dynamic loop:
# for c in centers:
#    indices = _get_window_indices(c, window_samples, n_samples)
#    if indices is None: continue
#    start, end = indices


def fit_chunk_dynamic(chunk: Chunk, config: Config, mode: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Performs dynamic windowed regression with NaN robustness.
    Returns: (uv_fit, delta_f)
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
    delta_f_all = np.zeros_like(chunk.sig_filt) * np.nan
    
    window_samples = int(config.window_sec * fs)
    step_samples = int(config.step_sec * fs)
    timing_metrics['roi_count'] = int(n_rois)
    
    if window_samples >= n_samples:
        # Fallback: perform exactly one regression on the entire chunk
        for i in range(n_rois):
            u_f = chunk.uv_filt[:, i]
            s_f = chunk.sig_filt[:, i]
            
            t_fallback_mask = time.perf_counter()
            m = np.isfinite(u_f) & np.isfinite(s_f)
            u_w = u_f[m]
            s_w = s_f[m]
            timing_buckets['fallback_mask_filter'] += (time.perf_counter() - t_fallback_mask)
            
            if len(u_w) < 2 or len(s_w) < 2:
                _ensure_metadata()
                chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD1] <2 samples in ROI {i} win fallback (var_u=NaN)")
                continue
                
            t_fallback_cov = time.perf_counter()
            cov = np.cov(u_w, s_w, bias=True)
            var_u = cov[0, 0]
            cov_us = cov[0, 1]
            timing_buckets['fallback_covariance_fit'] += (time.perf_counter() - t_fallback_cov)
            
            if not np.isfinite(var_u) or var_u <= 1e-12:
                _ensure_metadata()
                chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD2] var_u non-finite or too small in ROI {i} win fallback (var_u={var_u})")
                continue
                
            slope = cov_us / var_u
            intercept = np.mean(s_w) - slope * np.mean(u_w)
            
            t_fallback_apply = time.perf_counter()
            uv_fit_all[:, i] = intercept + slope * chunk.uv_filt[:, i]
            delta_f_all[:, i] = chunk.sig_filt[:, i] - uv_fit_all[:, i]
            timing_buckets['fallback_apply_fit'] += (time.perf_counter() - t_fallback_apply)
            timing_metrics['fallback_roi_count'] += 1
            
        _ensure_metadata()
        chunk.metadata['window_fallback_global'] = True
        timing_buckets['setup'] += (time.perf_counter() - t_setup)
        _finalize_and_attach()
        return uv_fit_all, delta_f_all
    
    # Safety Check for step
    if step_samples <= 0:
        step_samples = 1

    half_window = window_samples // 2
    centers = np.arange(half_window, n_samples - half_window, step_samples)
    timing_metrics['center_count'] = int(len(centers))
    
    min_samples = config.min_samples_per_window
    if min_samples <= 0:
        min_samples = int(window_samples * 0.8)
    timing_buckets['setup'] += (time.perf_counter() - t_setup)
        
    for r_idx in range(n_rois):
        u_f = chunk.uv_filt[:, r_idx]
        s_f = chunk.sig_filt[:, r_idx]
        u_r = chunk.uv_raw[:, r_idx]
        s_r = chunk.sig_raw[:, r_idx]
        
        stats = [] # (t, a_gated, u_mean, s_mean)
        
        # Calculate var_floor safely
        t_roi_prep = time.perf_counter()
        u_f_finite = u_f[np.isfinite(u_f)]
        try:
            med_val = np.median(u_f_finite) if len(u_f_finite) > 0 else 0.0
            var_floor = 1e-6 * (med_val**2)
            if var_floor < 1e-9: var_floor = 1e-9
        except Exception:
            var_floor = 1e-9
        timing_buckets['roi_prep'] += (time.perf_counter() - t_roi_prep)
        
        for c in centers:
            timing_metrics['window_iterations_total'] += 1
            t_window_extract = time.perf_counter()
            indices = _get_window_indices(c, window_samples, n_samples)
            if indices is None:
                timing_buckets['window_extract_mask'] += (time.perf_counter() - t_window_extract)
                continue
            start, end = indices
            
            u_win = u_f[start:end]
            s_win = s_f[start:end]
            
            # Mask NaNs
            m = np.isfinite(u_win) & np.isfinite(s_win)
            # Check min samples
            if np.sum(m) < min_samples:
                timing_buckets['window_extract_mask'] += (time.perf_counter() - t_window_extract)
                continue
                
            u_w = u_win[m]
            s_w = s_win[m]
            timing_buckets['window_extract_mask'] += (time.perf_counter() - t_window_extract)
            
            # Check Variance
            if len(u_w) < 2 or len(s_w) < 2:
                _ensure_metadata()
                chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD1] <2 samples in ROI {r_idx} win {c}")
                continue
            if np.var(u_w) < var_floor:
                _ensure_metadata()
                chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD2] var_u below floor in ROI {r_idx} win {c}")
                continue
                
            # OLS
            t_window_cov = time.perf_counter()
            cov = np.cov(u_w, s_w, bias=True)
            var_u = cov[0,0]
            cov_us = cov[0,1]
            
            if var_u <= 1e-12 or not np.isfinite(var_u): 
                _ensure_metadata()
                chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD2] var_u non-finite or too small in ROI {r_idx} win {c} (var_u={var_u})")
                timing_buckets['window_covariance_fit'] += (time.perf_counter() - t_window_cov)
                continue
            
            slope = cov_us / var_u
            timing_buckets['window_covariance_fit'] += (time.perf_counter() - t_window_cov)
            
            # Pearson-equivalent correlation coefficient from covariance terms.
            # This preserves gating semantics while avoiding the heavy pearsonr call.
            t_pearson_gate = time.perf_counter()
            timing_metrics['window_pearson_gating.calls_total'] += 1
            t_pearson_call = time.perf_counter()
            with np.errstate(all='ignore'):
                var_s = cov[1, 1]
                denom = np.sqrt(var_u * var_s)
                r = cov_us / denom
            timing_buckets['window_pearson_gating.pearson_call'] += (time.perf_counter() - t_pearson_call)
            
            t_finite_check = time.perf_counter()
            if not np.isfinite(r):
                timing_buckets['window_pearson_gating.finite_check'] += (time.perf_counter() - t_finite_check)
                timing_metrics['window_pearson_gating.calls_nonfinite'] += 1
                timing_buckets['window_pearson_gating'] += (time.perf_counter() - t_pearson_gate)
                continue
            timing_buckets['window_pearson_gating.finite_check'] += (time.perf_counter() - t_finite_check)
            
            # Gating
            t_gating_branch = time.perf_counter()
            g = 1.0
            if r <= config.r_low:
                g = config.g_min
                timing_metrics['window_pearson_gating.branch_low'] += 1
            elif r >= config.r_high:
                g = 1.0
                timing_metrics['window_pearson_gating.branch_high'] += 1
            else:
                g = config.g_min + ((1.0 - config.g_min) / (config.r_high - config.r_low)) * (r - config.r_low)
                timing_metrics['window_pearson_gating.branch_mid'] += 1
                
            a_gated = g * slope
            timing_buckets['window_pearson_gating.gating_branch'] += (time.perf_counter() - t_gating_branch)
            
            t_stats_append = time.perf_counter()
            stats.append((c, a_gated, np.mean(u_w), np.mean(s_w)))
            timing_buckets['window_pearson_gating.stats_append'] += (time.perf_counter() - t_stats_append)
            timing_metrics['window_pearson_gating.stats_appended'] += 1
            timing_buckets['window_pearson_gating'] += (time.perf_counter() - t_pearson_gate)

                
        # Post-Processing
        if len(stats) < config.min_valid_windows:
            continue
        timing_metrics['window_valid_total'] += int(len(stats))
            
        try:
            t_post = time.perf_counter()
            stats_arr = np.array(stats)
            t_valid = stats_arr[:, 0]
            a_valid = stats_arr[:, 1]
            u_means = stats_arr[:, 2]
            s_means = stats_arr[:, 3]
            
            p5 = np.percentile(a_valid, 5)
            p95 = np.percentile(a_valid, 95)
            a_clamped = np.clip(a_valid, p5, p95)
            
            b_clamped = s_means - a_clamped * u_means
            
            t_indices = np.arange(n_samples).astype(float)
            
            a_smooth = np.interp(t_indices, t_valid, a_clamped)
            b_smooth = np.interp(t_indices, t_valid, b_clamped)
            
            uv_fit_roi = a_smooth * u_r + b_smooth
            delta_f_roi = s_r - uv_fit_roi
            
            uv_fit_all[:, r_idx] = uv_fit_roi
            delta_f_all[:, r_idx] = delta_f_roi
            timing_buckets['postprocess_interp_apply'] += (time.perf_counter() - t_post)
        except Exception:
            # If post-processing fails, return NaNs for this ROI
            continue

    _finalize_and_attach()
    return uv_fit_all, delta_f_all
