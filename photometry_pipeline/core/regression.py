import numpy as np
from scipy.stats import pearsonr
import traceback
from typing import Tuple, Optional
from ..config import Config
from .types import Chunk

def fit_chunk_dynamic(chunk: Chunk, config: Config) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Performs dynamic windowed regression with NaN robustness.
    Returns: (uv_fit, delta_f)
    """
    if chunk.uv_filt is None or chunk.sig_filt is None:
        raise ValueError("Dynamic regression requires filtered arrays")
        
    n_samples = len(chunk.time_sec)
    n_rois = chunk.uv_filt.shape[1]
    fs = chunk.fs_hz
    
    uv_fit_all = np.zeros_like(chunk.uv_filt) * np.nan
    delta_f_all = np.zeros_like(chunk.sig_filt) * np.nan
    
    window_samples = int(config.window_sec * fs)
    step_samples = int(config.step_sec * fs)
    
    if window_samples > n_samples:
        return uv_fit_all, delta_f_all 
    
    # Safety Check for step
    if step_samples <= 0:
        step_samples = 1

    half_window = window_samples // 2
    centers = np.arange(half_window, n_samples - half_window, step_samples)
    
    min_samples = config.min_samples_per_window
    if min_samples <= 0:
        min_samples = int(window_samples * 0.8)
        
    for r_idx in range(n_rois):
        u_f = chunk.uv_filt[:, r_idx]
        s_f = chunk.sig_filt[:, r_idx]
        u_r = chunk.uv_raw[:, r_idx]
        s_r = chunk.sig_raw[:, r_idx]
        
        stats = [] # (t, a_gated, u_mean, s_mean)
        
        # Calculate var_floor safely
        try:
            med_val = np.nanmedian(u_f)
            if np.isnan(med_val): med_val = 0.0
            var_floor = 1e-6 * (med_val**2)
            if var_floor < 1e-9: var_floor = 1e-9
        except Exception:
            var_floor = 1e-9
        
        for c in centers:
            start = c - half_window
            end = c + half_window
            
            u_win = u_f[start:end]
            s_win = s_f[start:end]
            
            # Mask NaNs
            m = np.isfinite(u_win) & np.isfinite(s_win)
            # Check min samples
            if np.sum(m) < min_samples:
                continue
                
            u_w = u_win[m]
            s_w = s_win[m]
            
            # Check Variance
            if np.var(u_w) < var_floor:
                continue
                
            # OLS
            cov = np.cov(u_w, s_w, bias=True)
            var_u = cov[0,0]
            cov_us = cov[0,1]
            
            if var_u <= 1e-12: continue
            
            slope = cov_us / var_u
            
            # Pearson
            with np.errstate(all='ignore'):
                r, _ = pearsonr(u_w, s_w)
            
            if not np.isfinite(r): continue
            
            # Gating
            g = 1.0
            if r <= config.r_low:
                g = config.g_min
            elif r >= config.r_high:
                g = 1.0
            else:
                g = config.g_min + ((1.0 - config.g_min) / (config.r_high - config.r_low)) * (r - config.r_low)
                
            a_gated = g * slope
            
            stats.append((c, a_gated, np.mean(u_w), np.mean(s_w)))

                
        # Post-Processing
        if len(stats) < config.min_valid_windows:
            continue
            
        try:
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
        except Exception:
            # If post-processing fails, return NaNs for this ROI
            continue
        
    return uv_fit_all, delta_f_all
