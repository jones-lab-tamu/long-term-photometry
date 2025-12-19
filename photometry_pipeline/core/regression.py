import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Tuple, Dict, Optional
from ..config import Config
from .types import Chunk

def fit_chunk_dynamic(chunk: Chunk, config: Config) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Performs dynamic windowed regression.
    Uses FILTERED data for fits.
    Applies parameters to RAW data.
    
    Returns:
       uv_fit, delta_f, dff (None here, separate step), fit_params
       
    Strict Spec Implementation:
    - sliding windows (center aligned)
    - OLS
    - r gating
    - variance floor -> invalid window
    - slope clamping (p5-p95)
    - linear interp of a(t), b(t)
    - apply to raw
    """
    
    # Check if we have filtered data
    if chunk.uv_filt is None or chunk.sig_filt is None:
        raise ValueError("Dynamic regression requires filtered arrays")
        
    n_samples = len(chunk.time_sec)
    n_rois = chunk.uv_filt.shape[1]
    
    uv_filt = chunk.uv_filt
    sig_filt = chunk.sig_filt
    
    uv_raw = chunk.uv_raw
    sig_raw = chunk.sig_raw
    
    fs = chunk.fs_hz
    
    # Outputs
    uv_fit_all = np.zeros_like(uv_raw) * np.nan
    delta_f_all = np.zeros_like(sig_raw) * np.nan
    
    # Parameters for features/debugging? Not strictly required by spec output schema but good to have
    # We apply per ROI
    
    window_samples = int(config.window_sec * fs)
    step_samples = int(config.step_sec * fs)
    half_window = window_samples // 2
    
    # Center points for windows
    # Range: from half_window to n_samples - half_window
    if window_samples > n_samples:
         # Chunk too short for even one window?
         # Spec: "if valid windows >= 10: clamp... else ROI FAILS"
         # If < 1 window, ROI FAILS immediately.
         centers = []
    else:
        centers = np.arange(half_window, n_samples - half_window, step_samples)
        
    # We iterate ROIs
    for r_idx in range(n_rois):
        u_f = uv_filt[:, r_idx]
        s_f = sig_filt[:, r_idx]
        
        u_r = uv_raw[:, r_idx]
        s_r = sig_raw[:, r_idx]
        
        # Variance floor check (Session level? No, spec says "median(uv_filt_chunk)^2")
        # "var_floor = 1e-6 * median(uv_filt_chunk)^2"
        # "if var(uv_filt_window) < var_floor -> window invalid"
        
        med_sq = np.median(u_f)**2
        var_floor = 1e-6 * med_sq
        
        a_vals = []
        b_vals = []
        t_centers_valid = []
        
        # Pass 1: Collect Fits
        for c in centers:
            start = c - half_window
            end = c + half_window
            
            u_win = u_f[start:end]
            s_win = s_f[start:end]
            
            # Variance check
            if np.var(u_win) < var_floor:
                continue # Invalid window
                
            # OLS
            # sig = a*uv + b
            # a = cov(u,s)/var(u)
            # b = mean(s) - a*mean(u)
            # Or use polyfit(deg=1)
            
            slope, intercept = np.polyfit(u_win, s_win, 1)
            
            # Pearson r
            r, _ = pearsonr(u_win, s_win)
            
            # Gating
            # r <= r_low -> g = g_min
            # r >= r_high -> g = 1
            # else linear
            
            g = 1.0
            if r <= config.r_low:
                g = config.g_min
            elif r >= config.r_high:
                g = 1.0
            else:
                # Linear interp
                # slope = (1 - g_min) / (r_high - r_low)
                # g = g_min + slope * (r - r_low)
                m_g = (1.0 - config.g_min) / (config.r_high - config.r_low)
                g = config.g_min + m_g * (r - config.r_low)
                
            a_gated = g * slope
            b_gated = np.mean(s_win) - a_gated * np.mean(u_win)
            
            a_vals.append(a_gated)
            b_vals.append(b_gated)
            t_centers_valid.append(c) # store index
            
        # Slope Clamping
        # "if valid windows >= 10"
        if len(a_vals) < 10:
            # ROI FAILS
            # Outputs remain Nan
            continue
            
        a_arr = np.array(a_vals)
        b_arr = np.array(b_vals)
        
        p5 = np.percentile(a_arr, 5)
        p95 = np.percentile(a_arr, 95)
        
        # clamp slopes
        a_clamped = np.clip(a_arr, p5, p95)
        
        # Recompute b? Spec implies we clamp 'a' and 'b' moves? 
        # Actually spec says: "clamp slopes to [p5, p95]", doesn't mention b.
        # But b = mean(s) - a*mean(u). If a changes, b should change to maintain center?
        # Usually yes. But spec is silent. 
        # "Interpolate a(t), b(t)" implies we interpolate the values we computed.
        # If we clamp A, and don't update B, the line pivot changes.
        # Let's assume we just clamp A and keep B as is? Or recompute B?
        # "Gating... 4) a = g*a_hat 5) b = mean - a*mean". Gating updates B.
        # Clamping is a post-hoc filter on 'a'. 
        # Logic suggests if we force 'a', we should re-estimate 'b' or accept the shift.
        # Given "Strict Spec", and it says "clamp slopes", and nothing else...
        # I will just clamp 'a' and leave 'b'. This might introduce jumps. 
        # But if the slope is clamped, maybe it was an outlier anyway.
        # Let's strictly follow: "clamp slopes to [p5, p95]". 
        # It doesn't say "recompute b".
        
        # Interpolate a(t), b(t) to full length
        # "linear between window centers, hold constant at edges"
        
        t_indices = np.arange(n_samples)
        
        # np.interp uses constant extrapolation by default if we don't specify left/right? 
        # No, np.interp holds constant! "left: Value to return for x < xp[0], default is fp[0]."
        # Perfect.
        
        a_smooth = np.interp(t_indices, t_centers_valid, a_clamped)
        b_smooth = np.interp(t_indices, t_centers_valid, b_arr)
        
        # Apply to RAW
        # uv_fit = a(t) * uv_raw + b(t)
        uv_fit_roi = a_smooth * u_r + b_smooth
        delta_f_roi = s_r - uv_fit_roi
        
        uv_fit_all[:, r_idx] = uv_fit_roi
        delta_f_all[:, r_idx] = delta_f_roi
        
    return uv_fit_all, delta_f_all
