
import numpy as np
from typing import Tuple, Dict, Optional

def compute_global_iso_fit(uv_all: np.ndarray, sig_all: np.ndarray) -> Tuple[float, float, bool]:
    """
    Computes global OLS fit across the entire dataset: sig = a * uv + b
    Returns (slope, intercept, success).
    nan-safe.
    
    Arguments:
        uv_all: Independent variable (Isosbestic)
        sig_all: Dependent variable (Signal)
    """
    mask = np.isfinite(uv_all) & np.isfinite(sig_all)
    if np.sum(mask) < 50: # Require more samples for global fit
        return 0.0, 0.0, False
        
    u = uv_all[mask]
    s = sig_all[mask]
    
    var_u = np.var(u)
    if var_u < 1e-12:
        return 0.0, 0.0, False
        
    cov = np.cov(u, s, bias=True)
    slope = cov[0, 1] / cov[0, 0]
    intercept = np.mean(s) - slope * np.mean(u)
    
    return float(slope), float(intercept), True

def compute_global_iso_fit_robust(uv_all: np.ndarray, sig_all: np.ndarray, 
                                  max_points: int = 200000, 
                                  n_iter: int = 3, 
                                  z_thresh: float = 4.0) -> Tuple[float, float, bool, int]:
    """
    Computes global robust fit on RAW samples (uv_all, sig_all).
    
    Algorithm:
    1. Mask finite values.
    2. Deterministic downsampling to max_points (Unique indices).
    3. Initial OLS.
    4. Iterative re-weighting using MAD of residuals (z_thresh).
    
    Returns (slope, intercept, success, n_points_used).
    """
    mask = np.isfinite(uv_all) & np.isfinite(sig_all)
    n_total = np.sum(mask)
    
    if n_total < 1000:
        return 0.0, 0.0, False, 0
        
    u = uv_all[mask]
    s = sig_all[mask]
    
    # Downsample if needed (Deterministic & Unique)
    if len(u) > max_points:
        # Use np.unique(linspace) to ensure uniqueness? 
        # Actually linspace with endpoint=False and large N usually okay, but casting to int can dupe.
        # Better: slice it.
        step = len(u) / max_points
        indices = np.arange(0, len(u), step).astype(int)
        # Ensure unique just in case step < 1 (impossible if len > max)
        indices = np.unique(indices)
        # Cap at len
        indices = indices[indices < len(u)]
        
        u = u[indices]
        s = s[indices]
        
    n_used = len(u)
    
    # Initial OLS
    if np.var(u) < 1e-12:
        return 0.0, 0.0, False, n_used
        
    slope, intercept, ok = compute_global_iso_fit(u, s)
    if not ok:
        return 0.0, 0.0, False, n_used
        
    # Iterative Robust Refinement
    for i in range(n_iter):
        resid = s - (slope * u + intercept)
        med_resid = np.median(resid)
        mad = np.median(np.abs(resid - med_resid))
        robust_scale = 1.4826 * mad
        
        if robust_scale < 1e-9:
            # Degenerate variance?
            # Check if we have enough points and variance
            if n_used >= 1000 and np.var(u) > 1e-12:
                 return slope, intercept, True, n_used
            else:
                 return 0.0, 0.0, False, n_used
            
        # Filter
        keep = np.abs(resid) <= (z_thresh * robust_scale)
        n_kept = np.sum(keep)
        
        if n_kept < 1000: # Too few points left
            return 0.0, 0.0, False, n_kept
            
        u_iter = u[keep]
        s_iter = s[keep]
        
        if np.var(u_iter) < 1e-12:
            return 0.0, 0.0, False, n_kept
            
        slope, intercept, ok = compute_global_iso_fit(u_iter, s_iter)
        if not ok:
            return 0.0, 0.0, False, n_kept
            
        n_used = n_kept
        u = u_iter # Update for next iter verification or return
        s = s_iter
        
    # Final Validity Check
    if n_used < 1000 or np.var(u) < 1e-12:
        return 0.0, 0.0, False, n_used
        
    return slope, intercept, True, n_used

def apply_global_fit(uv: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """
    Applies global fit parameters to a local uv array.
    iso_fit = slope * uv + intercept
    """
    iso_fit = slope * uv + intercept
    # Propagate NaNs from input (basic arithmetic does this, but being explicit isn't hurtful)
    return iso_fit

def compute_session_tonic_dff_from_global(sig: np.ndarray, uv: np.ndarray, iso_fit: np.ndarray, percentile: float = 2.0) -> Dict:
    """
    DEPRECATED: Ratio Tonic dFF.
    Raises RuntimeError to enforce additive metric.
    """
    raise RuntimeError("Ratio tonic metric is deprecated. Use compute_session_tonic_df_from_global (additive).")

def compute_session_tonic_df_from_global(sig: np.ndarray, uv: np.ndarray, iso_fit: np.ndarray, percentile: float = 2.0) -> Dict:
    """
    Computes Tonic Delta-F (Additive) for a single session using a PRE-COMPUTED global fit.
    
    df = sig - iso_fit
    Scalar = Percentile(df, k)
    
    Returns dictionary with:
        'tonic_scalar': float
        'df': np.ndarray
        'valid_mask': np.ndarray (bool)
        'success': bool
    """
    results = {
        'tonic_scalar': np.nan,
        'df': np.full_like(sig, np.nan),
        'valid_mask': np.zeros_like(sig, dtype=bool),
        'success': False
    }
    
    # Validation mask
    # 1. Finite values (iso_fit doesn't need to be positive for additive, just finite)
    mask = np.isfinite(sig) & np.isfinite(uv) & np.isfinite(iso_fit)
    
    if np.sum(mask) == 0:
        return results
        
    df = np.full_like(sig, np.nan)
    df[mask] = sig[mask] - iso_fit[mask]
    
    # Compute Scalar on valid dfs
    df_finite = df[mask]
    if len(df_finite) < 10:
        scalar = np.nan
    else:
        scalar = np.percentile(df_finite, percentile)
        
    results['tonic_scalar'] = scalar
    results['df'] = df
    results['valid_mask'] = mask
    results['success'] = True
    
    return results

from scipy.interpolate import interp1d

def compute_slow_baselines(t_sec_all: np.ndarray, 
                         uv_all: np.ndarray, 
                         sig_all: np.ndarray, 
                         sessions_meta: list,
                         baseline_stat: str = 'median') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes smooth 'slow' baselines for UV and Signal by interpolating session-wise statistics.
    
    Returns:
        uv_slow: Interpolated slow UV baseline (same shape as input)
        sig_slow: Interpolated slow Signal baseline (same shape as input)
        uv_sess_stats: Array of session statistics used for UV
        sig_sess_stats: Array of session statistics used for Signal
    """
    # Validate inputs
    if len(t_sec_all) != len(uv_all) or len(t_sec_all) != len(sig_all):
        raise ValueError("Input arrays must have same length")
        
    t_mids = []
    uv_stats = []
    sig_stats = []
    
    for meta in sessions_meta:
        s = meta['start_sample']
        e = meta['end_sample']
        
        # Validate indices
        if s >= len(uv_all) or e > len(uv_all):
            continue
            
        uv_chunk = uv_all[s:e]
        sig_chunk = sig_all[s:e]
        mask = np.isfinite(uv_chunk) & np.isfinite(sig_chunk)
        
        if np.sum(mask) < 10:
            continue
            
        # Get Time Midpoint from t_sec_all
        # We use the time at the middle index of the chunk
        mid_idx = s + (e - s) // 2
        t_mids.append(t_sec_all[mid_idx])
        
        # Compute Stat
        if baseline_stat == 'median':
            uv_stats.append(np.median(uv_chunk[mask]))
            sig_stats.append(np.median(sig_chunk[mask]))
        elif baseline_stat == 'p10':
            uv_stats.append(np.percentile(uv_chunk[mask], 10))
            sig_stats.append(np.percentile(sig_chunk[mask], 10))
        else:
            raise ValueError(f"Unknown baseline_stat: {baseline_stat}")
            
    # Convert to arrays
    t_mids = np.array(t_mids)
    uv_stats = np.array(uv_stats)
    sig_stats = np.array(sig_stats)
    
    # Interpolate
    if len(t_mids) < 2:
        # Fallback: simple rolling median or just global median if very short
        # For safety in this specific verification context, warn and return raw or constant
        # But logic requests fallback to rolling median in verification tool.
        # Here we return full-length constant if fitting failed, or raise.
        # Let's return constants of the global median.
        mask_all = np.isfinite(uv_all)
        uv_const = np.median(uv_all[mask_all]) if np.sum(mask_all) > 0 else np.nan
        
        mask_sig = np.isfinite(sig_all)
        sig_const = np.median(sig_all[mask_sig]) if np.sum(mask_sig) > 0 else np.nan
        
        return np.full_like(uv_all, uv_const), np.full_like(sig_all, sig_const), uv_stats, sig_stats

    # Linear interpolation with extrapolation (fill_value="extrapolate")
    f_uv = interp1d(t_mids, uv_stats, kind='linear', bounds_error=False, fill_value="extrapolate")
    f_sig = interp1d(t_mids, sig_stats, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    uv_slow = f_uv(t_sec_all)
    sig_slow = f_sig(t_sec_all)
    
    return uv_slow, sig_slow, uv_stats, sig_stats
    
def _self_test():
    """Internal unit test for linearity and drift cancellation."""
    print("Running tonic_dff internal self-test...")
    
    # 1. Linearity Directionality Test
    N = 1000
    uv = np.linspace(0, 10, N)
    # Target: sig = 2.0 * uv + 10.0
    sig = 2.0 * uv + 10.0
    
    slope, intercept, ok = compute_global_iso_fit(uv, sig)
    
    print(f"Prop Fit Result: Slope={slope:.4f}, Intercept={intercept:.4f}, OK={ok}")
    
    if not ok:
        print("FAIL: Fit returned failure.")
        sys.exit(1)
        
    if abs(slope - 2.0) > 1e-5:
        print(f"FAIL: Slope mismatch. Expected 2.0, got {slope}")
        sys.exit(1)
        
    if abs(intercept - 10.0) > 1e-5:
        print(f"FAIL: Intercept mismatch. Expected 10.0, got {intercept}")
        sys.exit(1)
        
    fit_check = apply_global_fit(uv, slope, intercept)
    err = np.max(np.abs(fit_check - sig))
    if err > 1e-5:
        print(f"FAIL: Reconstruction error too high: {err}")
        sys.exit(1)
        
    # 2. Additive Drift Cancellation Test
    print("Running Additive Drift Cancellation Test...")
    t = np.linspace(0, 10, N)
    # Bleaching UV
    uv_bleach = 200.0 - 0.5 * t 
    # Bio Signal (Sine)
    bio = 5.0 * np.sin(2 * np.pi * t / 2.0)
    # Signal = 2.0 * UV + 10 + Bio
    sig_bleach = 2.0 * uv_bleach + 10.0 + bio
    
    # Fit
    slope2, intercept2, ok2 = compute_global_iso_fit(uv_bleach, sig_bleach)
    if not ok2:
        print("FAIL: Drift fit failed.")
        sys.exit(1)
        
    iso_fit2 = apply_global_fit(uv_bleach, slope2, intercept2)
    # Compute df
    df = sig_bleach - iso_fit2
    
    # Verify Slopes
    from scipy.stats import linregress
    res_df = linregress(t, df)
    slope_df = res_df.slope
    
    print(f"Drift Test: Slope={slope2:.4f}, Int={intercept2:.4f}, df_Slope={slope_df:.6f}")
    
    # Assertions
    # Slope of df should be near 0 (drift cancelled)
    if abs(slope_df) > 0.05:
        print(f"FAIL: Residual drift in df too high: {slope_df}")
        sys.exit(1)
        
    # Bio functionality
    # Should retain the sine wave
    rng = np.ptp(df)
    if rng < 5.0: # Expecting ~10 (amplitude 5 * 2)
        print(f"FAIL: Biological signal lost! Range={rng}")
        sys.exit(1)
        
    print("SUCCESS: Tests passed.")

if __name__ == '__main__':
    import sys
    _self_test()

