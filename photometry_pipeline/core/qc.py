import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

def compute_lowfreq_preservation_metric(x: np.ndarray, y: np.ndarray, fs_hz: float, cutoff_hz: float) -> float:
    """
    Computes the preservation of low-frequency components between x and y.
    
    Metric: Pearson correlation of low-passed signals.
    Invariant: Preprocessing should not attenuate slow tonic structure.
    """
    if len(x) != len(y):
        return np.nan
        
    # Mask NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    
    # Enforce minimum data requirements
    if len(x) != len(y) or len(x) < 10 or np.sum(mask) < 10:
        return np.nan

    # Strategy: Do NOT drop samples (violates uniform grid).
    # Instead, interpolate NaNs out to preserve the time axis structure.
    
    x_fill = x.copy()
    y_fill = y.copy()
    
    # Indices
    indices = np.arange(len(x))
    
    # Fill X
    mask_x = np.isfinite(x)
    if not np.all(mask_x):
        if np.sum(mask_x) < 2: return np.nan
        x_fill = np.interp(indices, indices[mask_x], x[mask_x])
        
    # Fill Y
    mask_y = np.isfinite(y)
    if not np.all(mask_y):
        if np.sum(mask_y) < 2: return np.nan
        y_fill = np.interp(indices, indices[mask_y], y[mask_y])
    
    # Create Lowpass Filter
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    
    # Safety: if cutoff is too high or low relative to fs/len
    if normal_cutoff >= 1.0 or normal_cutoff <= 0.0:
        return np.nan
        
    # Filter design
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    
    try:
        # Filter full uniform arrays
        x_lp = filtfilt(b, a, x_fill)
        y_lp = filtfilt(b, a, y_fill)
        
        # Correlation ONLY on originally valid samples
        r, _ = pearsonr(x_lp[mask], y_lp[mask])
        return float(r)
    except Exception:
        return np.nan
