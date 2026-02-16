import numpy as np
from scipy.signal import sosfiltfilt, butter
from typing import Tuple, Dict
from ..config import Config

def get_lowpass_sos(fs_hz: float, cutoff_hz: float, order: int):
    nyquist = 0.5 * fs_hz
    if cutoff_hz >= nyquist:
        # If cutoff is too high, return identity or warn?
        # For now, clamp or error. 1Hz vs 40Hz is fine.
        return None
    return butter(order, cutoff_hz / nyquist, btype='low', output='sos')

def filter_finite_blocks(data: np.ndarray, sos: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Applies sosfiltfilt only to continuous finite segments of data.
    data: 1D or 2D array. If 2D, filters columns independently.
    Gaps (NaNs, Infs) are preserved.
    
    Padlen rule: 3 * (2 * n_sections + 1). Segments shorter than this are skipped.
    """
    # 2D case
    if data.ndim == 2:
        out = np.full_like(data, np.nan, dtype=float)
        meta = {'n_gaps': 0, 'samples_skipped': 0, 'rois_affected': 0}
        
        for c in range(data.shape[1]):
            col_out, col_meta = filter_finite_blocks(data[:, c], sos)
            out[:, c] = col_out
            meta['n_gaps'] += col_meta['n_gaps']
            meta['samples_skipped'] += col_meta['samples_skipped']
            if col_meta['rois_affected'] > 0:
                meta['rois_affected'] += 1
        return out, meta
        
    # 1D case
    out = np.full_like(data, np.nan, dtype=float)
    is_finite = np.isfinite(data)
    
    meta = {'n_gaps': 0, 'samples_skipped': 0, 'rois_affected': 0}
    
    # Padlen calculation
    # Scipy sosfiltfilt default padlen logic involves 3 * (2*n_sections + 1).
    n_sections = sos.shape[0]
    padlen = 3 * (2 * n_sections + 1)
    
    # If all finite
    if np.all(is_finite):
        if len(data) > padlen:
            out[:] = sosfiltfilt(sos, data, padlen=padlen)
        else:
             meta['samples_skipped'] += len(data)
        return out, meta

    # We have gaps
    meta['rois_affected'] = 1
    
    # Identify continuous blocks
    padded = np.concatenate(([False], is_finite, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    meta['n_gaps'] = max(0, len(starts) - 1)
    if len(starts) > 0 and not is_finite[0]: 
        meta['n_gaps'] += 1 
    
    for s, e in zip(starts, ends):
        length = e - s
        if length > padlen:
            segment = data[s:e]
            filtered_seg = sosfiltfilt(sos, segment, padlen=padlen)
            out[s:e] = filtered_seg
        else:
            # Segment too short
            meta['samples_skipped'] += length
            
    return out, meta


def lowpass_filter(data: np.ndarray, fs_hz: float, config: Config) -> np.ndarray:
    """
    Applies zero-phase lowpass filter to columns of data.
    Legacy signature: returns only data.
    """
    out, _ = lowpass_filter_with_meta(data, fs_hz, config)
    return out

def lowpass_filter_with_meta(data: np.ndarray, fs_hz: float, config: Config) -> Tuple[np.ndarray, Dict]:
    """
    Applies zero-phase lowpass filter.
    Returns: (filtered_data, metadata)
    """
    meta = {}
    
    sos = get_lowpass_sos(fs_hz, config.lowpass_hz, config.filter_order)
    if sos is None:
        return data, meta
        
    # Policy 2: Permissive/Explicit - Segment-wise filtering
    if not np.all(np.isfinite(data)):
        filtered, meta = filter_finite_blocks(data, sos)
    else:
        # Fast path
        filtered = sosfiltfilt(sos, data, axis=0)
        
    return filtered, meta
