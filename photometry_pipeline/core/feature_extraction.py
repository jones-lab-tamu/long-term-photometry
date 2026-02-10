import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def compute_auc_above_threshold(dff, thresh, fs_hz=None, time_s=None):
    """
    Computes AUC as area above threshold (clamped to 0).
    Args:
        dff (np.array): Phasic dFF signal.
        thresh (float): Threshold value.
        fs_hz (float): Sampling rate (required if time_s is None).
        time_s (np.array): Time vector (optional).
    Returns:
        float: AUC value (>= 0).
    """
    if dff is None or len(dff) < 2:
        return 0.0
    
    if thresh is None or not np.isfinite(thresh):
        raise ValueError("thresh must be finite")
        
    rect = np.clip(dff - thresh, 0.0, None)
    
    if time_s is not None:
        if len(time_s) != len(dff):
            raise ValueError("time_s length mismatch")
        # Check strict monotonicity? Or just let trapz handle it?
        # User req: "strictly increasing (or at least non-decreasing...)"
        if np.any(np.diff(time_s) < 0):
            raise ValueError("time_s must be non-decreasing")
             
        auc = float(np.trapz(rect, x=time_s))
    else:
        if fs_hz is None or fs_hz <= 0:
            raise ValueError("fs_hz must be > 0 when time_s is None")
        dt = 1.0 / fs_hz
        auc = float(np.trapz(rect, dx=dt))
    
    # Single safeguard
    auc = max(auc, 0.0)
        
    return auc

def extract_features(chunk, config):
    """
    Extracts phasic features from a Chunk object.
    
    Implements a NaN-safe segmented iteration over finite runs.
    Peak detection uses scipy.signal.find_peaks with height threshold and minimum distance only.
    Supported threshold methods include mean_std, percentile, and median_mad.
    
    Args:
        chunk (Chunk): The data chunk.
        config (Config): Configuration object.
    
    Returns:
        pd.DataFrame: One row per ROI containing the extracted features.
        Columns: chunk_id, source_file, roi, mean, median, std, mad, peak_count, auc.
    """
    if chunk.dff is None:
        return pd.DataFrame()
        
    rows = []
    n_rois = chunk.dff.shape[1]
    
    for i in range(n_rois):
        roi = chunk.channel_names[i]
        trace = chunk.dff[:, i]
        time = chunk.time_sec
        
        # Valid segments
        is_valid = np.isfinite(trace)
        
        # Identify changes in validity
        # pad with false to detect edges
        padded = np.concatenate(([False], is_valid, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        total_peaks = 0
        total_auc = 0.0 
        
        clean_trace = trace[is_valid]
        
        if len(clean_trace) == 0:
             row = {
                'chunk_id': chunk.chunk_id, 'source_file': chunk.source_file, 'roi': roi,
                'mean': np.nan, 'median': np.nan, 'std': np.nan, 'mad': np.nan,
                'peak_count': 0, 'auc': np.nan,
            }
        else:
            mu = np.mean(clean_trace)
            med = np.median(clean_trace)
            sigma = np.std(clean_trace)
            mad = np.median(np.abs(clean_trace - med))
            sigma_robust = 1.4826 * mad
            
            # Determine Threshold
            if config.peak_threshold_method == 'mean_std':
                thresh = mu + config.peak_threshold_k * sigma
            elif config.peak_threshold_method == 'percentile':
                thresh = np.nanpercentile(clean_trace, config.peak_threshold_percentile)
                if np.isnan(thresh):
                    raise ValueError(f"Feature Extraction Error: Percentile threshold is NaN for ROI '{roi}' (Chunk {chunk.chunk_id}).")
            elif config.peak_threshold_method == 'median_mad':
                if sigma_robust == 0:
                     # If MAD is 0 (e.g. quantization or flat signal), threshold is effectively infinite unless k=0
                     if config.peak_threshold_k == 0:
                         thresh = med
                     else:
                         thresh = float('inf') 
                else:
                    thresh = med + config.peak_threshold_k * sigma_robust
            else:
                raise ValueError(f"Unknown peak_threshold_method: {config.peak_threshold_method}. Supported: ['mean_std', 'percentile', 'median_mad']")
                
            # Constraints
            dist_samples = max(1, int(config.peak_min_distance_sec * chunk.fs_hz))

            # Segment iteration
            for s, e in zip(starts, ends):
                seg_trace = trace[s:e]
                seg_time = time[s:e]
                
                # Find Peaks
                peaks, _ = find_peaks(seg_trace, 
                                      height=thresh, 
                                      distance=dist_samples)
                total_peaks += len(peaks)
                
                # AUC above threshold
                total_auc += compute_auc_above_threshold(seg_trace, thresh, 
                                                         fs_hz=chunk.fs_hz, 
                                                         time_s=seg_time)
            
            row = {
                'chunk_id': chunk.chunk_id,
                'source_file': chunk.source_file,
                'roi': roi,
                'mean': mu,
                'median': med,
                'std': sigma,
                'mad': mad,
                'peak_count': total_peaks,
                'auc': total_auc,
            }
            
        rows.append(row)
        
    return pd.DataFrame(rows)
