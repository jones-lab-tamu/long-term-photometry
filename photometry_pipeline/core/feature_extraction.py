import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .preprocessing import lowpass_filter

def compute_auc_above_threshold(dff, baseline_value, fs_hz=None, time_s=None):
    """
    Computes AUC as area above threshold (clamped to 0).
    Args:
        dff (np.array): Phasic dFF signal.
        baseline_value (float): Threshold/Baseline value for AUC integration.
        fs_hz (float): Sampling rate (required if time_s is None).
        time_s (np.array): Time vector (optional).
    Returns:
        float: AUC value (>= 0).
    """
    if dff is None or len(dff) < 2:
        return 0.0
    
    if baseline_value is None or not np.isfinite(baseline_value):
        raise ValueError("baseline_value must be finite")
        
    rect = np.clip(dff - baseline_value, 0.0, None)
    
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
        
        # Determine signal to use based on pre-filter setting
        use_filter = getattr(config, 'peak_pre_filter', 'none') == 'lowpass'
        if use_filter:
            trace_use = lowpass_filter(trace, chunk.fs_hz, config)
        else:
            trace_use = trace
            
        is_valid_use = np.isfinite(trace_use)
        clean_trace_use = trace_use[is_valid_use]
        
        if len(clean_trace_use) == 0:
             row = {
                'chunk_id': chunk.chunk_id, 'source_file': chunk.source_file, 'roi': roi,
                'mean': np.nan, 'median': np.nan, 'std': np.nan, 'mad': np.nan,
                'peak_count': 0, 'auc': np.nan,
            }
        else:
            mu = np.mean(clean_trace_use)
            med = np.median(clean_trace_use)
            sigma = np.std(clean_trace_use)
            mad = np.median(np.abs(clean_trace_use - med))
            sigma_robust = 1.4826 * mad
            
            # Determine Threshold
            if config.peak_threshold_method == 'mean_std':
                thresh = mu + config.peak_threshold_k * sigma
            elif config.peak_threshold_method == 'percentile':
                thresh = np.nanpercentile(clean_trace_use, config.peak_threshold_percentile)
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
                
            # AUC Baseline
            auc_baseline_method = getattr(config, 'event_auc_baseline', 'zero')
            if auc_baseline_method == 'median':
                auc_baseline = med
            else:
                auc_baseline = 0.0
                
            # Constraints
            dist_samples = max(1, int(config.peak_min_distance_sec * chunk.fs_hz))

            # Segment iteration
            for s, e in zip(starts, ends):
                seg_trace_use = trace_use[s:e]
                
                # Apply local valid mask to segment if needed for peak finding/AUC (though AUC handles NaNs or assumes finite depending on implementation). 
                # Our implementation relies on `find_peaks` which dislikes NaNs.
                seg_valid = is_valid_use[s:e]
                
                # If segment is entirely invalid, skip
                if not np.any(seg_valid):
                    continue
                    
                # Otherwise, interpolate or extract clean seg
                # Actually, find_peaks expects continuous array. Missing data in segments is an edge case.
                # If there are NaNs inside the segment, find_peaks might fail.
                # Since requirement is D1/D2, we need clean_trace_use for stats and baseline.
                seg_time = time[s:e]
                
                # For peaks, finding peaks on NaNs raises an error, so we remove NaNs or just skip 
                if not np.all(seg_valid):
                     # Skip segment if it contains any NaNs to be perfectly safe, or drop them
                     seg_trace_clean = seg_trace_use[seg_valid]
                else:
                     seg_trace_clean = seg_trace_use
                
                # Find Peaks
                peaks, _ = find_peaks(seg_trace_clean, 
                                      height=thresh, 
                                      distance=dist_samples)
                total_peaks += len(peaks)
                
                # AUC above threshold
                total_auc += compute_auc_above_threshold(seg_trace_clean, auc_baseline, 
                                                         fs_hz=chunk.fs_hz, 
                                                         time_s=seg_time[seg_valid] if not np.all(seg_valid) else seg_time)
            
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
