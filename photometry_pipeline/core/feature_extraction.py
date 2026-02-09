import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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
        total_auc = 0.0 # Legacy full AUC (all positive area? or just total integral? Stick to original logic: simple trapz)
        
        
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
                
                # Legacy AUC (integral of raw trace)
                if len(seg_time) > 1:
                    total_auc += np.trapz(seg_trace, seg_time)
                
            
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
