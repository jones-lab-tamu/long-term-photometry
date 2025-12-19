
import numpy as np
import pandas as pd
from typing import Dict, List
from ..config import Config
from .types import Chunk
from scipy.signal import find_peaks

def extract_features(chunk: Chunk, config: Config) -> pd.DataFrame:
    """
    Computes features per ROI.
    Strictly NaN-safe:
    - Iterates contiguous finite segments.
    - Sums peaks and AUC across segments.
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
                'peak_count': np.nan, 'auc': np.nan
            }
        else:
            mu = np.mean(clean_trace)
            med = np.median(clean_trace)
            sigma = np.std(clean_trace)
            mad = np.median(np.abs(clean_trace - med))
            
            if config.peak_threshold_method == 'mean_std':
                thresh = mu + config.peak_threshold_k * sigma
            elif config.peak_threshold_method == 'percentile':
                # B1: Use clean_trace (finite only)
                thresh = np.nanpercentile(clean_trace, config.peak_threshold_percentile)
                # B2: No silent inf. Raise ValueError if NaN.
                if np.isnan(thresh):
                    raise ValueError(f"Feature Extraction Error: Percentile threshold is NaN for ROI '{roi}' (Chunk {chunk.chunk_id}).")
            else:
                # B3: Strict support only
                raise ValueError(f"Unknown peak_threshold_method: {config.peak_threshold_method}. Supported: ['mean_std', 'percentile']")
                
            dist_samples = int(config.peak_min_distance_sec * chunk.fs_hz)
            
            # Segment iteration
            for s, e in zip(starts, ends):
                seg_trace = trace[s:e]
                seg_time = time[s:e]
                
                # Peaks
                # distance must be > 0. If dist_samples < 1, defaults to 1?
                d = max(1, dist_samples)
                peaks, _ = find_peaks(seg_trace, height=thresh, distance=d)
                total_peaks += len(peaks)
                
                # AUC
                if len(seg_time) > 1:
                    seg_auc = np.trapz(seg_trace, seg_time)
                    total_auc += seg_auc
            
            row = {
                'chunk_id': chunk.chunk_id,
                'source_file': chunk.source_file,
                'roi': roi,
                'mean': mu,
                'median': med,
                'std': sigma,
                'mad': mad,
                'peak_count': total_peaks,
                'auc': total_auc
            }
            
        rows.append(row)
        
    return pd.DataFrame(rows)
