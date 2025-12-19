import numpy as np
import pandas as pd
from typing import Dict, List
from ..config import Config
from .types import Chunk

def extract_features(chunk: Chunk, config: Config) -> pd.DataFrame:
    """
    Computes features per ROI for the chunk.
    Mean, Median, Std, MAD, Peaks, AUC.
    Returns DataFrame with one row per ROI? Or one row per chunk+ROI?
    Spec: "features/features.csv"
    Usually: chunk_id, roi, feat1, feat2...
    """
    if chunk.dff is None:
        return pd.DataFrame()
        
    rows = []
    
    n_rois = chunk.dff.shape[1]
    
    for i in range(n_rois):
        roi = chunk.channel_names[i]
        trace = chunk.dff[:, i]
        
        # If Analysis Failed (NaNs in trace)
        if np.isnan(trace).all():
            row = {
                'chunk_id': chunk.chunk_id,
                'source_file': chunk.source_file,
                'roi': roi,
                'mean': np.nan, 'median': np.nan, 'std': np.nan, 'mad': np.nan,
                'peak_count': np.nan, 'auc': np.nan
            }
        else:
            # Drop NaNs for stats? 
            # If any NaNs exist (e.g. from edges?), usually regression produces valid full trace or all NaN?
            # Regression produces full trace.
            # However, if we had NaNs in filtered input, we might have NaNs.
            # Let's use nan-safe ops
            
            clean_trace = trace[np.isfinite(trace)]
            
            if len(clean_trace) == 0:
                 # Should not happen if trace not all nan
                 row = {
                    'chunk_id': chunk.chunk_id,
                    'source_file': chunk.source_file,
                    'roi': roi,
                    'mean': np.nan, 'median': np.nan, 'std': np.nan, 'mad': np.nan,
                    'peak_count': np.nan, 'auc': np.nan
                }
            else:
                mu = np.mean(clean_trace)
                med = np.median(clean_trace)
                sigma = np.std(clean_trace)
                mad = np.median(np.abs(clean_trace - med))
                
                # Peaks
                # "threshold = mean + 2*std"
                # "min distance = peak_min_distance_sec"
                
                thresh = mu + config.peak_threshold_k * sigma
                
                # Simple peak finding
                # Find local maxima > thresh
                # Since min_distance is specified, we need a smarter finder or scipy.signal.find_peaks
                from scipy.signal import find_peaks
                
                dist_samples = int(config.peak_min_distance_sec * chunk.fs_hz)
                
                peaks, _ = find_peaks(trace, height=thresh, distance=dist_samples)
                n_peaks = len(peaks)
                
                # AUC
                # trapz(dff, time_sec)
                # handle NaNs? if trace valid, trapz.
                auc = np.trapz(trace, chunk.time_sec)
                
                row = {
                    'chunk_id': chunk.chunk_id,
                    'source_file': chunk.source_file,
                    'roi': roi,
                    'mean': mu,
                    'median': med,
                    'std': sigma,
                    'mad': mad,
                    'peak_count': n_peaks,
                    'auc': auc
                }
        
        rows.append(row)
        
    return pd.DataFrame(rows)
