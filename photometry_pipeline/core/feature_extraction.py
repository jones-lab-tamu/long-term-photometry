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
        clean_finite = clean_trace_use[np.isfinite(clean_trace_use)]
        
        if len(clean_finite) < 2:
             if not hasattr(chunk, 'metadata') or chunk.metadata is None: chunk.metadata = {}
             chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD3] < 2 finite samples in ROI '{roi}'")
             
             row = {
                'chunk_id': chunk.chunk_id, 'source_file': chunk.source_file, 'roi': roi,
                'mean': np.nan, 'median': np.nan, 'std': np.nan, 'mad': np.nan,
                'peak_count': 0, 'auc': np.nan,
            }
             rows.append(row)
             continue
        else:
            mu = np.mean(clean_finite)
            med = np.median(clean_finite)
            sigma = np.std(clean_finite)
            mad = np.median(np.abs(clean_finite - med))
            sigma_robust = 1.4826 * mad
            
            # Determine Threshold
            if config.peak_threshold_method == 'mean_std':
                thresh = mu + config.peak_threshold_k * sigma
            elif config.peak_threshold_method == 'percentile':
                thresh = np.nanpercentile(clean_finite, config.peak_threshold_percentile)
                if np.isnan(thresh):
                    raise ValueError(f"Feature Extraction Error: Percentile threshold is NaN for ROI '{roi}' (Chunk {chunk.chunk_id}).")
            elif config.peak_threshold_method == 'median_mad':
                if sigma_robust == 0:
                     if not hasattr(chunk, 'metadata') or chunk.metadata is None: chunk.metadata = {}
                     chunk.metadata.setdefault('qc_warnings', []).append(f"DEGENERATE[DD4] Zero robust variance in ROI '{roi}'")
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

            total_peaks = 0
            total_auc = 0.0

            # Segment iteration
            for s, e in zip(starts, ends):
                seg_y = trace_use[s:e]
                seg_t = time[s:e]
                seg_valid = np.isfinite(seg_y)
                
                if not np.any(seg_valid):
                    continue
                
                # Ensure qc_counts exists
                if not hasattr(chunk, 'metadata') or chunk.metadata is None: chunk.metadata = {}
                qc_counts = chunk.metadata.setdefault('qc_counts', {})
                
                # Check for NaNs within THIS segment window
                if not np.all(seg_valid):
                    qc_counts['DD6'] = qc_counts.get('DD6', 0) + 1
                    warning = f"DEGENERATE[DD6] Analysis signal has NaNs inside raw-finite segment {s}:{e} in ROI '{roi}', splitting into finite runs (nan_gap_split)."
                    if warning not in chunk.metadata.get('qc_warnings', []):
                         chunk.metadata.setdefault('qc_warnings', []).append(warning)
                    
                    # Filter-artifact distinction
                    if use_filter:
                        seg_valid_raw = np.isfinite(trace[s:e])
                        # If raw was all finite but filtered is not, it's a filter artifact
                        if np.all(seg_valid_raw):
                            qc_counts['FILTER_NAN'] = qc_counts.get('FILTER_NAN', 0) + 1
                            f_warning = f"DEGENERATE[FILTER_NAN] lowpass_filter introduced NaNs inside segment {s}:{e} in ROI '{roi}'."
                            if f_warning not in chunk.metadata.get('qc_warnings', []):
                                chunk.metadata.setdefault('qc_warnings', []).append(f_warning)
                
                # Split into contiguous finite runs WITHIN the segment window
                padded_run = np.concatenate(([False], seg_valid, [False]))
                diff_run = np.diff(padded_run.astype(int))
                run_starts = np.where(diff_run == 1)[0]
                run_ends = np.where(diff_run == -1)[0]
                
                for rs, re in zip(run_starts, run_ends):
                    run_y = seg_y[rs:re]
                    run_t = seg_t[rs:re]
                    
                    if len(run_y) < 2:
                        # Emit DD5 for EACH run shorter than 2 samples with absolute indices
                        chunk.metadata.setdefault('qc_warnings', []).append(
                            f"DEGENERATE[DD5] Finite run {s+rs}:{s+re} in ROI '{roi}' length < 2"
                        )
                        qc_counts['DD5'] = qc_counts.get('DD5', 0) + 1
                        continue
                
                    # Compute peaks and AUC for this run
                    peaks, _ = find_peaks(run_y, height=thresh, distance=dist_samples)
                    total_peaks += len(peaks)
                    total_auc += compute_auc_above_threshold(
                        run_y, auc_baseline, 
                        fs_hz=chunk.fs_hz, 
                        time_s=run_t
                    )
            
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
