import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from .preprocessing import lowpass_filter


# Retain short, fs-tied smoothing defaults for peak preview/calling.
_SAVGOL_WINDOW_SEC = 0.75
_SAVGOL_DEFAULT_POLYORDER = 2
_SAVGOL_MIN_WINDOW = 5

def _trapezoid_integral(y, *, x=None, dx=1.0):
    """NumPy compatibility shim: prefer np.trapezoid, fall back to np.trapz."""
    trapezoid = getattr(np, "trapezoid", None)
    if callable(trapezoid):
        return trapezoid(y, x=x, dx=dx)
    trapz = getattr(np, "trapz", None)
    if callable(trapz):
        return trapz(y, x=x, dx=dx)
    raise AttributeError("NumPy missing both trapezoidal integration APIs: trapezoid/trapz")

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
             
        auc = float(_trapezoid_integral(rect, x=time_s))
    else:
        if fs_hz is None or fs_hz <= 0:
            raise ValueError("fs_hz must be > 0 when time_s is None")
        dt = 1.0 / fs_hz
        auc = float(_trapezoid_integral(rect, dx=dt))
    
    # Single safeguard
    auc = max(auc, 0.0)
        
    return auc

def get_event_signal_array(chunk, config):
    signal_type = getattr(config, 'event_signal', 'dff')
    if signal_type == 'dff':
        if getattr(chunk, 'dff', None) is None:
            raise ValueError("chunk.dff was not computed but config.event_signal='dff'")
        return chunk.dff
    elif signal_type == 'delta_f':
        if getattr(chunk, 'delta_f', None) is None:
            raise ValueError("chunk.delta_f was not computed but config.event_signal='delta_f'")
        return chunk.delta_f
    else:
        raise ValueError(f"Unknown event_signal: {signal_type}")


def _compute_detection_threshold(clean_finite: np.ndarray, config) -> float:
    """Compute peak threshold from the configured method on finite detection samples."""
    method = config.peak_threshold_method
    if method == 'absolute':
        return float(getattr(config, 'peak_threshold_abs', 0.0))
    if method == 'mean_std':
        mu = np.mean(clean_finite)
        sigma = np.std(clean_finite)
        return float(mu + config.peak_threshold_k * sigma)
    if method == 'percentile':
        thresh = np.nanpercentile(clean_finite, config.peak_threshold_percentile)
        if np.isnan(thresh):
            raise ValueError("Feature Extraction Error: Percentile threshold is NaN.")
        return float(thresh)
    if method == 'median_mad':
        median = np.median(clean_finite)
        mad = np.median(np.abs(clean_finite - median))
        sigma_robust = 1.4826 * mad
        if sigma_robust == 0:
            if config.peak_threshold_k == 0:
                return float(median)
            return float('inf')
        return float(median + config.peak_threshold_k * sigma_robust)
    raise ValueError(
        f"Unknown peak_threshold_method: {config.peak_threshold_method}. "
        "Supported: ['mean_std', 'percentile', 'median_mad', 'absolute']"
    )


def _resolve_prominence_requirement(clean_finite: np.ndarray, config) -> float | None:
    """
    Resolve optional minimum prominence from robust (MAD-based) noise.

    Returns None when disabled; otherwise a non-negative scalar (or +inf).
    """
    k = float(getattr(config, "peak_min_prominence_k", 0.0))
    if k <= 0.0:
        return None
    if clean_finite.size < 2:
        return float("inf")

    median = float(np.median(clean_finite))
    mad = float(np.median(np.abs(clean_finite - median)))
    sigma_robust = float(1.4826 * mad)
    if not np.isfinite(sigma_robust) or sigma_robust <= 0.0:
        # Degenerate robust-noise estimate: fail closed for prominence-enabled calls.
        return float("inf")
    return float(k * sigma_robust)


def _resolve_width_samples(fs_hz: float, config) -> int | None:
    """Resolve optional minimum peak width (seconds -> samples)."""
    width_sec = float(getattr(config, "peak_min_width_sec", 0.0))
    if width_sec <= 0.0:
        return None
    return max(1, int(round(width_sec * fs_hz)))


def _normalize_peak_prefilter_mode(mode_raw: str) -> str:
    mode = str(mode_raw or "none").strip().lower()
    if mode in {"none", "lowpass", "smooth"}:
        return mode
    return "none"


def _resolve_savgol_params(n_samples: int, fs_hz: float) -> tuple[int, int] | None:
    """
    Resolve stable Savitzky-Golay parameters for a trace length and sample rate.

    Window policy:
    - target window ~= 0.75 s worth of samples (for visible denoising at 20 Hz)
    - enforce odd window
    - clamp to largest odd <= n_samples
    - require at least 3 samples
    Polyorder policy:
    - default 2
    - clamp to < window_length for short segments
    """
    try:
        n = int(n_samples)
    except (TypeError, ValueError):
        return None
    if n < 3:
        return None

    if np.isfinite(fs_hz) and fs_hz > 0:
        target = int(round(float(fs_hz) * _SAVGOL_WINDOW_SEC))
    else:
        target = _SAVGOL_MIN_WINDOW
    target = max(_SAVGOL_MIN_WINDOW, target)
    if target % 2 == 0:
        target += 1

    max_odd = n if (n % 2 == 1) else (n - 1)
    if max_odd < 3:
        return None
    window = min(target, max_odd)

    poly = min(_SAVGOL_DEFAULT_POLYORDER, window - 1)
    if poly < 1:
        return None
    return int(window), int(poly)


def _savgol_smooth_trace(trace: np.ndarray, fs_hz: float) -> tuple[np.ndarray, dict]:
    arr = np.asarray(trace, dtype=float)
    out = np.array(arr, copy=True, dtype=float)
    is_finite = np.isfinite(arr)
    meta = {
        "mode": "smooth",
        "prefilter_applied": False,
        "savgol_window_length": None,
        "savgol_polyorder": None,
    }
    if not np.any(is_finite):
        return out, meta

    padded = np.concatenate(([False], is_finite, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        seg = arr[s:e]
        params = _resolve_savgol_params(len(seg), fs_hz)
        if params is None:
            continue
        window, poly = params
        out[s:e] = savgol_filter(seg, window_length=window, polyorder=poly, mode="interp")
        if meta["savgol_window_length"] is None:
            meta["savgol_window_length"] = int(window)
            meta["savgol_polyorder"] = int(poly)
        meta["prefilter_applied"] = True
    return out, meta


def apply_peak_prefilter(trace: np.ndarray, fs_hz: float, config) -> tuple[np.ndarray, dict]:
    """
    Apply configured prefilter to a single trace for peak detection/calling.

    Supported modes:
    - none: raw trace
    - lowpass: existing zero-phase Butterworth behavior
    - smooth: zero-lag Savitzky-Golay smoothing
    """
    arr = np.asarray(trace, dtype=float)
    mode = _normalize_peak_prefilter_mode(getattr(config, "peak_pre_filter", "none"))
    if mode == "lowpass":
        return lowpass_filter(arr, fs_hz, config), {
            "mode": "lowpass",
            "prefilter_applied": True,
            "savgol_window_length": None,
            "savgol_polyorder": None,
        }
    if mode == "smooth":
        return _savgol_smooth_trace(arr, fs_hz)
    return arr, {
        "mode": "none",
        "prefilter_applied": False,
        "savgol_window_length": None,
        "savgol_polyorder": None,
    }


def get_peak_indices_for_trace(
    trace: np.ndarray,
    fs_hz: float,
    config,
    *,
    trace_use: np.ndarray | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Authoritative per-trace event index detection used by both analysis and plotting.

    The returned indices are derived from the same thresholding/min-distance logic as
    extract_features(). Display smoothing in plotting must never modify these indices.
    """
    trace_arr = np.asarray(trace)
    if trace_use is None:
        trace_use_arr, _ = apply_peak_prefilter(trace_arr, fs_hz, config)
    else:
        trace_use_arr = np.asarray(trace_use)

    if threshold is None:
        finite_vals = trace_use_arr[np.isfinite(trace_use_arr)]
        if len(finite_vals) < 2:
            return np.array([], dtype=int)
        threshold = _compute_detection_threshold(finite_vals, config)
    else:
        finite_vals = trace_use_arr[np.isfinite(trace_use_arr)]

    prominence_req = _resolve_prominence_requirement(finite_vals, config)
    width_samples = _resolve_width_samples(fs_hz, config)

    is_valid = np.isfinite(trace_arr)
    padded = np.concatenate(([False], is_valid, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    dist_samples = max(1, int(config.peak_min_distance_sec * fs_hz))
    all_peaks = []
    for s, e in zip(starts, ends):
        seg_y = trace_use_arr[s:e]
        seg_valid = np.isfinite(seg_y)
        if not np.any(seg_valid):
            continue

        padded_run = np.concatenate(([False], seg_valid, [False]))
        diff_run = np.diff(padded_run.astype(int))
        run_starts = np.where(diff_run == 1)[0]
        run_ends = np.where(diff_run == -1)[0]

        for rs, re in zip(run_starts, run_ends):
            run_y = seg_y[rs:re]
            if len(run_y) < 2:
                continue
            peak_kwargs = {
                "height": threshold,
                "distance": dist_samples,
            }
            if prominence_req is not None:
                peak_kwargs["prominence"] = prominence_req
            if width_samples is not None:
                peak_kwargs["width"] = width_samples
            peaks, _ = find_peaks(run_y, **peak_kwargs)
            if len(peaks):
                all_peaks.append((s + rs + peaks).astype(int))

    if not all_peaks:
        return np.array([], dtype=int)
    return np.concatenate(all_peaks)

def extract_features(chunk, config):
    """
    Extracts phasic features from a Chunk object.
    
    Implements a NaN-safe segmented iteration over finite runs.
    Peak detection uses the authoritative detector helper with:
    height threshold, minimum distance, and optional prominence/width criteria.
    Supported threshold methods include mean_std, percentile, and median_mad.
    
    Args:
        chunk (Chunk): The data chunk.
        config (Config): Configuration object.
    
    Returns:
        pd.DataFrame: One row per ROI containing the extracted features.
        Columns: chunk_id, source_file, roi, mean, median, std, mad, peak_count, auc.
    """
    signal_array = get_event_signal_array(chunk, config)
        
    rows = []
    n_rois = signal_array.shape[1]
    
    for i in range(n_rois):
        roi = chunk.channel_names[i]
        trace = signal_array[:, i]
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
        
        # Determine signal to use based on pre-filter setting.
        mode = _normalize_peak_prefilter_mode(getattr(config, "peak_pre_filter", "none"))
        trace_use, _ = apply_peak_prefilter(trace, chunk.fs_hz, config)
        use_filter = mode != "none"
            
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
            
            # Determine threshold from the same helper used by plotting verification.
            if config.peak_threshold_method == 'median_mad' and sigma_robust == 0:
                if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.setdefault('qc_warnings', []).append(
                    f"DEGENERATE[DD4] Zero robust variance in ROI '{roi}'"
                )
            thresh = _compute_detection_threshold(clean_finite, config)
                
            # AUC Baseline
            auc_baseline_method = getattr(config, 'event_auc_baseline', 'zero')
            if auc_baseline_method == 'median':
                auc_baseline = med
            else:
                auc_baseline = 0.0
                
            # Constraints
            peak_indices = get_peak_indices_for_trace(
                trace,
                chunk.fs_hz,
                config,
                trace_use=trace_use,
                threshold=thresh,
            )

            total_peaks = int(len(peak_indices))
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
                            f_warning = (
                                f"DEGENERATE[FILTER_NAN] peak_pre_filter '{mode}' "
                                f"introduced NaNs inside segment {s}:{e} in ROI '{roi}'."
                            )
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
                
                    # Compute AUC for this run. Peak count comes from
                    # get_peak_indices_for_trace() so plotting and analysis share one detector.
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
