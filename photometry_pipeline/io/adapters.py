
import pandas as pd
import numpy as np
import os
import warnings
import itertools
from typing import Optional, List, Dict, Tuple
from ..config import Config
from ..core.types import Chunk

def sniff_format(path: str, config: Config) -> Optional[str]:
    """
    Detects if a file is 'rwd' or 'npm' based on header/column analysis.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            head_iter = itertools.islice(f, 50)
            head = list(head_iter)
        
        for line in head:
            if config.rwd_time_col in line:
                return 'rwd'

        if head:
            first_line = head[0]
            if config.npm_frame_col in first_line and config.npm_system_ts_col in first_line:
                return 'npm'
        return None
    except Exception:
        return None

def _create_canonical_names(n_rois: int) -> List[str]:
    return [f"Region{i}" for i in range(n_rois)]

def _require_strict_coverage(t_relative: np.ndarray, time_sec: np.ndarray, target_fs_hz: float, context: str):
    """
    Ensures input data covers the entire target grid.
    Endpoint extrapolation is forbidden in strict mode.
    """
    if len(t_relative) == 0:
        raise ValueError(f"{context}: Empty input time array")
        
    grid_end = time_sec[-1]
    
    # 1) Use max, not last element
    raw_end = float(np.nanmax(t_relative))
    
    # 2) Monotonicity warning
    if np.any(np.diff(t_relative) < 0):
        warnings.warn(f"{context}: Timestamps are not monotonic!", UserWarning)
    
    tol = 1.0 / target_fs_hz
    
    # Check coverage
    if raw_end < (grid_end - tol):
        raise ValueError(f"{context}: raw_end {raw_end:.4f}s < grid_end {grid_end:.4f}s")

def _resample_strict_rwd(t_raw: np.ndarray, data_in: np.ndarray, config: Config, context: str) -> Tuple[np.ndarray, np.ndarray]:
    # 1. Build Strict Grid
    n_target = int(np.round(config.chunk_duration_sec * config.target_fs_hz))
    time_sec = np.arange(n_target) / config.target_fs_hz
    
    # 2. Build Raw Relative Time
    t_relative = t_raw - t_raw[0]
    
    # 3. Check Coverage
    if not config.allow_partial_final_chunk:
        _require_strict_coverage(t_relative, time_sec, config.target_fs_hz, context)
        
    # 4. Interpolate
    data_out = np.zeros((n_target, data_in.shape[1]))
    for i in range(data_in.shape[1]):
        data_out[:, i] = np.interp(time_sec, t_relative, data_in[:, i])
        
    return time_sec, data_out

def load_chunk(path: str, format_type: str, config: Config, chunk_id: int) -> Chunk:
    if format_type == 'rwd':
        return _load_rwd(path, config, chunk_id)
    elif format_type == 'npm':
        return _load_npm(path, config, chunk_id)
    else:
        raise ValueError(f"Unknown format: {format_type}")

def _load_rwd(path: str, config: Config, chunk_id: int) -> Chunk:
    # Parsing
    header_row = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if config.rwd_time_col in line:
                header_row = i
                break
    
    if header_row is None:
        raise ValueError(f"RWD: Could not find time column '{config.rwd_time_col}' in {path}")
        
    df = pd.read_csv(path, header=header_row)
    
    if config.rwd_time_col not in df.columns:
        raise ValueError(f"Column missing: {config.rwd_time_col}")
    if df[config.rwd_time_col].isnull().any():
         raise ValueError("Time column contains NaNs")

    t_raw = df[config.rwd_time_col].values
    
    # Channels
    cols = df.columns
    uv_cols = [c for c in cols if c.endswith(config.uv_suffix)]
    sig_cols = [c for c in cols if c.endswith(config.sig_suffix)]
    
    channel_data = [] # (base, uv, sig)
    for uv_c in uv_cols:
        base = uv_c[: -len(config.uv_suffix)]
        expected_sig = base + config.sig_suffix
        if expected_sig in sig_cols:
            channel_data.append((base, uv_c, expected_sig))
            
    if not channel_data:
        raise ValueError("No matched UV/Signal pairs found.")

    channel_data.sort(key=lambda x: x[0])
    n_rois = len(channel_data)
    canonical_names = _create_canonical_names(n_rois)
    
    roi_map = {}
    for i, (base, uv_c, sig_c) in enumerate(channel_data):
        roi_map[canonical_names[i]] = {"raw_uv": uv_c, "raw_sig": sig_c}
        
    uv_raw_cols = [x[1] for x in channel_data]
    sig_raw_cols = [x[2] for x in channel_data]
    
    uv_vals = df[uv_raw_cols].values
    sig_vals = df[sig_raw_cols].values
    
    # Strict Resampling
    # duration_tolerance warning
    raw_dur = t_raw[-1] - t_raw[0]
    expected = config.chunk_duration_sec
    if raw_dur < expected * (1.0 - config.duration_tolerance_frac):
         warnings.warn(f"RWD: Raw duration {raw_dur:.2f}s short of expected {expected}s")
         
    time_sec, data_out = _resample_strict_rwd(t_raw, np.hstack([uv_vals, sig_vals]), config, "RWD strict")
    
    uv_grid = data_out[:, :n_rois]
    sig_grid = data_out[:, n_rois:]
    
    chunk = Chunk(
        chunk_id=chunk_id, source_file=path, format='rwd',
        time_sec=time_sec, uv_raw=uv_grid, sig_raw=sig_grid,
        fs_hz=config.target_fs_hz, channel_names=canonical_names,
        metadata={"roi_map": roi_map}
    )
    chunk.validate()
    return chunk

def _load_npm(path: str, config: Config, chunk_id: int) -> Chunk:
    df = pd.read_csv(path)
    
    time_col = config.npm_system_ts_col if config.npm_time_axis == 'system_timestamp' else config.npm_computer_ts_col
    if time_col not in df.columns: raise ValueError(f"NPM: Time column '{time_col}' missing")
        
    t_full = df[time_col].values
    led_state = df[config.npm_led_col].values
    
    mask_uv = (led_state == 1)
    mask_sig = (led_state == 2)
    
    t_uv = t_full[mask_uv]
    t_sig = t_full[mask_sig]
    
    if len(t_uv) < 2 or len(t_sig) < 2: raise ValueError("NPM: Insufficient data points")
        
    # ROIs
    roi_cols = [c for c in df.columns if c.startswith(config.npm_region_prefix) and c.endswith(config.npm_region_suffix)]
    roi_cols.sort()
    if not roi_cols: raise ValueError("NPM: No Region columns found")
        
    n_rois = len(roi_cols)
    canonical_names = _create_canonical_names(n_rois)
    roi_map = {canonical_names[i]: {"raw_col": col} for i, col in enumerate(roi_cols)}
    
    uv_subset = df.loc[mask_uv, roi_cols].values
    sig_subset = df.loc[mask_sig, roi_cols].values
    
    # 1. Compute Overlap Bounds
    t0 = max(t_uv[0], t_sig[0])
    t1 = min(t_uv[-1], t_sig[-1])
    overlap_dur = t1 - t0
    
    if overlap_dur <= 0:
        raise ValueError("NPM: No overlap between UV and Signal")
    
    # 2. Build Strict Grid
    n_target = int(np.round(config.chunk_duration_sec * config.target_fs_hz))
    time_sec = np.arange(n_target) / config.target_fs_hz
    grid_end = time_sec[-1]
    
    # 3. Strict Check
    if not config.allow_partial_final_chunk:
        tol = 1.0 / config.target_fs_hz
        if overlap_dur < (grid_end - tol):
             raise ValueError(f"NPM strict: overlap insufficient ({overlap_dur:.4f}s < grid_end {grid_end:.4f}s)")
             
    # 4. Convert to relative
    t_uv_rel = t_uv - t0
    t_sig_rel = t_sig - t0
    
    # 5. Crop
    mask_uv_valid = (t_uv_rel >= 0) & (t_uv_rel <= overlap_dur)
    mask_sig_valid = (t_sig_rel >= 0) & (t_sig_rel <= overlap_dur)
    
    t_uv_crop = t_uv_rel[mask_uv_valid]
    uv_vals_crop = uv_subset[mask_uv_valid]
    
    t_sig_crop = t_sig_rel[mask_sig_valid]
    sig_vals_crop = sig_subset[mask_sig_valid]
    
    # 6. Check Cropped Coverage
    if not config.allow_partial_final_chunk:
        _require_strict_coverage(t_uv_crop, time_sec, config.target_fs_hz, "NPM UV strict")
        _require_strict_coverage(t_sig_crop, time_sec, config.target_fs_hz, "NPM SIG strict")
        
    # 7. Interpolate
    uv_out = np.zeros((n_target, n_rois))
    sig_out = np.zeros((n_target, n_rois))
    
    for i in range(n_rois):
        uv_out[:, i] = np.interp(time_sec, t_uv_crop, uv_vals_crop[:, i])
        sig_out[:, i] = np.interp(time_sec, t_sig_crop, sig_vals_crop[:, i])
        
    chunk = Chunk(
        chunk_id=chunk_id, source_file=path, format='npm',
        time_sec=time_sec, uv_raw=uv_out, sig_raw=sig_out,
        fs_hz=config.target_fs_hz, channel_names=canonical_names,
        metadata={"roi_map": roi_map}
    )
    chunk.validate()
    return chunk
