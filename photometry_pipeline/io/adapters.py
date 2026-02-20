import pandas as pd
import numpy as np
import os
import warnings
import itertools
from typing import Optional, List, Dict, Tuple
from ..config import Config
from ..core.types import Chunk, SessionTimeMetadata
from dataclasses import asdict
import pathlib
import logging

def _interp_with_nan_policy(time_sec, xp, fp, config, roi_idx, channel_name):
    mask = np.isfinite(xp) & np.isfinite(fp)
    n_nans = len(fp) - np.sum(mask)
    
    if n_nans > 0:
        if getattr(config, 'adapter_value_nan_policy', 'strict') == 'strict':
            raise ValueError(f"NPM strict: NaN values found in {channel_name} for ROI {roi_idx}")
        else:
            xp_use = xp[mask]
            fp_use = fp[mask]
            if len(xp_use) < 2:
                logging.warning(f"NPM mask: Too few points remain for {channel_name} ROI {roi_idx} after NaN masking ({n_nans} NaNs)")
                return np.full_like(time_sec, np.nan), n_nans
            else:
                return np.interp(time_sec, xp_use, fp_use, left=np.nan, right=np.nan), n_nans
    else:
        return np.interp(time_sec, xp, fp, left=np.nan, right=np.nan), 0


def discover_rwd_chunks(root_path: str) -> List[str]:
    """
    Discovers RWD chunks as timestamped subdirectories containing 'fluorescence.csv'.
    
    Rules:
    - Scans immediate subdirectories of root_path.
    - Valid chunk: subdirectory with 'fluorescence.csv'.
    - Sorting: Lexicographical by directory name (YYYY_MM_DD-HH_MM_SS).
    - Ignores: outputs.csv, events.csv, fluorescence-unaligned.csv.
    """
    if not os.path.isdir(root_path):
        raise ValueError(f"RWD Discovery: Root path must be a directory: {root_path}")
        
    chunks = []
    
    # Iterate immediate children
    with os.scandir(root_path) as it:
        entries = sorted([e for e in it if e.is_dir()], key=lambda x: x.name)
        
        for entry in entries:
            target_file = os.path.join(entry.path, "fluorescence.csv")
            if os.path.isfile(target_file):
                chunks.append(target_file)
                
    if not chunks:
        raise ValueError(f"RWD Discovery: No valid RWD chunk directories found in {root_path} (subfolders must contain fluorescence.csv)")
        
    return chunks

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

def _require_strict_check(t_relative: np.ndarray, time_sec: np.ndarray, target_fs_hz: float, context: str):
    """
    Strict Mode Checks:
    1. Monotonicity (Hard Fail)
    2. Coverage (Hard Fail)
    """
    if len(t_relative) == 0:
        raise ValueError(f"{context}: Empty input time array")

    # 1. Monotonicity (Hard Fail in Strict Mode)
    if np.any(np.diff(t_relative) <= 0):
        # Strict mode requires strictly increasing
        raise ValueError(f"{context}: Timestamps not strictly increasing")

    # 2. Coverage
    grid_end = time_sec[-1]
    raw_start = float(np.nanmin(t_relative))
    raw_end = float(np.nanmax(t_relative))
    
    tol = 1.0 / target_fs_hz
    
    if raw_start > (0.0 + tol):
        raise ValueError(f"{context}: raw_start {raw_start:.4f}s > 0.0s (Start Coverage Failure)")
        
    if raw_end < (grid_end - tol):
        raise ValueError(f"{context}: raw_end {raw_end:.4f}s < grid_end {grid_end:.4f}s (End Coverage Failure)")

def _resample_strict(t_rel: np.ndarray, data_in: np.ndarray, config: Config, context: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples to ONE strict grid defined by chunk_duration_sec * target_fs_hz.
    """
    # Grid Construction (Strict)
    n_target = int(np.round(config.chunk_duration_sec * config.target_fs_hz))
    time_sec = np.arange(n_target) / config.target_fs_hz
    
    # Strict Checks
    if not config.allow_partial_final_chunk:
        _require_strict_check(t_rel, time_sec, config.target_fs_hz, context)
        
    # Interpolation
    data_out = np.zeros((n_target, data_in.shape[1]))
    for i in range(data_in.shape[1]):
        data_out[:, i] = np.interp(time_sec, t_rel, data_in[:, i])
        
    return time_sec, data_out

def _ensure_session_time_metadata(chunk: Chunk):
    """
    Ensures 'session_time' metadata key exists and strictly adheres to SessionTimeMetadata schema.
    Backfills missing keys with defaults.
    Always enforces session_id and chunk_index.
    """
    # Explicit schema definition to avoid dataclass instantiation assumptions
    default_dict = {
        "session_id": "",
        "session_start_iso": "",
        "chunk_index": -1,
        "zt0_iso": "",
        "zt_offset_hours": float("nan"),
        "notes": ""
    }
    
    # 1. Ensure dict exists
    if "session_time" not in chunk.metadata:
        chunk.metadata["session_time"] = default_dict.copy()
    
    meta = chunk.metadata["session_time"]
    if not isinstance(meta, dict):
        # Recovery if it's somehow not a dict (cleanup)
        meta = default_dict.copy()
        chunk.metadata["session_time"] = meta
        
    # 2. Backfill missing keys
    for k, v in default_dict.items():
        if k not in meta:
            meta[k] = v
            
    # 3. Enforce derived identity fields
    # "Any user-provided fields... do NOT overwrite... expect session_id/chunk_index"
    # Actually, instructions say "Always set session_id, chunk_index... do NOT overwrite other..."
    # So we ALWAYS overwrite these two.
    meta["session_id"] = str(pathlib.Path(chunk.source_file).stem)
    meta["chunk_index"] = chunk.chunk_id
    
    # Ensure zt_offset_hours is explicitly NaN if default (it is in our defaults, but ensure type safety)
    if "zt_offset_hours" in meta and (meta["zt_offset_hours"] is None or meta["zt_offset_hours"] == ""):
         meta["zt_offset_hours"] = float('nan')

def load_chunk(path: str, format_type: str, config: Config, chunk_id: int) -> Chunk:
    if format_type == 'rwd':
        chunk = _load_rwd(path, config, chunk_id)
    elif format_type == 'npm':
        chunk = _load_npm(path, config, chunk_id)
    else:
        raise ValueError(f"Unknown format: {format_type}")
    
    _ensure_session_time_metadata(chunk)
    # Use config.timestamp_cv_max as tolerance fraction
    chunk.validate(tolerance_frac=config.timestamp_cv_max)
    return chunk

def _load_rwd(path: str, config: Config, chunk_id: int) -> Chunk:
    # Parsing (Simplified for brevity, assuming standard logic)
    header_row = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if config.rwd_time_col in line:
                header_row = i
                break
    if header_row is None: raise ValueError(f"RWD: Missing time col header in {path}")
    
    df = pd.read_csv(path, header=header_row)
    
    if config.rwd_time_col not in df.columns: raise ValueError(f"Missing col: {config.rwd_time_col}")
    if df[config.rwd_time_col].isnull().any(): raise ValueError("Time column contains NaNs")

    t_raw = df[config.rwd_time_col].values
    
    cols = df.columns
    uv_cols = [c for c in cols if c.endswith(config.uv_suffix)]
    sig_cols = [c for c in cols if c.endswith(config.sig_suffix)]
    
    channel_data = []
    for uv_c in uv_cols:
        base = uv_c[: -len(config.uv_suffix)]
        expected_sig = base + config.sig_suffix
        if expected_sig in sig_cols:
            channel_data.append((base, uv_c, expected_sig))
            
    if not channel_data: raise ValueError("No RWD pairs found")
    channel_data.sort(key=lambda x: x[0])
    
    n_rois = len(channel_data)
    names = _create_canonical_names(n_rois)
    roi_map = {names[i]: {"raw_uv": x[1], "raw_sig": x[2]} for i, x in enumerate(channel_data)}
    
    uv_raw = df[[x[1] for x in channel_data]].values
    sig_raw = df[[x[2] for x in channel_data]].values
    
    # Relative Time
    t_rel = t_raw - t_raw[0]
    
    # Strict Resampling
    time_sec, data_out = _resample_strict(t_rel, np.hstack([uv_raw, sig_raw]), config, "RWD strict")
    
    uv_grid = data_out[:, :n_rois]
    sig_grid = data_out[:, n_rois:]
    
    chunk = Chunk(
        chunk_id=chunk_id,
        source_file=path,
        format='rwd',
        time_sec=time_sec,
        uv_raw=uv_grid,
        sig_raw=sig_grid,
        fs_hz=config.target_fs_hz,
        channel_names=names,
        metadata={"roi_map": roi_map}
    )
    # chunk.validate() moved to load_chunk
    return chunk

def _load_npm(path: str, config: Config, chunk_id: int) -> Chunk:
    df = pd.read_csv(path)
    
    time_col = config.npm_system_ts_col if config.npm_time_axis == 'system_timestamp' else config.npm_computer_ts_col
    if time_col not in df.columns: raise ValueError(f"NPM: Missing {time_col}")
        
    t_full = df[time_col].values
    led = df[config.npm_led_col].values
    mask_uv = (led == 1)
    mask_sig = (led == 2)
    t_uv = t_full[mask_uv]
    t_sig = t_full[mask_sig]
    
    if len(t_uv) < 2 or len(t_sig) < 2: raise ValueError("NPM: Insufficient data")
    
    roi_cols = [c for c in df.columns if c.startswith(config.npm_region_prefix) and c.endswith(config.npm_region_suffix)]
    roi_cols.sort()
    if not roi_cols: raise ValueError("NPM: No Region columns")
        
    n_rois = len(roi_cols)
    names = _create_canonical_names(n_rois)
    roi_map = {names[i]: {"raw_col": c} for i, c in enumerate(roi_cols)}
    
    uv_vals = df.loc[mask_uv, roi_cols].values
    sig_vals = df.loc[mask_sig, roi_cols].values
    
    n_value_nans_uv = 0
    n_value_nans_sig = 0
    
    if not config.allow_partial_final_chunk:
        # Strict Mode Logic
        
        # 1. Finite Filtering & Minimum Data Check
        t_uv_f = t_uv[np.isfinite(t_uv)]
        t_sig_f = t_sig[np.isfinite(t_sig)]
        
        if len(t_uv_f) < 2 or len(t_sig_f) < 2:
            raise ValueError("NPM: Insufficient data")
            
        # 2. Strict Monotonicity Check (Pre-Align)
        if np.any(np.diff(t_uv_f) <= 0):
            raise ValueError("NPM UV strict (pre-align): Timestamps not strictly increasing")
        if np.any(np.diff(t_sig_f) <= 0):
            raise ValueError("NPM SIG strict (pre-align): Timestamps not strictly increasing")
            
        # 3. Compute t0 using EARLIEST validated timestamps
        t0 = max(float(np.nanmin(t_uv_f)), float(np.nanmin(t_sig_f)))
        
        # 4. Relative Time (Safe because strict increasing verified)
        t_uv_rel = t_uv - t0
        t_sig_rel = t_sig - t0
        
        # Grid Construction
        n_target = int(np.round(config.chunk_duration_sec * config.target_fs_hz))
        time_sec = np.arange(n_target) / config.target_fs_hz
        grid_end = time_sec[-1]
        
        # Filter -> Check -> Interp
        # 1. Create strict-valid masks (No negative times, no far-future times)
        tol = 1.0 / config.target_fs_hz
        mask_uv_ok = np.isfinite(t_uv_rel) & (t_uv_rel >= 0.0) & (t_uv_rel <= grid_end + tol)
        mask_sig_ok = np.isfinite(t_sig_rel) & (t_sig_rel >= 0.0) & (t_sig_rel <= grid_end + tol)
        
        t_uv_use = t_uv_rel[mask_uv_ok]
        uv_use = uv_vals[mask_uv_ok, :]
        
        t_sig_use = t_sig_rel[mask_sig_ok]
        sig_use = sig_vals[mask_sig_ok, :]
        
        # 2. Strict Check on USED arrays
        _require_strict_check(t_uv_use, time_sec, config.target_fs_hz, "NPM UV strict")
        _require_strict_check(t_sig_use, time_sec, config.target_fs_hz, "NPM SIG strict")
        
        # 3. Interpolate ONLY using filtered arrays
        uv_out = np.zeros((n_target, n_rois))
        sig_out = np.zeros((n_target, n_rois))
        
        for i in range(n_rois):
            uv_val, nans_uv = _interp_with_nan_policy(time_sec, t_uv_use, uv_use[:, i], config, i, "UV")
            uv_out[:, i] = uv_val
            n_value_nans_uv += nans_uv
            
            sig_val, nans_sig = _interp_with_nan_policy(time_sec, t_sig_use, sig_use[:, i], config, i, "SIG")
            sig_out[:, i] = sig_val
            n_value_nans_sig += nans_sig
            
    else:
        # Permissive Mode (Original/Fallback)
        
        # C: Finite-Safe Permissive Check
        mask_uv_fin = np.isfinite(t_uv)
        mask_sig_fin = np.isfinite(t_sig)
        
        t_uv_f = t_uv[mask_uv_fin]
        t_sig_f = t_sig[mask_sig_fin]
        
        if len(t_uv_f) < 2 or len(t_sig_f) < 2:
             raise ValueError(f"NPM Permissive: Insufficient finite data in {path}")

        if np.any(np.diff(t_uv_f) <= 0):
             raise ValueError(f"NPM Permissive: UV timestamps not strictly increasing (finite subset) in {path}")
        if np.any(np.diff(t_sig_f) <= 0):
             raise ValueError(f"NPM Permissive: SIG timestamps not strictly increasing (finite subset) in {path}")

        # Overlap uses validated finite starts
        t0 = max(t_uv_f[0], t_sig_f[0])
        
        # Grid Construction
        n_target = int(np.round(config.chunk_duration_sec * config.target_fs_hz))
        time_sec = np.arange(n_target) / config.target_fs_hz
        
        # Bounds logic (using finite support)
        # Note: relative to t0
        uv_min_t, uv_max_t = t_uv_f[0] - t0, t_uv_f[-1] - t0
        sig_min_t, sig_max_t = t_sig_f[0] - t0, t_sig_f[-1] - t0
        
        # Interpolate
        uv_out = np.zeros((n_target, n_rois))
        sig_out = np.zeros((n_target, n_rois))
        
        # Prepare relative arrays from FULL (non-finite preserved for shape? No, interp needs valid xp)
        # Actually interp needs xp to be increasing. Non-finite in xp breaks it?
        # Standard np.interp expects increasing xp.
        # So we MUST use t_uv_f for xp. And corresponding values.
        
        uv_vals_f = uv_vals[mask_uv_fin]
        sig_vals_f = sig_vals[mask_sig_fin]
        
        t_uv_rel = t_uv_f - t0
        t_sig_rel = t_sig_f - t0
        
        for i in range(n_rois):
            # UV
            uv_val, nans_uv = _interp_with_nan_policy(time_sec, t_uv_rel, uv_vals_f[:, i], config, i, "UV")
            uv_out[:, i] = uv_val
            n_value_nans_uv += nans_uv
            
            # SIG
            sig_val, nans_sig = _interp_with_nan_policy(time_sec, t_sig_rel, sig_vals_f[:, i], config, i, "SIG")
            sig_out[:, i] = sig_val
            n_value_nans_sig += nans_sig
        
    chunk = Chunk(
        chunk_id=chunk_id,
        source_file=path,
        format='npm',
        time_sec=time_sec,
        uv_raw=uv_out,
        sig_raw=sig_out,
        fs_hz=config.target_fs_hz,
        channel_names=names,
        metadata={
            "roi_map": roi_map,
            "n_value_nans_uv": int(n_value_nans_uv),
            "n_value_nans_sig": int(n_value_nans_sig),
            "adapter_value_nan_policy": getattr(config, 'adapter_value_nan_policy', 'strict')
        }
    )
    # chunk.validate() moved to load_chunk
    return chunk
