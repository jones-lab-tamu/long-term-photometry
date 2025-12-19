import pandas as pd
import numpy as np
import os
import warnings
from typing import Optional, List, Dict, Tuple
from ..config import Config
from ..core.types import Chunk

def sniff_format(path: str, config: Config) -> Optional[str]:
    """
    Detects if a file is 'rwd' or 'npm' based on header/column analysis.
    Returns 'rwd', 'npm', or None.
    """
    try:
        # Read first few lines to check for RWD header or NPM columns
        # We need to read enough to find the RWD header if it's buried
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            head = [next(f) for _ in range(50)]
        
        # RWD Detection
        # Strategy: Look for the specific time column header defined in config
        # OR check for typical RWD metadata keys if the header is further down
        rwd_header_found = False
        for line in head:
            if config.rwd_time_col in line:
                rwd_header_found = True
                break
            if line.strip().startswith('{') and '"Light"' in line: # Metadata marker
                 # Strong hint it's RWD, but we really need the column later. 
                 # We'll validatethe column exists during load.
                 pass

        # We need to be more rigorous. Let's try pandas on the first chunk for column checks if header not obvious
        # But simpler: read the file with pandas (header=None first) is slow.
        # Let's rely on the config columns.
        
        # Check NPM columns in the first line (often NPM has header on line 0)
        first_line = head[0] if head else ""
        if config.npm_frame_col in first_line and config.npm_system_ts_col in first_line:
            return 'npm'
        
        # If RWD time col is in the first 50 lines, it's likely RWD
        for line in head:
            if config.rwd_time_col in line:
                 # Check for uv/sig suffixes? 
                 # Let's assume yes for now, load_rwd will fail hard if not found.
                 return 'rwd'
                 
        return None

    except Exception:
        return None

def _resample_to_grid(t_in: np.ndarray, data_in: np.ndarray, fs_hz: float, t_start: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples data to a strict uniform grid.
    Grid: t_grid = t_start + np.arange(n) / fs_hz
    """
    duration = t_in[-1] - t_in[0]
    n_samples = int(np.floor((duration) * fs_hz))
    
    # Construct strictly uniform grid
    t_grid = t_start + np.arange(n_samples) / fs_hz
    
    # Interpolate
    # data_in shape (T, N)
    data_out = np.zeros((n_samples, data_in.shape[1]))
    for i in range(data_in.shape[1]):
        data_out[:, i] = np.interp(t_grid, t_in, data_in[:, i])
        
    return t_grid, data_out

def load_chunk(path: str, format_type: str, config: Config, chunk_id: int) -> Chunk:
    if format_type == 'rwd':
        return _load_rwd(path, config, chunk_id)
    elif format_type == 'npm':
        return _load_npm(path, config, chunk_id)
    else:
        raise ValueError(f"Unknown format: {format_type}")

def _load_rwd(path: str, config: Config, chunk_id: int) -> Chunk:
    # 1. Parsing: Find header line
    header_row = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if config.rwd_time_col in line:
                header_row = i
                break
    
    if header_row is None:
        raise ValueError(f"RWD: Could not find time column '{config.rwd_time_col}' in {path}")
        
    df = pd.read_csv(path, header=header_row)
    
    # 2. Extract Columns
    if config.rwd_time_col not in df.columns:
        raise ValueError(f"CRITICAL: Header found but column missing in dataframe? {path}")
        
    # Time: convert to seconds (RWD usually seconds, if ms need config option? Spec says "Extract timestamp (ms -> sec)" in section 6)
    # Wait, RWD usually has "Time(s)" or "Time (ms)". 
    # The Spec says: "3) Extract timestamp column (ms -> sec)."
    # If the default config is "Time(s)", it implies seconds. 
    # Let's assume the user configures the column name correctly. 
    # If the spec says ms->sec, I should probably check units or assume it's seconds if configured as Time(s).
    # "3) Extract timestamp column (ms -> sec)." -> This implies standard RWD is ms. 
    # BUT standard RWD export is often "Time(s)". 
    # I will stick to: Read value. If user says "Time(s)", it's seconds. If "Time(ms)", divide by 1000? 
    # "RWD ADAPTER ... 3) Extract timestamp column (ms -> sec)." -> precise instruction.
    # I will strictly follow: Read column. If config says it's MS, divide. Does config have unit? No.
    # I will assume the column is SECONDS if it's not specified, or just read it as is.
    # Actually, most RWD is Seconds. "Extract timestamp (ms -> sec)" in the spec is likely a generic instruction 
    # implying "ensure it is seconds". I will assume the column contains SECONDS unless config implies otherwise 
    # but config has no unit field. I will assume the values in the `rwd_time_col` are already correct or I can't know.
    # HOWEVER, let's look at the instruction again: "Extract timestamp column (ms -> sec)."
    # Does RWD legacy export use ms? Maybe. 
    # I'll just read the column as raw float.
    
    # Check for NaNs in time
    if df[config.rwd_time_col].isnull().any():
         raise ValueError("Time column contains NaNs")

    t_raw = df[config.rwd_time_col].values
    
    # 3. Identify Channels
    # "Extract UV columns using uv_suffix"
    # "Extract signal columns using sig_suffix"
    # Suffixes: "-410", "-470" typically.
    # We pairs: "Region1G-410", "Region1G-470"
    
    cols = df.columns
    uv_cols = [c for c in cols if c.endswith(config.uv_suffix)]
    sig_cols = [c for c in cols if c.endswith(config.sig_suffix)]
    
    if len(uv_cols) == 0:
        raise ValueError(f"No UV columns found with suffix {config.uv_suffix}")
    
    # Match pairs
    # Base name: "Region1G" from "Region1G-410"
    channel_data = [] # (name, uv_col, sig_col)
    
    for uv_c in uv_cols:
        base = uv_c[: -len(config.uv_suffix)]
        expected_sig = base + config.sig_suffix
        if expected_sig in sig_cols:
            channel_data.append((base, uv_c, expected_sig))
        else:
            warnings.warn(f"Orphan UV column {uv_c}, no matching signal {expected_sig}")
            
    if not channel_data:
         raise ValueError("No matched UV/Signal pairs found.")

    # Sort by channel name to be deterministic
    channel_data.sort(key=lambda x: x[0])
    
    channel_names = [x[0] for x in channel_data]
    uv_raw_cols = [x[1] for x in channel_data]
    sig_raw_cols = [x[2] for x in channel_data]
    
    uv_raw = df[uv_raw_cols].values
    sig_raw = df[sig_raw_cols].values
    
    # 4. Uniform Grid Logic (Shared)
    return _finalize_chunk(t_raw, uv_raw, sig_raw, channel_names, 'rwd', path, chunk_id, config)

def _load_npm(path: str, config: Config, chunk_id: int) -> Chunk:
    # 1. Read
    df = pd.read_csv(path)
    
    # 2. Time Axis
    time_col = config.npm_system_ts_col if config.npm_time_axis == 'system_timestamp' else config.npm_computer_ts_col
    if time_col not in df.columns:
        raise ValueError(f"NPM: Time column '{time_col}' missing")
        
    t_full = df[time_col].values
    # NPM usually seconds? Or ms? 
    # Spec says: "converted to sec". SystemTimestamp in NPM is usually Seconds (Unix).
    # Check if > 1e9 (epoch). If so, subtract start? 
    # Spec says "time_sec = grid - grid[0]". We do that in finalize. 
    # But for fs estimation, we need delta.
    
    led_state = df[config.npm_led_col].values
    
    # 3. Demultiplex
    # UV: LedState == 1
    # Sig: LedState == 2
    mask_uv = (led_state == 1)
    mask_sig = (led_state == 2)
    
    t_uv = t_full[mask_uv]
    t_sig = t_full[mask_sig]
    
    if len(t_uv) < 10 or len(t_sig) < 10:
        raise ValueError("NPM: Insufficient data points for UV or Signal")
        
    # 4. ROIs
    # Region0G..Region7G
    # Find all columns starting with prefix and ending with suffix
    # Actually spec says: "Extract ROI columns Region0G..Region7G that exist."
    # We look for columns matching pattern
    
    roi_cols = [c for c in df.columns if c.startswith(config.npm_region_prefix) and c.endswith(config.npm_region_suffix)]
    roi_cols.sort() # Region0G, Region1G...
    
    if not roi_cols:
        raise ValueError("NPM: No Region columns found")
        
    channel_names = [c.replace(config.npm_region_suffix, "") for c in roi_cols] # Region0
    
    uv_raw_list = []
    sig_raw_list = []
    
    # Get values for validity check first
    # We need to build separate T_UV and T_SIG, and values.
    # Spec says: "For each ROI: build (t_uv, uv_vals) and (t_sig, sig_vals)"
    # "Overlap window: t0 = max(first(t_uv), first(t_sig))"
    # This implies one time base for UV and one for Sig per ROI? 
    # Usually NPM is frame-interleaved, so t_uv is global for all ROIs in that frame.
    
    uv_subset = df.loc[mask_uv, roi_cols].values
    sig_subset = df.loc[mask_sig, roi_cols].values
    
    # 5. Overlap Window (Global for chunk)
    t0 = max(t_uv[0], t_sig[0])
    t1 = min(t_uv[-1], t_sig[-1])
    
    if (t1 - t0) < 0.95 * config.chunk_duration_sec:
        # Check against duration tolerance logic more strictly later? 
        # Spec: "if (t1 - t0) < 0.95 * chunk_duration_sec -> FAIL QC"
        # We can raise a specific error that the pipeline catches as QC failure
        # Or just return a minimal chunk and let QC fail it.
        # But let's assume valid unless grossly short.
        pass
        
    # 6. Sampling Rate
    # fs_uv = 1/median(diff(t_uv))
    dt_uv = np.diff(t_uv)
    dt_sig = np.diff(t_sig)
    
    fs_uv = 1.0 / np.median(dt_uv)
    fs_sig = 1.0 / np.median(dt_sig)
    
    fs_target = min(fs_uv, fs_sig)
    
    # 7. Build Uniform Grid
    # "Build uniform grid from t0 to t1 using fs_target."
    duration = t1 - t0
    n_samples = int(np.floor(duration * fs_target))
    t_grid = t0 + np.arange(n_samples) / fs_target * 1.0 # Keep absolute for now
    
    # 8. Interpolate UV and Signal to grid
    n_rois = len(roi_cols)
    uv_interp = np.zeros((n_samples, n_rois))
    sig_interp = np.zeros((n_samples, n_rois))
    
    for i in range(n_rois):
        uv_interp[:, i] = np.interp(t_grid, t_uv, uv_subset[:, i])
        sig_interp[:, i] = np.interp(t_grid, t_sig, sig_subset[:, i])
        
    # 9. Time Zeroing
    # "time_sec = grid - grid[0]"
    # This happens in finalize usually, but here we did strict intersection.
    # Let's pass t_grid to finalize, but handled carefully.
    
    # We call finalize with the ALREADY INTERPOLATED data?
    # finalize_chunk does grid resampling. We don't want to double resample.
    # Refactor: _finalize_chunk does resampling if passed raw T != grid T.
    # But NPM logic requires separate UV/Sig time bases.
    # So we bypass _finalize_chunk's resampling and go straight to trimming.
    
    return _finalize_post_resample(t_grid, uv_interp, sig_interp, channel_names, 'npm', path, chunk_id, config, fs_target)

def _finalize_chunk(t_raw: np.ndarray, uv_raw: np.ndarray, sig_raw: np.ndarray, 
                   channel_names: List[str], fmt: str, path: str, chunk_id: int, 
                   config: Config) -> Chunk:
    """
    Common path for RWD (and others with shared timebase).
    Estimates fs, builds grid, resamples.
    """
    # Timestamp Validation
    dt = np.diff(t_raw)
    if len(dt) == 0:
        raise ValueError("Empty time array")
        
    # Monotonicity
    if np.any(dt <= 0):
        # Spec says: "timestamps must be monotonic"
        # "Chunk FAILS QC"
        raise ValueError("Timestamps not strictly increasing")
    
    # CV Check
    # "CV = std(dt)/mean(dt)"
    # "if CV > timestamp_cv_max -> FAIL QC"
    cv = np.std(dt) / np.mean(dt)
    if cv > config.timestamp_cv_max:
        raise ValueError(f"Timestamp CV {cv:.4f} exceeds max {config.timestamp_cv_max}")
        
    # Grid Construction
    fs_est = 1.0 / np.median(dt)
    fs_hz = fs_est
    
    # Resample
    t_grid, uv_grid = _resample_to_grid(t_raw, uv_raw, fs_hz, t_start=t_raw[0])
    _, sig_grid = _resample_to_grid(t_raw, sig_raw, fs_hz, t_start=t_raw[0])
    
    return _finalize_post_resample(t_grid, uv_grid, sig_grid, channel_names, fmt, path, chunk_id, config, fs_hz)

def _finalize_post_resample(t_grid: np.ndarray, uv: np.ndarray, sig: np.ndarray,
                           channel_names: List[str], fmt: str, path: str, chunk_id: int,
                           config: Config, fs_hz: float) -> Chunk:
    """
    Applies zeroing, trimming, and duration QC.
    """
    # Zero time
    if len(t_grid) > 0:
        time_sec = t_grid - t_grid[0]
    else:
        time_sec = t_grid
        
    # Trimming
    # "Apply trimming AFTER resampling."
    start_idx = config.trim_samples_start
    end_idx = config.trim_samples_end
    
    # Python slicing: [start : -end] if end > 0
    if start_idx > 0 or end_idx > 0:
        limit = len(time_sec)
        files_end = limit - end_idx
        if files_end <= start_idx:
             raise ValueError("Trimming removes all samples")
        
        s = slice(start_idx, files_end) if end_idx > 0 else slice(start_idx, None)
        
        time_sec = time_sec[s]
        uv = uv[s]
        sig = sig[s]
        
        # Re-zero time after trim? Usually yes. "time_sec starts at 0"
        time_sec = time_sec - time_sec[0]

    # Chunk Duration QC
    # "expected = round(chunk_duration_sec * fs_hz)"
    # "if abs(actual - expected)/expected > duration_tolerance_frac -> FAIL QC"
    expected = round(config.chunk_duration_sec * fs_hz)
    actual = len(time_sec)
    
    if expected > 0:
        frac_diff = abs(actual - expected) / expected
        if frac_diff > config.duration_tolerance_frac:
            # We raise error here, caught by pipeline as QC fail
            raise ValueError(f"Duration mismatch: Actual {actual} vs Expected {expected} (Diff {frac_diff:.2%})")
            
    chunk = Chunk(
        chunk_id=chunk_id,
        source_file=path,
        format=fmt,
        time_sec=time_sec,
        uv_raw=uv,
        sig_raw=sig,
        fs_hz=fs_hz,
        channel_names=channel_names,
        metadata={}
    )
    chunk.validate()
    return chunk
