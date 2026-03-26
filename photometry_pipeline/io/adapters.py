import pandas as pd
import numpy as np
import os
import glob
import warnings
import itertools
import csv
import re
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from ..config import Config
from ..core.types import Chunk, SessionTimeMetadata
from ..core.utils import natural_sort_key
from dataclasses import asdict
import pathlib
import logging


_RWD_METADATA_LED_KEYS: Tuple[str, ...] = ("Led410Enable", "Led470Enable", "Led560Enable")


def _unique_ordered(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _rwd_time_candidates(config: Config) -> List[str]:
    return _unique_ordered(
        [
            str(getattr(config, "rwd_time_col", "")).strip(),
            "TimeStamp",
            "Time(s)",
            "Timestamp",
            "Time",
        ]
    )


def _rwd_suffix_candidates(config: Config) -> Tuple[List[str], List[str]]:
    uv = _unique_ordered(
        [
            str(getattr(config, "uv_suffix", "")).strip(),
            "-410",
            "-415",
        ]
    )
    sig = _unique_ordered(
        [
            str(getattr(config, "sig_suffix", "")).strip(),
            "-470",
        ]
    )
    return uv, sig


def _parse_csv_header_fields(line: str) -> List[str]:
    try:
        fields = next(csv.reader([line]))
    except Exception:
        fields = line.split(",")
    return [str(field).strip().lstrip("\ufeff") for field in fields]


def _extract_rwd_channel_pairs(
    columns: List[str], uv_suffixes: List[str], sig_suffixes: List[str]
) -> List[Tuple[str, str, str]]:
    col_set = set(columns)
    pairs: List[Tuple[str, str, str]] = []
    seen_bases = set()

    for uv_suffix in uv_suffixes:
        for sig_suffix in sig_suffixes:
            for col in columns:
                if not col.endswith(uv_suffix):
                    continue
                base = col[: -len(uv_suffix)]
                if not base or base in seen_bases:
                    continue
                expected_sig = f"{base}{sig_suffix}"
                if expected_sig in col_set:
                    pairs.append((base, col, expected_sig))
                    seen_bases.add(base)

    pairs.sort(key=lambda x: natural_sort_key(x[0]))
    return pairs


def _looks_like_rwd_channel_header(
    columns: List[str], uv_suffixes: List[str], sig_suffixes: List[str]
) -> bool:
    candidate_suffixes = tuple(_unique_ordered(uv_suffixes + sig_suffixes))
    return any(col.endswith(candidate_suffixes) for col in columns)


def _detect_rwd_header(path: str, config: Config) -> Optional[Tuple[int, str]]:
    time_candidates = _rwd_time_candidates(config)
    uv_suffixes, sig_suffixes = _rwd_suffix_candidates(config)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(itertools.islice(f, 60)):
                columns = _parse_csv_header_fields(line)
                if len(columns) < 3:
                    continue

                time_col = next((c for c in time_candidates if c in columns), None)
                if time_col is None:
                    continue

                if _looks_like_rwd_channel_header(columns, uv_suffixes, sig_suffixes):
                    return idx, time_col
    except Exception:
        return None

    return None


def _extract_rwd_metadata_context(path: str, header_row_idx: int) -> Tuple[Optional[float], Optional[int]]:
    """
    Extract vendor metadata fields from the row above detected RWD header.

    Returns (metadata_fps, enabled_excitation_count). Each value may be None
    when unavailable.
    """
    if header_row_idx <= 0:
        return None, None

    meta_cells: Optional[List[str]] = None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == header_row_idx - 1:
                    meta_cells = [str(cell).strip().lstrip("\ufeff") for cell in row]
                    break
    except Exception:
        return None, None

    if not meta_cells:
        return None, None

    meta_text = ",".join(cell for cell in meta_cells if cell)
    if not meta_text:
        return None, None

    fps_val: Optional[float] = None
    m_fps = re.search(r'"?Fps"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', meta_text, flags=re.IGNORECASE)
    if m_fps:
        try:
            parsed = float(m_fps.group(1))
            if np.isfinite(parsed) and parsed > 0:
                fps_val = parsed
        except Exception:
            fps_val = None

    enabled_count: Optional[int] = None
    seen_any_led = False
    enabled_acc = 0
    for led_key in _RWD_METADATA_LED_KEYS:
        m_led = re.search(
            rf'"?{re.escape(led_key)}"?\s*[:=]\s*(true|false|1|0)',
            meta_text,
            flags=re.IGNORECASE,
        )
        if not m_led:
            continue
        seen_any_led = True
        token = str(m_led.group(1)).strip().lower()
        if token in {"true", "1"}:
            enabled_acc += 1
    if seen_any_led:
        enabled_count = int(enabled_acc)

    return fps_val, enabled_count


def _resolve_rwd_timestamp_scale(
    t_raw: np.ndarray,
    metadata_fps: Optional[float],
    enabled_excitation_count: Optional[int],
) -> Tuple[float, str]:
    """
    Resolve raw RWD timestamp unit scale to canonical seconds.

    Returns (scale_to_seconds, unit_label), where scale_to_seconds is 1.0 for
    second timestamps and 0.001 for millisecond timestamps.
    """
    if len(t_raw) < 2:
        return 1.0, "seconds"

    dt = np.diff(t_raw.astype(float))
    if np.any(dt <= 0):
        raise ValueError("RWD: Timestamps not strictly increasing.")
    med_dt = float(np.median(dt))
    if not np.isfinite(med_dt) or med_dt <= 0:
        raise ValueError("RWD: Invalid timestamp cadence.")

    fs_seconds = 1.0 / med_dt
    fs_milliseconds = 1000.0 / med_dt

    if metadata_fps is not None and np.isfinite(metadata_fps) and metadata_fps > 0:
        if enabled_excitation_count is not None and enabled_excitation_count <= 0:
            raise ValueError(
                "RWD: Metadata indicates no enabled excitation LEDs; cannot reconcile FPS."
            )
        # Vendor multiplex semantics: metadata Fps can represent total frame
        # cadence across enabled excitation states, while row timestamps are
        # paired-row cadence. Accept direct match or multiplex-adjusted match.
        multiplier = (
            enabled_excitation_count
            if (enabled_excitation_count is not None and enabled_excitation_count > 1)
            else 1
        )
        rel_tol = 0.03

        def _match(candidate_row_fs: float) -> bool:
            direct_ok = abs(candidate_row_fs - metadata_fps) / metadata_fps <= rel_tol
            multiplex_ok = abs((candidate_row_fs * multiplier) - metadata_fps) / metadata_fps <= rel_tol
            return direct_ok or multiplex_ok

        sec_match = _match(fs_seconds)
        ms_match = _match(fs_milliseconds)

        if sec_match and not ms_match:
            return 1.0, "seconds"
        if ms_match and not sec_match:
            return 0.001, "milliseconds"
        if sec_match and ms_match:
            raise ValueError(
                "RWD: Ambiguous timestamp units; both second and millisecond "
                f"interpretations match metadata FPS={metadata_fps:.6f}."
            )
        raise ValueError(
            "RWD: Timestamps incompatible with metadata FPS. "
            f"median_dt={med_dt:.6f}, fs_seconds={fs_seconds:.6f}, "
            f"fs_milliseconds={fs_milliseconds:.6f}, metadata_fps={metadata_fps:.6f}, "
            f"enabled_excitation_count={enabled_excitation_count}."
        )

    # Backward-compatible fallback when metadata cannot disambiguate:
    # keep historical assumption (seconds) unless cadence is clearly in ms.
    if med_dt >= 5.0:
        return 0.001, "milliseconds"
    return 1.0, "seconds"

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


def discover_csv_or_rwd_chunks(input_dir: str, file_glob: str = "*.csv") -> List[str]:
    """Discover top-level CSV files, falling back to chunked RWD roots."""
    search_pattern = os.path.join(input_dir, file_glob)
    file_list = glob.glob(search_pattern)
    if file_list:
        return file_list
    try:
        return discover_rwd_chunks(input_dir)
    except Exception:
        return []

def sniff_format(path: str, config: Config) -> Optional[str]:
    """
    Detects if a file is 'rwd' or 'npm' based on header/column analysis.
    """
    try:
        if _detect_rwd_header(path, config) is not None:
            return 'rwd'

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            head_iter = itertools.islice(f, 50)
            head = list(head_iter)
        if head:
            first_cols = _parse_csv_header_fields(head[0])
            if _looks_like_npm_header(first_cols, config):
                return 'npm'
        return None
    except Exception:
        return None


def _looks_like_npm_header(columns: List[str], config: Config) -> bool:
    """Heuristic NPM header check aligned with actual _load_npm requirements."""
    if not columns:
        return False

    time_col = _resolve_npm_time_col(columns, config)
    has_time = time_col is not None
    has_led = config.npm_led_col in columns
    has_roi = any(
        c.startswith(config.npm_region_prefix) and c.endswith(config.npm_region_suffix)
        for c in columns
    )
    return has_time and has_led and has_roi


def _resolve_npm_time_col(columns: List[str], config: Config) -> Optional[str]:
    """
    Resolve active NPM time column for loader/sniff.

    Primary rule is config-driven. For system-timestamp mode only, allow the
    narrow vendor alias `Timestamp` when configured `npm_system_ts_col` is
    absent, so default config can ingest real vendor exports.
    """
    if config.npm_time_axis == "system_timestamp":
        primary = config.npm_system_ts_col
        if primary in columns:
            return primary
        if "Timestamp" in columns:
            return "Timestamp"
        return None

    primary = config.npm_computer_ts_col
    if primary in columns:
        return primary
    return None


def _npm_roi_sort_key(col: str, prefix: str, suffix: str) -> Tuple[int, int, str]:
    """
    Natural ROI ordering for NPM region columns.

    Preferred form is prefix + <integer> + suffix (e.g., Region10G). These
    sort by numeric index. Non-matching or non-numeric forms fall back after
    numeric columns and are ordered lexicographically.
    """
    if col.startswith(prefix) and col.endswith(suffix):
        end_idx = len(col) - len(suffix) if suffix else len(col)
        core = col[len(prefix):end_idx]
        if core.isdigit():
            return (0, int(core), col)
    return (1, 0, col)


def _parse_vendor_npm_timestamp(path: str) -> Optional[datetime]:
    """
    Parse vendor-style NPM timestamp from filename stem.
    Expected pattern fragment: YYYY-MM-DDTHH_MM_SS (e.g., photometryData2025-03-05T15_37_44.csv).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H_%M_%S")
    except ValueError:
        return None


def sort_npm_files(paths: List[str]) -> List[str]:
    """
    Deterministic NPM session ordering.

    If ALL candidate paths contain a parseable vendor-style timestamp in the
    filename, sort by parsed datetime (full timestamp semantics). Otherwise,
    fall back to natural_sort_key for backward compatibility.
    """
    if not paths:
        return []

    stamped: List[Tuple[datetime, str]] = []
    for p in paths:
        ts = _parse_vendor_npm_timestamp(p)
        if ts is None:
            return sorted(paths, key=natural_sort_key)
        stamped.append((ts, p))

    stamped.sort(key=lambda x: (x[0], natural_sort_key(x[1])))
    return [p for _, p in stamped]

def _create_canonical_names(n_rois: int) -> List[str]:
    return [f"Region{i}" for i in range(n_rois)]

def _require_strict_check(
    t_relative: np.ndarray,
    time_sec: np.ndarray,
    target_fs_hz: float,
    context: str,
    coverage_tol_sec: Optional[float] = None,
):
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
    if coverage_tol_sec is not None and np.isfinite(coverage_tol_sec) and coverage_tol_sec > 0.0:
        tol = max(tol, float(coverage_tol_sec))
    
    if raw_start > (0.0 + tol):
        raise ValueError(f"{context}: raw_start {raw_start:.4f}s > 0.0s (Start Coverage Failure)")
        
    if raw_end < (grid_end - tol):
        raise ValueError(f"{context}: raw_end {raw_end:.4f}s < grid_end {grid_end:.4f}s (End Coverage Failure)")


def _median_positive_dt_sec(t_relative: np.ndarray) -> Optional[float]:
    """Return median positive timestep from a strictly increasing time vector."""
    if t_relative.size < 2:
        return None
    diffs = np.diff(t_relative)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _resolve_npm_strict_grid(
    t_uv_rel: np.ndarray, t_sig_rel: np.ndarray, config: Config
) -> Tuple[np.ndarray, float, float]:
    """
    Build strict NPM target grid from actual common UV/SIG support.

    Rationale:
    - Vendor NPM chunks can be shorter than GUI nominal chunk_duration_sec.
    - Strict admission should enforce full coverage over the usable overlap
      window, not over an idealized endpoint outside channel support.
    - We still reject malformed/truncated per-channel support by enforcing that
      UV/SIG start and end supports do not diverge by more than one channel
      cadence (or one target sample, whichever is larger).
    """
    t_uv_f = t_uv_rel[np.isfinite(t_uv_rel)]
    t_sig_f = t_sig_rel[np.isfinite(t_sig_rel)]

    if t_uv_f.size < 2 or t_sig_f.size < 2:
        raise ValueError("NPM strict: Insufficient finite UV/SIG timestamps")

    uv_start = float(np.nanmin(t_uv_f))
    sig_start = float(np.nanmin(t_sig_f))
    uv_end = float(np.nanmax(t_uv_f))
    sig_end = float(np.nanmax(t_sig_f))

    uv_dt = _median_positive_dt_sec(t_uv_f) or 0.0
    sig_dt = _median_positive_dt_sec(t_sig_f) or 0.0
    target_dt = 1.0 / float(config.target_fs_hz)
    support_tol = max(target_dt, uv_dt, sig_dt)

    start_gap = abs(uv_start - sig_start)
    end_gap = abs(uv_end - sig_end)
    if start_gap > support_tol:
        raise ValueError(
            "NPM strict: UV/SIG start support mismatch "
            f"({start_gap:.4f}s > {support_tol:.4f}s)"
        )
    if end_gap > support_tol:
        raise ValueError(
            "NPM strict: UV/SIG end support mismatch "
            f"({end_gap:.4f}s > {support_tol:.4f}s)"
        )

    overlap_start = max(uv_start, sig_start)
    overlap_end = min(uv_end, sig_end)
    overlap_duration = overlap_end - overlap_start
    if overlap_duration <= 0.0:
        raise ValueError(
            "NPM strict: Empty UV/SIG overlap window "
            f"(start={overlap_start:.4f}, end={overlap_end:.4f})"
        )

    # Per-channel inner support: interleaved NPM channels (UV/SIG) may
    # not both cover the full overlap window at exact grid edges.  Clamp
    # the grid to the *inner* support where BOTH channels have actual
    # timestamps so np.interp never manufactures edge NaN.
    # Filter to [overlap_start, overlap_end] to match the caller's mask.
    fs = float(config.target_fs_hz)
    uv_in_overlap = (t_uv_f >= overlap_start) & (t_uv_f <= overlap_end)
    sig_in_overlap = (t_sig_f >= overlap_start) & (t_sig_f <= overlap_end)
    t_uv_inner = t_uv_f[uv_in_overlap] - overlap_start
    t_sig_inner = t_sig_f[sig_in_overlap] - overlap_start

    if t_uv_inner.size < 2 or t_sig_inner.size < 2:
        raise ValueError(
            "NPM strict: Insufficient per-channel data within overlap window"
        )

    inner_start = max(float(t_uv_inner[0]), float(t_sig_inner[0]))
    inner_end = min(float(t_uv_inner[-1]), float(t_sig_inner[-1]))

    if inner_end <= inner_start:
        raise ValueError(
            f"NPM strict: Empty inner support "
            f"(start={inner_start:.4f}, end={inner_end:.4f})"
        )

    first_sample = int(np.ceil(inner_start * fs))
    last_sample = int(np.floor(inner_end * fs))
    n_target = last_sample - first_sample + 1

    if n_target < 2:
        raise ValueError(
            "NPM strict: Inner support too short for strict interpolation "
            f"(inner_start={inner_start:.4f}s, inner_end={inner_end:.4f}s, "
            f"fs={config.target_fs_hz})"
        )

    time_sec = np.arange(first_sample, last_sample + 1, dtype=float) / fs
    return time_sec, overlap_start, support_tol


def _resolve_strict_target_sample_count(
    t_rel: np.ndarray, config: Config, context: str
) -> int:
    """
    Resolve strict-grid target sample count.

    For real/vendor RWD with over-precise inferred contracts, the strict config
    product can over-demand the grid by slightly more than the existing
    one-sample end-coverage tolerance.

    We permit reducing n_target by exactly one sample only when the additional
    shortfall beyond that one-sample tolerance is tiny:
      shortfall_samples = (end_threshold - raw_end) * fs <= 0.25
    This keeps strict rejection for genuine truncation while rescuing the
    real-vendor near-miss case (observed shortfall ~= 0.103 samples).

    Why 0.25 samples:
    - It is much tighter than a full extra sample; we still fail any case
      needing more than 1.25 samples of total end slack versus grid_end.
    - It absorbs small contract-overprecision / timestamp quantization jitter
      without turning strict mode into partial-chunk acceptance.
    """
    n_target = int(np.round(config.chunk_duration_sec * config.target_fs_hz))

    if n_target <= 0:
        raise ValueError(
            f"{context}: Non-positive strict target sample count "
            f"({n_target}) from chunk_duration_sec={config.chunk_duration_sec} "
            f"and target_fs_hz={config.target_fs_hz}"
        )

    if not context.lower().startswith("rwd"):
        return n_target

    finite_t = t_rel[np.isfinite(t_rel)]
    if finite_t.size == 0:
        return n_target

    if n_target < 3:
        return n_target

    raw_end = float(np.nanmax(finite_t))
    fs = float(config.target_fs_hz)
    tol = 1.0 / fs
    grid_end = (n_target - 1) / fs
    end_threshold = grid_end - tol
    shortfall = end_threshold - raw_end
    shortfall_samples = shortfall * fs

    # Reduce by one sample only for a tiny fractional-sample overshoot beyond
    # the existing 1-sample strict tolerance.
    if 0.0 < shortfall_samples <= 0.25:
        return n_target - 1

    return n_target

def _resample_strict(t_rel: np.ndarray, data_in: np.ndarray, config: Config, context: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples to ONE strict grid defined by chunk_duration_sec * target_fs_hz.
    """
    # Grid Construction (Strict)
    n_target = _resolve_strict_target_sample_count(t_rel, config, context)
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
    header_info = _detect_rwd_header(path, config)
    if header_info is None:
        raise ValueError(
            "RWD: No recognizable header row found. Expected a time column "
            "(e.g., TimeStamp/Time(s)) and paired UV/SIG channels (e.g., CH1-410/CH1-470)."
        )

    header_row, detected_time_col = header_info
    df = pd.read_csv(path, header=header_row)

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    time_candidates = _rwd_time_candidates(config)
    time_col = detected_time_col if detected_time_col in df.columns else None
    if time_col is None:
        time_col = next((c for c in time_candidates if c in df.columns), None)
    if time_col is None:
        raise ValueError("RWD: Missing supported time column after header parse.")

    t_raw = pd.to_numeric(df[time_col], errors='coerce').values
    if np.isnan(t_raw).any():
        raise ValueError(f"RWD: Time column '{time_col}' contains non-numeric or NaN values.")
    metadata_fps, enabled_excitation_count = _extract_rwd_metadata_context(path, header_row)
    scale_to_seconds, timestamp_unit = _resolve_rwd_timestamp_scale(
        t_raw,
        metadata_fps=metadata_fps,
        enabled_excitation_count=enabled_excitation_count,
    )
    if scale_to_seconds != 1.0:
        t_raw = t_raw * scale_to_seconds

    cols = [str(c) for c in df.columns]
    uv_suffixes, sig_suffixes = _rwd_suffix_candidates(config)
    channel_data = _extract_rwd_channel_pairs(cols, uv_suffixes, sig_suffixes)
    if not channel_data:
        raise ValueError(
            "RWD: Recognizable header found, but no valid UV/SIG channel pairs were found. "
            f"Checked UV suffixes={uv_suffixes} and SIG suffixes={sig_suffixes}."
        )
    
    n_rois = len(channel_data)
    # POLICY: We preserve ROI base names derived directly from column headers (e.g., Region_0).
    # This ensures user labels (including underscores) are not lost but rather round-trip 
    # successfully into roi_selection and downstream packaging.
    names = [x[0] for x in channel_data]
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
        metadata={
            "roi_map": roi_map,
            "rwd_time_col_resolved": time_col,
            "rwd_timestamp_unit": timestamp_unit,
            "rwd_metadata_fps": (float(metadata_fps) if metadata_fps is not None else np.nan),
            "rwd_enabled_excitation_count": (
                int(enabled_excitation_count) if enabled_excitation_count is not None else -1
            ),
        }
    )
    # chunk.validate() moved to load_chunk
    return chunk

def _load_npm(path: str, config: Config, chunk_id: int) -> Chunk:
    df = pd.read_csv(path)
    
    time_col = _resolve_npm_time_col([str(c) for c in df.columns], config)
    expected_time_col = (
        config.npm_system_ts_col
        if config.npm_time_axis == 'system_timestamp'
        else config.npm_computer_ts_col
    )
    if time_col is None:
        raise ValueError(f"NPM: Missing {expected_time_col}")
    if config.npm_led_col not in df.columns: raise ValueError(f"NPM: Missing {config.npm_led_col}")
        
    t_full = df[time_col].values
    led = df[config.npm_led_col].values
    mask_uv = (led == 1)
    mask_sig = (led == 2)
    t_uv = t_full[mask_uv]
    t_sig = t_full[mask_sig]
    
    if len(t_uv) < 2 or len(t_sig) < 2: raise ValueError("NPM: Insufficient data")
    
    roi_cols = [
        c for c in df.columns
        if c.startswith(config.npm_region_prefix) and c.endswith(config.npm_region_suffix)
    ]
    roi_cols.sort(key=lambda c: _npm_roi_sort_key(c, config.npm_region_prefix, config.npm_region_suffix))
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
        
        # Build strict grid from actual common UV/SIG support window.
        time_sec, overlap_start, support_tol = _resolve_npm_strict_grid(t_uv_rel, t_sig_rel, config)
        n_target = time_sec.size
        grid_end = time_sec[-1]
        t_uv_rel_use = t_uv_rel - overlap_start
        t_sig_rel_use = t_sig_rel - overlap_start
        
        # Filter -> Check -> Interp
        # 1. Create strict-valid masks (No negative times, no far-future times)
        tol = support_tol * 2.0  # Allow enough bounding points for interpolation
        mask_uv_ok = np.isfinite(t_uv_rel_use) & (t_uv_rel_use >= 0.0) & (t_uv_rel_use <= grid_end + tol)
        mask_sig_ok = np.isfinite(t_sig_rel_use) & (t_sig_rel_use >= 0.0) & (t_sig_rel_use <= grid_end + tol)
        
        t_uv_use = t_uv_rel_use[mask_uv_ok]
        uv_use = uv_vals[mask_uv_ok, :]
        
        t_sig_use = t_sig_rel_use[mask_sig_ok]
        sig_use = sig_vals[mask_sig_ok, :]
        
        # 2. Strict Check on USED arrays
        _require_strict_check(
            t_uv_use, time_sec, config.target_fs_hz, "NPM UV strict", coverage_tol_sec=support_tol
        )
        _require_strict_check(
            t_sig_use, time_sec, config.target_fs_hz, "NPM SIG strict", coverage_tol_sec=support_tol
        )
        
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
        
        # Grid Construction - clamp to actual usable support
        # The idealized grid length is chunk_duration_sec * target_fs_hz, but the
        # actual raw data may end earlier.  Using the idealized length produces
        # trailing NaN via np.interp(right=np.nan) for any grid point past the
        # last raw timestamp.  Clamp to the shared UV/SIG support so the
        # resampled chunk contains only interpolatable samples.
        n_target_ideal = int(np.round(config.chunk_duration_sec * config.target_fs_hz))
        uv_max_rel = t_uv_f[-1] - t0
        sig_max_rel = t_sig_f[-1] - t0
        support_end = min(uv_max_rel, sig_max_rel)
        if support_end <= 0.0:
            raise ValueError(
                f"NPM Permissive: No usable support after t0 alignment "
                f"(uv_end={uv_max_rel:.4f}s, sig_end={sig_max_rel:.4f}s)"
            )
        n_target_support = int(np.floor(support_end * config.target_fs_hz)) + 1
        n_target = min(n_target_ideal, n_target_support)
        if n_target < 1:
            raise ValueError(
                f"NPM Permissive: Usable support too short for interpolation "
                f"(support_end={support_end:.4f}s, n_target={n_target})"
            )
        time_sec = np.arange(n_target) / config.target_fs_hz
        
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
