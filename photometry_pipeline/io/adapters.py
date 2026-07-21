import pandas as pd
import numpy as np
import os
import glob
import warnings
import itertools
import csv
import io
import re
from typing import Optional, List, Dict, Tuple, Any, Iterable
from datetime import datetime
from ..config import Config
from ..core.types import Chunk, SessionTimeMetadata
from ..core.utils import natural_sort_key
from .rwd_chronology import RwdChronologyError, order_rwd_session_candidates
from .npm_contract import NpmParserContract, resolve_npm_support_geometry
from dataclasses import asdict
import pathlib
import logging


_RWD_METADATA_LED_KEYS: Tuple[str, ...] = ("Led410Enable", "Led470Enable", "Led560Enable")
_CONTINUOUS_TIME_SCAN_CHUNKSIZE = 200_000
_CONTINUOUS_WINDOW_READ_CHUNKSIZE = 200_000


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


def rwd_authorized_time_column_candidates(rwd_time_col: str) -> Tuple[str, ...]:
    """The exact ordered time-column candidates real RWD parsing will try.

    Same resolution logic as ``_rwd_time_candidates`` (identical fallback
    order via the shared ``_unique_ordered`` helper), parameterized by the
    resolved ``rwd_time_col`` value rather than a full ``Config`` object, so
    callers outside this module (Guided normalized-recording materialization,
    which only carries the resolved scalar field, not a ``Config`` instance)
    can capture the real execution-time candidate set -- not the separate,
    config-independent preflight contract in ``io.rwd_contract`` -- as
    durable authorized provenance, without duplicating this resolution
    logic.
    """
    return tuple(
        _unique_ordered(
            [str(rwd_time_col or "").strip(), "TimeStamp", "Time(s)", "Timestamp", "Time"]
        )
    )


def rwd_authorized_suffix_candidates(
    uv_suffix: str, sig_suffix: str
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """The exact ordered (uv, signal) suffix candidates real RWD parsing will try.

    See ``rwd_authorized_time_column_candidates`` for why this exists.
    """
    uv = tuple(_unique_ordered([str(uv_suffix or "").strip(), "-410", "-415"]))
    sig = tuple(_unique_ordered([str(sig_suffix or "").strip(), "-470"]))
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


def _resolve_rwd_timestamp_scale_from_median_dt(
    med_dt: float,
    metadata_fps: Optional[float],
    enabled_excitation_count: Optional[int],
) -> Tuple[float, str]:
    """
    Resolve raw RWD timestamp unit scale to canonical seconds.

    Returns (scale_to_seconds, unit_label), where scale_to_seconds is 1.0 for
    second timestamps and 0.001 for millisecond timestamps.
    """
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
    return _resolve_rwd_timestamp_scale_from_median_dt(
        med_dt,
        metadata_fps=metadata_fps,
        enabled_excitation_count=enabled_excitation_count,
    )

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
    - Ordering: authoritative chronological order parsed from each session
      folder's canonical acquisition-time name (YYYY_MM_DD-HH_MM_SS) --
      not filesystem enumeration order and not an incidental lexical sort.
      A folder name that does not match this format, or two folders that
      resolve to the identical timestamp, fail discovery rather than
      silently falling back to a lexical or enumeration order (see
      io.rwd_chronology.order_rwd_session_candidates).
    - Ignores: outputs.csv, events.csv, fluorescence-unaligned.csv.
    """
    if not os.path.isdir(root_path):
        raise ValueError(f"RWD Discovery: Root path must be a directory: {root_path}")

    # Iterate immediate children (unordered filesystem enumeration).
    valid_entries = [
        e for e in _scandir_rwd_entries(root_path)
        if e.is_dir() and os.path.isfile(os.path.join(e.path, "fluorescence.csv"))
    ]

    if not valid_entries:
        raise ValueError(f"RWD Discovery: No valid RWD chunk directories found in {root_path} (subfolders must contain fluorescence.csv)")

    try:
        ordered_entries = order_rwd_session_candidates(
            valid_entries, name_of=lambda e: e.name
        )
    except RwdChronologyError as exc:
        raise ValueError(f"RWD Discovery: {exc}") from exc

    return [os.path.join(entry.path, "fluorescence.csv") for entry in ordered_entries]


def _scandir_rwd_entries(root_path: str):
    """Materialize the raw RWD directory enumeration for deterministic tests."""
    with os.scandir(root_path) as entries:
        return list(entries)


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


def _resolve_custom_tabular_channel_pairs(
    columns: List[str],
    uv_suffix: str,
    sig_suffix: str,
) -> List[Tuple[str, str, str]]:
    """
    Resolve strict custom-tabular ROI pairs from exact suffix contract.

    Contract:
      - each ROI base must have BOTH "<base><uv_suffix>" and "<base><sig_suffix>"
      - no inference/guessing beyond explicit suffix matching
    """
    uv_cols = [c for c in columns if c.endswith(uv_suffix)]
    sig_cols = [c for c in columns if c.endswith(sig_suffix)]
    if not uv_cols or not sig_cols:
        raise ValueError(
            "custom_tabular: missing required paired signal/isosbestic columns. "
            f"Expected columns ending with uv_suffix='{uv_suffix}' and sig_suffix='{sig_suffix}'."
        )

    def _bases(cols: List[str], suffix: str) -> List[str]:
        out = []
        for col in cols:
            base = col[: -len(suffix)] if suffix else col
            if base:
                out.append(base)
        return out

    uv_bases = set(_bases(uv_cols, uv_suffix))
    sig_bases = set(_bases(sig_cols, sig_suffix))
    if not uv_bases or not sig_bases:
        raise ValueError(
            "custom_tabular: no valid ROI bases could be resolved from suffix contract."
        )

    missing_sig = sorted(uv_bases - sig_bases, key=natural_sort_key)
    missing_uv = sorted(sig_bases - uv_bases, key=natural_sort_key)
    if missing_sig or missing_uv:
        raise ValueError(
            "custom_tabular: unmatched ROI pairs detected. "
            f"Missing signal for bases={missing_sig or []}, "
            f"missing isosbestic for bases={missing_uv or []}. "
            "Each ROI must provide both channels in the same file."
        )

    ordered_bases = sorted(uv_bases & sig_bases, key=natural_sort_key)
    return [(base, f"{base}{uv_suffix}", f"{base}{sig_suffix}") for base in ordered_bases]


def _require_strict_check(
    t_relative: np.ndarray,
    time_sec: np.ndarray,
    target_fs_hz: float,
    context: str,
    coverage_tol_sec: Optional[float] = None,
    grid_start_sec: float = 0.0,
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
    
    if raw_start > (grid_start_sec + tol):
        raise ValueError(f"{context}: raw_start {raw_start:.4f}s > {grid_start_sec:.4f}s (Start Coverage Failure)")
        
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

    # The grid is relative to the UV/signal overlap origin.  In particular,
    # it is intentionally not re-zeroed to inner_start: staggered streams
    # may therefore produce a first output time greater than zero.
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


def _resample_strict_for_duration(
    t_rel: np.ndarray,
    data_in: np.ndarray,
    *,
    target_fs_hz: float,
    duration_sec: float,
    context: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample onto a strict grid defined by `duration_sec * target_fs_hz`.
    This is used by continuous-mode window loading where duration is window-specific.
    """
    fs = float(target_fs_hz)
    duration = float(duration_sec)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"{context}: target_fs_hz must be > 0")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{context}: duration_sec must be > 0")

    n_target = int(np.round(duration * fs))
    if n_target < 2:
        raise ValueError(
            f"{context}: window too short after resampling target "
            f"(duration_sec={duration:.6f}, target_fs_hz={fs:.6f}, n_target={n_target})"
        )
    time_sec = np.arange(n_target, dtype=float) / fs
    _require_strict_check(t_rel, time_sec, fs, context)

    data_out = np.zeros((n_target, data_in.shape[1]), dtype=float)
    for i in range(data_in.shape[1]):
        data_out[:, i] = np.interp(time_sec, t_rel, data_in[:, i])
    return time_sec, data_out


def _duration_summary_from_t_rel(t_rel: np.ndarray, *, context: str) -> Dict[str, float]:
    """
    Compute source-support duration summary from strictly increasing elapsed time.
    Returns duration_sec using right-open support semantics (end + median_dt).
    """
    t = np.asarray(t_rel, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError(f"{context}: requires at least 2 timestamps")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{context}: timestamps must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{context}: timestamps must be strictly increasing")
    dt = _median_positive_dt_sec(t)
    if dt is None or (not np.isfinite(dt)) or dt <= 0.0:
        raise ValueError(f"{context}: could not resolve positive timestamp cadence")
    end_sec = float(t[-1])
    duration_sec = float(end_sec + dt)
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError(f"{context}: invalid computed duration")
    return {
        "duration_sec": duration_sec,
        "median_dt_sec": float(dt),
        "end_sec": end_sec,
    }


_CONTINUOUS_DT_SAMPLE_CAPACITY = 100_000


def _scan_time_column_metadata_chunked(
    *,
    path: str,
    time_col: str,
    header_row: int,
    context: str,
) -> Dict[str, Any]:
    """
    Scan only one time column from CSV with chunked reads.
    Keeps continuous metadata scanning bounded and avoids full-column materialization.
    """
    use_chunks = pd.read_csv(
        path,
        header=header_row,
        usecols=[time_col],
        chunksize=_CONTINUOUS_TIME_SCAN_CHUNKSIZE,
    )

    n_rows = 0
    first_time: Optional[float] = None
    last_time: Optional[float] = None
    dt_samples: List[float] = []
    dt_seen = 0
    rng = np.random.default_rng(0)

    def _record_dt(dt_value: float) -> None:
        nonlocal dt_seen
        dt_seen += 1
        if len(dt_samples) < _CONTINUOUS_DT_SAMPLE_CAPACITY:
            dt_samples.append(float(dt_value))
            return
        replace_idx = int(rng.integers(0, dt_seen))
        if replace_idx < _CONTINUOUS_DT_SAMPLE_CAPACITY:
            dt_samples[replace_idx] = float(dt_value)

    for chunk in use_chunks:
        vals = pd.to_numeric(chunk[time_col], errors="coerce").to_numpy(dtype=float)
        if vals.size == 0:
            continue
        if not np.all(np.isfinite(vals)):
            raise ValueError(
                f"{context}: Time column '{time_col}' contains non-numeric or NaN values."
            )
        if first_time is None:
            first_time = float(vals[0])
        if last_time is not None:
            cross_dt = float(vals[0]) - float(last_time)
            if cross_dt <= 0.0:
                raise ValueError(f"{context}: timestamps must be strictly increasing")
            _record_dt(cross_dt)
        diffs = np.diff(vals)
        if np.any(diffs <= 0.0):
            raise ValueError(f"{context}: timestamps must be strictly increasing")
        for dt in diffs:
            _record_dt(float(dt))
        n_rows += int(vals.size)
        last_time = float(vals[-1])

    if n_rows < 2 or first_time is None or last_time is None:
        raise ValueError(f"{context}: requires at least 2 time samples")
    if not dt_samples:
        raise ValueError(f"{context}: could not resolve positive timestamp cadence")
    median_dt = float(np.median(np.asarray(dt_samples, dtype=float)))
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise ValueError(f"{context}: could not resolve positive timestamp cadence")
    return {
        "first_time_raw": float(first_time),
        "last_time_raw": float(last_time),
        "n_rows": int(n_rows),
        "median_dt_raw": median_dt,
        "dt_sample_count": int(len(dt_samples)),
        "dt_samples_seen": int(dt_seen),
        "dt_sampling_mode": (
            "exact"
            if dt_seen <= _CONTINUOUS_DT_SAMPLE_CAPACITY
            else "deterministic_bounded_reservoir"
        ),
        "dt_sample_capacity": int(_CONTINUOUS_DT_SAMPLE_CAPACITY),
    }


def _scan_time_column_chunked(
    *,
    path: str,
    time_col: str,
    header_row: int,
    context: str,
) -> np.ndarray:
    """
    Legacy exact time-vector helper retained for non-continuous callers/tests.
    Continuous metadata planning uses _scan_time_column_metadata_chunked instead.
    """
    use_chunks = pd.read_csv(
        path,
        header=header_row,
        usecols=[time_col],
        chunksize=_CONTINUOUS_TIME_SCAN_CHUNKSIZE,
    )
    parts: List[np.ndarray] = []
    for chunk in use_chunks:
        vals = pd.to_numeric(chunk[time_col], errors="coerce").to_numpy(dtype=float)
        parts.append(vals)
    if not parts:
        raise ValueError(f"{context}: requires at least 2 time samples")
    out = np.concatenate(parts, axis=0)
    if out.size < 2:
        raise ValueError(f"{context}: requires at least 2 time samples")
    return out


def _continuous_window_metadata(window: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "acquisition_mode": "continuous",
        "window_index": int(window["window_index"]),
        "window_start_sec": float(window["window_start_sec"]),
        "window_end_sec": float(window["window_end_sec"]),
        "window_duration_sec": float(window["window_duration_sec"]),
        "original_file_duration_sec": float(window["original_file_duration_sec"]),
        "is_partial_final_window": bool(window["is_partial_final_window"]),
        "continuous_window_sec": float(window["continuous_window_sec"]),
        "continuous_step_sec": float(window["continuous_step_sec"]),
    }

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

def _scan_rwd_source_metadata(path: str, config: Config) -> Dict[str, Any]:
    header_info = _detect_rwd_header(path, config)
    if header_info is None:
        raise ValueError(
            "RWD: No recognizable header row found. Expected a time column "
            "(e.g., TimeStamp/Time(s)) and paired UV/SIG channels (e.g., CH1-410/CH1-470)."
        )

    header_row, detected_time_col = header_info
    # Header/schema only: avoid loading full signal/control table in metadata scan.
    header_df = pd.read_csv(path, header=header_row, nrows=0)
    header_df.columns = [str(c).strip().lstrip("\ufeff") for c in header_df.columns]
    cols = [str(c) for c in header_df.columns]

    time_candidates = _rwd_time_candidates(config)
    time_col = detected_time_col if detected_time_col in cols else None
    if time_col is None:
        time_col = next((c for c in time_candidates if c in cols), None)
    if time_col is None:
        raise ValueError("RWD: Missing supported time column after header parse.")

    time_scan = _scan_time_column_metadata_chunked(
        path=path,
        time_col=time_col,
        header_row=header_row,
        context="RWD",
    )

    metadata_fps, enabled_excitation_count = _extract_rwd_metadata_context(path, header_row)
    scale_to_seconds, timestamp_unit = _resolve_rwd_timestamp_scale_from_median_dt(
        float(time_scan["median_dt_raw"]),
        metadata_fps=metadata_fps,
        enabled_excitation_count=enabled_excitation_count,
    )

    uv_suffixes, sig_suffixes = _rwd_suffix_candidates(config)
    channel_data = _extract_rwd_channel_pairs(cols, uv_suffixes, sig_suffixes)
    if not channel_data:
        raise ValueError(
            "RWD: Recognizable header found, but no valid UV/SIG channel pairs were found. "
            f"Checked UV suffixes={uv_suffixes} and SIG suffixes={sig_suffixes}."
        )

    names = [x[0] for x in channel_data]
    roi_map = {names[i]: {"raw_uv": x[1], "raw_sig": x[2]} for i, x in enumerate(channel_data)}
    first_time_sec = float(time_scan["first_time_raw"]) * float(scale_to_seconds)
    last_time_sec = float(time_scan["last_time_raw"]) * float(scale_to_seconds)
    median_dt_sec = float(time_scan["median_dt_raw"]) * float(scale_to_seconds)
    duration_sec = float((last_time_sec - first_time_sec) + median_dt_sec)
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("RWD: invalid computed duration")

    return {
        "format": "rwd",
        "path": path,
        "header_row": int(header_row),
        "columns": cols,
        "time_col": time_col,
        "first_time_raw": float(time_scan["first_time_raw"]),
        "last_time_raw": float(time_scan["last_time_raw"]),
        "first_time_sec": first_time_sec,
        "last_time_sec": last_time_sec,
        "duration_sec": duration_sec,
        "median_dt_sec": median_dt_sec,
        "n_rows": int(time_scan["n_rows"]),
        "n_time_samples": int(time_scan["n_rows"]),
        "time_scan": {
            "dt_sampling_mode": time_scan["dt_sampling_mode"],
            "dt_sample_capacity": int(time_scan["dt_sample_capacity"]),
            "dt_sample_count": int(time_scan["dt_sample_count"]),
            "dt_samples_seen": int(time_scan["dt_samples_seen"]),
        },
        "n_rois": len(channel_data),
        "channel_names": names,
        "roi_map": roi_map,
        "uv_cols": [x[1] for x in channel_data],
        "sig_cols": [x[2] for x in channel_data],
        "rwd_time_col_resolved": time_col,
        "rwd_timestamp_unit": timestamp_unit,
        "rwd_timestamp_scale_to_seconds": float(scale_to_seconds),
        "rwd_metadata_fps": (float(metadata_fps) if metadata_fps is not None else np.nan),
        "rwd_enabled_excitation_count": (
            int(enabled_excitation_count) if enabled_excitation_count is not None else -1
        ),
    }


def _scan_custom_tabular_source_metadata(path: str, config: Config) -> Dict[str, Any]:
    # Header/schema only: avoid loading full signal/control table in metadata scan.
    header_df = pd.read_csv(path, nrows=0)
    header_df.columns = [str(c).strip().lstrip("\ufeff") for c in header_df.columns]
    cols = [str(c) for c in header_df.columns]

    time_col = str(getattr(config, "custom_tabular_time_col", "time_sec")).strip()
    uv_suffix = str(getattr(config, "custom_tabular_uv_suffix", "_iso")).strip()
    sig_suffix = str(getattr(config, "custom_tabular_sig_suffix", "_sig")).strip()

    if not time_col:
        raise ValueError("custom_tabular: custom_tabular_time_col must be configured")
    if not uv_suffix or not sig_suffix:
        raise ValueError(
            "custom_tabular: custom_tabular_uv_suffix and custom_tabular_sig_suffix must be configured"
        )
    if uv_suffix == sig_suffix:
        raise ValueError(
            "custom_tabular: custom_tabular_uv_suffix and custom_tabular_sig_suffix must differ"
        )
    if time_col not in cols:
        raise ValueError(
            "custom_tabular: missing required time column "
            f"'{time_col}'. Configure custom_tabular_time_col or fix CSV header."
        )

    time_scan = _scan_time_column_metadata_chunked(
        path=path,
        time_col=time_col,
        header_row=0,
        context="custom_tabular",
    )

    channel_data = _resolve_custom_tabular_channel_pairs(cols, uv_suffix, sig_suffix)
    names = [x[0] for x in channel_data]
    roi_map = {names[i]: {"raw_uv": x[1], "raw_sig": x[2]} for i, x in enumerate(channel_data)}

    first_time_sec = float(time_scan["first_time_raw"])
    last_time_sec = float(time_scan["last_time_raw"])
    median_dt_sec = float(time_scan["median_dt_raw"])
    duration_sec = float((last_time_sec - first_time_sec) + median_dt_sec)
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("custom_tabular: invalid computed duration")

    return {
        "format": "custom_tabular",
        "path": path,
        "columns": cols,
        "time_col": time_col,
        "first_time_raw": float(time_scan["first_time_raw"]),
        "last_time_raw": float(time_scan["last_time_raw"]),
        "first_time_sec": first_time_sec,
        "last_time_sec": last_time_sec,
        "duration_sec": duration_sec,
        "median_dt_sec": median_dt_sec,
        "n_rows": int(time_scan["n_rows"]),
        "n_time_samples": int(time_scan["n_rows"]),
        "time_scan": {
            "dt_sampling_mode": time_scan["dt_sampling_mode"],
            "dt_sample_capacity": int(time_scan["dt_sample_capacity"]),
            "dt_sample_count": int(time_scan["dt_sample_count"]),
            "dt_samples_seen": int(time_scan["dt_samples_seen"]),
        },
        "n_rois": len(channel_data),
        "channel_names": names,
        "roi_map": roi_map,
        "uv_cols": [x[1] for x in channel_data],
        "sig_cols": [x[2] for x in channel_data],
        "custom_tabular_contract": {
            "session_model": "one_csv_per_session",
            "time_col": time_col,
            "uv_suffix": uv_suffix,
            "sig_suffix": sig_suffix,
        },
    }


def _resolve_source_data(
    path: str,
    format_type: str,
    config: Config,
    source_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fmt = str(format_type).strip().lower()
    key = f"{fmt}:{os.path.abspath(path)}"
    if source_cache is not None and key in source_cache:
        return source_cache[key]

    if fmt == "rwd":
        source = _scan_rwd_source_metadata(path, config)
    elif fmt == "custom_tabular":
        source = _scan_custom_tabular_source_metadata(path, config)
    else:
        raise ValueError(f"Continuous source duration is unsupported for format: {format_type}")

    if source_cache is not None:
        source_cache[key] = source
    return source


def resolve_continuous_source_metadata(
    path: str,
    format_type: str,
    config: Config,
    *,
    source_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return header/time metadata for a continuous source without reading signal arrays."""
    return _resolve_source_data(path, format_type, config, source_cache=source_cache)


def _compute_window_row_bounds(
    t_rel: np.ndarray,
    *,
    window_start_sec: float,
    window_end_sec: float,
    is_final_window: bool,
) -> Tuple[int, int]:
    t = np.asarray(t_rel, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("continuous window slice requires at least 2 timestamps")
    if np.any(np.diff(t) <= 0):
        raise ValueError("continuous window slice requires strictly increasing timestamps")

    start = float(window_start_sec)
    end = float(window_end_sec)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError(
            f"Invalid continuous window bounds: start={window_start_sec}, end={window_end_sec}"
        )

    core_left = int(np.searchsorted(t, start, side="left"))
    core_right_exclusive = int(
        np.searchsorted(t, end, side=("right" if is_final_window else "left"))
    )
    left_idx = int(np.searchsorted(t, start, side="right") - 1)
    right_idx = int(np.searchsorted(t, end, side="left"))

    candidates: List[int] = []
    if core_left < core_right_exclusive:
        candidates.extend([core_left, core_right_exclusive - 1])
    if 0 <= left_idx < t.size:
        candidates.append(left_idx)
    if 0 <= right_idx < t.size:
        candidates.append(right_idx)

    if not candidates:
        raise ValueError(
            f"Window [{start:.6f}, {end:.6f}] has no usable timestamp support"
        )

    start_idx = int(min(candidates))
    end_idx = int(max(candidates))
    if end_idx - start_idx + 1 < 2:
        raise ValueError(
            f"Window [{start:.6f}, {end:.6f}] has insufficient support rows "
            f"({end_idx - start_idx + 1})"
        )
    return start_idx, end_idx


def _time_values_to_relative_seconds(
    values: np.ndarray,
    source_data: Dict[str, Any],
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    scale = float(source_data.get("rwd_timestamp_scale_to_seconds", 1.0))
    first_time_sec = float(source_data["first_time_sec"])
    return (vals * scale) - first_time_sec


def _attach_streaming_window_row_bounds(
    source_data: Dict[str, Any],
    windows: List[Dict[str, Any]],
) -> None:
    """
    Attach inclusive/exclusive row bounds using a time-column-only streaming pass.

    Preserves the existing support semantics:
    - window intervals are right-open for full windows;
    - final/partial windows may include the final endpoint;
    - one support row immediately before a start or at/after an end is included
      when present so interpolation coverage checks behave as before.
    """
    if not windows:
        return

    path = str(source_data["path"])
    time_col = str(source_data["time_col"])
    header_row = int(source_data.get("header_row", 0))
    context = str(source_data.get("format", "continuous")).upper()
    state: List[Dict[str, Any]] = [
        {
            "window": win,
            "start": float(win["window_start_sec"]),
            "end": float(win["window_end_sec"]),
            "is_final": bool(
                win.get("is_partial_final_window", False)
                or (
                    abs(
                        float(win["window_end_sec"])
                        - float(win["original_file_duration_sec"])
                    )
                    <= 1e-6
                )
            ),
            "start_idx": None,
            "end_idx": None,
        }
        for idx, win in enumerate(windows)
    ]

    start_ptr = 0
    end_ptr = 0
    prev_idx: Optional[int] = None
    prev_t: Optional[float] = None
    last_idx: Optional[int] = None
    last_t: Optional[float] = None
    row_offset = 0

    chunks = pd.read_csv(
        path,
        header=header_row,
        usecols=[time_col],
        chunksize=_CONTINUOUS_TIME_SCAN_CHUNKSIZE,
    )
    for chunk in chunks:
        vals_raw = pd.to_numeric(chunk[time_col], errors="coerce").to_numpy(dtype=float)
        if vals_raw.size == 0:
            continue
        if not np.all(np.isfinite(vals_raw)):
            raise ValueError(
                f"{context}: Time column '{time_col}' contains non-numeric or NaN values."
            )
        vals = _time_values_to_relative_seconds(vals_raw, source_data)
        if prev_t is not None and vals.size and float(vals[0]) <= float(prev_t):
            raise ValueError(f"{context}: timestamps must be strictly increasing")
        if vals.size > 1 and np.any(np.diff(vals) <= 0.0):
            raise ValueError(f"{context}: timestamps must be strictly increasing")

        for local_idx, t_val in enumerate(vals):
            row_idx = int(row_offset + local_idx)
            t = float(t_val)

            while start_ptr < len(state) and state[start_ptr]["start_idx"] is None:
                start = float(state[start_ptr]["start"])
                if t < start:
                    break
                if abs(t - start) <= 1e-12:
                    state[start_ptr]["start_idx"] = row_idx
                elif prev_idx is not None and prev_t is not None and prev_t <= start:
                    state[start_ptr]["start_idx"] = int(prev_idx)
                else:
                    state[start_ptr]["start_idx"] = row_idx
                start_ptr += 1

            while end_ptr < len(state) and state[end_ptr]["end_idx"] is None:
                end = float(state[end_ptr]["end"])
                if t < end:
                    break
                state[end_ptr]["end_idx"] = row_idx
                end_ptr += 1

            prev_idx = row_idx
            prev_t = t
            last_idx = row_idx
            last_t = t

        row_offset += int(vals.size)

    if last_idx is None or last_t is None:
        raise ValueError(f"{context}: requires at least 2 time samples")

    for entry in state:
        if entry["start_idx"] is None:
            raise ValueError(
                f"Window [{entry['start']:.6f}, {entry['end']:.6f}] has no usable timestamp support"
            )
        if entry["end_idx"] is None:
            if bool(entry["is_final"]) and last_t <= float(entry["end"]) + 1e-9:
                entry["end_idx"] = int(last_idx)
            else:
                raise ValueError(
                    f"Window [{entry['start']:.6f}, {entry['end']:.6f}] has no usable timestamp support"
                )
        start_idx = int(entry["start_idx"])
        end_idx = int(entry["end_idx"])
        if end_idx - start_idx + 1 < 2:
            raise ValueError(
                f"Window [{entry['start']:.6f}, {entry['end']:.6f}] has insufficient support rows "
                f"({end_idx - start_idx + 1})"
            )
        win = entry["window"]
        win["row_start"] = start_idx
        win["row_stop"] = end_idx + 1


def _read_csv_window_rows(
    *,
    path: str,
    format_type: str,
    columns: List[str],
    usecols: List[str],
    start_idx: int,
    end_idx: int,
    header_row: int,
) -> pd.DataFrame:
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"Invalid row bounds [{start_idx}, {end_idx}]")
    n_rows = int(end_idx - start_idx + 1)
    if n_rows < 1:
        raise ValueError("Window row read resolved no rows")

    if format_type == "rwd":
        # RWD may include metadata lines above the detected header.
        skiprows = int(header_row + 1 + start_idx)
    else:
        # custom_tabular has a single header row at line 0.
        skiprows = int(1 + start_idx)

    return pd.read_csv(
        path,
        header=None,
        names=columns,
        usecols=usecols,
        skiprows=skiprows,
        nrows=n_rows,
    )


def _load_bounded_window_arrays(
    source_data: Dict[str, Any],
    *,
    continuous_window: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmt = str(source_data.get("format", "")).strip().lower()
    start = float(continuous_window["window_start_sec"])
    if "row_start" not in continuous_window or "row_stop" not in continuous_window:
        raise ValueError(
            "Continuous window is missing precomputed row bounds; "
            "plan_continuous_windows_for_source must be used before loading."
        )
    start_idx = int(continuous_window["row_start"])
    end_idx = int(continuous_window["row_stop"]) - 1
    read_cols = (
        [str(source_data["time_col"])]
        + [str(c) for c in source_data["uv_cols"]]
        + [str(c) for c in source_data["sig_cols"]]
    )
    window_df = _read_csv_window_rows(
        path=str(source_data["path"]),
        format_type=fmt,
        columns=[str(c) for c in source_data["columns"]],
        usecols=read_cols,
        start_idx=start_idx,
        end_idx=end_idx,
        header_row=int(source_data.get("header_row", 0)),
    )
    if len(window_df) != (end_idx - start_idx + 1):
        raise ValueError(
            f"{fmt.upper()} continuous window read length mismatch: "
            f"expected {end_idx - start_idx + 1}, got {len(window_df)}"
        )

    return _window_arrays_from_dataframe(
        source_data,
        continuous_window=continuous_window,
        window_df=window_df,
    )


def _window_arrays_from_dataframe(
    source_data: Dict[str, Any],
    *,
    continuous_window: Dict[str, Any],
    window_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmt = str(source_data.get("format", "")).strip().lower()
    start = float(continuous_window["window_start_sec"])
    time_vals = pd.to_numeric(window_df[str(source_data["time_col"])], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(time_vals)):
        raise ValueError(
            f"{fmt}: time column contains non-numeric/NaN values in continuous window."
        )
    t_window = _time_values_to_relative_seconds(time_vals, source_data) - start
    if t_window.size < 2 or np.any(np.diff(t_window) <= 0.0):
        raise ValueError(f"{fmt}: continuous window timestamps must be strictly increasing")

    uv_df = window_df[[str(c) for c in source_data["uv_cols"]]].apply(
        pd.to_numeric, errors="coerce"
    )
    sig_df = window_df[[str(c) for c in source_data["sig_cols"]]].apply(
        pd.to_numeric, errors="coerce"
    )
    if uv_df.isna().values.any():
        raise ValueError(
            f"{fmt}: isosbestic channel columns contain non-numeric/NaN values."
        )
    if sig_df.isna().values.any():
        raise ValueError(
            f"{fmt}: signal channel columns contain non-numeric/NaN values."
        )
    uv_raw = uv_df.to_numpy(dtype=float)
    sig_raw = sig_df.to_numpy(dtype=float)
    return t_window, uv_raw, sig_raw


def _build_continuous_chunk_from_window_arrays(
    path: str,
    format_type: str,
    config: Config,
    chunk_id: int,
    *,
    source_data: Dict[str, Any],
    continuous_window: Dict[str, Any],
    t_window: np.ndarray,
    uv_window: np.ndarray,
    sig_window: np.ndarray,
) -> Chunk:
    n_rois = int(source_data["n_rois"])
    names = list(source_data["channel_names"])
    roi_map = dict(source_data["roi_map"])
    metadata: Dict[str, Any] = {"roi_map": roi_map}
    metadata["output_time_basis"] = "relative_seconds_since_session_start"
    if str(format_type).lower() == "custom_tabular":
        metadata["custom_tabular_contract"] = dict(source_data.get("custom_tabular_contract", {}))

    duration_sec = float(continuous_window["window_duration_sec"])
    data_window = np.hstack([uv_window, sig_window])
    time_sec, data_out = _resample_strict_for_duration(
        t_window,
        data_window,
        target_fs_hz=float(config.target_fs_hz),
        duration_sec=duration_sec,
        context=f"{str(format_type).upper()} continuous window",
    )
    metadata.update(_continuous_window_metadata(continuous_window))
    uv_grid = data_out[:, :n_rois]
    sig_grid = data_out[:, n_rois:]
    chunk = Chunk(
        chunk_id=chunk_id,
        source_file=path,
        format=str(format_type).lower(),
        time_sec=time_sec,
        uv_raw=uv_grid,
        sig_raw=sig_grid,
        fs_hz=config.target_fs_hz,
        channel_names=names,
        metadata=metadata,
    )
    _ensure_session_time_metadata(chunk)
    chunk.validate(tolerance_frac=config.timestamp_cv_max)
    return chunk


def iter_continuous_custom_tabular_chunks(
    source_data: Dict[str, Any],
    windows: List[Dict[str, Any]],
    config: Config,
    *,
    chunk_ids: Optional[List[int]] = None,
    read_chunksize: int = _CONTINUOUS_WINDOW_READ_CHUNKSIZE,
) -> Iterable[Tuple[int, Dict[str, Any], Chunk]]:
    """
    Yield continuous custom_tabular chunks using one forward CSV scan.

    Memory remains bounded: pandas holds one source chunk, and this function only
    retains DataFrame slices for windows intersecting that source chunk. Current
    continuous planning disallows overlapping/sliding windows, so active windows
    are limited to the current window plus small row-boundary overlap.
    """
    if str(source_data.get("format", "")).strip().lower() != "custom_tabular":
        raise ValueError("Sequential continuous iteration is only implemented for custom_tabular")
    if not windows:
        return
    if chunk_ids is None:
        chunk_ids = list(range(len(windows)))
    if len(chunk_ids) != len(windows):
        raise ValueError("chunk_ids length must match windows length")
    if read_chunksize <= 0:
        raise ValueError("read_chunksize must be > 0")

    ordered = list(zip(chunk_ids, windows))
    previous_start = -1
    previous_stop = -1
    previous_window_end: Optional[float] = None
    for _, win in ordered:
        row_start = int(win["row_start"])
        row_stop = int(win["row_stop"])
        window_start = float(win.get("window_start_sec", np.nan))
        if row_stop <= row_start:
            raise ValueError(f"Invalid continuous window row bounds [{row_start}, {row_stop})")
        if row_start < previous_start:
            raise ValueError(
                "Sequential continuous custom_tabular iteration requires windows in source row order"
            )
        if row_start < previous_stop and (
            previous_window_end is None
            or not np.isfinite(previous_window_end)
            or not np.isfinite(window_start)
            or window_start < previous_window_end
        ):
            raise ValueError(
                "Sequential continuous custom_tabular iteration does not support overlapping windows"
            )
        previous_start = row_start
        previous_stop = row_stop
        previous_window_end = float(win.get("window_end_sec", np.nan))

    # Selected-ROI filtering currently happens after chunk creation in Pipeline.
    # Reading all ROI columns here preserves ROI-map consistency; a future
    # optimization can safely prune usecols once selected-ROI provenance is wired
    # through this adapter layer.
    read_cols = (
        [str(source_data["time_col"])]
        + [str(c) for c in source_data["uv_cols"]]
        + [str(c) for c in source_data["sig_cols"]]
    )
    chunks = pd.read_csv(
        str(source_data["path"]),
        usecols=read_cols,
        chunksize=int(read_chunksize),
    )

    active: Dict[int, List[pd.DataFrame]] = {}
    next_add = 0
    next_yield = 0
    row_offset = 0

    for source_chunk in chunks:
        chunk_start = int(row_offset)
        chunk_end = int(row_offset + len(source_chunk))
        row_offset = chunk_end
        if chunk_end <= chunk_start:
            continue

        while next_add < len(ordered) and int(ordered[next_add][1]["row_start"]) < chunk_end:
            active[next_add] = []
            next_add += 1

        for order_idx, parts in list(active.items()):
            _, win = ordered[order_idx]
            row_start = int(win["row_start"])
            row_stop = int(win["row_stop"])
            left = max(row_start, chunk_start)
            right = min(row_stop, chunk_end)
            if left < right:
                # Copy only the intersecting rows so later pandas chunk reuse
                # cannot mutate active window pieces.
                parts.append(source_chunk.iloc[left - chunk_start: right - chunk_start].copy())

        while next_yield in active and int(ordered[next_yield][1]["row_stop"]) <= chunk_end:
            chunk_id, win = ordered[next_yield]
            parts = active.pop(next_yield)
            if not parts:
                raise ValueError(
                    f"Sequential continuous window {win.get('window_index')} resolved no source rows"
                )
            window_df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            expected_rows = int(win["row_stop"]) - int(win["row_start"])
            if len(window_df) != expected_rows:
                raise ValueError(
                    f"custom_tabular sequential window read length mismatch: "
                    f"expected {expected_rows}, got {len(window_df)}"
                )
            t_window, uv_window, sig_window = _window_arrays_from_dataframe(
                source_data,
                continuous_window=win,
                window_df=window_df,
            )
            yield (
                int(chunk_id),
                win,
                _build_continuous_chunk_from_window_arrays(
                    str(source_data["path"]),
                    "custom_tabular",
                    config,
                    int(chunk_id),
                    source_data=source_data,
                    continuous_window=win,
                    t_window=t_window,
                    uv_window=uv_window,
                    sig_window=sig_window,
                ),
            )
            next_yield += 1

    while next_yield in active:
        chunk_id, win = ordered[next_yield]
        parts = active.pop(next_yield)
        if not parts:
            raise ValueError(
                f"Sequential continuous window {win.get('window_index')} resolved no source rows"
            )
        window_df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        expected_rows = int(win["row_stop"]) - int(win["row_start"])
        if len(window_df) != expected_rows:
            raise ValueError(
                f"custom_tabular sequential window read length mismatch: "
                f"expected {expected_rows}, got {len(window_df)}"
            )
        t_window, uv_window, sig_window = _window_arrays_from_dataframe(
            source_data,
            continuous_window=win,
            window_df=window_df,
        )
        yield (
            int(chunk_id),
            win,
            _build_continuous_chunk_from_window_arrays(
                str(source_data["path"]),
                "custom_tabular",
                config,
                int(chunk_id),
                source_data=source_data,
                continuous_window=win,
                t_window=t_window,
                uv_window=uv_window,
                sig_window=sig_window,
            ),
        )
        next_yield += 1

    if next_yield != len(ordered):
        raise ValueError(
            f"custom_tabular sequential iterator yielded {next_yield} of {len(ordered)} windows"
        )


def estimate_continuous_source_duration(
    path: str,
    format_type: str,
    config: Config,
    *,
    source_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source = _resolve_source_data(path, format_type, config, source_cache=source_cache)
    return {
        "source_file": path,
        "format": str(format_type).lower(),
        "duration_sec": float(source["duration_sec"]),
        "median_dt_sec": float(source["median_dt_sec"]),
        "end_sec": float(source["last_time_sec"] - source["first_time_sec"]),
    }


def plan_continuous_windows_for_source(
    path: str,
    format_type: str,
    config: Config,
    *,
    source_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    fmt = str(format_type).strip().lower()
    if fmt not in {"rwd", "custom_tabular"}:
        raise ValueError(
            f"Continuous acquisition mode is not yet implemented for format '{format_type}'."
        )

    window_sec = float(getattr(config, "continuous_window_sec", 600.0))
    step_sec = float(getattr(config, "continuous_step_sec", window_sec))
    allow_partial = bool(getattr(config, "allow_partial_final_window", False))

    if window_sec <= 0.0 or step_sec <= 0.0:
        raise ValueError("continuous_window_sec and continuous_step_sec must be > 0")
    if abs(step_sec - window_sec) > 1e-9:
        raise ValueError(
            "continuous_step_sec must equal continuous_window_sec in this version; "
            "overlapping/sliding windows are not yet supported."
        )

    source = _resolve_source_data(path, fmt, config, source_cache=source_cache)
    duration_sec = float(source["duration_sec"])
    fs = float(getattr(config, "target_fs_hz", 40.0))

    windows: List[Dict[str, Any]] = []
    eps = 1e-9
    n_full = int(np.floor((duration_sec + eps) / window_sec))

    for idx in range(n_full):
        start = float(idx * step_sec)
        end = float(start + window_sec)
        windows.append(
            {
                "source_file": path,
                "window_index": idx,
                "window_start_sec": start,
                "window_end_sec": end,
                "window_duration_sec": float(window_sec),
                "original_file_duration_sec": duration_sec,
                "is_partial_final_window": False,
                "acquisition_mode": "continuous",
                "continuous_window_sec": window_sec,
                "continuous_step_sec": step_sec,
            }
        )

    remainder = float(duration_sec - (n_full * window_sec))
    if remainder > eps:
        if allow_partial:
            n_target_partial = int(np.round(remainder * fs))
            if n_target_partial >= 2:
                start = float(n_full * step_sec)
                end = float(start + remainder)
                windows.append(
                    {
                        "source_file": path,
                        "window_index": len(windows),
                        "window_start_sec": start,
                        "window_end_sec": end,
                        "window_duration_sec": remainder,
                        "original_file_duration_sec": duration_sec,
                        "is_partial_final_window": True,
                        "acquisition_mode": "continuous",
                        "continuous_window_sec": window_sec,
                        "continuous_step_sec": step_sec,
                    }
                )
        elif n_full == 0:
            raise ValueError(
                "Continuous source shorter than configured continuous_window_sec and "
                "allow_partial_final_window is false. Either reduce continuous_window_sec "
                "or enable allow_partial_final_window."
            )

    if not windows:
        raise ValueError(
            "No valid continuous windows could be planned. "
            "Check recording duration and continuous window settings."
        )
    _attach_streaming_window_row_bounds(source, windows)
    return windows


def _load_full_arrays_from_metadata(
    source_data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmt = str(source_data.get("format", "")).strip().lower()
    if fmt == "rwd":
        df = pd.read_csv(str(source_data["path"]), header=int(source_data["header_row"]))
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    elif fmt == "custom_tabular":
        df = pd.read_csv(str(source_data["path"]))
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    else:
        raise ValueError(f"Unsupported format for full-source read: {fmt}")

    uv_df = df[[str(c) for c in source_data["uv_cols"]]].apply(pd.to_numeric, errors="coerce")
    sig_df = df[[str(c) for c in source_data["sig_cols"]]].apply(pd.to_numeric, errors="coerce")
    time_vals = pd.to_numeric(df[str(source_data["time_col"])], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(time_vals)):
        raise ValueError(f"{fmt}: time column contains non-numeric/NaN values.")
    t_rel = _time_values_to_relative_seconds(time_vals, source_data)
    if t_rel.size < 2 or np.any(np.diff(t_rel) <= 0.0):
        raise ValueError(f"{fmt}: timestamps must be strictly increasing")
    if uv_df.isna().values.any():
        raise ValueError(f"{fmt}: isosbestic channel columns contain non-numeric/NaN values.")
    if sig_df.isna().values.any():
        raise ValueError(f"{fmt}: signal channel columns contain non-numeric/NaN values.")
    return (
        t_rel,
        uv_df.to_numpy(dtype=float),
        sig_df.to_numpy(dtype=float),
    )


def _build_chunk_from_source(
    path: str,
    format_type: str,
    config: Config,
    chunk_id: int,
    *,
    source_data: Dict[str, Any],
    continuous_window: Optional[Dict[str, Any]] = None,
) -> Chunk:
    n_rois = int(source_data["n_rois"])
    names = list(source_data["channel_names"])
    roi_map = dict(source_data["roi_map"])

    metadata: Dict[str, Any] = {
        "roi_map": roi_map,
        "output_time_basis": "relative_seconds_since_session_start",
    }
    if str(format_type).lower() == "rwd":
        metadata.update(
            {
                "rwd_time_col_resolved": source_data.get("rwd_time_col_resolved"),
                "rwd_header_row_resolved": source_data.get("header_row"),
                "rwd_timestamp_unit": source_data.get("rwd_timestamp_unit"),
                "rwd_metadata_fps": source_data.get("rwd_metadata_fps"),
                "rwd_enabled_excitation_count": source_data.get("rwd_enabled_excitation_count"),
            }
        )
    if str(format_type).lower() == "custom_tabular":
        metadata["custom_tabular_contract"] = dict(source_data.get("custom_tabular_contract", {}))

    if continuous_window is None:
        t_rel, uv_raw, sig_raw = _load_full_arrays_from_metadata(source_data)
        stacked = np.hstack([uv_raw, sig_raw])
        time_sec, data_out = _resample_strict(
            t_rel,
            stacked,
            config,
            f"{str(format_type).upper()} strict",
        )
    else:
        duration_sec = float(continuous_window["window_duration_sec"])
        t_window, uv_window, sig_window = _load_bounded_window_arrays(
            source_data,
            continuous_window=continuous_window,
        )
        data_window = np.hstack([uv_window, sig_window])
        time_sec, data_out = _resample_strict_for_duration(
            t_window,
            data_window,
            target_fs_hz=float(config.target_fs_hz),
            duration_sec=duration_sec,
            context=f"{str(format_type).upper()} continuous window",
        )
        metadata.update(_continuous_window_metadata(continuous_window))

    uv_grid = data_out[:, :n_rois]
    sig_grid = data_out[:, n_rois:]
    return Chunk(
        chunk_id=chunk_id,
        source_file=path,
        format=str(format_type).lower(),
        time_sec=time_sec,
        uv_raw=uv_grid,
        sig_raw=sig_grid,
        fs_hz=config.target_fs_hz,
        channel_names=names,
        metadata=metadata,
    )


def load_chunk(
    path: str,
    format_type: str,
    config: Config,
    chunk_id: int,
    *,
    continuous_window: Optional[Dict[str, Any]] = None,
    source_cache: Optional[Dict[str, Any]] = None,
    selected_roi: Optional[str] = None,
) -> Chunk:
    fmt = str(format_type).strip().lower()
    if fmt in {"rwd", "custom_tabular"}:
        source_data = _resolve_source_data(path, fmt, config, source_cache=source_cache)
        chunk = _build_chunk_from_source(
            path,
            fmt,
            config,
            chunk_id,
            source_data=source_data,
            continuous_window=continuous_window,
        )
    elif fmt == "npm":
        if continuous_window is not None:
            raise ValueError(
                "Continuous acquisition mode is not yet implemented for NPM/interleaved inputs."
            )
        chunk = _load_npm(
            path,
            config,
            chunk_id,
            selected_roi=selected_roi,
        )
    else:
        raise ValueError(f"Unknown format: {format_type}")

    _ensure_session_time_metadata(chunk)
    chunk.validate(tolerance_frac=config.timestamp_cv_max)
    return chunk


def _load_rwd(path: str, config: Config, chunk_id: int) -> Chunk:
    source = _scan_rwd_source_metadata(path, config)
    return _build_chunk_from_source(path, "rwd", config, chunk_id, source_data=source)


def _load_custom_tabular(path: str, config: Config, chunk_id: int) -> Chunk:
    source = _scan_custom_tabular_source_metadata(path, config)
    return _build_chunk_from_source(path, "custom_tabular", config, chunk_id, source_data=source)


def _load_npm(
    path: str,
    config: Config,
    chunk_id: int,
    *,
    selected_roi: Optional[str] = None,
) -> Chunk:
    contract = NpmParserContract.from_config(
        config, session_duration_sec=float(config.chunk_duration_sec)
    )
    if selected_roi is None:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        columns = list(df.columns)
    else:
        header = pd.read_csv(path, nrows=0)
        header.columns = [
            str(c).strip().lstrip("\ufeff") for c in header.columns
        ]
        columns = list(header.columns)
    time_col = next(
        (candidate for candidate in contract.timestamp_column_candidates if candidate in columns),
        None,
    )
    if time_col is None:
        raise ValueError(f"NPM: Missing {contract.timestamp_column_candidates[0]}")
    if contract.npm_led_col not in columns:
        raise ValueError(f"NPM: Missing {contract.npm_led_col}")

    roi_cols = sorted(
        [
            c for c in columns
            if c.startswith(contract.npm_region_prefix)
            and c.endswith(contract.npm_region_suffix)
        ],
        key=lambda c: _npm_roi_sort_key(c, contract.npm_region_prefix, contract.npm_region_suffix),
    )
    if not roi_cols:
        raise ValueError("NPM: No Region columns")
    names = _create_canonical_names(len(roi_cols))
    if selected_roi is not None:
        selected_text = str(selected_roi).strip()
        if selected_text not in names:
            raise ValueError(
                f"NPM: Requested ROI '{selected_text}' is not present"
            )
        selected_index = names.index(selected_text)
        selected_column = roi_cols[selected_index]
        roi_cols = [selected_column]
        names = [selected_text]
        df = pd.read_csv(
            path,
            usecols=[time_col, contract.npm_led_col, selected_column],
        )
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return _build_npm_chunk_from_dataframe(
        path,
        df,
        config,
        chunk_id,
        contract=contract,
        time_col=time_col,
        reference_led_value=1,
        signal_led_value=2,
        roi_cols=roi_cols,
        canonical_names=names,
        observed_physical_roi_ids=roi_cols,
    )


def load_npm_authorized_bytes(
    path: str,
    content: bytes,
    config: Config,
    chunk_id: int,
    *,
    contract: NpmParserContract,
    resolved_timestamp_column: str,
    reference_led_value: int | float | str,
    signal_led_value: int | float | str,
    physical_to_canonical_roi_mapping: tuple[tuple[str, str], ...],
    authorized_timing_geometry: dict[str, float] | None = None,
) -> Chunk:
    """Load exact already-verified bytes without parser or ROI inference."""
    if not isinstance(content, bytes):
        raise TypeError("authorized_npm_content_invalid")
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    mapping = tuple(physical_to_canonical_roi_mapping)
    physical = tuple(item[0] for item in mapping)
    canonical = tuple(item[1] for item in mapping)
    if (
        not mapping
        or len(physical) != len(set(physical))
        or len(canonical) != len(set(canonical))
    ):
        raise ValueError("authorized_npm_roi_mapping_invalid")
    roi_like = tuple(
        column
        for column in df.columns
        if column.startswith(contract.npm_region_prefix)
        and column.endswith(contract.npm_region_suffix)
    )
    if set(roi_like) != set(physical) or len(roi_like) != len(physical):
        raise ValueError("authorized_npm_physical_roi_inventory_mismatch")
    if any(column not in df.columns for column in physical):
        raise ValueError("authorized_npm_physical_roi_inventory_mismatch")
    if resolved_timestamp_column not in df.columns:
        raise ValueError("authorized_npm_timestamp_column_missing")
    if contract.npm_led_col not in df.columns:
        raise ValueError("authorized_npm_led_column_missing")
    return _build_npm_chunk_from_dataframe(
        path,
        df,
        config,
        chunk_id,
        contract=contract,
        time_col=resolved_timestamp_column,
        reference_led_value=reference_led_value,
        signal_led_value=signal_led_value,
        roi_cols=list(physical),
        canonical_names=list(canonical),
        observed_physical_roi_ids=list(roi_like),
        authorized_timing_geometry=authorized_timing_geometry,
    )


def _build_npm_chunk_from_dataframe(
    path: str,
    df: pd.DataFrame,
    config: Config,
    chunk_id: int,
    *,
    contract: NpmParserContract,
    time_col: str,
    reference_led_value: int | float | str,
    signal_led_value: int | float | str,
    roi_cols: list[str],
    canonical_names: list[str],
    observed_physical_roi_ids: list[str],
    authorized_timing_geometry: dict[str, float] | None = None,
) -> Chunk:
    n_rois = len(roi_cols)
    names = list(canonical_names)
    if len(names) != n_rois or len(names) != len(set(names)):
        raise ValueError("authorized_npm_canonical_roi_inventory_invalid")
    roi_map = {names[i]: {"raw_col": c} for i, c in enumerate(roi_cols)}
    t_full = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    led_raw = df[contract.npm_led_col]
    if isinstance(reference_led_value, str) or isinstance(signal_led_value, str):
        led = led_raw.astype(str).to_numpy()
    else:
        led = pd.to_numeric(led_raw, errors="coerce").to_numpy(dtype=float)
    mask_uv = led == reference_led_value
    mask_sig = led == signal_led_value
    t_uv = t_full[mask_uv]
    t_sig = t_full[mask_sig]
    if t_uv.size < 2 or t_sig.size < 2:
        raise ValueError("NPM: Insufficient data")
    uv_vals = df.loc[mask_uv, roi_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sig_vals = df.loc[mask_sig, roi_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_value_nans_uv = 0
    n_value_nans_sig = 0
    if authorized_timing_geometry is None:
        geometry = resolve_npm_support_geometry(t_uv, t_sig, contract)
    else:
        required_geometry = {
            "overlap_origin_absolute",
            "inner_start_rel_overlap",
            "inner_end_rel_overlap",
            "resolved_support_start_absolute",
            "resolved_support_end_absolute",
            "observed_duration_sec",
        }
        if set(authorized_timing_geometry) != required_geometry:
            raise ValueError("authorized_npm_timing_geometry_invalid")
        geometry = dict(authorized_timing_geometry)

    if not contract.allow_partial_final_chunk:
        t_uv_f_mask = np.isfinite(t_uv)
        t_sig_f_mask = np.isfinite(t_sig)
        t_uv_f = t_uv[t_uv_f_mask]
        t_sig_f = t_sig[t_sig_f_mask]
        uv_vals_f = uv_vals[t_uv_f_mask, :]
        sig_vals_f = sig_vals[t_sig_f_mask, :]
        if not np.all(np.isfinite(uv_vals_f)) or not np.all(np.isfinite(sig_vals_f)):
            raise ValueError("NPM strict: ROI values contain non-finite values")
        t0 = (
            float(geometry["overlap_origin_absolute"])
            if authorized_timing_geometry is not None
            else max(float(t_uv_f[0]), float(t_sig_f[0]))
        )
        t_uv_rel = t_uv_f - t0
        t_sig_rel = t_sig_f - t0
        time_sec, overlap_start, support_tol = _resolve_npm_strict_grid(
            t_uv_rel, t_sig_rel, config
        )
        grid_start = float(time_sec[0])
        grid_end = float(time_sec[-1])
        t_uv_rel_use = t_uv_rel - overlap_start
        t_sig_rel_use = t_sig_rel - overlap_start
        tol = support_tol * 2.0
        mask_uv_ok = (t_uv_rel_use >= grid_start - tol) & (t_uv_rel_use <= grid_end + tol)
        mask_sig_ok = (t_sig_rel_use >= grid_start - tol) & (t_sig_rel_use <= grid_end + tol)
        t_uv_use = t_uv_rel_use[mask_uv_ok]
        sig_use = sig_vals_f[mask_sig_ok, :]
        t_sig_use = t_sig_rel_use[mask_sig_ok]
        uv_use = uv_vals_f[mask_uv_ok, :]
        _require_strict_check(
            t_uv_use, time_sec, config.target_fs_hz, "NPM UV strict",
            coverage_tol_sec=support_tol, grid_start_sec=grid_start,
        )
        _require_strict_check(
            t_sig_use, time_sec, config.target_fs_hz, "NPM SIG strict",
            coverage_tol_sec=support_tol, grid_start_sec=grid_start,
        )
        uv_out = np.zeros((time_sec.size, n_rois), dtype=float)
        sig_out = np.zeros((time_sec.size, n_rois), dtype=float)
        for i in range(n_rois):
            uv_out[:, i], nans_uv = _interp_with_nan_policy(
                time_sec, t_uv_use, uv_use[:, i], config, i, "UV"
            )
            sig_out[:, i], nans_sig = _interp_with_nan_policy(
                time_sec, t_sig_use, sig_use[:, i], config, i, "SIG"
            )
            n_value_nans_uv += nans_uv
            n_value_nans_sig += nans_sig
    else:
        t_uv_f_mask = np.isfinite(t_uv)
        t_sig_f_mask = np.isfinite(t_sig)
        t_uv_f = t_uv[t_uv_f_mask]
        t_sig_f = t_sig[t_sig_f_mask]
        uv_vals_f = uv_vals[t_uv_f_mask, :]
        sig_vals_f = sig_vals[t_sig_f_mask, :]
        t0 = (
            float(geometry["overlap_origin_absolute"])
            if authorized_timing_geometry is not None
            else max(float(t_uv_f[0]), float(t_sig_f[0]))
        )
        support_end = min(float(t_uv_f[-1] - t0), float(t_sig_f[-1] - t0))
        n_target_ideal = int(np.round(config.chunk_duration_sec * config.target_fs_hz))
        n_target_support = int(np.floor(support_end * config.target_fs_hz)) + 1
        n_target = min(n_target_ideal, n_target_support)
        if support_end <= 0.0 or n_target < 1:
            raise ValueError("NPM Permissive: No usable support after t0 alignment")
        time_sec = np.arange(n_target, dtype=float) / config.target_fs_hz
        uv_out = np.zeros((n_target, n_rois), dtype=float)
        sig_out = np.zeros((n_target, n_rois), dtype=float)
        for i in range(n_rois):
            uv_out[:, i], nans_uv = _interp_with_nan_policy(
                time_sec, t_uv_f - t0, uv_vals_f[:, i], config, i, "UV"
            )
            sig_out[:, i], nans_sig = _interp_with_nan_policy(
                time_sec, t_sig_f - t0, sig_vals_f[:, i], config, i, "SIG"
            )
            n_value_nans_uv += nans_uv
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
            "adapter_value_nan_policy": getattr(config, "adapter_value_nan_policy", "strict"),
            "npm_resolved_timestamp_column": time_col,
            "npm_timestamp_unit": "seconds",
            # Common key the shared HDF5 cache writer and the
            # adapter-neutral normalized-recording comparator both read
            # (io/hdf5_cache.py, guided_normalized_recording_consumption.py).
            # NPM keeps its own distinct value here -- this is not the same
            # convention as RWD/custom_tabular's
            # "relative_seconds_since_session_start".
            "output_time_basis": "relative_seconds_since_uv_signal_overlap_origin",
            "npm_overlap_origin_absolute": float(geometry["overlap_origin_absolute"]),
            "npm_resolved_support_start_offset_sec": float(geometry["inner_start_rel_overlap"]),
            "npm_resolved_support_end_offset_sec": float(geometry["inner_end_rel_overlap"]),
            "npm_resolved_support_start_absolute": float(geometry["resolved_support_start_absolute"]),
            "npm_resolved_support_end_absolute": float(geometry["resolved_support_end_absolute"]),
            "npm_observed_duration_sec": float(geometry["observed_duration_sec"]),
            "npm_support_policy": contract.support_policy,
            "npm_observed_physical_roi_ids": tuple(observed_physical_roi_ids),
        }
    )
    return chunk
