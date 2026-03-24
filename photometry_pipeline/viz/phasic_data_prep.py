"""
Phasic Plotting Data Preparation
=================================

Shared data-preparation layer for phasic plotting scripts.
Centralises chunk discovery, ROI resolution, day/hour grouping,
and feature-map construction that was previously duplicated across:

  - tools/verification/plot_phasic_qc_grid.py
  - tools/verification/plot_session_grid.py
  - tools/verification/plot_phasic_stacked_day_smoothed.py

This module is Part 1 of a two-part plotting optimisation.
It does NOT eliminate cross-script CSV reloading (each script
still reads its own trace data) — that is deferred to Part 2.
"""

import glob
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ======================================================================
# Data Model
# ======================================================================

@dataclass
class ChunkRecord:
    """Metadata for a single chunk trace file, with layout coordinates."""
    chunk_id: int
    trace_path: str
    datetime_inferred: Optional[datetime]
    source_file: str
    session_folder: str
    elapsed_from_start_sec: float
    day_idx: int
    hour_idx: int
    hour_rank: int
    within_hour_offset_sec: float


@dataclass
class PhasicDataSet:
    """
    Canonical prepared structure for phasic plotting.

    Attributes:
        roi: ROI name this dataset was prepared for.
        sessions_per_hour: resolved column count for grid layouts.
        chunks: all ChunkRecords in canonical order.
        chunks_by_day: day_idx → list of ChunkRecords for that day.
        feature_map: optional (chunk_id, roi) → feature row dict.
    """
    roi: str
    sessions_per_hour: int
    timeline_anchor_mode: str
    fixed_daily_anchor_clock: Optional[str]
    chunks: List[ChunkRecord]
    chunks_by_day: Dict[int, List[ChunkRecord]]
    feature_map: Optional[Dict] = field(default=None, repr=False)


# ======================================================================
# Chunk Discovery
# ======================================================================

_CHUNK_RE = re.compile(r'chunk_(\d+)\.csv')


def discover_chunks(traces_dir: str) -> List[Tuple[int, str]]:
    """
    Discover and sort chunk trace files.

    Returns a list of (chunk_id, absolute_path) tuples sorted by chunk_id.
    Raises RuntimeError if no chunk files are found.
    """
    files = sorted(glob.glob(os.path.join(traces_dir, 'chunk_*.csv')))
    if not files:
        raise RuntimeError(f"No trace files found in {traces_dir}")

    result = []
    for fpath in files:
        m = _CHUNK_RE.search(os.path.basename(fpath))
        if m:
            result.append((int(m.group(1)), fpath))
    result.sort(key=lambda x: x[0])
    return result


# ======================================================================
# Datetime Inference
# ======================================================================

_DT_PATTERNS = [
    re.compile(r'(\d{4})[-_](\d{2})[-_](\d{2})[-_](\d{2})[_:](\d{2})[_:](\d{2})'),
    re.compile(r'(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})'),
    re.compile(r'(\d{4})[-_](\d{2})[-_](\d{2})\s+(\d{2})[:](\d{2})[:](\d{2})'),
]
_SESSION_FOLDER_PATTERN = re.compile(r'^(\d{4}[_-]\d{2}[_-]\d{2}-\d{2}[_:]\d{2}[_:]\d{2})$')
_ANCHOR_MODE_VALUES = {"civil", "elapsed", "fixed_daily_anchor"}


def infer_datetime_from_string(s) -> Optional[datetime]:
    """
    Try to extract a datetime from a filename or source-file string.
    Returns None on failure.
    """
    if not isinstance(s, str):
        return None
    for pat in _DT_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                parts = list(map(int, m.groups()))
                return datetime(*parts)
            except ValueError:
                continue
    return None


def infer_session_folder_name(source: str) -> str:
    """
    Infer canonical session folder label from a source path.

    Prefer a parent folder token matching YYYY_MM_DD-HH_MM_SS. If not found,
    return basename stem as a stable fallback label.
    """
    if not isinstance(source, str):
        return ""
    norm = source.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    for token in reversed(parts):
        if _SESSION_FOLDER_PATTERN.match(token):
            return token
    base = os.path.basename(norm)
    stem, _ = os.path.splitext(base)
    return stem or base


def parse_fixed_daily_anchor_clock(clock_text: str) -> Tuple[int, str]:
    """
    Parse fixed daily anchor clock text.

    Accepts HH:MM or HH:MM:SS and returns:
      (seconds_from_midnight, canonical_HH:MM:SS).
    """
    if not isinstance(clock_text, str) or not clock_text.strip():
        raise ValueError("fixed_daily_anchor clock is required (format HH:MM or HH:MM:SS).")
    token = clock_text.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", token)
    if not m:
        raise ValueError(
            f"Invalid fixed_daily_anchor clock '{clock_text}'. Expected HH:MM or HH:MM:SS."
        )
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3)) if m.group(3) is not None else 0
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        raise ValueError(
            f"Invalid fixed_daily_anchor clock '{clock_text}'. Hours 0-23, minutes/seconds 0-59."
        )
    seconds = hh * 3600 + mm * 60 + ss
    canonical = f"{hh:02d}:{mm:02d}:{ss:02d}"
    return seconds, canonical


# ======================================================================
# Feature Map
# ======================================================================

def build_feature_map(
    feats_path: str,
    roi: Optional[str] = None,
) -> Dict:
    """
    Load features.csv and build a lookup dict keyed on (chunk_id, roi).

    If *roi* is given, only rows for that ROI are included.
    Returns an empty dict if features.csv does not exist.
    """
    if not os.path.exists(feats_path):
        return {}

    df = pd.read_csv(feats_path)
    if roi is not None:
        df = df[df['roi'] == roi]

    feat_map = {}
    for _, row in df.iterrows():
        feat_map[(row['chunk_id'], row['roi'])] = row
    return feat_map


# ======================================================================
# ROI Resolution
# ======================================================================

def resolve_roi(
    first_trace_path: str,
    requested_roi: Optional[str],
    column_suffix: str = '_dff',
) -> str:
    """
    Determine which ROI to use.

    If *requested_roi* is given, returns it directly (caller is responsible
    for validity).  Otherwise, auto-detects from the first trace file by
    looking for columns ending with *column_suffix*.

    Raises RuntimeError if auto-detection fails.
    """
    if requested_roi:
        return requested_roi

    df0 = pd.read_csv(first_trace_path, nrows=1)
    matched = [c for c in df0.columns if c.endswith(column_suffix)]
    if not matched:
        raise RuntimeError(
            f"No columns ending with '{column_suffix}' in {first_trace_path}. "
            f"Available: {list(df0.columns)}"
        )
    rois = sorted(c.replace(column_suffix, '') for c in matched)
    return rois[0]


# ======================================================================
# Day Layout Computation
# ======================================================================

def compute_day_layout(
    chunk_entries: List[Tuple[int, str]],
    feature_map: Optional[Dict],
    roi: str,
    sessions_per_hour: Optional[int] = None,
    timeline_anchor_mode: str = "civil",
    fixed_daily_anchor_clock: Optional[str] = None,
) -> PhasicDataSet:
    """
    Build the canonical day/hour/rank layout for phasic plotting.

    Parameters
    ----------
    chunk_entries : list of (chunk_id, trace_path)
        As returned by ``discover_chunks()``.
    feature_map : dict or None
        ``(chunk_id, roi) → row``, used to look up source_file for
        datetime inference.  May be None or empty.
    roi : str
        The target ROI.
    sessions_per_hour : int or None
        If None, will be inferred from data.

    Returns
    -------
    PhasicDataSet
        A fully populated canonical structure.
    """
    if not chunk_entries:
        raise RuntimeError("No chunk entries provided to compute_day_layout")

    anchor_mode = str(timeline_anchor_mode).strip().lower()
    if anchor_mode not in _ANCHOR_MODE_VALUES:
        raise ValueError(
            f"Unsupported timeline_anchor_mode='{timeline_anchor_mode}'. "
            f"Allowed: {sorted(_ANCHOR_MODE_VALUES)}"
        )

    anchor_seconds = 0
    anchor_clock_canonical = None
    if anchor_mode == "fixed_daily_anchor":
        anchor_seconds, anchor_clock_canonical = parse_fixed_daily_anchor_clock(
            fixed_daily_anchor_clock if fixed_daily_anchor_clock is not None else ""
        )

    # -- 1. Build raw records with datetime inference --
    raw_rows = []
    for cid, tpath in chunk_entries:
        source = tpath
        dt = None
        if feature_map and (cid, roi) in feature_map:
            src_val = feature_map[(cid, roi)].get('source_file', tpath)
            if isinstance(src_val, str) and src_val.strip():
                source = src_val
        dt = infer_datetime_from_string(source)
        session_folder = infer_session_folder_name(source)
        raw_rows.append({
            'chunk_id': cid,
            'trace_path': tpath,
            'datetime': dt,
            'source_file': source,
            'session_folder': session_folder,
            'elapsed_from_start_sec': float('nan'),
        })

    # -- 2. Try datetime-based layout --
    n_mapped = sum(1 for r in raw_rows if r['datetime'] is not None)
    pct_mapped = (n_mapped / len(raw_rows)) * 100 if raw_rows else 0
    sph = sessions_per_hour

    if pct_mapped > 90:
        # Datetime mode (authoritative default):
        # - Placement semantics controlled by explicit anchor mode
        # - Slot placement from within-hour clock offset (not occurrence rank)
        datetimes = [r['datetime'] for r in raw_rows if r['datetime'] is not None]
        t0 = min(datetimes)
        base_date = t0.date()

        def _civil_components(dt_obj: datetime) -> Tuple[int, int, float]:
            day_idx_val = int((dt_obj.date() - base_date).days)
            hour_idx_val = int(dt_obj.hour)
            within_hour_offset = (
                float(dt_obj.minute) * 60.0
                + float(dt_obj.second)
                + (float(dt_obj.microsecond) / 1_000_000.0)
            )
            return day_idx_val, hour_idx_val, within_hour_offset

        def _elapsed_components(dt_obj: datetime) -> Tuple[int, int, float]:
            elapsed = float((dt_obj - t0).total_seconds())
            day_idx_val = int(elapsed // 86400.0)
            rem = elapsed % 86400.0
            hour_idx_val = int(rem // 3600.0)
            within_hour_offset = rem % 3600.0
            return day_idx_val, hour_idx_val, within_hour_offset

        def _fixed_anchor_day_start(dt_obj: datetime) -> datetime:
            start = datetime(dt_obj.year, dt_obj.month, dt_obj.day) + timedelta(seconds=anchor_seconds)
            if dt_obj < start:
                start -= timedelta(days=1)
            return start

        if anchor_mode == "fixed_daily_anchor":
            base_anchor_start = _fixed_anchor_day_start(t0)
            base_anchor_date = base_anchor_start.date()

            def _fixed_daily_anchor_components(dt_obj: datetime) -> Tuple[int, int, float]:
                day_start = _fixed_anchor_day_start(dt_obj)
                rel = float((dt_obj - day_start).total_seconds())
                day_idx_val = int((day_start.date() - base_anchor_date).days)
                hour_idx_val = int(rel // 3600.0)
                within_hour_offset = rel % 3600.0
                return day_idx_val, hour_idx_val, within_hour_offset
        else:
            _fixed_daily_anchor_components = None

        from collections import Counter
        hour_counts = Counter()
        for r in raw_rows:
            dt = r['datetime']
            if dt is None:
                continue
            if anchor_mode == "civil":
                day_idx, hour_idx, _ = _civil_components(dt)
            elif anchor_mode == "elapsed":
                day_idx, hour_idx, _ = _elapsed_components(dt)
            else:
                day_idx, hour_idx, _ = _fixed_daily_anchor_components(dt)
            hour_counts[(day_idx, hour_idx)] += 1

        if sessions_per_hour is not None:
            sph = int(max(1, sessions_per_hour))
        else:
            if hour_counts:
                count_values = list(hour_counts.values())
                sph = int(max(1, max(set(count_values), key=count_values.count)))
            else:
                sph = 1

        slot_width_sec = 3600.0 / float(max(1, sph))

        for r in raw_rows:
            dt = r['datetime']
            if dt is not None:
                elapsed = (dt - t0).total_seconds()
                r['elapsed_from_start_sec'] = float(elapsed)
                if anchor_mode == "civil":
                    day_idx, hour_idx, offset_sec = _civil_components(dt)
                elif anchor_mode == "elapsed":
                    day_idx, hour_idx, offset_sec = _elapsed_components(dt)
                else:
                    day_idx, hour_idx, offset_sec = _fixed_daily_anchor_components(dt)
                r['day_idx'] = int(day_idx)
                r['hour_idx'] = int(hour_idx)
                r['within_hour_offset_sec'] = float(offset_sec)
                slot = int(offset_sec // slot_width_sec)
                if slot < 0:
                    slot = 0
                if slot >= sph:
                    slot = sph - 1
                r['hour_rank'] = int(slot)
                r['_sort_key'] = (dt, r['chunk_id'])
            else:
                r['day_idx'] = 0
                r['hour_idx'] = 0
                r['hour_rank'] = 0
                r['within_hour_offset_sec'] = 0.0
                r['elapsed_from_start_sec'] = float('nan')
                r['_sort_key'] = (datetime.min, r['chunk_id'])

        raw_rows.sort(key=lambda r: r['_sort_key'])

    else:
        # -- Fallback: sequential layout --
        if sph is None:
            n_chunks = len(raw_rows)
            n_days_est = max(1, math.ceil(n_chunks / 48))
            sph = max(1, round(n_chunks / (24 * n_days_est)))

        raw_rows.sort(key=lambda r: r['chunk_id'])
        chunks_per_day = 24 * sph
        for idx, r in enumerate(raw_rows):
            r['day_idx'] = idx // chunks_per_day
            r['hour_idx'] = (idx // sph) % 24
            r['hour_rank'] = idx % sph
            r['within_hour_offset_sec'] = float(r['hour_rank']) * (3600.0 / float(sph))
            r['elapsed_from_start_sec'] = float(idx * (3600.0 / float(sph)))

    sph = int(max(1, sph))

    # -- 3. Build ChunkRecords and group by day --
    chunks = []
    for r in raw_rows:
        chunks.append(ChunkRecord(
            chunk_id=r['chunk_id'],
            trace_path=r['trace_path'],
            datetime_inferred=r['datetime'],
            source_file=r['source_file'],
            session_folder=r['session_folder'],
            elapsed_from_start_sec=float(r['elapsed_from_start_sec']),
            day_idx=r['day_idx'],
            hour_idx=r['hour_idx'],
            hour_rank=r['hour_rank'],
            within_hour_offset_sec=float(r.get('within_hour_offset_sec', 0.0)),
        ))

    chunks_by_day: Dict[int, List[ChunkRecord]] = {}
    for c in chunks:
        chunks_by_day.setdefault(c.day_idx, []).append(c)

    return PhasicDataSet(
        roi=roi,
        sessions_per_hour=sph,
        timeline_anchor_mode=anchor_mode,
        fixed_daily_anchor_clock=anchor_clock_canonical,
        chunks=chunks,
        chunks_by_day=chunks_by_day,
        feature_map=feature_map,
    )
