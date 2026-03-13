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
from datetime import datetime
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
    day_idx: int
    hour_idx: int
    hour_rank: int


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

    # -- 1. Build raw records with datetime inference --
    raw_rows = []
    for cid, tpath in chunk_entries:
        source = tpath
        dt = None
        if feature_map and (cid, roi) in feature_map:
            src_val = feature_map[(cid, roi)].get('source_file', tpath)
            if src_val:
                source = src_val
            dt = infer_datetime_from_string(source)
        raw_rows.append({
            'chunk_id': cid,
            'trace_path': tpath,
            'datetime': dt,
            'source_file': source,
        })

    # -- 2. Try datetime-based layout --
    n_mapped = sum(1 for r in raw_rows if r['datetime'] is not None)
    pct_mapped = (n_mapped / len(raw_rows)) * 100 if raw_rows else 0
    sph = sessions_per_hour

    if pct_mapped > 90 and sph is None:
        # Datetime mode
        datetimes = [r['datetime'] for r in raw_rows if r['datetime'] is not None]
        t0 = min(datetimes)
        day_start = t0.replace(hour=0, minute=0, second=0, microsecond=0)

        for r in raw_rows:
            if r['datetime'] is not None:
                elapsed = (r['datetime'] - day_start).total_seconds()
                r['day_idx'] = int(elapsed // 86400)
                r['hour_idx'] = int((elapsed % 86400) // 3600)
                r['_sort_key'] = (r['day_idx'], r['hour_idx'], r['datetime'])
            else:
                r['day_idx'] = 0
                r['hour_idx'] = 0
                r['_sort_key'] = (0, 0, datetime.min)

        raw_rows.sort(key=lambda r: r['_sort_key'])

        # Infer SPH from mode of per-hour counts
        from collections import Counter
        hour_counts = Counter()
        for r in raw_rows:
            hour_counts[(r['day_idx'], r['hour_idx'])] += 1
        if hour_counts:
            count_values = list(hour_counts.values())
            sph = max(set(count_values), key=count_values.count)
        else:
            sph = 1

        # Assign hour_rank within each (day, hour) group
        rank_counter: Dict[Tuple[int, int], int] = {}
        for r in raw_rows:
            key = (r['day_idx'], r['hour_idx'])
            rank = rank_counter.get(key, 0)
            r['hour_rank'] = rank
            rank_counter[key] = rank + 1

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

    sph = int(max(1, sph))

    # -- 3. Build ChunkRecords and group by day --
    chunks = []
    for r in raw_rows:
        chunks.append(ChunkRecord(
            chunk_id=r['chunk_id'],
            trace_path=r['trace_path'],
            datetime_inferred=r['datetime'],
            source_file=r['source_file'],
            day_idx=r['day_idx'],
            hour_idx=r['hour_idx'],
            hour_rank=r['hour_rank'],
        ))

    chunks_by_day: Dict[int, List[ChunkRecord]] = {}
    for c in chunks:
        chunks_by_day.setdefault(c.day_idx, []).append(c)

    return PhasicDataSet(
        roi=roi,
        sessions_per_hour=sph,
        chunks=chunks,
        chunks_by_day=chunks_by_day,
        feature_map=feature_map,
    )
