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
import json
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
    # The cache contribution id is a storage identifier.  For current
    # intermittent runs ``chunk_id`` is the authoritative chronological session
    # index and ``cache_chunk_id`` points to the real HDF5 group.  Legacy callers
    # leave these equal through the defaults below.
    session_index: Optional[int] = None
    cache_chunk_id: Optional[int] = None
    status: str = "valid"
    expected_start_time: Optional[datetime] = None
    expected_duration_sec: Optional[float] = None
    missing_reason: str = ""


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


def parse_session_folder_datetime(session_folder: str) -> Optional[datetime]:
    """
    Parse canonical session folder datetime token (YYYY_MM_DD-HH_MM_SS).

    Returns None when the provided token does not match the canonical format.
    """
    if not isinstance(session_folder, str):
        return None
    token = session_folder.strip()
    m = _SESSION_FOLDER_PATTERN.fullmatch(token)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1).replace("-", "_"), "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return None


def infer_session_datetime(source: str) -> Optional[datetime]:
    """
    Infer authoritative session datetime for timeline placement.

    Priority:
      1) Canonical timestamped session folder token (authoritative).
      2) Generic fallback search over source string (legacy compatibility).
    """
    session_folder = infer_session_folder_name(source)
    dt_session = parse_session_folder_datetime(session_folder)
    if dt_session is not None:
        return dt_session
    return infer_datetime_from_string(source)


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
    key_column: str = "chunk_id",
) -> Dict:
    """
    Load features.csv and build a lookup dict keyed on (chunk_id, roi).

    If *roi* is given, only rows for that ROI are included.
    Returns an empty dict if features.csv does not exist.
    """
    if not os.path.exists(feats_path):
        return {}

    df = pd.read_csv(feats_path)
    if key_column not in df.columns:
        raise RuntimeError(
            f"features.csv is missing the authoritative key column '{key_column}'"
        )
    if roi is not None:
        df = df[df['roi'] == roi]

    feat_map = {}
    for _, row in df.iterrows():
        key = row[key_column]
        if pd.isna(key):
            continue
        feat_map[(int(key), row['roi'])] = row
    return feat_map


def load_authoritative_session_index(analysis_out_dir: str) -> Optional[List[dict]]:
    """Load the current intermittent session index when one is present.

    A current 4J16k41 analysis writes the same ordered expected-session record
    beside each analysis cache.  Plotters must use it when available, while
    legacy fixture/output directories without the record retain their historical
    cache-chunk behavior.  Structural validation is delegated to the shared
    completeness reader so plot code cannot invent a second accounting policy.
    """
    from photometry_pipeline.input_processing_completeness import (
        read_input_completeness,
    )

    payload, error = read_input_completeness(str(analysis_out_dir))
    if payload is None:
        if error == "missing":
            return None
        raise RuntimeError(
            "The analysis session index is unavailable or corrupted: " + str(error)
        )
    expected = payload.get("expected")
    if not isinstance(expected, list) or not expected:
        raise RuntimeError("The analysis session index contains no expected sessions")
    return [dict(entry) for entry in expected]


def build_authoritative_plot_sessions(
    analysis_out_dir: str,
    cache_chunk_ids: List[int],
    cache_source_files: Optional[List[str]] = None,
) -> Optional[List[dict]]:
    """Join the authoritative session index to real cache contributions.

    The returned list has one record per expected session, including approved
    missing/corrupted and authorized-final-exclusion slots.  Missing slots have
    ``cache_chunk_id=None`` and are never read from HDF5.  A current record that
    cannot be reconciled with the cache fails closed instead of silently falling
    back to dense cache order.
    """
    expected = load_authoritative_session_index(analysis_out_dir)
    if expected is None:
        return None

    normalized_cache_ids = {int(cid) for cid in cache_chunk_ids}
    source_by_cache_id = {
        int(cid): str(cache_source_files[i])
        for i, cid in enumerate(cache_chunk_ids)
        if cache_source_files is not None and i < len(cache_source_files)
    }
    processed_by_index: dict[int, dict] = {}
    for entry in expected:
        idx = int(entry["index"])
        if str(entry.get("disposition", "")) == "process":
            # The completeness reader already verifies exactly-once accounting;
            # this lookup only joins the storage id to the plotting session id.
            processed_by_index[idx] = {}

    # Read the compact processed mapping from the same analysis record.
    from photometry_pipeline.input_processing_completeness import (
        INPUT_COMPLETENESS_FILENAME,
    )
    record_path = os.path.join(str(analysis_out_dir), INPUT_COMPLETENESS_FILENAME)
    with open(record_path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    for processed in record.get("processed", []):
        processed_by_index[int(processed["index"])] = dict(processed)

    sessions: list[dict] = []
    used_cache_ids: set[int] = set()
    for entry in sorted(expected, key=lambda item: int(item["index"])):
        index = int(entry["index"])
        disposition = str(entry.get("disposition", ""))
        if disposition == "process":
            processed = processed_by_index.get(index) or {}
            if "cache_chunk_id" not in processed:
                raise RuntimeError(
                    f"Session index {index} has no processed cache contribution"
                )
            cache_id = int(processed["cache_chunk_id"])
            if cache_id not in normalized_cache_ids:
                raise RuntimeError(
                    f"Session index {index} points to missing cache contribution {cache_id}"
                )
            used_cache_ids.add(cache_id)
        else:
            cache_id = None

        source = str(entry.get("source", ""))
        session_label = os.path.basename(os.path.dirname(source)) or os.path.basename(source)
        status = {
            "process": "valid",
            "authorized_missing_corrupted": "missing_corrupted",
            "authorized_exclusion": "authorized_final_exclusion",
        }.get(disposition, disposition or "unknown")
        start_time = None
        start_text = str(entry.get("expected_start_time", "")).strip()
        if start_text:
            try:
                start_time = datetime.fromisoformat(start_text)
            except ValueError:
                raise RuntimeError(
                    f"Session index {index} has an invalid expected start time"
                )
        sessions.append(
            {
                "session_index": index,
                "cache_chunk_id": cache_id,
                "source_file": source,
                "cache_source_file": source_by_cache_id.get(cache_id, source),
                "status": status,
                "disposition": disposition,
                "expected_start_time": start_time,
                "expected_start_time_text": start_text,
                "expected_duration_sec": entry.get("expected_duration_sec"),
                "missing_reason": str(entry.get("reason", "")),
                "failure_category": str(entry.get("failure_category", "")),
                "authorization_source": str(entry.get("authorization_source", "")),
                "session_label": session_label,
            }
        )

    unexpected = normalized_cache_ids - used_cache_ids
    if unexpected:
        raise RuntimeError(
            "The cache contains contributions not represented by the authoritative "
            f"session index: {sorted(unexpected)}"
        )
    return sessions


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
    session_index_entries: Optional[List[dict]] = None,
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
    if session_index_entries is not None:
        cache_paths = {int(cid): tpath for cid, tpath in chunk_entries}
        for session in sorted(session_index_entries, key=lambda item: int(item["session_index"])):
            session_id = int(session["session_index"])
            cache_id = session.get("cache_chunk_id")
            cache_id = int(cache_id) if cache_id is not None else None
            source = str(session.get("source_file", ""))
            raw_rows.append({
                # Keep the public chunk_id as the chronological session id.  The
                # HDF5 storage id remains separate and is used by consumers when
                # loading actual arrays.
                "chunk_id": session_id,
                "cache_chunk_id": cache_id,
                "trace_path": cache_paths.get(cache_id, source),
                "datetime": session.get("expected_start_time"),
                "source_file": source,
                "session_folder": infer_session_folder_name(source),
                "elapsed_from_start_sec": float("nan"),
                "status": str(session.get("status", "unknown")),
                "expected_start_time": session.get("expected_start_time"),
                "expected_duration_sec": session.get("expected_duration_sec"),
                "missing_reason": str(session.get("missing_reason", "")),
            })
    for cid, tpath in ([] if session_index_entries is not None else chunk_entries):
        source = tpath
        dt = None
        if feature_map and (cid, roi) in feature_map:
            src_val = feature_map[(cid, roi)].get('source_file', tpath)
            if isinstance(src_val, str) and src_val.strip():
                source = src_val
        dt = infer_session_datetime(source)
        session_folder = infer_session_folder_name(source)
        raw_rows.append({
            'chunk_id': cid,
            'trace_path': tpath,
            'datetime': dt,
            'source_file': source,
            'session_folder': session_folder,
            'elapsed_from_start_sec': float('nan'),
            'cache_chunk_id': cid,
            'status': 'valid',
            'expected_start_time': dt,
            'expected_duration_sec': None,
            'missing_reason': '',
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
        # -- Fallback: identity-anchored layout --
        # Place each session by its OWN chronological id, never by dense row
        # position. A gap in ids (an approved missing session, absent from these
        # rows) then leaves an empty slot instead of pulling later sessions
        # backward in time or renumbering them (4J16k41c).
        raw_rows.sort(key=lambda r: r['chunk_id'])
        if sph is None:
            span = (raw_rows[-1]['chunk_id'] - raw_rows[0]['chunk_id'] + 1) if raw_rows else 1
            n_days_est = max(1, math.ceil(span / 48))
            sph = max(1, round(span / (24 * n_days_est)))

        base_id = raw_rows[0]['chunk_id'] if raw_rows else 0
        chunks_per_day = 24 * sph
        for r in raw_rows:
            slot = int(r['chunk_id']) - int(base_id)
            r['day_idx'] = slot // chunks_per_day
            r['hour_idx'] = (slot // sph) % 24
            r['hour_rank'] = slot % sph
            r['within_hour_offset_sec'] = float(r['hour_rank']) * (3600.0 / float(sph))
            r['elapsed_from_start_sec'] = float(slot * (3600.0 / float(sph)))

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
            session_index=int(r.get('session_index', r['chunk_id'])) if r.get('session_index') is not None else int(r['chunk_id']),
            cache_chunk_id=(int(r['cache_chunk_id']) if r.get('cache_chunk_id') is not None else None),
            status=str(r.get('status', 'valid')),
            expected_start_time=r.get('expected_start_time'),
            expected_duration_sec=(
                float(r['expected_duration_sec'])
                if r.get('expected_duration_sec') is not None else None
            ),
            missing_reason=str(r.get('missing_reason', '')),
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
