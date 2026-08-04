"""Session-level tonic summary for repeated-session (intermittent) recordings.

One tonic value per ROI per authoritative session, computed after ingestion from
the raw signal/reference channels already stored in the tonic trace cache. This
module never reads the phasic correction result and never consults the per-ROI
correction map, so its output does not depend on the selected phasic correction
strategy.

Exactly one method is chosen per ROI for the whole recording:

Primary -- global-isosbestic tonic dF/F0
    fitted_iso(t) = slope * uv_raw(t) + intercept      (one robust global fit)
    dff(t)        = (sig_raw(t) - fitted_iso(t)) / fitted_iso(t)
    tonic_s       = percentile(finite dff in session s, 2)
    Units: fraction dF/F0.

Fallback -- signal-only bleach-corrected tonic F
    Used only when the global isosbestic fit fails, returns a nonpositive slope,
    or yields no usable session at all once applied (an ROI whose fitted
    isosbestic is never safely positive). A single session that ends up invalid
    never switches the ROI's method. Per session,
    baseline_s = percentile(finite sig_raw, 2); a single
    constrained exponential bleach trend is fit across the usable session
    baselines, and the existing offset-preserving subtractive correction is
    applied:
        tonic_s = baseline_s - fitted_bleach_s + mean(fitted_bleach)
    Units: raw fluorescence AU. The reference channel is not used.

The two methods are never mixed within one ROI, and neither ever falls back to
the phasic correction trace.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from photometry_pipeline.core.tonic_dff import (
    apply_global_fit,
    compute_global_iso_fit_robust,
)
# The existing constrained tonic bleach model (already used in production by
# core.tonic_output.apply_tonic_output_mode_to_session). Reused here rather than
# reimplemented so both tonic paths share one bleach definition.
from photometry_pipeline.core.tonic_output import _fit_exponential_bleach_trend

TONIC_SESSION_SUMMARY_FILENAME = "tonic_session_summary.csv"

#: Percentile defining the tonic value within a session.
TONIC_PERCENTILE = 2.0

#: Minimum finite samples a session must contribute to yield a tonic value.
MIN_FINITE_SAMPLES = 10

#: Smallest fitted-isosbestic value accepted as a dF/F0 denominator. Matches the
#: magnitude of the established ``Config.f0_min_value`` default, which guards the
#: same kind of fluorescence denominator elsewhere in the pipeline. Absolute and
#: deterministic so the accepted sample set never depends on the data's scale.
MIN_FITTED_ISOSBESTIC = 1e-9

METHOD_GLOBAL_ISOSBESTIC = "global_isosbestic_tonic_dff"
METHOD_SIGNAL_ONLY = "signal_only_bleach_corrected_tonic_f"

UNITS_FRACTION_DFF = "fraction_dff"
UNITS_RAW_AU = "raw_fluorescence_AU"

STATUS_VALID = "valid"
STATUS_NO_FINITE = "no_finite_samples"
STATUS_INSUFFICIENT = "insufficient_samples"
STATUS_INVALID_DENOMINATOR = "invalid_denominator"
STATUS_UNAVAILABLE = "tonic_unavailable"

REASON_GLOBAL_FIT_FAILED = "global_fit_failed"
REASON_NONPOSITIVE_SLOPE = "nonpositive_global_slope"
REASON_INVALID_GLOBAL_DENOMINATOR = "invalid_global_denominator"

SUMMARY_COLUMNS = [
    "roi",
    "session_index",
    "source_file",
    "session_start_time",
    "tonic_value",
    "status",
    "tonic_method",
    "units",
    "n_finite_samples",
    "percentile",
    "global_slope",
    "global_intercept",
    "global_fit_n_used",
    "fallback_reason",
]


class TonicSessionSummaryError(RuntimeError):
    """The session-level tonic summary could not be produced at all."""


def _authoritative_sessions(tonic_out_dir: str, cache) -> list[dict]:
    """Authoritative session records joined to real cache contributions.

    Falls back to dense cache order only when no authoritative session index is
    present (e.g. a directly constructed cache); a present-but-inconsistent
    index still fails closed inside the existing builder.
    """
    from photometry_pipeline.io.hdf5_cache_reader import (
        list_cache_chunk_ids,
        list_cache_source_files,
    )
    from photometry_pipeline.viz.phasic_data_prep import (
        build_authoritative_plot_sessions,
    )

    chunk_ids = list_cache_chunk_ids(cache)
    try:
        source_files = list_cache_source_files(cache)
    except Exception:
        source_files = []

    sessions = build_authoritative_plot_sessions(tonic_out_dir, chunk_ids, source_files)
    if sessions is not None:
        return sessions

    return [
        {
            "session_index": position,
            "cache_chunk_id": int(chunk_id),
            "source_file": (
                str(source_files[position]) if position < len(source_files) else ""
            ),
            "status": STATUS_VALID,
            "expected_start_time": None,
            "expected_start_time_text": "",
        }
        for position, chunk_id in enumerate(chunk_ids)
    ]


def _session_time_axis(sessions: Sequence[dict]) -> np.ndarray:
    """Proportional recording-time axis for the contributing sessions.

    Real expected start times are used when the authoritative index supplies
    them. Otherwise the authoritative session index itself is the time axis: it
    already skips missing slots, so it stays proportional to real elapsed time
    for a regular acquisition schedule. Only relative spacing matters to the
    bleach model, whose fitted trend is subtracted with its own mean anchor.
    """
    starts = [item.get("expected_start_time") for item in sessions]
    if starts and all(start is not None for start in starts):
        origin = starts[0]
        return np.array(
            [float((start - origin).total_seconds()) for start in starts], dtype=float
        )
    return np.array(
        [float(item.get("session_index", position)) for position, item in enumerate(sessions)],
        dtype=float,
    )


def _blank_row(roi: str, record: dict) -> dict:
    return {
        "roi": str(roi),
        "session_index": int(record.get("session_index", -1)),
        "source_file": str(record.get("source_file", "")),
        "session_start_time": str(record.get("expected_start_time_text", "") or ""),
        "tonic_value": np.nan,
        "status": str(record.get("status", "") or STATUS_UNAVAILABLE),
        "tonic_method": "",
        "units": "",
        "n_finite_samples": 0,
        "percentile": TONIC_PERCENTILE,
        "global_slope": np.nan,
        "global_intercept": np.nan,
        "global_fit_n_used": np.nan,
        "fallback_reason": "",
    }


def _load_roi_sessions(cache, roi: str, sessions: Sequence[dict]) -> list[dict]:
    """Read raw signal/reference for every contributing session of one ROI."""
    from photometry_pipeline.io.hdf5_cache_reader import load_cache_chunk_fields

    loaded = []
    for record in sessions:
        chunk_id = record.get("cache_chunk_id")
        if chunk_id is None:
            continue
        sig_raw, uv_raw = load_cache_chunk_fields(
            cache, roi, int(chunk_id), ["sig_raw", "uv_raw"]
        )
        loaded.append(
            {
                "record": record,
                "sig": np.asarray(sig_raw, dtype=float).reshape(-1),
                "uv": np.asarray(uv_raw, dtype=float).reshape(-1),
            }
        )
    return loaded


def _primary_session_value(sig: np.ndarray, uv: np.ndarray, slope: float, intercept: float):
    """One session's tonic dF/F0 value, or an explicit invalid status."""
    finite_pairs = np.isfinite(sig) & np.isfinite(uv)
    n_pairs = int(np.sum(finite_pairs))
    if n_pairs == 0:
        return np.nan, STATUS_NO_FINITE, 0
    if n_pairs < MIN_FINITE_SAMPLES:
        return np.nan, STATUS_INSUFFICIENT, n_pairs

    fitted = apply_global_fit(uv, slope, intercept)
    usable = (
        finite_pairs
        & np.isfinite(fitted)
        & (fitted > MIN_FITTED_ISOSBESTIC)
    )
    n_usable = int(np.sum(usable))
    if n_usable < MIN_FINITE_SAMPLES:
        return np.nan, STATUS_INVALID_DENOMINATOR, n_usable

    dff = (sig[usable] - fitted[usable]) / fitted[usable]
    dff = dff[np.isfinite(dff)]
    if dff.size < MIN_FINITE_SAMPLES:
        return np.nan, STATUS_INVALID_DENOMINATOR, int(dff.size)
    return float(np.percentile(dff, TONIC_PERCENTILE)), STATUS_VALID, int(dff.size)


def _signal_baseline(sig: np.ndarray):
    """One session's signal-only baseline, or an explicit invalid status."""
    finite = sig[np.isfinite(sig)]
    if finite.size == 0:
        return np.nan, STATUS_NO_FINITE, 0
    if finite.size < MIN_FINITE_SAMPLES:
        return np.nan, STATUS_INSUFFICIENT, int(finite.size)
    return float(np.percentile(finite, TONIC_PERCENTILE)), STATUS_VALID, int(finite.size)


def _rows_for_primary(roi, sessions, loaded, slope, intercept, n_used):
    by_chunk = {int(item["record"]["cache_chunk_id"]): item for item in loaded}
    rows = []
    for record in sessions:
        row = _blank_row(roi, record)
        row["tonic_method"] = METHOD_GLOBAL_ISOSBESTIC
        row["units"] = UNITS_FRACTION_DFF
        row["global_slope"] = float(slope)
        row["global_intercept"] = float(intercept)
        row["global_fit_n_used"] = int(n_used)
        chunk_id = record.get("cache_chunk_id")
        if chunk_id is not None:
            item = by_chunk[int(chunk_id)]
            value, status, n_finite = _primary_session_value(
                item["sig"], item["uv"], slope, intercept
            )
            row["tonic_value"] = value
            row["status"] = status
            row["n_finite_samples"] = n_finite
        rows.append(row)
    return rows


def _rows_for_fallback(roi, sessions, loaded, fallback_reason):
    by_chunk = {int(item["record"]["cache_chunk_id"]): item for item in loaded}
    rows = []
    baselines: dict[int, float] = {}
    for record in sessions:
        row = _blank_row(roi, record)
        row["tonic_method"] = METHOD_SIGNAL_ONLY
        row["units"] = UNITS_RAW_AU
        row["fallback_reason"] = fallback_reason
        chunk_id = record.get("cache_chunk_id")
        if chunk_id is not None:
            baseline, status, n_finite = _signal_baseline(by_chunk[int(chunk_id)]["sig"])
            row["status"] = status
            row["n_finite_samples"] = n_finite
            if status == STATUS_VALID:
                baselines[int(record["session_index"])] = baseline
        rows.append(row)

    usable = [
        record
        for record in sessions
        if int(record.get("session_index", -1)) in baselines
    ]
    if len(usable) < 3:
        reason = f"{fallback_reason};bleach_fit_failed:insufficient_valid_samples"
        for row in rows:
            if row["status"] == STATUS_VALID:
                row["status"] = STATUS_UNAVAILABLE
            row["tonic_value"] = np.nan
            row["fallback_reason"] = reason
        return rows

    times = _session_time_axis(usable)
    values = np.array(
        [baselines[int(record["session_index"])] for record in usable], dtype=float
    )
    trend, meta = _fit_exponential_bleach_trend(times, values)
    if meta.get("fallback", False):
        reason = f"{fallback_reason};bleach_fit_failed:{meta.get('reason', 'unknown')}"
        for row in rows:
            if row["status"] == STATUS_VALID:
                row["status"] = STATUS_UNAVAILABLE
            row["tonic_value"] = np.nan
            row["fallback_reason"] = reason
        return rows

    anchor = float(meta["anchor"])
    corrected = {
        int(record["session_index"]): float(values[position] - trend[position] + anchor)
        for position, record in enumerate(usable)
        if np.isfinite(trend[position])
    }
    for row in rows:
        index = int(row["session_index"])
        if row["status"] == STATUS_VALID:
            if index in corrected:
                row["tonic_value"] = corrected[index]
            else:
                row["status"] = STATUS_UNAVAILABLE
                row["fallback_reason"] = (
                    f"{fallback_reason};bleach_fit_failed:nonfinite_trend"
                )
    return rows


def build_tonic_session_summary(
    tonic_out_dir: str,
    *,
    rois: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Compute one tonic row per ROI per authoritative session."""
    from photometry_pipeline.io.hdf5_cache_reader import (
        list_cache_rois,
        open_tonic_cache,
    )

    cache_path = os.path.join(str(tonic_out_dir), "tonic_trace_cache.h5")
    if not os.path.isfile(cache_path):
        raise TonicSessionSummaryError(
            f"The tonic trace cache required for session-level tonic is missing: {cache_path}"
        )

    rows: list[dict[str, Any]] = []
    cache = open_tonic_cache(cache_path)
    try:
        available = list_cache_rois(cache)
        selected = list(rois) if rois else list(available)
        missing = [roi for roi in selected if roi not in available]
        if missing:
            raise TonicSessionSummaryError(
                f"Requested ROIs are not present in the tonic cache: {missing}"
            )
        sessions = _authoritative_sessions(str(tonic_out_dir), cache)

        for roi in selected:
            loaded = _load_roi_sessions(cache, roi, sessions)
            if not loaded:
                rows.extend(
                    _rows_for_fallback(roi, sessions, loaded, REASON_GLOBAL_FIT_FAILED)
                )
                continue

            uv_all = np.concatenate([item["uv"] for item in loaded])
            sig_all = np.concatenate([item["sig"] for item in loaded])
            slope, intercept, ok, n_used = compute_global_iso_fit_robust(uv_all, sig_all)

            if not ok:
                rows.extend(
                    _rows_for_fallback(roi, sessions, loaded, REASON_GLOBAL_FIT_FAILED)
                )
            elif not (float(slope) > 0.0):
                rows.extend(
                    _rows_for_fallback(roi, sessions, loaded, REASON_NONPOSITIVE_SLOPE)
                )
            else:
                primary_rows = _rows_for_primary(
                    roi, sessions, loaded, float(slope), float(intercept), int(n_used)
                )
                # The global fit is only usable for this ROI if applying it
                # actually yields a session-level value somewhere. One isolated
                # invalid session never switches the ROI's method; an ROI whose
                # fitted isosbestic is never safely positive has no primary
                # result at all and takes the signal-only fallback instead.
                if any(row["status"] == STATUS_VALID for row in primary_rows):
                    rows.extend(primary_rows)
                else:
                    rows.extend(
                        _rows_for_fallback(
                            roi, sessions, loaded, REASON_INVALID_GLOBAL_DENOMINATOR
                        )
                    )
    finally:
        cache.close()

    return rows


def write_tonic_session_summary(
    tonic_out_dir: str,
    output_path: str,
    *,
    rois: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Build the session-level tonic summary and write it as one CSV."""
    rows = build_tonic_session_summary(tonic_out_dir, rois=rois)
    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["roi", "session_index"], kind="stable")
    directory = os.path.dirname(str(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    frame.to_csv(output_path, index=False)

    methods = {
        str(roi): str(group["tonic_method"].iloc[0]) if len(group) else ""
        for roi, group in frame.groupby("roi", sort=True)
    } if not frame.empty else {}
    return {
        "output_path": str(output_path),
        "row_count": int(len(frame)),
        "rois_processed": sorted(methods),
        "tonic_method_by_roi": methods,
    }
