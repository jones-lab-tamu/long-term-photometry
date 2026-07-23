"""CR1-E1-A: read-only reopening model for completed continuous-RWD runs.

This module is deliberately read-only, mirroring
:mod:`photometry_pipeline.completed_run_review`'s own contract: it never
invokes a correction engine or a peak detector; a current continuous run's
already-persisted artifacts are bound to the terminal verifier's
(:func:`photometry_pipeline.run_completion_contract.classify_run_terminal_state`)
own requirements, and this module only reopens what that verifier already
accepted.

Why a new module instead of extending ``completed_run_review.py``
-------------------------------------------------------------------
``completed_run_review._load_completed_branch_review`` (the existing
intermittent completed-run reader) requires a *requested-versus-consumed
correction provenance* double-write (``run_metadata.json`` plus
``run_report.json``'s ``derived_settings.correction_provenance``, both
carrying a ``schema_version``/``analysis_mode``/``requested_by_roi`` record)
before it will treat a current run's ROI/session data as trustworthy
(``current_native``); a current run without that record is refused outright
with "has no verified native correction settings". None of the four accepted
continuous-RWD run producers
(``guided_continuous_rwd_{correction,tonic,phasic,combined}_run.py``) write
that record -- it is an intermittent, per-ROI-correction-plan concept with no
continuous analogue -- so routing a continuous run through the existing
reader would either hard-refuse every continuous run, or require fabricating
that record from the writer side (out of scope: those four modules are
accepted and this patch does not touch them).

Separately, even if that gate were bypassed, the existing reader's session
model (``CompletedReviewSession``) falls back to treating **every HDF5 cache
chunk as one "session"** whenever no frozen per-session input manifest is
present (`` _load_sessions_index`` returns empty lists) -- exactly the
outcome for continuous runs, since they always declare
``chunked_input_processing=False``. Reusing that fallback would render "579
session(s) processed" for one 96.5-hour recording, precisely the false
session semantics this patch must avoid (see CR1-E1 handoff section 8).

This module therefore reads continuous runs directly from their own
authoritative, already-written artifacts -- the normalized ``run_mode`` (via
``classify_run_terminal_state``) and each producer's own root
``run_report.json`` (which already records recording identity, target-grid
identity, included ROI order via ``run_mode``, and per-family cache/summary/
event paths and row counts) -- with no session concept and no per-chunk
reinterpretation. It represents one continuous recording, never 579 windows
as sessions.

Scope: this is CR1-E1-A only (recognition and loading). It does not touch
``gui/run_report_viewer.py`` or ``gui/main_window.py`` -- presenting this
model in the existing Results UI is CR1-E1-B, a distinct, not-yet-implemented
patch, so that no half-wired UI element is introduced here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from photometry_pipeline.feature_event_provenance import (
    PROVENANCE_MODE_CURRENT,
    classify_provenance_contract,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
    open_tonic_cache,
)
from photometry_pipeline.run_completion_contract import (
    PROFILE_CONTINUOUS,
    classify_run_terminal_state,
)

_REQUIRED_EVENT_COLUMNS = ("roi", "chunk_id", "window_index", "global_time_sec", "polarity")


class CompletedContinuousRwdReviewError(RuntimeError):
    """A completed continuous-RWD run could not be reopened for Review.

    Every message is written to be shown to a scientist directly (see module
    docstring / CR1-E1 handoff section 19) -- callers should surface
    ``str(exc)`` as the user-facing error, not a traceback.
    """


@dataclass(frozen=True)
class ContinuousReviewFinalWindow:
    """The true start/end/duration of the recording's final analysis
    window -- which may legitimately be shorter than every earlier window
    (see CR1-E1 handoff section 16). Never padded, omitted, or flagged as
    incomplete."""

    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass(frozen=True)
class ContinuousRunOverview:
    """A truthful, one-recording summary of one completed continuous-RWD
    run. Cheap to build: reads ``run_report.json``, the persisted event
    table, and one HDF5 attribute group -- never a full-resolution trace
    array (see ``load_continuous_roi_trace`` for on-demand trace loading)."""

    run_dir: str
    run_id: str
    terminal_state: str
    acquisition_mode: str
    included_roi_ids: tuple[str, ...]
    correction_completed: bool
    tonic_analysis: bool
    phasic_analysis: bool
    feature_extraction_ran: bool
    recording_identity: str
    target_grid_identity: str
    target_sample_count: int
    corrected_segment_count: int
    corrected_cache_relative_path: str
    tonic_cache_relative_path: str | None = None
    tonic_window_row_counts: dict[str, int] = field(default_factory=dict)
    tonic_summary_relative_paths: dict[str, str] = field(default_factory=dict)
    phasic_cache_relative_path: str | None = None
    features_relative_path: str | None = None
    events_relative_path: str | None = None
    phasic_window_row_counts: dict[str, int] = field(default_factory=dict)
    phasic_summary_relative_paths: dict[str, str] = field(default_factory=dict)
    phasic_event_counts_by_roi: dict[str, int] = field(default_factory=dict)
    phasic_event_total: int = 0
    final_window: ContinuousReviewFinalWindow | None = None
    summary_narrative: str = ""


@dataclass(frozen=True)
class ContinuousRoiTrace:
    """One selected ROI's full continuous trace, concatenated in
    chronological order across every recorded window. Loading is always
    scoped to exactly one ROI and one analysis family -- never every ROI or
    every family at once (see CR1-E1 handoff section 18)."""

    roi_id: str
    family: str
    fs_hz: float
    time_sec: np.ndarray
    sig_raw: np.ndarray
    primary_trace: np.ndarray
    primary_trace_label: str


def _read_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis is missing its {os.path.basename(path)} record."
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis's {os.path.basename(path)} could not be read ({exc})."
        ) from exc
    if not isinstance(data, dict):
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis's {os.path.basename(path)} is not a valid record."
        )
    return data


def _final_window_from_cache(cache_path: str, roi_id: str, opener) -> ContinuousReviewFinalWindow:
    cache = opener(cache_path)
    try:
        chunk_ids = list_cache_chunk_ids(cache)
        if not chunk_ids:
            raise CompletedContinuousRwdReviewError(
                "This completed analysis's saved trace cache has no recorded analysis windows."
            )
        last_chunk_id = sorted(chunk_ids)[-1]
        attrs = load_cache_chunk_attrs(cache, roi_id, last_chunk_id)
    finally:
        cache.close()
    try:
        return ContinuousReviewFinalWindow(
            start_sec=float(attrs["window_start_sec"]),
            end_sec=float(attrs["window_end_sec"]),
            duration_sec=float(attrs["window_duration_sec"]),
        )
    except KeyError as exc:
        raise CompletedContinuousRwdReviewError(
            "This completed analysis's saved trace cache is missing timing for its "
            "final recording window."
        ) from exc


def _load_and_validate_events_csv(
    events_path: str,
    *,
    included_roi_ids: tuple[str, ...],
    recording_support_end_sec: float | None = None,
) -> pd.DataFrame:
    """Load the persisted recording-global event table and verify its own
    internal consistency -- schema, ROI membership, finite times within the
    recording's own time support, chronological order per region, and valid
    polarity values. This never reruns detection: it can only confirm the
    saved rows are internally coherent, not re-derive ground truth (see
    CR1-E1 handoff section 27).
    """
    if not os.path.isfile(events_path):
        raise CompletedContinuousRwdReviewError(
            "This completed analysis could not be reopened because its saved event "
            "results are missing."
        )
    try:
        df = pd.read_csv(events_path)
    except Exception as exc:
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis's saved event results are invalid ({exc})."
        ) from exc
    missing_columns = [c for c in _REQUIRED_EVENT_COLUMNS if c not in df.columns]
    if missing_columns:
        raise CompletedContinuousRwdReviewError(
            "This completed analysis's saved event results are invalid (missing "
            f"columns: {missing_columns})."
        )
    if df.empty:
        return df
    unknown_rois = sorted(set(df["roi"].astype(str)) - set(included_roi_ids))
    if unknown_rois:
        raise CompletedContinuousRwdReviewError(
            "This completed analysis's saved event results reference a region that "
            f"is not part of this recording ({unknown_rois})."
        )
    times = pd.to_numeric(df["global_time_sec"], errors="coerce")
    times_array = times.to_numpy(dtype=float)
    if times.isna().any() or not np.all(np.isfinite(times_array)):
        raise CompletedContinuousRwdReviewError(
            "This completed analysis's saved event results are invalid (a non-finite "
            "event time)."
        )
    if recording_support_end_sec is not None:
        tolerance_sec = 1e-3
        if np.any(times_array < -tolerance_sec) or np.any(
            times_array > recording_support_end_sec + tolerance_sec
        ):
            raise CompletedContinuousRwdReviewError(
                "This completed analysis's saved event results are invalid (an event "
                "time falls outside the recording's own time support)."
            )
    polarities = pd.to_numeric(df["polarity"], errors="coerce")
    if polarities.isna().any() or not set(polarities.unique().tolist()) <= {-1.0, 1.0}:
        raise CompletedContinuousRwdReviewError(
            "This completed analysis's saved event results are invalid (an event "
            "polarity outside the accepted +1/-1 values)."
        )
    for roi_id, roi_rows in df.groupby("roi"):
        roi_times = pd.to_numeric(roi_rows["global_time_sec"], errors="coerce").to_numpy(dtype=float)
        if np.any(np.diff(roi_times) < 0):
            raise CompletedContinuousRwdReviewError(
                "This completed analysis's saved event results are invalid (region "
                f"{roi_id} is not in chronological order)."
            )
    return df


def load_continuous_run_overview(run_dir: str) -> ContinuousRunOverview:
    """Load a compact, truthful overview of one completed continuous-RWD run.

    Fails closed (raising :class:`CompletedContinuousRwdReviewError` with a
    scientist-facing message) unless:

    * :func:`classify_run_terminal_state` reports current success;
    * the run mode declares ``acquisition_mode == "continuous"`` and
      ``deliverable_profile == PROFILE_CONTINUOUS``;
    * every artifact the run mode declares as completed (tonic and/or
      phasic) actually exists and is internally consistent.

    Never infers success from file presence alone, and never treats a
    partial, failed, cancelled, or non-continuous run as reopenable here.
    """
    resolved = os.path.realpath(str(run_dir))
    classification = classify_run_terminal_state(resolved)
    if not classification.is_success:
        raise CompletedContinuousRwdReviewError(
            "This analysis could not be reopened because it did not complete "
            f"successfully ({classification.reason})."
        )
    run_mode = classification.run_mode or {}
    acquisition_mode = str(run_mode.get("acquisition_mode", ""))
    if acquisition_mode != "continuous" or run_mode.get("deliverable_profile") != PROFILE_CONTINUOUS:
        raise CompletedContinuousRwdReviewError(
            "This analysis is not a continuous recording and cannot be opened here."
        )
    included_roi_ids = tuple(str(roi) for roi in (run_mode.get("expected_rois") or ()))
    if not included_roi_ids:
        raise CompletedContinuousRwdReviewError(
            "This completed analysis has no recorded regions of interest."
        )

    report = _read_json(os.path.join(resolved, "run_report.json"))
    source = report.get("source") if isinstance(report.get("source"), dict) else {}
    target_grid = report.get("target_grid") if isinstance(report.get("target_grid"), dict) else {}
    corrected_cache = (
        report.get("corrected_cache") if isinstance(report.get("corrected_cache"), dict) else {}
    )
    corrected_cache_relative_path = str(corrected_cache.get("relative_path", ""))
    corrected_cache_abs = (
        os.path.join(resolved, corrected_cache_relative_path) if corrected_cache_relative_path else ""
    )
    if not corrected_cache_relative_path or not os.path.isfile(corrected_cache_abs):
        raise CompletedContinuousRwdReviewError(
            "This completed analysis could not be reopened because its corrected "
            "trace cache is missing."
        )

    tonic_analysis = bool(run_mode.get("tonic_analysis"))
    phasic_analysis = bool(run_mode.get("phasic_analysis"))
    feature_extraction_ran = bool(run_mode.get("feature_extraction_ran"))

    tonic_cache_relative_path: str | None = None
    tonic_window_row_counts: dict[str, int] = {}
    tonic_summary_relative_paths: dict[str, str] = {}
    if tonic_analysis:
        tonic_section = report.get("tonic_analysis")
        if not isinstance(tonic_section, dict):
            raise CompletedContinuousRwdReviewError(
                "This completed analysis is missing its tonic analysis record."
            )
        tonic_cache_relative_path = str(tonic_section.get("trace_cache_relative_path", ""))
        if not tonic_cache_relative_path or not os.path.isfile(
            os.path.join(resolved, tonic_cache_relative_path)
        ):
            raise CompletedContinuousRwdReviewError(
                "This completed analysis could not be reopened because its tonic "
                "trace file is missing."
            )
        row_counts = tonic_section.get("window_row_counts")
        paths = tonic_section.get("output_relative_paths")
        if not isinstance(row_counts, dict) or not isinstance(paths, dict):
            raise CompletedContinuousRwdReviewError(
                "This completed analysis is missing its tonic window summary record."
            )
        tonic_window_row_counts = {str(k): int(v) for k, v in row_counts.items()}
        tonic_summary_relative_paths = {str(k): str(v) for k, v in paths.items()}
        for roi_id in included_roi_ids:
            if roi_id not in tonic_window_row_counts or roi_id not in tonic_summary_relative_paths:
                raise CompletedContinuousRwdReviewError(
                    f"This completed analysis is missing its tonic results for region {roi_id}."
                )
            if not os.path.isfile(os.path.join(resolved, tonic_summary_relative_paths[roi_id])):
                raise CompletedContinuousRwdReviewError(
                    "This completed analysis could not be reopened because its tonic "
                    f"summary for region {roi_id} is missing."
                )

    phasic_cache_relative_path: str | None = None
    features_relative_path: str | None = None
    events_relative_path: str | None = None
    phasic_window_row_counts: dict[str, int] = {}
    phasic_summary_relative_paths: dict[str, str] = {}
    phasic_event_counts_by_roi: dict[str, int] = {}
    phasic_event_total = 0
    if phasic_analysis:
        if not feature_extraction_ran:
            raise CompletedContinuousRwdReviewError(
                "This completed analysis records phasic analysis without event "
                "detection; it cannot be reopened."
            )
        phasic_section = report.get("phasic_analysis")
        if not isinstance(phasic_section, dict):
            raise CompletedContinuousRwdReviewError(
                "This completed analysis is missing its phasic analysis record."
            )
        phasic_cache_relative_path = str(phasic_section.get("trace_cache_relative_path", ""))
        features_relative_path = str(phasic_section.get("features_relative_path", ""))
        events_relative_path = str(phasic_section.get("events_relative_path", ""))
        for label, rel_path in (
            ("phasic trace file", phasic_cache_relative_path),
            ("feature table", features_relative_path),
            ("saved event results", events_relative_path),
        ):
            if not rel_path or not os.path.isfile(os.path.join(resolved, rel_path)):
                raise CompletedContinuousRwdReviewError(
                    f"This completed analysis could not be reopened because its {label} is missing."
                )
        row_counts = phasic_section.get("window_row_counts")
        paths = phasic_section.get("output_relative_paths")
        reported_event_counts = phasic_section.get("per_roi_event_counts")
        if (
            not isinstance(row_counts, dict)
            or not isinstance(paths, dict)
            or not isinstance(reported_event_counts, dict)
        ):
            raise CompletedContinuousRwdReviewError(
                "This completed analysis is missing its phasic window summary record."
            )
        phasic_window_row_counts = {str(k): int(v) for k, v in row_counts.items()}
        phasic_summary_relative_paths = {str(k): str(v) for k, v in paths.items()}
        for roi_id in included_roi_ids:
            if (
                roi_id not in phasic_window_row_counts
                or roi_id not in phasic_summary_relative_paths
                or roi_id not in reported_event_counts
            ):
                raise CompletedContinuousRwdReviewError(
                    f"This completed analysis is missing its phasic results for region {roi_id}."
                )
            if not os.path.isfile(os.path.join(resolved, phasic_summary_relative_paths[roi_id])):
                raise CompletedContinuousRwdReviewError(
                    "This completed analysis could not be reopened because its phasic "
                    f"summary for region {roi_id} is missing."
                )

        phasic_out_dir = os.path.join(resolved, "_analysis", "phasic_out")
        provenance_mode, _provenance_path, provenance_reason = classify_provenance_contract(
            phasic_out_dir
        )
        if provenance_mode != PROVENANCE_MODE_CURRENT:
            raise CompletedContinuousRwdReviewError(
                "This completed analysis's saved detection settings could not be "
                f"verified ({provenance_reason})."
            )

        # Computed before event validation so the persisted event table can
        # be checked against the recording's own true time support, not
        # merely finiteness.
        phasic_final_window = _final_window_from_cache(
            os.path.join(resolved, phasic_cache_relative_path),
            included_roi_ids[0],
            open_phasic_cache,
        )
        events_df = _load_and_validate_events_csv(
            os.path.join(resolved, events_relative_path),
            included_roi_ids=included_roi_ids,
            recording_support_end_sec=phasic_final_window.end_sec,
        )
        for roi_id in included_roi_ids:
            actual_count = int((events_df["roi"] == roi_id).sum())
            reported_count = int(reported_event_counts.get(roi_id, -1))
            if actual_count != reported_count:
                raise CompletedContinuousRwdReviewError(
                    "This completed analysis's saved event results are invalid: "
                    f"region {roi_id} has {actual_count} saved events but the run "
                    f"recorded {reported_count}."
                )
            phasic_event_counts_by_roi[roi_id] = actual_count
        phasic_event_total = int(len(events_df))

    if phasic_cache_relative_path:
        final_window = phasic_final_window
    elif tonic_cache_relative_path:
        final_window = _final_window_from_cache(
            os.path.join(resolved, tonic_cache_relative_path),
            included_roi_ids[0],
            open_tonic_cache,
        )
    else:
        final_window = _final_window_from_cache(
            corrected_cache_abs, included_roi_ids[0], open_phasic_cache
        )

    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    run_context = report.get("run_context") if isinstance(report.get("run_context"), dict) else {}

    return ContinuousRunOverview(
        run_dir=resolved,
        run_id=str(classification.run_id or run_context.get("run_id", "")),
        terminal_state=classification.state,
        acquisition_mode=acquisition_mode,
        included_roi_ids=included_roi_ids,
        correction_completed=True,
        tonic_analysis=tonic_analysis,
        phasic_analysis=phasic_analysis,
        feature_extraction_ran=feature_extraction_ran,
        recording_identity=str(source.get("recording_identity", "")),
        target_grid_identity=str(target_grid.get("target_grid_identity", "")),
        target_sample_count=int(target_grid.get("target_sample_count", 0)),
        corrected_segment_count=int(corrected_cache.get("corrected_segment_count", 0)),
        corrected_cache_relative_path=corrected_cache_relative_path,
        tonic_cache_relative_path=tonic_cache_relative_path,
        tonic_window_row_counts=tonic_window_row_counts,
        tonic_summary_relative_paths=tonic_summary_relative_paths,
        phasic_cache_relative_path=phasic_cache_relative_path,
        features_relative_path=features_relative_path,
        events_relative_path=events_relative_path,
        phasic_window_row_counts=phasic_window_row_counts,
        phasic_summary_relative_paths=phasic_summary_relative_paths,
        phasic_event_counts_by_roi=phasic_event_counts_by_roi,
        phasic_event_total=phasic_event_total,
        final_window=final_window,
        summary_narrative=str(summary.get("narrative", "")),
    )


def load_continuous_window_summary(run_dir: str, *, family: str, roi_id: str) -> pd.DataFrame:
    """Load one ROI's already-persisted per-window summary table
    (``continuous_tonic_window_summary.csv`` / ``continuous_phasic_window_
    summary.csv``) for the requested analysis family, on demand."""
    if family not in ("tonic", "phasic"):
        raise ValueError(f"Unsupported analysis family: {family!r}")
    overview = load_continuous_run_overview(run_dir)
    paths = (
        overview.tonic_summary_relative_paths
        if family == "tonic"
        else overview.phasic_summary_relative_paths
    )
    if roi_id not in paths:
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis has no {family} summary for region {roi_id}."
        )
    path = os.path.join(overview.run_dir, paths[roi_id])
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis's {family} summary for region {roi_id} could "
            f"not be read ({exc})."
        ) from exc


def load_continuous_phasic_events(run_dir: str, *, roi_id: str | None = None) -> pd.DataFrame:
    """Load the persisted recording-global event table, optionally filtered
    to one region. This is the sole source of phasic event markers/counts
    for a continuous run -- it never reruns detection (see module
    docstring)."""
    overview = load_continuous_run_overview(run_dir)
    if not overview.phasic_analysis or not overview.events_relative_path:
        raise CompletedContinuousRwdReviewError(
            "This completed analysis has no phasic event results."
        )
    events_path = os.path.join(overview.run_dir, overview.events_relative_path)
    df = _load_and_validate_events_csv(
        events_path,
        included_roi_ids=overview.included_roi_ids,
        recording_support_end_sec=(
            overview.final_window.end_sec if overview.final_window is not None else None
        ),
    )
    if roi_id is not None:
        df = df[df["roi"] == roi_id].reset_index(drop=True)
    return df


def load_continuous_roi_trace(run_dir: str, *, family: str, roi_id: str) -> ContinuousRoiTrace:
    """Load one selected ROI's full continuous trace from the accepted
    cache, concatenated in chronological order across every recorded
    window. Loads only the one requested ROI and family -- never every ROI
    or every family at once (see CR1-E1 handoff section 18)."""
    if family not in ("tonic", "phasic"):
        raise ValueError(f"Unsupported analysis family: {family!r}")
    overview = load_continuous_run_overview(run_dir)
    cache_relative_path = (
        overview.tonic_cache_relative_path if family == "tonic" else overview.phasic_cache_relative_path
    )
    if not cache_relative_path:
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis has no {family} trace results."
        )
    if roi_id not in overview.included_roi_ids:
        raise CompletedContinuousRwdReviewError(
            f"Region {roi_id!r} is not part of this completed analysis."
        )
    cache_path = os.path.join(overview.run_dir, cache_relative_path)
    opener = open_tonic_cache if family == "tonic" else open_phasic_cache
    field_name = "deltaF" if family == "tonic" else "dff"
    label = "Tonic dF/F" if family == "tonic" else "Phasic dF/F"

    cache = opener(cache_path)
    try:
        chunk_ids = sorted(list_cache_chunk_ids(cache))
        if not chunk_ids:
            raise CompletedContinuousRwdReviewError(
                f"This completed analysis's {family} trace cache has no recorded "
                "analysis windows."
            )
        time_parts: list[np.ndarray] = []
        sig_parts: list[np.ndarray] = []
        trace_parts: list[np.ndarray] = []
        fs_hz: float | None = None
        for chunk_id in chunk_ids:
            attrs = load_cache_chunk_attrs(cache, roi_id, chunk_id)
            local_time, sig_raw, primary = load_cache_chunk_fields(
                cache, roi_id, chunk_id, ["time_sec", "sig_raw", field_name]
            )
            if fs_hz is None:
                fs_hz = float(attrs["fs_hz"])
            window_start = float(attrs["window_start_sec"])
            time_parts.append(window_start + np.asarray(local_time, dtype=np.float64))
            sig_parts.append(np.asarray(sig_raw, dtype=np.float64))
            trace_parts.append(np.asarray(primary, dtype=np.float64))
    except Exception as exc:
        if isinstance(exc, CompletedContinuousRwdReviewError):
            raise
        raise CompletedContinuousRwdReviewError(
            f"This completed analysis's {family} trace for region {roi_id} could not "
            f"be read ({exc})."
        ) from exc
    finally:
        cache.close()

    assert fs_hz is not None
    return ContinuousRoiTrace(
        roi_id=roi_id,
        family=family,
        fs_hz=fs_hz,
        time_sec=np.concatenate(time_parts),
        sig_raw=np.concatenate(sig_parts),
        primary_trace=np.concatenate(trace_parts),
        primary_trace_label=label,
    )
