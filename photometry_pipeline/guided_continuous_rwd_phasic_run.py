"""CR1-D3b-B: publish the accepted D3b-A one-recording phasic detection
result through the existing phasic artifact contract and completed-run
lifecycle.

CR1-D3b-A (:mod:`photometry_pipeline.guided_continuous_rwd_phasic_detection`)
already solved the scientific problem: it reconstructs each ROI as one
unbroken continuous trace from an accepted D1 corrected cache and runs the
existing, unmodified detector (``core.feature_extraction.extract_features`` /
``get_peak_indices_for_trace``) exactly once per ROI, returning one
recording-global threshold/event result per ROI, entirely in memory. This
module does not repeat, adapt, or second-guess that computation. It only
publishes it.

Publication problem
--------------------
The existing production artifact contract for a phasic run
(``photometry_pipeline/pipeline.py``'s Pass 2 loop,
``photometry_pipeline/continuous_outputs.py``'s
``generate_continuous_phasic_summary``, and
``photometry_pipeline/run_completion_contract.py``) was built around a
chunk/session-scoped ``features.csv``: one row per ``(chunk_id, roi)``, and a
continuous window-summary generator that is a pure read/join over that file
plus the phasic trace cache's own per-chunk window metadata -- it never
reruns detection itself (inspected directly; see CR1-D3b-B handoff section 3).
Writing a single recording-level row (e.g. ``chunk_id=0``) would collapse the
per-window summary the scientist actually reads into one row spanning the
whole 96.5-hour recording, which is exactly the "chunk 0 implies only the
first 600 seconds" failure mode this integration must avoid.

The chosen, narrow resolution: keep the existing one-row-per-``(chunk_id,
roi)`` schema (so ``generate_continuous_phasic_summary`` runs unmodified and
produces one genuine per-window row per accepted D1 storage segment), but
derive every row's values from D3b-A's already-established global result
instead of rerunning anything chunk-local:

* ``mean``/``median``/``std``/``mad`` are D3b-A's recording-global threshold
  statistics, copied identically onto every row for that ROI. They describe
  the one global threshold this recording's detection actually used; they are
  never recomputed from a window-local slice, which would silently
  reintroduce the exact chunk-local-threshold defect D3b-A exists to avoid.
* ``peak_count`` is the count of D3b-A's already-detected global event
  positions whose integer global sample index falls inside this chunk's own
  contiguous sample range -- a partition of one already-established event
  list, not a second detection pass. Sample-index arithmetic (not
  floating-point time-interval comparison) decides ownership, so a boundary
  event is assigned to exactly one chunk by construction.
* ``auc`` is a direct numerical re-integration (``compute_auc_above_
  threshold``, the same helper ``extract_features`` itself calls for the AUC
  step) of this chunk's own corrected event-signal slice against D3b-A's
  established global baseline -- arithmetic, not peak/threshold detection.

Individual event times/polarities have no existing persisted artifact
anywhere in the application (verified directly: every existing consumer that
needs event markers recomputes them from the trace on demand). Since this
integration explicitly forbids that recomputation at 96.5-hour scale and the
per-window summary's window-boundary tests require checking exactly which
events landed where, this module adds one small new per-run CSV,
``continuous_phasic_events.csv``, published as an optional (not
completion-contract-mandatory) manifest artifact -- not a new event-storage
architecture, just the individual events D3b-A already computed.

Scope: this integrates exactly one downstream path (phasic). It does not run
or modify tonic/combined analysis, does not connect to the GUI or worker, and
does not enable Guided continuous Run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from photometry_pipeline.config import Config
from photometry_pipeline.continuous_outputs import (
    PHASIC_SUMMARY_FILENAME,
    generate_continuous_phasic_summary,
)
from photometry_pipeline.core.feature_extraction import (
    compute_auc_above_threshold,
    normalize_signal_excursion_polarity,
)
from photometry_pipeline.core.reporting import generate_run_report
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.feature_event_provenance import (
    FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
    FEATURE_EVENT_PROVENANCE_FILENAME,
    build_feature_event_provenance_payload,
)
from photometry_pipeline.guided_continuous_rwd_block_plan import (
    GuidedContinuousRwdBlockPlan,
)
from photometry_pipeline.guided_continuous_rwd_correction_pass import (
    GuidedContinuousRwdCorrectionPassCompletion,
    GuidedContinuousRwdCorrectionPassError,
    GuidedContinuousRwdCorrectionPassTraversal,
    iterate_guided_continuous_rwd_corrected_segments,
)
from photometry_pipeline.guided_continuous_rwd_correction_pass_persistence import (
    persist_guided_continuous_rwd_correction_pass,
)
from photometry_pipeline.guided_continuous_rwd_correction_run import (
    CORRECTED_CACHE_RELATIVE_PATH,
    _allocate_run_directory,
    _is_cancelled_traversal,
    _per_roi_provenance,
    _validate_persisted_cache,
    _write_json,
    _write_running_status,
    _write_terminal_failure_status,
)
from photometry_pipeline.guided_continuous_rwd_correction_segments import (
    GuidedContinuousRwdCorrectionSegmentPlan,
    GuidedContinuousRwdDynamicF0Authority,
)
from photometry_pipeline.guided_continuous_rwd_phasic_detection import (
    GuidedContinuousRwdPhasicDetectionError,
    GuidedContinuousRwdPhasicDetectionResult,
    detect_guided_continuous_rwd_phasic_features,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.guided_execution_payloads import (
    GuidedExecutionStartupMappingContract,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
)
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY,
    MANIFEST_FILENAME,
    PROFILE_CONTINUOUS,
    REPORT_COMPLETION_KEY,
    RUN_REPORT_FILENAME,
    STATUS_FILENAME,
    build_continuous_window_index,
    build_manifest_completion_block,
    build_report_completion_block,
    build_status_completion_block,
    classify_run_terminal_state,
    normalize_run_mode,
    sha256_file,
)

_RUN_ID_PREFIX = "continuous_rwd_phasic_run"
_RUN_PROFILE = "guided_continuous_rwd_phasic"
_RUN_TYPE = "phasic_only"
_TOOL_NAME = "photometry_pipeline.guided_continuous_rwd_phasic_run"
PHASIC_ANALYSIS_RELATIVE_DIR = os.path.join("_analysis", "phasic_out")
PHASIC_CACHE_FILENAME = "phasic_trace_cache.h5"
PHASIC_FEATURES_RELATIVE_DIR = "features"
PHASIC_FEATURES_FILENAME = "features.csv"
CONTINUOUS_PHASIC_EVENTS_FILENAME = "continuous_phasic_events.csv"

# Each reused lower-layer module raises its own cancellation exception type
# with its own narrow category name for "cancellation_requested() was true".
# Mapping the exact (type, category) pair -- rather than matching on category
# string alone, or on exc type alone -- avoids ever misclassifying a genuine
# same-type failure (e.g. a D3b-A cache-authority mismatch) as cancellation.
# Mirrors guided_continuous_rwd_correction_pass._is_lower_layer_cancellation.
_LOWER_LAYER_CANCELLATION_CATEGORIES: dict[type, str] = {
    GuidedContinuousRwdCorrectionPassError: "segment_correction_pass_interrupted",
    GuidedContinuousRwdPhasicDetectionError: "phasic_detection_interrupted",
}


def _is_lower_layer_cancellation(exc: Exception) -> bool:
    expected_category = _LOWER_LAYER_CANCELLATION_CATEGORIES.get(type(exc))
    return expected_category is not None and getattr(exc, "category", None) == expected_category


class GuidedContinuousRwdPhasicRunError(RuntimeError):
    """A narrow refusal while executing or publishing one continuous-RWD
    phasic run. Errors raised directly by C4c/D1/D3b-A propagate unchanged;
    this exception covers only run-directory-level and phasic-artifact-
    production concerns."""


@dataclass(frozen=True)
class GuidedContinuousRwdPhasicRunResult:
    """What a caller needs to know about one completed continuous-RWD
    phasic run."""

    run_dir: str
    run_id: str
    corrected_cache_path: str
    phasic_cache_path: str
    features_path: str
    events_path: str
    completion: GuidedContinuousRwdCorrectionPassCompletion
    detection: GuidedContinuousRwdPhasicDetectionResult
    terminal_state: str
    phasic_summary_paths: dict[str, str]
    phasic_summary_row_counts: dict[str, int]


def _build_run_mode(included_roi_ids: tuple[str, ...]) -> dict:
    return normalize_run_mode(
        run_profile=_RUN_PROFILE,
        run_type=_RUN_TYPE,
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=list(included_roi_ids),
        continuous_outputs_ran=True,
        chunked_input_processing=False,
        shared_input_manifest=False,
    )


def _window_auc(
    trace: np.ndarray,
    local_time_sec: np.ndarray,
    *,
    baseline: float,
    polarity: str,
) -> float:
    """Numerically re-integrate AUC over one D1 chunk's own corrected
    event-signal slice against an already-established baseline.

    Mirrors ``core.feature_extraction.extract_features``'s finite-run
    splitting exactly (so a window containing an internal NaN gap integrates
    identically to how the production detector itself would have summed that
    same slice), but performs no threshold estimation or peak search -- this
    is arithmetic over already-corrected samples, not detection.
    """
    is_valid = np.isfinite(trace)
    padded = np.concatenate(([False], is_valid, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    total = 0.0
    for s, e in zip(starts, ends):
        run_y = trace[s:e]
        run_t = local_time_sec[s:e]
        if run_y.shape[0] < 2:
            continue
        total += compute_auc_above_threshold(
            run_y, baseline, time_s=run_t, signal_excursion_polarity=polarity
        )
    return total


def _publish_phasic_cache_and_features(
    *,
    corrected_cache_path: str,
    phasic_cache_path: str,
    included_roi_ids: tuple[str, ...],
    detection: GuidedContinuousRwdPhasicDetectionResult,
    config: Config,
) -> tuple[list[dict], list[dict]]:
    """Republish D1's corrected per-segment arrays through the existing
    phasic-mode ``Hdf5TraceCacheWriter``, one storage chunk (all canonical
    ROIs) at a time, deriving one ``features.csv`` row and zero or more
    published-event rows per ``(chunk_id, roi)`` from D3b-A's already-
    established global result (see module docstring). Returns
    ``(feature_rows, event_rows)``.

    Raises ``GuidedContinuousRwdPhasicRunError`` if the per-window event
    partition does not exactly conserve D3b-A's own recording-global event
    count for any ROI.
    """
    event_signal_field = str(getattr(config, "event_signal", "dff"))
    cache_field = "dff" if event_signal_field == "dff" else "delta_f"
    auc_baseline_method = str(getattr(config, "event_auc_baseline", "zero"))
    polarity = normalize_signal_excursion_polarity(
        getattr(config, "signal_excursion_polarity", "positive")
    )

    source_cache = open_phasic_cache(corrected_cache_path)
    try:
        chunk_ids = list_cache_chunk_ids(source_cache)
        writer = Hdf5TraceCacheWriter(phasic_cache_path, "phasic", config)
        feature_rows: list[dict] = []
        event_rows: list[dict] = []
        counted_events: dict[str, int] = {roi_id: 0 for roi_id in included_roi_ids}
        cumulative_samples: dict[str, int] = {roi_id: 0 for roi_id in included_roi_ids}
        try:
            for chunk_id in chunk_ids:
                sig_cols, uv_cols, dff_cols, fit_ref_cols, delta_cols = [], [], [], [], []
                time_sec = None
                fs_hz_chunk = None
                source_file = ""
                window_meta: dict[str, float] = {}
                for roi_id in included_roi_ids:
                    attrs = load_cache_chunk_attrs(source_cache, roi_id, int(chunk_id))
                    (t, sig, uv, dff, fit_ref, delta_f) = load_cache_chunk_fields(
                        source_cache,
                        roi_id,
                        int(chunk_id),
                        ["time_sec", "sig_raw", "uv_raw", "dff", "fit_ref", "delta_f"],
                    )
                    local_time = np.asarray(t, dtype=np.float64)
                    if time_sec is None:
                        time_sec = local_time
                        fs_hz_chunk = float(attrs["fs_hz"])
                        source_file = str(attrs.get("source_file", ""))
                        window_meta = {
                            "acquisition_mode": "continuous",
                            "window_index": float(attrs["window_index"]),
                            "window_start_sec": float(attrs["window_start_sec"]),
                            "window_end_sec": float(attrs["window_end_sec"]),
                            "window_duration_sec": float(attrs["window_duration_sec"]),
                        }
                    sig_cols.append(np.asarray(sig, dtype=np.float64))
                    uv_cols.append(np.asarray(uv, dtype=np.float64))
                    dff_arr = np.asarray(dff, dtype=np.float64)
                    delta_arr = np.asarray(delta_f, dtype=np.float64)
                    dff_cols.append(dff_arr)
                    fit_ref_cols.append(np.asarray(fit_ref, dtype=np.float64))
                    delta_cols.append(delta_arr)

                    roi_detection = detection.per_roi[roi_id]
                    event_trace = dff_arr if cache_field == "dff" else delta_arr
                    n_samples = int(event_trace.shape[0])

                    auc_baseline = (
                        roi_detection.median if auc_baseline_method == "median" else 0.0
                    )
                    window_auc = _window_auc(
                        event_trace, local_time, baseline=auc_baseline, polarity=polarity
                    )

                    range_start = cumulative_samples[roi_id]
                    range_end = range_start + n_samples
                    global_sample_index = np.rint(
                        (roi_detection.peak_global_times_sec - detection.global_time_start_sec)
                        * detection.sampling_rate_hz
                    ).astype(np.int64)
                    in_window = (global_sample_index >= range_start) & (
                        global_sample_index < range_end
                    )
                    window_event_count = int(np.count_nonzero(in_window))
                    counted_events[roi_id] += window_event_count
                    cumulative_samples[roi_id] = range_end

                    for peak_time, peak_polarity in zip(
                        roi_detection.peak_global_times_sec[in_window],
                        roi_detection.peak_polarities[in_window],
                    ):
                        event_rows.append(
                            {
                                "roi": roi_id,
                                "chunk_id": int(chunk_id),
                                "window_index": int(chunk_id),
                                "global_time_sec": float(peak_time),
                                "polarity": int(peak_polarity),
                            }
                        )

                    feature_rows.append(
                        {
                            "chunk_id": int(chunk_id),
                            "source_file": source_file,
                            "roi": roi_id,
                            "mean": roi_detection.mean,
                            "median": roi_detection.median,
                            "std": roi_detection.std,
                            "mad": roi_detection.mad,
                            "peak_count": window_event_count,
                            "auc": window_auc,
                            "status": "valid",
                        }
                    )

                chunk = Chunk(
                    chunk_id=int(chunk_id),
                    source_file=source_file,
                    format="rwd",
                    time_sec=time_sec,
                    uv_raw=np.column_stack(uv_cols),
                    sig_raw=np.column_stack(sig_cols),
                    dff=np.column_stack(dff_cols),
                    uv_fit=np.column_stack(fit_ref_cols),
                    delta_f=np.column_stack(delta_cols),
                    fs_hz=float(fs_hz_chunk),
                    channel_names=list(included_roi_ids),
                    metadata=window_meta,
                )
                writer.add_chunk(chunk, chunk_id=int(chunk_id), source_file=source_file)
            writer.finalize()
        except Exception:
            writer.abort()
            raise
    finally:
        source_cache.close()

    for roi_id in included_roi_ids:
        expected = detection.per_roi[roi_id].event_count
        if counted_events[roi_id] != expected:
            raise GuidedContinuousRwdPhasicRunError(
                f"ROI {roi_id!r}: the per-window event partition ({counted_events[roi_id]}) "
                f"does not conserve D3b-A's recording-global event count ({expected})."
            )

    return feature_rows, event_rows


def _validate_phasic_cache(
    phasic_cache_path: str,
    *,
    included_roi_ids: tuple[str, ...],
    completion: GuidedContinuousRwdCorrectionPassCompletion,
) -> None:
    """Reopen the just-written phasic cache through the existing reader and
    confirm it faithfully represents the one continuous recording."""
    if not os.path.isfile(phasic_cache_path):
        raise GuidedContinuousRwdPhasicRunError(
            "The phasic trace cache is missing after phasic-cache production claimed success."
        )
    if os.path.isfile(phasic_cache_path + ".tmp"):
        raise GuidedContinuousRwdPhasicRunError(
            "A .tmp phasic-cache artifact remains after finalize."
        )
    cache = open_phasic_cache(phasic_cache_path)
    try:
        rois = list_cache_rois(cache)
        if rois != list(included_roi_ids):
            raise GuidedContinuousRwdPhasicRunError(
                "The phasic cache's ROI set/order does not match the accepted "
                f"review binding: cache={rois!r}, expected={list(included_roi_ids)!r}."
            )
        chunk_ids = list_cache_chunk_ids(cache)
        if len(chunk_ids) != completion.corrected_segment_count:
            raise GuidedContinuousRwdPhasicRunError(
                "The phasic cache's chunk count does not match the C4c "
                f"completion: cache={len(chunk_ids)}, "
                f"expected={completion.corrected_segment_count}."
            )
        if sorted(chunk_ids) != list(range(completion.corrected_segment_count)):
            raise GuidedContinuousRwdPhasicRunError(
                "The phasic cache's chunk identities are not a contiguous 0-based range."
            )
    finally:
        cache.close()


def _write_features_csv(features_dir: str, feature_rows: list[dict]) -> str:
    os.makedirs(features_dir, exist_ok=True)
    path = os.path.join(features_dir, PHASIC_FEATURES_FILENAME)
    columns = [
        "chunk_id",
        "source_file",
        "roi",
        "mean",
        "median",
        "std",
        "mad",
        "peak_count",
        "auc",
        "status",
    ]
    pd.DataFrame(feature_rows, columns=columns).to_csv(path, index=False)
    return path


def _write_events_csv(features_dir: str, event_rows: list[dict]) -> str:
    os.makedirs(features_dir, exist_ok=True)
    path = os.path.join(features_dir, CONTINUOUS_PHASIC_EVENTS_FILENAME)
    columns = ["roi", "chunk_id", "window_index", "global_time_sec", "polarity"]
    df = pd.DataFrame(event_rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["roi", "global_time_sec"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def _validate_events_csv(
    events_path: str,
    *,
    detection: GuidedContinuousRwdPhasicDetectionResult,
) -> None:
    """Reopen the published event-level CSV and confirm it exactly and
    losslessly represents D3b-A's recording-global event result."""
    df = pd.read_csv(events_path)
    for roi_id in detection.included_roi_ids:
        roi_detection = detection.per_roi[roi_id]
        roi_rows = df[df["roi"].astype(str) == str(roi_id)].sort_values("global_time_sec")
        if len(roi_rows) != roi_detection.event_count:
            raise GuidedContinuousRwdPhasicRunError(
                f"ROI {roi_id!r}: published event count ({len(roi_rows)}) does not "
                f"match D3b-A's recording-global event count ({roi_detection.event_count})."
            )
        published_times = roi_rows["global_time_sec"].to_numpy(dtype=np.float64)
        if not np.allclose(
            published_times, roi_detection.peak_global_times_sec, atol=1e-6
        ):
            raise GuidedContinuousRwdPhasicRunError(
                f"ROI {roi_id!r}: published event times do not match D3b-A's result."
            )
        published_polarities = roi_rows["polarity"].to_numpy(dtype=np.int64)
        if not np.array_equal(published_polarities, roi_detection.peak_polarities):
            raise GuidedContinuousRwdPhasicRunError(
                f"ROI {roi_id!r}: published event polarities do not match D3b-A's result."
            )
        if not roi_rows.empty:
            if published_times.min() < detection.global_time_start_sec - 1e-6:
                raise GuidedContinuousRwdPhasicRunError(
                    f"ROI {roi_id!r}: a published event time precedes the recording's start."
                )
            if published_times.max() > detection.global_time_end_sec + 1e-6:
                raise GuidedContinuousRwdPhasicRunError(
                    f"ROI {roi_id!r}: a published event time exceeds the recording's end."
                )


def _write_feature_event_provenance(
    features_dir: str,
    *,
    included_roi_ids: tuple[str, ...],
    config: Config,
    detection: GuidedContinuousRwdPhasicDetectionResult,
) -> dict:
    """Reuse the existing, unmodified provenance payload builder, then add
    one small additive, backward-compatible section distinguishing this
    recording-global continuous execution from ordinary per-chunk detection.

    Continuous Guided has no per-ROI feature-detection override path yet (the
    D3b-A kernel itself accepts a single ``config``, never a per-ROI feature
    config mapping), so every analyzed ROI's effective settings are the
    global Default -- exactly what the unmodified payload builder already
    records when given no per-ROI override.
    """
    payload = build_feature_event_provenance_payload(
        base_config=config, analyzed_rois=list(included_roi_ids)
    )
    payload["continuous_execution"] = {
        "execution_strategy": detection.execution_strategy,
        "recording_identity": detection.recording_identity,
        "target_grid_identity": detection.target_grid_identity,
        "completion_identity": detection.completion_identity,
        "detector_parameter_identity": detection.detector_parameter_identity,
        "sampling_rate_hz": detection.sampling_rate_hz,
    }
    path = os.path.join(features_dir, FEATURE_EVENT_PROVENANCE_FILENAME)
    _write_json(path, payload)
    return payload


def _stamp_feature_event_provenance_contract(phasic_out_dir: str, payload: dict) -> None:
    """Record the explicit contract-version signal in the phasic-out
    ``run_report.json``, matching ``Pipeline._stamp_feature_event_provenance_
    contract`` exactly so ``feature_event_provenance.classify_provenance_
    contract`` recognizes this as a current run."""
    report_path = os.path.join(phasic_out_dir, "run_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["feature_event_provenance"] = {
        "contract_version": FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
        "schema_version": payload.get("schema_version", ""),
        "relative_path": os.path.join(
            PHASIC_FEATURES_RELATIVE_DIR, FEATURE_EVENT_PROVENANCE_FILENAME
        ),
        "global_default_config_digest": payload.get("global_default_config_digest", ""),
        "roi_count": len(payload.get("rois", [])),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _generate_phasic_summary(
    run_dir: str, phasic_out_dir: str, included_roi_ids: tuple[str, ...]
) -> tuple[dict[str, str], dict[str, int]]:
    """Invoke the existing, unmodified continuous-mode phasic generator
    against the genuine phasic-mode cache and features.csv. This performs no
    detection: it only reads ``features.csv`` and joins it against the
    cache's own per-chunk window metadata (see module docstring)."""
    result = generate_continuous_phasic_summary(phasic_out_dir, run_dir)
    if result.get("skipped_outputs"):
        raise GuidedContinuousRwdPhasicRunError(
            "The phasic window-summary generator skipped its output: "
            f"{result['skipped_outputs']!r}"
        )
    processed = set(result.get("rois_processed") or ())
    if processed != set(included_roi_ids):
        raise GuidedContinuousRwdPhasicRunError(
            "The phasic window-summary generator did not cover every canonical "
            f"included ROI: processed={sorted(processed)!r}, "
            f"expected={sorted(included_roi_ids)!r}."
        )
    row_counts = result.get("row_counts") or {}
    relative_paths = {
        roi_id: f"{roi_id}/tables/{PHASIC_SUMMARY_FILENAME}" for roi_id in included_roi_ids
    }
    for roi_id, relative_path in relative_paths.items():
        if not os.path.isfile(os.path.join(run_dir, relative_path)):
            raise GuidedContinuousRwdPhasicRunError(
                f"Expected phasic window-summary artifact is missing: {relative_path}"
            )
    row_counts_by_roi = {roi_id: int(row_counts.get(roi_id, 0)) for roi_id in included_roi_ids}
    return relative_paths, row_counts_by_roi


def _validate_summary_conserves_events(
    run_dir: str,
    *,
    relative_paths: dict[str, str],
    detection: GuidedContinuousRwdPhasicDetectionResult,
) -> None:
    """Prove the published per-window summary's event counts sum to exactly
    D3b-A's recording-global event count for every ROI -- the summary must
    partition, never re-detect."""
    for roi_id, relative_path in relative_paths.items():
        df = pd.read_csv(os.path.join(run_dir, relative_path))
        total = int(df["event_count"].sum()) if not df.empty else 0
        expected = detection.per_roi[roi_id].event_count
        if total != expected:
            raise GuidedContinuousRwdPhasicRunError(
                f"ROI {roi_id!r}: the published continuous phasic window summary's "
                f"event counts sum to {total}, which does not conserve D3b-A's "
                f"recording-global event count ({expected})."
            )


def execute_guided_continuous_rwd_phasic_run(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    dynamic_f0_authority: GuidedContinuousRwdDynamicF0Authority,
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
    output_base: str,
    config: Config,
    cancellation_requested: Callable[[], bool] | None = None,
) -> GuidedContinuousRwdPhasicRunResult:
    """Produce one coherent continuous-RWD run whose phasic (event) analysis
    has completed and been published through the existing phasic artifact
    contract.

    Accepts exactly the same accepted continuous authorities as
    :func:`photometry_pipeline.guided_continuous_rwd_correction_run.
    execute_guided_continuous_rwd_correction_run` /
    :func:`photometry_pipeline.guided_continuous_rwd_tonic_run.
    execute_guided_continuous_rwd_tonic_run`, plus the same
    ``output_base``/``config``/``cancellation_requested``. Publication order:
    allocate the run directory -> write a running status -> build the C4c
    traversal and persist it through D1 -> cross-check the finalized
    correction cache -> run D3b-A's one-recording detection kernel once per
    ROI -> republish the corrected arrays into a genuine
    ``_analysis/phasic_out/phasic_trace_cache.h5`` while deriving one
    ``features.csv`` row and its published events per D1 storage chunk from
    D3b-A's already-established global result -> write the existing
    ``_analysis/phasic_out/{run_report.json, config_used.yaml}`` pair ->
    stamp the feature-event provenance contract -> generate the existing
    continuous phasic window summary -> validate every published artifact
    conserves D3b-A's recording-global event result -> write the run-level
    ``run_report.json`` -> build and write ``MANIFEST.json`` -> write the
    final success ``status.json`` -> run the existing completed-run validator
    as the last gate. Any failure at any step writes a terminal
    ``error``/``cancelled`` status instead and re-raises -- no run directory
    this function touches can be left claiming success after a failure or
    cancellation.

    Tonic analysis is not run and is explicitly recorded as such in the run
    mode and run report.
    """
    included_roi_ids = tuple(review_binding.recording.roi.included_roi_ids)
    run_mode = _build_run_mode(included_roi_ids)
    run_id, run_dir = _allocate_run_directory(output_base)
    _write_running_status(run_dir, run_id=run_id, run_mode=run_mode)

    cache_path = os.path.join(run_dir, CORRECTED_CACHE_RELATIVE_PATH)
    phasic_out_dir = os.path.join(run_dir, PHASIC_ANALYSIS_RELATIVE_DIR)
    phasic_cache_path = os.path.join(phasic_out_dir, PHASIC_CACHE_FILENAME)
    features_dir = os.path.join(phasic_out_dir, PHASIC_FEATURES_RELATIVE_DIR)
    traversal: GuidedContinuousRwdCorrectionPassTraversal | None = None
    try:
        traversal = iterate_guided_continuous_rwd_corrected_segments(
            review_binding,
            target_grid,
            block_plan,
            segment_plan,
            dynamic_f0_authority,
            accepted_draft=accepted_draft,
            startup_mapping_contract=startup_mapping_contract,
            cancellation_requested=cancellation_requested,
        )
        completion = persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=review_binding,
            target_grid=target_grid,
            output_path=cache_path,
            config=config,
        )
        _validate_persisted_cache(
            cache_path,
            review_binding=review_binding,
            target_grid=target_grid,
            completion=completion,
        )

        detection = detect_guided_continuous_rwd_phasic_features(
            cache_path,
            review_binding=review_binding,
            target_grid=target_grid,
            config=config,
            cancellation_requested=cancellation_requested,
        )

        os.makedirs(phasic_out_dir, exist_ok=True)
        feature_rows, event_rows = _publish_phasic_cache_and_features(
            corrected_cache_path=cache_path,
            phasic_cache_path=phasic_cache_path,
            included_roi_ids=included_roi_ids,
            detection=detection,
            config=config,
        )
        _validate_phasic_cache(
            phasic_cache_path, included_roi_ids=included_roi_ids, completion=completion
        )

        features_path = _write_features_csv(features_dir, feature_rows)
        events_path = _write_events_csv(features_dir, event_rows)
        _validate_events_csv(events_path, detection=detection)

        generate_run_report(config, phasic_out_dir, traces_only=False)
        provenance_payload = _write_feature_event_provenance(
            features_dir,
            included_roi_ids=included_roi_ids,
            config=config,
            detection=detection,
        )
        _stamp_feature_event_provenance_contract(phasic_out_dir, provenance_payload)

        phasic_paths, phasic_row_counts = _generate_phasic_summary(
            run_dir, phasic_out_dir, included_roi_ids
        )
        _validate_summary_conserves_events(
            run_dir, relative_paths=phasic_paths, detection=detection
        )
        correction_provenance = _per_roi_provenance(cache_path, included_roi_ids, first_chunk_id=0)

        total_events = sum(detection.per_roi[roi_id].event_count for roi_id in included_roi_ids)
        report = {
            "schema_name": "guided_continuous_rwd_phasic_run_report",
            "schema_version": "v1",
            "run_context": {"run_id": run_id, "stage": "continuous_correction_and_phasic"},
            "summary": {
                "narrative": (
                    "Continuous recording correction completed. Phasic (event) "
                    "analysis completed for this recording: event detection was "
                    "applied once over the full continuous recording for each of "
                    f"the {len(included_roi_ids)} region(s) of interest, producing "
                    f"a per-window event summary and {total_events} total detected "
                    "events. Tonic (slow, sustained-signal) analysis has not been "
                    "run for this recording."
                ),
            },
            "source": {
                "acquisition_mode": "continuous",
                "canonical_source_path": review_binding.recording.source.fluorescence_path_canonical,
                "source_content_identity": review_binding.recording.source.source_content_identity,
                "recording_identity": review_binding.recording.recording_identity,
            },
            "target_grid": {
                "target_grid_identity": target_grid.target_grid_identity,
                "target_sample_count": target_grid.target_sample_count,
            },
            "included_roi_ids": list(included_roi_ids),
            "per_roi_correction": correction_provenance,
            "corrected_cache": {
                "relative_path": CORRECTED_CACHE_RELATIVE_PATH,
                "corrected_segment_count": completion.corrected_segment_count,
            },
            "phasic_analysis": {
                "trace_cache_relative_path": f"{PHASIC_ANALYSIS_RELATIVE_DIR}/{PHASIC_CACHE_FILENAME}".replace(
                    "\\", "/"
                ),
                "features_relative_path": f"{PHASIC_ANALYSIS_RELATIVE_DIR}/{PHASIC_FEATURES_RELATIVE_DIR}/{PHASIC_FEATURES_FILENAME}".replace(
                    "\\", "/"
                ),
                "events_relative_path": f"{PHASIC_ANALYSIS_RELATIVE_DIR}/{PHASIC_FEATURES_RELATIVE_DIR}/{CONTINUOUS_PHASIC_EVENTS_FILENAME}".replace(
                    "\\", "/"
                ),
                "output_relative_paths": phasic_paths,
                "window_row_counts": phasic_row_counts,
                "per_roi_event_counts": {
                    roi_id: detection.per_roi[roi_id].event_count for roi_id in included_roi_ids
                },
                "execution_strategy": detection.execution_strategy,
            },
            "continuous_correction_pass_completion_identity": completion.completion_identity,
        }
        report[REPORT_COMPLETION_KEY] = build_report_completion_block(run_id=run_id)
        _write_json(os.path.join(run_dir, RUN_REPORT_FILENAME), report)

        continuous_index = build_continuous_window_index(
            run_dir,
            run_mode=run_mode,
            row_counts_by_family={
                FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY: dict(phasic_row_counts),
            },
        )
        finalized_utc = datetime.now(timezone.utc).isoformat()
        events_relative_path = os.path.join(
            PHASIC_ANALYSIS_RELATIVE_DIR, PHASIC_FEATURES_RELATIVE_DIR, CONTINUOUS_PHASIC_EVENTS_FILENAME
        )
        manifest = {
            "tool": _TOOL_NAME,
            "run_id": run_id,
            "run_profile": run_mode["run_profile"],
            "run_type": run_mode["run_type"],
            COMPLETION_KEY: build_manifest_completion_block(
                run_dir,
                run_id=run_id,
                run_mode=run_mode,
                finalized_utc=finalized_utc,
                optional_artifacts=[CORRECTED_CACHE_RELATIVE_PATH, events_relative_path],
                continuous_index=continuous_index,
            ),
        }
        manifest_path = os.path.join(run_dir, MANIFEST_FILENAME)
        _write_json(manifest_path, manifest)

        status = {
            "schema_version": 1,
            "run_id": run_id,
            "run_profile": run_mode["run_profile"],
            "run_type": run_mode["run_type"],
            "acquisition_mode": run_mode["acquisition_mode"],
            "traces_only": run_mode["traces_only"],
            "phase": "final",
            "status": "success",
            "errors": [],
            "warnings": [],
            COMPLETION_KEY: build_status_completion_block(
                run_id=run_id, manifest_sha256=sha256_file(manifest_path)
            ),
        }
        _write_json(os.path.join(run_dir, STATUS_FILENAME), status)

        classification = classify_run_terminal_state(run_dir)
        if not classification.is_success:
            raise GuidedContinuousRwdPhasicRunError(
                "The existing completed-run validator refused this run: "
                f"{classification.reason}"
            )
    except Exception as exc:
        cancelled = _is_lower_layer_cancellation(exc) or _is_cancelled_traversal(traversal)
        _write_terminal_failure_status(
            run_dir, run_id=run_id, run_mode=run_mode, cancelled=cancelled, message=str(exc)
        )
        raise

    return GuidedContinuousRwdPhasicRunResult(
        run_dir=run_dir,
        run_id=run_id,
        corrected_cache_path=cache_path,
        phasic_cache_path=phasic_cache_path,
        features_path=features_path,
        events_path=events_path,
        completion=completion,
        detection=detection,
        terminal_state=classification.state,
        phasic_summary_paths=phasic_paths,
        phasic_summary_row_counts=phasic_row_counts,
    )
