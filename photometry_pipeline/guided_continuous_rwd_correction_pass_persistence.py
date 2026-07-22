"""CR1-D1: persist one accepted C4c continuous-RWD correction pass into the
existing corrected-data HDF5 trace cache.

This module extends the existing, already-accepted corrected-data output
path (``photometry_pipeline.io.hdf5_cache.Hdf5TraceCacheWriter``) rather than
introducing a new storage format. It adapts each immutable
``GuidedContinuousRwdCorrectedSegment`` yielded by a C4c
``GuidedContinuousRwdCorrectionPassTraversal`` into the same ``Chunk``-shaped
input the writer already accepts from the classic per-file correction path,
and writes it through the exact same ``add_chunk()``/``finalize()``/
``abort()`` contract used there, one segment at a time -- never loading the
full continuous recording into memory.

Source-file and sampling-rate provenance are derived from the accepted C1/B3
authorities (``GuidedContinuousRwdReviewBinding``,
``GuidedContinuousRwdTargetGridDescription``), never taken as unbound caller
values: the recording's own canonical fluorescence path and the target
grid's own exact cadence are the only representations of those two facts
this codebase already accepts as authoritative.

Reader compatibility: no reader changes were needed. ``hdf5_cache_reader``'s
``load_cache_chunk_attrs`` already documents itself as "intentionally
tolerant of missing optional attrs so it can be used on both intermittent
and continuous caches" and already normalizes ``window_start_sec``/
``window_end_sec``/``window_duration_sec``. This module writes exactly the
``window_index``/``window_start_sec``/``window_end_sec``/``window_duration_sec``
attrs the existing continuous-window convention (and
``continuous_outputs._CONTINUOUS_REQUIRED_ATTRS``) already expects.

Scope: storage-component only. This module does not connect to the Guided
worker or GUI, does not perform tonic/phasic analysis, and does not enable
Guided continuous Run -- continuous Guided remains hidden and non-runnable.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.guided_continuous_rwd_correction_pass import (
    GuidedContinuousRwdCorrectionPassCompletion,
    GuidedContinuousRwdCorrectionPassTraversal,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_segment_correction import (
    GuidedContinuousRwdCorrectedSegment,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter


class GuidedContinuousRwdPersistenceError(RuntimeError):
    """A narrow refusal while persisting one accepted C4c correction pass."""


def _per_roi_consumed_metadata(
    segment: GuidedContinuousRwdCorrectedSegment,
) -> tuple[dict[str, dict], dict[str, np.ndarray], dict[str, dict]]:
    """Build the existing writer's per-ROI provenance contract from one segment.

    Returns ``(consumed_by_roi, signal_only_baseline_by_roi,
    signal_only_qc_by_roi)`` -- the exact shapes
    ``Hdf5TraceCacheWriter._validate_native_signal_only_evidence`` and
    ``add_chunk``'s correction-attribute loop already require/accept. The
    Signal-Only baseline is not recomputed: it is exactly the same
    ``correction_reference_values`` column C4b already produced and bound
    into ``result_identity``.
    """
    consumed_by_roi: dict[str, dict] = {}
    baseline_by_roi: dict[str, np.ndarray] = {}
    qc_by_roi: dict[str, dict] = {}
    for roi_index, result in enumerate(segment.per_roi_results):
        consumed_by_roi[result.roi_id] = {
            "roi_id": result.roi_id,
            "strategy_family": result.strategy_family,
            "selected_strategy": result.selected_strategy,
            "dynamic_fit_mode": result.dynamic_fit_mode,
            "parameter_identity": result.parameter_identity,
            "evidence_identity": result.evidence_identity,
            "execution_status": "consumed",
            "applied_strategy": result.applied_strategy,
            "fallback_path": ";".join(result.fallback_path),
            "qc_json": result.qc_json,
        }
        if result.strategy_family == "signal_only_f0":
            baseline_by_roi[result.roi_id] = np.asarray(
                segment.correction_reference_values[:, roi_index], dtype=np.float64
            )
            qc_payload = json.loads(result.qc_json)
            qc_by_roi[result.roi_id] = dict(qc_payload.get("production", {}))
    return consumed_by_roi, baseline_by_roi, qc_by_roi


def _segment_to_chunk(
    segment: GuidedContinuousRwdCorrectedSegment,
    *,
    source_file: str,
    sampling_rate_hz: float,
) -> Chunk:
    """Adapt one immutable corrected segment into the existing Chunk shape.

    ``time_sec`` is made segment-local (zero-based), matching ``Chunk``'s
    "uniform grid starting at 0" contract; the segment's true position in
    the recording is preserved separately via ``window_start_sec``/
    ``window_end_sec`` metadata, exactly like the existing continuous-window
    intermittent path already does for one file split into windows.
    """
    target_time = np.asarray(segment.target_elapsed_seconds, dtype=np.float64)
    local_time = target_time - target_time[0]
    consumed_by_roi, baseline_by_roi, qc_by_roi = _per_roi_consumed_metadata(segment)
    metadata: dict[str, Any] = {
        "acquisition_mode": "continuous",
        "window_index": segment.segment_index,
        "window_start_sec": float(target_time[0]),
        "window_end_sec": float(target_time[-1]),
        "window_duration_sec": float(target_time[-1] - target_time[0]),
        "correction_strategy_consumed_by_roi": consumed_by_roi,
    }
    if baseline_by_roi:
        metadata["signal_only_f0_production_baseline"] = baseline_by_roi
        metadata["signal_only_f0_production_qc"] = qc_by_roi
    return Chunk(
        chunk_id=segment.segment_index,
        source_file=source_file,
        format="rwd",
        time_sec=local_time,
        uv_raw=np.asarray(segment.raw_control_values, dtype=np.float64),
        sig_raw=np.asarray(segment.raw_signal_values, dtype=np.float64),
        uv_fit=np.asarray(segment.correction_reference_values, dtype=np.float64),
        delta_f=np.asarray(segment.delta_f_values, dtype=np.float64),
        dff=np.asarray(segment.dff_values, dtype=np.float64),
        fs_hz=float(sampling_rate_hz),
        channel_names=list(segment.included_roi_ids),
        metadata=metadata,
    )


def _validate_authorities_before_traversal(
    review_binding: object,
    target_grid: object,
) -> tuple[str, float]:
    """Fail closed before any ``.tmp`` file is created.

    Returns ``(source_file, sampling_rate_hz)`` derived from the accepted
    authorities -- never from a loose caller-supplied value.
    """
    if not isinstance(review_binding, GuidedContinuousRwdReviewBinding):
        raise GuidedContinuousRwdPersistenceError(
            "review_binding must be a GuidedContinuousRwdReviewBinding."
        )
    if not isinstance(target_grid, GuidedContinuousRwdTargetGridDescription):
        raise GuidedContinuousRwdPersistenceError(
            "target_grid must be a GuidedContinuousRwdTargetGridDescription."
        )
    if target_grid.recording_identity != review_binding.recording.recording_identity:
        raise GuidedContinuousRwdPersistenceError(
            "target_grid does not describe the same recording as review_binding: "
            f"target_grid.recording_identity={target_grid.recording_identity!r}, "
            f"review_binding.recording.recording_identity="
            f"{review_binding.recording.recording_identity!r}."
        )
    source_file = review_binding.recording.source.fluorescence_path_canonical
    sampling_rate_hz = 1.0 / float(target_grid.cadence_fraction)
    return source_file, sampling_rate_hz


def persist_guided_continuous_rwd_correction_pass(
    traversal: GuidedContinuousRwdCorrectionPassTraversal,
    *,
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    output_path: str,
    config: Config,
) -> GuidedContinuousRwdCorrectionPassCompletion:
    """Consume one C4c traversal and write it into the existing HDF5 cache format.

    Consumes ``traversal`` (a ``GuidedContinuousRwdCorrectionPassTraversal``)
    one provisional segment at a time -- never retaining more than one
    segment's arrays in memory -- and writes each through the existing
    ``Hdf5TraceCacheWriter`` (``mode="phasic"``, matching the dff/fit_ref/
    delta_f shape C4c segments already have).

    ``source_file`` and the ``fs_hz`` stamped into every chunk are derived
    from ``review_binding``/``target_grid`` (the same accepted authorities
    the traversal itself was built from), not accepted as free-form caller
    input: ``source_file`` is
    ``review_binding.recording.source.fluorescence_path_canonical``, and the
    sampling rate is ``1.0 / float(target_grid.cadence_fraction)`` -- the
    exact canonical cadence representation, not a value re-derived from
    rounded segment timestamps. ``review_binding`` and ``target_grid`` are
    validated (type, and that they describe the same recording) before the
    writer's ``.tmp`` file is ever created; every yielded segment (including
    the first) and the final completion authority are then checked against
    those same two authorities, not merely against each other.

    The artifact remains at its ``.tmp`` path (per the writer's existing
    atomic-write convention) until every check below passes; only then are
    the completion-binding ``/meta`` attrs written and ``finalize()`` called
    (atomic rename to ``output_path``). Any failure, cancellation, or
    validation mismatch calls ``abort()`` instead, deleting the ``.tmp`` file.
    No newly completed artifact is published after failure; the temporary
    write is aborted. (If a previously completed artifact already existed at
    ``output_path``, that pre-existing file is untouched by an aborted call --
    this function only ever removes its own ``.tmp`` file on failure. A
    successful call replaces ``output_path`` exactly as the existing writer
    already does for the classic per-file path; this is pre-existing
    ``Hdf5TraceCacheWriter`` behavior, not a D1-specific overwrite.)

    Returns the accepted ``GuidedContinuousRwdCorrectionPassCompletion`` on
    success. Raises ``GuidedContinuousRwdPersistenceError`` (or propagates
    the traversal's own error) otherwise.
    """
    if not isinstance(traversal, GuidedContinuousRwdCorrectionPassTraversal):
        raise GuidedContinuousRwdPersistenceError(
            "traversal must be a GuidedContinuousRwdCorrectionPassTraversal."
        )
    source_file, sampling_rate_hz = _validate_authorities_before_traversal(
        review_binding, target_grid
    )
    expected_recording_identity = review_binding.recording.recording_identity
    expected_source_content_identity = review_binding.recording.source.source_content_identity
    expected_target_grid_identity = target_grid.target_grid_identity
    expected_roi_order = review_binding.recording.roi.included_roi_ids

    writer = Hdf5TraceCacheWriter(output_path, "phasic", config)
    expected_next_start = 0
    expected_segment_plan_identity: str | None = None
    written_count = 0
    try:
        for segment in traversal:
            if not isinstance(segment, GuidedContinuousRwdCorrectedSegment):
                raise GuidedContinuousRwdPersistenceError(
                    "Traversal yielded an object that is not a "
                    "GuidedContinuousRwdCorrectedSegment."
                )
            if expected_segment_plan_identity is None:
                expected_segment_plan_identity = segment.correction_segment_plan_identity
            if (
                segment.recording_identity != expected_recording_identity
                or segment.source_content_identity != expected_source_content_identity
                or segment.target_grid_identity != expected_target_grid_identity
                or segment.included_roi_ids != expected_roi_order
                or segment.correction_segment_plan_identity != expected_segment_plan_identity
            ):
                raise GuidedContinuousRwdPersistenceError(
                    "Corrected segment does not match the accepted review_binding/"
                    "target_grid authorities (or an earlier segment's correction-"
                    f"segment-plan identity) (segment_index={segment.segment_index})."
                )
            if segment.start_target_index != expected_next_start:
                raise GuidedContinuousRwdPersistenceError(
                    "Corrected segment is out of order, overlapping, duplicated, "
                    f"or leaves a gap: expected start_target_index="
                    f"{expected_next_start}, got {segment.start_target_index} "
                    f"(segment_index={segment.segment_index})."
                )
            chunk = _segment_to_chunk(
                segment, source_file=source_file, sampling_rate_hz=sampling_rate_hz
            )
            writer.add_chunk(chunk, chunk_id=segment.segment_index, source_file=source_file)
            expected_next_start = segment.stop_target_index
            written_count += 1

        if traversal.state != "completed":
            raise GuidedContinuousRwdPersistenceError(
                "The C4c traversal did not reach a completed state "
                f"(state={traversal.state!r}); refusing to publish a completed "
                "corrected-data artifact."
            )
        completion = traversal.completion
        if (
            completion.corrected_segment_count != written_count
            or completion.target_sample_count != expected_next_start
            or completion.target_sample_count != target_grid.target_sample_count
            or completion.recording_identity != expected_recording_identity
            or completion.source_content_identity != expected_source_content_identity
            or completion.target_grid_identity != expected_target_grid_identity
            or completion.correction_segment_plan_identity != expected_segment_plan_identity
        ):
            raise GuidedContinuousRwdPersistenceError(
                "The C4c completion authority does not match the accepted "
                "review_binding/target_grid authorities or the segments actually "
                "written; refusing to publish a completed corrected-data artifact."
            )

        # Minimum new authority to prove the full continuous traversal
        # completed -- not every C4c identity, only what is not already
        # re-derivable/verifiable from the written chunk data itself.
        writer.meta.attrs["continuous_acquisition_mode"] = "continuous"
        writer.meta.attrs["continuous_completion_identity"] = completion.completion_identity
        writer.meta.attrs["continuous_recording_identity"] = completion.recording_identity
        writer.meta.attrs["continuous_target_grid_identity"] = completion.target_grid_identity
        writer.meta.attrs["continuous_correction_segment_plan_identity"] = (
            completion.correction_segment_plan_identity
        )
        writer.meta.attrs["continuous_target_sample_count"] = int(completion.target_sample_count)
        writer.meta.attrs["continuous_corrected_segment_count"] = int(
            completion.corrected_segment_count
        )
        writer.finalize()
        return completion
    except Exception:
        writer.abort()
        raise
