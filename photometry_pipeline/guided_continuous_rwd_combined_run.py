"""CR1-D4: one combined continuous-RWD run that completes correction, tonic,
and phasic analysis together.

This module composes three already-accepted pieces without repeating or
redesigning any of them:

* :mod:`photometry_pipeline.guided_continuous_rwd_correction_run` (D2) --
  the C4c traversal / D1 persistence / corrected-cache validation call
  sequence, and the run-directory-level lifecycle helpers
  (``_allocate_run_directory``, ``_write_running_status``,
  ``_write_terminal_failure_status``, ``_is_cancelled_traversal``,
  ``_per_roi_provenance``, ``_validate_persisted_cache``, ``_write_json``).
* :mod:`photometry_pipeline.guided_continuous_rwd_tonic_run` (D3a) -- its
  analysis-level publication helpers (``_write_tonic_trace_cache``,
  ``_validate_tonic_cache``, ``_generate_tonic_summary``), which read the
  already-corrected D1 arrays and republish them through the existing
  tonic-mode HDF5 writer; no correction mathematics runs there.
* :mod:`photometry_pipeline.guided_continuous_rwd_phasic_run` (D3b-B) -- its
  analysis-level publication helpers (``_publish_phasic_cache_and_features``,
  ``_validate_phasic_cache``, ``_write_features_csv``, ``_write_events_csv``,
  ``_validate_events_csv``, ``_write_feature_event_provenance``,
  ``_stamp_feature_event_provenance_contract``, ``_generate_phasic_summary``,
  ``_validate_summary_conserves_events``), which call D3b-A's recording-
  global detection kernel once per ROI and derive every published artifact
  from that one result.

None of the helpers listed above owns a run directory, writes a root
``run_report.json``/``MANIFEST.json``/``status.json``, or runs its own
``classify_run_terminal_state`` gate -- that lifecycle ownership belongs
entirely to each module's own public entry point
(``execute_guided_continuous_rwd_correction_run`` /
``execute_guided_continuous_rwd_tonic_run`` /
``execute_guided_continuous_rwd_phasic_run``), none of which this module
calls. Calling any of those three would allocate its own run directory and
run its own correction pass, which is exactly the duplicated-correction,
split-directory outcome this module exists to avoid. Instead this module
allocates exactly one run directory, runs C4c/D1 exactly once, and then
drives the tonic and phasic *analysis-level* helpers directly against that
one shared corrected cache, before writing exactly one root report,
manifest, and status.

Cancellation classification reuses
``guided_continuous_rwd_phasic_run._is_lower_layer_cancellation`` (and its
``(type, category)`` mapping for C4c and D3b-A cancellation) directly rather
than duplicating it -- tonic publication has no long-running interruptible
step of its own to classify.

Scope: this composes exactly the two remaining downstream paths (tonic +
phasic) into one run. It does not connect to the GUI or worker, does not
enable Guided continuous Run, and does not introduce a generalized
analysis-family list, stage registry, or workflow graph -- the two families
are named directly because there are exactly two of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from photometry_pipeline.config import Config
from photometry_pipeline.core.reporting import generate_run_report
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
    GuidedContinuousRwdPhasicDetectionResult,
    detect_guided_continuous_rwd_phasic_features,
)
from photometry_pipeline.guided_continuous_rwd_phasic_run import (
    CONTINUOUS_PHASIC_EVENTS_FILENAME,
    PHASIC_ANALYSIS_RELATIVE_DIR,
    PHASIC_CACHE_FILENAME,
    PHASIC_FEATURES_RELATIVE_DIR,
    PHASIC_FEATURES_FILENAME,
    _generate_phasic_summary,
    _is_lower_layer_cancellation,
    _publish_phasic_cache_and_features,
    _stamp_feature_event_provenance_contract,
    _validate_events_csv,
    _validate_phasic_cache,
    _validate_summary_conserves_events,
    _write_events_csv,
    _write_feature_event_provenance,
    _write_features_csv,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.guided_continuous_rwd_tonic_run import (
    TONIC_ANALYSIS_RELATIVE_DIR,
    TONIC_CACHE_FILENAME,
    _generate_tonic_summary,
    _validate_tonic_cache,
    _write_tonic_trace_cache,
)
from photometry_pipeline.guided_execution_payloads import (
    GuidedExecutionStartupMappingContract,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    open_phasic_cache,
    open_tonic_cache,
)
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY,
    FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY,
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

_RUN_ID_PREFIX = "continuous_rwd_combined_run"
_RUN_PROFILE = "guided_continuous_rwd_combined"
_RUN_TYPE = "tonic_and_phasic"
_TOOL_NAME = "photometry_pipeline.guided_continuous_rwd_combined_run"


class GuidedContinuousRwdCombinedRunError(RuntimeError):
    """A narrow refusal while executing or publishing one combined
    continuous-RWD tonic+phasic run. Errors raised directly by C4c/D1/D3a's
    or D3b-B's reused helpers propagate unchanged; this exception covers
    only combined-run-directory-level and cross-family-coherence concerns."""


@dataclass(frozen=True)
class GuidedContinuousRwdCombinedRunResult:
    """What a caller needs to know about one completed combined
    continuous-RWD tonic+phasic run."""

    run_dir: str
    run_id: str
    corrected_cache_path: str
    tonic_cache_path: str
    phasic_cache_path: str
    features_path: str
    events_path: str
    completion: GuidedContinuousRwdCorrectionPassCompletion
    detection: GuidedContinuousRwdPhasicDetectionResult
    terminal_state: str
    tonic_summary_paths: dict[str, str]
    tonic_summary_row_counts: dict[str, int]
    phasic_summary_paths: dict[str, str]
    phasic_summary_row_counts: dict[str, int]


def _build_run_mode(included_roi_ids: tuple[str, ...]) -> dict:
    return normalize_run_mode(
        run_profile=_RUN_PROFILE,
        run_type=_RUN_TYPE,
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=True,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=list(included_roi_ids),
        continuous_outputs_ran=True,
        chunked_input_processing=False,
        shared_input_manifest=False,
    )


def _validate_cross_family_coherence(
    *,
    tonic_cache_path: str,
    phasic_cache_path: str,
    included_roi_ids: tuple[str, ...],
    completion: GuidedContinuousRwdCorrectionPassCompletion,
) -> None:
    """Directly confirm the tonic and phasic caches describe the same
    corrected recording, rather than constructing a new shared authority
    object for it.

    Both caches were built in this same run from arrays read back from the
    one shared, already-validated D1 corrected cache (validated once against
    ``completion.completion_identity`` by :func:`_validate_persisted_cache`
    before either publication step ran) -- so "same correction completion
    identity" is architecturally guaranteed by construction (one traversal,
    one persisted cache, one ``completion`` object threaded into both
    publication calls). This function proves that guarantee actually landed
    in both artifacts: identical canonical ROI order, an identical
    contiguous chunk-ID range matching the one shared completion, and an
    identical final window boundary (the tonic and phasic writers each copy
    the same source D1 chunk's own ``window_start_sec``/``window_end_sec``/
    ``window_duration_sec`` attrs verbatim, so these must match exactly, not
    merely approximately).
    """
    tonic_cache = open_tonic_cache(tonic_cache_path)
    try:
        tonic_rois = list_cache_rois(tonic_cache)
        tonic_chunk_ids = list_cache_chunk_ids(tonic_cache)
        tonic_last_attrs = load_cache_chunk_attrs(
            tonic_cache, tonic_rois[0], tonic_chunk_ids[-1]
        )
    finally:
        tonic_cache.close()

    phasic_cache = open_phasic_cache(phasic_cache_path)
    try:
        phasic_rois = list_cache_rois(phasic_cache)
        phasic_chunk_ids = list_cache_chunk_ids(phasic_cache)
        phasic_last_attrs = load_cache_chunk_attrs(
            phasic_cache, phasic_rois[0], phasic_chunk_ids[-1]
        )
    finally:
        phasic_cache.close()

    if tonic_rois != list(included_roi_ids) or phasic_rois != list(included_roi_ids):
        raise GuidedContinuousRwdCombinedRunError(
            "The tonic and phasic caches do not both carry the accepted "
            f"canonical ROI order: tonic={tonic_rois!r}, phasic={phasic_rois!r}, "
            f"expected={list(included_roi_ids)!r}."
        )
    expected_chunk_ids = list(range(completion.corrected_segment_count))
    if tonic_chunk_ids != expected_chunk_ids or phasic_chunk_ids != expected_chunk_ids:
        raise GuidedContinuousRwdCombinedRunError(
            "The tonic and phasic caches do not both cover the one shared "
            f"correction pass's chunk range: tonic={tonic_chunk_ids!r}, "
            f"phasic={phasic_chunk_ids!r}, expected={expected_chunk_ids!r}."
        )
    for key in ("window_start_sec", "window_end_sec", "window_duration_sec"):
        if tonic_last_attrs[key] != phasic_last_attrs[key]:
            raise GuidedContinuousRwdCombinedRunError(
                f"The tonic and phasic caches' final window {key} disagree: "
                f"tonic={tonic_last_attrs[key]!r}, phasic={phasic_last_attrs[key]!r}."
            )


def execute_guided_continuous_rwd_combined_run(
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
) -> GuidedContinuousRwdCombinedRunResult:
    """Produce one coherent continuous-RWD run whose tonic AND phasic
    analysis have both completed and been published, sharing one correction
    pass.

    Accepts exactly the same accepted continuous authorities as
    :func:`photometry_pipeline.guided_continuous_rwd_tonic_run.
    execute_guided_continuous_rwd_tonic_run` /
    :func:`photometry_pipeline.guided_continuous_rwd_phasic_run.
    execute_guided_continuous_rwd_phasic_run`, plus the same
    ``output_base``/``config``/``cancellation_requested``. Publication order:
    allocate the run directory -> write a running status -> build the C4c
    traversal and persist it through D1 exactly once -> cross-check the
    finalized correction cache -> publish the tonic trace cache/summary from
    that one shared cache -> run D3b-A's one-recording detection kernel once
    per ROI -> publish the phasic trace cache/features/events/provenance/
    summary from that one shared cache and D3b-A's result -> cross-check
    that both published caches agree on ROI order, chunk coverage, and final
    window boundary -> write one root ``run_report.json`` -> build and write
    one ``MANIFEST.json`` covering both analysis families -> write the final
    success ``status.json`` -> run the existing completed-run validator as
    the last gate. Any failure at any step writes a terminal
    ``error``/``cancelled`` status instead and re-raises -- no run directory
    this function touches can be left claiming success after a failure or
    cancellation.
    """
    included_roi_ids = tuple(review_binding.recording.roi.included_roi_ids)
    run_mode = _build_run_mode(included_roi_ids)
    run_id, run_dir = _allocate_run_directory(output_base)
    _write_running_status(run_dir, run_id=run_id, run_mode=run_mode)

    cache_path = os.path.join(run_dir, CORRECTED_CACHE_RELATIVE_PATH)
    tonic_out_dir = os.path.join(run_dir, TONIC_ANALYSIS_RELATIVE_DIR)
    tonic_cache_path = os.path.join(tonic_out_dir, TONIC_CACHE_FILENAME)
    phasic_out_dir = os.path.join(run_dir, PHASIC_ANALYSIS_RELATIVE_DIR)
    phasic_cache_path = os.path.join(phasic_out_dir, PHASIC_CACHE_FILENAME)
    features_dir = os.path.join(phasic_out_dir, PHASIC_FEATURES_RELATIVE_DIR)
    traversal: GuidedContinuousRwdCorrectionPassTraversal | None = None
    try:
        # --- one correction pass, shared by both analysis families ---
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

        # --- tonic publication, reading the one shared corrected cache ---
        os.makedirs(tonic_out_dir, exist_ok=True)
        _write_tonic_trace_cache(
            corrected_cache_path=cache_path,
            tonic_cache_path=tonic_cache_path,
            included_roi_ids=included_roi_ids,
            config=config,
        )
        _validate_tonic_cache(
            tonic_cache_path, included_roi_ids=included_roi_ids, completion=completion
        )
        generate_run_report(config, tonic_out_dir, traces_only=False)
        tonic_paths, tonic_row_counts = _generate_tonic_summary(
            run_dir, tonic_out_dir, included_roi_ids
        )

        # --- phasic detection + publication, reading the same shared cache ---
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

        # --- cross-family coherence, before either family can claim success ---
        _validate_cross_family_coherence(
            tonic_cache_path=tonic_cache_path,
            phasic_cache_path=phasic_cache_path,
            included_roi_ids=included_roi_ids,
            completion=completion,
        )

        correction_provenance = _per_roi_provenance(cache_path, included_roi_ids, first_chunk_id=0)
        total_events = sum(detection.per_roi[roi_id].event_count for roi_id in included_roi_ids)
        report = {
            "schema_name": "guided_continuous_rwd_combined_run_report",
            "schema_version": "v1",
            "run_context": {"run_id": run_id, "stage": "continuous_correction_tonic_and_phasic"},
            "summary": {
                "narrative": (
                    "Continuous recording correction completed. Tonic (slow, "
                    "sustained-signal) analysis completed for this recording. "
                    "Phasic (event) analysis completed for this recording: event "
                    "detection was applied once over the full continuous recording "
                    f"for each of the {len(included_roi_ids)} region(s) of interest. "
                    "A per-window tonic summary and a per-window phasic event "
                    "summary were produced for every included region of interest, "
                    f"totaling {total_events} detected events across all regions."
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
            "tonic_analysis": {
                "trace_cache_relative_path": f"{TONIC_ANALYSIS_RELATIVE_DIR}/{TONIC_CACHE_FILENAME}".replace(
                    "\\", "/"
                ),
                "output_relative_paths": tonic_paths,
                "window_row_counts": tonic_row_counts,
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
                FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY: dict(tonic_row_counts),
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
            raise GuidedContinuousRwdCombinedRunError(
                "The existing completed-run validator refused this run: "
                f"{classification.reason}"
            )
    except Exception as exc:
        cancelled = _is_lower_layer_cancellation(exc) or _is_cancelled_traversal(traversal)
        _write_terminal_failure_status(
            run_dir, run_id=run_id, run_mode=run_mode, cancelled=cancelled, message=str(exc)
        )
        raise

    return GuidedContinuousRwdCombinedRunResult(
        run_dir=run_dir,
        run_id=run_id,
        corrected_cache_path=cache_path,
        tonic_cache_path=tonic_cache_path,
        phasic_cache_path=phasic_cache_path,
        features_path=features_path,
        events_path=events_path,
        completion=completion,
        detection=detection,
        terminal_state=classification.state,
        tonic_summary_paths=tonic_paths,
        tonic_summary_row_counts=tonic_row_counts,
        phasic_summary_paths=phasic_paths,
        phasic_summary_row_counts=phasic_row_counts,
    )
