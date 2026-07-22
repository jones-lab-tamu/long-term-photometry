"""CR1-D3a: integrate the established tonic computation with the continuous-
RWD correction-run lifecycle.

This module extends the existing D2 run lifecycle
(:mod:`photometry_pipeline.guided_continuous_rwd_correction_run`) with the
existing production tonic contract, not a descriptive-summary substitute for
it.

The established tonic scientific contract (inspected directly in
``photometry_pipeline/core/tonic_dff.py`` and ``photometry_pipeline/
pipeline.py``) has two distinct routes:

* a *legacy* route (``Pipeline.mode == "tonic"`` and ``per_roi_correction is
  None``): a recording-global robust isosbestic fit
  (``core.tonic_dff.compute_global_iso_fit_robust``) applied per session
  (``apply_global_fit`` + ``compute_session_tonic_df_from_global``);
* the *native* per-ROI-correction route, which every Guided continuous-RWD
  correction (C4b) uses: ``pipeline.py``'s own ``_apply_standard_analysis``
  dispatches tonic to the *exact same* call as phasic --
  ``regression.fit_chunk_dynamic(chunk, config, mode="phasic",
  per_roi_correction=dispatch_map)`` (see ``pipeline.py`` around
  ``_apply_standard_analysis``, comment: "Native tonic consumes the same
  canonical per-session correction engine as phasic"). C4b
  (``guided_continuous_rwd_segment_correction.py``) calls this identical
  function with the identical argument shape
  (``regression.fit_chunk_dynamic(chunk, config, mode="phasic",
  per_roi_correction={spec.roi_id: spec})``) to produce every corrected
  segment's ``delta_f``.

Because every Guided continuous-RWD recording is corrected with an explicit
per-ROI strategy (never the uniform/legacy route), the established tonic
computation for this backend *is* C4b's own per-segment ``delta_f`` -- not a
different or additional fit. This module does not re-invoke
``regression.fit_chunk_dynamic`` a second time (that would recompute an
already-established, already-validated result at real cost for no scientific
difference); it reuses the already-computed values D1 already validated and
persisted, and republishes them through the existing, unmodified tonic
trace-cache writer (``Hdf5TraceCacheWriter(path, "tonic", config)``) so the
run produces the actual artifact the application ordinarily treats as proof
tonic analysis ran: ``_analysis/tonic_out/tonic_trace_cache.h5``, alongside
the same ``run_report.json``/``config_used.yaml`` pair every tonic Pipeline
run writes (``core.reporting.generate_run_report``, reused unmodified). The
existing continuous window-summary generator
(``continuous_outputs.generate_continuous_tonic_summary``) is then called,
unmodified, against that genuine tonic-mode cache -- exactly as the classic
continuous pipeline already does.

Fitting is chunk-local by established design in native/per-ROI-correction
mode (each 600-second correction segment is fit independently, exactly as
C4b already does and as phasic already does); there is no continuous-wide
model to carry across storage-chunk boundaries in this route, and no
boundary risk to solve, because the destination signal (a percentile/mean/
median descriptive summary) has no thresholding or refractory state to reset.

Phasic remains unimplemented (see the CR1-D3a inspection report): its
continuous integration would require running peak/event detection for the
first time and inherits an acknowledged, unaddressed chunk-boundary
threshold/refractory-reset risk this module does not attempt to solve.

Scope: this integrates exactly one downstream path (tonic). It does not run
or modify phasic/feature analysis, does not connect to the GUI or worker,
and does not enable Guided continuous Run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.continuous_outputs import generate_continuous_tonic_summary
from photometry_pipeline.core.reporting import generate_run_report
from photometry_pipeline.core.types import Chunk
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
    _write_running_status,
    _write_terminal_failure_status,
)
from photometry_pipeline.guided_continuous_rwd_correction_segments import (
    GuidedContinuousRwdCorrectionSegmentPlan,
    GuidedContinuousRwdDynamicF0Authority,
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
    open_tonic_cache,
)
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
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
from photometry_pipeline.guided_continuous_rwd_correction_run import _write_json

_RUN_ID_PREFIX = "continuous_rwd_tonic_run"
_RUN_PROFILE = "guided_continuous_rwd_tonic"
_RUN_TYPE = "tonic_only"
_TOOL_NAME = "photometry_pipeline.guided_continuous_rwd_tonic_run"
TONIC_ANALYSIS_RELATIVE_DIR = os.path.join("_analysis", "tonic_out")
TONIC_CACHE_FILENAME = "tonic_trace_cache.h5"


class GuidedContinuousRwdTonicRunError(RuntimeError):
    """A narrow refusal while executing or publishing one continuous-RWD
    tonic run. Errors raised directly by C4c/D1 propagate unchanged; this
    exception covers only run-directory-level and tonic-cache-production
    concerns."""


@dataclass(frozen=True)
class GuidedContinuousRwdTonicRunResult:
    """What a caller needs to know about one completed continuous-RWD tonic run."""

    run_dir: str
    run_id: str
    corrected_cache_path: str
    tonic_cache_path: str
    completion: GuidedContinuousRwdCorrectionPassCompletion
    terminal_state: str
    tonic_summary_paths: dict[str, str]
    tonic_summary_row_counts: dict[str, int]


def _build_run_mode(included_roi_ids: tuple[str, ...]) -> dict:
    return normalize_run_mode(
        run_profile=_RUN_PROFILE,
        run_type=_RUN_TYPE,
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=False,
        tonic_analysis=True,
        feature_extraction_ran=False,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=list(included_roi_ids),
        continuous_outputs_ran=True,
        chunked_input_processing=False,
        shared_input_manifest=False,
    )


def _write_tonic_trace_cache(
    *,
    corrected_cache_path: str,
    tonic_cache_path: str,
    included_roi_ids: tuple[str, ...],
    config: Config,
) -> None:
    """Republish C4b's already-established per-segment ``delta_f`` (the
    native tonic result -- see module docstring) through the existing,
    unmodified tonic-mode ``Hdf5TraceCacheWriter``, one storage chunk (all
    canonical ROIs) at a time.

    No correction mathematics runs here: every array is read back verbatim
    from the already-validated D1 corrected cache and copied into the tonic
    cache's ``deltaF`` field via the writer's own existing dispatch
    (``mode="tonic"``). Bounded to one chunk's arrays across all ROIs at a
    time, never the full recording.
    """
    source_cache = open_phasic_cache(corrected_cache_path)
    try:
        chunk_ids = list_cache_chunk_ids(source_cache)
        writer = Hdf5TraceCacheWriter(tonic_cache_path, "tonic", config)
        try:
            for chunk_id in chunk_ids:
                sig_cols = []
                uv_cols = []
                delta_cols = []
                time_sec = None
                fs_hz = None
                source_file = ""
                window_meta: dict[str, float] = {}
                for roi_id in included_roi_ids:
                    attrs = load_cache_chunk_attrs(source_cache, roi_id, int(chunk_id))
                    t, sig, uv, delta_f = load_cache_chunk_fields(
                        source_cache, roi_id, int(chunk_id),
                        ["time_sec", "sig_raw", "uv_raw", "delta_f"],
                    )
                    if time_sec is None:
                        time_sec = np.asarray(t, dtype=np.float64)
                        fs_hz = float(attrs["fs_hz"])
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
                    delta_cols.append(np.asarray(delta_f, dtype=np.float64))

                chunk = Chunk(
                    chunk_id=int(chunk_id),
                    source_file=source_file,
                    format="rwd",
                    time_sec=time_sec,
                    uv_raw=np.column_stack(uv_cols),
                    sig_raw=np.column_stack(sig_cols),
                    delta_f=np.column_stack(delta_cols),
                    fs_hz=float(fs_hz),
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


def _validate_tonic_cache(
    tonic_cache_path: str,
    *,
    included_roi_ids: tuple[str, ...],
    completion: GuidedContinuousRwdCorrectionPassCompletion,
) -> None:
    """Reopen the just-written tonic cache through the existing reader and
    confirm it faithfully represents the one continuous recording: canonical
    ROI order, every corrected storage chunk present, no duplication."""
    if not os.path.isfile(tonic_cache_path):
        raise GuidedContinuousRwdTonicRunError(
            "The tonic trace cache is missing after tonic-cache production claimed success."
        )
    if os.path.isfile(tonic_cache_path + ".tmp"):
        raise GuidedContinuousRwdTonicRunError(
            "A .tmp tonic-cache artifact remains after finalize."
        )
    cache = open_tonic_cache(tonic_cache_path)
    try:
        rois = list_cache_rois(cache)
        if rois != list(included_roi_ids):
            raise GuidedContinuousRwdTonicRunError(
                "The tonic cache's ROI set/order does not match the accepted "
                f"review binding: cache={rois!r}, expected={list(included_roi_ids)!r}."
            )
        chunk_ids = list_cache_chunk_ids(cache)
        if len(chunk_ids) != completion.corrected_segment_count:
            raise GuidedContinuousRwdTonicRunError(
                "The tonic cache's chunk count does not match the C4c "
                f"completion: cache={len(chunk_ids)}, "
                f"expected={completion.corrected_segment_count}."
            )
        if sorted(chunk_ids) != list(range(completion.corrected_segment_count)):
            raise GuidedContinuousRwdTonicRunError(
                "The tonic cache's chunk identities are not a contiguous "
                "0-based range."
            )
    finally:
        cache.close()


def _generate_tonic_summary(
    run_dir: str, tonic_out_dir: str, included_roi_ids: tuple[str, ...]
) -> tuple[dict[str, str], dict[str, int]]:
    """Invoke the existing, unmodified continuous-mode tonic generator
    against the genuine tonic-mode cache.

    Returns ``(relative_paths_by_roi, row_counts_by_roi)``. Raises
    ``GuidedContinuousRwdTonicRunError`` if the generator skipped its output
    or did not cover every canonical included ROI.
    """
    result = generate_continuous_tonic_summary(tonic_out_dir, run_dir)
    if result.get("skipped_outputs"):
        raise GuidedContinuousRwdTonicRunError(
            "The tonic window-summary generator skipped its output: "
            f"{result['skipped_outputs']!r}"
        )
    processed = set(result.get("rois_processed") or ())
    if processed != set(included_roi_ids):
        raise GuidedContinuousRwdTonicRunError(
            "The tonic window-summary generator did not cover every canonical "
            f"included ROI: processed={sorted(processed)!r}, "
            f"expected={sorted(included_roi_ids)!r}."
        )
    row_counts = result.get("row_counts") or {}
    relative_paths = {
        roi_id: f"{roi_id}/tables/continuous_tonic_window_summary.csv"
        for roi_id in included_roi_ids
    }
    for roi_id, relative_path in relative_paths.items():
        if not os.path.isfile(os.path.join(run_dir, relative_path)):
            raise GuidedContinuousRwdTonicRunError(
                f"Expected tonic window-summary artifact is missing: {relative_path}"
            )
    row_counts_by_roi = {roi_id: int(row_counts.get(roi_id, 0)) for roi_id in included_roi_ids}
    return relative_paths, row_counts_by_roi


def execute_guided_continuous_rwd_tonic_run(
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
) -> GuidedContinuousRwdTonicRunResult:
    """Produce one coherent continuous-RWD run whose established tonic
    computation has completed and been published through the existing tonic
    trace-cache writer/reader.

    Accepts exactly the same accepted continuous authorities as
    :func:`photometry_pipeline.guided_continuous_rwd_correction_run.
    execute_guided_continuous_rwd_correction_run`, plus the same
    ``output_base``/``config``/``cancellation_requested``. Publication order:
    allocate the run directory -> write a running status -> build the C4c
    traversal and persist it through D1 -> cross-check the finalized
    correction cache against the accepted authorities and the C4c completion
    -> republish the established per-segment tonic result into a genuine
    ``_analysis/tonic_out/tonic_trace_cache.h5`` via the existing tonic-mode
    writer -> write the existing ``_analysis/tonic_out/{run_report.json,
    config_used.yaml}`` pair via the existing production report writer ->
    generate the existing continuous tonic window summary from that cache ->
    write the run-level ``run_report.json`` -> build and write
    ``MANIFEST.json`` (with the tonic continuous-window index) -> write the
    final success ``status.json`` -> run the existing completed-run
    validator as the last gate. Any failure at any step writes a terminal
    ``error``/``cancelled`` status instead and re-raises -- no run directory
    this function touches can be left claiming success after a failure or
    cancellation.

    Phasic/feature analysis is not run and is explicitly recorded as such in
    the run mode and run report.
    """
    included_roi_ids = tuple(review_binding.recording.roi.included_roi_ids)
    run_mode = _build_run_mode(included_roi_ids)
    run_id, run_dir = _allocate_run_directory(output_base)
    _write_running_status(run_dir, run_id=run_id, run_mode=run_mode)

    cache_path = os.path.join(run_dir, CORRECTED_CACHE_RELATIVE_PATH)
    tonic_out_dir = os.path.join(run_dir, TONIC_ANALYSIS_RELATIVE_DIR)
    tonic_cache_path = os.path.join(tonic_out_dir, TONIC_CACHE_FILENAME)
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
        provenance = _per_roi_provenance(cache_path, included_roi_ids, first_chunk_id=0)

        report = {
            "schema_name": "guided_continuous_rwd_tonic_run_report",
            "schema_version": "v1",
            "run_context": {"run_id": run_id, "stage": "continuous_correction_and_tonic"},
            "summary": {
                "narrative": (
                    "Continuous recording correction completed. Tonic (slow, "
                    "sustained-signal) analysis completed for this recording, "
                    f"producing a per-window tonic summary for each of the "
                    f"{len(included_roi_ids)} region(s) of interest. Phasic "
                    "(event) analysis has not been run for this recording."
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
            "per_roi_correction": provenance,
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
            "continuous_correction_pass_completion_identity": completion.completion_identity,
        }
        report[REPORT_COMPLETION_KEY] = build_report_completion_block(run_id=run_id)
        _write_json(os.path.join(run_dir, RUN_REPORT_FILENAME), report)

        continuous_index = build_continuous_window_index(
            run_dir,
            run_mode=run_mode,
            row_counts_by_family={
                FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY: dict(tonic_row_counts),
            },
        )
        finalized_utc = datetime.now(timezone.utc).isoformat()
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
                optional_artifacts=[CORRECTED_CACHE_RELATIVE_PATH],
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
            raise GuidedContinuousRwdTonicRunError(
                "The existing completed-run validator refused this run: "
                f"{classification.reason}"
            )
    except Exception as exc:
        cancelled = (
            isinstance(exc, GuidedContinuousRwdCorrectionPassError)
            and exc.category == "segment_correction_pass_interrupted"
        ) or _is_cancelled_traversal(traversal)
        _write_terminal_failure_status(
            run_dir, run_id=run_id, run_mode=run_mode, cancelled=cancelled, message=str(exc)
        )
        raise

    return GuidedContinuousRwdTonicRunResult(
        run_dir=run_dir,
        run_id=run_id,
        corrected_cache_path=cache_path,
        tonic_cache_path=tonic_cache_path,
        completion=completion,
        terminal_state=classification.state,
        tonic_summary_paths=tonic_paths,
        tonic_summary_row_counts=tonic_row_counts,
    )
