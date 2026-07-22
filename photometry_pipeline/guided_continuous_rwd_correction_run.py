"""CR1-D2: place one accepted C4c+D1 continuous-RWD correction pass into the
existing run-directory / completed-run lifecycle.

This module is pure orchestration over already-accepted authorities and
already-accepted entry points:

* :func:`photometry_pipeline.guided_continuous_rwd_correction_pass.
  iterate_guided_continuous_rwd_corrected_segments` (C4c) builds the
  second-pass corrected-segment traversal;
* :func:`photometry_pipeline.guided_continuous_rwd_correction_pass_persistence.
  persist_guided_continuous_rwd_correction_pass` (D1) consumes it into the
  existing HDF5 trace-cache format;
* :mod:`photometry_pipeline.run_completion_contract` -- the same completion
  contract library the production wrapper (``tools/run_full_pipeline_
  deliverables.py``) and every completed-run reader already use -- decides
  what a *successful* run directory looks like.

No new run-lifecycle architecture is introduced. The existing Guided startup
allocator/materializer/orchestrator (``guided_startup_allocation.py`` /
``guided_startup_materialization.py`` / ``guided_startup_orchestration.py``)
and the production wrapper subprocess are not reused here: both are hard-gated
to ``acquisition_mode="intermittent"`` (``guided_capabilities.
GUIDED_PRODUCTION_ACQUISITION_MODES`` and ``tools.run_full_pipeline_
deliverables.validate_guided_preallocated_mode_args``), and the Guided
allocator's own placeholder file (``guided_startup_status.json``) is one of
``run_completion_contract.GUIDED_DEFINITIVE_MARKER_FILENAMES`` -- writing it
would make this run classify as a Guided run and then require the full set of
Guided-only provenance artifacts this backend-only entry point does not have.
Continuous Guided must remain hidden and non-runnable, so this run is
deliberately built as an ordinary (non-Guided) current run: a run directory
containing only ``status.json``/``MANIFEST.json``/``run_report.json`` (plus
the corrected trace cache) classifies as ``not_guided`` under
``classify_guided_current_native_state`` and is judged solely by the
existing, mode-agnostic completion contract.

The run mode this entry point declares is deliberately minimal and truthful:
``deliverable_profile=PROFILE_CONTINUOUS`` (the existing profile for one
continuous source recording), ``phasic_analysis=False``,
``tonic_analysis=False``, ``continuous_outputs_ran=False``. Under the existing
contract this degenerates ``required_artifacts_for_run_mode`` to exactly
``run_report.json`` -- no per-ROI window tables, no phasic/tonic trace cache,
no chunked-input-processing ledger are required or claimed, because none of
that downstream work has run. The corrected trace cache is recorded as an
*optional* manifest artifact and is independently cross-checked against the
accepted authorities and the C4c completion (see
:func:`_validate_persisted_cache`) before any completed status is published.

Scope: run-directory/lifecycle placement only. This module does not perform
tonic/phasic/feature analysis, does not connect to the GUI or worker, and does
not enable Guided continuous Run.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from photometry_pipeline.config import Config
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
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    open_phasic_cache,
)
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    MANIFEST_FILENAME,
    PROFILE_CONTINUOUS,
    REPORT_COMPLETION_KEY,
    RUN_REPORT_FILENAME,
    STATUS_FILENAME,
    build_manifest_completion_block,
    build_report_completion_block,
    build_status_completion_block,
    classify_run_terminal_state,
    normalize_run_mode,
    sha256_file,
)

# Deliberately not "guided_run_..." -- this run carries none of the Guided
# markers (see module docstring), so naming it as one would misrepresent what
# actually produced the directory to anyone reading the folder tree.
_RUN_ID_PREFIX = "continuous_rwd_correction_run"
_RUN_PROFILE = "guided_continuous_rwd_correction"
_RUN_TYPE = "correction_only"
_TOOL_NAME = "photometry_pipeline.guided_continuous_rwd_correction_run"

CORRECTED_CACHE_RELATIVE_PATH = "continuous_corrected_trace_cache.h5"


class GuidedContinuousRwdCorrectionRunError(RuntimeError):
    """A narrow refusal while executing or publishing one continuous-RWD
    correction run. Errors raised directly by C4c/D1 propagate unchanged;
    this exception covers only run-directory-level concerns (allocation,
    cache/authority coherence, and post-hoc validator refusal)."""


@dataclass(frozen=True)
class GuidedContinuousRwdCorrectionRunResult:
    """What a caller needs to know about one completed continuous-RWD run."""

    run_dir: str
    run_id: str
    corrected_cache_path: str
    completion: GuidedContinuousRwdCorrectionPassCompletion
    terminal_state: str


def _new_run_id() -> str:
    now = datetime.now(timezone.utc)
    return f"{_RUN_ID_PREFIX}_{now.strftime('%Y%m%dT%H%M%S%f')}Z_{secrets.token_hex(6)}"


def _allocate_run_directory(output_base: str) -> tuple[str, str]:
    """Exclusively create one new run directory directly under ``output_base``."""
    if not isinstance(output_base, str) or not output_base or not os.path.isabs(output_base):
        raise GuidedContinuousRwdCorrectionRunError(
            "output_base must be an absolute path."
        )
    os.makedirs(output_base, exist_ok=True)
    last_error: OSError | None = None
    for _attempt in range(5):
        run_id = _new_run_id()
        run_dir = os.path.join(output_base, run_id)
        try:
            os.mkdir(run_dir)
        except FileExistsError as exc:
            last_error = exc
            continue
        return run_id, run_dir
    raise GuidedContinuousRwdCorrectionRunError(
        "Could not allocate a new run directory under output_base after "
        f"repeated name collisions: {last_error}"
    )


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _build_run_mode(included_roi_ids: tuple[str, ...]) -> dict:
    return normalize_run_mode(
        run_profile=_RUN_PROFILE,
        run_type=_RUN_TYPE,
        acquisition_mode="continuous",
        traces_only=True,
        phasic_analysis=False,
        tonic_analysis=False,
        feature_extraction_ran=False,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=list(included_roi_ids),
        continuous_outputs_ran=False,
        chunked_input_processing=False,
        shared_input_manifest=False,
    )


def _write_running_status(run_dir: str, *, run_id: str, run_mode: dict) -> None:
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "run_profile": run_mode["run_profile"],
        "run_type": run_mode["run_type"],
        "acquisition_mode": run_mode["acquisition_mode"],
        "traces_only": run_mode["traces_only"],
        "phase": "running",
        "status": "running",
        "errors": [],
        "warnings": [],
    }
    _write_json(os.path.join(run_dir, STATUS_FILENAME), status)


def _write_terminal_failure_status(
    run_dir: str, *, run_id: str, run_mode: dict, cancelled: bool, message: str
) -> None:
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "run_profile": run_mode["run_profile"],
        "run_type": run_mode["run_type"],
        "acquisition_mode": run_mode["acquisition_mode"],
        "traces_only": run_mode["traces_only"],
        "phase": "final",
        "status": "cancelled" if cancelled else "error",
        "errors": [] if cancelled else [message],
        "warnings": [],
    }
    _write_json(os.path.join(run_dir, STATUS_FILENAME), status)


def _is_cancelled_traversal(
    traversal: GuidedContinuousRwdCorrectionPassTraversal | None,
) -> bool:
    return traversal is not None and getattr(traversal, "state", None) == "cancelled"


def _per_roi_provenance(
    cache_path: str, included_roi_ids: tuple[str, ...], first_chunk_id: int
) -> dict[str, dict]:
    """Read back accepted per-ROI correction provenance through the existing reader.

    Reused for the run report only (human/machine provenance); it is not part
    of the completion contract's own required-artifact set.
    """
    cache = open_phasic_cache(cache_path)
    try:
        provenance: dict[str, dict] = {}
        for roi_id in included_roi_ids:
            attrs = load_cache_chunk_attrs(cache, roi_id, first_chunk_id)
            provenance[roi_id] = {
                "strategy_family": attrs.get("correction_strategy_family"),
                "selected_strategy": attrs.get("correction_selected_strategy"),
                "parameter_identity": attrs.get("correction_parameter_identity"),
                "evidence_identity": attrs.get("correction_evidence_identity"),
            }
        return provenance
    finally:
        cache.close()


def _validate_persisted_cache(
    cache_path: str,
    *,
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    completion: GuidedContinuousRwdCorrectionPassCompletion,
) -> None:
    """Cross-check the finalized cache against the accepted authorities and
    the C4c completion before any run-level record claims success."""
    if not os.path.isfile(cache_path):
        raise GuidedContinuousRwdCorrectionRunError(
            "The finalized corrected cache is missing after persistence claimed success."
        )
    if os.path.isfile(cache_path + ".tmp"):
        raise GuidedContinuousRwdCorrectionRunError(
            "A .tmp corrected-cache artifact remains after finalize."
        )
    expected_roi_order = list(review_binding.recording.roi.included_roi_ids)
    cache = open_phasic_cache(cache_path)
    try:
        rois = list_cache_rois(cache)
        if rois != expected_roi_order:
            raise GuidedContinuousRwdCorrectionRunError(
                "The corrected cache's ROI set/order does not match the accepted "
                f"review binding: cache={rois!r}, expected={expected_roi_order!r}."
            )
        chunk_ids = list_cache_chunk_ids(cache)
        if len(chunk_ids) != completion.corrected_segment_count:
            raise GuidedContinuousRwdCorrectionRunError(
                "The corrected cache's chunk count does not match the C4c "
                f"completion: cache={len(chunk_ids)}, "
                f"expected={completion.corrected_segment_count}."
            )
        meta = cache["meta"]
        checks = (
            ("continuous_completion_identity", completion.completion_identity),
            ("continuous_recording_identity", review_binding.recording.recording_identity),
            ("continuous_target_grid_identity", target_grid.target_grid_identity),
        )
        for attr_name, expected_value in checks:
            actual_value = meta.attrs.get(attr_name)
            if str(actual_value) != str(expected_value):
                raise GuidedContinuousRwdCorrectionRunError(
                    f"The corrected cache's {attr_name} does not match the accepted "
                    f"authorities: cache={actual_value!r}, expected={expected_value!r}."
                )
        if int(meta.attrs.get("continuous_target_sample_count", -1)) != target_grid.target_sample_count:
            raise GuidedContinuousRwdCorrectionRunError(
                "The corrected cache's target sample count does not match the "
                "accepted target grid."
            )
        if int(meta.attrs.get("continuous_corrected_segment_count", -1)) != completion.corrected_segment_count:
            raise GuidedContinuousRwdCorrectionRunError(
                "The corrected cache's corrected segment count does not match "
                "the C4c completion."
            )
    finally:
        cache.close()


def execute_guided_continuous_rwd_correction_run(
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
) -> GuidedContinuousRwdCorrectionRunResult:
    """Produce one coherent continuous-RWD correction run inside the existing
    run-directory / completed-run lifecycle.

    Accepts the same accepted continuous authorities C4c already requires
    (``review_binding``, ``target_grid``, ``block_plan``, ``segment_plan``,
    ``dynamic_f0_authority``, ``accepted_draft``, ``startup_mapping_contract``,
    an optional ``cancellation_requested`` callback) plus ``output_base`` (an
    absolute directory under which one new run directory is allocated) and
    ``config`` (forwarded to D1's writer exactly as D1's own signature
    already requires).

    Publication order: allocate the run directory -> write a running status
    -> build the C4c traversal and persist it through D1 -> cross-check the
    finalized cache against the accepted authorities and the C4c completion
    -> write ``run_report.json`` -> build and write ``MANIFEST.json`` -> write
    the final success ``status.json`` -> run the existing completed-run
    validator (``classify_run_terminal_state``) as the last gate before
    returning. Any failure at any step writes a terminal ``error`` (or
    ``cancelled``, when the traversal's own state says so) status instead,
    and the exception propagates -- no run directory this function touches
    can be left claiming success after a failure or cancellation.

    Returns a :class:`GuidedContinuousRwdCorrectionRunResult` only when the
    existing completed-run validator accepts the run as
    ``TERMINAL_SUCCESS_CURRENT``.
    """
    included_roi_ids = tuple(review_binding.recording.roi.included_roi_ids)
    run_mode = _build_run_mode(included_roi_ids)
    run_id, run_dir = _allocate_run_directory(output_base)
    _write_running_status(run_dir, run_id=run_id, run_mode=run_mode)

    cache_path = os.path.join(run_dir, CORRECTED_CACHE_RELATIVE_PATH)
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

        provenance = _per_roi_provenance(cache_path, included_roi_ids, first_chunk_id=0)
        report = {
            "schema_name": "guided_continuous_rwd_correction_run_report",
            "schema_version": "v1",
            "run_context": {"run_id": run_id, "stage": "continuous_correction_only"},
            "summary": {
                "narrative": (
                    "Continuous recording correction completed. A corrected trace "
                    "cache is available for this recording. Downstream tonic, "
                    "phasic, and feature-event analysis has not yet been run in "
                    "this backend integration stage."
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
            "continuous_correction_pass_completion_identity": completion.completion_identity,
        }
        report[REPORT_COMPLETION_KEY] = build_report_completion_block(run_id=run_id)
        _write_json(os.path.join(run_dir, RUN_REPORT_FILENAME), report)

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
            raise GuidedContinuousRwdCorrectionRunError(
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

    return GuidedContinuousRwdCorrectionRunResult(
        run_dir=run_dir,
        run_id=run_id,
        corrected_cache_path=cache_path,
        completion=completion,
        terminal_state=classification.state,
    )
