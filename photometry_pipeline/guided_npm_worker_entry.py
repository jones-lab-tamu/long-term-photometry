"""Production child entry for one canonical Guided NPM worker request.

The production route accepts one worker-request artifact and derives every
scientific input from its embedded B2-C6A projections.  The smoke-only route is
intentionally a different argument and is not accepted by the parent launch
invocation contract.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from photometry_pipeline.application_build_identity import (
    resolve_application_build_identity,
)
from photometry_pipeline.guided_npm_authorized_adapter import (
    build_guided_npm_authorized_runtime,
)
from photometry_pipeline.guided_npm_worker_acknowledgement import (
    GuidedNpmWorkerLaunchContext,
    build_guided_npm_worker_consumed_authority_receipt,
    publish_guided_npm_worker_consumed_authority_receipt,
    read_guided_npm_worker_launch_context,
)
from photometry_pipeline.guided_npm_worker_prelaunch_claim import stored_paths_equal
from photometry_pipeline.guided_npm_worker_request import (
    GuidedNpmWorkerRequest,
    decode_canonical_guided_npm_worker_request_bytes,
    verify_guided_npm_worker_request,
)
from photometry_pipeline.guided_npm_worker_request_materialization import (
    verify_guided_npm_source_freshness_live,
    verify_guided_npm_startup_artifact_live,
)
from photometry_pipeline.guided_npm_worker_terminal import (
    STAGE_AUTHORIZED_RUNTIME_BUILD,
    STAGE_COMPLETED_RUN_VERIFICATION,
    STAGE_CONSUMED_AUTHORITY_PUBLICATION,
    STAGE_LAUNCH_CONTEXT_VERIFICATION,
    STAGE_NUMERICAL_DISPATCH,
    STAGE_NUMERICAL_PIPELINE_RETURNED,
    STAGE_PASS_1,
    STAGE_TERMINAL_RECEIPT_PUBLICATION,
    TERMINAL_OUTCOME_FAILED_AFTER_CONSUMED,
    TERMINAL_OUTCOME_FAILED_BEFORE_CONSUMED,
    TERMINAL_OUTCOME_FAILED_DURING_OUTPUT_FINALIZATION,
    build_guided_npm_required_output_evidence,
    build_guided_npm_worker_terminal_failure_receipt,
    build_guided_npm_worker_terminal_success_receipt,
    publish_guided_npm_worker_terminal_receipt,
)
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity
from photometry_pipeline.pipeline import Pipeline


GUIDED_NPM_WORKER_ENTRY_SUCCESS = 0
GUIDED_NPM_WORKER_ENTRY_REFUSED = 2
GUIDED_NPM_WORKER_ENTRY_FAILED = 3
GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED = 4
GUIDED_NPM_WORKER_SMOKE_ARGUMENT = "--guided-npm-worker-entry-smoke-test-only"
GUIDED_NPM_LAUNCH_CONTEXT_ARGUMENT = "--guided-npm-launch-context"

def _project_root() -> Path:
    return Path(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))


def _resolve_current_build() -> ApplicationBuildIdentity:
    result = resolve_application_build_identity(project_root=_project_root())
    if result.status != "resolved" or result.build_identity is None:
        raise ValueError("worker_current_build_unavailable")
    return result.build_identity


def load_verified_guided_npm_worker_request(
    artifact_path: str,
    *,
    current_application_build_identity: ApplicationBuildIdentity | None = None,
) -> GuidedNpmWorkerRequest:
    """Reread and verify the child's sole authority before numerical dispatch."""
    path = Path(artifact_path)
    if not path.is_absolute():
        raise ValueError("worker_artifact_path_invalid")
    content = path.read_bytes()
    worker = decode_canonical_guided_npm_worker_request_bytes(content)
    verify_guided_npm_worker_request(worker)
    style = worker.execution_request.output_runtime_projection.output_base_path_style
    if not stored_paths_equal(artifact_path, worker.worker_request_artifact_path, style):
        raise ValueError("worker_artifact_path_mismatch")
    current = current_application_build_identity or _resolve_current_build()
    if not (
        current
        == worker.application_build_identity
        == worker.execution_request.application_build_identity
    ):
        raise ValueError("worker_current_build_mismatch")
    verify_guided_npm_startup_artifact_live(worker.execution_request)
    verify_guided_npm_source_freshness_live(worker.execution_request)
    return worker


def build_guided_npm_pipeline_runtime(worker: GuidedNpmWorkerRequest) -> dict[str, Any]:
    """Expose the exact authorized runtime to the repository-owned child."""
    authorized_runtime = build_guided_npm_authorized_runtime(worker)
    execution = worker.execution_request
    return {
        "authorized_runtime": authorized_runtime,
        "config": authorized_runtime.config,
        "mode": authorized_runtime.mode,
        "per_roi_correction": authorized_runtime.per_roi_correction,
        "per_roi_feature_config": authorized_runtime.per_roi_feature_config,
        "per_roi_feature_provenance": authorized_runtime.per_roi_feature_provenance,
        "input_dir": execution.source_runtime_projection.source_root_canonical,
        "output_dir": execution.output_runtime_projection.run_directory_path,
        "force_format": "npm",
        "recursive": False,
        "ordered_session_paths": authorized_runtime.authorized_input.ordered_session_paths,
        "selected_canonical_roi_ids": (
            authorized_runtime.authorized_input.selected_canonical_roi_ids
        ),
        "traces_only": False,
    }


def run_guided_npm_worker(
    worker: GuidedNpmWorkerRequest,
    *,
    launch_context: GuidedNpmWorkerLaunchContext | None = None,
    pipeline_factory: Callable[..., Pipeline] = Pipeline,
    on_pass_1_complete: Callable[[], None] | None = None,
    on_consumed_authority_published: Callable[[Any], None] | None = None,
) -> None:
    """Dispatch the verified worker through the existing numerical Pipeline.

    ``on_pass_1_complete`` and ``on_consumed_authority_published`` are optional
    B2-D2A observer hooks used only to build the child's own terminal receipt;
    they default to None and do not alter this function's existing dispatch,
    acknowledgement, or exception behavior for any existing caller.
    """
    runtime = build_guided_npm_pipeline_runtime(worker)
    pipeline = pipeline_factory(
        runtime["config"],
        mode=runtime["mode"],
        per_roi_correction=runtime["per_roi_correction"],
        per_roi_feature_config=runtime["per_roi_feature_config"],
        per_roi_feature_provenance=runtime["per_roi_feature_provenance"],
    )
    kwargs = {"traces_only": runtime["traces_only"]}
    if launch_context is not None:
        publication_attempted = False

        def publish_consumed_authority(evidence) -> None:
            nonlocal publication_attempted
            if on_pass_1_complete is not None:
                on_pass_1_complete()
            if publication_attempted:
                raise RuntimeError("consumed_authority_publication_repeated")
            publication_attempted = True
            receipt = build_guided_npm_worker_consumed_authority_receipt(
                worker_request=worker,
                launch_context=launch_context,
                evidence=evidence,
                observed_process_id=os.getpid(),
            )
            publish_guided_npm_worker_consumed_authority_receipt(
                receipt,
                receipt_path=launch_context.consumed_authority_receipt_path,
                launch_context=launch_context,
            )
            if on_consumed_authority_published is not None:
                on_consumed_authority_published(receipt)

        kwargs["on_consumed_authority_verified"] = publish_consumed_authority
    pipeline.run_guided_npm_authorized(
        runtime["authorized_runtime"], runtime["output_dir"], **kwargs
    )


def run_guided_npm_worker_to_terminal_receipt(
    worker: GuidedNpmWorkerRequest,
    *,
    launch_context: GuidedNpmWorkerLaunchContext,
    pipeline_factory: Callable[..., Pipeline] = Pipeline,
) -> tuple[int, Any]:
    """Dispatch the verified worker and durably record its own terminal receipt.

    Assumes ``worker`` and ``launch_context`` are already verified (as
    ``main()`` does before calling this).  Separated from ``main()`` so tests
    can drive the full B2-D2A lifecycle with a fixture-built worker/launch
    context directly, the same way existing B2-D1 tests call
    ``run_guided_npm_worker`` directly rather than through the CLI's own
    real-repository build-identity resolution.

    Returns the worker-entry exit code alongside the exact terminal receipt
    that was durably published, or ``None`` when no receipt could be.
    """
    # A process start timestamp anchors the required-output freshness floor:
    # every artifact this same process writes necessarily gets an mtime at or
    # after this instant, so a stale leftover from an earlier attempt reusing
    # this run directory (predating this process) can never pass as fresh,
    # while config_used.yaml/run_report.json -- written once, before Pass 1,
    # and never touched again for phasic runs whose mode-specific late
    # appends don't apply -- are correctly accepted since they were written
    # by *this* process, not an earlier one.
    process_start_time_ns = time.time_ns()
    observed_process_id = os.getpid()
    stage = STAGE_LAUNCH_CONTEXT_VERIFICATION
    consumed_authority_receipt = None
    terminal_outcome: str | None = None
    failure_category = ""
    failure_exception_type = ""

    def _on_pass_1_complete() -> None:
        nonlocal stage
        stage = STAGE_PASS_1

    def _on_consumed_authority_published(receipt: Any) -> None:
        nonlocal stage, consumed_authority_receipt
        consumed_authority_receipt = receipt
        stage = STAGE_CONSUMED_AUTHORITY_PUBLICATION

    # KeyboardInterrupt/SystemExit are deliberately not caught here: only
    # ordinary Exception subclasses are classified into terminal evidence, so
    # an interrupted or deliberately-exited process leaves no false success or
    # false failure receipt behind.
    try:
        stage = STAGE_AUTHORIZED_RUNTIME_BUILD
        traces_only = bool(build_guided_npm_pipeline_runtime(worker)["traces_only"])
        stage = STAGE_NUMERICAL_DISPATCH
        run_guided_npm_worker(
            worker,
            launch_context=launch_context,
            pipeline_factory=pipeline_factory,
            on_pass_1_complete=_on_pass_1_complete,
            on_consumed_authority_published=_on_consumed_authority_published,
        )
        stage = STAGE_NUMERICAL_PIPELINE_RETURNED
    except Exception as exc:
        if consumed_authority_receipt is not None:
            terminal_outcome = TERMINAL_OUTCOME_FAILED_AFTER_CONSUMED
            failure_category = "pipeline_execution_failed"
        else:
            terminal_outcome = TERMINAL_OUTCOME_FAILED_BEFORE_CONSUMED
            failure_category = (
                "authorized_runtime_build_failed"
                if stage == STAGE_AUTHORIZED_RUNTIME_BUILD
                else "numerical_dispatch_failed"
            )
        failure_exception_type = type(exc).__name__

    success_receipt = None
    if terminal_outcome is None:
        try:
            stage = STAGE_COMPLETED_RUN_VERIFICATION
            output_evidence = build_guided_npm_required_output_evidence(
                worker.run_directory_path,
                worker.execution_request.execution_mode,
                traces_only=traces_only,
                not_before_mtime_ns=process_start_time_ns,
            )
            success_receipt = build_guided_npm_worker_terminal_success_receipt(
                worker_request=worker,
                launch_context=launch_context,
                consumed_authority_receipt=consumed_authority_receipt,
                observed_process_id=observed_process_id,
                output_evidence=output_evidence,
            )
        except Exception as exc:
            terminal_outcome = TERMINAL_OUTCOME_FAILED_DURING_OUTPUT_FINALIZATION
            failure_category = "output_evidence_verification_failed"
            failure_exception_type = type(exc).__name__

    if success_receipt is not None:
        try:
            stage = STAGE_TERMINAL_RECEIPT_PUBLICATION
            publish_guided_npm_worker_terminal_receipt(success_receipt)
        except Exception:
            return GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED, None
        return GUIDED_NPM_WORKER_ENTRY_SUCCESS, success_receipt

    try:
        failure_receipt = build_guided_npm_worker_terminal_failure_receipt(
            worker_request=worker,
            launch_context=launch_context,
            observed_process_id=observed_process_id,
            terminal_outcome=terminal_outcome,
            terminal_stage=stage,
            consumed_authority_receipt=consumed_authority_receipt,
            failure_category=failure_category,
            failure_exception_type=failure_exception_type,
        )
        publish_guided_npm_worker_terminal_receipt(failure_receipt)
    except Exception:
        return GUIDED_NPM_WORKER_ENTRY_TERMINAL_PUBLICATION_FAILED, None
    return GUIDED_NPM_WORKER_ENTRY_FAILED, failure_receipt


def main(argv: tuple[str, ...] | list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guided NPM production worker")
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--guided-npm-worker-request")
    authority.add_argument(GUIDED_NPM_WORKER_SMOKE_ARGUMENT, action="store_true")
    parser.add_argument(GUIDED_NPM_LAUNCH_CONTEXT_ARGUMENT)
    args = parser.parse_args(argv)
    if getattr(args, GUIDED_NPM_WORKER_SMOKE_ARGUMENT[2:].replace("-", "_")):
        if args.guided_npm_launch_context is not None:
            return GUIDED_NPM_WORKER_ENTRY_REFUSED
        return GUIDED_NPM_WORKER_ENTRY_SUCCESS
    if args.guided_npm_launch_context is None:
        return GUIDED_NPM_WORKER_ENTRY_REFUSED
    try:
        worker = load_verified_guided_npm_worker_request(
            args.guided_npm_worker_request
        )
        launch_context = read_guided_npm_worker_launch_context(
            args.guided_npm_launch_context,
            worker_request=worker,
        )
    except (OSError, TypeError, ValueError):
        return GUIDED_NPM_WORKER_ENTRY_REFUSED
    exit_code, _receipt = run_guided_npm_worker_to_terminal_receipt(
        worker, launch_context=launch_context
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
