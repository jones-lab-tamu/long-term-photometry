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
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity
from photometry_pipeline.pipeline import Pipeline


GUIDED_NPM_WORKER_ENTRY_SUCCESS = 0
GUIDED_NPM_WORKER_ENTRY_REFUSED = 2
GUIDED_NPM_WORKER_ENTRY_FAILED = 3
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
) -> None:
    """Dispatch the verified worker through the existing numerical Pipeline."""
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

        kwargs["on_consumed_authority_verified"] = publish_consumed_authority
    pipeline.run_guided_npm_authorized(
        runtime["authorized_runtime"], runtime["output_dir"], **kwargs
    )


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
    try:
        run_guided_npm_worker(worker, launch_context=launch_context)
    except Exception:
        return GUIDED_NPM_WORKER_ENTRY_FAILED
    return GUIDED_NPM_WORKER_ENTRY_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
