"""Build a current Guided NPM worker prelaunch claim from an accepted
Guided backend validation outcome, without launching a worker process.

This module adds no new validation rules of its own. It chains the
already-committed, independently-tested NPM production-mapping/authority/
authorization/startup-payload/persistence/claim/materialization/prelaunch-
claim functions in the exact order their own tests already exercise them
(see tests/test_guided_npm_worker_prelaunch_claim.py and its fixture
chain), mirroring the existing RWD equivalent in
guided_execution_request_builder.py. It performs the same disk writes those
underlying functions already perform on their own (allocating one NPM run
directory and writing the startup-payload and worker-request artifacts);
it does not add, skip, or reorder any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from photometry_pipeline.application_build_identity import (
    resolve_application_build_identity,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowOutcome,
)
from photometry_pipeline.guided_npm_authorization import (
    GuidedNpmExecutionAuthorization,
    authorize_guided_npm_execution_authority,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GuidedNpmExecutionAuthority,
    build_guided_npm_execution_authority,
)
from photometry_pipeline.guided_npm_production_execution_request import (
    GuidedNpmProductionExecutionRequest,
    build_guided_npm_production_execution_request,
)
from photometry_pipeline.guided_npm_startup_claim import (
    GuidedNpmStartupClaimReceipt,
    claim_guided_npm_startup_artifact,
)
from photometry_pipeline.guided_npm_startup_payload import (
    GuidedNpmStartupPayload,
    build_guided_npm_startup_payload,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    GuidedNpmStartupPersistenceReceipt,
    persist_guided_npm_startup_payload,
)
from photometry_pipeline.guided_npm_worker_prelaunch_claim import (
    GuidedNpmWorkerPrelaunchClaim,
    claim_guided_npm_worker_for_prelaunch,
)
from photometry_pipeline.guided_npm_worker_request_materialization import (
    GuidedNpmWorkerRequestMaterializationReceipt,
    materialize_guided_npm_worker_request,
)
from photometry_pipeline.guided_production_mapping import (
    GuidedNpmProductionMappingSuccess,
    build_guided_production_mapping_contract,
    map_guided_npm_validation_outcome_to_execution_intent,
)


@dataclass(frozen=True)
class GuidedNpmRunLaunchBuildIssue:
    category: str
    section: str
    message: str


@dataclass(frozen=True)
class GuidedNpmRunLaunchBuildResult:
    status: str
    ok: bool
    prelaunch_claim: GuidedNpmWorkerPrelaunchClaim | None
    application_build_identity: object | None
    blocking_issues: tuple[GuidedNpmRunLaunchBuildIssue, ...]
    current_gui_revision: int
    no_execution_invoked: bool = True


def _refused(
    status: str,
    section: str,
    message: str,
    revision: int,
) -> GuidedNpmRunLaunchBuildResult:
    return GuidedNpmRunLaunchBuildResult(
        status=status,
        ok=False,
        prelaunch_claim=None,
        application_build_identity=None,
        blocking_issues=(
            GuidedNpmRunLaunchBuildIssue(status, section, message),
        ),
        current_gui_revision=revision,
    )


def build_guided_npm_worker_prelaunch_claim_from_validation(
    *,
    validation_context: GuidedBackendValidationGuiContext,
    validation_outcome: GuidedBackendValidationWorkflowOutcome,
    current_gui_revision: int,
    project_root: Path | str | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmRunLaunchBuildResult:
    """Build the exact NPM prelaunch claim in memory-plus-durable-artifacts."""
    if (
        isinstance(current_gui_revision, bool)
        or not isinstance(current_gui_revision, int)
        or current_gui_revision < 0
        or not isinstance(
            validation_context, GuidedBackendValidationGuiContext
        )
        or not isinstance(
            validation_outcome, GuidedBackendValidationWorkflowOutcome
        )
    ):
        return _refused(
            "invalid_context",
            "validation",
            "Current Guided validation context is invalid.",
            current_gui_revision
            if isinstance(current_gui_revision, int)
            else 0,
        )
    if (
        validation_context.revision != current_gui_revision
        or validation_outcome.stale is not False
    ):
        return _refused(
            "validation_not_current",
            "validation",
            "Guided validation is no longer current.",
            current_gui_revision,
        )
    if (
        validation_outcome.status != "validator_accepted"
        or validation_outcome.accepted_for_backend_validation is not True
    ):
        return _refused(
            "validation_not_accepted",
            "validation",
            "Guided validation was not accepted.",
            current_gui_revision,
        )
    if validation_context.draft.input_format != "npm":
        return _refused(
            "unsupported_format",
            "validation",
            "Only NPM analyses use this launch path.",
            current_gui_revision,
        )

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    build_result = resolve_application_build_identity(project_root=root)
    if build_result.build_identity is None:
        return _refused(
            "build_identity_unavailable",
            "build_identity",
            "Application build identity is unavailable.",
            current_gui_revision,
        )
    build_identity = build_result.build_identity
    plan_identity = validation_outcome.guided_plan_identity
    if not plan_identity:
        return _refused(
            "plan_identity_unavailable",
            "validation",
            "Guided plan identity is unavailable.",
            current_gui_revision,
        )

    try:
        mapping_contract = build_guided_production_mapping_contract()
        mapping_result = map_guided_npm_validation_outcome_to_execution_intent(
            validation_outcome,
            expected_validation_revision=current_gui_revision,
            expected_plan_identity=plan_identity,
            application_build_identity=build_identity,
            mapping_contract=mapping_contract,
        )
    except Exception:
        mapping_result = None
    if not isinstance(mapping_result, GuidedNpmProductionMappingSuccess):
        return _refused(
            "production_mapping_failed",
            "production_mapping",
            "Guided NPM production mapping failed.",
            current_gui_revision,
        )

    try:
        authority_result = build_guided_npm_execution_authority(
            mapping_result.intent
        )
    except Exception:
        authority_result = None
    if not isinstance(authority_result, GuidedNpmExecutionAuthority):
        return _refused(
            "execution_authority_failed",
            "execution_authority",
            "Guided NPM execution authority could not be built.",
            current_gui_revision,
        )

    try:
        authorization_result = authorize_guided_npm_execution_authority(
            authority_result,
            expected_validation_revision=authority_result.validation_revision,
            expected_plan_identity=authority_result.guided_plan_identity,
            current_application_build_identity=build_identity,
            cancellation_check=cancellation_check,
        )
    except Exception:
        authorization_result = None
    if not isinstance(authorization_result, GuidedNpmExecutionAuthorization):
        return _refused(
            "execution_authorization_failed",
            "execution_authorization",
            "Guided NPM execution authorization failed.",
            current_gui_revision,
        )

    try:
        startup_payload_result = build_guided_npm_startup_payload(
            authorization_result, authority_result
        )
    except Exception:
        startup_payload_result = None
    if not isinstance(startup_payload_result, GuidedNpmStartupPayload):
        return _refused(
            "startup_payload_failed",
            "startup_payload",
            "Guided NPM startup payload could not be built.",
            current_gui_revision,
        )

    try:
        persistence_result = persist_guided_npm_startup_payload(
            startup_payload_result, cancellation_check=cancellation_check
        )
    except Exception:
        persistence_result = None
    if not isinstance(persistence_result, GuidedNpmStartupPersistenceReceipt):
        return _refused(
            "startup_persistence_failed",
            "startup_persistence",
            "The NPM startup payload could not be saved.",
            current_gui_revision,
        )

    try:
        startup_claim_result = claim_guided_npm_startup_artifact(
            persistence_result,
            current_application_build_identity=build_identity,
            cancellation_check=cancellation_check,
        )
    except Exception:
        startup_claim_result = None
    if not isinstance(startup_claim_result, GuidedNpmStartupClaimReceipt):
        return _refused(
            "startup_claim_failed",
            "startup_claim",
            "The NPM startup artifact could not be claimed.",
            current_gui_revision,
        )

    try:
        execution_request_result = build_guided_npm_production_execution_request(
            startup_claim_result, startup_payload_result
        )
    except Exception:
        execution_request_result = None
    if not isinstance(
        execution_request_result, GuidedNpmProductionExecutionRequest
    ):
        return _refused(
            "production_execution_request_failed",
            "production_execution_request",
            "The NPM production execution request could not be built.",
            current_gui_revision,
        )

    try:
        materialization_result = materialize_guided_npm_worker_request(
            execution_request_result,
            current_application_build_identity=build_identity,
            cancellation_check=cancellation_check,
        )
    except Exception:
        materialization_result = None
    if not isinstance(
        materialization_result, GuidedNpmWorkerRequestMaterializationReceipt
    ):
        return _refused(
            "worker_request_materialization_failed",
            "worker_request_materialization",
            "The NPM worker request could not be saved.",
            current_gui_revision,
        )

    try:
        prelaunch_claim_result = claim_guided_npm_worker_for_prelaunch(
            materialization_result.worker_request_artifact_path,
            materialization_result,
            current_application_build_identity=build_identity,
            cancellation_check=cancellation_check,
        )
    except Exception:
        prelaunch_claim_result = None
    if not isinstance(prelaunch_claim_result, GuidedNpmWorkerPrelaunchClaim):
        return _refused(
            "prelaunch_claim_failed",
            "prelaunch_claim",
            "The NPM worker could not be claimed for launch.",
            current_gui_revision,
        )

    return GuidedNpmRunLaunchBuildResult(
        status="built",
        ok=True,
        prelaunch_claim=prelaunch_claim_result,
        application_build_identity=build_identity,
        blocking_issues=(),
        current_gui_revision=current_gui_revision,
    )
