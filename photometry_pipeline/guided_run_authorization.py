"""Pure composition of Guided validation, mapping, and execution preflights."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Callable

import photometry_pipeline.guided_backend_validation_workflow as validation_workflow
import photometry_pipeline.guided_execution_preflight as execution_preflight
import photometry_pipeline.guided_production_mapping as production_mapping
from photometry_pipeline.guided_backend_validation_request import (
    GuidedBackendValidationCompileSuccess,
    compute_guided_backend_validation_request_identity,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowOutcome,
)
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_production_mapping import (
    ApplicationBuildIdentity,
    GuidedProductionExecutionIntent,
    GuidedProductionMappingContract,
    GuidedProductionMappingSuccess,
)
from photometry_pipeline.guided_execution_preflight import (
    GuidedCandidateManifestExecutionPreflightResult,
    GuidedRoiExecutionPreflightResult,
)


GUIDED_RUN_AUTHORIZATION_SCHEMA_NAME = "guided_run_authorization"
GUIDED_RUN_AUTHORIZATION_SCHEMA_VERSION = "v1"
GUIDED_RUN_AUTHORIZATION_CONTRACT_VERSION = "guided_run_authorization.v1"
GUIDED_RUN_AUTHORIZATION_IDENTITY_DOMAIN = "guided-run-authorization:v1"

GUIDED_RUN_AUTHORIZATION_REFUSAL_CATEGORIES = (
    "authorization_request_invalid",
    "authorization_contract_unsupported",
    "stored_validation_missing",
    "stored_validation_not_accepted",
    "stored_validation_stale",
    "guided_revision_mismatch",
    "stored_request_identity_invalid",
    "stored_request_identity_inconsistent",
    "fresh_validation_refused",
    "fresh_request_identity_invalid",
    "fresh_request_identity_inconsistent",
    "fresh_request_identity_mismatch",
    "application_build_identity_missing",
    "application_build_identity_unusable",
    "production_mapping_refused",
    "production_intent_identity_inconsistent",
    "candidate_preflight_refused",
    "candidate_preflight_identity_inconsistent",
    "candidate_intent_binding_mismatch",
    "roi_preflight_refused",
    "roi_preflight_identity_inconsistent",
    "roi_candidate_binding_mismatch",
    "roi_intent_binding_mismatch",
    "authorization_cancelled",
    "authorization_internal_error",
)
GUIDED_RUN_AUTHORIZATION_REFUSAL_CATEGORY_SET = frozenset(
    GUIDED_RUN_AUTHORIZATION_REFUSAL_CATEGORIES
)
GUIDED_RUN_AUTHORIZATION_RESERVED_REFUSAL_CATEGORIES = (
    "application_build_provider_unavailable",
    "runner_consumption_contract_mismatch",
    "source_changed_after_authorization",
    "authorization_expired",
    "execution_transition_binding_mismatch",
)
GUIDED_RUN_AUTHORIZATION_IDENTITY_FIELDS = (
    "authorization_contract_version",
    "validation_request_identity",
    "production_execution_intent_identity",
    "application_build_identity",
    "production_mapping_contract_version",
    "runner_contract_version",
    "candidate_preflight_contract_version",
    "candidate_preflight_identity",
    "roi_preflight_contract_version",
    "roi_preflight_identity",
)

_HEX = frozenset("0123456789abcdef")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


@dataclass(frozen=True)
class GuidedRunAuthorizationRequest:
    authorization_schema_name: str
    authorization_schema_version: str
    authorization_contract_version: str
    stored_validation_outcome: GuidedBackendValidationWorkflowOutcome
    stored_validation_outcome_revision: int
    current_gui_revision: int
    current_validation_context: GuidedBackendValidationGuiContext
    application_build_identity: ApplicationBuildIdentity
    production_mapping_contract: GuidedProductionMappingContract

    def __post_init__(self) -> None:
        for name in (
            "authorization_schema_name",
            "authorization_schema_version",
            "authorization_contract_version",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} must be a non-empty string.")
        for name in (
            "stored_validation_outcome_revision",
            "current_gui_revision",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


def build_guided_run_authorization_request(
    *,
    stored_validation_outcome: GuidedBackendValidationWorkflowOutcome,
    stored_validation_outcome_revision: int,
    current_gui_revision: int,
    current_validation_context: GuidedBackendValidationGuiContext,
    application_build_identity: ApplicationBuildIdentity,
    production_mapping_contract: GuidedProductionMappingContract,
) -> GuidedRunAuthorizationRequest:
    return GuidedRunAuthorizationRequest(
        authorization_schema_name=GUIDED_RUN_AUTHORIZATION_SCHEMA_NAME,
        authorization_schema_version=GUIDED_RUN_AUTHORIZATION_SCHEMA_VERSION,
        authorization_contract_version=GUIDED_RUN_AUTHORIZATION_CONTRACT_VERSION,
        stored_validation_outcome=stored_validation_outcome,
        stored_validation_outcome_revision=stored_validation_outcome_revision,
        current_gui_revision=current_gui_revision,
        current_validation_context=current_validation_context,
        application_build_identity=application_build_identity,
        production_mapping_contract=production_mapping_contract,
    )


@dataclass(frozen=True)
class GuidedRunAuthorizationIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""

    def __post_init__(self) -> None:
        if self.category not in GUIDED_RUN_AUTHORIZATION_REFUSAL_CATEGORY_SET:
            raise ValueError("Unsupported authorization refusal category.")
        if not _text(self.section) or not _text(self.message):
            raise ValueError("Authorization issue requires section and message.")
        if not isinstance(self.detail_code, str):
            raise ValueError("detail_code must be a string.")


@dataclass(frozen=True)
class GuidedRunAuthorizationResult:
    status: str
    authorized: bool
    run_authorization: bool
    authorization_contract_version: str
    authorized_gui_revision: int | None
    stored_request_identity: str | None
    fresh_request_identity: str | None
    production_intent_identity: str | None
    application_build_identity: str | None
    candidate_preflight_identity: str | None
    roi_preflight_identity: str | None
    blocking_issues: tuple[GuidedRunAuthorizationIssue, ...]
    canonical_authorization_identity: str | None
    production_intent: GuidedProductionExecutionIntent | None
    candidate_preflight_result: (
        GuidedCandidateManifestExecutionPreflightResult | None
    )
    roi_preflight_result: GuidedRoiExecutionPreflightResult | None
    no_files_written: bool = True
    no_directories_created: bool = True
    no_artifacts_created: bool = True
    no_output_allocated: bool = True
    no_run_id_allocated: bool = True
    no_config_or_argv_generated: bool = True
    no_runner_invoked: bool = True

    def __post_init__(self) -> None:
        if self.status not in {"authorized", "refused", "cancelled"}:
            raise ValueError("Unsupported authorization status.")
        if not isinstance(self.blocking_issues, tuple):
            raise ValueError("blocking_issues must be a tuple.")
        if any(
            value is not True
            for value in (
                self.no_files_written,
                self.no_directories_created,
                self.no_artifacts_created,
                self.no_output_allocated,
                self.no_run_id_allocated,
                self.no_config_or_argv_generated,
                self.no_runner_invoked,
            )
        ):
            raise ValueError("Authorization results must assert no side effects.")
        if self.status == "authorized":
            identities = (
                self.stored_request_identity,
                self.fresh_request_identity,
                self.production_intent_identity,
                self.application_build_identity,
                self.candidate_preflight_identity,
                self.roi_preflight_identity,
                self.canonical_authorization_identity,
            )
            if (
                self.authorized is not True
                or self.run_authorization is not True
                or self.blocking_issues
                or any(not _sha256(value) for value in identities)
                or self.authorized_gui_revision is None
                or not isinstance(self.production_intent, GuidedProductionExecutionIntent)
                or not isinstance(
                    self.candidate_preflight_result,
                    GuidedCandidateManifestExecutionPreflightResult,
                )
                or not self.candidate_preflight_result.accepted
                or not isinstance(
                    self.roi_preflight_result,
                    GuidedRoiExecutionPreflightResult,
                )
                or not self.roi_preflight_result.accepted
            ):
                raise ValueError("Authorized result is incomplete.")
        elif (
            self.authorized is not False
            or self.run_authorization is not False
            or len(self.blocking_issues) != 1
            or self.canonical_authorization_identity is not None
            or self.production_intent is not None
            or self.candidate_preflight_result is not None
            or self.roi_preflight_result is not None
        ):
            raise ValueError("Non-authorized result is inconsistent.")


def _identity_payload(result: GuidedRunAuthorizationResult) -> dict[str, Any]:
    intent = result.production_intent
    candidate = result.candidate_preflight_result
    roi = result.roi_preflight_result
    if intent is None or candidate is None or roi is None:
        raise ValueError("Authorization proof bundle is incomplete.")
    return {
        "identity_domain": GUIDED_RUN_AUTHORIZATION_IDENTITY_DOMAIN,
        "authorization": {
            "authorization_contract_version": result.authorization_contract_version,
            "validation_request_identity": result.fresh_request_identity,
            "production_execution_intent_identity": (
                result.production_intent_identity
            ),
            "application_build_identity": result.application_build_identity,
            "production_mapping_contract_version": intent.mapping_contract_version,
            "runner_contract_version": intent.runner_contract_version,
            "candidate_preflight_contract_version": candidate.contract_version,
            "candidate_preflight_identity": result.candidate_preflight_identity,
            "roi_preflight_contract_version": roi.contract_version,
            "roi_preflight_identity": result.roi_preflight_identity,
        },
    }


def compute_guided_run_authorization_identity(
    result: GuidedRunAuthorizationResult,
) -> str:
    if not isinstance(result, GuidedRunAuthorizationResult) or not result.authorized:
        raise ValueError("Only an authorized result has an identity.")
    return hashlib.sha256(
        encode_canonical_value(_identity_payload(result))
    ).hexdigest()


def _issue(
    category: str, section: str, message: str, detail_code: str = ""
) -> GuidedRunAuthorizationIssue:
    return GuidedRunAuthorizationIssue(category, section, message, detail_code)


def _not_authorized(
    request: Any,
    issue: GuidedRunAuthorizationIssue,
    *,
    cancelled: bool = False,
    stored_identity: str | None = None,
    fresh_identity: str | None = None,
) -> GuidedRunAuthorizationResult:
    return GuidedRunAuthorizationResult(
        status="cancelled" if cancelled else "refused",
        authorized=False,
        run_authorization=False,
        authorization_contract_version=str(
            getattr(request, "authorization_contract_version", "") or ""
        ),
        authorized_gui_revision=None,
        stored_request_identity=stored_identity,
        fresh_request_identity=fresh_identity,
        production_intent_identity=None,
        application_build_identity=None,
        candidate_preflight_identity=None,
        roi_preflight_identity=None,
        blocking_issues=(issue,),
        canonical_authorization_identity=None,
        production_intent=None,
        candidate_preflight_result=None,
        roi_preflight_result=None,
    )


def _cancelled(
    request: Any,
    *,
    detail_code: str = "cancellation_requested",
    stored_identity: str | None = None,
    fresh_identity: str | None = None,
) -> GuidedRunAuthorizationResult:
    return _not_authorized(
        request,
        _issue(
            "authorization_cancelled",
            "authorization",
            "Guided Run authorization was cancelled.",
            detail_code,
        ),
        cancelled=True,
        stored_identity=stored_identity,
        fresh_identity=fresh_identity,
    )


def _is_cancelled(cancellation_check: Callable[[], bool] | None) -> bool:
    return bool(cancellation_check and cancellation_check())


def _lower_detail(result: Any, fallback: str) -> str:
    issues = getattr(result, "blocking_issues", ())
    if not issues:
        return fallback
    issue = issues[0]
    return str(
        getattr(issue, "category", "")
        or getattr(issue, "detail_code", "")
        or fallback
    )


def _lower_cancelled(result: Any) -> bool:
    if getattr(result, "status", "") == "cancelled":
        return True
    issues = getattr(result, "blocking_issues", ())
    return any(
        "cancel" in str(getattr(issue, "category", "")).lower()
        or "cancel" in str(getattr(issue, "detail_code", "")).lower()
        for issue in issues
    )


def _build_identity_usable(identity: Any) -> bool:
    if not isinstance(identity, ApplicationBuildIdentity):
        return False
    try:
        rebuilt = production_mapping.build_application_build_identity(
            distribution_name=identity.distribution_name,
            distribution_version=identity.distribution_version,
            source_revision_kind=identity.source_revision_kind,
            source_revision=identity.source_revision,
            source_tree_state=identity.source_tree_state,
            source_tree_digest=identity.source_tree_digest,
            build_artifact_digest=identity.build_artifact_digest,
            identity_provider_version=identity.identity_provider_version,
        )
    except Exception:
        return False
    return (
        rebuilt.schema_name == identity.schema_name
        and rebuilt.schema_version == identity.schema_version
        and rebuilt.canonical_identity == identity.canonical_identity
    )


def _accepted_workflow(outcome: Any) -> bool:
    return (
        isinstance(outcome, GuidedBackendValidationWorkflowOutcome)
        and outcome.status == "validator_accepted"
        and outcome.accepted_for_backend_validation is True
        and outcome.run_authorization is False
        and outcome.stale is False
        and isinstance(outcome.compile_result, GuidedBackendValidationCompileSuccess)
        and outcome.validation_result is not None
        and outcome.validation_result.accepted is True
    )


def _workflow_identity(outcome: GuidedBackendValidationWorkflowOutcome) -> tuple[
    str | None, str
]:
    identity = outcome.request_identity
    if not _sha256(identity):
        return None, "identity_invalid"
    compiled = outcome.compile_result
    validated = outcome.validation_result
    if not isinstance(compiled, GuidedBackendValidationCompileSuccess):
        return None, "compile_result_missing"
    try:
        recomputed = compute_guided_backend_validation_request_identity(
            compiled.request
        )
    except Exception:
        return None, "identity_recomputation_failed"
    if (
        compiled.canonical_request_identity != identity
        or validated is None
        or validated.request_identity != identity
        or recomputed != identity
    ):
        return None, "identity_chain_mismatch"
    return identity, ""


def authorize_guided_run(
    request: GuidedRunAuthorizationRequest,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedRunAuthorizationResult:
    if not isinstance(request, GuidedRunAuthorizationRequest):
        return _not_authorized(
            request,
            _issue(
                "authorization_request_invalid",
                "authorization",
                "Guided Run authorization request is invalid.",
                "request_invalid_type",
            ),
        )
    if (
        request.authorization_schema_name != GUIDED_RUN_AUTHORIZATION_SCHEMA_NAME
        or request.authorization_schema_version
        != GUIDED_RUN_AUTHORIZATION_SCHEMA_VERSION
        or request.authorization_contract_version
        != GUIDED_RUN_AUTHORIZATION_CONTRACT_VERSION
    ):
        return _not_authorized(
            request,
            _issue(
                "authorization_contract_unsupported",
                "authorization",
                "Guided Run authorization contract is unsupported.",
                "authorization_contract_mismatch",
            ),
        )
    try:
        if _is_cancelled(cancellation_check):
            return _cancelled(request)
        stored = request.stored_validation_outcome
        if not isinstance(stored, GuidedBackendValidationWorkflowOutcome):
            return _not_authorized(
                request,
                _issue(
                    "stored_validation_missing",
                    "stored_validation",
                    "Stored Guided validation is missing.",
                    "stored_outcome_invalid",
                ),
            )
        if stored.stale:
            return _not_authorized(
                request,
                _issue(
                    "stored_validation_stale",
                    "stored_validation",
                    "Stored Guided validation is stale.",
                    "stored_outcome_stale",
                ),
            )
        if not _accepted_workflow(stored):
            return _not_authorized(
                request,
                _issue(
                    "stored_validation_not_accepted",
                    "stored_validation",
                    "Stored Guided validation is not accepted.",
                    "stored_outcome_not_accepted",
                ),
            )
        context = request.current_validation_context
        if (
            not isinstance(context, GuidedBackendValidationGuiContext)
            or request.stored_validation_outcome_revision
            != request.current_gui_revision
            or context.revision != request.current_gui_revision
        ):
            return _not_authorized(
                request,
                _issue(
                    "guided_revision_mismatch",
                    "revision",
                    "Guided revision changed.",
                    "guided_revision_mismatch",
                ),
            )
        stored_identity, stored_problem = _workflow_identity(stored)
        if stored_identity is None:
            category = (
                "stored_request_identity_invalid"
                if stored_problem == "identity_invalid"
                else "stored_request_identity_inconsistent"
            )
            return _not_authorized(
                request,
                _issue(
                    category,
                    "stored_validation",
                    "Stored request identity is invalid or inconsistent.",
                    stored_problem,
                ),
            )
        if _is_cancelled(cancellation_check):
            return _cancelled(request, stored_identity=stored_identity)
        fresh = validation_workflow.validate_current_guided_draft_for_backend(
            context.draft,
            parser_contract=context.parser_contract,
            additional_protected_roots=context.additional_protected_roots,
            validator_contract=context.validator_contract,
            cancellation_check=cancellation_check,
        )
        if _lower_cancelled(fresh):
            return _cancelled(
                request,
                detail_code="fresh_validation_cancelled",
                stored_identity=stored_identity,
            )
        if not _accepted_workflow(fresh):
            return _not_authorized(
                request,
                _issue(
                    "fresh_validation_refused",
                    "fresh_validation",
                    "Fresh Guided validation was refused.",
                    _lower_detail(fresh, "fresh_validation_not_accepted"),
                ),
                stored_identity=stored_identity,
            )
        fresh_identity, fresh_problem = _workflow_identity(fresh)
        if fresh_identity is None:
            category = (
                "fresh_request_identity_invalid"
                if fresh_problem == "identity_invalid"
                else "fresh_request_identity_inconsistent"
            )
            return _not_authorized(
                request,
                _issue(
                    category,
                    "fresh_validation",
                    "Fresh request identity is invalid or inconsistent.",
                    fresh_problem,
                ),
                stored_identity=stored_identity,
            )
        if fresh_identity != stored_identity:
            return _not_authorized(
                request,
                _issue(
                    "fresh_request_identity_mismatch",
                    "fresh_validation",
                    "Fresh request identity differs from stored acceptance.",
                    "stored_fresh_identity_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if _is_cancelled(cancellation_check):
            return _cancelled(
                request,
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if not isinstance(request.application_build_identity, ApplicationBuildIdentity):
            return _not_authorized(
                request,
                _issue(
                    "application_build_identity_missing",
                    "build_identity",
                    "Application build identity is required.",
                    "build_identity_invalid_type",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if not _build_identity_usable(request.application_build_identity):
            return _not_authorized(
                request,
                _issue(
                    "application_build_identity_unusable",
                    "build_identity",
                    "Application build identity is unusable.",
                    "build_identity_invalid",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        compiled = fresh.compile_result
        assert isinstance(compiled, GuidedBackendValidationCompileSuccess)
        mapped = production_mapping.map_guided_validation_request_to_execution_intent(
            compiled.request,
            canonical_request_identity=fresh_identity,
            application_build_identity=request.application_build_identity,
            mapping_contract=request.production_mapping_contract,
        )
        if not isinstance(mapped, GuidedProductionMappingSuccess):
            return _not_authorized(
                request,
                _issue(
                    "production_mapping_refused",
                    "production_mapping",
                    "Guided production mapping was refused.",
                    _lower_detail(mapped, "mapping_refused"),
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        intent = mapped.intent
        try:
            recomputed_intent = (
                production_mapping.compute_guided_production_execution_intent_identity(
                    intent
                )
            )
        except Exception:
            recomputed_intent = ""
        if (
            mapped.source_request_identity != fresh_identity
            or intent.source_request_identity != fresh_identity
            or mapped.canonical_intent_identity != intent.canonical_intent_identity
            or recomputed_intent != intent.canonical_intent_identity
        ):
            return _not_authorized(
                request,
                _issue(
                    "production_intent_identity_inconsistent",
                    "production_mapping",
                    "Production intent identity is inconsistent.",
                    "production_intent_identity_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        contract = request.production_mapping_contract
        if (
            not isinstance(contract, GuidedProductionMappingContract)
            or intent.mapping_contract_version != contract.mapping_contract_version
            or intent.runner_contract_version != contract.runner_contract_version
            or intent.input_source.candidate_manifest_execution_contract_version
            != contract.candidate_manifest_execution_contract_version
            or intent.roi_scope.roi_execution_contract_version
            != contract.roi_execution_contract_version
        ):
            return _not_authorized(
                request,
                _issue(
                    "production_intent_identity_inconsistent",
                    "production_mapping",
                    "Production intent is not bound to the mapping contract.",
                    "production_contract_binding_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if _is_cancelled(cancellation_check):
            return _cancelled(
                request,
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        candidate_request = (
            execution_preflight.derive_candidate_manifest_preflight_request_from_intent(
                intent
            )
        )
        candidate = execution_preflight.run_candidate_manifest_execution_preflight(
            candidate_request,
            cancellation_check=cancellation_check,
        )
        if _lower_cancelled(candidate):
            return _cancelled(
                request,
                detail_code=_lower_detail(candidate, "candidate_preflight_cancelled"),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if (
            not isinstance(
                candidate, GuidedCandidateManifestExecutionPreflightResult
            )
            or not candidate.accepted
        ):
            return _not_authorized(
                request,
                _issue(
                    "candidate_preflight_refused",
                    "candidate_preflight",
                    "Candidate execution preflight was refused.",
                    _lower_detail(candidate, "candidate_preflight_refused"),
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        try:
            candidate_identity = (
                execution_preflight.compute_guided_candidate_preflight_identity(
                    candidate
                )
            )
        except Exception:
            candidate_identity = ""
        if candidate_identity != candidate.canonical_preflight_identity:
            return _not_authorized(
                request,
                _issue(
                    "candidate_preflight_identity_inconsistent",
                    "candidate_preflight",
                    "Candidate preflight identity is inconsistent.",
                    "candidate_preflight_identity_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        intent_candidates = tuple(
            (
                item.canonical_relative_path,
                item.size_bytes,
                item.sha256_content_digest,
            )
            for item in intent.input_source.candidate_files
        )
        actual_candidates = tuple(
            (
                item.canonical_relative_path,
                item.size_bytes,
                item.sha256_content_digest,
            )
            for item in candidate.actual_candidates
        )
        if (
            candidate.actual_candidate_set_digest
            != intent.input_source.source_candidate_set_digest
            or candidate.actual_candidate_content_digest
            != intent.input_source.source_candidate_content_digest
            or actual_candidates != intent_candidates
        ):
            return _not_authorized(
                request,
                _issue(
                    "candidate_intent_binding_mismatch",
                    "candidate_preflight",
                    "Candidate preflight is not bound to production intent.",
                    "candidate_intent_binding_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if _is_cancelled(cancellation_check):
            return _cancelled(
                request,
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        roi_request = execution_preflight.derive_roi_execution_preflight_request_from_intent(
            intent,
            accepted_candidate_preflight_identity=candidate_identity,
        )
        roi = execution_preflight.run_roi_execution_preflight(
            roi_request,
            cancellation_check=cancellation_check,
        )
        if _lower_cancelled(roi):
            return _cancelled(
                request,
                detail_code=_lower_detail(roi, "roi_preflight_cancelled"),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if (
            not isinstance(roi, GuidedRoiExecutionPreflightResult)
            or not roi.accepted
        ):
            return _not_authorized(
                request,
                _issue(
                    "roi_preflight_refused",
                    "roi_preflight",
                    "ROI execution preflight was refused.",
                    _lower_detail(roi, "roi_preflight_refused"),
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        try:
            roi_identity = execution_preflight.compute_guided_roi_preflight_identity(
                roi
            )
        except Exception:
            roi_identity = ""
        if roi_identity != roi.canonical_preflight_identity:
            return _not_authorized(
                request,
                _issue(
                    "roi_preflight_identity_inconsistent",
                    "roi_preflight",
                    "ROI preflight identity is inconsistent.",
                    "roi_preflight_identity_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if roi.accepted_candidate_preflight_identity != candidate_identity:
            return _not_authorized(
                request,
                _issue(
                    "roi_candidate_binding_mismatch",
                    "roi_preflight",
                    "ROI preflight is not bound to candidate acceptance.",
                    "roi_candidate_identity_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if (
            roi.source_candidate_content_digest
            != intent.input_source.source_candidate_content_digest
            or roi.parser_contract_digest != intent.parser.parser_contract_digest
            or roi.actual_discovered_roi_ids != intent.roi_scope.discovered_roi_ids
            or roi.actual_included_roi_ids != intent.roi_scope.included_roi_ids
            or roi.actual_excluded_roi_ids != intent.roi_scope.excluded_roi_ids
            or roi.actual_strict_roi_inventory_digest
            != roi_request.expected_strict_roi_inventory_digest
        ):
            return _not_authorized(
                request,
                _issue(
                    "roi_intent_binding_mismatch",
                    "roi_preflight",
                    "ROI preflight is not bound to production intent.",
                    "roi_intent_binding_mismatch",
                ),
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        if _is_cancelled(cancellation_check):
            return _cancelled(
                request,
                stored_identity=stored_identity,
                fresh_identity=fresh_identity,
            )
        provisional = GuidedRunAuthorizationResult(
            status="authorized",
            authorized=True,
            run_authorization=True,
            authorization_contract_version=request.authorization_contract_version,
            authorized_gui_revision=request.current_gui_revision,
            stored_request_identity=stored_identity,
            fresh_request_identity=fresh_identity,
            production_intent_identity=intent.canonical_intent_identity,
            application_build_identity=(
                request.application_build_identity.canonical_identity
            ),
            candidate_preflight_identity=candidate_identity,
            roi_preflight_identity=roi_identity,
            blocking_issues=(),
            canonical_authorization_identity="0" * 64,
            production_intent=intent,
            candidate_preflight_result=candidate,
            roi_preflight_result=roi,
        )
        return replace(
            provisional,
            canonical_authorization_identity=(
                compute_guided_run_authorization_identity(provisional)
            ),
        )
    except Exception:
        return _not_authorized(
            request,
            _issue(
                "authorization_internal_error",
                "authorization",
                "Guided Run authorization could not complete.",
                "authorization_exception",
            ),
        )
