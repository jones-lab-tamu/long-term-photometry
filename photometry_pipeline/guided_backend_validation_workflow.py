"""Backend-neutral orchestration for Guided validation without execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from photometry_pipeline.guided_backend_validation_materialization import (
    GuidedBackendValidationMaterializationFailure,
    GuidedBackendValidationMaterializationSuccess,
    materialize_guided_backend_validation_facts,
)
from photometry_pipeline.guided_backend_validation_request import (
    GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
    GUIDED_BACKEND_VALIDATION_SCOPE,
    GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
    GuidedBackendValidationCompileFailure,
    GuidedBackendValidationCompileSuccess,
    GuidedBackendValidatorContract,
    compile_guided_backend_validation_request,
)
from photometry_pipeline.guided_backend_validator import (
    GUIDED_BACKEND_VALIDATOR_CAPABILITY_VERSION,
    GuidedBackendValidationResult,
    validate_guided_backend_validation_request,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
)
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)
from photometry_pipeline.io.rwd_contract import RwdHeaderParsingContract
from photometry_pipeline.io.npm_contract import NpmParserContract


GUIDED_BACKEND_VALIDATION_WORKFLOW_STATUSES = frozenset(
    {
        "cancelled",
        "materialization_failed",
        "compile_failed",
        "validator_refused",
        "validator_accepted",
        "internal_error",
    }
)
GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES = ("Time(s)", "TimeStamp")
GUIDED_BACKEND_RWD_UV_SUFFIX_CANDIDATES = ("-410",)
GUIDED_BACKEND_RWD_SIGNAL_SUFFIX_CANDIDATES = ("-470",)


def build_guided_backend_validation_parser_contract(
    *,
    source_format: str = "rwd",
    npm_contract: NpmParserContract | None = None,
) -> RwdHeaderParsingContract | NpmParserContract:
    """Return the explicit application-owned parser contract."""
    if source_format == "npm":
        if not isinstance(npm_contract, NpmParserContract):
            raise TypeError("npm_contract is required for an NPM parser contract.")
        return npm_contract
    return RwdHeaderParsingContract(
        time_column_candidates=GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
        uv_suffix_candidates=GUIDED_BACKEND_RWD_UV_SUFFIX_CANDIDATES,
        signal_suffix_candidates=GUIDED_BACKEND_RWD_SIGNAL_SUFFIX_CANDIDATES,
    )


def build_guided_backend_validator_contract(
) -> GuidedBackendValidatorContract:
    """Return the explicit application-owned validator contract."""
    return GuidedBackendValidatorContract(
        validation_scope=GUIDED_BACKEND_VALIDATION_SCOPE,
        validation_contract_version=(
            GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
        ),
        validator_capability_version=(
            GUIDED_BACKEND_VALIDATOR_CAPABILITY_VERSION
        ),
        supported_subset_rule_version=(
            GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
        ),
    )


@dataclass(frozen=True)
class GuidedBackendValidationGuiContext:
    draft: GuidedNewAnalysisDraftPlan
    parser_contract: RwdHeaderParsingContract | NpmParserContract
    additional_protected_roots: tuple[tuple[str, str], ...]
    validator_contract: GuidedBackendValidatorContract
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.draft, GuidedNewAnalysisDraftPlan):
            raise TypeError("draft must be a GuidedNewAnalysisDraftPlan.")
        if not isinstance(self.parser_contract, (RwdHeaderParsingContract, NpmParserContract)):
            raise TypeError(
                "parser_contract must be an RwdHeaderParsingContract or NpmParserContract."
            )
        if not isinstance(self.additional_protected_roots, tuple):
            raise TypeError("additional_protected_roots must be a tuple.")
        if not all(
            isinstance(item, tuple)
            and len(item) == 2
            and all(isinstance(value, str) and value for value in item)
            for item in self.additional_protected_roots
        ):
            raise ValueError("additional_protected_roots contains an invalid root.")
        if not isinstance(self.validator_contract, GuidedBackendValidatorContract):
            raise TypeError(
                "validator_contract must be a GuidedBackendValidatorContract."
            )
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("revision must be a non-negative integer.")


@dataclass(frozen=True)
class GuidedBackendValidationWorkflowIssue:
    stage: str
    category: str
    section: str
    message: str
    detail_code: str = ""

    def __post_init__(self) -> None:
        for name in ("stage", "category", "section", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.detail_code, str):
            raise ValueError("detail_code must be a string.")


@dataclass(frozen=True)
class GuidedBackendValidationWorkflowOutcome:
    status: str
    accepted_for_backend_validation: bool
    run_authorization: bool
    request_identity: str | None
    validation_result: GuidedBackendValidationResult | None
    compile_result: GuidedBackendValidationCompileSuccess | None
    materialization_result: GuidedBackendValidationMaterializationSuccess | None
    blocking_issues: tuple[GuidedBackendValidationWorkflowIssue, ...]
    user_summary: str
    warning_categories: tuple[str, ...] = ()
    stale: bool = False
    no_files_written: bool = True
    no_directories_created: bool = True
    no_artifacts_created: bool = True
    no_run_id_allocated: bool = True
    no_runner_invoked: bool = True
    # Explicit accepted-outcome linkage for production-boundary consumers.
    # These remain optional for legacy RWD workflow callers that do not carry
    # a GUI revision into this function; the NPM accepted-outcome boundary
    # refuses an outcome when any required linkage is absent.
    accepted_request_identity: str | None = None
    validation_revision: int | None = None
    guided_plan_identity: str | None = None

    def __post_init__(self) -> None:
        if self.status not in GUIDED_BACKEND_VALIDATION_WORKFLOW_STATUSES:
            raise ValueError("Unsupported workflow status.")
        if not isinstance(self.blocking_issues, tuple):
            raise ValueError("blocking_issues must be a tuple.")
        if not all(
            isinstance(item, GuidedBackendValidationWorkflowIssue)
            for item in self.blocking_issues
        ):
            raise ValueError("blocking_issues contains an invalid issue.")
        if not isinstance(self.user_summary, str) or not self.user_summary:
            raise ValueError("user_summary must be a non-empty string.")
        if self.run_authorization is not False:
            raise ValueError("Workflow outcomes cannot authorize a run.")
        if any(
            value is not True
            for value in (
                self.no_files_written,
                self.no_directories_created,
                self.no_artifacts_created,
                self.no_run_id_allocated,
                self.no_runner_invoked,
            )
        ):
            raise ValueError("Workflow outcomes must assert no side effects.")
        accepted = self.status == "validator_accepted"
        if self.accepted_for_backend_validation is not accepted:
            raise ValueError("Workflow acceptance does not match status.")
        if accepted:
            if (
                not isinstance(self.request_identity, str)
                or not self.request_identity
                or (
                    self.accepted_request_identity is not None
                    and self.accepted_request_identity != self.request_identity
                )
                or self.validation_result is None
                or self.validation_result.accepted is not True
                or self.compile_result is None
                or self.materialization_result is None
                or self.blocking_issues
            ):
                raise ValueError("Accepted workflow outcome is incomplete.")
        elif self.status == "cancelled":
            if (
                self.request_identity is not None
                or self.validation_result is not None
                or self.compile_result is not None
                or self.materialization_result is not None
                or self.blocking_issues
            ):
                raise ValueError("Cancelled workflow outcome must be empty.")
        elif not self.blocking_issues:
            raise ValueError("Failed workflow outcome requires an issue.")


def _normalized_issues(
    stage: str,
    issues: tuple[object, ...],
) -> tuple[GuidedBackendValidationWorkflowIssue, ...]:
    return tuple(
        GuidedBackendValidationWorkflowIssue(
            stage=stage,
            category=str(getattr(issue, "category", "") or "unknown_issue"),
            section=str(getattr(issue, "section", "") or stage),
            message=str(getattr(issue, "message", "") or "Validation failed."),
            detail_code=str(getattr(issue, "detail_code", "") or ""),
        )
        for issue in issues
    )


def _cancelled() -> GuidedBackendValidationWorkflowOutcome:
    return GuidedBackendValidationWorkflowOutcome(
        status="cancelled",
        accepted_for_backend_validation=False,
        run_authorization=False,
        request_identity=None,
        validation_result=None,
        compile_result=None,
        materialization_result=None,
        blocking_issues=(),
        user_summary="The Guided setup check was cancelled.",
    )


def _internal_error(stage: str) -> GuidedBackendValidationWorkflowOutcome:
    return GuidedBackendValidationWorkflowOutcome(
        status="internal_error",
        accepted_for_backend_validation=False,
        run_authorization=False,
        request_identity=None,
        validation_result=None,
        compile_result=None,
        materialization_result=None,
        blocking_issues=(
            GuidedBackendValidationWorkflowIssue(
                stage=stage,
                category="workflow_internal_error",
                section="workflow",
                message="The Guided setup check could not complete.",
                detail_code=f"{stage}_exception",
            ),
        ),
        user_summary="The Guided setup check could not complete safely.",
    )


def make_guided_backend_validation_workflow_internal_error(
    stage: str = "context",
) -> GuidedBackendValidationWorkflowOutcome:
    """Return a safe in-memory outcome for an adapter boundary exception."""
    normalized_stage = (
        stage if isinstance(stage, str) and stage else "context"
    )
    return _internal_error(normalized_stage)


def _is_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> bool:
    return bool(cancellation_check and cancellation_check())


def validate_current_guided_draft_for_backend(
    draft: GuidedNewAnalysisDraftPlan,
    *,
    parser_contract: RwdHeaderParsingContract | NpmParserContract,
    additional_protected_roots: tuple[tuple[str, str], ...] = (),
    validator_contract: GuidedBackendValidatorContract,
    cancellation_check: Callable[[], bool] | None = None,
    validation_revision: int | None = None,
) -> GuidedBackendValidationWorkflowOutcome:
    """Materialize, compile, and validate one draft without writes or Run."""
    if validation_revision is not None and (
        isinstance(validation_revision, bool)
        or not isinstance(validation_revision, int)
        or validation_revision < 0
    ):
        return _internal_error("validation_revision")
    if _is_cancelled(cancellation_check):
        return _cancelled()
    try:
        materialized = materialize_guided_backend_validation_facts(
            draft,
            parser_contract=parser_contract,
            cancellation_check=cancellation_check,
            additional_protected_roots=additional_protected_roots,
        )
    except Exception:
        return _internal_error("materialization")
    if isinstance(materialized, GuidedBackendValidationMaterializationFailure):
        return GuidedBackendValidationWorkflowOutcome(
            status="materialization_failed",
            accepted_for_backend_validation=False,
            run_authorization=False,
            request_identity=None,
            validation_result=None,
            compile_result=None,
            materialization_result=None,
            blocking_issues=_normalized_issues(
                "materialization", materialized.blocking_issues
            ),
            user_summary="Guided setup is incomplete or stale.",
        )
    if not isinstance(materialized, GuidedBackendValidationMaterializationSuccess):
        return _internal_error("materialization")
    if _is_cancelled(cancellation_check):
        return _cancelled()

    try:
        compiled = compile_guided_backend_validation_request(
            draft,
            facts=materialized.facts,
            validator_contract=validator_contract,
        )
    except Exception:
        return _internal_error("compile")
    if isinstance(compiled, GuidedBackendValidationCompileFailure):
        return GuidedBackendValidationWorkflowOutcome(
            status="compile_failed",
            accepted_for_backend_validation=False,
            run_authorization=False,
            request_identity=None,
            validation_result=None,
            compile_result=None,
            materialization_result=materialized,
            blocking_issues=_normalized_issues(
                "compile", compiled.blocking_issues
            ),
            user_summary="The Guided request could not be compiled.",
        )
    if not isinstance(compiled, GuidedBackendValidationCompileSuccess):
        return _internal_error("compile")
    if _is_cancelled(cancellation_check):
        return _cancelled()

    try:
        validated = validate_guided_backend_validation_request(
            compiled.request,
            canonical_request_identity=compiled.canonical_request_identity,
            validator_contract=validator_contract,
        )
    except Exception:
        return _internal_error("validator")
    if validated.accepted is not True:
        return GuidedBackendValidationWorkflowOutcome(
            status="validator_refused",
            accepted_for_backend_validation=False,
            run_authorization=False,
            request_identity=compiled.canonical_request_identity,
            validation_result=validated,
            compile_result=compiled,
            materialization_result=materialized,
            blocking_issues=_normalized_issues(
                "validator", validated.blocking_issues
            ),
            user_summary=(
                "The NPM setup check found a problem with the current import "
                "settings."
                if draft.input_format == "npm"
                else "Backend validation refused the Guided request."
            ),
        )
    try:
        guided_plan_identity = compute_guided_new_analysis_draft_plan_identity(draft)
    except Exception:
        return _internal_error("guided_plan_identity")
    return GuidedBackendValidationWorkflowOutcome(
        status="validator_accepted",
        accepted_for_backend_validation=True,
        run_authorization=False,
        request_identity=compiled.canonical_request_identity,
        validation_result=validated,
        compile_result=compiled,
        materialization_result=materialized,
        blocking_issues=(),
        user_summary=(
            "This NPM recording setup was checked successfully. Running NPM "
            "analyses is not available yet."
            if draft.input_format == "npm"
            else "Backend validation accepted the Guided request."
        ),
        warning_categories=tuple(getattr(materialized, "warning_categories", ()) or ()),
        accepted_request_identity=compiled.canonical_request_identity,
        validation_revision=validation_revision,
        guided_plan_identity=guided_plan_identity,
    )
