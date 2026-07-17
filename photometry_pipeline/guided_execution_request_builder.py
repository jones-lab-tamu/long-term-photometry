"""Build a current Guided startup request without writes or execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Callable

from photometry_pipeline.application_build_identity import (
    resolve_application_build_identity,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowOutcome,
)
from photometry_pipeline.guided_completed_run_rejection_policy import (
    detect_guided_diagnostic_cache_candidate,
)
from photometry_pipeline.guided_execution_payloads import (
    GuidedExecutionPayloadDerivationResult,
    build_guided_execution_startup_mapping_contract,
    derive_guided_execution_payloads,
)
from photometry_pipeline.guided_production_mapping import (
    build_guided_production_mapping_contract,
)
from photometry_pipeline.guided_run_authorization import (
    GuidedRunAuthorizationResult,
    authorize_guided_run,
    build_guided_run_authorization_request,
)
from photometry_pipeline.guided_startup_transaction import (
    GuidedStartupFilesystemPolicy,
    GuidedStartupTransactionRequest,
    GuidedWrapperEntrypointIdentity,
)


@dataclass(frozen=True)
class GuidedExecutionRequestBuildIssue:
    category: str
    section: str
    message: str


@dataclass(frozen=True)
class GuidedExecutionRequestBuildResult:
    status: str
    ok: bool
    authorization_result: GuidedRunAuthorizationResult | None
    payload_result: GuidedExecutionPayloadDerivationResult | None
    startup_transaction_request: GuidedStartupTransactionRequest | None
    blocking_issues: tuple[GuidedExecutionRequestBuildIssue, ...]
    current_gui_revision: int
    request_ready: bool
    no_files_written: bool = True
    no_execution_invoked: bool = True


def _refused(
    status: str,
    section: str,
    message: str,
    revision: int,
    extra_issues: tuple[GuidedExecutionRequestBuildIssue, ...] = (),
) -> GuidedExecutionRequestBuildResult:
    return GuidedExecutionRequestBuildResult(
        status=status,
        ok=False,
        authorization_result=None,
        payload_result=None,
        startup_transaction_request=None,
        blocking_issues=(
            GuidedExecutionRequestBuildIssue(status, section, message),
            *extra_issues,
        ),
        current_gui_revision=revision,
        request_ready=False,
    )


def _payload_blocking_issues(
    payload_result: object,
) -> tuple[GuidedExecutionRequestBuildIssue, ...]:
    """Carry a failed payload result's own blocking issues into the builder
    result so the specific refusal reason is not lost. Never raises."""
    issues = getattr(payload_result, "blocking_issues", None) or ()
    preserved: list[GuidedExecutionRequestBuildIssue] = []
    for issue in issues:
        try:
            preserved.append(
                GuidedExecutionRequestBuildIssue(
                    category=str(getattr(issue, "category", "") or ""),
                    section=str(getattr(issue, "section", "") or "payload"),
                    message=str(getattr(issue, "message", "") or ""),
                )
            )
        except Exception:
            continue
    return tuple(preserved)


def _default_run_id(now: datetime) -> str:
    return (
        f"guided_run_{now.strftime('%Y%m%dT%H%M%S%fZ')}_"
        f"{secrets.token_hex(6)}"
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _output_base_creatability(output_base: Path) -> tuple[bool, bool]:
    """Return (exists_or_creatable, is_directory_or_creatable) for output_base.

    output_base is intentionally not created before Guided Run is pressed
    (see the no_directories_created contract flags asserted throughout
    authorization/payload derivation), so it will not exist yet for the
    standard "new analysis" case. A not-yet-created path is creatable if its
    immediate parent already exists as a writable directory -- matching the
    single-level contract already enforced when the output destination was
    selected (gui/main_window.py _validate_guided_new_analysis_output_
    policy_path / validate_output_write_safety's target_parent_missing
    check) and the single-level directory creation the startup-allocation
    layer performs once this check has passed.
    """
    if output_base.exists():
        is_dir = output_base.is_dir()
        return is_dir, is_dir
    parent = output_base.parent
    if not parent.is_dir():
        return False, False
    try:
        writable = os.access(str(parent), os.W_OK)
    except OSError:
        writable = False
    return writable, writable


def _is_successful_completed_run_root(path: Path) -> bool:
    def read_object(filename: str) -> dict | None:
        try:
            value = json.loads((path / filename).read_text(encoding="utf-8"))
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    status = read_object("status.json")
    if status is not None and (
        status.get("schema_version") == 1
        and str(status.get("phase", "")).strip().lower() == "final"
        and str(status.get("status", "")).strip().lower() == "success"
    ):
        return True
    report = read_object("run_report.json")
    if report is not None:
        state = str(
            report.get("status", report.get("run_status", ""))
        ).strip().lower()
        phase = str(
            report.get("phase", report.get("run_phase", ""))
        ).strip().lower()
        if state in {"success", "complete", "completed", "done"} and (
            not phase or phase in {"final", "complete", "completed", "done"}
        ):
            return True
    manifest = read_object("MANIFEST.json")
    return bool(
        manifest is not None
        and str(manifest.get("status", "")).strip().lower()
        in {"success", "complete", "completed", "done"}
    )


def build_guided_startup_request_from_validation(
    *,
    validation_context: GuidedBackendValidationGuiContext,
    validation_outcome: GuidedBackendValidationWorkflowOutcome,
    current_gui_revision: int,
    project_root: Path | str | None = None,
    current_time_utc: datetime | None = None,
    token_factory: Callable[[], str] | None = None,
    run_id_factory: Callable[[datetime], str] | None = None,
) -> GuidedExecutionRequestBuildResult:
    """Build the exact authorization/payload/startup bundle in memory."""
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
    try:
        mapping_contract = build_guided_production_mapping_contract()
        authorization_request = build_guided_run_authorization_request(
            stored_validation_outcome=validation_outcome,
            stored_validation_outcome_revision=current_gui_revision,
            current_gui_revision=current_gui_revision,
            current_validation_context=validation_context,
            application_build_identity=build_result.build_identity,
            production_mapping_contract=mapping_contract,
        )
        authorization_result = authorize_guided_run(authorization_request)
    except Exception:
        authorization_result = None
    if (
        not isinstance(authorization_result, GuidedRunAuthorizationResult)
        or authorization_result.status != "authorized"
        or authorization_result.authorized is not True
        or authorization_result.run_authorization is not True
        or authorization_result.production_intent is None
    ):
        return _refused(
            "authorization_failed",
            "authorization",
            "Guided execution authorization failed.",
            current_gui_revision,
        )
    try:
        startup_mapping_contract = (
            build_guided_execution_startup_mapping_contract()
        )
        payload_result = derive_guided_execution_payloads(
            authorization_result,
            startup_mapping_contract=startup_mapping_contract,
        )
    except Exception:
        payload_result = None
    if (
        not isinstance(
            payload_result, GuidedExecutionPayloadDerivationResult
        )
        or payload_result.ok is not True
    ):
        # Preserve the payload layer's own specific blocking issues (e.g. an
        # invalid saved Feature Detection profile) so the GUI can show a
        # truthful scientist-facing reason instead of collapsing every
        # payload failure into a single opaque message.
        return _refused(
            "payload_derivation_failed",
            "payload",
            "Guided execution payload derivation failed.",
            current_gui_revision,
            extra_issues=_payload_blocking_issues(payload_result),
        )

    try:
        intent = authorization_result.production_intent
        source_root_canonical = intent.input_source.source_root_canonical
        output_base_canonical = intent.output_policy.output_base_canonical
        source_root = Path(source_root_canonical).resolve(strict=False)
        output_base = Path(output_base_canonical).resolve(strict=False)
        now = current_time_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("current_time_utc must be timezone-aware")
        now = now.astimezone(timezone.utc)
        run_id = (
            run_id_factory(now)
            if run_id_factory is not None
            else _default_run_id(now)
        )
        one_shot_token = (
            token_factory()
            if token_factory is not None
            else secrets.token_urlsafe(32)
        )
        planned_run_dir = os.path.join(output_base_canonical, run_id)
        run_dir = Path(planned_run_dir).resolve(strict=False)
        wrapper_path = (
            root / "tools" / "run_full_pipeline_deliverables.py"
        ).resolve(strict=True)
        wrapper_digest = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
        output_exists_or_creatable, output_is_dir_or_creatable = (
            _output_base_creatability(output_base)
        )
        output_is_dir = output_base.is_dir() if output_base.exists() else False
        overlap = (
            output_base == source_root
            or _is_relative_to(output_base, source_root)
            or _is_relative_to(source_root, output_base)
        )
        filesystem_policy = GuidedStartupFilesystemPolicy(
            output_base_exists_or_creatable=output_exists_or_creatable,
            output_base_is_directory_or_creatable=output_is_dir_or_creatable,
            output_base_overlaps_source=overlap,
            output_base_is_completed_run_root=(
                _is_successful_completed_run_root(output_base)
                if output_is_dir
                else False
            ),
            output_base_is_guided_diagnostic_cache_root=(
                detect_guided_diagnostic_cache_candidate(output_base)
                is not None
                if output_is_dir
                else False
            ),
            output_base_is_protected_ineligible_root=False,
            planned_child_directly_under_base=run_dir.parent == output_base,
            planned_child_already_exists=os.path.lexists(run_dir),
            overwrite_requested=intent.output_policy.overwrite,
            protected_root_context_complete=(
                intent.output_policy.protected_root_context_complete
            ),
        )
        request = GuidedStartupTransactionRequest(
            authorization_result=authorization_result,
            payload_result=payload_result,
            startup_mapping_contract=startup_mapping_contract,
            application_build_identity=build_result.build_identity,
            current_guided_revision=current_gui_revision,
            explicit_user_run_transition=True,
            output_base_canonical=output_base_canonical,
            source_root_canonical=source_root_canonical,
            planned_run_id=run_id,
            planned_allocated_run_dir=planned_run_dir,
            wrapper_entrypoint=GuidedWrapperEntrypointIdentity(
                entrypoint_kind="script_path",
                entrypoint_value=os.fspath(wrapper_path),
                trusted_application_root=os.fspath(root),
                wrapper_identity_digest=wrapper_digest,
                supported_contract_version=(
                    "run_full_pipeline_deliverables.v1"
                ),
                supports_guided_preallocated_run_dir=True,
                supports_guided_candidate_manifest=True,
                trusted_entrypoint=True,
                python_executable=sys.executable,
            ),
            one_shot_consumption_token=one_shot_token,
            one_shot_token_current=True,
            one_shot_token_unused=True,
            current_time_utc_iso=now.isoformat(),
            filesystem_policy=filesystem_policy,
        )
    except Exception:
        return _refused(
            "startup_request_invalid",
            "startup",
            "Guided startup request could not be constructed.",
            current_gui_revision,
        )
    return GuidedExecutionRequestBuildResult(
        status="built",
        ok=True,
        authorization_result=authorization_result,
        payload_result=payload_result,
        startup_transaction_request=request,
        blocking_issues=(),
        current_gui_revision=current_gui_revision,
        request_ready=True,
    )
