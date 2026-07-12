"""Internal orchestration from Guided startup planning to wrapper invocation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

from photometry_pipeline.guided_startup_allocation import (
    GuidedStartupAllocationResult,
    allocate_guided_startup_directory,
)
from photometry_pipeline.guided_startup_materialization import (
    GuidedStartupMaterializationResult,
    materialize_guided_startup_artifacts,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_CANDIDATE_MANIFEST_FILENAME,
    GUIDED_COMMAND_RECORD_FILENAME,
    GUIDED_CONFIG_EFFECTIVE_FILENAME,
    GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    GUIDED_PER_ROI_CORRECTION_FILENAME,
    GUIDED_STARTUP_PROVENANCE_FILENAME,
    GUIDED_STARTUP_STATUS_FILENAME,
    GuidedStartupPlanResult,
    GuidedStartupTransactionRequest,
    plan_guided_startup_transaction,
)


GUIDED_STARTUP_ORCHESTRATION_FAILURE_FILENAME = (
    "guided_startup_orchestration_failure.json"
)
GUIDED_STARTUP_ORCHESTRATION_FAILURE_SCHEMA_NAME = (
    "guided_startup_orchestration_failure"
)
GUIDED_STARTUP_ORCHESTRATION_FAILURE_SCHEMA_VERSION = "v1"

_MATERIALIZED_FILENAMES = frozenset(
    (
        GUIDED_STARTUP_STATUS_FILENAME,
        GUIDED_CANDIDATE_MANIFEST_FILENAME,
        GUIDED_CONFIG_EFFECTIVE_FILENAME,
        GUIDED_COMMAND_RECORD_FILENAME,
        GUIDED_STARTUP_PROVENANCE_FILENAME,
    )
)
# Optional artifacts materialize_guided_startup_artifacts writes in addition
# to the required set above, conditioned on production_intent content (e.g.
# guided_correction_strategy_map.json for a per-ROI correction strategy map,
# GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME for a per-ROI feature/event map).
# Their absence is never required; their presence alone must not refuse
# materialization.
_OPTIONAL_MATERIALIZED_FILENAMES = frozenset(
    (
        "guided_correction_strategy_map.json",
        GUIDED_PER_ROI_CORRECTION_FILENAME,
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    )
)
_PROHIBITED_ARGUMENTS = frozenset(
    (
        "--out-base",
        "--overwrite",
        "tonic",
        "both",
        "--preview-first-n",
        "--discover",
        "--validate-only",
        "--traces-only",
        "--include-rois",
        "--exclude-rois",
    )
)


@dataclass(frozen=True)
class GuidedStartupOrchestrationIssue:
    category: str
    section: str
    message: str


@dataclass(frozen=True)
class GuidedWrapperProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    command: tuple[str, ...]
    started: bool
    completed: bool


@dataclass(frozen=True)
class GuidedStartupOrchestrationResult:
    status: str
    ok: bool
    wrapper_started: bool
    wrapper_completed: bool
    wrapper_returncode: int | None
    allocated_run_dir: str | None
    startup_transaction_identity: str | None
    pure_plan_status: str | None
    allocation_status: str | None
    materialization_status: str | None
    wrapper_command: tuple[str, ...] | None
    blocking_issues: tuple[GuidedStartupOrchestrationIssue, ...]
    failure_marker_written: bool
    failure_marker_path: str | None
    completed_run_claim: bool = False
    gui_run_enabled: bool = False
    user_facing_manifest_workflow: bool = False


def _result(
    *,
    status: str,
    ok: bool,
    issue: GuidedStartupOrchestrationIssue | None,
    plan: GuidedStartupPlanResult | None = None,
    allocation: GuidedStartupAllocationResult | None = None,
    materialization: GuidedStartupMaterializationResult | None = None,
    process: GuidedWrapperProcessResult | None = None,
    command: tuple[str, ...] | None = None,
    failure_marker_path: str | None = None,
) -> GuidedStartupOrchestrationResult:
    identity = None
    if plan is not None and plan.identities is not None:
        identity = plan.identities.startup_transaction_identity
    return GuidedStartupOrchestrationResult(
        status=status,
        ok=ok,
        wrapper_started=bool(process and process.started),
        wrapper_completed=bool(process and process.completed),
        wrapper_returncode=process.returncode if process else None,
        allocated_run_dir=(
            allocation.allocated_run_dir if allocation is not None else None
        ),
        startup_transaction_identity=identity,
        pure_plan_status=plan.status if plan is not None else None,
        allocation_status=allocation.status if allocation is not None else None,
        materialization_status=(
            materialization.status if materialization is not None else None
        ),
        wrapper_command=command,
        blocking_issues=() if issue is None else (issue,),
        failure_marker_written=failure_marker_path is not None,
        failure_marker_path=failure_marker_path,
    )


def _argument_value(argv: tuple[str, ...], flag: str) -> str | None:
    positions = tuple(index for index, value in enumerate(argv) if value == flag)
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return None
    return argv[positions[0] + 1]


def _same_path(left: str, right: str) -> bool:
    try:
        left_path = Path(left).resolve(strict=True)
        right_path = Path(right).resolve(strict=True)
    except OSError:
        return False
    return os.path.normcase(os.fspath(left_path)) == os.path.normcase(
        os.fspath(right_path)
    )


def _validate_plan(
    request: GuidedStartupTransactionRequest,
    supplied: GuidedStartupPlanResult | None,
) -> tuple[GuidedStartupPlanResult | None, GuidedStartupOrchestrationIssue | None]:
    recomputed = plan_guided_startup_transaction(request)
    plan = recomputed if supplied is None else supplied
    if supplied is not None and supplied != recomputed:
        return None, GuidedStartupOrchestrationIssue(
            "pure_plan_stale_or_tampered",
            "pure_plan",
            "Supplied startup plan does not match the current request.",
        )
    command = plan.command_plan
    if (
        plan.status != "planned_non_effectful"
        or plan.ok is not True
        or plan.ready_for_effectful_startup is not True
        or command is None
        or command.executable_now is not False
        or command.requires_future_wrapper_preallocated_mode is not True
        or request.payload_result.runner_request is not None
    ):
        if plan.blocking_issues and plan.blocking_issues[0].category in (
            "output_base_unavailable",
            "output_base_not_directory",
        ):
            return None, GuidedStartupOrchestrationIssue(
                "pure_plan_output_not_creatable",
                "output",
                "The selected output folder could not be found or created.",
            )
        return None, GuidedStartupOrchestrationIssue(
            "pure_plan_not_accepted",
            "pure_plan",
            "Accepted startup-limited planning is required.",
        )
    argv = command.argv
    required = (
        ("--input", request.source_root_canonical),
        ("--out", request.planned_allocated_run_dir),
        (
            "--config",
            os.path.join(
                request.planned_allocated_run_dir,
                GUIDED_CONFIG_EFFECTIVE_FILENAME,
            ),
        ),
        (
            "--guided-candidate-manifest",
            os.path.join(
                request.planned_allocated_run_dir,
                GUIDED_CANDIDATE_MANIFEST_FILENAME,
            ),
        ),
        ("--mode", "phasic"),
        ("--run-type", "full"),
    )
    for flag, expected in required:
        value = _argument_value(argv, flag)
        if value is None:
            return None, GuidedStartupOrchestrationIssue(
                "wrapper_command_invalid",
                "command",
                f"Planned wrapper command lacks {flag}.",
            )
        if flag in {
            "--input",
            "--out",
            "--config",
            "--guided-candidate-manifest",
        }:
            # Paths need not exist before allocation, so compare normalized
            # absolute forms without resolving.
            if os.path.normcase(os.path.abspath(value)) != os.path.normcase(
                os.path.abspath(expected)
            ):
                return None, GuidedStartupOrchestrationIssue(
                    "wrapper_command_invalid",
                    "command",
                    f"Planned wrapper command mismatches {flag}.",
                )
        elif value != expected:
            return None, GuidedStartupOrchestrationIssue(
                "wrapper_command_invalid",
                "command",
                f"Planned wrapper command mismatches {flag}.",
            )
    if (
        "--guided-preallocated-run-dir" not in argv
        or any(item in argv for item in _PROHIBITED_ARGUMENTS)
    ):
        return None, GuidedStartupOrchestrationIssue(
            "wrapper_command_invalid",
            "command",
            "Planned wrapper command contains unsupported execution state.",
        )
    return plan, None


def _validate_allocation(
    allocation: GuidedStartupAllocationResult,
    plan: GuidedStartupPlanResult,
    request: GuidedStartupTransactionRequest,
) -> GuidedStartupOrchestrationIssue | None:
    if (
        not isinstance(allocation, GuidedStartupAllocationResult)
        or allocation.status != "allocated_startup_status_written"
        or allocation.ok is not True
        or allocation.allocated is not True
        or allocation.startup_status_written is not True
        or allocation.allocated_run_dir is None
        or allocation.startup_status_sha256
        != plan.identities.startup_status_bytes_sha256
        or allocation.startup_transaction_identity
        != plan.identities.startup_transaction_identity
        or os.path.normcase(os.path.abspath(allocation.allocated_run_dir))
        != os.path.normcase(os.path.abspath(request.planned_allocated_run_dir))
    ):
        return GuidedStartupOrchestrationIssue(
            "allocation_not_accepted",
            "allocation",
            "Accepted startup allocation does not match the plan.",
        )
    status_path = Path(allocation.allocated_run_dir) / GUIDED_STARTUP_STATUS_FILENAME
    try:
        if status_path.read_bytes() != plan.startup_status_bytes:
            raise ValueError
    except (OSError, ValueError):
        return GuidedStartupOrchestrationIssue(
            "allocation_not_accepted",
            "allocation",
            "Allocated startup status does not match the plan.",
        )
    return None


def _validate_materialization(
    materialized: GuidedStartupMaterializationResult,
    plan: GuidedStartupPlanResult,
    allocation: GuidedStartupAllocationResult,
    request: GuidedStartupTransactionRequest,
) -> GuidedStartupOrchestrationIssue | None:
    if (
        not isinstance(materialized, GuidedStartupMaterializationResult)
        or materialized.status != "startup_artifacts_materialized"
        or materialized.ok is not True
        or materialized.materialized is not True
        or materialized.allocated_run_dir != allocation.allocated_run_dir
        or materialized.startup_transaction_identity
        != plan.identities.startup_transaction_identity
        or materialized.startup_status_updated is not False
        or materialized.no_runner_invoked is not True
        or materialized.no_wrapper_invoked is not True
        or materialized.no_completed_run_claim is not True
    ):
        return GuidedStartupOrchestrationIssue(
            "materialization_not_accepted",
            "materialization",
            "Accepted startup materialization is required.",
        )
    run_dir = Path(materialized.allocated_run_dir)
    expected_bytes = {
        GUIDED_STARTUP_STATUS_FILENAME: plan.startup_status_bytes,
        GUIDED_CANDIDATE_MANIFEST_FILENAME: plan.candidate_manifest_bytes,
        GUIDED_CONFIG_EFFECTIVE_FILENAME: plan.config_effective_bytes,
        GUIDED_COMMAND_RECORD_FILENAME: plan.command_record_bytes,
        GUIDED_STARTUP_PROVENANCE_FILENAME: plan.startup_provenance_bytes,
    }
    try:
        names = frozenset(item.name for item in run_dir.iterdir())
        correction = request.authorization_result.production_intent.correction
        native_current = bool(correction.production_strategy_map_version)
        positive_legacy = bool(
            not correction.production_strategy_map_version
            and not correction.per_roi_production_strategy_map
        )
        if (
            not _MATERIALIZED_FILENAMES.issubset(names)
            or names - _MATERIALIZED_FILENAMES - _OPTIONAL_MATERIALIZED_FILENAMES
            or (
                native_current
                and (
                    GUIDED_PER_ROI_CORRECTION_FILENAME not in names
                    or "guided_correction_strategy_map.json" in names
                )
            )
            or (
                positive_legacy
                and GUIDED_PER_ROI_CORRECTION_FILENAME in names
            )
            or (not native_current and not positive_legacy)
        ):
            raise ValueError
        for filename, content in expected_bytes.items():
            if (run_dir / filename).read_bytes() != content:
                raise ValueError
    except (OSError, ValueError):
        return GuidedStartupOrchestrationIssue(
            "materialization_not_accepted",
            "materialization",
            "Materialized startup artifacts do not match the plan.",
        )
    if hashlib.sha256(plan.command_record_bytes).hexdigest() != (
        plan.identities.command_record_sha256
    ):
        return GuidedStartupOrchestrationIssue(
            "materialization_not_accepted",
            "materialization",
            "Materialized command hash is inconsistent.",
        )
    return None


def _default_subprocess_runner(
    command: tuple[str, ...],
) -> GuidedWrapperProcessResult:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return GuidedWrapperProcessResult(
            returncode=None,
            stdout="",
            stderr=str(exc),
            command=command,
            started=False,
            completed=False,
        )
    return GuidedWrapperProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=command,
        started=True,
        completed=True,
    )


def _write_start_failure_marker(
    *,
    run_dir: str,
    transaction_identity: str,
    category: str,
    runner_started: bool,
) -> str | None:
    if (Path(run_dir) / "status.json").exists():
        return None
    path = Path(run_dir) / GUIDED_STARTUP_ORCHESTRATION_FAILURE_FILENAME
    payload = {
        "schema_name": GUIDED_STARTUP_ORCHESTRATION_FAILURE_SCHEMA_NAME,
        "schema_version": GUIDED_STARTUP_ORCHESTRATION_FAILURE_SCHEMA_VERSION,
        "failure_category": category,
        "startup_transaction_identity": transaction_identity,
        "runner_started": runner_started,
        "completed_run_claim": False,
    }
    content = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return None
    return os.fspath(path)


def run_guided_startup_to_wrapper(
    *,
    request: GuidedStartupTransactionRequest,
    pure_plan: GuidedStartupPlanResult | None = None,
    allocation_result: GuidedStartupAllocationResult | None = None,
    materialization_result: GuidedStartupMaterializationResult | None = None,
    subprocess_runner: (
        Callable[[tuple[str, ...]], GuidedWrapperProcessResult] | None
    ) = None,
) -> GuidedStartupOrchestrationResult:
    """Prepare and invoke the internal preallocated Guided wrapper command."""
    plan, issue = _validate_plan(request, pure_plan)
    if issue is not None:
        return _result(status="refused_before_allocation", ok=False, issue=issue)
    assert plan is not None and plan.identities is not None

    allocated = allocation_result
    if allocated is None:
        allocated = allocate_guided_startup_directory(
            request=request, pure_plan=plan
        )
    allocation_issue = _validate_allocation(allocated, plan, request)
    if allocation_issue is not None:
        orchestration_status = {
            "refused_before_allocation": "refused_before_allocation",
            "allocation_failed": "allocation_failed",
            "allocated_status_write_failed": "startup_status_write_failed",
        }.get(allocated.status, "allocation_failed")
        return _result(
            status=orchestration_status,
            ok=False,
            issue=allocation_issue,
            plan=plan,
            allocation=allocated,
        )

    materialized = materialization_result
    if materialized is None:
        materialized = materialize_guided_startup_artifacts(
            request=request,
            pure_plan=plan,
            allocation_result=allocated,
        )
    materialization_issue = _validate_materialization(
        materialized, plan, allocated, request
    )
    if materialization_issue is not None:
        return _result(
            status="materialization_failed",
            ok=False,
            issue=materialization_issue,
            plan=plan,
            allocation=allocated,
            materialization=materialized,
        )

    command = plan.command_plan.argv
    command_path = (
        Path(allocated.allocated_run_dir) / GUIDED_COMMAND_RECORD_FILENAME
    )
    try:
        recorded = tuple(command_path.read_text(encoding="utf-8").splitlines())
    except OSError:
        recorded = ()
    if recorded != command:
        return _result(
            status="refused_before_wrapper",
            ok=False,
            issue=GuidedStartupOrchestrationIssue(
                "wrapper_command_materialization_mismatch",
                "command",
                "Materialized command does not equal the planned wrapper command.",
            ),
            plan=plan,
            allocation=allocated,
            materialization=materialized,
            command=command,
        )

    runner = subprocess_runner or _default_subprocess_runner
    try:
        process = runner(command)
    except Exception as exc:
        process = GuidedWrapperProcessResult(
            returncode=None,
            stdout="",
            stderr=str(exc),
            command=command,
            started=False,
            completed=False,
        )
    if (
        not isinstance(process, GuidedWrapperProcessResult)
        or process.command != command
    ):
        process = GuidedWrapperProcessResult(
            returncode=None,
            stdout="",
            stderr="Runner returned an invalid process result.",
            command=command,
            started=False,
            completed=False,
        )
    if not process.started:
        marker = _write_start_failure_marker(
            run_dir=allocated.allocated_run_dir,
            transaction_identity=plan.identities.startup_transaction_identity,
            category="wrapper_start_failed",
            runner_started=False,
        )
        return _result(
            status="wrapper_start_failed",
            ok=False,
            issue=GuidedStartupOrchestrationIssue(
                "wrapper_start_failed",
                "wrapper",
                process.stderr or "Wrapper process did not start.",
            ),
            plan=plan,
            allocation=allocated,
            materialization=materialized,
            process=process,
            command=command,
            failure_marker_path=marker,
        )
    if not process.completed:
        return _result(
            status="wrapper_started",
            ok=True,
            issue=None,
            plan=plan,
            allocation=allocated,
            materialization=materialized,
            process=process,
            command=command,
        )
    if process.completed and process.returncode == 0:
        return _result(
            status="wrapper_completed",
            ok=True,
            issue=None,
            plan=plan,
            allocation=allocated,
            materialization=materialized,
            process=process,
            command=command,
        )
    return _result(
        status="wrapper_failed",
        ok=False,
        issue=GuidedStartupOrchestrationIssue(
            "wrapper_returned_nonzero",
            "wrapper",
            process.stderr or "Wrapper process returned a failure result.",
        ),
        plan=plan,
        allocation=allocated,
        materialization=materialized,
        process=process,
        command=command,
    )
