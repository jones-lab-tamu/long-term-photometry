"""Effectful allocation boundary for an accepted pure Guided startup plan."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from photometry_pipeline.guided_completed_run_rejection_policy import (
    detect_guided_diagnostic_cache_candidate,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_STARTUP_STATUS_FILENAME,
    GuidedStartupPlanResult,
    GuidedStartupTransactionRequest,
    plan_guided_startup_transaction,
)


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUCCESS_TOKENS = frozenset(("success", "complete", "completed", "done"))
_FINAL_TOKENS = frozenset(("final", "complete", "completed", "done"))


@dataclass(frozen=True)
class GuidedStartupAllocationIssue:
    category: str
    section: str
    message: str


@dataclass(frozen=True)
class GuidedStartupAllocationResult:
    status: str
    ok: bool
    allocated: bool
    startup_status_written: bool
    allocated_run_dir: str | None
    startup_status_path: str | None
    startup_status_sha256: str | None
    blocking_issues: tuple[GuidedStartupAllocationIssue, ...]
    startup_transaction_identity: str | None
    no_runner_invoked: bool = True
    no_manifest_written: bool = True
    no_config_written: bool = True
    no_provenance_written: bool = True
    no_command_written: bool = True
    no_gui_mutation: bool = True
    completed_run_claim: bool = False


def _result(
    *,
    status: str,
    ok: bool,
    allocated: bool,
    startup_status_written: bool,
    issue: GuidedStartupAllocationIssue | None,
    allocated_run_dir: str | None = None,
    startup_status_path: str | None = None,
    startup_status_sha256: str | None = None,
    startup_transaction_identity: str | None = None,
) -> GuidedStartupAllocationResult:
    return GuidedStartupAllocationResult(
        status=status,
        ok=ok,
        allocated=allocated,
        startup_status_written=startup_status_written,
        allocated_run_dir=allocated_run_dir,
        startup_status_path=startup_status_path,
        startup_status_sha256=startup_status_sha256,
        blocking_issues=() if issue is None else (issue,),
        startup_transaction_identity=startup_transaction_identity,
    )


def _refused(category: str, section: str, message: str) -> GuidedStartupAllocationResult:
    return _result(
        status="refused_before_allocation",
        ok=False,
        allocated=False,
        startup_status_written=False,
        issue=GuidedStartupAllocationIssue(category, section, message),
    )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _is_successful_completed_run_root(path: Path) -> bool:
    report = _read_json_object(path / "run_report.json")
    if report is not None:
        status_values = (
            report.get("status"),
            report.get("run_status"),
            report.get("final_status"),
            report.get("result"),
        )
        phase_values = (
            report.get("phase"),
            report.get("run_phase"),
            report.get("final_phase"),
        )
        status_ok = any(
            str(value).strip().lower() in _SUCCESS_TOKENS
            for value in status_values
        )
        phases = tuple(
            str(value).strip().lower() for value in phase_values if value is not None
        )
        if status_ok and (not any(phases) or any(x in _FINAL_TOKENS for x in phases)):
            return True
        context = report.get("run_context")
        if isinstance(context, dict):
            status = str(context.get("status", "")).strip().lower()
            phase = str(context.get("phase", "")).strip().lower()
            if status in _SUCCESS_TOKENS and (
                not phase or phase in _FINAL_TOKENS
            ):
                return True
    status = _read_json_object(path / "status.json")
    if status is not None and (
        status.get("schema_version") == 1
        and str(status.get("phase", "")).strip().lower() == "final"
        and str(status.get("status", "")).strip().lower() == "success"
    ):
        return True
    manifest = _read_json_object(path / "MANIFEST.json")
    return bool(
        manifest is not None
        and str(manifest.get("status", "")).strip().lower() in _SUCCESS_TOKENS
    )


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(left)) == os.path.normcase(os.fspath(right))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _protected_output_ancestor(
    output_base: Path,
) -> tuple[str, Path] | None:
    for candidate in (output_base, *output_base.parents):
        if _is_successful_completed_run_root(candidate):
            return "completed_run_root_prohibited", candidate
        if detect_guided_diagnostic_cache_candidate(candidate) is not None:
            return "diagnostic_cache_root_prohibited", candidate
    return None


def _validate_plan(
    request: GuidedStartupTransactionRequest,
    pure_plan: GuidedStartupPlanResult | None,
) -> tuple[GuidedStartupPlanResult | None, GuidedStartupAllocationIssue | None]:
    expected = plan_guided_startup_transaction(request)
    plan = expected if pure_plan is None else pure_plan
    if (
        not isinstance(plan, GuidedStartupPlanResult)
        or plan.status != "planned_non_effectful"
        or plan.ok is not True
        or plan.ready_for_effectful_startup is not True
        or plan.startup_status_bytes is None
        or plan.identities is None
        or plan.no_files_written is not True
        or plan.no_directories_created is not True
        or plan.no_runner_invoked is not True
        or plan.no_gui_mutation is not True
    ):
        return None, GuidedStartupAllocationIssue(
            "pure_plan_not_accepted",
            "pure_plan",
            "An accepted non-effectful startup plan is required.",
        )
    if (
        expected.status != "planned_non_effectful"
        or expected.ok is not True
        or expected.identities is None
        or plan.startup_status_bytes != expected.startup_status_bytes
        or plan.identities.startup_transaction_identity
        != expected.identities.startup_transaction_identity
        or plan.identities.startup_status_bytes_sha256
        != expected.identities.startup_status_bytes_sha256
    ):
        return None, GuidedStartupAllocationIssue(
            "pure_plan_request_mismatch",
            "pure_plan",
            "Startup plan does not match the current request.",
        )
    try:
        status = json.loads(plan.startup_status_bytes)
    except Exception:
        status = None
    if (
        not isinstance(status, dict)
        or status.get("allocated_run_dir") != request.planned_allocated_run_dir
        or status.get("run_id") != request.planned_run_id
        or status.get("completed_run_claim") is not False
        or status.get("runner_started") is not False
    ):
        return None, GuidedStartupAllocationIssue(
            "startup_status_plan_mismatch",
            "pure_plan",
            "Planned startup status is inconsistent.",
        )
    actual_hash = hashlib.sha256(plan.startup_status_bytes).hexdigest()
    if actual_hash != plan.identities.startup_status_bytes_sha256:
        return None, GuidedStartupAllocationIssue(
            "startup_status_identity_mismatch",
            "pure_plan",
            "Planned startup status hash is inconsistent.",
        )
    return plan, None


def allocate_guided_startup_directory(
    *,
    request: GuidedStartupTransactionRequest,
    pure_plan: GuidedStartupPlanResult | None = None,
) -> GuidedStartupAllocationResult:
    """Exclusively allocate one run directory and write its first status file."""
    plan, issue = _validate_plan(request, pure_plan)
    if issue is not None:
        return _refused(issue.category, issue.section, issue.message)
    assert plan is not None and plan.identities is not None

    if not _RUN_ID_RE.fullmatch(request.planned_run_id):
        return _refused(
            "planned_run_id_invalid",
            "paths",
            "Planned run ID contains unsafe characters.",
        )
    output_input = Path(request.output_base_canonical)
    source_input = Path(request.source_root_canonical)
    child_input = Path(request.planned_allocated_run_dir)
    if not output_input.is_absolute() or not source_input.is_absolute() or not child_input.is_absolute():
        return _refused(
            "startup_path_not_absolute",
            "paths",
            "Startup source and output paths must be absolute.",
        )
    if not output_input.exists():
        return _refused(
            "output_base_missing",
            "output",
            "Output base must already exist.",
        )
    if not output_input.is_dir():
        return _refused(
            "output_base_not_directory",
            "output",
            "Output base is not a directory.",
        )
    try:
        output_base = output_input.resolve(strict=True)
        source_root = source_input.resolve(strict=False)
        child = child_input.resolve(strict=False)
    except OSError:
        return _refused(
            "startup_path_resolution_failed",
            "paths",
            "Startup paths could not be safely resolved.",
        )
    if output_input.is_symlink():
        return _refused(
            "output_base_symlink_prohibited",
            "output",
            "Symlink output bases are not supported.",
        )
    if (
        not _same_path(child.parent, output_base)
        or child.name != request.planned_run_id
        or not _is_relative_to(child, output_base)
    ):
        return _refused(
            "planned_child_not_direct",
            "paths",
            "Planned run directory must be one direct child of the output base.",
        )
    if (
        _same_path(output_base, source_root)
        or _is_relative_to(output_base, source_root)
        or _same_path(child, source_root)
        or _is_relative_to(child, source_root)
        or _is_relative_to(source_root, child)
    ):
        return _refused(
            "source_output_overlap",
            "paths",
            "Source and startup output paths overlap.",
        )
    if request.filesystem_policy.overwrite_requested is not False:
        return _refused(
            "overwrite_prohibited", "output", "Startup allocation forbids overwrite."
        )
    protected_ancestor = _protected_output_ancestor(output_base)
    if protected_ancestor is not None:
        category, protected_root = protected_ancestor
        kind = (
            "completed-run"
            if category == "completed_run_root_prohibited"
            else "Guided diagnostic-cache"
        )
        return _refused(
            category,
            "output",
            f"Output base is within a protected {kind} root: {protected_root}",
        )
    if child.exists() or os.path.lexists(child):
        return _refused(
            "planned_child_exists",
            "output",
            "Planned run directory already exists.",
        )

    transaction_identity = plan.identities.startup_transaction_identity
    try:
        child.mkdir(exist_ok=False)
    except OSError as exc:
        return _result(
            status="allocation_failed",
            ok=False,
            allocated=False,
            startup_status_written=False,
            issue=GuidedStartupAllocationIssue(
                "exclusive_allocation_failed",
                "allocation",
                f"Exclusive startup allocation failed: {exc}",
            ),
            startup_transaction_identity=transaction_identity,
        )

    child_text = os.fspath(child)
    status_path = child / GUIDED_STARTUP_STATUS_FILENAME
    try:
        if any(child.iterdir()):
            raise FileExistsError("allocated directory is not empty")
        with status_path.open("xb") as handle:
            handle.write(plan.startup_status_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        written_hash = hashlib.sha256(status_path.read_bytes()).hexdigest()
        if written_hash != plan.identities.startup_status_bytes_sha256:
            raise OSError("written startup status hash does not match the plan")
        if tuple(item.name for item in child.iterdir()) != (
            GUIDED_STARTUP_STATUS_FILENAME,
        ):
            raise OSError("startup status is not the only allocated entry")
    except Exception as exc:
        return _result(
            status="allocated_status_write_failed",
            ok=False,
            allocated=True,
            startup_status_written=False,
            issue=GuidedStartupAllocationIssue(
                "startup_status_write_failed",
                "startup_status",
                f"Startup status could not be written: {exc}",
            ),
            allocated_run_dir=child_text,
            startup_status_path=os.fspath(status_path),
            startup_transaction_identity=transaction_identity,
        )

    return _result(
        status="allocated_startup_status_written",
        ok=True,
        allocated=True,
        startup_status_written=True,
        issue=None,
        allocated_run_dir=child_text,
        startup_status_path=os.fspath(status_path),
        startup_status_sha256=written_hash,
        startup_transaction_identity=transaction_identity,
    )
