"""Materialize deterministic Guided startup preparation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from photometry_pipeline.config import Config
from photometry_pipeline.guided_manifest_verification import (
    load_guided_candidate_manifest,
)
from photometry_pipeline.guided_startup_allocation import (
    GuidedStartupAllocationResult,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_CANDIDATE_MANIFEST_FILENAME,
    GUIDED_COMMAND_RECORD_FILENAME,
    GUIDED_CONFIG_EFFECTIVE_FILENAME,
    GUIDED_STARTUP_PROVENANCE_FILENAME,
    GUIDED_STARTUP_STATUS_FILENAME,
    GuidedStartupPlanResult,
    GuidedStartupTransactionRequest,
    plan_guided_startup_transaction,
)


_TARGET_FILENAMES = (
    GUIDED_CANDIDATE_MANIFEST_FILENAME,
    GUIDED_CONFIG_EFFECTIVE_FILENAME,
    GUIDED_COMMAND_RECORD_FILENAME,
    GUIDED_STARTUP_PROVENANCE_FILENAME,
)
_PRODUCTION_FILENAMES = ("status.json", "MANIFEST.json", "run_report.json")
_PRODUCTION_DIRECTORY_NAMES = (
    "qc",
    "cache",
    "events",
    "figures",
)


@dataclass(frozen=True)
class GuidedStartupMaterializationIssue:
    category: str
    section: str
    message: str


@dataclass(frozen=True)
class GuidedStartupMaterializationResult:
    status: str
    ok: bool
    materialized: bool
    allocated_run_dir: str
    files_written: tuple[str, ...]
    manifest_path: str | None
    config_path: str | None
    provenance_path: str | None
    command_record_path: str | None
    startup_status_path: str | None
    startup_status_updated: bool
    artifact_hashes: tuple[tuple[str, str], ...]
    startup_transaction_identity: str | None
    blocking_issues: tuple[GuidedStartupMaterializationIssue, ...]
    no_runner_invoked: bool = True
    no_wrapper_invoked: bool = True
    no_gui_mutation: bool = True
    no_completed_run_claim: bool = True
    no_production_status_written: bool = True
    no_manifest_json_production_written: bool = True


def _result(
    *,
    status: str,
    ok: bool,
    materialized: bool,
    run_dir: str,
    files_written: tuple[str, ...],
    issue: GuidedStartupMaterializationIssue | None,
    startup_transaction_identity: str | None,
    artifact_hashes: tuple[tuple[str, str], ...] = (),
) -> GuidedStartupMaterializationResult:
    written = frozenset(files_written)
    return GuidedStartupMaterializationResult(
        status=status,
        ok=ok,
        materialized=materialized,
        allocated_run_dir=run_dir,
        files_written=files_written,
        manifest_path=(
            os.path.join(run_dir, GUIDED_CANDIDATE_MANIFEST_FILENAME)
            if GUIDED_CANDIDATE_MANIFEST_FILENAME in written
            else None
        ),
        config_path=(
            os.path.join(run_dir, GUIDED_CONFIG_EFFECTIVE_FILENAME)
            if GUIDED_CONFIG_EFFECTIVE_FILENAME in written
            else None
        ),
        provenance_path=(
            os.path.join(run_dir, GUIDED_STARTUP_PROVENANCE_FILENAME)
            if GUIDED_STARTUP_PROVENANCE_FILENAME in written
            else None
        ),
        command_record_path=(
            os.path.join(run_dir, GUIDED_COMMAND_RECORD_FILENAME)
            if GUIDED_COMMAND_RECORD_FILENAME in written
            else None
        ),
        startup_status_path=(
            os.path.join(run_dir, GUIDED_STARTUP_STATUS_FILENAME)
            if run_dir
            else None
        ),
        startup_status_updated=False,
        artifact_hashes=artifact_hashes,
        startup_transaction_identity=startup_transaction_identity,
        blocking_issues=() if issue is None else (issue,),
    )


def _refused(
    category: str,
    section: str,
    message: str,
    *,
    run_dir: str = "",
) -> GuidedStartupMaterializationResult:
    return _result(
        status="refused_before_materialization",
        ok=False,
        materialized=False,
        run_dir=run_dir,
        files_written=(),
        issue=GuidedStartupMaterializationIssue(category, section, message),
        startup_transaction_identity=None,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _validate_preconditions(
    request: GuidedStartupTransactionRequest,
    pure_plan: GuidedStartupPlanResult,
    allocation_result: GuidedStartupAllocationResult,
) -> tuple[Path | None, GuidedStartupMaterializationIssue | None]:
    recomputed = plan_guided_startup_transaction(request)
    required_bytes = (
        pure_plan.startup_status_bytes,
        pure_plan.candidate_manifest_bytes,
        pure_plan.config_effective_bytes,
        pure_plan.startup_provenance_bytes,
        pure_plan.command_record_bytes,
    )
    if (
        not isinstance(pure_plan, GuidedStartupPlanResult)
        or pure_plan.status != "planned_non_effectful"
        or pure_plan.ok is not True
        or pure_plan.ready_for_effectful_startup is not True
        or any(value is None for value in required_bytes)
        or pure_plan.identities is None
        or pure_plan.command_plan is None
        or pure_plan.no_files_written is not True
        or pure_plan.no_directories_created is not True
        or pure_plan.no_runner_invoked is not True
        or pure_plan.no_gui_mutation is not True
    ):
        return None, GuidedStartupMaterializationIssue(
            "pure_plan_not_accepted",
            "pure_plan",
            "An accepted complete pure startup plan is required.",
        )
    if recomputed != pure_plan:
        return None, GuidedStartupMaterializationIssue(
            "pure_plan_stale_or_tampered",
            "pure_plan",
            "Pure startup plan no longer matches its request.",
        )
    if (
        not isinstance(allocation_result, GuidedStartupAllocationResult)
        or allocation_result.status != "allocated_startup_status_written"
        or allocation_result.ok is not True
        or allocation_result.allocated is not True
        or allocation_result.startup_status_written is not True
        or not allocation_result.allocated_run_dir
        or not allocation_result.startup_status_path
        or allocation_result.startup_status_sha256
        != pure_plan.identities.startup_status_bytes_sha256
        or allocation_result.startup_transaction_identity
        != pure_plan.identities.startup_transaction_identity
        or allocation_result.no_runner_invoked is not True
        or allocation_result.no_manifest_written is not True
        or allocation_result.no_config_written is not True
        or allocation_result.no_provenance_written is not True
        or allocation_result.no_command_written is not True
        or allocation_result.no_gui_mutation is not True
        or allocation_result.completed_run_claim is not False
    ):
        return None, GuidedStartupMaterializationIssue(
            "allocation_not_accepted",
            "allocation",
            "An accepted startup-status allocation is required.",
        )
    run_dir = Path(allocation_result.allocated_run_dir)
    planned_dir = Path(request.planned_allocated_run_dir)
    try:
        if (
            not run_dir.is_dir()
            or run_dir.resolve(strict=True) != planned_dir.resolve(strict=True)
        ):
            raise ValueError
    except (OSError, ValueError):
        return None, GuidedStartupMaterializationIssue(
            "allocated_directory_mismatch",
            "directory",
            "Allocated directory does not match the startup request.",
        )
    status_path = run_dir / GUIDED_STARTUP_STATUS_FILENAME
    if Path(allocation_result.startup_status_path) != status_path:
        return None, GuidedStartupMaterializationIssue(
            "startup_status_path_mismatch",
            "directory",
            "Startup status path is inconsistent.",
        )
    try:
        status_bytes = status_path.read_bytes()
        status = json.loads(status_bytes)
    except Exception:
        return None, GuidedStartupMaterializationIssue(
            "startup_status_missing_or_invalid",
            "directory",
            "Startup status is missing or invalid.",
        )
    if (
        status_bytes != pure_plan.startup_status_bytes
        or not isinstance(status, dict)
        or status.get("run_id") != request.planned_run_id
        or status.get("allocated_run_dir") != request.planned_allocated_run_dir
        or status.get("completed_run_claim") is not False
        or status.get("runner_started") is not False
    ):
        return None, GuidedStartupMaterializationIssue(
            "startup_status_mismatch",
            "directory",
            "Startup status does not match the pure plan.",
        )
    entries = tuple(run_dir.iterdir())
    for name in _PRODUCTION_FILENAMES:
        if (run_dir / name).exists():
            return None, GuidedStartupMaterializationIssue(
                "production_artifact_prohibited",
                "directory",
                f"Production-shaped artifact already exists: {name}",
            )
    for name in _PRODUCTION_DIRECTORY_NAMES:
        if (run_dir / name).exists():
            return None, GuidedStartupMaterializationIssue(
                "production_artifact_prohibited",
                "directory",
                f"Production-shaped directory already exists: {name}",
            )
    for name in _TARGET_FILENAMES:
        if os.path.lexists(run_dir / name):
            return None, GuidedStartupMaterializationIssue(
                "startup_artifact_already_exists",
                "directory",
                f"Startup artifact already exists: {name}",
            )
    if tuple(item.name for item in entries) != (GUIDED_STARTUP_STATUS_FILENAME,):
        return None, GuidedStartupMaterializationIssue(
            "allocated_directory_not_pristine",
            "directory",
            "Allocated directory contains unexpected entries.",
        )
    return run_dir, None


def _write_exclusive(path: Path, content: bytes) -> str:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    written = path.read_bytes()
    if written != content:
        raise OSError(f"Written bytes differ for {path.name}.")
    return hashlib.sha256(written).hexdigest()


def _validate_manifest(path: Path, plan: GuidedStartupPlanResult) -> None:
    loaded = load_guided_candidate_manifest(os.fspath(path))
    if (
        not loaded.accepted
        or loaded.manifest is None
        or loaded.manifest.canonical_candidate_manifest_payload_identity
        != plan.identities.candidate_manifest_payload_identity
    ):
        raise ValueError("Manifest round-trip identity validation failed.")


def _validate_config(
    path: Path,
    request: GuidedStartupTransactionRequest,
) -> None:
    loaded = Config.from_yaml(os.fspath(path))
    payload = request.payload_result.config_payload
    if payload is None:
        raise ValueError("Config payload is missing.")
    for item in payload.values:
        if _plain(getattr(loaded, item.name)) != _plain(item.value):
            raise ValueError(f"Config round-trip mismatch: {item.name}")


def _validate_command(path: Path, plan: GuidedStartupPlanResult) -> None:
    content = path.read_bytes()
    required = (
        b"--guided-candidate-manifest\n",
        b"--guided-preallocated-run-dir\n",
        b"--mode\nphasic\n",
        b"--run-type\nfull\n",
    )
    if any(value not in content for value in required):
        raise ValueError("Command record lacks required future Guided arguments.")
    if b"\ntonic\n" in content or b"\nboth\n" in content:
        raise ValueError("Command record contains a prohibited execution mode.")
    if (
        plan.command_plan is None
        or plan.command_plan.executable_now is not False
        or plan.command_plan.requires_future_wrapper_preallocated_mode is not True
    ):
        raise ValueError("Command plan incorrectly claims current executability.")


def _validate_provenance(path: Path, plan: GuidedStartupPlanResult) -> None:
    try:
        payload = json.loads(path.read_bytes())
    except Exception as exc:
        raise ValueError("Startup provenance is invalid JSON.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("state") != "prepared_runner_not_started"
        or payload.get("runner_started") is not False
        or payload.get("runner_start_uncertain") is not False
        or payload.get("completed_run_claim") is not False
        or payload.get("startup_transaction_identity")
        != plan.identities.startup_transaction_identity
    ):
        raise ValueError("Startup provenance makes an invalid state claim.")


def materialize_guided_startup_artifacts(
    *,
    request: GuidedStartupTransactionRequest,
    pure_plan: GuidedStartupPlanResult,
    allocation_result: GuidedStartupAllocationResult,
) -> GuidedStartupMaterializationResult:
    """Write only the four deterministic startup preparation artifacts."""
    run_dir, issue = _validate_preconditions(request, pure_plan, allocation_result)
    if issue is not None:
        return _refused(
            issue.category,
            issue.section,
            issue.message,
            run_dir=str(allocation_result.allocated_run_dir or ""),
        )
    assert run_dir is not None and pure_plan.identities is not None

    artifacts: tuple[
        tuple[str, bytes, str, Callable[[Path], None]], ...
    ] = (
        (
            GUIDED_CANDIDATE_MANIFEST_FILENAME,
            pure_plan.candidate_manifest_bytes,
            pure_plan.identities.candidate_manifest_bytes_sha256,
            lambda path: _validate_manifest(path, pure_plan),
        ),
        (
            GUIDED_CONFIG_EFFECTIVE_FILENAME,
            pure_plan.config_effective_bytes,
            pure_plan.identities.config_bytes_sha256,
            lambda path: _validate_config(path, request),
        ),
        (
            GUIDED_COMMAND_RECORD_FILENAME,
            pure_plan.command_record_bytes,
            pure_plan.identities.command_record_sha256,
            lambda path: _validate_command(path, pure_plan),
        ),
        (
            GUIDED_STARTUP_PROVENANCE_FILENAME,
            pure_plan.startup_provenance_bytes,
            pure_plan.identities.startup_provenance_bytes_sha256,
            lambda path: _validate_provenance(path, pure_plan),
        ),
    )
    written: list[str] = []
    hashes: list[tuple[str, str]] = []
    for filename, content, expected_hash, validate in artifacts:
        path = run_dir / filename
        try:
            actual_hash = _write_exclusive(path, content)
            written.append(filename)
            hashes.append((filename, actual_hash))
            if actual_hash != expected_hash:
                raise ValueError(f"Byte hash mismatch for {filename}.")
            validate(path)
            if (run_dir / GUIDED_STARTUP_STATUS_FILENAME).read_bytes() != (
                pure_plan.startup_status_bytes
            ):
                raise ValueError("Startup status changed during materialization.")
        except Exception as exc:
            return _result(
                status="materialization_failed_partial",
                ok=False,
                materialized=False,
                run_dir=os.fspath(run_dir),
                files_written=tuple(written),
                issue=GuidedStartupMaterializationIssue(
                    "startup_artifact_materialization_failed",
                    filename,
                    f"Startup artifact materialization failed: {exc}",
                ),
                artifact_hashes=tuple(hashes),
                startup_transaction_identity=(
                    pure_plan.identities.startup_transaction_identity
                ),
            )

    correction = request.authorization_result.production_intent.correction
    if correction.applied_dff_orchestration_enabled:
        strategy_map_payload = {
            "applied_dff_orchestration_enabled": True,
            "production_strategy_map_version": correction.production_strategy_map_version,
            "included_roi_ids": list(request.authorization_result.production_intent.roi_scope.included_roi_ids),
            "per_roi_production_strategy_map": [
                {
                    "roi_id": entry.roi_id,
                    "strategy_family": entry.strategy_family,
                    "dynamic_fit_mode": entry.dynamic_fit_mode,
                    "selected_strategy": entry.selected_strategy,
                    "evidence_source_type": entry.evidence_source_type,
                    "evidence_reference_json": entry.evidence_reference_json,
                    "explicit_user_mark": entry.explicit_user_mark,
                    "current_or_stale": entry.current_or_stale,
                }
                for entry in correction.per_roi_production_strategy_map
            ]
        }
        strategy_map_bytes = json.dumps(strategy_map_payload, indent=2).encode("utf-8")
        try:
            _write_exclusive(run_dir / "guided_correction_strategy_map.json", strategy_map_bytes)
            written.append("guided_correction_strategy_map.json")
        except Exception as exc:
            return _result(
                status="materialization_failed_partial",
                ok=False,
                materialized=False,
                run_dir=os.fspath(run_dir),
                files_written=tuple(written),
                issue=GuidedStartupMaterializationIssue(
                    "startup_artifact_materialization_failed",
                    "guided_correction_strategy_map.json",
                    f"Startup artifact materialization failed: {exc}",
                ),
                artifact_hashes=tuple(hashes),
                startup_transaction_identity=(
                    pure_plan.identities.startup_transaction_identity
                ),
            )

    return _result(
        status="startup_artifacts_materialized",
        ok=True,
        materialized=True,
        run_dir=os.fspath(run_dir),
        files_written=tuple(written),
        issue=None,
        artifact_hashes=tuple(hashes),
        startup_transaction_identity=(
            pure_plan.identities.startup_transaction_identity
        ),
    )
