"""Validate and exclusively claim a prepared Guided startup directory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from photometry_pipeline.config import Config
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_manifest_verification import (
    load_guided_candidate_manifest,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_CANDIDATE_MANIFEST_FILENAME,
    GUIDED_COMMAND_RECORD_FILENAME,
    GUIDED_CONFIG_EFFECTIVE_FILENAME,
    GUIDED_PER_ROI_CORRECTION_FILENAME,
    GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
    GUIDED_STARTUP_PROVENANCE_FILENAME,
    GUIDED_STARTUP_STATUS_FILENAME,
    GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION,
    LEGACY_GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION,
)


GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME = "guided_startup_wrapper_claim.json"
GUIDED_STARTUP_WRAPPER_CLAIM_SCHEMA_NAME = "guided_startup_wrapper_claim"
GUIDED_STARTUP_WRAPPER_CLAIM_SCHEMA_VERSION = "v1"
GUIDED_STARTUP_WRAPPER_CLAIM_CONTRACT_VERSION = (
    "guided_startup_wrapper_claim.4J14r.v1"
)

_PREPARED_FILENAMES = frozenset(
    (
        GUIDED_STARTUP_STATUS_FILENAME,
        GUIDED_CANDIDATE_MANIFEST_FILENAME,
        GUIDED_CONFIG_EFFECTIVE_FILENAME,
        GUIDED_STARTUP_PROVENANCE_FILENAME,
        GUIDED_COMMAND_RECORD_FILENAME,
    )
)


@dataclass(frozen=True)
class GuidedStartupClaimIssue:
    category: str
    section: str
    message: str


@dataclass(frozen=True)
class GuidedStartupClaimValidation:
    accepted: bool
    run_dir: str | None
    startup_transaction_identity: str | None
    blocking_issues: tuple[GuidedStartupClaimIssue, ...]


@dataclass(frozen=True)
class GuidedStartupClaimResult:
    status: str
    claimed: bool
    claim_path: str | None
    startup_transaction_identity: str | None
    blocking_issues: tuple[GuidedStartupClaimIssue, ...]
    no_production_artifacts_written: bool = True
    completed_run_claim: bool = False


def _refused(category: str, section: str, message: str) -> GuidedStartupClaimValidation:
    return GuidedStartupClaimValidation(
        accepted=False,
        run_dir=None,
        startup_transaction_identity=None,
        blocking_issues=(GuidedStartupClaimIssue(category, section, message),),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_bytes())
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    try:
        left_path = Path(left).resolve(strict=True)
        right_path = Path(right).resolve(strict=True)
    except OSError:
        return False
    return os.path.normcase(os.fspath(left_path)) == os.path.normcase(
        os.fspath(right_path)
    )


def _command_identity(argv: tuple[str, ...], wrapper_identity: str) -> str:
    value = {
        "schema_name": "guided_startup_command",
        "schema_version": "v1",
        "argv": argv,
        "wrapper_identity": wrapper_identity,
    }
    return hashlib.sha256(
        b"guided-startup-command:v1"
        + b"\x00"
        + encode_canonical_value(value)
    ).hexdigest()


def _argument_value(argv: tuple[str, ...], flag: str) -> str | None:
    positions = tuple(index for index, value in enumerate(argv) if value == flag)
    if len(positions) != 1:
        return None
    index = positions[0]
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def validate_guided_preallocated_startup(
    *,
    input_dir: str,
    output_dir: str,
    config_path: str,
    manifest_path: str,
) -> GuidedStartupClaimValidation:
    """Validate a fully prepared startup directory without writing."""
    run_dir = Path(output_dir)
    if not run_dir.is_dir() or run_dir.is_symlink():
        return _refused(
            "preallocated_directory_invalid",
            "directory",
            "Guided preallocated output must be an existing directory.",
        )
    try:
        resolved_run_dir = run_dir.resolve(strict=True)
    except OSError:
        return _refused(
            "preallocated_directory_invalid",
            "directory",
            "Guided preallocated output could not be resolved.",
        )
    expected_manifest = resolved_run_dir / GUIDED_CANDIDATE_MANIFEST_FILENAME
    expected_config = resolved_run_dir / GUIDED_CONFIG_EFFECTIVE_FILENAME
    if not _same_path(manifest_path, expected_manifest):
        return _refused(
            "manifest_path_mismatch",
            "paths",
            "Guided manifest must be the prepared manifest inside the run directory.",
        )
    if not _same_path(config_path, expected_config):
        return _refused(
            "config_path_mismatch",
            "paths",
            "Guided config must be the prepared config inside the run directory.",
        )
    names = frozenset(item.name for item in resolved_run_dir.iterdir())
    claim_path = resolved_run_dir / GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    if os.path.lexists(claim_path):
        return _refused(
            "startup_already_claimed",
            "claim",
            "Guided startup directory has already been claimed.",
        )
    provenance = _read_json(
        resolved_run_dir / GUIDED_STARTUP_PROVENANCE_FILENAME
    )
    optional = frozenset((
        GUIDED_PER_ROI_CORRECTION_FILENAME,
        GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME,
        "guided_correction_strategy_map.json",
    ))
    startup_contract = (
        provenance.get("startup_contract_version")
        if isinstance(provenance, dict) else None
    )
    native_required = startup_contract == GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
    positive_legacy = startup_contract == LEGACY_GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
    if (
        not _PREPARED_FILENAMES.issubset(names)
        or names - _PREPARED_FILENAMES - optional
        or (native_required and GUIDED_PER_ROI_CORRECTION_FILENAME not in names)
        or (not native_required and not positive_legacy)
        or (native_required and "guided_correction_strategy_map.json" in names)
        or (positive_legacy and GUIDED_PER_ROI_CORRECTION_FILENAME in names)
    ):
        return _refused(
            "prepared_directory_dirty_or_incomplete",
            "directory",
            "Prepared startup directory is incomplete or contains unexpected entries.",
        )
    if any((resolved_run_dir / name).is_symlink() for name in names):
        return _refused(
            "prepared_artifact_symlink_prohibited",
            "directory",
            "Prepared startup artifacts must be regular in-directory files.",
        )

    status = _read_json(resolved_run_dir / GUIDED_STARTUP_STATUS_FILENAME)
    if (
        status is None
        or status.get("schema_name") != "guided_startup_status"
        or status.get("completed_run_claim") is not False
        or status.get("runner_started") is not False
        or status.get("runner_start_uncertain") is not False
        or not status.get("authorization_identity")
        or not status.get("config_payload_identity")
        or not status.get("candidate_manifest_payload_identity")
        or not _same_path(status.get("allocated_run_dir", ""), resolved_run_dir)
        or not _same_path(status.get("source_root", ""), input_dir)
    ):
        return _refused(
            "startup_status_invalid",
            "startup_status",
            "Guided startup status is invalid or mismatched.",
        )
    if (
        provenance is None
        or provenance.get("state") != "prepared_runner_not_started"
        or provenance.get("runner_started") is not False
        or provenance.get("runner_start_uncertain") is not False
        or provenance.get("completed_run_claim") is not False
        or not provenance.get("startup_transaction_identity")
        or not _same_path(provenance.get("allocated_run_dir", ""), resolved_run_dir)
        or not _same_path(provenance.get("source_root", ""), input_dir)
        or provenance.get("authorization_identity")
        != status.get("authorization_identity")
        or provenance.get("config_payload_identity")
        != status.get("config_payload_identity")
        or provenance.get("candidate_manifest_payload_identity")
        != status.get("candidate_manifest_payload_identity")
        or (
            status.get("startup_transaction_identity") is not None
            and status.get("startup_transaction_identity")
            != provenance.get("startup_transaction_identity")
        )
    ):
        return _refused(
            "startup_provenance_invalid",
            "provenance",
            "Guided startup provenance is invalid or mismatched.",
        )

    manifest_file = resolved_run_dir / GUIDED_CANDIDATE_MANIFEST_FILENAME
    config_file = resolved_run_dir / GUIDED_CONFIG_EFFECTIVE_FILENAME
    command_file = resolved_run_dir / GUIDED_COMMAND_RECORD_FILENAME
    loaded_manifest = load_guided_candidate_manifest(os.fspath(manifest_file))
    if (
        not loaded_manifest.accepted
        or loaded_manifest.manifest is None
        or loaded_manifest.manifest.canonical_candidate_manifest_payload_identity
        != status.get("candidate_manifest_payload_identity")
    ):
        return _refused(
            "startup_manifest_invalid",
            "manifest",
            "Guided startup manifest is invalid or mismatched.",
        )
    try:
        Config.from_yaml(os.fspath(config_file))
    except Exception:
        return _refused(
            "startup_config_invalid",
            "config",
            "Guided startup config is invalid.",
        )
    manifest_hash = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    command_bytes = command_file.read_bytes()
    command_hash = hashlib.sha256(command_bytes).hexdigest()
    correction_file = resolved_run_dir / GUIDED_PER_ROI_CORRECTION_FILENAME
    correction_hash = (
        hashlib.sha256(correction_file.read_bytes()).hexdigest()
        if correction_file.is_file() else None
    )
    if (
        manifest_hash != provenance.get("serialized_manifest_sha256")
        or config_hash != provenance.get("serialized_config_sha256")
        or command_hash != provenance.get("command_record_sha256")
        or correction_hash
        != provenance.get("serialized_native_correction_sha256")
    ):
        return _refused(
            "startup_artifact_hash_mismatch",
            "provenance",
            "Prepared startup artifact hashes do not match provenance.",
        )
    try:
        command_argv = tuple(command_bytes.decode("utf-8").splitlines())
    except UnicodeDecodeError:
        command_argv = ()
    semantic_pairs = (
        ("--input", input_dir),
        ("--out", output_dir),
        ("--config", config_path),
        ("--guided-candidate-manifest", manifest_path),
        ("--mode", "phasic"),
        ("--run-type", "full"),
    )
    if (
        "--guided-preallocated-run-dir" not in command_argv
        or any(
            (value := _argument_value(command_argv, flag)) is None
            or (
                not _same_path(value, expected)
                if flag
                in {
                    "--input",
                    "--out",
                    "--config",
                    "--guided-candidate-manifest",
                }
                else value != expected
            )
            for flag, expected in semantic_pairs
        )
        or "tonic" in command_argv
        or "both" in command_argv
    ):
        return _refused(
            "startup_command_invalid",
            "command",
            "Recorded startup command does not match the wrapper handoff.",
        )
    command_identity = _command_identity(
        command_argv, str(provenance.get("wrapper_entrypoint_identity", ""))
    )
    if command_identity != provenance.get("command_identity"):
        return _refused(
            "startup_command_identity_mismatch",
            "command",
            "Recorded startup command identity is inconsistent.",
        )
    return GuidedStartupClaimValidation(
        accepted=True,
        run_dir=os.fspath(resolved_run_dir),
        startup_transaction_identity=str(
            provenance["startup_transaction_identity"]
        ),
        blocking_issues=(),
    )


def claim_guided_preallocated_startup(
    validation: GuidedStartupClaimValidation,
    *,
    claimed_utc: str,
    process_id: int | None = None,
) -> GuidedStartupClaimResult:
    """Exclusively consume a validated startup handoff."""
    if (
        not isinstance(validation, GuidedStartupClaimValidation)
        or not validation.accepted
        or not validation.run_dir
        or not validation.startup_transaction_identity
    ):
        return GuidedStartupClaimResult(
            status="claim_refused",
            claimed=False,
            claim_path=None,
            startup_transaction_identity=None,
            blocking_issues=(
                GuidedStartupClaimIssue(
                    "claim_validation_not_accepted",
                    "claim",
                    "Accepted startup validation is required before claim.",
                ),
            ),
        )
    path = Path(validation.run_dir) / GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    payload = {
        "schema_name": GUIDED_STARTUP_WRAPPER_CLAIM_SCHEMA_NAME,
        "schema_version": GUIDED_STARTUP_WRAPPER_CLAIM_SCHEMA_VERSION,
        "claim_contract_version": GUIDED_STARTUP_WRAPPER_CLAIM_CONTRACT_VERSION,
        "startup_transaction_identity": validation.startup_transaction_identity,
        "claimed_utc": claimed_utc,
        "wrapper_process_id": process_id,
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
    except FileExistsError:
        issue = GuidedStartupClaimIssue(
            "startup_already_claimed",
            "claim",
            "Guided startup directory has already been claimed.",
        )
    except OSError as exc:
        issue = GuidedStartupClaimIssue(
            "startup_claim_write_failed",
            "claim",
            f"Guided startup claim could not be written: {exc}",
        )
    else:
        return GuidedStartupClaimResult(
            status="claimed",
            claimed=True,
            claim_path=os.fspath(path),
            startup_transaction_identity=validation.startup_transaction_identity,
            blocking_issues=(),
        )
    return GuidedStartupClaimResult(
        status="claim_refused",
        claimed=False,
        claim_path=os.fspath(path),
        startup_transaction_identity=validation.startup_transaction_identity,
        blocking_issues=(issue,),
    )
