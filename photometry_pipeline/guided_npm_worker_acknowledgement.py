"""Durable B2-D1 acknowledgement of exact Guided NPM authority consumption.

This contract proves worker acceptance, exact source-byte admission, and entry
into the real numerical Pipeline.  It deliberately says nothing about process
liveness, terminal outcome, output completeness, or scientific success.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_worker_prelaunch_claim import stored_paths_equal
from photometry_pipeline.guided_npm_worker_request import (
    GuidedNpmWorkerRequest,
    verify_guided_npm_worker_request,
)
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity


GUIDED_NPM_LAUNCH_CONTEXT_FILENAME = "guided_npm_launch_context.json"
GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_FILENAME = (
    "guided_npm_consumed_authority_receipt.json"
)
GUIDED_NPM_LAUNCH_CONTEXT_SCHEMA_NAME = "guided_npm_worker_launch_context"
GUIDED_NPM_LAUNCH_CONTEXT_SCHEMA_VERSION = "v1"
GUIDED_NPM_LAUNCH_CONTEXT_CONTRACT_VERSION = "guided_npm_worker_launch_context.v1"
GUIDED_NPM_CONSUMED_SOURCE_RECORD_CONTRACT_VERSION = (
    "guided_npm_consumed_source_record.v1"
)
GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_SCHEMA_NAME = (
    "guided_npm_worker_consumed_authority_receipt"
)
GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_SCHEMA_VERSION = "v1"
GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_CONTRACT_VERSION = (
    "guided_npm_worker_consumed_authority_receipt.v1"
)
GUIDED_NPM_AUTHORIZED_RUNTIME_IDENTITY_DOMAIN = (
    "guided_npm_authorized_runtime.v1"
)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("acknowledgement_nonfinite")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("acknowledgement_mapping_key_invalid")
        return {key: _canonical(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"acknowledgement_value_unsupported:{type(value).__name__}")


def _identity(value: Any, field_name: str, domain: str) -> str:
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != field_name
    }
    return hashlib.sha256(
        domain.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_canonical(payload))
    ).hexdigest()


def _digest_payload(domain: str, payload: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_canonical(payload))
    ).hexdigest()


@dataclass(frozen=True)
class GuidedNpmWorkerLaunchContext:
    context_schema_name: str
    context_schema_version: str
    context_contract_version: str
    source_launch_invocation_identity: str
    source_prelaunch_freshness_evidence_identity: str
    source_worker_request_identity: str
    source_execution_request_identity: str
    worker_request_artifact_path: str
    run_directory_path: str
    consumed_authority_receipt_path: str
    canonical_launch_context_identity: str


@dataclass(frozen=True)
class GuidedNpmConsumedSourceRecord:
    chronological_position: int
    source_path: str
    canonical_relative_path: str
    source_size_bytes: int
    source_sha256: str
    source_mtime_ns: int
    source_device: int
    source_inode: int
    source_mode: int
    canonical_session_runtime_identity: str
    resolved_timestamp_column: str
    timestamp_unit: str
    resolved_led_column: str
    reference_led_value: int | float | str
    signal_led_value: int | float | str
    support_policy: str
    output_time_basis: str
    physical_roi_ids: tuple[str, ...]
    observed_physical_roi_ids: tuple[str, ...]
    canonical_roi_ids: tuple[str, ...]
    physical_to_canonical_roi_map: tuple[tuple[str, str], ...]
    actual_elapsed_sec: float
    recording_time_start_sec: float
    recording_time_end_sec: float
    canonical_consumed_source_record_identity: str


@dataclass(frozen=True)
class GuidedNpmConsumedAuthorityEvidence:
    authorized_runtime_identity: str
    correction_authority_identity: str
    feature_authority_identity: str
    consumed_source_records: tuple[GuidedNpmConsumedSourceRecord, ...]
    numerical_dispatch_status: str = "entered"


@dataclass(frozen=True)
class GuidedNpmWorkerConsumedAuthorityReceipt:
    receipt_schema_name: str
    receipt_schema_version: str
    receipt_contract_version: str
    source_worker_request_identity: str
    source_execution_request_identity: str
    source_launch_invocation_identity: str
    source_launch_context_identity: str
    source_prelaunch_freshness_evidence_identity: str
    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    execution_mode: str
    observed_process_id: int
    worker_request_artifact_path: str
    run_directory_path: str
    ordered_session_paths: tuple[str, ...]
    ordered_session_identities: tuple[str, ...]
    chronological_positions: tuple[int, ...]
    actual_elapsed_sec_by_chunk: tuple[float, ...]
    parser_contract_identity: str
    complete_physical_roi_ids: tuple[str, ...]
    complete_canonical_roi_ids: tuple[str, ...]
    selected_canonical_roi_ids: tuple[str, ...]
    physical_to_canonical_roi_map: tuple[tuple[str, str], ...]
    correction_authority_identity: str
    feature_authority_identity: str
    authorized_runtime_identity: str
    consumed_source_records: tuple[GuidedNpmConsumedSourceRecord, ...]
    worker_acceptance_status: str
    consumed_authority_status: str
    numerical_dispatch_status: str
    completion_status: str
    canonical_consumed_authority_receipt_identity: str


def compute_guided_npm_launch_context_identity(value: GuidedNpmWorkerLaunchContext) -> str:
    return _identity(
        value,
        "canonical_launch_context_identity",
        GUIDED_NPM_LAUNCH_CONTEXT_CONTRACT_VERSION,
    )


def compute_guided_npm_consumed_source_record_identity(
    value: GuidedNpmConsumedSourceRecord,
) -> str:
    return _identity(
        value,
        "canonical_consumed_source_record_identity",
        GUIDED_NPM_CONSUMED_SOURCE_RECORD_CONTRACT_VERSION,
    )


def compute_guided_npm_worker_consumed_authority_receipt_identity(
    value: GuidedNpmWorkerConsumedAuthorityReceipt,
) -> str:
    return _identity(
        value,
        "canonical_consumed_authority_receipt_identity",
        GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_CONTRACT_VERSION,
    )


def compute_guided_npm_authorized_runtime_identity(runtime, worker: GuidedNpmWorkerRequest) -> str:
    """Bind the exact child runtime without Python object hashes."""
    execution = worker.execution_request
    payload = {
        "worker_request_identity": worker.canonical_worker_request_identity,
        "execution_request_identity": execution.canonical_execution_request_identity,
        "source_runtime_identity": execution.source_runtime_projection.canonical_source_runtime_projection_identity,
        "parser_runtime_identity": execution.parser_runtime_projection.canonical_parser_runtime_projection_identity,
        "timing_runtime_identity": execution.timing_runtime_projection.canonical_timing_runtime_projection_identity,
        "roi_runtime_identity": execution.roi_runtime_projection.canonical_roi_runtime_projection_identity,
        "correction_runtime_identity": execution.correction_runtime_projection.canonical_correction_runtime_projection_identity,
        "feature_runtime_identity": execution.feature_runtime_projection.canonical_feature_runtime_projection_identity,
        "output_runtime_identity": execution.output_runtime_projection.canonical_output_runtime_projection_identity,
        "authorized_input": asdict(runtime.authorized_input),
        "config": asdict(runtime.config),
        "mode": runtime.mode,
        "per_roi_correction": runtime.per_roi_correction,
        "per_roi_feature_config": {
            key: asdict(value) for key, value in runtime.per_roi_feature_config.items()
        },
        "per_roi_feature_provenance": runtime.per_roi_feature_provenance,
        "config_field_audit": runtime.config_field_audit,
    }
    return _digest_payload(GUIDED_NPM_AUTHORIZED_RUNTIME_IDENTITY_DOMAIN, payload)


def expected_guided_npm_launch_context_path(run_directory: str) -> str:
    return os.path.join(run_directory, GUIDED_NPM_LAUNCH_CONTEXT_FILENAME)


def expected_guided_npm_consumed_authority_receipt_path(run_directory: str) -> str:
    return os.path.join(run_directory, GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_FILENAME)


def build_guided_npm_worker_launch_context(invocation) -> GuidedNpmWorkerLaunchContext:
    context = GuidedNpmWorkerLaunchContext(
        GUIDED_NPM_LAUNCH_CONTEXT_SCHEMA_NAME,
        GUIDED_NPM_LAUNCH_CONTEXT_SCHEMA_VERSION,
        GUIDED_NPM_LAUNCH_CONTEXT_CONTRACT_VERSION,
        invocation.canonical_launch_invocation_identity,
        invocation.source_prelaunch_freshness_evidence_identity,
        invocation.source_worker_request_identity,
        invocation.source_execution_request_identity,
        invocation.worker_request_artifact_path,
        invocation.run_directory_path,
        expected_guided_npm_consumed_authority_receipt_path(invocation.run_directory_path),
        "0" * 64,
    )
    return replace(
        context,
        canonical_launch_context_identity=compute_guided_npm_launch_context_identity(context),
    )


def verify_guided_npm_worker_launch_context(
    context: GuidedNpmWorkerLaunchContext,
    *,
    worker_request: GuidedNpmWorkerRequest,
    expected_invocation_identity: str | None = None,
) -> None:
    if type(context) is not GuidedNpmWorkerLaunchContext:
        raise ValueError("launch_context_type_invalid")
    verify_guided_npm_worker_request(worker_request)
    execution = worker_request.execution_request
    style = execution.output_runtime_projection.output_base_path_style
    if (
        (context.context_schema_name, context.context_schema_version, context.context_contract_version)
        != (GUIDED_NPM_LAUNCH_CONTEXT_SCHEMA_NAME, GUIDED_NPM_LAUNCH_CONTEXT_SCHEMA_VERSION, GUIDED_NPM_LAUNCH_CONTEXT_CONTRACT_VERSION)
        or context.source_worker_request_identity != worker_request.canonical_worker_request_identity
        or context.source_execution_request_identity != execution.canonical_execution_request_identity
        or len(context.source_prelaunch_freshness_evidence_identity) != 64
        or any(character not in "0123456789abcdef" for character in context.source_prelaunch_freshness_evidence_identity)
        or (expected_invocation_identity is not None and context.source_launch_invocation_identity != expected_invocation_identity)
        or not stored_paths_equal(context.worker_request_artifact_path, worker_request.worker_request_artifact_path, style)
        or not stored_paths_equal(context.run_directory_path, worker_request.run_directory_path, style)
        or not stored_paths_equal(
            context.consumed_authority_receipt_path,
            expected_guided_npm_consumed_authority_receipt_path(worker_request.run_directory_path),
            style,
        )
        or compute_guided_npm_launch_context_identity(context) != context.canonical_launch_context_identity
    ):
        raise ValueError("launch_context_invalid")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("acknowledgement_duplicate_json_key")
        result[key] = value
    return result


def _strict_fields(value: Any, cls: type) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
        raise ValueError("acknowledgement_field_set_invalid")
    return dict(value)


def _decode_build(value: Any) -> ApplicationBuildIdentity:
    return ApplicationBuildIdentity(**_strict_fields(value, ApplicationBuildIdentity))


def _decode_source_record(value: Any) -> GuidedNpmConsumedSourceRecord:
    data = _strict_fields(value, GuidedNpmConsumedSourceRecord)
    for name in ("physical_roi_ids", "observed_physical_roi_ids", "canonical_roi_ids"):
        if not isinstance(data[name], list):
            raise ValueError("consumed_source_sequence_invalid")
        data[name] = tuple(data[name])
    mapping = data["physical_to_canonical_roi_map"]
    if not isinstance(mapping, list) or any(not isinstance(item, list) or len(item) != 2 for item in mapping):
        raise ValueError("consumed_source_mapping_invalid")
    data["physical_to_canonical_roi_map"] = tuple(tuple(item) for item in mapping)
    return GuidedNpmConsumedSourceRecord(**data)


def serialize_guided_npm_worker_launch_context(value: GuidedNpmWorkerLaunchContext) -> bytes:
    return _canonical_bytes(value)


def decode_guided_npm_worker_launch_context_bytes(content: bytes) -> GuidedNpmWorkerLaunchContext:
    try:
        data = json.loads(content, object_pairs_hook=_strict_object)
        result = GuidedNpmWorkerLaunchContext(**_strict_fields(data, GuidedNpmWorkerLaunchContext))
        if serialize_guided_npm_worker_launch_context(result) != content:
            raise ValueError("launch_context_noncanonical")
        return result
    except Exception as exc:
        raise ValueError("launch_context_decode_invalid") from exc


def serialize_guided_npm_worker_consumed_authority_receipt(
    value: GuidedNpmWorkerConsumedAuthorityReceipt,
) -> bytes:
    return _canonical_bytes(value)


def decode_guided_npm_worker_consumed_authority_receipt_bytes(
    content: bytes,
) -> GuidedNpmWorkerConsumedAuthorityReceipt:
    try:
        data = _strict_fields(
            json.loads(content, object_pairs_hook=_strict_object),
            GuidedNpmWorkerConsumedAuthorityReceipt,
        )
        data["application_build_identity"] = _decode_build(data["application_build_identity"])
        for name in (
            "ordered_session_paths", "ordered_session_identities", "chronological_positions",
            "actual_elapsed_sec_by_chunk", "complete_physical_roi_ids",
            "complete_canonical_roi_ids", "selected_canonical_roi_ids",
        ):
            if not isinstance(data[name], list):
                raise ValueError("receipt_sequence_invalid")
            data[name] = tuple(data[name])
        mapping = data["physical_to_canonical_roi_map"]
        if not isinstance(mapping, list) or any(not isinstance(item, list) or len(item) != 2 for item in mapping):
            raise ValueError("receipt_mapping_invalid")
        data["physical_to_canonical_roi_map"] = tuple(tuple(item) for item in mapping)
        records = data["consumed_source_records"]
        if not isinstance(records, list):
            raise ValueError("receipt_records_invalid")
        data["consumed_source_records"] = tuple(_decode_source_record(item) for item in records)
        result = GuidedNpmWorkerConsumedAuthorityReceipt(**data)
        if serialize_guided_npm_worker_consumed_authority_receipt(result) != content:
            raise ValueError("receipt_noncanonical")
        return result
    except Exception as exc:
        raise ValueError("consumed_authority_receipt_decode_invalid") from exc


def _publish_new(path: str, content: bytes) -> None:
    target = Path(path)
    parent = target.parent
    if target.name not in {GUIDED_NPM_LAUNCH_CONTEXT_FILENAME, GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_FILENAME}:
        raise ValueError("acknowledgement_destination_invalid")
    try:
        parent_facts = parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("acknowledgement_destination_invalid") from exc
    if (
        stat.S_ISLNK(parent_facts.st_mode)
        or not stat.S_ISDIR(parent_facts.st_mode)
        or target.exists()
        or target.is_symlink()
    ):
        raise ValueError("acknowledgement_destination_conflict")
    temporary = parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("acknowledgement_temporary_conflict")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _stable_read(path: str) -> bytes:
    target = Path(path)
    before = target.stat(follow_symlinks=False)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ValueError("acknowledgement_artifact_not_regular")
    with target.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        content = handle.read()
        after_open = os.fstat(handle.fileno())
    after = target.stat(follow_symlinks=False)
    facts = lambda item: (item.st_size, item.st_mtime_ns, item.st_dev, item.st_ino, item.st_mode)
    if not (facts(before) == facts(opened) == facts(after_open) == facts(after)):
        raise ValueError("acknowledgement_artifact_changed_during_read")
    return content


def persist_guided_npm_worker_launch_context(context: GuidedNpmWorkerLaunchContext) -> str:
    path = expected_guided_npm_launch_context_path(context.run_directory_path)
    _publish_new(path, serialize_guided_npm_worker_launch_context(context))
    observed = decode_guided_npm_worker_launch_context_bytes(_stable_read(path))
    if observed != context:
        raise ValueError("launch_context_reread_mismatch")
    return path


class GuidedNpmLaunchContextCleanupError(RuntimeError):
    """A launch context could not be proven safe to remove; it was left in place."""


def remove_exact_guided_npm_worker_launch_context(
    context: GuidedNpmWorkerLaunchContext,
) -> bool:
    """Remove only the exact, unchanged launch context this transaction persisted.

    Used only when this launch transaction persisted ``context`` but no process
    was created (cancellation observed, or the process launcher raised). Returns
    True once the artifact is confirmed absent, whether because it was verified
    and removed here or because it was already gone. Raises
    ``GuidedNpmLaunchContextCleanupError`` and leaves the artifact untouched if
    anything at the expected path cannot be proven to be that exact context.
    """
    path = expected_guided_npm_launch_context_path(context.run_directory_path)
    target = Path(path)
    try:
        before = target.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise GuidedNpmLaunchContextCleanupError(
            "launch_context_cleanup_initial_stat_failed"
        ) from exc
    try:
        content = _stable_read(path)
        observed = decode_guided_npm_worker_launch_context_bytes(content)
    except (OSError, ValueError) as exc:
        raise GuidedNpmLaunchContextCleanupError(
            "launch_context_cleanup_verification_failed"
        ) from exc
    if (
        observed != context
        or compute_guided_npm_launch_context_identity(observed)
        != context.canonical_launch_context_identity
    ):
        raise GuidedNpmLaunchContextCleanupError("launch_context_cleanup_mismatch")
    facts = lambda item: (
        item.st_size,
        item.st_mtime_ns,
        item.st_dev,
        item.st_ino,
        item.st_mode,
    )
    try:
        pre_unlink = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuidedNpmLaunchContextCleanupError(
            "launch_context_cleanup_vanished_before_unlink"
        ) from exc
    if facts(pre_unlink) != facts(before):
        raise GuidedNpmLaunchContextCleanupError(
            "launch_context_cleanup_changed_before_unlink"
        )
    try:
        target.unlink()
    except OSError as exc:
        raise GuidedNpmLaunchContextCleanupError(
            "launch_context_cleanup_unlink_failed"
        ) from exc
    if target.exists() or target.is_symlink():
        raise GuidedNpmLaunchContextCleanupError(
            "launch_context_cleanup_still_present"
        )
    return True


def read_guided_npm_worker_launch_context(
    path: str, *, worker_request: GuidedNpmWorkerRequest
) -> GuidedNpmWorkerLaunchContext:
    expected = expected_guided_npm_launch_context_path(worker_request.run_directory_path)
    style = worker_request.execution_request.output_runtime_projection.output_base_path_style
    if not stored_paths_equal(path, expected, style):
        raise ValueError("launch_context_path_invalid")
    result = decode_guided_npm_worker_launch_context_bytes(_stable_read(path))
    verify_guided_npm_worker_launch_context(result, worker_request=worker_request)
    return result


def build_guided_npm_worker_consumed_authority_receipt(
    *,
    worker_request: GuidedNpmWorkerRequest,
    launch_context: GuidedNpmWorkerLaunchContext,
    evidence: GuidedNpmConsumedAuthorityEvidence,
    observed_process_id: int,
) -> GuidedNpmWorkerConsumedAuthorityReceipt:
    verify_guided_npm_worker_request(worker_request)
    verify_guided_npm_worker_launch_context(launch_context, worker_request=worker_request)
    execution = worker_request.execution_request
    authorized = execution.source_runtime_projection
    parser = execution.parser_runtime_projection
    roi = execution.roi_runtime_projection
    if evidence.numerical_dispatch_status != "entered":
        raise ValueError("numerical_dispatch_not_entered")
    receipt = GuidedNpmWorkerConsumedAuthorityReceipt(
        GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_SCHEMA_NAME,
        GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_SCHEMA_VERSION,
        GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_CONTRACT_VERSION,
        worker_request.canonical_worker_request_identity,
        execution.canonical_execution_request_identity,
        launch_context.source_launch_invocation_identity,
        launch_context.canonical_launch_context_identity,
        launch_context.source_prelaunch_freshness_evidence_identity,
        worker_request.application_build_identity,
        worker_request.guided_plan_identity,
        worker_request.validation_revision,
        execution.execution_mode,
        observed_process_id,
        worker_request.worker_request_artifact_path,
        worker_request.run_directory_path,
        authorized.ordered_source_paths,
        tuple(item.canonical_session_runtime_identity for item in authorized.ordered_sessions),
        tuple(item.chronological_position for item in authorized.ordered_sessions),
        tuple(item.actual_elapsed_sec for item in authorized.ordered_sessions),
        parser.parser_policy_identity,
        roi.complete_physical_source_columns,
        roi.complete_canonical_roi_ids,
        roi.selected_canonical_roi_ids,
        tuple((item.physical_source_column, item.canonical_roi_id) for item in roi.physical_to_canonical_roi_mapping),
        evidence.correction_authority_identity,
        evidence.feature_authority_identity,
        evidence.authorized_runtime_identity,
        evidence.consumed_source_records,
        "accepted_exact_worker_authority",
        "verified",
        "entered",
        "not_available",
        "0" * 64,
    )
    receipt = replace(
        receipt,
        canonical_consumed_authority_receipt_identity=(
            compute_guided_npm_worker_consumed_authority_receipt_identity(receipt)
        ),
    )
    verify_guided_npm_child_consumed_authority_receipt(
        receipt,
        worker_request=worker_request,
        launch_context=launch_context,
        evidence=evidence,
    )
    return receipt


def verify_guided_npm_child_consumed_authority_receipt(
    receipt: GuidedNpmWorkerConsumedAuthorityReceipt,
    *,
    worker_request: GuidedNpmWorkerRequest,
    launch_context: GuidedNpmWorkerLaunchContext,
    evidence: GuidedNpmConsumedAuthorityEvidence,
) -> None:
    """Pure child-side verification before the one publication attempt."""
    verify_guided_npm_worker_request(worker_request)
    verify_guided_npm_worker_launch_context(
        launch_context, worker_request=worker_request
    )
    execution = worker_request.execution_request
    source = execution.source_runtime_projection
    parser = execution.parser_runtime_projection
    roi = execution.roi_runtime_projection
    expected_mapping = tuple(
        (item.physical_source_column, item.canonical_roi_id)
        for item in roi.physical_to_canonical_roi_mapping
    )
    if (
        receipt.source_worker_request_identity != worker_request.canonical_worker_request_identity
        or receipt.source_execution_request_identity != execution.canonical_execution_request_identity
        or receipt.source_launch_invocation_identity != launch_context.source_launch_invocation_identity
        or receipt.source_launch_context_identity != launch_context.canonical_launch_context_identity
        or receipt.source_prelaunch_freshness_evidence_identity
        != launch_context.source_prelaunch_freshness_evidence_identity
        or isinstance(receipt.observed_process_id, bool)
        or not isinstance(receipt.observed_process_id, int)
        or receipt.observed_process_id <= 0
        or receipt.ordered_session_paths != source.ordered_source_paths
        or receipt.ordered_session_identities != tuple(item.canonical_session_runtime_identity for item in source.ordered_sessions)
        or receipt.chronological_positions != tuple(range(len(source.ordered_sessions)))
        or receipt.actual_elapsed_sec_by_chunk != tuple(item.actual_elapsed_sec for item in source.ordered_sessions)
        or receipt.parser_contract_identity != parser.parser_policy_identity
        or receipt.complete_physical_roi_ids != roi.complete_physical_source_columns
        or receipt.complete_canonical_roi_ids != roi.complete_canonical_roi_ids
        or receipt.selected_canonical_roi_ids != roi.selected_canonical_roi_ids
        or receipt.physical_to_canonical_roi_map != expected_mapping
        or receipt.correction_authority_identity != evidence.correction_authority_identity
        or receipt.feature_authority_identity != evidence.feature_authority_identity
        or receipt.authorized_runtime_identity != evidence.authorized_runtime_identity
        or receipt.consumed_source_records != evidence.consumed_source_records
        or len(receipt.consumed_source_records) != len(source.ordered_sessions)
        or (receipt.worker_acceptance_status, receipt.consumed_authority_status,
            receipt.numerical_dispatch_status, receipt.completion_status)
        != ("accepted_exact_worker_authority", "verified", "entered", "not_available")
        or compute_guided_npm_worker_consumed_authority_receipt_identity(receipt)
        != receipt.canonical_consumed_authority_receipt_identity
    ):
        raise ValueError("child_consumed_authority_receipt_invalid")
    for record, session in zip(receipt.consumed_source_records, source.ordered_sessions, strict=True):
        if (
            record.chronological_position != session.chronological_position
            or record.source_path != session.source_path
            or record.source_size_bytes != session.source_size_bytes
            or record.source_sha256 != session.source_sha256
            or record.canonical_session_runtime_identity != session.canonical_session_runtime_identity
            or record.resolved_timestamp_column != session.resolved_timestamp_column
            or record.physical_roi_ids != session.physical_roi_inventory
            or record.canonical_roi_ids != roi.complete_canonical_roi_ids
            or record.physical_to_canonical_roi_map != expected_mapping
            or record.actual_elapsed_sec != session.actual_elapsed_sec
            or compute_guided_npm_consumed_source_record_identity(record)
            != record.canonical_consumed_source_record_identity
        ):
            raise ValueError("child_consumed_source_record_invalid")


def verify_guided_npm_worker_consumed_authority_receipt(
    receipt: GuidedNpmWorkerConsumedAuthorityReceipt,
    *,
    worker_request: GuidedNpmWorkerRequest,
    launch_invocation,
    execution_start_receipt,
    authorized_runtime_identity: str | None = None,
) -> None:
    """Purely reconcile D1 evidence; perform no file or process access."""
    from photometry_pipeline.guided_npm_worker_launch import (
        compute_guided_npm_worker_execution_start_receipt_identity,
        compute_guided_npm_worker_launch_invocation_identity,
    )
    from photometry_pipeline.guided_npm_authorized_adapter import (
        build_guided_npm_authorized_runtime,
    )

    if type(receipt) is not GuidedNpmWorkerConsumedAuthorityReceipt:
        raise ValueError("consumed_authority_receipt_type_invalid")
    verify_guided_npm_worker_request(worker_request)
    # Invocation/start verification requires the original claim; their own
    # pure verifiers are exercised by the caller before this worker-only bind.
    execution = worker_request.execution_request
    source = execution.source_runtime_projection
    parser = execution.parser_runtime_projection
    roi = execution.roi_runtime_projection
    runtime = build_guided_npm_authorized_runtime(worker_request)
    expected_runtime_identity = runtime.canonical_guided_npm_authorized_runtime_identity
    expected_context = build_guided_npm_worker_launch_context(launch_invocation)
    records = receipt.consumed_source_records
    expected_mapping = tuple(
        (item.physical_source_column, item.canonical_roi_id)
        for item in roi.physical_to_canonical_roi_mapping
    )
    if (
        (receipt.receipt_schema_name, receipt.receipt_schema_version, receipt.receipt_contract_version)
        != (GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_SCHEMA_NAME, GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_SCHEMA_VERSION, GUIDED_NPM_CONSUMED_AUTHORITY_RECEIPT_CONTRACT_VERSION)
        or receipt.source_worker_request_identity != worker_request.canonical_worker_request_identity
        or receipt.source_execution_request_identity != execution.canonical_execution_request_identity
        or receipt.source_launch_invocation_identity != launch_invocation.canonical_launch_invocation_identity
        or receipt.source_launch_context_identity != expected_context.canonical_launch_context_identity
        or receipt.source_prelaunch_freshness_evidence_identity
        != launch_invocation.source_prelaunch_freshness_evidence_identity
        or receipt.application_build_identity != worker_request.application_build_identity
        or receipt.guided_plan_identity != worker_request.guided_plan_identity
        or receipt.validation_revision != worker_request.validation_revision
        or receipt.execution_mode != execution.execution_mode
        or isinstance(receipt.observed_process_id, bool)
        or not isinstance(receipt.observed_process_id, int)
        or receipt.observed_process_id <= 0
        or receipt.observed_process_id != execution_start_receipt.process_id
        or execution_start_receipt.source_launch_invocation_identity
        != launch_invocation.canonical_launch_invocation_identity
        or execution_start_receipt.source_worker_request_identity
        != worker_request.canonical_worker_request_identity
        or execution_start_receipt.source_execution_request_identity
        != execution.canonical_execution_request_identity
        or execution_start_receipt.argument_vector != launch_invocation.argument_vector
        or execution_start_receipt.launch_context_artifact_path
        != launch_invocation.launch_context_artifact_path
        or (execution_start_receipt.launch_status, execution_start_receipt.execution_status,
            execution_start_receipt.completion_status, execution_start_receipt.consumed_authority_status)
        != ("process_created", "start_unconfirmed", "not_available", "not_available")
        or receipt.worker_request_artifact_path != worker_request.worker_request_artifact_path
        or receipt.run_directory_path != worker_request.run_directory_path
        or receipt.ordered_session_paths != source.ordered_source_paths
        or receipt.ordered_session_identities != tuple(item.canonical_session_runtime_identity for item in source.ordered_sessions)
        or receipt.chronological_positions != tuple(range(len(source.ordered_sessions)))
        or receipt.actual_elapsed_sec_by_chunk != tuple(item.actual_elapsed_sec for item in source.ordered_sessions)
        or receipt.parser_contract_identity != parser.parser_policy_identity
        or receipt.complete_physical_roi_ids != roi.complete_physical_source_columns
        or receipt.complete_canonical_roi_ids != roi.complete_canonical_roi_ids
        or receipt.selected_canonical_roi_ids != roi.selected_canonical_roi_ids
        or receipt.physical_to_canonical_roi_map != expected_mapping
        or receipt.correction_authority_identity != execution.correction_runtime_projection.canonical_correction_runtime_projection_identity
        or receipt.feature_authority_identity != execution.feature_runtime_projection.canonical_feature_runtime_projection_identity
        or receipt.authorized_runtime_identity != expected_runtime_identity
        or (authorized_runtime_identity is not None and receipt.authorized_runtime_identity != authorized_runtime_identity)
        or len(records) != len(source.ordered_sessions)
        or (receipt.worker_acceptance_status, receipt.consumed_authority_status, receipt.numerical_dispatch_status, receipt.completion_status)
        != ("accepted_exact_worker_authority", "verified", "entered", "not_available")
    ):
        raise ValueError("consumed_authority_receipt_invalid")
    if (
        compute_guided_npm_worker_launch_invocation_identity(launch_invocation)
        != launch_invocation.canonical_launch_invocation_identity
        or compute_guided_npm_worker_execution_start_receipt_identity(execution_start_receipt)
        != execution_start_receipt.canonical_execution_start_receipt_identity
    ):
        raise ValueError("consumed_authority_launch_binding_invalid")
    for record, session in zip(records, source.ordered_sessions, strict=True):
        fs = float(runtime.config.target_fs_hz)
        if runtime.config.allow_partial_final_chunk:
            ideal = int(round(float(runtime.config.chunk_duration_sec) * fs))
            support = int(math.floor(float(session.support_end_offset_sec) * fs)) + 1
            expected_local_start = 0.0
            expected_local_end = (min(ideal, support) - 1) / fs
        else:
            expected_local_start = math.ceil(float(session.support_start_offset_sec) * fs) / fs
            expected_local_end = math.floor(float(session.support_end_offset_sec) * fs) / fs
        if (
            type(record) is not GuidedNpmConsumedSourceRecord
            or record.chronological_position != session.chronological_position
            or record.source_path != session.source_path
            or record.canonical_relative_path != session.canonical_relative_path
            or record.source_size_bytes != session.source_size_bytes
            or record.source_sha256 != session.source_sha256
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    record.source_mtime_ns,
                    record.source_device,
                    record.source_inode,
                    record.source_mode,
                )
            )
            or record.canonical_session_runtime_identity != session.canonical_session_runtime_identity
            or record.resolved_timestamp_column != session.resolved_timestamp_column
            or record.timestamp_unit != session.timestamp_unit
            or record.resolved_led_column != session.resolved_led_column
            or record.reference_led_value != parser.reference_led_value
            or record.signal_led_value != parser.signal_led_value
            or record.support_policy != session.support_policy
            or record.output_time_basis != session.output_time_basis
            or record.physical_roi_ids != session.physical_roi_inventory
            or record.observed_physical_roi_ids != session.physical_roi_inventory
            or record.canonical_roi_ids != roi.complete_canonical_roi_ids
            or record.physical_to_canonical_roi_map != expected_mapping
            or record.actual_elapsed_sec != session.actual_elapsed_sec
            or record.recording_time_start_sec
            != float(session.actual_elapsed_sec + expected_local_start)
            or record.recording_time_end_sec
            != float(session.actual_elapsed_sec + expected_local_end)
            or compute_guided_npm_consumed_source_record_identity(record) != record.canonical_consumed_source_record_identity
        ):
            raise ValueError("consumed_source_record_invalid")
    if (
        compute_guided_npm_worker_consumed_authority_receipt_identity(receipt)
        != receipt.canonical_consumed_authority_receipt_identity
    ):
        raise ValueError("consumed_authority_receipt_identity_invalid")


def publish_guided_npm_worker_consumed_authority_receipt(
    receipt: GuidedNpmWorkerConsumedAuthorityReceipt,
    *,
    receipt_path: str,
    launch_context: GuidedNpmWorkerLaunchContext,
) -> None:
    expected = expected_guided_npm_consumed_authority_receipt_path(
        launch_context.run_directory_path
    )
    if (
        receipt_path != expected
        or receipt_path != launch_context.consumed_authority_receipt_path
        or receipt.run_directory_path != launch_context.run_directory_path
        or receipt.source_launch_context_identity
        != launch_context.canonical_launch_context_identity
        or receipt.source_launch_invocation_identity
        != launch_context.source_launch_invocation_identity
    ):
        raise ValueError("consumed_authority_receipt_path_invalid")
    if (
        compute_guided_npm_worker_consumed_authority_receipt_identity(receipt)
        != receipt.canonical_consumed_authority_receipt_identity
        or (receipt.worker_acceptance_status, receipt.consumed_authority_status,
            receipt.numerical_dispatch_status, receipt.completion_status)
        != ("accepted_exact_worker_authority", "verified", "entered", "not_available")
    ):
        raise ValueError("consumed_authority_receipt_invalid")
    content = serialize_guided_npm_worker_consumed_authority_receipt(receipt)
    _publish_new(receipt_path, content)
    observed = decode_guided_npm_worker_consumed_authority_receipt_bytes(_stable_read(receipt_path))
    if observed != receipt:
        raise ValueError("consumed_authority_receipt_reread_mismatch")


def read_and_verify_guided_npm_consumed_authority_receipt(
    receipt_path: str,
    *,
    prelaunch_claim,
    launch_invocation,
    execution_start_receipt,
) -> GuidedNpmWorkerConsumedAuthorityReceipt:
    from photometry_pipeline.guided_npm_worker_launch import (
        verify_guided_npm_worker_execution_start_receipt,
        verify_guided_npm_worker_launch_invocation,
    )

    verify_guided_npm_worker_launch_invocation(launch_invocation, prelaunch_claim)
    verify_guided_npm_worker_execution_start_receipt(
        execution_start_receipt, prelaunch_claim, launch_invocation
    )
    worker = prelaunch_claim.worker_request
    expected = expected_guided_npm_consumed_authority_receipt_path(worker.run_directory_path)
    style = worker.execution_request.output_runtime_projection.output_base_path_style
    if not stored_paths_equal(receipt_path, expected, style):
        raise ValueError("consumed_authority_receipt_path_invalid")
    receipt = decode_guided_npm_worker_consumed_authority_receipt_bytes(
        _stable_read(receipt_path)
    )
    verify_guided_npm_worker_consumed_authority_receipt(
        receipt,
        worker_request=worker,
        launch_invocation=launch_invocation,
        execution_start_receipt=execution_start_receipt,
    )
    return receipt
