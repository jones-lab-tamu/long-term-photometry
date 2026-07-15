"""Pure B2-C6B1 live-evidence and worker-request contracts.

A valid ``guided_npm_worker_request.v1`` is a self-verifying, non-launched
execution authority.  It does not prove that any particular filesystem
artifact was successfully materialized.  Future B2-C6B2 launch must also
require a matching verified materialization receipt or a new independent
path-based materialization claim.  This module performs no filesystem access
and launches no work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
import ntpath
import posixpath
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_production_execution_request import (
    GuidedNpmProductionExecutionRequest,
    compute_guided_npm_production_session_sequence_identity,
    compute_guided_npm_production_source_content_identity,
    compute_guided_npm_production_source_membership_identity,
    deserialize_guided_npm_production_execution_request,
    serialize_guided_npm_production_execution_request,
    verify_guided_npm_production_execution_request,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    verify_application_build_identity,
)
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity


GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_NAME = "guided_npm_live_freshness_evidence"
GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_VERSION = "v1"
GUIDED_NPM_LIVE_FRESHNESS_CONTRACT_VERSION = "guided_npm_live_freshness_evidence.v1"
GUIDED_NPM_WORKER_REQUEST_SCHEMA_NAME = "guided_npm_worker_request"
GUIDED_NPM_WORKER_REQUEST_SCHEMA_VERSION = "v1"
GUIDED_NPM_WORKER_REQUEST_CONTRACT_VERSION = "guided_npm_worker_request.v1"
GUIDED_NPM_WORKER_REQUEST_IDENTITY_DOMAIN = "guided_npm_worker_request.v1"
GUIDED_NPM_WORKER_REQUEST_FILENAME = "guided_npm_worker_request.json"
GUIDED_NPM_DISCOVERY_CONTRACT_VERSION = "guided_npm_immediate_case_insensitive_csv.v1"

_HEX = frozenset("0123456789abcdef")


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("worker_request_nonfinite")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("worker_request_mapping_key_invalid")
        return {key: _canonical(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"worker_request_value_unsupported:{type(value).__name__}")


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + encode_canonical_value(_canonical(value))).hexdigest()


def _identity(value: Any, field_name: str, domain: str) -> str:
    return _digest(domain, {item.name: getattr(value, item.name) for item in fields(value) if item.name != field_name})


@dataclass(frozen=True)
class GuidedNpmLiveFileFacts:
    canonical_path: str
    size_bytes: int
    mtime_ns: int
    file_type: str
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class GuidedNpmLiveDirectoryFacts:
    canonical_path: str
    mtime_ns: int
    file_type: str
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class GuidedNpmLiveVerifiedSourceFile:
    chronological_position: int
    source_path: str
    canonical_relative_path: str
    expected_size_bytes: int
    observed_size_bytes: int
    expected_sha256: str
    observed_sha256: str
    pre_open_file_facts: GuidedNpmLiveFileFacts
    opened_file_facts: GuidedNpmLiveFileFacts
    post_hash_file_facts: GuidedNpmLiveFileFacts
    final_path_file_facts: GuidedNpmLiveFileFacts
    source_runtime_session_identity: str
    canonical_live_verified_source_file_identity: str


@dataclass(frozen=True)
class GuidedNpmLiveSourceFreshnessEvidence:
    freshness_schema_name: str
    freshness_schema_version: str
    freshness_contract_version: str
    source_root_canonical: str
    source_path_style: str
    discovery_contract_version: str
    expected_runtime_source_projection_identity: str
    expected_runtime_membership_identity: str
    expected_runtime_content_identity: str
    expected_runtime_session_sequence_identity: str
    ordered_verified_files: tuple[GuidedNpmLiveVerifiedSourceFile, ...]
    live_membership_identity: str
    live_content_identity: str
    live_session_sequence_identity: str
    live_verified_file_sequence_identity: str
    source_root_pre_facts: GuidedNpmLiveDirectoryFacts
    source_root_post_facts: GuidedNpmLiveDirectoryFacts
    freshness_status: str
    canonical_live_freshness_evidence_identity: str


@dataclass(frozen=True)
class GuidedNpmWorkerRequest:
    worker_request_schema_name: str
    worker_request_schema_version: str
    worker_request_contract_version: str
    source_execution_request_identity: str
    source_claim_receipt_identity: str
    source_startup_payload_identity: str
    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    startup_artifact_path: str
    startup_artifact_sha256: str
    startup_artifact_size_bytes: int
    execution_request: GuidedNpmProductionExecutionRequest
    live_freshness_evidence: GuidedNpmLiveSourceFreshnessEvidence
    run_directory_path: str
    worker_request_artifact_path: str
    worker_request_artifact_filename: str
    request_status: str
    launch_status: str
    execution_status: str
    completion_status: str
    runnable: bool
    canonical_worker_request_identity: str


def compute_guided_npm_live_verified_source_file_identity(value: GuidedNpmLiveVerifiedSourceFile) -> str:
    return _identity(value, "canonical_live_verified_source_file_identity", "guided_npm_live_verified_source_file.v1")


def compute_guided_npm_live_verified_file_sequence_identity(values: tuple[GuidedNpmLiveVerifiedSourceFile, ...]) -> str:
    return _digest("guided_npm_live_verified_file_sequence.v1", tuple(item.canonical_live_verified_source_file_identity for item in values))


def compute_guided_npm_live_freshness_evidence_identity(value: GuidedNpmLiveSourceFreshnessEvidence) -> str:
    return _identity(value, "canonical_live_freshness_evidence_identity", GUIDED_NPM_LIVE_FRESHNESS_CONTRACT_VERSION)


def compute_guided_npm_worker_request_identity(value: GuidedNpmWorkerRequest) -> str:
    return _identity(value, "canonical_worker_request_identity", GUIDED_NPM_WORKER_REQUEST_IDENTITY_DOMAIN)


def _same_path(left: str, right: str) -> bool:
    return ntpath.normcase(left) == ntpath.normcase(right) if ntpath.isabs(left) else left == right


def _verify_facts(value: Any, expected_type: type, expected_path: str) -> None:
    if type(value) is not expected_type or not _same_path(value.canonical_path, expected_path):
        raise ValueError("live_file_facts_invalid")
    integer_names = ("mtime_ns", "device", "inode", "mode")
    if expected_type is GuidedNpmLiveFileFacts:
        integer_names += ("size_bytes",)
    if any(isinstance(getattr(value, name), bool) or not isinstance(getattr(value, name), int) or getattr(value, name) < 0 for name in integer_names):
        raise ValueError("live_file_facts_invalid")
    required_type = "regular_file" if expected_type is GuidedNpmLiveFileFacts else "directory"
    if value.file_type != required_type:
        raise ValueError("live_file_facts_invalid")


def verify_guided_npm_live_source_freshness_evidence(
    evidence: GuidedNpmLiveSourceFreshnessEvidence,
    execution_request: GuidedNpmProductionExecutionRequest,
) -> None:
    if type(evidence) is not GuidedNpmLiveSourceFreshnessEvidence:
        raise ValueError("live_freshness_type_invalid")
    verify_guided_npm_production_execution_request(execution_request)
    source = execution_request.source_runtime_projection
    if (evidence.freshness_schema_name, evidence.freshness_schema_version, evidence.freshness_contract_version) != (
        GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_NAME, GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_VERSION, GUIDED_NPM_LIVE_FRESHNESS_CONTRACT_VERSION,
    ) or evidence.freshness_status != "live_verified":
        raise ValueError("live_freshness_schema_or_state_invalid")
    if (
        not _same_path(evidence.source_root_canonical, source.source_root_canonical)
        or evidence.source_path_style != source.source_path_style
        or evidence.discovery_contract_version != GUIDED_NPM_DISCOVERY_CONTRACT_VERSION
        or evidence.expected_runtime_source_projection_identity != source.canonical_source_runtime_projection_identity
        or evidence.expected_runtime_membership_identity != source.runtime_source_membership_identity
        or evidence.expected_runtime_content_identity != source.runtime_source_content_identity
        or evidence.expected_runtime_session_sequence_identity != source.runtime_session_sequence_identity
    ):
        raise ValueError("live_freshness_authority_mismatch")
    _verify_facts(evidence.source_root_pre_facts, GuidedNpmLiveDirectoryFacts, source.source_root_canonical)
    _verify_facts(evidence.source_root_post_facts, GuidedNpmLiveDirectoryFacts, source.source_root_canonical)
    if evidence.source_root_pre_facts != evidence.source_root_post_facts:
        raise ValueError("live_source_root_replaced")
    if len(evidence.ordered_verified_files) != len(source.ordered_sessions):
        raise ValueError("live_source_membership_mismatch")
    for live, session in zip(evidence.ordered_verified_files, source.ordered_sessions, strict=True):
        if type(live) is not GuidedNpmLiveVerifiedSourceFile:
            raise ValueError("live_verified_source_type_invalid")
        for facts in (live.pre_open_file_facts, live.opened_file_facts, live.post_hash_file_facts, live.final_path_file_facts):
            _verify_facts(facts, GuidedNpmLiveFileFacts, session.source_path)
        if not (live.pre_open_file_facts == live.opened_file_facts == live.post_hash_file_facts == live.final_path_file_facts):
            raise ValueError("live_source_file_mutated")
        if (
            (live.chronological_position, live.source_path, live.canonical_relative_path)
            != (session.chronological_position, session.source_path, session.canonical_relative_path)
            or live.expected_size_bytes != session.source_size_bytes
            or live.observed_size_bytes != session.source_size_bytes
            or live.expected_sha256 != session.source_sha256
            or live.observed_sha256 != session.source_sha256
            or live.source_runtime_session_identity != session.canonical_session_runtime_identity
            or live.pre_open_file_facts.size_bytes != session.source_size_bytes
            or compute_guided_npm_live_verified_source_file_identity(live) != live.canonical_live_verified_source_file_identity
        ):
            raise ValueError("live_verified_source_mismatch")
    sessions = source.ordered_sessions
    expected_membership = compute_guided_npm_production_source_membership_identity(sessions)
    expected_content = compute_guided_npm_production_source_content_identity(sessions)
    expected_sequence = compute_guided_npm_production_session_sequence_identity(sessions)
    if (
        evidence.live_membership_identity != expected_membership
        or evidence.live_content_identity != expected_content
        or evidence.live_session_sequence_identity != expected_sequence
        or evidence.live_verified_file_sequence_identity != compute_guided_npm_live_verified_file_sequence_identity(evidence.ordered_verified_files)
        or compute_guided_npm_live_freshness_evidence_identity(evidence) != evidence.canonical_live_freshness_evidence_identity
    ):
        raise ValueError("live_source_runtime_identity_mismatch")


def verify_guided_npm_worker_request(request: GuidedNpmWorkerRequest) -> None:
    if type(request) is not GuidedNpmWorkerRequest:
        raise ValueError("worker_request_type_invalid")
    if (request.worker_request_schema_name, request.worker_request_schema_version, request.worker_request_contract_version) != (
        GUIDED_NPM_WORKER_REQUEST_SCHEMA_NAME, GUIDED_NPM_WORKER_REQUEST_SCHEMA_VERSION, GUIDED_NPM_WORKER_REQUEST_CONTRACT_VERSION,
    ):
        raise ValueError("worker_request_schema_unsupported")
    verify_application_build_identity(request.application_build_identity)
    verify_guided_npm_production_execution_request(request.execution_request)
    verify_guided_npm_live_source_freshness_evidence(request.live_freshness_evidence, request.execution_request)
    execution = request.execution_request
    output = execution.output_runtime_projection
    module = ntpath if output.output_base_path_style == "windows_drive" else posixpath
    expected_artifact = module.join(output.run_directory_path, GUIDED_NPM_WORKER_REQUEST_FILENAME)
    if (
        request.source_execution_request_identity != execution.canonical_execution_request_identity
        or request.source_claim_receipt_identity != execution.source_claim_receipt_identity
        or request.source_startup_payload_identity != execution.source_startup_payload_identity
        or request.application_build_identity != execution.application_build_identity
        or request.guided_plan_identity != execution.guided_plan_identity
        or request.validation_revision != execution.validation_revision
        or request.startup_artifact_path != execution.startup_artifact_path
        or request.startup_artifact_sha256 != execution.startup_artifact_sha256
        or request.startup_artifact_size_bytes != execution.startup_artifact_size_bytes
        or request.run_directory_path != output.run_directory_path
        or request.worker_request_artifact_filename != GUIDED_NPM_WORKER_REQUEST_FILENAME
        or request.worker_request_artifact_path != expected_artifact
        or (request.request_status, request.launch_status, request.execution_status, request.completion_status, request.runnable)
        != ("constructed_for_worker", "not_launched", "not_started", "not_available", False)
    ):
        raise ValueError("worker_request_authority_or_state_mismatch")
    if compute_guided_npm_worker_request_identity(request) != request.canonical_worker_request_identity:
        raise ValueError("worker_request_identity_mismatch")


def serialize_guided_npm_worker_request(request: GuidedNpmWorkerRequest) -> dict[str, Any]:
    verify_guided_npm_worker_request(request)
    value = _canonical(request)
    value["execution_request"] = serialize_guided_npm_production_execution_request(request.execution_request)
    return {"identity_domain": GUIDED_NPM_WORKER_REQUEST_IDENTITY_DOMAIN, **value}


def canonical_guided_npm_worker_request_bytes(request: GuidedNpmWorkerRequest) -> bytes:
    return (json.dumps(serialize_guided_npm_worker_request(request), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _strict_fields(value: Any, cls: type) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
        raise ValueError("worker_request_field_set_invalid")
    return value


def _facts(value: Any, cls: type):
    value = _strict_fields(value, cls)
    return cls(**value)


def _live_file(value: Any) -> GuidedNpmLiveVerifiedSourceFile:
    value = dict(_strict_fields(value, GuidedNpmLiveVerifiedSourceFile))
    for name in ("pre_open_file_facts", "opened_file_facts", "post_hash_file_facts", "final_path_file_facts"):
        value[name] = _facts(value[name], GuidedNpmLiveFileFacts)
    return GuidedNpmLiveVerifiedSourceFile(**value)


def _freshness(value: Any) -> GuidedNpmLiveSourceFreshnessEvidence:
    value = dict(_strict_fields(value, GuidedNpmLiveSourceFreshnessEvidence))
    if not isinstance(value["ordered_verified_files"], list):
        raise ValueError("worker_request_sequence_invalid")
    value["ordered_verified_files"] = tuple(_live_file(item) for item in value["ordered_verified_files"])
    value["source_root_pre_facts"] = _facts(value["source_root_pre_facts"], GuidedNpmLiveDirectoryFacts)
    value["source_root_post_facts"] = _facts(value["source_root_post_facts"], GuidedNpmLiveDirectoryFacts)
    return GuidedNpmLiveSourceFreshnessEvidence(**value)


def deserialize_guided_npm_worker_request(payload: Mapping[str, Any]) -> GuidedNpmWorkerRequest:
    try:
        if not isinstance(payload, Mapping) or payload.get("identity_domain") != GUIDED_NPM_WORKER_REQUEST_IDENTITY_DOMAIN:
            raise ValueError("worker_request_identity_domain_invalid")
        value = dict(payload)
        del value["identity_domain"]
        value = dict(_strict_fields(value, GuidedNpmWorkerRequest))
        execution_payload = value["execution_request"]
        execution = deserialize_guided_npm_production_execution_request(execution_payload)
        value["execution_request"] = execution
        if value["application_build_identity"] != _canonical(execution.application_build_identity):
            raise ValueError("worker_request_build_mismatch")
        value["application_build_identity"] = execution.application_build_identity
        value["live_freshness_evidence"] = _freshness(value["live_freshness_evidence"])
        result = GuidedNpmWorkerRequest(**value)
        verify_guided_npm_worker_request(result)
        return result
    except Exception as exc:
        raise ValueError("worker_request_serialization_invalid") from exc


def decode_canonical_guided_npm_worker_request_bytes(content: bytes) -> GuidedNpmWorkerRequest:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite:{value}")),
        )
        request = deserialize_guided_npm_worker_request(payload)
        if canonical_guided_npm_worker_request_bytes(request) != content:
            raise ValueError("worker_request_noncanonical")
        return request
    except Exception as exc:
        raise ValueError("worker_request_noncanonical") from exc


def build_guided_npm_worker_request(
    execution_request: GuidedNpmProductionExecutionRequest,
    live_freshness_evidence: GuidedNpmLiveSourceFreshnessEvidence,
) -> GuidedNpmWorkerRequest:
    verify_guided_npm_production_execution_request(execution_request)
    verify_guided_npm_live_source_freshness_evidence(live_freshness_evidence, execution_request)
    output = execution_request.output_runtime_projection
    module = ntpath if output.output_base_path_style == "windows_drive" else posixpath
    value = GuidedNpmWorkerRequest(
        GUIDED_NPM_WORKER_REQUEST_SCHEMA_NAME,
        GUIDED_NPM_WORKER_REQUEST_SCHEMA_VERSION,
        GUIDED_NPM_WORKER_REQUEST_CONTRACT_VERSION,
        execution_request.canonical_execution_request_identity,
        execution_request.source_claim_receipt_identity,
        execution_request.source_startup_payload_identity,
        execution_request.application_build_identity,
        execution_request.guided_plan_identity,
        execution_request.validation_revision,
        execution_request.startup_artifact_path,
        execution_request.startup_artifact_sha256,
        execution_request.startup_artifact_size_bytes,
        execution_request,
        live_freshness_evidence,
        output.run_directory_path,
        module.join(output.run_directory_path, GUIDED_NPM_WORKER_REQUEST_FILENAME),
        GUIDED_NPM_WORKER_REQUEST_FILENAME,
        "constructed_for_worker",
        "not_launched",
        "not_started",
        "not_available",
        False,
        "0" * 64,
    )
    value = replace(value, canonical_worker_request_identity=compute_guided_npm_worker_request_identity(value))
    verify_guided_npm_worker_request(value)
    return value
