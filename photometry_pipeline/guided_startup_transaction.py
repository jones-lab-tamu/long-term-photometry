"""Pure planning for the future Guided startup transaction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import ntpath
import re
from typing import Any

import yaml

from photometry_pipeline.guided_execution_payloads import (
    GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE,
    GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_VERSION,
    GuidedExecutionConfigPayload,
    GuidedExecutionPayloadDerivationResult,
    GuidedExecutionStartupMappingContract,
    GuidedRunnerCandidateManifestPayload,
    compute_guided_execution_config_payload_identity,
    compute_guided_runner_candidate_manifest_payload_identity,
    compute_guided_startup_provenance_seed_identity,
)
from photometry_pipeline.guided_execution_preflight import (
    compute_guided_candidate_preflight_identity,
    compute_guided_roi_preflight_identity,
)
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_correction_payload import (
    GUIDED_PER_ROI_CORRECTION_FILENAME,
    correction_payload_identity,
    serialize_guided_correction_payload,
)
from photometry_pipeline.guided_manifest_verification import (
    GUIDED_CANDIDATE_CONSUMPTION_CONTRACT_VERSION,
)
from photometry_pipeline.guided_production_mapping import (
    ApplicationBuildIdentity,
    build_application_build_identity,
    compute_guided_production_execution_intent_identity,
)
from photometry_pipeline.guided_run_authorization import (
    GuidedRunAuthorizationResult,
    compute_guided_run_authorization_identity,
)
from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingError,
    compute_normalized_recording_description_identity,
    deserialize_normalized_recording_description,
    rebuild_normalized_recording_description_from_intent,
    serialize_normalized_recording_description,
)
from photometry_pipeline.guided_npm_execution_authority import (
    compute_guided_npm_execution_authority_identity,
    verify_guided_npm_execution_authority,
)
from photometry_pipeline.guided_npm_startup_bridge import GuidedStartupAuthority
from photometry_pipeline.guided_production_mapping import (
    compute_guided_npm_production_execution_intent_identity,
)


GUIDED_STARTUP_TRANSACTION_SCHEMA_NAME = "guided_startup_transaction"
GUIDED_STARTUP_TRANSACTION_SCHEMA_VERSION = "v1"
LEGACY_GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION = (
    "guided_startup_transaction.4J14o.pure.v1"
)
GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION = (
    "guided_startup_transaction.4J16k49.native_correction.v2"
)
GUIDED_STARTUP_STATUS_SCHEMA_NAME = "guided_startup_status"
GUIDED_STARTUP_STATUS_SCHEMA_VERSION = "v1"
GUIDED_STARTUP_PROVENANCE_SCHEMA_NAME = "guided_startup_provenance"
GUIDED_STARTUP_PROVENANCE_SCHEMA_VERSION = "v1"
GUIDED_STARTUP_COMMAND_SCHEMA_NAME = "guided_startup_command"
GUIDED_STARTUP_COMMAND_SCHEMA_VERSION = "v1"

GUIDED_STARTUP_STATUS_FILENAME = "guided_startup_status.json"
GUIDED_CANDIDATE_MANIFEST_FILENAME = "guided_candidate_manifest.json"
GUIDED_CONFIG_EFFECTIVE_FILENAME = "config_effective.yaml"
GUIDED_STARTUP_PROVENANCE_FILENAME = "guided_startup_provenance.json"
GUIDED_COMMAND_RECORD_FILENAME = "command_invoked.txt"
GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME = "guided_per_roi_feature_config.json"
# B1: the durable, mandatory Setup-check-authorized recording description.
# Unlike GUIDED_PER_ROI_CORRECTION_FILENAME (native-correction runs only),
# this is always applicable -- every accepted intent already requires
# source_format == "rwd" (see _gate_issue below).
GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME = (
    "guided_normalized_recording_description.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{16,256}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class GuidedWrapperEntrypointIdentity:
    entrypoint_kind: str
    entrypoint_value: str
    trusted_application_root: str
    wrapper_identity_digest: str
    supported_contract_version: str
    supports_guided_preallocated_run_dir: bool
    supports_guided_candidate_manifest: bool
    trusted_entrypoint: bool
    python_executable: str = "python"


@dataclass(frozen=True)
class GuidedStartupFilesystemPolicy:
    output_base_exists_or_creatable: bool
    output_base_is_directory_or_creatable: bool
    output_base_overlaps_source: bool
    output_base_is_completed_run_root: bool
    output_base_is_guided_diagnostic_cache_root: bool
    output_base_is_protected_ineligible_root: bool
    planned_child_directly_under_base: bool
    planned_child_already_exists: bool
    overwrite_requested: bool
    protected_root_context_complete: bool


@dataclass(frozen=True)
class GuidedStartupTransactionRequest:
    startup_authority: GuidedStartupAuthority
    payload_result: GuidedExecutionPayloadDerivationResult
    startup_mapping_contract: GuidedExecutionStartupMappingContract
    application_build_identity: ApplicationBuildIdentity
    current_guided_revision: int
    explicit_user_run_transition: bool
    output_base_canonical: str
    source_root_canonical: str
    planned_run_id: str
    planned_allocated_run_dir: str
    wrapper_entrypoint: GuidedWrapperEntrypointIdentity
    one_shot_consumption_token: str
    one_shot_token_current: bool
    one_shot_token_unused: bool
    current_time_utc_iso: str
    filesystem_policy: GuidedStartupFilesystemPolicy


@dataclass(frozen=True)
class GuidedStartupIssue:
    category: str
    section: str
    message: str


@dataclass(frozen=True)
class SerializedStartupArtifact:
    content_bytes: bytes
    byte_sha256: str
    semantic_identity: str


@dataclass(frozen=True)
class GuidedStartupCommandPlan:
    argv: tuple[str, ...]
    canonical_command_identity: str
    command_record_bytes: bytes
    command_record_sha256: str
    executable_now: bool = False
    requires_future_wrapper_preallocated_mode: bool = True


@dataclass(frozen=True)
class GuidedStartupIdentityBundle:
    authorization_identity: str
    production_intent_identity: str
    config_payload_identity: str
    candidate_manifest_payload_identity: str
    provenance_seed_identity: str
    config_bytes_sha256: str
    candidate_manifest_bytes_sha256: str
    startup_status_bytes_sha256: str
    provenance_identity_basis_sha256: str
    startup_provenance_bytes_sha256: str
    command_identity: str
    command_record_sha256: str
    normalized_recording_description_bytes_sha256: str
    startup_transaction_identity: str


@dataclass(frozen=True)
class GuidedStartupPlanResult:
    status: str
    ok: bool
    ready_for_effectful_startup: bool
    blocking_issues: tuple[GuidedStartupIssue, ...]
    startup_status_bytes: bytes | None
    candidate_manifest_bytes: bytes | None
    config_effective_bytes: bytes | None
    startup_provenance_bytes: bytes | None
    command_record_bytes: bytes | None
    normalized_recording_description_bytes: bytes | None
    planned_command_argv: tuple[str, ...]
    command_plan: GuidedStartupCommandPlan | None
    identities: GuidedStartupIdentityBundle | None
    no_files_written: bool = True
    no_directories_created: bool = True
    no_runner_invoked: bool = True
    no_gui_mutation: bool = True


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_identity(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + encode_canonical_value(value)
    ).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _plain_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _plain_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported config payload value: {type(value).__name__}")


def _manifest_document(
    payload: GuidedRunnerCandidateManifestPayload,
) -> dict[str, Any]:
    return {
        "manifest_schema_name": payload.manifest_schema_name,
        "manifest_schema_version": payload.manifest_schema_version,
        "candidate_consumption_contract_version": (
            payload.candidate_consumption_contract_version
        ),
        "source_root_canonical": payload.source_root_canonical,
        "source_candidate_set_digest": payload.source_candidate_set_digest,
        "source_candidate_content_digest": payload.source_candidate_content_digest,
        "candidate_files": [
            {
                "canonical_relative_path": item.canonical_relative_path,
                "size_bytes": item.size_bytes,
                "sha256_content_digest": item.sha256_content_digest,
            }
            for item in payload.candidate_files
        ],
        "parser_contract_digest": payload.parser_contract_digest,
        "discovered_roi_ids": list(payload.discovered_roi_ids),
        "included_roi_ids": list(payload.included_roi_ids),
        "excluded_roi_ids": list(payload.excluded_roi_ids),
        "strict_roi_inventory_digest": payload.strict_roi_inventory_digest,
        "candidate_preflight_identity": payload.candidate_preflight_identity,
        "roi_preflight_identity": payload.roi_preflight_identity,
        "canonical_candidate_manifest_payload_identity": (
            payload.canonical_candidate_manifest_payload_identity
        ),
    }


def serialize_guided_candidate_manifest_payload_to_bytes(
    payload: GuidedRunnerCandidateManifestPayload,
) -> SerializedStartupArtifact:
    if not isinstance(payload, GuidedRunnerCandidateManifestPayload):
        raise TypeError("payload must be GuidedRunnerCandidateManifestPayload.")
    identity = compute_guided_runner_candidate_manifest_payload_identity(payload)
    if identity != payload.canonical_candidate_manifest_payload_identity:
        raise ValueError("Candidate manifest payload identity is inconsistent.")
    content = _json_bytes(_manifest_document(payload))
    return SerializedStartupArtifact(content, _sha256_bytes(content), identity)


def serialize_guided_config_payload_to_yaml_bytes(
    payload: GuidedExecutionConfigPayload,
) -> SerializedStartupArtifact:
    if not isinstance(payload, GuidedExecutionConfigPayload):
        raise TypeError("payload must be GuidedExecutionConfigPayload.")
    identity = compute_guided_execution_config_payload_identity(payload)
    if identity != payload.canonical_config_payload_identity:
        raise ValueError("Config payload identity is inconsistent.")
    names = tuple(item.name for item in payload.values)
    if len(names) != len(set(names)):
        raise ValueError("Config payload contains duplicate fields.")
    document = {item.name: _plain_value(item.value) for item in payload.values}
    content = yaml.safe_dump(
        document,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode("utf-8")
    return SerializedStartupArtifact(content, _sha256_bytes(content), identity)


def build_guided_startup_command_plan(
    request: GuidedStartupTransactionRequest,
    *,
    config_path: str | None = None,
    manifest_path: str | None = None,
) -> GuidedStartupCommandPlan:
    run_dir = request.planned_allocated_run_dir
    config_path = config_path or ntpath.join(
        run_dir, GUIDED_CONFIG_EFFECTIVE_FILENAME
    )
    manifest_path = manifest_path or ntpath.join(
        run_dir, GUIDED_CANDIDATE_MANIFEST_FILENAME
    )
    wrapper = request.wrapper_entrypoint
    authority = request.startup_authority
    sessions_per_hour = authority.sessions_per_hour
    execution_mode = authority.execution_mode
    argv = (
        wrapper.python_executable,
        wrapper.entrypoint_value,
        "--input",
        request.source_root_canonical,
        "--out",
        run_dir,
        "--config",
        config_path,
        "--format",
        authority.source_format,
        "--mode",
        execution_mode,
        "--run-type",
        "full",
        "--sessions-per-hour",
        str(sessions_per_hour),
        "--guided-candidate-manifest",
        manifest_path,
        "--guided-preallocated-run-dir",
    )
    command_identity = _canonical_identity(
        "guided-startup-command:v1",
        {
            "schema_name": GUIDED_STARTUP_COMMAND_SCHEMA_NAME,
            "schema_version": GUIDED_STARTUP_COMMAND_SCHEMA_VERSION,
            "argv": argv,
            "wrapper_identity": wrapper.wrapper_identity_digest,
        },
    )
    record = "".join(f"{argument}\n" for argument in argv).encode("utf-8")
    return GuidedStartupCommandPlan(
        argv=argv,
        canonical_command_identity=command_identity,
        command_record_bytes=record,
        command_record_sha256=_sha256_bytes(record),
    )


def compute_guided_startup_transaction_identity(payload: dict[str, Any]) -> str:
    """Hash the nonrecursive startup identity envelope."""
    return _canonical_identity("guided-startup-transaction:v1", payload)


def _refused(category: str, section: str, message: str) -> GuidedStartupPlanResult:
    return GuidedStartupPlanResult(
        status="refused",
        ok=False,
        ready_for_effectful_startup=False,
        blocking_issues=(GuidedStartupIssue(category, section, message),),
        startup_status_bytes=None,
        candidate_manifest_bytes=None,
        config_effective_bytes=None,
        startup_provenance_bytes=None,
        command_record_bytes=None,
        normalized_recording_description_bytes=None,
        planned_command_argv=(),
        command_plan=None,
        identities=None,
    )


def _build_identity_usable(identity: Any) -> bool:
    if not isinstance(identity, ApplicationBuildIdentity):
        return False
    try:
        rebuilt = build_application_build_identity(
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
    return rebuilt.canonical_identity == identity.canonical_identity


def _config_values(payload: GuidedExecutionConfigPayload) -> dict[str, Any]:
    return {item.name: item.value for item in payload.values}


def _rwd_gate_authority(request: Any):
    """RWD's existing authorization/identity verification, unchanged.
    Returns (intent, auth_identity, intent_identity, candidate_identity,
    roi_identity, application_build_identity) on success, or a
    GuidedStartupIssue on refusal."""
    auth = request.startup_authority.rwd
    if (
        not isinstance(auth, GuidedRunAuthorizationResult)
        or auth.status != "authorized"
        or auth.authorized is not True
        or auth.run_authorization is not True
    ):
        return GuidedStartupIssue(
            "authorization_not_accepted",
            "authorization",
            "Accepted Guided authorization is required.",
        )
    if request.current_guided_revision != auth.authorized_gui_revision:
        return GuidedStartupIssue(
            "guided_revision_stale",
            "authorization",
            "Guided revision changed after authorization.",
        )
    try:
        auth_identity = compute_guided_run_authorization_identity(auth)
        intent_identity = compute_guided_production_execution_intent_identity(
            auth.production_intent
        )
        candidate_identity = compute_guided_candidate_preflight_identity(
            auth.candidate_preflight_result
        )
        roi_identity = compute_guided_roi_preflight_identity(
            auth.roi_preflight_result
        )
    except Exception:
        return GuidedStartupIssue(
            "authorization_identity_inconsistent",
            "authorization",
            "Authorization proof identities could not be recomputed.",
        )
    if (
        auth_identity != auth.canonical_authorization_identity
        or intent_identity != auth.production_intent_identity
        or candidate_identity != auth.candidate_preflight_identity
        or roi_identity != auth.roi_preflight_identity
    ):
        return GuidedStartupIssue(
            "authorization_identity_inconsistent",
            "authorization",
            "Authorization proof identities are inconsistent.",
        )
    if any(
        value is not True
        for value in (
            auth.no_files_written,
            auth.no_directories_created,
            auth.no_artifacts_created,
            auth.no_output_allocated,
            auth.no_run_id_allocated,
            auth.no_config_or_argv_generated,
            auth.no_runner_invoked,
        )
    ):
        return GuidedStartupIssue(
            "payload_side_effect_assertion_invalid",
            "payload",
            "Payload side-effect assertions are invalid.",
        )
    return (
        auth.production_intent,
        auth_identity,
        intent_identity,
        candidate_identity,
        roi_identity,
        auth.application_build_identity,
    )


def _npm_gate_authority(request: Any):
    """NPM's authorization/identity verification, using only NPM's own
    existing identity/verification functions. verify_guided_npm_execution_
    authority's round trip already proves the authority's full internal
    identity chain (ROI/correction/Feature-Detection sub-authorities
    included) in one call -- NPM's authority object is pure by
    construction (no filesystem access, no allocation), so there is no
    side-effect-flag equivalent to RWD's auth.no_files_written etc. to
    check."""
    intent = request.startup_authority.npm_intent
    authority = request.startup_authority.npm_authority
    if request.current_guided_revision != intent.validation_revision:
        return GuidedStartupIssue(
            "guided_revision_stale",
            "authorization",
            "Guided revision changed after authorization.",
        )
    try:
        intent_identity = compute_guided_npm_production_execution_intent_identity(
            intent
        )
    except Exception:
        return GuidedStartupIssue(
            "authorization_identity_inconsistent",
            "authorization",
            "Authorization proof identities could not be recomputed.",
        )
    if intent_identity != intent.canonical_intent_identity:
        return GuidedStartupIssue(
            "authorization_identity_inconsistent",
            "authorization",
            "Authorization proof identities are inconsistent.",
        )
    try:
        verify_guided_npm_execution_authority(authority)
        auth_identity = compute_guided_npm_execution_authority_identity(authority)
    except Exception:
        return GuidedStartupIssue(
            "authorization_identity_inconsistent",
            "authorization",
            "Authorization proof identities could not be recomputed.",
        )
    if (
        auth_identity != authority.canonical_authority_identity
        or authority.source_production_intent_identity != intent_identity
    ):
        return GuidedStartupIssue(
            "authorization_identity_inconsistent",
            "authorization",
            "Authorization proof identities are inconsistent.",
        )
    return (
        intent,
        auth_identity,
        intent_identity,
        intent.source_snapshot_identity,
        authority.roi_authority.canonical_roi_authority_identity,
        intent.application_build_identity.canonical_identity,
    )


def _gate_issue(request: Any) -> GuidedStartupIssue | None:
    if not isinstance(request, GuidedStartupTransactionRequest):
        return GuidedStartupIssue(
            "startup_request_invalid", "startup", "Startup request is invalid."
        )
    if request.explicit_user_run_transition is not True:
        return GuidedStartupIssue(
            "explicit_run_transition_required",
            "authority",
            "An explicit Guided Run transition is required.",
        )
    if not _TOKEN_RE.fullmatch(request.one_shot_consumption_token):
        return GuidedStartupIssue(
            "one_shot_token_invalid", "authority", "One-shot token is invalid."
        )
    if (
        request.one_shot_token_current is not True
        or request.one_shot_token_unused is not True
    ):
        return GuidedStartupIssue(
            "one_shot_token_unavailable",
            "authority",
            "One-shot token is stale or already consumed.",
        )
    if not _RUN_ID_RE.fullmatch(request.planned_run_id):
        return GuidedStartupIssue(
            "planned_run_id_invalid", "allocation_plan", "Planned run ID is invalid."
        )
    authority = request.startup_authority
    if not isinstance(authority, GuidedStartupAuthority):
        return GuidedStartupIssue(
            "authorization_not_accepted",
            "authorization",
            "Accepted Guided authorization is required.",
        )
    resolved = (
        _npm_gate_authority(request)
        if authority.is_npm
        else _rwd_gate_authority(request)
    )
    if isinstance(resolved, GuidedStartupIssue):
        return resolved
    intent, auth_identity, intent_identity, candidate_identity, roi_identity, build_id_from_authority = resolved
    payload = request.payload_result
    if (
        not isinstance(payload, GuidedExecutionPayloadDerivationResult)
        or payload.status != GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE
        or payload.ok is not True
        or payload.runnable is not False
    ):
        return GuidedStartupIssue(
            "payload_status_unsupported",
            "payload",
            "Expected nonrunnable startup-limited payloads.",
        )
    if payload.runner_request is not None or payload.runner_request_identity is not None:
        return GuidedStartupIssue(
            "payload_runner_request_prohibited",
            "payload",
            "Payload result must not contain a runner request.",
        )
    if (
        len(payload.limiting_issues) != 1
        or payload.limiting_issues[0].category != "startup_transaction_unavailable"
        or payload.blocking_issues
    ):
        return GuidedStartupIssue(
            "payload_limiter_unsupported",
            "payload",
            "Payload result has an unsupported limiter or blocking issue.",
        )
    if (
        payload.config_payload is None
        or payload.candidate_manifest_payload is None
        or payload.provenance_seed is None
    ):
        return GuidedStartupIssue(
            "payload_incomplete", "payload", "Startup payload bundle is incomplete."
        )
    contract = request.startup_mapping_contract
    if (
        not isinstance(contract, GuidedExecutionStartupMappingContract)
        or contract.contract_version
        != GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_VERSION
        or contract.exact_candidate_manifest_consumption_capable is not True
        or contract.exact_roi_consumption_capable is not True
    ):
        return GuidedStartupIssue(
            "startup_mapping_contract_unsupported",
            "payload",
            "Post-4J14l exact-consumption startup mapping is required.",
        )
    if any(
        value is not True
        for value in (
            payload.no_files_written,
            payload.no_directories_created,
            payload.no_artifacts_created,
            payload.no_output_allocated,
            payload.no_run_id_allocated,
            payload.no_config_file_generated,
            payload.no_argv_generated,
            payload.no_runner_invoked,
        )
    ):
        return GuidedStartupIssue(
            "payload_side_effect_assertion_invalid",
            "payload",
            "Payload side-effect assertions are invalid.",
        )
    manifest = payload.candidate_manifest_payload
    seed = payload.provenance_seed
    if not isinstance(request.application_build_identity, ApplicationBuildIdentity):
        return GuidedStartupIssue(
            "application_build_identity_mismatch",
            "build_identity",
            "Application build identity is invalid or inconsistent.",
        )
    build_id = request.application_build_identity.canonical_identity
    if (
        not _build_identity_usable(request.application_build_identity)
        or build_id != build_id_from_authority
        or build_id != seed.application_build_identity
    ):
        return GuidedStartupIssue(
            "application_build_identity_mismatch",
            "build_identity",
            "Application build identity is invalid or inconsistent.",
        )
    try:
        config_identity = compute_guided_execution_config_payload_identity(
            payload.config_payload
        )
        manifest_identity = (
            compute_guided_runner_candidate_manifest_payload_identity(manifest)
        )
        seed_identity = compute_guided_startup_provenance_seed_identity(seed)
    except Exception:
        return GuidedStartupIssue(
            "payload_identity_inconsistent",
            "payload",
            "Payload identities could not be recomputed.",
        )
    if (
        config_identity != payload.config_payload_identity
        or config_identity != payload.config_payload.canonical_config_payload_identity
        or manifest_identity != payload.candidate_manifest_payload_identity
        or manifest_identity
        != manifest.canonical_candidate_manifest_payload_identity
        or seed_identity != payload.provenance_seed_identity
        or seed_identity != seed.canonical_provenance_seed_identity
        or seed.authorization_identity != auth_identity
        or seed.production_intent_identity != intent_identity
        or seed.candidate_preflight_identity != candidate_identity
        or seed.roi_preflight_identity != roi_identity
        or seed.config_payload_identity != config_identity
        or seed.candidate_manifest_payload_identity != manifest_identity
        or manifest.candidate_preflight_identity != candidate_identity
        or manifest.roi_preflight_identity != roi_identity
    ):
        return GuidedStartupIssue(
            "payload_identity_inconsistent",
            "payload",
            "Payload identities are inconsistent.",
        )
    if authority.is_npm:
        source_binding_ok = (
            request.source_root_canonical == intent.source_root_canonical
            and request.source_root_canonical == manifest.source_root_canonical
            and request.output_base_canonical == intent.output_policy.output_base_canonical
            and manifest.source_candidate_set_digest == intent.source_snapshot_set_identity
            and manifest.source_candidate_content_digest == intent.source_snapshot_content_identity
            and manifest.parser_contract_digest == intent.parser_policy_identity
            and manifest.discovered_roi_ids == intent.discovered_roi_ids
            and manifest.included_roi_ids == intent.selected_roi_ids
            and manifest.excluded_roi_ids == intent.excluded_roi_ids
        )
    else:
        source_binding_ok = (
            request.source_root_canonical == intent.input_source.source_root_canonical
            and request.source_root_canonical == manifest.source_root_canonical
            and request.output_base_canonical == intent.output_policy.output_base_canonical
            and manifest.source_candidate_set_digest == intent.input_source.source_candidate_set_digest
            and manifest.source_candidate_content_digest == intent.input_source.source_candidate_content_digest
            and manifest.parser_contract_digest == intent.parser.parser_contract_digest
            and manifest.discovered_roi_ids == intent.roi_scope.discovered_roi_ids
            and manifest.included_roi_ids == intent.roi_scope.included_roi_ids
            and manifest.excluded_roi_ids == intent.roi_scope.excluded_roi_ids
        )
    if not source_binding_ok:
        return GuidedStartupIssue(
            "source_output_binding_mismatch",
            "paths",
            "Source or output path is not bound to authorization.",
        )
    config = _config_values(payload.config_payload)
    if authority.is_npm:
        # NPM has no RWD-only "selection_mode"/"strategy_scope"/global
        # dynamic-fit concepts to recreate (see guided_npm_startup_bridge
        # module docstring); their equivalent guarantees are already
        # proven by GuidedNpmRoiAuthority's and GuidedNpmCorrectionAuthority's
        # own construction-time invariants, verified above via
        # verify_guided_npm_execution_authority.
        npm_correction_values = {
            item.field_name: item.value for item in intent.correction_parameter_values
        }
        first_subset_ok = (
            intent.source_format == "npm"
            and intent.acquisition_mode == "intermittent"
            and intent.execution_mode == "both"
            and intent.run_type == "full"
            and intent.output_policy.overwrite is False
            and intent.output_policy.precreate is False
            and bool(intent.selected_roi_ids)
            and config.get("acquisition_mode") == "intermittent"
            and config.get("tonic_output_mode")
            == npm_correction_values.get("tonic_output_mode")
            and config.get("tonic_timeline_mode")
            == npm_correction_values.get("tonic_timeline_mode")
        )
    else:
        first_subset_ok = (
            intent.input_source.source_format == "rwd"
            and intent.acquisition.acquisition_mode == "intermittent"
            and intent.execution_profile.execution_mode in {"phasic", "tonic", "both"}
            and intent.execution_profile.run_type == "full"
            and intent.execution_profile.traces_only is False
            and intent.output_policy.overwrite is False
            and intent.output_policy.precreate is False
            and intent.roi_scope.selection_mode == "include"
            and bool(intent.roi_scope.included_roi_ids)
            and intent.correction.strategy_scope == "global"
            and config.get("acquisition_mode") == "intermittent"
            and config.get("dynamic_fit_mode") == intent.correction.global_dynamic_fit_mode
            and config.get("tonic_output_mode") == intent.correction.global_tonic_output_mode
            and config.get("tonic_timeline_mode") == intent.correction.global_tonic_timeline_mode
        )
    if not first_subset_ok:
        return GuidedStartupIssue(
            "first_subset_contract_unsupported",
            "execution_profile",
            "The selected analysis setup is not supported for this run.",
        )
    if manifest.candidate_consumption_contract_version != (
        GUIDED_CANDIDATE_CONSUMPTION_CONTRACT_VERSION
    ) or (
        request.startup_mapping_contract.supported_mapping_contract_version
        != intent.mapping_contract_version
        or request.startup_mapping_contract.candidate_consumption_contract_version
        != manifest.candidate_consumption_contract_version
    ):
        return GuidedStartupIssue(
            "manifest_contract_unsupported",
            "payload",
            "Manifest-consumption contract is unsupported.",
        )
    policy = request.filesystem_policy
    if not isinstance(policy, GuidedStartupFilesystemPolicy):
        return GuidedStartupIssue(
            "filesystem_policy_invalid", "output", "Filesystem policy is invalid."
        )
    policy_failures = (
        (not policy.output_base_exists_or_creatable, "output_base_unavailable"),
        (
            not policy.output_base_is_directory_or_creatable,
            "output_base_not_directory",
        ),
        (policy.output_base_overlaps_source, "output_source_overlap"),
        (policy.output_base_is_completed_run_root, "completed_run_root_prohibited"),
        (
            policy.output_base_is_guided_diagnostic_cache_root,
            "diagnostic_cache_root_prohibited",
        ),
        (
            policy.output_base_is_protected_ineligible_root,
            "protected_output_root_prohibited",
        ),
        (
            not policy.protected_root_context_complete,
            "protected_root_context_incomplete",
        ),
        (
            not policy.planned_child_directly_under_base,
            "planned_child_not_direct",
        ),
        (policy.planned_child_already_exists, "planned_child_exists"),
        (policy.overwrite_requested, "overwrite_prohibited"),
    )
    for failed, category in policy_failures:
        if failed:
            return GuidedStartupIssue(
                category, "output", "Output allocation plan is unsafe."
            )
    wrapper = request.wrapper_entrypoint
    if (
        not isinstance(wrapper, GuidedWrapperEntrypointIdentity)
        or not wrapper.entrypoint_value.strip()
        or not wrapper.trusted_application_root.strip()
        or not _SHA256_RE.fullmatch(wrapper.wrapper_identity_digest)
        or not wrapper.supported_contract_version.strip()
        or wrapper.supports_guided_preallocated_run_dir is not True
        or wrapper.supports_guided_candidate_manifest is not True
        or wrapper.trusted_entrypoint is not True
    ):
        return GuidedStartupIssue(
            "wrapper_contract_unsupported",
            "wrapper",
            "Trusted future Guided wrapper capabilities are required.",
        )
    return None


def plan_guided_startup_transaction(
    request: GuidedStartupTransactionRequest,
) -> GuidedStartupPlanResult:
    issue = _gate_issue(request)
    if issue is not None:
        return _refused(issue.category, issue.section, issue.message)
    authority = request.startup_authority
    payload = request.payload_result
    intent = authority.npm_intent if authority.is_npm else authority.rwd.production_intent
    per_roi_correction_strategy_map = authority.per_roi_correction_strategy_map
    if authority.is_npm:
        # NPM was never part of the RWD legacy (pre-native-correction)
        # era: it always resolves per-ROI, so this is a truthful fixed
        # condition, not an invented strategy-map version.
        native_current = True
        positive_legacy = False
    else:
        correction = authority.rwd.production_intent.correction
        native_current = bool(correction.production_strategy_map_version)
        positive_legacy = bool(
            not correction.production_strategy_map_version
            and not correction.per_roi_production_strategy_map
        )
    effective_startup_contract_version = (
        GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
        if native_current
        else LEGACY_GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
        if positive_legacy
        else GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
    )
    native_correction_bytes = None
    native_correction_sha256 = None
    native_correction_identity = None
    if native_current:
        try:
            native_correction_bytes = serialize_guided_correction_payload(
                authority.included_roi_ids,
                per_roi_correction_strategy_map,
            )
            native_correction_sha256 = _sha256_bytes(native_correction_bytes)
            native_correction_identity = json.loads(native_correction_bytes)[
                "canonical_correction_payload_identity"
            ]
        except Exception as exc:
            return _refused(
                "native_correction_payload_invalid",
                "correction",
                f"Native correction payload is invalid: {exc}",
            )
    manifest_artifact = serialize_guided_candidate_manifest_payload_to_bytes(
        payload.candidate_manifest_payload
    )
    config_artifact = serialize_guided_config_payload_to_yaml_bytes(
        payload.config_payload
    )
    command = build_guided_startup_command_plan(request)

    # B1: verify the exact authorized normalized recording description.
    # RWD rebuilds it from scratch from granular intent facts
    # (verify-by-rebuild, zero filesystem I/O -- every field this reads was
    # already frozen onto the intent at authorization time). NPM's intent
    # already carries the complete, already-serialized normalized
    # recording payload verbatim -- deserializing and recomputing its
    # identity is the equivalent proof, using the same existing
    # deserialize/identity functions, not a rebuild-from-scratch (NPM has
    # no granular facts to rebuild from at this layer; they were already
    # consumed earlier, at validation time). A mismatch either way means
    # the authorized intent's own frozen facts are no longer internally
    # consistent with its own normalized_recording_description_identity,
    # which should be impossible for an accepted authorization -- refused
    # rather than silently proceeding.
    if authority.is_npm:
        try:
            rebuilt_normalized_recording = deserialize_normalized_recording_description(
                json.loads(intent.normalized_recording_payload_json)
            )
            rebuilt_normalized_recording_identity = (
                compute_normalized_recording_description_identity(
                    rebuilt_normalized_recording
                )
            )
        except (NormalizedRecordingError, ValueError, TypeError, KeyError) as exc:
            return _refused(
                "normalized_recording_description_invalid",
                "normalized_recording",
                f"The authorized recording description could not be verified: {exc}",
            )
    else:
        try:
            rebuilt_normalized_recording = rebuild_normalized_recording_description_from_intent(
                intent
            )
            rebuilt_normalized_recording_identity = (
                compute_normalized_recording_description_identity(
                    rebuilt_normalized_recording
                )
            )
        except NormalizedRecordingError as exc:
            return _refused(
                "normalized_recording_description_invalid",
                "normalized_recording",
                f"The authorized recording description could not be rebuilt: {exc}",
            )
    if rebuilt_normalized_recording_identity != intent.normalized_recording_description_identity:
        return _refused(
            "normalized_recording_description_mismatch",
            "normalized_recording",
            "The rebuilt recording description does not match the authorized "
            "intent's own normalized_recording_description_identity.",
        )
    normalized_recording_bytes = _json_bytes(
        serialize_normalized_recording_description(rebuilt_normalized_recording)
    )
    normalized_recording_sha256 = _sha256_bytes(normalized_recording_bytes)
    status_document = {
        "schema_name": GUIDED_STARTUP_STATUS_SCHEMA_NAME,
        "schema_version": GUIDED_STARTUP_STATUS_SCHEMA_VERSION,
        "startup_contract_version": effective_startup_contract_version,
        "status": "allocated_preparation_pending",
        "created_utc": request.current_time_utc_iso,
        "run_id": request.planned_run_id,
        "allocated_run_dir": request.planned_allocated_run_dir,
        "output_base": request.output_base_canonical,
        "source_root": request.source_root_canonical,
        "application_build_identity": authority.application_build_identity,
        "authorization_identity": authority.canonical_authorization_identity,
        "production_intent_identity": authority.production_intent_identity,
        "config_payload_identity": payload.config_payload_identity,
        "candidate_manifest_payload_identity": (
            payload.candidate_manifest_payload_identity
        ),
        "provenance_seed_identity": payload.provenance_seed_identity,
        "runner_started": False,
        "runner_start_uncertain": False,
        "completed_run_claim": False,
    }
    status_bytes = _json_bytes(status_document)
    provenance_basis = {
        "schema_name": GUIDED_STARTUP_PROVENANCE_SCHEMA_NAME,
        "schema_version": GUIDED_STARTUP_PROVENANCE_SCHEMA_VERSION,
        "startup_contract_version": effective_startup_contract_version,
        "state": "prepared_runner_not_started",
        "validation_request_identity": authority.fresh_request_identity,
        "authorization_identity": authority.canonical_authorization_identity,
        "production_intent_identity": authority.production_intent_identity,
        "candidate_preflight_identity": authority.candidate_preflight_identity,
        "roi_preflight_identity": authority.roi_preflight_identity,
        "config_payload_identity": payload.config_payload_identity,
        "candidate_manifest_payload_identity": (
            payload.candidate_manifest_payload_identity
        ),
        "provenance_seed_identity": payload.provenance_seed_identity,
        "serialized_config_sha256": config_artifact.byte_sha256,
        "serialized_manifest_sha256": manifest_artifact.byte_sha256,
        "application_build_identity": authority.application_build_identity,
        "production_mapping_contract_version": authority.mapping_contract_version,
        "startup_mapping_contract_version": (
            request.startup_mapping_contract.contract_version
        ),
        "manifest_consumption_contract_version": (
            payload.candidate_manifest_payload.candidate_consumption_contract_version
        ),
        "runner_contract_version": authority.runner_contract_version,
        "wrapper_contract_version": (
            request.wrapper_entrypoint.supported_contract_version
        ),
        "run_id": request.planned_run_id,
        "allocated_run_dir": request.planned_allocated_run_dir,
        "output_base": request.output_base_canonical,
        "source_root": request.source_root_canonical,
        "wrapper_entrypoint_identity": (
            request.wrapper_entrypoint.wrapper_identity_digest
        ),
        "command_identity": command.canonical_command_identity,
        "command_record_sha256": command.command_record_sha256,
        "runner_started": False,
        "runner_start_uncertain": False,
        "completed_run_claim": False,
        "native_correction_schema_version": (
            "v1" if native_current else None
        ),
        "native_correction_payload_identity": native_correction_identity,
        "serialized_native_correction_sha256": native_correction_sha256,
        "normalized_recording_description_identity": (
            intent.normalized_recording_description_identity
        ),
        "serialized_normalized_recording_description_sha256": (
            normalized_recording_sha256
        ),
    }
    provenance_basis_bytes = _json_bytes(provenance_basis)
    provenance_basis_hash = _sha256_bytes(provenance_basis_bytes)
    identity_envelope = {
        "startup_contract_version": effective_startup_contract_version,
        "authorization_identity": authority.canonical_authorization_identity,
        "production_intent_identity": authority.production_intent_identity,
        "candidate_preflight_identity": authority.candidate_preflight_identity,
        "roi_preflight_identity": authority.roi_preflight_identity,
        "config_payload_identity": payload.config_payload_identity,
        "candidate_manifest_payload_identity": (
            payload.candidate_manifest_payload_identity
        ),
        "provenance_seed_identity": payload.provenance_seed_identity,
        "serialized_config_sha256": config_artifact.byte_sha256,
        "serialized_manifest_sha256": manifest_artifact.byte_sha256,
        "startup_status_sha256": _sha256_bytes(status_bytes),
        "provenance_identity_basis_sha256": provenance_basis_hash,
        "command_identity": command.canonical_command_identity,
        "command_record_sha256": command.command_record_sha256,
        "normalized_recording_description_bytes_sha256": normalized_recording_sha256,
        "application_build_identity": authority.application_build_identity,
        "wrapper_identity": request.wrapper_entrypoint.wrapper_identity_digest,
        "run_id": request.planned_run_id,
        "allocated_run_dir": request.planned_allocated_run_dir,
        "output_base": request.output_base_canonical,
        "source_root": request.source_root_canonical,
        "created_utc": request.current_time_utc_iso,
    }
    transaction_identity = compute_guided_startup_transaction_identity(
        identity_envelope
    )
    provenance_document = {
        **provenance_basis,
        "startup_transaction_identity": transaction_identity,
        "identity_basis_provenance_sha256": provenance_basis_hash,
    }
    provenance_bytes = _json_bytes(provenance_document)
    identities = GuidedStartupIdentityBundle(
        authorization_identity=authority.canonical_authorization_identity,
        production_intent_identity=authority.production_intent_identity,
        config_payload_identity=payload.config_payload_identity,
        candidate_manifest_payload_identity=(
            payload.candidate_manifest_payload_identity
        ),
        provenance_seed_identity=payload.provenance_seed_identity,
        config_bytes_sha256=config_artifact.byte_sha256,
        candidate_manifest_bytes_sha256=manifest_artifact.byte_sha256,
        startup_status_bytes_sha256=_sha256_bytes(status_bytes),
        provenance_identity_basis_sha256=provenance_basis_hash,
        startup_provenance_bytes_sha256=_sha256_bytes(provenance_bytes),
        command_identity=command.canonical_command_identity,
        command_record_sha256=command.command_record_sha256,
        normalized_recording_description_bytes_sha256=normalized_recording_sha256,
        startup_transaction_identity=transaction_identity,
    )
    return GuidedStartupPlanResult(
        status="planned_non_effectful",
        ok=True,
        ready_for_effectful_startup=True,
        blocking_issues=(),
        startup_status_bytes=status_bytes,
        candidate_manifest_bytes=manifest_artifact.content_bytes,
        config_effective_bytes=config_artifact.content_bytes,
        startup_provenance_bytes=provenance_bytes,
        command_record_bytes=command.command_record_bytes,
        normalized_recording_description_bytes=normalized_recording_bytes,
        planned_command_argv=command.argv,
        command_plan=command,
        identities=identities,
    )
