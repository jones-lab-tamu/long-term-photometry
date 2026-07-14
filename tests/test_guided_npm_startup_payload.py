from __future__ import annotations

import builtins
import copy
import inspect
import math
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import photometry_pipeline.guided_npm_startup_payload as startup_module
import photometry_pipeline.guided_normalized_recording as normalized_module
import photometry_pipeline.guided_npm_authorization as authorization_module
import photometry_pipeline.guided_npm_execution_authority as authority_module
import photometry_pipeline.io.npm_contract as npm_contract_module
import photometry_pipeline.io.npm_source_snapshot as source_snapshot_module
from photometry_pipeline.guided_npm_execution_authority import (
    compute_guided_npm_authorized_session_identity,
)
from photometry_pipeline.guided_npm_authorization import (
    compute_guided_npm_execution_authorization_identity,
    compute_guided_npm_verified_source_content_identity,
    compute_guided_npm_verified_source_file_identity,
    compute_guided_npm_verified_source_sequence_identity,
    compute_guided_npm_verified_source_set_identity,
    compute_guided_npm_verified_source_snapshot_identity,
)
from photometry_pipeline.guided_npm_startup_payload import (
    GUIDED_NPM_STARTUP_DEFERRED_EXECUTION_CAPABILITIES,
    GUIDED_NPM_STARTUP_PAYLOAD_CONTRACT_VERSION,
    GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME,
    GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION,
    GuidedNpmStartupPayload,
    GuidedNpmStartupPayloadFailure,
    build_guided_npm_startup_payload,
    compute_guided_npm_startup_execution_projection_identity,
    compute_guided_npm_startup_payload_identity,
    compute_guided_npm_startup_session_identity,
    compute_guided_npm_startup_session_sequence_identity,
    compute_guided_npm_startup_source_projection_identity,
    deserialize_guided_npm_startup_payload,
    serialize_guided_npm_startup_payload,
    verify_guided_npm_startup_payload,
)

from tests.test_guided_npm_authorization import _authorize, _rebind_sessions
from tests.test_guided_npm_execution_authority import (
    _accepted_authority,
    _accepted_two_session_authority,
)


def _pair(tmp_path: Path, *, two_sessions: bool = True):
    authority = (
        _accepted_two_session_authority(tmp_path)
        if two_sessions
        else _accepted_authority(tmp_path)
    )
    authorization = _authorize(authority)
    return authorization, authority


def _payload(tmp_path: Path) -> GuidedNpmStartupPayload:
    authorization, authority = _pair(tmp_path)
    result = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(result, GuidedNpmStartupPayload)
    return result


def _issue(result, category: str):
    assert isinstance(result, GuidedNpmStartupPayloadFailure)
    assert result.blocking_issues[0].category == category
    return result.blocking_issues[0]


def _reidentify_authorization(authorization, **changes):
    candidate = replace(
        authorization,
        **changes,
        canonical_authorization_identity="0" * 64,
    )
    return replace(
        candidate,
        canonical_authorization_identity=(
            compute_guided_npm_execution_authorization_identity(candidate)
        ),
    )


def _unsafe_replace(instance, **changes):
    clone = object.__new__(type(instance))
    for name, value in instance.__dict__.items():
        object.__setattr__(clone, name, changes.get(name, value))
    return clone


def _outer_reidentified_source_tamper(
    payload,
    *,
    verified_changes=None,
    session_changes=None,
    source_changes=None,
):
    verified_changes = verified_changes or {}
    session_changes = session_changes or {}
    source_changes = source_changes or {}
    sessions = list(payload.source_projection.ordered_sessions)
    verified = _unsafe_replace(
        sessions[0].verified_source_file,
        **verified_changes,
    )
    session = _unsafe_replace(
        sessions[0],
        verified_source_file=verified,
        canonical_startup_session_identity="0" * 64,
        **session_changes,
    )
    session = _unsafe_replace(
        session,
        canonical_startup_session_identity=(
            compute_guided_npm_startup_session_identity(session)
        ),
    )
    sessions[0] = session
    source = _unsafe_replace(
        payload.source_projection,
        ordered_sessions=tuple(sessions),
        ordered_session_sequence_identity=(
            compute_guided_npm_startup_session_sequence_identity(tuple(sessions))
        ),
        canonical_source_projection_identity="0" * 64,
        **source_changes,
    )
    source = _unsafe_replace(
        source,
        canonical_source_projection_identity=(
            compute_guided_npm_startup_source_projection_identity(source)
        ),
    )
    execution = _unsafe_replace(
        payload.execution_projection,
        source_projection_identity=source.canonical_source_projection_identity,
        canonical_execution_projection_identity="0" * 64,
    )
    execution = _unsafe_replace(
        execution,
        canonical_execution_projection_identity=(
            compute_guided_npm_startup_execution_projection_identity(execution)
        ),
    )
    altered = _unsafe_replace(
        payload,
        source_projection=source,
        execution_projection=execution,
        canonical_startup_payload_identity="0" * 64,
    )
    return _unsafe_replace(
        altered,
        canonical_startup_payload_identity=(
            compute_guided_npm_startup_payload_identity(altered)
        ),
    )


def _rebind_verified_files(authorization, files):
    rebound_files = []
    for position, item in enumerate(files):
        candidate = replace(
            item,
            chronological_position=position,
            canonical_verified_file_identity="0" * 64,
        )
        rebound_files.append(
            replace(
                candidate,
                canonical_verified_file_identity=(
                    compute_guided_npm_verified_source_file_identity(candidate)
                ),
            )
        )
    files = tuple(rebound_files)
    snapshot = replace(
        authorization.verified_source_snapshot,
        ordered_files=files,
        ordered_file_sequence_identity=(
            compute_guided_npm_verified_source_sequence_identity(files)
        ),
        source_set_identity=compute_guided_npm_verified_source_set_identity(files),
        source_content_identity=(
            compute_guided_npm_verified_source_content_identity(files)
        ),
        canonical_verified_snapshot_identity="0" * 64,
    )
    snapshot = replace(
        snapshot,
        canonical_verified_snapshot_identity=(
            compute_guided_npm_verified_source_snapshot_identity(snapshot)
        ),
    )
    return _reidentify_authorization(
        authorization,
        verified_source_snapshot=snapshot,
    )


def test_builds_frozen_in_memory_non_runnable_payload(tmp_path):
    authorization, authority = _pair(tmp_path)
    payload = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(payload, GuidedNpmStartupPayload)
    assert (
        payload.startup_schema_name,
        payload.startup_schema_version,
        payload.startup_contract_version,
    ) == (
        GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME,
        GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION,
        GUIDED_NPM_STARTUP_PAYLOAD_CONTRACT_VERSION,
    )
    assert payload.source_authorization_identity == authorization.canonical_authorization_identity
    assert payload.source_authority_identity == authority.canonical_authority_identity
    assert (payload.source_format, payload.acquisition_mode, payload.run_type) == (
        "npm",
        "intermittent",
        "full",
    )
    assert (payload.payload_status, payload.persistence_status, payload.claim_status) == (
        "constructed_in_memory",
        "not_persisted",
        "not_claimed",
    )
    assert payload.startup_status == "not_materialized"
    assert payload.runnable is False
    with pytest.raises(FrozenInstanceError):
        payload.runnable = True


@pytest.mark.parametrize(
    ("field", "value", "category"),
    (
        ("authorization_schema_name", "other", "authorization_schema_unsupported"),
        ("authorization_status", "persisted", "authorization_state_invalid"),
        ("startup_status", "materialized", "authorization_state_invalid"),
        ("runnable", True, "authorization_state_invalid"),
        ("canonical_authorization_identity", "1" * 64, "authorization_identity_mismatch"),
    ),
)
def test_refuses_invalid_authorization_contract(tmp_path, field, value, category):
    authorization, authority = _pair(tmp_path)
    _issue(
        build_guided_npm_startup_payload(
            _unsafe_replace(authorization, **{field: value}), authority
        ),
        category,
    )


def test_refuses_wrong_authorization_type(tmp_path):
    _, authority = _pair(tmp_path)
    _issue(
        build_guided_npm_startup_payload(object(), authority),
        "authorization_missing_or_invalid",
    )


@pytest.mark.parametrize(
    ("field", "value", "category"),
    (
        ("authority_schema_name", "other", "authority_schema_unsupported"),
        ("authorization_status", "authorized", "authority_state_invalid"),
        ("startup_status", "materialized", "authority_state_invalid"),
        ("runnable", True, "authority_state_invalid"),
        ("canonical_authority_identity", "1" * 64, "authority_identity_mismatch"),
    ),
)
def test_refuses_invalid_authority_contract(tmp_path, field, value, category):
    authorization, authority = _pair(tmp_path)
    _issue(
        build_guided_npm_startup_payload(
            authorization, _unsafe_replace(authority, **{field: value})
        ),
        category,
    )


def test_refuses_wrong_authority_type(tmp_path):
    authorization, _ = _pair(tmp_path)
    _issue(
        build_guided_npm_startup_payload(authorization, object()),
        "authority_missing_or_invalid",
    )


@pytest.mark.parametrize(
    ("field", "value", "category"),
    (
        ("source_request_identity", "1" * 64, "request_identity_mismatch"),
        ("source_production_intent_identity", "1" * 64, "production_intent_identity_mismatch"),
        ("validation_revision", 912, "validation_revision_mismatch"),
        ("guided_plan_identity", "1" * 64, "guided_plan_identity_mismatch"),
        ("execution_mode", "tonic", "execution_mode_mismatch"),
        ("selected_canonical_roi_ids", ("roi-999",), "selected_roi_scope_mismatch"),
        ("correction_authority_identity", "1" * 64, "correction_authority_identity_mismatch"),
        ("feature_authority_identity", "1" * 64, "feature_authority_identity_mismatch"),
        ("output_authority_identity", "1" * 64, "output_authority_identity_mismatch"),
    ),
)
def test_refuses_reidentified_cross_object_mismatch(
    tmp_path, field, value, category
):
    authorization, authority = _pair(tmp_path)
    authorization = _reidentify_authorization(authorization, **{field: value})
    _issue(build_guided_npm_startup_payload(authorization, authority), category)


def test_refuses_authorization_paired_with_different_authority(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    authorization, _ = _pair(tmp_path / "a")
    _, authority_b = _pair(tmp_path / "b")
    _issue(
        build_guided_npm_startup_payload(authorization, authority_b),
        "authorization_authority_mismatch",
    )


def test_authority_order_and_verified_files_bind_one_to_one(tmp_path):
    authorization, authority = _pair(tmp_path)
    payload = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(payload, GuidedNpmStartupPayload)
    sessions = payload.source_projection.ordered_sessions
    assert tuple(item.chronological_position for item in sessions) == (0, 1)
    assert tuple(item.canonical_relative_path for item in sessions) == tuple(
        item.canonical_relative_path for item in authority.sessions
    )
    assert tuple(
        item.verified_source_file.canonical_verified_file_identity
        for item in sessions
    ) == tuple(
        item.canonical_verified_file_identity
        for item in authorization.verified_source_snapshot.ordered_files
    )
    assert payload.execution_projection.ordered_source_paths == tuple(
        item.authorized_absolute_source_reference for item in sessions
    )
    assert payload.execution_projection.ordered_source_relative_paths == tuple(
        item.canonical_relative_path for item in sessions
    )
    assert payload.execution_projection.ordered_source_digests == tuple(
        item.sha256_content_digest for item in sessions
    )


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        ("missing", "verified_source_file_missing"),
        ("extra", "verified_source_file_extra"),
        ("order", "verified_source_file_path_mismatch"),
        ("relative_path", "verified_source_file_path_mismatch"),
        ("absolute_path", "verified_source_file_path_mismatch"),
        ("size", "verified_source_file_size_mismatch"),
        ("digest", "verified_source_file_digest_mismatch"),
    ),
)
def test_refuses_semantically_reidentified_source_correspondence_changes(
    tmp_path, mutation, category
):
    authorization, authority = _pair(tmp_path)
    files = list(authorization.verified_source_snapshot.ordered_files)
    if mutation == "missing":
        files.pop()
    elif mutation == "extra":
        files.append(
            replace(
                files[-1],
                canonical_relative_path="extra.csv",
                authorized_absolute_source_reference="C:/verified/extra.csv",
                inspected_absolute_path="C:/verified/extra.csv",
            )
        )
    elif mutation == "order":
        files.reverse()
    elif mutation == "relative_path":
        files[0] = replace(files[0], canonical_relative_path="changed.csv")
    elif mutation == "absolute_path":
        files[0] = replace(
            files[0],
            authorized_absolute_source_reference="C:/verified/changed.csv",
        )
    elif mutation == "size":
        files[0] = replace(
            files[0],
            expected_size_bytes=files[0].expected_size_bytes + 1,
            observed_size_bytes=files[0].observed_size_bytes + 1,
        )
    elif mutation == "digest":
        files[0] = replace(
            files[0],
            expected_sha256_content_digest="1" * 64,
            observed_sha256_content_digest="1" * 64,
        )
    changed = _rebind_verified_files(authorization, files)
    _issue(build_guided_npm_startup_payload(changed, authority), category)


def test_preserves_complete_recording_session_and_roi_projection(tmp_path):
    authorization, authority = _pair(tmp_path)
    payload = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(payload, GuidedNpmStartupPayload)
    policy = payload.recording_policy
    source_policy = authority.recording_policy
    assert policy.ordered_timestamp_candidates == source_policy.ordered_timestamp_candidates
    assert policy.parser_policy_content_json == source_policy.parser_policy_content_json
    assert policy.support_policy_identity == source_policy.support_policy_identity
    assert policy.target_fs_hz == source_policy.target_fs_hz
    assert policy.sessions_per_hour == source_policy.sessions_per_hour
    assert policy.output_time_basis == source_policy.output_time_basis
    assert tuple(item.resolved_timestamp_column for item in payload.source_projection.ordered_sessions) == (
        "Timestamp",
        "SystemTimestamp",
    )
    assert payload.roi_projection.selected_physical_source_columns == (
        authority.roi_authority.selected_physical_source_columns
    )
    assert payload.roi_projection.excluded_canonical_roi_ids == (
        authority.roi_authority.excluded_canonical_roi_ids
    )


def test_preserves_exact_heterogeneous_per_session_timestamp_columns(tmp_path):
    _, authority = _pair(tmp_path)
    sessions = []
    for session, column in zip(
        authority.sessions, ("SystemTimestamp", "Timestamp")
    ):
        candidate = replace(
            session,
            resolved_timestamp_column=column,
            canonical_session_identity="0" * 64,
        )
        sessions.append(
            replace(
                candidate,
                canonical_session_identity=(
                    compute_guided_npm_authorized_session_identity(candidate)
                ),
            )
        )
    authority = _rebind_sessions(
        authority,
        tuple(sessions),
    )
    authorization = _authorize(authority)
    payload = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(payload, GuidedNpmStartupPayload)
    assert tuple(
        item.resolved_timestamp_column
        for item in payload.source_projection.ordered_sessions
    ) == ("SystemTimestamp", "Timestamp")
    assert "resolved_timestamp_column" not in payload.recording_policy.__dataclass_fields__


def test_preserves_correction_feature_and_future_output_authority(tmp_path):
    _, authority = _pair(tmp_path)
    authorization = _authorize(authority)
    payload = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(payload, GuidedNpmStartupPayload)
    assert payload.correction_projection.per_roi_correction_strategy_map == (
        authority.correction_authority.per_roi_correction_strategy_map
    )
    assert payload.correction_projection.correction_parameter_values == (
        authority.correction_authority.correction_parameter_values
    )
    assert payload.feature_projection.per_roi_feature_event_map == (
        authority.feature_authority.per_roi_feature_event_map
    )
    assert payload.feature_projection.effective_values == authority.feature_authority.effective_values
    assert payload.feature_projection.inactive_for_execution == (
        authority.feature_authority.inactive_for_execution
    )
    assert payload.output_projection.creation_timing == "future_execution_start_only"
    assert payload.output_projection.overwrite is False
    assert payload.output_projection.precreate is False


def test_execution_projection_is_adapter_facing_and_contains_no_later_stage_data(tmp_path):
    payload = _payload(tmp_path)
    execution = payload.execution_projection
    assert execution.deferred_execution_capabilities == (
        GUIDED_NPM_STARTUP_DEFERRED_EXECUTION_CAPABILITIES
    )
    assert execution.parser_policy_identity == payload.recording_policy.parser_policy_identity
    assert execution.session_sequence_identity == (
        payload.source_projection.ordered_session_sequence_identity
    )
    assert execution.source_projection_identity == (
        payload.source_projection.canonical_source_projection_identity
    )
    assert execution.roi_projection_identity == payload.roi_projection.canonical_roi_projection_identity
    serialized = serialize_guided_npm_startup_payload(payload)
    serialized_text = repr(serialized).lower()
    for forbidden in (
        "wrapper_argv",
        "candidate_manifest",
        "manifest_path",
        "validation_request",
        "gui_draft",
        "claim_path",
    ):
        assert forbidden not in serialized_text


def test_serialization_round_trip_is_deterministic_and_reorders_mapping_keys(tmp_path):
    payload = _payload(tmp_path)
    first = serialize_guided_npm_startup_payload(payload)
    second = serialize_guided_npm_startup_payload(payload)
    assert first == second
    reordered = dict(reversed(tuple(first.items())))
    restored = deserialize_guided_npm_startup_payload(reordered)
    assert restored == payload
    assert restored.canonical_startup_payload_identity == payload.canonical_startup_payload_identity


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("startup_schema_version",), "v999"),
        (("canonical_startup_payload_identity",), "1" * 64),
        (("source_projection", "ordered_session_sequence_identity"), "1" * 64),
        (("execution_projection", "ordered_source_digests"), ["1" * 64, "2" * 64]),
        (("payload_status",), "persisted"),
        (("claim_status",), "claimed"),
        (("startup_status",), "materialized"),
        (("runnable",), True),
        (("recording_policy", "target_fs_hz"), math.inf),
    ),
)
def test_deserialization_refuses_tampering(tmp_path, path, value):
    serialized = copy.deepcopy(serialize_guided_npm_startup_payload(_payload(tmp_path)))
    target = serialized
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="startup_payload_serialization_invalid"):
        deserialize_guided_npm_startup_payload(serialized)


def test_deserialization_refuses_missing_required_field(tmp_path):
    serialized = serialize_guided_npm_startup_payload(_payload(tmp_path))
    del serialized["source_projection"]
    with pytest.raises(ValueError, match="startup_payload_serialization_invalid"):
        deserialize_guided_npm_startup_payload(serialized)


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    (
        ("pre_hash_stat_identity", None, True),
        ("pre_hash_stat_identity", "not-a-digest", False),
        ("pre_hash_stat_identity", "1" * 64, False),
        ("observed_size_bytes", 987654321, False),
        ("observed_sha256_content_digest", "1" * 64, False),
        ("canonical_verified_file_identity", "1" * 64, False),
    ),
)
def test_deserialization_refuses_invalid_embedded_verified_file_contract(
    tmp_path, field, value, remove
):
    serialized = serialize_guided_npm_startup_payload(_payload(tmp_path))
    verified = serialized["source_projection"]["ordered_sessions"][0][
        "verified_source_file"
    ]
    if remove:
        del verified[field]
    else:
        verified[field] = value
    with pytest.raises(ValueError, match="startup_payload_serialization_invalid"):
        deserialize_guided_npm_startup_payload(serialized)


def test_opaque_per_session_and_source_root_identities_were_removed(tmp_path):
    payload = _payload(tmp_path)
    session_fields = payload.source_projection.ordered_sessions[0].__dataclass_fields__
    source_fields = payload.source_projection.__dataclass_fields__
    assert "source_authorized_session_identity" not in session_fields
    assert "verified_file_identity" not in session_fields
    assert "source_root_verified_identity" not in source_fields
    assert "verified_source_file" in session_fields
    assert "source_root_inspected" in source_fields
    assert "discovery_contract_version" in source_fields


def test_public_verifier_detects_nested_and_top_level_tampering(tmp_path):
    payload = _payload(tmp_path)
    verify_guided_npm_startup_payload(payload)
    changed_source = replace(
        payload.source_projection,
        source_root_inspected="C:/changed/root",
    )
    with pytest.raises(
        ValueError,
        match="startup_verified_source_snapshot_identity_mismatch",
    ):
        verify_guided_npm_startup_payload(
            replace(payload, source_projection=changed_source)
        )
    with pytest.raises(ValueError, match="startup_payload_identity_mismatch"):
        verify_guided_npm_startup_payload(
            replace(payload, canonical_startup_payload_identity="1" * 64)
        )


@pytest.mark.parametrize(
    ("verified_changes", "session_changes", "source_changes"),
    (
        ({"canonical_verified_file_identity": "1" * 64}, {}, {}),
        (
            {
                "pre_hash_stat_identity": "1" * 64,
                "post_hash_stat_identity": "1" * 64,
            },
            {},
            {},
        ),
        (
            {
                "pre_hash_stat_identity": "2" * 64,
                "post_hash_stat_identity": "2" * 64,
            },
            {},
            {},
        ),
        ({"pre_hash_stat_identity": "1" * 64}, {}, {}),
        ({}, {}, {"verified_ordered_file_sequence_identity": "1" * 64}),
        ({}, {}, {"verified_source_set_identity": "1" * 64}),
        ({}, {}, {"verified_source_content_identity": "1" * 64}),
        ({}, {}, {"verified_source_snapshot_identity": "1" * 64}),
        ({}, {}, {"source_root_inspected": "C:/altered/inspected-root"}),
        ({}, {}, {"discovery_contract_version": "altered.discovery.v1"}),
        ({"observed_size_bytes": 987654321}, {}, {}),
        ({"observed_sha256_content_digest": "1" * 64}, {}, {}),
        ({}, {"size_bytes": 987654321}, {}),
        ({}, {"sha256_content_digest": "1" * 64}, {}),
        ({}, {"canonical_relative_path": "altered.csv"}, {}),
        ({}, {"chronological_position": 99}, {}),
    ),
)
def test_semantic_source_tampering_refuses_after_all_b2_c4_identities_are_recomputed(
    tmp_path,
    verified_changes,
    session_changes,
    source_changes,
):
    payload = _payload(tmp_path)
    altered = _outer_reidentified_source_tamper(
        payload,
        verified_changes=verified_changes,
        session_changes=session_changes,
        source_changes=source_changes,
    )
    assert (
        compute_guided_npm_startup_payload_identity(altered)
        == altered.canonical_startup_payload_identity
    )
    with pytest.raises(ValueError):
        verify_guided_npm_startup_payload(altered)


def test_top_level_upstream_identities_are_provenance_references_not_source_evidence(
    tmp_path,
):
    payload = _payload(tmp_path)
    for field in ("source_authorization_identity", "source_authority_identity"):
        candidate = replace(
            payload,
            **{field: "1" * 64},
            canonical_startup_payload_identity="0" * 64,
        )
        candidate = replace(
            candidate,
            canonical_startup_payload_identity=(
                compute_guided_npm_startup_payload_identity(candidate)
            ),
        )
        verify_guided_npm_startup_payload(candidate)
        assert (
            candidate.canonical_startup_payload_identity
            != payload.canonical_startup_payload_identity
        )
        assert (
            candidate.source_projection
            == payload.source_projection
        )


def test_serialized_payload_verifies_after_upstream_objects_are_discarded(tmp_path):
    authorization, authority = _pair(tmp_path)
    payload = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(payload, GuidedNpmStartupPayload)
    serialized = serialize_guided_npm_startup_payload(payload)
    del authorization, authority, payload
    restored = deserialize_guided_npm_startup_payload(serialized)
    verify_guided_npm_startup_payload(restored)
    assert serialize_guided_npm_startup_payload(restored) == serialized


def test_internal_execution_reference_mismatch_refuses_even_with_recomputed_identities(tmp_path):
    payload = _payload(tmp_path)
    execution = replace(
        payload.execution_projection,
        ordered_source_paths=tuple(reversed(payload.execution_projection.ordered_source_paths)),
        canonical_execution_projection_identity="0" * 64,
    )
    execution = replace(
        execution,
        canonical_execution_projection_identity=(
            compute_guided_npm_startup_execution_projection_identity(execution)
        ),
    )
    altered = replace(
        payload,
        execution_projection=execution,
        canonical_startup_payload_identity="0" * 64,
    )
    altered = replace(
        altered,
        canonical_startup_payload_identity=compute_guided_npm_startup_payload_identity(altered),
    )
    with pytest.raises(ValueError, match="execution_source_path_mismatch"):
        verify_guided_npm_startup_payload(altered)


def test_no_rwd_candidate_manifest_dependency_or_api_surface(tmp_path):
    source = inspect.getsource(startup_module).lower()
    assert "candidate_manifest" not in source
    assert "guided_candidate_manifest.json" not in source
    assert "rwd" not in source
    payload = _payload(tmp_path)
    assert all("manifest" not in field for field in payload.__dataclass_fields__)


def test_all_b2_c4_operations_are_pure_after_upstream_objects_exist(tmp_path, monkeypatch):
    authorization, authority = _pair(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("B2-C4 attempted filesystem or source reinterpretation")

    for target, name in (
        (builtins, "open"),
        (os, "stat"),
        (os, "listdir"),
        (os, "scandir"),
        (Path, "exists"),
        (Path, "is_file"),
        (Path, "is_dir"),
        (Path, "stat"),
        (Path, "iterdir"),
        (Path, "glob"),
        (Path, "rglob"),
        (Path, "resolve"),
        (Path, "mkdir"),
        (Path, "write_text"),
        (Path, "write_bytes"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    for target, name in (
        (authorization_module, "authorize_guided_npm_execution_authority"),
        (authorization_module, "_build_verified_snapshot"),
        (authorization_module, "_verify_one_file"),
        (authority_module, "build_guided_npm_execution_authority"),
        (source_snapshot_module, "parse_npm_filename_timestamp"),
        (source_snapshot_module, "build_npm_source_candidate_snapshot"),
        (npm_contract_module, "inspect_npm_csv"),
        (npm_contract_module, "_resolve_time_column"),
        (npm_contract_module, "resolve_npm_support_geometry"),
        (normalized_module, "build_npm_normalized_recording_description"),
        (normalized_module, "rebuild_normalized_recording_description_from_intent"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    payload = build_guided_npm_startup_payload(authorization, authority)
    assert isinstance(payload, GuidedNpmStartupPayload)
    identity = compute_guided_npm_startup_payload_identity(payload)
    assert identity == payload.canonical_startup_payload_identity
    serialized = serialize_guided_npm_startup_payload(payload)
    restored = deserialize_guided_npm_startup_payload(serialized)
    verify_guided_npm_startup_payload(restored)


def test_builder_signature_has_only_the_two_upstream_contracts():
    assert tuple(inspect.signature(build_guided_npm_startup_payload).parameters) == (
        "authorization",
        "authority",
    )


def test_builder_does_not_create_persistence_claim_or_output_artifacts(tmp_path):
    authorization, authority = _pair(tmp_path)
    before = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    payload = build_guided_npm_startup_payload(authorization, authority)
    after = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    assert isinstance(payload, GuidedNpmStartupPayload)
    assert after == before
    assert not any(
        item.name in {
            "guided_npm_startup_payload.json",
            "guided_candidate_manifest.json",
        }
        for item in tmp_path.rglob("*")
    )
