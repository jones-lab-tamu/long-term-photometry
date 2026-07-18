import builtins
from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from photometry_pipeline.guided_npm_execution_authority import (
    GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED,
    GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME,
    GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED,
    _AuthorityRefusal,
    _build_sessions,
    _digest,
    _parser_policy_from_intent,
    GuidedNpmExecutionAuthority,
    GuidedNpmExecutionAuthorityFailure,
    build_guided_npm_execution_authority,
    compute_guided_npm_authorized_session_identity,
    compute_guided_npm_execution_authority_identity,
    compute_guided_npm_session_sequence_identity,
    deserialize_guided_npm_execution_authority,
    serialize_guided_npm_execution_authority,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    validate_current_guided_draft_for_backend,
)
from photometry_pipeline.guided_backend_validator import GuidedBackendValidatorContract
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_normalized_recording import (
    compute_npm_parser_contract_digest,
    compute_npm_support_policy_identity,
    deserialize_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.io.npm_contract import NpmParserContract
from photometry_pipeline.guided_production_mapping import (
    GuidedNpmProductionMappingSuccess,
    GuidedProductionMappingFailure,
    compute_guided_npm_production_execution_intent_identity,
)

from tests.test_guided_backend_validation_materialization import _valid_npm_stage2c_draft
from tests.test_guided_npm_production_mapping import (
    _accepted_npm,
    _accepted_npm_default_initialized,
    _map,
)


def _accepted_authority(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    mapped = _map(outcome)
    return build_guided_npm_execution_authority(mapped.intent)


def _accepted_authority_with_default_initialized_features(tmp_path: Path):
    """Same real materialization path as _accepted_authority, but with a
    loaded Default Feature Detection profile left as
    "default_initialized" (never explicitly applied) -- an equally
    current, real, production-accepted state."""
    outcome, _request = _accepted_npm_default_initialized(tmp_path)
    return build_guided_npm_execution_authority(_map(outcome).intent)


def _default_initialized_mapped_intent(tmp_path: Path):
    """The mapped (but not yet authority-built) intent for the same
    default_initialized state as _accepted_authority_with_default_
    initialized_features, so callers can tamper per-ROI entries before
    building the authority."""
    outcome, _request = _accepted_npm_default_initialized(tmp_path)
    return _map(outcome).intent


def _accepted_two_session_authority(tmp_path: Path):
    draft = _valid_npm_stage2c_draft(tmp_path)
    source_root = Path(draft.input_source_path)
    (source_root / "photometryData2026-06-30T12_10_00.csv").write_text(
        "SystemTimestamp,LedState,Region2G\n"
        "100.0,1,10.0\n"
        "100.5,2,100.0\n"
        "101.0,1,11.0\n"
        "101.5,2,101.0\n"
        "102.0,1,12.0\n"
        "102.5,2,102.0\n",
        encoding="utf-8",
    )
    validator_contract = GuidedBackendValidatorContract(
        validation_scope="guided_rwd_intermittent_phasic_full_validate",
        validation_contract_version="guided_backend_validation_contract.v1",
        validator_capability_version="test_validator_capability.v1",
        supported_subset_rule_version="global_dynamic_fit_only.v1",
    )
    parser_contract = NpmParserContract(
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        target_fs_hz=2.0,
        session_duration_sec=2.0,
        allow_partial_final_chunk=False,
        adapter_value_nan_policy="strict",
    )
    outcome = validate_current_guided_draft_for_backend(
        draft,
        parser_contract=parser_contract,
        validator_contract=validator_contract,
        validation_revision=4,
    )
    assert outcome.status == "validator_accepted"
    return build_guided_npm_execution_authority(_map(outcome).intent)


def _assert_failure(result, category: str):
    assert isinstance(result, GuidedNpmExecutionAuthorityFailure)
    assert result.blocking_issues[0].category == category
    return result.blocking_issues[0]


def _unsafe_replace(instance, **changes):
    clone = object.__new__(type(instance))
    for name, value in instance.__dict__.items():
        object.__setattr__(clone, name, changes.get(name, value))
    for name, value in changes.items():
        if name not in instance.__dict__:
            object.__setattr__(clone, name, value)
    return clone


def _rebound_intent(intent, **changes):
    candidate = _unsafe_replace(intent, **changes)
    return _unsafe_replace(
        candidate,
        canonical_intent_identity=compute_guided_npm_production_execution_intent_identity(
            candidate
        ),
    )


def _normalized_from_intent(intent):
    return deserialize_normalized_recording_description(
        json.loads(intent.normalized_recording_payload_json)
    )


def _rebound_normalized_intent(intent, normalized, **changes):
    payload = serialize_normalized_recording_description(normalized)
    return _rebound_intent(
        intent,
        normalized_recording_description_identity=(
            payload["normalized_recording_description_identity"]
        ),
        normalized_recording_payload_json=json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ),
        **changes,
    )


def _session_build_refusal(intent, normalized):
    policy, _payload = _parser_policy_from_intent(intent)
    try:
        _build_sessions(intent, normalized, policy)
    except _AuthorityRefusal as exc:
        return exc
    raise AssertionError("Expected the per-session authority construction to refuse.")


def _permissive_intent(intent):
    normalized = _normalized_from_intent(intent)
    parser_content = json.loads(intent.parser_policy_content_json)
    parser_content["sampling"]["allow_partial_final_chunk"] = True
    parser_content["sampling"]["support_policy"] = "permissive_overlap_from_t0"
    parser_digest = compute_npm_parser_contract_digest(parser_content)

    adapter_evidence = copy.deepcopy(dict(normalized.adapter_evidence))
    permissive_identity = compute_npm_support_policy_identity(
        "permissive_overlap_from_t0"
    )
    for raw in adapter_evidence["npm_sessions"]:
        raw["support_policy"] = "permissive_overlap_from_t0"
        raw["support_policy_identity"] = permissive_identity
        raw["resolved_support_start_offset_sec"] = 0.0
        raw["resolved_support_end_offset_sec"] = 1.0
        raw["resolved_support_start_absolute"] = raw["overlap_origin_absolute"]
        raw["resolved_support_end_absolute"] = raw["overlap_origin_absolute"] + 1.0

    normalized = replace(
        normalized,
        sampling=replace(
            normalized.sampling,
            parser_contract_identity=parser_digest,
            parser_contract_content=parser_content,
        ),
        adapter_evidence=adapter_evidence,
    )
    payload = serialize_normalized_recording_description(normalized)
    parser_content_json = json.dumps(
        parser_content, sort_keys=True, separators=(",", ":")
    )
    return _rebound_intent(
        intent,
        normalized_recording_description_identity=(
            payload["normalized_recording_description_identity"]
        ),
        normalized_recording_payload_json=json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ),
        parser_policy_identity=parser_digest,
        parser_policy_content_json=parser_content_json,
        per_session_resolved_evidence_identity=_digest(
            "npm-per-session-resolved-evidence:v1",
            payload["npm_per_session_resolved_evidence"],
        ),
        support_policy_identity=permissive_identity,
    )


def test_accepted_intent_builds_frozen_unauthorized_authority(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    assert isinstance(authority, GuidedNpmExecutionAuthority)
    assert authority.authority_schema_name == GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME
    assert authority.authorization_status == GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED
    assert authority.startup_status == GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED
    assert authority.runnable is False
    assert authority.sessions[0].chronological_position == 0
    assert authority.sessions[0].canonical_relative_path
    assert authority.recording_policy.ordered_timestamp_candidates
    assert authority.roi_authority.selected_physical_source_columns
    with pytest.raises((AttributeError, TypeError)):
        authority.runnable = True


def test_default_initialized_feature_profile_without_apply_builds_authority(
    tmp_path: Path,
):
    """Repair regression: _build_feature_authority's completeness
    check must accept the same "default_initialized" (never explicitly
    applied) state _feature_event_profile_current_for_first_subset
    already treats as current, using the same real materialization path
    Check My Setup itself uses -- not a hand-built intent. The per-ROI
    entries' truthful explicit_user_mark=False is accepted because the
    enclosing profile is itself current and default_initialized (see
    feature_entry_provenance_valid); the check must not require a
    falsified explicit mark."""
    authority = _accepted_authority_with_default_initialized_features(tmp_path)
    assert isinstance(authority, GuidedNpmExecutionAuthority)
    assert authority.feature_authority.per_roi_feature_event_map
    for entry in authority.feature_authority.per_roi_feature_event_map:
        assert entry.source == "default"
        assert entry.explicit_user_mark is False
        assert entry.current_or_stale == "current"


def test_genuinely_stale_feature_profile_still_refuses_authority(tmp_path: Path):
    """The completeness check must continue refusing a genuinely
    incomplete/stale feature profile -- proves the repair did not widen
    the check beyond the one legitimate default_initialized case."""
    outcome, _request = _accepted_npm(tmp_path)
    mapped = _map(outcome)
    stale_feature_event = replace(
        mapped.intent.feature_event,
        profile_status="stale",
        explicitly_applied=False,
        current=False,
    )
    tampered_intent = _rebound_intent(
        mapped.intent, feature_event=stale_feature_event
    )
    result = build_guided_npm_execution_authority(tampered_intent)
    _assert_failure(result, "feature_authority_incomplete")


def test_applied_profile_entry_without_explicit_mark_refuses_authority(
    tmp_path: Path,
):
    """Negative provenance case A: under an "applied" enclosing profile,
    a per-ROI entry without an explicit mark must still refuse -- the
    narrow default_initialized acceptance does not apply here."""
    outcome, _request = _accepted_npm(tmp_path)
    mapped = _map(outcome)
    entries = mapped.intent.feature_event.per_roi_feature_event_map
    tampered_entries = (replace(entries[0], explicit_user_mark=False),) + entries[1:]
    tampered_feature_event = replace(
        mapped.intent.feature_event, per_roi_feature_event_map=tampered_entries
    )
    tampered_intent = _rebound_intent(
        mapped.intent, feature_event=tampered_feature_event
    )
    result = build_guided_npm_execution_authority(tampered_intent)
    _assert_failure(result, "feature_authority_incomplete")


def test_override_entry_without_explicit_mark_refuses_authority(tmp_path: Path):
    """Negative provenance case B: an override/custom-sourced entry must
    still carry an explicit mark; the narrow default_initialized
    acceptance only applies to source == "default"."""
    outcome, _request = _accepted_npm(tmp_path)
    mapped = _map(outcome)
    entries = mapped.intent.feature_event.per_roi_feature_event_map
    override_entry = replace(
        entries[0],
        source="override",
        feature_event_profile_id="feature-profile-custom",
        explicit_user_mark=False,
    )
    tampered_feature_event = replace(
        mapped.intent.feature_event,
        per_roi_feature_event_map=(override_entry,) + entries[1:],
    )
    tampered_intent = _rebound_intent(
        mapped.intent, feature_event=tampered_feature_event
    )
    result = build_guided_npm_execution_authority(tampered_intent)
    _assert_failure(result, "feature_authority_incomplete")


@pytest.mark.parametrize(
    "tamper",
    (
        lambda entry: replace(entry, current_or_stale="stale"),
        lambda entry: replace(entry, effective_config_fields=()),
        lambda entry: replace(entry, feature_event_profile_id="a-different-profile-id"),
    ),
)
def test_default_initialized_entry_edge_cases_refuse_authority(
    tmp_path: Path, tamper
):
    """Negative provenance cases C/D/E: even under the narrow
    default_initialized acceptance, a stale entry (C), an entry with no
    effective settings (D), or a default-sourced entry whose profile
    identity does not match the enclosing accepted default profile (E)
    must still refuse at the authority layer too -- proves
    feature_entry_provenance_valid does not infer validity from a
    non-explicit entry alone."""
    intent = _default_initialized_mapped_intent(tmp_path)
    entries = intent.feature_event.per_roi_feature_event_map
    tampered_entries = (tamper(entries[0]),) + entries[1:]
    tampered_feature_event = replace(
        intent.feature_event, per_roi_feature_event_map=tampered_entries
    )
    tampered_intent = _rebound_intent(intent, feature_event=tampered_feature_event)
    result = build_guided_npm_execution_authority(tampered_intent)
    _assert_failure(result, "feature_authority_incomplete")


def test_stray_feature_entry_roi_not_selected_refuses_authority(tmp_path: Path):
    """Negative provenance case F: a per-ROI feature entry for a ROI
    outside the selected set must not be silently accepted as
    authoritative -- the coverage check catches this before the
    per-entry provenance rule is ever reached."""
    intent = _default_initialized_mapped_intent(tmp_path)
    entries = intent.feature_event.per_roi_feature_event_map
    stray_entry = replace(entries[0], roi_id="NotSelectedRoi")
    tampered_feature_event = replace(
        intent.feature_event,
        per_roi_feature_event_map=(stray_entry,) + entries[1:],
    )
    tampered_intent = _rebound_intent(intent, feature_event=tampered_feature_event)
    result = build_guided_npm_execution_authority(tampered_intent)
    _assert_failure(result, "feature_selected_roi_coverage_mismatch")


def test_mapping_and_authority_agree_default_initialized_state_is_valid(
    tmp_path: Path,
):
    """Mapper/authority agreement (positive): the same accepted, current
    default_initialized feature state -- passed through production
    mapping and then execution authority -- is accepted by both layers,
    and both preserve the truthful explicit_user_mark=False rather than
    requiring or fabricating an explicit mark."""
    outcome, _request = _accepted_npm_default_initialized(tmp_path)
    mapped = _map(outcome)
    assert isinstance(mapped, GuidedNpmProductionMappingSuccess)
    mapped_entries = mapped.intent.feature_event.per_roi_feature_event_map
    assert mapped_entries
    for entry in mapped_entries:
        assert entry.source == "default"
        assert entry.explicit_user_mark is False

    authority = build_guided_npm_execution_authority(mapped.intent)
    assert isinstance(authority, GuidedNpmExecutionAuthority)
    authority_entries = authority.feature_authority.per_roi_feature_event_map
    assert authority_entries
    for entry in authority_entries:
        assert entry.source == "default"
        assert entry.explicit_user_mark is False


def test_mapping_and_authority_agree_applied_profile_non_explicit_entry_is_invalid(
    tmp_path: Path,
):
    """Mapper/authority agreement (negative): an applied-profile per-ROI
    entry without an explicit mark is rejected identically by production
    mapping and by execution authority -- the two layers cannot diverge
    on this mismatch."""
    from tests.test_guided_npm_production_mapping import (
        _outcome_with_request,
    )

    outcome, request = _accepted_npm(tmp_path)
    tampered_request = replace(
        request,
        feature_event=replace(
            request.feature_event,
            per_roi_feature_event_map=(
                replace(
                    request.feature_event.per_roi_feature_event_map[0],
                    explicit_user_mark=False,
                ),
            )
            + request.feature_event.per_roi_feature_event_map[1:],
        ),
    )
    mapping_result = _map(_outcome_with_request(outcome, tampered_request))
    assert isinstance(mapping_result, GuidedProductionMappingFailure)
    assert mapping_result.blocking_issues[0].category == "incomplete_feature_settings"
    assert mapping_result.blocking_issues[0].detail_code == (
        "per_roi_feature_entry_incomplete"
    )

    mapped = _map(outcome)
    entries = mapped.intent.feature_event.per_roi_feature_event_map
    tampered_feature_event = replace(
        mapped.intent.feature_event,
        per_roi_feature_event_map=(replace(entries[0], explicit_user_mark=False),)
        + entries[1:],
    )
    tampered_intent = _rebound_intent(
        mapped.intent, feature_event=tampered_feature_event
    )
    authority_result = build_guided_npm_execution_authority(tampered_intent)
    _assert_failure(authority_result, "feature_authority_incomplete")


def test_authority_round_trips_and_nested_identities_are_verified(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    payload = serialize_guided_npm_execution_authority(authority)
    restored = deserialize_guided_npm_execution_authority(payload)
    assert restored == authority
    assert compute_guided_npm_execution_authority_identity(restored) == (
        restored.canonical_authority_identity
    )

    tampered = dict(payload)
    tampered["canonical_authority_identity"] = "0" * 64
    with pytest.raises(ValueError, match="authority_serialization_invalid"):
        deserialize_guided_npm_execution_authority(tampered)


def test_two_sessions_preserve_exact_order_and_heterogeneous_timestamp_columns(
    tmp_path: Path,
):
    authority = _accepted_two_session_authority(tmp_path)
    assert isinstance(authority, GuidedNpmExecutionAuthority)
    assert tuple(item.chronological_position for item in authority.sessions) == (0, 1)
    assert tuple(item.resolved_timestamp_column for item in authority.sessions) == (
        "Timestamp",
        "SystemTimestamp",
    )
    assert not hasattr(authority.recording_policy, "resolved_timestamp_column")
    assert authority.sessions[0].canonical_session_identity != (
        authority.sessions[1].canonical_session_identity
    )

    changed = replace(
        authority.sessions[1],
        resolved_timestamp_column="Timestamp",
        canonical_session_identity="0" * 64,
    )
    changed = replace(
        changed,
        canonical_session_identity=compute_guided_npm_authorized_session_identity(
            changed
        ),
    )
    changed_authority = replace(
        authority,
        sessions=(authority.sessions[0], changed),
        session_sequence_identity="0" * 64,
        canonical_authority_identity="0" * 64,
    )
    changed_authority = replace(
        changed_authority,
        session_sequence_identity=compute_guided_npm_session_sequence_identity(
            changed_authority.sessions
        ),
    )
    assert compute_guided_npm_execution_authority_identity(changed_authority) != (
        authority.canonical_authority_identity
    )


def test_unauthorized_timestamp_column_refuses_with_precise_category(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    intent = _map(outcome).intent
    normalized = _normalized_from_intent(intent)
    adapter_evidence = copy.deepcopy(dict(normalized.adapter_evidence))
    adapter_evidence["npm_sessions"][0][
        "resolved_timestamp_column"
    ] = "AlternateTimestamp"
    bad_intent = _rebound_normalized_intent(
        intent,
        replace(normalized, adapter_evidence=adapter_evidence),
    )

    issue = _assert_failure(
        build_guided_npm_execution_authority(bad_intent),
        "session_timestamp_column_missing",
    )
    assert issue.detail_code == "resolved_timestamp_column_not_authorized"


def test_timestamp_candidate_removed_from_policy_refuses(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    intent = _map(outcome).intent
    normalized = _normalized_from_intent(intent)
    parser_content = json.loads(intent.parser_policy_content_json)
    parser_content["sampling"]["timestamp_column_candidates"] = ["SystemTimestamp"]
    parser_digest = compute_npm_parser_contract_digest(parser_content)
    normalized = replace(
        normalized,
        sampling=replace(
            normalized.sampling,
            parser_contract_identity=parser_digest,
            parser_contract_content=parser_content,
        ),
    )
    bad_intent = _rebound_normalized_intent(
        intent,
        normalized,
        parser_policy_identity=parser_digest,
        parser_policy_content_json=json.dumps(
            parser_content, sort_keys=True, separators=(",", ":")
        ),
    )

    issue = _assert_failure(
        build_guided_npm_execution_authority(bad_intent),
        "session_timestamp_column_missing",
    )
    assert issue.detail_code == "resolved_timestamp_column_not_authorized"


def test_recording_wide_support_policy_identity_is_not_an_aggregate(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    intent = _map(outcome).intent
    assert intent.support_policy_identity == compute_npm_support_policy_identity(
        "strict_overlap_inner_support"
    )
    authority = build_guided_npm_execution_authority(intent)
    assert isinstance(authority, GuidedNpmExecutionAuthority)
    policy = authority.recording_policy
    assert policy.support_policy_identity == compute_npm_support_policy_identity(
        policy.support_policy
    )

    old_aggregate = hashlib.sha256(
        b"npm-support-policy:v1\x00"
        + encode_canonical_value(
            [item.support_policy_identity for item in authority.sessions]
        )
    ).hexdigest()
    assert policy.support_policy_identity != old_aggregate
    bad_intent = _rebound_intent(
        intent, support_policy_identity=old_aggregate
    )
    issue = _assert_failure(
        build_guided_npm_execution_authority(bad_intent),
        "recording_policy_identity_mismatch",
    )
    assert issue.detail_code == "support_policy_identity_mismatch"


@pytest.mark.parametrize(
    ("changes", "category", "detail_code"),
    (
        (
            {"support_policy_identity": "0" * 64},
            "session_support_geometry_invalid",
            "session_support_policy_identity_mismatch",
        ),
        (
            {"support_policy": "permissive_overlap_from_t0"},
            "session_output_time_basis_mismatch",
            "session_support_policy_mismatch",
        ),
        (
            {
                "support_policy": "permissive_overlap_from_t0",
                "support_policy_identity": compute_npm_support_policy_identity(
                    "permissive_overlap_from_t0"
                ),
            },
            "session_output_time_basis_mismatch",
            "session_support_policy_mismatch",
        ),
        (
            {"support_policy_identity": "1" * 64},
            "session_support_geometry_invalid",
            "session_support_policy_identity_mismatch",
        ),
    ),
)
def test_session_support_policy_identity_checks(
    tmp_path: Path, changes: dict[str, str], category: str, detail_code: str
):
    outcome, _request = _accepted_npm(tmp_path)
    intent = _map(outcome).intent
    normalized = _normalized_from_intent(intent)
    adapter_evidence = copy.deepcopy(dict(normalized.adapter_evidence))
    adapter_evidence["npm_sessions"][0].update(changes)
    refusal = _session_build_refusal(
        intent, replace(normalized, adapter_evidence=adapter_evidence)
    )
    assert refusal.category == category
    assert refusal.detail_code == detail_code


def test_strict_and_permissive_support_policies_build_distinct_authorities(
    tmp_path: Path,
):
    outcome, _request = _accepted_npm(tmp_path)
    strict_intent = _map(outcome).intent
    strict = build_guided_npm_execution_authority(strict_intent)
    permissive = build_guided_npm_execution_authority(_permissive_intent(strict_intent))
    assert isinstance(strict, GuidedNpmExecutionAuthority)
    assert isinstance(permissive, GuidedNpmExecutionAuthority)
    assert strict.recording_policy.support_policy == "strict_overlap_inner_support"
    assert permissive.recording_policy.support_policy == "permissive_overlap_from_t0"
    assert strict.recording_policy.support_policy_identity != (
        permissive.recording_policy.support_policy_identity
    )
    assert strict.recording_policy.output_time_basis == (
        "relative_seconds_since_uv_signal_overlap_origin"
    )
    assert permissive.recording_policy.output_time_basis == (
        "relative_seconds_since_uv_signal_overlap_origin"
    )
    assert all(
        item.resolved_support_start_offset_sec > 0 for item in strict.sessions
    )
    for authority in (strict, permissive):
        assert all(
            item.support_policy_identity
            == authority.recording_policy.support_policy_identity
            for item in authority.sessions
        )


def test_intent_identity_and_state_refusals(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    mapped = _map(outcome)
    intent = mapped.intent

    wrong_identity = replace(intent, canonical_intent_identity="0" * 64)
    result = build_guided_npm_execution_authority(wrong_identity)
    assert _assert_failure(result, "intent_identity_mismatch").detail_code == (
        "intent_identity_mismatch"
    )

    wrong_format = _unsafe_replace(intent, source_format="rwd")
    wrong_format = _unsafe_replace(
        wrong_format,
        canonical_intent_identity=compute_guided_npm_production_execution_intent_identity(
            wrong_format
        ),
    )
    assert _assert_failure(
        build_guided_npm_execution_authority(wrong_format), "intent_not_npm"
    )

    startup_claim = replace(
        intent,
        deferred_capabilities=tuple(
            item for item in intent.deferred_capabilities if item != "npm_startup_orchestration"
        ),
    )
    startup_claim = replace(
        startup_claim,
        canonical_intent_identity=compute_guided_npm_production_execution_intent_identity(
            startup_claim
        ),
    )
    assert _assert_failure(
        build_guided_npm_execution_authority(startup_claim), "intent_startup_state_invalid"
    )


def test_builder_refuses_candidate_linkage_and_output_policy_tampering(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    intent = _map(outcome).intent

    changed_candidate = replace(
        intent.source_candidate_files[0],
        size_bytes=intent.source_candidate_files[0].size_bytes + 1,
    )
    bad_candidate = _rebound_intent(
        intent, source_candidate_files=(changed_candidate,)
    )
    assert _assert_failure(
        build_guided_npm_execution_authority(bad_candidate), "session_size_mismatch"
    )

    bad_output_policy = replace(intent.output_policy, overwrite=True)
    bad_output = _rebound_intent(intent, output_policy=bad_output_policy)
    assert _assert_failure(
        build_guided_npm_execution_authority(bad_output), "output_authority_invalid"
    )


@pytest.mark.parametrize("bad_path", ("../escape.csv", "C:/escape.csv"))
def test_builder_refuses_noncanonical_relative_source_paths(tmp_path: Path, bad_path: str):
    outcome, _request = _accepted_npm(tmp_path)
    intent = _map(outcome).intent
    changed_candidate = replace(
        intent.source_candidate_files[0], canonical_relative_path=bad_path
    )
    bad_intent = _rebound_intent(intent, source_candidate_files=(changed_candidate,))
    assert _assert_failure(
        build_guided_npm_execution_authority(bad_intent), "session_candidate_missing"
    )


def test_authority_identity_binds_revision_and_nested_facts(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    changed = replace(
        authority,
        validation_revision=authority.validation_revision + 1,
        canonical_authority_identity="0" * 64,
    )
    assert compute_guided_npm_execution_authority_identity(changed) != (
        authority.canonical_authority_identity
    )

    changed_session = replace(
        authority.sessions[0],
        actual_elapsed_sec=authority.sessions[0].actual_elapsed_sec + 0.25,
        canonical_session_identity="0" * 64,
    )
    changed_session = replace(
        changed_session,
        canonical_session_identity=(
            __import__(
                "photometry_pipeline.guided_npm_execution_authority",
                fromlist=["compute_guided_npm_authorized_session_identity"],
            ).compute_guided_npm_authorized_session_identity(changed_session)
        ),
    )
    changed_authority = replace(
        authority,
        sessions=(changed_session,),
        session_sequence_identity="0" * 64,
        canonical_authority_identity="0" * 64,
    )
    changed_authority = replace(
        changed_authority,
        session_sequence_identity=__import__(
            "photometry_pipeline.guided_npm_execution_authority",
            fromlist=["compute_guided_npm_session_sequence_identity"],
        ).compute_guided_npm_session_sequence_identity(changed_authority.sessions),
    )
    assert compute_guided_npm_execution_authority_identity(changed_authority) != (
        authority.canonical_authority_identity
    )


def test_nested_serialization_tampering_is_refused(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    payload = serialize_guided_npm_execution_authority(authority)
    tampered = dict(payload)
    tampered_policy = dict(tampered["recording_policy"])
    tampered_policy["canonical_policy_identity"] = "0" * 64
    tampered["recording_policy"] = tampered_policy
    with pytest.raises(ValueError, match="authority_serialization_invalid"):
        deserialize_guided_npm_execution_authority(tampered)

    unknown_schema = dict(payload)
    unknown_schema["authority_schema_version"] = "v999"
    with pytest.raises(ValueError, match="authority_serialization_invalid"):
        deserialize_guided_npm_execution_authority(unknown_schema)


@pytest.mark.parametrize(
    "mutation",
    (
        "unauthorized_timestamp",
        "removed_timestamp_candidate",
        "session_support_identity",
        "recording_support_identity",
        "support_policy_without_identity",
    ),
)
def test_serialized_support_and_timestamp_semantics_are_refused(
    tmp_path: Path, mutation: str
):
    authority = _accepted_authority(tmp_path)
    payload = copy.deepcopy(serialize_guided_npm_execution_authority(authority))
    if mutation == "unauthorized_timestamp":
        payload["sessions"][0]["resolved_timestamp_column"] = "AlternateTimestamp"
    elif mutation == "removed_timestamp_candidate":
        payload["recording_policy"]["ordered_timestamp_candidates"] = [
            item
            for item in payload["recording_policy"]["ordered_timestamp_candidates"]
            if item != authority.sessions[0].resolved_timestamp_column
        ]
    elif mutation == "session_support_identity":
        payload["sessions"][0]["support_policy_identity"] = "0" * 64
    elif mutation == "recording_support_identity":
        payload["recording_policy"]["support_policy_identity"] = "0" * 64
    else:
        payload["recording_policy"]["support_policy"] = "permissive_overlap_from_t0"

    with pytest.raises(ValueError, match="authority_serialization_invalid"):
        deserialize_guided_npm_execution_authority(payload)


def test_builder_and_serialization_perform_no_source_io(tmp_path: Path, monkeypatch):
    outcome, _request = _accepted_npm(tmp_path)
    mapped = _map(outcome)

    def fail(*_args, **_kwargs):
        raise AssertionError("B2-C2 attempted filesystem or source I/O")

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "open", fail)
        guarded.setattr(Path, "exists", fail)
        guarded.setattr(Path, "is_file", fail)
        guarded.setattr(Path, "is_dir", fail)
        guarded.setattr(Path, "stat", fail)
        guarded.setattr(Path, "resolve", fail)
        guarded.setattr(Path, "iterdir", fail)
        guarded.setattr(Path, "glob", fail)
        guarded.setattr(Path, "rglob", fail)
        guarded.setattr(os, "listdir", fail)

        authority = build_guided_npm_execution_authority(mapped.intent)
        assert isinstance(authority, GuidedNpmExecutionAuthority)
        payload = serialize_guided_npm_execution_authority(authority)
        assert deserialize_guided_npm_execution_authority(payload) == authority


def test_serialization_is_deterministic_for_mapping_key_order(tmp_path: Path):
    authority = _accepted_authority(tmp_path)
    first = serialize_guided_npm_execution_authority(authority)
    second = json.loads(json.dumps(first, sort_keys=True))
    assert serialize_guided_npm_execution_authority(
        deserialize_guided_npm_execution_authority(second)
    ) == first
