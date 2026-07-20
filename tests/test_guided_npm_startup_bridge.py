"""Permanent tests for GuidedStartupAuthority and
compile_npm_generic_execution_payloads (photometry_pipeline.guided_npm_startup_bridge).

These prove the two completed Phase 4C bridge pieces in isolation, using
real RWD and NPM fixtures, before either is wired into the shared startup
transaction.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import photometry_pipeline.guided_run_authorization as authorization
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
)
from photometry_pipeline.guided_execution_payloads import (
    GuidedExecutionPayloadDerivationResult,
    build_guided_execution_startup_mapping_contract,
    compute_guided_execution_config_payload_identity,
    compute_guided_runner_candidate_manifest_payload_identity,
    compute_guided_startup_provenance_seed_identity,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GuidedNpmExecutionAuthority,
    build_guided_npm_execution_authority,
)
from photometry_pipeline.guided_npm_startup_bridge import (
    GuidedStartupAuthority,
    compile_npm_generic_execution_payloads,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisExecutionIntent
from photometry_pipeline.guided_production_mapping import (
    GuidedNpmProductionExecutionIntent,
)
from photometry_pipeline.guided_run_authorization import GuidedRunAuthorizationResult

from tests.test_guided_run_authorization import (
    _accepted_candidate,
    _accepted_roi,
    _request as _rwd_request,
)
from tests.test_guided_npm_production_mapping import _map
from tests.test_guided_backend_validation_materialization import (
    _valid_npm_stage2c_draft,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    validate_current_guided_draft_for_backend,
)
from photometry_pipeline.guided_backend_validator import GuidedBackendValidatorContract
from photometry_pipeline.io.npm_contract import NpmParserContract


def _npm_validation_fixture_kwargs():
    return dict(
        validator_contract=GuidedBackendValidatorContract(
            validation_scope="guided_rwd_intermittent_phasic_full_validate",
            validation_contract_version="guided_backend_validation_contract.v1",
            validator_capability_version="test_validator_capability.v1",
            supported_subset_rule_version="global_dynamic_fit_only.v1",
        ),
        parser_contract=NpmParserContract(
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
        ),
    )


def _accepted_npm_with_mode(tmp_path: Path, execution_mode: str):
    """Same real materialization path as _accepted_npm, but with the
    authentic Guided execution mode this repair's bridge now requires
    ("both"), or an explicit non-Guided mode for negative coverage."""
    draft = _valid_npm_stage2c_draft(tmp_path)
    draft.execution_intent = GuidedNewAnalysisExecutionIntent(execution_mode=execution_mode)
    kwargs = _npm_validation_fixture_kwargs()
    outcome = validate_current_guided_draft_for_backend(
        draft, validation_revision=4, **kwargs
    )
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    assert outcome.compile_result is not None
    request = outcome.compile_result.request
    assert request is not None
    return outcome, request


def _accepted_npm_two_rois_with_mode(tmp_path: Path, execution_mode: str):
    """Same real materialization path as _accepted_npm_two_rois, but with
    an explicit Guided execution mode."""
    from photometry_pipeline.guided_new_analysis_plan import (
        compute_guided_local_preview_source_setup_signature,
    )

    draft = _valid_npm_stage2c_draft(tmp_path)
    draft.execution_intent = GuidedNewAnalysisExecutionIntent(execution_mode=execution_mode)
    source_root = Path(draft.input_source_path)
    source_file = next(source_root.glob("*.csv"))
    source_file.write_text(
        "Timestamp,LedState,Region2G,Region10G\n"
        "100.0,1,10.0,20.0\n"
        "100.5,2,100.0,200.0\n"
        "101.0,1,11.0,21.0\n"
        "101.5,2,101.0,201.0\n"
        "102.0,1,12.0,22.0\n"
        "102.5,2,102.0,202.0\n",
        encoding="utf-8",
    )
    draft.discovered_roi_ids = ["Region0", "Region1"]
    draft.included_roi_ids = ["Region0", "Region1"]
    draft.excluded_roi_ids = []
    draft.dataset_contract_snapshot = replace(
        draft.dataset_contract_snapshot,
        source_identity=replace(
            draft.dataset_contract_snapshot.source_identity,
            discovered_roi_ids=("Region0", "Region1"),
            included_roi_ids=("Region0", "Region1"),
        ),
    )
    local_preview_signature = compute_guided_local_preview_source_setup_signature(draft)
    first_choice = replace(
        draft.per_roi_correction_strategy_choices[0],
        source_setup_signature=local_preview_signature,
    )
    draft.per_roi_correction_strategy_choices = [
        first_choice,
        replace(first_choice, roi_id="Region1"),
    ]
    kwargs = _npm_validation_fixture_kwargs()
    outcome = validate_current_guided_draft_for_backend(
        draft, validation_revision=4, **kwargs
    )
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    assert outcome.compile_result is not None
    request = outcome.compile_result.request
    assert request is not None
    return outcome, request


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rwd_authority(monkeypatch) -> GuidedRunAuthorizationResult:
    request = _rwd_request()
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: request.stored_validation_outcome,
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_candidate_manifest_execution_preflight",
        lambda candidate_request, **_kwargs: _accepted_candidate(candidate_request),
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda roi_request, **_kwargs: _accepted_roi(roi_request),
    )
    result = authorization.authorize_guided_run(request)
    assert result.status == "authorized"
    return result


def _npm_pair(
    tmp_path: Path, *, execution_mode: str = "both"
) -> tuple[GuidedNpmProductionExecutionIntent, GuidedNpmExecutionAuthority]:
    """Real, accepted NPM intent/authority pair. Defaults to "both" -- the
    only execution mode an authentic Guided NPM run carries -- not the
    phasic-only default _accepted_npm itself still uses (that default
    remains a legitimate backend/Full Control state, just not a Guided
    one)."""
    outcome, _request = _accepted_npm_with_mode(tmp_path, execution_mode)
    mapped = _map(outcome)
    intent = mapped.intent
    authority = build_guided_npm_execution_authority(intent)
    assert isinstance(authority, GuidedNpmExecutionAuthority)
    return intent, authority


def _npm_pair_two_rois(
    tmp_path: Path, *, execution_mode: str = "both"
) -> tuple[GuidedNpmProductionExecutionIntent, GuidedNpmExecutionAuthority]:
    outcome, _request = _accepted_npm_two_rois_with_mode(tmp_path, execution_mode)
    mapped = _map(outcome)
    intent = mapped.intent
    authority = build_guided_npm_execution_authority(intent)
    assert isinstance(authority, GuidedNpmExecutionAuthority)
    return intent, authority


# ---------------------------------------------------------------------------
# GuidedStartupAuthority: construction
# ---------------------------------------------------------------------------


def test_valid_rwd_variant_constructs(rwd_authority):
    wrapper = GuidedStartupAuthority(rwd=rwd_authority)
    assert wrapper.rwd is rwd_authority
    assert wrapper.npm_intent is None
    assert wrapper.npm_authority is None
    assert wrapper.is_npm is False


def test_valid_npm_variant_constructs(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    wrapper = GuidedStartupAuthority(npm_intent=intent, npm_authority=authority)
    assert wrapper.rwd is None
    assert wrapper.npm_intent is intent
    assert wrapper.npm_authority is authority
    assert wrapper.is_npm is True


def test_refuses_when_neither_variant_provided():
    with pytest.raises(ValueError):
        GuidedStartupAuthority()


def test_refuses_when_both_variants_provided(rwd_authority, tmp_path):
    intent, authority = _npm_pair(tmp_path)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(rwd=rwd_authority, npm_intent=intent, npm_authority=authority)


def test_refuses_when_only_npm_intent_provided(tmp_path):
    intent, _authority = _npm_pair(tmp_path)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(npm_intent=intent)


def test_refuses_when_only_npm_authority_provided(tmp_path):
    _intent, authority = _npm_pair(tmp_path)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(npm_authority=authority)


def test_refuses_incorrect_rwd_type(tmp_path):
    intent, _authority = _npm_pair(tmp_path)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(rwd=intent)


def test_refuses_incorrect_npm_intent_type(rwd_authority):
    with pytest.raises(ValueError):
        GuidedStartupAuthority(npm_intent=rwd_authority, npm_authority=rwd_authority)


def test_refuses_incorrect_npm_authority_type(tmp_path):
    intent, _authority = _npm_pair(tmp_path)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(npm_intent=intent, npm_authority=intent)


# ---------------------------------------------------------------------------
# GuidedStartupAuthority: common accessors
# ---------------------------------------------------------------------------


def test_common_accessors_rwd(rwd_authority):
    wrapper = GuidedStartupAuthority(rwd=rwd_authority)
    intent = rwd_authority.production_intent
    assert wrapper.source_format == intent.input_source.source_format
    assert wrapper.execution_mode == intent.execution_profile.execution_mode
    assert wrapper.sessions_per_hour == intent.acquisition.sessions_per_hour
    assert wrapper.source_root_canonical == intent.input_source.source_root_canonical
    assert wrapper.output_base_canonical == intent.output_policy.output_base_canonical
    assert wrapper.overwrite == intent.output_policy.overwrite
    assert (
        wrapper.protected_root_context_complete
        == intent.output_policy.protected_root_context_complete
    )
    assert wrapper.included_roi_ids == intent.roi_scope.included_roi_ids
    assert (
        wrapper.per_roi_correction_strategy_map
        == intent.correction.per_roi_production_strategy_map
    )
    assert wrapper.feature_event is intent.feature_event


def test_common_accessors_npm(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    wrapper = GuidedStartupAuthority(npm_intent=intent, npm_authority=authority)
    assert wrapper.source_format == intent.source_format == "npm"
    assert wrapper.execution_mode == intent.execution_mode
    assert wrapper.sessions_per_hour == authority.recording_policy.sessions_per_hour
    assert wrapper.source_root_canonical == intent.source_root_canonical
    assert wrapper.output_base_canonical == intent.output_policy.output_base_canonical
    assert wrapper.overwrite == intent.output_policy.overwrite is False
    assert (
        wrapper.protected_root_context_complete
        == intent.output_policy.protected_root_context_complete
    )
    assert wrapper.included_roi_ids == intent.selected_roi_ids
    assert (
        wrapper.per_roi_correction_strategy_map
        == intent.per_roi_correction_strategy_map
    )
    assert wrapper.feature_event is intent.feature_event


# ---------------------------------------------------------------------------
# GuidedStartupAuthority: verify_self_consistent
# ---------------------------------------------------------------------------


def test_verify_self_consistent_accepts_rwd_fixture(rwd_authority):
    GuidedStartupAuthority(rwd=rwd_authority).verify_self_consistent()


def test_verify_self_consistent_accepts_npm_fixture(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    GuidedStartupAuthority(npm_intent=intent, npm_authority=authority).verify_self_consistent()


def test_verify_self_consistent_refuses_tampered_rwd(rwd_authority):
    tampered = replace(rwd_authority, canonical_authorization_identity="0" * 64)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(rwd=tampered).verify_self_consistent()


def test_verify_self_consistent_refuses_tampered_npm_authority(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    tampered = replace(authority, canonical_authority_identity="0" * 64)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(npm_intent=intent, npm_authority=tampered).verify_self_consistent()


def test_verify_self_consistent_refuses_tampered_npm_intent(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    tampered = replace(intent, canonical_intent_identity="0" * 64)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(npm_intent=tampered, npm_authority=authority).verify_self_consistent()


def test_verify_self_consistent_refuses_mismatched_npm_pair(tmp_path):
    sub_a = tmp_path / "a"
    sub_a.mkdir()
    sub_b = tmp_path / "b"
    sub_b.mkdir()
    intent_a, authority_a = _npm_pair(sub_a)
    intent_b, _authority_b = _npm_pair_two_rois(sub_b)
    with pytest.raises(ValueError):
        GuidedStartupAuthority(
            npm_intent=intent_b, npm_authority=authority_a
        ).verify_self_consistent()


# ---------------------------------------------------------------------------
# compile_npm_generic_execution_payloads: positive coverage
# ---------------------------------------------------------------------------


def _compile(intent, authority):
    contract = build_guided_execution_startup_mapping_contract()
    return compile_npm_generic_execution_payloads(
        intent, authority, startup_mapping_contract=contract
    )


def test_accepted_npm_produces_derivation_result(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    assert isinstance(result, GuidedExecutionPayloadDerivationResult)
    assert result.ok is True
    assert result.runnable is False
    assert result.config_payload is not None
    assert result.candidate_manifest_payload is not None
    assert result.provenance_seed is not None
    assert result.blocking_issues == ()
    assert len(result.limiting_issues) == 1
    assert result.limiting_issues[0].category == "startup_transaction_unavailable"


def test_generic_identities_recompute_exactly(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    assert (
        compute_guided_execution_config_payload_identity(result.config_payload)
        == result.config_payload_identity
        == result.config_payload.canonical_config_payload_identity
    )
    assert (
        compute_guided_runner_candidate_manifest_payload_identity(
            result.candidate_manifest_payload
        )
        == result.candidate_manifest_payload_identity
        == result.candidate_manifest_payload.canonical_candidate_manifest_payload_identity
    )
    assert (
        compute_guided_startup_provenance_seed_identity(result.provenance_seed)
        == result.provenance_seed_identity
        == result.provenance_seed.canonical_provenance_seed_identity
    )


def test_real_parser_values_preserved_not_rwd_placeholders(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    values = {item.name: item.value for item in result.config_payload.values}
    recording_policy = authority.recording_policy
    assert values["npm_time_axis"] == recording_policy.time_axis_mode
    assert values["npm_led_col"] == recording_policy.led_state_column
    assert values["npm_region_prefix"] == recording_policy.roi_prefix
    assert values["npm_region_suffix"] == recording_policy.roi_suffix
    assert values["target_fs_hz"] == recording_policy.target_fs_hz
    assert values["chunk_duration_sec"] == recording_policy.configured_session_duration_sec
    # RWD-context placeholders (guided_execution_payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS'
    # unsupported_defaults) must never silently stand in for real NPM values.
    assert values["npm_led_col"] != "LED"
    assert values["npm_time_axis"] != "" or recording_policy.time_axis_mode == ""


def test_compiler_does_not_mutate_normalized_recording_identity(tmp_path):
    """compile_npm_generic_execution_payloads does not read, verify, or
    emit intent.normalized_recording_description_identity into any of its
    three output payloads -- it only proves the identity is left
    untouched by compilation. Normalized-recording identity verification
    (deserializing normalized_recording_payload_json and recomputing its
    identity) belongs to plan_guided_startup_transaction's NPM branch, not
    this compiler; see the integration test added when that wiring lands."""
    intent, authority = _npm_pair(tmp_path)
    before = intent.normalized_recording_description_identity
    result = _compile(intent, authority)
    assert intent.normalized_recording_description_identity == before
    assert len(before) == 64
    assert "normalized_recording" not in vars(result.config_payload)
    assert "normalized_recording" not in vars(result.candidate_manifest_payload)
    assert "normalized_recording" not in vars(result.provenance_seed)


def test_source_snapshot_identities_preserved(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    manifest = result.candidate_manifest_payload
    assert manifest.source_candidate_set_digest == intent.source_snapshot_set_identity
    assert manifest.source_candidate_content_digest == intent.source_snapshot_content_identity
    assert manifest.source_root_canonical == intent.source_root_canonical


def test_parser_policy_identity_preserved(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    assert result.candidate_manifest_payload.parser_contract_digest == intent.parser_policy_identity


def test_included_and_excluded_roi_scope_correct(tmp_path):
    intent, authority = _npm_pair_two_rois(tmp_path)
    result = _compile(intent, authority)
    manifest = result.candidate_manifest_payload
    assert set(manifest.included_roi_ids) == set(authority.roi_authority.selected_canonical_roi_ids)
    assert set(manifest.excluded_roi_ids) == set(authority.roi_authority.excluded_canonical_roi_ids)
    assert set(manifest.discovered_roi_ids) == set(authority.roi_authority.complete_canonical_roi_ids)
    assert not (set(manifest.included_roi_ids) & set(manifest.excluded_roi_ids))


def test_per_roi_correction_settings_correct(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    values = {item.name: item.value for item in result.config_payload.values}
    correction_values = {
        item.field_name: item.value
        for item in authority.correction_authority.correction_parameter_values
    }
    for name, value in correction_values.items():
        mapped_name = {"slope_constraint": "dynamic_fit_slope_constraint", "min_slope": "dynamic_fit_min_slope"}.get(name, name)
        assert values[mapped_name] == value


def test_per_roi_feature_settings_correct(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    values = {item.name: item.value for item in result.config_payload.values}
    feature_values = {
        item.field_name: item.value
        for item in authority.feature_authority.effective_values
    }
    for name, value in feature_values.items():
        assert values[name] == value


def test_application_build_identity_correct(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    assert (
        result.provenance_seed.application_build_identity
        == intent.application_build_identity.canonical_identity
    )


def test_output_policy_correct(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    result = _compile(intent, authority)
    values = {item.name: item.value for item in result.config_payload.values}
    assert intent.output_policy.overwrite is False
    assert values["acquisition_mode"] == "intermittent"


# ---------------------------------------------------------------------------
# compile_npm_generic_execution_payloads: authentic Guided execution mode
# ---------------------------------------------------------------------------


def test_compiler_accepts_execution_mode_both(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    assert intent.execution_mode == "both"
    result = _compile(intent, authority)
    assert result.ok is True


@pytest.mark.parametrize("real_mode", ("phasic", "tonic"))
def test_compiler_refuses_non_both_execution_mode(tmp_path, real_mode):
    """Guided Mode exposes no phasic-versus-tonic choice: the startup
    bridge compiler must refuse anything but "both", even for a genuinely
    valid, self-consistent, real NPM intent/authority built with
    execution_mode=phasic/tonic (ordinary backend and Full Control NPM
    execution remain untouched by this restriction elsewhere)."""
    intent, authority = _npm_pair(tmp_path, execution_mode=real_mode)
    assert intent.execution_mode == real_mode
    result = _compile(intent, authority)
    assert result.ok is False
    assert result.blocking_issues[0].category == "config_field_unsupported"


def test_intent_construction_refuses_unknown_execution_mode(tmp_path):
    """An execution mode outside {"phasic","tonic","both"} is already
    refused at intent construction, before the Guided bridge compiler is
    ever reached -- this is GuidedNpmProductionExecutionIntent's own
    existing invariant, not a new check."""
    draft = _valid_npm_stage2c_draft(tmp_path)
    draft.execution_intent = GuidedNewAnalysisExecutionIntent(execution_mode="unknown_mode")
    kwargs = _npm_validation_fixture_kwargs()
    outcome = validate_current_guided_draft_for_backend(
        draft, validation_revision=4, **kwargs
    )
    assert outcome.status != "validator_accepted"


# ---------------------------------------------------------------------------
# compile_npm_generic_execution_payloads: refusal coverage
# ---------------------------------------------------------------------------


def test_refuses_mismatched_intent_and_authority(tmp_path):
    sub_a = tmp_path / "a"
    sub_a.mkdir()
    sub_b = tmp_path / "b"
    sub_b.mkdir()
    intent_a, authority_a = _npm_pair(sub_a)
    intent_b, _authority_b = _npm_pair_two_rois(sub_b)
    result = _compile(intent_b, authority_a)
    assert result.ok is False
    assert result.blocking_issues[0].category == "production_intent_identity_mismatch"


def test_refuses_tampered_authority_identity(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    tampered = replace(authority, canonical_authority_identity="0" * 64)
    result = _compile(intent, tampered)
    assert result.ok is False
    assert result.blocking_issues[0].category == "authorization_identity_mismatch"


def test_refuses_tampered_intent_identity(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    tampered = replace(intent, canonical_intent_identity="0" * 64)
    result = _compile(tampered, authority)
    assert result.ok is False
    assert result.blocking_issues[0].category == "production_intent_identity_mismatch"


def test_refuses_malformed_parser_policy_content(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    tampered_policy = replace(
        authority.recording_policy, parser_policy_content_json="{}"
    )
    # Bypass GuidedNpmRecordingPolicy.__post_init__'s own identity binding
    # (which would itself refuse this) via a raw object.__new__ patch so the
    # compiler's own defensive parsing is what is actually exercised here.
    tampered_authority = object.__new__(type(authority))
    for field_name, value in authority.__dict__.items():
        object.__setattr__(tampered_authority, field_name, value)
    object.__setattr__(tampered_authority, "recording_policy", tampered_policy)
    result = _compile(intent, tampered_authority)
    assert result.ok is False
    assert result.blocking_issues[0].category == "authorization_identity_mismatch"


def test_refuses_roi_authority_mismatch(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    # GuidedNpmRoiAuthority's own __post_init__ forbids a selected ROI
    # outside the complete inventory, so dataclasses.replace() cannot build
    # this tampered value -- bypass __post_init__ directly, since the point
    # of this test is proving the *compiler* refuses an internally
    # inconsistent authority, not re-testing GuidedNpmRoiAuthority's own
    # already-covered invariant.
    tampered_roi = object.__new__(type(authority.roi_authority))
    for field_name, value in authority.roi_authority.__dict__.items():
        object.__setattr__(tampered_roi, field_name, value)
    object.__setattr__(tampered_roi, "selected_canonical_roi_ids", ("SomeOtherRoi",))
    tampered_authority = object.__new__(type(authority))
    for field_name, value in authority.__dict__.items():
        object.__setattr__(tampered_authority, field_name, value)
    object.__setattr__(tampered_authority, "roi_authority", tampered_roi)
    result = _compile(intent, tampered_authority)
    assert result.ok is False
    assert result.blocking_issues[0].category == "authorization_identity_mismatch"


def test_refuses_correction_authority_mismatch(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    tampered_correction = replace(
        authority.correction_authority,
        correction_parameter_values=(),
    )
    tampered_authority = object.__new__(type(authority))
    for field_name, value in authority.__dict__.items():
        object.__setattr__(tampered_authority, field_name, value)
    object.__setattr__(tampered_authority, "correction_authority", tampered_correction)
    result = _compile(intent, tampered_authority)
    assert result.ok is False
    assert result.blocking_issues[0].category == "authorization_identity_mismatch"


def test_refuses_feature_authority_mismatch(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    tampered_feature = replace(
        authority.feature_authority,
        effective_values=(),
    )
    tampered_authority = object.__new__(type(authority))
    for field_name, value in authority.__dict__.items():
        object.__setattr__(tampered_authority, field_name, value)
    object.__setattr__(tampered_authority, "feature_authority", tampered_feature)
    result = _compile(intent, tampered_authority)
    assert result.ok is False
    assert result.blocking_issues[0].category == "authorization_identity_mismatch"


def test_refuses_mapping_contract_mismatch(tmp_path):
    intent, authority = _npm_pair(tmp_path)
    contract = build_guided_execution_startup_mapping_contract()
    tampered_contract = replace(
        contract, supported_mapping_contract_version="unsupported.v0"
    )
    result = compile_npm_generic_execution_payloads(
        intent, authority, startup_mapping_contract=tampered_contract
    )
    # The compiler itself does not gate on startup_mapping_contract's
    # supported_mapping_contract_version (that check belongs to the shared
    # _gate_issue, not the per-format compiler) -- confirm it still
    # succeeds here, and that the resulting payload carries the *intent's*
    # real mapping_contract_version untouched, so the mismatch remains
    # detectable downstream.
    assert result.ok is True
    assert result.provenance_seed.production_mapping_contract_version == intent.mapping_contract_version
    assert intent.mapping_contract_version != tampered_contract.supported_mapping_contract_version
