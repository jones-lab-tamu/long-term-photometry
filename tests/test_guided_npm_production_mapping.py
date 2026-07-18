from dataclasses import replace
import builtins
import hashlib
import inspect
import os
from pathlib import Path

import pytest

from photometry_pipeline.guided_backend_validation_request import (
    GuidedBackendNpmParserRequest,
    compute_guided_backend_validation_request_identity,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationWorkflowIssue,
    validate_current_guided_draft_for_backend,
)
from photometry_pipeline.guided_backend_validator import (
    GuidedBackendValidatorContract,
)
from photometry_pipeline.guided_normalized_recording import (
    deserialize_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.guided_production_mapping import (
    GuidedNpmProductionMappingSuccess,
    build_application_build_identity,
    build_guided_production_mapping_contract,
    compute_guided_npm_production_execution_intent_identity,
    map_guided_npm_validation_outcome_to_execution_intent,
)
import photometry_pipeline.guided_production_mapping as production_mapping
from photometry_pipeline.guided_run_readiness import evaluate_guided_run_readiness
from photometry_pipeline.io.npm_contract import NpmParserContract

from tests.test_guided_backend_validation_materialization import (
    _valid_npm_stage2c_draft,
)


def _accepted_npm(tmp_path: Path):
    draft = _valid_npm_stage2c_draft(tmp_path)
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
    assert outcome.compile_result is not None
    request = outcome.compile_result.request
    assert request is not None
    return outcome, request


def _accepted_npm_default_initialized(tmp_path: Path):
    draft = _valid_npm_stage2c_draft(tmp_path)
    draft.feature_event_profile_status = "default_initialized"
    draft.feature_event_explicitly_applied = False
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
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    assert outcome.compile_result is not None
    request = outcome.compile_result.request
    assert request is not None
    return outcome, request


def _accepted_npm_two_rois(tmp_path: Path):
    from photometry_pipeline.guided_new_analysis_plan import (
        compute_guided_local_preview_source_setup_signature,
    )

    draft = _valid_npm_stage2c_draft(tmp_path)
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
    local_preview_signature = compute_guided_local_preview_source_setup_signature(
        draft
    )
    first_choice = replace(
        draft.per_roi_correction_strategy_choices[0],
        source_setup_signature=local_preview_signature,
    )
    draft.per_roi_correction_strategy_choices = [
        first_choice,
        replace(first_choice, roi_id="Region1"),
    ]
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
    assert outcome.compile_result is not None
    assert outcome.compile_result.request is not None
    return outcome, outcome.compile_result.request


def _build_identity():
    return build_application_build_identity(
        distribution_name="photometry-test",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="test-revision",
        source_tree_state="clean",
    )


def _map(
    outcome,
    *,
    expected_validation_revision: int | None = 4,
    expected_plan_identity: str | None = None,
):
    return map_guided_npm_validation_outcome_to_execution_intent(
        outcome,
        expected_validation_revision=expected_validation_revision,
        expected_plan_identity=(
            outcome.guided_plan_identity
            if expected_plan_identity is None
            else expected_plan_identity
        ),
        application_build_identity=_build_identity(),
        mapping_contract=build_guided_production_mapping_contract(),
    )


def _outcome_with_request(outcome, request):
    """Rebind a synthetic identity-variation fixture for lower-level tests."""
    identity = compute_guided_backend_validation_request_identity(request)
    return replace(
        outcome,
        request_identity=identity,
        accepted_request_identity=identity,
        compile_result=replace(
            outcome.compile_result,
            request=request,
            canonical_request_identity=identity,
        ),
    )


def _unsafe_replace(instance, **changes):
    """Create malformed outcome fixtures to exercise boundary refusals."""
    clone = object.__new__(type(instance))
    for name, value in instance.__dict__.items():
        object.__setattr__(clone, name, changes.get(name, value))
    for name, value in changes.items():
        if name not in instance.__dict__:
            object.__setattr__(clone, name, value)
    return clone


def _boundary_failure(outcome, **kwargs):
    return map_guided_npm_validation_outcome_to_execution_intent(
        outcome,
        expected_validation_revision=kwargs.pop("expected_validation_revision", 4),
        expected_plan_identity=kwargs.pop(
            "expected_plan_identity", getattr(outcome, "guided_plan_identity", None)
        ),
        application_build_identity=_build_identity(),
        mapping_contract=build_guided_production_mapping_contract(),
    )


def _assert_mapped(result):
    assert isinstance(result, GuidedNpmProductionMappingSuccess)
    assert result.capability.status == (
        "supported_by_production_mapping_but_startup_not_implemented"
    )
    assert result.capability.production_mapping_supported is True
    assert result.capability.startup_available is False
    assert result.capability.runnable is False
    assert result.intent.runner_contract_version == (
        "guided_npm_runner_not_yet_startable.v1"
    )
    return result


def test_accepted_npm_maps_to_immutable_non_runnable_intent(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    result = _assert_mapped(_map(outcome))

    with pytest.raises((AttributeError, TypeError)):
        result.intent.execution_mode = "tonic"

    assert result.intent.source_format == "npm"
    assert result.intent.acquisition_mode == "intermittent"
    assert result.intent.source_request_identity == outcome.accepted_request_identity
    assert result.intent.validation_revision == 4
    assert result.intent.current_plan_identity == outcome.guided_plan_identity
    assert result.intent.normalized_recording_payload_json
    assert result.intent.parser_policy_content_json
    assert result.intent.per_session_resolved_evidence_identity
    assert result.intent.canonical_intent_identity == (
        compute_guided_npm_production_execution_intent_identity(result.intent)
    )


def test_default_initialized_feature_profile_without_apply_maps_successfully(
    tmp_path: Path,
):
    """Repair regression: a loaded Default Feature Detection
    profile left as "default_initialized" (never explicitly applied) is
    real production-accepted state -- the same real materialization path
    Check My Setup itself uses, not a hand-built intent. Production
    mapping accepts the per-ROI entries' truthful explicit_user_mark=False
    here because the enclosing profile is itself current and
    default_initialized (see feature_entry_provenance_valid); it must not
    require a falsified explicit mark."""
    outcome, _request = _accepted_npm_default_initialized(tmp_path)

    result = _assert_mapped(_map(outcome))
    entries = result.intent.feature_event.per_roi_feature_event_map
    assert entries
    for entry in entries:
        assert entry.source == "default"
        assert entry.explicit_user_mark is False
        assert entry.current_or_stale == "current"
        assert entry.effective_config_fields


@pytest.mark.parametrize(
    "mutation,detail_code",
    (
        ("wrong_type", "validation_outcome_type_invalid"),
        ("not_accepted", "validation_outcome_not_accepted"),
        ("stale", "validation_outcome_stale"),
        ("blockers", "validation_outcome_has_blockers"),
        ("missing_compile", "validation_compile_result_missing"),
        ("missing_request", "validation_request_missing"),
        ("missing_identity", "accepted_request_identity_missing"),
    ),
)
def test_outcome_boundary_refuses_invalid_or_incomplete_outcomes(
    tmp_path: Path, mutation: str, detail_code: str
):
    outcome, _request = _accepted_npm(tmp_path)
    if mutation == "wrong_type":
        malformed = object()
    elif mutation == "not_accepted":
        malformed = _unsafe_replace(
            outcome,
            status="validator_refused",
            accepted_for_backend_validation=False,
        )
    elif mutation == "stale":
        malformed = _unsafe_replace(outcome, stale=True)
    elif mutation == "blockers":
        malformed = _unsafe_replace(
            outcome,
            blocking_issues=(
                GuidedBackendValidationWorkflowIssue(
                    stage="validator",
                    category="test_blocker",
                    section="test",
                    message="blocked",
                    detail_code="test_blocker",
                ),
            ),
        )
    elif mutation == "missing_compile":
        malformed = _unsafe_replace(outcome, compile_result=None)
    elif mutation == "missing_request":
        malformed_compile = _unsafe_replace(
            outcome.compile_result, request=None
        )
        malformed = _unsafe_replace(outcome, compile_result=malformed_compile)
    else:
        malformed = _unsafe_replace(outcome, accepted_request_identity=None)

    result = _boundary_failure(malformed)
    assert result.status == "refused"
    assert result.blocking_issues[0].category == "stale_or_mismatched_validation"
    assert result.blocking_issues[0].detail_code == detail_code


def test_revision_and_plan_identity_are_explicit_and_current(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    mapped = _assert_mapped(_map(outcome))

    for expected_revision, detail_code in ((None, "validation_revision_missing"), (5, "validation_revision_mismatch"), (-1, "validation_revision_mismatch")):
        result = _boundary_failure(
            outcome, expected_validation_revision=expected_revision
        )
        assert result.blocking_issues[0].detail_code == detail_code

    missing_revision_outcome = _unsafe_replace(outcome, validation_revision=None)
    result = _boundary_failure(missing_revision_outcome)
    assert result.blocking_issues[0].detail_code == "validation_revision_missing"

    other_plan_identity = hashlib.sha256(b"different-guided-plan").hexdigest()
    assert other_plan_identity != outcome.accepted_request_identity
    assert _boundary_failure(
        outcome, expected_plan_identity=outcome.accepted_request_identity
    ).blocking_issues[0].detail_code == "request_identity_used_as_plan_identity"
    assert _boundary_failure(
        outcome, expected_plan_identity=other_plan_identity
    ).blocking_issues[0].detail_code == "guided_plan_identity_mismatch"
    missing_plan_outcome = _unsafe_replace(outcome, guided_plan_identity=None)
    assert _boundary_failure(missing_plan_outcome).blocking_issues[0].detail_code == (
        "guided_plan_identity_missing"
    )

    revision_mutation = _unsafe_replace(outcome, validation_revision=5)
    revision_mapped = _assert_mapped(
        _boundary_failure(revision_mutation, expected_validation_revision=5)
    )
    plan_mutation = _unsafe_replace(outcome, guided_plan_identity=other_plan_identity)
    plan_mapped = _assert_mapped(
        _boundary_failure(plan_mutation, expected_plan_identity=other_plan_identity)
    )
    assert revision_mapped.intent.canonical_intent_identity != (
        mapped.intent.canonical_intent_identity
    )
    assert plan_mapped.intent.canonical_intent_identity != (
        mapped.intent.canonical_intent_identity
    )


def test_request_only_public_mapper_and_caller_asserted_acceptance_are_unavailable():
    assert not hasattr(
        production_mapping, "map_guided_npm_validation_request_to_execution_intent"
    )
    public_parameters = inspect.signature(
        map_guided_npm_validation_outcome_to_execution_intent
    ).parameters
    assert "validation_status" not in public_parameters
    internal_parameters = inspect.signature(
        production_mapping._map_verified_guided_npm_request_to_execution_intent
    ).parameters
    for name in (
        "accepted_request_identity",
        "validation_revision",
        "current_plan_identity",
    ):
        assert internal_parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize("execution_mode", ("phasic", "tonic", "both"))
def test_signal_only_is_supported_without_global_dynamic_fit_requirement(
    tmp_path: Path, execution_mode: str
):
    outcome, request = _accepted_npm(tmp_path)
    correction = request.correction
    signal_only_entries = tuple(
        replace(
            entry,
            strategy_family="signal_only_f0",
            dynamic_fit_mode=None,
            selected_strategy="signal_only_f0",
        )
        for entry in correction.per_roi_production_strategy_map
    )
    request = replace(
        request,
        acquisition_dataset=replace(
            request.acquisition_dataset, execution_mode=execution_mode
        ),
        correction=replace(
            correction,
            # This legacy field remains present for request compatibility; the
            # NPM mapper must not use it as the capability decision.
            global_dynamic_fit_mode="global_linear_regression",
            per_roi_production_strategy_map=signal_only_entries,
        ),
    )

    result = _assert_mapped(_map(_outcome_with_request(outcome, request)))
    assert {entry.strategy_family for entry in result.intent.per_roi_correction_strategy_map} == {
        "signal_only_f0"
    }
    assert result.intent.execution_mode == execution_mode


def test_mixed_signal_only_and_dynamic_fit_is_bound_per_roi(tmp_path: Path):
    outcome, request = _accepted_npm_two_rois(tmp_path)
    entries = request.correction.per_roi_production_strategy_map
    mixed_entries = (
        replace(
            entries[0],
            strategy_family="signal_only_f0",
            dynamic_fit_mode=None,
            selected_strategy="signal_only_f0",
        ),
        entries[1],
    )
    feature_entries = request.feature_event.per_roi_feature_event_map
    mixed_feature_entries = (
        replace(
            feature_entries[0],
            source="override",
            feature_event_profile_id="feature-profile-roi0-custom",
            override_config_fields=feature_entries[0].effective_config_fields[:1],
        ),
        feature_entries[1],
    )
    result = _assert_mapped(
        _map(
            _outcome_with_request(
                outcome,
                replace(
                    request,
                    correction=replace(
                        request.correction,
                        per_roi_production_strategy_map=mixed_entries,
                    ),
                    feature_event=replace(
                        request.feature_event,
                        per_roi_feature_event_map=mixed_feature_entries,
                    ),
                ),
            )
        )
    )
    assert [
        entry.strategy_family for entry in result.intent.per_roi_correction_strategy_map
    ] == ["signal_only_f0", "dynamic_fit"]
    assert len(result.intent.feature_event.per_roi_feature_event_map) == 2
    assert [
        entry.source for entry in result.intent.feature_event.per_roi_feature_event_map
    ] == ["override", "default"]


def test_feature_defaults_and_override_identity_are_preserved(tmp_path: Path):
    outcome, request = _accepted_npm(tmp_path)
    default_result = _assert_mapped(_map(outcome))
    default_entry = default_result.intent.feature_event.per_roi_feature_event_map[0]
    assert default_entry.source == "default"
    assert default_entry.effective_config_fields

    override_entry = replace(
        request.feature_event.per_roi_feature_event_map[0],
        source="override",
        feature_event_profile_id="feature-profile-custom",
        override_config_fields=request.feature_event.per_roi_feature_event_map[
            0
        ].effective_config_fields[:1],
    )
    custom_request = replace(
        request,
        feature_event=replace(
            request.feature_event,
            per_roi_feature_event_map=(override_entry,),
        ),
    )
    custom_result = _assert_mapped(
        _map(_outcome_with_request(outcome, custom_request))
    )
    assert custom_result.intent.feature_event.per_roi_feature_event_map[0].source == "override"
    assert custom_result.intent.feature_payload_identity != default_result.intent.feature_payload_identity


@pytest.mark.parametrize(
    "tamper",
    (
        lambda entry: replace(entry, explicit_user_mark=False),
        lambda entry: replace(entry, current_or_stale="stale"),
        lambda entry: replace(entry, effective_config_fields=()),
    ),
)
def test_incomplete_or_stale_per_roi_feature_entry_still_refuses(
    tmp_path: Path, tamper
):
    """Repair regression (negative case): the strict per-entry
    completeness check in guided_production_mapping.py must continue
    refusing a genuinely non-explicit, stale, or empty per-ROI feature
    entry with per_roi_feature_entry_incomplete -- proves the repair
    (which only changed how a current default-sourced entry is marked
    explicit) did not widen this check to accept genuinely incomplete
    entries too."""
    outcome, request = _accepted_npm(tmp_path)
    tampered_entry = tamper(request.feature_event.per_roi_feature_event_map[0])
    tampered_request = replace(
        request,
        feature_event=replace(
            request.feature_event,
            per_roi_feature_event_map=(tampered_entry,),
        ),
    )
    result = _map(_outcome_with_request(outcome, tampered_request))
    assert result.status == "refused"
    assert result.blocking_issues[0].category == "incomplete_feature_settings"
    assert result.blocking_issues[0].detail_code == "per_roi_feature_entry_incomplete"


def test_override_entry_without_explicit_mark_still_refuses(tmp_path: Path):
    """Negative provenance case B: an override/custom-sourced entry must
    still carry an explicit mark -- the narrow default_initialized
    acceptance in feature_entry_provenance_valid only applies to
    source == "default", never to an override."""
    outcome, request = _accepted_npm(tmp_path)
    override_entry = replace(
        request.feature_event.per_roi_feature_event_map[0],
        source="override",
        feature_event_profile_id="feature-profile-custom",
        override_config_fields=request.feature_event.per_roi_feature_event_map[
            0
        ].effective_config_fields[:1],
        explicit_user_mark=False,
    )
    tampered_request = replace(
        request,
        feature_event=replace(
            request.feature_event,
            per_roi_feature_event_map=(override_entry,),
        ),
    )
    result = _map(_outcome_with_request(outcome, tampered_request))
    assert result.status == "refused"
    assert result.blocking_issues[0].category == "incomplete_feature_settings"
    assert result.blocking_issues[0].detail_code == "per_roi_feature_entry_incomplete"


@pytest.mark.parametrize(
    "tamper",
    (
        lambda entry: replace(entry, current_or_stale="stale"),
        lambda entry: replace(entry, effective_config_fields=()),
        lambda entry: replace(entry, feature_event_profile_id="a-different-profile-id"),
    ),
)
def test_default_initialized_entry_edge_cases_still_refuse(tmp_path: Path, tamper):
    """Negative provenance cases C/D/E: even under the narrow
    default_initialized acceptance, a stale entry (C), an entry with no
    effective settings (D), or a default-sourced entry whose profile
    identity does not match the enclosing accepted default profile (E)
    must still refuse -- proves the new acceptance path does not infer
    validity from a non-explicit entry alone."""
    outcome, request = _accepted_npm_default_initialized(tmp_path)
    tampered_entry = tamper(request.feature_event.per_roi_feature_event_map[0])
    tampered_request = replace(
        request,
        feature_event=replace(
            request.feature_event,
            per_roi_feature_event_map=(tampered_entry,),
        ),
    )
    result = _map(_outcome_with_request(outcome, tampered_request))
    assert result.status == "refused"
    assert result.blocking_issues[0].category == "incomplete_feature_settings"
    assert result.blocking_issues[0].detail_code == "per_roi_feature_entry_incomplete"


def test_stray_feature_entry_roi_not_in_included_set_still_refuses(tmp_path: Path):
    """Negative provenance case F: a per-ROI feature entry for a ROI
    outside the accepted included set must not be silently accepted as
    authoritative, even when it is otherwise a valid default_initialized
    entry -- the exact-coverage check catches this before the per-entry
    provenance rule is ever reached."""
    outcome, request = _accepted_npm_default_initialized(tmp_path)
    stray_entry = replace(
        request.feature_event.per_roi_feature_event_map[0], roi_id="NotIncludedRoi"
    )
    tampered_request = replace(
        request,
        feature_event=replace(
            request.feature_event,
            per_roi_feature_event_map=(stray_entry,),
        ),
    )
    result = _map(_outcome_with_request(outcome, tampered_request))
    assert result.status == "refused"
    assert result.blocking_issues[0].category == "incomplete_feature_settings"
    assert result.blocking_issues[0].detail_code == "per_roi_feature_map_incomplete"


def test_per_session_evidence_changes_npm_identity(tmp_path: Path):
    outcome, request = _accepted_npm(tmp_path)
    first = _assert_mapped(_map(outcome))
    description = deserialize_normalized_recording_description(
        request.normalized_recording_description
    )
    adapter_evidence = dict(description.adapter_evidence)
    npm_sessions = [dict(item) for item in adapter_evidence["npm_sessions"]]
    npm_sessions[0]["resolved_timestamp_column"] = "AlternateTimestamp"
    adapter_evidence["npm_sessions"] = npm_sessions
    changed_description = replace(description, adapter_evidence=adapter_evidence)
    changed_payload = serialize_normalized_recording_description(changed_description)
    changed_request = replace(
        request,
        normalized_recording_description=changed_payload,
        normalized_recording_description_identity=changed_payload[
            "normalized_recording_description_identity"
        ],
    )

    second = _assert_mapped(
        _map(_outcome_with_request(outcome, changed_request))
    )
    assert second.intent.normalized_recording_description_identity != (
        first.intent.normalized_recording_description_identity
    )
    assert second.intent.per_session_resolved_evidence_identity != (
        first.intent.per_session_resolved_evidence_identity
    )


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_missing_or_duplicate_session_evidence_refuses_mapping(
    tmp_path: Path, mutation: str
):
    outcome, request = _accepted_npm(tmp_path)
    payload = dict(request.normalized_recording_description)
    evidence = dict(payload["adapter_evidence"])
    sessions = [dict(item) for item in evidence["npm_sessions"]]
    if mutation == "missing":
        sessions[0].pop("resolved_timestamp_column")
    else:
        sessions.append(dict(sessions[0]))
    evidence["npm_sessions"] = sessions
    payload["adapter_evidence"] = evidence
    result = _map(
        _outcome_with_request(
            outcome, replace(request, normalized_recording_description=payload)
        )
    )

    assert result.status == "refused"
    assert result.blocking_issues[0].category == "per_session_evidence_not_identity_bound"


def test_identity_is_stable_for_parser_policy_mapping_order_and_rejects_nonfinite(
    tmp_path: Path,
):
    outcome, request = _accepted_npm(tmp_path)
    first = _assert_mapped(_map(outcome))
    parser = request.parser
    reordered_content = dict(reversed(list(parser.parser_contract_content.items())))
    reordered_parser = GuidedBackendNpmParserRequest(
        schema_name=parser.schema_name,
        schema_version=parser.schema_version,
        timestamp_column_candidates=parser.timestamp_column_candidates,
        parser_contract_digest=parser.parser_contract_digest,
        parser_contract_content=reordered_content,
    )
    reordered = _assert_mapped(
        _map(_outcome_with_request(outcome, replace(request, parser=reordered_parser)))
    )
    assert reordered.intent.canonical_intent_identity == (
        first.intent.canonical_intent_identity
    )

    with pytest.raises(ValueError):
        replace(first.intent, target_fs_hz=float("nan"))


def test_npm_mapping_performs_no_source_io(tmp_path: Path, monkeypatch):
    outcome, _request = _accepted_npm(tmp_path)

    def fail(*_args, **_kwargs):
        raise AssertionError("NPM production mapping attempted source I/O")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_bytes", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(Path, "iterdir", fail)
    monkeypatch.setattr(Path, "glob", fail)
    monkeypatch.setattr(Path, "rglob", fail)
    monkeypatch.setattr(os, "listdir", fail)
    _assert_mapped(_map(outcome))


def test_npm_readiness_text_and_visible_run_control_remain_disabled(tmp_path: Path):
    outcome, _request = _accepted_npm(tmp_path)
    result = evaluate_guided_run_readiness(
        validation_outcome=outcome,
        validation_revision=3,
        current_gui_revision=3,
    )
    assert result.status == "validated_npm_not_available"
    assert result.ready is False
    assert result.visible_run_control_enabled is False
    assert result.user_summary == (
        "This NPM recording setup was checked successfully. Running NPM analyses "
        "is not available yet."
    )
