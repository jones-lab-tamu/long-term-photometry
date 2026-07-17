import ast
import builtins
from dataclasses import fields, replace
import os
from pathlib import Path
import pytest

import photometry_pipeline.guided_backend_validation_request as validation_request
import photometry_pipeline.guided_backend_validator as validator
import photometry_pipeline.guided_execution_preflight as preflight
import photometry_pipeline.guided_production_mapping as mapping
import photometry_pipeline.guided_run_authorization as authorization
import photometry_pipeline.guided_execution_payloads as payloads
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowOutcome,
    build_guided_backend_validation_parser_contract,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.config import Config
from tests.test_guided_backend_validator import _contract as _validator_contract


# Test fixtures and builders adapted from test_guided_run_authorization.py
def _accepted_outcome():
    from tests.test_guided_backend_validator import (
        _request as _valid_request,
        _typed,
    )
    request = _valid_request()
    # Inject target_fs_hz into the request dataset semantic values
    new_dataset = replace(
        request.acquisition_dataset,
        semantic_values=request.acquisition_dataset.semantic_values + (_typed("target_fs_hz", 40.0),)
    )
    from tests.test_guided_backend_validator import _normalized_recording_identity_for
    request = replace(
        request,
        acquisition_dataset=new_dataset,
        normalized_recording_description_identity=(
            _normalized_recording_identity_for(
                request.source, new_dataset, request.roi_scope, request.parser
            )
        ),
    )

    identity = validation_request.compute_guided_backend_validation_request_identity(request)
    compiled = validation_request.GuidedBackendValidationCompileSuccess(request, identity)
    validated = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=_validator_contract(),
    )
    assert validated.accepted
    return GuidedBackendValidationWorkflowOutcome(
        status="validator_accepted",
        accepted_for_backend_validation=True,
        run_authorization=False,
        request_identity=identity,
        validation_result=validated,
        compile_result=compiled,
        materialization_result=object(),
        blocking_issues=(),
        user_summary="Accepted.",
    )


def _rebuild_normalized_recording_identity(intent) -> str:
    """Recompute the normalized recording description identity from an
    intent's own source/acquisition/ROI/parser fields, the same way
    guided_execution_payloads.derive_guided_execution_payloads verifies it
    -- used by tests that deliberately mutate one of those fields and must
    keep the identity self-consistent with it."""
    from types import SimpleNamespace

    from photometry_pipeline.guided_normalized_recording import (
        build_rwd_normalized_recording_description,
        compute_normalized_recording_description_identity,
    )

    excluded_path = (
        intent.input_source.candidate_files[-1].canonical_relative_path
        if intent.acquisition.exclude_incomplete_final_rwd_chunk
        and intent.input_source.candidate_files
        else None
    )
    description = build_rwd_normalized_recording_description(
        source_root_canonical=intent.input_source.source_root_canonical,
        candidate_snapshot=SimpleNamespace(
            candidates=intent.input_source.candidate_files,
            source_candidate_set_digest=intent.input_source.source_candidate_set_digest,
            source_candidate_content_digest=intent.input_source.source_candidate_content_digest,
        ),
        session_duration_sec=intent.acquisition.session_duration_sec,
        sessions_per_hour=intent.acquisition.sessions_per_hour,
        timeline_anchor_mode=intent.acquisition.timeline_anchor_mode,
        acquisition_mode=intent.acquisition.acquisition_mode,
        discovered_roi_ids=intent.roi_scope.discovered_roi_ids,
        included_roi_ids=intent.roi_scope.included_roi_ids,
        rwd_time_col=intent.acquisition.rwd_time_col,
        uv_suffix=intent.acquisition.uv_suffix,
        sig_suffix=intent.acquisition.sig_suffix,
        parser_contract_digest=intent.parser.parser_contract_digest,
        target_fs_hz=next(
            (
                item.value
                for item in intent.acquisition.semantic_values
                if item.field_name == "target_fs_hz"
            ),
            None,
        ),
        missing_canonical_relative_paths=tuple(
            item.canonical_relative_path
            for item in intent.input_source.approved_missing_candidates
        ),
        excluded_canonical_relative_path=excluded_path,
    )
    return compute_normalized_recording_description_identity(description)


def _build_app_identity():
    return mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )


def _request_auth(outcome=None):
    return authorization.build_guided_run_authorization_request(
        stored_validation_outcome=outcome or _accepted_outcome(),
        stored_validation_outcome_revision=3,
        current_gui_revision=3,
        current_validation_context=GuidedBackendValidationGuiContext(
            draft=GuidedNewAnalysisDraftPlan(),
            parser_contract=build_guided_backend_validation_parser_contract(),
            additional_protected_roots=(),
            validator_contract=_validator_contract(),
            revision=3,
        ),
        application_build_identity=_build_app_identity(),
        production_mapping_contract=mapping.build_guided_production_mapping_contract(),
    )


def _accepted_candidate(candidate_request):
    provisional = preflight.GuidedCandidateManifestExecutionPreflightResult(
        status="accepted",
        accepted=True,
        contract_version=candidate_request.contract_version,
        runner_contract_version=candidate_request.runner_contract_version,
        expected_candidate_set_digest=candidate_request.expected_candidate_set_digest,
        expected_candidate_content_digest=candidate_request.expected_candidate_content_digest,
        actual_candidate_set_digest=candidate_request.expected_candidate_set_digest,
        actual_candidate_content_digest=candidate_request.expected_candidate_content_digest,
        actual_candidates=candidate_request.expected_candidates,
        blocking_issues=(),
        canonical_preflight_identity="0" * 64,
    )
    return replace(
        provisional,
        canonical_preflight_identity=preflight.compute_guided_candidate_preflight_identity(provisional),
    )


def _accepted_roi(roi_request):
    provisional = preflight.GuidedRoiExecutionPreflightResult(
        status="accepted",
        accepted=True,
        contract_version=roi_request.contract_version,
        runner_contract_version=roi_request.runner_contract_version,
        accepted_candidate_preflight_identity=roi_request.accepted_candidate_preflight_identity,
        source_candidate_content_digest=roi_request.source_candidate_content_digest,
        parser_contract_digest=roi_request.parser_contract_digest,
        expected_strict_roi_inventory_digest=roi_request.expected_strict_roi_inventory_digest,
        actual_strict_roi_inventory_digest=roi_request.expected_strict_roi_inventory_digest,
        actual_discovered_roi_ids=roi_request.expected_discovered_roi_ids,
        actual_included_roi_ids=roi_request.expected_included_roi_ids,
        actual_excluded_roi_ids=roi_request.expected_excluded_roi_ids,
        blocking_issues=(),
        canonical_preflight_identity="0" * 64,
    )
    return replace(
        provisional,
        canonical_preflight_identity=preflight.compute_guided_roi_preflight_identity(provisional),
    )


def _unchecked(instance, **changes):
    result = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            result,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return result


@pytest.fixture
def auth_result(monkeypatch):
    req = _request_auth()
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: req.stored_validation_outcome,
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
    res = authorization.authorize_guided_run(req)
    assert res.status == "authorized"
    return res


# Test A: Default/current contract returns nonrunnable
def test_valid_authorized_result_derives_nonrunnable(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract()
    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)

    assert (
        contract.contract_version
        == "guided_execution_startup_mapping.post_4J14l.v2"
    )
    assert contract.exact_candidate_manifest_consumption_capable is True
    assert contract.exact_roi_consumption_capable is True
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE
    assert result.ok is True
    assert result.runnable is False
    assert result.config_payload is not None
    assert result.candidate_manifest_payload is not None
    assert result.provenance_seed is not None
    assert result.runner_request is None
    assert len(result.limiting_issues) == 1
    assert result.limiting_issues[0].category == "startup_transaction_unavailable"
    assert len(result.blocking_issues) == 0

    # Ensure no side effects are marked True
    assert result.no_files_written is True
    assert result.no_directories_created is True
    assert result.no_artifacts_created is True
    assert result.no_output_allocated is True
    assert result.no_run_id_allocated is True
    assert result.no_config_file_generated is True
    assert result.no_argv_generated is True
    assert result.no_runner_invoked is True


# Test B: Runner consumption capability does not imply startup/runnability.
def test_valid_authorized_result_remains_nonrunnable_when_booleans_true(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract(
        exact_candidate_manifest_consumption_capable=True,
        exact_roi_consumption_capable=True,
    )
    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)

    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE
    assert result.ok is True
    assert result.runnable is False
    assert result.config_payload is not None
    assert result.candidate_manifest_payload is not None
    assert result.provenance_seed is not None
    assert result.runner_request is None
    assert len(result.limiting_issues) == 1
    assert result.limiting_issues[0].category == "startup_transaction_unavailable"
    assert len(result.blocking_issues) == 0


def test_successful_nonrunnable_path_has_no_stale_runner_limiter():
    source = Path(payloads.__file__).read_text(encoding="utf-8")
    assert "runner_contract_missing_exact_manifest" not in source


# Test C
def test_refuses_when_authorization_status_is_invalid(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract()
    for bad_status in ("accepted", "refused", "cancelled", "unknown"):
        bad_auth = _unchecked(auth_result, status=bad_status)
        result = payloads.derive_guided_execution_payloads(bad_auth, startup_mapping_contract=contract)
        assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
        assert result.ok is False
        assert result.blocking_issues[0].category == "authorization_not_accepted"


# Test D
def test_refuses_when_authorization_boolean_gates_are_false(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract()
    
    # authorized is False
    bad_auth = _unchecked(auth_result, authorized=False)
    result = payloads.derive_guided_execution_payloads(bad_auth, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "authorization_not_accepted"

    # run_authorization is False
    bad_auth = _unchecked(auth_result, run_authorization=False)
    result = payloads.derive_guided_execution_payloads(bad_auth, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "authorization_not_accepted"

    # canonical_authorization_identity is missing
    bad_auth = _unchecked(auth_result, canonical_authorization_identity=None)
    result = payloads.derive_guided_execution_payloads(bad_auth, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "authorization_not_accepted"


# Test E
def test_refuses_when_parent_identities_mismatch(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract()

    # production intent identity mismatch
    bad_auth = _unchecked(auth_result, production_intent_identity="f" * 64)
    bad_auth = _unchecked(bad_auth, canonical_authorization_identity=authorization.compute_guided_run_authorization_identity(bad_auth))
    result = payloads.derive_guided_execution_payloads(bad_auth, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "production_intent_identity_mismatch"

    # candidate preflight identity mismatch
    bad_auth = _unchecked(auth_result, candidate_preflight_identity="f" * 64)
    bad_auth = _unchecked(bad_auth, canonical_authorization_identity=authorization.compute_guided_run_authorization_identity(bad_auth))
    result = payloads.derive_guided_execution_payloads(bad_auth, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "candidate_preflight_identity_mismatch"

    # ROI preflight identity mismatch
    bad_auth = _unchecked(auth_result, roi_preflight_identity="f" * 64)
    bad_auth = _unchecked(bad_auth, canonical_authorization_identity=authorization.compute_guided_run_authorization_identity(bad_auth))
    result = payloads.derive_guided_execution_payloads(bad_auth, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "roi_preflight_identity_mismatch"


# Test F
def test_config_field_dispositions_coverage():
    # every dataclasses.fields(Config) name must appear exactly once in GUIDED_CONFIG_FIELD_DISPOSITIONS
    config_fields = {f.name for f in fields(Config)}
    disposition_keys = set(payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS.keys())
    assert config_fields == disposition_keys

    # No unknown dispositions
    valid_dispositions = {
        payloads.CONFIG_DISPOSITION_INTENT,
        payloads.CONFIG_DISPOSITION_FIXED,
        payloads.CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
        payloads.CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
        payloads.CONFIG_DISPOSITION_PROHIBITED_FIRST_SUBSET,
        payloads.CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
        # 4J16k39a: feature-detection fields are sourced from the settings the
        # user confirmed in Guided Step 5, not from baked contract overrides.
        payloads.CONFIG_DISPOSITION_CONFIRMED_FEATURE,
        payloads.CONFIG_DISPOSITION_APPROVED_MISSING,
    }
    for disp in payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS.values():
        assert disp in valid_dispositions

    # The confirmed-feature disposition must cover exactly the shared
    # feature-detection field set, and none of those may be baked into the
    # contract's fixed overrides (that was the C1 defect).
    from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS

    confirmed = {
        k for k, v in payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS.items()
        if v == payloads.CONFIG_DISPOSITION_CONFIRMED_FEATURE
    }
    assert confirmed == set(FEATURE_EVENT_CONFIG_FIELDS)
    assert not (confirmed & set(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES))


def test_config_mapping_contract_completeness_refusal(auth_result):
    # missing override key refuses config_mapping_incomplete
    bad_overrides = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    del bad_overrides["chunk_duration_sec"]
    contract = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=bad_overrides)
    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "config_mapping_incomplete"

    # extra override key refuses config_mapping_incomplete
    bad_overrides = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    bad_overrides["extra_nonexistent_field"] = 42
    contract = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=bad_overrides)
    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "config_mapping_incomplete"

    # duplicate mapped intent field in overrides refuses config_mapping_incomplete
    bad_overrides = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    bad_overrides["exclude_incomplete_final_rwd_chunk"] = True
    contract = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=bad_overrides)
    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "config_mapping_incomplete"


def test_unsupported_or_prohibited_value_refusal(auth_result):
    # Test that modifying prohibited allow_partial_final_window refuses
    bad_overrides = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    bad_overrides["allow_partial_final_window"] = True
    contract = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=bad_overrides)
    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result.blocking_issues[0].category == "config_field_unsupported"

    # Test that modifying unsupported future npm_time_axis refuses
    bad_overrides2 = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    bad_overrides2["npm_time_axis"] = "custom_timestamp"
    contract2 = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=bad_overrides2)
    result2 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract2)
    assert result2.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED
    assert result2.blocking_issues[0].category == "config_field_unsupported"


def test_contract_immutability(auth_result):
    # 1. Existing scalar immutability test still passes
    overrides_dict = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    contract = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=overrides_dict)
    overrides_dict["chunk_duration_sec"] = 9999.0
    for item in contract.fixed_config_overrides:
        if item.name == "chunk_duration_sec":
            assert item.value == 600.0

    # 2. Passing a mutable list override, mutating original list does not change derivation/identity
    my_list = ["file1.csv", "file2.csv"]
    overrides_dict = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    overrides_dict["authorized_missing_sessions"] = my_list
    contract_list = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=overrides_dict)
    
    # Get initial identity/derivation
    res_list1 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract_list)
    assert res_list1.ok is True
    
    # Mutate original list
    my_list.append("file3.csv")
    
    # Derivation / identity remains exactly same
    res_list2 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract_list)
    assert res_list2.ok is True
    assert res_list2.config_payload_identity == res_list1.config_payload_identity

    # Verify that the value stored is immutable (tuple)
    for item in contract_list.fixed_config_overrides:
        if item.name == "authorized_missing_sessions":
            assert isinstance(item.value, tuple)
            assert item.value == ("file1.csv", "file2.csv")

    # 3. Passing a mutable dict override, mutating original dict does not change derivation/identity
    my_dict = {"key1": "val1"}
    overrides_dict = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    overrides_dict["rwd_contract_validation"] = my_dict
    contract_dict = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=overrides_dict)
    
    res_dict1 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract_dict)
    assert res_dict1.ok is True
    
    # Mutate original dict
    my_dict["key2"] = "val2"
    
    res_dict2 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract_dict)
    assert res_dict2.ok is True
    assert res_dict2.config_payload_identity == res_dict1.config_payload_identity

    # Verify stored value is immutable FrozenDict
    for item in contract_dict.fixed_config_overrides:
        if item.name == "rwd_contract_validation":
            assert isinstance(item.value, payloads.FrozenDict)
            assert item.value == (("key1", "val1"),)

    # 4. Default contract contains no mutable list/dict values inside fixed_config_overrides
    default_contract = payloads.build_guided_execution_startup_mapping_contract()
    for item in default_contract.fixed_config_overrides:
        val = item.value
        # Check no list/dict/set or other mutables are inside value
        assert not isinstance(val, (list, dict, set))


# Test G
def test_no_implicit_config_defaults(auth_result, monkeypatch):
    contract = payloads.build_guided_execution_startup_mapping_contract()
    
    # monkeypatch Config init and from_yaml to fail
    def _fail_init(*args, **kwargs):
        pytest.fail("Config was implicitly instantiated.")
        
    def _fail_from_yaml(*args, **kwargs):
        pytest.fail("Config.from_yaml was implicitly called.")

    monkeypatch.setattr(Config, "__init__", _fail_init)
    monkeypatch.setattr(Config, "from_yaml", _fail_from_yaml)

    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.ok is True


# Test H
def test_candidate_manifest_payload_binding(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract()
    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.ok is True

    manifest = result.candidate_manifest_payload
    intent = auth_result.production_intent
    cand_pre = auth_result.candidate_preflight_result
    roi_pre = auth_result.roi_preflight_result

    assert manifest.source_root_canonical == intent.input_source.source_root_canonical
    assert manifest.source_candidate_set_digest == cand_pre.actual_candidate_set_digest
    assert manifest.source_candidate_content_digest == cand_pre.actual_candidate_content_digest
    assert len(manifest.candidate_files) == len(cand_pre.actual_candidates)
    assert manifest.parser_contract_digest == roi_pre.parser_contract_digest
    assert manifest.discovered_roi_ids == roi_pre.actual_discovered_roi_ids
    assert manifest.included_roi_ids == roi_pre.actual_included_roi_ids
    assert manifest.excluded_roi_ids == roi_pre.actual_excluded_roi_ids
    assert manifest.strict_roi_inventory_digest == roi_pre.actual_strict_roi_inventory_digest
    assert manifest.candidate_preflight_identity == auth_result.candidate_preflight_identity
    assert manifest.roi_preflight_identity == auth_result.roi_preflight_identity


# Test I
def test_payload_identity_determinism_and_sensitivity(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract()
    
    res1 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    res2 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    
    assert res1.config_payload_identity == res2.config_payload_identity
    assert res1.candidate_manifest_payload_identity == res2.candidate_manifest_payload_identity
    assert res1.provenance_seed_identity == res2.provenance_seed_identity

    # Change sessions_per_hour -- and, as real production would, the
    # normalized recording description identity along with it (it is a
    # sampling-contract fact the description covers).
    intent_changed = replace(
        auth_result.production_intent,
        acquisition=replace(auth_result.production_intent.acquisition, sessions_per_hour=20)
    )
    intent_changed = replace(
        intent_changed,
        normalized_recording_description_identity=(
            _rebuild_normalized_recording_identity(intent_changed)
        ),
    )
    auth_changed = replace(
        auth_result,
        production_intent=intent_changed,
        production_intent_identity=mapping.compute_guided_production_execution_intent_identity(intent_changed)
    )
    # We must also recompute authorization identity so the gate passes
    auth_changed = replace(
        auth_changed,
        canonical_authorization_identity=authorization.compute_guided_run_authorization_identity(auth_changed)
    )
    res3 = payloads.derive_guided_execution_payloads(auth_changed, startup_mapping_contract=contract)
    assert res3.ok is True
    assert res3.provenance_seed_identity != res1.provenance_seed_identity

    # Change dynamic_fit_mode
    intent_changed2 = replace(
        auth_result.production_intent,
        correction=replace(auth_result.production_intent.correction, global_dynamic_fit_mode="different_mode")
    )
    auth_changed2 = replace(
        auth_result,
        production_intent=intent_changed2,
        production_intent_identity=mapping.compute_guided_production_execution_intent_identity(intent_changed2)
    )
    auth_changed2 = replace(
        auth_changed2,
        canonical_authorization_identity=authorization.compute_guided_run_authorization_identity(auth_changed2)
    )
    res4 = payloads.derive_guided_execution_payloads(auth_changed2, startup_mapping_contract=contract)
    assert res4.ok is True
    assert res4.config_payload_identity != res1.config_payload_identity

    # Change candidate file digest
    cand_pre = auth_result.candidate_preflight_result
    bad_cand = replace(
        cand_pre.actual_candidates[0],
        sha256_content_digest="f" * 64
    )
    new_cand_pre = replace(
        cand_pre,
        actual_candidates=(bad_cand,),
        actual_candidate_content_digest="f" * 64,
        actual_candidate_set_digest="f" * 64
    )
    new_cand_pre = replace(
        new_cand_pre,
        canonical_preflight_identity=preflight.compute_guided_candidate_preflight_identity(new_cand_pre)
    )
    auth_changed3 = replace(
        auth_result,
        candidate_preflight_result=new_cand_pre,
        candidate_preflight_identity=new_cand_pre.canonical_preflight_identity
    )
    auth_changed3 = replace(
        auth_changed3,
        canonical_authorization_identity=authorization.compute_guided_run_authorization_identity(auth_changed3)
    )
    res5 = payloads.derive_guided_execution_payloads(auth_changed3, startup_mapping_contract=contract)
    assert res5.ok is True
    assert res5.candidate_manifest_payload_identity != res1.candidate_manifest_payload_identity

    # Change included ROI tuple
    roi_pre = auth_result.roi_preflight_result
    new_roi_pre = replace(
        roi_pre,
        actual_included_roi_ids=("ROI1", "ROI2")
    )
    new_roi_pre = replace(
        new_roi_pre,
        canonical_preflight_identity=preflight.compute_guided_roi_preflight_identity(new_roi_pre)
    )
    auth_changed4 = replace(
        auth_result,
        roi_preflight_result=new_roi_pre,
        roi_preflight_identity=new_roi_pre.canonical_preflight_identity
    )
    auth_changed4 = replace(
        auth_changed4,
        canonical_authorization_identity=authorization.compute_guided_run_authorization_identity(auth_changed4)
    )
    res6 = payloads.derive_guided_execution_payloads(auth_changed4, startup_mapping_contract=contract)
    assert res6.ok is True
    assert res6.candidate_manifest_payload_identity != res1.candidate_manifest_payload_identity

    # Change startup contract version
    contract_changed = replace(contract, contract_version="different.v2")
    res7 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract_changed)
    assert res7.ok is True
    assert res7.provenance_seed_identity != res1.provenance_seed_identity

    # Change fixed config override value
    overrides_changed = dict(payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    overrides_changed["chunk_duration_sec"] = 1200.0
    contract_changed2 = payloads.build_guided_execution_startup_mapping_contract(fixed_config_overrides=overrides_changed)
    res8 = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract_changed2)
    assert res8.ok is True
    assert res8.config_payload_identity != res1.config_payload_identity


# Test J
def test_provenance_seed_excludes_allocation_and_runtime_fields():
    seed_fields = {f.name for f in fields(payloads.GuidedStartupProvenanceSeed)}
    forbidden_substrings = ("run_id", "run_dir", "allocated", "config_path", "manifest_path", "command_path", "timestamp", "process", "pid", "argv")
    for f in seed_fields:
        for forbidden in forbidden_substrings:
            assert forbidden not in f


# Test K: under Option A, runner request is not returned
def test_runner_request_is_none(auth_result):
    contract = payloads.build_guided_execution_startup_mapping_contract(
        exact_candidate_manifest_consumption_capable=True,
        exact_roi_consumption_capable=True,
    )
    res = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert res.ok is True
    assert res.runner_request is None


# Test L: Strengthened no-side-effects
def test_no_side_effects_during_derivation(auth_result, monkeypatch):
    contract = payloads.build_guided_execution_startup_mapping_contract()

    def _fail(*args, **kwargs):
        pytest.fail("Filesystem write/execution attempted!")

    def _fail_open(file, mode="r", *args, **kwargs):
        if any(ch in mode for ch in ("w", "a", "x", "+")):
            pytest.fail(f"Filesystem write/open attempted with mode {mode}!")
        return builtins.open(file, mode, *args, **kwargs)

    def _fail_mkdir(*args, **kwargs):
        pytest.fail("Directory creation attempted!")

    def _fail_tempfile(*args, **kwargs):
        pytest.fail("Tempfile creation attempted!")

    monkeypatch.setattr(Path, "write_text", _fail)
    monkeypatch.setattr(Path, "write_bytes", _fail)
    monkeypatch.setattr(Path, "mkdir", _fail_mkdir)
    monkeypatch.setattr(Path, "touch", _fail)
    monkeypatch.setattr(os, "makedirs", _fail)
    monkeypatch.setattr(os, "mkdir", _fail_mkdir)
    monkeypatch.setattr(builtins, "open", _fail_open)
    
    import tempfile
    monkeypatch.setattr(tempfile, "mktemp", _fail_tempfile)
    monkeypatch.setattr(tempfile, "mkstemp", _fail_tempfile)
    monkeypatch.setattr(tempfile, "mkdtemp", _fail_tempfile)
    monkeypatch.setattr(tempfile, "TemporaryFile", _fail_tempfile)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _fail_tempfile)
    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", _fail_tempfile)

    # Mock subprocess.run
    import subprocess
    monkeypatch.setattr(subprocess, "run", _fail)

    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.ok is True


# Test M
def test_no_preflight_rerun_during_derivation(auth_result, monkeypatch):
    contract = payloads.build_guided_execution_startup_mapping_contract()

    def _fail(*args, **kwargs):
        pytest.fail("Preflight run or file inspection function was called!")

    monkeypatch.setattr(preflight, "run_candidate_manifest_execution_preflight", _fail)
    monkeypatch.setattr(preflight, "run_roi_execution_preflight", _fail)

    result = payloads.derive_guided_execution_payloads(auth_result, startup_mapping_contract=contract)
    assert result.ok is True


# Test N: Strengthened import boundary check
def test_import_boundary():
    # AST check to confirm no forbidden imports in guided_execution_payloads.py
    file_path = Path(__file__).parent.parent / "photometry_pipeline" / "guided_execution_payloads.py"
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    forbidden_packages = (
        "gui",
        "RunSpec",
        "gui.run_spec",
        "runner",
        "process_runner",
        "output_allocator",
        "config_writer",
        "artifact_writer",
        "status_writer",
        "report_writer",
        "manifest_writer",
        "completed_run_loader",
        "subprocess",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                parts = name.name.split('.')
                for forbidden in forbidden_packages:
                    assert forbidden not in parts
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                parts = node.module.split('.')
                for forbidden in forbidden_packages:
                    assert forbidden not in parts


# Test O: Strengthened identity field coverage maps
def test_identity_field_coverage():
    # ConfigPayload
    config_map = {
        "config_schema_name": "included",
        "config_mapping_contract_version": "included",
        "values": "included",
        "canonical_config_payload_identity": "excluded",
    }
    config_fields = {f.name for f in fields(payloads.GuidedExecutionConfigPayload)}
    assert config_fields == set(config_map.keys())

    # ManifestPayload
    manifest_map = {
        "manifest_schema_name": "included",
        "manifest_schema_version": "included",
        "candidate_consumption_contract_version": "included",
        "source_root_canonical": "included",
        "source_candidate_set_digest": "included",
        "source_candidate_content_digest": "included",
        "candidate_files": "included",
        "parser_contract_digest": "included",
        "discovered_roi_ids": "included",
        "included_roi_ids": "included",
        "excluded_roi_ids": "included",
        "strict_roi_inventory_digest": "included",
        "candidate_preflight_identity": "included",
        "roi_preflight_identity": "included",
        "canonical_candidate_manifest_payload_identity": "excluded",
    }
    manifest_fields = {f.name for f in fields(payloads.GuidedRunnerCandidateManifestPayload)}
    assert manifest_fields == set(manifest_map.keys())

    # Request
    request_map = {
        "runner_request_schema_name": "included",
        "runner_request_schema_version": "included",
        "runner_contract_version": "included",
        "runner_entrypoint": "included",
        "input_source_root": "included",
        "output_base_canonical": "included",
        "run_type": "included",
        "mode": "included",
        "input_format": "included",
        "include_rois": "included",
        "traces_only": "included",
        "overwrite": "included",
        "config_filename": "included",
        "candidate_manifest_filename": "included",
        "required_future_argv_flags": "included",
        "prohibited_argv_flags": "included",
        "canonical_runner_request_identity": "excluded",
    }
    request_fields = {f.name for f in fields(payloads.GuidedRunnerExecutionRequest)}
    assert request_fields == set(request_map.keys())

    # Seed
    seed_map = {
        "provenance_schema_name": "included",
        "provenance_schema_version": "included",
        "startup_mapping_contract_version": "included",
        "validation_request_identity": "included",
        "authorization_identity": "included",
        "production_intent_identity": "included",
        "application_build_identity": "included",
        "production_mapping_contract_version": "included",
        "runner_contract_version": "included",
        "candidate_preflight_identity": "included",
        "roi_preflight_identity": "included",
        "config_payload_identity": "included",
        "candidate_manifest_payload_identity": "included",
        "runner_request_identity": "included",
        "runnable": "included",
        "canonical_provenance_seed_identity": "excluded",
    }
    seed_fields = {f.name for f in fields(payloads.GuidedStartupProvenanceSeed)}
    assert seed_fields == set(seed_map.keys())


# ---------------------------------------------------------------------------
# Feature Detection "loaded Defaults, no Apply" contract repair (B2 Phase 2)
#
# resolve_confirmed_feature_config_fields previously required
# `explicitly_applied is True`, which refused a valid loaded Default profile
# (`profile_status == "default_initialized"`) with
# "Feature-detection settings were never confirmed." -- the exact real
# interactive failure. It now reuses the shared
# is_saved_feature_event_profile_current predicate so the contract matches
# GUI readiness, the draft plan, and backend validation.
# ---------------------------------------------------------------------------

from types import SimpleNamespace as _NS
from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS
from tests.test_guided_backend_validator import CONFIRMED_FEATURE_PROFILE_VALUES


def _feature_event(*, profile_status, explicitly_applied, current=True, values=None):
    field_values = dict(CONFIRMED_FEATURE_PROFILE_VALUES)
    if values:
        field_values.update(values)
    return _NS(
        profile_status=profile_status,
        explicitly_applied=explicitly_applied,
        current=current,
        effective_values=tuple(
            _NS(field_name=name, value=value)
            for name, value in field_values.items()
        ),
    )


def _intent_with(feature_event):
    return _NS(feature_event=feature_event)


def test_loaded_default_profile_without_apply_is_accepted():
    # Case A: default_initialized + explicitly_applied=False is executable.
    # This is the exact old-code failure: the previous predicate returned
    # (None, "Feature-detection settings were never confirmed.") here.
    intent = _intent_with(
        _feature_event(
            profile_status="default_initialized", explicitly_applied=False
        )
    )
    result_fields, reason = payloads.resolve_confirmed_feature_config_fields(intent)
    assert reason == ""
    assert result_fields is not None
    assert result_fields["peak_threshold_method"] == "mean_std"
    assert set(result_fields) == set(FEATURE_EVENT_CONFIG_FIELDS)


def test_old_predicate_would_have_refused_loaded_default(monkeypatch):
    # Pin the fix: with the obsolete explicitly-applied gate restored, the
    # same loaded-Default profile is refused -- proving this exercises the
    # exact repaired predicate.
    import photometry_pipeline.guided_backend_validation_request as vr

    monkeypatch.setattr(
        vr,
        "is_saved_feature_event_profile_current",
        lambda status, applied: bool(status == "applied" and applied is True),
    )
    intent = _intent_with(
        _feature_event(
            profile_status="default_initialized", explicitly_applied=False
        )
    )
    result_fields, reason = payloads.resolve_confirmed_feature_config_fields(intent)
    assert result_fields is None
    assert "not ready" in reason.lower()


def test_applied_default_profile_is_accepted():
    # Case C: applied + explicitly_applied=True still works.
    intent = _intent_with(
        _feature_event(profile_status="applied", explicitly_applied=True)
    )
    result_fields, reason = payloads.resolve_confirmed_feature_config_fields(intent)
    assert reason == ""
    assert result_fields is not None


def test_unknown_or_unapplied_edit_profile_is_refused():
    # An `applied` profile that was never explicitly applied, an empty status,
    # and an unknown status are all refused (edits are not silently consumed).
    for status, applied in (("applied", False), ("", False), ("initialized", True)):
        intent = _intent_with(
            _feature_event(profile_status=status, explicitly_applied=applied)
        )
        result_fields, reason = payloads.resolve_confirmed_feature_config_fields(
            intent
        )
        assert result_fields is None
        assert "not ready" in reason.lower()


def test_stale_saved_profile_is_refused():
    intent = _intent_with(
        _feature_event(
            profile_status="default_initialized",
            explicitly_applied=False,
            current=False,
        )
    )
    result_fields, reason = payloads.resolve_confirmed_feature_config_fields(intent)
    assert result_fields is None
    assert "stale" in reason.lower()


def test_invalid_saved_default_is_refused_with_specific_reason():
    # Case D: a genuinely invalid saved Default still refuses, with a specific
    # Feature Detection reason (not the generic gate).
    intent = _intent_with(
        _feature_event(
            profile_status="default_initialized",
            explicitly_applied=False,
            values={"peak_threshold_k": -5.0},
        )
    )
    result_fields, reason = payloads.resolve_confirmed_feature_config_fields(intent)
    assert result_fields is None
    assert "invalid" in reason.lower()


def test_incomplete_saved_default_is_refused():
    intent = _intent_with(
        _feature_event(
            profile_status="default_initialized", explicitly_applied=False
        )
    )
    intent.feature_event.effective_values = tuple(
        item
        for item in intent.feature_event.effective_values
        if item.field_name != "peak_min_distance_sec"
    )
    result_fields, reason = payloads.resolve_confirmed_feature_config_fields(intent)
    assert result_fields is None
    assert "incomplete" in reason.lower()


def test_inactive_threshold_fields_do_not_block_loaded_default():
    # Case F: under mean_std, percentile/abs threshold fields are inactive and
    # must not block even if they carry dormant values.
    intent = _intent_with(
        _feature_event(
            profile_status="default_initialized",
            explicitly_applied=False,
            values={
                "peak_threshold_method": "mean_std",
                "peak_threshold_percentile": 999.0,
                "peak_threshold_abs": -123.0,
            },
        )
    )
    result_fields, reason = payloads.resolve_confirmed_feature_config_fields(intent)
    assert reason == ""
    assert result_fields is not None
    assert result_fields["peak_threshold_percentile"] == 999.0


def test_end_to_end_payload_accepts_default_initialized_intent(auth_result):
    # Full payload derivation with a default_initialized (no-Apply) intent
    # succeeds and carries the saved feature values into the config payload.
    intent = auth_result.production_intent
    loaded_feature_event = replace(
        intent.feature_event,
        profile_status="default_initialized",
        explicitly_applied=False,
    )
    loaded_intent = replace(intent, feature_event=loaded_feature_event)
    loaded_intent = replace(
        loaded_intent,
        canonical_intent_identity=(
            mapping.compute_guided_production_execution_intent_identity(
                loaded_intent
            )
        ),
    )
    loaded_auth = _unchecked(
        auth_result,
        production_intent=loaded_intent,
        production_intent_identity=loaded_intent.canonical_intent_identity,
    )
    loaded_auth = _unchecked(
        loaded_auth,
        canonical_authorization_identity=(
            authorization.compute_guided_run_authorization_identity(loaded_auth)
        ),
    )
    contract = payloads.build_guided_execution_startup_mapping_contract()
    result = payloads.derive_guided_execution_payloads(
        loaded_auth, startup_mapping_contract=contract
    )
    assert result.ok is True
    assert result.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE
    assert result.config_payload is not None
    assert result.provenance_seed is not None
    assert result.candidate_manifest_payload is not None
    config_names = {v.name for v in result.config_payload.values}
    assert "peak_threshold_method" in config_names
