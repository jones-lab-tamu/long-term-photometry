from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_backend_validation_request as contracts
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _validator_contract() -> contracts.GuidedBackendValidatorContract:
    return contracts.GuidedBackendValidatorContract(
        validation_scope=contracts.GUIDED_BACKEND_VALIDATION_SCOPE,
        validation_contract_version=(
            contracts.GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
        ),
        validator_capability_version="guided_backend_validator.test_fixture.v1",
        supported_subset_rule_version=(
            contracts.GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
        ),
    )


def _typed_value(
    name: str,
    value: str | bool | int | float,
) -> contracts.GuidedBackendTypedFieldValue:
    return contracts.GuidedBackendTypedFieldValue(
        field_name=name,
        value_type=type(value).__name__,
        value=value,
    )


def _request() -> contracts.GuidedBackendValidationRequest:
    candidate = contracts.GuidedBackendSourceCandidateFile(
        canonical_relative_path="session1/fluorescence.csv",
        size_bytes=42,
        sha256_content_digest=_DIGEST_A,
    )
    source = contracts.GuidedBackendSourceRequest(
        source_root_canonical=r"c:\source",
        source_root_path_style="windows_drive",
        source_format="rwd",
        snapshot_schema_name="guided_rwd_source_candidate_snapshot",
        snapshot_schema_version="v1",
        discovery_rule_version="immediate_child_exact_fluorescence_csv.v1",
        path_canonicalization_version="typed_json_utf8.v1",
        relative_path_rule_version="canonical_forward_slash_relative_path.v1",
        ignored_files_policy="ignore_non_target_entries.v1",
        build_mode="read_only",
        source_candidate_set_digest=_DIGEST_A,
        source_candidate_content_digest=_DIGEST_B,
        candidate_files=(candidate,),
    )
    acquisition = contracts.GuidedBackendAcquisitionDatasetRequest(
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
        allow_partial_final_window=False,
        exclude_incomplete_final_rwd_chunk=False,
        classification_schema_name=(
            "guided_rwd_incomplete_final_chunk_classification"
        ),
        classification_schema_version="v1",
        classifier_version="not_requested_only.v1",
        classification_status="not_requested",
        not_requested_classification_digest=_DIGEST_C,
        dataset_snapshot_schema_version=(
            "guided_new_analysis_dataset_contract_snapshot.v1"
        ),
        dataset_status="applied",
        dataset_current_applied=True,
        rwd_time_col="Time(s)",
        uv_suffix="-410",
        sig_suffix="-470",
        semantic_values=(_typed_value("rwd_time_col", "Time(s)"),),
        dataset_source_setup_signature="source-signature",
        diagnostic_cache_contract_identity="cache-contract",
    )
    parser = contracts.GuidedBackendRwdParserRequest(
        schema_name="rwd_header_parsing_contract",
        schema_version="v1",
        header_search_line_limit=60,
        time_column_candidates=("Time(s)",),
        uv_suffix_candidates=("-410",),
        signal_suffix_candidates=("-470",),
        column_normalization_rule="strip_whitespace_and_bom.v1",
        roi_name_rule="exact_case_sensitive_reject_casefold_collisions.v1",
        ambiguity_policy="reject_all.v1",
        parser_contract_digest=_DIGEST_A,
    )
    roi_scope = contracts.GuidedBackendRoiScopeRequest(
        discovered_roi_ids=("ROI0", "ROI1"),
        included_roi_ids=("ROI0",),
        excluded_roi_ids=("ROI1",),
        inventory_source_content_digest=_DIGEST_B,
    )
    mark = contracts.GuidedBackendConfirmedStrategyMark(
        roi_id="ROI0",
        selected_dynamic_fit_mode="global_linear_regression",
        diagnostic_cache_id="cache-001",
        source_setup_signature="source-signature",
        diagnostic_scope_signature="scope-signature",
        build_request_signature="request-signature",
        evidence_reference_id="evidence-001",
        evidence_chunk=0,
    )
    correction = contracts.GuidedBackendCorrectionRequest(
        strategy_scope="global",
        global_correction_strategy="dynamic_fit",
        global_dynamic_fit_mode="global_linear_regression",
        dynamic_fit_parameter_values=(
            _typed_value("dynamic_fit_mode", "global_linear_regression"),
        ),
        confirmed_marks=(mark,),
        mark_rule_version="explicit_confirmed_mark.v1",
        currentness_rule_version="cache_bound_currentness.v1",
        unanimity_rule_version="included_roi_unanimous_dynamic_fit.v1",
    )
    evidence = contracts.GuidedBackendEvidenceReference(
        evidence_reference_id="evidence-001",
        evidence_kind="correction_preview",
        diagnostic_cache_id="cache-001",
        source_setup_signature="source-signature",
    )
    diagnostic = contracts.GuidedBackendDiagnosticEvidenceRequest(
        cache_id="cache-001",
        cache_root_canonical=r"c:\output\_guided_diagnostic_cache\cache-001",
        source_setup_signature="source-signature",
        diagnostic_scope_signature="scope-signature",
        build_request_signature="request-signature",
        artifact_contract_version="guided_diagnostic_cache.v1",
        provenance_schema_version="guided_diagnostic_cache.v1",
        artifact_semantic_digest=_DIGEST_A,
        provenance_semantic_digest=_DIGEST_B,
        evidence_references=(evidence,),
        completed_run_rejection_category="guided_diagnostic_cache_ineligible",
        resolver_status="current",
        preliminary_cache=True,
        production_analysis=False,
    )
    feature_event = contracts.GuidedBackendFeatureEventRequest(
        profile_schema_version="guided_feature_event_profile.v1",
        profile_id="profile-001",
        effective_values=(_typed_value("event_signal", "dff"),),
        active_fields=("event_signal",),
        inactive_fields=(),
        profile_status="applied",
        explicitly_applied=True,
        current=True,
        visible_unapplied_changes=False,
    )
    output = contracts.GuidedBackendOutputRequest(
        output_base_canonical=r"c:\output",
        output_base_path_style="windows_drive",
        path_role="output_base",
        future_output_owner="runner",
        run_directory_strategy="derive_unique_run_id_under_output_base",
        creation_timing="future_execution_start_only",
        overwrite=False,
        precreate=False,
        policy_status="applied",
        policy_current=True,
        safety_classifier_version="guided_output_safety.v1",
        relationships=(
            contracts.GuidedBackendOutputRelationship(
                relationship="output_base_inside_source",
                root_kind="source",
                status="safe",
            ),
        ),
        protected_root_context_complete=True,
        blocker_categories=(),
        filesystem_fact_scope="read_only_supplied_facts",
    )
    local_contract = contracts.GuidedBackendLocalContractState(
        local_check_contract_version="guided_backend_local_checks.v1",
        blocking_issue_categories=(),
        warning_categories=(),
        unsupported_state_flags=(),
        unresolved_required_inputs=(),
        deferred_capabilities=(
            "app_build_identity",
            "full_source_manifest_identity",
            "run_authorization",
            "strict_roi_inventory_identity",
        ),
    )
    return contracts.GuidedBackendValidationRequest(
        request_schema_name=(
            contracts.GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME
        ),
        request_schema_version=(
            contracts.GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION
        ),
        validation_scope=contracts.GUIDED_BACKEND_VALIDATION_SCOPE,
        validation_contract_version=(
            contracts.GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
        ),
        validator_capability_version="guided_backend_validator.test_fixture.v1",
        compiler_version=contracts.GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
        subset_rule_version=(
            contracts.GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
        ),
        canonicalization_algorithm_version="typed_json_utf8.v1",
        source=source,
        acquisition_dataset=acquisition,
        parser=parser,
        roi_scope=roi_scope,
        correction=correction,
        diagnostic_evidence=diagnostic,
        feature_event=feature_event,
        output=output,
        local_contract=local_contract,
    )


def test_module_has_no_prohibited_imports():
    source = Path(contracts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    prohibited = {
        "PySide6",
        "gui",
        "subprocess",
        "pandas",
        "h5py",
        "yaml",
        "runner",
        "photometry_pipeline.runner",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name == root or name.startswith(f"{root}.")
        for name in imported
        for root in prohibited
    )


def test_constants_are_stable_and_non_empty():
    assert contracts.GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME == (
        "guided_backend_validation_request"
    )
    assert contracts.GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION == "v1"
    assert contracts.GUIDED_BACKEND_VALIDATION_SCOPE == (
        "guided_rwd_intermittent_phasic_full_validate"
    )
    assert contracts.GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION == (
        "guided_backend_validation_contract.v1"
    )
    assert contracts.GUIDED_BACKEND_VALIDATION_COMPILER_VERSION == (
        "guided_backend_validation_request_compiler.v1"
    )
    assert contracts.GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION == (
        "global_dynamic_fit_only.v1"
    )
    assert contracts.GUIDED_BACKEND_VALIDATION_IDENTITY_DOMAIN == (
        "guided-backend-validation-request:v1"
    )


def test_refusal_taxonomy_is_complete_and_duplicate_free():
    required = {
        "missing_source",
        "unsupported_source_format",
        "unsupported_acquisition_mode",
        "source_snapshot_unavailable",
        "source_snapshot_stale",
        "source_snapshot_digest_mismatch",
        "unsupported_incomplete_final_exclusion",
        "incomplete_final_classification_mismatch",
        "missing_or_stale_dataset_contract",
        "dataset_source_binding_mismatch",
        "invalid_sessions_per_hour",
        "invalid_session_duration",
        "missing_roi_inventory",
        "empty_included_roi_set",
        "roi_selection_stale",
        "included_roi_not_discovered",
        "included_excluded_roi_conflict",
        "duplicate_roi_id",
        "missing_confirmed_strategy_mark",
        "duplicate_confirmed_strategy_mark",
        "stale_strategy_mark",
        "non_explicit_strategy_mark",
        "mixed_dynamic_fit_modes",
        "signal_only_not_supported_for_validate",
        "forbidden_strategy_state",
        "global_intent_confirmed_marks_mismatch",
        "dynamic_fit_parameter_contract_mismatch",
        "unresolved_dynamic_fit_parameter",
        "missing_or_stale_diagnostic_cache",
        "diagnostic_cache_not_completed_run_ineligible",
        "diagnostic_cache_identity_mismatch",
        "evidence_reference_missing_or_stale",
        "missing_feature_event_profile",
        "invalid_feature_event_profile",
        "stale_feature_event_profile",
        "unapplied_feature_event_changes",
        "unresolved_feature_event_effective_value",
        "missing_output_policy",
        "stale_output_policy",
        "unapplied_output_policy_changes",
        "unsafe_output_base",
        "overwrite_not_allowed",
        "output_overlaps_source",
        "output_overlaps_completed_run",
        "output_overlaps_diagnostic_cache",
        "protected_root_context_incomplete",
        "parser_contract_missing",
        "parser_digest_unavailable",
        "parser_digest_mismatch",
        "unresolved_required_identity_input",
        "unsupported_first_subset_state",
        "compiler_contract_unavailable",
        "validator_contract_unavailable",
    }
    categories = contracts.GUIDED_BACKEND_VALIDATION_REFUSAL_CATEGORIES
    assert set(categories) == required
    assert len(categories) == len(set(categories))


def test_request_and_nested_models_are_frozen():
    request = _request()
    assert is_dataclass(request)
    with pytest.raises(FrozenInstanceError):
        request.validation_scope = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        request.roi_scope.included_roi_ids = ("ROI1",)  # type: ignore[misc]


def test_tuple_contracts_reject_mutable_lists():
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="must be a tuple",
    ):
        contracts.GuidedBackendRoiScopeRequest(
            discovered_roi_ids=["ROI0"],  # type: ignore[arg-type]
            included_roi_ids=("ROI0",),
            excluded_roi_ids=(),
            inventory_source_content_digest=_DIGEST_A,
        )
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="must be a tuple",
    ):
        contracts.GuidedBackendValidationMaterializedFacts(
            unresolved_required_inputs=[]  # type: ignore[arg-type]
        )


def test_prohibited_fields_are_absent_from_request_contract():
    request_field_names = {item.name for item in fields(_request())}
    assert request_field_names.isdisjoint(
        contracts.PROHIBITED_REQUEST_FIELD_NAMES
    )


def test_complete_synthetic_request_enforces_roi_output_and_strategy_invariants():
    request = _request()
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="disjoint",
    ):
        replace(
            request.roi_scope,
            excluded_roi_ids=("ROI0", "ROI1"),
        )
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="Overwrite",
    ):
        replace(request.output, overwrite=True)
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="path_role",
    ):
        replace(request.output, path_role="run_dir")
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="strategy_scope",
    ):
        replace(request.correction, strategy_scope="per_roi")
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="global_correction_strategy",
    ):
        replace(
            request.correction,
            global_correction_strategy="signal_only_f0",
        )


def test_compile_failure_shape_has_no_request_or_identity():
    failure = contracts.GuidedBackendValidationCompileFailure(
        blocking_issues=(
            contracts.GuidedBackendValidationCompileIssue(
                category="compiler_contract_unavailable",
                section="compiler",
                message="Compiler is unavailable.",
            ),
        )
    )
    assert failure.status == "refused"
    assert failure.no_partial_request is True
    assert failure.no_request_identity is True
    assert not hasattr(failure, "request")
    assert not hasattr(failure, "canonical_request_identity")


def test_none_draft_refuses_cleanly():
    result = contracts.compile_guided_backend_validation_request(
        None,
        facts=contracts.GuidedBackendValidationMaterializedFacts(),
        validator_contract=_validator_contract(),
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    issue = result.blocking_issues[0]
    assert issue.category == "compiler_contract_unavailable"
    assert issue.section == "compiler"
    assert issue.message


def test_none_facts_refuses_cleanly():
    result = contracts.compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(),
        facts=None,
        validator_contract=_validator_contract(),
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == (
        "unresolved_required_identity_input"
    )


@pytest.mark.parametrize("validator_contract", [None, object()])
def test_missing_or_invalid_validator_contract_refuses(validator_contract: object):
    result = contracts.compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(),
        facts=contracts.GuidedBackendValidationMaterializedFacts(),
        validator_contract=validator_contract,  # type: ignore[arg-type]
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == "validator_contract_unavailable"


def test_corrupted_validator_contract_refuses():
    invalid = object.__new__(contracts.GuidedBackendValidatorContract)
    object.__setattr__(invalid, "validation_scope", "")
    object.__setattr__(invalid, "validation_contract_version", "")
    object.__setattr__(invalid, "validator_capability_version", "unknown")
    object.__setattr__(invalid, "supported_subset_rule_version", "")

    result = contracts.compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(),
        facts=contracts.GuidedBackendValidationMaterializedFacts(),
        validator_contract=invalid,
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == "validator_contract_unavailable"


def test_stage_one_compiler_intentionally_never_returns_success():
    result = contracts.compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(),
        facts=contracts.GuidedBackendValidationMaterializedFacts(
            complete_for_compilation=True
        ),
        validator_contract=_validator_contract(),
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == "compiler_contract_unavailable"
    assert result.blocking_issues[0].detail_code == "stage_1_refusal_only"


def test_compiler_performs_no_filesystem_io(monkeypatch: pytest.MonkeyPatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem I/O is prohibited")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(os, "stat", fail)
    monkeypatch.setattr(os, "scandir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    result = contracts.compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(),
        facts=contracts.GuidedBackendValidationMaterializedFacts(),
        validator_contract=_validator_contract(),
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)


def test_compiler_creates_no_files_or_directories(tmp_path: Path):
    before = tuple(tmp_path.iterdir())
    contracts.compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(),
        facts=contracts.GuidedBackendValidationMaterializedFacts(),
        validator_contract=_validator_contract(),
    )
    assert tuple(tmp_path.iterdir()) == before


def test_refusal_message_does_not_expose_object_repr():
    result = contracts.compile_guided_backend_validation_request(
        object(),  # type: ignore[arg-type]
        facts=contracts.GuidedBackendValidationMaterializedFacts(),
        validator_contract=_validator_contract(),
    )
    message = result.blocking_issues[0].message
    assert "object at 0x" not in message
    assert "Traceback" not in message


def test_identity_is_explicitly_deferred():
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="deferred",
    ):
        contracts.compute_guided_backend_validation_request_identity(_request())
