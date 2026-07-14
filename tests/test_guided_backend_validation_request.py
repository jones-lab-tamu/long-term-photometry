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
_DIGEST_D = "d" * 64


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
        ignored_files_policy=contracts.GUIDED_BACKEND_SOURCE_IGNORED_FILES_POLICY,
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
        diagnostic_cache_contract_identity="request-signature",
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
        diagnostic_scope_signature="scope-signature",
        build_request_signature="request-signature",
        evidence_chunk=0,
        roi_id="ROI0",
        selected_dynamic_fit_mode="global_linear_regression",
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
        filesystem_fact_scope="read_only_path_relationships_no_writability_probe",
    )
    local_contract = contracts.GuidedBackendLocalContractState(
        local_check_contract_version="guided_backend_local_checks.v1",
        blocking_issue_categories=(),
        warning_categories=(),
        unsupported_state_flags=(),
        unresolved_required_inputs=(),
        deferred_capabilities=(
            "backend_validation",
            "run_authorization",
            "app_build_identity",
            "full_source_manifest_identity",
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
        normalized_recording_description_identity=_DIGEST_D,
    )


def _complete_facts() -> contracts.GuidedBackendValidationMaterializedFacts:
    request = _request()
    source = request.source
    acquisition = request.acquisition_dataset
    parser = request.parser
    roi = request.roi_scope
    correction = request.correction
    diagnostic = request.diagnostic_evidence
    feature = request.feature_event
    output = request.output
    return contracts.GuidedBackendValidationMaterializedFacts(
        source_snapshot=contracts.GuidedBackendSourceSnapshotFacts(
            available=True,
            source_root_canonical=source.source_root_canonical,
            source_root_path_style=source.source_root_path_style,
            source_candidate_set_digest=source.source_candidate_set_digest,
            source_candidate_content_digest=source.source_candidate_content_digest,
            candidate_files=source.candidate_files,
            stale=False,
        ),
        incomplete_final_classification=(
            contracts.GuidedBackendIncompleteFinalClassificationFacts(
                available=True,
                classification_status=acquisition.classification_status,
                classification_digest=acquisition.not_requested_classification_digest,
                source_candidate_set_digest=source.source_candidate_set_digest,
                source_candidate_content_digest=source.source_candidate_content_digest,
            )
        ),
        parser=contracts.GuidedBackendParserFacts(
            available=True,
            schema_name=parser.schema_name,
            schema_version=parser.schema_version,
            header_search_line_limit=parser.header_search_line_limit,
            time_column_candidates=parser.time_column_candidates,
            uv_suffix_candidates=parser.uv_suffix_candidates,
            signal_suffix_candidates=parser.signal_suffix_candidates,
            column_normalization_rule=parser.column_normalization_rule,
            roi_name_rule=parser.roi_name_rule,
            ambiguity_policy=parser.ambiguity_policy,
            parser_contract_digest=parser.parser_contract_digest,
            unresolved_inputs=(),
        ),
        acquisition_dataset=contracts.GuidedBackendAcquisitionDatasetFacts(
            available=True,
            acquisition_mode=acquisition.acquisition_mode,
            sessions_per_hour=acquisition.sessions_per_hour,
            session_duration_sec=acquisition.session_duration_sec,
            timeline_anchor_mode=acquisition.timeline_anchor_mode,
            fixed_daily_anchor_clock=acquisition.fixed_daily_anchor_clock,
            allow_partial_final_window=acquisition.allow_partial_final_window,
            exclude_incomplete_final_rwd_chunk=(
                acquisition.exclude_incomplete_final_rwd_chunk
            ),
            dataset_snapshot_schema_version=(
                acquisition.dataset_snapshot_schema_version
            ),
            dataset_status=acquisition.dataset_status,
            dataset_current_applied=acquisition.dataset_current_applied,
            rwd_time_col=acquisition.rwd_time_col,
            uv_suffix=acquisition.uv_suffix,
            sig_suffix=acquisition.sig_suffix,
            semantic_values=acquisition.semantic_values,
            dataset_source_setup_signature=(
                acquisition.dataset_source_setup_signature
            ),
            diagnostic_cache_contract_identity=(
                acquisition.diagnostic_cache_contract_identity
            ),
        ),
        roi_scope=contracts.GuidedBackendRoiScopeFacts(
            available=True,
            discovered_roi_ids=roi.discovered_roi_ids,
            included_roi_ids=roi.included_roi_ids,
            excluded_roi_ids=roi.excluded_roi_ids,
            selection_mode=roi.selection_mode,
            inventory_status=roi.inventory_status,
            inventory_source_content_digest=roi.inventory_source_content_digest,
            roi_inventory_identity_status=roi.roi_inventory_identity_status,
        ),
        correction=contracts.GuidedBackendCorrectionFacts(
            available=True,
            strategy_scope=correction.strategy_scope,
            global_correction_strategy=correction.global_correction_strategy,
            global_dynamic_fit_mode=correction.global_dynamic_fit_mode,
            dynamic_fit_parameter_values=correction.dynamic_fit_parameter_values,
            confirmed_marks=tuple(
                contracts.GuidedBackendConfirmedStrategyMarkFacts(
                    roi_id=mark.roi_id,
                    selected_dynamic_fit_mode=mark.selected_dynamic_fit_mode,
                    diagnostic_cache_id=mark.diagnostic_cache_id,
                    source_setup_signature=mark.source_setup_signature,
                    diagnostic_scope_signature=mark.diagnostic_scope_signature,
                    build_request_signature=mark.build_request_signature,
                    evidence_reference_id=mark.evidence_reference_id,
                    evidence_chunk=mark.evidence_chunk,
                    explicit_user_mark=mark.explicit_user_mark,
                    current=mark.current,
                )
                for mark in correction.confirmed_marks
            ),
            mark_rule_version=correction.mark_rule_version,
            currentness_rule_version=correction.currentness_rule_version,
            unanimity_rule_version=correction.unanimity_rule_version,
        ),
        diagnostic_cache=contracts.GuidedBackendDiagnosticCacheFacts(
            available=True,
            cache_id=diagnostic.cache_id,
            cache_root_canonical=diagnostic.cache_root_canonical,
            artifact_semantic_digest=diagnostic.artifact_semantic_digest,
            provenance_semantic_digest=diagnostic.provenance_semantic_digest,
            completed_run_rejection_category=(
                diagnostic.completed_run_rejection_category
            ),
            resolver_status=diagnostic.resolver_status,
            source_setup_signature=diagnostic.source_setup_signature,
            diagnostic_scope_signature=diagnostic.diagnostic_scope_signature,
            build_request_signature=diagnostic.build_request_signature,
            preliminary_cache=diagnostic.preliminary_cache,
            production_analysis=diagnostic.production_analysis,
        ),
        output=contracts.GuidedBackendOutputFacts(
            available=True,
            output_base_canonical=output.output_base_canonical,
            output_base_path_style=output.output_base_path_style,
            path_role=output.path_role,
            future_output_owner=output.future_output_owner,
            run_directory_strategy=output.run_directory_strategy,
            creation_timing=output.creation_timing,
            overwrite=output.overwrite,
            precreate=output.precreate,
            policy_status=output.policy_status,
            policy_current=output.policy_current,
            safety_classifier_version=output.safety_classifier_version,
            protected_root_context_complete=output.protected_root_context_complete,
            relationships=output.relationships,
            blocker_categories=output.blocker_categories,
            filesystem_fact_scope=output.filesystem_fact_scope,
        ),
        evidence_references=contracts.GuidedBackendEvidenceReferenceFacts(
            references=diagnostic.evidence_references,
            complete=True,
        ),
        feature_event=contracts.GuidedBackendFeatureEventFacts(
            available=True,
            profile_schema_version=feature.profile_schema_version,
            profile_id=feature.profile_id,
            effective_values=feature.effective_values,
            active_fields=feature.active_fields,
            inactive_fields=feature.inactive_fields,
            profile_status=feature.profile_status,
            explicitly_applied=feature.explicitly_applied,
            current=feature.current,
            visible_unapplied_changes=feature.visible_unapplied_changes,
        ),
        effective_feature_event_values=feature.effective_values,
        complete_for_compilation=True,
        unresolved_required_inputs=(),
        normalized_recording_description_identity=(
            request.normalized_recording_description_identity
        ),
    )


def _compiler_draft() -> GuidedNewAnalysisDraftPlan:
    return GuidedNewAnalysisDraftPlan(
        input_format="rwd",
        acquisition_mode="intermittent",
    )


def _unchecked_replace(instance, **changes):
    replacement = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            replacement,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return replacement


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
        "local_preview_setup_signature_mismatch",
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
        "incomplete_materialized_facts",
        "unresolved_materialized_inputs",
        "missing_source_snapshot",
        "unsupported_analysis_scope",
        "incomplete_final_policy_not_supported",
        "parser_contract_unavailable",
        "parser_unresolved_inputs",
        "feature_event_effective_value_unresolved",
        "unsupported_request_field",
        "compiler_internal_error",
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
        _compiler_draft(),
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


@pytest.mark.parametrize(
    "field,value,detail_code",
    [
        ("validation_scope", "wrong-scope", "validation_scope_mismatch"),
        (
            "validation_contract_version",
            "wrong-version",
            "validation_contract_version_mismatch",
        ),
        (
            "supported_subset_rule_version",
            "wrong-subset",
            "supported_subset_rule_version_mismatch",
        ),
        (
            "validator_capability_version",
            "unknown",
            "validator_capability_version_invalid",
        ),
        (
            "validator_capability_version",
            "",
            "validator_capability_version_invalid",
        ),
    ],
)
def test_validator_contract_value_gate_refusals(
    field: str,
    value: str,
    detail_code: str,
):
    contract = object.__new__(contracts.GuidedBackendValidatorContract)
    object.__setattr__(
        contract,
        "validation_scope",
        contracts.GUIDED_BACKEND_VALIDATION_SCOPE,
    )
    object.__setattr__(
        contract,
        "validation_contract_version",
        contracts.GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
    )
    object.__setattr__(
        contract,
        "supported_subset_rule_version",
        contracts.GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
    )
    object.__setattr__(
        contract,
        "validator_capability_version",
        "guided_backend_validator.test_fixture.v1",
    )
    object.__setattr__(contract, field, value)

    result = contracts.compile_guided_backend_validation_request(
        _compiler_draft(),
        facts=_complete_facts(),
        validator_contract=contract,
    )

    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    issue = result.blocking_issues[0]
    assert issue.category == "validator_contract_unavailable"
    assert issue.section == "validator_contract"
    assert issue.detail_code == detail_code
    assert result.no_partial_request is True
    assert result.no_request_identity is True
    assert not hasattr(result, "request")
    assert not hasattr(result, "canonical_request_identity")


def test_incomplete_fact_groups_refuse_even_when_flag_claims_complete():
    result = contracts.compile_guided_backend_validation_request(
        _compiler_draft(),
        facts=contracts.GuidedBackendValidationMaterializedFacts(
            complete_for_compilation=True
        ),
        validator_contract=_validator_contract(),
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == "missing_source_snapshot"


def test_complete_facts_compile_populated_request_with_identity():
    facts = _complete_facts()

    result = contracts.compile_guided_backend_validation_request(
        _compiler_draft(),
        facts=facts,
        validator_contract=_validator_contract(),
    )

    assert isinstance(result, contracts.GuidedBackendValidationCompileSuccess)
    assert result.status == "compiled"
    assert len(result.canonical_request_identity) == 64
    assert result.canonical_request_identity == (
        contracts.compute_guided_backend_validation_request_identity(result.request)
    )
    assert result.request_identity_deferred is False
    request = result.request
    assert isinstance(request, contracts.GuidedBackendValidationRequest)
    assert request.request_schema_name == contracts.GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME
    assert request.request_schema_version == contracts.GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION
    assert request.validation_scope == contracts.GUIDED_BACKEND_VALIDATION_SCOPE
    assert request.validation_contract_version == contracts.GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
    assert request.compiler_version == contracts.GUIDED_BACKEND_VALIDATION_COMPILER_VERSION
    assert request.subset_rule_version == contracts.GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
    assert request.canonicalization_algorithm_version == contracts.CANONICALIZATION_ALGORITHM_VERSION

    assert request.source.source_root_canonical == facts.source_snapshot.source_root_canonical
    assert request.source.source_root_path_style == facts.source_snapshot.source_root_path_style
    assert request.source.candidate_files is facts.source_snapshot.candidate_files
    assert request.source.source_identity_level == "content_bound_candidate_snapshot"

    assert request.acquisition_dataset.acquisition_mode == "intermittent"
    assert request.acquisition_dataset.sessions_per_hour == 6
    assert request.acquisition_dataset.timeline_anchor_mode == "civil"
    assert request.acquisition_dataset.classification_status == "not_requested"
    assert request.acquisition_dataset.semantic_values is facts.acquisition_dataset.semantic_values

    assert request.parser.time_column_candidates == facts.parser.time_column_candidates
    assert request.parser.parser_contract_digest == facts.parser.parser_contract_digest
    assert request.parser.unresolved_inputs == ()

    assert request.roi_scope.discovered_roi_ids == facts.roi_scope.discovered_roi_ids
    assert request.roi_scope.included_roi_ids == facts.roi_scope.included_roi_ids
    assert request.roi_scope.roi_inventory_identity_status == "deferred_not_authoritative"

    assert request.correction.global_dynamic_fit_mode == facts.correction.global_dynamic_fit_mode
    assert request.correction.dynamic_fit_parameter_values is facts.correction.dynamic_fit_parameter_values
    assert {mark.roi_id for mark in request.correction.confirmed_marks} == {"ROI0"}

    assert request.diagnostic_evidence.cache_id == facts.diagnostic_cache.cache_id
    assert request.diagnostic_evidence.evidence_references is facts.evidence_references.references
    evidence = request.diagnostic_evidence.evidence_references[0]
    assert evidence.roi_id == "ROI0"
    assert evidence.selected_dynamic_fit_mode == "global_linear_regression"

    assert request.feature_event.profile_id == facts.feature_event.profile_id
    assert request.feature_event.effective_values is facts.feature_event.effective_values
    assert request.output.output_base_canonical == facts.output.output_base_canonical
    assert request.output.relationships is facts.output.relationships
    assert request.local_contract.blocking_issue_categories == ()
    assert request.local_contract.unresolved_required_inputs == ()
    assert request.local_contract.deferred_capabilities == (
        "backend_validation",
        "run_authorization",
        "app_build_identity",
        "full_source_manifest_identity",
        "strict_roi_inventory_identity",
    )


def test_compiler_success_is_accepted_by_pure_backend_validator():
    from photometry_pipeline.guided_backend_validator import (
        validate_guided_backend_validation_request,
    )

    contract = _validator_contract()
    compiled = contracts.compile_guided_backend_validation_request(
        _compiler_draft(),
        facts=_complete_facts(),
        validator_contract=contract,
    )
    assert isinstance(compiled, contracts.GuidedBackendValidationCompileSuccess)

    result = validate_guided_backend_validation_request(
        compiled.request,
        canonical_request_identity=compiled.canonical_request_identity,
        validator_contract=contract,
    )
    assert result.accepted is True
    assert result.request_identity == compiled.canonical_request_identity
    assert result.run_authorization is False


@pytest.mark.parametrize(
    "identity,deferred",
    [
        (None, False),
        ("bad", False),
        ("A" * 64, False),
        (_DIGEST_A, True),
    ],
)
def test_compile_success_rejects_invalid_or_deferred_identity(
    identity: str | None,
    deferred: bool,
):
    with pytest.raises(contracts.GuidedBackendValidationRequestContractError):
        contracts.GuidedBackendValidationCompileSuccess(
            request=_request(),
            canonical_request_identity=identity,  # type: ignore[arg-type]
            request_identity_deferred=deferred,
        )


def test_compiler_uses_facts_not_mutable_draft_payload_fields():
    facts = _complete_facts()
    draft = _compiler_draft()
    draft.discovered_roi_ids = ["MUTATED"]
    draft.included_roi_ids = ["MUTATED"]
    draft.excluded_roi_ids = []
    draft.cache_root_path = "C:/mutated/cache"
    draft.cache_id = "mutated-cache"
    draft.feature_event_values = {"event_signal": "mutated"}
    draft.feature_event_profile_id = "mutated-profile"
    draft.output_policy_path = "C:/mutated/output"
    draft.per_roi_correction_strategy_choices = []

    result = contracts.compile_guided_backend_validation_request(
        draft,
        facts=facts,
        validator_contract=_validator_contract(),
    )

    assert isinstance(result, contracts.GuidedBackendValidationCompileSuccess)
    request = result.request
    assert request.roi_scope.included_roi_ids == ("ROI0",)
    assert request.diagnostic_evidence.cache_id == "cache-001"
    assert request.feature_event.profile_id == "profile-001"
    assert request.output.output_base_canonical == r"c:\output"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("incomplete", "incomplete_materialized_facts"),
        ("unresolved", "unresolved_materialized_inputs"),
        ("source", "missing_source_snapshot"),
        ("incomplete_final", "incomplete_final_policy_not_supported"),
        ("parser", "parser_contract_unavailable"),
        ("parser_unresolved", "parser_unresolved_inputs"),
        ("dataset", "missing_or_stale_dataset_contract"),
        ("roi", "missing_roi_inventory"),
        ("correction", "missing_confirmed_strategy_mark"),
        ("cache", "local_preview_setup_signature_mismatch"),
        ("cache_rejection", "diagnostic_cache_not_completed_run_ineligible"),
        ("evidence", "evidence_reference_missing_or_stale"),
        ("feature", "invalid_feature_event_profile"),
        ("output", "missing_output_policy"),
        ("output_blocker", "protected_root_context_incomplete"),
        ("protected_context", "protected_root_context_incomplete"),
    ],
)
def test_complete_fact_gate_refusals(mutation: str, expected: str):
    facts = _complete_facts()
    if mutation == "incomplete":
        facts = replace(facts, complete_for_compilation=False)
    elif mutation == "unresolved":
        facts = replace(facts, unresolved_required_inputs=("parser",))
    elif mutation == "source":
        facts = replace(
            facts, source_snapshot=replace(facts.source_snapshot, available=False)
        )
    elif mutation == "incomplete_final":
        facts = replace(
            facts,
            incomplete_final_classification=replace(
                facts.incomplete_final_classification,
                classification_status="excluded",
            ),
        )
    elif mutation == "parser":
        facts = replace(facts, parser=replace(facts.parser, available=False))
    elif mutation == "parser_unresolved":
        facts = replace(
            facts, parser=replace(facts.parser, unresolved_inputs=("time",))
        )
    elif mutation == "dataset":
        facts = replace(
            facts,
            acquisition_dataset=replace(
                facts.acquisition_dataset, available=False
            ),
        )
    elif mutation == "roi":
        facts = replace(facts, roi_scope=replace(facts.roi_scope, available=False))
    elif mutation == "correction":
        facts = replace(
            facts, correction=replace(facts.correction, available=False)
        )
    elif mutation == "cache":
        facts = replace(
            facts,
            diagnostic_cache=replace(facts.diagnostic_cache, available=False),
        )
    elif mutation == "cache_rejection":
        facts = replace(
            facts,
            diagnostic_cache=replace(
                facts.diagnostic_cache,
                completed_run_rejection_category="wrong",
            ),
        )
    elif mutation == "evidence":
        facts = replace(
            facts,
            evidence_references=replace(
                facts.evidence_references, complete=False
            ),
        )
    elif mutation == "feature":
        facts = replace(
            facts, feature_event=replace(facts.feature_event, available=False)
        )
    elif mutation == "output":
        facts = replace(facts, output=replace(facts.output, available=False))
    elif mutation == "output_blocker":
        facts = replace(
            facts,
            output=replace(facts.output, blocker_categories=("unsafe",)),
        )
    else:
        facts = replace(
            facts,
            output=replace(
                facts.output, protected_root_context_complete=False
            ),
        )

    result = contracts.compile_guided_backend_validation_request(
        _compiler_draft(),
        facts=facts,
        validator_contract=_validator_contract(),
    )

    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == expected
    assert result.no_partial_request is True
    assert result.no_request_identity is True
    assert not hasattr(result, "request")


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("input_format", "npm", "unsupported_source_format"),
        ("acquisition_mode", "continuous", "unsupported_acquisition_mode"),
        ("exclude_incomplete_final_rwd_chunk", True, None),
        ("allow_partial_final_window", True, "incomplete_final_policy_not_supported"),
    ],
)
def test_draft_first_subset_gate_refusals(field: str, value: object, expected: str):
    draft = _compiler_draft()
    setattr(draft, field, value)

    result = contracts.compile_guided_backend_validation_request(
        draft,
        facts=_complete_facts(),
        validator_contract=_validator_contract(),
    )

    if expected is None:
        assert isinstance(result, contracts.GuidedBackendValidationCompileSuccess)
    else:
        assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
        assert result.blocking_issues[0].category == expected


def test_compiler_performs_no_filesystem_io(monkeypatch: pytest.MonkeyPatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem I/O is prohibited")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(os, "stat", fail)
    monkeypatch.setattr(os, "scandir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os.path, "exists", fail)
    monkeypatch.setattr(os, "access", fail)
    result = contracts.compile_guided_backend_validation_request(
        GuidedNewAnalysisDraftPlan(),
        facts=contracts.GuidedBackendValidationMaterializedFacts(),
        validator_contract=_validator_contract(),
    )
    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)


def test_compiler_success_performs_no_filesystem_io(monkeypatch: pytest.MonkeyPatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem I/O is prohibited")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(Path, "read_bytes", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(os, "scandir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os.path, "exists", fail)

    with monkeypatch.context() as m:
        m.setattr(os, "stat", fail)
        m.setattr(os, "access", fail)
        result = contracts.compile_guided_backend_validation_request(
            _compiler_draft(),
            facts=_complete_facts(),
            validator_contract=_validator_contract(),
        )

    assert isinstance(result, contracts.GuidedBackendValidationCompileSuccess)


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


def test_identity_is_deterministic_digest_with_pinned_vector():
    request = _request()
    identical = replace(request)

    first = contracts.compute_guided_backend_validation_request_identity(request)

    # Pinned vector updated for B1: normalized_recording_description_identity
    # was added to the request identity payload (see
    # _GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS).
    assert first == (
        "71ba4c6e23ca8108f090f5db977f9ad515ffeedddc9cc100aab52fa069236e8f"
    )
    assert first == contracts.compute_guided_backend_validation_request_identity(
        request
    )
    assert first == contracts.compute_guided_backend_validation_request_identity(
        identical
    )
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_identity_payload_uses_domain_and_canonical_json_envelope():
    payload = contracts._guided_backend_validation_request_identity_payload(
        _request()
    )
    encoded = contracts.encode_canonical_value(payload)

    assert tuple(payload) == ("identity_domain", "request")
    assert (
        payload["identity_domain"]
        == contracts.GUIDED_BACKEND_VALIDATION_IDENTITY_DOMAIN
    )
    assert encoded.startswith(b'{"identity_domain":')
    assert b'"request":{' in encoded
    assert b" " not in encoded


def test_npm_acquisition_request_uses_format_neutral_disposition_policy():
    policy = contracts.GuidedBackendDispositionPolicyRequest(
        schema_name=contracts.GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_NAME,
        schema_version=contracts.GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_VERSION,
        admitted_dispositions=("process",),
        missing_session_policy="unsupported",
        excluded_session_policy="unsupported",
        partial_support_owner="parser_contract",
    )
    acquisition = contracts.GuidedBackendNpmAcquisitionDatasetRequest(
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
        allow_partial_final_window=False,
        dataset_snapshot_schema_version=(
            "guided_new_analysis_dataset_contract_snapshot.v1"
        ),
        dataset_status="applied",
        dataset_current_applied=True,
        semantic_values=(_typed_value("npm_led_col", "LedState"),),
        dataset_source_setup_signature="source-signature",
        diagnostic_cache_contract_identity="request-signature",
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        npm_target_fs_hz=40.0,
        npm_adapter_value_nan_policy="strict",
        disposition_policy=policy,
    )

    assert acquisition.disposition_policy is policy
    assert acquisition.disposition_policy.admitted_dispositions == ("process",)
    assert acquisition.disposition_policy.missing_session_policy == "unsupported"
    assert acquisition.disposition_policy.excluded_session_policy == "unsupported"
    assert acquisition.disposition_policy.partial_support_owner == "parser_contract"
    assert not hasattr(acquisition, "classification_schema_name")
    assert not hasattr(acquisition, "classifier_version")
    assert not hasattr(acquisition, "not_requested_classification_digest")


def test_npm_disposition_policy_is_part_of_parent_request_identity():
    from photometry_pipeline.guided_normalized_recording import (
        compute_npm_parser_contract_digest,
    )

    policy = contracts.GuidedBackendDispositionPolicyRequest(
        schema_name=contracts.GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_NAME,
        schema_version=contracts.GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_VERSION,
        admitted_dispositions=("process",),
        missing_session_policy="unsupported",
        excluded_session_policy="unsupported",
        partial_support_owner="parser_contract",
    )
    acquisition = contracts.GuidedBackendNpmAcquisitionDatasetRequest(
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
        allow_partial_final_window=False,
        dataset_snapshot_schema_version=(
            "guided_new_analysis_dataset_contract_snapshot.v1"
        ),
        dataset_status="applied",
        dataset_current_applied=True,
        semantic_values=(_typed_value("npm_led_col", "LedState"),),
        dataset_source_setup_signature="source-signature",
        diagnostic_cache_contract_identity="request-signature",
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        npm_target_fs_hz=40.0,
        npm_adapter_value_nan_policy="strict",
        disposition_policy=policy,
    )
    parser_content = {"parser": "npm"}
    parser = contracts.GuidedBackendNpmParserRequest(
        schema_name="npm_parser_contract",
        schema_version="v1",
        timestamp_column_candidates=("SystemTimestamp",),
        parser_contract_digest=compute_npm_parser_contract_digest(parser_content),
        parser_contract_content=parser_content,
    )
    request = replace(
        _request(),
        source=replace(_request().source, source_format="npm"),
        acquisition_dataset=acquisition,
        parser=parser,
        normalized_recording_description={"adapter_format": "npm"},
    )
    changed_policy = replace(policy, partial_support_owner="adapter_policy")
    changed = replace(
        request,
        acquisition_dataset=replace(
            acquisition,
            disposition_policy=changed_policy,
        ),
    )

    assert contracts.compute_guided_backend_validation_request_identity(changed) != (
        contracts.compute_guided_backend_validation_request_identity(request)
    )


def test_identity_mapper_covers_every_request_dataclass_field():
    expected_types = {
        contracts.GuidedBackendValidationRequest,
        contracts.GuidedBackendSourceRequest,
        contracts.GuidedBackendAcquisitionDatasetRequest,
        contracts.GuidedBackendNpmAcquisitionDatasetRequest,
        contracts.GuidedBackendDispositionPolicyRequest,
        contracts.GuidedBackendRwdParserRequest,
        contracts.GuidedBackendRoiScopeRequest,
        contracts.GuidedBackendCorrectionRequest,
        contracts.GuidedBackendDiagnosticEvidenceRequest,
        contracts.GuidedBackendFeatureEventRequest,
        contracts.GuidedBackendOutputRequest,
        contracts.GuidedBackendLocalContractState,
        contracts.GuidedBackendTypedFieldValue,
        contracts.GuidedBackendSourceCandidateFile,
        contracts.GuidedBackendConfirmedStrategyMark,
        contracts.GuidedBackendPerRoiProductionStrategy,
        contracts.GuidedBackendPerRoiFeatureEvent,
        contracts.GuidedBackendEvidenceReference,
        contracts.GuidedBackendOutputRelationship,
    }
    coverage = contracts._GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS

    assert set(coverage) == expected_types
    for model_type, mapped_names in coverage.items():
        assert mapped_names == tuple(item.name for item in fields(model_type))


@pytest.mark.parametrize(
    "field_name",
    [
        "request_schema_name",
        "request_schema_version",
        "validation_scope",
        "validation_contract_version",
        "validator_capability_version",
        "compiler_version",
        "subset_rule_version",
        "canonicalization_algorithm_version",
    ],
)
def test_identity_changes_for_every_top_level_contract_field(field_name: str):
    request = _request()
    changed = _unchecked_replace(
        request,
        **{field_name: f"{getattr(request, field_name)}.changed"},
    )
    assert contracts.compute_guided_backend_validation_request_identity(
        changed
    ) != contracts.compute_guided_backend_validation_request_identity(request)


@pytest.mark.parametrize(
    "section,mutate",
    [
        (
            "top_level",
            lambda request: _unchecked_replace(
                request, validator_capability_version="validator.v2"
            ),
        ),
        (
            "source",
            lambda request: _unchecked_replace(
                request,
                source=_unchecked_replace(
                    request.source, source_candidate_content_digest="d" * 64
                ),
            ),
        ),
        (
            "acquisition",
            lambda request: _unchecked_replace(
                request,
                acquisition_dataset=_unchecked_replace(
                    request.acquisition_dataset, session_duration_sec=121.0
                ),
            ),
        ),
        (
            "parser",
            lambda request: _unchecked_replace(
                request,
                parser=_unchecked_replace(
                    request.parser, parser_contract_digest="d" * 64
                ),
            ),
        ),
        (
            "roi",
            lambda request: _unchecked_replace(
                request,
                roi_scope=_unchecked_replace(
                    request.roi_scope, included_roi_ids=("ROI1",)
                ),
            ),
        ),
        (
            "correction",
            lambda request: _unchecked_replace(
                request,
                correction=_unchecked_replace(
                    request.correction,
                    global_dynamic_fit_mode="robust_global_event_reject",
                ),
            ),
        ),
        (
            "diagnostic",
            lambda request: _unchecked_replace(
                request,
                diagnostic_evidence=_unchecked_replace(
                    request.diagnostic_evidence,
                    artifact_semantic_digest="d" * 64,
                ),
            ),
        ),
        (
            "feature",
            lambda request: _unchecked_replace(
                request,
                feature_event=_unchecked_replace(
                    request.feature_event, profile_id="profile-002"
                ),
            ),
        ),
        (
            "output",
            lambda request: _unchecked_replace(
                request,
                output=_unchecked_replace(
                    request.output, output_base_canonical=r"c:\other"
                ),
            ),
        ),
        (
            "local_contract",
            lambda request: _unchecked_replace(
                request,
                local_contract=_unchecked_replace(
                    request.local_contract,
                    deferred_capabilities=("different",),
                ),
            ),
        ),
    ],
)
def test_identity_changes_for_every_request_section(section: str, mutate):
    request = _request()
    assert contracts.compute_guided_backend_validation_request_identity(
        mutate(request)
    ) != contracts.compute_guided_backend_validation_request_identity(request)


def test_identity_changes_for_nested_candidate_mark_evidence_and_relationship():
    request = _request()
    source = _unchecked_replace(
        request.source,
        candidate_files=(
            _unchecked_replace(
                request.source.candidate_files[0], size_bytes=999
            ),
        ),
    )
    mark = _unchecked_replace(
        request.correction.confirmed_marks[0], evidence_chunk=2
    )
    correction = _unchecked_replace(request.correction, confirmed_marks=(mark,))
    evidence = _unchecked_replace(
        request.diagnostic_evidence.evidence_references[0], evidence_chunk=2
    )
    diagnostic = _unchecked_replace(
        request.diagnostic_evidence, evidence_references=(evidence,)
    )
    relationship = _unchecked_replace(
        request.output.relationships[0], status="different"
    )
    output = _unchecked_replace(request.output, relationships=(relationship,))
    original = contracts.compute_guided_backend_validation_request_identity(request)

    for changed in (
        _unchecked_replace(request, source=source),
        _unchecked_replace(request, correction=correction),
        _unchecked_replace(request, diagnostic_evidence=diagnostic),
        _unchecked_replace(request, output=output),
    ):
        assert (
            contracts.compute_guided_backend_validation_request_identity(changed)
            != original
        )


def test_identity_preserves_tuple_order_and_scalar_types():
    request = _request()
    reordered = _unchecked_replace(
        request.parser,
        time_column_candidates=("Timestamp",)
        + request.parser.time_column_candidates,
    )
    base_value = request.feature_event.effective_values[0]
    scalar_values = (
        _unchecked_replace(base_value, value=None, value_type="NoneType"),
        _unchecked_replace(base_value, value=True, value_type="bool"),
        _unchecked_replace(base_value, value=1, value_type="int"),
        _unchecked_replace(base_value, value=1.0, value_type="float"),
        _unchecked_replace(base_value, value="1", value_type="str"),
    )

    identities = [
        contracts.compute_guided_backend_validation_request_identity(request),
        contracts.compute_guided_backend_validation_request_identity(
            _unchecked_replace(request, parser=reordered)
        ),
    ]
    identities.extend(
        contracts.compute_guided_backend_validation_request_identity(
            _unchecked_replace(
                request,
                feature_event=_unchecked_replace(
                    request.feature_event, effective_values=(scalar_value,)
                ),
            )
        )
        for scalar_value in scalar_values
    )
    assert len(set(identities)) == len(identities)


def test_identity_unicode_encoding_is_deterministic():
    request = _request()
    unicode_request = _unchecked_replace(
        request,
        validator_capability_version="válidator.Δ.v1",
    )
    first = contracts.compute_guided_backend_validation_request_identity(
        unicode_request
    )
    second = contracts.compute_guided_backend_validation_request_identity(
        _unchecked_replace(
            request,
            validator_capability_version="válidator.Δ.v1",
        )
    )
    assert first == second


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), object()])
def test_identity_rejects_unsupported_or_non_finite_values(bad_value):
    request = _request()
    typed = _unchecked_replace(
        request.feature_event.effective_values[0], value=bad_value
    )
    request = _unchecked_replace(
        request,
        feature_event=_unchecked_replace(
            request.feature_event, effective_values=(typed,)
        ),
    )

    with pytest.raises(contracts.GuidedBackendValidationRequestContractError):
        contracts.compute_guided_backend_validation_request_identity(request)


def test_identity_rejects_non_request_input():
    with pytest.raises(
        contracts.GuidedBackendValidationRequestContractError,
        match="GuidedBackendValidationRequest",
    ):
        contracts.compute_guided_backend_validation_request_identity(object())


def test_identity_failure_inside_compiler_returns_no_partial_request(monkeypatch):
    def fail(_request):
        raise contracts.GuidedBackendValidationRequestContractError("injected")

    monkeypatch.setattr(
        contracts,
        "compute_guided_backend_validation_request_identity",
        fail,
    )
    result = contracts.compile_guided_backend_validation_request(
        _compiler_draft(),
        facts=_complete_facts(),
        validator_contract=_validator_contract(),
    )

    assert isinstance(result, contracts.GuidedBackendValidationCompileFailure)
    assert result.blocking_issues[0].category == "compiler_internal_error"
    assert (
        result.blocking_issues[0].detail_code
        == "request_identity_computation_failed"
    )
    assert result.no_partial_request is True
    assert result.no_request_identity is True
    assert not hasattr(result, "request")
    assert not hasattr(result, "canonical_request_identity")


def test_identity_success_performs_no_filesystem_io(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem I/O is prohibited")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(Path, "read_bytes", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(os, "stat", fail)
    monkeypatch.setattr(os, "scandir", fail)
    monkeypatch.setattr(os.path, "exists", fail)
    monkeypatch.setattr(os, "access", fail)

    identity = contracts.compute_guided_backend_validation_request_identity(
        _request()
    )
    assert identity
    assert len(identity) == 64


def test_backend_validation_identity_includes_applied_dff_orchestration_enabled():
    import dataclasses
    req1 = _request()
    new_correction = dataclasses.replace(req1.correction, applied_dff_orchestration_enabled=not req1.correction.applied_dff_orchestration_enabled)
    req2 = dataclasses.replace(req1, correction=new_correction)
    id1 = contracts.compute_guided_backend_validation_request_identity(req1)
    id2 = contracts.compute_guided_backend_validation_request_identity(req2)
    assert id1 != id2
