from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, fields, replace
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_backend_validation_request as contracts
import photometry_pipeline.guided_backend_validator as validator


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_CAPABILITY = "guided_backend_validator.test.v1"


def _typed(name: str, value: object):
    return contracts.GuidedBackendTypedFieldValue(
        field_name=name,
        value_type=type(value).__name__,
        value=value,  # type: ignore[arg-type]
    )


def _contract() -> contracts.GuidedBackendValidatorContract:
    return contracts.GuidedBackendValidatorContract(
        validation_scope=contracts.GUIDED_BACKEND_VALIDATION_SCOPE,
        validation_contract_version=(
            contracts.GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
        ),
        validator_capability_version=_CAPABILITY,
        supported_subset_rule_version=(
            contracts.GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
        ),
    )


def _request() -> contracts.GuidedBackendValidationRequest:
    source = contracts.GuidedBackendSourceRequest(
        source_root_canonical=r"c:\source",
        source_root_path_style="windows_drive",
        source_format="rwd",
        snapshot_schema_name=contracts.GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_NAME,
        snapshot_schema_version=(
            contracts.GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_VERSION
        ),
        discovery_rule_version=(
            contracts.GUIDED_BACKEND_SOURCE_DISCOVERY_RULE_VERSION
        ),
        path_canonicalization_version=(
            contracts.CANONICALIZATION_ALGORITHM_VERSION
        ),
        relative_path_rule_version=(
            contracts.GUIDED_BACKEND_SOURCE_RELATIVE_PATH_RULE_VERSION
        ),
        ignored_files_policy=(
            contracts.GUIDED_BACKEND_SOURCE_IGNORED_FILES_POLICY
        ),
        build_mode="read_only",
        source_candidate_set_digest=_A,
        source_candidate_content_digest=_B,
        candidate_files=(
            contracts.GuidedBackendSourceCandidateFile(
                canonical_relative_path="session/fluorescence.csv",
                size_bytes=42,
                sha256_content_digest=_A,
            ),
        ),
    )
    dataset = contracts.GuidedBackendAcquisitionDatasetRequest(
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
        allow_partial_final_window=False,
        exclude_incomplete_final_rwd_chunk=False,
        classification_schema_name=(
            contracts.GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_NAME
        ),
        classification_schema_version=(
            contracts.GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_VERSION
        ),
        classifier_version=(
            contracts.GUIDED_BACKEND_INCOMPLETE_FINAL_CLASSIFIER_VERSION
        ),
        classification_status="not_requested",
        not_requested_classification_digest=_C,
        dataset_snapshot_schema_version="dataset_snapshot.v1",
        dataset_status="applied",
        dataset_current_applied=True,
        rwd_time_col="Time(s)",
        uv_suffix="-410",
        sig_suffix="-470",
        semantic_values=(_typed("rwd_time_col", "Time(s)"),),
        dataset_source_setup_signature="source-signature",
        diagnostic_cache_contract_identity="build-signature",
    )
    parser = contracts.GuidedBackendRwdParserRequest(
        schema_name="rwd_header_parsing_contract",
        schema_version="v1",
        header_search_line_limit=60,
        time_column_candidates=("Time(s)",),
        uv_suffix_candidates=("-410",),
        signal_suffix_candidates=("-470",),
        column_normalization_rule="normalization.v1",
        roi_name_rule="roi.v1",
        ambiguity_policy="reject.v1",
        parser_contract_digest=_A,
    )
    roi = contracts.GuidedBackendRoiScopeRequest(
        discovered_roi_ids=("ROI0", "ROI1"),
        included_roi_ids=("ROI0",),
        excluded_roi_ids=("ROI1",),
        inventory_source_content_digest=_B,
    )
    mark = contracts.GuidedBackendConfirmedStrategyMark(
        roi_id="ROI0",
        selected_dynamic_fit_mode="global_linear_regression",
        diagnostic_cache_id="cache-001",
        source_setup_signature="source-signature",
        diagnostic_scope_signature="scope-signature",
        build_request_signature="build-signature",
        evidence_reference_id="evidence-001",
        evidence_chunk=0,
    )
    correction = contracts.GuidedBackendCorrectionRequest(
        strategy_scope="global",
        global_correction_strategy="dynamic_fit",
        global_dynamic_fit_mode="global_linear_regression",
        dynamic_fit_parameter_values=(
            _typed("dynamic_fit_mode", "global_linear_regression"),
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
        build_request_signature="build-signature",
        evidence_chunk=0,
        roi_id="ROI0",
        selected_dynamic_fit_mode="global_linear_regression",
    )
    diagnostic = contracts.GuidedBackendDiagnosticEvidenceRequest(
        cache_id="cache-001",
        cache_root_canonical=r"c:\cache",
        source_setup_signature="source-signature",
        diagnostic_scope_signature="scope-signature",
        build_request_signature="build-signature",
        artifact_contract_version=(
            contracts.GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION
        ),
        provenance_schema_version=(
            contracts.GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION
        ),
        artifact_semantic_digest=_A,
        provenance_semantic_digest=_B,
        evidence_references=(evidence,),
        completed_run_rejection_category="guided_diagnostic_cache_ineligible",
        resolver_status="current",
        preliminary_cache=True,
        production_analysis=False,
    )
    feature = contracts.GuidedBackendFeatureEventRequest(
        profile_schema_version=(
            contracts.GUIDED_BACKEND_FEATURE_EVENT_PROFILE_SCHEMA_VERSION
        ),
        profile_id="profile-001",
        effective_values=(_typed("event_signal", "dff"),),
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
        safety_classifier_version="guided_output_base_safety_ownership.v1",
        relationships=(
            contracts.GuidedBackendOutputRelationship(
                relationship="output_base_equals_source",
                root_kind="source",
                status="safe",
            ),
        ),
        protected_root_context_complete=True,
        blocker_categories=(),
        filesystem_fact_scope=(
            "read_only_path_relationships_no_writability_probe"
        ),
    )
    local = contracts.GuidedBackendLocalContractState(
        local_check_contract_version=(
            contracts.GUIDED_BACKEND_LOCAL_CHECK_CONTRACT_VERSION
        ),
        blocking_issue_categories=(),
        warning_categories=(),
        unsupported_state_flags=(),
        unresolved_required_inputs=(),
        deferred_capabilities=(
            validator.GUIDED_BACKEND_VALIDATOR_DEFERRED_CAPABILITIES
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
        validator_capability_version=_CAPABILITY,
        compiler_version=contracts.GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
        subset_rule_version=(
            contracts.GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
        ),
        canonicalization_algorithm_version=(
            contracts.CANONICALIZATION_ALGORITHM_VERSION
        ),
        source=source,
        acquisition_dataset=dataset,
        parser=parser,
        roi_scope=roi,
        correction=correction,
        diagnostic_evidence=diagnostic,
        feature_event=feature,
        output=output,
        local_contract=local,
    )


def _local_preview_request() -> contracts.GuidedBackendValidationRequest:
    """A request variant using cache-free local_correction_preview evidence,
    proving currentness via source/setup signature instead of diagnostic-
    cache identity."""
    base = _request()
    mark = replace(
        base.correction.confirmed_marks[0],
        diagnostic_cache_id="",
        diagnostic_scope_signature="",
        build_request_signature="",
        source_setup_signature="local-preview-setup-signature",
    )
    evidence = replace(
        base.diagnostic_evidence.evidence_references[0],
        evidence_kind="local_correction_preview",
        diagnostic_cache_id="",
        diagnostic_scope_signature="",
        build_request_signature="",
        source_setup_signature="local-preview-setup-signature",
    )
    correction = replace(
        base.correction,
        confirmed_marks=(mark,),
        currentness_rule_version="source_setup_bound_currentness.v1",
    )
    diagnostic = replace(
        base.diagnostic_evidence,
        cache_id="",
        cache_root_canonical="",
        source_setup_signature="",
        diagnostic_scope_signature="",
        build_request_signature="",
        artifact_contract_version="",
        provenance_schema_version="",
        artifact_semantic_digest="",
        provenance_semantic_digest="",
        evidence_references=(evidence,),
        completed_run_rejection_category="",
        resolver_status="",
        preliminary_cache=False,
        available=False,
    )
    return replace(base, correction=correction, diagnostic_evidence=diagnostic)


def _unchecked(instance, **changes):
    replacement = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            replacement,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return replacement


def _validate(
    request: contracts.GuidedBackendValidationRequest,
    contract: contracts.GuidedBackendValidatorContract | None = None,
):
    identity = contracts.compute_guided_backend_validation_request_identity(
        request
    )
    return validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=contract or _contract(),
    )


def _category(result) -> str:
    return result.blocking_issues[0].category


def test_local_preview_request_accepted_without_diagnostic_cache():
    result = _validate(_local_preview_request())
    assert result.accepted is True
    assert result.status == "accepted"


def test_local_preview_request_rejects_cache_backed_evidence_kind():
    request = _local_preview_request()
    # Bypass GuidedBackendEvidenceReference.__post_init__ (which already
    # refuses to construct a cache-backed evidence_kind with an empty
    # diagnostic_cache_id) to prove the validator itself also rejects this
    # inconsistency, independent of the earlier construction-time guard.
    stale_evidence = _unchecked(
        request.diagnostic_evidence.evidence_references[0],
        evidence_kind="correction_preview",
    )
    request = replace(
        request,
        diagnostic_evidence=replace(
            request.diagnostic_evidence, evidence_references=(stale_evidence,)
        ),
    )
    result = _validate(request)
    assert result.accepted is False
    assert _category(result) == "evidence_reference_missing_or_stale"


def test_local_preview_request_rejects_signature_mismatch():
    request = _local_preview_request()
    mismatched_mark = replace(
        request.correction.confirmed_marks[0],
        source_setup_signature="different-signature",
    )
    request = replace(
        request,
        correction=replace(request.correction, confirmed_marks=(mismatched_mark,)),
    )
    result = _validate(request)
    assert result.accepted is False
    assert _category(result) == "local_preview_setup_signature_mismatch"


def test_diagnostic_cache_backed_request_still_requires_cache():
    request = _request()
    assert request.diagnostic_evidence.available is True
    result = _validate(request)
    assert result.accepted is True

    unavailable_but_cache_shaped = replace(
        request,
        diagnostic_evidence=_unchecked(
            request.diagnostic_evidence, cache_id="tampered-cache-id"
        ),
    )
    tampered_result = _validate(unavailable_but_cache_shaped)
    assert tampered_result.accepted is False


def test_module_has_strict_backend_neutral_import_boundary():
    source = Path(validator.__file__).read_text(encoding="utf-8")
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    prohibited = {
        "gui.",
        "RunSpec",
        "guided_backend_validation_materialization",
        "build_rwd_source_candidate_snapshot",
        "resolve_diagnostic_cache_source",
        "classify_output_base_safety_ownership",
        "validate_output_write_safety",
        "compile_guided_backend_validation_request",
    }
    assert not any(
        imported == "gui"
        or any(marker in imported for marker in prohibited)
        for imported in imports
    )


def test_taxonomy_is_exact_and_duplicate_free():
    assert len(validator.GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORIES) == len(
        set(validator.GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORIES)
    )
    assert validator.GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORY_SET == frozenset(
        validator.GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORIES
    )


def test_result_models_are_frozen_and_exclude_execution_fields():
    result = _validate(_request())
    with pytest.raises(FrozenInstanceError):
        result.accepted = False  # type: ignore[misc]
    prohibited = {
        "config",
        "argv",
        "command",
        "run_id",
        "artifact",
        "timestamp",
        "output_path",
    }
    result_fields = {item.name for item in fields(result)}
    assert not result_fields & prohibited
    assert result.run_authorization is False


def test_result_model_invariants_reject_inconsistent_states():
    issue = validator.GuidedBackendValidationIssue(
        category="validator_internal_error",
        section="validator",
        message="failed",
    )
    with pytest.raises(ValueError):
        validator.GuidedBackendValidationResult(
            status="accepted",
            request_identity=_A,
            accepted=True,
            blocking_issues=(issue,),
            warning_categories=(),
            validator_contract_version="v1",
            validator_capability_version="v1",
            validated_request_scope="scope",
        )
    with pytest.raises(ValueError):
        validator.GuidedBackendValidationResult(
            status="refused",
            request_identity=None,
            accepted=False,
            blocking_issues=(),
            warning_categories=(),
            validator_contract_version="v1",
            validator_capability_version="v1",
            validated_request_scope="scope",
        )
    with pytest.raises(ValueError):
        validator.GuidedBackendValidationResult(
            status="accepted",
            request_identity=_A,
            accepted=True,
            blocking_issues=(),
            warning_categories=(),
            validator_contract_version="v1",
            validator_capability_version="v1",
            validated_request_scope="scope",
            run_authorization=True,
        )


def test_valid_request_is_accepted_without_run_authorization():
    request = _request()
    identity = contracts.compute_guided_backend_validation_request_identity(
        request
    )
    result = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=_contract(),
    )
    assert result.status == "accepted"
    assert result.accepted is True
    assert result.request_identity == identity
    assert result.blocking_issues == ()
    assert result.warning_categories == ()
    assert result.no_files_written is True
    assert result.no_directories_created is True
    assert result.no_run_id_allocated is True
    assert result.no_runner_invoked is True
    assert result.no_artifacts_created is True
    assert result.run_authorization is False


@pytest.mark.parametrize(
    "identity,expected",
    [
        ("", "request_identity_missing"),
        ("bad", "request_identity_invalid"),
        ("A" * 64, "request_identity_invalid"),
        ("f" * 64, "request_identity_mismatch"),
    ],
)
def test_identity_refusals_have_no_request_identity(identity: str, expected: str):
    result = validator.validate_guided_backend_validation_request(
        _request(),
        canonical_request_identity=identity,
        validator_contract=_contract(),
    )
    assert result.accepted is False
    assert _category(result) == expected
    assert result.request_identity is None
    assert result.run_authorization is False


def test_non_request_refuses():
    result = validator.validate_guided_backend_validation_request(
        object(),  # type: ignore[arg-type]
        canonical_request_identity=_A,
        validator_contract=_contract(),
    )
    assert _category(result) == "request_missing_or_invalid"
    assert result.request_identity is None


def test_identity_computation_failure_refuses(monkeypatch):
    monkeypatch.setattr(
        validator,
        "compute_guided_backend_validation_request_identity",
        lambda _request: (_ for _ in ()).throw(ValueError("injected")),
    )
    result = validator.validate_guided_backend_validation_request(
        _request(),
        canonical_request_identity=_A,
        validator_contract=_contract(),
    )
    assert _category(result) == "request_identity_computation_failed"
    assert result.request_identity is None


@pytest.mark.parametrize(
    "field,value,detail_code",
    [
        ("validation_scope", "wrong", "validation_scope_mismatch"),
        (
            "validation_contract_version",
            "wrong",
            "validation_contract_version_mismatch",
        ),
        (
            "supported_subset_rule_version",
            "wrong",
            "supported_subset_rule_version_mismatch",
        ),
        (
            "validator_capability_version",
            "unknown",
            "validator_capability_version_invalid",
        ),
        (
            "validator_capability_version",
            "different",
            "validator_capability_version_mismatch",
        ),
    ],
)
def test_validator_contract_refusals(field: str, value: str, detail_code: str):
    contract = _unchecked(_contract(), **{field: value})
    result = _validate(_request(), contract)
    assert _category(result) == "validator_contract_unavailable"
    assert result.blocking_issues[0].detail_code == detail_code
    assert result.request_identity is not None


def test_invalid_validator_contract_type_refuses():
    request = _request()
    result = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=(
            contracts.compute_guided_backend_validation_request_identity(
                request
            )
        ),
        validator_contract=object(),  # type: ignore[arg-type]
    )
    assert _category(result) == "validator_contract_unavailable"


@pytest.mark.parametrize(
    "changes,detail_code",
    [
        ({"current": False}, "evidence_reference_not_current"),
        (
            {"evidence_reference_id": ""},
            "evidence_reference_identity_incomplete",
        ),
        ({"evidence_chunk": None}, "evidence_reference_chunk_invalid"),
        ({"evidence_chunk": -1}, "evidence_reference_chunk_invalid"),
        ({"evidence_chunk": True}, "evidence_reference_chunk_invalid"),
    ],
)
def test_incomplete_or_stale_evidence_reference_refuses(
    changes: dict[str, object],
    detail_code: str,
):
    request = _request()
    reference = _unchecked(
        request.diagnostic_evidence.evidence_references[0],
        **changes,
    )
    request = _unchecked(
        request,
        diagnostic_evidence=_unchecked(
            request.diagnostic_evidence,
            evidence_references=(reference,),
        ),
    )

    result = _validate(request)

    assert result.accepted is False
    assert _category(result) == "evidence_reference_missing_or_stale"
    assert result.blocking_issues[0].detail_code == detail_code


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            lambda request: _unchecked(request, request_schema_version="v2"),
            "unsupported_request_schema",
        ),
        (
            lambda request: _unchecked(
                request,
                source=_unchecked(request.source, source_format="npm"),
            ),
            "unsupported_source_format",
        ),
        (
            lambda request: _unchecked(
                request,
                source=_unchecked(
                    request.source,
                    candidate_files=(
                        _unchecked(
                            request.source.candidate_files[0],
                            canonical_relative_path="../bad.csv",
                        ),
                    ),
                ),
            ),
            "unsupported_request_field",
        ),
        (
            lambda request: _unchecked(
                request,
                acquisition_dataset=_unchecked(
                    request.acquisition_dataset, acquisition_mode="continuous"
                ),
            ),
            "unsupported_acquisition_mode",
        ),
        (
            lambda request: _unchecked(
                request,
                acquisition_dataset=_unchecked(
                    request.acquisition_dataset,
                    exclude_incomplete_final_rwd_chunk=True,
                ),
            ),
            "incomplete_final_policy_not_supported",
        ),
        (
            lambda request: _unchecked(
                request,
                parser=_unchecked(
                    request.parser, unresolved_inputs=("time",)
                ),
            ),
            "parser_unresolved_inputs",
        ),
        (
            lambda request: _unchecked(
                request,
                roi_scope=_unchecked(
                    request.roi_scope,
                    discovered_roi_ids=("ROI0", "ROI0"),
                ),
            ),
            "duplicate_roi_id",
        ),
        (
            lambda request: _unchecked(
                request,
                roi_scope=_unchecked(
                    request.roi_scope,
                    excluded_roi_ids=(),
                ),
            ),
            "included_excluded_roi_conflict",
        ),
        (
            lambda request: _unchecked(
                request,
                correction=_unchecked(
                    request.correction,
                    global_dynamic_fit_mode="signal_only_f0",
                ),
            ),
            "signal_only_not_supported_for_validate",
        ),
        (
            lambda request: _unchecked(
                request,
                correction=_unchecked(
                    request.correction,
                    confirmed_marks=(),
                ),
            ),
            "missing_confirmed_strategy_mark",
        ),
        (
            lambda request: _unchecked(
                request,
                correction=_unchecked(
                    request.correction,
                    confirmed_marks=(
                        _unchecked(
                            request.correction.confirmed_marks[0],
                            selected_dynamic_fit_mode=(
                                "robust_global_event_reject"
                            ),
                        ),
                    ),
                ),
            ),
            "mixed_dynamic_fit_modes",
        ),
        (
            lambda request: _unchecked(
                request,
                diagnostic_evidence=_unchecked(
                    request.diagnostic_evidence,
                    completed_run_rejection_category="wrong",
                ),
            ),
            "diagnostic_cache_not_completed_run_ineligible",
        ),
        (
            lambda request: _unchecked(
                request,
                diagnostic_evidence=_unchecked(
                    request.diagnostic_evidence, resolver_status="stale"
                ),
            ),
            "missing_or_stale_diagnostic_cache",
        ),
        (
            lambda request: _unchecked(
                request,
                diagnostic_evidence=_unchecked(
                    request.diagnostic_evidence, evidence_references=()
                ),
            ),
            "evidence_reference_missing_or_stale",
        ),
        (
            lambda request: _unchecked(
                request,
                feature_event=_unchecked(
                    request.feature_event, current=False
                ),
            ),
            "invalid_feature_event_profile",
        ),
        (
            lambda request: _unchecked(
                request,
                feature_event=_unchecked(
                    request.feature_event,
                    effective_values=(
                        _unchecked(
                            request.feature_event.effective_values[0],
                            source_classification="unresolved",
                        ),
                    ),
                ),
            ),
            "feature_event_effective_value_unresolved",
        ),
        (
            lambda request: _unchecked(
                request,
                output=_unchecked(request.output, overwrite=True),
            ),
            "overwrite_not_allowed",
        ),
        (
            lambda request: _unchecked(
                request,
                output=_unchecked(
                    request.output, protected_root_context_complete=False
                ),
            ),
            "protected_root_context_incomplete",
        ),
        (
            lambda request: _unchecked(
                request,
                local_contract=_unchecked(
                    request.local_contract, warning_categories=("warning",)
                ),
            ),
            "local_contract_not_clean",
        ),
    ],
)
def test_structural_section_refusals(mutation, expected: str):
    result = _validate(mutation(_request()))
    assert result.accepted is False
    assert _category(result) == expected
    assert result.request_identity is not None
    assert result.run_authorization is False


@pytest.mark.parametrize(
    "mutation,expected_detail",
    [
        (
            lambda request: _unchecked(
                request,
                roi_scope=_unchecked(
                    request.roi_scope,
                    inventory_source_content_digest=_C,
                ),
            ),
            "roi_source_content_digest_mismatch",
        ),
        (
            lambda request: _unchecked(
                request,
                acquisition_dataset=_unchecked(
                    request.acquisition_dataset,
                    dataset_source_setup_signature="different",
                ),
            ),
            "dataset_cache_identity_mismatch",
        ),
        (
            lambda request: _unchecked(
                request,
                acquisition_dataset=_unchecked(
                    request.acquisition_dataset,
                    diagnostic_cache_contract_identity="different",
                ),
            ),
            "dataset_cache_identity_mismatch",
        ),
        (
            lambda request: _unchecked(
                request,
                correction=_unchecked(
                    request.correction,
                    confirmed_marks=(
                        _unchecked(
                            request.correction.confirmed_marks[0],
                            diagnostic_cache_id="different",
                        ),
                    ),
                ),
            ),
            "mark_evidence_cache_binding_mismatch",
        ),
        (
            lambda request: _unchecked(
                request,
                diagnostic_evidence=_unchecked(
                    request.diagnostic_evidence,
                    evidence_references=(
                        _unchecked(
                            request.diagnostic_evidence.evidence_references[0],
                            selected_dynamic_fit_mode=(
                                "robust_global_event_reject"
                            ),
                        ),
                    ),
                ),
            ),
            "mark_evidence_mode_mismatch",
        ),
        (
            lambda request: _unchecked(
                request,
                diagnostic_evidence=_unchecked(
                    request.diagnostic_evidence,
                    evidence_references=(
                        _unchecked(
                            request.diagnostic_evidence.evidence_references[0],
                            evidence_reference_id="different",
                        ),
                    ),
                ),
            ),
            "mark_evidence_reference_mismatch",
        ),
        (
            lambda request: _unchecked(
                request,
                output=_unchecked(
                    request.output,
                    relationships=(
                        _unchecked(
                            request.output.relationships[0], status="unsafe"
                        ),
                    ),
                ),
            ),
            "output_safety_invalid",
        ),
    ],
)
def test_cross_section_refusals(mutation, expected_detail: str):
    result = _validate(mutation(_request()))
    assert result.accepted is False
    assert result.blocking_issues[0].detail_code == expected_detail


def _install_no_io_guards(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("validator I/O is prohibited")

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


def test_accepted_path_performs_no_io(monkeypatch):
    request = _request()
    identity = contracts.compute_guided_backend_validation_request_identity(
        request
    )
    _install_no_io_guards(monkeypatch)
    result = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=_contract(),
    )
    assert result.accepted is True


def test_refused_path_performs_no_io(monkeypatch):
    request = _request()
    _install_no_io_guards(monkeypatch)
    result = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity="bad",
        validator_contract=_contract(),
    )
    assert result.accepted is False


def test_validation_leaves_filesystem_snapshot_unchanged(tmp_path: Path):
    marker = tmp_path / "existing.txt"
    marker.write_text("unchanged", encoding="utf-8")
    before = tuple(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    )

    result = _validate(_request())

    after = tuple(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    )
    assert result.accepted is True
    assert after == before
