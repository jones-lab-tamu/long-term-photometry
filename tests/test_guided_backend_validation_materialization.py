from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, replace
import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisDatasetContractSnapshot,
    GuidedNewAnalysisDatasetContractSourceIdentity,
    GuidedPlanCorrectionChoice,
)
from photometry_pipeline.guided_backend_validation_request import (
    compile_guided_backend_validation_request,
    GuidedBackendValidatorContract,
    GuidedBackendValidationCompileSuccess,
    GuidedBackendValidationMaterializedFacts,
)
from photometry_pipeline.guided_backend_validation_materialization import (
    materialize_guided_backend_validation_facts,
    GuidedBackendValidationMaterializationIssue,
    GuidedBackendValidationMaterializationSuccess,
    GuidedBackendValidationMaterializationFailure,
    STAGE_2B_VALID_ISSUES,
)
from photometry_pipeline.io.rwd_contract import RwdHeaderParsingContract


def _write_session(root: Path, name: str, content: bytes) -> Path:
    session = root / name
    session.mkdir()
    target = session / "fluorescence.csv"
    target.write_bytes(content)
    return target


def _create_tiny_rwd_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    _write_session(root, "2026_06_30-12_00_00", b"time,uv,sig\n0.0,1.2,3.4\n")
    return root


def _valid_parser_contract() -> RwdHeaderParsingContract:
    return RwdHeaderParsingContract(
        time_column_candidates=("Time(s)", "TimeStamp"),
        uv_suffix_candidates=("-410", "-415"),
        signal_suffix_candidates=("-470",),
    )


def _apply_valid_feature_event_profile(draft: GuidedNewAnalysisDraftPlan) -> None:
    draft.feature_event_profile_status = "applied"
    draft.feature_event_profile_id = "feature-profile-001"
    draft.feature_event_explicitly_applied = True
    draft.feature_event_values = {
        "event_signal": "dff",
        "signal_excursion_polarity": "positive",
        "peak_threshold_method": "percentile",
        "peak_threshold_percentile": 90.0,
        "peak_min_distance_sec": 1.0,
        "peak_min_prominence_k": 2.0,
        "peak_min_width_sec": 0.5,
        "peak_pre_filter": "none",
        "event_auc_baseline": "zero",
        # Inactive threshold fields explicitly provided to avoid default falling
        "peak_threshold_k": 1.0,
        "peak_threshold_abs": 0.2,
    }
    draft.feature_event_validation_issues = []
    draft.feature_event_stale_reasons = []
    _apply_valid_diagnostic_cache_and_evidence(draft)


def _apply_valid_diagnostic_cache_and_evidence(
    draft: GuidedNewAnalysisDraftPlan,
) -> Path:
    source_root = Path(draft.resolved_input_source_path or draft.input_source_path or "")
    cache_root = source_root.parent / "diagnostic_cache"
    cache_root.mkdir(exist_ok=True)
    phasic_path = cache_root / "phasic_trace_cache.h5"
    config_path = cache_root / "config_used.yaml"
    request_path = cache_root / "guided_diagnostic_cache_request.json"
    phasic_path.write_bytes(b"cache")
    config_path.write_text("event_signal: dff\n", encoding="utf-8")
    request_path.write_text("{}", encoding="utf-8")

    cache_id = "cache-001"
    source_signature = "source-signature"
    scope_signature = "scope-signature"
    build_signature = "build-signature"
    evidence_id = "preview-001"
    roi_id = "CH1"
    strategy = "global_linear_regression"
    summary = {
        "preliminary_cache": True,
        "production_analysis": False,
        "evidence_references": [
            {
                "evidence_reference_id": evidence_id,
                "diagnostic_cache_id": cache_id,
                "roi_id": roi_id,
                "selected_strategy": strategy,
            }
        ],
    }
    artifact = {
        "artifact_contract_version": "guided_diagnostic_cache.v1",
        "cache_id": cache_id,
        "purpose": "guided_diagnostic_cache",
        "production_analysis": False,
        "source_path": str(source_root),
        "source_setup_signature": source_signature,
        "build_request_signature": build_signature,
        "diagnostic_scope_signature": scope_signature,
        "cache_root_path": str(cache_root.resolve()),
        "phasic_trace_cache_path": str(phasic_path),
        "config_used_path": str(config_path),
        "request_json_path": str(request_path),
        "included_roi_ids": [roi_id],
        "excluded_roi_ids": [],
        "roi_inventory": [roi_id],
        "session_chunk_inventory_summary": summary,
    }
    provenance = {
        "schema_version": "guided_diagnostic_cache.v1",
        "purpose": "guided_diagnostic_cache",
        "preliminary_cache": True,
        "production_analysis": False,
        "build_request": {},
        "artifact": dict(artifact),
    }
    artifact_path = cache_root / "guided_diagnostic_cache_artifact.json"
    provenance_path = cache_root / "guided_diagnostic_cache_provenance.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    draft.cache_id = cache_id
    draft.cache_root_path = str(cache_root)
    draft.artifact_record_path = str(artifact_path)
    draft.provenance_path = str(provenance_path)
    draft.request_json_path = str(request_path)
    draft.phasic_trace_cache_path = str(phasic_path)
    draft.config_used_path = str(config_path)
    draft.source_setup_signature = source_signature
    draft.diagnostic_scope_signature = scope_signature
    draft.build_request_signature = build_signature
    draft.stale_or_current = "current"
    draft.stale_reasons = []
    draft.preliminary_cache = True
    draft.production_analysis = False
    draft.discovered_roi_ids = [roi_id]
    draft.included_roi_ids = [roi_id]
    draft.excluded_roi_ids = []
    draft.sessions_per_hour = 6
    draft.session_duration_sec = 120.0
    draft.correction_preview_result_id = evidence_id
    draft.correction_preview_status = "current"
    draft.correction_preview_source_cache_id = cache_id
    draft.per_roi_correction_strategy_choices = [
        GuidedPlanCorrectionChoice(
            roi_id=roi_id,
            selected_strategy=strategy,
            source_type="diagnostic_cache",
            diagnostic_cache_id=cache_id,
            diagnostic_cache_root=str(cache_root),
            source_setup_signature=source_signature,
            diagnostic_scope_signature=scope_signature,
            build_request_signature=build_signature,
            evidence_chunk=0,
            current_or_stale="current",
            explicit_user_mark=True,
        )
    ]
    draft.dataset_contract_snapshot = GuidedNewAnalysisDatasetContractSnapshot(
        status="applied",
        input_format="rwd",
        resolved_input_format="rwd",
        acquisition_mode="intermittent",
        contract_values={
            "rwd_time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
            "exclude_incomplete_final_rwd_chunk": False,
        },
        source_identity=GuidedNewAnalysisDatasetContractSourceIdentity(
            input_source_path=str(source_root),
            resolved_input_source_path=str(source_root),
            input_format="rwd",
            resolved_input_format="rwd",
            acquisition_mode="intermittent",
            sessions_per_hour=6,
            session_duration_sec=120.0,
            allow_partial_final_window=False,
            exclude_incomplete_final_rwd_chunk=False,
            discovered_roi_ids=(roi_id,),
            included_roi_ids=(roi_id,),
            source_setup_signature=source_signature,
            diagnostic_cache_contract_identity=build_signature,
        ),
        explicitly_applied=True,
    )
    draft.output_policy_status = "applied"
    draft.output_policy_path = str(source_root.parent / "planned_outputs")
    draft.output_policy_validation_issues = []
    draft.output_policy_stale_reasons = []
    draft.output_policy_explicitly_applied = True
    return cache_root


def _valid_stage2c_draft(tmp_path: Path) -> GuidedNewAnalysisDraftPlan:
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(_create_tiny_rwd_fixture(tmp_path)),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    return draft


def _cache_payloads(draft: GuidedNewAnalysisDraftPlan) -> tuple[dict, dict]:
    artifact = json.loads(Path(draft.artifact_record_path or "").read_text(encoding="utf-8"))
    provenance = json.loads(Path(draft.provenance_path or "").read_text(encoding="utf-8"))
    return artifact, provenance


def _write_cache_payloads(
    draft: GuidedNewAnalysisDraftPlan,
    artifact: object,
    provenance: object,
) -> None:
    Path(draft.artifact_record_path or "").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    Path(draft.provenance_path or "").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


# A. Import and Boundary Tests
def test_import_boundaries():
    # Verify module does not import prohibited packages using AST
    prohibited = {
        "PySide6",
        "gui.main_window",
        "gui.run_spec",
        "gui.run_report_viewer",
        "runner",
        "subprocess",
    }
    module_path = Path(__file__).parent.parent / "photometry_pipeline" / "guided_backend_validation_materialization.py"
    with open(module_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                assert name.name not in prohibited, f"Prohibited import: {name.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for prohibited_name in prohibited:
                    if node.module == prohibited_name or node.module.startswith(prohibited_name + "."):
                        pytest.fail(f"Prohibited import from module: {node.module}")


# B. Result Model Tests
def test_result_models_frozen():
    issue = GuidedBackendValidationMaterializationIssue(
        category="missing_source",
        section="source",
        message="Source path not found.",
    )
    success = GuidedBackendValidationMaterializationSuccess(facts=GuidedBackendValidationMaterializedFacts())
    failure = GuidedBackendValidationMaterializationFailure(blocking_issues=(issue,))

    with pytest.raises(Exception):
        success.facts = {}  # type: ignore

    with pytest.raises(Exception):
        failure.blocking_issues = ()  # type: ignore

    with pytest.raises(TypeError):
        GuidedBackendValidationMaterializationSuccess(facts=None)  # type: ignore


def test_failure_model_properties():
    issue = GuidedBackendValidationMaterializationIssue(
        category="missing_source",
        section="source",
        message="Source path not found.",
    )
    failure = GuidedBackendValidationMaterializationFailure(blocking_issues=(issue,))
    assert failure.status == "refused"
    assert failure.no_usable_facts is True
    assert not hasattr(failure, "request")
    assert not hasattr(failure, "request_identity")
    assert not hasattr(failure, "validation_result")
    assert not hasattr(failure, "run_artifact_fields")


def test_failure_requires_blocking_issues():
    with pytest.raises(ValueError):
        GuidedBackendValidationMaterializationFailure(blocking_issues=())


def test_unknown_issue_category_rejected():
    with pytest.raises(ValueError):
        GuidedBackendValidationMaterializationIssue(
            category="invalid_category_xyz",
            section="source",
            message="Error",
        )


# C. Parser Materialization Success
def test_parser_materialization_success(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert result.facts.complete_for_compilation is True
    assert result.facts.parser.available is True
    assert result.facts.parser.parser_contract_digest != ""
    assert result.facts.parser.unresolved_inputs == ()


# D. Parser Contract Missing Blocks
def test_parser_contract_missing(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)

    result = materialize_guided_backend_validation_facts(draft, parser_contract=None)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "parser_contract_missing"


# E. Parser Unresolved Inputs Block
def test_parser_unresolved_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)

    # Disable validation temporarily to allow construction of incomplete contract
    import photometry_pipeline.io.rwd_contract as rwd_c
    monkeypatch.setattr(rwd_c, "_validate_parsing_contract", lambda *args, **kwargs: None)

    # Incomplete parser contract
    parser = RwdHeaderParsingContract(unresolved_inputs=("time_column_candidates",))
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "parser_unresolved_inputs"


# F. No Header Inspection in Materializer
def test_no_header_inspection_in_materializer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    parser = _valid_parser_contract()

    # Monkeypatch inspect_rwd_header_contract to fail if called
    import photometry_pipeline.io.rwd_contract as rwd_c
    def fail_if_called(*args, **kwargs):
        pytest.fail("inspect_rwd_header_contract should not be called in materializer.")
    monkeypatch.setattr(rwd_c, "inspect_rwd_header_contract", fail_if_called)

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)


# G. Feature/Event Materialization Success
def test_feature_event_materialization_success(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert len(result.facts.effective_feature_event_values) > 0
    # Assert explicitly applied values are classified as explicit
    for val in result.facts.effective_feature_event_values:
        assert val.field_name != ""
        assert val.value_type != ""
        if val.field_name == "event_signal":
            assert val.source_classification == "explicit"


# H. Missing Feature/Event Values Block (Active missing / default)
def test_missing_feature_event_values_block(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    # Remove a required active field so it falls back to backend_default
    draft.feature_event_values.pop("peak_threshold_percentile")
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "unresolved_feature_event_effective_value"


# I. Feature/Event Provenance Audits
def test_feature_event_provenance_mapping(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    
    # We explicitly pop an inactive field so it falls back to backend_config_default
    draft.feature_event_values.pop("peak_threshold_k", None)
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    
    found_default = False
    for val in result.facts.effective_feature_event_values:
        # No default value should be stored as applied_profile or explicit
        if val.source_classification == "backend_default":
            found_default = True
            assert val.field_name == "peak_threshold_k"
        else:
            assert val.source_classification == "explicit"

    assert found_default is True


# J. Unapplied Feature/Event Changes Block
def test_unapplied_feature_event_changes_block(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    # Set status to invalid / stale / not explicitly applied
    draft.feature_event_explicitly_applied = False
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "feature_event_unapplied_changes"


# K. Stage 2c Unresolved Inputs
def test_stage_2c_unresolved_inputs(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert result.facts.complete_for_compilation is True

    expected_unresolved = set()
    assert set(result.facts.unresolved_required_inputs) == expected_unresolved


def test_stage_2c_materializes_cache_and_evidence_facts(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    cache = result.facts.diagnostic_cache
    assert cache.available is True
    assert cache.cache_id == draft.cache_id
    assert cache.cache_root_canonical == os.path.realpath(draft.cache_root_path or "")
    assert len(cache.artifact_semantic_digest) == 64
    assert len(cache.provenance_semantic_digest) == 64
    assert cache.completed_run_rejection_category == "guided_diagnostic_cache_ineligible"
    assert cache.resolver_status == "current"
    assert cache.source_setup_signature == draft.source_setup_signature
    assert cache.diagnostic_scope_signature == draft.diagnostic_scope_signature
    assert cache.build_request_signature == draft.build_request_signature
    assert cache.preliminary_cache is True
    assert cache.production_analysis is False

    evidence = result.facts.evidence_references
    assert evidence.complete is True
    assert len(evidence.references) == 1
    reference = evidence.references[0]
    assert reference.evidence_reference_id == draft.correction_preview_result_id
    assert reference.diagnostic_cache_id == cache.cache_id
    assert reference.source_setup_signature == cache.source_setup_signature
    assert reference.diagnostic_scope_signature == cache.diagnostic_scope_signature
    assert reference.build_request_signature == cache.build_request_signature
    assert reference.evidence_chunk == 0
    assert reference.roi_id == "CH1"
    assert reference.selected_dynamic_fit_mode == "global_linear_regression"
    assert not hasattr(reference, "evidence_summary")
    assert not hasattr(reference, "selected_at_utc")
    assert result.facts.output.available is True
    assert result.facts.unresolved_required_inputs == ()


def test_missing_diagnostic_cache_pointer_blocks(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    draft.artifact_record_path = None
    draft.cache_root_path = None

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "diagnostic_cache_missing"


def test_cache_without_completed_run_rejection_blocks(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    Path(draft.artifact_record_path or "").unlink()
    Path(draft.provenance_path or "").unlink()

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert (
        result.blocking_issues[0].category
        == "diagnostic_cache_not_completed_run_ineligible"
    )


@pytest.mark.parametrize(
    "filename,value",
    [
        ("artifact", "{bad"),
        ("provenance", "{bad"),
        ("artifact", "[]"),
        ("provenance", "[]"),
    ],
)
def test_malformed_diagnostic_cache_metadata_blocks(
    tmp_path: Path,
    filename: str,
    value: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    path = (
        Path(draft.artifact_record_path or "")
        if filename == "artifact"
        else Path(draft.provenance_path or "")
    )
    path.write_text(value, encoding="utf-8")

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "diagnostic_cache_metadata_malformed"


@pytest.mark.parametrize(
    "target,field,value,expected",
    [
        ("artifact", "artifact_contract_version", "unsupported", "diagnostic_cache_schema_unsupported"),
        ("provenance", "schema_version", "unsupported", "diagnostic_cache_schema_unsupported"),
        ("artifact", "purpose", "wrong", "diagnostic_cache_purpose_mismatch"),
        ("artifact", "production_analysis", True, "diagnostic_cache_marked_production"),
        ("provenance", "production_analysis", True, "diagnostic_cache_marked_production"),
        ("provenance", "preliminary_cache", False, "diagnostic_cache_not_preliminary"),
    ],
)
def test_cache_boundary_mismatches_block(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    artifact, provenance = _cache_payloads(draft)
    payload = artifact if target == "artifact" else provenance
    payload[field] = value
    _write_cache_payloads(draft, artifact, provenance)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("cache_id", "evidence_reference_cache_mismatch"),
        ("source_setup_signature", "diagnostic_cache_source_setup_mismatch"),
        ("diagnostic_scope_signature", "diagnostic_cache_scope_mismatch"),
        ("build_request_signature", "diagnostic_cache_build_request_mismatch"),
    ],
)
def test_artifact_provenance_identity_mismatch_blocks(
    tmp_path: Path,
    field: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    artifact, provenance = _cache_payloads(draft)
    provenance["artifact"][field] = "different"
    _write_cache_payloads(draft, artifact, provenance)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing", "missing_confirmed_strategy_mark"),
        ("duplicate", "duplicate_confirmed_strategy_mark"),
        ("stale", "stale_strategy_mark"),
        ("non_explicit", "non_explicit_strategy_mark"),
        ("signal_only", "signal_only_not_supported_for_validate"),
        ("forbidden", "forbidden_strategy_state"),
    ],
)
def test_strategy_mark_failures_block(
    tmp_path: Path,
    mutation: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    choice = draft.per_roi_correction_strategy_choices[0]
    if mutation == "missing":
        draft.per_roi_correction_strategy_choices = []
    elif mutation == "duplicate":
        draft.per_roi_correction_strategy_choices.append(choice)
    elif mutation == "stale":
        draft.per_roi_correction_strategy_choices[0] = GuidedPlanCorrectionChoice(
            **{**choice.__dict__, "current_or_stale": "stale"}
        )
    elif mutation == "non_explicit":
        draft.per_roi_correction_strategy_choices[0] = GuidedPlanCorrectionChoice(
            **{**choice.__dict__, "explicit_user_mark": False}
        )
    elif mutation == "signal_only":
        draft.per_roi_correction_strategy_choices[0] = GuidedPlanCorrectionChoice(
            **{**choice.__dict__, "selected_strategy": "signal_only_f0"}
        )
    else:
        draft.per_roi_correction_strategy_choices[0] = GuidedPlanCorrectionChoice(
            **{**choice.__dict__, "selected_strategy": "auto"}
        )

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


def test_mixed_dynamic_fit_modes_block_and_excluded_marks_are_ignored(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    base = draft.per_roi_correction_strategy_choices[0]
    draft.included_roi_ids.append("CH2")
    draft.discovered_roi_ids.extend(["CH2", "EXCLUDED"])
    draft.excluded_roi_ids.append("EXCLUDED")
    draft.per_roi_correction_strategy_choices.extend(
        [
            GuidedPlanCorrectionChoice(
                **{
                    **base.__dict__,
                    "roi_id": "CH2",
                    "selected_strategy": "robust_global_event_reject",
                }
            ),
            GuidedPlanCorrectionChoice(
                **{
                    **base.__dict__,
                    "roi_id": "EXCLUDED",
                    "selected_strategy": "signal_only_f0",
                }
            ),
        ]
    )
    artifact, provenance = _cache_payloads(draft)
    artifact["session_chunk_inventory_summary"]["evidence_references"].append(
        {
            "evidence_reference_id": draft.correction_preview_result_id,
            "diagnostic_cache_id": draft.cache_id,
            "roi_id": "CH2",
            "selected_strategy": "robust_global_event_reject",
        }
    )
    provenance["artifact"] = dict(artifact)
    _write_cache_payloads(draft, artifact, provenance)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "mixed_dynamic_fit_modes"


def test_excluded_roi_strategy_marks_are_ignored(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    base = draft.per_roi_correction_strategy_choices[0]
    draft.discovered_roi_ids.append("EXCLUDED")
    draft.excluded_roi_ids.append("EXCLUDED")
    draft.per_roi_correction_strategy_choices.append(
        GuidedPlanCorrectionChoice(
            **{
                **base.__dict__,
                "roi_id": "EXCLUDED",
                "selected_strategy": "signal_only_f0",
            }
        )
    )
    draft.dataset_contract_snapshot = replace(
        draft.dataset_contract_snapshot,
        source_identity=replace(
            draft.dataset_contract_snapshot.source_identity,
            discovered_roi_ids=("CH1", "EXCLUDED"),
            included_roi_ids=("CH1",),
        ),
    )

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert [reference.roi_id for reference in result.facts.evidence_references.references] == [
        "CH1"
    ]


@pytest.mark.parametrize("inventory_value", [None, {}, "invalid"])
def test_missing_or_non_list_evidence_inventory_blocks(
    tmp_path: Path,
    inventory_value: object,
):
    draft = _valid_stage2c_draft(tmp_path)
    artifact, provenance = _cache_payloads(draft)
    summary = artifact["session_chunk_inventory_summary"]
    if inventory_value is None:
        summary.pop("evidence_references")
    else:
        summary["evidence_references"] = inventory_value
    provenance["artifact"] = dict(artifact)
    _write_cache_payloads(draft, artifact, provenance)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert (
        result.blocking_issues[0].category
        == "evidence_reference_missing_or_stale"
    )


def test_draft_marks_alone_cannot_materialize_evidence(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    artifact, provenance = _cache_payloads(draft)
    artifact["session_chunk_inventory_summary"].pop("evidence_references")
    provenance["artifact"] = dict(artifact)
    _write_cache_payloads(draft, artifact, provenance)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert (
        result.blocking_issues[0].category
        == "evidence_reference_missing_or_stale"
    )
    assert result.no_usable_facts is True


def test_stage_2d_materializes_output_facts(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    expected_base = os.path.realpath(os.path.abspath(draft.output_policy_path or ""))

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    output = result.facts.output
    assert output.available is True
    assert output.output_base_canonical == expected_base
    assert output.output_base_path_style == "windows_drive"
    assert output.path_role == "output_base"
    assert output.future_output_owner == "runner"
    assert output.run_directory_strategy == "derive_unique_run_id_under_output_base"
    assert output.creation_timing == "future_execution_start_only"
    assert output.overwrite is False
    assert output.precreate is False
    assert output.policy_status == "applied"
    assert output.policy_current is True
    assert output.safety_classifier_version
    assert output.protected_root_context_complete is True
    assert output.blocker_categories == ()
    assert (
        output.filesystem_fact_scope
        == "read_only_path_relationships_no_writability_probe"
    )
    assert result.facts.unresolved_required_inputs == ()
    assert result.facts.complete_for_compilation is True


def test_request_ready_enriched_facts_are_complete_and_immutable(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=parser
    )

    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    facts = result.facts
    assert facts.complete_for_compilation is True
    assert facts.unresolved_required_inputs == ()
    assert facts.source_snapshot.source_root_path_style == "windows_drive"

    parser_facts = facts.parser
    assert parser_facts.available is True
    assert parser_facts.schema_name == parser.schema_name
    assert parser_facts.schema_version == parser.schema_version
    assert parser_facts.header_search_line_limit == parser.header_search_line_limit
    assert parser_facts.time_column_candidates == parser.time_column_candidates
    assert parser_facts.uv_suffix_candidates == parser.uv_suffix_candidates
    assert parser_facts.signal_suffix_candidates == parser.signal_suffix_candidates
    assert parser_facts.column_normalization_rule == parser.column_normalization_rule
    assert parser_facts.roi_name_rule == parser.roi_name_rule
    assert parser_facts.ambiguity_policy == parser.ambiguity_policy

    dataset = facts.acquisition_dataset
    assert dataset.available is True
    assert dataset.acquisition_mode == "intermittent"
    assert dataset.sessions_per_hour == 6
    assert dataset.session_duration_sec == pytest.approx(120.0)
    assert dataset.timeline_anchor_mode == "civil"
    assert dataset.fixed_daily_anchor_clock is None
    assert dataset.rwd_time_col == "Time(s)"
    assert dataset.uv_suffix == "-410"
    assert dataset.sig_suffix == "-470"
    assert dataset.dataset_source_setup_signature == facts.diagnostic_cache.source_setup_signature
    assert dataset.diagnostic_cache_contract_identity == facts.diagnostic_cache.build_request_signature

    roi = facts.roi_scope
    assert roi.available is True
    assert roi.discovered_roi_ids == ("CH1",)
    assert roi.included_roi_ids == ("CH1",)
    assert roi.excluded_roi_ids == ()

    correction = facts.correction
    assert correction.available is True
    assert correction.global_dynamic_fit_mode == "global_linear_regression"
    assert correction.dynamic_fit_parameter_values
    assert {mark.roi_id for mark in correction.confirmed_marks} == {"CH1"}
    assert all(mark.explicit_user_mark and mark.current for mark in correction.confirmed_marks)
    assert correction.production_strategy_map_version == (
        "per_roi_correction_strategy_map.v1"
    )
    assert len(correction.per_roi_production_strategy_map) == 1
    strategy_entry = correction.per_roi_production_strategy_map[0]
    assert strategy_entry.roi_id == "CH1"
    assert strategy_entry.strategy_family == "dynamic_fit"
    assert strategy_entry.dynamic_fit_mode == "global_linear_regression"

    feature = facts.feature_event
    assert feature.available is True
    assert feature.profile_id == "feature-profile-001"
    assert feature.profile_schema_version == "guided_feature_event_profile.v1"
    assert feature.profile_status == "applied"
    assert feature.current is True
    assert feature.active_fields
    assert feature.inactive_fields
    assert not hasattr(feature, "feature_event_updated_at_utc")
    assert not hasattr(feature, "evidence_summary")

    with pytest.raises(FrozenInstanceError):
        facts.roi_scope.included_roi_ids = ("changed",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        facts.correction.global_dynamic_fit_mode = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing_semantic", "dataset_semantic_value_unresolved"),
        ("stale", "dataset_facts_stale"),
        ("source_signature", "dataset_source_signature_mismatch"),
        ("missing_timing", "dataset_semantic_value_unresolved"),
    ],
)
def test_dataset_fact_enrichment_failures(
    tmp_path: Path,
    mutation: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    snapshot = draft.dataset_contract_snapshot
    if mutation == "missing_semantic":
        values = dict(snapshot.contract_values)
        values.pop("rwd_time_col")
        draft.dataset_contract_snapshot = replace(snapshot, contract_values=values)
    elif mutation == "stale":
        draft.dataset_contract_snapshot = replace(
            snapshot, status="stale", stale_reasons=("changed",)
        )
    elif mutation == "source_signature":
        draft.dataset_contract_snapshot = replace(
            snapshot,
            source_identity=replace(
                snapshot.source_identity,
                source_setup_signature="different",
            ),
        )
    else:
        draft.dataset_contract_snapshot = replace(
            snapshot,
            source_identity=replace(
                snapshot.source_identity,
                sessions_per_hour=None,
            ),
        )

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected
    assert result.no_usable_facts is True


@pytest.mark.parametrize(
    "mutation",
    ["no_discovered", "overlap", "incomplete_partition", "duplicate"],
)
def test_roi_scope_fact_enrichment_failures(tmp_path: Path, mutation: str):
    draft = _valid_stage2c_draft(tmp_path)
    if mutation == "no_discovered":
        draft.discovered_roi_ids = []
    elif mutation == "overlap":
        draft.excluded_roi_ids = ["CH1"]
    elif mutation == "incomplete_partition":
        draft.discovered_roi_ids = ["CH1", "CH2"]
    else:
        draft.discovered_roi_ids = ["CH1", "CH1"]

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category in {
        "roi_scope_missing",
        "roi_scope_invalid",
    }


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("unresolved", "dynamic_fit_parameter_unresolved"),
        ("mode_mismatch", "dynamic_fit_parameter_contract_mismatch"),
    ],
)
def test_dynamic_fit_fact_enrichment_failures(
    tmp_path: Path,
    mutation: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    if mutation == "unresolved":
        draft.dynamic_fit_parameter_contract = replace(
            draft.dynamic_fit_parameter_contract,
            unresolved_parameters=("min_slope",),
        )
    else:
        draft.dynamic_fit_parameter_contract = replace(
            draft.dynamic_fit_parameter_contract,
            dynamic_fit_mode="robust_global_event_reject",
        )

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


def test_feature_event_profile_identity_is_required_for_complete_facts(
    tmp_path: Path,
):
    draft = _valid_stage2c_draft(tmp_path)
    draft.feature_event_profile_id = None

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert (
        result.blocking_issues[0].category
        == "feature_event_profile_identity_unavailable"
    )


@pytest.mark.parametrize("existing", [False, True])
def test_output_base_existing_or_nonexistent_is_allowed_without_changes(
    tmp_path: Path,
    existing: bool,
):
    draft = _valid_stage2c_draft(tmp_path)
    output_base = Path(draft.output_policy_path or "")
    if existing:
        output_base.mkdir()
        marker = output_base / "existing.txt"
        marker.write_text("unchanged", encoding="utf-8")
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert after == before
    assert output_base.exists() is existing


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing", "output_policy_missing"),
        ("stale", "output_policy_stale"),
        ("validation_issue", "output_policy_unapplied_changes"),
        ("not_explicit", "output_policy_unapplied_changes"),
    ],
)
def test_output_policy_failures_block(
    tmp_path: Path,
    mutation: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    if mutation == "missing":
        draft.output_policy_status = "missing"
    elif mutation == "stale":
        draft.output_policy_stale_reasons = ["source changed"]
    elif mutation == "validation_issue":
        draft.output_policy_validation_issues = ["invalid"]
    else:
        draft.output_policy_explicitly_applied = False

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected
    assert result.no_usable_facts is True


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing_base", "output_base_missing"),
        ("relative_base", "output_base_invalid"),
        ("mixed_path_style", "output_base_invalid"),
        ("path_role", "output_safety_facts_unavailable"),
        ("creation_timing", "output_safety_facts_unavailable"),
        ("run_strategy", "output_safety_facts_unavailable"),
    ],
)
def test_invalid_output_base_or_ownership_policy_blocks(
    tmp_path: Path,
    mutation: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    if mutation == "missing_base":
        draft.output_policy_path = None
    elif mutation == "relative_base":
        draft.output_policy_path = "relative/output"
    elif mutation == "mixed_path_style":
        draft.output_policy_path = "/posix/output"
    elif mutation == "path_role":
        draft.output_creation_policy = replace(
            draft.output_creation_policy, path_role="run_directory"
        )
    elif mutation == "creation_timing":
        draft.output_creation_policy = replace(
            draft.output_creation_policy, creation_timing="preview"
        )
    else:
        draft.output_creation_policy = replace(
            draft.output_creation_policy, run_directory_strategy="fixed"
        )

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("overwrite", True, "output_overwrite_not_allowed"),
        ("precreate_during_preview", True, "output_precreate_not_allowed"),
        ("gui_preflight_writes_enabled", True, "output_safety_facts_unavailable"),
    ],
)
def test_output_write_intent_is_rejected(
    tmp_path: Path,
    field: str,
    value: bool,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    draft.output_creation_policy = replace(
        draft.output_creation_policy, **{field: value}
    )

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


@pytest.mark.parametrize("relationship", ["equal", "output_inside", "source_inside"])
def test_output_source_overlap_blocks(tmp_path: Path, relationship: str):
    draft = _valid_stage2c_draft(tmp_path)
    source = Path(draft.input_source_path or "")
    if relationship == "equal":
        draft.output_policy_path = str(source)
    elif relationship == "output_inside":
        draft.output_policy_path = str(source / "outputs")
    else:
        draft.output_policy_path = str(source.parent)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "output_overlaps_source"


@pytest.mark.parametrize("relationship", ["equal", "output_inside", "cache_inside"])
def test_output_diagnostic_cache_overlap_blocks(
    tmp_path: Path,
    relationship: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    cache = Path(draft.cache_root_path or "")
    additional_roots: tuple[tuple[str, str], ...] = ()
    if relationship == "equal":
        draft.output_policy_path = str(cache)
    elif relationship == "output_inside":
        draft.output_policy_path = str(cache / "outputs")
    else:
        containing_root = tmp_path / "cache_output_parent"
        draft.output_policy_path = str(containing_root)
        additional_roots = (("diagnostic_cache", str(containing_root / "cache")),)

    result = materialize_guided_backend_validation_facts(
        draft,
        parser_contract=_valid_parser_contract(),
        additional_protected_roots=additional_roots,
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert (
        result.blocking_issues[0].category
        == "output_overlaps_diagnostic_cache"
    )


@pytest.mark.parametrize(
    "root_kind,expected",
    [
        ("completed_run", "output_overlaps_completed_run"),
        ("legacy_output", "output_overlaps_protected_root"),
    ],
)
def test_additional_backend_neutral_protected_root_overlap_blocks(
    tmp_path: Path,
    root_kind: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    protected = tmp_path / f"{root_kind}_root"
    draft.output_policy_path = str(protected / "future")

    result = materialize_guided_backend_validation_facts(
        draft,
        parser_contract=_valid_parser_contract(),
        additional_protected_roots=((root_kind, str(protected)),),
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


def test_invalid_additional_protected_root_context_blocks(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)

    result = materialize_guided_backend_validation_facts(
        draft,
        parser_contract=_valid_parser_contract(),
        additional_protected_roots=(("", ""),),
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert (
        result.blocking_issues[0].category
        == "output_protected_root_context_incomplete"
    )


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing", "evidence_reference_missing_or_stale"),
        ("cache", "evidence_reference_cache_mismatch"),
        ("roi", "evidence_reference_roi_mismatch"),
        ("strategy", "evidence_reference_strategy_mismatch"),
    ],
)
def test_evidence_inventory_mismatches_block(
    tmp_path: Path,
    mutation: str,
    expected: str,
):
    draft = _valid_stage2c_draft(tmp_path)
    artifact, provenance = _cache_payloads(draft)
    entries = artifact["session_chunk_inventory_summary"]["evidence_references"]
    if mutation == "missing":
        entries.clear()
    elif mutation == "cache":
        entries[0]["diagnostic_cache_id"] = "different"
    elif mutation == "roi":
        entries[0]["roi_id"] = "different"
    else:
        entries[0]["selected_strategy"] = "robust_global_event_reject"
    provenance["artifact"] = dict(artifact)
    _write_cache_payloads(draft, artifact, provenance)

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == expected


# L. No-Write Guarantee (extended)
def test_no_write_guarantee_stage_2b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    parser = _valid_parser_contract()
    output_base = Path(draft.output_policy_path or "")
    assert not output_base.exists()

    original_open = builtins.open

    def mock_open(file, mode="r", *args, **kwargs):
        if any(char in mode for char in "w+ax"):
            raise IOError("Write mode open is forbidden during materialization.")
        return original_open(file, mode, *args, **kwargs)

    def raise_write_error(*args, **kwargs):
        raise IOError("Write operations are forbidden during materialization.")

    monkeypatch.setattr(builtins, "open", mock_open)
    monkeypatch.setattr(Path, "write_text", raise_write_error)
    monkeypatch.setattr(Path, "write_bytes", raise_write_error)
    monkeypatch.setattr(Path, "touch", raise_write_error)
    monkeypatch.setattr(Path, "mkdir", raise_write_error)
    monkeypatch.setattr(os, "mkdir", raise_write_error)
    monkeypatch.setattr(os, "makedirs", raise_write_error)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", raise_write_error)
    import photometry_pipeline.workflow_safety as workflow_safety
    monkeypatch.setattr(
        workflow_safety,
        "validate_output_write_safety",
        lambda *args, **kwargs: pytest.fail(
            "validate_output_write_safety must not be called by materialization."
        ),
    )

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert not output_base.exists()


# M. Compiler Handoff Populates Request With Identity Deferred
def test_compiler_handoff_succeeds_with_identity_deferred(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    _apply_valid_feature_event_profile(draft)
    parser = _valid_parser_contract()

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert result.facts.complete_for_compilation is True

    validator_contract = GuidedBackendValidatorContract(
        validation_scope="guided_rwd_intermittent_phasic_full_validate",
        validation_contract_version="guided_backend_validation_contract.v1",
        validator_capability_version="test_validator_capability.v1",
        supported_subset_rule_version="global_dynamic_fit_only.v1",
    )

    compile_result = compile_guided_backend_validation_request(
        draft,
        facts=result.facts,
        validator_contract=validator_contract,
    )
    assert isinstance(compile_result, GuidedBackendValidationCompileSuccess)
    assert compile_result.status == "compiled"
    assert compile_result.request is not None
    assert len(compile_result.canonical_request_identity) == 64
    assert compile_result.request_identity_deferred is False


def test_backend_validation_workflow_accepts_real_materialized_draft(
    tmp_path: Path,
):
    from photometry_pipeline.guided_backend_validation_workflow import (
        validate_current_guided_draft_for_backend,
    )

    draft = _valid_stage2c_draft(tmp_path)
    validator_contract = GuidedBackendValidatorContract(
        validation_scope="guided_rwd_intermittent_phasic_full_validate",
        validation_contract_version="guided_backend_validation_contract.v1",
        validator_capability_version="test_validator_capability.v1",
        supported_subset_rule_version="global_dynamic_fit_only.v1",
    )
    outcome = validate_current_guided_draft_for_backend(
        draft,
        parser_contract=_valid_parser_contract(),
        validator_contract=validator_contract,
    )

    assert outcome.status == "validator_accepted"
    assert outcome.accepted_for_backend_validation is True
    assert outcome.request_identity
    assert outcome.run_authorization is False


# Preservation: Exclusion True Refused
def test_exclusion_true_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
        exclude_incomplete_final_rwd_chunk=True,
    )
    parser = _valid_parser_contract()

    def fail_if_called(*args, **kwargs):
        pytest.fail("build_rwd_source_candidate_snapshot should not be called when exclusion is enabled.")

    import photometry_pipeline.guided_backend_validation_materialization as mat
    monkeypatch.setattr(mat, "build_rwd_source_candidate_snapshot", fail_if_called)

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "unsupported_incomplete_final_exclusion"


# Preservation: Cancellation
def test_cancellation_post_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    parser = _valid_parser_contract()

    cancelled = False

    import photometry_pipeline.guided_backend_validation_materialization as mat
    original_build = mat.build_rwd_source_candidate_snapshot

    def mock_build(*args, **kwargs):
        res = original_build(*args, **kwargs)
        nonlocal cancelled
        cancelled = True
        return res

    monkeypatch.setattr(mat, "build_rwd_source_candidate_snapshot", mock_build)

    def cancellation_check():
        return cancelled

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=parser, cancellation_check=cancellation_check
    )
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "materialization_cancelled"


def test_missing_source_blocks():
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path="C:/nonexistent_directory_path_xyz",
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category in ("missing_source", "source_snapshot_unavailable")


def test_unsupported_source_format(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="npm",
        acquisition_mode="intermittent",
    )
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "unsupported_source_format"


def test_cancellation_preflight(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    parser = _valid_parser_contract()

    def cancel_always():
        return True

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser, cancellation_check=cancel_always)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "materialization_cancelled"


def _setup_fallback_path(draft: GuidedNewAnalysisDraftPlan, tmp_path: Path):
    artifact, provenance = _cache_payloads(draft)

    if "evidence_references" in artifact["session_chunk_inventory_summary"]:
        del artifact["session_chunk_inventory_summary"]["evidence_references"]

    _write_cache_payloads(draft, artifact, provenance)

    preview_dir = (
        Path(draft.cache_root_path)
        / "_guided_workflow"
        / "previews"
        / draft.correction_preview_result_id
    )
    preview_dir.mkdir(parents=True, exist_ok=True)
    draft.correction_preview_path = str(preview_dir)

    prov_data = {
        "preview_id": draft.correction_preview_result_id,
        "diagnostic_cache": {
            "cache_id": draft.cache_id,
            "build_request_signature": draft.build_request_signature,
            "source_setup_signature": draft.source_setup_signature,
            "diagnostic_scope_signature": draft.diagnostic_scope_signature,
        }
    }
    prov_file = preview_dir / "preview_provenance.json"
    prov_file.write_text(json.dumps(prov_data), encoding="utf-8")
    return preview_dir, prov_file


def test_fallback_validation_success(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    artifact_path = Path(draft.artifact_record_path)
    artifact_bytes_before = artifact_path.read_bytes()
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert artifact_path.read_bytes() == artifact_bytes_before


def test_fallback_missing_provenance(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    prov_file.unlink()
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "cache_evidence_inventory_missing"


def test_fallback_malformed_provenance(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    prov_file.write_text("not json", encoding="utf-8")
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "cache_evidence_inventory_missing"


def test_fallback_cache_id_mismatch(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    data = json.loads(prov_file.read_text(encoding="utf-8"))
    data["diagnostic_cache"]["cache_id"] = "different-cache-id"
    prov_file.write_text(json.dumps(data), encoding="utf-8")
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_provenance_cache_id_mismatch"


def test_fallback_build_request_signature_mismatch(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    data = json.loads(prov_file.read_text(encoding="utf-8"))
    data["diagnostic_cache"]["build_request_signature"] = "different-sig"
    prov_file.write_text(json.dumps(data), encoding="utf-8")
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_provenance_build_request_mismatch"


def test_fallback_source_setup_signature_mismatch(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    data = json.loads(prov_file.read_text(encoding="utf-8"))
    data["diagnostic_cache"]["source_setup_signature"] = "different-sig"
    prov_file.write_text(json.dumps(data), encoding="utf-8")
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_provenance_source_setup_mismatch"


def test_fallback_diagnostic_scope_signature_mismatch(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    data = json.loads(prov_file.read_text(encoding="utf-8"))
    data["diagnostic_cache"]["diagnostic_scope_signature"] = "different-sig"
    prov_file.write_text(json.dumps(data), encoding="utf-8")
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_provenance_scope_mismatch"


def test_fallback_preview_id_mismatch(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    data = json.loads(prov_file.read_text(encoding="utf-8"))
    data["preview_id"] = "different-preview-id"
    prov_file.write_text(json.dumps(data), encoding="utf-8")
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_id_mismatch"


def test_fallback_preview_status_stale(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.correction_preview_status = "stale"
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "correction_preview_missing_or_stale"


def test_fallback_missing_preview_path(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.correction_preview_path = ""
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "correction_preview_path_missing"


def test_fallback_preview_path_outside_cache(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    outside_dir = tmp_path.parent / "outside_previews"
    outside_dir.mkdir(exist_ok=True)
    draft.correction_preview_path = str(outside_dir)
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_path_outside_cache"


def test_fallback_preview_path_in_cache_parent_sibling_rejected(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    sibling_dir = (
        Path(draft.cache_root_path).parent
        / "previews"
        / draft.correction_preview_result_id
    )
    sibling_dir.mkdir(parents=True)
    (sibling_dir / "preview_provenance.json").write_bytes(prov_file.read_bytes())
    draft.correction_preview_path = str(sibling_dir)
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=parser
    )
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_path_outside_cache"


def test_fallback_wrong_preview_id_directory_rejected(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _preview_dir, prov_file = _setup_fallback_path(draft, tmp_path)
    wrong_id_dir = (
        Path(draft.cache_root_path)
        / "_guided_workflow"
        / "previews"
        / "different-preview-id"
    )
    wrong_id_dir.mkdir(parents=True)
    (wrong_id_dir / "preview_provenance.json").write_bytes(
        prov_file.read_bytes()
    )
    draft.correction_preview_path = str(wrong_id_dir)
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=parser
    )
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "preview_path_outside_cache"


def test_fallback_roi_missing_confirmed_strategy_choice(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.per_roi_correction_strategy_choices = []
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "strategy_mark_missing"


def test_fallback_unconfirmed_strategy_choice(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.per_roi_correction_strategy_choices[0] = replace(
        draft.per_roi_correction_strategy_choices[0],
        explicit_user_mark=False
    )
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "strategy_mark_not_explicit"


def test_fallback_stale_strategy_choice(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.per_roi_correction_strategy_choices[0] = replace(
        draft.per_roi_correction_strategy_choices[0],
        current_or_stale="stale"
    )
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "strategy_mark_stale"


def test_fallback_wrong_unsupported_strategy(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.per_roi_correction_strategy_choices[0] = replace(
        draft.per_roi_correction_strategy_choices[0],
        selected_strategy="signal_only_f0"
    )
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "signal_only"


def test_fallback_roi_not_in_cache_inventory(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)

    # Remove ROI from inventory
    artifact, provenance = _cache_payloads(draft)
    artifact["roi_inventory"] = []
    _write_cache_payloads(draft, artifact, provenance)

    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "roi_not_in_cache_inventory"


def test_fallback_existing_artifact_evidence_references_path_passes(tmp_path: Path):
    # This path has evidence_references present in artifact and it passes as before
    draft = _valid_stage2c_draft(tmp_path)
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)


def test_fallback_correction_preview_source_cache_id_missing(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.correction_preview_source_cache_id = ""
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "correction_preview_source_cache_mismatch"


def test_fallback_correction_preview_source_cache_id_mismatch(tmp_path: Path):
    draft = _valid_stage2c_draft(tmp_path)
    _setup_fallback_path(draft, tmp_path)
    draft.correction_preview_source_cache_id = "completely-different-cache-id"
    parser = _valid_parser_contract()
    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].detail_code == "correction_preview_source_cache_mismatch"
