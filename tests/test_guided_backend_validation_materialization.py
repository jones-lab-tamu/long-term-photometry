from __future__ import annotations

import ast
import builtins
import os
import sys
from pathlib import Path
import pytest

from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.guided_backend_validation_request import (
    compile_guided_backend_validation_request,
    GuidedBackendValidatorContract,
    GuidedBackendValidationCompileFailure,
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


# K. Stage 2b Unresolved Inputs
def test_stage_2b_unresolved_inputs(tmp_path: Path):
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
    assert result.facts.complete_for_compilation is False

    expected_unresolved = {
        "diagnostic_cache_facts",
        "output_facts",
        "evidence_references",
    }
    assert set(result.facts.unresolved_required_inputs) == expected_unresolved


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
    monkeypatch.setattr(Path, "mkdir", raise_write_error)
    monkeypatch.setattr(os, "mkdir", raise_write_error)
    monkeypatch.setattr(os, "makedirs", raise_write_error)

    result = materialize_guided_backend_validation_facts(draft, parser_contract=parser)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)


# M. Compiler Handoff Still Refuses
def test_compiler_handoff_refuses_stage_2b(tmp_path: Path):
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
    assert isinstance(compile_result, GuidedBackendValidationCompileFailure)
    assert compile_result.status == "refused"
    assert compile_result.no_request_identity is True


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
