from __future__ import annotations

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
)
from photometry_pipeline.guided_backend_validation_materialization import (
    materialize_guided_backend_validation_facts,
    GuidedBackendValidationMaterializationIssue,
    GuidedBackendValidationMaterializationSuccess,
    GuidedBackendValidationMaterializationFailure,
    STAGE_2A_VALID_ISSUES,
)


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


# A. Import and Boundary Tests
def test_import_boundaries():
    import ast
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
                # check both absolute and relative module namespaces
                for prohibited_name in prohibited:
                    if node.module == prohibited_name or node.module.startswith(prohibited_name + "."):
                        pytest.fail(f"Prohibited import from module: {node.module}")


# B. Result Model Tests
def test_result_models_frozen():
    from photometry_pipeline.guided_backend_validation_request import GuidedBackendValidationMaterializedFacts
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

    # Success facts must be instances of GuidedBackendValidationMaterializedFacts
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


# C. Stage 2a Success Path with Tiny RWD Fixture
def test_stage_2a_success_path(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
        exclude_incomplete_final_rwd_chunk=False,
    )

    result = materialize_guided_backend_validation_facts(draft)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)
    assert result.facts.source_snapshot.available is True
    assert result.facts.source_snapshot.source_root_canonical != ""
    assert result.facts.source_snapshot.source_candidate_content_digest != ""
    assert len(result.facts.source_snapshot.candidate_files) == 1
    assert result.facts.source_snapshot.stale is False

    assert result.facts.incomplete_final_classification.available is True
    assert result.facts.incomplete_final_classification.classification_status == "not_requested"
    assert result.facts.incomplete_final_classification.classification_digest != ""

    # Other facts remain unavailable
    assert result.facts.parser.available is False
    assert result.facts.diagnostic_cache.available is False
    assert result.facts.output.available is False
    assert result.facts.evidence_references.complete is False
    assert result.facts.complete_for_compilation is False

    expected_unresolved = {
        "parser_facts",
        "diagnostic_cache_facts",
        "output_facts",
        "evidence_references",
        "effective_feature_event_values",
    }
    assert set(result.facts.unresolved_required_inputs) == expected_unresolved


# D. Exclusion True Blocks
def test_exclusion_true_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
        exclude_incomplete_final_rwd_chunk=True,
    )

    # Monkeypatch snapshot helper to fail if called, proving it is never executed
    def fail_if_called(*args, **kwargs):
        pytest.fail("build_rwd_source_candidate_snapshot should not be called when exclusion is enabled.")

    import photometry_pipeline.guided_backend_validation_materialization as mat
    monkeypatch.setattr(mat, "build_rwd_source_candidate_snapshot", fail_if_called)

    result = materialize_guided_backend_validation_facts(draft)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "unsupported_incomplete_final_exclusion"


# E. Missing / Invalid Source Blocks
def test_missing_source_blocks():
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path="C:/nonexistent_directory_path_xyz",
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    result = materialize_guided_backend_validation_facts(draft)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category in ("missing_source", "source_snapshot_unavailable")


def test_unsupported_source_format(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="npm",
        acquisition_mode="intermittent",
    )
    result = materialize_guided_backend_validation_facts(draft)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "unsupported_source_format"


# F. Cancellation
def test_cancellation_preflight(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )

    def cancel_always():
        return True

    result = materialize_guided_backend_validation_facts(draft, cancellation_check=cancel_always)
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "materialization_cancelled"


def test_cancellation_post_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )

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
        draft, cancellation_check=cancellation_check
    )
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == "materialization_cancelled"


# G. No-Write Guarantee
def test_no_write_guarantee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )

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

    result = materialize_guided_backend_validation_facts(draft)
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)


# H. Compiler Handoff
def test_compiler_handoff_refuses(tmp_path: Path):
    source_root = _create_tiny_rwd_fixture(tmp_path)
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(source_root),
        input_format="rwd",
        acquisition_mode="intermittent",
    )
    result = materialize_guided_backend_validation_facts(draft)
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
