from __future__ import annotations

import ast
import builtins
import json
import os
import sys
from pathlib import Path
import pytest

from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedPlanCorrectionChoice,
)
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
    assert result.facts.complete_for_compilation is False

    expected_unresolved = {"output_facts"}
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
    assert result.facts.output.available is False
    assert result.facts.unresolved_required_inputs == ("output_facts",)


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
