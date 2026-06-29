import os
import sys
from pathlib import Path
import pytest
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisExecutionIntent,
    GuidedNewAnalysisOutputCreationPolicy,
)
from photometry_pipeline.guided_validation_request import (
    GuidedValidationRequest,
    build_guided_validation_request_from_plan,
    compute_request_identity,
    validate_guided_validation_request,
)

def test_gui_dependency_is_not_imported():
    # Verify that importing guided_validation_request does not pull in any gui modules
    # Clear GUI modules from sys.modules first (if loaded) to test pure import pathway
    gui_mods = [mod for mod in sys.modules if mod.startswith("gui.")]
    for mod in gui_mods:
        del sys.modules[mod]
    
    import photometry_pipeline.guided_validation_request
    
    # Assert no gui modules have been loaded as side-effect
    gui_mods_after = [mod for mod in sys.modules if mod.startswith("gui.")]
    assert len(gui_mods_after) == 0

def test_guided_validation_request_forbidden_fields():
    # Verify that forbidden fields are absent
    forbidden = {
        "run_dir", "run_id", "production_run_id", "command", "argv",
        "config_path", "status_path", "run_report_path", "manifest_path",
        "validation_timestamp"
    }
    fields = GuidedValidationRequest.__dataclass_fields__
    for f in forbidden:
        assert f not in fields

def test_build_guided_validation_request_from_plan(tmp_path: Path):
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path=str(tmp_path / "raw_input"),
        resolved_input_source_path=str(tmp_path / "raw_input"),
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        included_roi_ids=["ROI0", "ROI1"],
        output_base_path=str(tmp_path / "output_base"),
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        execution_intent=GuidedNewAnalysisExecutionIntent(
            timeline_anchor_mode="civil",
            execution_mode="phasic",
            run_profile="full"
        ),
        output_creation_policy=GuidedNewAnalysisOutputCreationPolicy(
            path_role="output_base",
            creation_timing="future_execution_start_only",
            run_directory_strategy="derive_unique_run_id_under_output_base"
        )
    )

    request = build_guided_validation_request_from_plan(plan)
    assert request.source_path == str(tmp_path / "raw_input")
    assert request.source_format == "rwd"
    assert request.acquisition_mode == "intermittent"
    assert request.sessions_per_hour == 6
    assert request.session_duration_sec == 120.0
    assert request.exclude_incomplete_final_rwd_chunk is True
    assert request.timeline_anchor_mode == "civil"
    assert request.included_roi_ids == ["ROI0", "ROI1"]
    assert request.execution_mode == "phasic"
    assert request.run_profile == "full"
    assert request.traces_only is False
    assert request.strategy_scope == "global"
    assert request.global_correction_strategy == "dynamic_fit"
    assert request.dynamic_fit_mode == "global_linear_regression"
    assert request.output_base_path == str(tmp_path / "output_base")
    assert request.output_overwrite is False
    assert request.output_path_role == "output_base"
    assert request.output_creation_timing == "future_execution_start_only"
    assert request.run_directory_strategy == "derive_unique_run_id_under_output_base"

def test_validation_missing_output_base(tmp_path: Path):
    request = GuidedValidationRequest(
        source_path=str(tmp_path / "raw_input"),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        output_base_path=None
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "missing_output_base" for iss in issues)

def test_validation_missing_strategy(tmp_path: Path):
    request = GuidedValidationRequest(
        source_path=str(tmp_path / "raw_input"),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy=None,
        dynamic_fit_mode=None,
        output_base_path=str(tmp_path / "output")
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "unsupported_correction_strategy" for iss in issues)

def test_validation_signal_only_f0_blocks(tmp_path: Path):
    request = GuidedValidationRequest(
        source_path=str(tmp_path / "raw_input"),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="signal_only_f0",
        dynamic_fit_mode=None,
        output_base_path=str(tmp_path / "output")
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "unsupported_correction_strategy" for iss in issues)

def test_validation_unsupported_dynamic_fit_mode(tmp_path: Path):
    request = GuidedValidationRequest(
        source_path=str(tmp_path / "raw_input"),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="unsupported_mode",
        output_base_path=str(tmp_path / "output")
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "unsupported_dynamic_fit_mode" for iss in issues)

def test_validation_output_source_same_path(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    request = GuidedValidationRequest(
        source_path=str(raw),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        output_base_path=str(raw)
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "output_source_overlap" for iss in issues)

def test_validation_output_nested_in_source(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    nested_out = raw / "nested_out"
    request = GuidedValidationRequest(
        source_path=str(raw),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        output_base_path=str(nested_out)
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "output_source_overlap" for iss in issues)

def test_validation_source_nested_in_output(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    nested_raw = out / "nested_raw"
    nested_raw.mkdir()
    request = GuidedValidationRequest(
        source_path=str(nested_raw),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        output_base_path=str(out)
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "output_source_overlap" for iss in issues)

def test_validation_output_overwrite_blocks(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    request = GuidedValidationRequest(
        source_path=str(raw),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        output_base_path=str(tmp_path / "out"),
        output_overwrite=True
    )
    issues = validate_guided_validation_request(request)
    assert any(iss.category == "unsupported_overwrite" for iss in issues)

def test_validation_successful_case_passes(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    request = GuidedValidationRequest(
        source_path=str(raw),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        output_base_path=str(tmp_path / "out")
    )
    issues = validate_guided_validation_request(request)
    assert len(issues) == 0

def test_nonwriting_invariant_compile_and_validate(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path=str(raw),
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        included_roi_ids=["ROI0"],
        output_base_path=str(tmp_path / "out"),
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression"
    )
    
    def _snapshot():
        return sorted(list(tmp_path.glob("**/*")))
    
    before = _snapshot()
    req = build_guided_validation_request_from_plan(plan)
    validate_guided_validation_request(req)
    after = _snapshot()
    
    assert before == after

def test_request_identity_hashing(tmp_path: Path):
    req1 = GuidedValidationRequest(
        source_path=str(tmp_path / "raw"),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        output_base_path=str(tmp_path / "out")
    )
    req2 = GuidedValidationRequest(
        source_path=str(tmp_path / "raw"),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["ROI0"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="robust_global_event_reject",
        output_base_path=str(tmp_path / "out")
    )
    assert compute_request_identity(req1) != compute_request_identity(req2)
