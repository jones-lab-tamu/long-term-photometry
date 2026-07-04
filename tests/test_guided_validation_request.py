import os
import sys
from pathlib import Path
import pytest
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisExecutionIntent,
    GuidedNewAnalysisOutputCreationPolicy,
    GuidedPlanCorrectionChoice,
    PER_ROI_PRODUCTION_STRATEGY_MAP_VERSION,
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
        ),
        applied_dff_orchestration_enabled=True
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
    assert request.applied_dff_orchestration_enabled is True


def test_validation_request_preserves_authoritative_per_roi_strategy_map(
    tmp_path: Path,
):
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path=str(tmp_path / "raw_input"),
        input_format="rwd",
        acquisition_mode="intermittent",
        included_roi_ids=["CH1", "CH2"],
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="robust_global_event_reject",
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id=roi,
                selected_strategy="robust_global_event_reject",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
                evidence_reference={
                    "evidence_source_type": "local_correction_preview",
                    "preview_id": f"preview-{roi}",
                },
            )
            for roi in ("CH1", "CH2")
        ],
        output_base_path=str(tmp_path / "output"),
    )

    request = build_guided_validation_request_from_plan(plan)

    assert request.production_strategy_map_version == (
        PER_ROI_PRODUCTION_STRATEGY_MAP_VERSION
    )
    assert [entry["roi_id"] for entry in request.per_roi_production_strategy_map] == [
        "CH1", "CH2"
    ]
    assert {
        entry["strategy_family"]
        for entry in request.per_roi_production_strategy_map
    } == {"dynamic_fit"}
    assert request.legacy_global_dynamic_fit_mode == (
        "robust_global_event_reject"
    )
    assert not {
        issue.category
        for issue in validate_guided_validation_request(request)
    } & {
        "missing_strategy_for_included_roi",
        "mixed_dynamic_fit_modes_not_enabled",
        "signal_only_f0_production_routing_not_enabled",
    }


def test_new_strategy_map_signal_only_is_truthful_and_fail_closed(tmp_path: Path):
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path=str(tmp_path / "raw_input"),
        input_format="rwd",
        acquisition_mode="intermittent",
        included_roi_ids=["CH1"],
        global_correction_strategy="signal_only_f0",
        dynamic_fit_mode=None,
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="signal_only_f0",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
                evidence_reference={
                    "evidence_source_type": "local_correction_preview",
                },
            )
        ],
        output_base_path=str(tmp_path / "output"),
    )

    request = build_guided_validation_request_from_plan(plan)
    entry = request.per_roi_production_strategy_map[0]
    issues = validate_guided_validation_request(request)

    assert entry["strategy_family"] == "signal_only_f0"
    assert entry["dynamic_fit_mode"] is None
    assert request.legacy_global_dynamic_fit_mode is None
    assert "signal_only_f0_production_routing_not_enabled" in {
        issue.category for issue in issues
    }
    assert "unsupported_correction_strategy" in {
        issue.category for issue in issues
    }

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


def test_guided_validation_result_state_4J11q():
    from photometry_pipeline.guided_validation_request import (
        GuidedValidationResultState,
        make_unvalidated_guided_validation_state,
        make_passed_guided_validation_state,
        make_failed_guided_validation_state,
        make_error_guided_validation_state,
        is_guided_validation_state_stale,
        can_guided_run_unlock,
    )
    from photometry_pipeline.guided_new_analysis_plan import GuidedPlanIssue

    # 1. default state is unvalidated
    state = make_unvalidated_guided_validation_state()
    assert state.backend_validation_status == "unvalidated"
    assert state.backend_validated_request_identity == ""

    # 2. forbidden fields are absent
    forbidden = {
        "run_dir", "run_id", "production_run_id", "output_artifact_path",
        "config_path", "command_path", "status_path", "run_report_path",
        "manifest_path", "completed_run_path", "argv", "command",
        "status_json", "run_report_json", "manifest_json"
    }
    fields = GuidedValidationResultState.__dataclass_fields__
    for f in forbidden:
        assert f not in fields

    # 3. passed state stores values but no run IDs/paths
    passed = make_passed_guided_validation_state(
        req_identity="hash123",
        result_identity="res456",
        warnings=["warning1"],
        info=["info1"],
        validator_version="v2"
    )
    assert passed.backend_validation_status == "passed"
    assert passed.backend_validated_request_identity == "hash123"
    assert passed.validation_result_identity == "res456"
    assert passed.backend_warnings == ["warning1"]
    assert passed.backend_info == ["info1"]
    assert passed.validator_version == "v2"

    # 4. failed/error states store errors and cannot unlock
    failed = make_failed_guided_validation_state("hash123", ["error1"])
    assert failed.backend_validation_status == "failed"
    assert failed.backend_errors == ["error1"]
    assert not can_guided_run_unlock(failed, "hash123", [])

    err_state = make_error_guided_validation_state("hash123", ["critical_error"])
    assert err_state.backend_validation_status == "error"
    assert err_state.backend_errors == ["critical_error"]
    assert not can_guided_run_unlock(err_state, "hash123", [])

    # 5. unvalidated state is stale
    assert is_guided_validation_state_stale(state, "hash123")

    # 6. matching passed identity is not stale, mismatched is stale
    assert not is_guided_validation_state_stale(passed, "hash123")
    assert is_guided_validation_state_stale(passed, "hash999")

    # 7. failed/error matching identity is not stale but cannot unlock
    assert not is_guided_validation_state_stale(failed, "hash123")
    assert not is_guided_validation_state_stale(err_state, "hash123")

    # 8. can_guided_run_unlock triggers
    # passed + matching + zero blocking issues + no run allocated -> True
    assert can_guided_run_unlock(passed, "hash123", [])

    # passed state with empty backend identity cannot unlock
    passed_empty_backend = make_passed_guided_validation_state(
        req_identity="",
        result_identity="res456"
    )
    assert not can_guided_run_unlock(passed_empty_backend, "hash123", [])

    # passed state with non-empty backend identity but empty current identity cannot unlock
    assert not can_guided_run_unlock(passed, "", [])

    # passed state with matching non-empty identity still can unlock Run
    assert can_guided_run_unlock(passed, "hash123", [])

    # warning-only local issues still do not block
    warn_issue = GuidedPlanIssue(category="some_warning", message="warning", severity="warning")
    assert can_guided_run_unlock(passed, "hash123", [warn_issue])

    # blocking local issues still block
    block_issue = GuidedPlanIssue(category="some_block", message="blocking issue", severity="blocking")
    assert not can_guided_run_unlock(passed, "hash123", [block_issue])

    # local checks passing without backend state -> False
    assert not can_guided_run_unlock(state, "hash123", [])

    # production run allocated -> False
    assert not can_guided_run_unlock(passed, "hash123", [], production_run_allocated=True)


def _base_request_with_strategy_map(tmp_path, entries, *, dynamic_fit_mode="robust_global_event_reject"):
    return GuidedValidationRequest(
        source_path=str(tmp_path / "raw_input"),
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        included_roi_ids=["CH1"],
        strategy_scope="global",
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode=dynamic_fit_mode,
        production_strategy_map_version=PER_ROI_PRODUCTION_STRATEGY_MAP_VERSION,
        per_roi_production_strategy_map=tuple(entries),
        legacy_global_dynamic_fit_mode=dynamic_fit_mode,
        output_base_path=str(tmp_path / "output"),
        execution_mode="phasic",
        run_profile="full",
        traces_only=False,
        output_overwrite=False,
        output_path_role="output_base",
        output_creation_timing="future_execution_start_only",
        run_directory_strategy="derive_unique_run_id_under_output_base"
    )

def test_valid_dynamic_fit_entry_passes(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "dynamic_fit",
        "dynamic_fit_mode": "robust_global_event_reject",
        "selected_strategy": "robust_global_event_reject",
        "evidence_source_type": "local_correction_preview",
        "evidence_reference": {},
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}

    assert "unsupported_production_strategy_family" not in cats
    assert "invalid_dynamic_fit_strategy_entry" not in cats
    assert "missing_or_invalid_dynamic_fit_mode" not in cats
    assert "dynamic_fit_strategy_mode_mismatch" not in cats
    assert "invalid_signal_only_f0_strategy_entry" not in cats
    assert "signal_only_f0_dynamic_fit_mode_invalid" not in cats
    assert "signal_only_f0_production_routing_not_enabled" not in cats
    assert "mixed_dynamic_fit_modes_not_enabled" not in cats
    assert "mixed_strategy_families_not_enabled" not in cats

def test_dynamic_fit_row_missing_dynamic_fit_mode_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "dynamic_fit",
        "selected_strategy": "robust_global_event_reject",
        "dynamic_fit_mode": None,
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "missing_or_invalid_dynamic_fit_mode" in cats

def test_dynamic_fit_row_unsupported_dynamic_fit_mode_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "dynamic_fit",
        "selected_strategy": "robust_global_event_reject",
        "dynamic_fit_mode": "bad_mode",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "missing_or_invalid_dynamic_fit_mode" in cats

def test_dynamic_fit_row_unsupported_selected_strategy_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "dynamic_fit",
        "selected_strategy": "bad_strategy",
        "dynamic_fit_mode": "robust_global_event_reject",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "invalid_dynamic_fit_strategy_entry" in cats

def test_dynamic_fit_row_mismatch_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "dynamic_fit",
        "selected_strategy": "robust_global_event_reject",
        "dynamic_fit_mode": "adaptive_event_gated_regression",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "dynamic_fit_strategy_mode_mismatch" in cats
    assert "mixed_dynamic_fit_modes_not_enabled" not in cats

def test_unsupported_strategy_family_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "unsupported",
        "selected_strategy": "robust_global_event_reject",
        "dynamic_fit_mode": "robust_global_event_reject",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "unsupported_production_strategy_family" in cats

def test_missing_strategy_family_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "selected_strategy": "robust_global_event_reject",
        "dynamic_fit_mode": "robust_global_event_reject",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "unsupported_production_strategy_family" in cats

def test_signal_only_f0_dynamic_fit_mode_populated_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "signal_only_f0",
        "selected_strategy": "signal_only_f0",
        "dynamic_fit_mode": "robust_global_event_reject",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "signal_only_f0_dynamic_fit_mode_invalid" in cats
    assert "signal_only_f0_production_routing_not_enabled" in cats

def test_signal_only_f0_wrong_selected_strategy_blocks(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "signal_only_f0",
        "selected_strategy": "robust_global_event_reject",
        "dynamic_fit_mode": None,
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry])
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "invalid_signal_only_f0_strategy_entry" in cats
    assert "signal_only_f0_production_routing_not_enabled" in cats

def test_unsupported_selected_strategy_blocks_even_when_legacy_valid(tmp_path: Path):
    entry = {
        "roi_id": "CH1",
        "strategy_family": "dynamic_fit",
        "selected_strategy": "bad_strategy",
        "dynamic_fit_mode": "robust_global_event_reject",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }
    request = _base_request_with_strategy_map(tmp_path, [entry], dynamic_fit_mode="robust_global_event_reject")
    issues = validate_guided_validation_request(request)
    cats = {i.category for i in issues}
    assert "invalid_dynamic_fit_strategy_entry" in cats


def test_compute_request_identity_includes_applied_dff_orchestration_enabled():
    import dataclasses
    req1 = GuidedValidationRequest(
        source_path="/path/to/source",
        source_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        timeline_anchor_mode="civil",
        applied_dff_orchestration_enabled=False
    )
    req2 = dataclasses.replace(req1, applied_dff_orchestration_enabled=True)
    assert compute_request_identity(req1) != compute_request_identity(req2)
