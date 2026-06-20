"""Contract tests for the Guided new_analysis draft plan state and validation helpers."""

import pytest
from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_EXECUTION_SUBSET_NAME,
    GuidedNewAnalysisDraftPlan,
    GuidedPlanCorrectionChoice,
    GuidedPlanIssue,
    NEW_ANALYSIS_ISSUE_CATEGORY_TO_SECTION,
    RUN_PREVIEW_SCHEMA_VERSION,
    build_guided_new_analysis_run_preview,
    evaluate_guided_new_analysis_execution_subset_readiness,
    evaluate_new_analysis_plan_issues,
    evaluate_new_analysis_plan_readiness,
)


def _complete_new_analysis_plan(**overrides):
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path="C:/raw/input",
        resolved_input_source_path="C:/raw/input",
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        acquisition_structure_status="ready",
        discovered_roi_ids=["ROI1", "ROI2"],
        included_roi_ids=["ROI1"],
        excluded_roi_ids=["ROI2"],
        cache_id="cache-1",
        cache_root_path="C:/cache",
        artifact_record_path="C:/cache/guided_diagnostic_cache_artifact.json",
        request_json_path="C:/cache/guided_diagnostic_cache_request.json",
        provenance_path="C:/cache/guided_diagnostic_cache_provenance.json",
        phasic_trace_cache_path="C:/cache/phasic_trace_cache.h5",
        config_used_path="C:/cache/config_used.json",
        source_setup_signature="setup-1",
        diagnostic_scope_signature="scope-1",
        build_request_signature="build-1",
        stale_or_current="current",
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="global_linear_regression",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            )
        ],
        feature_event_profile_status="applied",
        feature_event_profile_id="feature-profile-1",
        feature_event_values={"signal_column": "dff"},
        feature_event_explicitly_applied=True,
        output_policy_status="applied",
        output_policy_path="C:/planned/output",
        output_policy_explicitly_applied=True,
    )
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def test_can_construct_incomplete_new_analysis_draft_plan():
    plan = GuidedNewAnalysisDraftPlan()
    assert plan.mode == "new_analysis"
    assert plan.execution_ready is False
    assert plan.executable is False
    assert plan.production_run_enabled is False


def test_missing_input_produces_issue():
    plan = GuidedNewAnalysisDraftPlan(input_source_path=None)
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "missing_input_source" for iss in issues)

    plan_empty = GuidedNewAnalysisDraftPlan(input_source_path="   ")
    issues_empty = evaluate_new_analysis_plan_issues(plan_empty)
    assert any(iss.category == "missing_input_source" for iss in issues_empty)


def test_no_included_rois_produces_issue():
    plan = GuidedNewAnalysisDraftPlan(
        discovered_roi_ids=["ROI1", "ROI2"],
        included_roi_ids=[]
    )
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "no_included_rois" for iss in issues)


def test_missing_diagnostic_cache_produces_issue():
    plan = GuidedNewAnalysisDraftPlan(cache_id=None)
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "missing_diagnostic_cache" for iss in issues)


def test_stale_diagnostic_cache_produces_issue():
    plan = GuidedNewAnalysisDraftPlan(
        cache_id="some_cache_id",
        stale_or_current="stale",
        stale_reasons=["ROI inclusion/exclusion changed"]
    )
    issues = evaluate_new_analysis_plan_issues(plan)
    stale_issues = [iss for iss in issues if iss.category == "stale_diagnostic_cache"]
    assert len(stale_issues) == 1
    assert "ROI inclusion/exclusion changed" in stale_issues[0].message


def test_missing_strategy_choice_for_included_roi_produces_issue():
    plan = GuidedNewAnalysisDraftPlan(
        included_roi_ids=["ROI1"],
        per_roi_correction_strategy_choices=[]
    )
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "missing_strategy_choice_for_included_roi" for iss in issues)


def test_forbidden_strategy_is_reported():
    # Forbidden strategy (e.g. auto)
    choice_forbidden = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="auto",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        current_or_stale="current"
    )
    plan_forbidden = GuidedNewAnalysisDraftPlan(
        cache_id="active_cache",
        included_roi_ids=["ROI1"],
        per_roi_correction_strategy_choices=[choice_forbidden]
    )
    issues_forbidden = evaluate_new_analysis_plan_issues(plan_forbidden)
    assert any(iss.category == "forbidden_strategy" for iss in issues_forbidden)

    # Invalid/unknown strategy
    choice_invalid = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="invalid_strat",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        current_or_stale="current"
    )
    plan_invalid = GuidedNewAnalysisDraftPlan(
        cache_id="active_cache",
        included_roi_ids=["ROI1"],
        per_roi_correction_strategy_choices=[choice_invalid]
    )
    issues_invalid = evaluate_new_analysis_plan_issues(plan_invalid)
    assert any(iss.category == "forbidden_strategy" for iss in issues_invalid)


def test_plan_reports_execution_not_implemented():
    plan = GuidedNewAnalysisDraftPlan()
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "execution_not_implemented" and iss.severity == "info" for iss in issues)


def test_readiness_complete_plan_allows_future_handoff_but_not_execution():
    readiness = evaluate_new_analysis_plan_readiness(_complete_new_analysis_plan())

    assert readiness.plan_complete_for_handoff is True
    assert readiness.execution_available is False
    assert "Final Guided Run/RunSpec is not implemented" in readiness.execution_blocked_reason
    assert not readiness.blocking_issues
    execution_section = next(section for section in readiness.sections if section.key == "execution")
    assert execution_section.status == "info"


@pytest.mark.parametrize(
    ("override", "category", "section_key"),
    [
        ({"cache_id": None}, "missing_diagnostic_cache", "diagnostic_cache"),
        ({"stale_or_current": "stale", "stale_reasons": ["ROI changed"]}, "stale_diagnostic_cache", "diagnostic_cache"),
        ({"per_roi_correction_strategy_choices": []}, "missing_strategy_choice_for_included_roi", "correction_strategies"),
        ({"feature_event_profile_status": "default_initialized", "feature_event_explicitly_applied": False}, "feature_event_profile_not_applied", "feature_event"),
        ({"feature_event_profile_status": "stale", "feature_event_stale_reasons": ["baseline changed"]}, "stale_feature_event_profile", "feature_event"),
        ({"output_policy_status": "missing", "output_policy_path": None, "output_policy_explicitly_applied": False}, "missing_output_policy", "output_policy"),
        ({"output_policy_status": "stale", "output_policy_stale_reasons": ["target appeared"]}, "stale_output_policy", "output_policy"),
    ],
)
def test_readiness_blocking_sections_prevent_future_handoff(override, category, section_key):
    readiness = evaluate_new_analysis_plan_readiness(_complete_new_analysis_plan(**override))

    assert readiness.plan_complete_for_handoff is False
    assert any(issue.category == category for issue in readiness.blocking_issues)
    section = next(section for section in readiness.sections if section.key == section_key)
    assert section.status in {"missing", "stale", "invalid", "blocked"}
    assert any(issue.category == category for issue in section.blocking_issues)


def test_readiness_warning_only_evidence_does_not_block_future_handoff():
    plan = _complete_new_analysis_plan(
        correction_preview_result_id="preview-1",
        correction_preview_path=None,
        correction_preview_source_cache_id=None,
        signal_only_f0_result_id="signal-1",
        signal_only_f0_path=None,
        signal_only_f0_source_cache_id=None,
    )
    readiness = evaluate_new_analysis_plan_readiness(plan)

    assert readiness.plan_complete_for_handoff is True
    assert not readiness.blocking_issues
    evidence_section = next(section for section in readiness.sections if section.key == "evidence_references")
    assert evidence_section.status == "warning"
    assert {issue.category for issue in evidence_section.warning_issues} == {
        "correction_preview_evidence_path_missing",
        "correction_preview_source_identity_missing",
        "signal_only_f0_evidence_path_missing",
        "signal_only_f0_source_identity_missing",
    }


def test_readiness_category_map_contains_evidence_section():
    assert NEW_ANALYSIS_ISSUE_CATEGORY_TO_SECTION["correction_preview_evidence_path_missing"] == "evidence_references"
    assert NEW_ANALYSIS_ISSUE_CATEGORY_TO_SECTION["signal_only_f0_source_identity_missing"] == "evidence_references"


def test_run_preview_returns_object_for_incomplete_plan_with_unresolved_items():
    preview = build_guided_new_analysis_run_preview(GuidedNewAnalysisDraftPlan())

    assert preview.preview_schema_version == RUN_PREVIEW_SCHEMA_VERSION
    assert preview.readiness_snapshot["plan_complete_for_handoff"] is False
    assert preview.execution_available is False
    categories = {item.category for item in preview.unresolved_items}
    assert "missing_input_source" in categories
    assert "missing_diagnostic_cache" in categories
    assert "missing_output_policy" in categories
    assert preview.provenance["no_gui_runspec"] is True
    assert preview.provenance["no_argv_generated"] is True
    assert preview.provenance["no_config_written"] is True


def test_run_preview_keeps_handoff_readiness_separate_from_execution_contract_unresolved_items():
    plan = _complete_new_analysis_plan()
    preview = build_guided_new_analysis_run_preview(plan)

    assert preview.readiness_snapshot["plan_complete_for_handoff"] is True
    assert preview.execution_available is False
    categories = {item.category for item in preview.unresolved_items}
    assert categories == {"per_roi_correction_execution_contract_unresolved"}
    assert preview.correction_strategy["global_strategy_collapsed"] is False
    assert preview.correction_strategy["per_roi_choices"][0]["selected_strategy"] == "global_linear_regression"
    assert preview.output_policy["path"] == plan.output_policy_path
    assert preview.output_policy["directory_created"] is False
    assert preview.output_policy["files_written"] is False


def test_run_preview_marks_signal_only_f0_production_routing_unresolved():
    plan = _complete_new_analysis_plan(
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="signal_only_f0",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            )
        ]
    )

    preview = build_guided_new_analysis_run_preview(plan)

    assert preview.readiness_snapshot["plan_complete_for_handoff"] is True
    categories = {item.category for item in preview.unresolved_items}
    assert "per_roi_correction_execution_contract_unresolved" in categories
    assert "signal_only_f0_production_routing_unresolved" in categories
    assert preview.correction_strategy["per_roi_choices"][0]["selected_strategy"] == "signal_only_f0"


def test_run_preview_uses_only_plan_fields_and_marks_missing_execution_fields_unresolved():
    plan = _complete_new_analysis_plan(
        input_source_path="C:/typed/raw",
        resolved_input_source_path="C:/resolved/raw",
        output_policy_path="C:/planned/future_output",
        feature_event_values={"event_signal": "delta_f", "peak_threshold_method": "mean_std"},
    )

    preview = build_guided_new_analysis_run_preview(plan)

    assert preview.source["input_source_path"] == "C:/typed/raw"
    assert preview.source["resolved_input_source_path"] == "C:/resolved/raw"
    assert preview.source["authoritative_input_source_path"] == "C:/resolved/raw"
    assert preview.feature_event["values"] == {"event_signal": "delta_f", "peak_threshold_method": "mean_std"}
    assert preview.acquisition["timeline_anchor_mode"]["status"] == "unresolved"
    assert "timeline anchor mode" in preview.acquisition["timeline_anchor_mode"]["reason"]


def test_run_preview_does_not_create_output_directory(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_output"
    plan = _complete_new_analysis_plan(output_policy_path=str(target))
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    preview = build_guided_new_analysis_run_preview(plan)

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before
    assert not target.exists()
    assert preview.output_policy["path"] == str(target)
    assert preview.provenance["no_output_directory_created"] is True


def test_run_preview_rejects_malformed_input():
    with pytest.raises(TypeError, match="GuidedNewAnalysisDraftPlan"):
        build_guided_new_analysis_run_preview(object())


def test_execution_subset_same_dynamic_strategy_preserves_planning_readiness_but_blocks_missing_fields():
    readiness = evaluate_guided_new_analysis_execution_subset_readiness(_complete_new_analysis_plan())

    assert readiness.subset_name == FIRST_EXECUTION_SUBSET_NAME
    assert readiness.planning_complete_for_handoff is True
    assert readiness.execution_available is False
    assert readiness.first_subset_executable is False
    assert readiness.allowed_dynamic_fit_strategy == "global_linear_regression"
    categories = {issue.category for issue in readiness.blocking_issues}
    assert "mixed_per_roi_strategies" not in categories
    assert "signal_only_f0_execution_not_supported" not in categories
    assert "missing_rwd_dataset_contract" in categories
    assert "missing_execution_mode" in categories
    assert "missing_run_profile" in categories
    assert "missing_output_creation_policy" in categories


def test_execution_subset_mixed_dynamic_strategies_block_subset_not_planning_readiness():
    plan = _complete_new_analysis_plan(
        included_roi_ids=["ROI1", "ROI2"],
        excluded_roi_ids=[],
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="global_linear_regression",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
            GuidedPlanCorrectionChoice(
                roi_id="ROI2",
                selected_strategy="robust_global_event_reject",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ],
    )

    planning = evaluate_new_analysis_plan_readiness(plan)
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    assert planning.plan_complete_for_handoff is True
    assert subset.planning_complete_for_handoff is True
    assert subset.first_subset_executable is False
    assert subset.allowed_dynamic_fit_strategy is None
    assert any(issue.category == "mixed_per_roi_strategies" for issue in subset.blocking_issues)


def test_execution_subset_signal_only_blocks_subset_not_planning_readiness():
    plan = _complete_new_analysis_plan(
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="signal_only_f0",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            )
        ]
    )

    planning = evaluate_new_analysis_plan_readiness(plan)
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    assert planning.plan_complete_for_handoff is True
    assert subset.planning_complete_for_handoff is True
    assert subset.first_subset_executable is False
    assert subset.allowed_dynamic_fit_strategy is None
    assert any(issue.category == "signal_only_f0_execution_not_supported" for issue in subset.blocking_issues)


def test_execution_subset_duplicate_strategy_choice_for_included_roi_blocks_subset():
    plan = _complete_new_analysis_plan(
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="global_linear_regression",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="robust_global_event_reject",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ]
    )

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    assert subset.first_subset_executable is False
    assert any(
        issue.category == "duplicate_strategy_choice_for_execution_subset"
        for issue in subset.blocking_issues
    )


def test_execution_subset_forbidden_strategy_blocks_planning_and_subset():
    plan = _complete_new_analysis_plan(
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="auto",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            )
        ]
    )

    planning = evaluate_new_analysis_plan_readiness(plan)
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    assert planning.plan_complete_for_handoff is False
    assert subset.planning_complete_for_handoff is False
    categories = {issue.category for issue in subset.blocking_issues}
    assert "incomplete_planning_readiness" in categories
    assert "forbidden_strategy_state" in categories


@pytest.mark.parametrize(
    ("input_format", "acquisition_mode", "expected_category"),
    [
        ("rwd", "intermittent", "missing_rwd_dataset_contract"),
        ("rwd", "continuous", "missing_rwd_dataset_contract"),
        ("npm", "intermittent", "missing_npm_channel_mapping"),
        ("npm", "continuous", "unsupported_npm_continuous"),
        ("custom_tabular", "intermittent", "missing_custom_tabular_column_mapping"),
        ("custom_tabular", "continuous", "missing_custom_tabular_column_mapping"),
        ("auto", "intermittent", "unsupported_auto_format_for_execution_subset"),
    ],
)
def test_execution_subset_reports_specific_format_acquisition_field_gaps(input_format, acquisition_mode, expected_category):
    plan = _complete_new_analysis_plan(
        input_format=input_format,
        acquisition_mode=acquisition_mode,
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
    )

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    assert any(issue.category == expected_category for issue in subset.blocking_issues)
    assert any(
        field.issue_category == expected_category and field.blocks_subset
        for field in subset.field_classifications
    )


def test_execution_subset_fixed_defaults_are_reported_as_provenance():
    subset = evaluate_guided_new_analysis_execution_subset_readiness(_complete_new_analysis_plan())
    fields = {field.field_name: field for field in subset.field_classifications}

    assert fields["timeline_anchor_mode"].status == "required_missing"
    assert fields["timeline_anchor_mode"].value is None
    assert fields["timeline_anchor_mode"].blocks_subset is True
    assert fields["timeline_anchor_mode"].issue_category == "missing_timeline_anchor_mode"
    assert "no executable default is verified" in fields["timeline_anchor_mode"].provenance
    assert fields["traces_only"].status == "fixed_default"
    assert fields["traces_only"].value is False
    assert fields["preview_first_n"].status == "fixed_default"
    assert fields["preview_first_n"].value is None


def test_execution_subset_is_pure_no_files_and_no_plan_mutation(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_output"
    plan = _complete_new_analysis_plan(output_policy_path=str(target))
    before_state = dict(plan.__dict__)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after_files == before_files
    assert not target.exists()
    assert dict(plan.__dict__) == before_state
    assert subset.execution_available is False


def test_execution_subset_rejects_malformed_input():
    with pytest.raises(TypeError, match="GuidedNewAnalysisDraftPlan"):
        evaluate_guided_new_analysis_execution_subset_readiness(object())


def test_feature_event_and_output_policy_missing_are_represented():
    plan = GuidedNewAnalysisDraftPlan(
        feature_event_profile_status="unavailable",
        output_policy_status="unavailable"
    )
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "missing_feature_event_profile" for iss in issues)
    assert any(iss.category == "missing_output_policy" for iss in issues)


def test_output_policy_statuses_are_represented():
    plan_selected = GuidedNewAnalysisDraftPlan(
        output_policy_status="selected",
        output_policy_path="C:/planned/out",
        output_policy_explicitly_applied=False,
    )
    issues_selected = evaluate_new_analysis_plan_issues(plan_selected)
    assert any(iss.category == "output_policy_not_applied" and iss.severity == "blocking" for iss in issues_selected)

    plan_invalid = GuidedNewAnalysisDraftPlan(
        output_policy_status="invalid",
        output_policy_validation_issues=["inside source"],
    )
    issues_invalid = evaluate_new_analysis_plan_issues(plan_invalid)
    assert any(iss.category == "invalid_output_policy" and "inside source" in iss.message for iss in issues_invalid)

    plan_stale = GuidedNewAnalysisDraftPlan(
        output_policy_status="stale",
        output_policy_path="C:/planned/out",
        output_policy_stale_reasons=["diagnostic cache root changed"],
        output_policy_explicitly_applied=True,
    )
    issues_stale = evaluate_new_analysis_plan_issues(plan_stale)
    assert any(iss.category == "stale_output_policy" and "diagnostic cache root changed" in iss.message for iss in issues_stale)

    plan_applied = GuidedNewAnalysisDraftPlan(
        output_policy_status="applied",
        output_policy_path="C:/planned/out",
        output_policy_explicitly_applied=True,
    )
    issues_applied = evaluate_new_analysis_plan_issues(plan_applied)
    assert not any(iss.category.startswith("missing_output_policy") for iss in issues_applied)
    assert not any(iss.category in {"output_policy_not_applied", "invalid_output_policy", "stale_output_policy"} for iss in issues_applied)


def test_strict_signature_matching_in_strategy_choices():
    # Test that mismatches in setup/scope/build request signatures raise blocking stale issues
    plan = GuidedNewAnalysisDraftPlan(
        cache_id="active_cache",
        cache_root_path="C:/some/path",
        source_setup_signature="setup_sig_123",
        diagnostic_scope_signature="scope_sig_456",
        build_request_signature="build_sig_789",
        included_roi_ids=["ROI1"],
    )

    # All matching - should not produce stale choice issue
    choice_ok = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="global_linear_regression",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        diagnostic_cache_root="C:/some/path",
        source_setup_signature="setup_sig_123",
        diagnostic_scope_signature="scope_sig_456",
        build_request_signature="build_sig_789",
        current_or_stale="current",
    )
    plan.per_roi_correction_strategy_choices = [choice_ok]
    issues = evaluate_new_analysis_plan_issues(plan)
    assert not any(iss.category == "stale_strategy_choice" for iss in issues)

    # Setup signature mismatch
    choice_mismatch_setup = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="global_linear_regression",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        diagnostic_cache_root="C:/some/path",
        source_setup_signature="different_setup_sig",
        diagnostic_scope_signature="scope_sig_456",
        build_request_signature="build_sig_789",
        current_or_stale="current",
    )
    plan.per_roi_correction_strategy_choices = [choice_mismatch_setup]
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "stale_strategy_choice" and "source setup signature mismatch" in iss.message for iss in issues)

    # Scope signature mismatch
    choice_mismatch_scope = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="global_linear_regression",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        diagnostic_cache_root="C:/some/path",
        source_setup_signature="setup_sig_123",
        diagnostic_scope_signature="different_scope_sig",
        build_request_signature="build_sig_789",
        current_or_stale="current",
    )
    plan.per_roi_correction_strategy_choices = [choice_mismatch_scope]
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "stale_strategy_choice" and "diagnostic scope signature mismatch" in iss.message for iss in issues)

    # Build request signature mismatch
    choice_mismatch_build = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="global_linear_regression",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        diagnostic_cache_root="C:/some/path",
        source_setup_signature="setup_sig_123",
        diagnostic_scope_signature="scope_sig_456",
        build_request_signature="different_build_sig",
        current_or_stale="current",
    )
    plan.per_roi_correction_strategy_choices = [choice_mismatch_build]
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "stale_strategy_choice" and "build request signature mismatch" in iss.message for iss in issues)


def test_path_comparison_with_normcase_normpath():
    # Cache root path comparison should use normpath and normcase
    plan = GuidedNewAnalysisDraftPlan(
        cache_id="active_cache",
        cache_root_path="c:\\Some/Path/Subdir\\",
        included_roi_ids=["ROI1"],
    )

    # Matches after normpath/normcase
    choice_ok = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="global_linear_regression",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        diagnostic_cache_root="C:/some\\path/subdir",
        current_or_stale="current",
    )
    plan.per_roi_correction_strategy_choices = [choice_ok]
    issues = evaluate_new_analysis_plan_issues(plan)
    assert not any(iss.category == "stale_strategy_choice" for iss in issues)

    # Real mismatch
    choice_mismatch = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="global_linear_regression",
        source_type="diagnostic_cache",
        diagnostic_cache_id="active_cache",
        diagnostic_cache_root="C:/some\\other/subdir",
        current_or_stale="current",
    )
    plan.per_roi_correction_strategy_choices = [choice_mismatch]
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "stale_strategy_choice" and "source cache root mismatch" in iss.message for iss in issues)


def test_completed_run_choices_rejected():
    plan = GuidedNewAnalysisDraftPlan(
        cache_id="active_cache",
        cache_root_path="C:/some/path",
        included_roi_ids=["ROI1"],
    )

    # completed_run source type is rejected
    choice_completed_run = GuidedPlanCorrectionChoice(
        roi_id="ROI1",
        selected_strategy="global_linear_regression",
        source_type="completed_run",
        diagnostic_cache_id="active_cache",
        diagnostic_cache_root="C:/some/path",
        current_or_stale="current",
    )
    plan.per_roi_correction_strategy_choices = [choice_completed_run]
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "stale_strategy_choice" and "invalid_choice_source_type" in iss.message for iss in issues)


def test_warning_issues_for_missing_paths_and_evidence_cache_identities():
    # Cache generated but missing artifact record / provenance paths
    plan = GuidedNewAnalysisDraftPlan(
        cache_id="active_cache",
        artifact_record_path=None,
        provenance_path=None,
    )
    issues = evaluate_new_analysis_plan_issues(plan)
    assert any(iss.category == "missing_diagnostic_cache_artifact_path" and iss.severity == "warning" for iss in issues)
    assert any(iss.category == "missing_diagnostic_cache_provenance_path" and iss.severity == "warning" for iss in issues)

    # Evidence paths / cache identities missing
    plan_ev = GuidedNewAnalysisDraftPlan(
        cache_id="active_cache",
        correction_preview_result_id="prev_id",
        correction_preview_path=None,
        correction_preview_source_cache_id=None,
        signal_only_f0_result_id="sig_id",
        signal_only_f0_path=None,
        signal_only_f0_source_cache_id=None,
    )
    issues_ev = evaluate_new_analysis_plan_issues(plan_ev)
    assert any(iss.category == "correction_preview_evidence_path_missing" and iss.severity == "warning" for iss in issues_ev)
    assert any(iss.category == "correction_preview_source_identity_missing" and iss.severity == "warning" for iss in issues_ev)
    assert any(iss.category == "signal_only_f0_evidence_path_missing" and iss.severity == "warning" for iss in issues_ev)
    assert any(iss.category == "signal_only_f0_source_identity_missing" and iss.severity == "warning" for iss in issues_ev)


def test_feature_event_profile_statuses_validation():
    # Case A: Applied status with no validation issues has no feature/event issues
    plan_a = GuidedNewAnalysisDraftPlan(
        feature_event_profile_status="applied",
        feature_event_validation_issues=[],
        feature_event_explicitly_applied=True,
    )
    issues_a = evaluate_new_analysis_plan_issues(plan_a)
    assert not any(iss.category.startswith("missing_feature_event") or iss.category.startswith("feature_event") or iss.category.startswith("invalid_feature_event") or iss.category.startswith("stale_feature_event") for iss in issues_a)

    # Case B: default_initialized status results in blocking feature_event_profile_not_applied
    plan_b = GuidedNewAnalysisDraftPlan(
        feature_event_profile_status="default_initialized"
    )
    issues_b = evaluate_new_analysis_plan_issues(plan_b)
    iss_b = [iss for iss in issues_b if iss.category == "feature_event_profile_not_applied"]
    assert len(iss_b) == 1
    assert iss_b[0].severity == "blocking"
    assert "explicitly applied" in iss_b[0].message

    # Case C: invalid status results in blocking invalid_feature_event_profile
    plan_c = GuidedNewAnalysisDraftPlan(
        feature_event_profile_status="invalid",
        feature_event_validation_issues=["Window size must be positive", "Invalid threshold"]
    )
    issues_c = evaluate_new_analysis_plan_issues(plan_c)
    iss_c = [iss for iss in issues_c if iss.category == "invalid_feature_event_profile"]
    assert len(iss_c) == 1
    assert iss_c[0].severity == "blocking"
    assert "Window size must be positive" in iss_c[0].message

    # Case D: stale status results in blocking stale_feature_event_profile
    plan_d = GuidedNewAnalysisDraftPlan(
        feature_event_profile_status="stale",
        feature_event_stale_reasons=["active baseline config changed", "format changed"]
    )
    issues_d = evaluate_new_analysis_plan_issues(plan_d)
    iss_d = [iss for iss in issues_d if iss.category == "stale_feature_event_profile"]
    assert len(iss_d) == 1
    assert iss_d[0].severity == "blocking"
    assert "active baseline config changed" in iss_d[0].message

    # Case E: unavailable status results in blocking missing_feature_event_profile
    plan_e = GuidedNewAnalysisDraftPlan(
        feature_event_profile_status="unavailable"
    )
    issues_e = evaluate_new_analysis_plan_issues(plan_e)
    iss_e = [iss for iss in issues_e if iss.category == "missing_feature_event_profile"]
    assert len(iss_e) == 1
    assert iss_e[0].severity == "blocking"
