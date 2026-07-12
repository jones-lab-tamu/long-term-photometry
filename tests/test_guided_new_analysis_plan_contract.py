"""Contract tests for the Guided new_analysis draft plan state and validation helpers."""

import dataclasses

import pytest
import photometry_pipeline.guided_new_analysis_plan as guided_plan_model
from photometry_pipeline.config import Config
from photometry_pipeline.guided_new_analysis_plan import (
    BACKEND_DYNAMIC_FIT_DEFAULT_FIELD_MAP,
    BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
    BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE,
    DATASET_CONTRACT_SNAPSHOT_SCHEMA_VERSION,
    DYNAMIC_FIT_PARAMETER_CONTRACT_SCHEMA_VERSION,
    EXECUTION_SPEC_PREVIEW_SCHEMA_VERSION,
    EXECUTION_INTENT_SCHEMA_VERSION,
    FEATURE_EVENT_BACKEND_DEFAULT_SOURCE,
    FEATURE_EVENT_CONFIG_FIELDS,
    FEATURE_EVENT_EFFECTIVE_VALUE_FIELDS,
    FIRST_SUBSET_MAPPING_PREVIEW_SCHEMA_VERSION,
    RUNNER_REQUEST_PREVIEW_SCHEMA_VERSION,
    OUTPUT_CREATION_POLICY_SCHEMA_VERSION,
    RWD_NORMALIZATION_BACKEND_CONFIG_FIELDS,
    RWD_NORMALIZATION_REQUIRED_SNAPSHOT_FIELDS,
    FIRST_EXECUTION_SUBSET_NAME,
    GuidedFirstSubsetExecutableMappingPreview,
    GuidedRunnerRequestPreview,
    GuidedNewAnalysisDatasetContractSnapshot,
    GuidedNewAnalysisDatasetContractSourceIdentity,
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisDynamicFitParameterContract,
    GuidedNewAnalysisExecutionIntent,
    GuidedNewAnalysisOutputCreationPolicy,
    GuidedPlanCorrectionChoice,
    GuidedPlanFeatureEventChoice,
    GuidedPlanIssue,
    PER_ROI_FEATURE_EVENT_MAP_VERSION,
    PER_ROI_PRODUCTION_STRATEGY_MAP_VERSION,
    NEW_ANALYSIS_ISSUE_CATEGORY_TO_SECTION,
    RUN_PREVIEW_SCHEMA_VERSION,
    build_guided_new_analysis_execution_spec_preview,
    build_guided_per_roi_feature_event_map,
    build_guided_per_roi_production_strategy_map,
    build_per_roi_effective_feature_config_fields_for_overrides,
    build_per_roi_feature_backend_config,
    build_per_roi_feature_event_provenance,
    build_guided_feature_event_effective_values_preview,
    build_guided_first_subset_executable_mapping_preview,
    build_guided_runner_request_preview,
    build_guided_new_analysis_run_preview,
    build_guided_output_base_safety_ownership_preview,
    build_guided_rwd_dataset_contract_normalization_preview,
    classify_output_base_safety_ownership,
    canonical_feature_event_backend_defaults,
    canonical_dynamic_fit_backend_defaults,
    evaluate_guided_new_analysis_execution_subset_readiness,
    evaluate_new_analysis_plan_issues,
    evaluate_new_analysis_plan_readiness,
)
from photometry_pipeline.workflow_safety import feature_event_defaults_from_config


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
        feature_event_values={"event_signal": "dff"},
        feature_event_explicitly_applied=True,
        output_policy_status="applied",
        output_policy_path="C:/planned/output",
        output_policy_explicitly_applied=True,
    )
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def test_per_roi_production_strategy_map_unanimous_dynamic_fit_projection():
    choices = [
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
        for roi in ("CH1", "CH2", "CH3")
    ]
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2", "CH3"],
        included_roi_ids=["CH1", "CH2", "CH3"],
        excluded_roi_ids=[],
        per_roi_correction_strategy_choices=choices,
    )

    strategy_map = build_guided_per_roi_production_strategy_map(plan)

    assert strategy_map.version == PER_ROI_PRODUCTION_STRATEGY_MAP_VERSION
    assert [entry.roi_id for entry in strategy_map.entries] == [
        "CH1", "CH2", "CH3"
    ]
    assert {entry.strategy_family for entry in strategy_map.entries} == {
        "dynamic_fit"
    }
    assert {entry.dynamic_fit_mode for entry in strategy_map.entries} == {
        "robust_global_event_reject"
    }
    assert strategy_map.legacy_global_dynamic_fit_mode == (
        "robust_global_event_reject"
    )
    assert strategy_map.execution_routing_supported is True
    assert strategy_map.blocking_categories == ()


def test_per_roi_strategy_map_structurally_accepts_signal_only():
    plan = _complete_new_analysis_plan(
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="signal_only_f0",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
                evidence_reference={
                    "evidence_source_type": "local_correction_preview",
                },
            )
        ],
    )

    strategy_map = build_guided_per_roi_production_strategy_map(plan)

    assert len(strategy_map.entries) == 1
    entry = strategy_map.entries[0]
    assert entry.strategy_family == "signal_only_f0"
    assert entry.dynamic_fit_mode is None
    assert strategy_map.legacy_global_dynamic_fit_mode is None
    assert strategy_map.execution_routing_supported is True
    assert strategy_map.blocking_categories == ()


def test_per_roi_strategy_map_represents_mixed_dynamic_modes_without_collapse():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="robust_global_event_reject",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
            GuidedPlanCorrectionChoice(
                roi_id="CH2",
                selected_strategy="adaptive_event_gated_regression",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ],
    )

    strategy_map = build_guided_per_roi_production_strategy_map(plan)

    assert [entry.dynamic_fit_mode for entry in strategy_map.entries] == [
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
    ]
    assert strategy_map.legacy_global_dynamic_fit_mode is None
    assert strategy_map.execution_routing_supported is True
    assert strategy_map.blocking_categories == ()


def test_per_roi_strategy_map_allows_enabled_uniform_mixed_families():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        applied_dff_orchestration_enabled=True,
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="robust_global_event_reject",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
            GuidedPlanCorrectionChoice(
                roi_id="CH2",
                selected_strategy="signal_only_f0",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ],
    )

    strategy_map = build_guided_per_roi_production_strategy_map(plan)

    assert [entry.strategy_family for entry in strategy_map.entries] == [
        "dynamic_fit", "signal_only_f0"
    ]
    assert strategy_map.entries[1].dynamic_fit_mode is None
    assert strategy_map.legacy_global_dynamic_fit_mode == (
        "robust_global_event_reject"
    )
    assert strategy_map.execution_routing_supported is True
    assert strategy_map.blocking_categories == ()


def test_per_roi_strategy_map_enabled_all_signal_only_is_supported():
    plan = _complete_new_analysis_plan(
        applied_dff_orchestration_enabled=True,
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="signal_only_f0",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
            )
        ],
    )

    strategy_map = build_guided_per_roi_production_strategy_map(plan)

    assert strategy_map.execution_routing_supported is True
    assert strategy_map.legacy_global_dynamic_fit_mode is None
    assert strategy_map.blocking_categories == ()


def test_per_roi_feature_event_map_no_overrides_resolves_to_default():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2", "CH3"],
        included_roi_ids=["CH1", "CH2", "CH3"],
        excluded_roi_ids=[],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert feature_map.version == PER_ROI_FEATURE_EVENT_MAP_VERSION
    assert [entry.roi_id for entry in feature_map.entries] == ["CH1", "CH2", "CH3"]
    assert all(entry.source == "default" for entry in feature_map.entries)
    assert all(
        entry.feature_event_profile_id == "feature-profile-1"
        for entry in feature_map.entries
    )
    assert all(
        entry.config_fields == {"event_signal": "dff"}
        for entry in feature_map.entries
    )
    assert feature_map.resolution_supported is True
    assert feature_map.blocking_categories == ()


def test_per_roi_feature_event_map_single_roi_override_resolves_custom_others_default():
    override = GuidedPlanFeatureEventChoice(
        roi_id="CH1",
        feature_event_profile_id="custom-ch1",
        config_fields={
            "peak_threshold_method": "percentile",
            "peak_threshold_percentile": 90.0,
        },
        current_or_stale="current",
        explicit_user_mark=True,
    )
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2", "CH3"],
        included_roi_ids=["CH1", "CH2", "CH3"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[override],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert feature_map.resolution_supported is True
    assert feature_map.blocking_categories == ()
    assert [entry.roi_id for entry in feature_map.entries] == ["CH1", "CH2", "CH3"]
    by_roi = {entry.roi_id: entry for entry in feature_map.entries}
    assert by_roi["CH1"].source == "override"
    assert by_roi["CH1"].feature_event_profile_id == "custom-ch1"
    assert by_roi["CH1"].config_fields == {
        "peak_threshold_method": "percentile",
        "peak_threshold_percentile": 90.0,
    }
    assert by_roi["CH2"].source == "default"
    assert by_roi["CH2"].feature_event_profile_id == "feature-profile-1"
    assert by_roi["CH3"].source == "default"


def test_per_roi_feature_event_map_duplicate_override_for_roi_blocks():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[
            GuidedPlanFeatureEventChoice(
                roi_id="CH1",
                feature_event_profile_id="custom-a",
                config_fields={"event_signal": "delta_f"},
                current_or_stale="current",
                explicit_user_mark=True,
            ),
            GuidedPlanFeatureEventChoice(
                roi_id="CH1",
                feature_event_profile_id="custom-b",
                config_fields={"event_signal": "dff"},
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert "duplicate_feature_event_override_for_included_roi" in (
        feature_map.blocking_categories
    )
    assert feature_map.resolution_supported is False
    assert "CH1" not in {entry.roi_id for entry in feature_map.entries}


def test_per_roi_feature_event_map_stale_override_blocks():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[
            GuidedPlanFeatureEventChoice(
                roi_id="CH1",
                feature_event_profile_id="custom-ch1",
                config_fields={"event_signal": "dff"},
                current_or_stale="stale",
                explicit_user_mark=True,
            ),
        ],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert "stale_feature_event_override" in feature_map.blocking_categories
    assert feature_map.resolution_supported is False
    assert "CH1" not in {entry.roi_id for entry in feature_map.entries}


def test_per_roi_feature_event_map_non_explicit_override_blocks():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[
            GuidedPlanFeatureEventChoice(
                roi_id="CH1",
                feature_event_profile_id="custom-ch1",
                config_fields={"event_signal": "dff"},
                current_or_stale="current",
                explicit_user_mark=False,
            ),
        ],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert "non_explicit_feature_event_override" in feature_map.blocking_categories
    assert feature_map.resolution_supported is False
    assert "CH1" not in {entry.roi_id for entry in feature_map.entries}


def test_per_roi_feature_event_map_unknown_roi_override_blocks_and_is_dropped():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[
            GuidedPlanFeatureEventChoice(
                roi_id="CH99",
                feature_event_profile_id="custom-ghost",
                config_fields={"event_signal": "dff"},
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert "unknown_roi_feature_event_override" in feature_map.blocking_categories
    # Known included ROIs still resolve to default despite the unrelated
    # unknown-ROI override.
    assert [entry.roi_id for entry in feature_map.entries] == ["CH1", "CH2"]
    assert all(entry.source == "default" for entry in feature_map.entries)
    assert feature_map.resolution_supported is False


def test_per_roi_feature_event_map_excluded_roi_override_is_ignored_not_blocking():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1"],
        excluded_roi_ids=["CH2"],
        per_roi_feature_event_choices=[
            GuidedPlanFeatureEventChoice(
                roi_id="CH2",
                feature_event_profile_id="custom-excluded",
                config_fields={"event_signal": "dff"},
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert feature_map.blocking_categories == ()
    assert feature_map.resolution_supported is True
    assert [entry.roi_id for entry in feature_map.entries] == ["CH1"]
    assert feature_map.entries[0].source == "default"


def test_per_roi_feature_event_map_missing_default_blocks_rois_without_override():
    override = GuidedPlanFeatureEventChoice(
        roi_id="CH1",
        feature_event_profile_id="custom-ch1",
        config_fields={"event_signal": "dff"},
        current_or_stale="current",
        explicit_user_mark=True,
    )
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[override],
        feature_event_profile_status="missing",
        feature_event_explicitly_applied=False,
        feature_event_values={},
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert "missing_default_feature_event_profile_for_included_roi" in (
        feature_map.blocking_categories
    )
    assert feature_map.resolution_supported is False
    by_roi = {entry.roi_id: entry for entry in feature_map.entries}
    # CH1 still resolves via its own explicit, current override even though
    # the plan has no valid default profile.
    assert by_roi["CH1"].source == "override"
    assert "CH2" not in by_roi


def test_per_roi_feature_event_map_invalid_override_config_fields_blocks():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[
            GuidedPlanFeatureEventChoice(
                roi_id="CH1",
                feature_event_profile_id="custom-ch1",
                config_fields={"event_signal": "not_a_real_signal"},
                current_or_stale="current",
                explicit_user_mark=True,
            ),
        ],
    )

    feature_map = build_guided_per_roi_feature_event_map(plan)

    assert "invalid_feature_event_override_config_fields" in (
        feature_map.blocking_categories
    )
    assert feature_map.resolution_supported is False
    assert "CH1" not in {entry.roi_id for entry in feature_map.entries}


def test_backend_glue_global_only_plan_materializes_as_before():
    """No per-ROI choices: backend glue functions carry nothing extra."""
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
    )
    feature_map = build_guided_per_roi_feature_event_map(plan)
    base_config = Config()

    backend_config = build_per_roi_feature_backend_config(feature_map, base_config)
    effective_overrides = build_per_roi_effective_feature_config_fields_for_overrides(
        feature_map, base_config
    )
    provenance = build_per_roi_feature_event_provenance(feature_map, base_config)

    # No ROI needs a distinct backend Config or a feature-config file: both
    # ROIs resolve to the plan's single default profile, exactly like today.
    assert backend_config == {}
    assert effective_overrides == {}
    assert set(provenance.keys()) == {"CH1", "CH2"}
    assert provenance["CH1"]["source"] == "default"
    assert provenance["CH2"]["source"] == "default"
    assert provenance["CH1"]["feature_event_profile_id"] == "feature-profile-1"
    # Provenance must not under-report: effective_config_fields carries every
    # feature-detection field actually in effect (base_config's values, since
    # no override applies), not just whatever sparse fields the default
    # profile happened to record in override_config_fields.
    assert set(provenance["CH1"]["effective_config_fields"]) == FEATURE_EVENT_CONFIG_FIELDS
    assert (
        provenance["CH1"]["effective_config_fields"]["peak_threshold_method"]
        == base_config.peak_threshold_method
    )


def test_backend_glue_per_roi_override_carried_through_deterministically():
    override = GuidedPlanFeatureEventChoice(
        roi_id="CH1",
        feature_event_profile_id="custom-ch1",
        config_fields={
            "peak_threshold_method": "percentile",
            "peak_threshold_percentile": 90.0,
        },
        current_or_stale="current",
        explicit_user_mark=True,
    )
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[override],
    )
    feature_map = build_guided_per_roi_feature_event_map(plan)
    base_config = Config()
    base_config.peak_threshold_method = "mean_std"
    base_config.peak_min_distance_sec = 2.5  # a field CH1's override never sets

    backend_config = build_per_roi_feature_backend_config(feature_map, base_config)
    effective_overrides = build_per_roi_effective_feature_config_fields_for_overrides(
        feature_map, base_config
    )
    provenance = build_per_roi_feature_event_provenance(feature_map, base_config)

    # Only CH1 (the overridden ROI) gets its own backend Config; CH2 is
    # absent so it keeps using base_config unchanged, matching how
    # extract_features/Pipeline resolve a ROI absent from a per-ROI map.
    assert set(backend_config.keys()) == {"CH1"}
    assert backend_config["CH1"].peak_threshold_method == "percentile"
    assert backend_config["CH1"].peak_threshold_percentile == 90.0
    # Everything else on the override Config still comes from base_config,
    # including fields CH1's own override never mentioned.
    assert backend_config["CH1"].target_fs_hz == base_config.target_fs_hz
    assert backend_config["CH1"].peak_min_distance_sec == 2.5

    # The applied-dF/F feature-config-file payload must be COMPLETE, not the
    # sparse override alone: it has to carry peak_min_distance_sec=2.5 too,
    # because run_applied_dff_features.py has no other base config to fall
    # back to and would otherwise silently use the bare Config() default.
    assert effective_overrides.keys() == {"CH1"}
    ch1_effective = effective_overrides["CH1"]
    assert set(ch1_effective) == FEATURE_EVENT_CONFIG_FIELDS
    assert ch1_effective["peak_threshold_method"] == "percentile"
    assert ch1_effective["peak_threshold_percentile"] == 90.0
    assert ch1_effective["peak_min_distance_sec"] == 2.5

    assert provenance["CH1"]["source"] == "override"
    assert provenance["CH1"]["feature_event_profile_id"] == "custom-ch1"
    # override_config_fields stays sparse (exactly what the profile set)...
    assert provenance["CH1"]["override_config_fields"] == {
        "peak_threshold_method": "percentile",
        "peak_threshold_percentile": 90.0,
    }
    # ...but effective_config_fields is always complete and does not
    # under-report the settings actually used.
    assert provenance["CH1"]["effective_config_fields"]["peak_min_distance_sec"] == 2.5
    assert provenance["CH2"]["source"] == "default"


def test_backend_glue_omits_rois_blocked_by_invalid_map_does_not_silently_execute():
    """A stale/duplicate/invalid override must not silently reach backend glue."""
    stale_override = GuidedPlanFeatureEventChoice(
        roi_id="CH1",
        feature_event_profile_id="custom-ch1",
        config_fields={"peak_threshold_method": "percentile"},
        current_or_stale="stale",
        explicit_user_mark=True,
    )
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        per_roi_feature_event_choices=[stale_override],
    )
    feature_map = build_guided_per_roi_feature_event_map(plan)
    assert feature_map.resolution_supported is False
    assert "stale_feature_event_override" in feature_map.blocking_categories

    base_config = Config()
    backend_config = build_per_roi_feature_backend_config(feature_map, base_config)
    effective_overrides = build_per_roi_effective_feature_config_fields_for_overrides(
        feature_map, base_config
    )
    provenance = build_per_roi_feature_event_provenance(feature_map, base_config)

    # CH1's stale override never reaches any backend glue output: no Config
    # override, no feature-config file entry, and no provenance row, because
    # it was never added to feature_map.entries in the first place. A caller
    # must check feature_map.resolution_supported before executing; nothing
    # here silently substitutes the stale override or a default for CH1.
    assert "CH1" not in backend_config
    assert "CH1" not in effective_overrides
    assert "CH1" not in provenance
    assert set(provenance.keys()) == {"CH2"}


def test_signal_only_f0_missing_preview_evidence_blocks_without_crash():
    plan = _complete_new_analysis_plan(
        discovered_roi_ids=["CH1", "CH2"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=[],
        applied_dff_orchestration_enabled=True,
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="robust_global_event_reject",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
                evidence_reference={
                    "evidence_source_type": "local_correction_preview",
                    "preview_only": True,
                    "production_analysis": False,
                },
            ),
            GuidedPlanCorrectionChoice(
                roi_id="CH2",
                selected_strategy="signal_only_f0",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
                evidence_reference=None,
            ),
        ],
    )

    issues = evaluate_new_analysis_plan_issues(plan)

    assert "signal_only_f0_preview_evidence_missing" in {
        issue.category for issue in issues
    }
    assert evaluate_new_analysis_plan_readiness(
        plan
    ).plan_complete_for_handoff is False


def test_signal_only_f0_invalid_preview_evidence_blocks_without_crash():
    plan = _complete_new_analysis_plan(
        applied_dff_orchestration_enabled=True,
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="signal_only_f0",
                source_type="local_correction_preview",
                current_or_stale="current",
                explicit_user_mark=True,
                evidence_reference={
                    "evidence_source_type": "local_preview",
                    "preview_only": False,
                    "production_analysis": False,
                    "strategy_family": "signal_only_f0",
                    "selected_strategy": "signal_only_f0",
                    "dynamic_fit_mode": None,
                    "valid": False,
                    "current_or_stale": "current",
                },
            )
        ],
    )

    categories = {
        issue.category for issue in evaluate_new_analysis_plan_issues(plan)
    }

    assert "signal_only_f0_preview_evidence_invalid" in categories


def test_signal_only_f0_stale_preview_evidence_blocks_without_crash():
    plan = _complete_new_analysis_plan(
        applied_dff_orchestration_enabled=True,
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="ROI1",
                selected_strategy="signal_only_f0",
                source_type="local_correction_preview",
                current_or_stale="stale",
                explicit_user_mark=True,
                evidence_reference={
                    "evidence_source_type": "local_preview",
                    "preview_only": True,
                    "production_analysis": False,
                    "strategy_family": "signal_only_f0",
                    "selected_strategy": "signal_only_f0",
                    "dynamic_fit_mode": None,
                    "valid": True,
                    "current_or_stale": "stale",
                },
            )
        ],
    )

    categories = {
        issue.category for issue in evaluate_new_analysis_plan_issues(plan)
    }

    assert "signal_only_f0_preview_evidence_stale" in categories


def _current_applied_snapshot_for_plan(plan, **overrides):
    source_identity = GuidedNewAnalysisDatasetContractSourceIdentity(
        input_source_path=plan.input_source_path,
        resolved_input_source_path=plan.resolved_input_source_path,
        input_format=plan.input_format,
        resolved_input_format=plan.input_format,
        acquisition_mode=plan.acquisition_mode,
        sessions_per_hour=plan.sessions_per_hour,
        session_duration_sec=plan.session_duration_sec,
        continuous_window_sec=plan.continuous_window_sec,
        continuous_step_sec=plan.continuous_step_sec,
        allow_partial_final_window=plan.allow_partial_final_window,
        exclude_incomplete_final_rwd_chunk=plan.exclude_incomplete_final_rwd_chunk,
        discovered_roi_ids=tuple(plan.discovered_roi_ids),
        included_roi_ids=tuple(plan.included_roi_ids),
        source_setup_signature=plan.source_setup_signature,
        diagnostic_cache_contract_identity=plan.build_request_signature,
    )
    snapshot = GuidedNewAnalysisDatasetContractSnapshot(
        status="applied",
        input_format=plan.input_format,
        resolved_input_format=plan.input_format,
        acquisition_mode=plan.acquisition_mode,
        contract_values={
            "input_format": plan.input_format,
            "resolved_input_format": plan.input_format,
            "acquisition_mode": plan.acquisition_mode,
        },
        source_identity=source_identity,
        explicitly_applied=True,
        provenance={"explicit_guided_apply": True, "no_files_written": True},
    )
    return dataclasses.replace(snapshot, **overrides) if overrides else snapshot


def test_can_construct_incomplete_new_analysis_draft_plan():
    plan = GuidedNewAnalysisDraftPlan()
    assert plan.mode == "new_analysis"
    assert plan.execution_ready is False
    assert plan.executable is False
    assert plan.production_run_enabled is False
    assert plan.dataset_contract_snapshot.status == "missing"
    assert plan.dataset_contract_snapshot.schema_version == DATASET_CONTRACT_SNAPSHOT_SCHEMA_VERSION


def test_dataset_contract_snapshot_represents_reviewed_applied_planning_state():
    source_identity = GuidedNewAnalysisDatasetContractSourceIdentity(
        input_source_path="C:/raw/input",
        resolved_input_source_path="C:/raw/input",
        input_format="rwd",
        resolved_input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        exclude_incomplete_final_rwd_chunk=True,
        discovered_roi_ids=("ROI1", "ROI2"),
        included_roi_ids=("ROI1",),
        source_setup_signature="setup-1",
        config_fingerprint="dataset-config-1",
    )
    snapshot = GuidedNewAnalysisDatasetContractSnapshot(
        status="applied",
        input_format="rwd",
        resolved_input_format="rwd",
        acquisition_mode="intermittent",
        contract_values={
            "rwd_time_col": "Time",
            "uv_suffix": "_UV",
            "sig_suffix": "_Signal",
            "exclude_incomplete_final_rwd_chunk": True,
        },
        format_specific={"excluded_source_files": ["chunk_99.csv"]},
        source_identity=source_identity,
        explicitly_applied=True,
        provenance={"reviewed_inference": True, "no_files_written": True},
    )
    plan = GuidedNewAnalysisDraftPlan(dataset_contract_snapshot=snapshot)

    assert plan.dataset_contract_snapshot.current_applied is True
    assert plan.dataset_contract_snapshot.contract_values["rwd_time_col"] == "Time"
    assert plan.dataset_contract_snapshot.source_identity.config_fingerprint == "dataset-config-1"
    assert plan.dataset_contract_snapshot.provenance["reviewed_inference"] is True


def test_dataset_contract_snapshot_stale_invalid_and_unsupported_statuses_are_structural():
    stale = GuidedNewAnalysisDatasetContractSnapshot(
        status="stale",
        stale_reasons=("input source path changed",),
        explicitly_applied=True,
    )
    invalid = GuidedNewAnalysisDatasetContractSnapshot(
        status="invalid",
        validation_issues=("missing npm LED column",),
    )
    unsupported = GuidedNewAnalysisDatasetContractSnapshot(
        status="unsupported",
        input_format="npm",
        acquisition_mode="continuous",
        validation_issues=("unsupported_npm_continuous",),
    )

    assert stale.current_applied is False
    assert invalid.current_applied is False
    assert unsupported.current_applied is False
    assert stale.stale_reasons == ("input source path changed",)
    assert invalid.validation_issues == ("missing npm LED column",)
    assert unsupported.validation_issues == ("unsupported_npm_continuous",)


def test_dataset_contract_snapshot_rejects_unknown_status():
    with pytest.raises(ValueError, match="Unsupported dataset contract snapshot status"):
        GuidedNewAnalysisDatasetContractSnapshot(status="ready")


def test_execution_intent_defaults_and_provenance_are_model_only():
    intent = GuidedNewAnalysisExecutionIntent()

    assert intent.schema_version == EXECUTION_INTENT_SCHEMA_VERSION
    assert intent.timeline_anchor_mode == "civil"
    assert intent.fixed_daily_anchor_clock is None
    assert intent.execution_mode == "phasic"
    assert intent.run_profile == "full"
    assert intent.provenance["execution_mode"] == "first_subset_fixed_default_phasic_for_global_dynamic_fit_only"
    assert intent.provenance["no_runspec"] is True
    assert intent.provenance["no_argv"] is True
    assert intent.provenance["no_config_written"] is True
    assert intent.provenance["no_files_written"] is True


def test_output_creation_policy_defaults_and_provenance_are_non_writing():
    policy = GuidedNewAnalysisOutputCreationPolicy()

    assert policy.schema_version == OUTPUT_CREATION_POLICY_SCHEMA_VERSION
    assert policy.path_role == "output_base"
    assert policy.creation_timing == "future_execution_start_only"
    assert policy.run_directory_strategy == "derive_unique_run_id_under_output_base"
    assert policy.overwrite is False
    assert policy.precreate_during_preview is False
    assert policy.config_write_timing == "future_execution_or_validation_only"
    assert policy.gui_preflight_writes_enabled is False
    assert policy.provenance["no_runspec"] is True
    assert policy.provenance["no_argv"] is True
    assert policy.provenance["no_config_written"] is True
    assert policy.provenance["no_files_written"] is True


def test_output_creation_policy_can_represent_unsafe_state_for_readiness_refusal():
    policy = GuidedNewAnalysisOutputCreationPolicy(
        overwrite=True,
        precreate_during_preview=True,
        gui_preflight_writes_enabled=True,
    )

    assert policy.overwrite is True
    assert policy.precreate_during_preview is True
    assert policy.gui_preflight_writes_enabled is True


def test_dynamic_fit_parameter_contract_defaults_are_mechanically_derived_from_backend_config():
    cfg = Config()
    backend_defaults = canonical_dynamic_fit_backend_defaults()
    contract = GuidedNewAnalysisDynamicFitParameterContract()

    assert contract.schema_version == DYNAMIC_FIT_PARAMETER_CONTRACT_SCHEMA_VERSION
    assert BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE == "photometry_pipeline.config.Config"
    assert backend_defaults == {
        contract_field: getattr(cfg, config_field)
        for contract_field, config_field in BACKEND_DYNAMIC_FIT_DEFAULT_FIELD_MAP.items()
    }

    # The first subset uses an explicit model default strategy label. It is not
    # falsely claimed as a backend Config mirror.
    assert backend_defaults["dynamic_fit_mode"] == "robust_global_event_reject"
    assert contract.dynamic_fit_mode == "global_linear_regression"
    assert "not mirrored from backend Config dynamic_fit_mode" in contract.provenance["dynamic_fit_mode"]
    assert contract.provenance["backend_config_dynamic_fit_mode"] == backend_defaults["dynamic_fit_mode"]

    mirrored_contract_fields = tuple(
        field for field in BACKEND_DYNAMIC_FIT_DEFAULT_FIELD_MAP
        if field != "dynamic_fit_mode"
    )
    for field in mirrored_contract_fields:
        assert getattr(contract, field) == backend_defaults[field]
        assert contract.provenance[field] == BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE

    assert contract.unresolved_parameters == ()
    assert contract.provenance["no_runspec"] is True
    assert contract.provenance["no_argv"] is True
    assert contract.provenance["no_config_written"] is True
    assert contract.provenance["no_files_written"] is True


def test_dynamic_fit_parameter_contract_rejects_invalid_values():
    with pytest.raises(ValueError, match="Unsupported dynamic_fit_mode"):
        GuidedNewAnalysisDynamicFitParameterContract(dynamic_fit_mode="rolling_local_regression")
    with pytest.raises(ValueError, match="Unsupported dynamic_fit slope_constraint"):
        GuidedNewAnalysisDynamicFitParameterContract(slope_constraint="bad")
    with pytest.raises(ValueError, match="robust_event_reject_max_iters"):
        GuidedNewAnalysisDynamicFitParameterContract(robust_event_reject_max_iters=0)


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
    execution_issues = [
        iss
        for iss in issues
        if iss.category == "execution_not_implemented" and iss.severity == "info"
    ]
    assert execution_issues
    # Guided Run is implemented; the user-visible message must direct to the
    # Run step instead of falsely claiming Final Run is not implemented.
    assert execution_issues[0].message == (
        "This draft plan does not run analysis by itself. "
        "Use the Run step to validate and start Guided Run."
    )
    assert "not implemented" not in execution_issues[0].message


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
    assert "per_roi_correction_execution_contract_unresolved" not in categories
    assert preview.correction_strategy["global_strategy_collapsed"] is False
    assert preview.correction_strategy["per_roi_choices"][0]["selected_strategy"] == "global_linear_regression"
    assert preview.output_policy["path"] == plan.output_policy_path
    assert preview.output_policy["directory_created"] is False
    assert preview.output_policy["files_written"] is False
    assert preview.execution_intent["timeline_anchor_mode"] == "civil"
    assert preview.execution_intent["fixed_daily_anchor_clock"] is None
    assert preview.execution_intent["execution_mode"] == "phasic"
    assert preview.execution_intent["run_profile"] == "full"
    assert preview.execution_intent["execution_consumption_enabled"] is True
    assert preview.feature_event_consumption["execution_mode"] == "phasic"
    assert preview.feature_event_consumption["run_profile"] == "full"
    assert preview.feature_event_consumption["traces_only"] is False
    assert preview.feature_event_consumption["feature_event_profile_required"] is True
    assert preview.feature_event_consumption["feature_event_profile_current_applied"] is True
    assert preview.feature_event_consumption["feature_event_values_consumed"] is True
    assert preview.feature_event_consumption["feature_extraction_in_scope"] is True
    assert preview.feature_event_consumption["feature_dependent_phasic_summaries_in_scope"] is True
    assert preview.feature_event_consumption["tonic_outputs_in_scope"] is False
    assert preview.feature_event_consumption["full_both_mode_outputs_in_scope"] is False
    assert preview.feature_event_consumption["execution_consumption_enabled"] is True
    assert preview.feature_event_consumption["effective_values_preview"]["backend_config_mapping_status"] == (
        "effective_values_ready_for_future_mapping"
    )
    assert preview.feature_event_consumption["provenance"] == (
        "first subset phasic full execution preview includes phasic feature extraction "
        "and feature-dependent summaries"
    )
    assert preview.feature_event_consumption["no_runspec"] is True
    assert preview.feature_event_consumption["no_argv"] is True
    assert preview.feature_event_consumption["no_config_written"] is True
    assert preview.feature_event_consumption["no_files_written"] is True
    assert preview.output_creation_policy["path_role"] == "output_base"
    assert preview.output_creation_policy["creation_timing"] == "future_execution_start_only"
    assert preview.output_creation_policy["run_directory_strategy"] == "derive_unique_run_id_under_output_base"
    assert preview.output_creation_policy["overwrite"] is False
    assert preview.output_creation_policy["precreate_during_preview"] is False
    assert preview.output_creation_policy["config_write_timing"] == "future_execution_or_validation_only"
    assert preview.output_creation_policy["gui_preflight_writes_enabled"] is False
    assert preview.output_creation_policy["execution_consumption_enabled"] is True
    assert preview.output_creation_policy["directory_created"] is False
    assert preview.output_creation_policy["files_written"] is False


def test_run_preview_serializes_stored_dataset_contract_snapshot_only():
    snapshot = GuidedNewAnalysisDatasetContractSnapshot(
        status="applied",
        input_format="rwd",
        resolved_input_format="rwd",
        acquisition_mode="intermittent",
        contract_values={"rwd_time_col": "Time"},
        explicitly_applied=True,
        provenance={"explicit_guided_apply": True},
    )
    plan = _complete_new_analysis_plan(dataset_contract_snapshot=snapshot)

    preview = build_guided_new_analysis_run_preview(plan)

    assert preview.dataset_contract["status"] == "applied"
    assert preview.dataset_contract["current_applied"] is True
    assert preview.dataset_contract["explicitly_applied"] is True
    assert preview.dataset_contract["input_format"] == "rwd"
    assert preview.dataset_contract["contract_values"]["rwd_time_col"] == "Time"
    assert preview.dataset_contract["provenance"]["explicit_guided_apply"] is True
    assert preview.dataset_contract["provenance"]["no_runspec"] is True
    assert preview.dataset_contract["provenance"]["no_config_written"] is True
    assert preview.dataset_contract["execution_consumption_enabled"] is False


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
    assert "all_signal_only_f0_backend_validation_not_enabled" not in categories
    assert "signal_only_f0_production_routing_unresolved" not in categories
    assert preview.correction_strategy["per_roi_choices"][0]["selected_strategy"] == "signal_only_f0"


def test_run_preview_marks_mixed_dynamic_strategies_unresolved_without_generic_mapping_item():
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

    preview = build_guided_new_analysis_run_preview(plan)
    categories = {item.category for item in preview.unresolved_items}

    assert "mixed_dynamic_fit_modes_execution_not_enabled" not in categories
    assert "per_roi_correction_execution_contract_unresolved" not in categories


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
    assert preview.acquisition["timeline_anchor_mode"]["status"] == "represented"
    assert preview.acquisition["timeline_anchor_mode"]["value"] == "civil"
    assert preview.acquisition["timeline_anchor_mode"]["source"] == "GuidedNewAnalysisDraftPlan.execution_intent"


def test_run_preview_feature_event_consumption_contract_requires_current_applied_profile():
    plan = _complete_new_analysis_plan(
        feature_event_profile_status="default_initialized",
        feature_event_explicitly_applied=False,
    )

    readiness = evaluate_new_analysis_plan_readiness(plan)
    preview = build_guided_new_analysis_run_preview(plan)
    categories = {issue.category for issue in readiness.blocking_issues}

    assert "feature_event_profile_not_applied" in categories
    assert preview.feature_event_consumption["execution_mode"] == "phasic"
    assert preview.feature_event_consumption["run_profile"] == "full"
    assert preview.feature_event_consumption["traces_only"] is False
    assert preview.feature_event_consumption["feature_event_profile_required"] is True
    assert preview.feature_event_consumption["feature_event_profile_current_applied"] is False
    assert preview.feature_event_consumption["feature_event_values_consumed"] is False
    assert preview.feature_event_consumption["feature_extraction_in_scope"] is True
    assert preview.feature_event_consumption["feature_dependent_phasic_summaries_in_scope"] is True
    assert preview.feature_event_consumption["tonic_outputs_in_scope"] is False
    assert preview.feature_event_consumption["full_both_mode_outputs_in_scope"] is False
    assert preview.feature_event_consumption["execution_consumption_enabled"] is False
    assert preview.execution_available is False


@pytest.mark.parametrize(
    ("overrides", "category"),
    [
        ({"feature_event_profile_status": "missing"}, "missing_feature_event_profile"),
        (
            {
                "feature_event_profile_status": "invalid",
                "feature_event_validation_issues": ["bad threshold"],
            },
            "invalid_feature_event_profile",
        ),
        (
            {
                "feature_event_profile_status": "stale",
                "feature_event_stale_reasons": ["baseline changed"],
            },
            "stale_feature_event_profile",
        ),
        (
            {
                "feature_event_profile_status": "applied",
                "feature_event_explicitly_applied": False,
            },
            "feature_event_profile_not_applied",
        ),
    ],
)
def test_feature_event_non_current_states_still_block_first_subset_preview_contract(overrides, category):
    plan = _complete_new_analysis_plan(**overrides)

    readiness = evaluate_new_analysis_plan_readiness(plan)
    preview = build_guided_new_analysis_run_preview(plan)
    categories = {issue.category for issue in readiness.blocking_issues}

    assert category in categories
    assert preview.feature_event_consumption["feature_event_profile_required"] is True
    assert preview.feature_event_consumption["feature_event_profile_current_applied"] is False
    assert preview.feature_event_consumption["feature_event_values_consumed"] is False
    assert preview.feature_event_consumption["execution_consumption_enabled"] is False
    assert preview.execution_available is False


def _effective_values_by_field(preview: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["field_name"]): item
        for item in preview["effective_values"]
    }


def _mapping_entries_by_field(section: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["field"]): item
        for item in section["entries"]
    }


def test_feature_event_effective_values_defaults_are_mechanically_derived_from_backend_config():
    cfg = Config()
    defaults = canonical_feature_event_backend_defaults()

    assert set(FEATURE_EVENT_EFFECTIVE_VALUE_FIELDS) == set(FEATURE_EVENT_CONFIG_FIELDS)
    assert defaults == feature_event_defaults_from_config(cfg)

    preview = build_guided_feature_event_effective_values_preview(_complete_new_analysis_plan())
    assert preview["backend_default_source"] == FEATURE_EVENT_BACKEND_DEFAULT_SOURCE
    assert preview["backend_default_values"] == defaults
    assert preview["backend_config_mapping_status"] == "effective_values_ready_for_future_mapping"
    assert preview["execution_consumption_enabled"] is True
    assert preview["unresolved_fields"] == []
    assert preview["blocker_categories"] == []


def test_feature_event_effective_values_applied_values_override_backend_defaults():
    plan = _complete_new_analysis_plan(
        feature_event_values={
            "event_signal": "delta_f",
            "peak_threshold_method": "percentile",
            "peak_threshold_percentile": 91.0,
            "peak_pre_filter": "lowpass",
            "event_auc_baseline": "median",
        }
    )

    preview = build_guided_feature_event_effective_values_preview(plan)
    values = _effective_values_by_field(preview)

    assert preview["backend_config_mapping_status"] == "effective_values_ready_for_future_mapping"
    assert values["event_signal"]["effective_value"] == "delta_f"
    assert values["event_signal"]["source"] == "applied_guided_profile"
    assert values["peak_threshold_percentile"]["effective_value"] == pytest.approx(91.0)
    assert values["peak_threshold_percentile"]["active_or_inactive"] == "active"
    assert values["peak_threshold_k"]["active_or_inactive"] == "inactive_for_threshold_method"
    assert values["peak_threshold_abs"]["active_or_inactive"] == "inactive_for_threshold_method"
    assert values["peak_pre_filter"]["effective_value"] == "lowpass"
    assert values["event_auc_baseline"]["effective_value"] == "median"


def test_feature_event_effective_values_fill_missing_applied_fields_from_verified_defaults():
    plan = _complete_new_analysis_plan(feature_event_values={"event_signal": "dff"})

    preview = build_guided_feature_event_effective_values_preview(plan)
    values = _effective_values_by_field(preview)

    assert values["event_signal"]["source"] == "applied_guided_profile"
    assert values["peak_threshold_method"]["source"] == "backend_config_default"
    assert values["peak_threshold_method"]["provenance"].startswith("backend Config default mechanically derived")
    assert values["peak_pre_filter"]["source"] == "backend_config_default"
    assert values["event_auc_baseline"]["source"] == "backend_config_default"
    assert preview["unresolved_fields"] == []


def test_feature_event_effective_values_missing_unverified_default_blocks(monkeypatch):
    defaults = dict(canonical_feature_event_backend_defaults())
    defaults.pop("peak_min_width_sec")
    monkeypatch.setattr(guided_plan_model, "_CANONICAL_FEATURE_EVENT_BACKEND_DEFAULTS", defaults)
    plan = _complete_new_analysis_plan(feature_event_values={"event_signal": "dff"})

    preview = build_guided_feature_event_effective_values_preview(plan)
    spec_preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert "peak_min_width_sec" in preview["unresolved_fields"]
    assert "unresolved_feature_event_effective_values" in preview["blocker_categories"]
    assert preview["backend_config_mapping_status"] == "effective_values_unresolved"
    assert spec_preview.spec_preview_available is False
    assert "unresolved_feature_event_effective_values" in spec_preview.blocking_issue_categories


def test_feature_event_effective_values_unsupported_threshold_method_blocks_spec_preview():
    plan = _complete_new_analysis_plan(
        feature_event_values={
            "event_signal": "dff",
            "peak_threshold_method": "not_supported",
        }
    )

    preview = build_guided_new_analysis_execution_spec_preview(plan)
    effective = preview.feature_event["feature_event_effective_values"]

    assert preview.spec_preview_available is False
    assert "unsupported_threshold_method" in preview.blocking_issue_categories
    assert effective["backend_config_mapping_status"] == "unsupported_threshold_method"
    assert effective["threshold_activity"]["supported"] is False


def test_feature_event_effective_values_threshold_specific_inactive_fields_are_shown():
    plan = _complete_new_analysis_plan(
        feature_event_values={
            "event_signal": "dff",
            "peak_threshold_method": "absolute",
            "peak_threshold_abs": 0.25,
        }
    )

    preview = build_guided_feature_event_effective_values_preview(plan)
    values = _effective_values_by_field(preview)

    assert values["peak_threshold_abs"]["active_or_inactive"] == "active"
    assert values["peak_threshold_abs"]["consumed_by_first_subset"] is True
    assert values["peak_threshold_k"]["active_or_inactive"] == "inactive_for_threshold_method"
    assert values["peak_threshold_percentile"]["active_or_inactive"] == "inactive_for_threshold_method"
    assert "peak_threshold_k" in preview["inactive_fields"]
    assert "peak_threshold_percentile" in preview["inactive_fields"]


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
    assert preview.output_creation_policy["precreate_during_preview"] is False
    assert preview.output_creation_policy["gui_preflight_writes_enabled"] is False
    assert preview.output_creation_policy["directory_created"] is False
    assert preview.output_creation_policy["files_written"] is False


def test_run_preview_rejects_malformed_input():
    with pytest.raises(TypeError, match="GuidedNewAnalysisDraftPlan"):
        build_guided_new_analysis_run_preview(object())


def test_execution_subset_same_dynamic_strategy_preserves_planning_readiness_but_blocks_missing_dataset_contract():
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
    assert "missing_timeline_anchor_mode" not in categories
    assert "missing_execution_mode" not in categories
    assert "missing_run_profile" not in categories
    assert "missing_output_creation_policy" not in categories


def test_execution_subset_rwd_current_applied_snapshot_satisfies_rwd_dataset_contract_blocker():
    plan = _complete_new_analysis_plan()
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        contract_values={
            "input_format": "rwd",
            "resolved_input_format": "rwd",
            "acquisition_mode": "intermittent",
            "rwd_time_col": "Time",
            "sig_suffix": "_Signal",
            "uv_suffix": "_UV",
        },
    )

    readiness = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    fields = {field.field_name: field for field in readiness.field_classifications}
    categories = {issue.category for issue in readiness.blocking_issues}

    assert fields["dataset_contract_snapshot"].status == "present"
    assert fields["dataset_contract_snapshot"].value["current_applied"] is True
    assert fields["dataset_contract_snapshot"].blocks_subset is False
    assert fields["dataset_contract_overrides"].status == "present"
    assert fields["dataset_contract_overrides"].blocks_subset is False
    preview = build_guided_new_analysis_run_preview(plan)

    assert "missing_rwd_dataset_contract" not in categories
    assert "missing_timeline_anchor_mode" not in categories
    assert "missing_execution_mode" not in categories
    assert "missing_run_profile" not in categories
    assert "missing_output_creation_policy" not in categories
    assert preview.feature_event_consumption["execution_mode"] == "phasic"
    assert preview.feature_event_consumption["run_profile"] == "full"
    assert preview.feature_event_consumption["traces_only"] is False
    assert preview.feature_event_consumption["feature_event_values_consumed"] is True
    assert readiness.first_subset_executable is True
    assert readiness.execution_available is False
    assert "actual execution remains unavailable" in readiness.execution_blocked_reason


def test_execution_subset_default_execution_intent_removes_intent_blockers():
    readiness = evaluate_guided_new_analysis_execution_subset_readiness(_complete_new_analysis_plan())
    fields = {field.field_name: field for field in readiness.field_classifications}
    categories = {issue.category for issue in readiness.blocking_issues}

    assert fields["timeline_anchor_mode"].status == "fixed_default"
    assert fields["timeline_anchor_mode"].value == "civil"
    assert fields["timeline_anchor_mode"].blocks_subset is False
    assert fields["mode"].status == "fixed_default"
    assert fields["mode"].value == "phasic"
    assert fields["mode"].blocks_subset is False
    assert fields["run_profile"].status == "fixed_default"
    assert fields["run_profile"].value == "full"
    assert fields["run_profile"].blocks_subset is False
    assert "missing_timeline_anchor_mode" not in categories
    assert "missing_execution_mode" not in categories
    assert "missing_run_profile" not in categories


def test_execution_subset_default_output_creation_policy_removes_output_creation_blocker():
    readiness = evaluate_guided_new_analysis_execution_subset_readiness(_complete_new_analysis_plan())
    fields = {field.field_name: field for field in readiness.field_classifications}
    categories = {issue.category for issue in readiness.blocking_issues}

    assert fields["output_creation_policy"].status == "present"
    assert fields["output_creation_policy"].value["path_role"] == "output_base"
    assert fields["output_creation_policy"].value["overwrite"] is False
    assert fields["output_creation_policy"].value["precreate_during_preview"] is False
    assert fields["output_creation_policy"].value["gui_preflight_writes_enabled"] is False
    assert fields["output_creation_policy"].blocks_subset is False
    assert "missing_output_creation_policy" not in categories


def test_execution_subset_invalid_execution_intent_blocks_subset():
    plan = _complete_new_analysis_plan(
        execution_intent=GuidedNewAnalysisExecutionIntent(
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock=None,
            execution_mode="both",
            run_profile="tuning_prep",
        )
    )

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    categories = {issue.category for issue in subset.blocking_issues}
    fields = {field.field_name: field for field in subset.field_classifications}

    assert "invalid_timeline_anchor_mode" in categories
    assert "invalid_execution_mode" in categories
    assert "unsupported_run_profile_for_first_subset" in categories
    assert fields["timeline_anchor_mode"].blocks_subset is True
    assert fields["mode"].blocks_subset is True
    assert fields["run_profile"].blocks_subset is True
    assert subset.execution_available is False


def test_execution_subset_unsafe_output_creation_policy_blocks_without_writes(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_output"
    plan = _complete_new_analysis_plan(
        output_policy_path=str(target),
        output_creation_policy=GuidedNewAnalysisOutputCreationPolicy(
            overwrite=True,
            precreate_during_preview=True,
            gui_preflight_writes_enabled=True,
        ),
    )
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    categories = {issue.category for issue in subset.blocking_issues}
    fields = {field.field_name: field for field in subset.field_classifications}

    assert after == before
    assert not target.exists()
    assert "unsafe_output_creation_policy" in categories
    assert fields["output_creation_policy"].status == "invalid"
    assert fields["output_creation_policy"].blocks_subset is True
    assert subset.execution_available is False


def test_execution_subset_rwd_missing_snapshot_still_blocks_dataset_contract():
    readiness = evaluate_guided_new_analysis_execution_subset_readiness(_complete_new_analysis_plan())
    categories = {issue.category for issue in readiness.blocking_issues}
    fields = {field.field_name: field for field in readiness.field_classifications}

    assert "missing_rwd_dataset_contract" in categories
    assert fields["dataset_contract_snapshot"].status == "required_missing"
    assert fields["dataset_contract_overrides"].status == "required_missing"


def test_execution_subset_rwd_stale_or_inconsistent_snapshot_blocks_dataset_contract():
    plan = _complete_new_analysis_plan()
    stale_snapshot = _current_applied_snapshot_for_plan(
        plan,
        stale_reasons=("input_source_path changed",),
    )
    inconsistent_identity = dataclasses.replace(
        _current_applied_snapshot_for_plan(plan).source_identity,
        input_source_path="C:/different/input",
    )
    inconsistent_snapshot = _current_applied_snapshot_for_plan(
        plan,
        source_identity=inconsistent_identity,
    )

    stale_readiness = evaluate_guided_new_analysis_execution_subset_readiness(
        _complete_new_analysis_plan(dataset_contract_snapshot=stale_snapshot)
    )
    inconsistent_readiness = evaluate_guided_new_analysis_execution_subset_readiness(
        _complete_new_analysis_plan(dataset_contract_snapshot=inconsistent_snapshot)
    )

    stale_categories = {issue.category for issue in stale_readiness.blocking_issues}
    inconsistent_categories = {issue.category for issue in inconsistent_readiness.blocking_issues}
    stale_fields = {field.field_name: field for field in stale_readiness.field_classifications}
    inconsistent_fields = {field.field_name: field for field in inconsistent_readiness.field_classifications}

    assert "stale_dataset_contract_snapshot" in stale_categories
    assert stale_fields["dataset_contract_snapshot"].status == "stale"
    assert "inconsistent_dataset_contract_snapshot" in inconsistent_categories
    assert inconsistent_fields["dataset_contract_snapshot"].status == "stale"
    assert inconsistent_fields["dataset_contract_snapshot"].value["consistency_reasons"]


def test_execution_subset_rwd_applied_without_explicit_apply_does_not_satisfy():
    plan = _complete_new_analysis_plan()
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        explicitly_applied=False,
    )

    readiness = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    fields = {field.field_name: field for field in readiness.field_classifications}
    categories = {issue.category for issue in readiness.blocking_issues}

    assert fields["dataset_contract_snapshot"].status == "selected"
    assert fields["dataset_contract_overrides"].status == "required_missing"
    assert "missing_rwd_dataset_contract" in categories


def test_execution_subset_rwd_invalid_snapshot_does_not_satisfy():
    plan = _complete_new_analysis_plan()
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        status="invalid",
        validation_issues=("missing rwd time column",),
    )

    readiness = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    fields = {field.field_name: field for field in readiness.field_classifications}
    categories = {issue.category for issue in readiness.blocking_issues}

    assert fields["dataset_contract_snapshot"].status == "invalid"
    assert fields["dataset_contract_overrides"].status == "invalid"
    assert "invalid_dataset_contract_snapshot" in categories


@pytest.mark.parametrize(
    ("snapshot", "expected_status", "expected_provenance"),
    [
        (
            GuidedNewAnalysisDatasetContractSnapshot(status="missing"),
            "required_missing",
            "not represented as applied",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="applied",
                contract_values={"rwd_time_col": "Time"},
                explicitly_applied=True,
            ),
            "stale",
            "source identity is inconsistent",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="applied",
                validation_issues=("missing rwd time column",),
                explicitly_applied=True,
            ),
            "invalid",
            "failed structural or reviewed validation",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="applied",
                stale_reasons=("input source path changed",),
                explicitly_applied=True,
            ),
            "stale",
            "no longer current",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="applied",
                contract_values={"rwd_time_col": "Time"},
                explicitly_applied=False,
            ),
            "selected",
            "explicit apply provenance is missing",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="inferred",
                contract_values={"npm_time_axis": "system_time"},
                explicitly_applied=False,
            ),
            "selected",
            "visible/reviewable but not explicitly applied",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="stale",
                stale_reasons=("input source path changed",),
                explicitly_applied=True,
            ),
            "stale",
            "no longer current",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="unsupported",
                input_format="npm",
                acquisition_mode="continuous",
                validation_issues=("unsupported_npm_continuous",),
            ),
            "unsupported",
            "unsupported format/acquisition",
        ),
    ],
)
def test_execution_subset_classifies_dataset_contract_snapshot_structurally(
    snapshot,
    expected_status,
    expected_provenance,
):
    readiness = evaluate_guided_new_analysis_execution_subset_readiness(
        _complete_new_analysis_plan(dataset_contract_snapshot=snapshot)
    )
    field = next(field for field in readiness.field_classifications if field.field_name == "dataset_contract_snapshot")

    assert field.status == expected_status
    assert field.blocks_subset is (expected_status in {"invalid", "stale", "unsupported"})
    assert snapshot.current_applied is (snapshot.status == "applied" and snapshot.explicitly_applied and not snapshot.validation_issues and not snapshot.stale_reasons)
    assert expected_provenance in field.provenance


def test_execution_subset_classifies_consistent_current_applied_snapshot_as_present():
    plan = _complete_new_analysis_plan()
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(plan)

    readiness = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    field = next(field for field in readiness.field_classifications if field.field_name == "dataset_contract_snapshot")

    assert plan.dataset_contract_snapshot.current_applied is True
    assert field.status == "present"
    assert field.blocks_subset is False
    assert "applied GuidedNewAnalysisDraftPlan" in field.provenance


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
    assert not planning.blocking_issues
    assert subset.planning_complete_for_handoff is True
    assert subset.first_subset_executable is False
    assert subset.allowed_dynamic_fit_strategy is None
    assert not any(
        issue.category == "mixed_dynamic_fit_modes_execution_not_enabled"
        for issue in subset.blocking_issues
    )


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
    assert not any(
        issue.category == "signal_only_f0_execution_not_supported"
        for issue in subset.blocking_issues
    )
    assert not any(
        issue.category == "all_signal_only_f0_backend_validation_not_enabled"
        for issue in subset.blocking_issues
    )


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


def test_execution_subset_npm_intermittent_without_mapping_remains_blocked():
    plan = _complete_new_analysis_plan(input_format="npm", acquisition_mode="intermittent")
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(plan)

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    categories = {issue.category for issue in subset.blocking_issues}
    fields = {field.field_name: field for field in subset.field_classifications}

    assert "missing_npm_channel_mapping" in categories
    assert "missing_npm_dataset_contract" in categories
    assert fields["npm_channel_mapping"].status == "required_missing"
    assert fields["dataset_contract_overrides"].status == "required_missing"


def test_execution_subset_npm_intermittent_with_explicit_mapping_satisfies_npm_dataset_blockers():
    plan = _complete_new_analysis_plan(input_format="npm", acquisition_mode="intermittent")
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        contract_values={
            "signal_channel": "465",
            "control_channel": "405",
            "time_column": "time_sec",
        },
    )

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    categories = {issue.category for issue in subset.blocking_issues}
    fields = {field.field_name: field for field in subset.field_classifications}

    assert "missing_npm_channel_mapping" not in categories
    assert "missing_npm_dataset_contract" not in categories
    assert fields["npm_channel_mapping"].status == "present"
    assert fields["dataset_contract_overrides"].status == "present"
    assert subset.execution_available is False


def test_execution_subset_custom_tabular_without_mapping_remains_blocked():
    plan = _complete_new_analysis_plan(input_format="custom_tabular", acquisition_mode="intermittent")
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(plan)

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    categories = {issue.category for issue in subset.blocking_issues}
    fields = {field.field_name: field for field in subset.field_classifications}

    assert "missing_custom_tabular_column_mapping" in categories
    assert "missing_custom_tabular_dataset_contract" in categories
    assert fields["custom_tabular_column_mapping"].status == "required_missing"
    assert fields["dataset_contract_overrides"].status == "required_missing"


def test_execution_subset_custom_tabular_with_explicit_mapping_satisfies_custom_dataset_blockers():
    plan = _complete_new_analysis_plan(input_format="custom_tabular", acquisition_mode="intermittent")
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        contract_values={
            "signal_column": "signal",
            "control_column": "control",
            "time_column": "time_sec",
            "roi_column": "roi",
        },
    )

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    categories = {issue.category for issue in subset.blocking_issues}
    fields = {field.field_name: field for field in subset.field_classifications}

    assert "missing_custom_tabular_column_mapping" not in categories
    assert "missing_custom_tabular_dataset_contract" not in categories
    assert fields["custom_tabular_column_mapping"].status == "present"
    assert fields["dataset_contract_overrides"].status == "present"
    assert subset.execution_available is False


def test_execution_subset_npm_continuous_remains_unsupported_with_current_snapshot():
    plan = _complete_new_analysis_plan(
        input_format="npm",
        acquisition_mode="continuous",
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
    )
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        contract_values={
            "signal_channel": "465",
            "control_channel": "405",
            "time_column": "time_sec",
        },
    )

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    categories = {issue.category for issue in subset.blocking_issues}

    assert "unsupported_npm_continuous" in categories
    assert subset.execution_available is False


def test_execution_subset_auto_remains_blocked_even_with_concrete_snapshot_resolution():
    plan = _complete_new_analysis_plan(input_format="auto", acquisition_mode="intermittent")
    identity = GuidedNewAnalysisDatasetContractSourceIdentity(
        input_source_path=plan.input_source_path,
        resolved_input_source_path=plan.resolved_input_source_path,
        input_format="auto",
        resolved_input_format="rwd",
        acquisition_mode=plan.acquisition_mode,
        sessions_per_hour=plan.sessions_per_hour,
        session_duration_sec=plan.session_duration_sec,
        continuous_window_sec=plan.continuous_window_sec,
        continuous_step_sec=plan.continuous_step_sec,
        allow_partial_final_window=plan.allow_partial_final_window,
        exclude_incomplete_final_rwd_chunk=plan.exclude_incomplete_final_rwd_chunk,
        included_roi_ids=tuple(plan.included_roi_ids),
    )
    plan.dataset_contract_snapshot = GuidedNewAnalysisDatasetContractSnapshot(
        status="applied",
        input_format="auto",
        resolved_input_format="rwd",
        acquisition_mode="intermittent",
        source_identity=identity,
        explicitly_applied=True,
    )

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    categories = {issue.category for issue in subset.blocking_issues}

    assert "unsupported_auto_format_for_execution_subset" in categories
    assert subset.execution_available is False


def test_execution_subset_fixed_defaults_are_reported_as_provenance():
    subset = evaluate_guided_new_analysis_execution_subset_readiness(_complete_new_analysis_plan())
    fields = {field.field_name: field for field in subset.field_classifications}

    assert fields["timeline_anchor_mode"].status == "fixed_default"
    assert fields["timeline_anchor_mode"].value == "civil"
    assert fields["timeline_anchor_mode"].blocks_subset is False
    assert fields["timeline_anchor_mode"].issue_category is None
    assert "matches backend/Full Control civil timeline anchor default" in fields["timeline_anchor_mode"].provenance
    assert fields["mode"].status == "fixed_default"
    assert fields["mode"].value == "phasic"
    assert fields["mode"].blocks_subset is False
    assert fields["run_profile"].status == "fixed_default"
    assert fields["run_profile"].value == "full"
    assert fields["run_profile"].blocks_subset is False
    assert fields["output_creation_policy"].status == "present"
    assert fields["output_creation_policy"].blocks_subset is False
    assert fields["dynamic_fit_parameter_contract"].status == "present"
    assert fields["dynamic_fit_parameter_contract"].value["dynamic_fit_mode"] == "global_linear_regression"
    assert fields["dynamic_fit_parameter_contract"].value["backend_default_source"] == BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE
    assert fields["dynamic_fit_parameter_contract"].value["backend_default_values"] == canonical_dynamic_fit_backend_defaults()
    assert fields["dynamic_fit_parameter_contract"].value["unresolved_parameters"] == []
    assert fields["dynamic_fit_parameter_contract"].blocks_subset is False
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


def _complete_new_analysis_plan_with_current_snapshot(**overrides):
    plan = _complete_new_analysis_plan(**overrides)
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        contract_values={
            "input_format": plan.input_format,
            "resolved_input_format": plan.input_format,
            "acquisition_mode": plan.acquisition_mode,
            "rwd_time_col": "Time",
            "sig_suffix": "_Signal",
            "uv_suffix": "_UV",
        },
    )
    return plan


def test_rwd_dataset_normalization_best_case_is_ready_and_separates_field_roles():
    plan = _complete_new_analysis_plan_with_current_snapshot()
    plan.dataset_contract_snapshot = dataclasses.replace(
        plan.dataset_contract_snapshot,
        contract_values={
            **plan.dataset_contract_snapshot.contract_values,
            "rwd_contract_validation": {"status": "reviewed"},
            "rwd_excluded_source_files": ["chunk_99.rwd"],
            "unknown_structural_note": "reviewed manually",
        },
    )

    normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert normalization["normalization_status"] == "ready_for_future_mapping"
    assert normalization["backend_config_mapping_status"] == "rwd_dataset_contract_ready_for_future_mapping"
    assert normalization["execution_consumption_enabled"] is True
    assert normalization["backend_config_values"] == {
        "rwd_time_col": "Time",
        "sig_suffix": "_Signal",
        "uv_suffix": "_UV",
        "exclude_incomplete_final_rwd_chunk": False,
    }
    assert set(normalization["backend_config_values"]) == RWD_NORMALIZATION_BACKEND_CONFIG_FIELDS
    assert normalization["structural_values"]["input_format"] == "rwd"
    assert normalization["structural_values"]["acquisition_mode"] == "intermittent"
    assert "input_format" not in normalization["backend_config_values"]
    assert normalization["provenance_values"]["rwd_contract_validation"] == {"status": "reviewed"}
    assert normalization["provenance_values"]["rwd_excluded_source_files"] == ["chunk_99.rwd"]
    assert "rwd_contract_validation" not in normalization["backend_config_values"]
    assert normalization["rejected_fields"] == {"unknown_structural_note": "reviewed manually"}
    assert normalization["missing_required_fields"] == []
    assert normalization["inconsistent_fields"] == []
    assert normalization["unresolved_fields"] == []
    assert normalization["blocker_categories"] == []
    assert normalization["no_file_inspection"] is True
    assert normalization["no_runspec"] is True
    assert normalization["no_argv"] is True
    assert normalization["no_config_written"] is True
    assert normalization["no_files_written"] is True
    assert preview.spec_preview_available is True
    assert preview.execution_available is False
    assert preview.dataset_contract["rwd_normalization"]["execution_consumption_enabled"] is True


@pytest.mark.parametrize("missing_field", RWD_NORMALIZATION_REQUIRED_SNAPSHOT_FIELDS)
def test_rwd_dataset_normalization_missing_required_backend_field_blocks(missing_field):
    plan = _complete_new_analysis_plan_with_current_snapshot()
    contract_values = dict(plan.dataset_contract_snapshot.contract_values)
    contract_values.pop(missing_field)
    plan.dataset_contract_snapshot = dataclasses.replace(
        plan.dataset_contract_snapshot,
        contract_values=contract_values,
    )

    normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert missing_field in normalization["missing_required_fields"]
    assert missing_field in normalization["unresolved_fields"]
    assert "missing_required_rwd_contract_field" in normalization["blocker_categories"]
    assert "unresolved_rwd_dataset_contract_normalization" in normalization["blocker_categories"]
    assert normalization["backend_config_mapping_status"] == "rwd_dataset_contract_unresolved"
    assert preview.spec_preview_available is False
    assert "missing_required_rwd_contract_field" in preview.blocking_issue_categories


@pytest.mark.parametrize(
    ("snapshot_overrides", "identity_overrides", "expected_field"),
    [
        ({"input_format": "npm"}, {}, "input_format"),
        ({"resolved_input_format": "npm"}, {}, "resolved_input_format"),
        ({"acquisition_mode": "continuous"}, {}, "acquisition_mode"),
        ({}, {"exclude_incomplete_final_rwd_chunk": True}, "exclude_incomplete_final_rwd_chunk"),
    ],
)
def test_rwd_dataset_normalization_inconsistent_duplicate_fields_block(
    snapshot_overrides,
    identity_overrides,
    expected_field,
):
    plan = _complete_new_analysis_plan_with_current_snapshot()
    identity = dataclasses.replace(
        plan.dataset_contract_snapshot.source_identity,
        **identity_overrides,
    )
    plan.dataset_contract_snapshot = dataclasses.replace(
        plan.dataset_contract_snapshot,
        source_identity=identity,
        **snapshot_overrides,
    )

    normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert expected_field in {item["field_name"] for item in normalization["inconsistent_fields"]}
    assert "inconsistent_rwd_contract_field" in normalization["blocker_categories"]
    assert normalization["backend_config_mapping_status"] == "rwd_dataset_contract_inconsistent"
    assert preview.spec_preview_available is False
    assert "inconsistent_rwd_contract_field" in preview.blocking_issue_categories


@pytest.mark.parametrize("input_format", ["npm", "custom_tabular", "auto"])
def test_rwd_dataset_normalization_unsupported_formats_are_not_coerced(input_format):
    plan = _complete_new_analysis_plan_with_current_snapshot(input_format=input_format)

    normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert normalization["structural_values"]["input_format"] == input_format
    assert normalization["normalization_status"] == "unsupported_format_for_rwd_first_subset"
    assert "unsupported_format_for_rwd_first_subset" in normalization["blocker_categories"]
    assert preview.spec_preview_available is False
    assert "unsupported_format_for_rwd_first_subset" in preview.blocking_issue_categories


def test_rwd_dataset_normalization_continuous_rwd_is_unsupported():
    plan = _complete_new_analysis_plan_with_current_snapshot(acquisition_mode="continuous")

    normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert normalization["structural_values"]["input_format"] == "rwd"
    assert normalization["structural_values"]["acquisition_mode"] == "continuous"
    assert normalization["normalization_status"] == "unsupported_acquisition_mode_for_rwd_first_subset"
    assert "unsupported_acquisition_mode_for_rwd_first_subset" in normalization["blocker_categories"]
    assert preview.spec_preview_available is False


@pytest.mark.parametrize(
    "snapshot_overrides",
    [
        {"status": "missing", "explicitly_applied": False},
        {"status": "stale", "stale_reasons": ("source changed",)},
        {"status": "invalid", "validation_issues": ("invalid contract",)},
        {"status": "applied", "explicitly_applied": False},
    ],
)
def test_rwd_dataset_normalization_noncurrent_snapshot_blocks(snapshot_overrides):
    plan = _complete_new_analysis_plan_with_current_snapshot()
    plan.dataset_contract_snapshot = dataclasses.replace(
        plan.dataset_contract_snapshot,
        **snapshot_overrides,
    )

    normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert normalization["normalization_status"] == "dataset_contract_snapshot_not_current"
    assert "dataset_contract_snapshot_not_current" in normalization["blocker_categories"]
    assert normalization["execution_consumption_enabled"] is False
    assert preview.spec_preview_available is False
    assert "dataset_contract_snapshot_not_current" in preview.blocking_issue_categories


def test_rwd_dataset_normalization_is_pure_and_performs_no_file_inspection(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_output"
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path=str(target))
    before_state = dict(plan.__dict__)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after_files == before_files
    assert not target.exists()
    assert dict(plan.__dict__) == before_state
    assert normalization["no_file_inspection"] is True
    assert normalization["no_runspec"] is True
    assert normalization["no_argv"] is True
    assert normalization["no_config_written"] is True
    assert normalization["no_files_written"] is True
    assert preview.execution_available is False


def test_output_safety_ownership_best_case_is_runner_owned_pattern_only():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    safety = build_guided_output_base_safety_ownership_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)
    preview_safety = preview.output["output_safety_ownership"]

    assert safety["output_safety_status"] == "output_base_ready_for_runner_owned_future_mapping"
    assert safety["backend_config_mapping_status"] == "output_base_ready_for_runner_owned_future_mapping"
    assert safety["future_output_owner"] == "runner"
    assert safety["path_role"] == "output_base"
    assert safety["run_directory_strategy"] == "derive_unique_run_id_under_output_base"
    assert safety["future_run_dir"] == "unresolved_until_execution_start"
    assert safety["future_run_directory_pattern"] == "<output_base>/<runner-generated-run-id>"
    assert safety["preview_path_kind"] == "pattern_only"
    assert safety["concrete_run_dir_known"] is False
    assert safety["overwrite_requested"] is False
    assert safety["filesystem_facts"]["facts_source"] == "not_inspected"
    assert safety["filesystem_facts"]["writability_status"] == "unknown"
    assert safety["filesystem_facts"]["filesystem_facts_required_for_planning_mapping"] is False
    assert safety["blocker_categories"] == []
    assert safety["execution_consumption_enabled"] is True
    assert safety["no_directory_creation"] is True
    assert safety["no_directory_reservation"] is True
    assert safety["no_config_written"] is True
    assert safety["no_command_written"] is True
    assert safety["no_files_written"] is True
    assert safety["no_validation"] is True
    assert safety["no_run"] is True
    assert preview.spec_preview_available is True
    assert preview.execution_available is False
    assert preview_safety["execution_consumption_enabled"] is True
    assert preview_safety["future_run_dir"] == "unresolved_until_execution_start"
    assert preview_safety["concrete_run_dir_known"] is False


@pytest.mark.parametrize(
    ("overrides", "expected_status"),
    [
        (
            {
                "output_policy_status": "missing",
                "output_policy_path": None,
                "output_policy_explicitly_applied": False,
            },
            "output_policy_not_current",
        ),
        (
            {
                "output_policy_status": "stale",
                "output_policy_stale_reasons": ["target changed"],
            },
            "output_policy_not_current",
        ),
        (
            {
                "output_policy_status": "invalid",
                "output_policy_validation_issues": ["inside source"],
            },
            "output_policy_not_current",
        ),
        (
            {
                "output_policy_status": "applied",
                "output_policy_explicitly_applied": False,
            },
            "output_policy_not_current",
        ),
    ],
)
def test_output_safety_noncurrent_output_policy_blocks(overrides, expected_status):
    plan = _complete_new_analysis_plan_with_current_snapshot(**overrides)

    safety = build_guided_output_base_safety_ownership_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert safety["output_safety_status"] == expected_status
    assert "output_policy_not_current" in safety["blocker_categories"]
    assert preview.spec_preview_available is False
    assert "output_policy_not_current" in preview.blocking_issue_categories
    assert preview.output["output_safety_ownership"]["execution_consumption_enabled"] is False


def test_output_safety_missing_output_base_blocks_even_when_policy_claims_applied():
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path=None)

    safety = build_guided_output_base_safety_ownership_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert safety["output_safety_status"] == "output_base_missing"
    assert "output_base_missing" in safety["blocker_categories"]
    assert preview.spec_preview_available is False
    assert "output_base_missing" in preview.blocking_issue_categories


def test_output_safety_relative_output_base_blocks():
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path="relative/output")

    safety = build_guided_output_base_safety_ownership_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert safety["output_safety_status"] == "output_base_relative"
    assert "output_base_relative" in safety["blocker_categories"]
    assert preview.spec_preview_available is False
    assert "output_base_relative" in preview.blocking_issue_categories


def test_output_safety_windows_drive_paths_are_platform_stable():
    safety = classify_output_base_safety_ownership(
        output_base="C:/raw/input/future_output",
        source_path="C:/raw/input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
    )

    relationship_status = {
        item["relationship"]: item["status"]
        for item in safety["path_relationships"]
    }
    relationship_evidence = {
        item["relationship"]: item["evidence"]
        for item in safety["path_relationships"]
    }
    assert safety["normalized_output_base"] == "c:/raw/input/future_output"
    assert relationship_evidence["output_base_inside_source"]["output_path_style"] == "windows_drive"
    assert relationship_evidence["output_base_inside_source"]["source_path_style"] == "windows_drive"
    assert relationship_status["output_base_equals_source"] == "safe"
    assert relationship_status["output_base_inside_source"] == "unsafe"
    assert relationship_status["source_inside_output_base"] == "safe"
    assert safety["output_safety_status"] == "unsafe_source_output_relationship"


def test_output_safety_windows_backslash_paths_normalize_like_windows_drive_paths():
    safety = classify_output_base_safety_ownership(
        output_base=r"C:\raw\input\future_output",
        source_path=r"C:\raw\input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
    )

    relationship_status = {
        item["relationship"]: item["status"]
        for item in safety["path_relationships"]
    }
    assert safety["normalized_output_base"] == "c:/raw/input/future_output"
    assert relationship_status["output_base_inside_source"] == "unsafe"
    assert "unsafe_source_output_relationship" in safety["blocker_categories"]


def test_output_safety_posix_absolute_paths_are_platform_stable():
    safety = classify_output_base_safety_ownership(
        output_base="/raw/input/future_output",
        source_path="/raw/input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
    )

    relationship_status = {
        item["relationship"]: item["status"]
        for item in safety["path_relationships"]
    }
    relationship_evidence = {
        item["relationship"]: item["evidence"]
        for item in safety["path_relationships"]
    }
    assert safety["normalized_output_base"] == "/raw/input/future_output"
    assert relationship_evidence["output_base_inside_source"]["output_path_style"] == "posix"
    assert relationship_evidence["output_base_inside_source"]["source_path_style"] == "posix"
    assert relationship_status["output_base_inside_source"] == "unsafe"
    assert safety["output_safety_status"] == "unsafe_source_output_relationship"


def test_output_safety_mixed_path_styles_block_instead_of_silent_safe():
    safety = classify_output_base_safety_ownership(
        output_base="C:/planned/output",
        source_path="/raw/input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
    )

    relationship_status = {
        item["relationship"]: item["status"]
        for item in safety["path_relationships"]
    }
    assert set(relationship_status.values()) == {"unknown_mixed_path_style"}
    assert safety["output_safety_status"] == "output_path_style_mismatch"
    assert "output_path_style_mismatch" in safety["blocker_categories"]


@pytest.mark.parametrize(
    ("output_path", "relationship"),
    [
        ("C:/raw/input", "output_base_equals_source"),
        ("C:/raw/input/future_output", "output_base_inside_source"),
        ("C:/raw", "source_inside_output_base"),
    ],
)
def test_output_safety_source_output_relationships_block(output_path, relationship):
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path=output_path)

    safety = build_guided_output_base_safety_ownership_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    relationship_status = {
        item["relationship"]: item["status"]
        for item in safety["path_relationships"]
    }
    assert relationship_status[relationship] == "unsafe"
    assert safety["output_safety_status"] == "unsafe_source_output_relationship"
    assert "unsafe_source_output_relationship" in safety["blocker_categories"]
    assert preview.spec_preview_available is False
    assert "unsafe_source_output_relationship" in preview.blocking_issue_categories


@pytest.mark.parametrize(
    ("root_kind", "root_path", "output_path"),
    [
        ("completed_run", "C:/completed/run-1", "C:/completed/run-1/future_output"),
        ("legacy_output", "C:/legacy/output", "C:/legacy/output/future_output"),
        ("diagnostic_cache", "C:/cache", "C:/cache/future_output"),
    ],
)
def test_output_safety_protected_roots_block(root_kind, root_path, output_path):
    safety = classify_output_base_safety_ownership(
        output_base=output_path,
        source_path="C:/raw/input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
        protected_roots=((root_kind, root_path),),
        protected_root_context_complete=True,
    )

    assert safety["output_safety_status"] == "unsafe_protected_output_location"
    assert safety["protected_root_status"] == "unsafe"
    assert "unsafe_protected_output_location" in safety["blocker_categories"]
    assert safety["protected_root_relationships"][0]["root_kind"] == root_kind
    assert safety["protected_root_relationships"][0]["status"] == "unsafe"


def test_output_safety_mixed_protected_root_style_blocks_when_context_present():
    safety = classify_output_base_safety_ownership(
        output_base="C:/planned/output",
        source_path="C:/raw/input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
        protected_roots=(("completed_run", "/completed/run-1"),),
        protected_root_context_complete=True,
    )

    assert safety["output_safety_status"] == "unsafe_protected_output_location"
    assert safety["protected_root_status"] == "unsafe"
    assert safety["protected_root_relationships"][0]["status"] == "unknown_mixed_path_style"
    assert "unsafe_protected_output_location" in safety["blocker_categories"]


def test_output_safety_overwrite_intent_blocks():
    plan = _complete_new_analysis_plan_with_current_snapshot(
        output_creation_policy=GuidedNewAnalysisOutputCreationPolicy(overwrite=True),
    )

    safety = build_guided_output_base_safety_ownership_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert safety["output_safety_status"] == "unsafe_overwrite_for_guided_first_subset"
    assert "unsafe_overwrite_for_guided_first_subset" in safety["blocker_categories"]
    assert preview.spec_preview_available is False
    assert "unsafe_overwrite_for_guided_first_subset" in preview.blocking_issue_categories


def test_output_safety_filesystem_facts_are_explicit_and_non_required():
    safety = classify_output_base_safety_ownership(
        output_base="C:/planned/output",
        source_path="C:/raw/input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
        filesystem_facts={
            "exists": True,
            "is_directory": True,
            "is_file": False,
            "directory_empty": False,
            "writable": None,
        },
    )

    assert safety["output_safety_status"] == "output_base_ready_for_runner_owned_future_mapping"
    assert safety["filesystem_facts"]["facts_source"] == "supplied_read_only_facts"
    assert safety["filesystem_facts"]["exists"] is True
    assert safety["filesystem_facts"]["is_directory"] is True
    assert safety["filesystem_facts"]["directory_empty"] is False
    assert safety["filesystem_facts"]["directory_non_empty"] is True
    assert safety["filesystem_facts"]["requires_creation"] is False
    assert safety["filesystem_facts"]["writability_status"] == "unknown"
    assert safety["filesystem_facts"]["filesystem_facts_required_for_planning_mapping"] is False
    assert safety["blocker_categories"] == []


def test_output_safety_supplied_file_or_not_writable_facts_block():
    safety = classify_output_base_safety_ownership(
        output_base="C:/planned/output.txt",
        source_path="C:/raw/input",
        output_policy_status="applied",
        output_policy_explicitly_applied=True,
        filesystem_facts={
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "writable": False,
        },
    )

    assert safety["output_safety_status"] == "output_base_safety_unresolved"
    assert "output_base_is_file" in safety["blocker_categories"]
    assert "output_base_not_writable" in safety["blocker_categories"]


def test_output_safety_preview_is_pure_no_files_no_concrete_run_dir(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_output_base"
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path=str(target))
    before_state = dict(plan.__dict__)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    safety = build_guided_output_base_safety_ownership_preview(plan)
    preview = build_guided_new_analysis_execution_spec_preview(plan)

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after_files == before_files
    assert not target.exists()
    assert dict(plan.__dict__) == before_state
    assert safety["future_run_dir"] == "unresolved_until_execution_start"
    assert safety["preview_path_kind"] == "pattern_only"
    assert safety["concrete_run_dir_known"] is False
    assert preview.output["future_run_dir"] == "unresolved_until_execution_start"
    assert preview.output["output_safety_ownership"]["concrete_run_dir_known"] is False
    assert preview.output["output_safety_ownership"]["no_directory_creation"] is True
    assert preview.output["output_safety_ownership"]["no_directory_reservation"] is True
    assert preview.output["output_safety_ownership"]["no_files_written"] is True
    assert preview.execution_available is False


def test_execution_spec_preview_best_case_rwd_available_but_never_executable():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert preview.spec_preview_schema_version == EXECUTION_SPEC_PREVIEW_SCHEMA_VERSION
    assert preview.plan_schema_version == plan.schema_version
    assert preview.subset_name == FIRST_EXECUTION_SUBSET_NAME
    assert preview.spec_preview_available is True
    assert preview.first_subset_executable is True
    assert preview.execution_available is False
    assert preview.backend_mapping_status == "preview_only_not_mapped_to_RunSpec"
    assert preview.blocking_issue_categories == ()
    assert preview.source_acquisition["authoritative_input_source_path"] == "C:/raw/input"
    assert preview.source_acquisition["input_format"] == "rwd"
    assert preview.source_acquisition["acquisition_mode"] == "intermittent"
    assert preview.dataset_contract["current_applied"] is True
    assert preview.dataset_contract["execution_consumption_enabled"] is True
    assert preview.roi["included_roi_ids"] == ["ROI1"]
    assert preview.roi["include_list_is_authoritative"] is True
    assert preview.correction["selected_global_dynamic_fit_strategy"] == "global_linear_regression"
    assert preview.correction["global_strategy_derivation"] == "unanimous_explicit_per_roi_choices"
    assert preview.correction["global_strategy_collapsed"] is False
    assert preview.correction["per_roi_choices"][0]["roi_id"] == "ROI1"
    dyn_contract = preview.correction["dynamic_fit_parameter_contract"]
    assert dyn_contract["dynamic_fit_mode"] == "global_linear_regression"
    assert dyn_contract["selected_strategy"] == "global_linear_regression"
    assert dyn_contract["active_parameter_set"] == "global_linear_regression"
    assert dyn_contract["active_parameters"]["slope_constraint"] == "unconstrained"
    assert dyn_contract["active_parameters"]["window_sec"] == 60.0
    assert dyn_contract["backend_default_source"] == BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE
    assert dyn_contract["backend_default_values"] == canonical_dynamic_fit_backend_defaults()
    assert "robust_event_rejection" in dyn_contract["inactive_parameter_sets"]
    assert "adaptive_event_gate" in dyn_contract["inactive_parameter_sets"]
    assert dyn_contract["unresolved_parameters"] == []
    assert dyn_contract["execution_consumption_enabled"] is True
    assert dyn_contract["backend_config_mapping_status"] == "label_and_parameters_ready_for_future_mapping"
    assert dyn_contract["no_runspec"] is True
    assert dyn_contract["no_argv"] is True
    assert dyn_contract["no_config_written"] is True
    assert dyn_contract["no_files_written"] is True
    assert preview.execution_intent["timeline_anchor_mode"] == "civil"
    assert preview.execution_intent["execution_mode"] == "phasic"
    assert preview.execution_intent["run_profile"] == "full"
    assert preview.execution_intent["traces_only"] is False
    assert preview.execution_intent["tonic_outputs_in_scope"] is False
    assert preview.feature_event["values"] == {"event_signal": "dff"}
    assert preview.feature_event["consumption"]["feature_event_values_consumed"] is True
    feature_effective = preview.feature_event["feature_event_effective_values"]
    assert feature_effective["backend_config_mapping_status"] == "effective_values_ready_for_future_mapping"
    assert feature_effective["execution_consumption_enabled"] is True
    assert feature_effective["backend_default_source"] == FEATURE_EVENT_BACKEND_DEFAULT_SOURCE
    assert feature_effective["backend_default_values"] == canonical_feature_event_backend_defaults()
    assert feature_effective["unresolved_fields"] == []
    assert feature_effective["blocker_categories"] == []
    feature_effective_values = _effective_values_by_field(feature_effective)
    assert feature_effective_values["event_signal"]["effective_value"] == "dff"
    assert feature_effective_values["event_signal"]["source"] == "applied_guided_profile"
    assert feature_effective_values["peak_threshold_method"]["source"] == "backend_config_default"
    assert feature_effective_values["peak_pre_filter"]["active_or_inactive"] == "active"
    assert feature_effective_values["event_auc_baseline"]["active_or_inactive"] == "active"
    assert preview.output["output_base"] == "C:/planned/output"
    assert preview.output["path_role"] == "output_base"
    assert preview.output["future_run_directory_strategy"] == "derive_unique_run_id_under_output_base"
    assert preview.output["future_run_dir"] == "unresolved_until_execution_start"
    assert preview.output["overwrite"] is False
    output_safety = preview.output["output_safety_ownership"]
    assert output_safety["output_safety_status"] == "output_base_ready_for_runner_owned_future_mapping"
    assert output_safety["backend_config_mapping_status"] == "output_base_ready_for_runner_owned_future_mapping"
    assert output_safety["future_output_owner"] == "runner"
    assert output_safety["path_role"] == "output_base"
    assert output_safety["run_directory_strategy"] == "derive_unique_run_id_under_output_base"
    assert output_safety["future_run_dir"] == "unresolved_until_execution_start"
    assert output_safety["preview_path_kind"] == "pattern_only"
    assert output_safety["concrete_run_dir_known"] is False
    assert output_safety["blocker_categories"] == []
    assert output_safety["execution_consumption_enabled"] is True
    assert output_safety["filesystem_facts"]["facts_source"] == "not_inspected"
    assert output_safety["filesystem_facts"]["writability_status"] == "unknown"
    assert output_safety["no_directory_creation"] is True
    assert output_safety["no_directory_reservation"] is True
    assert output_safety["no_files_written"] is True
    assert output_safety["no_validation"] is True
    assert output_safety["no_run"] is True
    assert preview.output["directory_created"] is False
    assert preview.output["config_written"] is False
    assert preview.output["RunSpec_instantiated"] is False
    assert preview.output["argv_generated"] is False
    assert preview.diagnostic_cache_provenance["cache_id"] == "cache-1"
    assert preview.diagnostic_cache_provenance["execution_consumes_cache_artifacts"] is False
    assert preview.provenance["no_gui_runspec"] is True
    assert preview.provenance["no_argv_generated"] is True
    assert preview.provenance["no_config_written"] is True
    assert preview.provenance["no_output_directory_created"] is True
    assert preview.provenance["no_validation_run"] is True
    assert preview.provenance["no_pipeline_run"] is True


def test_first_subset_mapping_preview_best_case_is_available_and_pure():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    mapping = build_guided_first_subset_executable_mapping_preview(plan)
    spec_preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert isinstance(mapping, GuidedFirstSubsetExecutableMappingPreview)
    assert type(mapping).__name__ != "RunSpec"
    assert mapping.mapping_preview_schema_version == FIRST_SUBSET_MAPPING_PREVIEW_SCHEMA_VERSION
    assert mapping.mapping_preview_available is True
    assert mapping.first_subset_name == FIRST_EXECUTION_SUBSET_NAME
    assert spec_preview.first_subset_executable_mapping_preview is not None
    assert spec_preview.first_subset_executable_mapping_preview.mapping_preview_available is True
    assert spec_preview.execution_available is False
    assert mapping.supported_scope == {
        "input_format": "rwd",
        "acquisition_mode": "intermittent",
        "execution_mode": "phasic",
        "run_profile": "full",
        "traces_only": False,
        "correction_scope": "unanimous_global_dynamic_fit_only",
        "output_owner": "runner",
        "future_cli_target_concept": "out_base",
        "no_validation": True,
        "no_execution": True,
    }
    assert mapping.no_runspec is True
    assert mapping.no_argv is True
    assert mapping.no_config_dict is True
    assert mapping.no_config_serialization is True
    assert mapping.no_yaml is True
    assert mapping.no_json is True
    assert mapping.no_directory_creation is True
    assert mapping.no_directory_reservation is True
    assert mapping.no_validation is True
    assert mapping.no_run is True


def test_first_subset_mapping_dataset_and_source_are_rwd_intermittent_only():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    mapping = build_guided_first_subset_executable_mapping_preview(plan)
    source_entries = _mapping_entries_by_field(mapping.source_acquisition_mapping)
    dataset_entries = _mapping_entries_by_field(mapping.dataset_mapping)

    assert mapping.source_acquisition_mapping["section_status"] == "ready_for_first_subset_mapping"
    assert source_entries["input_source_path"]["value"] == "C:/raw/input"
    assert source_entries["input_source_path"]["mapping_status"] == "would_emit_override"
    assert source_entries["input_format"]["value"] == "rwd"
    assert source_entries["acquisition_mode"]["value"] == "intermittent"
    assert source_entries["continuous_window_sec"]["mapping_status"] == "not_applicable"
    assert source_entries["continuous_step_sec"]["mapping_status"] == "not_applicable"
    assert mapping.dataset_mapping["section_status"] == "ready_for_first_subset_mapping"
    assert dataset_entries["rwd_time_col"]["value"] == "Time"
    assert dataset_entries["rwd_time_col"]["mapping_status"] == "would_emit_override"
    assert dataset_entries["sig_suffix"]["value"] == "_Signal"
    assert dataset_entries["uv_suffix"]["value"] == "_UV"
    assert dataset_entries["npm_channel_mapping"]["mapping_status"] == "unsupported_first_subset"
    assert dataset_entries["custom_tabular_column_mapping"]["mapping_status"] == "unsupported_first_subset"


def test_first_subset_mapping_roi_semantics_subset_and_all_rois():
    subset_plan = _complete_new_analysis_plan_with_current_snapshot()
    subset_mapping = build_guided_first_subset_executable_mapping_preview(subset_plan)
    subset_entries = _mapping_entries_by_field(subset_mapping.roi_mapping)

    assert subset_mapping.roi_mapping["include_filter_kind"] == "explicit_include_filter"
    assert subset_mapping.roi_mapping["would_emit_include_filter"] is True
    assert subset_mapping.roi_mapping["would_emit_exclude_filter"] is False
    assert subset_entries["included_roi_ids"]["mapping_status"] == "would_emit_override"
    assert subset_entries["excluded_roi_ids"]["mapping_status"] == "display_only_provenance"
    assert subset_entries["excluded_roi_ids"]["production_input"] is False

    all_roi_plan = _complete_new_analysis_plan_with_current_snapshot(
        included_roi_ids=["ROI1", "ROI2"],
        excluded_roi_ids=[],
        per_roi_correction_strategy_choices=[_strategy_choice("global_linear_regression", roi_id="ROI1"),
                                             _strategy_choice("global_linear_regression", roi_id="ROI2")],
    )
    all_roi_plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        all_roi_plan,
        contract_values={
            "input_format": "rwd",
            "resolved_input_format": "rwd",
            "acquisition_mode": "intermittent",
            "rwd_time_col": "Time",
            "sig_suffix": "_Signal",
            "uv_suffix": "_UV",
        },
    )
    all_roi_mapping = build_guided_first_subset_executable_mapping_preview(all_roi_plan)
    all_entries = _mapping_entries_by_field(all_roi_mapping.roi_mapping)

    assert all_roi_mapping.roi_mapping["include_filter_kind"] == "all_rois"
    assert all_roi_mapping.roi_mapping["all_rois_policy"] == "omit_include_filter_and_rely_on_backend_all_rois"
    assert all_roi_mapping.roi_mapping["would_emit_include_filter"] is False
    assert all_entries["included_roi_ids"]["mapping_status"] == "would_rely_on_backend_default"


def test_first_subset_mapping_dynamic_fit_emit_omit_policy():
    plan = _complete_new_analysis_plan_with_current_snapshot(
        dynamic_fit_parameter_contract=GuidedNewAnalysisDynamicFitParameterContract(
            dynamic_fit_mode="global_linear_regression",
            window_sec=90.0,
        ),
    )

    mapping = build_guided_first_subset_executable_mapping_preview(plan)
    entries = _mapping_entries_by_field(mapping.correction_mapping)

    assert mapping.correction_mapping["section_status"] == "ready_for_first_subset_mapping"
    assert entries["dynamic_fit_mode"]["mapping_status"] == "would_emit_override"
    assert entries["slope_constraint"]["mapping_status"] == "would_rely_on_backend_default"
    assert entries["window_sec"]["mapping_status"] == "would_emit_override"
    assert entries["robust_event_rejection.robust_event_reject_max_iters"]["mapping_status"] == "display_only_inactive"
    assert entries["adaptive_event_gate.adaptive_event_gate_smooth_window_sec"]["mapping_status"] == "display_only_inactive"
    assert entries["robust_event_rejection.robust_event_reject_max_iters"]["production_input"] is False
    assert mapping.correction_mapping["no_config_dict"] is True


def test_first_subset_mapping_feature_event_emit_omit_policy():
    plan = _complete_new_analysis_plan_with_current_snapshot(
        feature_event_values={
            "event_signal": "dff",
            "peak_threshold_method": "absolute",
            "peak_threshold_abs": 0.25,
        },
    )

    mapping = build_guided_first_subset_executable_mapping_preview(plan)
    entries = _mapping_entries_by_field(mapping.feature_event_mapping)

    assert mapping.feature_event_mapping["section_status"] == "ready_for_first_subset_mapping"
    assert mapping.feature_event_mapping["active_threshold_field"] == "peak_threshold_abs"
    assert entries["event_signal"]["mapping_status"] == "would_rely_on_backend_default"
    assert entries["peak_threshold_method"]["mapping_status"] == "would_emit_override"
    assert entries["peak_threshold_abs"]["mapping_status"] == "would_emit_override"
    assert entries["peak_threshold_k"]["mapping_status"] == "display_only_inactive"
    assert entries["peak_threshold_k"]["production_input"] is False
    assert entries["peak_pre_filter"]["mapping_status"] == "would_rely_on_backend_default"
    assert mapping.feature_event_mapping["no_config_dict"] is True


def test_first_subset_mapping_output_is_runner_owned_out_base_concept_only():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    mapping = build_guided_first_subset_executable_mapping_preview(plan)
    entries = _mapping_entries_by_field(mapping.output_mapping)

    assert mapping.output_mapping["section_status"] == "ready_for_first_subset_mapping"
    assert mapping.output_mapping["future_output_owner"] == "runner"
    assert mapping.output_mapping["future_cli_target_concept"] == "out_base"
    assert mapping.output_mapping["concrete_run_dir_known"] is False
    assert entries["output_base"]["value"] == "C:/planned/output"
    assert entries["future_cli_target_concept"]["mapping_status"] == "should_not_emit"
    assert entries["future_run_dir"]["value"] == "unresolved_until_execution_start"
    assert entries["future_run_dir"]["mapping_status"] == "should_not_emit"
    assert mapping.output_mapping["no_argv"] is True
    assert mapping.output_mapping["no_directory_creation"] is True
    assert mapping.output_mapping["no_directory_reservation"] is True


def test_first_subset_mapping_diagnostic_and_evidence_are_provenance_only():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    mapping = build_guided_first_subset_executable_mapping_preview(plan)
    entries = _mapping_entries_by_field(mapping.provenance_mapping)

    assert mapping.provenance_mapping["section_status"] == "display_only_provenance"
    assert mapping.provenance_mapping["execution_consumes_cache_artifacts"] is False
    assert mapping.provenance_mapping["evidence_chunks_are_production_scope"] is False
    assert entries["diagnostic_cache_identity"]["mapping_status"] == "display_only_provenance"
    assert entries["diagnostic_cache_identity"]["production_input"] is False
    assert entries["diagnostic_cache_root_path"]["production_input"] is False


@pytest.mark.parametrize(
    ("overrides", "expected_category"),
    [
        ({"input_format": "npm"}, "unsupported_input_format_for_first_subset_mapping"),
        ({"input_format": "custom_tabular"}, "unsupported_input_format_for_first_subset_mapping"),
        ({"input_format": "auto"}, "unsupported_input_format_for_first_subset_mapping"),
        ({"acquisition_mode": "continuous"}, "unsupported_acquisition_mode_for_first_subset_mapping"),
    ],
)
def test_first_subset_mapping_unsupported_formats_and_modes_are_unavailable(overrides, expected_category):
    plan = _complete_new_analysis_plan_with_current_snapshot(**overrides)

    mapping = build_guided_first_subset_executable_mapping_preview(plan)

    assert mapping.mapping_preview_available is False
    assert expected_category in mapping.blocking_issue_categories
    assert expected_category in mapping.unsupported_reasons


@pytest.mark.parametrize(
    ("plan_overrides", "expected_category"),
    [
        (
            {
                "dynamic_fit_parameter_contract": GuidedNewAnalysisDynamicFitParameterContract(
                    unresolved_parameters=("legacy_global_settings",)
                )
            },
            "unresolved_dynamic_fit_parameter_contract",
        ),
        ({"feature_event_profile_status": "stale", "feature_event_stale_reasons": ["baseline changed"]},
         "incomplete_planning_readiness"),
        ({"output_policy_path": "relative/output"}, "output_base_relative"),
    ],
)
def test_first_subset_mapping_upstream_blockers_keep_mapping_unavailable(plan_overrides, expected_category):
    plan = _complete_new_analysis_plan_with_current_snapshot(**plan_overrides)

    mapping = build_guided_first_subset_executable_mapping_preview(plan)

    assert mapping.mapping_preview_available is False
    assert expected_category in mapping.blocking_issue_categories
    assert mapping.no_runspec is True
    assert mapping.no_argv is True
    assert mapping.no_config_dict is True
    assert mapping.no_directory_creation is True
    assert mapping.no_validation is True
    assert mapping.no_run is True


def test_first_subset_mapping_signal_only_and_mixed_strategies_are_unavailable():
    signal_plan = _complete_new_analysis_plan_with_current_snapshot(
        per_roi_correction_strategy_choices=[_strategy_choice("signal_only_f0")]
    )
    mixed_plan = _complete_new_analysis_plan_with_current_snapshot(
        included_roi_ids=["ROI1", "ROI2"],
        excluded_roi_ids=[],
        per_roi_correction_strategy_choices=[
            _strategy_choice("global_linear_regression", roi_id="ROI1"),
            _strategy_choice("robust_global_event_reject", roi_id="ROI2"),
        ],
    )
    mixed_plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        mixed_plan,
        contract_values={
            "input_format": "rwd",
            "resolved_input_format": "rwd",
            "acquisition_mode": "intermittent",
            "rwd_time_col": "Time",
            "sig_suffix": "_Signal",
            "uv_suffix": "_UV",
        },
    )

    signal_mapping = build_guided_first_subset_executable_mapping_preview(signal_plan)
    mixed_mapping = build_guided_first_subset_executable_mapping_preview(mixed_plan)

    assert signal_mapping.mapping_preview_available is False
    assert "signal_only_f0_production_routing_not_supported" not in signal_mapping.unsupported_reasons
    assert mixed_mapping.mapping_preview_available is False
    assert "mixed_per_roi_strategy_execution_not_supported" not in mixed_mapping.unsupported_reasons


def test_first_subset_mapping_preview_is_pure_no_files_and_no_plan_mutation(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_output_base"
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path=str(target))
    before_state = dict(plan.__dict__)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    mapping = build_guided_first_subset_executable_mapping_preview(plan)

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after_files == before_files
    assert not target.exists()
    assert dict(plan.__dict__) == before_state
    assert mapping.no_runspec is True
    assert mapping.no_argv is True
    assert mapping.no_config_dict is True
    assert mapping.no_config_serialization is True
    assert mapping.no_yaml is True
    assert mapping.no_json is True
    assert mapping.no_directory_creation is True
    assert mapping.no_directory_reservation is True
    assert mapping.no_validation is True
    assert mapping.no_run is True


def test_guided_runner_request_preview_best_case_is_available_and_pure():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    runner_preview = build_guided_runner_request_preview(plan)
    spec_preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert isinstance(runner_preview, GuidedRunnerRequestPreview)
    assert type(runner_preview).__name__ != "RunSpec"
    assert runner_preview.runner_request_preview_schema_version == RUNNER_REQUEST_PREVIEW_SCHEMA_VERSION
    assert runner_preview.runner_request_preview_available is True
    assert runner_preview.request_kind == "guided_first_subset_runner_request_preview"
    assert runner_preview.first_subset_name == FIRST_EXECUTION_SUBSET_NAME
    assert runner_preview.future_runner_owner == "runner"
    assert runner_preview.future_cli_target_concept == "out_base"
    assert spec_preview.guided_runner_request_preview is not None
    assert spec_preview.guided_runner_request_preview.runner_request_preview_available is True
    assert spec_preview.execution_available is False
    assert runner_preview.no_real_runner_request is True
    assert runner_preview.no_runspec is True
    assert runner_preview.no_argv is True
    assert runner_preview.no_command_text is True
    assert runner_preview.no_config_dict is True
    assert runner_preview.no_config_serialization is True
    assert runner_preview.no_yaml is True
    assert runner_preview.no_json is True
    assert runner_preview.no_file_writes is True
    assert runner_preview.no_directory_creation is True
    assert runner_preview.no_directory_reservation is True
    assert runner_preview.no_validation is True
    assert runner_preview.no_run is True


def test_guided_runner_request_sections_preserve_mapping_entries_without_config_dict():
    plan = _complete_new_analysis_plan_with_current_snapshot()
    mapping = build_guided_first_subset_executable_mapping_preview(plan)

    runner_preview = build_guided_runner_request_preview(plan, mapping_preview=mapping)

    source_request = runner_preview.request_sections["source_input_request"]
    dataset_request = runner_preview.request_sections["dataset_contract_request"]
    provenance_request = runner_preview.request_sections["decision_provenance_request"]
    assert source_request["section_status"] == "ready_for_runner_request_preview"
    assert source_request["source_mapping_section"] == "source_acquisition_mapping"
    assert source_request["entries"] == mapping.source_acquisition_mapping["entries"]
    assert source_request["would_contribute_to_future_config"] is True
    assert source_request["would_contribute_to_future_cli"] is True
    assert dataset_request["entries"] == mapping.dataset_mapping["entries"]
    assert provenance_request["section_status"] == "metadata_only"
    assert provenance_request["would_contribute_to_future_config"] is False
    assert provenance_request["would_contribute_to_future_cli"] is False
    assert provenance_request["would_contribute_to_future_metadata"] is True
    assert runner_preview.config_conversion_preview["no_config_dict"] is True
    assert "config" not in runner_preview.request_sections


def test_guided_runner_request_config_conversion_preview_classifies_without_generating_config():
    plan = _complete_new_analysis_plan_with_current_snapshot(
        dynamic_fit_parameter_contract=GuidedNewAnalysisDynamicFitParameterContract(
            dynamic_fit_mode="global_linear_regression",
            window_sec=90.0,
        ),
        feature_event_values={
            "event_signal": "dff",
            "peak_threshold_method": "absolute",
            "peak_threshold_abs": 0.25,
        },
    )

    runner_preview = build_guided_runner_request_preview(plan)
    conversion = runner_preview.config_conversion_preview

    assert conversion["conversion_by_mapping_status"]["would_emit_override"] == "future_config_override_candidate"
    assert conversion["conversion_by_mapping_status"]["would_rely_on_backend_default"] == "future_backend_default_reliance"
    assert conversion["conversion_by_mapping_status"]["display_only_inactive"] == "omit_from_future_config_inactive"
    assert conversion["conversion_by_mapping_status"]["display_only_provenance"] == "future_metadata_only"
    assert conversion["conversion_by_mapping_status"]["should_not_emit"] == "omit_from_future_config"
    assert conversion["conversion_by_mapping_status"]["not_applicable"] == "omit_not_applicable"
    assert conversion["conversion_by_mapping_status"]["unsupported_first_subset"] == "block_future_request"
    override_fields = {
        item["field"]
        for item in conversion["fields_by_conversion_class"]["future_config_override_candidate"]
    }
    default_fields = {
        item["field"]
        for item in conversion["fields_by_conversion_class"]["future_backend_default_reliance"]
    }
    inactive_fields = {
        item["field"]
        for item in conversion["fields_by_conversion_class"]["omit_from_future_config_inactive"]
    }
    assert "window_sec" in override_fields
    assert "peak_threshold_abs" in override_fields
    assert "slope_constraint" in default_fields
    assert "peak_pre_filter" in default_fields
    assert "robust_event_rejection.robust_event_reject_max_iters" in inactive_fields
    assert "peak_threshold_k" in inactive_fields
    assert conversion["config_payload_generated"] is False
    assert conversion["config_serialization_generated"] is False
    assert conversion["default_reliance_recorded"] is True
    assert conversion["inactive_fields_omitted"] is True
    assert conversion["provenance_fields_metadata_only"] is True
    assert conversion["blocking_fields"] == ()


def test_guided_runner_request_argv_and_output_previews_are_conceptual_only():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    runner_preview = build_guided_runner_request_preview(plan)
    argv_preview = runner_preview.argv_conversion_preview
    output_preview = runner_preview.output_request_preview

    assert argv_preview["argv_generated"] is False
    assert argv_preview["command_text_generated"] is False
    assert argv_preview["future_cli_target_concept"] == "out_base"
    assert argv_preview["uses_out_base_concept"] is True
    assert argv_preview["uses_out_concept"] is False
    assert argv_preview["future_format_concept"] == "rwd"
    assert argv_preview["future_mode_concept"] == "phasic"
    assert argv_preview["future_profile_concept"] == "full"
    assert argv_preview["roi_filter_concept"]["included_roi_ids"] == ["ROI1"]
    assert argv_preview["no_argv_list"] is True
    assert argv_preview["no_arg_ordering_decided"] is True
    assert argv_preview["no_path_quoting_decided"] is True
    assert output_preview["output_base"] == "C:/planned/output"
    assert output_preview["future_output_owner"] == "runner"
    assert output_preview["future_cli_target_concept"] == "out_base"
    assert output_preview["future_run_directory_strategy"] == "derive_unique_run_id_under_output_base"
    assert output_preview["future_run_dir"] == "unresolved_until_execution_start"
    assert output_preview["concrete_run_dir_known"] is False
    assert output_preview["preview_path_kind"] == "pattern_only"
    assert output_preview["overwrite"] is False
    assert output_preview["directory_creation_timing"] == "future_explicit_action_only"
    assert output_preview["validation_side_effects"] == "deferred_unresolved"
    assert output_preview["directory_created"] is False
    assert output_preview["directory_reserved"] is False


def test_guided_runner_request_identity_and_boundaries_are_preview_only():
    plan = _complete_new_analysis_plan_with_current_snapshot()

    runner_preview = build_guided_runner_request_preview(plan)
    identity = runner_preview.identity_preview
    validation = runner_preview.validation_boundary_preview
    execution = runner_preview.execution_boundary_preview
    provenance = runner_preview.provenance_preview

    assert identity["identity_inputs"]["source_setup_signature"] == "setup-1"
    assert identity["identity_inputs"]["mapping_preview_schema_version"] == FIRST_SUBSET_MAPPING_PREVIEW_SCHEMA_VERSION
    assert identity["identity_inputs"]["runner_request_preview_schema_version"] == RUNNER_REQUEST_PREVIEW_SCHEMA_VERSION
    assert identity["preview_identity_hash"]
    assert identity["preview_identity_hash_source"] == "in_memory_guided_preview_state_only"
    assert identity["config_hash_generated"] is False
    assert identity["argv_hash_generated"] is False
    assert identity["validation_identity_generated"] is False
    assert identity["run_identity_generated"] is False
    assert identity["no_file_hashes"] is True
    assert validation["validation_available"] is False
    assert validation["validation_namespace_designed"] is False
    assert validation["validation_artifact_ownership_designed"] is False
    assert validation["no_validation_subprocess"] is True
    assert execution["execution_available"] is False
    assert execution["explicit_run_action_implemented"] is False
    assert execution["production_run_id_final_directory_capture_implemented"] is False
    assert execution["no_pipeline_run"] is True
    assert provenance["diagnostic_cache_identity"] == "cache-1"
    assert provenance["execution_consumes_cache_artifacts"] is False
    assert provenance["evidence_chunks_are_production_scope"] is False
    assert provenance["decision_provenance_only"] is True


@pytest.mark.parametrize(
    ("case_name", "plan_overrides", "expected_category"),
    [
        ("npm", {"input_format": "npm"}, "unsupported_input_format_for_first_subset_mapping"),
        ("custom_tabular", {"input_format": "custom_tabular"}, "unsupported_input_format_for_first_subset_mapping"),
        ("auto", {"input_format": "auto"}, "unsupported_input_format_for_first_subset_mapping"),
        ("continuous", {"acquisition_mode": "continuous"}, "unsupported_acquisition_mode_for_first_subset_mapping"),
        ("signal_only_f0", {}, "correction_mapping_not_ready"),
        ("mixed_strategies", {}, "correction_mapping_not_ready"),
    ],
)
def test_guided_runner_request_unsupported_mapping_keeps_request_unavailable(
    case_name,
    plan_overrides,
    expected_category,
):
    if case_name == "signal_only_f0":
        plan_overrides = {
            "per_roi_correction_strategy_choices": [_strategy_choice("signal_only_f0")]
        }
    elif case_name == "mixed_strategies":
        plan_overrides = {
            "included_roi_ids": ["ROI1", "ROI2"],
            "excluded_roi_ids": [],
            "per_roi_correction_strategy_choices": [
                _strategy_choice("global_linear_regression", roi_id="ROI1"),
                _strategy_choice("robust_global_event_reject", roi_id="ROI2"),
            ],
        }
    plan = _complete_new_analysis_plan_with_current_snapshot(**plan_overrides)
    if plan_overrides.get("included_roi_ids") == ["ROI1", "ROI2"]:
        plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
            plan,
            contract_values={
                "input_format": "rwd",
                "resolved_input_format": "rwd",
                "acquisition_mode": "intermittent",
                "rwd_time_col": "Time",
                "sig_suffix": "_Signal",
                "uv_suffix": "_UV",
            },
        )

    runner_preview = build_guided_runner_request_preview(plan)

    assert runner_preview.runner_request_preview_available is False
    assert expected_category in runner_preview.blocking_issue_categories
    if case_name in {"signal_only_f0", "mixed_strategies"}:
        assert expected_category not in runner_preview.unsupported_reasons
    else:
        assert expected_category in runner_preview.unsupported_reasons
    assert runner_preview.no_real_runner_request is True
    assert runner_preview.no_runspec is True
    assert runner_preview.no_argv is True
    assert runner_preview.no_command_text is True
    assert runner_preview.no_config_dict is True
    assert runner_preview.no_file_writes is True
    assert runner_preview.no_directory_creation is True
    assert runner_preview.no_directory_reservation is True
    assert runner_preview.no_validation is True
    assert runner_preview.no_run is True


def test_guided_runner_request_preview_is_pure_no_files_and_no_plan_mutation(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_output_base"
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path=str(target))
    before_state = dict(plan.__dict__)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    runner_preview = build_guided_runner_request_preview(plan)

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after_files == before_files
    assert not target.exists()
    assert dict(plan.__dict__) == before_state
    assert runner_preview.no_real_runner_request is True
    assert runner_preview.no_runspec is True
    assert runner_preview.no_argv is True
    assert runner_preview.no_command_text is True
    assert runner_preview.no_config_dict is True
    assert runner_preview.no_config_serialization is True
    assert runner_preview.no_yaml is True
    assert runner_preview.no_json is True
    assert runner_preview.no_file_writes is True
    assert runner_preview.no_directory_creation is True
    assert runner_preview.no_directory_reservation is True
    assert runner_preview.no_validation is True
    assert runner_preview.no_run is True


def _strategy_choice(strategy: str, *, roi_id: str = "ROI1"):
    return GuidedPlanCorrectionChoice(
        roi_id=roi_id,
        selected_strategy=strategy,
        source_type="diagnostic_cache",
        diagnostic_cache_id="cache-1",
        diagnostic_cache_root="C:/cache",
        source_setup_signature="setup-1",
        diagnostic_scope_signature="scope-1",
        build_request_signature="build-1",
        current_or_stale="current",
        explicit_user_mark=True,
    )


@pytest.mark.parametrize(
    ("strategy", "active_parameter_set", "expected_active_key"),
    [
        ("robust_global_event_reject", "robust_event_rejection", "robust_event_reject_residual_z_thresh"),
        ("adaptive_event_gated_regression", "adaptive_event_gate", "adaptive_event_gate_smooth_window_sec"),
    ],
)
def test_execution_spec_preview_matching_robust_and_adaptive_contracts_are_ready(
    strategy,
    active_parameter_set,
    expected_active_key,
):
    plan = _complete_new_analysis_plan_with_current_snapshot(
        per_roi_correction_strategy_choices=[_strategy_choice(strategy)],
        dynamic_fit_parameter_contract=GuidedNewAnalysisDynamicFitParameterContract(
            dynamic_fit_mode=strategy
        ),
    )

    preview = build_guided_new_analysis_execution_spec_preview(plan)
    dyn_contract = preview.correction["dynamic_fit_parameter_contract"]

    assert preview.spec_preview_available is True
    assert preview.correction["selected_global_dynamic_fit_strategy"] == strategy
    assert dyn_contract["dynamic_fit_mode"] == strategy
    assert dyn_contract["selected_strategy"] == strategy
    assert dyn_contract["active_parameter_set"] == active_parameter_set
    assert expected_active_key in dyn_contract["active_parameters"]
    assert dyn_contract["backend_config_mapping_status"] == "label_and_parameters_ready_for_future_mapping"
    assert dyn_contract["execution_consumption_enabled"] is True
    assert preview.execution_available is False


def test_execution_spec_preview_dynamic_fit_contract_mismatch_blocks():
    plan = _complete_new_analysis_plan_with_current_snapshot(
        per_roi_correction_strategy_choices=[_strategy_choice("robust_global_event_reject")],
        dynamic_fit_parameter_contract=GuidedNewAnalysisDynamicFitParameterContract(
            dynamic_fit_mode="global_linear_regression"
        ),
    )

    preview = build_guided_new_analysis_execution_spec_preview(plan)
    dyn_contract = preview.correction["dynamic_fit_parameter_contract"]

    assert preview.spec_preview_available is True
    assert "dynamic_fit_parameter_contract_mismatch" not in preview.blocking_issue_categories
    assert preview.correction["selected_global_dynamic_fit_strategy"] == "robust_global_event_reject"
    assert dyn_contract["dynamic_fit_mode"] == "global_linear_regression"
    assert dyn_contract["selected_strategy"] == "robust_global_event_reject"
    assert dyn_contract["backend_config_mapping_status"] == "label_and_parameters_ready_for_future_mapping"
    assert dyn_contract["execution_consumption_enabled"] is True
    assert preview.execution_available is False


def test_execution_spec_preview_unresolved_dynamic_fit_parameter_contract_blocks():
    plan = _complete_new_analysis_plan_with_current_snapshot(
        dynamic_fit_parameter_contract=GuidedNewAnalysisDynamicFitParameterContract(
            unresolved_parameters=("legacy_global_settings",)
        ),
    )

    preview = build_guided_new_analysis_execution_spec_preview(plan)
    dyn_contract = preview.correction["dynamic_fit_parameter_contract"]

    assert preview.spec_preview_available is False
    assert "unresolved_dynamic_fit_parameter_contract" in preview.blocking_issue_categories
    assert dyn_contract["unresolved_parameters"] == ["legacy_global_settings"]
    assert dyn_contract["backend_config_mapping_status"] == "label_ready_parameters_unresolved"
    assert dyn_contract["execution_consumption_enabled"] is False


def test_execution_spec_preview_signal_only_f0_blocks_without_global_strategy():
    plan = _complete_new_analysis_plan_with_current_snapshot(
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

    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert preview.spec_preview_available is True
    assert "signal_only_f0_execution_not_supported" not in preview.blocking_issue_categories
    assert preview.correction["selected_global_dynamic_fit_strategy"] is None
    assert preview.correction["signal_only_f0_production_routing_supported"] is False
    assert "signal_only_f0_execution_not_supported" not in preview.correction["blocker_categories"]
    assert preview.execution_available is False


def test_execution_spec_preview_mixed_per_roi_strategies_block_without_collapse():
    plan = _complete_new_analysis_plan_with_current_snapshot(
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
    plan.dataset_contract_snapshot = _current_applied_snapshot_for_plan(
        plan,
        contract_values={
            "input_format": "rwd",
            "resolved_input_format": "rwd",
            "acquisition_mode": "intermittent",
            "rwd_time_col": "Time",
        },
    )

    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert preview.spec_preview_available is False
    assert "mixed_dynamic_fit_modes_execution_not_enabled" not in preview.blocking_issue_categories
    assert preview.correction["selected_global_dynamic_fit_strategy"] is None
    assert preview.correction["global_strategy_collapsed"] is False
    assert preview.correction["mixed_strategy_supported"] is False
    assert len(preview.correction["per_roi_choices"]) == 2


@pytest.mark.parametrize(
    ("snapshot", "expected_category"),
    [
        (GuidedNewAnalysisDatasetContractSnapshot(status="missing"), "missing_rwd_dataset_contract"),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="applied",
                input_format="rwd",
                resolved_input_format="rwd",
                acquisition_mode="intermittent",
                validation_issues=("missing rwd time column",),
                explicitly_applied=True,
            ),
            "invalid_dataset_contract_snapshot",
        ),
        (
            GuidedNewAnalysisDatasetContractSnapshot(
                status="applied",
                input_format="rwd",
                resolved_input_format="rwd",
                acquisition_mode="intermittent",
                stale_reasons=("input changed",),
                explicitly_applied=True,
            ),
            "stale_dataset_contract_snapshot",
        ),
    ],
)
def test_execution_spec_preview_dataset_contract_blockers(snapshot, expected_category):
    preview = build_guided_new_analysis_execution_spec_preview(
        _complete_new_analysis_plan(dataset_contract_snapshot=snapshot)
    )

    assert preview.spec_preview_available is False
    assert expected_category in preview.blocking_issue_categories
    assert preview.dataset_contract["execution_consumption_enabled"] is False
    assert preview.execution_available is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_event_profile_status": "missing", "feature_event_explicitly_applied": False},
        {"feature_event_profile_status": "default_initialized", "feature_event_explicitly_applied": False},
        {
            "feature_event_profile_status": "invalid",
            "feature_event_validation_issues": ["bad threshold"],
            "feature_event_explicitly_applied": False,
        },
        {
            "feature_event_profile_status": "stale",
            "feature_event_stale_reasons": ["baseline changed"],
            "feature_event_explicitly_applied": True,
        },
        {"feature_event_profile_status": "applied", "feature_event_explicitly_applied": False},
    ],
)
def test_execution_spec_preview_feature_event_blockers(overrides):
    plan = _complete_new_analysis_plan_with_current_snapshot(**overrides)

    preview = build_guided_new_analysis_execution_spec_preview(plan)

    assert preview.spec_preview_available is False
    assert "incomplete_planning_readiness" in preview.blocking_issue_categories
    assert any(
        category.startswith("planning_")
        and (
            "feature_event" in category
            or category == "planning_missing_feature_event_profile"
        )
        for category in preview.blocking_issue_categories
    )
    assert preview.feature_event["consumption"]["feature_event_profile_required"] is True
    assert preview.feature_event["consumption"]["feature_event_values_consumed"] is False
    assert preview.execution_available is False


def test_execution_spec_preview_unsafe_output_creation_policy_blocks_without_writes(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_output"
    plan = _complete_new_analysis_plan_with_current_snapshot(
        output_policy_path=str(target),
        output_creation_policy=GuidedNewAnalysisOutputCreationPolicy(
            overwrite=True,
            precreate_during_preview=True,
            gui_preflight_writes_enabled=True,
        ),
    )
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    preview = build_guided_new_analysis_execution_spec_preview(plan)

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before
    assert not target.exists()
    assert preview.spec_preview_available is False
    assert "unsafe_output_creation_policy" in preview.blocking_issue_categories
    assert preview.output["directory_created"] is False
    assert preview.output["files_written"] is False
    assert preview.output["config_written"] is False
    assert preview.output["command_written"] is False
    assert preview.output["validation_run"] is False
    assert preview.output["execution_started"] is False


def test_execution_spec_preview_is_pure_no_files_and_no_plan_mutation(tmp_path):
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_output"
    plan = _complete_new_analysis_plan_with_current_snapshot(output_policy_path=str(target))
    before_state = dict(plan.__dict__)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    preview = build_guided_new_analysis_execution_spec_preview(plan)

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after_files == before_files
    assert not target.exists()
    assert dict(plan.__dict__) == before_state
    assert preview.execution_available is False
    assert preview.provenance["no_gui_runspec"] is True
    assert preview.provenance["no_argv_generated"] is True
    assert preview.provenance["no_config_written"] is True
    assert preview.provenance["no_output_directory_created"] is True


def test_execution_spec_preview_rejects_malformed_input():
    with pytest.raises(TypeError, match="GuidedNewAnalysisDraftPlan"):
        build_guided_new_analysis_execution_spec_preview(object())


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


def test_guided_new_analysis_durable_fields_4J11i():
    # 1. Defaults
    plan = GuidedNewAnalysisDraftPlan()
    assert plan.output_base_path is None
    assert plan.output_overwrite is False
    assert plan.global_correction_strategy is None
    assert plan.dynamic_fit_mode is None

    # 2. Assignment
    plan_assigned = GuidedNewAnalysisDraftPlan(
        output_base_path="C:/some/output",
        output_overwrite=True,
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression"
    )
    assert plan_assigned.output_base_path == "C:/some/output"
    assert plan_assigned.output_overwrite is True
    assert plan_assigned.global_correction_strategy == "dynamic_fit"
    assert plan_assigned.dynamic_fit_mode == "global_linear_regression"

    # 3. Serialization simulation
    from dataclasses import asdict
    d = asdict(plan_assigned)
    assert d["output_base_path"] == "C:/some/output"
    assert d["output_overwrite"] is True
    assert d["global_correction_strategy"] == "dynamic_fit"
    assert d["dynamic_fit_mode"] == "global_linear_regression"

    # 4. Deserialization backward compatibility simulation
    # Older plan dict lacking new fields
    old_dict = {
        "schema_version": "guided_new_analysis_plan.v1",
        "mode": "new_analysis",
        "input_source_path": "C:/raw/input"
    }
    # Should load cleanly without crash, defaulting the missing fields
    restored = GuidedNewAnalysisDraftPlan(
        input_source_path=old_dict.get("input_source_path"),
        output_base_path=old_dict.get("output_base_path", None),
        output_overwrite=old_dict.get("output_overwrite", False),
        global_correction_strategy=old_dict.get("global_correction_strategy", None),
        dynamic_fit_mode=old_dict.get("dynamic_fit_mode", None)
    )
    assert restored.input_source_path == "C:/raw/input"
    assert restored.output_base_path is None
    assert restored.output_overwrite is False
    assert restored.global_correction_strategy is None
    assert restored.dynamic_fit_mode is None
