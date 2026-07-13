"""Determinism and coverage tests for the canonical Guided draft-plan identity.

This is the authoritative "current canonical plan identity" that
gui/main_window.py uses to detect stale validation without relying on a
manually maintained revision counter (see guided_plan_identity.py).
"""

from __future__ import annotations

import dataclasses

import pytest

from photometry_pipeline.guided_new_analysis_plan import (
    GuidedApprovedMissingSession,
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisDynamicFitParameterContract,
    GuidedNewAnalysisExecutionIntent,
    GuidedNewAnalysisOutputCreationPolicy,
    GuidedPlanCorrectionChoice,
    GuidedPlanFeatureEventChoice,
)
from photometry_pipeline.guided_plan_identity import (
    build_guided_new_analysis_draft_plan_identity_payload,
    compute_guided_new_analysis_draft_plan_identity,
)


def _base_plan(**overrides) -> GuidedNewAnalysisDraftPlan:
    defaults = dict(
        mode="new_analysis",
        input_source_path=r"C:\data\recording_1",
        resolved_input_source_path=r"C:\data\recording_1",
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=["CH3"],
        output_base_path=r"C:\data\output",
        global_correction_strategy="dynamic_fit",
        dynamic_fit_mode="robust_global_event_reject",
    )
    defaults.update(overrides)
    return GuidedNewAnalysisDraftPlan(**defaults)


def _correction_choice(roi_id, strategy="robust_global_event_reject", **overrides):
    defaults = dict(
        roi_id=roi_id,
        selected_strategy=strategy,
        source_type="local_correction_preview",
        current_or_stale="current",
        explicit_user_mark=True,
        evidence_reference={"evidence_source_type": "local_correction_preview"},
    )
    defaults.update(overrides)
    return GuidedPlanCorrectionChoice(**defaults)


def _feature_choice(roi_id, **overrides):
    defaults = dict(
        roi_id=roi_id,
        feature_event_profile_id="custom",
        config_fields={"peak_threshold_k": 8.0},
        current_or_stale="current",
        explicit_user_mark=True,
    )
    defaults.update(overrides)
    return GuidedPlanFeatureEventChoice(**defaults)


def _missing_session_approval(path, index, **overrides):
    defaults = dict(
        canonical_relative_path=path,
        size_bytes=1024,
        sha256_content_digest="a" * 64,
        session_index=index,
        expected_start_time="2026-01-01T00:00:00Z",
        expected_duration_sec=120.0,
        reason="could not be processed",
    )
    defaults.update(overrides)
    return GuidedApprovedMissingSession(**defaults)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_identical_plans_built_repeatedly_yield_identical_identity():
    plan_a = _base_plan()
    plan_b = _base_plan()
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)
    # Also stable across repeated computation on the *same* object.
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_a)


def test_dict_insertion_order_does_not_change_identity():
    plan_a = _base_plan(feature_event_values={"a": 1, "b": 2, "c": 3})
    plan_b = _base_plan(feature_event_values={"c": 3, "a": 1, "b": 2})
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_roi_ordering_is_canonicalized():
    plan_a = _base_plan(included_roi_ids=["CH1", "CH2"], excluded_roi_ids=["CH3"])
    plan_b = _base_plan(included_roi_ids=["CH2", "CH1"], excluded_roi_ids=["CH3"])
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_correction_choice_list_order_is_canonicalized():
    choices_a = [_correction_choice("CH1"), _correction_choice("CH2", "global_linear_regression")]
    choices_b = [_correction_choice("CH2", "global_linear_regression"), _correction_choice("CH1")]
    plan_a = _base_plan(per_roi_correction_strategy_choices=choices_a)
    plan_b = _base_plan(per_roi_correction_strategy_choices=choices_b)
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_missing_session_approval_list_order_is_canonicalized_but_session_index_is_not_resorted():
    a1 = _missing_session_approval("session_1/fluorescence.csv", index=0)
    a2 = _missing_session_approval("session_3/fluorescence.csv", index=2)
    plan_a = _base_plan(approved_missing_sessions=[a1, a2])
    plan_b = _base_plan(approved_missing_sessions=[a2, a1])
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)

    # A different session_index for the *same* path is a scientifically
    # meaningful difference and must not be silently sorted away.
    a1_wrong_index = _missing_session_approval("session_1/fluorescence.csv", index=1)
    plan_c = _base_plan(approved_missing_sessions=[a1_wrong_index, a2])
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_c)


def test_missing_session_expected_start_time_change_changes_identity():
    """Path, digest, size, and session index unchanged -- only the
    expected start time differs -- must still change identity."""
    a1 = _missing_session_approval(
        "session_1/fluorescence.csv", index=0,
        expected_start_time="2026-01-01T00:00:00Z",
    )
    a2 = _missing_session_approval(
        "session_1/fluorescence.csv", index=0,
        expected_start_time="2026-01-01T01:00:00Z",
    )
    plan_a = _base_plan(approved_missing_sessions=[a1])
    plan_b = _base_plan(approved_missing_sessions=[a2])
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_missing_session_expected_duration_change_changes_identity():
    """Path, digest, size, session index, and start time unchanged --
    only the expected duration differs -- must still change identity."""
    a1 = _missing_session_approval(
        "session_1/fluorescence.csv", index=0, expected_duration_sec=600.0,
    )
    a2 = _missing_session_approval(
        "session_1/fluorescence.csv", index=0, expected_duration_sec=900.0,
    )
    plan_a = _base_plan(approved_missing_sessions=[a1])
    plan_b = _base_plan(approved_missing_sessions=[a2])
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_channel_pairing_change_changes_identity():
    from photometry_pipeline.guided_new_analysis_plan import (
        GuidedNewAnalysisDatasetContractSnapshot,
    )

    plan_a = _base_plan(
        dataset_contract_snapshot=GuidedNewAnalysisDatasetContractSnapshot(
            contract_values={
                "rwd_time_col": "Time(s)",
                "uv_suffix": "-410",
                "sig_suffix": "-470",
            },
        )
    )
    plan_b = _base_plan(
        dataset_contract_snapshot=GuidedNewAnalysisDatasetContractSnapshot(
            contract_values={
                "rwd_time_col": "Time(s)",
                "uv_suffix": "-470",
                "sig_suffix": "-410",
            },
        )
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_source_manifest_signature_change_changes_identity():
    """A source-content change under the same root (a different
    source_setup_signature/config_fingerprint) cannot reuse old
    validation, even if every top-level plan field looks identical."""
    from photometry_pipeline.guided_new_analysis_plan import (
        GuidedNewAnalysisDatasetContractSnapshot,
        GuidedNewAnalysisDatasetContractSourceIdentity,
    )

    plan_a = _base_plan(
        dataset_contract_snapshot=GuidedNewAnalysisDatasetContractSnapshot(
            source_identity=GuidedNewAnalysisDatasetContractSourceIdentity(
                source_setup_signature="a" * 64,
                config_fingerprint="config-1",
            ),
        )
    )
    plan_b = _base_plan(
        dataset_contract_snapshot=GuidedNewAnalysisDatasetContractSnapshot(
            source_identity=GuidedNewAnalysisDatasetContractSourceIdentity(
                source_setup_signature="b" * 64,
                config_fingerprint="config-1",
            ),
        )
    )
    plan_c = _base_plan(
        dataset_contract_snapshot=GuidedNewAnalysisDatasetContractSnapshot(
            source_identity=GuidedNewAnalysisDatasetContractSourceIdentity(
                source_setup_signature="a" * 64,
                config_fingerprint="config-2",
            ),
        )
    )
    ids = {
        compute_guided_new_analysis_draft_plan_identity(plan)
        for plan in (plan_a, plan_b, plan_c)
    }
    assert len(ids) == 3


def test_path_normalization_is_deterministic():
    plan_a = _base_plan(
        input_source_path=r"C:\data\recording_1",
        resolved_input_source_path=r"C:\data\recording_1",
    )
    plan_b = _base_plan(
        input_source_path=r"c:\DATA\Recording_1",
        resolved_input_source_path=r"c:\DATA\Recording_1",
    )
    # Windows paths are casefolded by canonicalize_absolute_path.
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_equivalent_scalar_representations_do_not_mismatch():
    plan_a = _base_plan(sessions_per_hour=6, session_duration_sec=120.0)
    plan_b = _base_plan(sessions_per_hour=6, session_duration_sec=120.0)
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_no_timestamp_or_cache_id_or_object_repr_contaminates_identity():
    plan_a = _base_plan()
    plan_b = dataclasses.replace(
        plan_a,
        cache_id="cache-" + "1" * 20,
        cache_root_path=r"C:\cache\somewhere",
        created_at_utc="2026-01-01T00:00:00Z",
        updated_at_utc="2026-06-01T00:00:00Z",
        request_json_path=r"C:\cache\request.json",
        provenance_path=r"C:\cache\provenance.json",
        warnings=["some warning"],
        blocking_issues=["some issue"],
        stale_or_current="stale",
        stale_reasons=["setup changed"],
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)


# ---------------------------------------------------------------------------
# Meaningful changes DO change identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    (
        {"input_source_path": r"C:\data\recording_2", "resolved_input_source_path": r"C:\data\recording_2"},
        {"input_format": "npm"},
        {"acquisition_mode": "continuous"},
        {"sessions_per_hour": 4},
        {"session_duration_sec": 90.0},
        {"included_roi_ids": ["CH1"]},
        {"excluded_roi_ids": ["CH2", "CH3"]},
        {"global_correction_strategy": "signal_only_f0", "dynamic_fit_mode": None},
        {"dynamic_fit_mode": "global_linear_regression"},
        {"output_base_path": r"C:\data\output_2"},
        {"exclude_incomplete_final_rwd_chunk": True},
        {"feature_event_profile_id": "custom_profile"},
        {"feature_event_values": {"peak_threshold_k": 12.0}},
    ),
)
def test_meaningful_field_change_changes_identity(overrides):
    plan_a = _base_plan()
    plan_b = _base_plan(**overrides)
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_per_roi_correction_choice_change_changes_identity():
    plan_a = _base_plan(
        per_roi_correction_strategy_choices=[_correction_choice("CH1")]
    )
    plan_b = _base_plan(
        per_roi_correction_strategy_choices=[
            _correction_choice("CH1", "signal_only_f0")
        ]
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_per_roi_feature_override_change_changes_identity():
    plan_a = _base_plan(per_roi_feature_event_choices=[_feature_choice("CH1")])
    plan_b = _base_plan(
        per_roi_feature_event_choices=[
            _feature_choice("CH1", config_fields={"peak_threshold_k": 20.0})
        ]
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_removing_a_per_roi_feature_override_changes_identity():
    plan_a = _base_plan(
        per_roi_feature_event_choices=[_feature_choice("CH1"), _feature_choice("CH2")]
    )
    plan_b = _base_plan(per_roi_feature_event_choices=[_feature_choice("CH1")])
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_approved_missing_session_set_change_changes_identity():
    plan_a = _base_plan(approved_missing_sessions=[])
    plan_b = _base_plan(
        approved_missing_sessions=[_missing_session_approval("session_1/fluorescence.csv", 0)]
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_execution_mode_change_changes_identity():
    plan_a = _base_plan(
        execution_intent=GuidedNewAnalysisExecutionIntent(execution_mode="phasic")
    )
    plan_b = _base_plan(
        execution_intent=GuidedNewAnalysisExecutionIntent(execution_mode="tonic")
    )
    plan_c = _base_plan(
        execution_intent=GuidedNewAnalysisExecutionIntent(execution_mode="both")
    )
    ids = {
        compute_guided_new_analysis_draft_plan_identity(plan)
        for plan in (plan_a, plan_b, plan_c)
    }
    assert len(ids) == 3


def test_dynamic_fit_parameter_change_changes_identity():
    plan_a = _base_plan(
        dynamic_fit_parameter_contract=GuidedNewAnalysisDynamicFitParameterContract(
            dynamic_fit_mode="robust_global_event_reject",
            robust_event_reject_max_iters=5,
        )
    )
    plan_b = _base_plan(
        dynamic_fit_parameter_contract=GuidedNewAnalysisDynamicFitParameterContract(
            dynamic_fit_mode="robust_global_event_reject",
            robust_event_reject_max_iters=9,
        )
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_output_overwrite_change_changes_identity():
    plan_a = _base_plan(
        output_creation_policy=GuidedNewAnalysisOutputCreationPolicy(overwrite=False)
    )
    plan_b = _base_plan(
        output_creation_policy=GuidedNewAnalysisOutputCreationPolicy(overwrite=True)
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) != compute_guided_new_analysis_draft_plan_identity(plan_b)


# ---------------------------------------------------------------------------
# Irrelevant UI state does NOT change identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    (
        {"cache_id": "some-cache-id"},
        {"cache_root_path": r"C:\cache\root"},
        {"artifact_record_path": r"C:\cache\artifact.json"},
        {"request_json_path": r"C:\cache\request.json"},
        {"provenance_path": r"C:\cache\provenance.json"},
        {"phasic_trace_cache_path": r"C:\cache\phasic_trace_cache.h5"},
        {"config_used_path": r"C:\cache\config_used.yaml"},
        {"source_setup_signature": "sig-1"},
        {"diagnostic_scope_signature": "sig-2"},
        {"build_request_signature": "sig-3"},
        {"stale_or_current": "stale"},
        {"stale_reasons": ("setup changed",)},
        {"warnings": ["a warning"]},
        {"blocking_issues": ["a blocking issue"]},
        {"informational_issues": ["an info issue"]},
        {"completeness_readiness_summary": "some summary"},
        {"execution_ready": True},
        {"executable": True},
        {"production_run_enabled": True},
        {"created_at_utc": "2026-01-01T00:00:00Z"},
        {"updated_at_utc": "2026-06-01T00:00:00Z"},
        {"correction_preview_result_id": "preview-1"},
        {"correction_preview_path": r"C:\cache\preview"},
        {"correction_preview_status": "current"},
        {"signal_only_f0_result_id": "signal-1"},
        {"signal_only_f0_path": r"C:\cache\signal_only"},
        {"feature_event_profile_status": "applied"},
        {"feature_event_updated_at_utc": "2026-01-01T00:00:00Z"},
        {"output_policy_status": "applied"},
        {"output_policy_updated_at_utc": "2026-01-01T00:00:00Z"},
        {"output_policy_safety_summary": "some safety text"},
    ),
)
def test_display_only_field_change_does_not_change_identity(overrides):
    plan_a = _base_plan()
    plan_b = dataclasses.replace(plan_a, **overrides)
    assert compute_guided_new_analysis_draft_plan_identity(
        plan_a
    ) == compute_guided_new_analysis_draft_plan_identity(plan_b)


def test_payload_is_a_plain_json_serializable_structure():
    payload = build_guided_new_analysis_draft_plan_identity_payload(_base_plan())
    import json

    # Must round-trip through JSON without error -- no object repr, no
    # non-finite floats, no non-string dict keys.
    json.dumps(payload)


def test_rejects_non_draft_plan_input():
    with pytest.raises(TypeError):
        compute_guided_new_analysis_draft_plan_identity(object())
