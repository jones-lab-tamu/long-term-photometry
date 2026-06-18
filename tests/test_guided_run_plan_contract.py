import sys

import pytest

from photometry_pipeline.feature_event_config import validate_feature_event_config_fields
from photometry_pipeline.guided_run_plan import (
    CorrectionStrategyChoice,
    EvidenceChunkReview,
    FeatureEventProfile,
    GuidedPlanSource,
    GuidedRunPlan,
    GuidedRunPlanContractError,
    OutputPolicy,
    PlanProvenanceFlags,
    RoiPlanEntry,
    assert_valid_plan_contract,
    deserialize_plan_from_dict,
    evaluate_guided_plan_checklist,
    feature_event_profile_summary_lines,
    is_runnable_correction_strategy,
    serialize_plan_to_dict,
    validate_correction_strategy,
    validate_plan_contract,
)


def _valid_plan() -> GuidedRunPlan:
    return GuidedRunPlan(
        plan_id="plan-1",
        mode="completed_run_planning",
        source=GuidedPlanSource(
            source_mode="completed_run",
            completed_run_dir="C:/runs/example",
            phasic_out_dir="C:/runs/example/_analysis/phasic_out",
        ),
        roi_plan=[
            RoiPlanEntry(
                roi="CH1",
                correction_strategy=CorrectionStrategyChoice(
                    strategy="robust_global_event_reject",
                    strategy_label="Robust Global Event-Reject Fit",
                ),
                evidence=[
                    EvidenceChunkReview(
                        chunk_id=0,
                        preview_artifact_paths=["C:/runs/example/_guided_workflow/previews/p1/summary.json"],
                    )
                ],
            ),
            RoiPlanEntry(
                roi="CH2",
                correction_strategy=CorrectionStrategyChoice(
                    strategy="signal_only_f0",
                    strategy_label="Signal-Only F0",
                ),
                evidence=[
                    EvidenceChunkReview(
                        chunk_id=3,
                        diagnostic_artifact_paths=[
                            "C:/runs/example/_guided_workflow/signal_only_f0_diagnostics/d1/summary.json"
                        ],
                    )
                ],
            ),
        ],
        feature_event_profiles=[
            FeatureEventProfile(
                profile_id="default",
                scope="run",
                config_fields={
                    "event_signal": "dff",
                    "peak_threshold_method": "mean_std",
                    "peak_threshold_k": 2.5,
                },
                evidence_previews=[EvidenceChunkReview(chunk_id=0, summary="preview only")],
            )
        ],
    )


def _checklist_items(plan: GuidedRunPlan | None, errors: list[str] | None = None):
    return evaluate_guided_plan_checklist(plan, errors).item_by_key()


def _valid_feature_event_config() -> dict:
    return {
        "event_signal": "dff",
        "signal_excursion_polarity": "positive",
        "peak_threshold_method": "mean_std",
        "peak_threshold_k": 2.5,
        "peak_threshold_percentile": 95.0,
        "peak_min_distance_sec": 1.0,
        "peak_min_prominence_k": 2.0,
        "peak_min_width_sec": 0.3,
        "peak_pre_filter": "none",
        "event_auc_baseline": "zero",
    }


def _valid_feature_event_profile(**overrides) -> FeatureEventProfile:
    values = {
        "profile_id": "default",
        "profile_label": "Default feature detection",
        "scope": "run",
        "config_fields": _valid_feature_event_config(),
        "evidence_previews": [EvidenceChunkReview(chunk_id=0, summary="preview only")],
        "choice_source": "explicit_user_profile_edit",
        "status": "draft",
        "target_rois": [],
        "resolved_rois": [],
        "provenance_references": ["preview://feature/default"],
    }
    values.update(overrides)
    return FeatureEventProfile(**values)


def test_round_trip_serialization_preserves_roi_level_correction_choices():
    plan = _valid_plan()

    payload = serialize_plan_to_dict(plan)
    restored = deserialize_plan_from_dict(payload)

    assert validate_plan_contract(restored) == []
    by_roi = {entry.roi: entry for entry in restored.roi_plan}
    assert set(by_roi) == {"CH1", "CH2"}
    assert by_roi["CH1"].correction_strategy.strategy == "robust_global_event_reject"
    assert by_roi["CH2"].correction_strategy.strategy == "signal_only_f0"
    assert by_roi["CH1"].evidence[0].chunk_id == 0
    assert by_roi["CH2"].evidence[0].chunk_id == 3


def test_checklist_for_none_plan_is_blocked():
    checklist = evaluate_guided_plan_checklist(None)
    items = checklist.item_by_key()

    assert items["source"].status == "not_configured"
    assert items["contract"].status == "fail"
    assert items["execution"].status == "blocked"
    assert checklist.execution_ready is False


def test_checklist_for_valid_plan_with_strategy_and_evidence():
    plan = _valid_plan()
    plan.feature_event_profiles = []
    items = _checklist_items(plan)
    checklist = evaluate_guided_plan_checklist(plan)

    assert items["contract"].status == "pass"
    assert items["roi_choices"].status == "pass"
    assert items["evidence"].status == "pass"
    assert items["feature_event"].status == "not_configured"
    assert items["output_destination"].status == "not_configured"
    assert items["execution"].status == "blocked"
    assert checklist.execution_ready is False


def test_checklist_for_valid_plan_with_no_roi_choices():
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir="C:/runs/example"),
        roi_plan=[RoiPlanEntry(roi="CH1")],
    )

    items = _checklist_items(plan)

    assert items["contract"].status == "pass"
    assert items["roi_choices"].status == "not_configured"
    assert items["execution"].status == "blocked"


def test_checklist_with_contract_errors_reports_blocking_messages():
    plan = _valid_plan()
    plan.roi_plan[0].correction_strategy = CorrectionStrategyChoice(strategy="auto")
    errors = validate_plan_contract(plan)
    checklist = evaluate_guided_plan_checklist(plan, errors)
    items = checklist.item_by_key()

    assert items["contract"].status == "fail"
    assert any("forbidden runnable correction strategy: auto" in msg for msg in checklist.blocking_messages)
    assert items["execution"].status == "blocked"
    assert checklist.execution_ready is False


def test_checklist_warns_for_roi_strategy_without_evidence():
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir="C:/runs/example"),
        roi_plan=[
            RoiPlanEntry(
                roi="CH1",
                correction_strategy=CorrectionStrategyChoice(strategy="signal_only_f0"),
                evidence=[],
            )
        ],
    )

    items = _checklist_items(plan)

    assert items["roi_choices"].status == "pass"
    assert items["evidence"].status == "warning"
    assert items["execution"].status == "blocked"


def test_checklist_fails_when_non_execution_provenance_flags_are_false():
    plan = _valid_plan()
    plan.provenance = PlanProvenanceFlags(no_pipeline_execution=False, no_applied_dff_outputs=False)
    items = _checklist_items(plan)

    assert items["non_execution"].status == "fail"
    assert items["execution"].status == "blocked"


def test_checklist_feature_event_profile_readiness():
    no_profiles = _valid_plan()
    no_profiles.feature_event_profiles = []
    assert _checklist_items(no_profiles)["feature_event"].status == "not_configured"

    valid_profile = _valid_plan()
    valid_profile.feature_event_profiles = [_valid_feature_event_profile()]
    assert _checklist_items(valid_profile)["feature_event"].status == "pass"

    invalid_profile = _valid_plan()
    invalid_profile.feature_event_profiles = [_valid_feature_event_profile()]
    invalid_profile.feature_event_profiles[0].config_fields["unknown_feature_setting"] = True
    errors = validate_plan_contract(invalid_profile)
    items = _checklist_items(invalid_profile, errors)
    assert items["contract"].status == "fail"
    assert items["feature_event"].status == "fail"


def test_valid_run_level_feature_event_profile_passes_and_execution_remains_blocked():
    plan = _valid_plan()
    plan.feature_event_profiles = [_valid_feature_event_profile()]

    errors = validate_plan_contract(plan)
    checklist = evaluate_guided_plan_checklist(plan, errors)
    items = checklist.item_by_key()

    assert errors == []
    assert items["feature_event"].status == "pass"
    assert checklist.execution_ready is False
    assert items["execution"].status == "blocked"


def test_feature_event_profile_summary_for_no_profiles():
    plan = _valid_plan()
    plan.feature_event_profiles = []

    lines = feature_event_profile_summary_lines(plan)
    items = _checklist_items(plan)

    assert lines == ["Feature/event profiles: none configured"]
    assert items["feature_event"].status == "not_configured"


def test_feature_event_profile_summary_for_valid_profile():
    plan = _valid_plan()
    plan.feature_event_profiles = [
        _valid_feature_event_profile(
            profile_id="roi-profile",
            profile_label="ROI CH1 feature profile",
            scope="roi",
            status="complete",
            resolved_rois=["CH1"],
            evidence_previews=[EvidenceChunkReview(chunk_id=4)],
        )
    ]

    lines = feature_event_profile_summary_lines(plan)

    assert len(lines) == 1
    assert "roi-profile" in lines[0]
    assert "ROI CH1 feature profile" in lines[0]
    assert "scope=roi" in lines[0]
    assert "status=complete" in lines[0]
    assert "resolved_rois=CH1" in lines[0]
    assert "config_fields=10" in lines[0]
    assert "event_signal" in lines[0]
    assert "evidence preview chunks=4" in lines[0]


def test_feature_event_profile_summary_does_not_execute_or_write(tmp_path):
    plan = _valid_plan()
    plan.feature_event_profiles = [_valid_feature_event_profile()]
    before = sorted(tmp_path.rglob("*"))

    lines = feature_event_profile_summary_lines(plan)
    errors = validate_plan_contract(plan)
    payload = serialize_plan_to_dict(plan)
    restored = deserialize_plan_from_dict(payload)

    assert lines
    assert errors == []
    assert validate_plan_contract(restored) == []
    assert sorted(tmp_path.rglob("*")) == before
    assert "photometry_pipeline.core.feature_extraction" not in sys.modules
    assert not (tmp_path / "features.csv").exists()
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "MANIFEST.json").exists()


def test_invalid_feature_event_profile_summary_still_displays_and_checklist_fails():
    plan = _valid_plan()
    plan.feature_event_profiles = [
        _valid_feature_event_profile(profile_id="bad-profile", scope="chunk")
    ]
    errors = validate_plan_contract(plan)
    lines = feature_event_profile_summary_lines(plan)
    items = _checklist_items(plan, errors)

    assert any("bad-profile" in line for line in lines)
    assert any("scope=chunk" in line for line in lines)
    assert items["contract"].status == "fail"
    assert items["feature_event"].status == "fail"
    assert items["execution"].status == "blocked"


def test_duplicate_feature_event_profile_id_is_rejected():
    plan = _valid_plan()
    plan.feature_event_profiles = [
        _valid_feature_event_profile(profile_id="events"),
        _valid_feature_event_profile(profile_id="events"),
    ]

    errors = validate_plan_contract(plan)

    assert any("duplicate feature_event profile_id: events" in err for err in errors)


def test_checklist_output_destination_readiness_does_not_touch_filesystem(tmp_path):
    plan = _valid_plan()
    output_root = tmp_path / "future_output"
    plan.output_policy = OutputPolicy(output_root=str(output_root))

    items = _checklist_items(plan)

    assert items["output_destination"].status == "pass"
    assert not output_root.exists()

    unsafe_policy = _valid_plan()
    unsafe_policy.output_policy = OutputPolicy(
        output_root=str(output_root),
        separate_from_source_required=False,
    )
    assert _checklist_items(unsafe_policy)["output_destination"].status == "fail"


def test_evidence_chunk_is_not_production_scope():
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir="C:/runs/example"),
        roi_plan=[
            RoiPlanEntry(
                roi="CH1",
                correction_strategy=CorrectionStrategyChoice(strategy="global_linear_regression"),
                evidence=[EvidenceChunkReview(chunk_id=0), EvidenceChunkReview(chunk_id=1)],
            )
        ],
    )

    assert validate_plan_contract(plan) == []
    payload = serialize_plan_to_dict(plan)
    assert len(payload["roi_plan"]) == 1
    assert payload["roi_plan"][0]["roi"] == "CH1"
    assert "chunk_id" not in payload["roi_plan"][0]
    assert [item["chunk_id"] for item in payload["roi_plan"][0]["evidence"]] == [0, 1]


@pytest.mark.parametrize("strategy", ["auto", "needs_review", "no_correction"])
def test_forbidden_strategies_are_rejected(strategy):
    assert is_runnable_correction_strategy(strategy) is False
    with pytest.raises(GuidedRunPlanContractError):
        validate_correction_strategy(strategy)

    plan = _valid_plan()
    plan.roi_plan[0].correction_strategy = CorrectionStrategyChoice(strategy=strategy)
    errors = validate_plan_contract(plan)
    assert any("forbidden runnable correction strategy" in err for err in errors)


def test_signal_only_f0_is_allowed_only_as_explicit_user_mark():
    plan = _valid_plan()
    assert validate_plan_contract(plan) == []

    plan.roi_plan[1].correction_strategy = CorrectionStrategyChoice(
        strategy="signal_only_f0",
        choice_source="diagnostic_success",
        no_auto_selection=True,
    )
    errors = validate_plan_contract(plan)
    assert any("choice_source must be explicit_user_mark" in err for err in errors)

    plan.roi_plan[1].correction_strategy = CorrectionStrategyChoice(
        strategy="signal_only_f0",
        choice_source="explicit_user_mark",
        no_auto_selection=False,
    )
    errors = validate_plan_contract(plan)
    assert any("no_auto_selection must be true" in err for err in errors)


def test_feature_event_profile_scope_rejects_chunk_but_allows_evidence_preview_chunks():
    for scope in ("run", "roi", "selected_roi_group"):
        plan = _valid_plan()
        plan.feature_event_profiles = [_valid_feature_event_profile(scope=scope)]
        if scope == "roi":
            plan.feature_event_profiles[0].resolved_rois = ["CH1"]
        assert validate_plan_contract(plan) == []

    plan = _valid_plan()
    plan.feature_event_profiles = [
        _valid_feature_event_profile(scope="chunk", evidence_previews=[EvidenceChunkReview(chunk_id=2)])
    ]
    errors = validate_plan_contract(plan)
    assert any("invalid scope: chunk" in err for err in errors)
    assert not any("chunk_id" in err for err in errors)
    assert _checklist_items(plan, errors)["feature_event"].status == "fail"


def test_feature_event_profile_unknown_config_fields_are_rejected():
    plan = _valid_plan()
    plan.feature_event_profiles = [_valid_feature_event_profile()]
    plan.feature_event_profiles[0].config_fields["new_detector_threshold"] = 1.0

    errors = validate_plan_contract(plan)

    assert any("unknown config fields" in err for err in errors)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("event_signal", "raw", "invalid event_signal"),
        ("signal_excursion_polarity", "upward", "invalid signal_excursion_polarity"),
        ("peak_threshold_method", "magic", "invalid peak_threshold_method"),
        ("peak_threshold_k", 0.0, "peak_threshold_k must be > 0"),
        ("peak_threshold_k", -1.0, "peak_threshold_k must be > 0"),
        ("peak_threshold_percentile", -0.1, "peak_threshold_percentile must be >= 0"),
        ("peak_threshold_percentile", 100.1, "peak_threshold_percentile must be <= 100"),
        ("peak_min_distance_sec", -1.0, "peak_min_distance_sec must be >= 0"),
        ("peak_min_prominence_k", -1.0, "peak_min_prominence_k must be >= 0"),
        ("peak_min_width_sec", -1.0, "peak_min_width_sec must be >= 0"),
        ("peak_pre_filter", "smooth", "invalid peak_pre_filter"),
        ("event_auc_baseline", "mean", "invalid event_auc_baseline"),
    ],
)
def test_feature_event_profile_invalid_config_values_are_rejected(field, value, message):
    plan = _valid_plan()
    cfg = _valid_feature_event_config()
    cfg[field] = value
    plan.feature_event_profiles = [_valid_feature_event_profile(config_fields=cfg)]

    errors = validate_plan_contract(plan)

    assert any(message in err for err in errors)


def test_feature_event_profile_validation_uses_shared_feature_event_semantics():
    plan = _valid_plan()
    cfg = _valid_feature_event_config()
    cfg["peak_pre_filter"] = "smooth"
    plan.feature_event_profiles = [_valid_feature_event_profile(config_fields=cfg)]
    shared_issue = validate_feature_event_config_fields(cfg)[0]

    errors = validate_plan_contract(plan)

    assert shared_issue == "invalid peak_pre_filter: smooth"
    assert any(shared_issue in err for err in errors)


def test_feature_event_profile_absolute_threshold_requires_positive_abs():
    plan = _valid_plan()
    cfg = _valid_feature_event_config()
    cfg["peak_threshold_method"] = "absolute"
    cfg.pop("peak_threshold_abs", None)
    plan.feature_event_profiles = [_valid_feature_event_profile(config_fields=cfg)]
    errors = validate_plan_contract(plan)
    assert any("peak_threshold_abs is required" in err for err in errors)

    cfg["peak_threshold_abs"] = 0.0
    plan.feature_event_profiles = [_valid_feature_event_profile(config_fields=cfg)]
    errors = validate_plan_contract(plan)
    assert any("peak_threshold_abs must be > 0" in err for err in errors)


@pytest.mark.parametrize("source", ["diagnostic_success", "preview_success", "auto"])
def test_feature_event_profile_automatic_choice_source_is_rejected(source):
    plan = _valid_plan()
    plan.feature_event_profiles = [_valid_feature_event_profile(choice_source=source)]

    errors = validate_plan_contract(plan)

    assert any("feature/event choice_source must be explicit_user_profile_edit" in err for err in errors)


@pytest.mark.parametrize("status", ["needs_review", "ready", ""])
def test_feature_event_profile_invalid_status_is_rejected(status):
    plan = _valid_plan()
    plan.feature_event_profiles = [_valid_feature_event_profile(status=status)]

    errors = validate_plan_contract(plan)

    assert any("invalid status" in err for err in errors)


def test_feature_event_profile_roi_scope_requires_single_roi_when_rois_are_provided():
    one_roi = _valid_plan()
    one_roi.feature_event_profiles = [
        _valid_feature_event_profile(scope="roi", resolved_rois=["CH1"], target_rois=["CH1"])
    ]
    assert validate_plan_contract(one_roi) == []

    multi_resolved = _valid_plan()
    multi_resolved.feature_event_profiles = [
        _valid_feature_event_profile(scope="roi", resolved_rois=["CH1", "CH2"])
    ]
    assert any("roi scope resolved_rois must contain exactly one ROI" in err for err in validate_plan_contract(multi_resolved))

    group = _valid_plan()
    group.feature_event_profiles = [
        _valid_feature_event_profile(scope="selected_roi_group", resolved_rois=["CH1", "CH2"])
    ]
    assert validate_plan_contract(group) == []


def test_feature_event_evidence_preview_chunk_is_provenance_only():
    plan = _valid_plan()
    plan.feature_event_profiles = [
        _valid_feature_event_profile(scope="run", evidence_previews=[EvidenceChunkReview(chunk_id=0)])
    ]

    assert validate_plan_contract(plan) == []
    payload = serialize_plan_to_dict(plan)
    profile = payload["feature_event_profiles"][0]
    assert profile["scope"] == "run"
    assert "chunk_id" not in profile
    assert profile["evidence_previews"][0]["chunk_id"] == 0


def test_feature_event_profile_round_trip_preserves_profile_fields():
    plan = _valid_plan()
    plan.feature_event_profiles = [
        _valid_feature_event_profile(
            profile_id="roi-profile",
            profile_label="ROI CH1 feature profile",
            scope="roi",
            target_rois=["CH1"],
            resolved_rois=["CH1"],
            choice_source="explicit_user_profile_edit",
            status="complete",
            provenance_references=["preview://feature/roi-profile"],
        )
    ]

    restored = deserialize_plan_from_dict(serialize_plan_to_dict(plan))
    profile = restored.feature_event_profiles[0]

    assert validate_plan_contract(restored) == []
    assert profile.profile_id == "roi-profile"
    assert profile.profile_label == "ROI CH1 feature profile"
    assert profile.scope == "roi"
    assert profile.target_rois == ["CH1"]
    assert profile.resolved_rois == ["CH1"]
    assert profile.config_fields == _valid_feature_event_config()
    assert profile.evidence_previews[0].chunk_id == 0
    assert profile.choice_source == "explicit_user_profile_edit"
    assert profile.status == "complete"
    assert profile.provenance_references == ["preview://feature/roi-profile"]


def test_source_mode_boundary_is_explicit():
    raw_plan = GuidedRunPlan(
        mode="new_analysis",
        source=GuidedPlanSource(source_mode="raw_input", raw_input_dir="C:/raw"),
        roi_plan=[RoiPlanEntry(roi="CH1")],
    )
    assert validate_plan_contract(raw_plan) == []

    completed_plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir="C:/runs/example"),
        roi_plan=[RoiPlanEntry(roi="CH1")],
    )
    assert validate_plan_contract(completed_plan) == []

    missing_raw = GuidedRunPlan(
        mode="new_analysis",
        source=GuidedPlanSource(source_mode="raw_input"),
        roi_plan=[RoiPlanEntry(roi="CH1")],
    )
    assert any("raw_input source requires raw_input_dir" in err for err in validate_plan_contract(missing_raw))

    missing_completed = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run"),
        roi_plan=[RoiPlanEntry(roi="CH1")],
    )
    assert any(
        "completed_run source requires completed_run_dir" in err
        for err in validate_plan_contract(missing_completed)
    )

    mismatched = GuidedRunPlan(
        mode="new_analysis",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir="C:/runs/example"),
        roi_plan=[RoiPlanEntry(roi="CH1")],
    )
    assert any("new_analysis plans must use source_mode raw_input" in err for err in validate_plan_contract(mismatched))

    explicit_with_both_paths = GuidedRunPlan(
        mode="new_analysis",
        source=GuidedPlanSource(
            source_mode="raw_input",
            raw_input_dir="C:/raw",
            completed_run_dir="C:/runs/reference_for_review",
        ),
        roi_plan=[RoiPlanEntry(roi="CH1")],
    )
    assert validate_plan_contract(explicit_with_both_paths) == []


def test_non_execution_guarantee(tmp_path):
    plan = _valid_plan()
    before = sorted(tmp_path.rglob("*"))

    assert_valid_plan_contract(plan)
    payload = serialize_plan_to_dict(plan)
    restored = deserialize_plan_from_dict(payload)
    assert validate_plan_contract(restored) == []

    after = sorted(tmp_path.rglob("*"))
    assert after == before
    assert not (tmp_path / "MANIFEST.json").exists()
    assert not (tmp_path / "manifest.csv").exists()
    assert not (tmp_path / "applied_dff").exists()
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "features.csv").exists()
    assert not (tmp_path / "validation").exists()
    assert "photometry_pipeline.core.feature_extraction" not in sys.modules
    assert "photometry_pipeline.pipeline" not in sys.modules


def test_current_selection_firewall_contract():
    plan = _valid_plan()
    before = serialize_plan_to_dict(plan)

    current_roi = "CH2"
    current_evidence_chunk = 99
    assert current_roi == "CH2"
    assert current_evidence_chunk == 99

    after = serialize_plan_to_dict(plan)
    assert after == before
    by_roi = {entry["roi"]: entry for entry in after["roi_plan"]}
    assert by_roi["CH1"]["correction_strategy"]["strategy"] == "robust_global_event_reject"
    assert by_roi["CH1"]["evidence"][0]["chunk_id"] == 0


def test_duplicate_roi_entries_fail_validation():
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir="C:/runs/example"),
        roi_plan=[RoiPlanEntry(roi="CH1"), RoiPlanEntry(roi="CH1")],
    )

    errors = validate_plan_contract(plan)

    assert any("duplicate ROI plan entry: CH1" in err for err in errors)


def test_serialized_non_dict_roi_plan_item_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["roi_plan"].append("CH3")

    with pytest.raises(GuidedRunPlanContractError, match="roi_plan item must be an object"):
        deserialize_plan_from_dict(payload)


def test_serialized_non_dict_evidence_item_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["roi_plan"][0]["evidence"].append("chunk 2")

    with pytest.raises(GuidedRunPlanContractError, match="evidence item must be an object"):
        deserialize_plan_from_dict(payload)


def test_serialized_missing_evidence_chunk_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    del payload["roi_plan"][0]["evidence"][0]["chunk_id"]

    with pytest.raises(GuidedRunPlanContractError, match="chunk_id is required"):
        deserialize_plan_from_dict(payload)


def test_serialized_non_integer_evidence_chunk_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["roi_plan"][0]["evidence"][0]["chunk_id"] = "bad"

    with pytest.raises(GuidedRunPlanContractError, match="chunk_id must be an integer"):
        deserialize_plan_from_dict(payload)


def test_serialized_negative_evidence_chunk_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["roi_plan"][0]["evidence"][0]["chunk_id"] = -1

    with pytest.raises(GuidedRunPlanContractError, match="chunk_id must be non-negative"):
        deserialize_plan_from_dict(payload)


def test_manually_constructed_invalid_evidence_chunks_fail_validation():
    missing = _valid_plan()
    missing.roi_plan[0].evidence = [EvidenceChunkReview(chunk_id=None)]  # type: ignore[arg-type]
    assert any("chunk_id is required" in err for err in validate_plan_contract(missing))

    non_integer = _valid_plan()
    non_integer.roi_plan[0].evidence = [EvidenceChunkReview(chunk_id="bad")]  # type: ignore[arg-type]
    assert any("chunk_id must be an integer" in err for err in validate_plan_contract(non_integer))

    negative = _valid_plan()
    negative.roi_plan[0].evidence = [EvidenceChunkReview(chunk_id=-1)]
    assert any("chunk_id must be non-negative" in err for err in validate_plan_contract(negative))


def test_serialized_malformed_feature_event_profile_item_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["feature_event_profiles"].append(["default"])

    with pytest.raises(
        GuidedRunPlanContractError,
        match="feature_event_profiles item must be an object",
    ):
        deserialize_plan_from_dict(payload)


def test_serialized_malformed_feature_event_evidence_preview_item_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["feature_event_profiles"][0]["evidence_previews"].append("chunk 3")

    with pytest.raises(GuidedRunPlanContractError, match="evidence item must be an object"):
        deserialize_plan_from_dict(payload)


def test_serialized_malformed_correction_strategy_structure_is_rejected():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["roi_plan"][0]["correction_strategy"] = "signal_only_f0"

    with pytest.raises(GuidedRunPlanContractError, match="correction_strategy must be an object"):
        deserialize_plan_from_dict(payload)


def test_serialized_forbidden_and_automatic_correction_strategy_still_fail_validation():
    payload = serialize_plan_to_dict(_valid_plan())
    payload["roi_plan"][0]["correction_strategy"]["strategy"] = "auto"
    plan = deserialize_plan_from_dict(payload)
    errors = validate_plan_contract(plan)
    assert any("forbidden runnable correction strategy: auto" in err for err in errors)

    payload = serialize_plan_to_dict(_valid_plan())
    payload["roi_plan"][0]["correction_strategy"]["choice_source"] = "diagnostic_success"
    payload["roi_plan"][0]["correction_strategy"]["no_auto_selection"] = False
    plan = deserialize_plan_from_dict(payload)
    errors = validate_plan_contract(plan)
    assert any("choice_source must be explicit_user_mark" in err for err in errors)
    assert any("no_auto_selection must be true" in err for err in errors)


def test_roi_entry_missing_roi_and_invalid_status_fail_validation():
    plan = _valid_plan()
    plan.roi_plan[0].roi = ""
    plan.roi_plan[0].roi_status = "needs_review"

    errors = validate_plan_contract(plan)

    assert any("roi_plan[0] missing roi" in err for err in errors)
    assert any("invalid roi_status: needs_review" in err for err in errors)


def test_readiness_summary_empty_incomplete_plan():
    from photometry_pipeline.guided_run_plan import (
        GuidedRunPlan,
        GuidedPlanSource,
        summarize_guided_plan_readiness,
        guided_plan_readiness_summary_lines,
    )
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(
            source_mode="completed_run",
            completed_run_dir="C:/runs/example",
        ),
        roi_plan=[],
    )

    summary = summarize_guided_plan_readiness(plan)
    assert "source" in summary.configured
    assert "ROI correction strategies" in summary.missing
    assert "feature/event profile" in summary.missing
    assert "output destination" in summary.missing
    assert any("execution intentionally unavailable" in b for b in summary.blocked)
    assert summary.execution_ready is False

    lines = guided_plan_readiness_summary_lines(plan)
    assert "Configured: source" in lines[0]
    assert "Missing: ROI correction strategies; feature/event profile; output destination" in lines[1]
    assert "Blocked: execution intentionally unavailable until a later Guided Run/RunSpec stage" in lines[2]
    assert "Files written: none" in lines[3]


def test_readiness_summary_partially_configured_plan():
    from photometry_pipeline.guided_run_plan import (
        GuidedRunPlan,
        GuidedPlanSource,
        RoiPlanEntry,
        CorrectionStrategyChoice,
        summarize_guided_plan_readiness,
        guided_plan_readiness_summary_lines,
    )
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(
            source_mode="completed_run",
            completed_run_dir="C:/runs/example",
        ),
        roi_plan=[
            RoiPlanEntry(
                roi="CH1",
                correction_strategy=CorrectionStrategyChoice(
                    strategy="robust_global_event_reject",
                    strategy_label="Robust Global Event-Reject Fit",
                ),
            )
        ],
    )

    summary = summarize_guided_plan_readiness(plan)
    assert "source" in summary.configured
    assert any("1 ROI correction strategy" in c for c in summary.configured)
    assert "feature/event profile" in summary.missing
    assert "output destination" in summary.missing
    assert summary.execution_ready is False

    lines = guided_plan_readiness_summary_lines(plan)
    assert "Configured: source; 1 ROI correction strategy" in lines[0]
    assert "Missing: feature/event profile; output destination" in lines[1]


def test_readiness_summary_fully_configured_plan():
    from photometry_pipeline.guided_run_plan import (
        GuidedRunPlan,
        GuidedPlanSource,
        RoiPlanEntry,
        CorrectionStrategyChoice,
        EvidenceChunkReview,
        FeatureEventProfile,
        OutputPolicy,
        summarize_guided_plan_readiness,
        guided_plan_readiness_summary_lines,
    )
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(
            source_mode="completed_run",
            completed_run_dir="C:/runs/example",
        ),
        roi_plan=[
            RoiPlanEntry(
                roi="CH1",
                correction_strategy=CorrectionStrategyChoice(
                    strategy="robust_global_event_reject",
                    strategy_label="Robust Global Event-Reject Fit",
                ),
                evidence=[
                    EvidenceChunkReview(
                        chunk_id=0,
                    )
                ],
            )
        ],
        feature_event_profiles=[
            FeatureEventProfile(
                profile_id="run_profile",
                scope="run",
                status="complete",
                config_fields={
                    "event_signal": "dff",
                    "signal_excursion_polarity": "positive",
                    "peak_threshold_method": "mean_std",
                    "peak_threshold_k": 3.0,
                    "peak_min_distance_sec": 0.5,
                    "peak_min_prominence_k": 1.0,
                    "peak_min_width_sec": 0.2,
                    "peak_pre_filter": "none",
                    "event_auc_baseline": "zero",
                },
            )
        ],
        output_policy=OutputPolicy(
            output_root="C:/outputs",
        ),
    )

    summary = summarize_guided_plan_readiness(plan)
    assert "source" in summary.configured
    assert any("1 ROI correction strategy" in c for c in summary.configured)
    assert "feature/event profile" in summary.configured
    assert "output destination" in summary.configured
    assert not summary.missing
    assert summary.execution_ready is False

    lines = guided_plan_readiness_summary_lines(plan)
    assert "Configured: source; 1 ROI correction strategy; feature/event profile; output destination" in lines[0]
    assert "Missing: none" in lines[1]
    assert "Blocked: execution intentionally unavailable until a later Guided Run/RunSpec stage" in lines[2]


def test_readiness_summary_contract_errors():
    from photometry_pipeline.guided_run_plan import (
        GuidedRunPlan,
        GuidedPlanSource,
        OutputPolicy,
        summarize_guided_plan_readiness,
        guided_plan_readiness_summary_lines,
    )
    # Invalid output policy (e.g. separate_from_source_required is False but output_root is set)
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(
            source_mode="completed_run",
            completed_run_dir="C:/runs/example",
        ),
        roi_plan=[],
        output_policy=OutputPolicy(
            output_root="C:/runs/example",
            separate_from_source_required=False,
        ),
    )

    summary = summarize_guided_plan_readiness(plan)
    assert "Invalid output destination policy" in summary.warnings
    assert summary.execution_ready is False

    lines = guided_plan_readiness_summary_lines(plan)
    # The problems line should display
    problems_line = [l for l in lines if l.startswith("Problems:")]
    assert len(problems_line) == 1
    assert "Invalid output destination policy" in problems_line[0]


def test_readiness_summary_pure_non_executing_guarantee(tmp_path):
    import sys
    from photometry_pipeline.guided_run_plan import (
        GuidedRunPlan,
        GuidedPlanSource,
        summarize_guided_plan_readiness,
    )
    plan = GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(
            source_mode="completed_run",
            completed_run_dir=str(tmp_path),
        ),
        roi_plan=[],
    )

    # Before calling, assert no production/execution pipeline modules are loaded
    assert "photometry_pipeline.core.feature_extraction" not in sys.modules
    assert "photometry_pipeline.core.pipeline" not in sys.modules

    summary = summarize_guided_plan_readiness(plan)
    assert summary is not None

    # After calling, they must still not be loaded
    assert "photometry_pipeline.core.feature_extraction" not in sys.modules
    assert "photometry_pipeline.core.pipeline" not in sys.modules
    assert not list(tmp_path.rglob("*"))


def test_plan_export_json_helper():
    import json
    from photometry_pipeline.guided_run_plan import (
        plan_export_json_text,
        deserialize_plan_from_dict,
        validate_plan_contract,
    )
    plan = _valid_plan()
    json_text = plan_export_json_text(plan)

    parsed = json.loads(json_text)
    assert parsed["schema_version"] == "guided_run_plan.v1"

    restored = deserialize_plan_from_dict(parsed)
    assert restored.plan_id == plan.plan_id
    assert restored.source.completed_run_dir == plan.source.completed_run_dir
    assert len(restored.roi_plan) == len(plan.roi_plan)
    assert restored.roi_plan[0].roi == plan.roi_plan[0].roi

    errors = validate_plan_contract(restored)
    assert errors == []
