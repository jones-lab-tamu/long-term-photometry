import sys

import pytest

from photometry_pipeline.guided_run_plan import (
    CorrectionStrategyChoice,
    EvidenceChunkReview,
    FeatureEventProfile,
    GuidedPlanSource,
    GuidedRunPlan,
    GuidedRunPlanContractError,
    RoiPlanEntry,
    assert_valid_plan_contract,
    deserialize_plan_from_dict,
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
        plan.feature_event_profiles[0].scope = scope
        assert validate_plan_contract(plan) == []

    plan = _valid_plan()
    plan.feature_event_profiles[0].scope = "chunk"
    plan.feature_event_profiles[0].evidence_previews = [EvidenceChunkReview(chunk_id=2)]
    errors = validate_plan_contract(plan)
    assert any("invalid scope: chunk" in err for err in errors)
    assert not any("chunk_id" in err for err in errors)


def test_feature_event_profile_unknown_config_fields_are_rejected():
    plan = _valid_plan()
    plan.feature_event_profiles[0].config_fields["new_detector_threshold"] = 1.0

    errors = validate_plan_contract(plan)

    assert any("unknown config fields" in err for err in errors)


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
    assert not (tmp_path / "validation").exists()
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
