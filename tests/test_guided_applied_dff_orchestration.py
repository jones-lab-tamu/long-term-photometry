"""Tests for retained Guided applied-dF/F manifest validation helpers."""

import pytest

from photometry_pipeline.guided_applied_dff_orchestration import (
    GuidedAppliedDffOrchestrationError,
    build_guided_applied_dff_manifest_rows,
)


def test_build_guided_applied_dff_manifest_rows_happy_path(tmp_path):
    """It builds correct rows with valid map."""
    strategy_map_payload = {
        "applied_dff_orchestration_enabled": True,
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1", "CH2"],
        "per_roi_production_strategy_map": [
            {
                "roi_id": "CH1",
                "strategy_family": "dynamic_fit",
                "dynamic_fit_mode": "robust_global_event_reject",
                "selected_strategy": "robust_global_event_reject",
                "evidence_source_type": "diagnostic_cache",
                "evidence_reference_json": '{"diagnostic_cache_id": "cache123", "evidence_reference_id": "ev1"}',
                "explicit_user_mark": True,
                "current_or_stale": "current",
            },
            {
                "roi_id": "CH2",
                "strategy_family": "signal_only_f0",
                "dynamic_fit_mode": None,
                "selected_strategy": "signal_only_f0",
                "evidence_source_type": "diagnostic_cache",
                "evidence_reference_json": '{"diagnostic_cache_id": "cache123", "evidence_reference_id": "ev2"}',
                "explicit_user_mark": True,
                "current_or_stale": "current",
            },
        ],
    }

    rows = build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)
    assert len(rows) == 2
    assert rows[0] == {
        "roi": "CH1",
        "strategy": "dynamic_fit",
        "output_name": "CH1_dynamic_fit",
        "feature_config": "",
    }
    assert rows[1] == {
        "roi": "CH2",
        "strategy": "signal_only_f0",
        "output_name": "CH2_signal_only_f0",
        "feature_config": "",
    }


def _valid_dynamic_fit_entry(roi_id="CH1"):
    return {
        "roi_id": roi_id,
        "strategy_family": "dynamic_fit",
        "dynamic_fit_mode": "robust_global_event_reject",
        "selected_strategy": "robust_global_event_reject",
        "evidence_source_type": "diagnostic_cache",
        "evidence_reference_json": "{}",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }


def _valid_signal_only_f0_entry(roi_id="CH2"):
    return {
        "roi_id": roi_id,
        "strategy_family": "signal_only_f0",
        "dynamic_fit_mode": None,
        "selected_strategy": "signal_only_f0",
        "evidence_source_type": "diagnostic_cache",
        "evidence_reference_json": "{}",
        "explicit_user_mark": True,
        "current_or_stale": "current",
    }


def test_build_guided_applied_dff_manifest_rows_empty_included(tmp_path):
    """It raises if included_roi_ids is empty."""
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Included ROI set cannot be empty"):
        build_guided_applied_dff_manifest_rows({"production_strategy_map_version": "per_roi_correction_strategy_map.v1"}, tmp_path)


def test_build_guided_applied_dff_manifest_rows_missing_roi(tmp_path):
    """It raises if an included ROI is missing from the map."""
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1", "CH2"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1")
        ],
    }
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Missing ROIs in strategy map"):
        build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)


def test_build_guided_applied_dff_manifest_rows_duplicate_roi(tmp_path):
    """It raises if the map has duplicate ROIs."""
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1"),
            _valid_signal_only_f0_entry("CH1")
        ],
    }
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Duplicate ROIs in strategy map"):
        build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)


def test_build_guided_applied_dff_manifest_rows_extra_roi(tmp_path):
    """It raises if the map has ROIs not in included set."""
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1"),
            _valid_dynamic_fit_entry("CH2")
        ],
    }
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Extra ROIs in strategy map"):
        build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)


def test_build_guided_applied_dff_manifest_rows_unsupported_strategy_family(tmp_path):
    """It rejects unsupported strategy_family."""
    entry = _valid_dynamic_fit_entry("CH1")
    entry["strategy_family"] = "unsupported"
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Unsupported strategy_family"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_dynamic_fit_selected_strategy_invalid(tmp_path):
    """It rejects dynamic_fit with invalid selected_strategy."""
    entry = _valid_dynamic_fit_entry("CH1")
    entry["selected_strategy"] = "dynamic_fit"
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Unsupported dynamic_fit selected_strategy"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_dynamic_fit_mode_invalid(tmp_path):
    """It rejects dynamic_fit with invalid dynamic_fit_mode."""
    entry = _valid_dynamic_fit_entry("CH1")
    entry["dynamic_fit_mode"] = "bic_heuristic"
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Unsupported dynamic_fit_mode"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_dynamic_fit_mismatch(tmp_path):
    """It rejects dynamic_fit selected_strategy != dynamic_fit_mode."""
    entry = _valid_dynamic_fit_entry("CH1")
    entry["selected_strategy"] = "robust_global_event_reject"
    entry["dynamic_fit_mode"] = "global_linear_regression"
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Mismatch between selected_strategy"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_signal_only_f0_selected_strategy_invalid(tmp_path):
    """It rejects signal_only_f0 with invalid selected_strategy."""
    entry = _valid_signal_only_f0_entry("CH1")
    entry["selected_strategy"] = "signal_only_F0"
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Unsupported signal_only_f0 selected_strategy"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_signal_only_f0_mode_populated(tmp_path):
    """It rejects signal_only_f0 with dynamic_fit_mode populated."""
    entry = _valid_signal_only_f0_entry("CH1")
    entry["dynamic_fit_mode"] = "robust_global_event_reject"
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="must have dynamic_fit_mode=None"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_explicit_user_mark_false(tmp_path):
    """It rejects explicit_user_mark = False."""
    entry = _valid_dynamic_fit_entry("CH1")
    entry["explicit_user_mark"] = False
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Non-explicit entry"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_current_or_stale_stale(tmp_path):
    """It rejects current_or_stale = stale."""
    entry = _valid_dynamic_fit_entry("CH1")
    entry["current_or_stale"] = "stale"
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Stale entry"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_blank_roi(tmp_path):
    """It rejects blank ROI."""
    entry = _valid_dynamic_fit_entry("")
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Strategy map entry missing roi_id"):
        build_guided_applied_dff_manifest_rows({
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": [""],
            "per_roi_production_strategy_map": [entry]
        }, tmp_path)


def test_build_guided_applied_dff_manifest_rows_duplicate_output_dir(tmp_path):
    """It raises if path sanitization causes duplicate outputs."""
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH 1", "CH_1"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH 1"),
            _valid_dynamic_fit_entry("CH_1"),
        ],
    }
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Duplicate output_name"):
        build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)


def test_build_guided_applied_dff_manifest_rows_sanitizes_path(tmp_path):
    """It sanitizes paths to prevent escaping."""
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["../CH1"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("../CH1")
        ],
    }
    rows = build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)
    assert rows[0]["output_name"] == ".._CH1_dynamic_fit"


def test_build_guided_applied_dff_manifest_rows_missing_version(tmp_path):
    strategy_map_payload = {
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [_valid_dynamic_fit_entry("CH1")]
    }
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="must be exactly per_roi_correction_strategy_map.v1"):
        build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)


def test_build_guided_applied_dff_manifest_rows_unsupported_version(tmp_path):
    strategy_map_payload = {
        "production_strategy_map_version": "v2",
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [_valid_dynamic_fit_entry("CH1")]
    }
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="must be exactly per_roi_correction_strategy_map.v1"):
        build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)


def test_build_guided_applied_dff_manifest_rows_all_signal_only_f0_accepted(tmp_path):
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [_valid_signal_only_f0_entry("CH1")]
    }
    rows = build_guided_applied_dff_manifest_rows(
        strategy_map_payload, tmp_path
    )
    assert rows == [
        {
            "roi": "CH1",
            "strategy": "signal_only_f0",
            "output_name": "CH1_signal_only_f0",
            "feature_config": "",
        }
    ]


def test_build_guided_applied_dff_manifest_rows_mixed_dynamic_fit_rejected(tmp_path):
    entry1 = _valid_dynamic_fit_entry("CH1")
    entry2 = _valid_dynamic_fit_entry("CH2")
    entry2["dynamic_fit_mode"] = "global_linear_regression"
    entry2["selected_strategy"] = "global_linear_regression"

    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1", "CH2"],
        "per_roi_production_strategy_map": [entry1, entry2]
    }
    with pytest.raises(
        GuidedAppliedDffOrchestrationError,
        match="Mixed dynamic_fit modes cannot be executed",
    ):
        build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)


def test_build_guided_applied_dff_manifest_rows_mixed_dynamic_fit_and_signal_only_accepted(tmp_path):
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1", "CH2"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1"),
            _valid_signal_only_f0_entry("CH2")
        ]
    }
    rows = build_guided_applied_dff_manifest_rows(strategy_map_payload, tmp_path)
    assert len(rows) == 2


def test_build_guided_applied_dff_manifest_rows_per_roi_feature_config_routes_only_overridden_roi(tmp_path):
    strategy_map_payload = {
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1", "CH2"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1"),
            _valid_dynamic_fit_entry("CH2"),
        ],
    }

    rows = build_guided_applied_dff_manifest_rows(
        strategy_map_payload,
        tmp_path,
        per_roi_feature_config_paths={"CH1": str(tmp_path / "ch1_feature_config.json")},
    )

    by_roi = {row["roi"]: row for row in rows}
    assert by_roi["CH1"]["feature_config"] == str(tmp_path / "ch1_feature_config.json")
    # CH2 has no override: empty cell, same as today's default-only behavior.
    assert by_roi["CH2"]["feature_config"] == ""
