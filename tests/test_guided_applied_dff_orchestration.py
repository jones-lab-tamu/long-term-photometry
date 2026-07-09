"""Tests for Guided applied-dF/F orchestration helper."""

import json
from pathlib import Path

import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS
from photometry_pipeline.guided_applied_dff_orchestration import (
    GuidedAppliedDffOrchestrationError,
    build_guided_applied_dff_manifest_rows,
    run_guided_applied_dff_orchestration_if_enabled,
    write_per_roi_feature_config_files,
)


def _complete_feature_config_fields(**overrides):
    """A complete FEATURE_EVENT_CONFIG_FIELDS dict, shaped like
    guided_new_analysis_plan.build_per_roi_effective_feature_config_fields_for_overrides'
    output: every required field present, starting from Config() defaults
    with the given fields overridden.
    """
    cfg = Config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return {field_name: getattr(cfg, field_name) for field_name in FEATURE_EVENT_CONFIG_FIELDS}


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


def test_wrapper_skip_behavior_no_file(tmp_path):
    """A. No file: wrapper helper skips orchestration, no folder created."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()
    
    ran = run_guided_applied_dff_orchestration_if_enabled(str(run_dir), str(phasic_out))
    assert not ran
    assert not (run_dir / "applied_dff").exists()


def test_wrapper_skip_behavior_enabled_false(tmp_path):
    """B. File present, enabled false: wrapper helper skips orchestration, no folder created."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()
    
    map_file = run_dir / "guided_correction_strategy_map.json"
    map_file.write_text(json.dumps({"applied_dff_orchestration_enabled": False}), encoding="utf-8")
    
    ran = run_guided_applied_dff_orchestration_if_enabled(str(run_dir), str(phasic_out))
    assert not ran
    assert not (run_dir / "applied_dff").exists()


def test_wrapper_behavior_malformed_json(tmp_path):
    """C. File present, malformed JSON: wrapper helper fails closed."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()
    
    map_file = run_dir / "guided_correction_strategy_map.json"
    map_file.write_text("invalid json", encoding="utf-8")
    
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="malformed"):
        run_guided_applied_dff_orchestration_if_enabled(str(run_dir), str(phasic_out))


def test_wrapper_behavior_missing_enabled_flag(tmp_path):
    """D. File present, missing enabled flag: wrapper helper fails closed."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()
    
    map_file = run_dir / "guided_correction_strategy_map.json"
    map_file.write_text(json.dumps({"some_other_key": True}), encoding="utf-8")
    
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="missing applied_dff_orchestration_enabled flag"):
        run_guided_applied_dff_orchestration_if_enabled(str(run_dir), str(phasic_out))


def test_wrapper_behavior_missing_phasic_cache(tmp_path):
    """E. File present, enabled true, missing phasic cache: writes provenance, raises."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()
    
    map_file = run_dir / "guided_correction_strategy_map.json"
    map_file.write_text(json.dumps({"applied_dff_orchestration_enabled": True}), encoding="utf-8")
    
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Missing phasic cache file"):
        run_guided_applied_dff_orchestration_if_enabled(str(run_dir), str(phasic_out))
        
    prov_path = run_dir / "applied_dff" / "guided_applied_dff_provenance.json"
    assert prov_path.exists()
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov_data["overall_status"] == "failed"
    assert "Missing phasic cache file" in prov_data["error"]


def test_wrapper_behavior_fake_successful_batch(tmp_path):
    """F. File present, enabled true, fake successful batch."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()
    
    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"fake_h5")
    
    map_file = run_dir / "guided_correction_strategy_map.json"
    strategy_map_payload = {
        "applied_dff_orchestration_enabled": True,
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1")
        ],
    }
    map_file.write_text(json.dumps(strategy_map_payload), encoding="utf-8")
    
    cmd_called = []
    def fake_run(cmd):
        cmd_called.extend(cmd)
        return 0

    on_enabled_called = []
    def fake_on_enabled():
        on_enabled_called.append(True)
        
    ran = run_guided_applied_dff_orchestration_if_enabled(
        str(run_dir), str(phasic_out), run_cmd_callable=fake_run, on_enabled=fake_on_enabled
    )
    
    assert ran
    assert on_enabled_called == [True]
    assert any(str(c).endswith("tools/run_applied_dff_batch.py") or str(c).endswith("tools\\run_applied_dff_batch.py") for c in cmd_called)
    assert "--phasic-out" in cmd_called
    assert "--manifest" in cmd_called
    assert "--output-root" in cmd_called
    
    manifest_path = run_dir / "applied_dff" / "batch_manifest.csv"
    assert manifest_path.exists()
    content = manifest_path.read_text(encoding="utf-8")
    assert "roi,strategy,output_name" in content
    assert "CH1,dynamic_fit,CH1_dynamic_fit" in content
    
    prov_path = run_dir / "applied_dff" / "guided_applied_dff_provenance.json"
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov_data["overall_status"] == "succeeded"
    assert prov_data["batch_returncode"] == 0
    assert len(prov_data["rows"]) == 1
    assert all(r["status"] == "succeeded" for r in prov_data["rows"])


def test_wrapper_behavior_fake_failed_batch(tmp_path):
    """G. File present, enabled true, fake failed batch."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()
    
    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"fake_h5")
    
    map_file = run_dir / "guided_correction_strategy_map.json"
    strategy_map_payload = {
        "applied_dff_orchestration_enabled": True,
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1")
        ],
    }
    map_file.write_text(json.dumps(strategy_map_payload), encoding="utf-8")
    
    def fake_run(cmd):
        return 1
        
    with pytest.raises(GuidedAppliedDffOrchestrationError, match="Subprocess returned 1"):
        run_guided_applied_dff_orchestration_if_enabled(
            str(run_dir), str(phasic_out), run_cmd_callable=fake_run
        )
        
    prov_path = run_dir / "applied_dff" / "guided_applied_dff_provenance.json"
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov_data["overall_status"] == "failed"
    assert prov_data["batch_returncode"] == 1
    assert len(prov_data["rows"]) == 1
    assert all(r["status"] == "failed" for r in prov_data["rows"])


def test_write_per_roi_feature_config_files_writes_one_file_per_roi(tmp_path):
    out_dir = tmp_path / "feature_configs"
    ch1_fields = _complete_feature_config_fields(
        peak_threshold_method="percentile", peak_threshold_percentile=90.0
    )
    ch2_fields = _complete_feature_config_fields(event_signal="delta_f")
    paths = write_per_roi_feature_config_files(
        {
            "CH1": ch1_fields,
            "CH 2/x": ch2_fields,
        },
        out_dir,
    )

    assert set(paths.keys()) == {"CH1", "CH 2/x"}
    ch1_payload = json.loads(Path(paths["CH1"]).read_text(encoding="utf-8"))
    assert ch1_payload == ch1_fields
    assert set(ch1_payload.keys()) == FEATURE_EVENT_CONFIG_FIELDS
    # Path components are sanitized the same way ROI output dirs already are.
    assert Path(paths["CH 2/x"]).name == "CH_2_x_feature_config.json"


def test_write_per_roi_feature_config_files_empty_input_writes_nothing(tmp_path):
    out_dir = tmp_path / "feature_configs"
    paths = write_per_roi_feature_config_files(None, out_dir)

    assert paths == {}
    assert not out_dir.exists()


def test_write_per_roi_feature_config_files_rejects_sparse_dict_and_writes_nothing(tmp_path):
    """A sparse override dict must fail closed, not silently write a
    feature-config file that run_applied_dff_features.py would misread as
    complete."""
    out_dir = tmp_path / "feature_configs"
    sparse = {"peak_threshold_method": "percentile", "peak_threshold_percentile": 90.0}

    with pytest.raises(
        GuidedAppliedDffOrchestrationError,
        match="must contain the complete FEATURE_EVENT_CONFIG_FIELDS set",
    ):
        write_per_roi_feature_config_files({"CH1": sparse}, out_dir)

    assert not out_dir.exists()


def test_write_per_roi_feature_config_files_one_sparse_roi_blocks_all_writes(tmp_path):
    """A single incomplete ROI blocks writing files for every ROI, not just
    the incomplete one, so a partial batch never gets to disk."""
    out_dir = tmp_path / "feature_configs"
    complete = _complete_feature_config_fields(peak_threshold_method="percentile")
    sparse = {"event_signal": "delta_f"}

    with pytest.raises(GuidedAppliedDffOrchestrationError):
        write_per_roi_feature_config_files({"CH1": complete, "CH2": sparse}, out_dir)

    assert not out_dir.exists()


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


def test_run_guided_applied_dff_orchestration_routes_per_roi_feature_config(tmp_path):
    """Per-ROI feature config paths reach the manifest CSV and provenance."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()

    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"fake_h5")

    map_file = run_dir / "guided_correction_strategy_map.json"
    strategy_map_payload = {
        "applied_dff_orchestration_enabled": True,
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1", "CH2"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1"),
            _valid_dynamic_fit_entry("CH2"),
        ],
    }
    map_file.write_text(json.dumps(strategy_map_payload), encoding="utf-8")

    def fake_run(cmd):
        return 0

    ch1_fields = _complete_feature_config_fields(
        peak_threshold_method="percentile", peak_threshold_percentile=90.0
    )
    ran = run_guided_applied_dff_orchestration_if_enabled(
        str(run_dir),
        str(phasic_out),
        run_cmd_callable=fake_run,
        per_roi_feature_event_overrides={"CH1": ch1_fields},
    )
    assert ran

    manifest_path = run_dir / "applied_dff" / "batch_manifest.csv"
    content = manifest_path.read_text(encoding="utf-8")
    assert "roi,strategy,output_name,feature_config" in content
    lines = {line.split(",")[0]: line for line in content.strip().splitlines()[1:]}
    assert "CH1_feature_config.json" in lines["CH1"]
    assert lines["CH2"].endswith(",")  # CH2 has no override: empty trailing cell

    config_path = run_dir / "applied_dff" / "feature_configs" / "CH1_feature_config.json"
    assert config_path.exists()
    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written == ch1_fields
    assert set(written.keys()) == FEATURE_EVENT_CONFIG_FIELDS

    prov_path = run_dir / "applied_dff" / "guided_applied_dff_provenance.json"
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov_data["per_roi_feature_config_paths"] == {"CH1": str(config_path)}


def test_run_guided_applied_dff_orchestration_rejects_sparse_per_roi_feature_config(tmp_path):
    """A sparse per-ROI feature config fails the whole orchestration closed:
    no manifest, no feature-config files, and the failure is recorded in
    provenance -- it must not silently route an incomplete config to the
    batch."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()

    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"fake_h5")

    map_file = run_dir / "guided_correction_strategy_map.json"
    strategy_map_payload = {
        "applied_dff_orchestration_enabled": True,
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1", "CH2"],
        "per_roi_production_strategy_map": [
            _valid_dynamic_fit_entry("CH1"),
            _valid_dynamic_fit_entry("CH2"),
        ],
    }
    map_file.write_text(json.dumps(strategy_map_payload), encoding="utf-8")

    def fake_run(cmd):
        return 0

    sparse = {"peak_threshold_method": "percentile", "peak_threshold_percentile": 90.0}
    with pytest.raises(
        GuidedAppliedDffOrchestrationError,
        match="must contain the complete FEATURE_EVENT_CONFIG_FIELDS set",
    ):
        run_guided_applied_dff_orchestration_if_enabled(
            str(run_dir),
            str(phasic_out),
            run_cmd_callable=fake_run,
            per_roi_feature_event_overrides={"CH1": sparse},
        )

    assert not (run_dir / "applied_dff" / "feature_configs").exists()
    manifest_path = run_dir / "applied_dff" / "batch_manifest.csv"
    assert not manifest_path.exists()

    prov_path = run_dir / "applied_dff" / "guided_applied_dff_provenance.json"
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov_data["overall_status"] == "failed"
    assert "must contain the complete FEATURE_EVENT_CONFIG_FIELDS set" in prov_data["error"]


def test_run_guided_applied_dff_orchestration_default_only_unaffected_by_new_param(tmp_path):
    """Omitting per_roi_feature_event_overrides is identical to today's behavior."""
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "phasic"
    run_dir.mkdir()
    phasic_out.mkdir()

    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"fake_h5")

    map_file = run_dir / "guided_correction_strategy_map.json"
    strategy_map_payload = {
        "applied_dff_orchestration_enabled": True,
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "included_roi_ids": ["CH1"],
        "per_roi_production_strategy_map": [_valid_dynamic_fit_entry("CH1")],
    }
    map_file.write_text(json.dumps(strategy_map_payload), encoding="utf-8")

    def fake_run(cmd):
        return 0

    ran = run_guided_applied_dff_orchestration_if_enabled(
        str(run_dir), str(phasic_out), run_cmd_callable=fake_run
    )
    assert ran
    assert not (run_dir / "applied_dff" / "feature_configs").exists()

    prov_path = run_dir / "applied_dff" / "guided_applied_dff_provenance.json"
    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov_data["per_roi_feature_config_paths"] == {}
