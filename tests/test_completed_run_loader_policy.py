import os
import json
import pytest
from pathlib import Path
from gui.run_report_parser import (
    is_successful_completed_run_dir,
    is_preflight_or_ineligible,
    classify_completed_run_candidate,
)

# Call-chain testing for the actual completed-run acceptance path
class MockMainWindow:
    from gui.main_window import MainWindow
    _is_openable_completed_results_dir = MainWindow._is_openable_completed_results_dir

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")

def _write_region_deliverable(run_dir: Path, region_name: str, subfolder: str = "summary") -> None:
    sub = run_dir / region_name / subfolder
    sub.mkdir(parents=True, exist_ok=True)

# 1. Positive Acceptance Tests (protecting Full Control compatibility)

def test_positive_run_report_success(tmp_path: Path):
    run_dir = tmp_path / "run_report_success"
    # Positive run_report.json fixture includes explicit phase final
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success", "phase": "final"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True
    assert "run_report.json" in reason

def test_positive_status_json_success(tmp_path: Path):
    run_dir = tmp_path / "status_json_success"
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "success"})
    _write_region_deliverable(run_dir, "Region0", "day_plots")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True
    assert "status.json" in reason

def test_positive_manifest_json_success(tmp_path: Path):
    run_dir = tmp_path / "manifest_json_success"
    _write_json(run_dir / "MANIFEST.json", {"status": "success"})
    _write_region_deliverable(run_dir, "Region0", "tables")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True
    assert "MANIFEST.json" in reason

def test_positive_supplemental_metadata_allowed(tmp_path: Path):
    run_dir = tmp_path / "supplemental_metadata"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    # Supplemental metadata files
    (run_dir / "config_effective.yaml").write_text("dummy: config", encoding="utf-8")
    (run_dir / "command_invoked.txt").write_text("dummy command", encoding="utf-8")
    _write_json(run_dir / "gui_run_spec.json", {"mode": "guided"})
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True

def test_positive_multiple_regions(tmp_path: Path):
    run_dir = tmp_path / "multiple_regions"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    _write_region_deliverable(run_dir, "Region1", "tables")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True

def test_positive_minimally_acceptable_region(tmp_path: Path):
    run_dir = tmp_path / "minimal_region"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_region_deliverable(run_dir, "Region0", "tables")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True

# 2. Negative Rejection Tests for Current Fail-Closed Behavior

def test_negative_raw_non_output_folder(tmp_path: Path):
    run_dir = tmp_path / "raw_folder"
    run_dir.mkdir()
    (run_dir / "raw_recording.csv").write_text("time,sig\n0,1", encoding="utf-8")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_saved_guided_plan_only(tmp_path: Path):
    run_dir = tmp_path / "plan_only"
    _write_json(run_dir / "gui_run_spec.json", {"mode": "guided"})
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_cache_artifacts_only(tmp_path: Path):
    run_dir = tmp_path / "cache_only"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "phasic_trace_cache.h5").write_text("h5 content", encoding="utf-8")
    (run_dir / "config_used.yaml").write_text("config info", encoding="utf-8")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_regions_without_success_metadata(tmp_path: Path):
    run_dir = tmp_path / "regions_no_metadata"
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_success_metadata_without_regions(tmp_path: Path):
    run_dir = tmp_path / "metadata_no_regions"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_failed_metadata(tmp_path: Path):
    run_dir = tmp_path / "failed_metadata"
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "failed"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_aborted_metadata(tmp_path: Path):
    run_dir = tmp_path / "aborted_metadata"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success", "phase": "aborted"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_corrupted_json(tmp_path: Path):
    run_dir = tmp_path / "corrupted_json"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text("{corrupted json}", encoding="utf-8")
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    # Should not raise exception, but fail gracefully
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

def test_negative_unsupported_schema_version(tmp_path: Path):
    run_dir = tmp_path / "unsupported_version"
    _write_json(run_dir / "status.json", {"schema_version": 2, "phase": "final", "status": "success"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False

# 3. Future Preflight / Completed-Run-Ineligible Rejection Tests

def test_preflight_marker_file_rejection(tmp_path: Path):
    run_dir = tmp_path / "preflight_file_marker"
    # Even with successful production-looking metadata and region deliverables
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    # Create marker file
    (run_dir / "preflight_marker.json").write_text("{}", encoding="utf-8")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "non-production/completed-run-ineligible" in reason

def test_status_json_ineligible_rejection(tmp_path: Path):
    run_dir = tmp_path / "status_ineligible"
    _write_json(run_dir / "status.json", {
        "schema_version": 1,
        "phase": "final",
        "status": "success",
        "completed_run_ineligible": True
    })
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "non-production/completed-run-ineligible" in reason

def test_run_report_preflight_rejection(tmp_path: Path):
    run_dir = tmp_path / "run_report_preflight"
    _write_json(run_dir / "run_report.json", {
        "run_context": {
            "status": "success",
            "preflight": True
        }
    })
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "non-production/completed-run-ineligible" in reason

# 4. Actual Path Rejection and Reserved Rejection Markers Verification

def test_actual_loader_path_rejection_of_preflight(tmp_path: Path):
    run_dir = tmp_path / "actual_path_preflight_rejection"
    # Valid output structure
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "success"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    # Assert accepted normally
    ok, reason = MockMainWindow()._is_openable_completed_results_dir(str(run_dir))
    assert ok is True
    
    # Add preflight file marker
    (run_dir / "preflight.manifest").write_text("{}", encoding="utf-8")
    
    # Assert rejected by actual loader path
    ok, reason = MockMainWindow()._is_openable_completed_results_dir(str(run_dir))
    assert ok is False
    assert "non-production/completed-run-ineligible" in reason

def test_reserved_rejection_markers_never_cause_acceptance(tmp_path: Path):
    run_dir = tmp_path / "rejection_markers_only"
    run_dir.mkdir()
    # Adding preflight/ineligible indicators only (no production success)
    (run_dir / "preflight_marker.json").write_text("{}", encoding="utf-8")
    _write_json(run_dir / "status.json", {"completed_run_ineligible": True})
    
    # Must reject (does not cause acceptance)
    ok, reason = MockMainWindow()._is_openable_completed_results_dir(str(run_dir))
    assert ok is False

def test_existing_valid_full_control_outputs_remain_accepted(tmp_path: Path):
    run_dir = tmp_path / "valid_full_control_output"
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "success"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    # No Guided preflight tags. Must load successfully.
    ok, reason = MockMainWindow()._is_openable_completed_results_dir(str(run_dir))
    assert ok is True

# 5. Read-Only / No-Side-Effect Tests

def _get_dir_snapshot(path: Path) -> dict:
    snapshot = {}
    for p in path.glob("**/*"):
        if p.is_file():
            stat = p.stat()
            snapshot[str(p.relative_to(path))] = (stat.st_mtime, stat.st_size)
    return snapshot

def test_loader_has_no_side_effects(tmp_path: Path):
    run_dir = tmp_path / "side_effects_test"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    # Capture snapshot before running classification
    before = _get_dir_snapshot(run_dir)
    
    # Run classification on both paths
    classify_completed_run_candidate(str(run_dir))
    MockMainWindow()._is_openable_completed_results_dir(str(run_dir))
    
    # Capture snapshot after running classification
    after = _get_dir_snapshot(run_dir)
    
    # Ensure they match exactly
    assert before == after

# 6. Hardening Conflict Rejection Tests

def test_conflict_run_report_success_status_failed(tmp_path: Path):
    run_dir = tmp_path / "conflict_1"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "failed"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "conflicting metadata" in reason

def test_conflict_run_report_success_status_active(tmp_path: Path):
    run_dir = tmp_path / "conflict_2"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "active", "status": "success"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "conflicting metadata" in reason

def test_conflict_status_success_manifest_failed(tmp_path: Path):
    run_dir = tmp_path / "conflict_3"
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "success"})
    _write_json(run_dir / "MANIFEST.json", {"status": "failed"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "conflicting metadata" in reason

def test_conflict_run_report_success_manifest_failed(tmp_path: Path):
    run_dir = tmp_path / "conflict_4"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_json(run_dir / "MANIFEST.json", {"status": "failed"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "conflicting metadata" in reason

def test_status_success_missing_others(tmp_path: Path):
    run_dir = tmp_path / "missing_others_1"
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "success"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True

def test_run_report_success_missing_others(tmp_path: Path):
    run_dir = tmp_path / "missing_others_2"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True

def test_corrupted_run_report_status_success(tmp_path: Path):
    run_dir = tmp_path / "corrupt_report"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text("{corrupt json}", encoding="utf-8")
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "success"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    # Intended compatibility: skips corrupt run_report and succeeds based on status.json
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is True
    assert "status.json" in reason

def test_preflight_marker_with_successful_metadata_replaces_all(tmp_path: Path):
    run_dir = tmp_path / "preflight_vs_success"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_region_deliverable(run_dir, "Region0", "summary")
    # Preflight marker tag
    (run_dir / "preflight.manifest").write_text("{}", encoding="utf-8")
    
    # Must reject early
    ok, reason = classify_completed_run_candidate(str(run_dir))
    assert ok is False
    assert "non-production/completed-run-ineligible" in reason

def test_actual_open_results_path_rejects_conflict(tmp_path: Path):
    run_dir = tmp_path / "actual_path_conflict"
    _write_json(run_dir / "run_report.json", {"run_context": {"status": "success"}})
    _write_json(run_dir / "status.json", {"schema_version": 1, "phase": "final", "status": "failed"})
    _write_region_deliverable(run_dir, "Region0", "summary")
    
    ok, reason = MockMainWindow()._is_openable_completed_results_dir(str(run_dir))
    assert ok is False
    assert "conflicting metadata" in reason
