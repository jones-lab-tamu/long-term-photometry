"""Tests for guided_completed_applied_dff_reload module and RunReportViewer integration."""

import json
import os
import tempfile
from pathlib import Path
import pytest

from photometry_pipeline.guided_completed_applied_dff_reload import (
    load_guided_completed_applied_dff_state,
    format_guided_completed_applied_dff_summary,
    GuidedCompletedAppliedDffIssue,
    GuidedCompletedAppliedDffRow,
    GuidedCompletedAppliedDffState,
)
from gui.run_report_viewer import RunReportViewer
from PySide6.QtWidgets import QApplication


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _create_completed_run(tmp_path: Path, prov_payload: dict | str | None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write standard run report success metadata
    run_report_path = run_dir / "run_report.json"
    run_report_path.write_text(json.dumps({
        "status": "success",
        "run_context": {
            "run_type": "full"
        }
    }), encoding="utf-8")

    # Write a region deliverable to satisfy is_successful_completed_run_dir / RunReportViewer
    reg_dir = run_dir / "Region0"
    (reg_dir / "summary").mkdir(parents=True, exist_ok=True)
    (reg_dir / "day_plots").mkdir(parents=True, exist_ok=True)
    (reg_dir / "tables").mkdir(parents=True, exist_ok=True)

    if prov_payload is not None:
        applied_dff_dir = run_dir / "applied_dff"
        applied_dff_dir.mkdir(parents=True, exist_ok=True)
        prov_path = applied_dff_dir / "guided_applied_dff_provenance.json"
        if isinstance(prov_payload, str):
            prov_path.write_text(prov_payload, encoding="utf-8")
        else:
            prov_path.write_text(json.dumps(prov_payload), encoding="utf-8")

    return run_dir


def _valid_provenance_payload(run_dir: Path) -> dict:
    applied_dff_root = run_dir / "applied_dff"
    applied_dff_root.mkdir(parents=True, exist_ok=True)
    phasic_cache = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    phasic_cache.parent.mkdir(parents=True, exist_ok=True)
    phasic_cache.write_bytes(b"fake_h5")

    manifest_csv = applied_dff_root / "batch_manifest.csv"
    manifest_csv.write_text("roi,strategy,output_name\nCH1,dynamic_fit,CH1_dynamic_fit", encoding="utf-8")

    roi_dir = applied_dff_root / "CH1_dynamic_fit"
    roi_dir.mkdir(parents=True, exist_ok=True)

    return {
        "schema_version": "v1",
        "created_at_utc": "2026-07-04T12:00:00Z",
        "completed_at_utc": "2026-07-04T12:05:00Z",
        "run_dir": str(run_dir),
        "phasic_cache_path": str(phasic_cache),
        "phasic_cache_exists_at_start": True,
        "phasic_cache_size_bytes": len(b"fake_h5"),
        "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
        "requested_strategy_map": {
            "applied_dff_orchestration_enabled": True,
            "production_strategy_map_version": "per_roi_correction_strategy_map.v1",
            "included_roi_ids": ["CH1"],
            "per_roi_production_strategy_map": [
                {
                    "roi_id": "CH1",
                    "strategy_family": "dynamic_fit",
                    "dynamic_fit_mode": "robust_global_event_reject",
                    "selected_strategy": "robust_global_event_reject",
                    "evidence_source_type": "diagnostic_cache",
                    "evidence_reference_json": "{}",
                    "explicit_user_mark": True,
                    "current_or_stale": "current"
                }
            ]
        },
        "batch_manifest_path": str(manifest_csv),
        "applied_output_root": str(applied_dff_root),
        "batch_command": ["python", "tools/run_applied_dff_batch.py"],
        "batch_returncode": 0,
        "batch_summary_path": None,
        "orchestration_capability_enabled": True,
        "failure_policy": "fail_whole_guided_run_on_any_applied_failure",
        "overall_status": "succeeded",
        "production_analysis": True,
        "preview_only": False,
        "error": None,
        "rows": [
            {
                "roi_id": "CH1",
                "batch_roi": "CH1",
                "strategy_family": "dynamic_fit",
                "selected_strategy": "robust_global_event_reject",
                "dynamic_fit_mode": "robust_global_event_reject",
                "batch_strategy": "dynamic_fit",
                "output_name": "CH1_dynamic_fit",
                "output_dir": str(roi_dir),
                "status": "succeeded",
                "pipeline_summary_path": None,
                "error": None
            }
        ]
    }


def test_no_provenance_file(tmp_path):
    """1. No provenance file -> returns present=False, no blocking issues, does not create directory."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is False
    assert len(state.blocking_issues()) == 0
    assert not (run_dir / "applied_dff").exists()


def test_format_summary_absent_state():
    text = format_guided_completed_applied_dff_summary(
        GuidedCompletedAppliedDffState.absent()
    )

    assert "Applied dF/F" in text
    assert "not present" in text
    assert "blocking" not in text


def test_format_summary_valid_succeeded_state():
    state = GuidedCompletedAppliedDffState(
        present=True,
        overall_status="succeeded",
        production_analysis=True,
        preview_only=False,
        orchestration_capability_enabled=True,
        rows=(
            GuidedCompletedAppliedDffRow(
                roi_id="CH1",
                batch_roi="CH1",
                strategy_family="dynamic_fit",
                selected_strategy="robust_global_event_reject",
                dynamic_fit_mode="robust_global_event_reject",
                batch_strategy="dynamic_fit",
                output_name="CH1_dynamic_fit",
                output_dir="applied_dff/CH1_dynamic_fit",
                status="succeeded",
                pipeline_summary_path=None,
                output_dir_exists=True,
                pipeline_summary_exists=False,
            ),
        ),
    )

    text = format_guided_completed_applied_dff_summary(state)

    assert "Applied dF/F: succeeded" in text
    assert "Production analysis: yes" in text
    assert "CH1" in text
    assert "dynamic_fit" in text
    assert "output present" in text


def test_format_summary_invalid_provenance():
    state = GuidedCompletedAppliedDffState(
        present=True,
        overall_status="succeeded",
        issues=(
            GuidedCompletedAppliedDffIssue(
                category="test_blocker",
                severity="blocking",
                message="Contract mismatch.",
            ),
        ),
    )

    text = format_guided_completed_applied_dff_summary(state)

    assert "invalid provenance" in text
    assert "test_blocker" in text
    assert "Contract mismatch." in text


def test_format_summary_failed_orchestration_warning():
    state = GuidedCompletedAppliedDffState(
        present=True,
        overall_status="failed",
        issues=(
            GuidedCompletedAppliedDffIssue(
                category="applied_dff_orchestration_failed",
                severity="warning",
                message="Subprocess returned 1.",
            ),
        ),
    )

    text = format_guided_completed_applied_dff_summary(state)

    assert "Applied dF/F: failed" in text
    assert "applied_dff_orchestration_failed" in text


def test_malformed_provenance_json(tmp_path):
    """2. Malformed JSON -> present=True, blocking issue applied_dff_provenance_malformed, no crash."""
    run_dir = _create_completed_run(tmp_path, prov_payload="invalid json")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_malformed" for issue in state.blocking_issues())


def test_minimal_valid_succeeded_provenance(tmp_path):
    """3. Minimal valid succeeded provenance -> present=True, succeeded, production=True, preview=False, parsed rows, exists flags."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    # Write updated payload
    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.overall_status == "succeeded"
    assert state.production_analysis is True
    assert state.preview_only is False
    assert len(state.rows) == 1
    assert state.rows[0].roi_id == "CH1"
    assert state.rows[0].output_dir_exists is True
    assert state.phasic_cache_exists_at_load is True
    assert state.batch_manifest_exists_at_load is True
    assert state.has_blocking_issues is False


def test_missing_required_field(tmp_path):
    """4. Missing required field -> blocking issue applied_dff_provenance_missing_required_field."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    del payload["schema_version"]

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_missing_required_field" for issue in state.blocking_issues())


def test_unsupported_schema_version(tmp_path):
    """5. Unsupported schema version -> blocking issue applied_dff_provenance_unsupported_schema_version."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["schema_version"] = "v2"

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_unsupported_schema_version" for issue in state.blocking_issues())


def test_unsupported_top_level_strategy_map_version(tmp_path):
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["production_strategy_map_version"] = "v2"
    _write_provenance(run_dir, payload)

    state = load_guided_completed_applied_dff_state(run_dir)

    assert state.present is True
    assert state.has_blocking_issues is True
    assert (
        "applied_dff_provenance_unsupported_strategy_map_version"
        in {issue.category for issue in state.blocking_issues()}
    )


def test_production_analysis_false(tmp_path):
    """6. production_analysis False -> blocking issue applied_dff_provenance_not_production_analysis."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["production_analysis"] = False

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_not_production_analysis" for issue in state.blocking_issues())


def test_preview_only_true(tmp_path):
    """7. preview_only True -> blocking issue applied_dff_provenance_marked_preview_only."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["preview_only"] = True

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_marked_preview_only" for issue in state.blocking_issues())


def test_running_status_in_completed_run(tmp_path):
    """8. running status -> blocking issue applied_dff_provenance_still_running_in_completed_run."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["overall_status"] = "running"

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_still_running_in_completed_run" for issue in state.blocking_issues())


def test_failed_status(tmp_path):
    """9. failed status -> issue applied_dff_orchestration_failed (warning)."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["overall_status"] = "failed"
    payload["error"] = "Subprocess returned 1"

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.overall_status == "failed"
    # overall_status=failed is a warning, not blocking
    assert state.has_blocking_issues is False
    assert any(issue.category == "applied_dff_orchestration_failed" and issue.severity == "warning" for issue in state.issues)


def test_succeeded_overall_but_row_pending_or_failed(tmp_path):
    """10. succeeded overall but row pending/failed -> blocking issue applied_dff_row_status_inconsistent."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["rows"][0]["status"] = "failed"

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_row_status_inconsistent" for issue in state.blocking_issues())


def test_row_output_dir_escapes_applied_output_root(tmp_path):
    """11. row output_dir escapes root -> blocking issue applied_dff_row_output_dir_escapes_root."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["rows"][0]["output_dir"] = "/outside/directory"

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_row_output_dir_escapes_root" for issue in state.blocking_issues())


def test_requested_strategy_map_all_signal_only_f0(tmp_path):
    """12. all-signal_only_f0 is valid when its evidence is current."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["requested_strategy_map"]["per_roi_production_strategy_map"] = [
        {
            "roi_id": "CH1",
            "strategy_family": "signal_only_f0",
            "dynamic_fit_mode": None,
            "selected_strategy": "signal_only_f0",
            "explicit_user_mark": True,
            "current_or_stale": "current"
        }
    ]

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is False


def test_requested_strategy_map_mixed_dynamic_fit_modes(tmp_path):
    """13. mixed dynamic_fit modes -> blocking issue."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["requested_strategy_map"]["included_roi_ids"] = ["CH1", "CH2"]
    payload["requested_strategy_map"]["per_roi_production_strategy_map"] = [
        {
            "roi_id": "CH1",
            "strategy_family": "dynamic_fit",
            "dynamic_fit_mode": "robust_global_event_reject",
            "selected_strategy": "robust_global_event_reject",
            "explicit_user_mark": True,
            "current_or_stale": "current"
        },
        {
            "roi_id": "CH2",
            "strategy_family": "dynamic_fit",
            "dynamic_fit_mode": "global_linear_regression",
            "selected_strategy": "global_linear_regression",
            "explicit_user_mark": True,
            "current_or_stale": "current"
        }
    ]
    payload["rows"].append({
        "roi_id": "CH2",
        "batch_roi": "CH2",
        "strategy_family": "dynamic_fit",
        "selected_strategy": "global_linear_regression",
        "dynamic_fit_mode": "global_linear_regression",
        "batch_strategy": "dynamic_fit",
        "output_name": "CH2_dynamic_fit",
        "output_dir": str(run_dir / "applied_dff" / "CH2_dynamic_fit"),
        "status": "succeeded",
        "pipeline_summary_path": None,
        "error": None
    })

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_mixed_dynamic_fit" for issue in state.blocking_issues())


def test_requested_strategy_map_mixed_dynamic_fit_and_signal_only_f0_valid(tmp_path):
    """14. mixed dynamic_fit + signal_only_f0 valid -> no blocking issue."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["requested_strategy_map"]["included_roi_ids"] = ["CH1", "CH2"]
    payload["requested_strategy_map"]["per_roi_production_strategy_map"] = [
        {
            "roi_id": "CH1",
            "strategy_family": "dynamic_fit",
            "dynamic_fit_mode": "robust_global_event_reject",
            "selected_strategy": "robust_global_event_reject",
            "explicit_user_mark": True,
            "current_or_stale": "current"
        },
        {
            "roi_id": "CH2",
            "strategy_family": "signal_only_f0",
            "dynamic_fit_mode": None,
            "selected_strategy": "signal_only_f0",
            "explicit_user_mark": True,
            "current_or_stale": "current"
        }
    ]
    payload["rows"].append({
        "roi_id": "CH2",
        "batch_roi": "CH2",
        "strategy_family": "signal_only_f0",
        "selected_strategy": "signal_only_f0",
        "dynamic_fit_mode": None,
        "batch_strategy": "signal_only_f0",
        "output_name": "CH2_signal_only_f0",
        "output_dir": str(run_dir / "applied_dff" / "CH2_signal_only_f0"),
        "status": "succeeded",
        "pipeline_summary_path": None,
        "error": None
    })

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is False


def test_requested_strategy_map_stale_entry(tmp_path):
    """15. requested strategy map stale entry -> blocking issue."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["requested_strategy_map"]["per_roi_production_strategy_map"][0]["current_or_stale"] = "stale"

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_stale_strategy" for issue in state.blocking_issues())


def test_requested_strategy_map_non_explicit_entry(tmp_path):
    """16. requested strategy map non-explicit entry -> blocking issue."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)
    payload["requested_strategy_map"]["per_roi_production_strategy_map"][0]["explicit_user_mark"] = False

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    state = load_guided_completed_applied_dff_state(run_dir)
    assert state.present is True
    assert state.has_blocking_issues is True
    assert any(issue.category == "applied_dff_provenance_non_explicit_strategy" for issue in state.blocking_issues())


def test_no_filesystem_writes_during_load(tmp_path):
    """19. No filesystem writes during load -> loading does not create/write files."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)

    # Capture list of files under run_dir before loading
    before = sorted(os.walk(run_dir))

    load_guided_completed_applied_dff_state(run_dir)

    after = sorted(os.walk(run_dir))
    assert before == after


def _write_provenance(run_dir: Path, payload: dict) -> None:
    path = run_dir / "applied_dff" / "guided_applied_dff_provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _blocking_categories(run_dir: Path) -> set[str]:
    return {
        issue.category
        for issue in load_guided_completed_applied_dff_state(
            run_dir
        ).blocking_issues()
    }


def test_unexpected_phasic_cache_location_blocks(tmp_path):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    payload["phasic_cache_path"] = str(
        run_dir / "phasic" / "phasic_trace_cache.h5"
    )
    _write_provenance(run_dir, payload)

    assert "applied_dff_phasic_cache_path_unexpected" in (
        _blocking_categories(run_dir)
    )


def test_rows_must_be_a_list(tmp_path):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    payload["rows"] = {"CH1": payload["rows"][0]}
    _write_provenance(run_dir, payload)

    assert "applied_dff_rows_malformed" in _blocking_categories(run_dir)


@pytest.mark.parametrize("field_name", ["output_dir", "batch_strategy"])
def test_row_required_fields_must_be_nonblank(tmp_path, field_name):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    payload["rows"][0][field_name] = ""
    _write_provenance(run_dir, payload)

    assert "applied_dff_row_missing_required_field" in (
        _blocking_categories(run_dir)
    )


def test_row_status_domain_is_fail_closed(tmp_path):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    payload["rows"][0]["status"] = "done"
    _write_provenance(run_dir, payload)

    assert "applied_dff_row_status_invalid" in _blocking_categories(run_dir)


@pytest.mark.parametrize(
    ("family", "selected", "dynamic_mode"),
    [
        ("unsupported", "robust_global_event_reject", "robust_global_event_reject"),
        ("signal_only_f0", "signal_only_f0", "robust_global_event_reject"),
    ],
)
def test_requested_map_reuses_production_manifest_validation(
    tmp_path, family, selected, dynamic_mode
):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    entry = payload["requested_strategy_map"][
        "per_roi_production_strategy_map"
    ][0]
    entry.update(
        strategy_family=family,
        selected_strategy=selected,
        dynamic_fit_mode=dynamic_mode,
    )
    _write_provenance(run_dir, payload)

    assert "applied_dff_requested_strategy_map_invalid" in (
        _blocking_categories(run_dir)
    )


@pytest.mark.parametrize(
    "field_name",
    ["batch_manifest_path", "applied_output_root"],
)
def test_relative_top_level_path_escape_blocks(tmp_path, field_name):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    payload[field_name] = "../outside"
    _write_provenance(run_dir, payload)

    assert "path_escapes_run_dir" in _blocking_categories(run_dir)


def test_relative_row_output_escape_blocks(tmp_path):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    payload["rows"][0]["output_dir"] = "../../outside"
    _write_provenance(run_dir, payload)

    assert "applied_dff_row_output_dir_escapes_root" in (
        _blocking_categories(run_dir)
    )


def test_integration_valid_completed_run_no_provenance(qapp, tmp_path):
    """17A. RunReportViewer integration with absent provenance."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)

    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run_dir)) is True
        assert viewer.applied_dff_state.present is False
        assert len(viewer.applied_dff_state.blocking_issues()) == 0
        assert "not present" in viewer.applied_dff_summary_text
    finally:
        viewer.close()


def test_failed_viewer_load_does_not_expose_applied_dff_state(qapp, tmp_path):
    run_dir = _create_completed_run(tmp_path, None)
    payload = _valid_provenance_payload(run_dir)
    _write_provenance(run_dir, payload)
    (run_dir / "run_report.json").write_text("{malformed", encoding="utf-8")
    for child in (run_dir / "Region0").iterdir():
        child.rmdir()
    (run_dir / "Region0").rmdir()

    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run_dir)) is False
        assert viewer.applied_dff_state.present is False
        assert "succeeded" not in viewer.applied_dff_summary_text
    finally:
        viewer.close()


def test_integration_valid_completed_run_valid_provenance(qapp, tmp_path):
    """17B. RunReportViewer integration with valid provenance."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)
    payload = _valid_provenance_payload(run_dir)

    prov_dir = run_dir / "applied_dff"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / "guided_applied_dff_provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run_dir)) is True
        assert viewer.applied_dff_state.present is True
        assert viewer.applied_dff_state.overall_status == "succeeded"
        assert len(viewer.applied_dff_state.rows) == 1
        assert viewer.applied_dff_state.rows[0].roi_id == "CH1"
        assert viewer.applied_dff_state.has_blocking_issues is False
        assert "succeeded" in viewer.applied_dff_summary_text
        assert "CH1" in viewer.applied_dff_summary_text
        assert "dynamic_fit" in viewer.applied_dff_summary_text
    finally:
        viewer.close()


def test_integration_valid_completed_run_malformed_provenance(qapp, tmp_path):
    """17C. RunReportViewer integration with malformed provenance (should not crash ordinary load)."""
    run_dir = _create_completed_run(tmp_path, prov_payload="malformed json content")

    viewer = RunReportViewer()
    try:
        # Ordinary load still succeeds because run_report.json and Region0 exist
        assert viewer.load_report(str(run_dir)) is True
        assert viewer.applied_dff_state.present is True
        assert viewer.applied_dff_state.has_blocking_issues is True
        assert "invalid provenance" in viewer.applied_dff_summary_text
        assert (
            "applied_dff_provenance_malformed"
            in viewer.applied_dff_summary_text
        )
    finally:
        viewer.close()


def test_legacy_completed_run_compatibility(qapp, tmp_path):
    """18. Legacy run has present=False, no changes to normal load behavior."""
    run_dir = _create_completed_run(tmp_path, prov_payload=None)

    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run_dir)) is True
        assert viewer.applied_dff_state.present is False
        assert len(viewer.applied_dff_state.blocking_issues()) == 0
    finally:
        viewer.close()
