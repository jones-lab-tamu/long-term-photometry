"""Tests for guided_completed_feature_event_reload module and RunReportViewer
integration (4J16k35: completed-run Review display of per-ROI feature-detection
settings)."""

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from gui.run_report_viewer import RunReportViewer
from photometry_pipeline.guided_completed_feature_event_reload import (
    GuidedCompletedFeatureEventRow,
    GuidedCompletedFeatureEventState,
    format_guided_completed_feature_event_summary,
    format_guided_completed_feature_event_technical_details,
    load_guided_completed_feature_event_state,
)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _create_completed_run(tmp_path: Path, provenance_payload=None) -> Path:
    """Mirrors tests.test_guided_completed_applied_dff_reload._create_completed_run."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "success", "run_context": {"run_type": "full"}}),
        encoding="utf-8",
    )

    reg_dir = run_dir / "Region0"
    (reg_dir / "summary").mkdir(parents=True, exist_ok=True)
    (reg_dir / "day_plots").mkdir(parents=True, exist_ok=True)
    (reg_dir / "tables").mkdir(parents=True, exist_ok=True)

    if provenance_payload is not None:
        features_dir = run_dir / "_analysis" / "phasic_out" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        path = features_dir / "feature_event_provenance.json"
        if isinstance(provenance_payload, str):
            path.write_text(provenance_payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(provenance_payload), encoding="utf-8")

    return run_dir


def _valid_v2_payload():
    return {
        "schema_version": "guided_feature_event_provenance.v2",
        "rois": [
            {
                "roi": "CH1",
                "source": "override",
                "feature_event_profile_id": "custom-ch1",
                # Sparse: only the field the Custom profile actually set.
                "override_config_fields": {"peak_threshold_k": 10.0},
                "effective_config_fields": {
                    "event_signal": "dff",
                    "signal_excursion_polarity": "positive",
                    "peak_threshold_method": "mean_std",
                    "peak_threshold_k": 10.0,
                    "peak_threshold_percentile": 95.0,
                    "peak_threshold_abs": 0.0,
                    "peak_min_distance_sec": 1.0,
                    "peak_min_prominence_k": 2.0,
                    "peak_min_width_sec": 0.3,
                    "peak_pre_filter": "none",
                    "event_auc_baseline": "zero",
                },
            },
            {
                "roi": "CH2",
                "source": "default",
                "feature_event_profile_id": "profile-001",
                "override_config_fields": {},
                "effective_config_fields": {
                    "event_signal": "dff",
                    "signal_excursion_polarity": "positive",
                    "peak_threshold_method": "mean_std",
                    "peak_threshold_k": 2.5,
                    "peak_threshold_percentile": 95.0,
                    "peak_threshold_abs": 0.0,
                    "peak_min_distance_sec": 1.0,
                    "peak_min_prominence_k": 2.0,
                    "peak_min_width_sec": 0.3,
                    "peak_pre_filter": "none",
                    "event_auc_baseline": "zero",
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# A. Loader/parsing
# ---------------------------------------------------------------------------


def test_missing_file_returns_absent_state_no_error(tmp_path):
    run_dir = _create_completed_run(tmp_path, provenance_payload=None)

    state = load_guided_completed_feature_event_state(run_dir)

    assert state.present is False
    assert state.valid is True
    assert state.warning is None
    assert state.rows == ()


def test_valid_v2_file_returns_rows_for_all_rois(tmp_path):
    run_dir = _create_completed_run(tmp_path, provenance_payload=_valid_v2_payload())

    state = load_guided_completed_feature_event_state(run_dir)

    assert state.present is True
    assert state.valid is True
    assert state.warning is None
    assert {row.roi for row in state.rows} == {"CH1", "CH2"}
    by_roi = {row.roi: row for row in state.rows}
    assert by_roi["CH1"].source == "override"
    assert by_roi["CH2"].source == "default"


def test_malformed_file_returns_non_crashing_warning_state(tmp_path):
    run_dir = _create_completed_run(tmp_path, provenance_payload="{not json")

    state = load_guided_completed_feature_event_state(run_dir)

    assert state.present is True
    assert state.valid is False
    assert state.warning
    assert state.rows == ()


def test_unknown_schema_returns_non_crashing_warning_state(tmp_path):
    payload = _valid_v2_payload()
    payload["schema_version"] = "guided_feature_event_provenance.v99"
    run_dir = _create_completed_run(tmp_path, provenance_payload=payload)

    state = load_guided_completed_feature_event_state(run_dir)

    assert state.present is True
    assert state.valid is False
    assert state.warning
    assert state.rows == ()


def test_absent_state_factory_matches_missing_file_loader_result():
    assert GuidedCompletedFeatureEventState.absent() == load_guided_completed_feature_event_state(
        None
    )


# ---------------------------------------------------------------------------
# B. Completed-run Review display
# ---------------------------------------------------------------------------


def test_review_shows_custom_and_default_labels_and_hides_internal_language(
    tmp_path, qapp
):
    run_dir = _create_completed_run(tmp_path, provenance_payload=_valid_v2_payload())

    viewer = RunReportViewer()
    assert viewer.load_report(str(run_dir)) is True

    summary = viewer.feature_event_summary_text
    assert "CH1: Custom" in summary
    assert "CH2: Default" in summary
    assert not viewer._feature_event_summary_label.isHidden()

    for banned in (
        "backend",
        "manifest",
        "artifact",
        "schema",
        "contract",
        "resolver",
        "materialization",
        "Config",
        "dataclass",
        "typed field",
    ):
        assert banned.lower() not in summary.lower()

    viewer.close()


def test_review_missing_file_loads_normally_without_alarming_error(tmp_path, qapp):
    run_dir = _create_completed_run(tmp_path, provenance_payload=None)

    viewer = RunReportViewer()
    assert viewer.load_report(str(run_dir)) is True

    # Section hidden entirely -- global-only runs stay visually unchanged.
    assert viewer._feature_event_summary_label.isHidden() is True
    assert viewer._feature_event_details_toggle.isHidden() is True
    assert "error" not in viewer._status_label.text().lower()

    viewer.close()


def test_review_malformed_file_still_loads_with_calm_message(tmp_path, qapp):
    run_dir = _create_completed_run(tmp_path, provenance_payload="{not json")

    viewer = RunReportViewer()
    assert viewer.load_report(str(run_dir)) is True

    # Not a crash, and not an alarming top-level error -- the plain-language
    # note lives in the summary line, with the real detail in technical
    # details only.
    summary = viewer.feature_event_summary_text
    assert "available in technical details" in summary
    assert "error" not in summary.lower()
    assert "could not be read" not in summary.lower()

    details = viewer.feature_event_technical_details_text
    assert details  # non-empty plain-language technical note
    assert "traceback" not in details.lower()

    viewer.close()


# ---------------------------------------------------------------------------
# C. Effective-settings source (sparse override vs complete effective fields)
# ---------------------------------------------------------------------------


def test_sparse_override_fields_do_not_cause_blank_display():
    """A row with sparse override_config_fields but complete
    effective_config_fields must display the inherited effective fields, not
    a blank summary caused by reading the sparse fields instead."""
    row = GuidedCompletedFeatureEventRow(
        roi="CH1",
        source="override",
        feature_event_profile_id="custom-ch1",
        effective_config_fields={
            "event_signal": "dff",
            "peak_threshold_method": "percentile",
            "peak_threshold_percentile": 80.0,
        },
    )
    state = GuidedCompletedFeatureEventState(present=True, valid=True, rows=(row,))

    summary = format_guided_completed_feature_event_summary(state)
    assert "percentile threshold (80.0)" in summary
    assert "dff signal" in summary

    details = format_guided_completed_feature_event_technical_details(state)
    assert "peak_threshold_percentile: 80.0" in details


def test_default_only_summary_uses_approved_phrasing():
    row = GuidedCompletedFeatureEventRow(
        roi="CH1",
        source="default",
        feature_event_profile_id="profile-001",
        effective_config_fields={"event_signal": "dff", "peak_threshold_k": 2.5},
    )
    state = GuidedCompletedFeatureEventState(present=True, valid=True, rows=(row,))

    summary = format_guided_completed_feature_event_summary(state)
    assert "one default setting set was used for all ROIs" in summary


def test_absent_state_summary_is_empty_and_technical_details_are_plain():
    state = GuidedCompletedFeatureEventState.absent()

    assert format_guided_completed_feature_event_summary(state) == ""
    details = format_guided_completed_feature_event_technical_details(state)
    assert "No per-ROI feature-detection settings were recorded" in details
