"""The natural Guided path for one continuous generic CSV recording.

Nothing on the source side is mocked: the real candidate-file resolution, the
real CSV mapping controls, the real inspector, the real normalized recording
builder, and the real review binding all run.
"""

import json
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.synthetic_demo_generator import (
    GUIDED_CONTINUOUS_DEMO_FILE_NAME,
    generate_guided_continuous_demo,
)


pytestmark = pytest.mark.usefixtures("no_real_modals")

BOUNDED_DURATION_SEC = 1200.0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def continuous_csv_folder(tmp_path_factory) -> Path:
    result = generate_guided_continuous_demo(
        tmp_path_factory.mktemp("gui_csv"), _duration_sec=BOUNDED_DURATION_SEC
    )
    assert result.success, result.message
    return result.input_dir


@pytest.fixture
def tmp_output(tmp_path) -> Path:
    destination = tmp_path / "results"
    destination.mkdir()
    return destination


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _pump(predicate, *, limit: int = 20000) -> bool:
    for _ in range(limit):
        if predicate():
            return True
        QCoreApplication.processEvents()
    return predicate()


def _select_continuous_csv(window, folder: Path) -> None:
    """Drive the ordinary Select-data controls, as a scientist would."""
    window._set_guided_workflow_mode("new_analysis")
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_format_combo.setCurrentText("custom_tabular")
    window._refresh_guided_csv_source_interpretation()
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )


def _discover_rois(window) -> None:
    """Press Find ROIs and wait for the real threaded discovery to finish."""
    window._on_guided_discover_rois()
    assert _pump(
        lambda: not window._guided_roi_discovery_running
        and window._guided_roi_discovery_thread is None
    ), window._guided_discovery_summary_label.text()


def _roi_texts(roi_list) -> list[str]:
    return [roi_list.item(index).text() for index in range(roi_list.count())]


def _map_columns(window) -> None:
    """Use the same CSV interpretation panel the intermittent workflow uses."""
    combo = window._guided_csv_time_column_combo
    combo.setCurrentIndex(combo.findData("ElapsedSeconds"))
    units = window._guided_csv_time_units_combo
    units.setCurrentIndex(units.findData("seconds"))
    window._guided_csv_order_confirm_cb.setChecked(True)
    while len(window._guided_csv_mapping_rows) < 2:
        window._add_guided_csv_mapping_row()
    for row, roi in zip(window._guided_csv_mapping_rows, ("ROI1", "ROI2")):
        row["name"].setText(roi)
        row["signal"].setCurrentIndex(row["signal"].findData(f"{roi}_Signal"))
        row["reference"].setCurrentIndex(
            row["reference"].findData(f"{roi}_Reference")
        )


def _set_timeline(window) -> None:
    """A generic CSV knows only elapsed seconds, so the clock is stated here."""
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    window._guided_recording_start_clock_edit.setText("12:00:00")
    QCoreApplication.processEvents()


def _confirm_corrections(window) -> None:
    """Drive the ordinary Step 4 preview-and-confirm path for every ROI."""
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()
    for checkbox in window._guided_preview_method_checkboxes.values():
        checkbox.setChecked(True)
    for index in range(window._guided_preview_roi_combo.count()):
        window._guided_preview_roi_combo.setCurrentIndex(index)
        window._on_generate_guided_correction_preview()
        assert _pump(
            lambda: not window._guided_correction_preview_running
            and window._guided_correction_preview_thread is None,
            limit=400000,
        ), window._guided_preview_status_label.text()
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()
    rows = dict(window._guided_local_preview_confirmation_rows)
    assert sorted(rows) == ["ROI1", "ROI2"], window._guided_preview_status_label.text()
    for row in rows.values():
        combo = row["strategy_combo"]
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            if combo.currentData():
                break
        row["action_button"].click()
        QCoreApplication.processEvents()


def test_one_csv_file_reaches_a_bound_continuous_recording(
    window, continuous_csv_folder, tmp_output
):
    _select_continuous_csv(window, continuous_csv_folder)

    # The continuous choice is offered for exactly one CSV file.
    available, reason = window._guided_continuous_csv_availability()
    assert available is True, reason

    _map_columns(window)

    # Real discovery reads the CSV and is the only thing that names the ROIs.
    _discover_rois(window)
    assert _roi_texts(window._roi_list) == ["ROI1", "ROI2"]
    assert _roi_texts(window._guided_roi_list) == ["ROI1", "ROI2"]

    # The mapping the panel shows is the mapping the check will use.
    fields = window._guided_continuous_csv_check_fields()
    assert fields["source_format"] == "custom_tabular"
    assert Path(fields["csv_source_path"]).name == GUIDED_CONTINUOUS_DEMO_FILE_NAME
    assert fields["csv_time_column"] == "ElapsedSeconds"
    assert fields["csv_time_unit"] == "seconds"
    assert fields["csv_roi_columns"] == (
        ("ROI1", "ROI1_Signal", "ROI1_Reference"),
        ("ROI2", "ROI2_Signal", "ROI2_Reference"),
    )

    draft = window._guided_continuous_rwd_live_draft()
    assert draft is not None
    assert draft.input_format == "custom_tabular"
    assert draft.acquisition_mode == "continuous"
    # ROI identity reaches the draft through discovery, not through the panel.
    assert tuple(draft.discovered_roi_ids) == ("ROI1", "ROI2")
    assert tuple(draft.included_roi_ids) == ("ROI1", "ROI2")
    assert tuple(roi for roi, _sig, _ref in fields["csv_roi_columns"]) == tuple(
        draft.included_roi_ids
    )

    # The real recording check runs the real CSV inspector on a worker thread.
    assert window._maybe_start_guided_continuous_rwd_recording_check() is True
    assert _pump(
        lambda: window._guided_continuous_rwd_check_thread is None
    ), window._guided_continuous_rwd_status_message

    prepared = window._guided_continuous_rwd_review_binding
    assert prepared is not None, window._guided_continuous_rwd_status_message
    recording = prepared.recording
    assert recording.source_format == "custom_tabular"
    assert recording.acquisition_mode == "continuous"
    assert recording.source.header_row_index == 0
    assert recording.source.selected_time_column == "ElapsedSeconds"
    assert tuple(recording.roi.included_roi_ids) == ("ROI1", "ROI2")
    assert Path(prepared.current_source_path).name == GUIDED_CONTINUOUS_DEMO_FILE_NAME

    # The CSV source has carried all the way to an accepted continuous plan,
    # which is the state the later Guided steps read.
    window._guided_output_dir_edit.setText(str(tmp_output))
    assert window._guided_continuous_rwd_accepted_plan() is not None

    # Run stays closed until the correction and Feature Detection steps are
    # done. The CSV source must never open Run on its own.
    assert window._guided_run_btn.isEnabled() is False

    # Scientist-facing text carries no internal vocabulary.
    visible = (
        window._guided_continuous_rwd_check_status_label.text(),
        window._guided_discovery_summary_label.text(),
    )
    assert all(text.strip() for text in visible)
    for text in visible:
        for forbidden in ("RWD", "CR1", "custom_tabular", "contract", "target grid"):
            assert forbidden not in text, text


def test_mapping_change_during_the_check_discards_the_result(
    window, continuous_csv_folder
):
    _select_continuous_csv(window, continuous_csv_folder)
    _map_columns(window)
    _discover_rois(window)
    assert window._maybe_start_guided_continuous_rwd_recording_check() is True

    # Change the mapping while the inspection is still in flight.
    row = window._guided_csv_mapping_rows[0]
    row["signal"].setCurrentIndex(row["signal"].findData("ROI2_Signal"))
    row["reference"].setCurrentIndex(row["reference"].findData("ROI2_Reference"))

    assert _pump(lambda: window._guided_continuous_rwd_check_thread is None)

    # The completed inspection must not be installed, and Run must not open.
    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_continuous_rwd_prepared_run is None
    assert window._guided_run_btn.isEnabled() is False

    # The scientist is told the setup moved, not that the source was bad.
    status = window._guided_continuous_rwd_check_status_label.text().lower()
    assert "changed" in status, status
    assert "check the recording again" in status, status


def _reach_review(window, folder, output):
    """Everything up to and including review binding, through real controls."""
    _select_continuous_csv(window, folder)
    window._guided_output_dir_edit.setText(str(output))
    _map_columns(window)
    _set_timeline(window)
    _discover_rois(window)
    for index in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(index).setCheckState(Qt.Checked)
    QCoreApplication.processEvents()
    assert window._maybe_start_guided_continuous_rwd_recording_check() is True
    assert _pump(lambda: window._guided_continuous_rwd_check_thread is None)
    assert window._guided_continuous_rwd_review_binding is not None


def test_continuous_csv_reaches_run_through_the_ordinary_later_steps(
    window, continuous_csv_folder, tmp_output
):
    """Correction, Feature Detection, and preparation are format-neutral.

    Nothing is injected: the correction preview really reads the CSV, the
    strategies are really confirmed per ROI, and the real preparation worker
    builds the run.
    """
    _reach_review(window, continuous_csv_folder, tmp_output)

    _confirm_corrections(window)
    draft = window._build_guided_new_analysis_draft_plan()
    assert [
        (choice.roi_id, choice.selected_strategy)
        for choice in draft.per_roi_correction_strategy_choices
    ] == [
        ("ROI1", "robust_global_event_reject"),
        ("ROI2", "robust_global_event_reject"),
    ]
    assert window._guided_correction_continue_btn.isEnabled() is True

    window._on_guided_continue_to_feature_detection()
    QCoreApplication.processEvents()
    ready, message = window._guided_feature_detection_readiness()
    assert ready is True, message
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.feature_event_values

    assert window._maybe_start_guided_continuous_rwd_preparation() is True
    assert _pump(
        lambda: not window._guided_continuous_rwd_preparation_active()
    ), window._guided_continuous_rwd_status_message
    assert (
        window._guided_continuous_rwd_prepared_run is not None
    ), window._guided_continuous_rwd_status_message
    assert window._guided_run_btn.isEnabled() is True


def test_continuous_csv_runs_to_a_completed_continuous_run(
    window, continuous_csv_folder, tmp_output
):
    """One real bounded execution, ending in a successful completed run."""
    _reach_review(window, continuous_csv_folder, tmp_output)
    _confirm_corrections(window)
    window._on_guided_continue_to_feature_detection()
    QCoreApplication.processEvents()
    assert window._maybe_start_guided_continuous_rwd_preparation() is True
    assert _pump(lambda: not window._guided_continuous_rwd_preparation_active())
    assert window._guided_run_btn.isEnabled() is True

    window._on_guided_run_clicked_backend_guarded()
    assert _pump(
        lambda: not window._guided_continuous_rwd_execution_active, limit=400000
    ), window._guided_continuous_rwd_status_message

    run_dir = Path(window._guided_continuous_rwd_completed_run_dir)
    assert run_dir.is_dir(), window._guided_continuous_rwd_status_message
    assert json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )["status"] == "success"

    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["source"]["acquisition_mode"] == "continuous"
    assert report["timeline"]["fixed_daily_anchor_clock"] == "07:00"
    assert report["timeline"]["recording_start_clock"] == "12:00"

    # Both ROIs corrected, with tonic and phasic summaries and Day Plots.
    for roi in ("ROI1", "ROI2"):
        assert (run_dir / roi / "tables" / "continuous_tonic_window_summary.csv").exists()
        assert (run_dir / roi / "tables" / "continuous_phasic_window_summary.csv").exists()
        assert list((run_dir / roi / "day_plots").glob("*.png"))
    events = run_dir / "_analysis" / "phasic_out" / "features" / "continuous_phasic_events.csv"
    assert sum(1 for _ in events.open(encoding="utf-8")) - 1 > 0


def _confirm_dataset_settings(window) -> None:
    """Press the visible Review Plan confirmation, as continuous RWD needs too."""
    button = window._guided_review_dataset_contract_action_btn
    assert button.isHidden() is False, window._guided_review_plan_status_label.text()
    button.click()
    _pump(lambda: not window._guided_dataset_contract_confirmation_active)
    QCoreApplication.processEvents()


def _reach_review_plan(window, folder, output):
    """The ordinary workflow, driven by the real Continue buttons."""
    _select_continuous_csv(window, folder)
    window._guided_output_dir_edit.setText(str(output))
    _map_columns(window)
    _set_timeline(window)
    _discover_rois(window)
    for index in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(index).setCheckState(Qt.Checked)
    QCoreApplication.processEvents()

    window._on_guided_continue_to_recording_structure()
    QCoreApplication.processEvents()
    window._on_guided_continue_to_correction_approach()
    QCoreApplication.processEvents()
    assert _pump(
        lambda: window._guided_continuous_rwd_check_thread is None, limit=200000
    ), window._guided_continuous_rwd_check_status_label.text()
    assert window._guided_continuous_rwd_review_binding is not None

    _confirm_corrections(window)
    window._on_guided_continue_to_feature_detection()
    QCoreApplication.processEvents()
    window._guided_feature_event_apply_btn.click()
    QCoreApplication.processEvents()
    window._on_guided_continue_to_review_plan()
    QCoreApplication.processEvents()


def _go_to_run_and_check_setup(window) -> None:
    """Go to Run, then press Check my setup, which prepares the run.

    Run-page entry is deliberately passive for both Guided modes: continuous
    preparation starts only from this button, exactly as it does for RWD.
    """
    window._guided_review_go_to_run_btn.click()
    QCoreApplication.processEvents()
    window._guided_backend_validate_btn.click()
    QCoreApplication.processEvents()
    assert _pump(
        lambda: not window._guided_continuous_rwd_preparation_active(), limit=400000
    ), window._guided_continuous_rwd_status_message


def test_continuous_csv_review_plan_reaches_an_enabled_go_to_run(
    window, continuous_csv_folder, tmp_output
):
    """Review Plan must not call a supported continuous CSV plan unsupported."""
    from photometry_pipeline.guided_new_analysis_plan import (
        evaluate_guided_new_analysis_execution_subset_readiness,
        evaluate_new_analysis_plan_readiness,
    )

    _reach_review_plan(window, continuous_csv_folder, tmp_output)

    plan = window._build_guided_new_analysis_draft_plan()
    assert evaluate_new_analysis_plan_readiness(plan).plan_complete_for_handoff is True

    # Before confirmation the scientist is offered the same confirmation
    # continuous RWD is offered -- never "Guided Run does not yet support".
    status = window._guided_review_plan_status_label.text()
    assert "Plan completeness: Complete" in status
    assert "does not yet support this configuration" not in status
    assert "detected dataset settings have not been confirmed yet" in status

    _confirm_dataset_settings(window)

    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.input_format == "custom_tabular"
    assert plan.acquisition_mode == "continuous"
    assert plan.dataset_contract_snapshot.current_applied is True
    values = dict(plan.dataset_contract_snapshot.contract_values or {})
    # The confirmed contract is the mapping the recording check accepted, and
    # carries no session-ordering fields, which one file does not have.
    assert values["custom_tabular_time_col"] == "ElapsedSeconds"
    assert values["custom_tabular_time_unit"] == "seconds"
    assert "ROI1_Signal" in values["custom_tabular_roi_mapping_json"]
    assert "ROI2_Signal" in values["custom_tabular_roi_mapping_json"]
    assert "custom_tabular_ordered_source_files_json" not in values

    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    assert subset.first_subset_executable is True, subset.blocking_issues

    status = window._guided_review_plan_status_label.text()
    assert "This plan is ready" in status
    assert "does not yet support this configuration" not in status
    assert "Full Control" not in status
    for forbidden in ("custom_tabular", "CR1", "RWD", "producer"):
        assert forbidden not in status
    assert window._guided_review_go_to_run_btn.isEnabled() is True


def test_continuous_csv_runs_from_the_guided_run_button(
    window, continuous_csv_folder, tmp_output
):
    """Go to Run, then the ordinary Run button, then a completed run."""
    _reach_review_plan(window, continuous_csv_folder, tmp_output)
    _confirm_dataset_settings(window)
    assert window._guided_review_go_to_run_btn.isEnabled() is True

    _go_to_run_and_check_setup(window)
    assert (
        window._guided_continuous_rwd_prepared_run is not None
    ), window._guided_continuous_rwd_status_message
    assert window._guided_run_btn.isEnabled() is True

    window._guided_run_btn.click()
    assert _pump(
        lambda: not window._guided_continuous_rwd_execution_active, limit=600000
    ), window._guided_continuous_rwd_status_message

    run_dir = Path(window._guided_continuous_rwd_completed_run_dir)
    assert run_dir.is_dir(), window._guided_continuous_rwd_status_message
    assert json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )["status"] == "success"
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["source"]["acquisition_mode"] == "continuous"
    for roi in ("ROI1", "ROI2"):
        assert (run_dir / roi / "tables" / "continuous_tonic_window_summary.csv").exists()
        assert (run_dir / roi / "tables" / "continuous_phasic_window_summary.csv").exists()
        assert list((run_dir / roi / "day_plots").glob("*.png"))
    events = (
        run_dir / "_analysis" / "phasic_out" / "features" / "continuous_phasic_events.csv"
    )
    assert sum(1 for _ in events.open(encoding="utf-8")) - 1 > 0


def test_changing_the_setup_after_confirmation_closes_run_again(
    window, continuous_csv_folder, tmp_output
):
    """Currentness is not weakened: a later edit must withdraw Run."""
    _reach_review_plan(window, continuous_csv_folder, tmp_output)
    _confirm_dataset_settings(window)
    assert window._guided_review_go_to_run_btn.isEnabled() is True

    _go_to_run_and_check_setup(window)
    assert window._guided_run_btn.isEnabled() is True

    # The scientist changes the analysis window after preparing. The plan is
    # still valid, so continuous readiness still owns Run -- and must withdraw
    # it, because the prepared run belongs to the previous plan.
    window._continuous_window_sec_spin.setValue(
        float(window._continuous_window_sec_spin.value()) + 60.0
    )
    QCoreApplication.processEvents()

    assert window._guided_continuous_rwd_prepared_run is None
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_continuous_rwd_run_readiness()[0] is False
