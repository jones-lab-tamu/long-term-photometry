import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

import gui.main_window as main_window_module
import photometry_pipeline.preview.correction_preview as correction_preview_module
from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.guided_run_plan import (
    CorrectionStrategyChoice,
    EvidenceChunkReview,
    FeatureEventProfile,
    GuidedPlanSource,
    GuidedRunPlan,
    OutputPolicy,
    RoiPlanEntry,
    plan_export_json_text,
)

# An unstubbed modal blocks an unattended run on a dialog nobody can dismiss.
pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _tab_labels(window: MainWindow) -> list[str]:
    tabs = window._workflow_mode_tabs
    return [tabs.tabText(i) for i in range(tabs.count())]


def _label_texts(widget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel)]


def _make_window(qapp) -> MainWindow:
    return MainWindow()


def _close_window(window: MainWindow) -> None:
    window.close()
    window.deleteLater()


def _populate_fake_discovery(window: MainWindow) -> None:
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": 2,
        "n_preview": 2,
        "sessions": [{"session_id": "s1"}, {"session_id": "s2"}],
        "rois": [{"roi_id": "CH1"}, {"roi_id": "CH2"}, {"roi_id": "CH3"}],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)


class _FakeDiagnosticCacheRunner:
    def __init__(self):
        self.argv = None
        self.run_dir = ""
        self.state = main_window_module.RunnerState.IDLE
        self.final_status_code = None
        self.final_errors = []
        self._running = False

    def is_running(self):
        return self._running

    def set_run_dir(self, run_dir):
        self.run_dir = run_dir

    def start(self, argv, state):
        self.argv = list(argv)
        self.state = state
        self._running = True

    def succeed(self):
        self.state = main_window_module.RunnerState.SUCCESS
        self.final_status_code = "success"
        self.final_errors = []
        self._running = False

    def fail(self, errors=None):
        self.state = main_window_module.RunnerState.FAILED
        self.final_status_code = "error"
        self.final_errors = list(errors or [])
        self._running = False


def _configure_guided_raw_cache_setup(window: MainWindow, tmp_path, monkeypatch):
    input_dir = tmp_path / "raw_input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._mode_combo.setCurrentText("both")
    idx = window._format_combo.findText("rwd")
    window._format_combo.setCurrentIndex(idx)
    _populate_fake_discovery(window)
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        lambda _fmt: {
            "rwd_time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        },
    )
    return input_dir, output_dir


def _write_minimal_guided_cache_outputs(
    cache_dir, rois=("CH1", "CH2", "CH3")
):
    phasic = cache_dir / "_analysis" / "phasic_out"
    phasic.mkdir(parents=True)
    (phasic / "config_used.yaml").write_text(
        "target_fs_hz: 20.0\nlowpass_hz: 1.0\nfilter_order: 3\n"
        "dynamic_fit_mode: robust_global_event_reject\n",
        encoding="utf-8",
    )
    t = np.arange(400, dtype=float) / 20.0
    uv = 1.0 + 0.02 * np.sin(t * 0.7)
    sig = 1.2 * uv + 0.05 * np.exp(-0.5 * ((t - 8.0) / 0.5) ** 2)
    with h5py.File(phasic / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset(
            "rois", data=np.asarray([roi.encode("utf-8") for roi in rois])
        )
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"mock0.csv", b"mock1.csv"]))
        for roi in rois:
            roi_group = h5.create_group(f"roi/{roi}")
            for chunk_id in (0, 1):
                grp = roi_group.create_group(f"chunk_{chunk_id}")
                grp.create_dataset("time_sec", data=t)
                grp.create_dataset("sig_raw", data=sig + chunk_id)
                grp.create_dataset("uv_raw", data=uv + 0.1 * chunk_id)
    (cache_dir / "run_report.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (cache_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )


def _make_preview_completed_run(tmp_path):
    run_dir = tmp_path / "completed_preview"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "success", "configuration": {}, "analytical_contract": {}}),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (run_dir / "CH1" / "summary").mkdir(parents=True)
    (phasic_out / "config_used.yaml").write_text(
        "target_fs_hz: 20.0\nlowpass_hz: 1.0\nfilter_order: 3\n"
        "dynamic_fit_mode: robust_global_event_reject\n",
        encoding="utf-8",
    )
    t = np.arange(400, dtype=float) / 20.0
    uv = 1.0 + 0.02 * np.sin(t * 0.7)
    sig = 1.2 * uv + 0.05 * np.exp(-0.5 * ((t - 8.0) / 0.5) ** 2)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([b"CH1", b"CH2"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"mock0.csv", b"mock1.csv"]))
        for roi in ("CH1", "CH2"):
            roi_group = h5.create_group(f"roi/{roi}")
            for chunk_id in (0, 1):
                grp = roi_group.create_group(f"chunk_{chunk_id}")
                grp.create_dataset("time_sec", data=t)
                grp.create_dataset("sig_raw", data=sig + chunk_id)
                grp.create_dataset("uv_raw", data=uv + 0.1 * chunk_id)
    return run_dir


def _load_preview_completed_run(window, run_dir, monkeypatch):
    window._current_run_dir = str(run_dir)
    monkeypatch.setattr(window._report_viewer, "has_loaded_results", lambda: True)
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_diagnostics_panel()


def _write_guided_plan_json(path, plan: GuidedRunPlan) -> None:
    path.write_text(plan_export_json_text(plan), encoding="utf-8")


def _review_plan(window: MainWindow, path) -> str:
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_imported_plan_path_edit.setText(str(path))
    window._guided_imported_plan_open_btn.click()
    return window._guided_imported_plan_summary_label.text()


def _adoption_status_text(window: MainWindow) -> str:
    return window._guided_imported_plan_adoption_status_label.text()


def _candidate_plan(run_dir, *, rois=None, strategy="robust_global_event_reject", profile=False, output_policy=None):
    entries = []
    for roi in rois or []:
        entries.append(
            RoiPlanEntry(
                roi=roi,
                correction_strategy=CorrectionStrategyChoice(
                    strategy=strategy,
                    strategy_label="Signal-Only F0" if strategy == "signal_only_f0" else "Robust Global Event-Reject Fit",
                ),
                evidence=[EvidenceChunkReview(chunk_id=0)],
            )
        )
    profiles = []
    if profile:
        profiles.append(
            FeatureEventProfile(
                profile_id="default",
                profile_label="Default",
                scope="run",
                status="draft",
                config_fields={
                    "event_signal": "dff",
                    "signal_excursion_polarity": "positive",
                    "peak_threshold_method": "mean_std",
                    "peak_threshold_k": 3.0,
                    "peak_threshold_percentile": 95.0,
                    "peak_threshold_abs": 0.1,
                    "peak_min_distance_sec": 1.0,
                    "peak_min_prominence_k": 2.0,
                    "peak_min_width_sec": 0.3,
                    "peak_pre_filter": "none",
                    "event_auc_baseline": "zero",
                },
            )
        )
    return GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir=str(run_dir.resolve())),
        roi_plan=entries,
        feature_event_profiles=profiles,
        output_policy=output_policy or OutputPolicy(),
    )


def _assert_no_active_imported_plan_candidate(window: MainWindow) -> None:
    summary = window._guided_imported_plan_summary_label.text()
    adoption = _adoption_status_text(window)
    assert window._guided_imported_plan_candidate is None
    assert window._guided_imported_plan_file_path == ""
    assert window._guided_imported_plan_status == ""
    assert "Candidate review: none." in summary
    assert "Adoption: unavailable in this read-only stage" in summary
    assert "Execution: blocked" in summary
    assert "Files written: none" in summary
    assert "Future adoption eligible: No" in adoption
    assert "no active candidate" in adoption
    assert "Adoption action: unavailable in this stage" in adoption
    assert "Execution: blocked" in adoption
    assert "Files written: none" in adoption


def _state_for_equivalence(window: MainWindow) -> dict[str, object]:
    state = dict(window._guided_setup_summary_state())
    return {
        key: state[key]
        for key in (
            "input_dir",
            "output_dir",
            "format",
            "resolved_format",
            "acquisition_mode",
            "sessions_per_hour",
            "session_duration_s",
            "continuous_window_sec",
            "continuous_step_sec",
            "allow_partial_final_window",
            "exclude_incomplete_final_rwd_chunk",
            "selected_roi_count",
            "total_roi_count",
            "selected_rois",
            "reference_correction_method",
            "reference_correction_label",
            "guided_correction_intent",
        )
    }


GUIDED_CARD_TO_DYNAMIC_MODE = {
    "Robust Global Event-Reject Fit": "robust_global_event_reject",
    "Adaptive Event-Gated Fit": "adaptive_event_gated_regression",
    "Global Linear Regression": "global_linear_regression",
}


def test_guided_workflow_and_full_control_tabs_are_accessible(window):
    assert _tab_labels(window) == ["Guided Workflow", "Full Control"]
    assert window._guided_workflow_tab.objectName() == "guidedWorkflowShell"
    assert window._full_control_tab.objectName() == "fullControlShell"


def test_guided_workflow_stepper_has_expected_steps(window):
    stepper = window._guided_workflow_stepper
    assert stepper.count() == len(GUIDED_WORKFLOW_STEPS)
    assert [stepper.item(i).data(0x0100) for i in range(stepper.count())] == list(GUIDED_WORKFLOW_STEPS)
    assert [stepper.item(i).text() for i in range(stepper.count())] == [
        f"{idx}. {main_window_module.GUIDED_WORKFLOW_STEP_DISPLAY_LABELS.get(step, step)}"
        for idx, step in enumerate(GUIDED_WORKFLOW_STEPS, start=1)
    ]


def test_guided_workflow_stepper_switches_placeholder_panels(window):
    expected_panels = [
        "guidedStepStart",
        "guidedStepSelectData",
        "guidedStepRecordingStructure",
        "guidedStepCorrectionApproach",
        "guidedStepFeatureDetection",
        "guidedStepDraftPlan",
        "guidedStepRun",
        "guidedStepReview",
    ]
    for idx, expected_name in enumerate(expected_panels):
        window._guided_workflow_stepper.setCurrentRow(idx)
        assert window._guided_workflow_stack.currentWidget().objectName() == expected_name


def test_guided_step_scroll_areas_are_width_resizable_without_page_horizontal_scroll(window):
    for step_name in GUIDED_WORKFLOW_STEPS:
        idx = list(GUIDED_WORKFLOW_STEPS).index(step_name)
        window._guided_workflow_stepper.setCurrentRow(idx)
        scroll = window._guided_workflow_stack.currentWidget()
        assert isinstance(scroll, QScrollArea)
        assert scroll.widgetResizable() is True
        assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


def test_guided_start_step_exists_first_with_raw_setup_and_open_results_choices(window):
    stepper = window._guided_workflow_stepper
    assert stepper.item(0).data(Qt.UserRole) == "Start"
    assert stepper.item(1).data(Qt.UserRole) == "Select data"
    window._guided_workflow_stepper.setCurrentRow(0)

    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepStart"
    setup_card = window._guided_workflow_tab.findChild(QGroupBox, "guidedStartSetupNewAnalysisCard")
    open_card = window._guided_workflow_tab.findChild(QGroupBox, "guidedStartOpenResultsCard")
    assert setup_card is not None
    assert open_card is not None
    assert window._guided_workflow_tab.findChild(QGroupBox, "guidedStartStatusPanel") is None
    assert "Set up a new analysis" in setup_card.title()
    assert "Open results from a completed run" in open_card.title()
    assert "Input folder:" in window._guided_start_setup_status_label.text()
    assert "Completed run:" in window._guided_start_open_status_label.text()
    assert window._guided_start_setup_btn.text() == "Set up new analysis"
    assert window._guided_start_open_results_btn.text() == "Open Results..."


def test_guided_mode_banner_initially_distinguishes_input_from_completed_results(window):
    text = window._guided_mode_banner_label.text()
    assert text.startswith("Mode: choose a starting path")
    assert "new analysis" in text
    assert "completed results" in text
    assert "\n" not in text


def test_guided_start_setup_new_analysis_navigates_without_loading_or_generating(
    window, tmp_path, monkeypatch
):
    input_dir = tmp_path / "raw_input"
    input_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))
    calls = {"open": 0, "preview": 0, "signal": 0}
    monkeypatch.setattr(window, "_prompt_open_completed_results", lambda: calls.__setitem__("open", calls["open"] + 1) or True)
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )

    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepSelectData"
    assert window._guided_mode_banner_label.text().startswith("New analysis using:")
    assert "Completed results:" not in window._guided_mode_banner_label.text()
    assert "Mode:" not in window._guided_mode_banner_label.text()
    assert window._guided_input_dir_edit.text() == str(input_dir)
    assert all(banner.isHidden() for banner in window._guided_skipped_setup_banners.values())
    assert all(not controls.isHidden() for controls in window._guided_raw_setup_controls.values())
    assert calls == {"open": 0, "preview": 0, "signal": 0}


def test_guided_steps_one_through_four_use_reached_navigation_contract(
    window, tmp_path
):
    stepper = window._guided_workflow_stepper
    enabled = Qt.ItemIsEnabled
    selectable = Qt.ItemIsSelectable

    assert stepper.currentRow() == 0
    assert stepper.item(0).flags() & enabled
    assert stepper.item(0).flags() & selectable
    assert all(
        not stepper.item(index).flags() & enabled
        for index in range(1, stepper.count())
    )
    step_two_rect = stepper.visualItemRect(stepper.item(1))
    QTest.mouseClick(
        stepper.viewport(), Qt.LeftButton, pos=step_two_rect.center()
    )
    assert stepper.currentRow() == 0

    window._guided_start_setup_btn.click()
    assert stepper.currentRow() == 1
    assert stepper.item(1).flags() & enabled
    assert not stepper.item(2).flags() & enabled
    assert window._guided_select_data_continue_btn.isEnabled() is False

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._guided_sessions_per_hour_edit.setText("")
    window._guided_session_duration_edit.setText("")
    _populate_fake_discovery(window)

    assert window._guided_select_data_continue_btn.isEnabled() is True
    window._guided_select_data_continue_btn.click()
    assert stepper.currentRow() == 2
    assert stepper.item(2).flags() & enabled
    assert window._guided_recording_continue_btn.isEnabled() is False
    assert not stepper.item(3).flags() & enabled

    window._guided_sessions_per_hour_edit.setText("12")
    window._guided_session_duration_edit.setText("60")
    assert window._guided_recording_continue_btn.isEnabled() is True
    window._guided_recording_continue_btn.click()

    assert stepper.currentRow() == 3
    assert stepper.item(3).flags() & enabled
    assert not stepper.item(4).flags() & enabled
    stepper.setCurrentRow(1)
    assert window._guided_workflow_stack.currentWidget().objectName() == (
        "guidedStepSelectData"
    )
    stepper.setCurrentRow(2)
    assert window._guided_workflow_stack.currentWidget().objectName() == (
        "guidedStepRecordingStructure"
    )


def test_guided_start_open_results_uses_shared_loader_and_navigates_to_diagnostics(
    window, tmp_path, monkeypatch
):
    run_dir = _make_preview_completed_run(tmp_path)
    raw_input = tmp_path / "raw_input"
    raw_input.mkdir()
    window._guided_input_dir_edit.setText(str(raw_input))
    calls = {"open": 0}

    def _fake_open(path):
        calls["open"] += 1
        assert path == str(run_dir)
        window._current_run_dir = str(run_dir)
        return True

    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_args: str(run_dir))
    monkeypatch.setattr(window, "_open_completed_results_dir", _fake_open)

    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_open_results_btn.click()

    assert calls["open"] == 1
    assert window._guided_workflow_mode == "open_results"
    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepCorrectionApproach"
    assert window._guided_mode_banner_label.text().startswith("Mode: Open Results")
    assert "Raw input setup unchanged" in window._guided_mode_banner_label.text()
    assert window._guided_input_dir_edit.text() == str(raw_input)
    assert window._input_dir.text() == str(raw_input)


def test_guided_start_open_results_populates_diagnostics_without_overloading_input(
    window, tmp_path, monkeypatch
):
    run_dir = _make_preview_completed_run(tmp_path)
    raw_input = tmp_path / "raw_input"
    raw_input.mkdir()
    window._guided_input_dir_edit.setText(str(raw_input))
    calls = {"preview": 0, "signal": 0}
    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_args: str(run_dir))
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )

    window._guided_start_open_results_btn.click()

    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepCorrectionApproach"
    assert str(run_dir) in window._guided_diagnostics_completed_run_label.text()
    assert [window._guided_preview_roi_combo.itemText(i) for i in range(window._guided_preview_roi_combo.count())] == [
        "CH1",
        "CH2",
    ]
    assert [window._guided_signal_f0_roi_combo.itemText(i) for i in range(window._guided_signal_f0_roi_combo.count())] == [
        "CH1",
        "CH2",
    ]
    assert window._guided_preview_generate_btn.isEnabled() is True
    assert window._guided_signal_f0_generate_btn.isEnabled() is True
    assert calls == {"preview": 0, "signal": 0}
    assert window._guided_workflow_mode == "open_results"
    assert window._guided_mode_banner_label.text().startswith("Mode: Open Results")
    assert window._guided_input_dir_edit.text() == str(raw_input)
    assert window._input_dir.text() == str(raw_input)
    assert window._input_dir.text() != str(run_dir)
    assert str(run_dir) in window._guided_start_status_label.text()
    assert str(raw_input) in window._guided_start_status_label.text()


def test_guided_open_results_mode_marks_setup_steps_skipped_and_can_switch_back(
    window, tmp_path, monkeypatch
):
    run_dir = _make_preview_completed_run(tmp_path)
    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_args: str(run_dir))

    window._guided_start_open_results_btn.click()

    for step_name, object_name in [
        ("Select data", "guidedStepSelectData"),
        ("Recording structure", "guidedStepRecordingStructure"),
        ("Correction approach", "guidedStepCorrectionApproach"),
    ]:
        idx = list(GUIDED_WORKFLOW_STEPS).index(step_name)
        window._guided_workflow_stepper.setCurrentRow(idx)
        assert window._guided_workflow_stack.currentWidget().objectName() == object_name
        banner = window._guided_skipped_setup_banners[step_name]
        assert banner.isHidden() is False
        if step_name in window._guided_raw_setup_controls:
            assert window._guided_raw_setup_controls[step_name].isHidden() is True
        text = " ".join(label.text() for label in banner.findChildren(QLabel))
        assert "reviewing a completed run" in text
        assert "raw/input data" in text
        assert "do not configure the loaded completed run" in text
    assert window._guided_method_configuration_group.isHidden() is True

    switch_btn = window._guided_skipped_setup_banners["Select data"].findChild(
        QPushButton,
        "guidedSwitchToNewAnalysisSelectdata",
    )
    assert switch_btn is not None
    switch_btn.click()

    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepSelectData"
    assert all(not banner.isVisible() for banner in window._guided_skipped_setup_banners.values())
    assert all(not controls.isHidden() for controls in window._guided_raw_setup_controls.values())


def test_guided_confirm_strategy_is_real_planning_ui_and_run_stays_skipped_in_open_results_mode(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_args: str(run_dir))
    window._guided_start_open_results_btn.click()

    idx = list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepCorrectionApproach"
    assert window._guided_workflow_tab.findChild(QGroupBox, "guidedConfirmStrategyOpenResultsSkipped") is None
    assert [window._guided_confirm_roi_combo.itemText(i) for i in range(window._guided_confirm_roi_combo.count())] == [
        "CH1",
        "CH2",
    ]
    assert [window._guided_confirm_chunk_combo.itemData(i) for i in range(window._guided_confirm_chunk_combo.count())] == [
        0,
        1,
    ]
    assert window._guided_confirm_chunk_combo.isHidden() is False
    assert window._guided_confirm_chunk_label.isHidden() is False
    strategy_values = {
        window._guided_confirm_strategy_combo.itemData(i)
        for i in range(window._guided_confirm_strategy_combo.count())
    }
    assert {
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
        "global_linear_regression",
        "signal_only_f0",
    } <= strategy_values
    assert "auto" not in strategy_values
    assert "needs_review" not in strategy_values
    assert "no_correction" not in strategy_values
    assert window._guided_confirm_strategy_combo.currentData() == ""
    assert window._guided_confirm_mark_btn.isEnabled() is False

    run_panel = window._guided_workflow_tab.findChild(QGroupBox, "guidedRunOpenResultsSkipped")
    idx = list(GUIDED_WORKFLOW_STEPS).index("Run")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert run_panel is not None
    assert run_panel.isHidden() is False
    text = " ".join(label.text() for label in run_panel.findChildren(QLabel))
    assert "Open Results mode" in run_panel.title()
    assert "does not validate" in text
    button_texts = {button.text() for button in run_panel.findChildren(QPushButton)}
    assert {
        "Go to correction approach",
        "Switch to new analysis setup",
    } <= button_texts
    run_panel.findChild(QPushButton, "guidedRunOpenResultsSkippedGoToDiagnostics").click()
    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepCorrectionApproach"

    window._guided_workflow_stepper.setCurrentRow(idx)
    run_panel.findChild(QPushButton, "guidedRunOpenResultsSkippedSwitchToNewAnalysis").click()
    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepSelectData"


def test_guided_confirm_strategy_requires_completed_run_and_does_not_generate(window, monkeypatch):
    calls = {"preview": 0, "signal": 0}
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._set_guided_workflow_mode("start")

    assert window._guided_confirm_roi_combo.isEnabled() is False
    assert window._guided_confirm_chunk_combo.isEnabled() is False
    assert window._guided_confirm_strategy_combo.isEnabled() is False
    assert window._guided_confirm_mark_btn.isEnabled() is False
    assert "Open Results must be used first" in window._guided_confirm_context_label.text()
    assert "Correction preview: not generated" in window._guided_confirm_evidence_label.text()
    assert "Signal-Only F0 diagnostic: not generated" in window._guided_confirm_evidence_label.text()
    assert calls == {"preview": 0, "signal": 0}


def test_guided_confirm_strategy_never_auto_selects_from_loaded_or_generated_evidence(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    raw_input = tmp_path / "raw_input"
    raw_input.mkdir()
    window._guided_input_dir_edit.setText(str(raw_input))
    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_args: str(run_dir))
    window._guided_start_open_results_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    assert window._guided_confirm_strategy_combo.currentData() == ""

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_preview_generate_btn.click()
    assert window._guided_confirm_strategy_combo.currentData() == ""
    window._guided_signal_f0_generate_btn.click()
    assert window._guided_confirm_strategy_combo.currentData() == ""
    assert "Signal-Only F0 diagnostic: success" in window._guided_confirm_evidence_label.text()
    assert "recommend" not in window._guided_confirm_evidence_label.text().lower()
    assert window._guided_input_dir_edit.text() == str(raw_input)


def test_guided_confirm_strategy_explicit_mark_is_ui_state_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    calls = {"preview": 0, "signal": 0}
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )
    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_args: str(run_dir))
    window._guided_start_open_results_btn.click()
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_confirm_mark_btn.isEnabled() is False
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    assert idx >= 0
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    assert window._guided_confirm_mark_btn.isEnabled() is True
    assert window._guided_strategy_choices == {}

    window._guided_confirm_mark_btn.click()

    key = (str(run_dir.resolve()), "CH1")
    assert key in window._guided_strategy_choices
    entry = window._guided_strategy_choices[key]
    assert entry["strategy"] == "signal_only_f0"
    assert entry["strategy_label"] == "Signal-Only F0"
    assert entry["confirmed"] is True
    assert entry["completed_run_dir"] == str(run_dir.resolve())
    assert entry["roi"] == "CH1"
    assert entry["evidence_chunk"] == 0
    assert "not generated" in entry["evidence_summary"]["preview"]
    assert "not generated" in entry["evidence_summary"]["signal_only_f0"]
    assert "ROI: CH1" in window._guided_confirm_marked_choice_label.text()
    assert "Evidence reviewed: chunk 0" in window._guided_confirm_marked_choice_label.text()
    assert "marked for later planning only" in window._guided_confirm_marked_choice_label.text()
    assert "no manifest written" in window._guided_confirm_marked_choice_label.text()
    after = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after == before
    assert calls == {"preview": 0, "signal": 0}
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()


def _guided_feature_profile_config() -> dict:
    return {
        "event_signal": "dff",
        "signal_excursion_polarity": "positive",
        "peak_threshold_method": "mean_std",
        "peak_threshold_k": 2.5,
        "peak_min_distance_sec": 1.0,
        "peak_min_prominence_k": 2.0,
        "peak_min_width_sec": 0.3,
        "peak_pre_filter": "none",
        "event_auc_baseline": "zero",
    }


def _guided_plan_with_feature_profile(run_dir, *, scope="run", chunk_id=0) -> GuidedRunPlan:
    return GuidedRunPlan(
        mode="completed_run_planning",
        source=GuidedPlanSource(source_mode="completed_run", completed_run_dir=str(run_dir.resolve())),
        roi_plan=[],
        feature_event_profiles=[
            FeatureEventProfile(
                profile_id="default-events",
                profile_label="Default event profile",
                scope=scope,
                status="complete",
                config_fields=_guided_feature_profile_config(),
                evidence_previews=[EvidenceChunkReview(chunk_id=chunk_id)],
            )
        ],
    )


def test_guided_draft_run_plan_preview_appears_only_from_marked_roi_choices(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert "Status: no marked ROI choices" in window._guided_draft_run_plan_preview_label.text()
    assert "Planned ROIs: 0" in window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profiles: none configured" in window._guided_draft_run_plan_preview_label.text()
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "Source: pass" in checklist
    assert "ROI choices: not_configured" in checklist
    assert "Feature/event settings: not_configured" in checklist
    assert "Execution readiness: blocked" in checklist
    assert "Execution ready: false" in checklist

    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    text = window._guided_draft_run_plan_preview_label.text()
    assert "Status: draft valid" in text
    assert "Planned ROIs: 1" in text
    assert "- CH1: Signal-Only F0 | evidence reviewed chunk 0" in text
    assert "Preview only. This plan is in memory only" in text
    assert "cannot run, write manifests, create applied-dF/F outputs, or extract features" in text
    assert str(run_dir.resolve()) in window._guided_draft_run_plan_preview_label.toolTip()
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "ROI choices: pass" in checklist
    assert "Evidence provenance: pass" in checklist
    assert "Feature/event settings: not_configured" in checklist
    assert "Output destination: not_configured" in checklist
    assert "Execution readiness: blocked" in checklist


def test_guided_feature_event_profile_editor_creates_no_profile_by_default(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    plan, errors = window._build_guided_draft_run_plan()

    assert errors == []
    assert plan.feature_event_profiles == []
    assert "Feature/event profiles: none configured" in window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event settings: not_configured" in window._guided_draft_run_plan_checklist_label.text()
    assert window._guided_feature_event_status_label.text() == (
        "Default settings have not been confirmed."
    )


def test_guided_new_analysis_feature_event_profile_requires_apply_without_open_results(
    window,
):
    from gui.main_window import (
        GUIDED_DEFAULT_SETTINGS_NOT_CONFIRMED,
        GUIDED_DEFAULT_SETTINGS_READY,
    )

    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )

    assert window._current_run_dir == ""
    assert window._guided_feature_event_status_label.text() == (
        GUIDED_DEFAULT_SETTINGS_NOT_CONFIRMED
    )
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.feature_event_profile_status == "default_initialized"
    assert plan.feature_event_explicitly_applied is False

    window._guided_feature_event_apply_btn.click()

    assert window._guided_feature_event_status_label.text() == (
        GUIDED_DEFAULT_SETTINGS_READY
    )
    assert "Open Results must be used first" not in (
        window._guided_feature_event_status_label.text()
    )
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.feature_event_profile_status == "applied"
    assert plan.feature_event_explicitly_applied is True
    assert plan.feature_event_profile_id
    assert plan.feature_event_values["event_signal"]
    visible = window._guided_feature_event_status_label.text().lower()
    prohibited = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
        "guided_candidate_manifest",
        "guided_startup",
        "wrapper_claim",
        "backend adapter",
        "orchestration",
        "subprocess",
        "raw command",
    )
    assert not any(term in visible for term in prohibited)


def test_guided_feature_detection_step_gates_review_and_writes_nothing(
    window, tmp_path
):
    window._set_guided_workflow_mode("new_analysis")
    feature_index = list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    draft_index = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    window._reach_guided_step("Feature detection")
    window._guided_workflow_stepper.setCurrentRow(feature_index)
    feature_widget = window._guided_workflow_stack.widget(feature_index)

    intro = feature_widget.findChild(
        QWidget, "guidedFeatureDetectionStepExplanation"
    )
    # Scientist-facing Step 5 intro: names Default and Custom, no "outputs"
    # jargon (4J16k38).
    assert "Default settings" in intro.text()
    assert "Custom settings" in intro.text()
    assert feature_widget.findChild(
        QWidget, "guidedFeatureEventProfileEditorPanel"
    ) is not None
    assert window._guided_feature_detection_continue_btn.isEnabled() is False
    assert window._guided_feature_detection_continue_status.text() == (
        "Confirm the Default settings before continuing."
    )
    assert list(tmp_path.iterdir()) == []

    window._guided_feature_event_apply_btn.click()
    assert window._guided_feature_detection_continue_btn.isEnabled() is True
    assert "Status: Confirmed" in (
        window._guided_feature_detection_summary_label.text()
    )
    assert list(tmp_path.iterdir()) == []

    window._guided_feature_detection_continue_btn.click()
    assert window._guided_workflow_stepper.currentRow() == draft_index
    assert list(tmp_path.iterdir()) == []

    window._guided_feature_event_peak_k_edit.setFocus()
    window._guided_feature_event_peak_k_edit.selectAll()
    QTest.keyClicks(window._guided_feature_event_peak_k_edit, "4")
    assert window._guided_feature_detection_continue_btn.isEnabled() is False
    assert window._guided_feature_detection_continue_status.text() == (
        "Default settings changed. Confirm them again before continuing."
    )

    window._guided_feature_event_peak_k_edit.selectAll()
    QTest.keyClicks(window._guided_feature_event_peak_k_edit, "0")
    window._guided_feature_event_apply_btn.click()
    assert window._guided_feature_detection_continue_btn.isEnabled() is False
    assert window._guided_feature_detection_continue_status.text().startswith(
        "Default settings need attention before continuing."
    )


def test_guided_new_analysis_feature_event_profile_invalid_status_is_actionable(
    window,
):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_feature_event_peak_method_combo.setCurrentText("mean_std")
    window._guided_feature_event_peak_k_edit.setText("0")

    window._guided_feature_event_apply_btn.click()

    status_text = window._guided_feature_event_status_label.text()
    assert status_text.startswith(
        "Default settings need attention before continuing."
    )
    # The invalid state must stay actionable: the field-level validation detail
    # is still shown (only the raw loader diagnostic is sanitized away).
    assert "peak_threshold_k" in status_text.lower() or "k must be" in status_text.lower()
    assert (
        window._build_guided_new_analysis_draft_plan().feature_event_profile_status
        == "invalid"
    )


def test_guided_new_analysis_configured_feature_event_profile_materializes(
    window, tmp_path
):
    from photometry_pipeline.guided_backend_validation_materialization import (
        GuidedBackendValidationMaterializationSuccess,
        materialize_guided_backend_validation_facts,
    )
    from tests.test_guided_backend_validation_materialization import (
        _valid_parser_contract,
        _valid_stage2c_draft,
    )

    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_feature_event_apply_btn.click()
    gui_plan = window._build_guided_new_analysis_draft_plan()

    backend_root = tmp_path / "configured_profile_backend"
    backend_root.mkdir()
    backend_draft = _valid_stage2c_draft(backend_root)
    backend_draft.feature_event_profile_status = (
        gui_plan.feature_event_profile_status
    )
    backend_draft.feature_event_profile_id = gui_plan.feature_event_profile_id
    backend_draft.feature_event_values = dict(gui_plan.feature_event_values)
    backend_draft.feature_event_validation_issues = list(
        gui_plan.feature_event_validation_issues
    )
    backend_draft.feature_event_stale_reasons = list(
        gui_plan.feature_event_stale_reasons
    )
    backend_draft.feature_event_updated_at_utc = (
        gui_plan.feature_event_updated_at_utc
    )
    backend_draft.feature_event_explicitly_applied = (
        gui_plan.feature_event_explicitly_applied
    )

    result = materialize_guided_backend_validation_facts(
        backend_draft,
        parser_contract=_valid_parser_contract(),
    )
    assert isinstance(result, GuidedBackendValidationMaterializationSuccess)


def test_guided_new_analysis_cleared_feature_event_profile_still_blocks_validation(
    window, tmp_path
):
    from photometry_pipeline.guided_backend_validation_materialization import (
        GuidedBackendValidationMaterializationFailure,
        materialize_guided_backend_validation_facts,
    )
    from tests.test_guided_backend_validation_materialization import (
        _valid_parser_contract,
        _valid_stage2c_draft,
    )

    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_feature_event_apply_btn.click()
    window._guided_feature_event_clear_btn.click()
    gui_plan = window._build_guided_new_analysis_draft_plan()
    assert gui_plan.feature_event_profile_status == "default_initialized"
    assert gui_plan.feature_event_explicitly_applied is False

    backend_root = tmp_path / "cleared_profile_backend"
    backend_root.mkdir()
    backend_draft = _valid_stage2c_draft(backend_root)
    backend_draft.feature_event_profile_status = (
        gui_plan.feature_event_profile_status
    )
    backend_draft.feature_event_profile_id = gui_plan.feature_event_profile_id
    backend_draft.feature_event_values = dict(gui_plan.feature_event_values)
    backend_draft.feature_event_validation_issues = list(
        gui_plan.feature_event_validation_issues
    )
    backend_draft.feature_event_stale_reasons = list(
        gui_plan.feature_event_stale_reasons
    )
    backend_draft.feature_event_explicitly_applied = (
        gui_plan.feature_event_explicitly_applied
    )

    result = materialize_guided_backend_validation_facts(
        backend_draft,
        parser_contract=_valid_parser_contract(),
    )
    assert isinstance(result, GuidedBackendValidationMaterializationFailure)
    assert result.blocking_issues[0].category == (
        "feature_event_unapplied_changes"
    )


def test_guided_feature_event_profile_apply_valid_run_level_profile(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._guided_feature_event_polarity_combo.setCurrentText("both")
    window._guided_feature_event_peak_method_combo.setCurrentText("percentile")
    window._guided_feature_event_peak_pct_edit.setText("90.0")
    window._guided_feature_event_pre_filter_combo.setCurrentText("lowpass")
    window._guided_feature_event_auc_baseline_combo.setCurrentText("median")
    window._guided_feature_event_apply_btn.click()

    plan, errors = window._build_guided_draft_run_plan()
    assert errors == []
    assert len(plan.feature_event_profiles) == 1
    profile = plan.feature_event_profiles[0]
    assert profile.profile_id == "default-events"
    assert profile.profile_label == "Default feature/event profile"
    assert profile.scope == "run"
    assert profile.status == "complete"
    assert profile.choice_source == "explicit_user_profile_edit"
    assert profile.evidence_previews == []
    assert profile.target_rois == []
    assert profile.resolved_rois == []
    assert profile.config_fields["event_signal"] == "delta_f"
    assert profile.config_fields["signal_excursion_polarity"] == "both"
    assert profile.config_fields["peak_threshold_method"] == "percentile"
    assert profile.config_fields["peak_threshold_percentile"] == 90.0
    assert profile.config_fields["peak_pre_filter"] == "lowpass"
    assert profile.config_fields["event_auc_baseline"] == "median"
    preview = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile default-events (Default feature/event profile)" in preview
    assert "scope=run" in preview
    assert "config_fields=9" in preview
    assert "evidence preview chunks=none" in preview
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "Feature/event settings: pass" in checklist
    assert "Execution readiness: blocked" in checklist
    assert "Execution ready: false" in checklist


def test_full_control_report_viewer_unaffected_by_guided_review_viewer(
    window, tmp_path
):
    """4J16k20: Full Control's own completed-run loading path
    (_open_completed_results_dir -> self._report_viewer) must keep working
    exactly as before the Guided-native review viewer was added, and the
    two RunReportViewer instances must not share mutable state -- loading
    a run through Full Control's own entry point must not populate the
    separate Guided Review viewer."""
    run_dir = _make_preview_completed_run(tmp_path / "full_control_run")

    loaded = window._open_completed_results_dir(str(run_dir))

    assert loaded is True
    assert window._report_viewer._region_paths
    assert set(window._report_viewer._region_paths) == {"CH1"}
    assert window._report_viewer._region_combo.count() == 1
    # The Guided Review viewer is a separate instance and must remain in
    # its untouched idle state -- Full Control loading a run must not
    # leak into it.
    assert window._guided_report_viewer._region_paths == {}
    assert window._guided_report_viewer._current_run_dir == ""


def test_guided_feature_event_profiles_are_scoped_to_completed_run(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "feature_run_a_parent")
    run_b = _make_preview_completed_run(tmp_path / "feature_run_b_parent")
    defaults = Config()

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._guided_feature_event_apply_btn.click()

    plan_a, errors_a = window._build_guided_draft_run_plan()
    assert errors_a == []
    assert len(plan_a.feature_event_profiles) == 1
    run_a_config = dict(plan_a.feature_event_profiles[0].config_fields)
    assert run_a_config["event_signal"] == "delta_f"
    assert "Feature/event profile default-events" in window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event settings: pass" in window._guided_draft_run_plan_checklist_label.text()

    window._open_completed_results_dir(str(run_b))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()

    plan_b, errors_b = window._build_guided_draft_run_plan()
    assert errors_b == []
    assert plan_b.feature_event_profiles == []
    assert window._guided_feature_event_signal_combo.currentText() == defaults.event_signal
    assert "Feature/event profiles: none configured" in window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event settings: not_configured" in window._guided_draft_run_plan_checklist_label.text()

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()

    restored_plan, restored_errors = window._build_guided_draft_run_plan()
    assert restored_errors == []
    assert len(restored_plan.feature_event_profiles) == 1
    assert restored_plan.feature_event_profiles[0].config_fields == run_a_config
    assert window._guided_feature_event_signal_combo.currentText() == "delta_f"
    assert "Feature/event profile default-events" in window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event settings: pass" in window._guided_draft_run_plan_checklist_label.text()


def test_guided_feature_event_unsaved_same_run_edits_survive_refresh_without_plan_mutation(
    window,
    tmp_path,
    monkeypatch,
):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._refresh_guided_confirm_strategy_panel()

    plan, errors = window._build_guided_draft_run_plan()
    assert errors == []
    assert plan.feature_event_profiles == []
    assert window._guided_feature_event_signal_combo.currentText() == "delta_f"
    assert "Feature/event settings: not_configured" in window._guided_draft_run_plan_checklist_label.text()


def test_guided_feature_event_unsaved_edits_do_not_leak_across_completed_runs(
    window,
    tmp_path,
    monkeypatch,
):
    run_a = _make_preview_completed_run(tmp_path / "unsaved_run_a_parent")
    run_b = _make_preview_completed_run(tmp_path / "unsaved_run_b_parent")
    defaults = Config()

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._refresh_guided_confirm_strategy_panel()
    assert window._guided_feature_event_signal_combo.currentText() == "delta_f"

    window._open_completed_results_dir(str(run_b))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()
    plan_b, errors_b = window._build_guided_draft_run_plan()
    assert errors_b == []
    assert plan_b.feature_event_profiles == []
    assert window._guided_feature_event_signal_combo.currentText() == defaults.event_signal

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()
    plan_a, errors_a = window._build_guided_draft_run_plan()
    assert errors_a == []
    assert plan_a.feature_event_profiles == []
    assert window._guided_feature_event_signal_combo.currentText() == defaults.event_signal


def test_guided_feature_event_profile_controls_do_not_live_bind_before_apply(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._guided_feature_event_apply_btn.click()
    plan, _ = window._build_guided_draft_run_plan()
    assert plan.feature_event_profiles[0].config_fields["event_signal"] == "dff"

    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    plan, _ = window._build_guided_draft_run_plan()
    assert plan.feature_event_profiles[0].config_fields["event_signal"] == "dff"

    window._guided_feature_event_apply_btn.click()
    plan, _ = window._build_guided_draft_run_plan()
    assert plan.feature_event_profiles[0].config_fields["event_signal"] == "delta_f"


def test_guided_feature_event_profile_invalid_values_are_rejected_without_updating_plan(
    window,
    tmp_path,
    monkeypatch,
):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_feature_event_peak_method_combo.setCurrentText("mean_std")
    window._guided_feature_event_peak_k_edit.setText("0")
    window._guided_feature_event_apply_btn.click()

    plan, errors = window._build_guided_draft_run_plan()
    assert errors == []
    assert plan.feature_event_profiles == []
    assert "Feature/event profile not applied: Peak Threshold K must be > 0." in (
        window._guided_feature_event_status_label.text()
    )
    assert "Feature/event settings: not_configured" in window._guided_draft_run_plan_checklist_label.text()

    window._guided_feature_event_peak_k_edit.setText("2.5")
    window._guided_feature_event_apply_btn.click()
    plan, _ = window._build_guided_draft_run_plan()
    assert len(plan.feature_event_profiles) == 1
    previous = dict(plan.feature_event_profiles[0].config_fields)

    window._guided_feature_event_peak_method_combo.setCurrentText("absolute")
    window._guided_feature_event_peak_abs_edit.setText("0")
    window._guided_feature_event_apply_btn.click()
    plan, _ = window._build_guided_draft_run_plan()
    assert plan.feature_event_profiles[0].config_fields == previous
    assert "Peak Threshold Absolute must be > 0." in window._guided_feature_event_status_label.text()
    assert window._guided_feature_event_peak_method_combo.currentText() == "absolute"
    assert window._guided_feature_event_peak_abs_edit.text() == "0"
    assert "Feature/event settings: pass" in window._guided_draft_run_plan_checklist_label.text()


def test_guided_feature_event_profile_is_not_scoped_or_changed_by_roi_chunk_selection(
    window,
    tmp_path,
    monkeypatch,
):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_feature_event_apply_btn.click()
    before_plan, _ = window._build_guided_draft_run_plan()
    before_config = dict(before_plan.feature_event_profiles[0].config_fields)

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData("CH2"))
    window._guided_confirm_chunk_combo.setCurrentIndex(window._guided_confirm_chunk_combo.findData(1))

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    after_plan, _ = window._build_guided_draft_run_plan()
    profile = after_plan.feature_event_profiles[0]

    assert profile.scope == "run"
    assert profile.evidence_previews == []
    assert profile.target_rois == []
    assert profile.resolved_rois == []
    assert profile.config_fields == before_config


def test_guided_diagnostics_do_not_create_or_mutate_feature_event_profile(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_preview_generate_btn.click()
    window._guided_signal_f0_generate_btn.click()
    plan, _ = window._build_guided_draft_run_plan()
    assert plan.feature_event_profiles == []

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_feature_event_apply_btn.click()
    before_plan, _ = window._build_guided_draft_run_plan()
    before_config = dict(before_plan.feature_event_profiles[0].config_fields)

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_preview_generate_btn.click()
    window._guided_signal_f0_generate_btn.click()
    after_plan, _ = window._build_guided_draft_run_plan()

    assert after_plan.feature_event_profiles[0].config_fields == before_config
    assert len(after_plan.feature_event_profiles) == 1


def test_guided_feature_event_profile_clear_removes_only_in_memory_profile(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path / "clear_run_a_parent")
    run_b = _make_preview_completed_run(tmp_path / "clear_run_b_parent")
    before = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    before_b = sorted(p.relative_to(run_b).as_posix() for p in run_b.rglob("*"))
    window._open_completed_results_dir(str(run_dir))
    window._set_guided_workflow_mode("open_results")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_feature_event_apply_btn.click()
    assert window._build_guided_draft_run_plan()[0].feature_event_profiles
    run_a_config = dict(window._build_guided_draft_run_plan()[0].feature_event_profiles[0].config_fields)

    window._open_completed_results_dir(str(run_b))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._guided_feature_event_apply_btn.click()
    assert window._build_guided_draft_run_plan()[0].feature_event_profiles[0].config_fields["event_signal"] == "delta_f"
    window._guided_feature_event_clear_btn.click()

    plan, errors = window._build_guided_draft_run_plan()
    assert errors == []
    assert plan.feature_event_profiles == []
    assert window._guided_feature_event_signal_combo.currentText() == Config().event_signal
    assert "Feature/event profiles: none configured" in window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event settings: not_configured" in window._guided_draft_run_plan_checklist_label.text()
    window._open_completed_results_dir(str(run_dir))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()
    plan_a, errors_a = window._build_guided_draft_run_plan()
    assert errors_a == []
    assert len(plan_a.feature_event_profiles) == 1
    assert plan_a.feature_event_profiles[0].config_fields == run_a_config
    assert window._guided_feature_event_signal_combo.currentText() == run_a_config["event_signal"]
    after = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    after_b = sorted(p.relative_to(run_b).as_posix() for p in run_b.rglob("*"))
    assert after == before
    assert after_b == before_b
    assert not list(run_dir.rglob("guided_run_plan*.json"))
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not list(run_b.rglob("guided_run_plan*.json"))
    assert not (run_b / "features.csv").exists()


def test_guided_draft_run_plan_preview_displays_injected_feature_event_profile(
    window,
    tmp_path,
    monkeypatch,
):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan = _guided_plan_with_feature_profile(run_dir, chunk_id=1)
    monkeypatch.setattr(window, "_build_guided_draft_run_plan", lambda: (plan, []))

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_confirm_strategy_panel()

    preview = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile default-events (Default event profile)" in preview
    assert "scope=run" in preview
    assert "status=complete" in preview
    assert "config_fields=9" in preview
    assert "event_signal" in preview
    assert "evidence preview chunks=1" in preview
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "Feature/event settings: pass" in checklist
    assert "Execution readiness: blocked" in checklist
    assert "Execution ready: false" in checklist


def test_guided_draft_run_plan_preview_displays_invalid_feature_event_profile_errors(
    window,
    tmp_path,
    monkeypatch,
):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan = _guided_plan_with_feature_profile(run_dir, scope="chunk")
    monkeypatch.setattr(window, "_build_guided_draft_run_plan", lambda: (plan, []))

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_confirm_strategy_panel()

    preview = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile default-events" in preview
    assert "scope=chunk" in preview
    assert "Status: draft has errors" in preview
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "Contract: fail" in checklist
    assert "Feature/event settings: fail" in checklist
    assert "Execution readiness: blocked" in checklist


def test_guided_feature_event_profile_display_is_not_mutated_by_visible_selection(
    window,
    tmp_path,
    monkeypatch,
):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan = _guided_plan_with_feature_profile(run_dir, chunk_id=0)
    monkeypatch.setattr(window, "_build_guided_draft_run_plan", lambda: (plan, []))

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_confirm_strategy_panel()
    before = serialize_feature_preview_text = window._guided_draft_run_plan_preview_label.text()

    # Navigate to Confirm strategy to interact with confirm widgets
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_confirm_roi_combo.setCurrentText("CH2")
    window._guided_confirm_chunk_combo.setCurrentIndex(window._guided_confirm_chunk_combo.findData(1))
    window._refresh_guided_confirm_strategy_panel()

    # Navigate back to Draft plan to check output
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    after = window._guided_draft_run_plan_preview_label.text()
    assert "scope=run" in after
    assert "evidence preview chunks=0" in after
    assert "evidence preview chunks=1" not in after
    assert "config_fields=9" in after
    assert len(plan.feature_event_profiles) == 1
    assert plan.feature_event_profiles[0].scope == "run"
    assert plan.feature_event_profiles[0].evidence_previews[0].chunk_id == 0
    assert before == after


def test_guided_feature_event_profile_display_writes_no_outputs(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan = _guided_plan_with_feature_profile(run_dir)
    monkeypatch.setattr(window, "_build_guided_draft_run_plan", lambda: (plan, []))

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_confirm_strategy_panel()

    after = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after == before
    assert not list(run_dir.rglob("guided_run_plan*.json"))
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "validation").exists()


def test_guided_confirm_strategy_evidence_marks_stale_for_selection_change(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_preview_generate_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    assert "Correction preview: success" in window._guided_confirm_evidence_label.text()
    window._guided_confirm_chunk_combo.setCurrentIndex(window._guided_confirm_chunk_combo.findData(1))
    assert "Correction preview: success stale" in window._guided_confirm_evidence_label.text()
    assert "Displayed evidence is stale for the current selection" in window._guided_confirm_evidence_label.text()


def test_guided_confirm_acknowledgment_resets_when_chunk_changes(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    assert window._guided_confirm_mark_btn.isEnabled() is True

    window._guided_confirm_chunk_combo.setCurrentIndex(window._guided_confirm_chunk_combo.findData(1))

    assert window._guided_confirm_ack_cb.isChecked() is False
    assert window._guided_confirm_strategy_combo.currentData() == "signal_only_f0"
    assert window._guided_confirm_mark_btn.isEnabled() is True


def test_guided_confirm_choice_is_roi_level_and_evidence_chunk_can_update(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    key = (str(run_dir.resolve()), "CH1")
    assert key in window._guided_strategy_choices
    assert window._guided_strategy_choices[key]["evidence_chunk"] == 0
    assert "Evidence reviewed: chunk 0" in window._guided_confirm_marked_choice_label.text()
    assert "- CH1: Signal-Only F0 | evidence reviewed chunk 0" in (
        window._guided_draft_run_plan_preview_label.text()
    )

    window._guided_confirm_chunk_combo.setCurrentIndex(window._guided_confirm_chunk_combo.findData(1))

    assert "Signal-Only F0" in window._guided_confirm_marked_choice_label.text()
    assert "Evidence reviewed: chunk 0" in window._guided_confirm_marked_choice_label.text()
    assert "- CH1: Signal-Only F0 | evidence reviewed chunk 0" in (
        window._guided_draft_run_plan_preview_label.text()
    )
    assert "chunk 1" not in window._guided_draft_run_plan_preview_label.text()
    assert window._guided_confirm_ack_cb.isChecked() is False
    assert window._guided_confirm_mark_btn.isEnabled() is True

    window._guided_confirm_ack_cb.setChecked(True)
    assert window._guided_confirm_mark_btn.isEnabled() is True
    window._guided_confirm_mark_btn.click()

    assert key in window._guided_strategy_choices
    assert window._guided_strategy_choices[key]["evidence_chunk"] == 1
    assert "Evidence reviewed: chunk 1" in window._guided_confirm_marked_choice_label.text()
    plan_text = window._guided_draft_run_plan_preview_label.text()
    assert plan_text.count("- CH1:") == 1
    assert "- CH1: Signal-Only F0 | evidence reviewed chunk 1" in plan_text


def test_guided_confirm_choices_are_independent_by_roi(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()
    assert (str(run_dir.resolve()), "CH1") in window._guided_strategy_choices

    window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData("CH2"))

    assert "Current marked choice: none." in window._guided_confirm_marked_choice_label.text()
    assert window._guided_confirm_ack_cb.isChecked() is False
    assert window._guided_confirm_mark_btn.isEnabled() is True

    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    assert (str(run_dir.resolve()), "CH1") in window._guided_strategy_choices
    assert (str(run_dir.resolve()), "CH2") in window._guided_strategy_choices
    assert "ROI: CH2" in window._guided_confirm_marked_choice_label.text()
    plan_text = window._guided_draft_run_plan_preview_label.text()
    assert "Planned ROIs: 2" in plan_text
    assert plan_text.count("- CH1:") == 1
    assert plan_text.count("- CH2:") == 1


def test_guided_confirm_strategy_choices_are_scoped_to_loaded_completed_run(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a_parent")
    run_b = _make_preview_completed_run(tmp_path / "run_b_parent")

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()
    assert (str(run_a.resolve()), "CH1") in window._guided_strategy_choices
    assert "Signal-Only F0" in window._guided_confirm_marked_choice_label.text()

    window._open_completed_results_dir(str(run_b))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()

    assert window._guided_confirm_roi_combo.currentText() == "CH1"
    assert window._guided_confirm_chunk_combo.currentData() == 0
    assert "Current marked choice: none." in window._guided_confirm_marked_choice_label.text()
    assert "Status: no marked ROI choices" in window._guided_draft_run_plan_preview_label.text()
    assert (str(run_a.resolve()), "CH1") in window._guided_strategy_choices

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()
    assert window._guided_confirm_chunk_combo.currentData() == 0
    assert "Signal-Only F0" in window._guided_confirm_marked_choice_label.text()
    assert "Evidence reviewed: chunk 0" in window._guided_confirm_marked_choice_label.text()
    assert "- CH1: Signal-Only F0 | evidence reviewed chunk 0" in (
        window._guided_draft_run_plan_preview_label.text()
    )


def test_guided_diagnostics_do_not_auto_populate_draft_run_plan_preview(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)

    window._guided_preview_generate_btn.click()
    window._guided_signal_f0_generate_btn.click()

    # Navigate to Confirm strategy to verify evidence labels
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    assert "Correction preview: success" in window._guided_confirm_evidence_label.text()
    assert "Signal-Only F0 diagnostic: success" in window._guided_confirm_evidence_label.text()

    # Navigate to Draft plan to verify draft plan labels
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    assert "Status: no marked ROI choices" in window._guided_draft_run_plan_preview_label.text()
    assert "Planned ROIs: 0" in window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profiles: none configured" in window._guided_draft_run_plan_preview_label.text()
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "ROI choices: not_configured" in checklist
    assert "Feature/event settings: not_configured" in checklist
    assert "Execution readiness: blocked" in checklist
    assert window._guided_strategy_choices == {}


def test_guided_draft_run_plan_preview_reports_contract_errors(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    run_key = str(run_dir.resolve())
    window._guided_strategy_choices[(run_key, "CH1")] = {
        "strategy": "auto",
        "strategy_label": "auto",
        "choice_source": "diagnostic_success",
        "no_auto_selection": False,
        "confirmed": True,
        "completed_run_dir": run_key,
        "roi": "CH1",
        "evidence_chunk": 0,
        "evidence_summary": {"preview": "Correction preview: success", "signal_only_f0": "", "stale": False},
    }

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_confirm_strategy_panel()

    text = window._guided_draft_run_plan_preview_label.text()
    assert "Status: draft has errors" in text
    assert "forbidden runnable correction strategy: auto" in text
    assert "choice_source must be explicit_user_mark" in text
    assert "no_auto_selection must be true" in text


def test_guided_draft_run_plan_preview_reports_malformed_stored_evidence_chunk(
    window,
    tmp_path,
    monkeypatch,
):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    run_key = str(run_dir.resolve())
    window._guided_strategy_choices[(run_key, "CH1")] = {
        "strategy": "signal_only_f0",
        "strategy_label": "Signal-Only F0",
        "choice_source": "explicit_user_mark",
        "no_auto_selection": True,
        "confirmed": True,
        "completed_run_dir": run_key,
        "roi": "CH1",
        "evidence_chunk": "bad",
        "evidence_summary": {"preview": "Correction preview: success", "signal_only_f0": "", "stale": False},
    }

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_confirm_strategy_panel()

    text = window._guided_draft_run_plan_preview_label.text()
    assert "Status: draft has errors" in text
    assert "guided strategy choice for ROI CH1 has invalid evidence_chunk" in text
    assert "chunk_id must be an integer evidence reference" in text
    assert "- CH1: Signal-Only F0 | evidence reviewed chunk bad" in text
    assert "evidence reviewed chunk 0" not in text
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "Contract: fail" in checklist
    assert "Evidence provenance: fail" in checklist
    assert "Execution readiness: blocked" in checklist
    assert "Execution ready: false" in checklist


def test_guided_draft_run_plan_preview_writes_no_outputs(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    # Navigate to Confirm strategy to mark strategy
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    # Navigate to Draft plan
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_confirm_strategy_panel()

    after = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after == before
    assert not list(run_dir.rglob("guided_run_plan*.json"))
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    assert not (run_dir / "validation").exists()


def test_guided_confirm_acknowledgment_and_strategy_reset_when_completed_run_changes(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "ack_run_a_parent")
    run_b = _make_preview_completed_run(tmp_path / "ack_run_b_parent")

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    assert window._guided_confirm_mark_btn.isEnabled() is True

    window._open_completed_results_dir(str(run_b))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()

    assert window._guided_confirm_ack_cb.isChecked() is False
    assert window._guided_confirm_strategy_combo.currentData() == ""
    assert window._guided_confirm_mark_btn.isEnabled() is False
    assert "Current marked choice: none." in window._guided_confirm_marked_choice_label.text()


def test_guided_confirm_returning_to_marked_run_still_requires_fresh_ack(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "return_ack_run_a_parent")
    run_b = _make_preview_completed_run(tmp_path / "return_ack_run_b_parent")

    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()
    assert "Signal-Only F0" in window._guided_confirm_marked_choice_label.text()

    window._open_completed_results_dir(str(run_b))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()
    window._open_completed_results_dir(str(run_a))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()

    assert "Signal-Only F0" in window._guided_confirm_marked_choice_label.text()
    assert window._guided_confirm_ack_cb.isChecked() is False
    assert window._guided_confirm_strategy_combo.currentData() == ""
    assert window._guided_confirm_mark_btn.isEnabled() is False


def test_guided_confirm_strategy_evidence_is_scoped_to_loaded_completed_run(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "evidence_run_a_parent")
    run_b = _make_preview_completed_run(tmp_path / "evidence_run_b_parent")

    _load_preview_completed_run(window, run_a, monkeypatch)
    window._guided_preview_generate_btn.click()
    window._guided_signal_f0_generate_btn.click()
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    assert "Correction preview: success" in window._guided_confirm_evidence_label.text()
    assert "Signal-Only F0 diagnostic: success" in window._guided_confirm_evidence_label.text()

    window._open_completed_results_dir(str(run_b))
    window._set_guided_workflow_mode("open_results")
    window._refresh_guided_confirm_strategy_panel()
    text = window._guided_confirm_evidence_label.text()
    assert "Correction preview: not generated for current completed run" in text
    assert "Signal-Only F0 diagnostic: not generated for current completed run" in text
    assert "Correction preview: success" not in text
    assert "Signal-Only F0 diagnostic: success" not in text


def test_full_control_open_results_still_uses_same_completed_loader(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    calls = {"open": 0}

    def _fake_open(path):
        calls["open"] += 1
        assert path == str(run_dir)
        return True

    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_args: str(run_dir))
    monkeypatch.setattr(window, "_open_completed_results_dir", _fake_open)

    window._on_open_results()

    assert calls["open"] == 1


def test_guided_correction_step_shows_expected_non_executing_cards(window):
    cards = window._guided_correction_cards
    assert list(cards) == [
        "Robust Global Event-Reject Fit",
        "Adaptive Event-Gated Fit",
        "Global Linear Regression",
        "Signal-Only F0",
        "Decision-Support Audit",
    ]
    assert "not recommended" in " ".join(_label_texts(cards["Global Linear Regression"])).lower()
    assert cards["Decision-Support Audit"].property("guidedCorrectionCardNonExecuting") is True
    assert "read-only evidence" in " ".join(_label_texts(cards["Decision-Support Audit"])).lower()
    assert "No Correction" not in cards
    assert "Decision-Support Audit" not in window._guided_correction_select_buttons


def test_no_correction_is_not_a_normal_guided_correction_card(window):
    card_titles = [
        card.property("guidedCorrectionCardTitle")
        for card in window._guided_workflow_tab.findChildren(QGroupBox)
        if card.property("guidedCorrectionCardTitle")
    ]
    assert "No Correction" not in card_titles


def test_full_control_preserves_existing_applied_dff_controls(window):
    tabs = window._workflow_mode_tabs
    tabs.setCurrentIndex(_tab_labels(window).index("Full Control"))
    assert window._applied_dff_group.isVisible() or window._applied_dff_group is not None
    assert window._applied_dff_save_manifest_btn.text() == "Save Manifest"
    assert window._applied_dff_dry_run_btn.text() == "Dry Run"
    assert window._applied_dff_run_batch_btn.text() == "Run Batch"


def test_guided_workflow_has_no_run_validate_or_manifest_action_buttons(window):
    forbidden = {"Run Pipeline", "Validate Only", "Save Manifest", "Dry Run", "Run Batch"}
    guided_button_texts = {
        button.text()
        for button in window._guided_workflow_tab.findChildren(QPushButton)
        if button.text()
    }
    assert guided_button_texts.isdisjoint(forbidden)


def test_guided_select_data_syncs_to_full_control_state(window, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    target_format = "custom_tabular"
    idx = window._guided_format_combo.findText(target_format)
    assert idx >= 0
    window._guided_format_combo.setCurrentIndex(idx)

    assert window._input_dir.text() == str(input_dir)
    assert window._output_dir.text() == str(output_dir)
    assert window._format_combo.currentText() == target_format


def test_full_control_select_data_syncs_to_guided_display(window, tmp_path):
    input_dir = tmp_path / "full_input"
    output_dir = tmp_path / "full_output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(output_dir))
    idx = window._format_combo.findText("auto")
    assert idx >= 0
    window._format_combo.setCurrentIndex(idx)

    assert window._guided_input_dir_edit.text() == str(input_dir)
    assert window._guided_output_dir_edit.text() == str(output_dir)
    assert window._guided_format_combo.currentText() == "auto"


def test_guided_recording_structure_syncs_to_full_control_state(window):
    idx = window._guided_acquisition_mode_combo.findData("continuous")
    assert idx >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_continuous_window_sec_spin.setValue(900.0)
    window._guided_allow_partial_final_window_cb.setChecked(True)
    window._guided_exclude_incomplete_final_rwd_chunk_cb.setChecked(True)

    assert window._selected_acquisition_mode() == "continuous"
    assert float(window._continuous_window_sec_spin.value()) == 900.0
    assert float(window._continuous_step_sec_spin.value()) == 900.0
    assert window._allow_partial_final_window_cb.isChecked() is True
    assert window._exclude_incomplete_final_rwd_chunk_cb.isChecked() is True

    idx = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("300")

    assert window._selected_acquisition_mode() == "intermittent"
    assert window._sph_edit.text() == "6"
    assert window._duration_edit.text() == "300"


def _set_guided_rwd_intermittent(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("rwd")
    index = window._guided_acquisition_mode_combo.findData("intermittent")
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)


def test_guided_rwd_intermittent_duration_is_visibly_required(window):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.clear()

    assert window._guided_session_duration_label.text() == (
        "Session duration (s):"
    )
    assert "required, seconds > 0" in (
        window._guided_session_duration_edit.placeholderText()
    )
    status = window._guided_recording_structure_help_label.text()
    assert "needs attention" in status
    assert "session duration are required" in status
    assert "optional" not in status.lower()


@pytest.mark.parametrize("duration", ("", "0", "-1", "not-a-number"))
def test_guided_rwd_intermittent_invalid_duration_remains_incomplete(
    window,
    duration,
):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.setText(duration)

    assert "needs attention" in (
        window._guided_recording_structure_help_label.text()
    )
    assert "positive sessions/hour and session duration are required" in (
        window._guided_recording_structure_help_label.text()
    )


def test_guided_rwd_intermittent_valid_duration_completes_step(window):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.setText("600")

    status = window._guided_recording_structure_help_label.text()
    # Redundant restatement of already-visible field values was removed;
    # "Recording structure is ready." (shown near Continue) now covers this.
    assert status == ""
    assert "needs attention" not in status
    assert window._guided_recording_continue_status.text() == (
        "Recording structure is ready."
    )


def test_guided_recording_continue_reports_missing_and_invalid_intermittent_fields(
    window,
):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.clear()
    window._guided_session_duration_edit.clear()

    assert window._guided_recording_continue_btn.isEnabled() is False
    assert window._guided_recording_continue_status.text() == (
        "Enter sessions per hour and session duration to continue."
    )

    window._guided_sessions_per_hour_edit.setText("1.5")
    window._guided_session_duration_edit.setText("600")
    assert window._guided_recording_continue_btn.isEnabled() is False
    assert window._guided_recording_continue_status.text() == (
        "Sessions per hour must be a positive whole number."
    )

    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.setText("0")
    assert window._guided_recording_continue_btn.isEnabled() is False
    assert window._guided_recording_continue_status.text() == (
        "Session duration must be greater than 0 seconds."
    )


@pytest.mark.parametrize(
    ("sessions_per_hour", "duration", "ready"),
    [
        ("2", "600", True),
        ("6", "600", True),
        ("600", "1000", False),
    ],
)
def test_guided_recording_continue_enforces_intermittent_timing_plausibility(
    window, sessions_per_hour, duration, ready
):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.setText(sessions_per_hour)
    window._guided_session_duration_edit.setText(duration)

    assert window._guided_recording_continue_btn.isEnabled() is ready
    if ready:
        assert window._guided_recording_continue_status.text() == (
            "Recording structure is ready."
        )
    else:
        assert window._guided_recording_continue_status.text() == (
            "Session timing is impossible: sessions per hour × session "
            "duration cannot exceed 3600 seconds."
        )


def test_guided_recording_continuous_readiness_and_mode_visibility(window):
    _set_guided_rwd_intermittent(window)
    assert (
        window._guided_recording_timing_inference_label.isHidden() is False
    )
    assert window._guided_sessions_per_hour_edit.isHidden() is False
    assert window._guided_continuous_window_sec_spin.isHidden() is True
    assert window._guided_incomplete_final_rwd_group.isHidden() is False
    assert window._guided_exclude_incomplete_final_rwd_chunk_cb.text() == (
        "Allow one incomplete final recording session"
    )
    assert "Earlier incomplete sessions still stop validation" in (
        window._guided_incomplete_final_rwd_help_label.text()
    )
    assert "Raw files are not modified" in (
        window._guided_incomplete_final_rwd_help_label.text()
    )

    continuous = window._guided_acquisition_mode_combo.findData(
        "continuous"
    )
    window._guided_acquisition_mode_combo.setCurrentIndex(continuous)
    assert window._guided_sessions_per_hour_edit.isHidden() is True
    assert window._guided_session_duration_edit.isHidden() is True
    assert (
        window._guided_recording_timing_inference_label.isHidden() is True
    )
    assert window._guided_use_detected_timing_btn.isHidden() is True
    assert window._guided_continuous_window_sec_spin.isHidden() is False
    assert window._guided_incomplete_final_rwd_group.isHidden() is True
    assert window._guided_recording_continue_btn.isEnabled() is True
    assert window._guided_recording_continue_status.text() == (
        "Recording structure is ready."
    )

    blocker = QSignalBlocker(window._guided_continuous_window_sec_spin)
    window._guided_continuous_window_sec_spin.setMinimum(0)
    window._guided_continuous_window_sec_spin.setValue(0)
    del blocker
    window._refresh_guided_navigation_state()
    assert window._guided_recording_continue_btn.isEnabled() is False
    assert window._guided_recording_continue_status.text() == (
        "Continuous analysis window must be greater than 0 seconds."
    )


def test_guided_incomplete_final_rwd_option_hidden_for_non_rwd(window):
    _set_guided_rwd_intermittent(window)
    window._guided_format_combo.setCurrentText("custom_tabular")

    assert window._guided_incomplete_final_rwd_group.isHidden() is True


def _timed_discovery(input_dir, session_ids, resolved_format="RWD"):
    return {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "resolved_format": resolved_format,
        "sessions": [
            {
                "index": index,
                "session_id": session_id,
                "path": str(
                    input_dir / session_id / "fluorescence.csv"
                ),
                "included_in_preview": True,
            }
            for index, session_id in enumerate(session_ids)
        ],
        "n_total_discovered": len(session_ids),
        "n_preview": len(session_ids),
        "rois": [{"roi_id": "CH1"}],
    }


def test_guided_rwd_discovery_autofills_recording_timing(
    window, tmp_path, monkeypatch
):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.clear()
    window._guided_session_duration_edit.clear()
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {
            "csv_path": path,
            "chunk_duration_sec": 599.988,
        },
    )
    discovery = _timed_discovery(
        tmp_path / "first",
        [
            "2025_01_01-00_00_00",
            "2025_01_01-00_30_00",
            "2025_01_01-01_00_00",
        ],
    )
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    assert window._guided_sessions_per_hour_edit.text() == "2"
    assert window._guided_session_duration_edit.text() == "600"
    assert window._guided_recording_timing_inference.evidence[
        "raw_session_duration_sec"
    ] == 599.988
    assert window._guided_recording_timing_inference.evidence[
        "display_session_duration_sec"
    ] == 600.0
    assert window._sph_edit.text() == "2"
    assert window._duration_edit.text() == "600"
    assert window._guided_recording_continue_btn.isEnabled() is True
    assert window._guided_recording_timing_inference_label.text() == (
        "Detected 2 sessions/hour and approximately 600 s/session. "
        "Please confirm."
    )


def test_guided_recording_timing_preserves_manual_edit_until_new_source(
    window, tmp_path, monkeypatch
):
    _set_guided_rwd_intermittent(window)
    duration = {"value": 600.0}
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {
            "csv_path": path,
            "chunk_duration_sec": duration["value"],
        },
    )
    first = _timed_discovery(
        tmp_path / "first",
        ["2025_01_01-00_00_00", "2025_01_01-00_30_00"],
    )
    window._discovery_cache = first
    window._populate_discovery_ui(first)

    window._guided_session_duration_edit.setFocus()
    window._guided_session_duration_edit.selectAll()
    QTest.keyClicks(window._guided_session_duration_edit, "700")
    assert window._guided_recording_timing_inference_label.text() == (
        "Using manually entered session timing."
    )
    window._sync_guided_discovery_from_full()
    assert window._guided_session_duration_edit.text() == "700"
    assert window._guided_use_detected_timing_btn.isHidden() is False
    assert window._guided_use_detected_timing_btn.isEnabled() is True

    window._guided_session_duration_edit.setFocus()
    window._guided_session_duration_edit.selectAll()
    QTest.keyClick(
        window._guided_session_duration_edit, Qt.Key_Backspace
    )
    window._sync_guided_discovery_from_full()
    assert window._guided_session_duration_edit.text() == ""
    assert window._guided_recording_continue_btn.isEnabled() is False
    assert window._guided_use_detected_timing_btn.isHidden() is False

    window._guided_use_detected_timing_btn.click()
    assert window._guided_sessions_per_hour_edit.text() == "2"
    assert window._guided_session_duration_edit.text() == "600"
    assert window._sph_edit.text() == "2"
    assert window._duration_edit.text() == "600"
    assert window._guided_recording_continue_btn.isEnabled() is True
    assert window._guided_use_detected_timing_btn.isHidden() is True
    assert window._guided_recording_timing_inference_label.text() == (
        "Detected 2 sessions/hour and approximately 600 s/session. "
        "Please confirm."
    )

    duration["value"] = 300.0
    second = _timed_discovery(
        tmp_path / "second",
        ["2025_01_02-00_00_00", "2025_01_02-00_15_00"],
    )
    window._discovery_cache = second
    window._populate_discovery_ui(second)
    assert window._guided_sessions_per_hour_edit.text() == "4"
    assert window._guided_session_duration_edit.text() == "300"


def test_guided_ambiguous_or_unsupported_timing_does_not_overwrite_values(
    window, tmp_path, monkeypatch
):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.setText("3")
    window._guided_session_duration_edit.setText("500")
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {"csv_path": path, "chunk_duration_sec": 600.0},
    )
    ambiguous = _timed_discovery(
        tmp_path / "ambiguous",
        ["2025_01_01-00_00_00"],
    )
    window._discovery_cache = ambiguous
    window._populate_discovery_ui(ambiguous)

    assert window._guided_sessions_per_hour_edit.text() == "3"
    assert window._guided_session_duration_edit.text() == "500"
    assert "Could not confidently detect session timing" in (
        window._guided_recording_timing_inference_label.text()
    )
    assert window._guided_use_detected_timing_btn.isHidden() is True

    window._guided_sessions_per_hour_edit.clear()
    window._guided_session_duration_edit.clear()
    unsupported = _timed_discovery(
        tmp_path / "custom",
        [],
        resolved_format="custom_tabular",
    )
    window._discovery_cache = unsupported
    window._populate_discovery_ui(unsupported)
    assert window._guided_sessions_per_hour_edit.text() == ""
    assert window._guided_session_duration_edit.text() == ""
    assert window._guided_recording_timing_inference_label.text() == (
        "Automatic timing detection is not available for this format yet."
    )
    assert window._guided_use_detected_timing_btn.isHidden() is True


def test_guided_rwd_timing_duration_reads_are_bounded(
    window, tmp_path, monkeypatch
):
    _set_guided_rwd_intermittent(window)
    start = datetime(2025, 1, 1)
    session_ids = [
        (start + timedelta(minutes=30 * index)).strftime(
            "%Y_%m_%d-%H_%M_%S"
        )
        for index in range(30)
    ]
    calls = []

    def infer_contract(path):
        calls.append(path)
        return {"csv_path": path, "chunk_duration_sec": 600.0}

    monkeypatch.setattr(
        window, "_infer_rwd_chunk_contract", infer_contract
    )
    discovery = _timed_discovery(tmp_path / "many", session_ids)
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    assert len(calls) == 3
    assert calls == [
        discovery["sessions"][0]["path"],
        discovery["sessions"][15]["path"],
        discovery["sessions"][29]["path"],
    ]
    assert window._guided_sessions_per_hour_edit.text() == "2"
    evidence = window._guided_recording_timing_inference.evidence
    assert evidence["duration_sample_size"] == 3
    assert evidence["n_sessions_available"] == 30
    assert evidence["duration_sampling_strategy"] == "first_middle_last"
    assert evidence["duration_sample_paths"] == calls


def test_guided_rwd_disagreeing_duration_sample_is_ambiguous(
    window, tmp_path, monkeypatch
):
    _set_guided_rwd_intermittent(window)
    window._guided_sessions_per_hour_edit.setText("3")
    window._guided_session_duration_edit.setText("500")
    start = datetime(2025, 1, 1)
    session_ids = [
        (start + timedelta(minutes=30 * index)).strftime(
            "%Y_%m_%d-%H_%M_%S"
        )
        for index in range(30)
    ]
    calls = []

    def infer_contract(path):
        calls.append(path)
        duration = 650.0 if len(calls) == 2 else 600.0
        return {"csv_path": path, "chunk_duration_sec": duration}

    monkeypatch.setattr(
        window, "_infer_rwd_chunk_contract", infer_contract
    )
    discovery = _timed_discovery(tmp_path / "mixed", session_ids)
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    assert len(calls) == 3
    assert window._guided_recording_timing_inference.status == "ambiguous"
    assert window._guided_sessions_per_hour_edit.text() == "3"
    assert window._guided_session_duration_edit.text() == "500"


def test_guided_recording_requiredness_updates_with_format_and_mode(window):
    _set_guided_rwd_intermittent(window)
    assert "required for RWD intermittent input" in (
        window._guided_recording_structure_help_label.text()
    )

    window._guided_format_combo.setCurrentText("custom_tabular")
    assert "outside the current Guided Run scope" in (
        window._guided_recording_structure_help_label.text()
    )
    assert "required for RWD intermittent input" not in (
        window._guided_recording_structure_help_label.text()
    )

    window._guided_format_combo.setCurrentText("rwd")
    continuous = window._guided_acquisition_mode_combo.findData("continuous")
    window._guided_acquisition_mode_combo.setCurrentIndex(continuous)
    assert "Continuous acquisition is outside the current Guided Run scope" in (
        window._guided_recording_structure_help_label.text()
    )
    assert "required for RWD intermittent input" not in (
        window._guided_recording_structure_help_label.text()
    )


def test_guided_recording_structure_text_is_user_safe(window):
    _set_guided_rwd_intermittent(window)
    text = " ".join(
        (
            window._guided_session_duration_label.text(),
            window._guided_session_duration_edit.placeholderText(),
            window._guided_session_duration_edit.toolTip(),
            window._guided_recording_structure_help_label.text(),
        )
    ).lower()
    prohibited = (
        "backend validator",
        "manifest",
        "run spec",
        "startup",
        "materialization",
        "contract snapshot",
        "diagnostic cache",
        "completed run",
        "open results",
    )
    assert not any(term in text for term in prohibited)


def test_full_control_recording_structure_syncs_to_guided_display(window):
    idx = window._acquisition_mode_combo.findData("continuous")
    assert idx >= 0
    window._acquisition_mode_combo.setCurrentIndex(idx)
    window._continuous_window_sec_spin.setValue(1200.0)
    window._allow_partial_final_window_cb.setChecked(True)

    assert window._guided_acquisition_mode_combo.currentData() == "continuous"
    assert float(window._guided_continuous_window_sec_spin.value()) == 1200.0
    assert window._guided_allow_partial_final_window_cb.isChecked() is True

    idx = window._acquisition_mode_combo.findData("intermittent")
    window._acquisition_mode_combo.setCurrentIndex(idx)
    window._sph_edit.setText("12")
    window._duration_edit.setText("100")

    assert window._guided_acquisition_mode_combo.currentData() == "intermittent"
    assert window._guided_sessions_per_hour_edit.text() == "12"
    assert window._guided_session_duration_edit.text() == "100"


def test_guided_roi_discovery_mirrors_existing_discovery_state(window):
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": 2,
        "n_preview": 2,
        "sessions": [{"session_id": "s1"}, {"session_id": "s2"}],
        "rois": [{"roi_id": "CH1"}, {"roi_id": "CH2"}],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    assert window._guided_discovery_summary_label.text() == (
        "Detected RWD data with 2 recording sessions. Channels detected "
        "below."
    )
    assert [window._guided_roi_list.item(i).text() for i in range(2)] == ["CH1", "CH2"]

    window._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
    assert window._roi_list.item(1).checkState() == Qt.Unchecked

    window._roi_list.item(0).setCheckState(Qt.Unchecked)
    assert window._guided_roi_list.item(0).checkState() == Qt.Unchecked


def test_guided_format_change_is_lightweight_and_clears_stale_discovery(
    window, monkeypatch
):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("custom_tabular")
    discovery = {
        "resolved_format": "custom_tabular",
        "n_total_discovered": 1,
        "n_preview": 1,
        "sessions": [{"session_id": "s1"}],
        "rois": [{"roi_id": "ROI_A"}],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)
    assert window._guided_roi_list.count() == 1

    def fail(*_args, **_kwargs):
        raise AssertionError("format change must not inspect data or discover ROIs")

    window._validation_passed = True
    validation_revision = window._guided_backend_validation_revision
    full_format_signals = []
    representative_signals = []
    window._format_combo.currentIndexChanged.connect(
        lambda index: full_format_signals.append(index)
    )
    window._rep_session_combo.currentIndexChanged.connect(
        lambda index: representative_signals.append(index)
    )
    monkeypatch.setattr(window, "_on_discover", fail)
    monkeypatch.setattr(window, "_infer_dataset_contract_overrides", fail)
    monkeypatch.setattr(window, "_refresh_guided_mode_display", fail)
    monkeypatch.setattr(window, "_refresh_splitter_workspace_policy", fail)
    monkeypatch.setattr(window, "_update_button_states", fail)
    original_summary_refresh = window._refresh_guided_setup_summary
    summary_refresh_modes = []

    def record_summary_refresh(*, lightweight=False):
        summary_refresh_modes.append(lightweight)
        return original_summary_refresh(lightweight=lightweight)

    monkeypatch.setattr(
        window,
        "_refresh_guided_setup_summary",
        record_summary_refresh,
    )
    window._guided_format_combo.setCurrentText("rwd")

    assert window._format_combo.currentText() == "rwd"
    # Guided synchronization blocks the mirrored Full Control combo signals,
    # then invalidates shared validation state directly. Full Control's broad
    # synchronous _on_config_changed refresh cascade is not entered.
    assert window._validation_passed is False
    assert window._guided_backend_validation_revision == validation_revision + 1
    assert window._run_btn.isEnabled() is False
    assert full_format_signals == []
    assert representative_signals == []
    assert summary_refresh_modes == [True]
    assert window._discovery_cache is None
    assert window._roi_list.count() == 0
    assert window._guided_roi_list.count() == 0
    assert not hasattr(window, "_guided_resolved_format_label")
    assert window._guided_discovery_summary_label.text() == (
        "Format changed. Select ROIs to run discovery for the selected input."
    )


def test_guided_roi_discovery_button_reuses_existing_discovery_handler(
    window, tmp_path, monkeypatch
):
    import threading

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    window._set_guided_workflow_mode("new_analysis")
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))

    started = threading.Event()
    release = threading.Event()
    discovery = {
        "input_dir": str(input_dir),
        "resolved_format": "rwd",
        "n_total_discovered": 1,
        "n_preview": 1,
        "sessions": [{"session_id": "s1"}],
        "rois": [{"roi_id": "ROI_A"}],
    }

    class BlockingSpec:
        def run_discovery(self):
            started.set()
            assert release.wait(timeout=5)
            return discovery

    monkeypatch.setattr(
        window,
        "_on_discover",
        lambda: pytest.fail("Guided worker must not call _on_discover"),
    )
    monkeypatch.setattr(
        window,
        "_build_discovery_spec",
        lambda: pytest.fail(
            "Guided handler must not build discovery spec on GUI thread"
        ),
    )
    monkeypatch.setattr(
        window,
        "_build_discovery_spec_from_snapshot",
        lambda snapshot: BlockingSpec(),
    )
    monkeypatch.setattr(
        main_window_module, "GUIDED_DISCOVERY_DIAGNOSTICS", True
    )
    window._on_guided_discover_rois()

    assert started.wait(timeout=5)
    assert window._guided_roi_discovery_running is True
    assert window._guided_discover_rois_btn.isEnabled() is False
    assert window._guided_discover_rois_btn.text() == "Discovering..."
    assert window._guided_roi_discovery_progress.isHidden() is False
    assert window._guided_input_dir_edit.isEnabled() is False
    assert window._guided_format_combo.isEnabled() is False
    release.set()
    for _ in range(100):
        QApplication.processEvents()
        if not window._guided_roi_discovery_running:
            break
        QTest.qWait(10)

    assert window._guided_roi_discovery_running is False
    assert window._guided_roi_list.item(0).text() == "ROI_A"
    assert window._guided_discovery_summary_label.text() == (
        "Detected RWD data with 1 recording session. Channels detected "
        "below."
    )
    assert window._guided_roi_discovery_progress.isHidden() is True
    assert window._guided_discover_rois_btn.isEnabled() is True
    assert window._guided_discover_rois_btn.text() == "Select ROIs..."
    assert window._guided_input_dir_edit.isEnabled() is True
    assert window._guided_format_combo.isEnabled() is True
    assert window._guided_select_data_continue_btn.isEnabled() is True
    for _ in range(100):
        QApplication.processEvents()
        if window._guided_roi_discovery_thread is None:
            break
        QTest.qWait(10)
    assert window._guided_roi_discovery_thread is None
    assert window._guided_roi_discovery_worker is None
    diagnostic_log = window._log_view.toPlainText()
    for label in (
        "handler_entered",
        "worker_run_entered",
        "worker_build_spec_start",
        "worker_build_spec_end",
        "run_discovery_start",
        "run_discovery_end",
        "success_slot_entered",
        "complete",
    ):
        assert f"GUIDED_DISCOVERY_DIAG" in diagnostic_log
        assert f"label={label}" in diagnostic_log
    assert "worker_is_gui_thread=False" in diagnostic_log


def test_guided_roi_discovery_restores_busy_state_after_failure(
    window, tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))

    class FailingSpec:
        def run_discovery(self):
            raise RuntimeError("simulated discovery failure")

    critical = []
    monkeypatch.setattr(
        window,
        "_build_discovery_spec_from_snapshot",
        lambda snapshot: FailingSpec(),
    )
    monkeypatch.setattr(
        main_window_module, "GUIDED_DISCOVERY_DIAGNOSTICS", True
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        lambda *_args: critical.append(_args[-1]),
    )
    window._on_guided_discover_rois()
    for _ in range(100):
        QApplication.processEvents()
        if not window._guided_roi_discovery_running:
            break
        QTest.qWait(10)

    assert window._guided_roi_discovery_running is False
    assert window._guided_roi_discovery_progress.isHidden() is True
    assert window._guided_discover_rois_btn.isEnabled() is True
    assert window._guided_discover_rois_btn.text() == "Select ROIs..."
    assert window._guided_discovery_summary_label.text() == (
        "ROI discovery failed. Check the selected input and try again."
    )
    assert "simulated discovery failure" in critical[0]
    for _ in range(100):
        QApplication.processEvents()
        if window._guided_roi_discovery_thread is None:
            break
        QTest.qWait(10)
    assert window._guided_roi_discovery_thread is None
    assert window._guided_roi_discovery_worker is None
    diagnostic_log = window._log_view.toPlainText()
    assert "GUIDED_DISCOVERY_DIAG" in diagnostic_log
    assert "label=worker_failure" in diagnostic_log
    assert "label=failure_slot_entered" in diagnostic_log
    assert "label=complete" in diagnostic_log


def test_guided_roi_discovery_worker_spec_build_failure_restores_ui(
    window, tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))

    def fail_builder(snapshot):
        raise RuntimeError("simulated spec build failure")

    critical = []
    monkeypatch.setattr(
        window,
        "_build_discovery_spec_from_snapshot",
        fail_builder,
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        lambda *_args: critical.append(_args[-1]),
    )
    window._on_guided_discover_rois()
    for _ in range(100):
        QApplication.processEvents()
        if not window._guided_roi_discovery_running:
            break
        QTest.qWait(10)

    assert window._guided_roi_discovery_running is False
    assert window._guided_roi_discovery_progress.isHidden() is True
    assert window._guided_discover_rois_btn.isEnabled() is True
    assert "simulated spec build failure" in critical[0]


def test_guided_roi_discovery_running_guard_blocks_duplicate_call(
    window, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        window, "_build_discovery_spec", lambda: calls.append("discover")
    )
    window._guided_roi_discovery_running = True

    window._on_guided_discover_rois()

    assert calls == []


def test_guided_format_change_status_uses_only_select_data_language(
    window,
):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("custom_tabular")
    text = window._guided_discovery_summary_label.text().lower()
    assert "select rois" in text
    prohibited = (
        "open results",
        "completed run",
        "manifest",
        "backend startup",
        "diagnostic cache",
    )
    assert not any(term in text for term in prohibited)


def test_guided_setup_values_are_run_spec_relevant_state_equivalent(window, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._guided_format_combo.setCurrentText("custom_tabular")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    window._guided_sessions_per_hour_edit.setText("4")
    window._guided_session_duration_edit.setText("600")

    assert window._input_dir.text() == str(input_dir)
    assert window._output_dir.text() == str(output_dir)
    assert window._format_combo.currentText() == "custom_tabular"
    assert window._selected_acquisition_mode() == "intermittent"
    assert window._sph_edit.text() == "4"
    assert window._duration_edit.text() == "600"

    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )
    window._guided_continuous_window_sec_spin.setValue(750.0)

    assert window._selected_acquisition_mode() == "continuous"
    assert float(window._continuous_window_sec_spin.value()) == 750.0
    assert float(window._continuous_step_sec_spin.value()) == 750.0


def test_guided_setup_summary_state_is_read_only_and_tracks_current_state(window, tmp_path):
    """4J16k28: the raw-state debug panel was removed, but the underlying
    state-collection method it used to render must keep tracking setup
    changes without writing anything -- it still backs Guided/Full Control
    equivalence checks and the Correction Approach page's own details panel."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._guided_format_combo.setCurrentText("custom_tabular")
    window._guided_sessions_per_hour_edit.setText("8")

    state = window._guided_setup_summary_state()
    assert state["input_dir"] == str(input_dir)
    assert state["output_dir"] == str(output_dir)
    assert state["format"] == "custom_tabular"
    assert state["sessions_per_hour"] == "8"
    assert not list(output_dir.rglob("*"))


def test_guided_setup_summary_panel_removed_from_normal_pages(window, tmp_path, monkeypatch):
    """4J16k28: the raw internal "Current setup summary" debug panel (which
    dumped e.g. "RWD incomplete final chunk exclusion: on", "Guided
    correction intent: ...", "Diagnostics: not_generated") no longer
    appears anywhere in the Guided shell. Review Plan's scientist-facing
    summaries and technical-details toggle cover that need instead."""
    assert window._guided_workflow_tab.findChild(QGroupBox, "guidedSetupSummaryPanel") is None
    assert not hasattr(window, "_guided_setup_summary_label")
    assert not hasattr(window, "_guided_setup_summary_content")

    # The internal "Planned stages / not yet wired" roadmap panel has been
    # removed from Guided Mode and must not reappear.
    assert window._guided_workflow_tab.findChild(QGroupBox, "guidedWorkflowPlannedStages") is None
    assert not hasattr(window, "_guided_planned_stages_content")

    # Nothing about removing the display panel should touch state tracking
    # (still used for Guided/Full Control equivalence checks) or trigger
    # backend preview calls as a side effect of ordinary setup edits.
    state_before = _state_for_equivalence(window)
    calls = {"preview": 0}

    def _fake_preview_backend(*_args, **_kwargs):
        calls["preview"] += 1
        return {}

    monkeypatch.setattr(main_window_module, "run_guided_correction_preview_comparison", _fake_preview_backend)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._guided_format_combo.setCurrentText("custom_tabular")
    assert calls["preview"] == 0

    state_after = _state_for_equivalence(window)
    assert state_after != state_before
    assert window._guided_setup_summary_state()["input_dir"] == str(input_dir)
    assert window._guided_setup_summary_state()["output_dir"] == str(output_dir)
    assert window._guided_setup_summary_state()["format"] == "custom_tabular"


def test_guided_and_full_control_intermittent_setup_summary_equivalence(qapp, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        guided._guided_input_dir_edit.setText(str(input_dir))
        guided._guided_output_dir_edit.setText(str(output_dir))
        guided._guided_format_combo.setCurrentText("custom_tabular")
        guided._guided_acquisition_mode_combo.setCurrentIndex(
            guided._guided_acquisition_mode_combo.findData("intermittent")
        )
        guided._guided_sessions_per_hour_edit.setText("4")
        guided._guided_session_duration_edit.setText("600")

        full._input_dir.setText(str(input_dir))
        full._output_dir.setText(str(output_dir))
        full._format_combo.setCurrentText("custom_tabular")
        full._acquisition_mode_combo.setCurrentIndex(
            full._acquisition_mode_combo.findData("intermittent")
        )
        full._sph_edit.setText("4")
        full._duration_edit.setText("600")

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
    finally:
        _close_window(guided)
        _close_window(full)


def test_guided_and_full_control_continuous_setup_summary_equivalence(qapp, tmp_path):
    input_dir = tmp_path / "continuous_input"
    output_dir = tmp_path / "continuous_output"
    input_dir.mkdir()
    output_dir.mkdir()

    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        guided._guided_input_dir_edit.setText(str(input_dir))
        guided._guided_output_dir_edit.setText(str(output_dir))
        guided._guided_format_combo.setCurrentText("auto")
        guided._guided_acquisition_mode_combo.setCurrentIndex(
            guided._guided_acquisition_mode_combo.findData("continuous")
        )
        guided._guided_continuous_window_sec_spin.setValue(900.0)
        guided._guided_allow_partial_final_window_cb.setChecked(True)

        full._input_dir.setText(str(input_dir))
        full._output_dir.setText(str(output_dir))
        full._format_combo.setCurrentText("auto")
        full._acquisition_mode_combo.setCurrentIndex(
            full._acquisition_mode_combo.findData("continuous")
        )
        full._continuous_window_sec_spin.setValue(900.0)
        full._allow_partial_final_window_cb.setChecked(True)

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
    finally:
        _close_window(guided)
        _close_window(full)


def test_guided_and_full_control_roi_selection_summary_equivalence(qapp):
    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        _populate_fake_discovery(guided)
        _populate_fake_discovery(full)

        guided._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
        full._roi_list.item(1).setCheckState(Qt.Unchecked)

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
    finally:
        _close_window(guided)
        _close_window(full)


def test_guided_and_full_control_rwd_final_chunk_option_equivalence(qapp):
    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        guided._guided_exclude_incomplete_final_rwd_chunk_cb.setChecked(True)
        full._exclude_incomplete_final_rwd_chunk_cb.setChecked(True)

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
        assert guided._exclude_incomplete_final_rwd_chunk_cb.isChecked() is True
    finally:
        _close_window(guided)
        _close_window(full)


@pytest.mark.parametrize("card_title,mode", GUIDED_CARD_TO_DYNAMIC_MODE.items())
def test_guided_reference_correction_cards_sync_to_dynamic_fit_mode(window, card_title, mode):
    button = window._guided_correction_select_buttons[card_title]
    button.click()

    assert window._selected_dynamic_fit_mode() == mode
    assert window._guided_correction_cards[card_title].property("guidedCorrectionSelected") is True
    assert window._guided_correction_intent == card_title


@pytest.mark.parametrize("card_title,mode", GUIDED_CARD_TO_DYNAMIC_MODE.items())
def test_full_control_dynamic_fit_mode_syncs_to_guided_reference_card(window, card_title, mode):
    idx = window._dynamic_fit_mode_combo.findData(mode)
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)

    assert window._guided_correction_cards[card_title].property("guidedCorrectionSelected") is True
    assert window._guided_correction_intent == card_title


def test_guided_signal_only_f0_intent_does_not_change_dynamic_fit_mode_or_write_manifest(window, tmp_path):
    idx = window._dynamic_fit_mode_combo.findData("robust_global_event_reject")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    before_mode = window._selected_dynamic_fit_mode()
    output_root = tmp_path / "out"
    window._applied_dff_output_root_edit.setText(str(output_root))

    window._guided_correction_select_buttons["Signal-Only F0"].click()

    assert window._selected_dynamic_fit_mode() == before_mode
    assert window._guided_correction_intent == "Signal-Only F0"
    assert window._guided_correction_cards["Signal-Only F0"].property("guidedCorrectionSelected") is True
    assert not (output_root / "gui_manifest").exists()
    assert not (output_root / "applied_dff_gui_provenance.json").exists()


def test_decision_support_audit_does_not_alter_dynamic_fit_mode(window):
    idx = window._dynamic_fit_mode_combo.findData("global_linear_regression")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    before_mode = window._selected_dynamic_fit_mode()

    assert "Decision-Support Audit" not in window._guided_correction_select_buttons
    assert window._guided_correction_cards["Decision-Support Audit"].property("guidedCorrectionCardNonExecuting") is True
    assert window._selected_dynamic_fit_mode() == before_mode


def test_guided_setup_summary_state_reports_correction_state(window):
    """4J16k28: the display panel that used to render this as raw text
    ("Reference correction method: ...", "Guided correction intent: ...")
    was removed; the underlying state-collection method must still report
    the correction choice accurately."""
    window._guided_correction_select_buttons["Adaptive Event-Gated Fit"].click()
    state = window._guided_setup_summary_state()

    assert state["reference_correction_method"] == "adaptive_event_gated_regression"
    assert state["guided_correction_intent"] == "Adaptive Event-Gated Fit"


def test_guided_diagnostics_step_has_status_context_and_slots(window):
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._set_guided_workflow_mode("start")

    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepCorrectionApproach"
    assert window._guided_workflow_tab.findChild(QGroupBox, "guidedDiagnosticsCompletedRunSection").title() == "Completed run"
    assert window._guided_workflow_tab.findChild(
        QWidget, "guidedDiagnosticsActionsSection"
    ) is not None
    assert (
        window._guided_workflow_tab.findChild(QGroupBox, "guidedDiagnosticsGeneratedOutputsSection").title()
        == "Technical generated-output details"
    )
    actions_section = window._guided_workflow_tab.findChild(
        QWidget, "guidedDiagnosticsActionsSection"
    )
    assert isinstance(actions_section.layout(), QVBoxLayout)
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: not generated; no completed run loaded"
    assert "Reference correction method:" in window._guided_diagnostics_context_label.text()
    assert "Decision-Support Audit: coming later / read-only evidence" in window._guided_diagnostics_context_label.text()
    assert "No Correction: not available in Guided Workflow" in window._guided_diagnostics_context_label.text()
    assert "Global linear baseline comparison" in window._guided_diagnostics_slot_labels
    assert "Decision-Support Audit evidence" in window._guided_diagnostics_slot_labels
    assert "not generated" in window._guided_diagnostics_slot_labels["Fit stability"].text()
    assert window._guided_diagnostics_completed_run_content.isHidden() is True
    assert window._guided_diagnostics_context_content.isHidden() is True
    assert window._guided_diagnostics_slot_labels["Fit stability"].isHidden() is True
    assert "No completed run is loaded" in window._guided_diagnostics_completed_run_label.text()
    assert "Load a completed run to generate preview-only correction comparisons" in window._guided_preview_source_status_label.text()
    assert window._guided_preview_generate_btn.text() == (
        "Generate correction preview"
    )
    assert window._guided_preview_generate_btn.isEnabled() is False
    assert window._guided_preview_result_label.text() == ""
    signal_panel = window._guided_workflow_tab.findChild(QGroupBox, "guidedSignalOnlyF0DiagnosticPanel")
    preview_panel = window._guided_workflow_tab.findChild(QGroupBox, "guidedCorrectionPreviewPanel")
    assert signal_panel is not None
    assert preview_panel is not None
    assert signal_panel is not preview_panel
    assert window._guided_preview_technical_details_group.isHidden() is True
    assert window._guided_signal_f0_technical_details_group.isHidden() is True
    assert "Load a completed run to generate Signal-Only F0 diagnostic review artifacts" in (
        window._guided_signal_f0_source_status_label.text()
    )
    assert window._guided_signal_f0_generate_btn.text() == (
        "Generate Signal-Only F0 preview"
    )
    assert window._guided_signal_f0_generate_btn.isEnabled() is False
    output_summary = window._guided_generated_outputs_summary_label.text()
    assert "Correction preview: not generated." in output_summary
    assert "Signal-Only F0 diagnostic: not generated." in output_summary
    assert "Load a completed run" not in output_summary


def test_guided_diagnostic_cache_action_blocks_without_roi_discovery(window, tmp_path):
    input_dir = tmp_path / "raw_input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._refresh_guided_diagnostic_cache_panel()

    assert window._guided_diagnostic_cache_build_btn.text() == (
        "Build correction evidence"
    )
    assert window._guided_diagnostic_cache_build_btn.isEnabled() is False
    assert "Run ROI discovery" in window._guided_diagnostic_cache_readiness_label.text()
    assert window._guided_diagnostic_cache_record is None
    assert window._current_run_dir == ""


def test_guided_full_evidence_controls_remain_internal_and_hidden(window):
    panel = window._guided_workflow_tab.findChild(QGroupBox, "guidedDiagnosticCachePanel")
    assert panel is not None
    assert panel.parentWidget() is not None
    assert panel.isWindow() is False
    assert panel.isHidden() is True
    assert panel.isChecked() is False
    assert window._guided_diagnostic_cache_build_btn.isVisible() is False


def test_guided_diagnostic_cache_build_launches_tuning_prep_without_completed_run(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    before_run_dir = window._current_run_dir

    window._guided_diagnostic_cache_build_btn.click()

    assert fake_runner.argv is not None
    assert fake_runner.run_dir
    cache_dir = tmp_path / "output" / "_guided_diagnostic_cache"
    assert str(fake_runner.run_dir).startswith(str(cache_dir))
    assert "--run-type" in fake_runner.argv
    assert fake_runner.argv[fake_runner.argv.index("--run-type") + 1] == "tuning_prep"
    assert "--include-rois" in fake_runner.argv
    assert fake_runner.argv[fake_runner.argv.index("--include-rois") + 1] == "CH1,CH2,CH3"
    assert "--out" in fake_runner.argv
    assert fake_runner.argv[fake_runner.argv.index("--out") + 1] == fake_runner.run_dir
    assert window._current_run_dir == before_run_dir
    request_json = tmp_path / "output" / "_guided_diagnostic_cache" / Path(fake_runner.run_dir).name / "guided_diagnostic_cache_request.json"
    assert request_json.exists()
    request = json.loads(request_json.read_text(encoding="utf-8"))
    assert request["diagnostic_scope"] == "full_selected_input"
    assert request["included_roi_ids"] == ["CH1", "CH2", "CH3"]
    assert request["baseline_config_source_kind"] == "lab_default"
    assert "Diagnostic cache building" in window._guided_diagnostic_cache_status_label.text()
    assert window._guided_diagnostic_cache_record is None
    assert window._report_viewer.has_loaded_results() is False


def test_guided_diagnostic_cache_success_rewrites_identity_files_after_cleanup(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner

    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    prelaunch_request = cache_path / "guided_diagnostic_cache_request.json"
    assert prelaunch_request.exists()
    prelaunch_request.unlink()
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    assert window._guided_diagnostic_cache_request is not None
    assert window._guided_diagnostic_cache_record is not None
    assert window._guided_diagnostic_cache_status.ok is True
    assert "Diagnostic cache ready" in window._guided_diagnostic_cache_status_label.text()
    summary = window._guided_diagnostic_cache_summary_label.text()
    assert "preliminary; not final analysis" in summary
    assert "ROI count: 3" in summary
    assert str(cache_path) in summary
    assert (cache_path / "guided_diagnostic_cache_request.json").exists()
    assert (cache_path / "guided_diagnostic_cache_artifact.json").exists()
    assert (cache_path / "guided_diagnostic_cache_provenance.json").exists()
    request = json.loads((cache_path / "guided_diagnostic_cache_request.json").read_text(encoding="utf-8"))
    artifact = json.loads((cache_path / "guided_diagnostic_cache_artifact.json").read_text(encoding="utf-8"))
    provenance = json.loads((cache_path / "guided_diagnostic_cache_provenance.json").read_text(encoding="utf-8"))
    assert request["included_roi_ids"] == ["CH1", "CH2", "CH3"]
    assert artifact["purpose"] == "guided_diagnostic_cache"
    assert artifact["production_analysis"] is False
    assert artifact["included_roi_ids"] == ["CH1", "CH2", "CH3"]
    assert provenance["purpose"] == "guided_diagnostic_cache"
    assert provenance["production_analysis"] is False
    assert window._current_run_dir == ""
    assert window._report_viewer.has_loaded_results() is False


def test_guided_diagnostic_cache_missing_request_json_after_finish_blocks_ready(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()

    def _raise_request_write(_path, _request):
        raise OSError("request write blocked")

    monkeypatch.setattr(main_window_module, "write_build_request_json", _raise_request_write)
    window._on_guided_diagnostic_cache_finished(0)

    assert window._guided_diagnostic_cache_record is None
    assert window._guided_diagnostic_cache_status.ok is False
    assert "request JSON could not be written" in window._guided_diagnostic_cache_status_label.text()
    assert not (cache_path / "guided_diagnostic_cache_artifact.json").exists()
    assert not (cache_path / "guided_diagnostic_cache_provenance.json").exists()


def test_guided_diagnostic_cache_missing_identity_file_blocks_ready(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()

    real_write_json_file = main_window_module.write_json_file

    def _skip_provenance(path, payload):
        if str(path).endswith("guided_diagnostic_cache_provenance.json"):
            return None
        return real_write_json_file(path, payload)

    monkeypatch.setattr(main_window_module, "write_json_file", _skip_provenance)
    window._on_guided_diagnostic_cache_finished(0)

    assert window._guided_diagnostic_cache_record is None
    assert window._guided_diagnostic_cache_status.ok is False
    assert "identity/provenance files are missing" in window._guided_diagnostic_cache_status_label.text()
    assert (cache_path / "guided_diagnostic_cache_request.json").exists()
    assert (cache_path / "guided_diagnostic_cache_artifact.json").exists()
    assert not (cache_path / "guided_diagnostic_cache_provenance.json").exists()


def test_guided_diagnostic_cache_success_then_setup_change_marks_stale(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    item = window._guided_roi_list.item(2)
    item.setCheckState(Qt.Unchecked)
    window._refresh_guided_diagnostic_cache_panel()

    assert "Diagnostic cache stale" in window._guided_diagnostic_cache_status_label.text()
    assert "ROI inclusion/exclusion changed" in window._guided_diagnostic_cache_summary_label.text()
    assert window._guided_diagnostic_cache_record is not None


def test_guided_correction_preview_new_analysis_blocks_without_roi_selection(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_preview_generate_btn.isEnabled() is False
    assert "Complete ROI selection before generating a local correction preview" in (
        window._guided_preview_source_status_label.text()
    )


def test_guided_correction_preview_new_analysis_uses_current_diagnostic_cache(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._set_guided_workflow_mode("new_analysis")
    before_run_dir = window._current_run_dir
    monkeypatch.setattr(window._report_viewer, "load_report", lambda _path: pytest.fail("completed-run workspace loaded"))

    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    assert window._guided_preview_generate_btn.isEnabled() is True
    assert window._guided_preview_source_type == "diagnostic_cache"
    assert window._guided_preview_source_path == str(cache_path)
    assert "preliminary diagnostic cache" in window._guided_preview_source_status_label.text()
    assert "not final production analysis" in window._guided_preview_source_status_label.text()
    assert [window._guided_preview_roi_combo.itemText(i) for i in range(window._guided_preview_roi_combo.count())] == [
        "CH1",
        "CH2",
        "CH3",
    ]
    assert [window._guided_preview_chunk_combo.itemData(i) for i in range(window._guided_preview_chunk_combo.count())] == [
        0,
        1,
    ]
    assert window._current_run_dir == before_run_dir
    assert window._report_viewer.has_loaded_results() is False


def test_guided_correction_preview_new_analysis_stale_cache_blocks_preview(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._set_guided_workflow_mode("new_analysis")
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)
    assert window._guided_preview_generate_btn.isEnabled() is True

    window._guided_roi_list.item(2).setCheckState(Qt.Unchecked)
    window._refresh_guided_diagnostics_panel()

    assert window._guided_preview_generate_btn.isEnabled() is False
    assert "Diagnostic cache is stale" in window._guided_preview_source_status_label.text()
    assert "must be rebuilt" in window._guided_preview_source_status_label.text()


def test_guided_correction_preview_generation_passes_diagnostic_cache_identity(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._set_guided_workflow_mode("new_analysis")
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)
    before_run_dir = window._current_run_dir
    before_choices = dict(window._guided_strategy_choices)
    calls = {"args": None, "kwargs": None}

    def _fake_preview_backend(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "ok": True,
            "status": "success",
            "preview_id": kwargs.get("preview_id", "preview"),
            "preview_output_dir": kwargs.get("preview_output_dir", ""),
            "preview_summary_path": "summary.json",
            "preview_provenance_path": "provenance.json",
            "generated_artifacts": {},
            "method_statuses": {
                "robust_global_event_reject": {
                    "status": "success",
                    "errors": [],
                    "warnings": [],
                    "diagnostics_json": "diagnostics.json",
                    "trace_csv": "trace.csv",
                }
            },
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr(main_window_module, "run_guided_correction_preview_comparison", _fake_preview_backend)

    window._guided_preview_generate_btn.click()

    assert calls["args"] == (str(cache_path),)
    assert calls["kwargs"]["source_type"] == "diagnostic_cache"
    assert calls["kwargs"]["preview_output_dir"].startswith(str(cache_path / "_guided_workflow" / "previews"))
    assert calls["kwargs"]["roi"] == "CH1"
    assert calls["kwargs"]["chunk_index"] == 0
    assert window._guided_preview_last_result["source_type"] == "diagnostic_cache"
    assert window._guided_preview_last_result["diagnostic_cache_root"] == str(cache_path.resolve())
    assert "completed_run_dir" not in window._guided_preview_last_result
    assert window._current_run_dir == before_run_dir
    assert window._report_viewer.has_loaded_results() is False
    assert window._guided_strategy_choices == before_choices


def _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch) -> Path:
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._set_guided_workflow_mode("new_analysis")
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)
    return cache_path


def _generate_ready_guided_correction_preview(window) -> Path:
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    assert window._guided_preview_generate_btn.isEnabled()
    window._guided_preview_generate_btn.click()
    result = window._guided_preview_last_result
    assert result["status"] in {"success", "partial"}
    preview_dir = Path(result["preview_output_dir"])
    assert (preview_dir / "preview_provenance.json").is_file()
    return preview_dir


def test_guided_confirm_strategy_new_analysis_blocks_without_diagnostic_cache(window, monkeypatch):
    calls = {"preview": 0, "signal": 0}
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_confirm_roi_combo.isEnabled() is False
    assert window._guided_confirm_strategy_combo.isEnabled() is False
    assert window._guided_confirm_mark_btn.isEnabled() is False
    assert "Generate a local correction preview before confirming this strategy" in (
        window._guided_confirm_context_label.text()
    )
    assert "Build a diagnostic cache" not in window._guided_confirm_context_label.text()
    progress = window._guided_confirm_strategy_progress_label.text()
    assert progress == "0 of 0 ROIs confirmed."
    assert "Open Results must be used first" not in progress
    assert window._current_run_dir == ""
    assert calls == {"preview": 0, "signal": 0}


def test_guided_confirm_strategy_progress_none_confirmed(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._refresh_guided_confirm_strategy_panel()

    assert window._guided_confirm_strategy_progress_label.text() == (
        "0 of 3 ROIs confirmed."
    )
    assert window._guided_confirm_mark_btn.isEnabled() is False


def test_guided_confirm_strategy_cannot_reach_complete_without_preview_evidence(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    strategy_index = window._guided_confirm_strategy_combo.findData(
        "global_linear_regression"
    )
    for roi_index in range(window._guided_confirm_roi_combo.count()):
        window._guided_confirm_roi_combo.setCurrentIndex(roi_index)
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_confirm_ack_cb.setChecked(True)
        assert window._guided_confirm_mark_btn.isEnabled() is False
        window._guided_confirm_mark_btn.click()

    assert window._guided_strategy_choices == {}
    progress = window._guided_confirm_strategy_progress_label.text()
    assert "0 of 3 ROIs confirmed" in progress
    assert "Correction strategies confirmed for all included ROIs" not in progress
    assert "Open Results must be used first" not in progress


def test_guided_confirm_strategy_stale_preview_evidence_blocks_marks(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    _generate_ready_guided_correction_preview(window)
    window._guided_preview_mark_stale("Preview selection changed.")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    strategy_index = window._guided_confirm_strategy_combo.findData(
        "global_linear_regression"
    )
    window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
    window._guided_confirm_ack_cb.setChecked(True)

    assert window._guided_confirm_mark_btn.isEnabled() is False
    assert window._guided_confirm_strategy_progress_label.text() == (
        "The preview needs to be regenerated before confirming strategies. "
        "0 of 3 ROIs confirmed."
    )
    assert window._guided_correction_next_action_label.text() == (
        "The preview needs to be regenerated before confirming strategies."
    )


def test_guided_preview_evidence_populates_new_analysis_draft_fields(
    window, tmp_path, monkeypatch
):
    cache_path = _build_ready_guided_diagnostic_cache(
        window, tmp_path, monkeypatch
    )
    preview_dir = _generate_ready_guided_correction_preview(window)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    plan = window._build_guided_new_analysis_draft_plan()

    assert plan.correction_preview_result_id
    assert plan.correction_preview_path == str(preview_dir)
    assert plan.correction_preview_status == "current"
    assert plan.correction_preview_source_cache_id == (
        window._guided_diagnostic_cache_record.cache_id
    )
    assert preview_dir.parent == cache_path / "_guided_workflow" / "previews"
    assert (preview_dir / "preview_provenance.json").is_file()
    assert window._guided_confirm_strategy_progress_label.text() == (
        "0 of 3 ROIs confirmed."
    )
    assert "Preview ready" in (
        window._guided_correction_next_action_label.text()
    )


def test_guided_confirm_strategy_progress_partial_confirmation(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    _generate_ready_guided_correction_preview(window)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    strategy_index = window._guided_confirm_strategy_combo.findData(
        "global_linear_regression"
    )
    window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    assert window._guided_confirm_strategy_progress_label.text() == (
        "1 of 3 ROIs confirmed."
    )
    assert window._guided_correction_next_action_label.text() == (
        "Preview ready. Continue confirming methods. "
        "1 of 3 ROIs confirmed."
    )


def test_guided_confirm_strategy_progress_all_confirmed(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    _generate_ready_guided_correction_preview(window)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    strategy_index = window._guided_confirm_strategy_combo.findData(
        "global_linear_regression"
    )
    for roi_index in range(window._guided_confirm_roi_combo.count()):
        window._guided_confirm_roi_combo.setCurrentIndex(roi_index)
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_confirm_ack_cb.setChecked(True)
        window._guided_confirm_mark_btn.click()

    assert window._guided_confirm_strategy_progress_label.text() == (
        "3 of 3 ROIs confirmed."
    )
    assert window._guided_correction_next_action_label.text() == (
        "Correction method confirmed for 3 of 3 ROIs. Next step: "
        "Draft plan."
    )


def test_guided_confirm_strategy_progress_stale_choice_does_not_count(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    _generate_ready_guided_correction_preview(window)
    window._guided_strategy_choices[
        (("diagnostic_cache", "superseded-cache"), "CH1")
    ] = {
        "source_type": "diagnostic_cache",
        "strategy": "global_linear_regression",
        "confirmed": True,
        "choice_source": "explicit_user_mark",
        "current": False,
        "stale": True,
    }
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._refresh_guided_confirm_strategy_panel()

    assert window._guided_confirm_strategy_progress_label.text() == (
        "Some correction method choices need to be reconfirmed before Run. "
        "0 of 3 ROIs confirmed."
    )


def test_guided_confirm_strategy_progress_uses_included_rois_only(
    window, tmp_path, monkeypatch
):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._guided_roi_list.item(2).setCheckState(Qt.Unchecked)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._set_guided_workflow_mode("new_analysis")
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path, rois=("CH1", "CH2"))
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )

    assert window._guided_confirm_strategy_progress_label.text().endswith(
        "0 of 2 ROIs confirmed."
    )


def test_guided_confirm_strategy_progress_text_excludes_internal_terms(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    text = window._guided_confirm_strategy_progress_label.text().lower()
    prohibited = (
        "open results must be used first",
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
        "guided_candidate_manifest",
        "guided_startup",
        "wrapper_claim",
        "backend adapter",
        "orchestration",
        "subprocess",
        "raw command",
    )
    assert not any(term in text for term in prohibited)


def test_guided_confirm_strategy_new_analysis_uses_diagnostic_cache_roi_inventory(window, tmp_path, monkeypatch):
    cache_path = _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    before_run_dir = window._current_run_dir
    monkeypatch.setattr(window._report_viewer, "load_report", lambda _path: pytest.fail("completed-run workspace loaded"))

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_confirm_roi_combo.isEnabled() is True
    assert window._guided_confirm_strategy_combo.isEnabled() is True
    assert window._guided_confirm_source_type == "diagnostic_cache"
    assert window._guided_confirm_source.cache_root_path == str(cache_path)
    assert [window._guided_confirm_roi_combo.itemText(i) for i in range(window._guided_confirm_roi_combo.count())] == [
        "CH1",
        "CH2",
        "CH3",
    ]
    assert [window._guided_confirm_chunk_combo.itemData(i) for i in range(window._guided_confirm_chunk_combo.count())] == [
        0,
        1,
    ]
    assert window._guided_confirm_chunk_combo.isHidden() is False
    assert window._guided_confirm_chunk_label.isHidden() is False
    assert "preliminary diagnostic cache" in window._guided_confirm_context_label.text()
    assert "not final production analysis" in window._guided_confirm_context_label.text()
    assert window._current_run_dir == before_run_dir
    assert window._report_viewer.has_loaded_results() is False


def test_guided_confirm_strategy_new_analysis_stale_cache_blocks_and_marks_choices_stale(window, tmp_path, monkeypatch):
    cache_path = _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    _generate_ready_guided_correction_preview(window)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("robust_global_event_reject")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()
    choice_key = next(iter(window._guided_strategy_choices))
    assert choice_key[0][0] == "diagnostic_cache"
    assert window._guided_strategy_choices[choice_key]["cache_root_path"] == str(cache_path.resolve())
    assert window._guided_strategy_choices[choice_key]["current"] is True

    window._guided_roi_list.item(2).setCheckState(Qt.Unchecked)
    window._refresh_guided_diagnostics_panel()

    assert window._guided_confirm_roi_combo.isEnabled() is False
    assert window._guided_confirm_mark_btn.isEnabled() is False
    assert "Diagnostic cache is stale" in window._guided_confirm_context_label.text()
    assert "must be rebuilt" in window._guided_confirm_context_label.text()
    assert window._guided_strategy_choices[choice_key]["current"] is False
    assert window._guided_strategy_choices[choice_key]["stale"] is True


def test_guided_confirm_strategy_new_analysis_rejects_completed_run_like_folder_without_cache_identity(
    window, tmp_path, monkeypatch
):
    run_dir = _make_preview_completed_run(tmp_path)
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._current_run_dir = str(run_dir)
    window._set_guided_workflow_mode("new_analysis")

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_confirm_roi_combo.isEnabled() is False
    assert "Generate a local correction preview before confirming this strategy" in window._guided_confirm_context_label.text()
    assert window._guided_confirm_source_type == ""


def test_guided_confirm_strategy_new_analysis_rejects_production_diagnostic_cache(window, tmp_path, monkeypatch):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    window._guided_diagnostic_cache_record = replace(
        window._guided_diagnostic_cache_record,
        production_analysis=True,
    )

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._refresh_guided_confirm_strategy_panel()

    assert window._guided_confirm_roi_combo.isEnabled() is False
    assert "must not be marked as production analysis" in window._guided_confirm_context_label.text()
    assert window._guided_confirm_source_type == ""


def test_guided_confirm_strategy_new_analysis_roi_mismatch_blocks(window, tmp_path, monkeypatch):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    window._guided_diagnostic_cache_record = replace(
        window._guided_diagnostic_cache_record,
        included_roi_ids=("CH1", "CH2"),
    )

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    window._refresh_guided_confirm_strategy_panel()

    assert window._guided_confirm_roi_combo.isEnabled() is False
    assert window._guided_confirm_mark_btn.isEnabled() is False
    assert "Diagnostic cache ROI mismatch" in window._guided_confirm_context_label.text()
    assert window._guided_confirm_roi_combo.count() == 0


def test_guided_confirm_strategy_new_analysis_marks_source_scoped_choice_without_execution(
    window, tmp_path, monkeypatch
):
    cache_path = _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    _generate_ready_guided_correction_preview(window)
    calls = {"preview": 0, "signal": 0}
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )
    before_run_dir = window._current_run_dir
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("signal_only_f0")
    assert idx >= 0
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)

    window._guided_confirm_mark_btn.click()

    assert calls == {"preview": 0, "signal": 0}
    assert window._current_run_dir == before_run_dir
    assert len(window._guided_strategy_choices) == 1
    key, entry = next(iter(window._guided_strategy_choices.items()))
    assert key[0][0] == "diagnostic_cache"
    assert key[1] == "CH1"
    assert entry["source_type"] == "diagnostic_cache"
    assert entry["cache_id"] == window._guided_diagnostic_cache_record.cache_id
    assert entry["cache_root_path"] == str(cache_path.resolve())
    assert entry["diagnostic_scope_signature"] == window._guided_diagnostic_cache_record.diagnostic_scope_signature
    assert entry["build_request_signature"] == window._guided_diagnostic_cache_record.build_request_signature
    assert entry["strategy"] == "signal_only_f0"
    assert entry["choice_source"] == "explicit_user_mark"
    assert entry["no_auto_selection"] is True
    assert entry["production_analysis"] is False
    assert entry["preliminary_cache"] is True
    assert "not generated for current diagnostic cache" in entry["evidence_summary"]["signal_only_f0"]
    assert "Confirmed strategy for CH1" in (
        window._guided_confirm_marked_choice_label.text()
    )
    assert not (cache_path / "MANIFEST.csv").exists()
    assert not (cache_path / "_analysis" / "phasic_out" / "applied_dff").exists()


def test_guided_run_method_configuration_is_separate_from_evidence_state(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    _generate_ready_guided_correction_preview(window)
    idx = window._guided_confirm_strategy_combo.findData(
        "robust_global_event_reject"
    )
    assert idx >= 0
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    record = window._guided_diagnostic_cache_record
    preview_result = window._guided_preview_last_result
    choice_key, choice_before = next(
        iter(window._guided_strategy_choices.items())
    )

    window._guided_correction_select_buttons[
        "Adaptive Event-Gated Fit"
    ].click()

    assert window._guided_diagnostic_cache_record is record
    assert window._guided_preview_last_result is preview_result
    assert window._guided_preview_result_stale is False
    assert window._guided_strategy_choices[choice_key] == choice_before
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.global_correction_strategy == "dynamic_fit"
    assert plan.dynamic_fit_mode == "adaptive_event_gated_regression"


def test_guided_signal_only_f0_new_analysis_blocks_without_diagnostic_cache(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_signal_f0_generate_btn.isEnabled() is False
    assert "Build a diagnostic cache before running Signal-Only F0 diagnostic review" in (
        window._guided_signal_f0_source_status_label.text()
    )
    assert window._current_run_dir == ""


def test_guided_signal_only_f0_new_analysis_uses_current_diagnostic_cache(window, tmp_path, monkeypatch):
    cache_path = _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    before_run_dir = window._current_run_dir
    monkeypatch.setattr(window._report_viewer, "load_report", lambda _path: pytest.fail("completed-run workspace loaded"))

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_signal_f0_generate_btn.isEnabled() is True
    assert window._guided_signal_f0_source_type == "diagnostic_cache"
    assert window._guided_signal_f0_source_path == str(cache_path)
    assert "preliminary diagnostic cache" in window._guided_signal_f0_source_status_label.text()
    assert "not final production analysis" in window._guided_signal_f0_source_status_label.text()
    assert [window._guided_signal_f0_roi_combo.itemText(i) for i in range(window._guided_signal_f0_roi_combo.count())] == [
        "CH1",
        "CH2",
        "CH3",
    ]
    assert [
        window._guided_signal_f0_chunk_combo.itemData(i)
        for i in range(window._guided_signal_f0_chunk_combo.count())
    ] == [0, 1]
    assert window._current_run_dir == before_run_dir
    assert window._report_viewer.has_loaded_results() is False


def test_guided_signal_only_f0_new_analysis_stale_cache_blocks_diagnostic(window, tmp_path, monkeypatch):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    assert window._guided_signal_f0_generate_btn.isEnabled() is True

    window._guided_roi_list.item(2).setCheckState(Qt.Unchecked)
    window._refresh_guided_diagnostics_panel()

    assert window._guided_signal_f0_generate_btn.isEnabled() is False
    assert "Diagnostic cache is stale" in window._guided_signal_f0_source_status_label.text()
    assert "must be rebuilt" in window._guided_signal_f0_source_status_label.text()


def test_guided_signal_only_f0_new_analysis_rejects_completed_run_like_folder_without_cache_identity(
    window, tmp_path, monkeypatch
):
    run_dir = _make_preview_completed_run(tmp_path)
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._current_run_dir = str(run_dir)
    window._set_guided_workflow_mode("new_analysis")

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert window._guided_signal_f0_generate_btn.isEnabled() is False
    assert "Build a diagnostic cache before running Signal-Only F0 diagnostic review" in (
        window._guided_signal_f0_source_status_label.text()
    )
    assert window._guided_signal_f0_source_type == ""


def test_guided_signal_only_f0_new_analysis_rejects_production_diagnostic_cache(window, tmp_path, monkeypatch):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    window._guided_diagnostic_cache_record = replace(
        window._guided_diagnostic_cache_record,
        production_analysis=True,
    )

    window._refresh_guided_signal_f0_panel(window._guided_completed_run_diagnostic_artifacts())

    assert window._guided_signal_f0_generate_btn.isEnabled() is False
    assert "must not be marked as production analysis" in window._guided_signal_f0_source_status_label.text()
    assert window._guided_signal_f0_source_type == ""


def test_guided_signal_only_f0_generation_passes_diagnostic_cache_identity(window, tmp_path, monkeypatch):
    cache_path = _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    before_run_dir = window._current_run_dir
    before_choices = dict(window._guided_strategy_choices)
    calls = {"args": None, "kwargs": None}

    def _fake_backend(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "ok": True,
            "status": "success",
            "diagnostic_id": kwargs.get("diagnostic_id", "diagnostic"),
            "output_dir": str(kwargs.get("output_dir", "")),
            "provenance_path": "provenance.json",
            "summary_path": "summary.json",
            "chunk_csv_path": "",
            "trace_csv_paths": [],
            "warnings": [],
            "errors": [],
            "chunk_statuses": {"0": {"status": "success", "error": ""}},
            "source_type": "diagnostic_cache",
            "diagnostic_cache": {"cache_id": window._guided_diagnostic_cache_record.cache_id},
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_backend)

    window._guided_signal_f0_generate_btn.click()

    assert calls["args"] == (str(cache_path),)
    assert calls["kwargs"]["source_type"] == "diagnostic_cache"
    assert calls["kwargs"]["roi"] == "CH1"
    assert calls["kwargs"]["chunk_ids"] == [0]
    assert str(calls["kwargs"]["output_dir"]).startswith(
        str(cache_path / "_guided_workflow" / "signal_only_f0_diagnostics")
    )
    assert calls["kwargs"]["diagnostic_id"].startswith("diagnostic_cache_signal_only_f0_")
    assert window._guided_signal_f0_last_result["source_type"] == "diagnostic_cache"
    assert window._guided_signal_f0_last_result["diagnostic_cache_root"] == str(cache_path.resolve())
    assert "completed_run_dir" not in window._guided_signal_f0_last_result
    assert window._current_run_dir == before_run_dir
    assert window._report_viewer.has_loaded_results() is False
    assert window._guided_strategy_choices == before_choices
    assert not (cache_path / "MANIFEST.csv").exists()
    assert not (cache_path / "_analysis" / "phasic_out" / "applied_dff").exists()


def test_guided_confirm_strategy_sees_current_signal_only_f0_diagnostic_cache_evidence(
    window, tmp_path, monkeypatch
):
    cache_path = _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)

    def _fake_backend(*_args, **kwargs):
        return {
            "ok": True,
            "status": "success",
            "diagnostic_id": kwargs.get("diagnostic_id", "diagnostic"),
            "output_dir": str(kwargs.get("output_dir", "")),
            "provenance_path": "provenance.json",
            "summary_path": "summary.json",
            "chunk_csv_path": "",
            "trace_csv_paths": [],
            "warnings": [],
            "errors": [],
            "chunk_statuses": {"0": {"status": "success", "error": ""}},
            "source_type": "diagnostic_cache",
            "diagnostic_cache": {"cache_id": window._guided_diagnostic_cache_record.cache_id},
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_backend)
    window._guided_signal_f0_generate_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    assert "Signal-Only F0 diagnostic: success" in window._guided_confirm_evidence_label.text()
    assert window._guided_confirm_strategy_combo.currentData() == ""

    window._guided_roi_list.item(2).setCheckState(Qt.Unchecked)
    window._refresh_guided_diagnostics_panel()
    assert "Signal-Only F0 diagnostic: not generated for current diagnostic cache" in (
        window._guided_confirm_evidence_label.text()
    ) or "stale" in window._guided_confirm_evidence_label.text()
    assert window._guided_confirm_strategy_combo.currentData() == ""
    assert not (cache_path / "MANIFEST.csv").exists()


def test_guided_correction_preview_panel_populates_from_loaded_completed_run(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)

    _load_preview_completed_run(window, run_dir, monkeypatch)

    assert "Preview is generated from the loaded completed run" in window._guided_preview_source_status_label.text()
    if len(str(run_dir)) > 60:
        assert str(run_dir) not in window._guided_preview_source_status_label.text()
    assert window._display_path(str(run_dir)) in window._guided_preview_source_status_label.text()
    assert window._guided_preview_source_status_label.toolTip() == str(run_dir)
    assert [window._guided_preview_roi_combo.itemText(i) for i in range(window._guided_preview_roi_combo.count())] == [
        "CH1",
        "CH2",
    ]
    assert [window._guided_preview_chunk_combo.itemData(i) for i in range(window._guided_preview_chunk_combo.count())] == [
        0,
        1,
    ]
    assert set(window._guided_preview_method_checkboxes) == {
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
        "global_linear_regression",
    }
    assert all(cb.isChecked() for cb in window._guided_preview_method_checkboxes.values())
    assert window._guided_preview_generate_btn.isEnabled() is True
    method_text = " ".join(cb.text() for cb in window._guided_preview_method_checkboxes.values())
    assert "Signal-Only F0" not in method_text
    assert "Decision-Support Audit" not in method_text
    assert "No Correction" not in method_text
    assert window._guided_preview_technical_details_group.isHidden() is True


def test_guided_signal_only_f0_panel_populates_from_loaded_completed_run(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)

    _load_preview_completed_run(window, run_dir, monkeypatch)

    assert "Diagnostic review is generated from the loaded completed run" in (
        window._guided_signal_f0_source_status_label.text()
    )
    if len(str(run_dir)) > 60:
        assert str(run_dir) not in window._guided_signal_f0_source_status_label.text()
    assert window._display_path(str(run_dir)) in window._guided_signal_f0_source_status_label.text()
    assert window._guided_signal_f0_source_status_label.toolTip() == str(run_dir)
    assert [window._guided_signal_f0_roi_combo.itemText(i) for i in range(window._guided_signal_f0_roi_combo.count())] == [
        "CH1",
        "CH2",
    ]
    assert [
        window._guided_signal_f0_chunk_combo.itemData(i)
        for i in range(window._guided_signal_f0_chunk_combo.count())
    ] == [0, 1]
    assert window._guided_signal_f0_chunk_combo.currentData() == 0
    assert window._guided_signal_f0_generate_btn.isEnabled() is True
    method_text = " ".join(cb.text() for cb in window._guided_preview_method_checkboxes.values())
    assert "Signal-Only F0" not in method_text
    assert window._guided_signal_f0_technical_details_group.isHidden() is True


def test_guided_diagnostics_long_completed_run_paths_are_compact_in_visible_labels(window, tmp_path, monkeypatch):
    long_base = tmp_path / ("long_completed_run_path_segment_" * 3)
    long_base.mkdir()
    run_dir = _make_preview_completed_run(long_base)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._set_guided_workflow_mode("open_results")

    full_path = str(run_dir)
    assert len(full_path) > 60
    compact_path = window._display_path(full_path)

    visible_labels = [
        window._guided_mode_banner_label,
        window._guided_preview_source_status_label,
        window._guided_signal_f0_source_status_label,
    ]
    for label in visible_labels:
        assert compact_path in label.text()
        assert full_path not in label.text()

    assert window._guided_preview_source_status_label.toolTip() == full_path
    assert window._guided_signal_f0_source_status_label.toolTip() == full_path
    assert full_path in window._guided_diagnostics_completed_run_label.text()
    assert window._guided_diagnostics_completed_run_content.isHidden() is True


def test_guided_correction_preview_button_generates_backend_preview_read_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)

    window._guided_preview_generate_btn.click()

    assert window._guided_preview_status_label.text() == (
        "Preview ready."
    )
    artifacts_text = window._guided_preview_artifacts_label.text()
    assert "Preview directory:" in artifacts_text
    assert "Summary:" in artifacts_text
    assert "Provenance:" in artifacts_text
    table = window._guided_preview_method_table
    assert table.rowCount() == 3
    table_text = " ".join(
        table.item(row, col).text()
        for row in range(table.rowCount())
        for col in range(table.columnCount())
        if table.item(row, col) is not None
    )
    assert "Robust Global Event-Reject Fit" in table_text
    assert "Adaptive Event-Gated Fit" in table_text
    assert "Global Linear Regression" in table_text
    assert "method_global_linear_regression_diagnostics.json" in table_text
    assert "method_global_linear_regression_trace.csv" in table_text
    assert "Signal-Only F0" not in table_text
    assert "Decision-Support Audit" not in table_text
    assert "No Correction" not in table_text
    assert "auto" not in table_text
    assert "needs_review" not in table_text
    assert "Errors/warnings: none reported" in window._guided_preview_messages_label.text()
    text = window._guided_preview_result_label.text()
    assert "Preview-only correction comparison." in text
    assert "Strategy recommendation" not in text
    preview_dir = run_dir / "_guided_workflow" / "previews"
    assert preview_dir.exists()
    assert list(preview_dir.glob("*/preview_summary.json"))
    assert list(preview_dir.glob("*/preview_provenance.json"))
    assert not (preview_dir / "MANIFEST.json").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    output_summary = window._guided_generated_outputs_summary_label.text()
    assert "Correction preview: success" in output_summary
    assert "Signal-Only F0 diagnostic: not generated." in output_summary
    assert "Preview comparison ready" not in output_summary
    preview_output_dir = str(window._guided_preview_last_result["preview_output_dir"])
    assert window._display_path(preview_output_dir) in output_summary
    if len(preview_output_dir) > 60:
        assert preview_output_dir not in output_summary


def test_guided_signal_only_f0_button_generates_backend_diagnostic_read_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    before = {
        str(path.relative_to(phasic)): path.read_bytes()
        for path in sorted(phasic.rglob("*"))
        if path.is_file()
    }
    _load_preview_completed_run(window, run_dir, monkeypatch)

    window._guided_signal_f0_generate_btn.click()

    assert window._guided_signal_f0_status_label.text() == (
        "Signal-Only F0 preview: ready."
    )
    artifacts_text = window._guided_signal_f0_artifacts_label.text()
    assert "Diagnostic directory:" in artifacts_text
    assert "Provenance JSON:" in artifacts_text
    assert "Summary JSON:" in artifacts_text
    assert "Chunk CSV:" in artifacts_text
    assert "Strategy recommendation: none; not selected." in artifacts_text
    table = window._guided_signal_f0_chunk_table
    assert table.rowCount() == 1
    table_text = " ".join(
        table.item(row, col).text()
        for row in range(table.rowCount())
        for col in range(table.columnCount())
        if table.item(row, col) is not None
    )
    assert "0" in table_text
    assert "success" in table_text
    assert "best" not in table_text.lower()
    diagnostic_dir = run_dir / "_guided_workflow" / "signal_only_f0_diagnostics"
    assert diagnostic_dir.exists()
    assert list(diagnostic_dir.glob("*/signal_only_f0_diagnostic_provenance.json"))
    assert list(diagnostic_dir.glob("*/signal_only_f0_diagnostic_summary.json"))
    assert list(diagnostic_dir.glob("*/signal_only_f0_diagnostic_chunks.csv"))
    assert not list(diagnostic_dir.glob("*.png"))
    after = {
        str(path.relative_to(phasic)): path.read_bytes()
        for path in sorted(phasic.rglob("*"))
        if path.is_file()
    }
    assert after == before
    assert not (phasic / "qc").exists()
    assert not (phasic / "features").exists()
    assert not (phasic / "applied_dff").exists()
    assert not (run_dir / "manifest.csv").exists()
    output_summary = window._guided_generated_outputs_summary_label.text()
    assert "Correction preview: not generated." in output_summary
    assert "Signal-Only F0 diagnostic: success" in output_summary
    assert "diagnostic review ready" not in output_summary.lower()


def test_guided_correction_preview_does_not_auto_generate(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    calls = {"count": 0}

    def _fake_backend(*_args, **_kwargs):
        calls["count"] += 1
        return {
            "ok": True,
            "status": "success",
            "preview_output_dir": "preview",
            "preview_summary_path": "summary",
            "preview_provenance_path": "provenance",
            "method_statuses": {},
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr(main_window_module, "run_guided_correction_preview_comparison", _fake_backend)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    assert calls["count"] == 0
    _load_preview_completed_run(window, run_dir, monkeypatch)
    assert calls["count"] == 0
    window._guided_preview_roi_combo.setCurrentIndex(1)
    window._guided_preview_chunk_combo.setCurrentIndex(1)
    window._guided_preview_method_checkboxes["global_linear_regression"].setChecked(False)
    window._guided_correction_select_buttons["Adaptive Event-Gated Fit"].click()
    assert calls["count"] == 0

    window._guided_preview_generate_btn.click()
    assert calls["count"] == 1


def test_guided_signal_only_f0_diagnostic_is_explicit_button_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    calls = {"count": 0, "kwargs": None, "args": None}

    def _fake_backend(*args, **kwargs):
        calls["count"] += 1
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "ok": True,
            "status": "success",
            "diagnostic_id": "signal_only_f0_test",
            "output_dir": "diagnostic_dir",
            "provenance_path": "provenance.json",
            "summary_path": "summary.json",
            "chunk_csv_path": "chunks.csv",
            "trace_csv_paths": [],
            "warnings": [],
            "errors": [],
            "chunk_statuses": {"1": {"status": "success", "error": ""}},
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_backend)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    assert calls["count"] == 0
    _load_preview_completed_run(window, run_dir, monkeypatch)
    assert calls["count"] == 0
    window._guided_signal_f0_roi_combo.setCurrentIndex(1)
    window._guided_signal_f0_chunk_combo.setCurrentIndex(1)
    assert calls["count"] == 0
    window._guided_correction_select_buttons["Adaptive Event-Gated Fit"].click()
    assert calls["count"] == 0
    window._guided_preview_generate_btn.click()
    assert calls["count"] == 0

    window._guided_signal_f0_generate_btn.click()

    assert calls["count"] == 1
    assert calls["args"] == (str(run_dir),)
    assert calls["kwargs"]["roi"] == "CH2"
    assert calls["kwargs"]["chunk_ids"] == [1]
    assert calls["kwargs"]["allow_existing"] is False
    assert "output_dir" not in calls["kwargs"]
    assert "diagnostic_id" not in calls["kwargs"]


def test_guided_correction_preview_result_marks_stale_on_selection_change(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)

    window._guided_preview_generate_btn.click()
    assert window._guided_preview_status_label.text() == (
        "Preview ready."
    )

    window._guided_preview_chunk_combo.setCurrentIndex(1)

    assert "Preview needs to be regenerated" in window._guided_preview_status_label.text()
    output_summary = window._guided_generated_outputs_summary_label.text()
    assert "Correction preview: success stale" in output_summary
    assert "Displayed preview is stale" not in output_summary


def test_guided_signal_only_f0_result_displays_partial_failed_and_does_not_select_strategy(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    before_intent = window._guided_correction_intent

    def _fake_backend(*_args, **_kwargs):
        return {
            "ok": False,
            "status": "partial",
            "diagnostic_id": "signal_only_f0_test",
            "output_dir": "diagnostic_dir",
            "provenance_path": "provenance.json",
            "summary_path": "summary.json",
            "chunk_csv_path": "",
            "trace_csv_paths": [],
            "warnings": ["caution"],
            "errors": ["chunk 1: failed"],
            "chunk_statuses": {
                "0": {"status": "success", "error": ""},
                "1": {"status": "failed", "error": "failed"},
            },
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_backend)

    window._guided_signal_f0_generate_btn.click()

    assert window._guided_signal_f0_status_label.text() == (
        "Signal-Only F0 preview: ready."
    )
    assert "chunk 1: failed" in window._guided_signal_f0_messages_label.text()
    table_text = " ".join(
        window._guided_signal_f0_chunk_table.item(row, col).text()
        for row in range(window._guided_signal_f0_chunk_table.rowCount())
        for col in range(window._guided_signal_f0_chunk_table.columnCount())
        if window._guided_signal_f0_chunk_table.item(row, col) is not None
    )
    assert "failed" in table_text
    output_summary = window._guided_generated_outputs_summary_label.text()
    assert "Signal-Only F0 diagnostic: partial" in output_summary
    assert "chunk 1: failed" in output_summary
    assert window._guided_correction_intent == before_intent

    def _fake_failed(*_args, **_kwargs):
        return {
            "ok": False,
            "status": "failed",
            "diagnostic_id": "signal_only_f0_test",
            "output_dir": "",
            "provenance_path": "",
            "summary_path": "",
            "chunk_csv_path": "",
            "trace_csv_paths": [],
            "warnings": [],
            "errors": ["source failed"],
            "chunk_statuses": {},
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_failed)
    window._guided_signal_f0_generate_btn.click()
    assert "failed" in window._guided_signal_f0_status_label.text().lower()
    assert "source failed" in window._guided_signal_f0_messages_label.text()
    output_summary = window._guided_generated_outputs_summary_label.text()
    assert "Signal-Only F0 diagnostic: failed" in output_summary
    assert "source failed" in output_summary
    assert window._guided_correction_intent == before_intent


def test_guided_signal_only_f0_result_marks_stale_on_selection_change(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_signal_f0_generate_btn.click()
    assert window._guided_signal_f0_status_label.text() == (
        "Signal-Only F0 preview: ready."
    )

    window._guided_signal_f0_chunk_combo.setCurrentIndex(1)

    assert "Displayed Signal-Only F0 diagnostic review is stale because the selection changed" in (
        window._guided_signal_f0_status_label.text()
    )
    output_summary = window._guided_generated_outputs_summary_label.text()
    assert "Signal-Only F0 diagnostic: success stale" in output_summary
    assert "Displayed Signal-Only F0" not in output_summary


def test_guided_correction_preview_refresh_preserves_non_default_selection_with_result(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_preview_roi_combo.setCurrentIndex(window._guided_preview_roi_combo.findData("CH2"))
    window._guided_preview_chunk_combo.setCurrentIndex(window._guided_preview_chunk_combo.findData(1))
    window._guided_preview_has_result = True
    window._guided_preview_result_stale = False
    window._guided_preview_status_label.setText("Preview comparison generated: success.")
    window._guided_preview_result_label.setText("Preview status: success")

    window._refresh_guided_diagnostics_panel()

    assert window._guided_preview_roi_combo.currentData() == "CH2"
    assert window._guided_preview_chunk_combo.currentData() == 1
    assert "Preview comparison generated: success." in window._guided_preview_status_label.text()
    assert window._guided_preview_result_stale is False


def test_guided_diagnostics_reports_existing_completed_run_artifacts_read_only(window, tmp_path):
    run_dir = tmp_path / "completed"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "success", "configuration": {}, "analytical_contract": {}}),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (run_dir / "config_effective.yaml").write_text("event_signal: dff\n", encoding="utf-8")
    (run_dir / "gui_run_spec.json").write_text(json.dumps({"run": "spec"}), encoding="utf-8")
    (run_dir / "command_invoked.txt").write_text("python main.py\n", encoding="utf-8")
    summary_dir = run_dir / "CH1" / "summary"
    summary_dir.mkdir(parents=True)
    before = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))

    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir)) is True
    window._refresh_guided_diagnostics_panel()

    assert window._guided_diagnostics_status == "available"
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: available from loaded completed-run artifacts"
    text = window._guided_diagnostics_completed_run_label.text()
    assert "Loaded completed run artifacts; separate from the active editable setup" in text
    assert "Run summary/report: run_report.json" in text
    assert "Status: status.json" in text
    assert "Manifest/provenance: MANIFEST.json" in text
    assert "Effective config: config_effective.yaml" in text
    assert "GUI run spec: gui_run_spec.json" in text
    assert "Command log: command_invoked.txt" in text
    assert "Region deliverable: Summary: CH1" in text
    after = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after == before


def test_guided_diagnostics_loaded_run_without_recognized_artifacts_is_unavailable(window, tmp_path, monkeypatch):
    run_dir = tmp_path / "empty_loaded_run"
    run_dir.mkdir()

    window._current_run_dir = str(run_dir)
    monkeypatch.setattr(window._report_viewer, "has_loaded_results", lambda: True)
    window._refresh_guided_diagnostics_panel()

    assert window._guided_diagnostics_status == "unavailable"
    assert "Diagnostics: unavailable" in window._guided_diagnostics_status_label.text()
    text = window._guided_diagnostics_completed_run_label.text()
    assert str(run_dir) in text
    assert "No recognized completed-run diagnostic artifacts were found" in text


def test_guided_diagnostics_scope_loaded_artifacts_as_separate_from_active_setup(window, tmp_path):
    run_dir = tmp_path / "completed"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "success", "configuration": {}, "analytical_contract": {}}),
        encoding="utf-8",
    )
    input_dir = tmp_path / "active_input"
    input_dir.mkdir()

    window._current_run_dir = str(run_dir)
    (run_dir / "CH1" / "summary").mkdir(parents=True)
    assert window._report_viewer.load_report(str(run_dir)) is True
    window._guided_input_dir_edit.setText(str(input_dir))
    window._refresh_guided_diagnostics_panel()

    assert window._guided_diagnostics_status == "available"
    text = window._guided_diagnostics_completed_run_label.text()
    assert "Loaded completed run artifacts; separate from the active editable setup" in text
    assert str(run_dir) in text
    assert str(input_dir) not in text


def test_guided_new_analysis_mode_skips_completed_run_diagnostic_scan(window, tmp_path, monkeypatch):
    """4J16k22: the diagnostics/confirm-strategy panels that consume the
    completed-run artifact scan are hidden in new-analysis mode, so
    switching into that mode (e.g. "Set up new analysis") must not
    re-scan a previously loaded completed run's filesystem. Regression
    for a hang on that transition when the referenced completed-run
    directory was large or on a slow/cloud-synced filesystem."""
    run_dir = tmp_path / "completed"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "success", "configuration": {}, "analytical_contract": {}}),
        encoding="utf-8",
    )
    (run_dir / "CH1" / "summary").mkdir(parents=True)

    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir)) is True

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "completed-run filesystem scan must not run in new-analysis mode"
        )

    monkeypatch.setattr(
        main_window_module, "resolve_region_deliverables", _fail_if_called
    )
    monkeypatch.setattr(
        main_window_module, "is_successful_completed_run_dir", _fail_if_called
    )

    window._guided_workflow_mode = "new_analysis"
    result = window._guided_completed_run_diagnostic_artifacts()

    assert result == {"status": "not_generated", "run_dir": "", "artifacts": []}


@pytest.mark.parametrize("card_title,mode", GUIDED_CARD_TO_DYNAMIC_MODE.items())
def test_guided_diagnostics_context_tracks_reference_correction_cards(window, card_title, mode):
    window._guided_correction_select_buttons[card_title].click()
    context = window._guided_diagnostics_context_label.text()

    assert mode in context
    assert f"Guided correction intent: {card_title}" in context
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: not generated; no completed run loaded"


def test_guided_diagnostics_context_tracks_signal_only_intent(window):
    window._guided_correction_select_buttons["Signal-Only F0"].click()
    context = window._guided_diagnostics_context_label.text()

    assert "Guided correction intent: Signal-Only F0" in context
    assert "Signal-Only F0 intent: selected for later explicit confirmation" in context
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: not generated; no completed run loaded"


def test_guided_diagnostics_step_has_no_generation_or_execution_buttons(window):
    forbidden = {
        "Run diagnostics",
        "Generate previews",
        "Compare fits",
        "Run correction tuning",
        "Apply diagnostics",
        "Auto choose strategy",
        "Validate Only",
        "Run Pipeline",
        "Save Manifest",
        "Run Batch",
    }
    button_texts = {
        button.text()
        for button in window._guided_workflow_tab.findChildren(QPushButton)
        if button.text()
    }
    assert button_texts.isdisjoint(forbidden)
    assert "Generate correction preview" in button_texts


def test_guided_visible_text_does_not_use_stale_shell_or_completed_loader_wording(window):
    labels = [
        label.text()
        for label in window._guided_workflow_tab.findChildren(QLabel)
        if label.text() and not label.isHidden()
    ]
    visible_text = "\n".join(labels)
    assert "Stage 1 shell only" not in visible_text
    assert "does not wire a completed-run loader into the Guided Workflow" not in visible_text
    # Stale prototype-era shell wording must be gone: it implied Guided Mode
    # could not perform real validation/runs at all.
    assert "Production runs and applied-dF/F routing still use Full Control" not in visible_text
    assert "not yet wired" not in visible_text
    # Honest replacement orientation copy: Guided Mode can validate and run
    # supported setups, while some configurations may still need Full Control.
    assert "running supported analyses" in visible_text
    assert "may still need Full Control" in visible_text


def test_guided_output_policy_no_policy_by_default(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    plan, errors = window._build_guided_draft_run_plan()
    assert errors == []
    assert plan.output_policy.output_root is None

    preview = window._guided_draft_run_plan_preview_label.text()
    assert "Output destination: none configured" in preview
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "Output destination: not_configured" in checklist
    assert "Execution ready: false" in checklist


def test_guided_new_analysis_draft_plan_distinguishes_select_output_from_run_output(
    window, tmp_path, monkeypatch
):
    _input_dir, output_dir = _configure_guided_raw_cache_setup(
        window, tmp_path, monkeypatch
    )
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )

    status = window._guided_output_status_label.text()
    assert status == (
        "Select data output folder is configured. Required before Run: choose "
        "and apply the run output parent folder in Draft Plan."
    )
    assert "No output root is configured" not in status
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.output_base_path == str(output_dir)
    assert plan.output_policy_status == "missing"
    assert plan.output_policy_path is None


def test_guided_new_analysis_draft_plan_reports_missing_select_output(window):
    window._guided_output_dir_edit.setText("")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )

    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_output_status_label.text() == (
        "Required before Run: choose an output folder in Select data, then "
        "apply the run output parent folder in Draft Plan."
    )
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.output_base_path is None
    assert plan.output_policy_status == "missing"


def test_guided_new_analysis_output_status_survives_navigation(
    window, tmp_path, monkeypatch
):
    _input_dir, output_dir = _configure_guided_raw_cache_setup(
        window, tmp_path, monkeypatch
    )
    window._set_guided_workflow_mode("new_analysis")
    draft_index = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    window._guided_workflow_stepper.setCurrentRow(draft_index)
    before = window._guided_output_status_label.text()

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._guided_workflow_stepper.setCurrentRow(draft_index)

    assert window._guided_output_dir_edit.text() == str(output_dir)
    assert window._guided_output_status_label.text() == before
    assert "No output root is configured" not in before


def test_guided_new_analysis_applied_output_parent_is_real_draft_state(
    window, tmp_path, monkeypatch
):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    output_parent = tmp_path / "planned_runs"
    output_parent.mkdir()
    run_output_base = output_parent / "future_guided_runs"
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_output_path_edit.setText(str(run_output_base))

    window._guided_output_apply_btn.click()

    status = window._guided_output_status_label.text()
    assert status.startswith(
        "Output parent folder is configured. Guided Run will create a new run "
        "folder inside it."
    )
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.output_policy_status == "applied"
    assert plan.output_policy_explicitly_applied is True
    assert plan.output_policy_path == str(run_output_base.resolve())
    assert not run_output_base.exists()
    prohibited = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
        "guided_candidate_manifest",
        "guided_startup",
        "wrapper_claim",
        "backend adapter",
        "orchestration",
        "subprocess",
        "raw command",
    )
    assert not any(term in status.lower() for term in prohibited)


def test_guided_output_policy_apply_valid(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    target_out = tmp_path / "future_output"
    assert not target_out.exists()
    window._guided_output_path_edit.setText(str(target_out))
    window._guided_output_apply_btn.click()

    plan, errors = window._build_guided_draft_run_plan()
    assert errors == []
    assert plan.output_policy.output_root == str(target_out.resolve())
    assert plan.output_policy.overwrite is False
    assert plan.output_policy.legacy_outputs_protected is True

    preview = window._guided_draft_run_plan_preview_label.text()
    assert f"Output destination: {str(target_out.resolve())}" in preview
    assert "Legacy outputs protected: true" in preview
    assert "Overwrite existing: false" in preview
    checklist = window._guided_draft_run_plan_checklist_label.text()
    assert "Output destination: pass" in checklist
    assert "Execution ready: false" in checklist
    assert not target_out.exists()


def test_guided_output_policy_empty_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_output_path_edit.setText("   ")
    window._guided_output_apply_btn.click()

    plan, errors = window._build_guided_draft_run_plan()
    assert plan.output_policy.output_root is None
    assert "Output destination not applied: Output root path cannot be empty." in window._guided_output_status_label.text()
    assert "Output destination: not_configured" in window._guided_draft_run_plan_checklist_label.text()


def test_guided_output_policy_completed_run_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_output_path_edit.setText(str(run_dir))
    window._guided_output_apply_btn.click()

    plan, errors = window._build_guided_draft_run_plan()
    assert plan.output_policy.output_root is None
    assert "Output root cannot be the completed run directory itself." in window._guided_output_status_label.text()


def test_guided_output_policy_legacy_paths_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    legacy_subpaths = [
        "_analysis",
        "_analysis/phasic_out",
        "_analysis/phasic_out/features",
        "_analysis/phasic_out/applied_dff",
    ]
    for sub in legacy_subpaths:
        legacy_path = run_dir / sub
        window._guided_output_path_edit.setText(str(legacy_path))
        window._guided_output_apply_btn.click()

        plan, errors = window._build_guided_draft_run_plan()
        assert plan.output_policy.output_root is None
        assert "Output root cannot be inside the completed run directory." in window._guided_output_status_label.text()


def test_guided_output_policy_scoped_by_run(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    _load_preview_completed_run(window, run_a, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(out_a))
    window._guided_output_apply_btn.click()

    plan_a, _ = window._build_guided_draft_run_plan()
    assert plan_a.output_policy.output_root == str(out_a.resolve())

    _load_preview_completed_run(window, run_b, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    plan_b_initial, _ = window._build_guided_draft_run_plan()
    assert plan_b_initial.output_policy.output_root is None
    assert window._guided_output_path_edit.text() == ""

    window._guided_output_path_edit.setText(str(out_b))
    window._guided_output_apply_btn.click()
    plan_b_applied, _ = window._build_guided_draft_run_plan()
    assert plan_b_applied.output_policy.output_root == str(out_b.resolve())

    _load_preview_completed_run(window, run_a, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    plan_a_restored, _ = window._build_guided_draft_run_plan()
    assert plan_a_restored.output_policy.output_root == str(out_a.resolve())
    assert window._guided_output_path_edit.text() == str(out_a.resolve())


def test_guided_output_policy_unsaved_edits_do_not_leak(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")

    _load_preview_completed_run(window, run_a, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_output_path_edit.setText("typed_path_a")
    plan_a, _ = window._build_guided_draft_run_plan()
    assert plan_a.output_policy.output_root is None

    window._refresh_guided_confirm_strategy_panel()
    assert window._guided_output_path_edit.text() == "typed_path_a"

    _load_preview_completed_run(window, run_b, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    assert window._guided_output_path_edit.text() == ""
    plan_b, _ = window._build_guided_draft_run_plan()
    assert plan_b.output_policy.output_root is None


def test_guided_output_policy_invalid_apply_preserves_previous(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    out_valid = tmp_path / "out_valid"
    window._guided_output_path_edit.setText(str(out_valid))
    window._guided_output_apply_btn.click()
    assert window._build_guided_draft_run_plan()[0].output_policy.output_root == str(out_valid.resolve())

    window._guided_output_path_edit.setText(str(run_dir))
    window._guided_output_apply_btn.click()

    assert "Output root cannot be the completed run directory itself." in window._guided_output_status_label.text()
    assert window._build_guided_draft_run_plan()[0].output_policy.output_root == str(out_valid.resolve())
    assert window._guided_output_path_edit.text() == str(run_dir)


def test_guided_output_policy_clear_affects_only_current(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    _load_preview_completed_run(window, run_a, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(out_a))
    window._guided_output_apply_btn.click()

    _load_preview_completed_run(window, run_b, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    window._guided_output_path_edit.setText(str(out_b))
    window._guided_output_apply_btn.click()

    window._guided_output_clear_btn.click()
    assert window._build_guided_draft_run_plan()[0].output_policy.output_root is None
    assert window._guided_output_path_edit.text() == ""

    _load_preview_completed_run(window, run_a, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    assert window._build_guided_draft_run_plan()[0].output_policy.output_root == str(out_a.resolve())




def test_gui_readiness_summary_default(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    summary = window._guided_plan_readiness_summary_label.text()
    assert "Configured: source" in summary
    assert "ROI correction strategies" in summary
    assert "feature/event profile" in summary
    assert "output destination" in summary
    assert "Blocked: execution intentionally unavailable" in summary
    assert "Files written: none" in summary


def test_gui_readiness_summary_updates_on_mark(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    # Select strategy
    idx = window._guided_confirm_strategy_combo.findData("robust_global_event_reject")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    summary = window._guided_plan_readiness_summary_label.text()
    assert "1 ROI correction strategy" in summary
    assert "feature/event profile" in summary
    assert "output destination" in summary
    assert "Blocked: execution intentionally unavailable" in summary


def test_gui_readiness_summary_updates_on_profile(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._guided_feature_event_polarity_combo.setCurrentText("positive")
    window._guided_feature_event_peak_method_combo.setCurrentText("mean_std")
    window._guided_feature_event_peak_k_edit.setText("3.0")
    window._guided_feature_event_apply_btn.click()

    summary = window._guided_plan_readiness_summary_label.text()
    assert "feature/event profile" in summary
    assert "ROI correction strategies" in summary
    assert "output destination" in summary
    assert "Blocked: execution intentionally unavailable" in summary


def test_gui_readiness_summary_updates_on_output(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    out_dest = tmp_path / "future_gui_out"
    window._guided_output_path_edit.setText(str(out_dest))
    window._guided_output_apply_btn.click()

    summary = window._guided_plan_readiness_summary_label.text()
    assert "output destination" in summary
    assert "ROI correction strategies" in summary
    assert "feature/event profile" in summary
    assert "Blocked: execution intentionally unavailable" in summary
    assert not out_dest.exists()


def test_gui_readiness_summary_full_and_non_output_guarantee(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    # 1. Mark Strategy
    idx = window._guided_confirm_strategy_combo.findData("robust_global_event_reject")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # 2. Apply Profile
    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._guided_feature_event_polarity_combo.setCurrentText("positive")
    window._guided_feature_event_peak_method_combo.setCurrentText("mean_std")
    window._guided_feature_event_peak_k_edit.setText("3.0")
    window._guided_feature_event_apply_btn.click()

    # 3. Apply Output Destination
    out_dest = tmp_path / "future_gui_out_full"
    window._guided_output_path_edit.setText(str(out_dest))
    window._guided_output_apply_btn.click()

    summary = window._guided_plan_readiness_summary_label.text()
    assert "Configured: source; 1 ROI correction strategy; feature/event profile; output destination" in summary
    assert "Missing: none" in summary
    assert "Blocked: execution intentionally unavailable" in summary
    assert "Files written: none" in summary

    # Ensure no files/folders were created
    assert not out_dest.exists()
    after_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after_files == before_files


def test_gui_readiness_summary_source_switching(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    out_a = tmp_path / "out_a"

    # Load Run A and configure output policy
    _load_preview_completed_run(window, run_a, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(out_a))
    window._guided_output_apply_btn.click()

    summary_a = window._guided_plan_readiness_summary_label.text()
    assert "output destination" in summary_a.split("Missing:")[0]

    # Load Run B and check summary (should reset/report missing output destination)
    _load_preview_completed_run(window, run_b, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    summary_b = window._guided_plan_readiness_summary_label.text()

    assert "output destination" not in summary_b.split("Missing:")[0]
    assert "output destination" in summary_b.split("Missing:")[1]

    # Re-load Run A and verify output destination is configured again
    _load_preview_completed_run(window, run_a, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    summary_a_restored = window._guided_plan_readiness_summary_label.text()
    assert "output destination" in summary_a_restored.split("Missing:")[0]


def test_gui_readiness_summary_unsaved_widget_edits_ignored(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # Type/edit but do not click Apply
    window._guided_output_path_edit.setText(str(tmp_path / "unsaved_out"))
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")

    # Refresh panel
    window._refresh_guided_confirm_strategy_panel()

    summary = window._guided_plan_readiness_summary_label.text()
    assert "output destination" not in summary.split("Missing:")[0]
    assert "output destination" in summary.split("Missing:")[1]
    assert "feature/event profile" in summary.split("Missing:")[1]


def test_gui_readiness_summary_invalid_apply_does_not_make_configured(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # Apply invalid output destination (same as completed run)
    window._guided_output_path_edit.setText(str(run_dir))
    window._guided_output_apply_btn.click()

    summary = window._guided_plan_readiness_summary_label.text()
    assert "output destination" in summary.split("Missing:")[1]
    assert "Problems" not in summary


def test_guided_output_policy_non_output_guarantee(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_run_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))

    out_dest = tmp_path / "future_guided_output"
    assert not out_dest.exists()

    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # 4. Assert proposed output root does not exist before Apply
    assert not out_dest.exists()

    # 5. Apply the proposed output root
    window._guided_output_path_edit.setText(str(out_dest))
    window._guided_output_apply_btn.click()

    # 6. Assert proposed output root still does not exist
    assert not out_dest.exists()

    # 7. Assert no production files were created
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    assert not any(p.name.startswith("guided_run_plan") and p.name.endswith(".json") for p in run_dir.rglob("*"))

    # 8. Attempt invalid Apply (completed run dir)
    window._guided_output_path_edit.setText(str(run_dir))
    window._guided_output_apply_btn.click()

    # 9. Clear the output policy
    window._guided_output_clear_btn.click()

    # 10. Assert proposed output root still does not exist
    assert not out_dest.exists()

    # 11. Assert completed-run file tree is unchanged
    after_run_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after_run_files == before_run_files

    # 12. Assert no other directories or exports exist
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    assert not any(p.name.startswith("guided_run_plan") and p.name.endswith(".json") for p in run_dir.rglob("*"))


def test_gui_export_no_file_by_default(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    assert window._guided_export_status_label.text() == "No export performed yet."
    assert not any(p.name.startswith("guided_run_plan") and p.name.endswith(".json") for p in run_dir.rglob("*"))


def test_gui_export_incomplete_but_valid_plan(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    export_file = tmp_path / "plan.json"
    assert not export_file.exists()

    window._guided_export_path_edit.setText(str(export_file))
    window._guided_export_btn.click()

    assert export_file.exists()
    import json
    from photometry_pipeline.guided_run_plan import deserialize_plan_from_dict, validate_plan_contract
    with open(export_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["schema_version"] == "guided_run_plan.v1"

    restored = deserialize_plan_from_dict(payload)
    assert validate_plan_contract(restored) == []
    assert restored.roi_plan == []
    assert restored.output_policy.output_root is None


def test_gui_export_fully_configured_plan(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    # 1. Mark Strategy
    idx = window._guided_confirm_strategy_combo.findData("robust_global_event_reject")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # 2. Apply Profile
    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._guided_feature_event_polarity_combo.setCurrentText("positive")
    window._guided_feature_event_peak_method_combo.setCurrentText("mean_std")
    window._guided_feature_event_peak_k_edit.setText("3.0")
    window._guided_feature_event_apply_btn.click()

    # 3. Apply Output Policy
    out_dest = tmp_path / "future_output"
    window._guided_output_path_edit.setText(str(out_dest))
    window._guided_output_apply_btn.click()

    # 4. Export
    export_file = tmp_path / "full_plan.json"
    window._guided_export_path_edit.setText(str(export_file))
    window._guided_export_btn.click()

    assert export_file.exists()
    import json
    from photometry_pipeline.guided_run_plan import deserialize_plan_from_dict, validate_plan_contract
    with open(export_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    restored = deserialize_plan_from_dict(payload)
    assert validate_plan_contract(restored) == []
    assert len(restored.roi_plan) == 1
    assert restored.roi_plan[0].roi == "CH1"
    assert restored.roi_plan[0].correction_strategy.strategy == "robust_global_event_reject"
    assert len(restored.feature_event_profiles) == 1
    assert restored.output_policy.output_root == str(out_dest.resolve())
    assert not out_dest.exists()


def test_gui_export_empty_path_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_export_path_edit.setText("    ")
    window._guided_export_btn.click()

    assert "Export failed: Export path cannot be empty." in window._guided_export_status_label.text()


def test_gui_export_non_json_suffix_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_export_path_edit.setText(str(tmp_path / "plan.txt"))
    window._guided_export_btn.click()

    assert "Export failed: Export path must have a .json suffix." in window._guided_export_status_label.text()


def test_gui_export_existing_file_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    existing_file = tmp_path / "existing.json"
    with open(existing_file, "w", encoding="utf-8") as f:
        f.write("original content")

    window._guided_export_path_edit.setText(str(existing_file))
    window._guided_export_btn.click()

    with open(existing_file, "r", encoding="utf-8") as f:
        assert f.read() == "original content"
    assert "Export failed: Export path already exists." in window._guided_export_status_label.text()


def test_gui_export_missing_parent_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    non_existent_dir = tmp_path / "no_such_folder"
    export_file = non_existent_dir / "plan.json"

    window._guided_export_path_edit.setText(str(export_file))
    window._guided_export_btn.click()

    assert not non_existent_dir.exists()
    assert not export_file.exists()
    assert "Export failed: Parent directory of export path does not exist." in window._guided_export_status_label.text()


def test_gui_export_completed_run_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    window._guided_export_path_edit.setText(str(run_dir))
    window._guided_export_btn.click()
    assert "Export failed: Export path must have a .json suffix." in window._guided_export_status_label.text()

    sub_path = run_dir / "plan.json"
    window._guided_export_path_edit.setText(str(sub_path))
    window._guided_export_btn.click()
    assert "Export failed: Export path cannot be inside the completed run directory." in window._guided_export_status_label.text()
    assert not sub_path.exists()


def test_gui_export_legacy_paths_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    legacy_subpaths = [
        "_analysis",
        "_analysis/phasic_out",
        "_analysis/phasic_out/features",
        "_analysis/phasic_out/applied_dff",
    ]
    for sub in legacy_subpaths:
        legacy_path = run_dir / sub
        window._guided_export_path_edit.setText(str(legacy_path / "plan.json"))
        window._guided_export_btn.click()
        assert "Export path cannot be inside legacy output directories." in window._guided_export_status_label.text()
        assert not (legacy_path / "plan.json").exists()


def test_gui_export_source_switching_clears_path(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    export_a = tmp_path / "plan_a.json"

    _load_preview_completed_run(window, run_a, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_export_path_edit.setText(str(export_a))

    window._refresh_guided_confirm_strategy_panel()
    assert window._guided_export_path_edit.text() == str(export_a)

    _load_preview_completed_run(window, run_b, monkeypatch)
    window._refresh_guided_confirm_strategy_panel()
    assert window._guided_export_path_edit.text() == ""


def test_gui_export_path_does_not_affect_output_policy(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    export_path = tmp_path / "plan.json"
    window._guided_export_path_edit.setText(str(export_path))
    window._guided_export_btn.click()

    plan, _ = window._build_guided_draft_run_plan()
    assert plan.output_policy.output_root is None


def test_gui_export_contract_invalid_plan_rejected(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    from photometry_pipeline.guided_run_plan import OutputPolicy
    monkeypatch.setattr(
        window,
        "_guided_output_policy_for_current_run",
        lambda: OutputPolicy(output_root=str(run_dir), separate_from_source_required=False)
    )

    export_path = tmp_path / "invalid_plan.json"
    window._guided_export_path_edit.setText(str(export_path))
    window._guided_export_btn.click()

    assert not export_path.exists()
    assert "Export failed due to plan contract validation errors:" in window._guided_export_status_label.text()


def test_gui_export_non_production_output_guarantee(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    export_file = tmp_path / "plan.json"
    window._guided_export_path_edit.setText(str(export_file))
    window._guided_export_btn.click()

    assert export_file.exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    after_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after_files == before_files


def test_gui_export_resolved_path_writing(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    export_path_raw = "  " + str(tmp_path / "." / "plan.json") + "  "
    resolved_path = tmp_path.resolve() / "plan.json"

    window._guided_export_path_edit.setText(export_path_raw)
    window._guided_export_btn.click()

    assert resolved_path.exists()
    assert str(resolved_path) in window._guided_export_status_label.text()
    assert "Draft plan successfully exported to:" in window._guided_export_status_label.text()


def test_gui_export_existing_file_resolved_rejection(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    resolved_path = tmp_path.resolve() / "plan_existing.json"
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write("should not be overwritten")

    export_path_raw = "  " + str(tmp_path / "." / "plan_existing.json") + "  "
    window._guided_export_path_edit.setText(export_path_raw)
    window._guided_export_btn.click()

    with open(resolved_path, "r", encoding="utf-8") as f:
        assert f.read() == "should not be overwritten"
    assert "Export failed: Export path already exists." in window._guided_export_status_label.text()


def test_gui_export_missing_parent_resolved_rejection(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    missing_dir = tmp_path / "missing_parent"
    export_path = missing_dir / "plan.json"

    window._guided_export_path_edit.setText(str(export_path))
    window._guided_export_btn.click()

    assert not missing_dir.exists()
    assert not export_path.exists()
    assert "Export failed: Parent directory of export path does not exist." in window._guided_export_status_label.text()


def test_gui_stepper_order_has_feature_detection_before_review_plan():
    steps = list(GUIDED_WORKFLOW_STEPS)
    assert "Draft plan" in steps
    assert "Diagnostics" not in steps
    assert "Confirm strategy" not in steps
    confirm_idx = steps.index("Correction approach")
    feature_idx = steps.index("Feature detection")
    draft_idx = steps.index("Draft plan")
    assert feature_idx == confirm_idx + 1
    assert draft_idx == feature_idx + 1


def _configure_correction_to_draft_gate(
    window, monkeypatch, strategies, *, invalid_signal_evidence=False
):
    included = list(strategies)
    window._set_guided_workflow_mode("new_analysis")
    window._discovery_cache = {"rois": [{"roi_id": roi} for roi in included]}
    for roi_list in (window._roi_list, window._guided_roi_list):
        roi_list.clear()
        for roi in included:
            roi_list.addItem(roi)
            roi_list.item(roi_list.count() - 1).setCheckState(Qt.Checked)
    monkeypatch.setattr(
        window,
        "_refresh_guided_local_preview_choice_currency",
        lambda: None,
    )
    references = {}
    choices = {}
    for roi, strategy in strategies.items():
        reference = {
            "evidence_source_type": (
                "local_preview"
                if strategy == "signal_only_f0"
                else "local_correction_preview"
            ),
            "preview_only": True,
            "production_analysis": False,
            "roi": roi,
            "roi_id": roi,
            "preview_id": f"preview-{roi}",
            "selected_strategy": strategy,
        }
        if strategy == "signal_only_f0":
            reference.update(
                strategy_family="signal_only_f0",
                dynamic_fit_mode=None,
                valid=not invalid_signal_evidence,
                current_or_stale="current",
            )
        references[(roi, strategy)] = reference
        choices[("local_correction_preview", roi)] = {
            "strategy": strategy,
            "strategy_family": (
                "signal_only_f0"
                if strategy == "signal_only_f0"
                else "dynamic_fit"
            ),
            "dynamic_fit_mode": (
                None if strategy == "signal_only_f0" else strategy
            ),
            "confirmed": True,
            "choice_source": "explicit_user_mark",
            "source_type": "local_correction_preview",
            "current": True,
            "stale": False,
            "roi": roi,
            "local_preview_evidence": reference,
        }
    window._guided_strategy_choices = choices
    monkeypatch.setattr(
        window,
        "_guided_local_preview_evidence_reference",
        lambda strategy, *, roi=None: references.get((roi, strategy)),
    )
    window._refresh_guided_correction_continue_state()
    return choices


def test_guided_correction_to_draft_plan_starts_locked(window):
    window._set_guided_workflow_mode("new_analysis")
    correction_index = list(GUIDED_WORKFLOW_STEPS).index(
        "Correction approach"
    )
    draft_index = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    window._reach_guided_step("Correction approach")
    window._guided_workflow_stepper.setCurrentRow(correction_index)

    assert not (
        window._guided_workflow_stepper.item(draft_index).flags()
        & Qt.ItemIsEnabled
    )
    draft_item = window._guided_workflow_stepper.item(draft_index)
    item_rect = window._guided_workflow_stepper.visualItemRect(draft_item)
    QTest.mouseClick(
        window._guided_workflow_stepper.viewport(),
        Qt.LeftButton,
        pos=item_rect.center(),
    )
    assert window._guided_workflow_stepper.currentRow() == correction_index
    assert window._guided_correction_continue_btn.isEnabled() is False
    assert window._guided_correction_continue_status.text() == (
        "Confirm correction strategies for all included ROIs to continue."
    )


def test_guided_correction_gate_ready_unlocks_only_feature_detection(
    window, monkeypatch
):
    _configure_correction_to_draft_gate(
        window,
        monkeypatch,
        {
            "CH1": "robust_global_event_reject",
            "CH2": "robust_global_event_reject",
        },
    )

    assert window._guided_correction_continue_btn.isEnabled() is True
    assert window._guided_correction_continue_status.text() == (
        "Correction strategies are ready. Continue to Feature Detection."
    )
    assert window._guided_correction_continue_btn.text() == (
        "Continue to Feature Detection"
    )
    window._guided_correction_continue_btn.click()

    feature_index = list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    draft_index = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    run_index = list(GUIDED_WORKFLOW_STEPS).index("Run")
    review_index = list(GUIDED_WORKFLOW_STEPS).index("Review")
    assert window._guided_workflow_stepper.currentRow() == feature_index
    assert (
        window._guided_workflow_stepper.item(feature_index).flags()
        & Qt.ItemIsEnabled
    )
    assert not (
        window._guided_workflow_stepper.item(draft_index).flags()
        & Qt.ItemIsEnabled
    )
    assert not (
        window._guided_workflow_stepper.item(run_index).flags()
        & Qt.ItemIsEnabled
    )
    assert not (
        window._guided_workflow_stepper.item(review_index).flags()
        & Qt.ItemIsEnabled
    )


def test_guided_correction_gate_requires_reconfirmation_after_change_or_stale(
    window, monkeypatch
):
    choices = _configure_correction_to_draft_gate(
        window,
        monkeypatch,
        {"CH1": "robust_global_event_reject"},
    )
    choice = choices[("local_correction_preview", "CH1")]
    choice.update(
        current=False,
        stale=True,
        stale_reason="Selection changed; confirm the new strategy.",
    )
    window._refresh_guided_correction_continue_state()
    assert window._guided_correction_continue_btn.isEnabled() is False
    assert window._guided_correction_continue_status.text() == (
        "Review and confirm changed correction strategies before continuing."
    )

    choice["stale_reason"] = "Correction-relevant setup changed."
    window._refresh_guided_correction_continue_state()
    assert window._guided_correction_continue_status.text() == (
        "Regenerate the correction preview and reconfirm stale strategies "
        "before continuing."
    )

    choice.update(current=True, stale=False, stale_reason="")
    window._refresh_guided_correction_continue_state()
    assert window._guided_correction_continue_btn.isEnabled() is True


@pytest.mark.parametrize(
    ("strategies", "invalid_signal", "ready", "status"),
    [
        (
            {
                "CH1": "robust_global_event_reject",
                "CH2": "signal_only_f0",
            },
            False,
            True,
            "Correction strategies are ready. Continue to Feature Detection.",
        ),
        (
            {"CH1": "signal_only_f0", "CH2": "signal_only_f0"},
            False,
            True,
            "Correction strategies are ready. Continue to Feature Detection.",
        ),
        (
            {
                "CH1": "robust_global_event_reject",
                "CH2": "global_linear_regression",
            },
            False,
            True,
            "Correction strategies are ready. Continue to Feature Detection.",
        ),
        (
            {
                "CH1": "adaptive_event_gated_regression",
                "CH2": "global_linear_regression",
                "CH3": "signal_only_f0",
            },
            False,
            True,
            "Correction strategies are ready. Continue to Feature Detection.",
        ),
        (
            {
                "CH1": "robust_global_event_reject",
                "CH2": "signal_only_f0",
            },
            True,
            False,
            "Signal-Only F0 choices require current local preview evidence.",
        ),
    ],
)
def test_guided_correction_gate_strategy_contracts(
    window,
    monkeypatch,
    strategies,
    invalid_signal,
    ready,
    status,
):
    _configure_correction_to_draft_gate(
        window,
        monkeypatch,
        strategies,
        invalid_signal_evidence=invalid_signal,
    )

    assert window._guided_correction_continue_btn.isEnabled() is ready
    assert window._guided_correction_continue_status.text() == status


def test_gui_merged_correction_page_contains_existing_workflow_controls(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    correction_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )

    assert correction_widget.findChild(
        QWidget, "guidedDiagnosticCacheBuildButton"
    ) is not None
    assert correction_widget.findChild(
        QWidget, "guidedCorrectionPreviewGenerateButton"
    ) is not None
    assert correction_widget.findChild(QWidget, "guidedConfirmStrategyRoiCombo") is not None
    assert correction_widget.findChild(QWidget, "guidedConfirmStrategyChunkCombo") is not None
    assert correction_widget.findChild(QWidget, "guidedConfirmStrategyChoiceCombo") is not None
    assert window._guided_confirm_ack_cb.isVisible() is False
    assert correction_widget.findChild(QWidget, "guidedConfirmStrategyMarkButton") is not None

    assert correction_widget.findChild(QWidget, "guidedFeatureEventProfileEditorPanel") is None
    assert correction_widget.findChild(QWidget, "guidedOutputDestinationPanel") is None
    assert correction_widget.findChild(QWidget, "guidedDraftPlanExportPanel") is None
    assert correction_widget.findChild(QWidget, "guidedPlanReadinessSummaryPanel") is None
    assert correction_widget.findChild(QWidget, "guidedDraftRunPlanPreviewPanel") is None
    assert correction_widget.findChild(QWidget, "guidedDraftRunPlanChecklistPanel") is None


def test_gui_merged_correction_page_makes_local_preview_primary_and_user_safe(window):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    correction_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    actions = correction_widget.findChild(
        QWidget, "guidedDiagnosticsActionsSection"
    )
    assert actions is not None
    actions_margins = actions.layout().contentsMargins()
    assert actions_margins.left() == 0
    assert actions_margins.right() == 0
    group_titles = [
        group.title()
        for group in correction_widget.findChildren(QGroupBox)
        if group.isVisibleTo(correction_widget)
    ]
    assert "Optional: prepare reusable full correction evidence" not in group_titles
    assert "1. Preview correction methods" in group_titles
    assert "2. Choose correction strategy" in group_titles
    assert window._guided_full_evidence_group.isVisible() is False
    assert window._guided_diagnostic_cache_build_btn.isVisible() is False
    assert "Choose a correction candidate" not in group_titles

    labels = [
        label.text()
        for label in correction_widget.findChildren(QLabel)
        if label.isVisibleTo(correction_widget)
    ]
    buttons = [
        button.text()
        for button in correction_widget.findChildren(QPushButton)
        if button.isVisibleTo(correction_widget)
    ]
    visible_text = " ".join(group_titles + labels + buttons)
    assert "Setup intent only; not wired to execution." not in visible_text
    assert "Compare correction methods and choose one strategy" in (
        visible_text
    )
    assert "Optional: prepare reusable full correction evidence" not in visible_text
    assert "Correction evidence: not built" not in visible_text
    assert "reusable full correction evidence" not in visible_text
    assert "Build correction evidence" not in visible_text
    assert "Generate correction preview" not in visible_text
    assert "Complete Select data, Recording structure" in visible_text
    assert "Locked until a correction preview is generated" in visible_text
    assert "I reviewed" not in visible_text
    assert "Mark strategy choice" not in visible_text
    prohibited = (
        "diagnostic cache",
        "evidence chunk",
        "artifact",
        "backend validator",
        "manifest",
        "materialization",
        "contract snapshot",
        "run spec",
        "startup",
        "wrapper",
        "completed-run cache",
        "open results",
    )
    lowered = visible_text.lower()
    found = [term for term in prohibited if term in lowered]
    assert found == []


def test_compact_correction_sections_unlock_in_order(
    window, tmp_path, monkeypatch
):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    assert window._guided_preview_locked_label.isHidden() is False
    assert window._guided_preview_roi_combo.isHidden() is True
    assert window._guided_confirm_locked_label.isHidden() is False
    assert window._guided_confirm_strategy_combo.parentWidget().isHidden()
    assert window._guided_diagnostic_cache_build_btn.isEnabled() is False
    assert window._guided_correction_next_action_label.text() != (
        "Next step: build correction evidence."
    )

    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)

    assert window._guided_preview_locked_label.isHidden() is True
    assert window._guided_preview_roi_combo.isHidden() is False
    assert window._guided_preview_chunk_combo.isHidden() is False
    assert window._guided_preview_generate_btn.isHidden() is False
    assert window._guided_confirm_locked_label.isHidden() is False
    assert window._guided_confirm_strategy_combo.parentWidget().isHidden()
    assert window._guided_diagnostic_cache_build_btn.text() == (
        "Rebuild correction evidence"
    )
    assert "Next step: generate a correction preview" in (
        window._guided_correction_next_action_label.text()
    )

    _generate_ready_guided_correction_preview(window)

    assert window._guided_confirm_locked_label.isHidden() is True
    assert window._guided_confirm_roi_combo.parentWidget().isHidden() is False
    assert window._guided_confirm_strategy_combo.parentWidget().isHidden() is False
    assert "Next step: confirm a method" in (
        window._guided_correction_next_action_label.text()
    )


def test_signal_only_f0_is_one_preview_subsection_with_plain_limit(window):
    window._set_guided_workflow_mode("new_analysis")
    correction_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    signal_groups = [
        group
        for group in correction_widget.findChildren(QGroupBox)
        if group.title() == "Signal-Only F0 preview"
    ]
    assert len(signal_groups) == 1
    assert signal_groups[0].isAncestorOf(
        window._guided_signal_f0_generate_btn
    )
    assert "Guided Run cannot execute it yet" in (
        window._guided_signal_f0_status_label.parentWidget().findChild(
            QLabel, "guidedSignalOnlyF0DiagnosticIntro"
        ).text()
    )


def test_hidden_method_configuration_never_becomes_top_level(window, qapp):
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._refresh_guided_mode_display()
    qapp.processEvents()

    group = window._guided_method_configuration_group
    assert group.parentWidget() is not None
    assert group.isWindow() is False
    assert group.isHidden() is True
    assert group not in QApplication.topLevelWidgets()
    assert "Correction approach" not in window._guided_raw_setup_controls
    assert not any(
        isinstance(widget, QGroupBox)
        and widget.title() == "Correction method reference (preview only)"
        for widget in QApplication.topLevelWidgets()
    )


def test_correction_approach_page_avoids_developer_facing_wording(window, qapp):
    """4J16k26: Correction Approach must read as preview -> compare -> choose,
    not as internal run-method-configuration/diagnostic-cache staging."""
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._refresh_guided_mode_display()
    qapp.processEvents()

    correction_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )

    # Scan every group title, label, and button on this step -- including
    # currently-hidden widgets (e.g. the reference method cards) -- plus
    # their tooltips, so removed wording cannot quietly reappear later.
    texts = []
    for widget in correction_widget.findChildren(QGroupBox):
        texts.append(widget.title())
        texts.append(widget.toolTip())
    for widget in correction_widget.findChildren(QLabel):
        texts.append(widget.text())
        texts.append(widget.toolTip())
    for widget in correction_widget.findChildren(QPushButton):
        texts.append(widget.text())
        texts.append(widget.toolTip())
    visible_text = " ".join(t for t in texts if t)
    lowered = visible_text.lower()

    forbidden = (
        "run method configuration",
        "diagnostic intent",
        "diagnostic-cache identity",
        "cache identity",
    )
    found = [term for term in forbidden if term in lowered]
    assert found == []

    # The reference-method cards should read as preview affordances.
    assert "Preview this correction method" in visible_text
    assert "Correction method reference (preview only)" in visible_text


def _guided_step_visible_texts(window, step_name: str) -> tuple[str, str]:
    """Return (raw_joined_text, lowercased_text) for every group title,
    label, and button (including tooltips) on the given Guided step."""
    step_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index(step_name)
    )
    texts = []
    for widget in step_widget.findChildren(QGroupBox):
        texts.append(widget.title())
        texts.append(widget.toolTip())
    for widget in step_widget.findChildren(QLabel):
        texts.append(widget.text())
        texts.append(widget.toolTip())
    for widget in step_widget.findChildren(QPushButton):
        texts.append(widget.text())
        texts.append(widget.toolTip())
    visible_text = " ".join(t for t in texts if t)
    return visible_text, visible_text.lower()


def test_select_data_page_avoids_developer_facing_wording(window, qapp):
    """4J16k27: Select Data must read as plain instructions, not as Full
    Control state-mirroring, and must not show 'Resolved format'/'Preview: N'."""
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Select data")
    )
    qapp.processEvents()

    visible_text, lowered = _guided_step_visible_texts(window, "Select data")

    forbidden = (
        "full control",
        "shared discovery state",
        "existing discovery path",
        "mirror the existing setup state",
        "resolved format",
        "preview:",
    )
    found = [term for term in forbidden if term in lowered]
    assert found == []

    assert "Choose the input folder" in visible_text
    assert "Select ROIs scans the input folder" in visible_text
    assert "detected this format automatically" in visible_text
    assert "Review Plan page" in visible_text


def test_select_data_roi_discovery_status_is_plain_language(window):
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": 581,
        "n_preview": 581,
        "sessions": [{"session_id": f"s{i}"} for i in range(3)],
        "rois": [{"roi_id": "CH1"}],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    status = window._guided_discovery_summary_label.text()
    assert status == (
        "Detected RWD data with 581 recording sessions. Channels detected "
        "below."
    )
    assert "Preview:" not in status
    assert "Format:" not in status
    assert "|" not in status


def test_recording_structure_page_avoids_developer_facing_wording(window, qapp):
    """4J16k27: Recording Structure must read as plain instructions, not as
    Full Control state-mirroring or continuous-window implementation detail."""
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Recording structure")
    )
    qapp.processEvents()

    visible_text, lowered = _guided_step_visible_texts(
        window, "Recording structure"
    )

    forbidden = (
        "full control",
        "mirror the acquisition and timing state",
        "non-overlapping continuous windows",
        "continuous step remains derived",
        "chunks occur per hour",
        "final rwd recording",
    )
    found = [term for term in forbidden if term in lowered]
    assert found == []

    assert "align sessions correctly" in visible_text
    assert "Final recording session" in visible_text
    assert "Sessions per hour:" in visible_text
    # QCheckBox text isn't covered by the generic label/button/group scan
    # above; check it directly.
    assert window._guided_exclude_incomplete_final_rwd_chunk_cb.text() == (
        "Allow one incomplete final recording session"
    )
    assert "Sessions per hour (required)" not in visible_text


def test_start_page_labels_carried_over_input_folder_before_new_analysis(
    window, tmp_path
):
    """4J16k27: a folder restored from a previous session must not look like
    an already-configured new analysis until the user actually starts one."""
    prior_dir = tmp_path / "prior_session_input"
    prior_dir.mkdir()
    window._input_dir.setText(str(prior_dir))
    assert window._guided_workflow_mode == "start"
    window._refresh_guided_start_panel()

    status = window._guided_start_setup_status_label.text()
    assert status.startswith("Last input folder used:")
    assert "Input folder: " + str(prior_dir) not in status

    window._on_guided_start_setup_new_analysis()
    window._refresh_guided_start_panel()
    status_after = window._guided_start_setup_status_label.text()
    assert status_after.startswith("Input folder:")
    assert "Last input folder" not in status_after


def test_guided_mode_banner_hides_completed_results_during_new_analysis_setup(
    window, tmp_path
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    window._input_dir.setText(str(input_dir))
    window._set_guided_workflow_mode("new_analysis")

    banner_text = window._guided_mode_banner_label.text()
    assert banner_text == f"New analysis using: {input_dir.name}"
    assert "Completed results:" not in banner_text
    assert "Mode:" not in banner_text
    assert window._guided_mode_banner_label.toolTip() == (
        f"New analysis using: {input_dir}"
    )


def test_status_strip_hidden_while_guided_idle_visible_in_full_control(window, qapp):
    """4J16k28/4J16k30: the Full-Control-only run status/phase/elapsed
    labels, the milestone progress bar, the pane-toggle buttons, and (as of
    4J16k30) the status header card itself should not be visible while the
    Guided Workflow tab is active and idle with no PREVIEW badge showing --
    Guided Run has its own Run-step "Analysis progress" panel, Guided has
    no left/right splitter for the pane-toggle buttons to control, and an
    empty rounded card with no content is a visual artifact, not useful
    chrome. Everything must still be fully present and visible in Full
    Control."""
    strip_widgets = (
        window._status_label,
        window._phase_label,
        window._elapsed_label,
        window._left_pane_toggle_btn,
        window._right_pane_toggle_btn,
        window._progress_caption_label,
        window._progress_bar,
    )

    # Default state: Guided tab active, idle, no PREVIEW badge.
    assert window._workflow_mode_tabs.currentWidget() is window._guided_workflow_tab
    assert window._ui_state == main_window_module.RunnerState.IDLE
    assert window._preview_badge.isHidden() is True
    window._refresh_status_strip_visibility()
    qapp.processEvents()
    for widget in strip_widgets:
        assert widget.isHidden() is True
    # 4J16k30: with no useful content left inside it, the card itself is
    # now hidden too -- not just its children -- so no empty rounded
    # rectangle remains under the Tools menu.
    assert window._status_header_card.isHidden() is True

    # Switching to Full Control must restore all of them, including the card.
    window._workflow_mode_tabs.setCurrentWidget(window._full_control_tab)
    qapp.processEvents()
    assert window._status_header_card.isHidden() is False
    for widget in strip_widgets:
        assert widget.isHidden() is False


def test_status_strip_stays_visible_during_active_full_control_run_even_on_guided_tab(
    window, qapp
):
    """4J16k28/4J16k30: an active Full Control run/validation must never be
    hidden -- including the card itself -- even if the user is looking at
    the Guided tab."""
    assert window._workflow_mode_tabs.currentWidget() is window._guided_workflow_tab
    window._ui_state = main_window_module.RunnerState.RUNNING
    window._refresh_status_strip_visibility()
    qapp.processEvents()

    assert window._status_header_card.isHidden() is False
    assert window._status_label.isHidden() is False
    assert window._progress_bar.isHidden() is False
    assert window._left_pane_toggle_btn.isHidden() is False
    assert window._right_pane_toggle_btn.isHidden() is False

    window._ui_state = main_window_module.RunnerState.IDLE
    window._refresh_status_strip_visibility()
    assert window._status_header_card.isHidden() is True
    assert window._status_label.isHidden() is True


def test_status_strip_card_stays_visible_for_preview_badge_during_idle_guided(
    window, qapp
):
    """4J16k30: if the PREVIEW badge is showing (a loaded completed run was
    a preview-type run), the card must not be hidden out from under it,
    even while the Guided tab is idle -- that badge is information the
    user still needs. The other Full-Control-only children (run status,
    progress bar, pane-toggle buttons) remain hidden since they are still
    not meaningful in Guided setup."""
    assert window._workflow_mode_tabs.currentWidget() is window._guided_workflow_tab
    assert window._ui_state == main_window_module.RunnerState.IDLE

    window._preview_badge.show()
    window._refresh_status_strip_visibility()
    qapp.processEvents()

    assert window._status_header_card.isHidden() is False
    assert window._preview_badge.isHidden() is False
    assert window._status_label.isHidden() is True
    assert window._progress_bar.isHidden() is True
    assert window._left_pane_toggle_btn.isHidden() is True

    # Once the badge goes away again, the card goes back to fully hidden.
    window._preview_badge.hide()
    window._refresh_status_strip_visibility()
    assert window._status_header_card.isHidden() is True


def test_guided_run_live_status_panel_unaffected_by_status_strip_visibility(
    window, qapp
):
    """4J16k30: the Guided Run step's own "Analysis progress" panel is a
    separate widget tree driven by a separate StatusFollower, not by
    self._ui_state, so hiding/showing the top status card and its children
    must never touch it."""
    assert window._workflow_mode_tabs.currentWidget() is window._guided_workflow_tab
    window._guided_run_live_status_group.setVisible(True)
    window._guided_run_live_status_label.setText(
        "Analysis is running. Do not close this window."
    )

    for ui_state in (
        main_window_module.RunnerState.IDLE,
        main_window_module.RunnerState.RUNNING,
        main_window_module.RunnerState.SUCCESS,
    ):
        window._ui_state = ui_state
        window._refresh_status_strip_visibility()
        qapp.processEvents()
        assert window._guided_run_live_status_group.isHidden() is False
        assert window._guided_run_live_status_label.text() == (
            "Analysis is running. Do not close this window."
        )

    window._workflow_mode_tabs.setCurrentWidget(window._full_control_tab)
    qapp.processEvents()
    assert window._guided_run_live_status_group.isHidden() is False
    window._workflow_mode_tabs.setCurrentWidget(window._guided_workflow_tab)


def _setup_guided_local_raw_preview_ready(
    window, tmp_path, monkeypatch, *, n_rois: int = 3
) -> list[Path]:
    """Reach the reachable new-analysis local-raw-segment preview state
    (ROI/segment selected, "Generate correction preview" enabled but not
    yet clicked) without going through the diagnostic-cache path -- this
    is the flow a real Guided user actually exercises."""
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    source_files = []
    for index in range(3):
        session_dir = input_dir / f"2025_01_01-{index:02d}_00_00"
        session_dir.mkdir()
        source_file = session_dir / "fluorescence.csv"
        source_file.write_text(
            f"local preview source {index}", encoding="utf-8"
        )
        source_files.append(source_file)
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._format_combo.setCurrentText("rwd")
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": 3,
        "n_preview": 3,
        "sessions": [
            {
                "index": index,
                "session_id": f"2025_01_01-{index:02d}_00_00",
                "path": str(source_file),
                "included_in_preview": True,
            }
            for index, source_file in enumerate(source_files)
        ],
        "rois": [{"roi_id": f"CH{i + 1}"} for i in range(n_rois)],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)
    time_sec = np.arange(2400, dtype=float) / 20.0
    uv = 1.0 + 0.03 * np.sin(time_sec * 0.15)
    sig = 1.25 * uv + 0.04 * np.sin(time_sec * 0.7)

    def fake_load_chunk(path, input_format, _config, chunk_id):
        return Chunk(
            chunk_id=chunk_id,
            source_file=path,
            format=input_format,
            time_sec=time_sec,
            uv_raw=np.column_stack([uv] * n_rois),
            sig_raw=np.column_stack([sig] * n_rois),
            fs_hz=20.0,
            channel_names=[f"CH{i + 1}" for i in range(n_rois)],
            metadata={},
        )

    monkeypatch.setattr(
        correction_preview_module, "load_chunk", fake_load_chunk
    )
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {
            "csv_path": path,
            "fs_hz": 20.0,
            "chunk_duration_sec": 600.0,
            "time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        },
    )
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._refresh_guided_diagnostics_panel()
    assert window._guided_preview_source_type == "local_raw_segment"
    assert window._guided_preview_generate_btn.isEnabled() is True
    return source_files


def test_local_preview_bypasses_full_evidence_and_unlocks_explicit_confirmation(
    window, tmp_path, monkeypatch
):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    source_files = []
    for index in range(3):
        session_dir = input_dir / f"2025_01_01-{index:02d}_00_00"
        session_dir.mkdir()
        source_file = session_dir / "fluorescence.csv"
        source_file.write_text(
            f"local preview source {index}", encoding="utf-8"
        )
        source_files.append(source_file)
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._format_combo.setCurrentText("rwd")
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": 3,
        "n_preview": 3,
        "sessions": [
            {
                "index": index,
                "session_id": f"2025_01_01-{index:02d}_00_00",
                "path": str(source_file),
                "included_in_preview": True,
            }
            for index, source_file in enumerate(source_files)
        ],
        "rois": [
            {"roi_id": "CH1"},
            {"roi_id": "CH2"},
            {"roi_id": "CH3"},
        ],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)
    time_sec = np.arange(2400, dtype=float) / 20.0
    uv = 1.0 + 0.03 * np.sin(time_sec * 0.15)
    sig = 1.25 * uv + 0.04 * np.sin(time_sec * 0.7)
    loads = []

    def fake_load_chunk(path, input_format, _config, chunk_id):
        loads.append((path, input_format, chunk_id))
        return Chunk(
            chunk_id=chunk_id,
            source_file=path,
            format=input_format,
            time_sec=time_sec,
            uv_raw=np.column_stack((uv, uv, uv)),
            sig_raw=np.column_stack((sig, sig * 1.01, sig * 0.99)),
            fs_hz=20.0,
            channel_names=["CH1", "CH2", "CH3"],
            metadata={},
        )

    monkeypatch.setattr(
        correction_preview_module, "load_chunk", fake_load_chunk
    )
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {
            "csv_path": path,
            "fs_hz": 20.0,
            "chunk_duration_sec": 600.0,
            "time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        },
    )
    runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = runner
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._refresh_guided_diagnostics_panel()

    assert window._guided_preview_source_type == "local_raw_segment"
    assert window._guided_preview_generate_btn.isEnabled() is True
    assert window._guided_preview_generate_btn.isHidden() is False
    assert window._guided_full_evidence_group.isChecked() is False
    assert window._guided_full_evidence_group.isVisible() is False
    assert window._guided_diagnostic_cache_build_btn.isVisible() is False
    assert window._guided_preview_signal_f0_cb.text() == "Signal-Only F0"
    assert window._guided_preview_signal_f0_cb.isChecked() is True
    window._guided_preview_chunk_combo.setCurrentIndex(2)
    window._guided_preview_generate_btn.click()

    result = window._guided_preview_last_result
    assert result["source_type"] == "local_raw_segment"
    assert result["preview_only"] is True
    assert result["production_analysis"] is False
    assert loads == [(str(source_files[2].resolve()), "rwd", 0)]
    assert runner.argv is None
    assert window._guided_diagnostic_cache_record is None
    assert Path(result["visual_preview_path"]).is_file()
    assert result["visual_panel_titles"] == [
        "Source segment",
        "Dynamic-fit corrected signal comparison",
        "Dynamic-fit fitted reference comparison",
        "Signal-Only F0 baseline",
        "Signal-Only F0-corrected dF/F preview",
    ]
    assert result["visual_trace_labels"] == [
        "Raw signal",
        "Reference/control signal",
        "Robust Global Event-Reject Fit",
        "Adaptive Event-Gated Fit",
        "Global Linear Regression",
        "Signal-only F0 baseline",
        "Signal-only dF/F",
    ]
    assert window._guided_preview_visual_label.pixmap().isNull() is False
    provenance = json.loads(
        Path(result["preview_provenance_path"]).read_text(encoding="utf-8")
    )
    assert provenance["local_preview_scope"] == (
        "single_discovered_session_single_roi"
    )
    assert provenance["selected_roi"] == "CH1"
    assert provenance["selected_chunk"] == 2
    assert provenance["selected_segment_index"] == 2
    assert provenance["selected_segment_label"] == "session-2"
    assert provenance["adapter_local_chunk_id"] == 0
    assert provenance["source_file"] == str(source_files[2].resolve())
    assert provenance["production_analysis"] is False
    assert provenance["reference_fit_scope"] == "selected_session"
    assert provenance["effective_config_values"]["target_fs_hz"] == 20.0
    signal_f0_evidence = result["signal_only_f0_preview_evidence"]
    assert signal_f0_evidence["strategy_family"] == "signal_only_f0"
    assert signal_f0_evidence["preview_only"] is True
    assert signal_f0_evidence["production_analysis"] is False
    assert signal_f0_evidence["explicit_user_mark"] is False
    # 4J16k29: Signal-Only F0 is included and valid, so it is shown like
    # any other method in the summary above -- no separate callout.
    assert window._guided_local_signal_f0_preview_label.text() == ""
    assert window._guided_local_signal_f0_preview_label.isHidden() is True
    review_text = window._guided_preview_review_label.text()
    assert "segment session-2" in review_text
    assert "Signal-Only F0" in review_text
    preview_dir = Path(result["preview_output_dir"])
    assert not (preview_dir / "status.json").exists()
    assert not (preview_dir / "run_report.json").exists()
    assert not (preview_dir / "MANIFEST.json").exists()
    assert not (preview_dir / "_analysis").exists()
    assert not (output_dir / "applied_dff").exists()
    assert not list(output_dir.rglob("batch_manifest.csv"))
    assert not list(
        output_dir.rglob("guided_applied_dff_provenance.json")
    )
    assert window._guided_confirm_locked_label.isHidden() is True
    assert window._guided_confirm_selection_group.isHidden() is True
    assert window._guided_confirm_choice_group.isHidden() is True
    assert window._guided_local_preview_confirmation_group.isHidden() is False
    rows = window._guided_local_preview_confirmation_rows
    assert set(rows) == {"CH1", "CH2", "CH3"}
    assert rows["CH1"]["evidence_label"].text() == (
        "Local preview, segment session-2"
    )
    assert rows["CH1"]["strategy_combo"].isEnabled() is True
    assert rows["CH2"]["evidence_label"].text() == "Preview first"
    assert rows["CH2"]["strategy_combo"].isEnabled() is False
    assert (
        rows["CH2"]["strategy_combo"].findData("signal_only_f0")
        == -1
    )
    assert rows["CH2"]["action_button"].isEnabled() is False
    assert rows["CH3"]["evidence_label"].text() == "Preview first"
    assert rows["CH3"]["action_button"].isEnabled() is False
    registered_ch1 = window._guided_local_preview_evidence_by_roi["CH1"]
    locked_signal_f0 = window._guided_local_preview_locked_evidence_for_roi(
        "CH1", "signal_only_f0"
    )
    assert locked_signal_f0 is not None
    assert locked_signal_f0["roi_id"] == "CH1"
    assert locked_signal_f0["strategy_family"] == "signal_only_f0"
    assert locked_signal_f0["selected_strategy"] == "signal_only_f0"
    assert locked_signal_f0["dynamic_fit_mode"] is None
    assert locked_signal_f0["evidence_source_type"] == "local_preview"
    assert locked_signal_f0["preview_only"] is True
    assert locked_signal_f0["production_analysis"] is False
    assert locked_signal_f0["explicit_user_mark"] is False
    assert locked_signal_f0["current_or_stale"] == "current"
    assert locked_signal_f0["locked"] is False
    assert locked_signal_f0["selectable"] is True
    assert locked_signal_f0["metrics"]["sample_count"] == len(time_sec)
    ch1_signal_f0_preview_id = locked_signal_f0["preview_id"]
    locked_signal_f0.update(
        valid=False,
        selectable=False,
        locked=True,
        issues=["invalid denominator"],
    )
    window._rebuild_guided_local_preview_confirmation_rows()
    assert window._guided_local_preview_confirmation_rows["CH1"][
        "strategy_combo"
    ].findData("signal_only_f0") == -1
    assert "0 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    locked_signal_f0.update(
        valid=True,
        selectable=True,
        locked=False,
        issues=[],
    )
    registered_ch1["result"]["method_statuses"][
        "adaptive_event_gated_regression"
    ]["status"] = "failed"
    window._rebuild_guided_local_preview_confirmation_rows()
    ch1_methods = window._guided_local_preview_confirmation_rows["CH1"][
        "strategy_combo"
    ]
    assert ch1_methods.findData("adaptive_event_gated_regression") == -1
    assert ch1_methods.findData("signal_only_f0") >= 0
    assert window._guided_strategy_choices == {}

    window._guided_preview_roi_combo.setCurrentIndex(
        window._guided_preview_roi_combo.findData("CH2")
    )
    assert window._guided_preview_result_stale is True
    # 4J16k29: no separate "stale" callout for Signal-Only F0 -- the label
    # is simply cleared, consistent with the rest of the stale preview.
    assert window._guided_local_signal_f0_preview_label.text() == ""
    assert window._guided_local_signal_f0_preview_label.isHidden() is True
    window._guided_preview_generate_btn.click()
    ch2_result = window._guided_preview_last_result
    assert ch2_result["roi"] == "CH2"
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["evidence_label"].text() == (
        "Local preview, segment session-2"
    )
    assert rows["CH2"]["evidence_label"].text() == (
        "Local preview, segment session-2"
    )
    assert rows["CH3"]["evidence_label"].text() == "Preview first"
    assert rows["CH2"]["strategy_combo"].findData(
        "signal_only_f0"
    ) >= 0
    persisted_signal_f0 = (
        window._guided_local_preview_locked_evidence_for_roi(
            "CH1", "signal_only_f0"
        )
    )
    assert persisted_signal_f0 is not None
    assert persisted_signal_f0["preview_id"] == ch1_signal_f0_preview_id

    ch1_combo = rows["CH1"]["strategy_combo"]
    ch1_strategy_index = ch1_combo.findData(
        "robust_global_event_reject"
    )
    ch1_combo.setCurrentIndex(ch1_strategy_index)
    assert window._guided_strategy_choices == {}
    assert rows["CH1"]["action_button"].isEnabled() is True
    assert rows["CH1"]["action_button"].text() == (
        "Confirm method"
    )
    rows["CH1"]["action_button"].click()
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].isEnabled() is False
    assert rows["CH1"]["strategy_combo"].currentData() == (
        "robust_global_event_reject"
    )
    assert "1 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    ch1_choice = window._guided_strategy_choices[
        ("local_correction_preview", "CH1")
    ]
    ch1_reference = ch1_choice["local_preview_evidence"]
    assert ch1_reference["preview_id"] == result["preview_id"]
    assert ch1_reference["roi"] == "CH1"
    assert ch1_reference["selected_segment_label"] == "session-2"
    assert ch1_reference["selected_segment_index"] == 2
    assert ch1_reference["source_file"] == str(source_files[2].resolve())
    assert ch1_reference["source_file_hash"]
    assert ch1_reference["preview_only"] is True
    assert ch1_reference["production_analysis"] is False

    ch1_combo = rows["CH1"]["strategy_combo"]
    ch1_combo.setCurrentIndex(ch1_combo.findData("signal_only_f0"))
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == (
        "Selection changed. Confirm this method."
    )
    assert rows["CH1"]["action_button"].text() == (
        "Confirm method"
    )
    assert rows["CH1"]["action_button"].isEnabled() is True
    assert rows["CH1"]["strategy_combo"].currentData() == "signal_only_f0"
    assert "0 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    rows["CH1"]["action_button"].click()
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].isEnabled() is False
    assert "1 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )

    ch1_combo = rows["CH1"]["strategy_combo"]
    ch1_combo.setCurrentIndex(
        ch1_combo.findData("global_linear_regression")
    )
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == (
        "Selection changed. Confirm this method."
    )
    assert rows["CH1"]["action_button"].text() == (
        "Confirm method"
    )
    assert rows["CH1"]["action_button"].isEnabled() is True
    assert "0 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    rows["CH1"]["action_button"].click()
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].isEnabled() is False
    assert "1 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )

    ch1_combo = rows["CH1"]["strategy_combo"]
    ch1_combo.setCurrentIndex(
        ch1_combo.findData("robust_global_event_reject")
    )
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == (
        "Selection changed. Confirm this method."
    )
    assert rows["CH1"]["action_button"].text() == (
        "Confirm method"
    )
    assert rows["CH1"]["action_button"].isEnabled() is True
    assert "0 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    rows["CH1"]["action_button"].click()
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].text() == "Confirmed"
    assert rows["CH1"]["action_button"].isEnabled() is False
    assert "1 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )

    rows = window._guided_local_preview_confirmation_rows
    ch2_combo = rows["CH2"]["strategy_combo"]
    ch2_combo.setCurrentIndex(
        ch2_combo.findData("signal_only_f0")
    )
    assert len(window._guided_strategy_choices) == 1
    rows["CH2"]["action_button"].click()
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH2"]["status_label"].text() == "Confirmed"
    assert rows["CH2"]["action_button"].text() == "Confirmed"
    assert rows["CH2"]["action_button"].isEnabled() is False
    assert rows["CH2"]["strategy_combo"].currentData() == "signal_only_f0"
    assert "2 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    window._rebuild_guided_local_preview_confirmation_rows()
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH2"]["status_label"].text() == "Confirmed"
    assert rows["CH2"]["strategy_combo"].currentData() == "signal_only_f0"
    assert "2 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    assert len(window._guided_strategy_choices) == 2
    ch2_choice = window._guided_strategy_choices[
        ("local_correction_preview", "CH2")
    ]
    assert ch2_choice["local_preview_evidence"]["preview_id"] == (
        ch2_result["preview_id"]
    )
    assert ch2_choice["local_preview_evidence"]["roi"] == "CH2"
    assert ch2_choice["strategy_family"] == "signal_only_f0"
    assert ch2_choice["strategy"] == "signal_only_f0"
    assert ch2_choice["dynamic_fit_mode"] is None
    assert ch2_choice["confirmed"] is True
    assert ch2_choice["local_preview_evidence"][
        "evidence_source_type"
    ] == "local_preview"
    assert ch2_choice["local_preview_evidence"][
        "current_or_stale"
    ] == "current"

    window._guided_preview_roi_combo.setCurrentIndex(
        window._guided_preview_roi_combo.findData("CH3")
    )
    window._guided_preview_generate_btn.click()
    rows = window._guided_local_preview_confirmation_rows
    ch3_combo = rows["CH3"]["strategy_combo"]
    ch3_combo.setCurrentIndex(
        ch3_combo.findData("robust_global_event_reject")
    )
    rows["CH3"]["action_button"].click()
    assert "3 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    assert window._guided_correction_continue_btn.isEnabled() is True
    assert window._guided_correction_continue_status.text() == (
        "Correction strategies are ready. Continue to Feature Detection."
    )

    rows = window._guided_local_preview_confirmation_rows
    ch3_combo = rows["CH3"]["strategy_combo"]
    ch3_combo.setCurrentIndex(
        ch3_combo.findData("global_linear_regression")
    )
    assert window._guided_correction_continue_btn.isEnabled() is False
    assert window._guided_correction_continue_status.text() == (
        "Review and confirm changed correction strategies before continuing."
    )
    rows = window._guided_local_preview_confirmation_rows
    rows["CH3"]["action_button"].click()
    assert window._guided_correction_continue_btn.isEnabled() is True
    assert window._guided_correction_continue_status.text() == (
        "Correction strategies are ready. Continue to Feature Detection."
    )
    rows = window._guided_local_preview_confirmation_rows
    ch3_combo = rows["CH3"]["strategy_combo"]
    ch3_combo.setCurrentIndex(
        ch3_combo.findData("robust_global_event_reject")
    )
    assert window._guided_correction_continue_btn.isEnabled() is False
    assert window._guided_correction_continue_status.text() == (
        "Review and confirm changed correction strategies before continuing."
    )
    rows = window._guided_local_preview_confirmation_rows
    rows["CH3"]["action_button"].click()
    assert window._guided_correction_continue_btn.isEnabled() is True
    assert window._guided_correction_continue_status.text() == (
        "Correction strategies are ready. Continue to Feature Detection."
    )

    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.cache_id is None
    assert len(plan.per_roi_correction_strategy_choices) == 3
    assert {
        choice.roi_id: choice.current_or_stale
        for choice in plan.per_roi_correction_strategy_choices
    } == {"CH1": "current", "CH2": "current", "CH3": "current"}
    # The obsolete Guided post-hoc applied-dF/F route has been retired;
    # the GUI no longer derives/sets this deprecated field for new plans.
    assert plan.applied_dff_orchestration_enabled is False
    assert plan.global_correction_strategy == "dynamic_fit"
    assert plan.dynamic_fit_mode == "robust_global_event_reject"
    plan_choices = {
        choice.roi_id: choice
        for choice in plan.per_roi_correction_strategy_choices
    }
    assert plan_choices["CH2"].selected_strategy == "signal_only_f0"
    assert plan_choices["CH2"].evidence_reference[
        "evidence_source_type"
    ] == "local_preview"
    from photometry_pipeline.guided_new_analysis_plan import (
        evaluate_new_analysis_plan_readiness,
    )

    ready_blocking = {
        issue.category
        for issue in evaluate_new_analysis_plan_readiness(
            plan
        ).blocking_issues
    }
    assert "missing_diagnostic_cache" not in ready_blocking
    assert "missing_strategy_choice_for_included_roi" not in ready_blocking

    window._guided_preview_roi_combo.setCurrentIndex(
        window._guided_preview_roi_combo.findData("CH1")
    )
    window._guided_preview_generate_btn.click()
    stale_plan = window._build_guided_new_analysis_draft_plan()
    stale_by_roi = {
        choice.roi_id: choice.current_or_stale
        for choice in stale_plan.per_roi_correction_strategy_choices
    }
    assert stale_by_roi == {
        "CH1": "stale",
        "CH2": "current",
        "CH3": "current",
    }
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH1"]["status_label"].text() == "The preview changed. Confirm this method again."
    assert rows["CH2"]["status_label"].text() == "Confirmed"
    replacement_signal_f0 = (
        window._guided_local_preview_locked_evidence_for_roi(
            "CH1", "signal_only_f0"
        )
    )
    assert replacement_signal_f0 is not None
    assert replacement_signal_f0["preview_id"] != ch1_signal_f0_preview_id
    assert replacement_signal_f0["current_or_stale"] == "current"
    assert replacement_signal_f0["locked"] is False
    assert replacement_signal_f0["selectable"] is True
    ch1_combo = rows["CH1"]["strategy_combo"]
    ch1_combo.setCurrentIndex(
        ch1_combo.findData("robust_global_event_reject")
    )
    rows["CH1"]["action_button"].click()
    refreshed_choice = window._guided_strategy_choices[
        ("local_correction_preview", "CH1")
    ]
    assert refreshed_choice["stale"] is False
    assert refreshed_choice["local_preview_evidence"]["preview_id"] == (
        window._guided_preview_last_result["preview_id"]
    )
    window._duration_edit.setText("601")
    changed_plan = window._build_guided_new_analysis_draft_plan()
    assert all(
        choice.current_or_stale == "stale"
        for choice in changed_plan.per_roi_correction_strategy_choices
    )
    blocking = {
        issue.category
        for issue in evaluate_new_analysis_plan_readiness(
            changed_plan
        ).blocking_issues
    }
    assert "stale_strategy_choice" in blocking
    assert "missing_diagnostic_cache" in blocking
    validation_context = window._capture_guided_backend_validation_context()
    assert all(
        choice.current_or_stale == "stale"
        for choice in
        validation_context.draft.per_roi_correction_strategy_choices
    )

    window._guided_preview_roi_combo.setCurrentIndex(
        window._guided_preview_roi_combo.findData("CH2")
    )
    for checkbox in window._guided_preview_method_checkboxes.values():
        checkbox.setChecked(False)
    assert window._guided_preview_signal_f0_cb.isChecked() is True
    assert window._guided_preview_generate_btn.isEnabled() is True
    window._guided_preview_generate_btn.click()
    signal_only_result = window._guided_preview_last_result
    assert signal_only_result["status"] == "success"
    assert signal_only_result["method_statuses"] == {}
    assert signal_only_result["signal_only_f0_preview_requested"] is True
    assert signal_only_result["signal_only_f0_preview_evidence"][
        "valid"
    ] is True
    assert Path(signal_only_result["visual_preview_path"]).is_file()
    assert signal_only_result["visual_panel_titles"] == [
        "Signal-Only F0 baseline",
        "Signal-Only F0-corrected dF/F preview",
    ]
    assert signal_only_result["visual_trace_labels"] == [
        "Raw signal",
        "Signal-only F0 baseline",
        "Signal-only dF/F",
    ]
    rows = window._guided_local_preview_confirmation_rows
    ch2_combo = rows["CH2"]["strategy_combo"]
    assert ch2_combo.findData("signal_only_f0") >= 0
    ch2_combo.setCurrentIndex(ch2_combo.findData("signal_only_f0"))
    rows["CH2"]["action_button"].click()
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH2"]["status_label"].text() == "Confirmed"
    assert rows["CH2"]["action_button"].text() == "Confirmed"
    assert rows["CH2"]["action_button"].isEnabled() is False

    window._guided_preview_method_checkboxes[
        "robust_global_event_reject"
    ].setChecked(True)
    window._guided_preview_signal_f0_cb.setChecked(False)
    assert window._guided_preview_generate_btn.isEnabled() is True
    window._guided_preview_generate_btn.click()
    without_signal_f0 = window._guided_preview_last_result
    assert without_signal_f0["signal_only_f0_preview_requested"] is False
    assert "signal_only_f0_preview_evidence" not in without_signal_f0
    assert (
        window._guided_local_preview_locked_evidence_for_roi(
            "CH2", "signal_only_f0"
        )
        is None
    )
    rows = window._guided_local_preview_confirmation_rows
    assert rows["CH2"]["strategy_combo"].findData("signal_only_f0") == -1
    assert rows["CH2"]["status_label"].text() != "Confirmed"
    assert window._guided_local_signal_f0_preview_label.text() == (
        "Signal-Only F0 was not included in this preview."
    )
    assert window._guided_local_signal_f0_preview_label.isHidden() is False
    assert runner.argv is None
    assert all(
        choice["stale"] is True
        for choice in window._guided_strategy_choices.values()
    )
    window._refresh_guided_confirm_strategy_panel()
    assert "0 of 3 ROIs confirmed" in (
        window._guided_confirm_strategy_progress_label.text()
    )
    assert any(
        choice.selected_strategy == "signal_only_f0"
        for choice in changed_plan.per_roi_correction_strategy_choices
    )
    assert "final run will recompute correction" in " ".join(
        _label_texts(
            window._guided_workflow_stack.widget(
                list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
            )
        )
    )


def test_new_local_preview_only_stales_confirmation_for_its_roi(
    window, monkeypatch
):
    window._set_guided_workflow_mode("new_analysis")
    monkeypatch.setattr(
        window, "_guided_local_preview_setup_signature", lambda: "setup-1"
    )
    monkeypatch.setattr(
        window, "_guided_selected_roi_ids", lambda: (["CH1", "CH2"], ["CH1", "CH2"])
    )
    window._guided_strategy_choices = {
        ("local_correction_preview", "CH1"): {
            "source_type": "local_correction_preview",
            "roi": "CH1",
            "strategy": "robust_global_event_reject",
            "setup_signature": "setup-1",
            "current": True,
            "stale": False,
            "confirmed": True,
        }
    }

    window._mark_guided_local_preview_choices_stale(
        "A newer local preview was generated for this ROI.",
        roi="CH2",
    )
    window._build_guided_new_analysis_draft_plan()

    ch1 = window._guided_strategy_choices[
        ("local_correction_preview", "CH1")
    ]
    assert ch1["current"] is True
    assert ch1["stale"] is False
    assert not any(
        isinstance(choice, dict) and choice.get("roi") == "CH2"
        for choice in window._guided_strategy_choices.values()
    )


def test_local_preview_failure_reports_selected_segment_without_fallback(
    window, tmp_path, monkeypatch
):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    files = []
    for index in range(2):
        path = input_dir / f"2025_01_01-{index:02d}_00_00" / "fluorescence.csv"
        path.parent.mkdir()
        path.write_text(f"source {index}", encoding="utf-8")
        files.append(path)
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._discovery_cache = {
        "resolved_format": "RWD",
        "n_total_discovered": 2,
        "n_preview": 2,
        "sessions": [
            {
                "index": index,
                "session_id": f"2025_01_01-{index:02d}_00_00",
                "path": str(path),
                "included_in_preview": True,
            }
            for index, path in enumerate(files)
        ],
        "rois": [{"roi_id": "CH1"}],
    }
    window._populate_discovery_ui(window._discovery_cache)
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {
            "csv_path": path,
            "fs_hz": 20.0,
            "chunk_duration_sec": 600.0,
            "time_col": "TimeStamp",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        },
    )
    loaded = []

    def fail_load(path, _format, _config, chunk_id):
        loaded.append((path, chunk_id))
        raise ValueError("selected session has invalid timestamps")

    monkeypatch.setattr(
        correction_preview_module, "load_chunk", fail_load
    )
    runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = runner
    window._set_guided_workflow_mode("new_analysis")
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    window._refresh_guided_diagnostics_panel()
    window._guided_preview_chunk_combo.setCurrentIndex(1)

    window._guided_preview_generate_btn.click()

    assert loaded == [(str(files[1].resolve()), 0)]
    assert all(path != str(files[0].resolve()) for path, _ in loaded)
    assert runner.argv is None
    assert window._guided_preview_status_label.text() == (
        "Could not load the selected preview segment. Try another segment or "
        "check that the source file is still available."
    )
    details = window._guided_preview_messages_label.text()
    assert "Selected segment: session-1" in details
    assert "Discovered session index: 1" in details
    assert f"Source path: {files[1].resolve()}" in details
    assert "Adapter-local chunk ID: 0" in details
    assert "Input format: rwd" in details
    assert "selected session has invalid timestamps" in details
    assert window._guided_strategy_choices == {}


def test_correction_preview_ready_has_openable_review_summary(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    opened = []
    monkeypatch.setattr(
        window,
        "_open_guided_preview_output",
        lambda path: opened.append(path),
    )
    _generate_ready_guided_correction_preview(window)
    result = window._guided_preview_last_result
    assert opened == []

    text = window._guided_preview_review_label.text()
    assert "Preview for CH1, segment 0." in text
    assert "Methods compared:" in text
    assert "Robust Global Event-Reject Fit" in text
    assert window._guided_preview_open_btn.text() == "Open exported report"
    assert window._guided_preview_technical_details_group.isHidden() is True
    assert window._guided_preview_technical_toggle_btn.text() == (
        "Show technical details"
    )
    report_path = Path(result["user_report_path"])
    assert report_path.name == "correction_preview_report.html"
    report_text = report_path.read_text(encoding="utf-8")
    assert "Correction preview" in report_text
    assert "ROI:</strong> CH1" in report_text
    assert "Preview segment:</strong> 0" in report_text
    assert "Methods compared:" in report_text
    assert "First 12 trace samples" in report_text
    assert '"comparison_plot"' not in report_text
    visual_path = Path(result["visual_preview_path"])
    assert visual_path.name == "correction_preview_visual.png"
    assert visual_path.is_file()
    assert set(result["visual_trace_sources"]) == {
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
        "global_linear_regression",
    }
    assert result["visual_panel_titles"] == [
        "Source segment",
        "Dynamic-fit corrected signal comparison",
        "Dynamic-fit fitted reference comparison",
    ]
    assert result["visual_trace_labels"] == [
        "Raw signal",
        "Reference/control signal",
        "Robust Global Event-Reject Fit",
        "Adaptive Event-Gated Fit",
        "Global Linear Regression",
    ]
    assert all(
        count > 0
        for count in result["visual_trace_sample_counts"].values()
    )
    assert window._guided_preview_visual_label.pixmap().isNull() is False
    assert window._guided_preview_visual_display_width <= 820
    assert window._guided_preview_visual_display_width <= max(
        320, window._guided_preview_visual_viewport_width - 72
    )
    visual_text = window._guided_preview_visual_status_label.text()
    assert "Use the plot below to choose the correction method for CH1." == (
        visual_text
    )

    assert opened == []
    window._guided_preview_technical_toggle_btn.click()
    assert window._guided_preview_technical_details_group.isHidden() is False
    assert window._guided_preview_technical_toggle_btn.text() == (
        "Hide technical details"
    )
    window._guided_preview_open_btn.click()

    assert opened == [str(report_path)]
    window._guided_preview_technical_toggle_btn.click()
    assert window._guided_preview_technical_details_group.isHidden() is True


def test_signal_only_f0_preview_has_plain_review_affordance(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    expected_summary = str(tmp_path / "signal-summary.json")

    def fake_signal_preview(*_args, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_csv = output_dir / "signal_only_f0_diagnostic_chunks.csv"
        chunk_csv.write_text(
            "roi,chunk_id,status,n_samples,signal_finite_fraction,"
            "f0_median,dff_median,signal_only_f0_candidate_viability,"
            "signal_only_f0_candidate_confidence,warning_flags,error\n"
            "CH1,0,success,100,1.0,2.0,0.1,viable,high,,\n",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "status": "success",
            "diagnostic_id": "signal-preview",
            "output_dir": str(output_dir),
            "provenance_path": str(tmp_path / "provenance.json"),
            "summary_path": expected_summary,
            "chunk_csv_path": str(chunk_csv),
            "trace_csv_paths": [],
            "warnings": [],
            "errors": [],
            "chunk_statuses": {"0": {"status": "success", "error": ""}},
        }

    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        fake_signal_preview,
    )
    opened = []
    monkeypatch.setattr(
        window,
        "_open_guided_preview_output",
        lambda path: opened.append(path),
    )
    assert window._guided_signal_f0_generate_btn.text() == (
        "Generate Signal-Only F0 preview"
    )

    window._guided_signal_f0_generate_btn.click()

    text = window._guided_signal_f0_review_label.text()
    assert "Signal-Only F0 preview: ready." in text
    assert "ROI: CH1" in text
    assert "Preview segment: 0" in text
    assert "no trace preview is available yet" in text
    assert "Guided Run cannot execute it yet" in (
        window._guided_signal_f0_status_label.parentWidget().findChild(
            QLabel, "guidedSignalOnlyF0DiagnosticIntro"
        ).text()
    )
    assert window._guided_signal_f0_technical_details_group.isHidden() is True
    assert window._guided_signal_f0_technical_toggle_btn.text() == (
        "Show technical details"
    )
    report_path = Path(
        window._guided_signal_f0_last_result["user_report_path"]
    )
    assert report_path.name == "signal_only_f0_preview_report.html"
    assert "Signal-Only F0 preview" in report_path.read_text(
        encoding="utf-8"
    )
    assert opened == []
    window._guided_signal_f0_technical_toggle_btn.click()
    assert window._guided_signal_f0_technical_details_group.isHidden() is False
    assert window._guided_signal_f0_open_btn.text() == "Open exported report"
    window._guided_signal_f0_open_btn.click()
    assert opened == [str(report_path)]


def test_visual_preview_failure_is_plain_and_keeps_strategy_locked(
    window, tmp_path, monkeypatch
):
    _build_ready_guided_diagnostic_cache(window, tmp_path, monkeypatch)
    monkeypatch.setattr(
        window,
        "_render_guided_correction_preview_visual",
        lambda _result: "",
    )

    window._guided_preview_generate_btn.click()

    assert window._guided_preview_status_label.text() == (
        "Preview ready, but the plot could not be generated."
    )
    assert "preview plot could not be loaded" in (
        window._guided_preview_visual_status_label.text()
    )
    assert window._guided_preview_visual_label.isHidden() is True
    assert window._guided_confirm_locked_label.isHidden() is False
    assert window._guided_confirm_strategy_combo.parentWidget().isHidden()
    assert "needs to be regenerated" in (
        window._guided_correction_next_action_label.text()
    )


def _guided_correction_approach_normal_text(window) -> str:
    """Group titles, labels, and buttons (with tooltips) on the Correction
    Approach step, excluding technical-details/legacy panels that are
    permanently hidden or collapsed behind an explicit toggle."""
    step_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    excluded_roots = [
        getattr(window, "_guided_preview_technical_details_group", None),
        getattr(window, "_guided_signal_f0_group", None),
        getattr(window, "_guided_full_evidence_group", None),
        getattr(window, "_guided_diagnostics_slots_group", None),
        getattr(window, "_guided_generated_outputs_group", None),
        getattr(window, "_guided_confirm_evidence_group", None),
    ]
    excluded_roots = [w for w in excluded_roots if w is not None]

    def is_excluded(widget) -> bool:
        parent = widget
        while parent is not None:
            if parent in excluded_roots:
                return True
            parent = parent.parentWidget()
        return False

    texts = []
    for widget in step_widget.findChildren(QGroupBox):
        if is_excluded(widget):
            continue
        texts.append(widget.title())
        texts.append(widget.toolTip())
    for widget in step_widget.findChildren(QLabel):
        if is_excluded(widget):
            continue
        texts.append(widget.text())
        texts.append(widget.toolTip())
    for widget in step_widget.findChildren(QPushButton):
        if is_excluded(widget):
            continue
        texts.append(widget.text())
        texts.append(widget.toolTip())
    return " ".join(t for t in texts if t)


def test_correction_approach_before_generate_has_no_premature_or_stale_text(
    window, tmp_path, monkeypatch
):
    """4J16k29 (H.1): before the user presses "Generate correction preview",
    the page must not claim the preview is ready, must not say "stale", and
    must not show a misleading visual-display error."""
    _setup_guided_local_raw_preview_ready(window, tmp_path, monkeypatch)

    assert getattr(window, "_guided_preview_has_result", False) is False
    assert window._guided_preview_status_label.text() == "No preview generated yet."

    normal_text = _guided_correction_approach_normal_text(window)
    lowered = normal_text.lower()
    assert "preview ready" not in lowered
    assert "preview comparison ready" not in lowered
    assert "stale" not in lowered
    assert "could not be displayed" not in lowered


def test_correction_approach_selection_change_requests_regeneration_not_stale(
    window, tmp_path, monkeypatch
):
    """4J16k29 (H.2): after the ROI/segment/method selection changes on an
    existing preview, the page must say the preview needs to be
    regenerated, must not say "stale", and must not show the "visual
    preview could not be displayed" error (the preview simply is not
    current, not broken)."""
    _setup_guided_local_raw_preview_ready(window, tmp_path, monkeypatch)
    window._guided_preview_generate_btn.click()
    assert window._guided_preview_status_label.text() == "Preview ready."

    window._guided_preview_chunk_combo.setCurrentIndex(
        1 if window._guided_preview_chunk_combo.count() > 1 else 0
    )

    assert window._guided_preview_result_stale is True
    assert window._guided_preview_status_label.text() == (
        "Preview needs to be regenerated because the ROI, segment, or "
        "selected methods changed."
    )
    assert window._guided_preview_review_label.isHidden() is True
    assert window._guided_preview_visual_status_label.isHidden() is True

    normal_text = _guided_correction_approach_normal_text(window)
    lowered = normal_text.lower()
    assert "stale" not in lowered
    assert "could not be displayed" not in lowered


def test_correction_approach_signal_only_f0_not_selected_has_one_plain_note(
    window, tmp_path, monkeypatch
):
    """4J16k29 (H.3): when Signal-Only F0 is left unselected, the page shows
    at most one plain note about it -- no separate debug block with sample
    counts, dF/F metrics, or "production output" language."""
    _setup_guided_local_raw_preview_ready(window, tmp_path, monkeypatch)
    window._guided_preview_signal_f0_cb.setChecked(False)
    window._guided_preview_generate_btn.click()

    assert window._guided_local_signal_f0_preview_label.text() == (
        "Signal-Only F0 was not included in this preview."
    )
    assert window._guided_local_signal_f0_preview_label.isHidden() is False

    normal_text = _guided_correction_approach_normal_text(window)
    lowered = normal_text.lower()
    for forbidden in (
        "sample count",
        "minimum df/f",
        "negative df/f",
        "production output",
        "does not write production",
    ):
        assert forbidden not in lowered


def test_confirm_strategy_table_has_no_stale_wording_and_short_button(
    window, tmp_path, monkeypatch
):
    """4J16k29 (H.4): the per-ROI confirm table must not use "stale"
    wording, and its action button must be short enough not to force the
    table wider than a normal button (the original wide label -- "Confirm
    changed strategy for this ROI" -- was the reported cause of layout
    stretching)."""
    _setup_guided_local_raw_preview_ready(window, tmp_path, monkeypatch)
    window._guided_preview_generate_btn.click()
    rows = window._guided_local_preview_confirmation_rows
    roi = next(iter(rows))
    combo = rows[roi]["strategy_combo"]
    combo.setCurrentIndex(combo.findData("global_linear_regression"))
    rows[roi]["action_button"].click()

    rows = window._guided_local_preview_confirmation_rows
    assert rows[roi]["status_label"].text() == "Confirmed"

    other_strategy = combo.findData("robust_global_event_reject")
    combo.setCurrentIndex(other_strategy)
    rows = window._guided_local_preview_confirmation_rows
    status_text = rows[roi]["status_label"].text()
    button_text = rows[roi]["action_button"].text()

    assert "stale" not in status_text.lower()
    assert status_text == "Selection changed. Confirm this method."
    assert button_text == "Confirm method"
    # The reported layout-stretch cause was specifically an overly wide
    # button label; a short fixed phrase avoids it regardless of window size.
    assert len(button_text) <= len("Confirm method")


def test_correction_approach_technical_details_hide_raw_internals(
    window, tmp_path, monkeypatch
):
    """4J16k29 (H.5): raw paths, adapter-local chunk ID, provenance, load
    error, sample counts, and "strategy recommendation" must not appear in
    the normal visible Correction Approach text -- only behind "Show
    technical details"."""
    _setup_guided_local_raw_preview_ready(window, tmp_path, monkeypatch)
    window._guided_preview_generate_btn.click()

    assert window._guided_preview_technical_details_group.isHidden() is True
    normal_text = _guided_correction_approach_normal_text(window)
    lowered = normal_text.lower()
    for forbidden in (
        "preview directory:",
        "provenance:",
        "adapter-local chunk id",
        "load error:",
        "strategy recommendation",
    ):
        assert forbidden not in lowered

    # The same details remain reachable behind the toggle.
    assert "Preview directory:" in window._guided_preview_artifacts_label.text()
    assert "Provenance:" in window._guided_preview_artifacts_label.text()


def test_correction_preview_report_failure_falls_back_to_folder(
    window, tmp_path, monkeypatch
):
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    window._guided_preview_has_result = True
    window._guided_preview_result_stale = False
    window._guided_preview_last_result = {
        "status": "success",
        "preview_output_dir": str(preview_dir),
        "preview_summary_path": str(preview_dir / "preview_summary.json"),
        "user_report_path": "",
        "roi": "CH1",
        "chunk_index": 0,
        "method_statuses": {},
    }
    opened = []
    monkeypatch.setattr(
        window,
        "_open_guided_preview_output",
        lambda path: opened.append(path),
    )

    window._refresh_guided_preview_review_affordances()
    window._guided_preview_open_btn.click()

    assert window._guided_preview_open_btn.text() == "Open preview folder"
    assert "report could not be generated" in (
        window._guided_preview_review_label.text()
    )
    assert opened == [str(preview_dir)]
    assert opened != [
        window._guided_preview_last_result["preview_summary_path"]
    ]


def test_gui_draft_plan_contains_moved_plan_panels(window):
    # Navigate to Draft Plan
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # Draft Plan step layout wrapper
    draft_step_widget = window._guided_workflow_stack.widget(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # Verify moved panels are present
    assert draft_step_widget.findChild(
        QWidget, "guidedFeatureEventProfileEditorPanel"
    ) is None
    assert draft_step_widget.findChild(
        QWidget, "guidedReviewFeatureDetectionPanel"
    ) is not None
    assert draft_step_widget.findChild(QWidget, "guidedOutputDestinationPanel") is not None
    assert draft_step_widget.findChild(QWidget, "guidedDraftPlanExportPanel") is not None
    assert draft_step_widget.findChild(QWidget, "guidedPlanReadinessSummaryPanel") is not None
    assert draft_step_widget.findChild(QWidget, "guidedDraftRunPlanPreviewPanel") is not None
    assert draft_step_widget.findChild(QWidget, "guidedDraftRunPlanChecklistPanel") is not None

    # Verify correction controls are NOT part of the Draft Plan step
    assert draft_step_widget.findChild(QWidget, "guidedConfirmStrategyRoiCombo") is None
    assert draft_step_widget.findChild(QWidget, "guidedConfirmStrategyChoiceCombo") is None
    feature_step_widget = window._guided_workflow_stack.widget(
        list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    )
    assert feature_step_widget.findChild(
        QWidget, "guidedFeatureEventProfileEditorPanel"
    ) is not None


def test_gui_new_analysis_run_preview_panel_hidden_in_completed_run_mode(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._set_guided_workflow_mode("open_results")

    draft_step_widget = window._guided_workflow_stack.widget(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    preview_group = draft_step_widget.findChild(QWidget, "guidedNewAnalysisRunPreviewPanel")

    assert preview_group is not None
    assert preview_group.isHidden()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    assert preview_group.isHidden()


def test_gui_real_split_workflow_flow(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))

    # 1. Navigate to Confirm Strategy step
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))

    # 2. Select correction strategy
    idx = window._guided_confirm_strategy_combo.findData("robust_global_event_reject")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    # Assert stored choice
    plan, _ = window._build_guided_draft_run_plan()
    assert len(plan.roi_plan) == 1
    assert plan.roi_plan[0].roi == "CH1"
    assert plan.roi_plan[0].correction_strategy.strategy == "robust_global_event_reject"

    # 3. Navigate to Draft Plan step
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    # Assert readiness, preview, checklist show 1 ROI choice
    summary = window._guided_plan_readiness_summary_label.text()
    assert "1 ROI correction strategy" in summary
    assert "feature/event profile" in summary
    assert "output destination" in summary
    assert "Blocked: execution intentionally unavailable" in summary

    # 4. Apply a feature/event profile
    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._guided_feature_event_polarity_combo.setCurrentText("positive")
    window._guided_feature_event_peak_method_combo.setCurrentText("mean_std")
    window._guided_feature_event_peak_k_edit.setText("3.0")
    window._guided_feature_event_apply_btn.click()

    # 5. Apply an output destination
    out_dest = tmp_path / "future_workflow_out"
    window._guided_output_path_edit.setText(str(out_dest))
    window._guided_output_apply_btn.click()

    # 6. Export JSON
    export_file = tmp_path / "workflow_plan.json"
    window._guided_export_path_edit.setText(str(export_file))
    window._guided_export_btn.click()

    # Assert exported JSON exists and round-trips
    assert export_file.exists()
    import json
    from photometry_pipeline.guided_run_plan import deserialize_plan_from_dict, validate_plan_contract
    with open(export_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    restored = deserialize_plan_from_dict(payload)
    assert validate_plan_contract(restored) == []
    assert len(restored.roi_plan) == 1
    assert restored.feature_event_profiles[0].config_fields["event_signal"] == "dff"
    assert restored.output_policy.output_root == str(out_dest.resolve())

    # Assert non-production outputs guarantee (only export file written)
    assert not out_dest.exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "MANIFEST.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()
    after_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after_files == before_files


def test_gui_imported_plan_review_panel_is_draft_plan_only(window):
    draft_idx = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    confirm_idx = list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    draft_step = window._guided_workflow_stack.widget(draft_idx)
    confirm_step = window._guided_workflow_stack.widget(confirm_idx)

    panel = draft_step.findChild(QWidget, "guidedImportedPlanReviewPanel")
    assert panel is not None
    assert draft_step.findChild(QWidget, "guidedImportedPlanPathEdit") is not None
    assert draft_step.findChild(QWidget, "guidedImportedPlanOpenButton") is not None
    assert draft_step.findChild(QWidget, "guidedImportedPlanStatusLabel") is not None
    assert draft_step.findChild(QWidget, "guidedImportedPlanSummaryLabel") is not None
    assert draft_step.findChild(QWidget, "guidedImportedPlanAdoptionStatusLabel") is not None
    assert "Open an exported completed-run review-plan JSON for read-only review" in "\n".join(_label_texts(panel))
    assert "Eligibility is informational only" in _adoption_status_text(window)
    assert confirm_step.findChild(QWidget, "guidedImportedPlanReviewPanel") is None

    panel_buttons = [button.text() for button in panel.findChildren(QPushButton)]
    assert panel_buttons == ["Open plan for review"]
    assert not any("Adopt" in text for text in panel_buttons)
    assert not any("Apply imported" in text for text in panel_buttons)
    assert not any("Load into draft" in text for text in panel_buttons)
    assert not any("RunSpec" in text for text in panel_buttons)
    assert not any(text == "Run" or "Guided Run" in text for text in panel_buttons)


def test_gui_imported_plan_adoption_status_no_candidate_initial(window):
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    text = _adoption_status_text(window)

    assert "Future adoption eligibility" in text
    assert "Eligibility is informational only" in text
    assert "Future adoption eligible: No" in text
    assert "no active candidate" in text
    assert "Adoption action: unavailable in this stage" in text
    assert "Execution: blocked" in text
    assert "Files written: none" in text


def test_gui_imported_plan_adoption_status_source_matched_candidate_eligible(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "eligible_plan.json"
    out_root = tmp_path / "future_output"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(out_root)),
        ),
    )
    before_plan = window._build_guided_draft_run_plan()[0].to_dict()

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: Yes" in text
    assert "Blocking reasons:\n- none" in text
    assert "source matched" in text
    assert "imported ROI choices match current inventory" in text
    assert window._build_guided_draft_run_plan()[0].to_dict() == before_plan
    assert not out_root.exists()
    assert sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*")) == before_files


def test_gui_imported_plan_adoption_status_source_mismatch_blocks(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    _load_preview_completed_run(window, run_a, monkeypatch)
    before_source = window._current_guided_completed_run_dir()
    path = tmp_path / "source_mismatch_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_b,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert "source mismatch between imported plan and current completed run" in text
    assert window._current_guided_completed_run_dir() == before_source


def test_gui_imported_plan_adoption_status_no_current_run_blocks(window, tmp_path):
    run_dir = _make_preview_completed_run(tmp_path)
    window._current_run_dir = ""
    path = tmp_path / "no_current_run_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert "no current completed run loaded" in text
    assert window._current_guided_completed_run_dir() == ""


def test_gui_imported_plan_adoption_status_missing_imported_roi_blocks(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "missing_roi_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH3"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert "missing imported ROI(s) in current run: CH3" in text


def test_gui_imported_plan_adoption_status_zero_roi_choices_block(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "zero_roi_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=[],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert "zero ROI choices cannot be adopted in first adoption implementation" in text


def test_gui_imported_plan_adoption_status_extra_current_roi_warns_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "partial_roi_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: Yes" in text
    assert "Blocking reasons:\n- none" in text
    assert "extra current ROI(s) not present in candidate" in text
    assert "CH2" in text


def test_gui_imported_plan_adoption_status_signal_only_f0_warns_without_execution(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    calls = {"signal": 0, "preview": 0}
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    path = tmp_path / "signal_only_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            strategy="signal_only_f0",
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: Yes" in text
    assert "Signal-Only F0 is explicit, not fallback; no diagnostic will run" in text
    assert calls == {"signal": 0, "preview": 0}
    assert sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*")) == before_files


@pytest.mark.parametrize("strategy", ["auto", "needs_review", "no_correction"])
def test_gui_imported_plan_adoption_status_forbidden_strategy_blocks(window, tmp_path, monkeypatch, strategy):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / f"{strategy}_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            strategy=strategy,
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert f"forbidden runnable correction strategy: {strategy}" in text


def test_gui_imported_plan_adoption_status_unsafe_output_root_blocks(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    unsafe_root = run_dir / "_analysis" / "phasic_out" / "features" / "future"
    path = tmp_path / "unsafe_root_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(unsafe_root)),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert "unsafe OutputPolicy" in text
    assert "inside legacy output directory" in text
    assert not unsafe_root.exists()


def test_gui_imported_plan_adoption_status_unsafe_output_flags_block(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "unsafe_flags_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(
                output_root=str(tmp_path / "future_output"),
                overwrite=True,
                separate_from_source_required=False,
                legacy_outputs_protected=False,
            ),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert "Overwrite is enabled" in text
    assert "separate_from_source_required is disabled" in text
    assert "legacy_outputs_protected is disabled" in text


def test_gui_imported_plan_adoption_status_no_output_root_warns_not_blocks(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "no_output_root_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_dir, rois=["CH1", "CH2"], profile=True))

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: Yes" in text
    assert "no output_root; candidate would remain incomplete" in text
    assert "Output policy: no output destination configured" in text


def test_gui_imported_plan_adoption_status_no_feature_profile_warns_not_blocks(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "no_profile_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=False,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: Yes" in text
    assert "no feature/event profile; candidate would remain incomplete" in text
    assert "Feature/event profile: no feature/event profile configured" in text


def test_gui_imported_plan_adoption_status_failed_open_clears_eligibility(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    valid_path = tmp_path / "eligible_plan.json"
    bad_path = tmp_path / "bad_plan.json"
    _write_guided_plan_json(
        valid_path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )
    bad_path.write_text("{bad", encoding="utf-8")

    _review_plan(window, valid_path)
    assert "Future adoption eligible: Yes" in _adoption_status_text(window)
    _review_plan(window, bad_path)

    _assert_no_active_imported_plan_candidate(window)


def test_gui_imported_plan_adoption_status_candidate_replacement_recomputes(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    eligible_path = tmp_path / "eligible_plan.json"
    ineligible_path = tmp_path / "ineligible_plan.json"
    _write_guided_plan_json(
        eligible_path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )
    _write_guided_plan_json(
        ineligible_path,
        _candidate_plan(
            run_dir,
            rois=["CH3"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output_2")),
        ),
    )

    _review_plan(window, eligible_path)
    assert "Future adoption eligible: Yes" in _adoption_status_text(window)
    _review_plan(window, ineligible_path)
    text = _adoption_status_text(window)

    assert "Future adoption eligible: No" in text
    assert "missing imported ROI(s) in current run: CH3" in text
    assert str(eligible_path.resolve()) not in text


def test_gui_imported_plan_adoption_status_completed_run_switch_clears(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    _load_preview_completed_run(window, run_a, monkeypatch)
    path = tmp_path / "eligible_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_a,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )

    _review_plan(window, path)
    assert "Future adoption eligible: Yes" in _adoption_status_text(window)
    _load_preview_completed_run(window, run_b, monkeypatch)

    _assert_no_active_imported_plan_candidate(window)


def test_gui_imported_plan_adoption_status_does_not_mutate_live_draft_state(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("robust_global_event_reject")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._guided_output_path_edit.setText(str(tmp_path / "unsaved_output"))
    window._guided_export_path_edit.setText(str(tmp_path / "export_target.json"))
    path = tmp_path / "eligible_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(tmp_path / "future_output")),
        ),
    )
    before_plan = window._build_guided_draft_run_plan()[0].to_dict()
    before_readiness = window._guided_plan_readiness_summary_label.text()
    before_preview = window._guided_draft_run_plan_preview_label.text()
    before_checklist = window._guided_draft_run_plan_checklist_label.text()
    before_export_path = window._guided_export_path_edit.text()
    before_output_path = window._guided_output_path_edit.text()
    before_feature_signal = window._guided_feature_event_signal_combo.currentText()
    before_source = window._current_guided_completed_run_dir()

    _review_plan(window, path)

    assert "Future adoption eligible:" in _adoption_status_text(window)
    assert window._build_guided_draft_run_plan()[0].to_dict() == before_plan
    assert window._guided_plan_readiness_summary_label.text() == before_readiness
    assert window._guided_draft_run_plan_preview_label.text() == before_preview
    assert window._guided_draft_run_plan_checklist_label.text() == before_checklist
    assert window._guided_export_path_edit.text() == before_export_path
    assert window._guided_output_path_edit.text() == before_output_path
    assert window._guided_feature_event_signal_combo.currentText() == before_feature_signal
    assert window._current_guided_completed_run_dir() == before_source


def test_gui_imported_plan_adoption_status_no_output_guarantee(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    out_root = tmp_path / "future_output"
    path = tmp_path / "eligible_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1", "CH2"],
            profile=True,
            output_policy=OutputPolicy(output_root=str(out_root)),
        ),
    )

    _review_plan(window, path)

    assert sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*")) == before_files
    assert not out_root.exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()


def test_gui_imported_plan_adoption_status_no_adoption_controls(window):
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    panel = window._guided_workflow_tab.findChild(QWidget, "guidedImportedPlanReviewPanel")
    button_texts = [button.text() for button in panel.findChildren(QPushButton)]

    assert button_texts == ["Open plan for review"]
    assert not any("Adopt" in text for text in button_texts)
    assert not any("Apply imported" in text for text in button_texts)
    assert not any("Load into draft" in text for text in button_texts)
    assert not any("RunSpec" in text for text in button_texts)
    assert not any(text == "Run" or "Guided Run" in text for text in button_texts)


@pytest.mark.parametrize(
    "path_text, expected",
    [
        ("   ", "Path cannot be empty"),
        ("plan.txt", "Plan file must have .json extension"),
        ("missing.json", "File does not exist"),
    ],
)
def test_gui_imported_plan_review_invalid_paths_rejected(window, tmp_path, path_text, expected):
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_imported_plan_path_edit.setText(path_text)
    window._guided_imported_plan_open_btn.click()

    assert "Open plan failed" in window._guided_imported_plan_status_label.text()
    assert expected in window._guided_imported_plan_status_label.text()
    _assert_no_active_imported_plan_candidate(window)


def test_gui_imported_plan_review_directory_path_rejected(window, tmp_path):
    json_dir = tmp_path / "candidate.json"
    json_dir.mkdir()
    _review_plan(window, json_dir)

    assert "Open plan failed: Path points to a directory, not a file." in window._guided_imported_plan_status_label.text()
    _assert_no_active_imported_plan_candidate(window)


@pytest.mark.parametrize(
    "content, expected",
    [
        ("{not valid", "Invalid JSON format"),
        ("[]", "JSON root must be an object"),
        (json.dumps({"mode": "completed_run_planning"}), "Missing plan schema version"),
        (json.dumps({"schema_version": 1}), "schema_version must be a string"),
        (json.dumps({"schema_version": "guided_run_plan.v999"}), "Unsupported schema version"),
    ],
)
def test_gui_imported_plan_review_invalid_json_and_schema_rejected(window, tmp_path, content, expected):
    path = tmp_path / "bad_plan.json"
    path.write_text(content, encoding="utf-8")

    _review_plan(window, path)

    assert "Open plan failed" in window._guided_imported_plan_status_label.text()
    assert expected in window._guided_imported_plan_status_label.text()
    _assert_no_active_imported_plan_candidate(window)


def test_gui_imported_plan_review_failed_open_clears_prior_candidate_invalid_json(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan_a = _candidate_plan(run_dir, rois=["CH1"])
    plan_a.plan_id = "candidate-a"
    valid_path = tmp_path / "candidate_a.json"
    bad_path = tmp_path / "bad_candidate.json"
    _write_guided_plan_json(valid_path, plan_a)
    bad_path.write_text("{not json", encoding="utf-8")

    valid_summary = _review_plan(window, valid_path)
    assert "Plan ID: candidate-a" in valid_summary
    assert window._guided_imported_plan_candidate is not None
    assert window._guided_imported_plan_file_path == str(valid_path.resolve())

    _review_plan(window, bad_path)

    assert "Open plan failed" in window._guided_imported_plan_status_label.text()
    assert "Invalid JSON format" in window._guided_imported_plan_status_label.text()
    _assert_no_active_imported_plan_candidate(window)
    assert "candidate-a" not in window._guided_imported_plan_summary_label.text()
    assert str(valid_path.resolve()) not in window._guided_imported_plan_summary_label.text()


def test_gui_imported_plan_review_failed_open_clears_prior_candidate_bad_schema(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan_a = _candidate_plan(run_dir, rois=["CH1"])
    plan_a.plan_id = "candidate-a"
    valid_path = tmp_path / "candidate_a.json"
    bad_schema_path = tmp_path / "bad_schema.json"
    _write_guided_plan_json(valid_path, plan_a)
    bad_schema_path.write_text(json.dumps({"schema_version": "guided_run_plan.v999"}), encoding="utf-8")

    assert "Plan ID: candidate-a" in _review_plan(window, valid_path)
    assert window._guided_imported_plan_candidate is not None

    _review_plan(window, bad_schema_path)

    assert "Unsupported schema version" in window._guided_imported_plan_status_label.text()
    _assert_no_active_imported_plan_candidate(window)
    assert "candidate-a" not in window._guided_imported_plan_summary_label.text()


def test_gui_imported_plan_review_failed_open_clears_prior_candidate_missing_path(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan_a = _candidate_plan(run_dir, rois=["CH1"])
    plan_a.plan_id = "candidate-a"
    valid_path = tmp_path / "candidate_a.json"
    missing_path = tmp_path / "missing_candidate.json"
    _write_guided_plan_json(valid_path, plan_a)
    before_plan = window._build_guided_draft_run_plan()[0].to_dict()

    assert "Plan ID: candidate-a" in _review_plan(window, valid_path)
    assert window._guided_imported_plan_candidate is not None

    _review_plan(window, missing_path)

    assert "File does not exist" in window._guided_imported_plan_status_label.text()
    _assert_no_active_imported_plan_candidate(window)
    assert window._build_guided_draft_run_plan()[0].to_dict() == before_plan


def test_gui_imported_plan_review_failed_open_does_not_mutate_live_draft_state(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Correction approach"))
    idx = window._guided_confirm_strategy_combo.findData("robust_global_event_reject")
    window._guided_confirm_strategy_combo.setCurrentIndex(idx)
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._guided_output_path_edit.setText(str(tmp_path / "unsaved_output"))
    window._guided_export_path_edit.setText(str(tmp_path / "export_target.json"))
    valid_path = tmp_path / "candidate_a.json"
    bad_path = tmp_path / "bad_candidate.json"
    _write_guided_plan_json(valid_path, _candidate_plan(run_dir, rois=["CH2"]))
    bad_path.write_text("{bad", encoding="utf-8")

    before_plan = window._build_guided_draft_run_plan()[0].to_dict()
    before_readiness = window._guided_plan_readiness_summary_label.text()
    before_preview = window._guided_draft_run_plan_preview_label.text()
    before_checklist = window._guided_draft_run_plan_checklist_label.text()
    before_export_path = window._guided_export_path_edit.text()
    before_output_path = window._guided_output_path_edit.text()
    before_feature_signal = window._guided_feature_event_signal_combo.currentText()
    before_source = window._current_guided_completed_run_dir()

    _review_plan(window, valid_path)
    _review_plan(window, bad_path)

    _assert_no_active_imported_plan_candidate(window)
    assert window._build_guided_draft_run_plan()[0].to_dict() == before_plan
    assert window._guided_plan_readiness_summary_label.text() == before_readiness
    assert window._guided_draft_run_plan_preview_label.text() == before_preview
    assert window._guided_draft_run_plan_checklist_label.text() == before_checklist
    assert window._guided_export_path_edit.text() == before_export_path
    assert window._guided_output_path_edit.text() == before_output_path
    assert window._guided_feature_event_signal_combo.currentText() == before_feature_signal
    assert window._current_guided_completed_run_dir() == before_source


def test_gui_imported_plan_review_success_after_failure_sets_new_candidate(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    plan_a = _candidate_plan(run_dir, rois=["CH1"])
    plan_a.plan_id = "candidate-a"
    plan_b = _candidate_plan(run_dir, rois=["CH2"])
    plan_b.plan_id = "candidate-b"
    path_a = tmp_path / "candidate_a.json"
    path_b = tmp_path / "candidate_b.json"
    bad_path = tmp_path / "bad_candidate.json"
    _write_guided_plan_json(path_a, plan_a)
    _write_guided_plan_json(path_b, plan_b)
    bad_path.write_text("{bad", encoding="utf-8")

    assert "Plan ID: candidate-a" in _review_plan(window, path_a)
    _review_plan(window, bad_path)
    _assert_no_active_imported_plan_candidate(window)

    summary_b = _review_plan(window, path_b)

    assert window._guided_imported_plan_candidate is not None
    assert window._guided_imported_plan_file_path == str(path_b.resolve())
    assert "Plan ID: candidate-b" in summary_b
    assert "Plan ID: candidate-a" not in summary_b


def test_gui_imported_plan_review_valid_incomplete_plan_opens_without_live_mutation(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "incomplete_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_dir))
    before_plan = window._build_guided_draft_run_plan()[0].to_dict()
    before_readiness = window._guided_plan_readiness_summary_label.text()
    before_preview = window._guided_draft_run_plan_preview_label.text()
    before_checklist = window._guided_draft_run_plan_checklist_label.text()

    summary = _review_plan(window, path)

    assert "Contract: valid" in summary
    assert "Completeness: incomplete" in summary
    assert "ROI compatibility: incomplete: zero imported ROI choices" in summary
    assert window._build_guided_draft_run_plan()[0].to_dict() == before_plan
    assert window._guided_plan_readiness_summary_label.text() == before_readiness
    assert window._guided_draft_run_plan_preview_label.text() == before_preview
    assert window._guided_draft_run_plan_checklist_label.text() == before_checklist


def test_gui_imported_plan_review_source_matched_no_mutation(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_output_path_edit.setText(str(tmp_path / "unsaved_output"))
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")
    window._guided_export_path_edit.setText(str(tmp_path / "export_target.json"))
    path = tmp_path / "matched_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_dir, rois=["CH1"], profile=True))
    before_plan = window._build_guided_draft_run_plan()[0].to_dict()
    before_labels = (
        window._guided_plan_readiness_summary_label.text(),
        window._guided_draft_run_plan_preview_label.text(),
        window._guided_draft_run_plan_checklist_label.text(),
    )

    summary = _review_plan(window, path)

    assert "Source: source matched" in summary
    assert "ROI compatibility: compatible partial" in summary
    assert "Feature/event profile default" in summary
    assert window._build_guided_draft_run_plan()[0].to_dict() == before_plan
    assert (
        window._guided_plan_readiness_summary_label.text(),
        window._guided_draft_run_plan_preview_label.text(),
        window._guided_draft_run_plan_checklist_label.text(),
    ) == before_labels
    assert window._guided_feature_event_signal_combo.currentText() == "delta_f"
    assert window._guided_output_path_edit.text() == str(tmp_path / "unsaved_output")
    assert window._guided_export_path_edit.text() == str(tmp_path / "export_target.json")


def test_gui_imported_plan_review_source_mismatch_no_source_switch(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    _load_preview_completed_run(window, run_a, monkeypatch)
    path = tmp_path / "mismatch_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_b, rois=["CH1"]))
    before_source = window._current_guided_completed_run_dir()
    before_plan = window._build_guided_draft_run_plan()[0].to_dict()

    summary = _review_plan(window, path)

    assert "Source: source mismatch" in summary
    assert window._current_guided_completed_run_dir() == before_source
    assert window._build_guided_draft_run_plan()[0].to_dict() == before_plan


def test_gui_imported_plan_review_no_current_run_loaded(window, tmp_path):
    run_dir = _make_preview_completed_run(tmp_path)
    path = tmp_path / "no_current_run_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_dir, rois=["CH1"]))
    window._current_run_dir = ""

    summary = _review_plan(window, path)

    assert "Source: not matched: no active completed run loaded" in summary
    assert "ROI compatibility: unknown: no active completed run loaded" in summary
    assert window._current_guided_completed_run_dir() == ""
    assert window._build_guided_draft_run_plan()[0] is None


def test_gui_imported_plan_review_roi_mismatch_display(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    path = tmp_path / "roi_mismatch_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_dir, rois=["CH3"]))

    summary = _review_plan(window, path)

    assert "ROI compatibility: missing imported ROIs" in summary
    assert "Missing imported ROIs: CH3" in summary


def test_gui_imported_plan_review_signal_only_f0_display_without_execution(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    calls = {"signal": 0, "preview": 0}
    monkeypatch.setattr(
        main_window_module,
        "run_signal_only_f0_diagnostic_review",
        lambda *_args, **_kwargs: calls.__setitem__("signal", calls["signal"] + 1),
    )
    monkeypatch.setattr(
        main_window_module,
        "run_guided_correction_preview_comparison",
        lambda *_args, **_kwargs: calls.__setitem__("preview", calls["preview"] + 1),
    )
    path = tmp_path / "signal_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_dir, rois=["CH1"], strategy="signal_only_f0"))

    summary = _review_plan(window, path)

    assert "Signal-Only F0 (explicit user mark; not fallback)" in summary
    assert calls == {"signal": 0, "preview": 0}
    assert sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*")) == before_files
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()


def test_gui_imported_plan_review_output_policy_warnings_no_live_policy_change(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    unsafe_out = run_dir / "_analysis" / "phasic_out" / "features" / "future"
    path = tmp_path / "unsafe_output_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(
            run_dir,
            rois=["CH1"],
            output_policy=OutputPolicy(
                output_root=str(unsafe_out),
                overwrite=True,
                separate_from_source_required=False,
                legacy_outputs_protected=False,
            ),
        ),
    )

    summary = _review_plan(window, path)

    assert "Output policy warnings:" in summary
    assert "Overwrite is enabled" in summary
    assert "separate_from_source_required is disabled" in summary
    assert "legacy_outputs_protected is disabled" in summary
    assert "inside legacy output directory" in summary
    assert window._build_guided_draft_run_plan()[0].output_policy.output_root is None
    assert not unsafe_out.exists()


def test_gui_imported_plan_review_candidate_clears_on_completed_run_switch(window, tmp_path, monkeypatch):
    run_a = _make_preview_completed_run(tmp_path / "run_a")
    run_b = _make_preview_completed_run(tmp_path / "run_b")
    _load_preview_completed_run(window, run_a, monkeypatch)
    path = tmp_path / "run_a_plan.json"
    _write_guided_plan_json(path, _candidate_plan(run_a, rois=["CH1"]))
    _review_plan(window, path)
    assert window._guided_imported_plan_candidate is not None

    _load_preview_completed_run(window, run_b, monkeypatch)

    assert window._guided_imported_plan_candidate is None
    assert "cleared because the loaded completed run changed" in window._guided_imported_plan_status_label.text()
    assert window._build_guided_draft_run_plan()[0].source.completed_run_dir == str(run_b.resolve())


def test_gui_imported_plan_review_candidate_replaced_on_new_open(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    first = tmp_path / "first_plan.json"
    second = tmp_path / "second_plan.json"
    first_plan = _candidate_plan(run_dir, rois=["CH1"])
    first_plan.plan_id = "first"
    second_plan = _candidate_plan(run_dir, rois=["CH2"])
    second_plan.plan_id = "second"
    _write_guided_plan_json(first, first_plan)
    _write_guided_plan_json(second, second_plan)

    first_summary = _review_plan(window, first)
    second_summary = _review_plan(window, second)

    assert "Plan ID: first" in first_summary
    assert "Plan ID: second" in second_summary
    assert "Plan ID: first" not in second_summary
    assert window._guided_imported_plan_file_path == str(second.resolve())


def test_gui_imported_plan_review_non_output_guarantee(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    before_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    _load_preview_completed_run(window, run_dir, monkeypatch)
    output_root = tmp_path / "candidate_output"
    path = tmp_path / "readonly_plan.json"
    _write_guided_plan_json(
        path,
        _candidate_plan(run_dir, rois=["CH1"], profile=True, output_policy=OutputPolicy(output_root=str(output_root))),
    )

    _review_plan(window, path)

    after_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after_files == before_files
    assert not output_root.exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "features.csv").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()


def test_gui_no_draft_plan_tests_use_hidden_confirm_controls():
    import ast
    import os
    test_file = __file__
    with open(test_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=test_file)

    confirm_widgets = {
        "_guided_confirm_roi_combo",
        "_guided_confirm_chunk_combo",
        "_guided_confirm_strategy_combo",
        "_guided_confirm_ack_cb",
        "_guided_confirm_mark_btn",
        "_guided_confirm_evidence_label",
        "_guided_confirm_marked_choice_label",
    }

    class TestVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if not node.name.startswith("test_"):
                return

            current_step = None

            for stmt in node.body:
                for subnode in ast.walk(stmt):
                    if isinstance(subnode, ast.Call):
                        if (isinstance(subnode.func, ast.Attribute) and
                            subnode.func.attr == "setCurrentRow"):
                            for arg in ast.walk(subnode):
                                if isinstance(arg, ast.Constant) and arg.value in ("Correction approach", "Draft plan"):
                                    current_step = arg.value
                                    break

                    if isinstance(subnode, ast.Attribute):
                        if subnode.attr in confirm_widgets:
                            if current_step == "Draft plan":
                                raise AssertionError(
                                    f"Test {node.name} accesses confirm widget {subnode.attr} "
                                    f"while selected step is 'Draft plan' (line {subnode.lineno})"
                                )
            self.generic_visit(node)

    TestVisitor().visit(tree)


def test_stepper_routing_fresh_mainwindow(window):
    assert window._guided_workflow_mode == "start"


def test_stepper_routing_select_data_promotes(window):
    idx = list(GUIDED_WORKFLOW_STEPS).index("Select data")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_workflow_stack.currentIndex() == idx


def test_stepper_routing_draft_plan_promotes(window):
    idx = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_workflow_stack.currentIndex() == idx


def test_stepper_routing_apply_profile_no_guard_after_promotion(window):
    # Navigate to Select data first to promote
    idx = list(GUIDED_WORKFLOW_STEPS).index("Select data")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert window._guided_workflow_mode == "new_analysis"

    # Now navigate to Draft plan and try to apply profile
    draft_idx = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    window._guided_workflow_stepper.setCurrentRow(draft_idx)

    # Let's set some dummy inputs in the editor so it doesn't fail on empty validation
    window._guided_feature_event_peak_k_edit.setText("2.0")
    window._guided_feature_event_apply_btn.click()
    # It should not show the Open Results guard message
    assert "Open Results must be used first" not in window._guided_feature_event_status_label.text()


def test_stepper_routing_setup_new_analysis_preserves(window):
    window._on_guided_start_setup_new_analysis()
    assert window._guided_workflow_mode == "new_analysis"
    idx = list(GUIDED_WORKFLOW_STEPS).index("Select data")
    assert window._guided_workflow_stepper.currentRow() == idx


def test_stepper_routing_open_results_retains_mode(window):
    # Simulates entering open_results mode (e.g. via helper or directly)
    window._set_guided_workflow_mode("open_results")
    assert window._guided_workflow_mode == "open_results"

    # Navigate to confirm strategy / draft plan
    idx = list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert window._guided_workflow_mode == "open_results"

    # Navigate to Diagnostics
    diag_idx = list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    window._guided_workflow_stepper.setCurrentRow(diag_idx)
    assert window._guided_workflow_mode == "open_results"


def test_stepper_routing_diagnostics_promotes(window):
    idx = list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_workflow_stack.currentIndex() == idx


def test_stepper_routing_confirm_strategy_promotes(window):
    idx = list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    window._guided_workflow_stepper.setCurrentRow(idx)
    assert window._guided_workflow_mode == "new_analysis"
    assert window._guided_workflow_stack.currentIndex() == idx


def test_stepper_routing_invalid_indices_no_crash(window):
    # Test negative index
    window._on_guided_stepper_row_changed(-1)
    # Test out of bounds index
    window._on_guided_stepper_row_changed(100)
    # Mode should still be start
    assert window._guided_workflow_mode == "start"


def test_guided_diagnostics_guidance_new_analysis_no_cache(window, tmp_path, monkeypatch):
    # Set to new_analysis mode
    window._set_guided_workflow_mode("new_analysis")
    # Refresh panel
    window._refresh_guided_diagnostics_panel()

    # Diagnostics page must show cache build is required before Run
    status_text = window._guided_diagnostic_cache_status_label.text()
    readiness_text = window._guided_diagnostic_cache_readiness_label.text()

    assert status_text == "Correction evidence: not built."
    assert "Select raw input folder." in readiness_text or "Run ROI discovery" in readiness_text or "Input directory is required." in readiness_text
    # Ensure it doesn't instruct the user to Open Results for new-analysis cache build
    assert "Open Results" not in status_text


def test_guided_diagnostics_guidance_new_analysis_ready_cache(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._set_guided_workflow_mode("new_analysis")

    # Build/simulate a ready diagnostic cache
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    prelaunch_request = cache_path / "guided_diagnostic_cache_request.json"
    if prelaunch_request.exists():
        prelaunch_request.unlink()
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    # Refresh panel
    window._refresh_guided_diagnostics_panel()

    # Must say diagnostic cache is ready and not present required message
    status_text = window._guided_diagnostic_cache_status_label.text()
    assert status_text == "Correction evidence: ready."
    assert "Required before Run" not in status_text


def test_guided_diagnostics_guidance_new_analysis_stale_cache(window, tmp_path, monkeypatch):
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._set_guided_workflow_mode("new_analysis")

    # Build/simulate a ready diagnostic cache
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    prelaunch_request = cache_path / "guided_diagnostic_cache_request.json"
    if prelaunch_request.exists():
        prelaunch_request.unlink()
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    # Now make it stale by changing a setting (e.g. adding an excluded ROI)
    item = window._roi_list.item(0)
    item.setCheckState(Qt.Unchecked)  # change roi inclusion to trigger stale status

    window._refresh_guided_diagnostics_panel()

    # Must say diagnostic cache is missing or stale. Rebuild the diagnostic cache before Run.
    status_text = window._guided_diagnostic_cache_status_label.text()
    assert status_text == (
        "Correction evidence: stale after setup changes. Rebuild before "
        "continuing."
    )
    assert window._guided_diagnostic_cache_build_btn.text() == (
        "Rebuild correction evidence"
    )


def test_guided_diagnostics_guidance_new_analysis_failed_cache(window):
    from photometry_pipeline.guided_diagnostic_cache import DiagnosticCacheStatus
    window._set_guided_workflow_mode("new_analysis")
    window._guided_diagnostic_cache_status = DiagnosticCacheStatus(
        ok=False,
        code="failed",
        message="Failure reason"
    )
    window._refresh_guided_diagnostics_panel()
    status_text = window._guided_diagnostic_cache_status_label.text()
    assert status_text == (
        "Correction evidence: build failed. Fix the issue and try again."
    )
