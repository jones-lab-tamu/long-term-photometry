import json
import os
import sys
import tempfile

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.main_window import MainWindow
from gui.process_runner import RunnerState
import gui.main_window as main_window_module


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _set_minimally_valid_paths(w: MainWindow):
    w._input_dir.setText("tests/out_manual_complex_5roi_5day_2sph_shared")
    w._config_path.setText("tests/qc_universal_config.yaml")
    w._output_dir.setText(tempfile.mkdtemp(prefix="gui_usability_out_"))


def test_effective_run_summary_updates(window):
    text0 = window._effective_summary_label.text()
    assert "Mode: both" in text0
    assert "Analysis: Full analysis" in text0
    assert "Preview: off" in text0
    assert "Plotting Mode: Standard" in text0

    window._traces_only_cb.setChecked(True)
    window._preview_enabled_cb.setChecked(True)
    window._preview_n_spin.setValue(7)
    window._plotting_mode_combo.setCurrentText("Full")
    text1 = window._effective_summary_label.text()
    assert "Analysis: Traces-only" in text1
    assert "Preview: first N = 7" in text1
    assert "Plotting Mode: Full" in text1

    discovered = {
        "n_total_discovered": 3,
        "n_preview": 3,
        "resolved_format": "rwd",
        "sessions": [
            {"session_id": "S0", "included_in_preview": True},
            {"session_id": "S1", "included_in_preview": True},
            {"session_id": "S2", "included_in_preview": False},
        ],
        "rois": [{"roi_id": "Region0"}, {"roi_id": "Region1"}, {"roi_id": "Region2"}],
    }
    window._discovery_cache = discovered
    window._populate_discovery_ui(discovered)
    window._roi_list.item(0).setCheckState(Qt.Unchecked)  # Include subset
    window._rep_session_combo.setCurrentIndex(2)  # discovery index 1 (S1)
    text2 = window._effective_summary_label.text()
    assert "ROI Filter: Include subset (2/3)" in text2
    assert "Representative Session: Session index 1 (S1)" in text2


def test_run_disabled_reason_text_states(window):
    _set_minimally_valid_paths(window)

    window._validation_passed = False
    window._update_button_states()
    assert "Validation required" in window._run_reason_label.text()

    window._validation_passed = True
    window._update_button_states()
    assert "Ready to run" in window._run_reason_label.text()

    window._sph_edit.setText("0")
    window._validation_passed = True
    window._update_button_states()
    assert "Invalid setting combination:" in window._run_reason_label.text()


def test_fail_closed_terminal_message_is_actionable(window, tmp_path):
    window._current_run_dir = str(tmp_path)
    window._is_validate_only = False
    window._did_finalize_run_ui = False
    window._runner._state = RunnerState.FAIL_CLOSED
    window._runner.fail_closed_code = "MISSING_FILE"
    window._runner.fail_closed_detail = "status.json was never created or is unreadable."
    window._runner.fail_closed_remediation = "Verify run directory is writeable and disk is not full."
    window._runner.final_status_code = "MISSING_FILE"

    window._finalize_run_ui()
    log = window._log_view.toPlainText()
    assert "Run failed (FAIL_CLOSED): MISSING_FILE" in log
    assert "Reason: status.json was never created or is unreadable." in log
    assert "Next step: Verify run directory is writeable and disk is not full." in log
    assert window._last_status_msg.startswith("FAIL_CLOSED:")


def test_advanced_tooltips_present(window):
    def _label(text: str):
        for candidate in window.findChildren(type(window._status_label)):
            if candidate.text() == text:
                return candidate
        return None

    # Run Configuration + Plotting representative rows: label and control both must carry tooltips.
    run_plot_pairs = [
        ("Input Directory:", window._input_dir),
        ("Output Directory:", window._output_dir),
        ("Format:", window._format_combo),
        ("Sessions/Hour:", window._sph_edit),
        ("Session Duration (s):", window._duration_edit),
        ("Mode:", window._mode_combo),
        ("Plotting Mode:", window._plotting_mode_combo),
        ("Smooth Window (s):", window._smooth_spin),
    ]
    for label_text, control in run_plot_pairs:
        label = _label(label_text)
        assert label is not None, f"Missing label {label_text}"
        assert label.toolTip().strip(), f"Missing tooltip on label {label_text}"
        assert control.toolTip().strip(), f"Missing tooltip on control for {label_text}"

    # Advanced rows: guard against control-only tooltips by asserting both label and control.
    advanced_pairs = [
        ("Regression Window:", window._window_sec_edit),
        ("Regression Step:", window._step_sec_edit),
        ("Min Valid Windows:", window._min_valid_windows_spin),
        ("Min Samples per Window:", window._min_samples_per_window_spin),
        ("R-Low Threshold:", window._r_low_edit),
        ("R-High Threshold:", window._r_high_edit),
        ("G-Min Threshold:", window._g_min_edit),
        ("Lowpass Filter:", window._lowpass_hz_edit),
        ("Baseline Method:", window._baseline_method_combo),
        ("Baseline Percentile:", window._baseline_percentile_edit),
        ("F0 Min Value:", window._f0_min_value_edit),
        ("Event Signal:", window._event_signal_combo),
        ("Peak Threshold Method:", window._peak_method_combo),
        ("Peak Threshold K:", window._peak_k_edit),
        ("Peak Threshold Percentile:", window._peak_pct_edit),
        ("Peak Threshold Absolute:", window._peak_abs_edit),
        ("Peak Min Distance:", window._peak_dist_edit),
        ("Peak Pre-Filter:", window._peak_pre_filter_combo),
        ("Event AUC Baseline:", window._event_auc_combo),
        ("Baseline Source:", window._use_custom_config_cb),
        ("Custom Config YAML:", window._config_path),
    ]
    for label_text, control in advanced_pairs:
        label = _label(label_text)
        assert label is not None, f"Missing label {label_text}"
        assert label.toolTip().strip(), f"Missing tooltip on label {label_text}"
        assert control.toolTip().strip(), f"Missing tooltip on control for {label_text}"

    # ROI row-level affordances that are not form rows still need usable tooltips.
    assert window._discover_btn.toolTip().strip()
    assert window._roi_list.toolTip().strip()
    assert window._roi_checked_label.toolTip().strip()
    assert window._config_browse_btn.toolTip().strip()


def test_window_title_progress_bar_and_progress_cap(window, tmp_path):
    assert window.windowTitle() == "Long-Term Photometry Analysis"
    assert window._progress_bar.minimumHeight() >= 24
    assert window._progress_bar.maximumHeight() >= 24
    assert "QProgressBar::chunk" in window._progress_bar.styleSheet()

    # Active running phase should not render 100% prematurely.
    window._ui_state = RunnerState.RUNNING
    window._state_str = RunnerState.RUNNING.value
    window._last_status_phase = "tonic"
    window._last_status_state = "running"
    window._ui_progress_pct = 100  # stale high-water mark should still be clamped in active state
    window._last_status_pct = 100
    window._render_status_label()
    assert window._progress_bar.value() < 100
    assert "100" not in window._progress_bar.format()
    assert "%" in window._progress_bar.format()
    assert "%%" not in window._progress_bar.format()

    # Terminal success should still show 100%.
    window._ui_state = RunnerState.SUCCESS
    window._state_str = RunnerState.SUCCESS.value
    window._render_status_label()
    assert window._progress_bar.value() == 100
    assert "100" in window._progress_bar.format()
    assert "%" in window._progress_bar.format()
    assert "%%" not in window._progress_bar.format()

    # Preview suffix still uses the new base title and resets on New Run.
    run_dir = tmp_path / "run_preview"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text(
        json.dumps({"run_context": {"run_type": "preview"}}),
        encoding="utf-8",
    )
    window._current_run_dir = str(run_dir)
    window._apply_preview_labeling()
    assert window.windowTitle() == "Long-Term Photometry Analysis [PREVIEW]"
    window._on_new_run()
    assert window.windowTitle() == "Long-Term Photometry Analysis"


def test_validate_to_run_progress_does_not_start_at_stale_99(window, monkeypatch, tmp_path):
    """Regression: a successful validate must not make the next run start at 99%."""
    # Simulate completed validate terminal state carrying stale completion progress.
    window._ui_state = RunnerState.SUCCESS
    window._state_str = RunnerState.SUCCESS.value
    window._last_status_phase = "final"
    window._last_status_state = "success"
    window._ui_progress_pct = 100
    window._last_status_pct = 100

    run_dir = tmp_path / "run_after_validate"
    run_dir.mkdir()

    monkeypatch.setattr(window, "_validate_gui_inputs", lambda: None)
    monkeypatch.setattr(window, "_exit_complete_state_workspace", lambda: None)
    monkeypatch.setattr(window, "_save_widgets_to_settings", lambda: None)
    monkeypatch.setattr(window, "_append_run_log", lambda _msg: None)
    monkeypatch.setattr(window, "_start_status_follower", lambda: None)
    monkeypatch.setattr(window, "_start_log_follower", lambda _run_dir: None)

    def _fake_build_argv(validate_only=False, overwrite=False):
        window._current_run_dir = str(run_dir)
        return ["python", "dummy.py"]

    monkeypatch.setattr(window, "_build_argv", _fake_build_argv)
    monkeypatch.setattr(main_window_module, "compute_run_signature", lambda _rd: "sig-now")
    monkeypatch.setattr(main_window_module, "is_validation_current", lambda _a, _b: True)
    window._validated_run_signature = "sig-prev"

    monkeypatch.setattr(window._runner, "set_run_dir", lambda _rd: None)
    monkeypatch.setattr(window._runner, "start", lambda _argv, state: None)

    window._on_run()
    # Simulate first RUNNING render that follows the start call.
    window._on_state_changed(RunnerState.RUNNING.value)

    assert window._progress_bar.value() < 50
    assert window._progress_bar.value() not in (99, 100)
    assert "running" in window._progress_bar.format().lower()


def test_fresh_run_reset_ignores_stale_high_water_state(window):
    """Run-start reset must clear stale completion values before active rendering."""
    window._ui_state = RunnerState.SUCCESS
    window._state_str = RunnerState.SUCCESS.value
    window._last_status_phase = "final"
    window._last_status_state = "success"
    window._ui_progress_pct = 100
    window._last_status_pct = 100

    window._reset_status_flags(next_state=RunnerState.RUNNING)

    assert window._ui_state == RunnerState.RUNNING
    assert window._progress_bar.value() < 50
    assert window._progress_bar.value() not in (99, 100)
    assert "running" in window._progress_bar.format().lower()
