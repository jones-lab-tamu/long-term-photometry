import json
import os
import sys
import tempfile

import pytest
import yaml
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSizePolicy, QGroupBox, QScrollArea, QSplitter, QToolButton

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
    candidate_input = "tests/out_manual_complex_5roi_5day_2sph_shared"
    if not os.path.isdir(candidate_input):
        candidate_input = tempfile.mkdtemp(prefix="gui_usability_in_")
    w._input_dir.setText(candidate_input)
    w._config_path.setText("tests/qc_universal_config.yaml")
    w._output_dir.setText(tempfile.mkdtemp(prefix="gui_usability_out_"))


def _write_png(path: str, width: int = 360, height: int = 220):
    pix = QPixmap(width, height)
    pix.fill(Qt.white)
    assert pix.save(path)


def _make_completed_results_fixture(base_dir: str, *, run_type: str = "full") -> str:
    run_dir = os.path.join(base_dir, "run_complete_results_m3")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "phase": "final", "status": "success"}, f)
    with open(os.path.join(run_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "success"}, f)
    with open(os.path.join(run_dir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump({"run_context": {"run_type": run_type}}, f)

    for region in ("Region0", "Region1"):
        summary = os.path.join(run_dir, region, "summary")
        day_plots = os.path.join(run_dir, region, "day_plots")
        tables = os.path.join(run_dir, region, "tables")
        os.makedirs(summary, exist_ok=True)
        os.makedirs(day_plots, exist_ok=True)
        os.makedirs(tables, exist_ok=True)
        _write_png(os.path.join(summary, "phasic_correction_impact.png"))
        _write_png(os.path.join(summary, "tonic_overview.png"))
        _write_png(os.path.join(summary, "phasic_auc_timeseries.png"))
        _write_png(os.path.join(summary, "phasic_peak_rate_timeseries.png"))
        _write_png(os.path.join(day_plots, "phasic_sig_iso_day_000.png"), 600, 1500)
        _write_png(os.path.join(day_plots, "phasic_dynamic_fit_day_000.png"), 600, 1500)
        _write_png(os.path.join(day_plots, "phasic_dFF_day_000.png"), 600, 1500)
        _write_png(os.path.join(day_plots, "phasic_stacked_day_000.png"), 600, 1500)
    return run_dir


def test_effective_run_summary_updates(window):
    text0 = window._effective_summary_label.text()
    assert "Run Type: Full Run" in text0
    assert "Mode: both" in text0
    assert "Analysis: Full analysis" in text0
    assert "Dynamic Fit Mode: Rolling regression (filtered→raw)" in text0
    assert "Baseline subtract before fit: off" in text0
    assert "Preview: off" in text0
    assert "Plotting Mode: Standard" in text0
    assert "Timeline Anchor: Civil clock" in text0

    window._traces_only_cb.setChecked(True)
    window._preview_enabled_cb.setChecked(True)
    window._preview_n_spin.setValue(7)
    window._plotting_mode_combo.setCurrentText("Full")
    window._timeline_anchor_mode_combo.setCurrentIndex(
        window._timeline_anchor_mode_combo.findData("fixed_daily_anchor")
    )
    window._fixed_daily_anchor_time_edit.setText("07:00")
    text1 = window._effective_summary_label.text()
    assert "Analysis: Traces-only" in text1
    assert "Preview: first N = 7" in text1
    assert "Plotting Mode: Full" in text1
    assert "Timeline Anchor: Fixed daily anchor (07:00)" in text1

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


def test_run_profile_selector_plumbs_tuning_prep_into_run_spec(window):
    _set_minimally_valid_paths(window)
    idx = window._run_profile_combo.findData("tuning_prep")
    assert idx >= 0
    window._run_profile_combo.setCurrentIndex(idx)

    spec = window._build_run_spec(validate_only=True)
    assert spec.run_profile == "tuning_prep"

    summary = window._effective_summary_label.text()
    assert "Run Type: Tuning Prep Run" in summary
    assert "Artifact Contract:" in summary
    assert "tuning" in window._run_profile_note_label.text().lower()


def test_run_profile_selector_rejects_tuning_prep_with_tonic_only(window):
    _set_minimally_valid_paths(window)
    idx_profile = window._run_profile_combo.findData("tuning_prep")
    assert idx_profile >= 0
    window._run_profile_combo.setCurrentIndex(idx_profile)
    window._mode_combo.setCurrentText("tonic")
    err = window._validate_gui_inputs()
    assert err is not None
    assert "phasic-capable mode" in err


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

    def _help_icon(label):
        parent = label.parentWidget()
        if parent is None:
            return None
        return parent.findChild(QToolButton, "formRowHelpIcon")

    # Run Configuration + Plotting representative rows: label and control both must carry tooltips.
    run_plot_pairs = [
        ("Input Directory:", window._input_dir),
        ("Output Directory:", window._output_dir),
        ("Format:", window._format_combo),
        ("Sessions/Hour:", window._sph_edit),
        ("Session Duration (s):", window._duration_edit),
        ("Mode:", window._mode_combo),
        ("Run Type:", window._run_profile_combo),
        ("Plotting Mode:", window._plotting_mode_combo),
        ("Smoothing Window Duration (s):", window._smooth_spin),
        ("Timeline Anchor:", window._timeline_anchor_mode_combo),
        ("Fixed Anchor Time:", window._fixed_daily_anchor_time_edit),
    ]
    for label_text, control in run_plot_pairs:
        label = _label(label_text)
        assert label is not None, f"Missing label {label_text}"
        assert label.toolTip().strip(), f"Missing tooltip on label {label_text}"
        assert control.toolTip().strip(), f"Missing tooltip on control for {label_text}"
        icon = _help_icon(label)
        assert icon is not None, f"Missing help icon for {label_text}"
        assert icon.toolTip().strip(), f"Missing help icon tooltip for {label_text}"

    # Advanced rows: guard against control-only tooltips by asserting both label and control.
    advanced_pairs = [
        ("Dynamic Fit Mode:", window._dynamic_fit_mode_combo),
        ("Baseline subtract before fit:", window._baseline_subtract_before_fit_cb),
        ("Regression Window:", window._window_sec_edit),
        ("Min Samples per Window:", window._min_samples_per_window_spin),
        ("Lowpass Filter:", window._lowpass_hz_edit),
        ("Baseline Method:", window._baseline_method_combo),
        ("Baseline Percentile:", window._baseline_percentile_edit),
        ("F0 Min Value:", window._f0_min_value_edit),
        ("Event Signal:", window._event_signal_combo),
        ("Signal Excursion Polarity:", window._signal_excursion_polarity_combo),
        ("Peak Threshold Method:", window._peak_method_combo),
        ("Peak Threshold K:", window._peak_k_edit),
        ("Peak Threshold Percentile:", window._peak_pct_edit),
        ("Peak Threshold Absolute:", window._peak_abs_edit),
        ("Peak Min Distance:", window._peak_dist_edit),
        ("Peak Min Prominence K:", window._peak_min_prominence_k_edit),
        ("Peak Min Width (s):", window._peak_min_width_sec_edit),
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
        icon = _help_icon(label)
        assert icon is not None, f"Missing help icon for {label_text}"
        assert icon.toolTip().strip(), f"Missing help icon tooltip for {label_text}"
    timeline_tip = window._timeline_anchor_mode_combo.toolTip().lower()
    assert "civil clock" in timeline_tip
    assert "elapsed from first session" in timeline_tip
    assert "fixed daily anchor" in timeline_tip
    polarity_tip = window._signal_excursion_polarity_combo.toolTip().lower()
    assert "event detection" in polarity_tip
    assert "dynamic-fit protection" in polarity_tip
    assert "both polarities" in polarity_tip
    auc_tip = window._event_auc_combo.toolTip().lower()
    assert "auc sign follows signal excursion polarity" in auc_tip
    assert "signed net area" in auc_tip
    inline_anchor_help_lines = window._timeline_anchor_help_label.text().splitlines()
    assert inline_anchor_help_lines == [
        "Civil clock: real time-of-day",
        "Elapsed from first session: starts at 0",
        "Fixed daily anchor: align each day to one clock time",
    ]

    removed_legacy_labels = [
        "Regression Step:",
        "Min Valid Windows:",
        "R-Low Threshold:",
        "R-High Threshold:",
        "G-Min Threshold:",
    ]
    for label_text in removed_legacy_labels:
        assert _label(label_text) is None, f"Legacy label should be removed: {label_text}"

    # ROI row-level affordances that are not form rows still need usable tooltips.
    assert window._discover_btn.toolTip().strip()
    assert window._roi_list.toolTip().strip()
    assert window._roi_checked_label.toolTip().strip()
    assert window._config_browse_btn.toolTip().strip()
    assert window.findChildren(QToolButton, "formRowHelpIcon")


def test_gui_dynamic_fit_mode_default_and_global_override_in_run_spec(window):
    _set_minimally_valid_paths(window)
    assert window._dynamic_fit_mode_combo.currentData() == "rolling_filtered_to_raw"
    assert not window._baseline_subtract_before_fit_cb.isChecked()

    visible_modes = [
        window._dynamic_fit_mode_combo.itemData(i)
        for i in range(window._dynamic_fit_mode_combo.count())
    ]
    assert visible_modes[:3] == [
        "rolling_filtered_to_raw",
        "rolling_filtered_to_filtered",
        "global_linear_regression",
    ]
    assert "robust_global_event_reject" in visible_modes
    assert "adaptive_event_gated_regression" in visible_modes

    spec_default = window._build_run_spec(validate_only=True)
    assert spec_default.config_overrides.get("dynamic_fit_mode") is None
    assert "window_sec" not in spec_default.config_overrides

    window._baseline_subtract_before_fit_cb.setChecked(True)
    spec_roll_baseline = window._build_run_spec(validate_only=True)
    assert spec_roll_baseline.config_overrides.get("baseline_subtract_before_fit") is True

    idx = window._dynamic_fit_mode_combo.findData("rolling_filtered_to_filtered")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    spec_filtered = window._build_run_spec(validate_only=True)
    overrides_filtered = dict(spec_filtered.config_overrides)
    assert overrides_filtered.get("dynamic_fit_mode") == "rolling_filtered_to_filtered"
    assert overrides_filtered.get("baseline_subtract_before_fit") is True
    assert window._window_sec_edit.isEnabled()
    assert window._min_samples_per_window_spin.isEnabled()
    assert window._baseline_subtract_before_fit_cb.isEnabled()

    idx = window._dynamic_fit_mode_combo.findData("global_linear_regression")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)

    spec_global = window._build_run_spec(validate_only=True)
    overrides = dict(spec_global.config_overrides)
    assert overrides.get("dynamic_fit_mode") == "global_linear_regression"
    assert "window_sec" not in overrides
    assert "min_samples_per_window" not in overrides
    assert "baseline_subtract_before_fit" not in overrides
    assert not window._window_sec_edit.isEnabled()
    assert not window._min_samples_per_window_spin.isEnabled()
    assert not window._baseline_subtract_before_fit_cb.isEnabled()

    run_dir = tempfile.mkdtemp(prefix="gui_dynamic_fit_mode_")
    cfg_path = spec_global.generate_derived_config(run_dir)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    assert cfg.get("dynamic_fit_mode") == "global_linear_regression"


def test_gui_dynamic_fit_mode_robust_event_reject_plumbs_mode_specific_overrides(window):
    _set_minimally_valid_paths(window)

    idx = window._dynamic_fit_mode_combo.findData("robust_global_event_reject")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    window._robust_event_reject_max_iters_spin.setValue(4)
    window._robust_event_reject_residual_z_spin.setValue(3.1)
    window._robust_event_reject_local_var_window_spin.setValue(9.0)
    window._robust_event_reject_local_var_ratio_enable_cb.setChecked(True)
    window._robust_event_reject_local_var_ratio_spin.setValue(4.2)
    window._robust_event_reject_min_keep_fraction_spin.setValue(0.6)

    assert not window._window_sec_edit.isEnabled()
    assert not window._min_samples_per_window_spin.isEnabled()
    assert not window._baseline_subtract_before_fit_cb.isEnabled()
    assert window._robust_event_reject_max_iters_spin.isEnabled()
    assert window._robust_event_reject_residual_z_spin.isEnabled()
    assert window._robust_event_reject_local_var_window_spin.isEnabled()
    assert window._robust_event_reject_local_var_ratio_enable_cb.isEnabled()
    assert window._robust_event_reject_local_var_ratio_spin.isEnabled()
    assert window._robust_event_reject_min_keep_fraction_spin.isEnabled()

    spec = window._build_run_spec(validate_only=True)
    overrides = dict(spec.config_overrides)
    assert overrides.get("dynamic_fit_mode") == "robust_global_event_reject"
    assert overrides.get("robust_event_reject_max_iters") == 4
    assert overrides.get("robust_event_reject_residual_z_thresh") == pytest.approx(3.1)
    assert overrides.get("robust_event_reject_local_var_window_sec") == pytest.approx(9.0)
    assert overrides.get("robust_event_reject_local_var_ratio_thresh") == pytest.approx(4.2)
    assert overrides.get("robust_event_reject_min_keep_fraction") == pytest.approx(0.6)
    assert "window_sec" not in overrides
    assert "min_samples_per_window" not in overrides
    assert "baseline_subtract_before_fit" not in overrides
    summary = window._effective_summary_label.text()
    assert "Robust event-reject settings:" in summary
    assert "max_iters=4" in summary
    assert "residual_z=3.1" in summary
    assert "local_var_window_s=9" in summary
    assert "local_var_ratio=4.2" in summary
    assert "min_keep=0.6" in summary


def test_gui_dynamic_fit_mode_adaptive_event_gated_plumbs_mode_specific_overrides(window):
    _set_minimally_valid_paths(window)

    idx = window._dynamic_fit_mode_combo.findData("adaptive_event_gated_regression")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    window._adaptive_event_gate_residual_z_spin.setValue(3.3)
    window._adaptive_event_gate_local_var_window_spin.setValue(9.5)
    window._adaptive_event_gate_local_var_ratio_enable_cb.setChecked(True)
    window._adaptive_event_gate_local_var_ratio_spin.setValue(4.1)
    window._adaptive_event_gate_smooth_window_spin.setValue(72.0)
    window._adaptive_event_gate_min_trust_fraction_spin.setValue(0.62)
    idx_interp = window._adaptive_event_gate_freeze_interp_combo.findData("linear_hold")
    assert idx_interp >= 0
    window._adaptive_event_gate_freeze_interp_combo.setCurrentIndex(idx_interp)

    assert not window._window_sec_edit.isEnabled()
    assert not window._min_samples_per_window_spin.isEnabled()
    assert not window._baseline_subtract_before_fit_cb.isEnabled()
    assert not window._robust_event_reject_max_iters_spin.isEnabled()
    assert window._adaptive_event_gate_residual_z_spin.isEnabled()
    assert window._adaptive_event_gate_local_var_window_spin.isEnabled()
    assert window._adaptive_event_gate_local_var_ratio_enable_cb.isEnabled()
    assert window._adaptive_event_gate_local_var_ratio_spin.isEnabled()
    assert window._adaptive_event_gate_smooth_window_spin.isEnabled()
    assert window._adaptive_event_gate_min_trust_fraction_spin.isEnabled()
    assert window._adaptive_event_gate_freeze_interp_combo.isEnabled()

    spec = window._build_run_spec(validate_only=True)
    overrides = dict(spec.config_overrides)
    assert overrides.get("dynamic_fit_mode") == "adaptive_event_gated_regression"
    assert overrides.get("adaptive_event_gate_residual_z_thresh") == pytest.approx(3.3)
    assert overrides.get("adaptive_event_gate_local_var_window_sec") == pytest.approx(9.5)
    assert overrides.get("adaptive_event_gate_local_var_ratio_thresh") == pytest.approx(4.1)
    assert overrides.get("adaptive_event_gate_smooth_window_sec") == pytest.approx(72.0)
    assert overrides.get("adaptive_event_gate_min_trust_fraction") == pytest.approx(0.62)
    assert overrides.get("adaptive_event_gate_freeze_interp_method") == "linear_hold"
    assert "window_sec" not in overrides
    assert "min_samples_per_window" not in overrides
    assert "baseline_subtract_before_fit" not in overrides
    summary = window._effective_summary_label.text()
    assert "Adaptive event-gated settings:" in summary
    assert "residual_z=3.3" in summary
    assert "local_var_window_s=9.5" in summary
    assert "local_var_ratio=4.1" in summary
    assert "smooth_window_s=72" in summary
    assert "min_trust=0.62" in summary
    assert "freeze_interp=linear_hold" in summary
    assert "Baseline subtract before fit: inactive in adaptive event-gated mode" in summary


def test_gui_dynamic_fit_mode_legacy_alias_in_settings_normalizes_to_filtered_to_raw(window):
    window._settings.beginGroup("run_config")
    window._settings.setValue("dynamic_fit_mode", "rolling_local_regression")
    window._settings.endGroup()
    window._settings.sync()
    window._load_settings_into_widgets()
    assert window._dynamic_fit_mode_combo.currentData() == "rolling_filtered_to_raw"


def test_gui_build_argv_accepts_baseline_subtract_before_fit_override(window, tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(output_dir))
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText("tests/qc_universal_config.yaml")

    idx_mode = window._dynamic_fit_mode_combo.findData("rolling_filtered_to_raw")
    assert idx_mode >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx_mode)
    window._baseline_subtract_before_fit_cb.setChecked(True)

    # Keep this test focused on override allowlisting instead of dataset inference.
    monkeypatch.setattr(window, "_infer_dataset_contract_overrides", lambda _fmt: {})

    assert window._validate_gui_inputs() is None
    argv = window._build_argv(validate_only=True)
    assert "--validate-only" in argv

    cfg_path = os.path.join(window._current_run_dir, "config_effective.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    assert cfg.get("baseline_subtract_before_fit") is True


def test_gui_build_argv_accepts_adaptive_event_gated_overrides(window, tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(output_dir))
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText("tests/qc_universal_config.yaml")

    idx_mode = window._dynamic_fit_mode_combo.findData("adaptive_event_gated_regression")
    assert idx_mode >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx_mode)
    window._adaptive_event_gate_residual_z_spin.setValue(3.2)
    window._adaptive_event_gate_local_var_window_spin.setValue(9.0)
    window._adaptive_event_gate_local_var_ratio_enable_cb.setChecked(True)
    window._adaptive_event_gate_local_var_ratio_spin.setValue(4.2)
    window._adaptive_event_gate_smooth_window_spin.setValue(75.0)
    window._adaptive_event_gate_min_trust_fraction_spin.setValue(0.6)
    idx_interp = window._adaptive_event_gate_freeze_interp_combo.findData("linear_hold")
    assert idx_interp >= 0
    window._adaptive_event_gate_freeze_interp_combo.setCurrentIndex(idx_interp)

    # Keep this test focused on override allowlisting instead of dataset inference.
    monkeypatch.setattr(window, "_infer_dataset_contract_overrides", lambda _fmt: {})

    assert window._validate_gui_inputs() is None
    argv = window._build_argv(validate_only=True)
    assert "--validate-only" in argv

    cfg_path = os.path.join(window._current_run_dir, "config_effective.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    assert cfg.get("dynamic_fit_mode") == "adaptive_event_gated_regression"
    assert cfg.get("adaptive_event_gate_residual_z_thresh") == pytest.approx(3.2)
    assert cfg.get("adaptive_event_gate_local_var_window_sec") == pytest.approx(9.0)
    assert cfg.get("adaptive_event_gate_local_var_ratio_thresh") == pytest.approx(4.2)
    assert cfg.get("adaptive_event_gate_smooth_window_sec") == pytest.approx(75.0)
    assert cfg.get("adaptive_event_gate_min_trust_fraction") == pytest.approx(0.6)
    assert cfg.get("adaptive_event_gate_freeze_interp_method") == "linear_hold"


def test_gui_chunk_discovery_survives_cache_reader_runtime_error(window, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    cache_path = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"placeholder")
    window._current_run_dir = str(run_dir)
    window._roi_chunk_ids_cache = {}

    def _raise_runtime_error(_cache_path):
        raise RuntimeError("malformed cache")

    monkeypatch.setattr(main_window_module, "open_phasic_cache", _raise_runtime_error)

    assert window._chunk_ids_for_roi("Region0") == []


def test_gui_timeline_anchor_controls_propagate_to_run_spec(window):
    _set_minimally_valid_paths(window)

    # Default is explicit civil anchor.
    spec_default = window._build_run_spec(validate_only=True)
    assert spec_default.timeline_anchor_mode == "civil"
    assert spec_default.fixed_daily_anchor_clock is None
    argv_default = spec_default.build_runner_argv()
    assert "--timeline-anchor-mode" not in argv_default
    assert "--fixed-daily-anchor-clock" not in argv_default

    # Fixed daily anchor: carry mode + clock through run spec/argv.
    window._timeline_anchor_mode_combo.setCurrentIndex(
        window._timeline_anchor_mode_combo.findData("fixed_daily_anchor")
    )
    window._fixed_daily_anchor_time_edit.setText("07:00")
    spec_fixed = window._build_run_spec(validate_only=True)
    assert spec_fixed.timeline_anchor_mode == "fixed_daily_anchor"
    assert spec_fixed.fixed_daily_anchor_clock == "07:00"
    argv_fixed = spec_fixed.build_runner_argv()
    assert "--timeline-anchor-mode" in argv_fixed
    assert argv_fixed[argv_fixed.index("--timeline-anchor-mode") + 1] == "fixed_daily_anchor"
    assert "--fixed-daily-anchor-clock" in argv_fixed
    assert argv_fixed[argv_fixed.index("--fixed-daily-anchor-clock") + 1] == "07:00"

    # Elapsed anchor: mode flag only.
    window._timeline_anchor_mode_combo.setCurrentIndex(
        window._timeline_anchor_mode_combo.findData("elapsed")
    )
    spec_elapsed = window._build_run_spec(validate_only=True)
    assert spec_elapsed.timeline_anchor_mode == "elapsed"
    assert spec_elapsed.fixed_daily_anchor_clock is None
    argv_elapsed = spec_elapsed.build_runner_argv()
    assert "--timeline-anchor-mode" in argv_elapsed
    assert argv_elapsed[argv_elapsed.index("--timeline-anchor-mode") + 1] == "elapsed"
    assert "--fixed-daily-anchor-clock" not in argv_elapsed


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


def test_results_workspace_idle_placeholder_is_deliberate(window):
    status_text = window._report_viewer._status_label.text()
    assert "Results workspace" in status_text
    assert "Run the pipeline or open a completed run folder." in status_text
    assert "Results and plots appear here after completion." in status_text


def test_results_workspace_m3_sections_are_structured(window):
    assert hasattr(window, "_results_summary_group")
    assert hasattr(window, "_results_views_group")
    assert hasattr(window, "_tuning_group")
    assert hasattr(window, "_open_results_btn")
    assert window._open_results_btn.text() == "Open Results..."
    assert window._results_summary_group.title() == "Run Summary"
    assert window._results_views_group.title() == "Analysis Outputs"
    assert window._results_summary_group.parentWidget() is window._results_pane
    assert window._results_views_group.parentWidget() is window._results_pane
    assert window._report_viewer.parentWidget() is window._results_views_group
    assert hasattr(window._report_viewer, "_region_combo")
    assert hasattr(window._report_viewer, "_open_run_report_btn")
    assert window._report_viewer._open_run_report_btn.text() == "Run Report"
    assert (
        window._report_viewer._open_run_report_btn.parentWidget()
        is window._report_viewer._region_combo.parentWidget()
    )


def test_results_workspace_m3_idle_summary_is_intentional(window):
    assert "No completed run loaded" in window._results_summary_title_label.text()
    assert window._results_summary_run_value.text() == "(none)"
    assert window._results_summary_state_value.text() == "Idle"
    assert not hasattr(window, "_results_summary_roi_value")
    assert not hasattr(window, "_results_summary_views_value")
    assert not window._results_summary_compact_label.isVisible()
    assert "open completed results" in window._results_summary_hint_label.text().lower()
    assert not window._results_summary_details_widget.isHidden()


def test_results_workspace_m3_loaded_summary_and_roi_context(window, qapp, tmp_path):
    run_dir = _make_completed_results_fixture(str(tmp_path))
    window._current_run_dir = run_dir
    assert window._report_viewer.load_report(run_dir)
    window._enter_complete_state_workspace()
    qapp.processEvents()

    assert "Completed results loaded." in window._results_summary_title_label.text()
    assert window._results_summary_run_value.text() == os.path.basename(run_dir)
    assert "Complete-state workspace" in window._results_summary_state_value.text()
    assert not window._results_summary_details_widget.isVisibleTo(window)
    assert window._results_summary_group.maximumHeight() <= 80
    assert window._results_summary_compact_label.isVisibleTo(window)
    assert "Run:" in window._results_summary_compact_label.text()
    assert "Complete-state workspace" in window._results_summary_compact_label.text()
    assert not window._results_views_hint_label.isVisibleTo(window)
    assert window._report_viewer._open_run_report_btn.isEnabled()

    window._report_viewer._region_combo.setCurrentText("Region1")
    qapp.processEvents()
    assert window._report_viewer._region_combo.currentText() == "Region1"


def test_results_workspace_m3_loaded_summary_labels_tuning_prep(window, qapp, tmp_path):
    run_dir = _make_completed_results_fixture(str(tmp_path), run_type="tuning_prep")
    window._current_run_dir = run_dir
    assert window._report_viewer.load_report(run_dir)
    window._enter_complete_state_workspace()
    window._apply_preview_labeling()
    qapp.processEvents()

    assert "TUNING PREP" in window._results_summary_state_value.text()
    assert "Tuning Prep Run" in window._results_summary_compact_label.text()
    assert "Run type: Tuning Prep Run" in window._complete_summary_label.text()
    assert window.windowTitle().endswith("[TUNING PREP]")


def test_results_workspace_m3_fit_layout_prioritizes_outputs(window, qapp, tmp_path):
    run_dir = _make_completed_results_fixture(str(tmp_path))
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()

    window._current_run_dir = run_dir
    assert window._report_viewer.load_report(run_dir)
    window._enter_complete_state_workspace()
    qapp.processEvents()

    assert window._results_summary_group.maximumHeight() <= 130
    assert window._results_summary_group.sizePolicy().verticalPolicy() == QSizePolicy.Maximum
    assert window._results_views_group.sizePolicy().verticalPolicy() == QSizePolicy.Expanding
    assert window._results_layout.stretch(1) == 1
    assert window._tuning_group.maximumHeight() <= 200

    viewer = window._report_viewer
    assert viewer._tabs.maximumHeight() <= 50
    assert viewer._image_scroll.sizePolicy().verticalPolicy() == QSizePolicy.Expanding
    group_titles = {g.title() for g in viewer.findChildren(QGroupBox)}
    assert "Selected Region Actions" not in group_titles
    assert viewer._open_run_report_btn.isEnabled()
    assert viewer._open_region_summary_btn.isEnabled()
    assert viewer._open_region_day_plots_btn.isEnabled()
    assert viewer._open_region_tables_btn.isEnabled()

    results_geo = window._results_views_group.geometry()
    tuning_geo = window._tuning_group.geometry()
    assert tuning_geo.top() >= results_geo.bottom() - 1
    assert viewer._image_scroll.geometry().height() >= 250


def test_results_workspace_running_placeholder_is_present(window):
    window.resize(1200, 850)
    window.show()
    window._ui_state = RunnerState.RUNNING
    window._on_run_started()
    qapp = QApplication.instance()
    if qapp is not None:
        qapp.processEvents()
    status_text = window._report_viewer._status_label.text()
    assert "Results workspace" in status_text
    assert "Run in progress..." in status_text
    assert "Results and plots will appear here after completion." in status_text
    assert window._results_summary_details_widget.isVisibleTo(window)
    assert not window._results_summary_compact_label.isVisibleTo(window)
    assert window._results_summary_run_value.height() >= max(1, window._results_summary_run_value.sizeHint().height() - 2)
    assert window._results_summary_state_value.height() >= max(1, window._results_summary_state_value.sizeHint().height() - 2)


def test_m1_shell_sections_and_header_structure(window, qapp):
    window.show()
    qapp.processEvents()

    group_titles = {g.title() for g in window.findChildren(QGroupBox)}
    assert "Run Configuration" in group_titles
    assert "Plotting" in group_titles
    assert "Advanced" in group_titles
    assert "Live Log" in group_titles
    assert "Results" in group_titles

    assert window._status_header_card is not None
    assert window._status_header_card.objectName() == "statusHeaderCard"
    assert window._status_label.parentWidget() is not None
    assert window._progress_bar.parentWidget() is not None
    assert window._status_header_card.isVisibleTo(window)


def test_sessions_per_hour_warning_is_contextual_and_wrapped(window, qapp):
    window.show()
    qapp.processEvents()

    assert hasattr(window, "_sph_edit")
    assert hasattr(window, "_sph_warning")
    assert hasattr(window, "_sph_field_container")
    assert window._sph_warning.text() == "Required for duty-cycled data unless timestamps are available."
    assert window._sph_warning.wordWrap()
    assert window._sph_warning.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert window._sph_warning.isVisibleTo(window)
    assert window._sph_warning.parentWidget() is window._sph_field_container
    assert window._sph_edit.parentWidget() is window._sph_field_container


def test_live_log_disclosure_defaults_collapsed(window, qapp):
    window.show()
    qapp.processEvents()

    assert hasattr(window, "_live_log_disclosure_btn")
    assert hasattr(window, "_live_log_content_container")
    assert hasattr(window, "_live_log_collapsed_hint")
    assert hasattr(window, "_log_view")
    assert not window._live_log_disclosure_btn.isChecked()
    assert not window._live_log_content_container.isVisibleTo(window)
    assert window._live_log_collapsed_hint.isVisibleTo(window)


def test_live_log_disclosure_toggle_and_log_content_persists(window, qapp):
    window.show()
    qapp.processEvents()

    window._log_view.clear()
    window._live_log_disclosure_btn.setChecked(False)
    qapp.processEvents()
    assert not window._live_log_content_container.isVisibleTo(window)

    window._append_log("line while collapsed")
    assert "line while collapsed" in window._log_view.toPlainText()

    window._live_log_disclosure_btn.setChecked(True)
    qapp.processEvents()
    assert window._live_log_content_container.isVisibleTo(window)
    assert "line while collapsed" in window._log_view.toPlainText()

    window._append_log("line while expanded")
    window._live_log_disclosure_btn.setChecked(False)
    qapp.processEvents()
    window._live_log_disclosure_btn.setChecked(True)
    qapp.processEvents()
    log_text = window._log_view.toPlainText()
    assert "line while collapsed" in log_text
    assert "line while expanded" in log_text


def test_live_log_disclosure_keeps_workflow_sections_accessible(window, qapp):
    window.show()
    qapp.processEvents()

    window._live_log_disclosure_btn.setChecked(False)
    qapp.processEvents()
    assert window._run_config_group.isVisibleTo(window)
    assert window._plotting_group.isVisibleTo(window)
    assert window._advanced_group.isVisibleTo(window)


def _left_column_width_snapshot(window):
    controls_scroll = window.findChild(QScrollArea, "workflowControlsScroll")
    assert controls_scroll is not None
    live_log = window.findChild(QGroupBox, "liveLogSection")
    assert live_log is not None
    return {
        "left_pane": window._left_pane.width(),
        "controls_viewport": controls_scroll.viewport().width(),
        "run_config": window._run_config_group.width(),
        "plotting": window._plotting_group.width(),
        "advanced": window._advanced_group.width(),
        "live_log": live_log.width(),
    }


def test_left_column_width_clamps_with_advanced_collapsed_and_expanded(window, qapp):
    window.show()
    window.resize(1100, 850)
    qapp.processEvents()

    window._advanced_disclosure_btn.setChecked(False)
    qapp.processEvents()
    collapsed = _left_column_width_snapshot(window)

    assert collapsed["run_config"] <= collapsed["controls_viewport"] + 2
    assert collapsed["plotting"] <= collapsed["controls_viewport"] + 2
    assert collapsed["advanced"] <= collapsed["controls_viewport"] + 2
    assert collapsed["live_log"] <= collapsed["left_pane"] + 2

    window._advanced_disclosure_btn.setChecked(True)
    qapp.processEvents()
    qapp.processEvents()
    expanded = _left_column_width_snapshot(window)

    assert expanded["run_config"] <= expanded["controls_viewport"] + 2
    assert expanded["plotting"] <= expanded["controls_viewport"] + 2
    assert expanded["advanced"] <= expanded["controls_viewport"] + 2
    assert expanded["live_log"] <= expanded["left_pane"] + 2


def test_advanced_toggle_does_not_widen_sibling_sections(window, qapp):
    window.show()
    window.resize(1100, 850)
    qapp.processEvents()

    window._advanced_disclosure_btn.setChecked(False)
    qapp.processEvents()
    collapsed = _left_column_width_snapshot(window)

    window._advanced_disclosure_btn.setChecked(True)
    qapp.processEvents()
    qapp.processEvents()
    expanded = _left_column_width_snapshot(window)

    assert expanded["run_config"] <= collapsed["run_config"] + 2
    assert expanded["plotting"] <= collapsed["plotting"] + 2
    assert expanded["live_log"] <= collapsed["live_log"] + 2


def test_shell_splitter_handle_is_non_interactive(window, qapp):
    window.show()
    window.resize(1400, 900)
    qapp.processEvents()

    splitter = window._main_splitter
    assert isinstance(splitter, QSplitter)
    handle = splitter.handle(1)
    assert handle is not None
    assert not handle.isEnabled()
    assert handle.testAttribute(Qt.WA_TransparentForMouseEvents)

    before = splitter.sizes()
    center = handle.rect().center()
    drag_to = QPoint(center.x() + 160, center.y())
    QTest.mousePress(handle, Qt.LeftButton, Qt.NoModifier, center)
    QTest.mouseMove(handle, drag_to)
    QTest.mouseRelease(handle, Qt.LeftButton, Qt.NoModifier, drag_to)
    qapp.processEvents()
    after = splitter.sizes()
    assert abs(after[0] - before[0]) <= 2
    assert abs(after[1] - before[1]) <= 2


def test_splitter_left_pane_collapse_expand_restores_prior_width(window, qapp):
    window.show()
    window.resize(1400, 900)
    qapp.processEvents()

    window._left_pane_collapsed = False
    window._right_pane_collapsed = False
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    before_left, before_right = window._main_splitter.sizes()
    assert before_left > 0
    assert before_right > 0
    assert window._left_pane_toggle_btn.text() == "Hide Workflow"

    window._toggle_left_pane_collapsed()
    qapp.processEvents()
    collapsed_left, collapsed_right = window._main_splitter.sizes()
    assert collapsed_left <= 2
    assert collapsed_right >= before_right
    assert window._left_pane_toggle_btn.text() == "Show Workflow"

    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    collapsed_left_2, _ = window._main_splitter.sizes()
    assert collapsed_left_2 <= 2

    window._toggle_left_pane_collapsed()
    qapp.processEvents()
    restored_left, restored_right = window._main_splitter.sizes()
    assert restored_left > 0
    assert restored_right > 0
    assert abs(restored_left - before_left) <= 24
    assert window._left_pane_toggle_btn.text() == "Hide Workflow"


def test_splitter_right_pane_collapse_expand_restores_prior_width_and_survives_policy(window, qapp):
    window.show()
    window.resize(1400, 900)
    qapp.processEvents()

    window._left_pane_collapsed = False
    window._right_pane_collapsed = False
    window._validation_passed = False
    window._ui_state = RunnerState.IDLE
    window._is_complete_workspace_active = False
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    before_left, before_right = window._main_splitter.sizes()
    assert before_left > 0
    assert before_right > 0
    assert window._right_pane_toggle_btn.text() == "Hide Results"

    window._toggle_right_pane_collapsed()
    qapp.processEvents()
    collapsed_left, collapsed_right = window._main_splitter.sizes()
    assert collapsed_right <= 2
    assert collapsed_left >= before_left
    assert window._right_pane_toggle_btn.text() == "Show Results"

    window._validation_passed = True
    window._ui_state = RunnerState.SUCCESS
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    after_policy_left, after_policy_right = window._main_splitter.sizes()
    assert after_policy_right <= 2
    assert after_policy_left > 0

    window.resize(1280, 880)
    qapp.processEvents()
    after_resize_left, after_resize_right = window._main_splitter.sizes()
    assert after_resize_right <= 2
    assert after_resize_left > 0

    window._toggle_right_pane_collapsed()
    qapp.processEvents()
    restored_left, restored_right = window._main_splitter.sizes()
    assert restored_left > 0
    assert restored_right > 0
    assert abs(restored_right - before_right) <= 32
    assert window._right_pane_toggle_btn.text() == "Hide Results"


def test_setup_vs_workspace_right_pane_width(window, qapp):
    window.show()
    window.resize(1400, 900)
    qapp.processEvents()

    window._validation_passed = False
    window._ui_state = RunnerState.IDLE
    window._is_complete_workspace_active = False
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    setup_left = window._left_pane.width()
    setup_right = window._results_pane.width()

    window._validation_passed = True
    window._ui_state = RunnerState.SUCCESS
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    workspace_left = window._left_pane.width()
    workspace_right = window._results_pane.width()

    assert workspace_right > setup_right
    assert workspace_left < setup_left


def test_complete_results_width_not_narrower_than_validate(window, qapp):
    window.show()
    window.resize(1400, 900)
    qapp.processEvents()

    window._validation_passed = True
    window._ui_state = RunnerState.SUCCESS
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    validate_right = window._results_pane.width()

    window._validation_passed = False
    window._ui_state = RunnerState.RUNNING
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    running_right = window._results_pane.width()

    window._enter_complete_state_workspace()
    qapp.processEvents()
    complete_right = window._results_pane.width()

    assert abs(running_right - validate_right) <= 4
    assert complete_right >= validate_right - 2
    assert complete_right >= running_right - 2


def test_shell_left_pane_width_floor_in_setup_and_workspace(window, qapp):
    window.show()
    window.resize(1300, 900)
    qapp.processEvents()

    setup_floor = max(420, getattr(window, "_shell_left_width_floor", 500))
    workspace_floor = max(
        420,
        min(
            setup_floor,
            getattr(window, "_shell_workspace_left_floor", setup_floor),
        ),
    )

    window._validation_passed = False
    window._ui_state = RunnerState.IDLE
    window._is_complete_workspace_active = False
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    setup_left = window._left_pane.width()

    window._validation_passed = True
    window._ui_state = RunnerState.SUCCESS
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    workspace_left = window._left_pane.width()

    assert setup_left >= setup_floor
    assert workspace_left >= workspace_floor


def test_shell_restores_setup_mode_after_new_run(window, qapp):
    window.show()
    window.resize(1400, 900)
    qapp.processEvents()

    window._validation_passed = False
    window._ui_state = RunnerState.IDLE
    window._is_complete_workspace_active = False
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    setup_left = window._left_pane.width()
    setup_right = window._results_pane.width()

    window._validation_passed = True
    window._ui_state = RunnerState.SUCCESS
    window._refresh_splitter_workspace_policy()
    qapp.processEvents()
    workspace_left = window._left_pane.width()
    workspace_right = window._results_pane.width()

    window._enter_complete_state_workspace()
    qapp.processEvents()
    window._on_new_run()
    qapp.processEvents()

    restored_left = window._left_pane.width()
    restored_right = window._results_pane.width()

    assert workspace_right > setup_right
    assert restored_right < workspace_right
    assert restored_left > workspace_left
    assert abs(restored_left - setup_left) <= 8


def test_peak_hardening_controls_follow_active_baseline_config(window, tmp_path):
    custom_cfg = tmp_path / "custom_peak_hardening.yaml"
    custom_cfg.write_text(
        yaml.safe_dump(
            {
                "event_signal": "delta_f",
                "peak_threshold_method": "percentile",
                "peak_threshold_k": 3.5,
                "peak_threshold_percentile": 91.0,
                "peak_threshold_abs": 0.75,
                "peak_min_distance_sec": 1.25,
                "peak_min_prominence_k": 2.75,
                "peak_min_width_sec": 0.42,
                "peak_pre_filter": "lowpass",
                "event_auc_baseline": "median",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Force stale mixed-state values first (legacy + new), then ensure full-section sync.
    window._event_signal_combo.setCurrentText("dff")
    window._peak_method_combo.setCurrentText("mean_std")
    window._peak_k_edit.setText("8.0")
    window._peak_pct_edit.setText("33.0")
    window._peak_abs_edit.setText("4.0")
    window._peak_dist_edit.setText("9.0")
    window._peak_min_prominence_k_edit.setText("0.1")
    window._peak_min_width_sec_edit.setText("9.9")
    window._peak_pre_filter_combo.setCurrentText("none")
    window._event_auc_combo.setCurrentText("zero")

    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(custom_cfg))
    window._update_config_source_ui()

    assert window._event_signal_combo.currentText() == "delta_f"
    assert window._peak_method_combo.currentText() == "percentile"
    assert float(window._peak_k_edit.text()) == pytest.approx(3.5)
    assert float(window._peak_pct_edit.text()) == pytest.approx(91.0)
    assert float(window._peak_abs_edit.text()) == pytest.approx(0.75)
    assert float(window._peak_dist_edit.text()) == pytest.approx(1.25)
    assert float(window._peak_min_prominence_k_edit.text()) == pytest.approx(2.75)
    assert float(window._peak_min_width_sec_edit.text()) == pytest.approx(0.42)
    assert window._peak_pre_filter_combo.currentText() == "lowpass"
    assert window._event_auc_combo.currentText() == "median"

    # Switching back to lab/default baseline should restore full-section consistency.
    window._use_custom_config_cb.setChecked(False)
    window._update_config_source_ui()
    expected_cfg = window._active_baseline_config()
    assert window._event_signal_combo.currentText() == str(expected_cfg.event_signal)
    assert window._peak_method_combo.currentText() == str(expected_cfg.peak_threshold_method)
    assert float(window._peak_k_edit.text()) == pytest.approx(float(expected_cfg.peak_threshold_k))
    assert float(window._peak_pct_edit.text()) == pytest.approx(float(expected_cfg.peak_threshold_percentile))
    assert float(window._peak_abs_edit.text()) == pytest.approx(float(expected_cfg.peak_threshold_abs))
    assert float(window._peak_dist_edit.text()) == pytest.approx(float(expected_cfg.peak_min_distance_sec))
    assert float(window._peak_min_prominence_k_edit.text()) == pytest.approx(
        float(getattr(expected_cfg, "peak_min_prominence_k", 0.0))
    )
    assert float(window._peak_min_width_sec_edit.text()) == pytest.approx(
        float(getattr(expected_cfg, "peak_min_width_sec", 0.0))
    )
    assert window._peak_pre_filter_combo.currentText() == str(getattr(expected_cfg, "peak_pre_filter", "none"))
    assert window._event_auc_combo.currentText() == str(expected_cfg.event_auc_baseline)


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


def test_tuning_apply_back_affordance_and_scope_text(window):
    assert hasattr(window, "_apply_tuning_btn")
    assert window._apply_tuning_btn.text() == "Apply to Next-Run Settings"
    assert hasattr(window, "_tuning_applyback_scope_label")
    assert hasattr(window, "_tuning_applyback_status_label")
    apply_scope = window._tuning_applyback_scope_label.text().lower()
    assert "temporary" in apply_scope
    assert "durable yaml reuse" in apply_scope

    scope_msg = window._tuning_scope_note.text().lower()
    assert "retunes downstream event detection from cached phasic traces" in scope_msg
    assert "before deciding whether to rerun" in scope_msg
    assert "not applied" in window._tuning_applyback_status_label.text().lower()
    assert window._tuning_scope_note.wordWrap()
    assert window._tuning_availability_label.wordWrap()
    assert window._tuning_summary_label.wordWrap()
    assert window._tuning_scope_note.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert window._tuning_availability_label.sizePolicy().horizontalPolicy() in (
        QSizePolicy.Ignored,
        QSizePolicy.Expanding,
    )
    assert window._tuning_summary_label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored


def test_complete_state_panel_results_mode_framing(window, qapp, tmp_path):
    run_dir = _make_completed_results_fixture(str(tmp_path))
    window._current_run_dir = run_dir
    assert window._report_viewer.load_report(run_dir)
    window._enter_complete_state_workspace()
    qapp.processEvents()

    assert window._controls_stack.currentWidget() is window._complete_state_panel
    assert window._complete_mode_title_label.text() == "Results Mode"
    assert "completed outputs are loaded on the right" in window._complete_mode_subtitle_label.text().lower()
    assert "optional" not in window._complete_mode_subtitle_label.text().lower()
    assert "apply back to next-run settings" in window._complete_mode_next_steps_label.text().lower()
    assert "as needed" in window._complete_mode_next_steps_label.text().lower()
    assert "read-only" in window._complete_summary_label.text().lower()
    assert window._new_run_btn.text() == "Start New Run"


def test_post_run_tuning_hierarchy_and_feedback_copy(window):
    assert window._tuning_group.title() == "Post-Run Tuning"
    assert "Primary:" in window._tuning_disclosure_btn.text()
    assert "Secondary:" in window._correction_tuning_disclosure_btn.text()
    assert "advanced" in window._correction_tuning_disclosure_btn.text().lower()
    assert "post-run tuning tools" in window._tuning_phase_note.text().lower()
    assert "optional" not in window._tuning_group.title().lower()
    assert "advanced path" in window._correction_tuning_role_note.text().lower()

    window._tuning_last_result = {
        "selected_roi": "Region0",
        "inspection_chunk_id": 2,
        "event_signal_used": "delta_f",
        "retune_dir": "C:/tmp/retune",
    }
    window._tuning_last_changed_fields = ["event signal", "threshold method"]
    window._tuning_applyback_applied = False
    window._tuning_applyback_timestamp = ""
    window._refresh_tuning_feedback_summary()
    assert "Changed vs baseline: event signal, threshold method" in window._tuning_summary_label.text()
    assert "not applied" in window._tuning_summary_label.text().lower()

    window._tuning_applyback_applied = True
    window._tuning_applyback_timestamp = "12:00:00"
    window._refresh_tuning_feedback_summary()
    assert "applied (12:00:00)" in window._tuning_summary_label.text()
    assert "completed run is unchanged" in window._tuning_applyback_status_label.text().lower()
