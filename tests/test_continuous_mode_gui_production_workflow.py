import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.run_report_parser import resolve_region_deliverables
from gui.run_report_viewer import RunReportViewer


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _write_custom_tabular_csv(path: Path, duration_sec: float = 1200.0, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig}).to_csv(
        path,
        index=False,
    )


def _write_custom_tabular_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "target_fs_hz: 10.0",
                "chunk_duration_sec: 600.0",
                "allow_partial_final_chunk: false",
                "custom_tabular_time_col: time_sec",
                "custom_tabular_uv_suffix: _iso",
                "custom_tabular_sig_suffix: _sig",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _select_combo_data(combo, value: str) -> None:
    idx = combo.findData(value)
    if idx < 0:
        idx = combo.findText(value)
    assert idx >= 0, f"{value!r} not found in combo"
    combo.setCurrentIndex(idx)


def _configure_gui_for_custom_tabular(
    window: MainWindow,
    tmp_path: Path,
    *,
    acquisition_mode: str,
    validate_mode: str = "both",
) -> tuple[Path, Path, Path]:
    input_dir = tmp_path / f"input_{acquisition_mode}_{validate_mode}"
    output_base = tmp_path / f"runs_{acquisition_mode}_{validate_mode}"
    config_path = tmp_path / f"config_{acquisition_mode}_{validate_mode}.yaml"
    _write_custom_tabular_csv(input_dir / "session_000.csv")
    _write_custom_tabular_config(config_path)
    output_base.mkdir(parents=True, exist_ok=True)

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(output_base))
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(config_path))
    window._format_combo.setCurrentText("custom_tabular")
    window._mode_combo.setCurrentText(validate_mode)
    _select_combo_data(window._acquisition_mode_combo, acquisition_mode)
    window._continuous_window_sec_spin.setValue(600.0)
    window._allow_partial_final_window_cb.setChecked(False)
    window._sph_edit.setText("2")
    window._duration_edit.setText("600")
    window._update_context_sensitive_controls()
    return input_dir, output_base, config_path


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def test_gui_runspec_serializes_continuous_mode_and_widget_state(window, tmp_path: Path):
    _configure_gui_for_custom_tabular(window, tmp_path, acquisition_mode="continuous")
    window._continuous_window_sec_spin.setValue(900.0)
    window._allow_partial_final_window_cb.setChecked(True)
    window._update_context_sensitive_controls()

    assert window._continuous_window_sec_spin.isEnabled()
    assert not window._continuous_step_sec_spin.isVisible()
    assert not window._continuous_step_sec_spin.isEnabled()
    assert window._continuous_step_sec_spin.value() == pytest.approx(900.0)
    assert window._allow_partial_final_window_cb.isEnabled()
    assert not window._sph_edit.isEnabled()
    assert not window._duration_edit.isEnabled()
    assert "ignored in continuous mode" in window._sph_warning.text().lower()

    spec = window._build_run_spec(validate_only=True)
    assert spec.acquisition_mode == "continuous"
    assert spec.continuous_window_sec == pytest.approx(900.0)
    assert spec.continuous_step_sec == pytest.approx(900.0)
    assert spec.allow_partial_final_window is True
    assert spec.format == "custom_tabular"
    assert "acquisition_mode" in spec.user_set_fields
    assert "continuous_window_sec" in spec.user_set_fields
    assert "continuous_step_sec" in spec.user_set_fields
    assert "allow_partial_final_window" in spec.user_set_fields

    _select_combo_data(window._acquisition_mode_combo, "intermittent")
    window._update_context_sensitive_controls()
    assert not window._continuous_window_sec_spin.isEnabled()
    assert not window._continuous_step_sec_spin.isVisible()
    assert not window._continuous_step_sec_spin.isEnabled()
    assert not window._allow_partial_final_window_cb.isEnabled()


def test_continuous_window_tooltip_and_step_sync(window, tmp_path: Path):
    _configure_gui_for_custom_tabular(window, tmp_path, acquisition_mode="continuous")
    tooltip = window._continuous_window_sec_spin.toolTip().lower()
    assert "analysis window" in tooltip
    assert "continuous recording" in tooltip
    assert "non-overlapping" in tooltip
    assert "chunk" in tooltip

    step_tooltip = window._continuous_step_sec_spin.toolTip().lower()
    assert "automatically set to match continuous window" in step_tooltip
    assert "non-overlapping windows only" in step_tooltip
    assert "sliding/overlapping windows are not supported" in step_tooltip

    window._continuous_window_sec_spin.setValue(300.0)
    window._update_context_sensitive_controls()

    assert not window._continuous_step_sec_spin.isVisible()
    assert not window._continuous_step_sec_spin.isEnabled()
    assert window._continuous_step_sec_spin.value() == pytest.approx(300.0)
    spec = window._build_run_spec(validate_only=True)
    assert spec.continuous_window_sec == pytest.approx(300.0)
    assert spec.continuous_step_sec == pytest.approx(300.0)
    argv = spec.build_runner_argv()
    assert "--continuous-window-sec" in argv
    assert argv[argv.index("--continuous-window-sec") + 1] == "300.0"
    assert "--continuous-step-sec" in argv
    assert argv[argv.index("--continuous-step-sec") + 1] == "300.0"


def test_partial_final_window_tooltip(window):
    tooltip = window._allow_partial_final_window_cb.toolTip().lower()
    assert "keeps a final shorter window" in tooltip
    assert "trailing partial window is dropped" in tooltip


def test_continuous_tuning_terminology_uses_window_chunk(window, tmp_path: Path):
    run_dir = tmp_path / "continuous_completed"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    (phasic_out / "config_used.yaml").write_text(
        "acquisition_mode: continuous\ncontinuous_window_sec: 600\ncontinuous_step_sec: 600\n",
        encoding="utf-8",
    )
    window._current_run_dir = str(run_dir)
    window._refresh_tuning_chunk_terminology()

    assert "window" in window._tuning_chunk_label.text().lower()
    assert "chunk" in window._tuning_chunk_label.text().lower()
    assert "session" not in window._tuning_chunk_label.text().lower()
    assert "fixed-length analysis windows" in window._tuning_chunk_combo.toolTip()
    assert "not acquisition sessions" in window._tuning_chunk_combo.toolTip()
    assert "window" in window._correction_tuning_chunk_label.text().lower()
    assert "chunk" in window._correction_tuning_chunk_label.text().lower()
    assert "session" not in window._correction_tuning_chunk_label.text().lower()
    assert "fixed-length analysis windows" in window._correction_tuning_chunk_combo.toolTip()


def test_intermittent_tuning_terminology_remains_session_based(window, tmp_path: Path):
    _configure_gui_for_custom_tabular(window, tmp_path, acquisition_mode="intermittent")
    window._current_run_dir = ""
    window._refresh_tuning_chunk_terminology()
    window._update_context_sensitive_controls()

    assert window._tuning_chunk_label.text() == "Preview session:"
    assert "session" in window._tuning_chunk_combo.toolTip().lower()
    assert window._correction_tuning_chunk_label.text() == "Preview session:"
    assert "session" in window._correction_tuning_chunk_combo.toolTip().lower()
    assert window._sph_edit.isEnabled()
    assert window._duration_edit.isEnabled()

    spec_inter = window._build_run_spec(validate_only=False)
    assert spec_inter.acquisition_mode == "intermittent"
    assert spec_inter.sessions_per_hour == 2
    assert spec_inter.session_duration_s == pytest.approx(600.0)


def test_gui_validate_continuous_builds_and_executes_wrapper_command(window, tmp_path: Path):
    _configure_gui_for_custom_tabular(window, tmp_path, acquisition_mode="continuous")

    argv = window._build_argv(validate_only=True, overwrite=True)
    run_dir = Path(window._current_run_dir)
    assert "--validate-only" in argv
    assert "--acquisition-mode" in argv
    assert argv[argv.index("--acquisition-mode") + 1] == "continuous"
    assert "--continuous-window-sec" in argv
    assert argv[argv.index("--continuous-window-sec") + 1] == "600.0"
    assert "--continuous-step-sec" in argv
    assert argv[argv.index("--continuous-step-sec") + 1] == "600.0"
    assert "--no-allow-partial-final-window" in argv

    for name in ("config_effective.yaml", "gui_run_spec.json", "command_invoked.txt"):
        assert (run_dir / name).exists()

    cfg = yaml.safe_load((run_dir / "config_effective.yaml").read_text(encoding="utf-8"))
    assert cfg["acquisition_mode"] == "continuous"
    assert float(cfg["continuous_window_sec"]) == pytest.approx(600.0)
    assert float(cfg["continuous_step_sec"]) == pytest.approx(600.0)
    assert cfg["allow_partial_final_window"] is False

    gui_spec = json.loads((run_dir / "gui_run_spec.json").read_text(encoding="utf-8"))
    assert gui_spec["acquisition_mode"] == "continuous"
    assert gui_spec["format"] == "custom_tabular"
    assert gui_spec["validate_only"] is True

    res = _run(argv)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["run_type"] == "validate_only"
    assert status["acquisition_mode"] == "continuous"
    plan = status["continuous_plan"]
    assert plan["acquisition_mode"] == "continuous"
    assert plan["source_file_count"] == 1
    assert plan["planned_window_count"] == 2
    assert float(plan["continuous_window_sec"]) == pytest.approx(600.0)
    assert float(plan["continuous_step_sec"]) == pytest.approx(600.0)
    assert plan["allow_partial_final_window"] is False
    assert not (run_dir / "_analysis").exists()


def test_gui_equivalent_continuous_full_run_outputs_are_viewer_visible(
    window,
    tmp_path: Path,
    qapp,
):
    _configure_gui_for_custom_tabular(window, tmp_path, acquisition_mode="continuous")

    argv = window._build_argv(validate_only=False, overwrite=True)
    run_dir = Path(window._current_run_dir)
    assert "--validate-only" not in argv
    res = _run(argv)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"

    assert (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").exists()
    assert (run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5").exists()
    assert (run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (run_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert (run_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (run_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()
    assert (run_dir / "Region0" / "summary" / "tonic_overview.png").exists()
    assert not (run_dir / "Region0" / "day_plots").exists()

    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    for payload in (manifest, status):
        continuous_outputs = payload["continuous_outputs"]
        assert continuous_outputs["summary_tables_generated"] is True
        assert continuous_outputs["summary_plots_generated"] is True
        assert "Region0/tables/continuous_phasic_window_summary.csv" in continuous_outputs["summary_tables"]
        assert "Region0/summary/phasic_auc_timeseries.png" in continuous_outputs["summary_plots"]
    assert manifest["continuous_outputs"]["intermittent_only_outputs_skipped"]

    regions = resolve_region_deliverables(str(run_dir))
    assert len(regions) == 1
    labels = {label for label, _path, status_text in regions[0]["subfolders"] if status_text == "ok"}
    assert labels == {"Summary", "Tables"}

    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run_dir)) is True
        assert viewer.available_regions() == ["Region0"]
        tab_map = {
            tab: [Path(path).name for path in paths]
            for tab, paths in viewer._region_tab_images["Region0"].items()
        }
        assert tab_map["Tonic"] == ["tonic_overview.png"]
        assert tab_map["Phasic Summary"] == [
            "phasic_auc_timeseries.png",
            "phasic_peak_rate_timeseries.png",
        ]
        assert tab_map["Phasic Sig/Iso"] == []
        assert tab_map["Dynamic Fit"] == []
        assert tab_map["Phasic dFF"] == []
        assert tab_map["Phasic Stacked"] == []
    finally:
        viewer.close()


def test_gui_intermittent_runspec_and_command_remain_session_based(window, tmp_path: Path):
    _configure_gui_for_custom_tabular(window, tmp_path, acquisition_mode="intermittent")

    spec = window._build_run_spec(validate_only=False)
    assert spec.acquisition_mode == "intermittent"
    assert spec.sessions_per_hour == 2
    assert spec.session_duration_s == pytest.approx(600.0)
    assert "acquisition_mode" not in spec.user_set_fields

    argv = window._build_argv(validate_only=True, overwrite=True)
    assert "--acquisition-mode" not in argv
    assert "--continuous-window-sec" not in argv
    assert "--continuous-step-sec" not in argv
    assert "--allow-partial-final-window" not in argv
    assert "--no-allow-partial-final-window" not in argv
    assert "--sessions-per-hour" in argv
    assert argv[argv.index("--sessions-per-hour") + 1] == "2"
    assert "--session-duration-s" in argv
    assert argv[argv.index("--session-duration-s") + 1] == "600.0"

    run_dir = Path(window._current_run_dir)
    cfg = yaml.safe_load((run_dir / "config_effective.yaml").read_text(encoding="utf-8"))
    assert cfg["acquisition_mode"] == "intermittent"
    assert float(cfg["continuous_window_sec"]) == pytest.approx(600.0)
    assert float(cfg["continuous_step_sec"]) == pytest.approx(600.0)
    assert cfg["allow_partial_final_window"] is False


def test_retune_dirs_not_misclassified_as_production_regions(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "Region0" / "summary").mkdir(parents=True)
    (run_dir / "Region0" / "tables").mkdir(parents=True)
    retune_dir = run_dir / "tuning_retune" / "retune_20260101_000000_abcdef"
    retune_dir.mkdir(parents=True)
    (retune_dir / "retuned_phasic_auc_timeseries_Region0.png").write_bytes(b"fake")
    correction_dir = run_dir / "tuning_correction_retune" / "retune_20260101_000001_abcdef"
    correction_dir.mkdir(parents=True)
    (correction_dir / "retuned_phasic_peak_rate_timeseries_Region0.png").write_bytes(b"fake")

    regions = resolve_region_deliverables(str(run_dir))
    assert [r["name"] for r in regions] == ["Region0"]
    labels = {label for label, _path, status in regions[0]["subfolders"] if status == "ok"}
    assert labels == {"Summary", "Tables"}
