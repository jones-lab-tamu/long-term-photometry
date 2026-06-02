import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from gui.main_window import MainWindow, RunnerState


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _write_png(path: Path, width: int = 360, height: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = QPixmap(width, height)
    pix.fill(Qt.white)
    assert pix.save(str(path))


def _write_completed_markers(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "run_report.json").write_text(
        json.dumps({"run_context": {"run_type": "full"}}),
        encoding="utf-8",
    )


def _make_continuous_completed_run(run_dir: Path, regions: tuple[str, ...]) -> Path:
    _write_completed_markers(run_dir)
    for region in regions:
        summary = run_dir / region / "summary"
        tables = run_dir / region / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        (tables / "continuous_phasic_window_summary.csv").write_text(
            "roi,chunk_id,acquisition_mode\n"
            f"{region},0,continuous\n",
            encoding="utf-8",
        )
        (tables / "continuous_tonic_window_summary.csv").write_text(
            "roi,chunk_id,acquisition_mode\n"
            f"{region},0,continuous\n",
            encoding="utf-8",
        )
        _write_png(summary / "phasic_peak_rate_timeseries.png")
        _write_png(summary / "phasic_peak_count_timeseries.png")
        _write_png(summary / "phasic_auc_timeseries.png")
        _write_png(summary / "tonic_overview.png")
    return run_dir


def _make_intermittent_completed_run(run_dir: Path) -> Path:
    _write_completed_markers(run_dir)
    for region in ("Region0", "Region1"):
        summary = run_dir / region / "summary"
        day_plots = run_dir / region / "day_plots"
        tables = run_dir / region / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        (tables / "features.csv").write_text("roi,chunk_id\nRegion0,0\n", encoding="utf-8")
        _write_png(summary / "phasic_correction_impact.png")
        _write_png(summary / "tonic_overview.png")
        _write_png(day_plots / "phasic_sig_iso_day_000.png", 600, 1500)
        _write_png(day_plots / "phasic_dynamic_fit_day_000.png", 600, 1500)
        _write_png(day_plots / "phasic_dFF_day_000.png", 600, 1500)
        _write_png(day_plots / "phasic_stacked_day_000.png", 600, 1500)
    return run_dir


def _patch_picker(monkeypatch, selected: Path | str, calls: dict[str, int] | None = None):
    def _pick_dir(*args, **kwargs):
        if calls is not None:
            calls["picker"] = calls.get("picker", 0) + 1
            calls["title"] = args[1] if len(args) > 1 else kwargs.get("caption", "")
            calls["start_dir"] = args[2] if len(args) > 2 else kwargs.get("dir", "")
        return str(selected)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _pick_dir)


def _capture_messages(monkeypatch) -> list[tuple[str, str]]:
    messages = []

    def _capture_message(_parent, title, text):
        messages.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "information", _capture_message)
    return messages


def test_open_results_always_uses_picker_even_with_valid_current_run_dir(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    current_run = _make_continuous_completed_run(tmp_path / "current_run", ("Region0",))
    selected_run = _make_continuous_completed_run(
        tmp_path / "selected_run",
        ("Region0", "Region1"),
    )
    calls = {}
    _patch_picker(monkeypatch, selected_run, calls)
    window._current_run_dir = str(current_run)
    window._update_button_states()

    window._on_open_results()
    qapp.processEvents()

    assert calls["picker"] == 1
    assert calls["title"] == "Select Completed Pipeline Run Folder"
    assert window._current_run_dir == str(selected_run)
    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["Region0", "Region1"]


def test_open_results_always_uses_picker_even_with_valid_output_directory(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    output_run = _make_continuous_completed_run(tmp_path / "output_run", ("Region0",))
    selected_run = _make_continuous_completed_run(
        tmp_path / "selected_output_override",
        ("CH1", "CH2"),
    )
    calls = {}
    _patch_picker(monkeypatch, selected_run, calls)
    window._output_dir.setText(str(output_run))
    window._update_button_states()

    window._on_open_results()
    qapp.processEvents()

    assert calls["picker"] == 1
    assert window._current_run_dir == str(selected_run)
    assert window._output_dir.text() == str(selected_run)
    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["CH1", "CH2"]


def test_open_results_cancel_does_nothing(window, tmp_path: Path, monkeypatch):
    prior_run = _make_continuous_completed_run(tmp_path / "prior_run", ("Region0",))
    _patch_picker(monkeypatch, "")
    window._current_run_dir = str(prior_run)
    window._output_dir.setText(str(tmp_path / "output_base"))
    before_current = window._current_run_dir
    before_output = window._output_dir.text()

    window._on_open_results()

    assert window._current_run_dir == before_current
    assert window._output_dir.text() == before_output
    assert not window._report_viewer.has_loaded_results()
    assert not window._runner.is_running()


def test_invalid_selected_folder_rejected_without_starting_pipeline(
    window,
    tmp_path: Path,
    monkeypatch,
):
    input_dir = tmp_path / "continuous_input_only"
    input_dir.mkdir(parents=True)
    (input_dir / "continuous_recording.csv").write_text(
        "time_sec,Region0_iso,Region0_sig\n0,1,2\n",
        encoding="utf-8",
    )
    (input_dir / "generation_manifest.yaml").write_text(
        "acquisition_mode: continuous\n",
        encoding="utf-8",
    )
    prior_run = _make_continuous_completed_run(tmp_path / "prior_run", ("Region0",))
    prior_output = str(tmp_path / "output_base")
    _patch_picker(monkeypatch, input_dir)
    messages = _capture_messages(monkeypatch)
    window._current_run_dir = str(prior_run)
    window._output_dir.setText(prior_output)

    window._on_open_results()

    assert window._current_run_dir == str(prior_run)
    assert window._output_dir.text() == prior_output
    assert not window._report_viewer.has_loaded_results()
    assert not window._runner.is_running()
    assert messages
    assert messages[0][0] == "Results Not Opened"
    assert "does not look like a completed pipeline run" in messages[0][1]
    assert "status.json/MANIFEST.json" in messages[0][1]


def test_continuous_completed_run_without_day_plots_opens_from_picker(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _make_continuous_completed_run(tmp_path / "continuous_no_day_plots", ("Region0",))
    assert not (run_dir / "Region0" / "day_plots").exists()
    _patch_picker(monkeypatch, run_dir)

    window._on_open_results()
    qapp.processEvents()

    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["Region0"]


def test_intermittent_completed_run_opens_from_picker(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _make_intermittent_completed_run(tmp_path / "intermittent_run")
    _patch_picker(monkeypatch, run_dir)

    window._on_open_results()
    qapp.processEvents()

    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["Region0", "Region1"]


def test_open_results_disabled_while_running_or_validating(window, monkeypatch):
    monkeypatch.setattr(window._runner, "is_running", lambda: True)

    window._ui_state = RunnerState.RUNNING
    window._update_button_states()
    assert not window._open_results_btn.isEnabled()

    window._ui_state = RunnerState.VALIDATING
    window._update_button_states()
    assert not window._open_results_btn.isEnabled()

    monkeypatch.setattr(window._runner, "is_running", lambda: False)
    window._ui_state = RunnerState.IDLE
    window._update_button_states()
    assert window._open_results_btn.isEnabled()
