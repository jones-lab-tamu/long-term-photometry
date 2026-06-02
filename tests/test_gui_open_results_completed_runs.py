import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from gui.main_window import MainWindow


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


def _set_output_path_and_refresh(window: MainWindow, path: Path) -> None:
    window._output_dir.setText(str(path))
    window._update_button_states()


def _open_results_from(window: MainWindow, monkeypatch, path: Path) -> None:
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(path))
    window._on_open_results()


def test_idle_open_results_browses_and_loads_continuous_completed_run(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _make_continuous_completed_run(
        tmp_path / "continuous_custom_tabular_run",
        ("Region0", "Region1"),
    )

    calls = {"picker": 0}

    def _pick_dir(*args, **kwargs):
        calls["picker"] += 1
        return str(run_dir)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _pick_dir)
    window._output_dir.setText("")
    window._current_run_dir = ""
    window._update_button_states()

    assert window._open_results_btn.isEnabled()
    window._on_open_results()
    qapp.processEvents()
    assert calls["picker"] == 1
    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["Region0", "Region1"]
    tab_map = window._report_viewer._region_tab_images["Region0"]
    assert "phasic_peak_rate_timeseries.png" in [Path(p).name for p in tab_map["Phasic Summary"]]
    assert "tonic_overview.png" in [Path(p).name for p in tab_map["Tonic"]]


def test_current_successful_run_opens_directly_without_picker(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _make_continuous_completed_run(
        tmp_path / "current_completed_run",
        ("Region0", "Region1"),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: pytest.fail("picker should not be used for valid current run"),
    )
    window._current_run_dir = str(run_dir)
    window._output_dir.setText("")
    window._update_button_states()

    assert window._open_results_btn.isEnabled()
    window._on_open_results()
    qapp.processEvents()
    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["Region0", "Region1"]


def test_output_directory_completed_run_opens_directly_without_picker(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _make_continuous_completed_run(
        tmp_path / "output_completed_run",
        ("Region0", "Region1"),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: pytest.fail("picker should not be used for valid output dir"),
    )
    _set_output_path_and_refresh(window, run_dir)

    assert window._open_results_btn.isEnabled()
    window._on_open_results()
    qapp.processEvents()
    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["Region0", "Region1"]


def test_open_results_enabled_and_loads_continuous_rwd_completed_run(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _make_continuous_completed_run(
        tmp_path / "continuous_rwd_run",
        ("CH1", "CH2"),
    )

    _set_output_path_and_refresh(window, run_dir)

    assert window._open_results_btn.isEnabled()
    _open_results_from(window, monkeypatch, run_dir)
    qapp.processEvents()
    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["CH1", "CH2"]


def test_open_results_still_enabled_for_intermittent_completed_run(
    window,
    qapp,
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _make_intermittent_completed_run(tmp_path / "intermittent_run")

    _set_output_path_and_refresh(window, run_dir)

    assert window._open_results_btn.isEnabled()
    _open_results_from(window, monkeypatch, run_dir)
    qapp.processEvents()
    assert window._report_viewer.has_loaded_results()
    assert window._report_viewer.available_regions() == ["Region0", "Region1"]


def test_open_results_rejects_input_dataset_folder_without_starting_pipeline(
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

    _set_output_path_and_refresh(window, input_dir)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(input_dir))
    messages = []

    def _capture_message(_parent, title, text):
        messages.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "information", _capture_message)

    assert window._open_results_btn.isEnabled()
    window._on_open_results()

    assert not window._report_viewer.has_loaded_results()
    assert not window._runner.is_running()
    assert messages
    assert messages[0][0] == "Results Not Opened"
    assert "does not look like a completed pipeline run" in messages[0][1]
    assert "status.json/MANIFEST.json" in messages[0][1]


def test_open_results_enabled_when_completed_run_has_no_day_plots(window, tmp_path: Path):
    run_dir = _make_continuous_completed_run(
        tmp_path / "continuous_no_day_plots",
        ("Region0",),
    )
    assert not (run_dir / "Region0" / "day_plots").exists()

    _set_output_path_and_refresh(window, run_dir)

    assert window._open_results_btn.isEnabled()
