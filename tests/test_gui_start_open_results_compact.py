from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.main_window import MainWindow
from tests.test_completed_run_review_overview import _compact_completed_run


pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    while instance._guided_start_open_results_loading:
        QTest.qWait(10)
    instance.close()
    instance.deleteLater()


def _reviewable_current_run(root: Path) -> Path:
    run = _compact_completed_run(root)
    summary = run / "CH1" / "summary"
    summary.mkdir(parents=True)
    image = QPixmap(4, 4)
    image.fill()
    assert image.save(str(summary / "phasic_correction_impact.png"), "PNG")
    return run


def _wait_for_open(window: MainWindow, timeout_ms: int = 3000) -> None:
    elapsed = 0
    while window._guided_start_open_results_loading and elapsed < timeout_ms:
        QTest.qWait(10)
        elapsed += 10
    assert window._guided_start_open_results_loading is False
    while window._guided_start_open_results_thread is not None and elapsed < timeout_ms:
        QTest.qWait(10)
        elapsed += 10


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int, bytes]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_start_open_uses_overview_worker_and_passes_overview_to_viewer(
    window, tmp_path, monkeypatch
):
    run = _reviewable_current_run(tmp_path / "run")
    before = _snapshot_tree(run)
    overview_calls: list[tuple[str, int]] = []
    viewer_overviews: list[dict] = []
    real_overview = main_window_module.load_completed_review_overview
    real_viewer_load = window._guided_report_viewer.load_report

    def tracked_overview(path):
        overview_calls.append((os.path.realpath(path), threading.get_ident()))
        return real_overview(path)

    def tracked_viewer(path, *, review_overview=None):
        viewer_overviews.append(dict(review_overview or {}))
        return real_viewer_load(path, review_overview=review_overview)

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args: str(run),
    )
    monkeypatch.setattr(
        main_window_module, "load_completed_review_overview", tracked_overview
    )
    monkeypatch.setattr(
        window._guided_report_viewer, "load_report", tracked_viewer
    )
    monkeypatch.setattr(
        main_window_module,
        "is_successful_completed_run_dir",
        lambda *_args: pytest.fail("strict classification repeated"),
    )
    monkeypatch.setattr(
        main_window_module,
        "resolve_completed_run_preview_source",
        lambda *_args: pytest.fail("strategy refresh reclassified the run"),
    )
    monkeypatch.setattr(
        main_window_module,
        "open_phasic_cache",
        lambda *_args: pytest.fail("initial open touched phasic HDF5"),
    )
    image_decodes: list[str] = []
    real_set_image = window._guided_report_viewer._set_image
    monkeypatch.setattr(
        window._guided_report_viewer,
        "_set_image",
        lambda path: (image_decodes.append(path), real_set_image(path))[1],
    )

    window._guided_start_open_results_btn.click()
    assert window._guided_start_open_results_loading is True
    assert window._guided_start_open_results_btn.isEnabled() is False
    assert window._open_results_btn.isEnabled() is False
    _wait_for_open(window)

    assert len(overview_calls) == 1
    assert overview_calls[0][0] == os.path.realpath(str(run))
    assert overview_calls[0][1] != threading.get_ident()
    assert len(viewer_overviews) == 1
    assert viewer_overviews[0]["run_dir"] == str(run.resolve())
    assert window._guided_report_viewer.has_loaded_results() is True
    assert window._guided_report_viewer.active_image_path() == ""
    assert image_decodes == []
    assert window._guided_start_open_results_btn.isEnabled() is True
    assert _snapshot_tree(run) == before

    window._is_complete_workspace_active = True
    monkeypatch.setattr(window, "_roi_chunk_ids_map", lambda *_args: {})
    window._refresh_tuning_workspace_availability()

    window._guided_report_viewer._region_combo.setCurrentIndex(1)
    QApplication.processEvents()
    assert len(image_decodes) == 1
    assert Path(image_decodes[0]).name == "phasic_correction_impact.png"


def test_start_open_worker_keeps_event_loop_responsive(
    window, tmp_path, monkeypatch
):
    run = _reviewable_current_run(tmp_path / "run")
    overview = main_window_module.load_completed_review_overview(run)
    heartbeat = {"count": 0}

    def slow_overview(_path):
        time.sleep(0.15)
        return overview

    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(
        lambda: heartbeat.__setitem__("count", heartbeat["count"] + 1)
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args: str(run),
    )
    monkeypatch.setattr(
        main_window_module, "load_completed_review_overview", slow_overview
    )

    timer.start()
    window._guided_start_open_results_btn.click()
    _wait_for_open(window)
    timer.stop()

    assert heartbeat["count"] >= 3
    assert window._guided_report_viewer.has_loaded_results() is True


def test_start_open_failure_returns_to_start_and_restores_action(
    window, tmp_path, monkeypatch
):
    selected = tmp_path / "partial_run"
    selected.mkdir()
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args: str(selected),
    )

    window._guided_start_open_results_btn.click()
    assert window._guided_start_open_results_btn.isEnabled() is False
    _wait_for_open(window)

    assert window._guided_workflow_mode == "start"
    assert (
        window._guided_workflow_stack.currentWidget().objectName()
        == "guidedStepStart"
    )
    message = window._guided_start_open_status_label.text()
    assert "This completed run could not be opened." in message
    assert str(selected) in message
    assert "The run was not changed." in message
    assert window._guided_start_open_results_btn.isEnabled() is True
