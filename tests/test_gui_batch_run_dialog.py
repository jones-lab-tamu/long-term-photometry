import json
import time
from pathlib import Path

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.batch_run_dialog import BatchRunDialog, BatchRunWorker
from gui.batch_spec import BatchDatasetRow
from gui.main_window import MainWindow, RunnerState
from gui.run_spec import RunSpec


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _base_spec(tmp_path: Path, *, validate_only: bool = False) -> RunSpec:
    cfg = tmp_path / "base.yaml"
    cfg.write_text("target_fs_hz: 10.0\n", encoding="utf-8")
    return RunSpec(
        input_dir=str(tmp_path / "single_input_placeholder"),
        run_dir=str(tmp_path / "single_output_placeholder"),
        format="custom_tabular",
        config_source_path=str(cfg),
        validate_only=validate_only,
        acquisition_mode="continuous",
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
        allow_partial_final_window=True,
        mode="phasic",
        include_roi_ids=["Region0"],
        run_profile="full",
    )


def _dialog(tmp_path: Path, *, opened: list[str] | None = None) -> BatchRunDialog:
    return BatchRunDialog(
        build_base_run_spec=lambda validate_only, **_kwargs: _base_spec(
            tmp_path, validate_only=validate_only
        ),
        open_completed_run=lambda path: opened.append(path) is None if opened is not None else True,
        open_folder=lambda _path: None,
    )


def _detect_two_datasets(dialog: BatchRunDialog, tmp_path: Path) -> tuple[Path, Path]:
    input_root = tmp_path / "batch_inputs"
    output_root = tmp_path / "batch_outputs"
    (input_root / "AnimalB").mkdir(parents=True)
    (input_root / "AnimalA").mkdir(parents=True)
    (input_root / "root_file.csv").write_text("ignored\n", encoding="utf-8")
    (input_root / ".hidden").mkdir()
    dialog._input_root_edit.setText(str(input_root))
    dialog._output_root_edit.setText(str(output_root))
    dialog._on_detect_datasets()
    return input_root, output_root


def _make_openable_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "Region0" / "summary").mkdir(parents=True)


def _wait_until(qapp, predicate, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def test_dialog_discovery_lists_immediate_subfolders_and_reports_ignored_files(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        assert "same current analysis settings" in dialog._scope_label.text()
        assert (
            "does not perform group statistics, averaging, or multi-recording visualization"
            in dialog._scope_label.text()
        )
        _input_root, output_root = _detect_two_datasets(dialog, tmp_path)

        assert dialog._table.rowCount() == 2
        assert [row.dataset_name for row in dialog._rows] == ["AnimalA", "AnimalB"]
        assert all(Path(row.output_path).parent == output_root / "runs" for row in dialog._rows)
        assert "Ignored 1 root-level file" in dialog._message_label.text()
        assert "Ignored 1 hidden folder" in dialog._message_label.text()
    finally:
        dialog.close()


def test_dialog_builds_frozen_batch_spec_from_current_settings_without_dataset_path_leak(
    qapp,
    tmp_path: Path,
):
    dialog = _dialog(tmp_path)
    try:
        input_root, output_root = _detect_two_datasets(dialog, tmp_path)
        spec = dialog.build_batch_run_spec(validate_only=True)

        assert spec.batch_input_root == str(input_root.resolve())
        assert spec.batch_output_root == str(output_root.resolve())
        assert spec.shared_settings["format"] == "custom_tabular"
        assert spec.shared_settings["mode"] == "phasic"
        assert spec.shared_settings["acquisition_mode"] == "continuous"
        assert spec.shared_settings["continuous_window_sec"] == 600.0
        assert spec.shared_settings["include_roi_ids"] == ["Region0"]
        assert spec.base_run_spec["input_dir"].endswith("single_input_placeholder")
        assert spec.base_run_spec["run_dir"].endswith("single_output_placeholder")
        assert all(row.input_path.startswith(str(input_root.resolve())) for row in spec.datasets)
        assert all(row.output_path.startswith(str(output_root.resolve())) for row in spec.datasets)
    finally:
        dialog.close()


def test_main_window_batch_base_spec_uses_placeholders_without_valid_single_run_paths(
    qapp,
    tmp_path: Path,
):
    window = MainWindow()
    dialog = None
    try:
        stale_input = tmp_path / "stale_single_input"
        stale_output = tmp_path / "stale_single_output"
        window._input_dir.setText(str(stale_input))
        window._output_dir.setText(str(stale_output))
        window._format_combo.setCurrentText("custom_tabular")
        continuous_idx = window._acquisition_mode_combo.findData("continuous")
        window._acquisition_mode_combo.setCurrentIndex(continuous_idx)
        window._continuous_window_sec_spin.setValue(900.0)
        window._allow_partial_final_window_cb.setChecked(True)
        window._mode_combo.setCurrentText("phasic")
        idx_constraint = window._dynamic_fit_slope_constraint_combo.findData("nonnegative")
        assert idx_constraint >= 0
        window._dynamic_fit_slope_constraint_combo.setCurrentIndex(idx_constraint)
        window._dynamic_fit_min_slope_spin.setValue(0.0)

        dialog = BatchRunDialog(
            build_base_run_spec=window._build_batch_base_run_spec,
            open_completed_run=lambda _path: True,
            open_folder=lambda _path: None,
            parent=window,
        )
        input_root, output_root = _detect_two_datasets(dialog, tmp_path)
        spec = dialog.build_batch_run_spec(validate_only=True)

        assert spec.batch_input_root == str(input_root.resolve())
        assert spec.batch_output_root == str(output_root.resolve())
        assert all(row.input_path.startswith(str(input_root.resolve())) for row in spec.datasets)
        assert all(row.output_path.startswith(str(output_root.resolve())) for row in spec.datasets)
        assert spec.base_run_spec["input_dir"] != str(stale_input)
        assert spec.base_run_spec["run_dir"] != str(stale_output)
        assert "_batch_placeholders" in spec.base_run_spec["input_dir"]
        assert "_batch_placeholders" in spec.base_run_spec["run_dir"]
        assert spec.shared_settings["format"] == "custom_tabular"
        assert spec.shared_settings["mode"] == "phasic"
        assert spec.shared_settings["acquisition_mode"] == "continuous"
        assert spec.shared_settings["continuous_window_sec"] == 900.0
        assert spec.shared_settings["allow_partial_final_window"] is True
        assert spec.shared_settings["config_overrides"]["dynamic_fit_slope_constraint"] == "nonnegative"
        assert spec.shared_settings["config_overrides"]["dynamic_fit_min_slope"] == 0.0
        assert spec.base_run_spec["config_overrides"]["dynamic_fit_slope_constraint"] == "nonnegative"
        assert spec.base_run_spec["config_overrides"]["dynamic_fit_min_slope"] == 0.0
        assert window._input_dir.text() == str(stale_input)
        assert window._output_dir.text() == str(stale_output)
    finally:
        if dialog is not None:
            dialog.close()
        window.close()
        window.deleteLater()


def test_changing_output_root_after_detection_clears_planned_rows(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        _input_root, output_root = _detect_two_datasets(dialog, tmp_path)
        assert dialog._rows

        dialog._output_root_edit.setText(str(tmp_path / "batch_outputs_b"))

        assert dialog._rows == []
        assert dialog._table.rowCount() == 0
        assert "Click Detect Datasets again" in dialog._message_label.text()
        assert not dialog._run_batch_btn.isEnabled()
        assert not any(str(output_root) in dialog._table.item(r, 3).text() for r in range(dialog._table.rowCount()))
    finally:
        dialog.close()


def test_changing_input_root_after_detection_clears_planned_rows(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        input_root, _output_root = _detect_two_datasets(dialog, tmp_path)
        assert dialog._rows
        other_root = tmp_path / "other_inputs"
        (other_root / "AnimalC").mkdir(parents=True)

        dialog._input_root_edit.setText(str(other_root))

        assert dialog._rows == []
        assert dialog._table.rowCount() == 0
        assert "Click Detect Datasets again" in dialog._message_label.text()
        assert not any(str(input_root) in dialog._table.item(r, 2).text() for r in range(dialog._table.rowCount()))
    finally:
        dialog.close()


def test_changing_overwrite_from_false_to_true_clears_collision_status(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        input_root = tmp_path / "batch_inputs"
        output_root = tmp_path / "batch_outputs"
        (input_root / "AnimalA").mkdir(parents=True)
        (output_root / "runs" / "AnimalA_001").mkdir(parents=True)
        dialog._input_root_edit.setText(str(input_root))
        dialog._output_root_edit.setText(str(output_root))
        dialog._overwrite_cb.setChecked(False)
        dialog._on_detect_datasets()

        assert [row.status for row in dialog._rows] == ["skipped"]
        dialog._overwrite_cb.setChecked(True)

        assert dialog._rows == []
        assert dialog._table.rowCount() == 0
        assert "Click Detect Datasets again" in dialog._message_label.text()
    finally:
        dialog.close()


def test_changing_overwrite_from_true_to_false_clears_pending_collision(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        input_root = tmp_path / "batch_inputs"
        output_root = tmp_path / "batch_outputs"
        (input_root / "AnimalA").mkdir(parents=True)
        (output_root / "runs" / "AnimalA_001").mkdir(parents=True)
        dialog._input_root_edit.setText(str(input_root))
        dialog._output_root_edit.setText(str(output_root))
        dialog._overwrite_cb.setChecked(True)
        dialog._on_detect_datasets()

        assert [row.status for row in dialog._rows] == ["pending"]
        dialog._overwrite_cb.setChecked(False)

        assert dialog._rows == []
        assert dialog._table.rowCount() == 0
        assert "Click Detect Datasets again" in dialog._message_label.text()
    finally:
        dialog.close()


def test_validate_batch_uses_batch_runner_with_validate_only_true(qapp, tmp_path: Path, monkeypatch):
    calls = []

    class FakeBatchRunner:
        def __init__(self, batch_spec, **callbacks):
            self.batch_spec = batch_spec
            self.callbacks = callbacks

        def run(self, *, validate_only=False):
            calls.append(validate_only)
            for row in self.batch_spec.datasets:
                row.status = "success"
                self.callbacks["on_row_update"](row)
            self.callbacks["on_finished"](self.batch_spec)
            return self.batch_spec

        def request_cancel(self):
            calls.append("cancel")

    monkeypatch.setattr("gui.batch_run_dialog.BatchRunner", FakeBatchRunner)
    dialog = _dialog(tmp_path)
    try:
        _detect_two_datasets(dialog, tmp_path)
        dialog._start_batch(validate_only=True)
        _wait_until(qapp, lambda: not dialog._batch_active)

        assert calls == [True]
        assert [row.status for row in dialog._rows] == ["success", "success"]
    finally:
        dialog.close()


def test_run_batch_uses_batch_runner_with_validate_only_false(qapp, tmp_path: Path, monkeypatch):
    calls = []

    class FakeBatchRunner:
        def __init__(self, batch_spec, **callbacks):
            self.batch_spec = batch_spec
            self.callbacks = callbacks

        def run(self, *, validate_only=False):
            calls.append(validate_only)
            self.callbacks["on_finished"](self.batch_spec)
            return self.batch_spec

        def request_cancel(self):
            pass

    monkeypatch.setattr("gui.batch_run_dialog.BatchRunner", FakeBatchRunner)
    dialog = _dialog(tmp_path)
    try:
        _detect_two_datasets(dialog, tmp_path)
        dialog._start_batch(validate_only=False)
        _wait_until(qapp, lambda: not dialog._batch_active)

        assert calls == [False]
    finally:
        dialog.close()


def test_table_status_update_changes_status_message_and_elapsed(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        _detect_two_datasets(dialog, tmp_path)
        row = dialog._rows[0]
        row.status = "failed"
        row.message = "validation failed"
        row.elapsed_sec = 1.25

        dialog._on_worker_row_updated(row)

        assert dialog._table.item(0, 4).text() == "failed"
        assert dialog._table.item(0, 5).text() == "validation failed"
        assert dialog._table.item(0, 6).text() == "1.250"
    finally:
        dialog.close()


def test_cancel_button_emits_queued_cancellation_request(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)

    class FakeWorker:
        def __init__(self):
            self.cancel_requested = False

        def request_cancel(self):
            self.cancel_requested = True

    try:
        worker = FakeWorker()
        emitted = []
        dialog.cancel_requested.connect(lambda: emitted.append(True))
        dialog._worker = worker
        dialog._set_batch_active(True)
        dialog._on_cancel()

        assert worker.cancel_requested is True
        assert emitted == [True]
        assert "cancellation requested" in dialog._message_label.text().lower()
        assert not dialog._cancel_btn.isEnabled()
    finally:
        dialog._set_batch_active(False)
        dialog.close()


def test_worker_pending_cancel_before_runner_creation_is_applied(monkeypatch, tmp_path: Path):
    applied = []

    class FakeBatchRunner:
        def __init__(self, batch_spec, **callbacks):
            self.batch_spec = batch_spec
            self.cancel_requested = callbacks["cancel_requested"]

        def run(self, *, validate_only=False):
            applied.append(("pre_cancel", self.cancel_requested()))
            applied.append(("run", validate_only))
            return self.batch_spec

    monkeypatch.setattr("gui.batch_run_dialog.BatchRunner", FakeBatchRunner)
    row = BatchDatasetRow(
        dataset_id="dataset_001",
        dataset_name="AnimalA",
        input_path=str(tmp_path / "in" / "AnimalA"),
        output_path=str(tmp_path / "out" / "runs" / "AnimalA_001"),
    )
    from gui.batch_spec import make_batch_run_spec

    spec = make_batch_run_spec(
        batch_id="batch_test",
        batch_input_root=str(tmp_path / "in"),
        batch_output_root=str(tmp_path / "out"),
        datasets=[row],
    )
    worker = BatchRunWorker(spec, validate_only=True)

    worker.request_cancel()
    worker.run()

    assert applied == [("pre_cancel", True), ("run", True)]


def test_worker_exception_emits_message_and_finished(monkeypatch, tmp_path: Path):
    messages = []
    finished = []

    class FailingBatchRunner:
        def __init__(self, batch_spec, **_callbacks):
            self.batch_spec = batch_spec

        def run(self, *, validate_only=False):
            raise RuntimeError("synthetic batch failure")

    monkeypatch.setattr("gui.batch_run_dialog.BatchRunner", FailingBatchRunner)
    row = BatchDatasetRow(
        dataset_id="dataset_001",
        dataset_name="AnimalA",
        input_path=str(tmp_path / "in" / "AnimalA"),
        output_path=str(tmp_path / "out" / "runs" / "AnimalA_001"),
    )
    from gui.batch_spec import make_batch_run_spec

    spec = make_batch_run_spec(
        batch_id="batch_test",
        batch_input_root=str(tmp_path / "in"),
        batch_output_root=str(tmp_path / "out"),
        datasets=[row],
    )
    worker = BatchRunWorker(spec, validate_only=False)
    worker.message.connect(lambda msg: messages.append(msg))
    worker.finished.connect(lambda final_spec: finished.append(final_spec))

    worker.run()

    assert len(finished) == 1
    assert finished[0] is spec
    assert any("synthetic batch failure" in msg for msg in messages)


def test_close_event_is_ignored_while_batch_is_active(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        dialog._set_batch_active(True)
        event = QCloseEvent()

        dialog.closeEvent(event)

        assert not event.isAccepted()
        assert "Cancel the batch before closing" in dialog._message_label.text()
    finally:
        dialog._set_batch_active(False)
        dialog.close()


def test_close_event_is_ignored_while_thread_is_still_running(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)

    class FakeThread:
        def isRunning(self):
            return True

    try:
        dialog._batch_active = False
        dialog._thread = FakeThread()
        event = QCloseEvent()

        dialog.closeEvent(event)

        assert not event.isAccepted()
        assert "Cancel the batch before closing" in dialog._message_label.text()
    finally:
        dialog._thread = None
        dialog.close()


def test_open_selected_run_uses_existing_completed_run_callback(qapp, tmp_path: Path):
    opened: list[str] = []
    dialog = _dialog(tmp_path, opened=opened)
    try:
        _detect_two_datasets(dialog, tmp_path)
        row = dialog._rows[0]
        row.status = "success"
        _make_openable_run(Path(row.output_path))
        dialog._populate_table()
        dialog._table.selectRow(0)
        qapp.processEvents()

        assert dialog._open_selected_btn.isEnabled()
        dialog._on_open_selected_run()

        assert opened == [row.output_path]
    finally:
        dialog.close()


def test_non_success_row_cannot_be_opened(qapp, tmp_path: Path):
    opened: list[str] = []
    dialog = _dialog(tmp_path, opened=opened)
    try:
        _detect_two_datasets(dialog, tmp_path)
        dialog._table.selectRow(0)
        qapp.processEvents()

        assert not dialog._open_selected_btn.isEnabled()
        dialog._on_open_selected_run()
        assert opened == []
    finally:
        dialog.close()


def test_controls_disable_during_active_batch_and_restore_after_finish(qapp, tmp_path: Path):
    dialog = _dialog(tmp_path)
    try:
        _detect_two_datasets(dialog, tmp_path)
        dialog._set_batch_active(True)

        assert not dialog._detect_btn.isEnabled()
        assert not dialog._validate_batch_btn.isEnabled()
        assert not dialog._run_batch_btn.isEnabled()
        assert dialog._cancel_btn.isEnabled()

        dialog._set_batch_active(False)
        assert dialog._detect_btn.isEnabled()
        assert dialog._validate_batch_btn.isEnabled()
        assert dialog._run_batch_btn.isEnabled()
        assert not dialog._cancel_btn.isEnabled()
    finally:
        dialog.close()


def test_main_window_has_batch_run_button_and_disables_it_while_runner_active(qapp, monkeypatch):
    window = MainWindow()
    try:
        assert window._batch_run_btn.text() == "Batch Run..."

        monkeypatch.setattr(window._runner, "is_running", lambda: True)
        window._ui_state = RunnerState.RUNNING
        window._update_button_states()
        assert not window._batch_run_btn.isEnabled()

        monkeypatch.setattr(window._runner, "is_running", lambda: False)
        window._ui_state = RunnerState.IDLE
        window._update_button_states()
        assert window._batch_run_btn.isEnabled()
    finally:
        window.close()
        window.deleteLater()


def test_open_results_invalid_selection_logs_blocked_once(qapp, tmp_path: Path, monkeypatch):
    window = MainWindow()
    try:
        selected = tmp_path / "raw_input_folder"
        selected.mkdir()
        messages = []
        monkeypatch.setattr(window, "_append_run_log", lambda msg: messages.append(msg))
        monkeypatch.setattr(
            main_window_module.QFileDialog,
            "getExistingDirectory",
            lambda *_args, **_kwargs: str(selected),
        )
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "information",
            lambda *_args, **_kwargs: None,
        )

        window._on_open_results()

        blocked = [msg for msg in messages if msg.startswith("Open Results blocked:")]
        assert len(blocked) == 1
    finally:
        window.close()
        window.deleteLater()
