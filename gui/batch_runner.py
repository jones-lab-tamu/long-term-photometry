"""
Sequential batch execution orchestration for GUI batch processing.

This module intentionally adds no Batch Run dialog and no analysis-specific
batch command.  Each dataset is converted back into a normal RunSpec and run
through RunSpec.build_runner_argv(), preserving the existing single-run wrapper
path and completed-run artifact format.

default_subprocess_command_runner is the non-Qt execution backend for batch
orchestration.  It writes stdout.txt/stderr.txt in each dataset run directory
to match the single-run GUI's PipelineRunner artifacts; a future dialog should
wrap BatchRunner rather than create a separate analysis path.
"""

from __future__ import annotations

import copy
import os
import subprocess
import threading
import time
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Callable

from gui.batch_spec import (
    BatchDatasetRow,
    BatchRunSpec,
    utc_now_iso,
    write_batch_config_used_yaml,
    write_batch_manifest_csv,
    write_batch_manifest_json,
    write_batch_run_spec_json,
)
from gui.process_runner import _read_final_status
from gui.run_report_parser import is_successful_completed_run_dir
from gui.run_spec import RunSpec


@dataclass
class BatchCommandResult:
    """Result returned by a per-dataset command runner."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    cancelled: bool = False
    message: str = ""


BatchCommandRunner = Callable[
    [list[str], BatchDatasetRow, RunSpec, Callable[[], bool]],
    BatchCommandResult,
]
BatchRowCallback = Callable[[BatchDatasetRow], None]
BatchSpecCallback = Callable[[BatchRunSpec], None]
BatchMessageCallback = Callable[[str], None]


class BatchCancelToken:
    """Thread-safe cancellation flag shared across GUI and worker threads."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_cancel(self) -> None:
        self._event.set()

    def is_cancel_requested(self) -> bool:
        return self._event.is_set()


def derive_dataset_run_spec(
    base_run_spec: BatchRunSpec | RunSpec | dict,
    dataset_row: BatchDatasetRow,
    *,
    validate_only: bool | None = None,
    overwrite: bool | None = None,
) -> RunSpec:
    """
    Derive one normal RunSpec for a dataset row.

    Only the dataset input/output paths and optional execution controls are
    replaced.  Shared fields from the frozen base spec remain unchanged.
    """
    shared_settings: dict = {}
    if isinstance(base_run_spec, BatchRunSpec):
        source = copy.deepcopy(base_run_spec.base_run_spec)
        shared_settings = copy.deepcopy(base_run_spec.shared_settings)
        if overwrite is None:
            overwrite = bool(base_run_spec.overwrite)
    elif isinstance(base_run_spec, RunSpec):
        source = base_run_spec.to_dict()
    elif isinstance(base_run_spec, dict):
        source = copy.deepcopy(base_run_spec)
    else:
        raise TypeError(
            "base_run_spec must be a BatchRunSpec, RunSpec, or serialized RunSpec dict"
        )

    run_spec_fields = {field.name for field in fields(RunSpec)}
    payload = {k: v for k, v in source.items() if k in run_spec_fields}

    # Shared settings may be supplied separately from the serialized RunSpec.
    # Only apply keys that are actual RunSpec constructor fields.
    for key, value in shared_settings.items():
        if key in run_spec_fields:
            payload[key] = value

    payload["input_dir"] = dataset_row.input_path
    payload["run_dir"] = dataset_row.output_path
    if validate_only is not None:
        payload["validate_only"] = bool(validate_only)
    if overwrite is not None:
        payload["overwrite"] = bool(overwrite)

    return RunSpec(**payload)


class BatchRunner:
    """
    Sequential batch orchestrator with injectable command execution.

    Patch 2 uses simple callbacks rather than Qt signals so the orchestration
    layer stays testable without PySide.  A future GUI dialog can wrap this
    class or adapt the callbacks to signals.
    """

    def __init__(
        self,
        batch_spec: BatchRunSpec,
        *,
        command_runner: BatchCommandRunner | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_row_update: BatchRowCallback | None = None,
        on_batch_update: BatchSpecCallback | None = None,
        on_message: BatchMessageCallback | None = None,
        on_finished: BatchSpecCallback | None = None,
    ) -> None:
        self.batch_spec = batch_spec
        self.command_runner = command_runner or default_subprocess_command_runner
        self._external_cancel_requested = cancel_requested
        self.on_row_update = on_row_update
        self.on_batch_update = on_batch_update
        self.on_message = on_message
        self.on_finished = on_finished
        self._cancel_requested = False
        self._active_row: BatchDatasetRow | None = None

    @property
    def batch_manifest_csv_path(self) -> str:
        return os.path.join(self.batch_spec.batch_output_root, "batch_manifest.csv")

    @property
    def batch_manifest_json_path(self) -> str:
        return os.path.join(self.batch_spec.batch_output_root, "batch_manifest.json")

    @property
    def batch_run_spec_json_path(self) -> str:
        return os.path.join(self.batch_spec.batch_output_root, "batch_run_spec.json")

    @property
    def batch_config_used_yaml_path(self) -> str:
        return os.path.join(self.batch_spec.batch_output_root, "batch_config_used.yaml")

    def request_cancel(self) -> None:
        """Request batch cancellation and write the active run cancel flag if possible."""
        self._cancel_requested = True
        if self._active_row is not None:
            _write_cancel_flag(self._active_row.output_path)

    def is_cancel_requested(self) -> bool:
        external_requested = False
        if self._external_cancel_requested is not None:
            try:
                external_requested = bool(self._external_cancel_requested())
            except Exception:
                external_requested = False
        requested = bool(self._cancel_requested or external_requested)
        if requested and self._active_row is not None:
            _write_cancel_flag(self._active_row.output_path)
        return requested

    def run(self, *, validate_only: bool = False) -> BatchRunSpec:
        """Run pending rows sequentially and update manifests incrementally."""
        os.makedirs(self.batch_spec.batch_output_root, exist_ok=True)
        os.makedirs(self.batch_spec.runs_dir, exist_ok=True)

        # Frozen intent artifacts are written before execution mutates row status.
        frozen_intent = copy.deepcopy(self.batch_spec)
        write_batch_run_spec_json(frozen_intent, self.batch_run_spec_json_path)
        write_batch_config_used_yaml(frozen_intent, self.batch_config_used_yaml_path)
        self._write_live_manifests()

        for index, row in enumerate(self.batch_spec.datasets):
            if self.is_cancel_requested():
                self._cancel_remaining_from(index)
                break
            if row.status == "skipped":
                self._emit_message(f"Skipping {row.dataset_id}: {row.message}")
                self._emit_row_update(row)
                self._write_live_manifests()
                continue
            if row.status != "pending":
                previous_status = row.status
                row.status = "failed"
                row.message = (
                    "Unexpected non-pending row state before execution: "
                    f"{previous_status}"
                )
                self._finalize_row_timing(row, row.started_at)
                self._emit_row_update(row)
                self._write_live_manifests()
                if self.batch_spec.stop_on_failure:
                    self._skip_remaining_after_failure(index + 1)
                    break
                continue

            started = utc_now_iso()
            row.started_at = started
            row.finished_at = None
            row.elapsed_sec = None
            row.status = "validating" if validate_only else "running"
            row.message = ""
            self._active_row = row
            self._emit_row_update(row)
            self._write_live_manifests()

            try:
                run_spec = derive_dataset_run_spec(
                    self.batch_spec,
                    row,
                    validate_only=validate_only,
                    overwrite=self.batch_spec.overwrite,
                )
                argv = self._prepare_dataset_run(run_spec, row)
                result = self.command_runner(
                    argv,
                    row,
                    run_spec,
                    self.is_cancel_requested,
                )
            except Exception as exc:
                result = BatchCommandResult(
                    exit_code=1,
                    stderr=str(exc),
                    message=str(exc),
                )

            if self.is_cancel_requested() or result.cancelled:
                row.status = "cancelled"
                row.message = result.message or "Batch cancellation requested."
                self._finalize_row_paths(row)
                self._finalize_row_timing(row, started)
                self._emit_row_update(row)
                self._write_live_manifests()
                self._cancel_remaining_from(index + 1)
                break

            ok, message = _resolve_dataset_success(
                row.output_path,
                validate_only=validate_only,
                command_result=result,
            )
            if ok:
                row.status = "success"
                row.message = message
            else:
                row.status = "failed"
                row.message = message
            self._finalize_row_paths(row)
            self._finalize_row_timing(row, started)
            self._emit_row_update(row)
            self._write_live_manifests()

            if row.status == "failed" and self.batch_spec.stop_on_failure:
                self._skip_remaining_after_failure(index + 1)
                break

        self._active_row = None
        self.batch_spec.finished_at = utc_now_iso()
        self._write_live_manifests()
        if self.on_finished is not None:
            self.on_finished(self.batch_spec)
        return self.batch_spec

    def _prepare_dataset_run(self, run_spec: RunSpec, row: BatchDatasetRow) -> list[str]:
        """Write normal per-run GUI provenance and return wrapper argv."""
        os.makedirs(row.output_path, exist_ok=True)
        config_path = run_spec.generate_derived_config(row.output_path)
        RunSpec.validate_effective_config(config_path)
        argv = run_spec.build_runner_argv()
        run_spec.write_gui_run_spec(row.output_path)
        run_spec.write_command_invoked(row.output_path, argv)
        return argv

    def _write_live_manifests(self) -> None:
        write_batch_manifest_csv(self.batch_spec, self.batch_manifest_csv_path)
        write_batch_manifest_json(self.batch_spec, self.batch_manifest_json_path)
        if self.on_batch_update is not None:
            self.on_batch_update(self.batch_spec)

    def _emit_row_update(self, row: BatchDatasetRow) -> None:
        if self.on_row_update is not None:
            self.on_row_update(copy.deepcopy(row))

    def _emit_message(self, message: str) -> None:
        if self.on_message is not None:
            self.on_message(message)

    def _finalize_row_paths(self, row: BatchDatasetRow) -> None:
        row.run_dir = row.output_path
        status_path = os.path.join(row.output_path, "status.json")
        report_path = os.path.join(row.output_path, "run_report.json")
        row.status_json_path = status_path if os.path.isfile(status_path) else None
        row.run_report_path = report_path if os.path.isfile(report_path) else None

    def _finalize_row_timing(self, row: BatchDatasetRow, started_at: str | None) -> None:
        row.finished_at = utc_now_iso()
        if started_at:
            row.elapsed_sec = _elapsed_sec_from_iso(started_at, row.finished_at)

    def _skip_remaining_after_failure(self, start_index: int) -> None:
        for row in self.batch_spec.datasets[start_index:]:
            if row.status == "pending":
                row.status = "skipped"
                row.message = "Skipped because stop_on_failure was enabled."
                self._emit_row_update(row)
        self._write_live_manifests()

    def _cancel_remaining_from(self, start_index: int) -> None:
        for row in self.batch_spec.datasets[start_index:]:
            if row.status == "pending":
                row.status = "cancelled"
                row.message = "Cancelled before dataset execution started."
                self._emit_row_update(row)
        self._write_live_manifests()


def default_subprocess_command_runner(
    argv: list[str],
    row: BatchDatasetRow,
    _run_spec: RunSpec,
    cancel_requested: Callable[[], bool],
) -> BatchCommandResult:
    """Run one dataset command and persist GUI-style stdout/stderr artifacts."""
    os.makedirs(row.output_path, exist_ok=True)
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        message = f"Failed to launch batch dataset command: {exc}"
        _write_process_logs(row.output_path, "", message)
        return BatchCommandResult(exit_code=1, stderr=message, message=message)

    while proc.poll() is None:
        if cancel_requested():
            _write_cancel_flag(row.output_path)
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            stdout = stdout or ""
            stderr = stderr or ""
            _write_process_logs(row.output_path, stdout, stderr)
            return BatchCommandResult(
                exit_code=proc.returncode if proc.returncode is not None else 130,
                stdout=stdout,
                stderr=stderr,
                cancelled=True,
                message="Batch cancellation requested.",
            )
        time.sleep(0.1)

    stdout, stderr = proc.communicate()
    stdout = stdout or ""
    stderr = stderr or ""
    _write_process_logs(row.output_path, stdout, stderr)
    return BatchCommandResult(
        exit_code=int(proc.returncode or 0),
        stdout=stdout,
        stderr=stderr,
    )


def _resolve_dataset_success(
    run_dir: str,
    *,
    validate_only: bool,
    command_result: BatchCommandResult,
) -> tuple[bool, str]:
    status_path = os.path.join(run_dir, "status.json")
    status_code, status_errors = _read_final_status(status_path, is_finished=True)
    if status_code == "success":
        if validate_only:
            return True, "validate-only success"
        return True, "success"
    if not validate_only:
        completed_ok, completed_reason = is_successful_completed_run_dir(run_dir)
        if completed_ok:
            return True, completed_reason

    details = []
    if status_code:
        details.append(f"status={status_code}")
    if status_errors:
        details.append("; ".join(str(e) for e in status_errors))
    if command_result.message:
        details.append(command_result.message)
    stderr_tail = _tail_text(command_result.stderr)
    if stderr_tail:
        details.append(stderr_tail)
    details.append(f"exit_code={command_result.exit_code}")
    return False, " | ".join(details)


def _write_cancel_flag(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "CANCEL.REQUESTED"), "w", encoding="utf-8") as f:
        f.write("cancelled by batch runner\n")


def _write_process_logs(run_dir: str, stdout: str, stderr: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout or "")
    with open(os.path.join(run_dir, "stderr.txt"), "w", encoding="utf-8") as f:
        f.write(stderr or "")


def _tail_text(text: str, *, max_chars: int = 500) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    return clean[-max_chars:]


def _elapsed_sec_from_iso(started_at: str, finished_at: str) -> float:
    try:
        start = _parse_iso(started_at)
        finish = _parse_iso(finished_at)
        return max(0.0, (finish - start).total_seconds())
    except Exception:
        return 0.0


def _parse_iso(value: str):
    return datetime.fromisoformat(value)
