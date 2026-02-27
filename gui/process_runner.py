"""
PipelineRunner, QProcess wrapper for tools/run_full_pipeline_deliverables.py

Streams stdout/stderr line-by-line, supports cancel with process-tree kill on Windows.
Cancel flag file is the primary cancellation mechanism; QProcess kill is fallback.

Tracks explicit RunnerState and persists stdout/stderr to run_dir log files.
"""

import enum
import os
import sys
import subprocess
import json

from PySide6.QtCore import QObject, QProcess, Signal


class RunnerState(enum.Enum):
    IDLE = "IDLE"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    FAIL_CLOSED = "FAIL_CLOSED"
    CANCELLED = "CANCELLED"


class PipelineRunner(QObject):
    """Runs the deliverables pipeline via QProcess with line-buffered log streaming."""

    # Signals
    log_line = Signal(str)      # Each emission is ONE complete line with "OUT: " or "ERR: " prefix
    started = Signal()
    finished = Signal(int)      # exit code
    error = Signal(str)         # fatal launch errors only
    state_changed = Signal(str) # RunnerState.value string

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._stdout_buf = b""
        self._stderr_buf = b""
        self._run_dir = ""
        self._state = RunnerState.IDLE
        self._cancel_requested = False
        # Log file handles
        self._stdout_log = None
        self._stderr_log = None
        # Fail-closed feedback
        self.final_status_code = None
        self.final_errors = []
        self.fail_closed_code = None
        self.fail_closed_detail = None
        self.fail_closed_remediation = None
        self.validation_summary = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def was_cancel_requested(self) -> bool:
        return self._cancel_requested

    def set_run_dir(self, run_dir: str) -> None:
        """Set the run directory and reset all per-run parsing state."""
        self._run_dir = run_dir
        self._stdout_buf = b""
        self._stderr_buf = b""
        self.final_status_code = None
        self.final_errors = []
        self.fail_closed_code = None
        self.fail_closed_detail = None
        self.fail_closed_remediation = None
        self.validation_summary = None

    def _set_state(self, new_state: RunnerState) -> None:
        self._state = new_state
        self.state_changed.emit(new_state.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, argv: list, state: RunnerState = RunnerState.RUNNING) -> None:
        """Start the pipeline. argv[0] is the program, argv[1:] are arguments."""
        if self.is_running():
            self.error.emit("A process is already running.")
            return

        self._stdout_buf = b""
        self._stderr_buf = b""
        self._cancel_requested = False

        # Open log files if run_dir exists
        self._open_log_files()

        self._set_state(state)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.SeparateChannels)

        # Connect signals
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.started.connect(self._on_started)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

        program = argv[0]
        args = argv[1:] if len(argv) > 1 else []
        self._process.start(program, args)

    def cancel(self) -> None:
        """Cancel the running process. Writes cancel flag first, then fallback kill."""
        if not self.is_running():
            return

        self._cancel_requested = True

        # Step 0: write cancel flag (primary cancellation mechanism)
        if self._run_dir:
            cancel_flag = os.path.join(self._run_dir, "CANCEL.REQUESTED")
            try:
                with open(cancel_flag, "w") as f:
                    f.write("cancelled by gui\n")
                self.log_line.emit("OUT: Cancel flag written, waiting for engine to stop...")
            except Exception as e:
                self.log_line.emit(f"ERR: Could not write cancel flag: {e}")

        self.log_line.emit("ERR: Cancel requested, terminating process...")

        # Step 1: polite terminate
        self._process.terminate()
        if self._process.waitForFinished(3000):
            return

        # Step 2: forceful kill
        if sys.platform == "win32":
            pid = self._process.processId()
            if pid:
                try:
                    subprocess.run(
                        ["taskkill", "/T", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=5
                    )
                except Exception as e:
                    self.log_line.emit(f"ERR: taskkill failed: {e}")
        else:
            self.log_line.emit(
                "ERR: Cancel requested, child processes may survive on POSIX "
                "without psutil/process-group support."
            )

        self._process.kill()
        self._process.waitForFinished(2000)

    def is_running(self) -> bool:
        return (self._process is not None
                and self._process.state() != QProcess.NotRunning)

    # ------------------------------------------------------------------
    # Log file management
    # ------------------------------------------------------------------

    def _open_log_files(self) -> None:
        """Open stdout.log and stderr.log in run_dir if it exists.

        Uses write mode ("w") so each process start truncates old content,
        preventing false validation gating from stale log data.
        """
        self._close_log_files()
        if self._run_dir and os.path.isdir(self._run_dir):
            try:
                self._stdout_log = open(
                    os.path.join(self._run_dir, "stdout.log"),
                    "w", encoding="utf-8"
                )
            except OSError:
                self._stdout_log = None
            try:
                self._stderr_log = open(
                    os.path.join(self._run_dir, "stderr.log"),
                    "w", encoding="utf-8"
                )
            except OSError:
                self._stderr_log = None

    def _close_log_files(self) -> None:
        """Close log file handles if open."""
        for fh in (self._stdout_log, self._stderr_log):
            if fh:
                try:
                    fh.close()
                except OSError:
                    pass
        self._stdout_log = None
        self._stderr_log = None

    # ------------------------------------------------------------------
    # Private slots and test hooks
    # ------------------------------------------------------------------

    def _handle_stdout_line(self, line: str) -> None:
        """Hook for stdout line processing. Emits signal and writes to log file."""
        self.log_line.emit("OUT: " + line)
        if self._stdout_log:
            try:
                self._stdout_log.write(line + "\n")
                self._stdout_log.flush()
            except OSError:
                pass

    def _handle_stderr_line(self, line: str) -> None:
        """Hook for stderr line processing. Emits signal and writes to log file."""
        self.log_line.emit("ERR: " + line)
        if self._stderr_log:
            try:
                self._stderr_log.write(line + "\n")
                self._stderr_log.flush()
            except OSError:
                pass

    def _on_stdout(self):
        if self._process:
            data = self._process.readAllStandardOutput().data()
            self._stdout_buf += data
            self._flush_buffer("stdout")

    def _on_stderr(self):
        if self._process:
            data = self._process.readAllStandardError().data()
            self._stderr_buf += data
            self._flush_buffer("stderr")

    def _flush_buffer(self, channel: str):
        """Split buffer on newlines, emit complete lines, keep partial tail."""
        if channel == "stdout":
            buf = self._stdout_buf
            handler = self._handle_stdout_line
        else:
            buf = self._stderr_buf
            handler = self._handle_stderr_line

        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            handler(line)

        if channel == "stdout":
            self._stdout_buf = buf
        else:
            self._stderr_buf = buf

    def _flush_remaining(self):
        """Emit any trailing partial lines on process exit."""
        if self._stdout_buf:
            line = self._stdout_buf.decode("utf-8", errors="replace").rstrip("\r")
            if line:
                self._handle_stdout_line(line)
            self._stdout_buf = b""

        if self._stderr_buf:
            line = self._stderr_buf.decode("utf-8", errors="replace").rstrip("\r")
            if line:
                self._handle_stderr_line(line)
            self._stderr_buf = b""

    def _determine_final_state(self, exit_code: int) -> RunnerState:
        """Determine final state using evidence order per design spec C.

        Evidence order:
        1) cancel_requested flag + nonzero exit -> CANCELLED
        2) exit code 0 -> SUCCESS
        3) else -> FAILED
        Note: events-based cancellation detection is handled at the GUI layer.
        """
        if self._cancel_requested and exit_code != 0:
            return RunnerState.CANCELLED
        if exit_code == 0:
            return RunnerState.SUCCESS
        return RunnerState.FAILED

    def _resolve_final_state(self, code: str | None, exit_code: int) -> RunnerState:
        """Determines final state from status code and populates fail-closed details."""
        self.fail_closed_code = None
        self.fail_closed_detail = None
        self.fail_closed_remediation = None

        # 1. Authoritative status check
        if code in ("success", "error", "cancelled"):
            if code == "success":
                return RunnerState.SUCCESS
            if code == "cancelled":
                return RunnerState.CANCELLED
            return RunnerState.FAILED

        # 2. Sentinels -> FAIL_CLOSED
        if code in ("MISSING_FILE", "MALFORMED_STATUS", "SCHEMA_MISMATCH",
                    "NOT_FINAL", "NONFINAL_WITH_EXIT", "MISSING_STATUS", "BAD_STATUS"):
            self.fail_closed_code = code
            mapping = {
                "MISSING_FILE": ("status.json was never created or is unreadable.", "Verify run directory is writeable and disk is not full."),
                "MALFORMED_STATUS": ("status.json is malformed or invalid JSON.", "Inspect status.json for corruption or partial writes."),
                "SCHEMA_MISMATCH": ("status.json schema_version is missing or incorrect.", "Check tool version or clear output directory."),
                "NOT_FINAL": ("status.json exists but phase is not 'final'.", "Wait for the process to fully complete or check for hangs."),
                "NONFINAL_WITH_EXIT": ("Process exited but status never reached 'final' phase.", "Check stderr.log for crashes or early exits."),
                "MISSING_STATUS": ("status.json is missing required 'status' field.", "Review run script logic or inspect status.json."),
                "BAD_STATUS": ("status.json 'status' field has unrecognized value.", "Verify contract adherence in run script.")
            }
            msg, rem = mapping.get(code, ("Incomplete or invalid status contract.", "Check logs."))
            self.fail_closed_detail = msg
            self.fail_closed_remediation = rem
            return RunnerState.FAIL_CLOSED

        # 3. Fallback for cancellation
        if self._cancel_requested and exit_code != 0:
            return RunnerState.CANCELLED

        # 4. Strictly fail-closed fallback
        self.fail_closed_code = code or "UNKNOWN_ERROR"
        self.fail_closed_detail = "Strict contract check failed; status.json is missing or invalid."
        self.fail_closed_remediation = "Consult logs to determine why the status contract was not satisfied."
        return RunnerState.FAIL_CLOSED

    def _on_started(self):
        self.started.emit()

    def _on_finished(self, exit_code, _exit_status):
        # FINAL DRAIN: Ensure we capture any bytes remaining in QProcess buffers.
        if self._process:
            self._stdout_buf += self._process.readAllStandardOutput().data()
            self._flush_buffer("stdout")
            
            self._stderr_buf += self._process.readAllStandardError().data()
            self._flush_buffer("stderr")

        self._flush_remaining()
        
        # Resolve final state using status contract if available
        code = None
        errors = []
        if self._run_dir:
            status_path = os.path.join(self._run_dir, "status.json")
            code, errors = _read_final_status(status_path, is_finished=True)
            self.final_status_code = code
            self.final_errors = errors
        
        if self._run_dir and code is not None:
            final = self._resolve_final_state(code, exit_code)
        else:
            final = self._determine_final_state(exit_code)
            
        self._set_state(final)
        self._close_log_files()
        self.finished.emit(exit_code)

    def _on_error(self, proc_error):
        error_map = {
            QProcess.FailedToStart: "Process failed to start",
            QProcess.Crashed: "Process crashed",
            QProcess.Timedout: "Process timed out",
            QProcess.WriteError: "Write error",
            QProcess.ReadError: "Read error",
            QProcess.UnknownError: "Unknown error",
        }
        msg = error_map.get(proc_error, f"QProcess error code {proc_error}")
        self.error.emit(msg)


def _read_final_status(status_path: str, is_finished: bool = False) -> tuple[str | None, list[str]]:
    """Read and validate status.json, returning (status_or_code, errors_list).

    Returns (status_or_code, errors_list).

    Codes:
      "MISSING_FILE": File does not exist
      "MALFORMED_STATUS": JSON parse failed
      "NOT_FINAL": phase != "final" (process still running)
      "NONFINAL_WITH_EXIT": phase != "final" but process has exited
      "SCHEMA_MISMATCH": schema_version missing OR != 1 OR not int
      "MISSING_STATUS": phase="final" but status key missing
      "BAD_STATUS": status present but not in {success, error, cancelled}
      "success"|"error"|"cancelled": Valid final status
    """
    if not os.path.isfile(status_path):
        return "MISSING_FILE", []

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "MALFORMED_STATUS", []

    phase = data.get("phase")
    if phase != "final":
        if is_finished:
            return "NONFINAL_WITH_EXIT", []
        return "NOT_FINAL", []

    # Enforce schema versioning
    sv = data.get("schema_version")
    if sv is None or not isinstance(sv, int) or sv != 1:
        return "SCHEMA_MISMATCH", []

    status = data.get("status")
    if status is None:
        return "MISSING_STATUS", []

    if status not in ("success", "error", "cancelled"):
        return "BAD_STATUS", []

    # Valid final status
    errors = data.get("errors", [])
    return status, errors if isinstance(errors, list) else []
