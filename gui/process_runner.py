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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def was_cancel_requested(self) -> bool:
        return self._cancel_requested

    def _set_state(self, new_state: RunnerState) -> None:
        self._state = new_state
        self.state_changed.emit(new_state.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_run_dir(self, path: str) -> None:
        """Set the run directory so cancel can write the flag file."""
        self._run_dir = path

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
    # Private slots
    # ------------------------------------------------------------------

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data()
        self._stdout_buf += data
        self._flush_buffer("stdout")

    def _on_stderr(self):
        data = self._process.readAllStandardError().data()
        self._stderr_buf += data
        self._flush_buffer("stderr")

    def _flush_buffer(self, channel: str):
        """Split buffer on newlines, emit complete lines, keep partial tail."""
        if channel == "stdout":
            buf = self._stdout_buf
            prefix = "OUT: "
            log_fh = self._stdout_log
        else:
            buf = self._stderr_buf
            prefix = "ERR: "
            log_fh = self._stderr_log

        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            self.log_line.emit(prefix + line)
            if log_fh:
                try:
                    log_fh.write(line + "\n")
                    log_fh.flush()
                except OSError:
                    pass

        if channel == "stdout":
            self._stdout_buf = buf
        else:
            self._stderr_buf = buf

    def _flush_remaining(self):
        """Emit any trailing partial lines on process exit."""
        for channel, buf, prefix, log_fh in [
            ("stdout", self._stdout_buf, "OUT: ", self._stdout_log),
            ("stderr", self._stderr_buf, "ERR: ", self._stderr_log),
        ]:
            if buf:
                line = buf.decode("utf-8", errors="replace").rstrip("\r")
                if line:
                    self.log_line.emit(prefix + line)
                    if log_fh:
                        try:
                            log_fh.write(line + "\n")
                            log_fh.flush()
                        except OSError:
                            pass
        self._stdout_buf = b""
        self._stderr_buf = b""

    def _on_started(self):
        self.started.emit()

    def _on_error(self, error: QProcess.ProcessError):
        """Handle launch errors (e.g. executable not found)."""
        if self._state != RunnerState.FAILED:
            self._set_state(RunnerState.FAILED)
            self.error.emit(f"Launch error: {error}")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        self.log_line.emit(f"OUT: Process finished with exit code {exit_code}")
        self._flush_remaining()
        self._close_log_files()

        final_state = self._compute_final_state(exit_code)
        self._set_state(final_state)
        self.finished.emit(exit_code)

    def _compute_final_state(self, exit_code: int) -> RunnerState:
        """Determines final state from status.json (authoritative) or process fallback.

        FAIL-CLOSED POLICY:
        1. If status.json exists and is valid/final:
           - "success" -> SUCCESS
           - "cancelled" -> CANCELLED
           - "error" -> FAILED
        2. If status.json exists but is malformed, non-final, or missing required fields:
           - return FAILED (Fail-Closed)
        3. Only if status.json is missing entirely:
           - cancel_requested + exit_code != 0 -> CANCELLED
           - exit_code == 0 -> SUCCESS
           - else -> FAILED
        """
        # 1. Authoritative check: status.json
        if self._run_dir:
            status_path = os.path.join(self._run_dir, "status.json")
            st, _ = _read_final_status(status_path)

            if st is not None:
                if st == "success":
                    return RunnerState.SUCCESS
                if st == "cancelled":
                    return RunnerState.CANCELLED
                # Any sentinel code ("__MALFORMED__", "__NOT_FINAL__", 
                # "__MISSING_SCHEMA__", "__BAD_SCHEMA__", etc.)
                # or "error" status -> FAILED (Fail-Closed)
                return RunnerState.FAILED

        # 2. Fallback (only if status.json missing)
        if self._cancel_requested and exit_code != 0:
            return RunnerState.CANCELLED

        if exit_code == 0:
            return RunnerState.SUCCESS
        return RunnerState.FAILED


def _read_final_status(status_path: str) -> tuple[str | None, list[str]]:
    """
    Reads status.json and determines the final status.
    Returns (status_or_code, errors_list).
    
    Codes:
      None: File missing
      "__MALFORMED__": JSON parse failed
      "__NOT_FINAL__": phase != "final"
      "__MISSING_SCHEMA__": schema_version field missing
      "__BAD_SCHEMA__": schema_version not integer or not 1
      "__MISSING_STATUS__": phase="final" but status missing
      "__BAD_STATUS__": status present but not in {success, error, cancelled}
      "success"|"error"|"cancelled": Valid final status
    """
    if not os.path.isfile(status_path):
        return None, []
    
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "__MALFORMED__", []

    phase = data.get("phase")
    if phase != "final":
        return "__NOT_FINAL__", []
    
    # Enforce schema versioning (Requirement 1)
    sv = data.get("schema_version")
    if sv is None:
        return "__MISSING_SCHEMA__", []
    if not isinstance(sv, int) or sv != 1:
        return "__BAD_SCHEMA__", []
    
    status = data.get("status")
    if status is None:
        return "__MISSING_STATUS__", []
    
    if status not in ("success", "error", "cancelled"):
        return "__BAD_STATUS__", []
        
    # Valid final status
    errors = data.get("errors", [])
    if not isinstance(errors, list):
        errors = []
        
    return status, errors
