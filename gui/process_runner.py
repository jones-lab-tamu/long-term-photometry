"""
PipelineRunner — QProcess wrapper for tools/run_full_pipeline_deliverables.py

Streams stdout/stderr line-by-line, supports cancel with process-tree kill on Windows.
"""

import sys
import subprocess

from PySide6.QtCore import QObject, QProcess, Signal


class PipelineRunner(QObject):
    """Runs the deliverables pipeline via QProcess with line-buffered log streaming."""

    # Signals
    log_line = Signal(str)      # Each emission is ONE complete line with "OUT: " or "ERR: " prefix
    started = Signal()
    finished = Signal(int)      # exit code
    error = Signal(str)         # fatal launch errors only

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._stdout_buf = b""
        self._stderr_buf = b""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, argv: list) -> None:
        """Start the pipeline. argv[0] is the program, argv[1:] are arguments."""
        if self.is_running():
            self.error.emit("A process is already running.")
            return

        self._stdout_buf = b""
        self._stderr_buf = b""

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
        """Terminate the running process (and its children on Windows)."""
        if not self.is_running():
            return

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
        else:
            buf = self._stderr_buf
            prefix = "ERR: "

        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            self.log_line.emit(prefix + line)

        if channel == "stdout":
            self._stdout_buf = buf
        else:
            self._stderr_buf = buf

    def _flush_remaining(self):
        """Emit any trailing partial lines on process exit."""
        for channel, buf, prefix in [
            ("stdout", self._stdout_buf, "OUT: "),
            ("stderr", self._stderr_buf, "ERR: "),
        ]:
            if buf:
                line = buf.decode("utf-8", errors="replace").rstrip("\r")
                if line:
                    self.log_line.emit(prefix + line)
        self._stdout_buf = b""
        self._stderr_buf = b""

    def _on_started(self):
        self.started.emit()

    def _on_finished(self, exit_code, _exit_status):
        self._flush_remaining()
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
