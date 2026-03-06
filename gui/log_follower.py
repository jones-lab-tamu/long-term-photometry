"""
LogFollower - QTimer-based tailing of stdout.txt and stderr.txt.

Monitors output files in the run directory and emits new lines as they appear.
Maintains byte offsets for each file to ensure efficient polling.
"""

import os
from PySide6.QtCore import QObject, QTimer, Signal

class LogFollower(QObject):
    """Tails stdout.txt and stderr.txt from a run directory."""

    # signal emitted for each new line, prefixed with "OUT: " or "ERR: "
    line_received = Signal(str)

    def __init__(self, run_dir: str, poll_ms: int = 500, parent=None):
        super().__init__(parent)
        self._run_dir = run_dir
        self._poll_ms = poll_ms
        
        self._stdout_path = os.path.join(run_dir, "stdout.txt")
        self._stderr_path = os.path.join(run_dir, "stderr.txt")
        
        self._stdout_offset = 0
        self._stderr_offset = 0

        self._stdout_carry = ""
        self._stderr_carry = ""
        
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        """Begin tailing the log files."""
        self._stdout_offset = 0
        self._stderr_offset = 0
        self._stdout_carry = ""
        self._stderr_carry = ""
        self._timer.start()

    def stop(self) -> None:
        """Stop tailing immediately and flush remains."""
        self._timer.stop()
        # Final poll to catch any last bits
        self._poll()
        # Flush any remaining carry
        if self._stdout_carry:
            self.line_received.emit(f"OUT: {self._stdout_carry}")
            self._stdout_carry = ""
        if self._stderr_carry:
            self.line_received.emit(f"ERR: {self._stderr_carry}")
            self._stderr_carry = ""

    def _poll(self) -> None:
        """Check both files for new bytes."""
        self._tail_file(self._stdout_path, "OUT")
        self._tail_file(self._stderr_path, "ERR")

    def _tail_file(self, path: str, prefix: str) -> None:
        """Read any new bytes from path and emit complete lines."""
        if not os.path.isfile(path):
            return

        try:
            file_size = os.path.getsize(path)
            
            offset_attr = f"_{'stdout' if prefix == 'OUT' else 'stderr'}_offset"
            carry_attr = f"_{'stdout' if prefix == 'OUT' else 'stderr'}_carry"
            current_offset = getattr(self, offset_attr)

            if file_size < current_offset:
                # If file shrank, reset offset and carry
                setattr(self, offset_attr, 0)
                setattr(self, carry_attr, "")
                current_offset = 0

            if file_size <= current_offset:
                return

            with open(path, "rb") as f:
                f.seek(current_offset)
                new_data = f.read(file_size - current_offset)
                
            if not new_data:
                return

            # Update offset
            setattr(self, offset_attr, current_offset + len(new_data))

            # Decode and merge with carry
            text = new_data.decode("utf-8", errors="replace")
            combined = getattr(self, carry_attr) + text
            
            # splitlines(keepends=True) lets us identify terminated vs unterminated lines
            lines = combined.splitlines(keepends=True)
            
            if not lines:
                return

            # Determine if the last item is a complete line
            if lines[-1].endswith(('\n', '\r')):
                # All are complete or last is also complete
                emit_lines = [line.rstrip('\r\n') for line in lines]
                setattr(self, carry_attr, "")
            else:
                # Last item is a fragment
                emit_lines = [line.rstrip('\r\n') for line in lines[:-1]]
                setattr(self, carry_attr, lines[-1])

            for line in emit_lines:
                self.line_received.emit(f"{prefix}: {line}")
                
        except Exception:
            pass
