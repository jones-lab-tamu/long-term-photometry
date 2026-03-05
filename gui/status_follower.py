"""
StatusFollower - QTimer-based status.json polling.

Polls the status.json file at a configurable interval, parses the JSON,
and emits signals for state changes. Handles:
  - File not yet existing (keeps polling)
  - Partial writes / Invalid JSON (keeps last known good state, shows updating)
  - Missing keys (preserves last known good fields)
  - Terminal state detection
"""

import json
import os

from PySide6.QtCore import QObject, QTimer, Signal


class StatusFollower(QObject):
    """Polls a status.json file for progress updates."""

    # Signals
    status_received = Signal(dict)   # Emits the merged last-known-good status dict
    terminal_reached = Signal(str)   # Emits the terminal status string (e.g. "success")
    parse_error = Signal(str)        # Emits when a read fails or is incomplete with reason msg
    status_warning = Signal(str)     # Emits for unknown status values or non-critical anomalies

    # Terminal states strictly emitted by tools/run_full_pipeline_deliverables.py
    # fail_closed is NOT in this set as it is interpreted by the GUI on runner exit.
    TERMINAL_SET = {"success", "error", "cancelled"}
    KNOWN_NONTERMINAL_SET = {"running"}

    def __init__(self, status_path: str, poll_ms: int = 500, parent=None):
        super().__init__(parent)
        self._status_path = status_path
        self._poll_ms = poll_ms
        
        # Last known good state preservation
        self._last_good_status = {}
        self._last_warning_val = None
        
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin polling the status file."""
        self._last_good_status = {}
        self._last_warning_val = None
        self._timer.start()

    def stop(self) -> None:
        """Stop polling immediately."""
        self._timer.stop()
        
    def get_last_good_status(self) -> dict:
        return dict(self._last_good_status)

    @property
    def is_active(self) -> bool:
        return self._timer.isActive()

    # ------------------------------------------------------------------
    # Polling logic
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Read and parse status.json while handling partial writes."""
        if not os.path.isfile(self._status_path):
            self.parse_error.emit("status.json missing")
            return

        try:
            with open(self._status_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            # json.JSONDecodeError or OSError (e.g., file locked during write)
            self.parse_error.emit(f"read/parse error: {e}")
            return
            
        if not isinstance(data, dict):
            self.parse_error.emit("status.json is not an object")
            return
            
        # Merge valid keys into last_good_status (preserves existing valid keys if missing from new payload)
        self._last_good_status.update(data)
        
        self.status_received.emit(self._last_good_status)
        
        # Check for terminal state
        status_val = self._last_good_status.get("status")
        if status_val in self.TERMINAL_SET:
            self.stop()
            self.terminal_reached.emit(status_val)
        elif status_val is not None and status_val not in self.KNOWN_NONTERMINAL_SET:
            # Unknown status policy: emit warning but continue polling.
            # Deduplicate to avoid spamming the UI.
            if status_val != self._last_warning_val:
                self._last_warning_val = status_val
                self.status_warning.emit(f"unknown status: {status_val}")
