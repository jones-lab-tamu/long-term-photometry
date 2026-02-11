"""
EventsFollower — QTimer-based NDJSON tailer for events.ndjson.

Polls the events file at a configurable interval, parses new complete lines,
and emits signals for each event.  Handles:
  - File not yet existing (keeps polling)
  - Partial lines (remainder buffer, never emits incomplete JSON)
  - Malformed JSON (emits parse_error, continues)
  - Drain window after process finishes
"""

import json
import os

from PySide6.QtCore import QObject, QTimer, Signal


class EventsFollower(QObject):
    """Tails an NDJSON events file using byte-offset tracking."""

    # Signals
    event_received = Signal(dict)   # Each fully parsed JSON event
    parse_error = Signal(str)       # Non-fatal: bad JSON line description

    def __init__(self, events_path: str, poll_ms: int = 300, parent=None):
        super().__init__(parent)
        self._events_path = events_path
        self._poll_ms = poll_ms
        self._offset = 0           # byte offset into the file
        self._remainder = b""      # trailing partial line from last poll
        self._idle_polls = 0       # consecutive polls with no new data
        self._drain_mode = False   # set after process finishes
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin polling the events file."""
        self._offset = 0
        self._remainder = b""
        self._idle_polls = 0
        self._drain_mode = False
        self._timer.start()

    def stop(self) -> None:
        """Stop polling immediately."""
        self._timer.stop()

    def begin_drain(self) -> None:
        """Enter drain mode: keep polling until no new data for 2 polls."""
        self._drain_mode = True
        self._idle_polls = 0

    @property
    def is_active(self) -> bool:
        return self._timer.isActive()

    # ------------------------------------------------------------------
    # Polling logic
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Read new bytes from events file, parse complete lines."""
        if not os.path.isfile(self._events_path):
            # File not yet created; keep polling
            if self._drain_mode:
                self._idle_polls += 1
                if self._idle_polls >= 2:
                    self.stop()
            return

        try:
            with open(self._events_path, "rb") as fh:
                fh.seek(self._offset)
                new_data = fh.read()
        except OSError:
            # File might be locked or unavailable momentarily
            return

        if not new_data:
            self._idle_polls += 1
            if self._drain_mode and self._idle_polls >= 2:
                # Drain complete: flush remainder if any, then stop
                self._flush_remainder()
                self.stop()
            return

        self._idle_polls = 0
        self._offset += len(new_data)

        # Prepend prior remainder
        data = self._remainder + new_data
        self._remainder = b""

        # Split on newlines
        segments = data.split(b"\n")

        # All segments except the last are complete lines.
        # The last segment is a remainder (could be empty if data ended with \n).
        for segment in segments[:-1]:
            self._parse_line(segment)

        # Store trailing partial (may be empty bytes)
        self._remainder = segments[-1]

    def _flush_remainder(self) -> None:
        """Drain-safe: attempt to parse trailing remainder.

        Only emits event_received if json.loads succeeds.
        Partial/invalid JSON is silently dropped (never emits parse_error)
        to uphold the 'never emits incomplete JSON' contract.
        """
        if self._remainder:
            try:
                text = self._remainder.decode("utf-8", errors="replace").strip()
            except Exception:
                self._remainder = b""
                return
            self._remainder = b""
            if not text:
                return
            try:
                obj = json.loads(text)
                self.event_received.emit(obj)
            except json.JSONDecodeError:
                pass  # silently drop partial fragment during drain

    def _parse_line(self, raw: bytes) -> None:
        """Decode and parse a single NDJSON line."""
        try:
            text = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        try:
            obj = json.loads(text)
            self.event_received.emit(obj)
        except json.JSONDecodeError as exc:
            self.parse_error.emit(f"Bad JSON: {text[:120]}... ({exc})")
