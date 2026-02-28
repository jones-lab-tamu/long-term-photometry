"""
EventEmitter — NDJSON event writer for pipeline lifecycle events.

This module provides a single writer class that appends structured
NDJSON event lines to a file.  It is used by:
  - tools/run_full_pipeline_deliverables.py  (parent orchestrator)
  - analyze_photometry.py                    (child, append-mode)
"""

import os
import json
from datetime import datetime


def _normalize_event_dict(event: dict) -> dict:
    """Ensure event dict has schema_version: 1 (int), overriding any existing key."""
    obj = event.copy()
    obj["schema_version"] = 1
    return obj


class EventEmitter:
    """Writes NDJSON events to events_path, one JSON object per line."""

    def __init__(self, events_path, run_id, run_dir, file_mode="a",
                 allow_makedirs=True):
        self._run_id = run_id
        self._run_dir = run_dir
        self._fh = None
        if events_path:
            parent = os.path.dirname(events_path) or "."
            if allow_makedirs:
                os.makedirs(parent, exist_ok=True)
                self._fh = open(events_path, file_mode, encoding="utf-8")
            else:
                # Side-effect free: only open if parent already exists
                if os.path.isdir(parent):
                    self._fh = open(events_path, file_mode, encoding="utf-8")
                # else: stay disabled (self._fh remains None)

    def emit(self, stage, event_type, message, **kwargs):
        """Emit one NDJSON event line. schema_version: 1 is forced."""
        raw_obj = {
            "time_iso": datetime.now().isoformat(),
            "run_id": self._run_id,
            "run_dir": self._run_dir,
            "stage": stage,
            "type": event_type,
            "message": message,
            **kwargs,
        }

        # Producer-side discipline: force schema_version: 1 (int)
        obj = _normalize_event_dict(raw_obj)

        if self._fh:
            # separators=(",", ":") for compact NDJSON (standard compliant)
            self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
