
import sys
import os
import json
import time
import unittest
import tempfile
import shutil
from PySide6.QtWidgets import QApplication
from gui.events_follower import EventsFollower

class TestEventsFollowerDrain(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        self.test_dir = tempfile.mkdtemp()
        self.events_path = os.path.join(self.test_dir, "test_events_drain.ndjson")
        self.status_path = os.path.join(self.test_dir, "status.json")
        self.received = []
        self.errors = []

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _on_event(self, obj):
        self.received.append(obj)

    def _on_error(self, msg):
        self.errors.append(msg)

    def _connect(self, follower):
        follower.event_received.connect(self._on_event)
        follower.parse_error.connect(self._on_error)

    def test_missing_file_drain_success(self):
        """Test that trailing valid remainder is emitted even if file is missing during drain."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        follower.start()

        # 1. Write a partial fragment with schema_version: 1
        with open(self.events_path, "wb") as f:
            f.write(b'{"schema_version":1,"key":"value"}') # Valid JSON fragment but no newline
        
        # Poll once to read it into remainder
        for _ in range(5):
            self.app.processEvents()
            time.sleep(0.02)
        
        self.assertEqual(len(self.received), 0, "Should not emit partial line yet")

        # 2. Delete file and enter drain mode
        os.remove(self.events_path)
        follower.begin_drain()

        # 3. Wait for drain (2 idle polls)
        timeout = time.time() + 2.0
        while follower.is_active and time.time() < timeout:
            self.app.processEvents()
            time.sleep(0.05)
        
        self.assertFalse(follower.is_active, "Follower should have stopped")
        self.assertEqual(len(self.received), 1, "Should have flushed remainder on missing-file drain")
        self.assertEqual(self.received[0]["key"], "value")
        self.assertEqual(len(self.errors), 0, "Should not produce parse error")

    def test_missing_file_drain_silent_drop_malformed(self):
        """Test that invalid remainder is silently dropped (no error) if file missing during drain."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        follower.start()

        # 1. Write a malformed fragment
        with open(self.events_path, "wb") as f:
            f.write(b'{"incomplete": ')
        
        # Poll to buffer
        for _ in range(5):
            self.app.processEvents()
            time.sleep(0.02)
        
        # 2. Delete and drain
        os.remove(self.events_path)
        follower.begin_drain()

        # 3. Wait for drain
        timeout = time.time() + 2.0
        while follower.is_active and time.time() < timeout:
            self.app.processEvents()
            time.sleep(0.05)
            
        self.assertFalse(follower.is_active)
        # We expect exactly one event: the stream_warning
        self.assertEqual(len(self.received), 1, "Should receive exactly one event (stream_warning)")
        self.assertEqual(self.received[0]["type"], "stream_warning")
        self.assertEqual(self.received[0]["dropped_lines_count"], 1)
        
        self.assertEqual(len(self.errors), 0, "Invalid remainder during drain MUST be silent (no parse_error)")

    def test_normal_poll_error_still_emits(self):
        """Regression check: invalid FULL line during normal polling NO LONGER emits parse_error under new contract."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        follower.start()

        with open(self.events_path, "wb") as f:
            f.write(b'invalid_json_with_newline\n')
            
        for _ in range(5):
            self.app.processEvents()
            time.sleep(0.02)
            
        self.assertEqual(len(self.errors), 0, "Normal poll error should NOT emit signal under new contract")
        self.assertEqual(follower._dropped_lines_count, 1)

if __name__ == "__main__":
    unittest.main()
