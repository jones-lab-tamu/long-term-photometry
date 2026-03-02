
import sys
import os
import json
import time
import unittest
import tempfile
import shutil
from PySide6.QtWidgets import QApplication
from gui.events_follower import EventsFollower

class TestEventsFollowerSchema(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        self.test_dir = tempfile.mkdtemp()
        self.events_path = os.path.join(self.test_dir, "test_events.ndjson")
        self.received = []
        self.warnings = []
        self.parse_errors = []

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _connect(self, follower):
        follower.event_received.connect(lambda obj: self.received.append(obj))
        follower.warning.connect(lambda msg: self.warnings.append(msg))
        follower.parse_error.connect(lambda msg: self.parse_errors.append(msg))

    def pump(self, ms_total=250, step_ms=10):
        """Helper to run the event loop briefly while waiting for polling."""
        deadline = time.time() + ms_total / 1000.0
        while time.time() < deadline:
            self.app.processEvents()
            time.sleep(step_ms / 1000.0)

    def test_case_a_mixed_stream_warn_once(self):
        """Test 1: Mixed stream (non-dict + bad schema + good schema + trailing bad) warns once, accepts valid."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        
        with open(self.events_path, "wb") as f:
            # a) Non-dict JSON
            f.write(b'"not a dict string"\n')
            # b) Dict with bad schema
            f.write(json.dumps({"schema_version": 999, "stage": "bad"}).encode() + b"\n")
            # c) Valid dict
            f.write(json.dumps({"schema_version": 1, "stage": "ok"}).encode() + b"\n")
            # d) Trailing violation (non-dict)
            f.write(b'["another list"]\n')

        follower.start()
        self.pump()
        follower.stop()

        self.assertEqual(len(self.received), 1, "Should receive exactly one valid event")
        self.assertTrue(isinstance(self.received[0], dict), "Received event must be a dict")
        self.assertEqual(self.received[0].get("schema_version"), 1, "Schema version must be 1")
        self.assertEqual(self.received[0].get("stage"), "ok")
        
        self.assertEqual(len(self.warnings), 1, "Should warn exactly once total (warn-once policy)")
        self.assertTrue(self.warnings[0].startswith("WARN: "), "Warning must start with WARN: prefix")
        self.assertEqual(len(self.parse_errors), 0, "Advisory schema/type issues must NOT trigger parse_error")

    def test_case_b_malformed_json_produces_parse_error_only(self):
        """Test 2: Malformed JSON line must produce parse_error only (json.loads failure)."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        
        with open(self.events_path, "wb") as f:
            f.write(b"{malformed json\n")
            f.write(json.dumps({"schema_version": 1, "event": "valid"}).encode() + b"\n")

        follower.start()
        self.pump()
        follower.stop()
        self.pump()

        self.assertEqual(len(self.received), 2, "Should receive valid event AND stream_warning at stop")
        
        # Last event must be the stream_warning
        warning = self.received[-1]
        self.assertEqual(warning.get("type"), "stream_warning")
        self.assertIn("dropped 1", warning.get("message"))
        
        self.assertEqual(len(self.warnings), 0, "Malformed JSON is not an advisory schema warning")
        self.assertEqual(len(self.parse_errors), 0, "Truly malformed JSON no longer triggers parse_error signal")

    def test_case_c_warn_once_resets_per_start(self):
        """Test 3: Warn-once resets per start() on same instance with new content."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)

        # Run 1: Write an advisory violation and a valid event
        with open(self.events_path, "wb") as f:
            f.write(json.dumps({"schema_version": 999}).encode() + b"\n")
            f.write(json.dumps({"schema_version": 1, "run": 1}).encode() + b"\n")
        
        follower.start()
        self.pump()
        follower.stop()
        
        self.assertEqual(len(self.warnings), 1)
        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0].get("run"), 1)
        self.assertEqual(len(self.parse_errors), 0)

        # Run 2: start() resets flag and offset. We truncate and rewrite with NEW content.
        with open(self.events_path, "wb") as f:
            f.write(b'"new string violation"\n')
            f.write(json.dumps({"schema_version": 1, "run": 2}).encode() + b"\n")
            
        follower.start()
        self.pump()
        follower.stop()
        
        self.assertEqual(len(self.warnings), 2, "Warn-once flag must reset on every start()")
        self.assertEqual(len(self.received), 2, "Should have processed the new valid event")
        self.assertEqual(self.received[1].get("run"), 2)
        self.assertEqual(len(self.parse_errors), 0)

    def test_schema_types_and_missing_field_warn(self):
        """Verify varied advisory mismatches route to warning signal only, with zero other emissions."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        
        with open(self.events_path, "wb") as f:
            # 1. Missing version
            f.write(json.dumps({"event": "no_version"}).encode() + b"\n")
            # 2. String instead of int
            f.write(json.dumps({"schema_version": "1"}).encode() + b"\n")
            # 3. List
            f.write(b"[null, 1, 2]\n")
            
        follower.start()
        self.pump()
        follower.stop()
        
        self.assertEqual(len(self.received), 0, "No events should be emitted for bad schemas")
        self.assertEqual(len(self.warnings), 1, "Warn-once across diverse types/mismatches in one run")
        self.assertEqual(len(self.parse_errors), 0, "No parse errors for advisory failures")

    def test_valid_event_received_only(self):
        """Purely valid events must not produce warnings or parse errors, with deep check."""
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        
        with open(self.events_path, "wb") as f:
            f.write(json.dumps({"schema_version": 1, "status": "perfect"}).encode() + b"\n")
            
        follower.start()
        self.pump()
        follower.stop()
        
        self.assertEqual(len(self.received), 1)
        self.assertTrue(isinstance(self.received[0], dict))
        self.assertEqual(self.received[0].get("schema_version"), 1)
        self.assertEqual(self.received[0].get("status"), "perfect")
        self.assertEqual(len(self.warnings), 0, "Valid events must not trigger warnings")
        self.assertEqual(len(self.parse_errors), 0, "Valid events must not trigger parse errors")

    def test_case_d_append_after_start_tailing_warn_once(self):
        """Test 4: Append-after-start (tailing) with warn-once enforced."""
        # Ensure file exists
        open(self.events_path, "wb").close()
        
        follower = EventsFollower(self.events_path, poll_ms=50)
        self._connect(follower)
        follower.start()
        try:
            # 1. Pump empty
            self.pump(ms_total=100)
            
            # 2. Phase 1: Append an advisory violation
            with open(self.events_path, "ab") as f:
                f.write(b'"late string"\n')
            self.pump(ms_total=250)
            
            self.assertEqual(len(self.warnings), 1)
            self.assertEqual(len(self.parse_errors), 0)
            self.assertEqual(len(self.received), 0)
            
            # 3. Phase 2: Append another advisory violation (warn-once)
            with open(self.events_path, "ab") as f:
                f.write(json.dumps({"schema_version": 999, "late": 2}).encode() + b"\n")
            self.pump(ms_total=250)
            
            self.assertEqual(len(self.warnings), 1, "Warn-once must hold for tailed content")
            self.assertEqual(len(self.parse_errors), 0)
            
            # 4. Phase 3: Append valid event
            with open(self.events_path, "ab") as f:
                f.write(json.dumps({"schema_version": 1, "late": "ok"}).encode() + b"\n")
            self.pump(ms_total=250)
            
            self.assertEqual(len(self.received), 1)
            self.assertEqual(self.received[0]["schema_version"], 1)
            self.assertEqual(self.received[0]["late"], "ok")
        finally:
            follower.stop()

if __name__ == "__main__":
    unittest.main()
