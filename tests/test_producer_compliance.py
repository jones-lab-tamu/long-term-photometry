import sys
import os
import json
import unittest
import tempfile
import shutil
from PySide6.QtWidgets import QApplication
from tools.run_full_pipeline_deliverables import EventEmitter
from gui.events_follower import EventsFollower

class TestProducerCompliance(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        self.test_dir = tempfile.mkdtemp()
        self.events_path = os.path.join(self.test_dir, "events.ndjson")
        self.received = []
        self.warnings = []

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def pump(self, ms_total=250, step_ms=10):
        deadline = __import__("time").time() + ms_total / 1000.0
        while __import__("time").time() < deadline:
            self.app.processEvents()
            __import__("time").sleep(step_ms / 1000.0)

    def test_event_emitter_includes_schema_version_int(self):
        """Verify EventEmitter adds/forces schema_version=1 (int) to all dict events."""
        emitter = EventEmitter(self.events_path, "test_run", self.test_dir)
        # Attempt violations via kwargs
        emitter.emit("stage1", "type1", "msg1", extra="data")
        emitter.emit("stage2", "type2", "msg2", schema_version="wrong_type") 
        emitter.emit("stage3", "type3", "msg3", schema_version=999)
        emitter.close()

        with open(self.events_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.assertEqual(len(lines), 3)
        for line in lines:
            obj = json.loads(line)
            self.assertIn("schema_version", obj)
            self.assertEqual(obj["schema_version"], 1, f"Failed for line: {line}")
            self.assertIsInstance(obj["schema_version"], int)
            # Verify separators=(",", ":") - no spaces after commas/colons in production code
            self.assertNotIn(": ", line)
            self.assertNotIn(", ", line)

    def test_compliance_results_in_zero_warnings(self):
        """Prove a compliant producer run yields zero EventsFollower warnings."""
        emitter = EventEmitter(self.events_path, "test_run", self.test_dir)
        emitter.emit("engine", "start", "hello")
        emitter.emit("engine", "done", "goodbye")
        emitter.close()

        follower = EventsFollower(self.events_path, poll_ms=50)
        follower.event_received.connect(lambda e: self.received.append(e))
        follower.warning.connect(lambda m: self.warnings.append(m))
        
        follower.start()
        self.pump(ms_total=250)
        follower.stop()

        self.assertEqual(len(self.received), 2)
        self.assertEqual(len(self.warnings), 0, "Normal compliant runs must NOT emit schema warnings")

if __name__ == "__main__":
    unittest.main()
