
import unittest
import tempfile
import shutil
import os
import json
from gui.process_runner import _read_final_status

class TestGuiRunnerStatusJson(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_status(self, content):
        path = os.path.join(self.test_dir, "status.json")
        with open(path, "w") as f:
            if isinstance(content, str):
                f.write(content)
            else:
                json.dump(content, f)
        return path

    def test_missing_file_returns_none(self):
        # File not created
        path = os.path.join(self.test_dir, "status.json")
        st, errors = _read_final_status(path)
        self.assertEqual(st, "MISSING_FILE")
        self.assertEqual(errors, [])

    def test_malformed_json_returns_malformed(self):
        path = self._write_status("{") # Invalid JSON
        st, errors = _read_final_status(path)
        self.assertEqual(st, "MALFORMED_STATUS")
        self.assertEqual(errors, [])

    def test_final_missing_status_returns_missing(self):
        path = self._write_status({"schema_version": 1, "phase": "final"}) # Missing status field
        st, errors = _read_final_status(path)
        self.assertEqual(st, "MISSING_STATUS")
        self.assertEqual(errors, [])

    def test_running_returns_not_final(self):
        path = self._write_status({"schema_version": 1, "phase": "running"})
        st, errors = _read_final_status(path)
        self.assertEqual(st, "NOT_FINAL")
        self.assertEqual(errors, [])

    def test_final_error_valid(self):
        path = self._write_status({
            "schema_version": 1,
            "phase": "final",
            "status": "error",
            "errors": ["x"]
        })
        st, errors = _read_final_status(path)
        self.assertEqual(st, "error")
        self.assertEqual(errors, ["x"])

    def test_final_success_valid(self):
        path = self._write_status({
            "schema_version": 1,
            "phase": "final",
            "status": "success",
            "errors": []
        })
        st, errors = _read_final_status(path)
        self.assertEqual(st, "success")
        self.assertEqual(errors, [])

    def test_final_cancelled_valid(self):
        path = self._write_status({
            "schema_version": 1,
            "phase": "final",
            "status": "cancelled",
            "errors": ["CANCELLED"]
        })
        st, errors = _read_final_status(path)
        self.assertEqual(st, "cancelled")
        self.assertEqual(errors, ["CANCELLED"])

    def test_bad_status_value(self):
        path = self._write_status({
            "schema_version": 1,
            "phase": "final",
            "status": "weird_state",
            "errors": []
        })
        st, errors = _read_final_status(path)
        self.assertEqual(st, "BAD_STATUS")
        self.assertEqual(errors, [])
