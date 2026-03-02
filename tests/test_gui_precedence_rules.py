
import unittest
import os
import json
import shutil
import tempfile
from gui.process_runner import PipelineRunner, RunnerState, _read_final_status

class TestPrecedenceRules(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.runner = PipelineRunner()
        self.runner.set_run_dir(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_status(self, phase, status, schema_version=1):
        status_path = os.path.join(self.test_dir, "status.json")
        data = {"phase": phase, "status": status}
        if schema_version is not None:
            data["schema_version"] = schema_version
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_precedence_case_g_missing_schema_fail_closed(self):
        # Case G: status.json exists but schema_version missing -> FAIL_CLOSED
        self._write_status("final", "success", schema_version=None)
        state = self.runner._resolve_final_state(None, exit_code=0)
        # Wait, the test calls it with code directly?
        # _resolve_final_state expects (code, exit_code) where code is from _read_final_status
        code, _ = _read_final_status(os.path.join(self.test_dir, "status.json"))
        state = self.runner._resolve_final_state(code, exit_code=0)
        self.assertEqual(state, RunnerState.FAIL_CLOSED)

    def test_precedence_case_h_bad_schema_fail_closed(self):
        # Case H: status.json exists but schema_version != 1 -> FAIL_CLOSED
        self._write_status("final", "success", schema_version=999)
        code, _ = _read_final_status(os.path.join(self.test_dir, "status.json"))
        state = self.runner._resolve_final_state(code, exit_code=0)
        self.assertEqual(state, RunnerState.FAIL_CLOSED)

    def test_precedence_case_i_string_schema_fail_closed(self):
        # Case I: status.json exists but schema_version is a string "1" -> FAIL_CLOSED
        self._write_status("final", "success", schema_version="1")
        code, _ = _read_final_status(os.path.join(self.test_dir, "status.json"))
        state = self.runner._resolve_final_state(code, exit_code=0)
        self.assertEqual(state, RunnerState.FAIL_CLOSED)

    def test_precedence_case_a_status_json_success_over_exit_1(self):
        # Case A: exit_code != 0, status.json says "success" -> SUCCESS
        self._write_status("final", "success")
        code, _ = _read_final_status(os.path.join(self.test_dir, "status.json"))
        state = self.runner._resolve_final_state(code, exit_code=1)
        self.assertEqual(state, RunnerState.SUCCESS)

    def test_precedence_case_b_status_json_error_over_exit_0(self):
        # Case B: exit_code == 0, status.json says "error" -> FAILED
        self._write_status("final", "error")
        code, _ = _read_final_status(os.path.join(self.test_dir, "status.json"))
        state = self.runner._resolve_final_state(code, exit_code=0)
        self.assertEqual(state, RunnerState.FAILED)

    def test_precedence_case_e_malformed_json_fail_closed(self):
        # Case E: status.json exists but is malformed -> FAIL_CLOSED (even if exit 0)
        status_path = os.path.join(self.test_dir, "status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            f.write("invalid json {")
        code, _ = _read_final_status(status_path)
        state = self.runner._resolve_final_state(code, exit_code=0)
        self.assertEqual(state, RunnerState.FAIL_CLOSED)

    def test_precedence_case_f_non_final_fail_closed(self):
        # Case F: status.json exists but phase is not "final" -> FAIL_CLOSED (even if exit 0)
        self._write_status("running", "success")
        code, _ = _read_final_status(os.path.join(self.test_dir, "status.json"))
        state = self.runner._resolve_final_state(code, exit_code=0)
        self.assertEqual(state, RunnerState.FAIL_CLOSED)

    def test_precedence_case_c_cancel_fallback(self):
        # Case C: status.json missing, cancel_requested true, exit_code 1 -> CANCELLED
        self.runner._cancel_requested = True
        state = self.runner._resolve_final_state(None, exit_code=1)
        self.assertEqual(state, RunnerState.CANCELLED)

    def test_precedence_case_d_exit_0_fallback(self):
        # Case D: status.json missing, exit_code 0 -> FAIL_CLOSED (Requirement 1.1)
        # Old behavior was fail-open (SUCCESS). New behavior is fail-closed.
        state = self.runner._resolve_final_state("MISSING_FILE", exit_code=0)
        self.assertEqual(state, RunnerState.FAIL_CLOSED)
        self.assertEqual(self.runner.fail_closed_code, "MISSING_FILE")

    def test_status_json_cancelled_over_exit_1(self):
        # Additional case: status.json says "cancelled", exit 1
        self._write_status("final", "cancelled")
        code, _ = _read_final_status(os.path.join(self.test_dir, "status.json"))
        state = self.runner._resolve_final_state(code, exit_code=1)
        self.assertEqual(state, RunnerState.CANCELLED)

if __name__ == "__main__":
    unittest.main()
