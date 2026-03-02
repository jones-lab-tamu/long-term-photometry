import unittest
import os
import json
import tempfile
import shutil
from gui.process_runner import PipelineRunner, RunnerState
from PySide6.QtWidgets import QApplication

class MockProcess:
    def readAllStandardOutput(self):
        class D:
            def data(self): return b""
        return D()
    def readAllStandardError(self):
        class D:
            def data(self): return b""
        return D()

class TestPipelineRunnerFinalStateContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance()
        if cls._app is None:
            cls._app = QApplication([])

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.runner = PipelineRunner()
        self.runner.set_run_dir(self.test_dir)
        self.runner._process = MockProcess()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_status(self, phase="final", status="success", schema=1):
        data = {"phase": phase, "status": status, "schema_version": schema}
        with open(os.path.join(self.test_dir, "status.json"), "w") as f:
            json.dump(data, f)

    def test_cancelled_beats_exit_zero(self):
        """T1: status.json 'cancelled' + exit_code 0 -> runner.state == CANCELLED"""
        self._write_status(status="cancelled")
        self.runner._on_finished(0, 0)
        self.assertEqual(self.runner.state, RunnerState.CANCELLED)
        self.assertEqual(self.runner.final_status_code, "cancelled")

    def test_missing_status_fail_closed(self):
        """T2: status.json missing + exit_code 0 -> runner.state == FAIL_CLOSED (MISSING_FILE)"""
        # No status.json written
        self.runner._on_finished(0, 0)
        self.assertEqual(self.runner.state, RunnerState.FAIL_CLOSED)
        self.assertEqual(self.runner.fail_closed_code, "MISSING_FILE")

    def test_nonfinal_with_exit_fail_closed(self):
        """T3: status.json phase != final + exit_code nonzero -> runner.state == FAIL_CLOSED (NONFINAL_WITH_EXIT)"""
        self._write_status(phase="running")
        self.runner._on_finished(1, 0)
        self.assertEqual(self.runner.state, RunnerState.FAIL_CLOSED)
        self.assertEqual(self.runner.fail_closed_code, "NONFINAL_WITH_EXIT")

    def test_success_beats_exit_nonzero(self):
        """Proof: status.json 'success' + exit_code 1 -> runner.state == SUCCESS (contract beats exit code)"""
        self._write_status(status="success")
        self.runner._on_finished(1, 0)
        self.assertEqual(self.runner.state, RunnerState.SUCCESS)

if __name__ == "__main__":
    unittest.main()
