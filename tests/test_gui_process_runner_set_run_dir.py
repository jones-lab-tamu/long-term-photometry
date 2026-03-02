import unittest
import os
from gui.process_runner import PipelineRunner
from PySide6.QtWidgets import QApplication

class TestPipelineRunnerSetRunDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance()
        if cls._app is None:
            cls._app = QApplication([])

    def test_exactly_one_set_run_dir_definition(self):
        """Assert that exactly one 'def set_run_dir' exists in the source file."""
        import gui.process_runner as pr
        source_path = pr.__file__
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        count = content.count("def set_run_dir")
        self.assertEqual(count, 1, f"Expected exactly 1 'def set_run_dir', found {count}")

    def test_set_run_dir_canonical_reset(self):
        """set_run_dir must update run_dir and clear ALL parsing buffers/state."""
        runner = PipelineRunner()
        
        # Inject state into internal fields
        runner._run_dir = "prior_dir"
        runner._stdout_buf = b"partial stdout"
        runner._stderr_buf = b"partial stderr"
        runner.final_status_code = "success"
        runner.final_errors = ["error1"]
        runner.fail_closed_code = "BUG"
        runner.fail_closed_detail = "detail"
        runner.fail_closed_remediation = "fix"
        runner.validation_summary = {"ok": False}
        
        # Call set_run_dir
        new_dir = "run_1"
        runner.set_run_dir(new_dir)
        
        self.assertEqual(runner._run_dir, new_dir)
        self.assertEqual(runner._stdout_buf, b"")
        self.assertEqual(runner._stderr_buf, b"")
        self.assertIsNone(runner.final_status_code)
        self.assertEqual(runner.final_errors, [])
        self.assertIsNone(runner.fail_closed_code)
        self.assertIsNone(runner.fail_closed_detail)
        self.assertIsNone(runner.fail_closed_remediation)
        self.assertIsNone(runner.validation_summary)

if __name__ == "__main__":
    unittest.main()
