import unittest
from gui.process_runner import PipelineRunner
from PySide6.QtWidgets import QApplication

class MockProcess:
    def __init__(self, stdout_data=b"", stderr_data=b""):
        self._stdout_data = stdout_data
        self._stderr_data = stderr_data

    def readAllStandardOutput(self):
        d = self._stdout_data
        self._stdout_data = b"" # Drain on read
        return self._Data(d)

    def readAllStandardError(self):
        d = self._stderr_data
        self._stderr_data = b"" # Drain on read
        return self._Data(d)

    class _Data:
        def __init__(self, d): self._d = d
        def data(self): return self._d

class TestPipelineRunnerEofDrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance()
        if cls._app is None:
            cls._app = QApplication([])

    def test_eof_drain_finish_only_fragments(self):
        """Test 1: Process returns fragments only at finish."""
        class RecordingRunner(PipelineRunner):
            def __init__(self):
                super().__init__()
                self.out_lines = []
                self.err_lines = []
            def _handle_stdout_line(self, line):
                self.out_lines.append(line)
            def _handle_stderr_line(self, line):
                self.err_lines.append(line)

        runner = RecordingRunner()
        runner._process = MockProcess(stdout_data=b"FINAL_OUT", stderr_data=b"FINAL_ERR")
        
        runner._on_finished(0, 0)
        
        self.assertEqual(runner.out_lines, ["FINAL_OUT"])
        self.assertEqual(runner.err_lines, ["FINAL_ERR"])
        self.assertEqual(runner._stdout_buf, b"")
        self.assertEqual(runner._stderr_buf, b"")

    def test_eof_drain_mixed_readyread_and_finish(self):
        """Test 2: Buffers already have data, finish returns tail."""
        class RecordingRunner(PipelineRunner):
            def __init__(self):
                super().__init__()
                self.out_lines = []
                self.err_lines = []
            def _handle_stdout_line(self, line):
                self.out_lines.append(line)
            def _handle_stderr_line(self, line):
                self.err_lines.append(line)

        runner = RecordingRunner()
        # Initial partial lines in buffers
        runner._stdout_buf = b"PARTIAL_OUT"
        runner._stderr_buf = b"PARTIAL_ERR"
        
        # Finish returns tails
        runner._process = MockProcess(stdout_data=b"_TAIL", stderr_data=b"_TAIL")
        
        runner._on_finished(0, 0)
        
        self.assertEqual(runner.out_lines, ["PARTIAL_OUT_TAIL"])
        self.assertEqual(runner.err_lines, ["PARTIAL_ERR_TAIL"])
        self.assertEqual(runner._stdout_buf, b"")
        self.assertEqual(runner._stderr_buf, b"")

if __name__ == "__main__":
    unittest.main()
