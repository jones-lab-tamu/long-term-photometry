import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.run_full_pipeline_deliverables import main


class TestRunnerModeSmoke(unittest.TestCase):
    """
    Prove that runner mode gating works without running the full pipeline.
    Test by mocking run_cmd and allowing it to crash on 'No traces found'
    after the subprocess command list has been constructed for the required modes.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.tmp_dir, "in")
        self.out_dir = os.path.join(self.tmp_dir, "out")
        os.makedirs(self.input_dir)
        self.config_path = os.path.join(self.tmp_dir, "cfg.yaml")
        # Basic config to pass validation
        with open(self.config_path, "w") as f:
            f.write("target_fs_hz: 40.0\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch('tools.run_full_pipeline_deliverables.run_cmd')
    def test_mode_tonic(self, mock_run_cmd):
        argv = ["cmd", "--input", self.input_dir, "--out", self.out_dir, "--config", self.config_path, "--format", "auto", "--mode", "tonic"]
        with patch.object(sys, 'argv', argv):
            with self.assertRaises((SystemExit, RuntimeError)):
                main()

        cmds = [call[0][0] for call in mock_run_cmd.call_args_list]
        self.assertEqual(len(cmds), 1)
        self.assertIn("--mode", cmds[0])
        self.assertEqual(cmds[0][cmds[0].index("--mode") + 1], "tonic")

    @patch('tools.run_full_pipeline_deliverables.run_cmd')
    def test_mode_phasic(self, mock_run_cmd):
        argv = ["cmd", "--input", self.input_dir, "--out", self.out_dir, "--config", self.config_path, "--format", "auto", "--mode", "phasic"]
        with patch.object(sys, 'argv', argv):
            with self.assertRaises((SystemExit, RuntimeError)):
                main()

        cmds = [call[0][0] for call in mock_run_cmd.call_args_list]
        self.assertEqual(len(cmds), 1)
        self.assertIn("--mode", cmds[0])
        self.assertEqual(cmds[0][cmds[0].index("--mode") + 1], "phasic")

    @patch('tools.run_full_pipeline_deliverables.run_cmd')
    def test_mode_both(self, mock_run_cmd):
        argv = ["cmd", "--input", self.input_dir, "--out", self.out_dir, "--config", self.config_path, "--format", "auto", "--mode", "both"]
        with patch.object(sys, 'argv', argv):
            with self.assertRaises((SystemExit, RuntimeError)):
                main()

        cmds = [call[0][0] for call in mock_run_cmd.call_args_list]
        self.assertEqual(len(cmds), 2)
        # Verify modes for each invocation
        modes = []
        for cmd in cmds:
            if "--mode" in cmd:
                modes.append(cmd[cmd.index("--mode") + 1])
            else:
                modes.append(None)
        
        self.assertIn("tonic", modes)
        self.assertIn("phasic", modes)

if __name__ == '__main__':
    unittest.main()
