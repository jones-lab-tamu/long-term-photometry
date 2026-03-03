import unittest
import os
import shutil
import tempfile
import subprocess
import sys
import json
import yaml
import pandas as pd
import numpy as np

class TestRepresentativeSessionIndex(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.in_dir = os.path.join(self.tmp_dir, "in")
        os.makedirs(self.in_dir)
        
        # Create 2 valid sessions
        for i in range(2):
            sess_dir = os.path.join(self.in_dir, f"chunk_{i:04d}")
            os.makedirs(sess_dir)
            fs = 100.0
            n_samples = 1500
            t = np.arange(n_samples) / fs
            df = pd.DataFrame({
                'Timestamp': t,
                'Region0G': np.sin(t) + 1.0 + np.linspace(0, 0.1, n_samples),
                'Region0R': np.cos(t) + 1.0 + np.linspace(0, 0.1, n_samples)
            })
            df.to_csv(os.path.join(sess_dir, "fluorescence.csv"), index=False)
            
        self.cfg_path = os.path.join(self.tmp_dir, "config.yaml")
        with open(self.cfg_path, 'w') as f:
            yaml.dump({
                'rwd_time_col': 'Timestamp',
                'target_fs_hz': 100.0,
                'chunk_duration_sec': 15.0,
                'uv_suffix': 'G',
                'sig_suffix': 'R',
                'window_sec': 2.0,
                'step_sec': 1.0,
                'min_valid_windows': 2,
                'lowpass_hz': 5.0
            }, f)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_out_of_range_fails_closed(self):
        # N=2 sessions, so index 2 is out of range
        out_dir = os.path.join(self.tmp_dir, "out_fail")
        cmd = [sys.executable, "analyze_photometry.py",
               "--input", self.in_dir, "--out", out_dir,
               "--config", self.cfg_path, "--format", "rwd",
               "--representative-session-index", "2"]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("representative_session_index out of range", res.stdout + res.stderr)

    def test_default_selection_first_loadable(self):
        # Create a malformed first session
        malformed_dir = os.path.join(self.in_dir, "chunk_0000_bad")
        # Rename existing chunk_0000 to move it out of the way
        shutil.move(os.path.join(self.in_dir, "chunk_0000"), os.path.join(self.tmp_dir, "temp_chunk"))
        os.makedirs(malformed_dir)
        with open(os.path.join(malformed_dir, "fluorescence.csv"), "w") as f:
            f.write("corrupt,data\n1,2,3\n")
            
        out_dir = os.path.join(self.tmp_dir, "out_default")
        cmd = [sys.executable, "analyze_photometry.py",
               "--input", self.in_dir, "--out", out_dir,
               "--config", self.cfg_path, "--format", "rwd"] # No index provided
        
        subprocess.run(cmd, check=True)
        
        with open(os.path.join(out_dir, "run_report.json"), "r") as f:
            report = json.load(f)
        
        # Should have picked chunk_0001 as the first loadable
        self.assertEqual(report["run_context"]["representative_session_index"], 1)
        self.assertEqual(report["run_context"]["representative_session_id"], "chunk_0001")
        self.assertEqual(report["run_context"]["user_provided_representative_session_index"], False)

    def test_index_changes_id(self):
        # Run with index 0
        out0 = os.path.join(self.tmp_dir, "out0")
        cmd0 = [sys.executable, "analyze_photometry.py",
                "--input", self.in_dir, "--out", out0,
                "--config", self.cfg_path, "--format", "rwd",
                "--representative-session-index", "0", "--overwrite"]
        subprocess.run(cmd0, check=True)
        
        with open(os.path.join(out0, "run_report.json"), "r") as f:
            report0 = json.load(f)
        id0 = report0["run_context"]["representative_session_id"]
        
        # Run with index 1
        out1 = os.path.join(self.tmp_dir, "out1")
        cmd1 = [sys.executable, "analyze_photometry.py",
                "--input", self.in_dir, "--out", out1,
                "--config", self.cfg_path, "--format", "rwd",
                "--representative-session-index", "1", "--overwrite"]
        subprocess.run(cmd1, check=True)
        
        with open(os.path.join(out1, "run_report.json"), "r") as f:
            report1 = json.load(f)
        id1 = report1["run_context"]["representative_session_id"]
        
        self.assertNotEqual(id0, id1)
        self.assertEqual(report1["run_context"]["user_provided_representative_session_index"], True)

    def test_audit_events_schema(self):
        out_dir = os.path.join(self.tmp_dir, "out_audit")
        events_path = os.path.join(self.tmp_dir, "events.ndjson")
        cmd = [sys.executable, "analyze_photometry.py",
               "--input", self.in_dir, "--out", out_dir,
               "--config", self.cfg_path, "--format", "rwd",
               "--representative-session-index", "0",
               "--events-path", events_path]
        
        subprocess.run(cmd, check=True)
        
        found_event = False
        with open(events_path, "r") as f:
            for line in f:
                event = json.loads(line)
                if event.get("stage") == "inputs" and event.get("type") == "representative_session":
                    found_event = True
                    payload = event["payload"]
                    self.assertEqual(payload["representative_session_index"], 0)
                    self.assertEqual(payload["representative_session_id"], "chunk_0000")
                    self.assertEqual(payload["user_provided"], True)
                    self.assertIn("resolved_session_ids_preview", payload)
                    self.assertLessEqual(len(payload["resolved_session_ids_preview"]), 5)
        self.assertTrue(found_event)

if __name__ == "__main__":
    unittest.main()
