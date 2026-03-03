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


class TestPreviewModeFirstN(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.in_dir = os.path.join(self.tmp_dir, "in")
        os.makedirs(self.in_dir)

        # Create 3 valid sessions with distinct names
        for i in range(3):
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

    def test_preview_limits_processing_and_stamps_everywhere(self):
        out_dir = os.path.join(self.tmp_dir, "out_preview")
        events_path = os.path.join(self.tmp_dir, "events.ndjson")
        cmd = [sys.executable, "analyze_photometry.py",
               "--input", self.in_dir, "--out", out_dir,
               "--config", self.cfg_path, "--format", "rwd",
               "--preview-first-n", "1",
               "--events-path", events_path]

        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0,
                         f"Pipeline failed:\nstdout: {res.stdout[-500:]}\nstderr: {res.stderr[-500:]}")

        # ---- run_report.json ----
        with open(os.path.join(out_dir, "run_report.json"), "r") as f:
            report = json.load(f)

        ctx = report["run_context"]
        self.assertEqual(ctx["run_type"], "preview")
        self.assertIsNotNone(ctx["preview"])
        self.assertEqual(ctx["preview"]["selector"], "first_n")
        self.assertEqual(ctx["preview"]["first_n"], 1)
        self.assertEqual(ctx["preview"]["n_total_discovered"], 3)
        self.assertEqual(ctx["preview"]["n_sessions_resolved"], 1)
        self.assertEqual(ctx["representative_session_index"], 0)
        self.assertEqual(ctx["representative_session_id"], "chunk_0000")

        # ---- events.ndjson ----
        events = []
        with open(events_path, "r") as f:
            for line in f:
                events.append(json.loads(line))

        # engine:context
        ctx_events = [e for e in events
                      if e.get("stage") == "engine" and e.get("type") == "context"]
        self.assertTrue(len(ctx_events) >= 1, "Missing engine:context event")
        ctx_payload = ctx_events[0]["payload"]
        self.assertEqual(ctx_payload["run_type"], "preview")
        self.assertIsNotNone(ctx_payload["preview"])
        self.assertEqual(ctx_payload["preview"]["selector"], "first_n")
        self.assertEqual(ctx_payload["preview"]["first_n"], 1)

        # inputs:preview
        prev_events = [e for e in events
                       if e.get("stage") == "inputs" and e.get("type") == "preview"]
        self.assertTrue(len(prev_events) >= 1, "Missing inputs:preview event")
        prev_payload = prev_events[0]["payload"]
        self.assertEqual(prev_payload["n_total_discovered"], 3)
        self.assertEqual(prev_payload["n_sessions_resolved"], 1)
        self.assertEqual(prev_payload["selector"], "first_n")
        self.assertEqual(prev_payload["first_n"], 1)

        # inputs:representative_session
        rep_events = [e for e in events
                      if e.get("stage") == "inputs"
                      and e.get("type") == "representative_session"]
        self.assertTrue(len(rep_events) >= 1, "Missing inputs:representative_session event")
        rep_payload = rep_events[0]["payload"]
        self.assertEqual(rep_payload["representative_session_index"], 0)
        self.assertEqual(rep_payload["representative_session_id"], "chunk_0000")

        # ---- Hard-wall proof: excluded sessions must not appear anywhere ----
        excluded = ["chunk_0001", "chunk_0002"]
        for dirpath, dirnames, filenames in os.walk(out_dir):
            for name in dirnames + filenames:
                for ex in excluded:
                    self.assertNotIn(
                        ex, name,
                        f"Excluded session '{ex}' found in output: {os.path.join(dirpath, name)}"
                    )
            # Also check the directory path itself for excluded session names
            for ex in excluded:
                # Only check the relative part under out_dir
                rel = os.path.relpath(dirpath, out_dir)
                self.assertNotIn(
                    ex, rel,
                    f"Excluded session '{ex}' found in output path: {dirpath}"
                )

    def test_invalid_preview_first_n_rejected(self):
        cfg_bad = os.path.join(self.tmp_dir, "config_bad.yaml")
        with open(self.cfg_path, 'r') as f:
            base = yaml.safe_load(f)

        from photometry_pipeline.config import Config

        # 0 should be rejected
        base['preview_first_n'] = 0
        with open(cfg_bad, 'w') as f:
            yaml.dump(base, f)
        with self.assertRaises(ValueError) as cm:
            Config.from_yaml(cfg_bad)
        self.assertIn("preview_first_n", str(cm.exception))
        self.assertIn("> 0", str(cm.exception))

        # -1 should be rejected
        base['preview_first_n'] = -1
        with open(cfg_bad, 'w') as f:
            yaml.dump(base, f)
        with self.assertRaises(ValueError) as cm:
            Config.from_yaml(cfg_bad)
        self.assertIn("preview_first_n", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
