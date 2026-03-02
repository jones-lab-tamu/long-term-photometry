
import unittest
import os
import sys
import shutil
import subprocess
import glob
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestFullPipelineDeliverables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        # Strict hygiene: Use unique temp directory
        cls.out_dir = tempfile.mkdtemp(prefix="out_deliverables_test_")
        cls.input_dir = os.path.join(cls.out_dir, "input_RWD")
        import uuid
        cls.output_package = os.path.join(cls.out_dir, f"package_{uuid.uuid4().hex[:8]}")
        
        # Original config
        orig_config_path = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
        
        # Create temp config (copy of original)
        cls.config_path = os.path.join(cls.out_dir, "qc_universal_config.yaml")
        
        # No need to cleanup hardcoded dir, we have a fresh one.
        if not os.path.exists(cls.out_dir):
            os.makedirs(cls.out_dir)

        shutil.copy2(orig_config_path, cls.config_path)

        # 1. Synthetic Data (2.0 days, 2 rec/hr = 96 chunks)
        print("Generating synthetic data (2.0 days)...")
        gen_cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", cls.input_dir,
            "--format", "rwd",
            "--config", cls.config_path,
            "--total-days", "2.0",
            "--recordings-per-hour", "2", 
            "--recording-duration-min", "10.0",
            "--n-rois", "2",
            "--phasic-mode", "phase_locked_to_tonic",
            "--seed", "42",
            "--preset", "biological_shared_nuisance"
        ]
        subprocess.check_call(gen_cmd)

    def test_pipeline_deliverables(self):
        # Run Tool with EXPLICIT sessions-per-hour (Required for duty cycle)
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", self.output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2"
        ]
        print(f"Running pipeline command: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Manifest
        man_path = os.path.join(self.output_package, "MANIFEST.json")
        self.assertTrue(os.path.exists(man_path))
        with open(man_path, 'r') as f:
            manifest = json.load(f)
            
        self.assertEqual(manifest['sessions_per_hour'], 2)
        
        # Check duration AND stride are recorded
        self.assertIn('session_duration_s', manifest)
        self.assertIn('session_stride_s', manifest)
        
        dur = manifest['session_duration_s']
        stride = manifest['session_stride_s']
        
        # Real/Synth Expectation:
        # Duration ~ 600s
        # Stride ~ 1800s
        self.assertTrue(590 < dur < 610, f"Duration {dur} not ~600")
        self.assertTrue(1790 < stride < 1810, f"Stride {stride} not ~1800")
        
        # Region0 Checks
        reg0 = os.path.join(self.output_package, "Region0")
        self.assertTrue(os.path.exists(reg0))
        
        # Multi-day plots
        for day in ["000", "001"]:
            self.assertTrue(os.path.exists(os.path.join(reg0, f"phasic_sig_iso_day_{day}.png")))
            self.assertTrue(os.path.exists(os.path.join(reg0, f"phasic_dFF_day_{day}.png")))
            self.assertTrue(os.path.exists(os.path.join(reg0, f"phasic_stacked_day_{day}.png")))
            
        # 3-Panel Correction Impact
        self.assertTrue(os.path.exists(os.path.join(reg0, "phasic_correction_impact.png")))
        self.assertTrue(os.path.exists(os.path.join(reg0, "phasic_correction_impact_session.csv")))
        
        # Time Series CSV Row Counts
        ts_csv = os.path.join(reg0, "phasic_peak_rate_timeseries.csv")
        self.assertTrue(os.path.exists(ts_csv))
        df_ts = pd.read_csv(ts_csv)
        # Expected: 96 rows 
        self.assertEqual(len(df_ts), 96)
        
        # Check columns
        req_cols = ['time_hours', 'day', 'hour', 'session_in_hour', 'window_seconds']
        for c in req_cols:
            self.assertIn(c, df_ts.columns)
            
        # Check window_seconds matches duration (approx 600s)
        # NOT stride (1800s)
        first_dur = df_ts.iloc[0]['window_seconds']
        self.assertTrue(590 < first_dur < 610, f"Window seconds {first_dur} should be ~600")

        # Check Peak Rate Columns
        self.assertIn('peak_rate_per_min', df_ts.columns)
        self.assertIn('peak_count', df_ts.columns)
        self.assertNotIn('n_peaks', df_ts.columns)
        
        # Check AUC CSV
        auc_csv = os.path.join(reg0, "phasic_auc_timeseries.csv")
        self.assertTrue(os.path.exists(auc_csv))
        df_auc = pd.read_csv(auc_csv)
        self.assertIn('auc_above_threshold_dff_s', df_auc.columns)
        for c in req_cols:
            self.assertIn(c, df_auc.columns)
            
        # Manifest Consistency
        deliv = manifest['deliverables']['Region0']
        self.assertIn('days_dff', deliv)
        self.assertIn('days_sig_iso', deliv)
        self.assertIn('days_stacked', deliv)
        self.assertIn('days_generated', deliv)
        self.assertEqual(deliv['days_dff'], deliv['days_generated'])
        self.assertEqual(deliv['days_sig_iso'], deliv['days_generated'])
        self.assertEqual(deliv['days_stacked'], deliv['days_generated'])

    def test_impossible_schedule(self):
        """Ensure failure when session_duration_s > stride_s."""
        # Trace duration is ~600s.
        # We need a valid duration (matches trace) that is > stride.
        # Set sessions_per_hour=10 -> Stride = 360s.
        # Duration 600s > Stride 360s -> Impossible.
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", os.path.join(self.out_dir, "test_impossible"),
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "10",
            "--session-duration-s", "600"
        ]
        print(f"Running impossible schedule test: {' '.join(cmd)}")
        
        # Expect failure
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Command should have failed")
        
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        print(f"Captured Output:\n{combined}")
        
        self.assertIn("Impossible schedule", combined)
        self.assertIn("Duration", combined)
        self.assertIn("Stride", combined)
            
        print("Successfully caught impossible schedule error with correct message.")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.out_dir):
            try:
                shutil.rmtree(cls.out_dir, ignore_errors=True)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
