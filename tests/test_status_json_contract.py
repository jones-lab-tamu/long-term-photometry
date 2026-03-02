
import time
import unittest
import os
import shutil
import tempfile
import sys
import subprocess
import json
import logging

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestStatusJsonContract(unittest.TestCase):
    """
    Validation of status.json contract and deliverables.
    Requirements:
    - Deterministic dataset generation (guaranteed >0 chunks).
    - Status.json exists and adheres to schema (phase, status, errors).
    - Output package integrity (MANIFEST.json, Region dirs).
    - Failure path validation.
    """
    
    def setUp(self):
        self.temp_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "temp_status_test"))
        
        # Robust cleanup
        if os.path.exists(self.temp_dir):
            for i in range(5):
                try:
                    shutil.rmtree(self.temp_dir)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                 # If failed to delete, try to continue or fail?
                 # If we can't delete, we might have stale data.
                 # Try to reuse if existing? No, unsafe.
                 pass

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Give Windows a moment to verify existence
        time.sleep(1.0)
        self.input_dir = os.path.join(self.temp_dir, "input_RWD")
        self.out_base = os.path.join(self.temp_dir, "out_base")
        self.config_path = os.path.join(self.temp_dir, "tests_config.yaml")

        # Copy canonical config
        src_config = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
        if not os.path.exists(src_config):
            self.fail(f"Test config not found at {src_config}")
        shutil.copy2(src_config, self.config_path)

        # Append specific settings for deterministic chunks
        # total-days=0.01 (14.4 min), 6 recs/hr -> 1.44 recs.
        # Actually, let's use 0.05 days (72 mins) to be ultra safe.
        # Window size must be compatible with chunk duration.
        with open(self.config_path, "a") as f:
            f.write("\nchunk_duration_sec: 600.0\n") # Default 10 mins
            f.write("window_sec: 60.0\n") # 1 min window

        # Generate dataset
        gen_script = os.path.join(PROJECT_ROOT, "tools", "synth_photometry_dataset.py")
        gen_cmd = [
            sys.executable,
            gen_script,
            "--out", self.input_dir,
            "--format", "rwd",
            "--config", self.config_path,
            "--total-days", "0.02", # ~28.8 mins
            "--recordings-per-hour", "6", # 10 min intervals
            "--recording-duration-min", "10.0",
            "--n-rois", "1",
            "--preset", "biological_shared_nuisance"
        ]
        
        # msg = "Synth bypassed for debugging"
        subprocess.check_call(gen_cmd)
        
        # if not os.path.exists(self.input_dir):
        #    os.makedirs(self.input_dir)
        #    # Create dummy CSV to satisfy any content checks if any (validate_inputs doesn't check content)
        #    with open(os.path.join(self.input_dir, "fluorescence.csv"), "w") as f:
        #        f.write("dummy,header\n0,0")

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass # Windows file locking flakiness

    def test_success_contract(self):
        """Verify success path produces all deliverables and correct status."""
        run_id = "run_success"
        script = os.path.join(PROJECT_ROOT, "tools", "run_full_pipeline_deliverables.py")
        
        cmd = [
            sys.executable,
            script,
            "--input", self.input_dir,
            "--out-base", self.out_base,
            "--run-id", run_id,
            "--config", self.config_path,
            "--format", "rwd",
            "--events", "auto",
            "--sessions-per-hour", "6"
        ]
        
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"RETCODE: {e.returncode}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise e
        
        run_root = os.path.join(self.out_base, run_id)
        
        # 1. status.json exists
        status_path = os.path.join(run_root, "status.json")
        self.assertTrue(os.path.exists(status_path), "status.json missing")
        
        with open(status_path, "r") as f:
            s = json.load(f)
            
        # 2. Schema check
        self.assertEqual(s.get("phase"), "final", "Phase must be 'final' on completion")
        self.assertEqual(s.get("status"), "success", "Status must be 'success'")
        self.assertIsInstance(s.get("errors"), list)
        self.assertEqual(len(s.get("errors")), 0, "Errors dict should be empty on success")
        # UTC Check
        self.assertIn("+00:00", s.get("created_utc", ""), "created_utc must be UTC (+00:00)")
        self.assertIn("+00:00", s.get("finished_utc", ""), "finished_utc must be UTC (+00:00)")
        
        # 3. Output Package
        self.assertTrue(os.path.exists(os.path.join(run_root, "MANIFEST.json")), "MANIFEST.json missing")
        
        # 4. Region Deliverables
        # Determine region dir name (likely Region0G)
        regions = [d for d in os.listdir(run_root) if d.startswith("Region") and os.path.isdir(os.path.join(run_root, d))]
        self.assertTrue(len(regions) > 0, "No Region directories found")
        r_dir = os.path.join(run_root, regions[0])
        
        required_files = [
            "tonic_df_timeseries.csv",
            "tonic_overview.png"
        ]
        for fname in required_files:
            self.assertTrue(os.path.exists(os.path.join(r_dir, fname)), f"Missing {fname}")
            
        # Check for at least one phasic file set
        phasic_csvs = [f for f in os.listdir(r_dir) if "phasic_" in f and f.endswith(".csv")]
        phasic_pngs = [f for f in os.listdir(r_dir) if "phasic_" in f and f.endswith(".png")]
        self.assertTrue(len(phasic_csvs) > 0, "No phasic CSVs found")
        self.assertTrue(len(phasic_pngs) > 0, "No phasic PNGs found")

    def test_failure_contract(self):
        """Verify failure path produces error status and messages."""
        run_id = "run_failure"
        script = os.path.join(PROJECT_ROOT, "tools", "run_full_pipeline_deliverables.py")
        
        # Point to non-existent input
        bad_input = os.path.join(self.temp_dir, "NON_EXISTENT_DIR")
        
        cmd = [
            sys.executable,
            script,
            "--input", bad_input,
            "--out-base", self.out_base,
            "--run-id", run_id,
            "--config", self.config_path,
            "--format", "rwd"
        ]
        
        # Expect clean fail (exit code non-zero, but we catch it)
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.fail("Command should have failed")
        except subprocess.CalledProcessError:
            pass # Expected
            
        run_root = os.path.join(self.out_base, run_id)
        status_path = os.path.join(run_root, "status.json")
        
        # status.json MUST exist even on early failure (if run_dir created)
        # Note: run_full_pipeline_deliverables creates run_dir very early.
        self.assertTrue(os.path.exists(status_path), "status.json missing on failure")
        
        with open(status_path, "r") as f:
            s = json.load(f)
            
        self.assertEqual(s.get("phase"), "final")
        self.assertEqual(s.get("status"), "error")
        # UTC Check
        self.assertIn("+00:00", s.get("created_utc", ""), "created_utc must be UTC (+00:00)")
        self.assertIn("+00:00", s.get("finished_utc", ""), "finished_utc must be UTC (+00:00)")
        self.assertTrue(len(s.get("errors")) > 0, "Errors list must not be empty on failure")

if __name__ == "__main__":
    unittest.main()
