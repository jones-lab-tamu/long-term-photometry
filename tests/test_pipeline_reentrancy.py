import unittest
import os
import shutil
import tempfile
import sys
import subprocess

# Secure imports and backend setup
import matplotlib
matplotlib.use('Agg')

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    # Remove existing project root entries to prevent duplicates/confusion (cleanup)
    # We insist on insertion at 0
    sys.path.insert(0, PROJECT_ROOT)

from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline

class TestPipelineReentrancy(unittest.TestCase):
    """
    Regression test for file locking / global state issues.
    Ensures pipeline can be run multiple times in the same process
    with different output directories.
    Harden checks: Absolute paths, Agg backend, minimal data.
    """
    
    def setUp(self):
        self.temp_roots = []
        
    def tearDown(self):
        for path in self.temp_roots:
            shutil.rmtree(path, ignore_errors=True)
            
    def _make_temp_dir(self):
        d = tempfile.mkdtemp(prefix="reentrancy_test_")
        self.temp_roots.append(d)
        return d
        
    def test_double_run_same_process(self):
        """Invoke pipeline.run() twice in sequence with fresh directories."""
        
        # 1. Setup Input Data (Tiny)
        input_dir = self._make_temp_dir()
        config_path = os.path.join(input_dir, "config.yaml")
        
        # Write Minimal Config 
        with open(config_path, "w") as f:
            f.write("""
target_fs_hz: 10
chunk_duration_sec: 60
rwd_time_col: Time(s)
uv_suffix: "-410"
sig_suffix: "-470"
window_sec: 5.0
step_sec: 1.0
baseline_method: uv_globalfit_percentile_session
peak_threshold_method: mean_std
""")
        
        # Generate 1 small chunk (1 min)
        # Use absolute path to tool to avoid CWD dependency
        synth_script = os.path.join(PROJECT_ROOT, "tools", "synth_photometry_dataset.py")
        
        gen_cmd = [
            sys.executable, synth_script,
            "--out", input_dir,
            "--format", "rwd",
            "--config", config_path,
            "--recording-duration-min", "1.0",
            # 0.002 days * 1440 min/day = 2.88 mins. 
            # With 60 recs/hr (1 min intervals), expected ~2 chunks. perfect.
            "--total-days", "0.002", 
            "--recordings-per-hour", "60", 
            "--n-rois", "1"
        ]
        # Capture output for debugging
        result = subprocess.run(gen_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Synth Gen Failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            self.fail("Synthetic data generation failed")
        
        # 2. Run 1
        out_1 = self._make_temp_dir()
        cfg = Config.from_yaml(config_path)
        pl = Pipeline(cfg)
        print(f"Run 1 -> {out_1}")
        pl.run(input_dir, out_1, force_format='rwd', recursive=False)
        
        self.assertTrue(os.path.exists(os.path.join(out_1, "run_report.json")))
        
        # 3. Run 2 (Fresh args)
        out_2 = self._make_temp_dir()
        print(f"Run 2 -> {out_2}")
        
        # Re-instantiate or reuse? Testing both is good, but reuse is harder.
        # Let's test FRESH instantiation first (standard usage).
        pl2_fresh = Pipeline(cfg)
        pl2_fresh.run(input_dir, out_2, force_format='rwd', recursive=False)
        
        self.assertTrue(os.path.exists(os.path.join(out_2, "run_report.json")))
        
if __name__ == "__main__":
    unittest.main()
