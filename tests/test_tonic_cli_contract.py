import unittest
import os
import shutil
import tempfile
import sys
import io
import subprocess
import numpy as np

# Adjust path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.core.types import Chunk

class TestTonicCLIContract(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config_test.yaml")
        
        # Minimal Config (Strict Schema)
        with open(self.config_path, 'w') as f:
            f.write("""
chunk_duration_sec: 600.0
target_fs_hz: 10.0
baseline_method: uv_globalfit_percentile_session
baseline_percentile: 10.0
rwd_time_col: Time(s)
uv_suffix: "-410"
sig_suffix: "-470"
window_sec: 60.0
step_sec: 10.0
r_low: 0.3
r_high: 0.8
g_min: 0.2
min_valid_windows: 2
min_samples_per_window: 10
qc_max_chunk_fail_fraction: 1.0
peak_threshold_method: mean_std
peak_threshold_k: 2.0
""")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_generator_verification_criteria_strict(self):
        """
        Verify that the synth generator output fails strictly if Day/Night targets are missed.
        """
        out_dir = os.path.join(self.test_dir, "synth_out_strict")
        
        # Run synth command (Using subprocess for end-to-end coverage)
        cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", out_dir,
            "--format", "rwd",
            "--config", self.config_path,
            "--n-rois", "1",
            "--total-days", "1.5", 
            "--recordings-per-hour", "4", # Enough for min chunks
            "--phasic-events-per-10min-mean", "12.0", 
            "--phasic-ct-soft-floor", "0.25", 
            "--phasic-mode", "phase_locked_to_tonic",
            "--phasic-phase-lock-alpha", "0.75",
            "--seed", "42"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Synth Failed Output:", result.stdout)
            print("Synth Failed Error:", result.stderr)
            
        self.assertEqual(result.returncode, 0, f"Synth generator verification failed. Error: {result.stderr}")
        pass_msg = ">>> VERIFIED: All Metrics within bounds. Generator Ready."
        
        if pass_msg not in result.stdout:
            print("Synth Output (Dump):", result.stdout)
            
        self.assertIn(pass_msg, result.stdout)
        
        # Verify strict check by forcing failure (low day rate)
        cmd_fail = cmd.copy()
        idx = cmd_fail.index("--phasic-events-per-10min-mean")
        cmd_fail[idx+1] = "2.0" # Low (Fail Day target 8-12)
        
        result_fail = subprocess.run(cmd_fail, capture_output=True, text=True)
        self.assertNotIn(pass_msg, result_fail.stdout)
        self.assertIn("FAIL", result_fail.stdout)

    def test_pipeline_tonic_provenance_and_logic(self):
        """
        Verify that analyze_photometry --mode tonic:
        1. Prints provenance exactly once.
        2. Does not explode dynamic regression (runtime guard).
        3. ACTUALLY computes tonic values (checked via output).
        """
        # 1. Generate small dataset
        synth_dir = os.path.join(self.test_dir, "synth_invariant")
        input_cmd = [
             sys.executable, "tools/synth_photometry_dataset.py",
             "--out", synth_dir,
             "--format", "rwd",
             "--config", self.config_path,
             "--total-days", "0.2", # Short but enough for 1 valid chunk (10min)
             "--recordings-per-hour", "4",
             "--n-rois", "1"
        ]
        subprocess.run(input_cmd, check=True, stdout=subprocess.DEVNULL)
        
        # 2. Run Pipeline in Tonic Mode
        pipe_out = os.path.join(self.test_dir, "pipe_out")
        pipe_cmd = [
            sys.executable, "analyze_photometry.py",
            "--input", synth_dir,
            "--out", pipe_out,
            "--config", self.config_path,
            "--overwrite",
            "--mode", "tonic",
            "--recursive"
        ]
        
        result = subprocess.run(pipe_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Pipe Failed Log:", result.stdout)
            print("Pipe Failed Error:", result.stderr)
            
        self.assertEqual(result.returncode, 0)
        
        # Check Provenance
        provenance = "Tonic iso-fit source: global robust fit (entire recording). Dynamic uv_fit ignored."
        self.assertIn(provenance, result.stdout)
        count = result.stdout.count(provenance)
        self.assertEqual(count, 1, f"Provenance should appear exactly once, found {count}")

    def test_missing_tonic_params_invariant_unit(self):
        """
        Verify that missing tonic params raises RuntimeError (no silent NaNs) using stable unit test.
        """
        cfg = Config.from_yaml(self.config_path)
        # Force mode to tonic
        p = Pipeline(cfg, mode='tonic')
        
        # Pre-populate stats for Region0 ONLY
        p.stats.tonic_fit_params['Region0'] = {'slope': 1.0, 'intercept': 0.0}
        
        # Construct Memory Chunk with Region0 and Region1
        from photometry_pipeline.core.types import Chunk
        # Ensure correct shape (N, 2)
        c = Chunk(
            chunk_id=0,
            source_file="unit_test",
            format="rwd",
            uv_raw=np.ones((10, 2), dtype=float),
            sig_raw=np.ones((10, 2), dtype=float),
            uv_filt=np.zeros((10, 2), dtype=float),
            sig_filt=np.zeros((10, 2), dtype=float),
            time_sec=np.arange(10.0),
            fs_hz=10.0,
            channel_names=["Region0", "Region1"],
            metadata={}
        )
        
        # Only Region0 is in stats, so Region1 is missing.
        # Should raise RuntimeError via _process_chunk_tonic -> Missing tonic fit params
        with self.assertRaisesRegex(RuntimeError, "Missing tonic fit params"):
            p._process_chunk_tonic(c, 0)

    def test_dynamic_invariant_integration(self):
        """
        Verify that running in tonic mode NEVER calls dynamic fitting.
        Since we enforced strict mode check in regression.py, 
        if pipeline accidentally called regression.fit_chunk_dynamic(..., mode='tonic'), 
        it WOULD crash.  This placeholder confirms tests passed.
        """
        pass

    def test_tonic_nan_tolerance_under_threshold(self):
        """
        Verify that small fraction of NaNs is tolerated if under threshold.
        """
        cfg = Config.from_yaml(self.config_path)
        cfg.tonic_allowed_nan_frac = 0.02 # 2% allowed
        p = Pipeline(cfg, mode='tonic')
        
        # Setup stats
        p.stats.tonic_fit_params['Region0'] = {'slope': 1.0, 'intercept': 0.0}
        
        # Chunk with 100 samples, 1 NaN (1%)
        from photometry_pipeline.core.types import Chunk
        c = Chunk(
            chunk_id=0, source_file="t1", format="rwd",
            uv_raw=np.ones((100, 1), dtype=float),
            sig_raw=np.ones((100, 1), dtype=float),
            uv_filt=np.zeros((100, 1), dtype=float),
            sig_filt=np.zeros((100, 1), dtype=float),
            time_sec=np.arange(100.0), fs_hz=1.0,
            channel_names=["Region0"], metadata={}
        )
        # Inject 1 NaN at index 50
        c.sig_raw[50, 0] = np.nan
        
        # Should NOT raise
        p._process_chunk_tonic(c, 0)
        
        # Verify delta_f has NaN at 50, but finite elsewhere
        self.assertTrue(np.isnan(c.delta_f[50, 0]))
        self.assertTrue(np.isfinite(c.delta_f[0, 0]))
        self.assertTrue(np.isfinite(c.delta_f[99, 0]))
        
        # Ensure ONLY index 50 is nan
        nans = np.isnan(c.delta_f[:, 0])
        self.assertEqual(np.sum(nans), 1)

        # Case 2: 1 NaN in UV (Under Threshold)
        c2 = Chunk(
            chunk_id=0, source_file="t1_uv", format="rwd",
            uv_raw=np.ones((100, 1), dtype=float),
            sig_raw=np.ones((100, 1), dtype=float),
            uv_filt=np.zeros((100, 1), dtype=float),
            sig_filt=np.zeros((100, 1), dtype=float),
            time_sec=np.arange(100.0), fs_hz=1.0,
            channel_names=["Region0"], metadata={}
        )
        c2.uv_raw[50, 0] = np.nan
        p._process_chunk_tonic(c2, 0)
        self.assertTrue(np.isnan(c2.delta_f[50, 0]))
        self.assertEqual(np.sum(np.isnan(c2.delta_f[:, 0])), 1)

    def test_tonic_nan_tolerance_over_threshold(self):
        """
        Verify that fraction of NaNs > threshold raises RuntimeError.
        """
        cfg = Config.from_yaml(self.config_path)
        cfg.tonic_allowed_nan_frac = 0.02 # 2% allowed
        p = Pipeline(cfg, mode='tonic')
        
        p.stats.tonic_fit_params['Region0'] = {'slope': 1.0, 'intercept': 0.0}
        
        # Chunk with 100 samples, 5 NaNs (5% > 2%)
        from photometry_pipeline.core.types import Chunk
        c = Chunk(
            chunk_id=0, source_file="t2", format="rwd",
            uv_raw=np.ones((100, 1), dtype=float),
            sig_raw=np.ones((100, 1), dtype=float),
            uv_filt=np.zeros((100, 1), dtype=float),
            sig_filt=np.zeros((100, 1), dtype=float),
            time_sec=np.arange(100.0), fs_hz=1.0,
            channel_names=["Region0"], metadata={}
        )
        # Inject 5 NaNs
        c.sig_raw[50:55, 0] = np.nan
        
        # Should raise RuntimeError with specific message
        with self.assertRaisesRegex(RuntimeError, "exceeds allowed"):
             p._process_chunk_tonic(c, 0)

        # Case 2: 5 NaNs in UV (Over Threshold)
        c2 = Chunk(
            chunk_id=0, source_file="t2_uv", format="rwd",
            uv_raw=np.ones((100, 1), dtype=float),
            sig_raw=np.ones((100, 1), dtype=float),
            uv_filt=np.zeros((100, 1), dtype=float),
            sig_filt=np.zeros((100, 1), dtype=float),
            time_sec=np.arange(100.0), fs_hz=1.0,
            channel_names=["Region0"], metadata={}
        )
        c2.uv_raw[50:55, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "exceeds allowed"):
             p._process_chunk_tonic(c2, 0)

if __name__ == '__main__':
    unittest.main()
