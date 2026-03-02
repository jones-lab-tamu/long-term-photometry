
import unittest
import numpy as np
from photometry_pipeline.core.qc import compute_lowfreq_preservation_metric
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.config import Config

from photometry_pipeline.core.regression import fit_chunk_dynamic
import pandas as pd
import os
import shutil

class TestInfrastructureHardening(unittest.TestCase):
    
    def test_tonic_preservation_qc_metric_smoke(self):
        """
        Validates the QC metric logic itself using a known signal + identity transform.
        Does NOT rely on regression configuration.
        """
        fs = 40.0
        T = 600.0
        n = int(T * fs)
        t = np.arange(n) / fs
        
        np.random.seed(42)
        f_slow = 0.005
        slow_sine = np.sin(2 * np.pi * f_slow * t)
        
        # Test Case 1: Identity (Perfect preservation)
        # Metric should be 1.0
        metric = compute_lowfreq_preservation_metric(slow_sine, slow_sine, fs, 0.02)
        self.assertAlmostEqual(metric, 1.0, places=4)
        
        # Test Case 2: Signal + Noise vs Signal
        # Should still be high
        noisy = slow_sine + 0.1 * np.random.randn(n)
        metric_noise = compute_lowfreq_preservation_metric(noisy, slow_sine, fs, 0.02)
        self.assertGreater(metric_noise, 0.95)
        
    def test_regression_attenuates_tonic_with_short_window(self):
        """
        Demonstrates that standard artifact correction (linear regression)
        with a window shorter than the tonic period WILL attenuate the tonic signal.
        This limits the ability to claim 'preservation' as a strict invariant under default settings.
        """
        fs = 40.0
        T = 600.0
        n = int(T * fs)
        t = np.arange(n) / fs
        
        np.random.seed(42)
        
        # Slow signal (Period 200s)
        f_slow = 0.005
        slow_sine = np.sin(2 * np.pi * f_slow * t).reshape(-1, 1)
        
        # Artifact (Faster)
        artifact = 0.2 * np.sin(2 * np.pi * 0.05 * t).reshape(-1, 1)
        noise = 0.05 * np.random.randn(n).reshape(-1, 1)
        
        uv_raw = artifact + noise
        sig_raw = slow_sine + artifact + noise
        
        chunk = Chunk(
            chunk_id=0,
            source_file="dummy",
            format="rwd",
            time_sec=t,
            uv_raw=uv_raw,
            sig_raw=sig_raw,
            fs_hz=fs,
            channel_names=["Region0"],
            metadata={}
        )
        chunk.uv_filt = uv_raw.copy()
        chunk.sig_filt = sig_raw.copy()
        
        config = Config()
        # Default window is 60s (<< 200s period)
        # This will regress out the slow sine locally
        config.window_sec = 60.0
        config.step_sec = 10.0
        config.min_valid_windows = 2
        config.min_samples_per_window = 0
        
        uv_fit, delta_f = fit_chunk_dynamic(chunk, config, mode='phasic')
        delta_f_roi = delta_f[:, 0]
        sig_roi = sig_raw[:, 0]
        
        # Regression should succeed (not all NaNs)
        self.assertTrue(np.any(np.isfinite(delta_f_roi)), "Regression returned all NaNs")
        
        metric = compute_lowfreq_preservation_metric(sig_roi, delta_f_roi, fs, 0.02)
        
        # Expect significant attenuation (low correlation with original slow sine)
        self.assertTrue(np.isfinite(metric), "Metric is NaN")
        self.assertLess(metric, 0.8, f"Expected attenuation with short window, got r={metric:.3f}")

    def test_session_time_metadata_presence(self):
        """
        Verifies that load_chunk populates session_time metadata.
        Uses a temporary CSV file to exercise the real loader path.
        """
        test_dir = "tests/temp_infra"
        os.makedirs(test_dir, exist_ok=True)
        try:
            # Create minimal RWD CSV
            path = os.path.join(test_dir, "mysession.csv")
            t = np.arange(0, 5.0, 1.0/40.0) # 5 seconds
            df = pd.DataFrame({
                "Time(s)": t,
                "Region0-410": np.zeros(len(t)),
                "Region0-470": np.zeros(len(t))
            })
            df.to_csv(path, index=False)
            
            config = Config()
            # Standard config defaults are fine, maybe set suffix matches
            config.rwd_time_col = "Time(s)"
            config.uv_suffix = "-410"
            config.sig_suffix = "-470"
            config.chunk_duration_sec = 10.0 # Will underfill but that's allowed in permissive default or we use short
            config.allow_partial_final_chunk = True
            
            from photometry_pipeline.io.adapters import load_chunk
            
            # Load
            chunk = load_chunk(path, 'rwd', config, 0)
            
            # Check presence without helper call
            self.assertIn("session_time", chunk.metadata)
            meta = chunk.metadata["session_time"]
            
            # Verify basic fields
            self.assertEqual(meta["session_id"], "mysession")
            self.assertEqual(meta["chunk_index"], 0)
            
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        
if __name__ == '__main__':
    unittest.main()
