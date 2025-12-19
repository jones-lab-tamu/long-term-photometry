
import unittest
import numpy as np
import pandas as pd
import os
import shutil
from photometry_pipeline.config import Config
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.core.regression import fit_chunk_dynamic

class TestStrictCompliance(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_strict"
        os.makedirs(self.test_dir, exist_ok=True)
        self.config = Config()
        self.config.target_fs_hz = 40.0
        self.config.chunk_duration_sec = 10.0
        self.config.allow_partial_final_chunk = False
        self.config.rwd_time_col = "Time"
        self.config.uv_suffix = "-410"
        self.config.sig_suffix = "-470"
        self.config.window_sec = 2.0
        self.config.step_sec = 1.0
        self.config.min_samples_per_window = 0
        self.config.min_valid_windows = 2
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_rwd_strict_rejection(self):
        # Create short RWD file (9.0s instead of 10.0s)
        # Grid end for 10s is 10.0. 
        # Raw end 9.0 < 9.975 -> Fail
        t = np.arange(0, 9.0, 1.0/40.0)
        df = pd.DataFrame({
            "Time": t,
            "Region0-410": np.random.randn(len(t)),
            "Region0-470": np.random.randn(len(t))
        })
        path = os.path.join(self.test_dir, "short_rwd.csv")
        df.to_csv(path, index=False)
        
        with self.assertRaises(ValueError) as cm:
            load_chunk(path, 'rwd', self.config, 0)
        self.assertIn("RWD strict: raw_end", str(cm.exception))
        
    def test_npm_strict_overlap_rejection(self):
        # Construction: 80Hz total (40Hz per channel)
        # 0 to 10.5 seconds
        # Overlap mismatch by shifting Signal +0.8s
        
        t_base = np.arange(0, 10.5, 1.0/80.0)
        n = len(t_base)
        
        # Interleaved indices
        # Even = UV, Odd = Signal
        led_state = np.array([1 if i%2 == 0 else 2 for i in range(n)])
        
        # Base timestamps
        sys_ts = t_base.copy()
        
        # Shift Signal timestamps (LedState == 2)
        sys_ts[led_state == 2] += 0.8
        
        # Create DataFrame
        df = pd.DataFrame({
            "SystemTimestamp": sys_ts,
            "FrameCounter": np.arange(n),
            "LedState": led_state,
            "Region0G": np.random.randn(n)
        })
        
        path = os.path.join(self.test_dir, "mismatch_npm.csv")
        df.to_csv(path, index=False)
        
        self.config.npm_system_ts_col = "SystemTimestamp"
        self.config.npm_led_col = "LedState"
        self.config.npm_region_prefix = "Region"
        self.config.npm_region_suffix = "G"
        self.config.npm_time_axis = "system_timestamp"
        
        # t0 = max(0, 0.8) = 0.8
        # t1 = min(t_uv_end, t_sig_end). 
        # t_uv ends at ~10.5. t_sig ends at ~10.5 + 0.8 = 11.3.
        # t1 = 10.5.
        # overlap = 10.5 - 0.8 = 9.7s.
        # Target = 10.0s. 
        # 9.7 < 10.0 -> Fail.
        
        with self.assertRaises(ValueError) as cm:
            load_chunk(path, 'npm', self.config, 0)
        self.assertIn("NPM strict: overlap insufficient", str(cm.exception))

    def test_regression_nan_robustness(self):
        # Create chunk
        n = 400 # 10s at 40Hz
        t = np.arange(n)/40.0
        uv = np.random.randn(n, 1)
        sig = np.random.randn(n, 1)
        
        # Correlated signal
        sig = uv * 2.0 + 1.0 + np.random.randn(n, 1)*0.1
        
        chunk = Chunk(
            chunk_id=0, source_file="dummy", format="rwd",
            time_sec=t, uv_raw=uv, sig_raw=sig, fs_hz=40.0,
            channel_names=["Region0"], metadata={}
        )
        # filtered same as raw for test
        chunk.uv_filt = uv.copy()
        chunk.sig_filt = sig.copy()
        
        # Inject NaNs in first 2 seconds (first window)
        # This forces the regression to skip or struggle in the first window
        chunk.uv_filt[0:80, 0] = np.nan
        
        uv_fit, delta_f = fit_chunk_dynamic(chunk, self.config)
        
        # Assertions
        # 1. Correct Shapes
        self.assertEqual(uv_fit.shape, (n, 1))
        self.assertEqual(delta_f.shape, (n, 1))
        
        # 2. At least SOME finite values (regression succeeded somewhere)
        self.assertTrue(np.any(np.isfinite(uv_fit)))
        self.assertTrue(np.any(np.isfinite(delta_f)))
        
        # 3. Do NOT assert finiteness in the NaN region, as strict masking might leave NaNs.


if __name__ == '__main__':
    unittest.main()
