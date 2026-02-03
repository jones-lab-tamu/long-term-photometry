
import unittest
import numpy as np
import traceback
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.core.regression import fit_chunk_dynamic

class TestRegressionLimit(unittest.TestCase):
    def test_dynamic_regression_nan_resilience(self):
        """
        Verify that fit_chunk_dynamic does not crash when encountering
        windows with insufficient valid samples or heavy NaN contamination.
        """
        config = Config()
        config.target_fs_hz = 40.0
        config.chunk_duration_sec = 10.0
        config.window_sec = 2.0
        config.step_sec = 1.0
        config.min_valid_windows = 1 # Allow even a single valid window to pass
        
        n = 400
        t = np.arange(n)/40.0
        uv = np.random.randn(n, 1)
        sig = uv * 2.0 + 1.0
        
        chunk = Chunk(0, "dummy", "rwd", t, uv, sig, 40.0, ["Region0"], {})
        chunk.uv_filt = uv.copy()
        chunk.sig_filt = sig.copy()
        
        # Inject NaNs to cause insufficient data in the first window
        # Window size is 2s * 40Hz = 80 samples.
        # Wiping out 50 samples leaves 30.
        # If min_samples_per_window > 30, it should skip.
        # Default dynamic min_samples is 80% (64). So this should skip.
        chunk.uv_filt[0:50, 0] = np.nan
        
        print("Starting fit (NaN resilience test)...")
        try:
            uv_fit, delta_f = fit_chunk_dynamic(chunk, config)
            print("Fit complete.")
        except Exception:
            self.fail(f"fit_chunk_dynamic crashed with:\n{traceback.format_exc()}")
            
        # Assertions to ensure we got result arrays back
        self.assertIsNotNone(uv_fit)
        self.assertIsNotNone(delta_f)
        self.assertEqual(uv_fit.shape, chunk.uv_filt.shape)
        
        # Since we wiped the first bit, the output should probably be NaN there
        # But we mostly care that it didn't crash.
        
if __name__ == '__main__':
    unittest.main()
