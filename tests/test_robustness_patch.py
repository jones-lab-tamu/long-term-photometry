
import unittest
import os
import shutil
import json
import numpy as np
import pandas as pd
from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.core.reporting import generate_run_report
from photometry_pipeline.pipeline import Pipeline
# Import helper explicitly
from photometry_pipeline.core.regression import _get_window_indices, fit_chunk_dynamic

class TestRobustnessPatch(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_robustness_final"
        os.makedirs(self.test_dir, exist_ok=True)
        self.config = Config()
        self.config.seed = 42 # Determinism

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_peak_method_consistency(self):
        """Test A: Strict peak method names and default."""
        # 1. Check Default
        self.assertEqual(self.config.peak_threshold_method, 'mean_std')
        
        # 2. Strict Supported List
        t = np.arange(5)
        chunk = Chunk(
            chunk_id=0, source_file="dummy", format='rwd',
            time_sec=t, uv_raw=np.zeros((5,1)), sig_raw=np.zeros((5,1)),
            fs_hz=1.0, channel_names=["R0"]
        )
        chunk.dff = np.zeros((5,1)) 
        
        # 'mean_plus_k_std' alias removed -> should error
        self.config.peak_threshold_method = 'mean_plus_k_std'
        with self.assertRaises(ValueError) as cm:
            extract_features(chunk, self.config)
        self.assertIn("Supported: ['mean_std', 'percentile']", str(cm.exception))

    def test_regression_window_boundaries(self):
        """Test 2: Regression window helper exactness."""
        n_samples = 100
        w = 10
        
        # Valid Center (Interior)
        # Center 50. Half 5. Start 45. End 55. [45, 55]. Valid.
        s, e = _get_window_indices(50, w, n_samples)
        self.assertEqual(e - s, w)
        self.assertEqual(s, 45)
        
        # Left Edge Fail
        # Center=4. Start=-1. End=9. Fail.
        res = _get_window_indices(4, w, n_samples)
        self.assertIsNone(res)
        
        # Right Edge Fail
        # Center=96. Start=91. End=101. Fail.
        res = _get_window_indices(96, w, n_samples)
        self.assertIsNone(res)
        
        # Left Edge Exact Fit? 
        # Center=5. Start=0. End=10. Valid.
        s, e = _get_window_indices(5, w, n_samples)
        self.assertEqual(s, 0)
        self.assertEqual(e, 10) # 0..10 is 10 samples (0 to 9 indices)

    def test_baseline_qc_counts_stable(self):
        """Test 3: Baseline QC counts schema."""
        inp = os.path.join(self.test_dir, "input")
        os.makedirs(inp, exist_ok=True)
        # Need VALID strict RWD coverage (start 0, end duration) 
        # Duration default 600.
        # Minimal file with 2 points: 0, 1. Duration=1. 
        # Set config to match.
        with open(os.path.join(inp, "file.csv"), "w") as f:
            f.write("Time(s),Region0-410,Region0-470\n0,1,1\n1,1,1\n") 
            
        self.config.chunk_duration_sec = 1.0 
        self.config.target_fs_hz = 1.0
        self.config.allow_partial_final_chunk = True
        self.config.f0_min_value = 2.0 # Force failure (F0=1.0)
        
        pipe = Pipeline(self.config)
        out = os.path.join(self.test_dir, "output")
        pipe.run(inp, out)
        
        with open(os.path.join(out, "qc", "qc_summary.json")) as f:
            qc = json.load(f)
            
        self.assertIn("invalid_baseline_rois", qc)
        self.assertEqual(qc["baseline_invalid_roi_count"], 1)
        self.assertEqual(qc["baseline_invalid_roi_chunk_pairs"], 1)

if __name__ == '__main__':
    unittest.main()
