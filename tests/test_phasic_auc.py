
import unittest
import numpy as np
from photometry_pipeline.core.feature_extraction import compute_auc_above_threshold

class TestPhasicAUC(unittest.TestCase):
    """
    Test suite for Phasic AUC computation (Area Above Threshold).
    """

    def test_auc_zeros(self):
        # Case 1: All zeros, thresh=1.0 -> auc=0
        dff = np.array([0, 0, 0, 0], dtype=float)
        thresh = 1.0
        fs_hz = 1.0
        
        auc = compute_auc_above_threshold(dff, thresh, fs_hz=fs_hz)
        self.assertAlmostEqual(auc, 0.0, places=9)
        self.assertGreaterEqual(auc, 0.0)

    def test_auc_positive(self):
        # Case 2: Pulse [0, 2, 2, 0], thresh=1.0
        # Rectified: [0, 1, 1, 0]
        # Trapz (inclusive, dt=1): (0+1)/2 + (1+1)/2 + (1+0)/2 = 0.5 + 1.0 + 0.5 = 2.0
        dff = np.array([0, 2, 2, 0], dtype=float)
        thresh = 1.0
        fs_hz = 1.0
        
        auc = compute_auc_above_threshold(dff, thresh, fs_hz=fs_hz)
        self.assertAlmostEqual(auc, 2.0, places=9)
        self.assertGreaterEqual(auc, 0.0)
        
    def test_auc_negative_input(self):
        # Case 3: All negative, thresh=0.0 -> rect=[0,0,0,0], auc=0
        dff = np.array([-5, -4, -3, -2], dtype=float)
        thresh = 0.0
        fs_hz = 1.0
        
        auc = compute_auc_above_threshold(dff, thresh, fs_hz=fs_hz)
        self.assertAlmostEqual(auc, 0.0, places=9)
        self.assertGreaterEqual(auc, 0.0)
        
    def test_auc_below_threshold(self):
        # Signal positive but below threshold -> auc=0
        dff = np.array([0.5, 0.6, 0.5], dtype=float)
        thresh = 1.0
        fs_hz = 1.0
        
        auc = compute_auc_above_threshold(dff, thresh, fs_hz=fs_hz)
        self.assertAlmostEqual(auc, 0.0, places=9)

    def test_auc_time_vector(self):
        # Case with time_s provided
        dff = np.array([0, 2, 2, 0], dtype=float)
        thresh = 1.0
        # time_s with dt=0.5 -> Should halve the Area
        time_s = np.array([0.0, 0.5, 1.0, 1.5]) 
        
        # Rect: [0, 1, 1, 0]
        # Trapz: 0.5*(0.5+0.5) + ... -> 2.0 * 0.5 = 1.0
        auc = compute_auc_above_threshold(dff, thresh, time_s=time_s)
        self.assertAlmostEqual(auc, 1.0, places=9)

    def test_auc_validation_errors(self):
        dff = np.array([0, 1, 0], dtype=float)
        
        # Invalid fs_hz when time_s is None
        with self.assertRaisesRegex(ValueError, "fs_hz must be > 0"):
            compute_auc_above_threshold(dff, baseline_value=0.0, fs_hz=0)
            
        # Invalid baseline_value
        with self.assertRaisesRegex(ValueError, "baseline_value must be finite"):
            compute_auc_above_threshold(dff, baseline_value=np.nan, fs_hz=1.0)
            
        # Time mismatch
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            compute_auc_above_threshold(dff, baseline_value=0.0, time_s=np.array([0, 1]))

if __name__ == '__main__':
    unittest.main()
