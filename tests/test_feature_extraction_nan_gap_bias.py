import numpy as np
import pandas as pd
import unittest
from photometry_pipeline.core.feature_extraction import extract_features

class MockChunk:
    def __init__(self, dff, time_sec, fs_hz):
        self.dff = dff
        self.time_sec = time_sec
        self.fs_hz = fs_hz
        self.chunk_id = "test_chunk"
        self.source_file = "test.raw"
        self.channel_names = ["ROI1"]
        self.metadata = {"qc_warnings": []}

class MockConfig:
    def __init__(self):
        self.peak_threshold_method = "mean_std"
        self.peak_threshold_k = 2.0
        self.peak_min_distance_sec = 0.5
        self.event_auc_baseline = "zero"
        self.peak_pre_filter = "none"

class TestFeatureExtractionNanGap(unittest.TestCase):
    def test_nan_gap_split_within_segment(self):
        """
        Verify that result is aggregated across finite runs and DD5 emits absolute indices.
        """
        fs = 100.0
        t = np.arange(100) / fs
        
        # Scenario: A run of 1 at start, then a gap, then a run of 98.
        y = np.full(100, np.nan)
        y[0] = 5.0      # Run 1: index 0:1
        y[2:100] = 0.0  # Run 2: index 2:100
        y[50] = 10.0    # Peak in second run
        
        chunk = MockChunk(y.reshape(-1, 1), t, fs)
        config = MockConfig()
        config.peak_pre_filter = "none"
        
        df = extract_features(chunk, config)
        
        warnings = chunk.metadata.get('qc_warnings', [])
        # Run 1 (len=1) should emit DD5 with absolute indices "0:1"
        dd5 = [w for w in warnings if "DD5" in w and "0:1" in w]
        self.assertEqual(len(dd5), 1, f"Expected DD5 '0:1', got: {warnings}")
        
        # Audit counts
        qc_counts = chunk.metadata.get('qc_counts', {})
        # Note: Raw NaNs become segment boundaries, so DD6 (inner gap) 
        # is not triggered for raw NaNs. Only DD5 (short runs).
        self.assertEqual(qc_counts.get('DD6', 0), 0)
        self.assertEqual(qc_counts.get('DD5', 0), 1)

        # Result must be aggregated (peak_count=1)
        self.assertEqual(df.loc[0, 'peak_count'], 1)

    def test_dd6_filter_nan_distinction(self):
        """Verify DD6 + FILTER_NAN when filter introduces a gap."""
        fs = 100.0
        t = np.arange(100) / fs
        y = np.zeros(100) # One contiguous finite segment [0:100]
        
        chunk = MockChunk(y.reshape(-1, 1), t, fs)
        config = MockConfig()
        
        import photometry_pipeline.core.feature_extraction as fe
        original_filter = fe.lowpass_filter
        def mock_filter(data, fs_hz, config):
            out = data.copy()
            out[50] = np.nan # Introduce a gap
            return out
        
        fe.lowpass_filter = mock_filter
        config.peak_pre_filter = "lowpass"
        
        try:
            df = extract_features(chunk, config)
            
            warnings = chunk.metadata.get('qc_warnings', [])
            qc_counts = chunk.metadata.get('qc_counts', {})

            # 1. DD6 precise wording
            self.assertTrue(any("Analysis signal has NaNs inside raw-finite segment 0:100" in w for w in warnings))
            
            # 2. FILTER_NAN warning
            self.assertTrue(any("DEGENERATE[FILTER_NAN]" in w and "0:100" in w for w in warnings))
            
            # 3. Counters
            self.assertEqual(qc_counts.get('DD6', 0), 1)
            self.assertEqual(qc_counts.get('FILTER_NAN', 0), 1)
        finally:
            fe.lowpass_filter = original_filter

    def test_dd6_raw_data_no_filter_nan(self):
        """Verify DD6 happens for raw NaNs, but FILTER_NAN does NOT."""
        fs = 100.0
        t = np.arange(100) / fs
        y = np.full(100, 0.0)
        y[50] = np.nan # Raw NaN gap
        
    def test_dd6_raw_data_no_filter_nan(self):
        """Verify DD6 happens 0 times for raw NaNs because boundaries split segments."""
        fs = 100.0
        t = np.arange(100) / fs
        
        # Scenario: trace has a NaN in the middle.
        # This creates two segments: [0:50] and [51:100].
        # Neither segment contains a NaN inside its boundaries.
        y = np.full(100, 5.0)
        y[50] = np.nan 
        
        # Make one finite run short to trigger DD5
        # Segment 1 is now [0:1]. Segment 2 is [51:100].
        # (The original segment 0:50 was split by the introduction of NaNs at y[1:50])
        y[1:50] = np.nan
        
        chunk = MockChunk(y.reshape(-1, 1), t, fs)
        config = MockConfig()
        config.peak_pre_filter = "none"
        
        df = extract_features(chunk, config)
        
        warnings = chunk.metadata.get('qc_warnings', [])
        qc_counts = chunk.metadata.get('qc_counts', {})
        
        # 1. DD6 must be 0 (no gaps INSIDE segments)
        self.assertEqual(qc_counts.get('DD6', 0), 0)
        
        # 2. FILTER_NAN must be 0
        self.assertEqual(qc_counts.get('FILTER_NAN', 0), 0)
        
        # 3. DD5 must be 1 (for Segment 1 which is length 1)
        # Absolute index of first segment start is 0, end is 1.
        self.assertTrue(any("DEGENERATE[DD5]" in w and "0:1" in w for w in warnings))
        self.assertEqual(qc_counts.get('DD5', 0), 1)
        
        # 4. Aggregation: peak_count is non-negative int
        self.assertEqual(len(df), 1)
        self.assertIsInstance(df.loc[0, 'peak_count'], (int, np.integer))
        self.assertGreaterEqual(df.loc[0, 'peak_count'], 0)

if __name__ == "__main__":
    unittest.main()
