
import unittest
import sys
import os
import io
import contextlib
import numpy as np

# Ensure we can import tools robustly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from tools.plot_phasic_intermediate_chain import evaluate_synth_gating

class TestSynthGatingCornerCases(unittest.TestCase):
    """
    Tests for Tier 2 Gating Logic in evaluate_synth_gating.
    """
    
    def test_pass_standard(self):
        """Standard high modulation should pass."""
        # 10 chunks high (100 peaks), 10 chunks low (10 peaks)
        # Ratio 10 > 1.25 -> Pass
        stats = []
        for i in range(10): stats.append({'tonic': 100, 'peaks': 100})
        for i in range(10): stats.append({'tonic': 0, 'peaks': 10})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should pass")
        self.assertIn("Gating Check: PASS", out)
        self.assertIn("Modulation Ratio:", out)

    def test_fail_zero_both(self):
        """Both groups zero peaks should fail."""
        stats = []
        for i in range(10): stats.append({'tonic': 100, 'peaks': 0})
        for i in range(10): stats.append({'tonic': 0, 'peaks': 0})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertFalse(res, "Should fail when both zero")
        self.assertIn("FAIL", out)
        self.assertIn("Both groups have zero peaks", out)

    def test_pass_zero_min(self):
        """Zero min peaks (infinite modulation) should pass."""
        stats = []
        for i in range(10): stats.append({'tonic': 100, 'peaks': 50})
        for i in range(10): stats.append({'tonic': 0, 'peaks': 0})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should pass with infinite ratio")
        self.assertIn("Modulation Ratio: inf", out)
        self.assertIn("PASS", out)

    def test_skip_no_separation(self):
        """No tonic separation should skip."""
        # All same tonic
        stats = []
        for i in range(20): stats.append({'tonic': 50, 'peaks': 100})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should skip (return True)")
        self.assertIn("SKIPPED (No Tonic Separation)", out)
        # q_hi should equal q_lo

    def test_skip_insufficient_data(self):
        """Not enough chunks total."""
        stats = []
        for i in range(5): stats.append({'tonic': 50, 'peaks': 100})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should skip")
        self.assertIn("SKIPPED (Insufficient Data)", out)

    def test_skip_small_groups(self):
        """Total enough, but split creates small groups."""
        # 18 chunks. Min chunks 8.
        # Need q_hi > q_lo but n_up < 8.
        # If 5 items are high (100) and 13 are low (0).
        stats = []
        for i in range(5): stats.append({'tonic': 100, 'peaks': 100})
        for i in range(13): stats.append({'tonic': 0, 'peaks': 10})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 8, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, f"Should skip (return True).\nOut:\n{out}")
        self.assertIn("SKIPPED (Small Groups)", out)

    def test_skip_no_finite_tonic(self):
        """All tonic values NaN/Inf should skip."""
        stats = []
        for i in range(10): stats.append({'tonic': float('nan'), 'peaks': 100})
        for i in range(10): stats.append({'tonic': float('inf'), 'peaks': 100})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should skip for no finite tonic")
        self.assertIn("SKIPPED (No Finite Tonic)", out)

    def test_nonfinite_tonics_do_not_contaminate_groups(self):
        """Non-finite tonics should be excluded from groups and means."""
        # 10 High (100, peaks=50), 10 Low (0, peaks=10).
        # Plus 5 Infinite tonics with peaks=999.
        # If included, peaks mean would skyrocket.
        # q_hi(0.75 of 20) = 100.
        stats = []
        for i in range(10): stats.append({'tonic': 100, 'peaks': 50})
        for i in range(10): stats.append({'tonic': 0, 'peaks': 10})
        for i in range(5): stats.append({'tonic': float('inf'), 'peaks': 999})
        for i in range(5): stats.append({'tonic': float('nan'), 'peaks': 999})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should pass")
        self.assertIn("Gating Check: PASS", out)
        # Groups should be N=10, N=10 (from the 20 finite)
        # Or similar (depending on quantile interpolation). 
        # With 20 finite points, q(0.75) is 100. 
        # High group (>= 100) -> 10 items.
        # Low group (<= 0.25 -> 0) -> 10 items.
        
        self.assertIn("High Tonic (N=10)", out)
        self.assertIn("Low Tonic (N=10)", out)
        
        # Means should be 50 and 10.
        self.assertIn("High Tonic=50.00", out)
        self.assertIn("Low Tonic=10.00", out)

    def test_insufficient_finite_data_skips_even_if_total_large(self):
        """Total chunks large, but finite chunks insufficient."""
        stats = []
        # 5 finite chunks (valid). Min chunks 4. Total need 8.
        # 50 infinite chunks.
        # Total len = 55 (>> 8).
        # Finite len = 5 (< 8). -> Skip Insufficient Data.
        for i in range(5): stats.append({'tonic': 100, 'peaks': 50})
        for i in range(50): stats.append({'tonic': float('inf'), 'peaks': 50})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should skip")
        self.assertIn("SKIPPED (Insufficient Data)", out)
        self.assertIn("Not enough audited chunks (5)", out)

    def test_string_tonics_are_handled(self):
        """String tonics should be coerced to float and processed correctly."""
        stats = []
        for i in range(10): stats.append({'tonic': '100', 'peaks': 50})
        for i in range(10): stats.append({'tonic': '0',   'peaks': 10})
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            res = evaluate_synth_gating(stats, 0.75, 4, 1.25)
            out = buf.getvalue()
            
        self.assertTrue(res, "Should pass")
        self.assertIn("Gating Check: PASS", out)
        self.assertIn("Groups: High Tonic (N=10), Low Tonic (N=10)", out)
        self.assertIn("Mean Peak Counts: High Tonic=50.00, Low Tonic=10.00", out)

if __name__ == '__main__':
    unittest.main()
