"""
Tests for the optimized tools/plot_tonic_48h.py.

Covers:
1. Explicit --out writes exactly there
2. Default output path when --out is omitted
3. Multi-chunk synthetic tonic input
4. Success and basic structural correctness (file exists, non-zero size)
"""

import os
import sys
import unittest
import tempfile
import subprocess
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _create_synthetic_tonic_chunks(traces_dir, n_chunks=4, n_pts=200, roi='Region0'):
    """Create n_chunks worth of synthetic tonic trace CSVs."""
    os.makedirs(traces_dir, exist_ok=True)
    for i in range(n_chunks):
        t = np.linspace(0, 10, n_pts)
        df = pd.DataFrame({
            'time_sec': t,
            f'{roi}_sig_raw': np.sin(t + i),
            f'{roi}_uv_raw': np.cos(t + i),
            f'{roi}_deltaF': np.sin(t + i) - np.cos(t + i),
            # Extra columns that should NOT be loaded by the optimized path
            f'{roi}_dff': np.random.randn(n_pts),
            'extra_col_1': np.random.randn(n_pts),
            'extra_col_2': np.random.randn(n_pts),
        })
        df.to_csv(os.path.join(traces_dir, f'chunk_{i}.csv'), index=False)


class TestPlotTonic48h(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.analysis_out = os.path.join(self.test_dir.name, '_analysis')
        self.traces_dir = os.path.join(self.analysis_out, 'traces')
        _create_synthetic_tonic_chunks(self.traces_dir, n_chunks=4)

    def _run_script(self, extra_args=None):
        script = os.path.join(PROJECT_ROOT, 'tools', 'plot_tonic_48h.py')
        cmd = [sys.executable, script, '--analysis-out', self.analysis_out]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        return result

    def test_explicit_out_path(self):
        """With --out, the script writes exactly to that path."""
        out_path = os.path.join(self.test_dir.name, 'custom', 'my_plot.png')
        result = self._run_script(['--out', out_path])
        self.assertEqual(result.returncode, 0, f"Script failed:\n{result.stderr}")
        self.assertTrue(os.path.exists(out_path), f"Expected output at {out_path}")
        self.assertGreater(os.path.getsize(out_path), 0)

    def test_default_out_path(self):
        """Without --out, the script writes to tonic_qc/ inside analysis-out."""
        result = self._run_script()
        self.assertEqual(result.returncode, 0, f"Script failed:\n{result.stderr}")
        default_path = os.path.join(self.analysis_out, 'tonic_qc', 'tonic_48h_overview_Region0.png')
        self.assertTrue(os.path.exists(default_path), f"Expected default output at {default_path}")
        self.assertGreater(os.path.getsize(default_path), 0)

    def test_multi_chunk_stitching(self):
        """Multi-chunk input produces a valid output and exercises stitching."""
        out_path = os.path.join(self.test_dir.name, 'stitched.png')
        result = self._run_script(['--out', out_path, '--roi', 'Region0'])
        self.assertEqual(result.returncode, 0, f"Script failed:\n{result.stderr}")
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

    def test_timing_instrumentation(self):
        """Verify that the expected timing lines are present in stdout."""
        result = self._run_script()
        self.assertEqual(result.returncode, 0, f"Script failed:\n{result.stderr}")
        out = result.stdout
        self.assertIn("PLOT_TIMING START script=plot_tonic_48h.py", out)
        self.assertIn("step=discovery", out)
        self.assertIn("step=csv_read", out)
        self.assertIn("step=assembly", out)
        self.assertIn("step=plotting", out)
        self.assertIn("step=figure_save", out)
        self.assertIn("PLOT_TIMING DONE script=plot_tonic_48h.py total_sec=", out)


if __name__ == '__main__':
    unittest.main()
