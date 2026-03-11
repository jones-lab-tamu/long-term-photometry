"""
Tests for the optimized tools/plot_tonic_48h.py using HDF5 cache data backend.

Covers:
1. Explicit --out writes exactly there
2. Default output path when --out is omitted
3. Multi-chunk synthetic tonic input
4. Success and basic structural correctness (file exists, non-zero size)
5. Missing cache hard-fails
6. Missing ROI hard-fails
7. Missing required dataset in cache hard-fails
8. Timing instrumentation verification
"""

import os
import sys
import unittest
import tempfile
import subprocess
import numpy as np
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _create_synthetic_tonic_cache(tonic_out_dir, n_chunks=4, n_pts=200, rois=['Region0']):
    """Create a synthetic tonic HDF5 cache to mock the Phase 1 trace output."""
    os.makedirs(tonic_out_dir, exist_ok=True)
    cache_path = os.path.join(tonic_out_dir, 'tonic_trace_cache.h5')
    
    with h5py.File(cache_path, 'w') as f:
        meta = f.create_group('meta')
        meta.attrs['mode'] = 'tonic'
        meta.attrs['schema_version'] = '1.0'
        
        dt_str = h5py.string_dtype(encoding='utf-8')
        meta.create_dataset('rois', data=np.array(rois, dtype=object), dtype=dt_str)
        meta.create_dataset('chunk_ids', data=np.array(list(range(n_chunks)), dtype=int))
        meta.create_dataset('source_files', data=np.array([f"f{i}.csv" for i in range(n_chunks)], dtype=object), dtype=dt_str)
        
        roi_group = f.create_group('roi')
        for roi in rois:
            grp_r = roi_group.create_group(roi)
            for i in range(n_chunks):
                grp_c = grp_r.create_group(f'chunk_{i}')
                t = np.linspace(0, 10, n_pts)
                grp_c.create_dataset('time_sec', data=t)
                grp_c.create_dataset('sig_raw', data=np.sin(t + i))
                grp_c.create_dataset('uv_raw', data=np.cos(t + i))
                grp_c.create_dataset('deltaF', data=np.sin(t + i) - np.cos(t + i))
                # extra arrays
                grp_c.create_dataset('dff', data=np.random.randn(n_pts))
                
    return cache_path


class TestPlotTonic48h(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.analysis_out = os.path.join(self.test_dir.name, '_analysis')
        self.tonic_out = os.path.join(self.analysis_out, 'tonic_out')
        self.cache_path = _create_synthetic_tonic_cache(self.tonic_out, n_chunks=4)

    def _run_script(self, extra_args=None):
        script = os.path.join(PROJECT_ROOT, 'tools', 'plot_tonic_48h.py')
        cmd = [sys.executable, script, '--analysis-out', self.tonic_out]
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
        default_path = os.path.join(self.tonic_out, 'tonic_qc', 'tonic_48h_overview_Region0.png')
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
        self.assertIn("step=cache_read", out)
        self.assertIn("step=assembly", out)
        self.assertIn("step=plotting", out)
        self.assertIn("step=figure_save", out)
        self.assertIn("PLOT_TIMING DONE script=plot_tonic_48h.py total_sec=", out)

    def test_missing_cache_fails(self):
        """Script hard-fails if cache does not exist."""
        os.remove(self.cache_path)
        result = self._run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CRITICAL: Cache file not found", result.stdout)

    def test_missing_roi_fails(self):
        """Script hard-fails if requested ROI is missing."""
        result = self._run_script(['--roi', 'MissingROI'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CRITICAL: Requested ROI 'MissingROI' not found in cache", result.stdout)

    def test_missing_dataset_fails(self):
        """Script hard-fails if chunk is missing a required dataset."""
        # Corrupt the cache
        with h5py.File(self.cache_path, 'r+') as f:
            del f['roi/Region0/chunk_1/deltaF']
            
        result = self._run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CRITICAL: Missing dataset deltaF", result.stdout)

    def test_regression_local_time_synthesis(self):
        """
        Verify that even when individual chunks reuse the same local time vector (0 to 10),
        the final assembled time vector is continuous and properly synthesized.
        """
        from tools.plot_tonic_48h import assemble_arrays
        from photometry_pipeline.io.hdf5_cache_reader import open_tonic_cache
        
        cache = open_tonic_cache(self.cache_path)
        continuous_time, sig_raw, uv_raw, deltaf_val = assemble_arrays(cache, 'Region0')
        cache.close()
        
        # 4 chunks of 200 pts each were created
        self.assertEqual(len(continuous_time), 800)
        self.assertEqual(len(continuous_time), len(sig_raw))
        
        # Should be strictly monotonically increasing despite the chunks all being 0 to 10 internally
        diffs = np.diff(continuous_time)
        self.assertTrue(np.all(diffs > 0), "Assembled time must be monotonically increasing.")
        
        # dt should be inferred from the first chunk's 0 to 10 over 200 pts
        # dt = 10.0 / (200 - 1) = ~0.05025
        dt_expected = 10.0 / 199.0
        self.assertTrue(np.allclose(diffs, dt_expected))

if __name__ == '__main__':
    unittest.main()
