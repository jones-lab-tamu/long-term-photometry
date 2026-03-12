import unittest
import os
import sys
import shutil
import tempfile
import subprocess
import numpy as np
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestPlotSessionGridMigration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.analysis_out = os.path.join(self.test_dir, "analysis")
        os.makedirs(self.analysis_out)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_session_grid_cache_only(self):
        """
        Verify plot_session_grid.py can run using only the HDF5 cache, 
        with NO 'traces' folder present.
        """
        cache_path = os.path.join(self.analysis_out, "phasic_trace_cache.h5")
        
        # Create a mock cache
        with h5py.File(cache_path, 'w') as f:
            meta = f.create_group('meta')
            meta.attrs['mode'] = 'phasic'
            meta.create_dataset('schema_version', data=np.array([1]))
            meta.create_dataset('rois', data=np.array([b'Region0']))
            meta.create_dataset('chunk_ids', data=np.array([101, 102]))
            meta.create_dataset('source_files', data=np.array([b'file1.raw', b'file2.raw'], dtype=object))
            
            roi_grp = f.create_group('roi/Region0')
            for cid in [101, 102]:
                cgrp = roi_grp.create_group(f"chunk_{cid}")
                t = np.linspace(0, 600, 1000)
                y_sig = np.sin(t * 0.1) + 5.0
                y_uv = np.cos(t * 0.1) + 4.0
                cgrp.create_dataset('time_sec', data=t)
                cgrp.create_dataset('dff', data=y_sig - y_uv)
                cgrp.create_dataset('sig_raw', data=y_sig)
                cgrp.create_dataset('uv_raw', data=y_uv)

        # Run the tool
        cmd = [
            sys.executable, "tools/plot_session_grid.py",
            "--analysis-out", self.analysis_out,
            "--roi", "Region0",
            "--sessions-per-hour", "1"
        ]
        
        # Ensure no 'traces' folder exists
        self.assertFalse(os.path.exists(os.path.join(self.analysis_out, "traces")))
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"Plotter failed: {result.stderr}")
        
        # Verify output
        out_path = os.path.join(self.analysis_out, "session_qc", "day_000_raw_iso_Region0.png")
        self.assertTrue(os.path.exists(out_path), f"Session grid PNG was not generated at {out_path}")
        print("Successfully verified plot_session_grid.py cache-only operation.")

if __name__ == "__main__":
    unittest.main()
