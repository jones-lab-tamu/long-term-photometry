import unittest
import os
import sys
import shutil
import tempfile
import subprocess
import numpy as np
import h5py
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestPlotPhasicQCGridMigration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.analysis_out = os.path.join(self.test_dir, "analysis")
        os.makedirs(self.analysis_out)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_qc_grid_cache_only(self):
        """
        Verify plot_phasic_qc_grid.py can run using ONLY the HDF5 cache and features.csv, 
        with NO 'traces' folder present.
        """
        # 1. Create a mock config
        config_path = os.path.join(self.analysis_out, "config_used.yaml")
        config = {
            'peak_threshold_method': 'absolute',
            'peak_threshold_abs': 0.5,
            'peak_min_distance_sec': 1.0,
            'target_fs_hz': 100.0
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        # 2. Create a mock features.csv
        feats_dir = os.path.join(self.analysis_out, "features")
        os.makedirs(feats_dir)
        feats_path = os.path.join(feats_dir, "features.csv")
        # We need 1 peak in each chunk to match our mock data
        df_feat = pd.DataFrame({
            'chunk_id': [101, 102],
            'roi': ['Region0', 'Region0'],
            'peak_count': [1, 1],
            'source_file': ['file1.raw', 'file2.raw']
        })
        df_feat.to_csv(feats_path, index=False)
        
        # 3. Create a mock cache
        cache_path = os.path.join(self.analysis_out, "phasic_trace_cache.h5")
        with h5py.File(cache_path, 'w') as f:
            meta = f.create_group('meta')
            meta.attrs['mode'] = 'phasic'
            meta.attrs['schema_version'] = '1.0'
            meta.create_dataset('rois', data=np.array([b'Region0']))
            meta.create_dataset('chunk_ids', data=np.array([101, 102]))
            meta.create_dataset('source_files', data=np.array([b'file1.raw', b'file2.raw'], dtype=object))
            
            roi_grp = f.create_group('roi/Region0')
            for cid in [101, 102]:
                cgrp = roi_grp.create_group(f"chunk_{cid}")
                t = np.linspace(0, 10, 1000)
                # One clear peak at t=5.0
                y = np.zeros_like(t)
                y[500] = 1.0 
                cgrp.create_dataset('time_sec', data=t)
                cgrp.create_dataset('dff', data=y)
                cgrp.create_dataset('sig_raw', data=y + 5.0)
                cgrp.create_dataset('uv_raw', data=np.zeros_like(y) + 4.0)

        # 4. Run the tool
        cmd = [
            sys.executable, "tools/plot_phasic_qc_grid.py",
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
        
        # 5. Verify output
        out_path = os.path.join(self.analysis_out, "phasic_qc", "day_000.png")
        self.assertTrue(os.path.exists(out_path), f"QC grid PNG was not generated at {out_path}")
        print("Successfully verified plot_phasic_qc_grid.py cache-only operation.")

if __name__ == "__main__":
    unittest.main()
