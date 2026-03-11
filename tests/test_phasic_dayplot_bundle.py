import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tools.plot_phasic_dayplot_bundle as bundle

class TestPhasicDayplotBundle(unittest.TestCase):
    
    @patch('tools.plot_phasic_dayplot_bundle.sys.argv', ['plot_phasic_dayplot_bundle.py', '--analysis-out', '/fake', '--roi', 'Region0', '--output-dir', '/fake_out', '--sessions-per-hour', '2'])
    def test_parse_args_defaults(self):
        args = bundle.parse_args()
        self.assertEqual(args.analysis_out, '/fake')
        self.assertEqual(args.roi, 'Region0')
        self.assertEqual(args.output_dir, '/fake_out')
        self.assertEqual(args.sessions_per_hour, 2)
        self.assertTrue(args.write_dff_grid)
        self.assertTrue(args.write_sig_iso_grid)
        self.assertTrue(args.write_stacked)

    @patch('tools.plot_phasic_dayplot_bundle.sys.argv', ['plot_phasic_dayplot_bundle.py', '--analysis-out', '/f', '--roi', 'R0', '--output-dir', '/o', '--sessions-per-hour', '1', '--no-write-stacked'])
    def test_parse_args_overrides(self):
        args = bundle.parse_args()
        self.assertTrue(args.write_dff_grid)
        self.assertTrue(args.write_sig_iso_grid)
        self.assertFalse(args.write_stacked)

    def test_check_monotonicity(self):
        self.assertTrue(bundle.check_monotonicity([0, 1, 2, 3]))
        self.assertFalse(bundle.check_monotonicity([0, 2, 1, 3]))

    def test_check_continuity(self):
        self.assertTrue(bundle.check_continuity([0, 1, 2, 3], 1.0))
        self.assertFalse(bundle.check_continuity([0, 1, 4, 5], 1.0))

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.analysis_out = os.path.join(self.test_dir.name, '_analysis')
        self.traces_dir = os.path.join(self.analysis_out, 'traces')
        self.features_dir = os.path.join(self.analysis_out, 'features')
        self.output_dir = os.path.join(self.test_dir.name, 'day_plots')
        
        os.makedirs(self.traces_dir)
        os.makedirs(self.output_dir)
        
        config_path = os.path.join(self.analysis_out, 'config_used.yaml')
        with open(config_path, 'w') as f:
            f.write("target_fs_hz: 10.0\n")
            f.write("peak_threshold_method: 'absolute'\n")
            f.write("peak_threshold_abs: 0.5\n")
            f.write("peak_min_distance_sec: 1.0\n")

    def create_synthetic_chunk(self, cid=0, include_dff=False, include_sig=True):
        t = np.arange(0, 600, 0.1)
        data = {'time_sec': t}
        if include_sig:
            data['Region0_sig_raw'] = np.sin(t)
            data['Region0_uv_raw'] = np.cos(t)
        if include_dff:
            data['Region0_dff'] = np.sin(t) - np.cos(t)
            
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.traces_dir, f"chunk_{cid}.csv")
        df.to_csv(csv_path, index=False)

    def create_features_csv(self, peak_count=0):
        os.makedirs(self.features_dir, exist_ok=True)
        df = pd.DataFrame([{
            'chunk_id': 0, 'roi': 'Region0', 'peak_count': peak_count, 'auc': 0.0
        }])
        df.to_csv(os.path.join(self.features_dir, 'features.csv'), index=False)

    def test_sig_iso_only_mode_no_dff(self):
        # A. sig/iso-only mode: no features.csv, no dff column -> success
        self.create_synthetic_chunk(cid=0, include_dff=False)
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--no-write-dff-grid',
            '--no-write-stacked'
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main() # Should not raise
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_sig_iso_day_000.png')))

    def test_stacked_only_mode_no_features(self):
        # B. stacked-only mode: no features.csv, dff necessary but feature shouldn't be read 
        self.create_synthetic_chunk(cid=0, include_dff=True)
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--no-write-dff-grid',
            '--no-write-sig-iso-grid',
            '--write-stacked'
        ]
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main() # Should not raise
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_stacked_day_000.png')))

    def test_dff_grid_mode_requires_features(self):
        # C. dFF-grid mode requires features.csv
        self.create_synthetic_chunk(cid=0, include_dff=True)
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--write-dff-grid',
            '--no-write-sig-iso-grid',
            '--no-write-stacked'
        ]
        
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                bundle.main()
            self.assertEqual(cm.exception.code, 1)

    def test_full_dff_mode(self):
        # D. full dFF mode passes when feature and dff traces match
        test_args = [
            'plot_phasic_dayplot_bundle.py',
            '--analysis-out', self.analysis_out,
            '--roi', 'Region0',
            '--output-dir', self.output_dir,
            '--sessions-per-hour', '1',
            '--write-dff-grid',
            '--write-stacked'
        ]
        
        t = np.arange(0, 600, 0.1)
        df_dff = pd.DataFrame({
            'time_sec': t,
            'Region0_sig_raw': np.zeros_like(t),
            'Region0_uv_raw': np.zeros_like(t),
            'Region0_dff': np.zeros_like(t)
        })
        # Add 2 peaks
        df_dff.loc[100, 'Region0_dff'] = 100.0
        df_dff.loc[200, 'Region0_dff'] = 100.0
        df_dff.to_csv(os.path.join(self.traces_dir, 'chunk_0.csv'), index=False)
        
        self.create_features_csv(peak_count=2)
        
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()
            
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dFF_day_000.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_stacked_day_000.png')))

    @patch('tools.plot_phasic_dayplot_bundle.pd.read_csv')
    @patch('tools.plot_phasic_dayplot_bundle.plt.savefig')
    def test_usecols_all_modes(self, mock_savefig, mock_read_csv):
        df_header = pd.DataFrame(columns=['time_sec', 'Region0_sig_raw', 'Region0_uv_raw', 'Region0_dff', 'ignored_col'])
        df_data = pd.DataFrame({
            'time_sec': [0.1, 0.2], 'Region0_sig_raw': [1, 2], 'Region0_uv_raw': [1, 2], 'Region0_dff': [0.5, 0.6]
        })
        
        def side_effect(*args, **kwargs):
            filepath = str(args[0])
            if 'features.csv' in filepath:
                return pd.DataFrame([{'chunk_id': 0, 'roi': 'Region0', 'peak_count': 0, 'auc': 0.0}])
            if kwargs.get('nrows') == 1: return df_header
            return df_data
            
        self.create_synthetic_chunk(cid=0, include_dff=True)
        self.create_features_csv(peak_count=0) # Must exist for dFF Grid mode
        
        cases = [
            ("sig_iso_only", ['--write-sig-iso-grid', '--no-write-dff-grid', '--no-write-stacked'], 
             ['time_sec', 'Region0_sig_raw', 'Region0_uv_raw'], ['Region0_dff', 'ignored_col']),
             
            ("stacked_only", ['--no-write-sig-iso-grid', '--no-write-dff-grid', '--write-stacked'], 
             ['time_sec', 'Region0_dff'], ['Region0_sig_raw', 'Region0_uv_raw', 'ignored_col']),
             
            ("dff_grid_only", ['--no-write-sig-iso-grid', '--write-dff-grid', '--no-write-stacked'], 
             ['time_sec', 'Region0_dff'], ['Region0_sig_raw', 'Region0_uv_raw', 'ignored_col']),
             
            ("full_mode", ['--write-sig-iso-grid', '--write-dff-grid', '--write-stacked'], 
             ['time_sec', 'Region0_sig_raw', 'Region0_uv_raw', 'Region0_dff'], ['ignored_col'])
        ]
        
        for name, flags, expected_in, expected_out in cases:
            mock_read_csv.reset_mock()
            mock_read_csv.side_effect = side_effect
            
            test_args = [
                'plot_phasic_dayplot_bundle.py', '--analysis-out', self.analysis_out,
                '--roi', 'Region0', '--output-dir', self.output_dir, '--sessions-per-hour', '1'
            ]
            test_args.extend(flags)
            test_args.extend(['--session-duration-s', '0.1'])
            
            with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
                bundle.main()
                
            chunk_calls = [c for c in mock_read_csv.call_args_list if 'chunk_0.csv' in str(c.args[0]) and c.kwargs.get('nrows') != 1]
            self.assertGreater(len(chunk_calls), 0, f"{name}: No chunk CSV read calls found")
            usecols = chunk_calls[0].kwargs.get('usecols')
            
            self.assertIsNotNone(usecols, f"{name}: usecols should not be None")
            for col in expected_in:
                self.assertIn(col, usecols, f"{name}: missing {col} in usecols")
            for col in expected_out:
                self.assertNotIn(col, usecols, f"{name}: {col} should not be in usecols")

if __name__ == '__main__':
    unittest.main()
