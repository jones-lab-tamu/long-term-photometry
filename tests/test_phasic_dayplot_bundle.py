import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.core.feature_extraction import extract_features

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

    def create_synthetic_phasic_cache(
        self,
        cid=0,
        include_dff=False,
        include_sig=True,
        include_delta_f=False,
        dff_data=None,
        delta_f_data=None,
        sig_data=None,
        uv_data=None,
        time_data=None,
    ):
        import h5py
        cache_path = os.path.join(self.analysis_out, 'phasic_trace_cache.h5')
        
        # Open in append mode so we can add multiple chunks if tests ever needed it
        with h5py.File(cache_path, 'a') as f:
            if 'meta' not in f:
                meta = f.create_group('meta')
                meta.attrs['mode'] = 'phasic'
                meta.attrs['schema_version'] = '1.0'
                dt_str = h5py.string_dtype(encoding='utf-8')
                meta.create_dataset('rois', data=np.array(['Region0'], dtype=object), dtype=dt_str)
                meta.create_dataset('chunk_ids', data=np.array([cid], dtype=int), maxshape=(None,))
            else:
                meta = f['meta']
                cids = list(meta['chunk_ids'][()])
                if cid not in cids:
                    cids.append(cid)
                    del meta['chunk_ids']
                    meta.create_dataset('chunk_ids', data=np.array(cids, dtype=int))
                
            if 'roi' not in f:
                f.create_group('roi')
            roi_grp = f.require_group('roi/Region0')
            c_grp = roi_grp.require_group(f'chunk_{cid}')
            
            t = np.arange(0, 600, 0.1) if time_data is None else np.asarray(time_data)
            
            if 'time_sec' in c_grp: del c_grp['time_sec']
            c_grp.create_dataset('time_sec', data=t)
            
            if include_sig:
                if 'sig_raw' in c_grp: del c_grp['sig_raw']
                if 'uv_raw' in c_grp: del c_grp['uv_raw']
                c_grp.create_dataset('sig_raw', data=np.sin(t) if sig_data is None else np.asarray(sig_data))
                c_grp.create_dataset('uv_raw', data=np.cos(t) if uv_data is None else np.asarray(uv_data))
                
            if include_dff:
                if 'dff' in c_grp: del c_grp['dff']
                # The prompt tests with some high peaks. Let's make sure test_full_dff_mode can overwrite this.
                c_grp.create_dataset('dff', data=(np.sin(t) - np.cos(t)) if dff_data is None else np.asarray(dff_data))

            if include_delta_f:
                if 'delta_f' in c_grp: del c_grp['delta_f']
                c_grp.create_dataset('delta_f', data=np.sin(t) if delta_f_data is None else np.asarray(delta_f_data))

    def test_sig_iso_only_mode_no_dff(self):
        # A. sig/iso-only mode: no features.csv, no dff column -> success
        self.create_synthetic_chunk(cid=0, include_dff=False)
        self.create_synthetic_phasic_cache(cid=0, include_dff=False)
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
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
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
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
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
        
        # Minimal trace CSV required for discovery logic
        t = np.arange(0, 600, 0.1)
        df_dff = pd.DataFrame({
            'time_sec': t,
            'Region0_sig_raw': np.zeros_like(t),
            'Region0_uv_raw': np.zeros_like(t),
            'Region0_dff': np.zeros_like(t)
        })
        df_dff.to_csv(os.path.join(self.traces_dir, 'chunk_0.csv'), index=False)
        
        # Add 2 peaks into the cache instead (but zeros otherwise)
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
        import h5py
        cache_path = os.path.join(self.analysis_out, 'phasic_trace_cache.h5')
        with h5py.File(cache_path, 'a') as f:
            dff_data = np.zeros_like(t)
            dff_data[100] = 100.0
            dff_data[200] = 100.0
            del f['roi/Region0/chunk_0/dff']
            f['roi/Region0/chunk_0'].create_dataset('dff', data=dff_data)
        
        self.create_features_csv(peak_count=2)
        
        with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
            bundle.main()
            
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dFF_day_000.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_stacked_day_000.png')))


    def test_missing_cache_fails_with_csvs_present(self):
        # Even if chunk_0.csv exists, missing HDF5 cache should hard fail
        self.create_synthetic_chunk(cid=0, include_dff=True)
        # We explicitly DO NOT create the phasic cache
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
            with self.assertRaises(SystemExit) as cm:
                bundle.main()
            self.assertEqual(cm.exception.code, 1)

    @patch('tools.plot_phasic_dayplot_bundle.discover_chunks')
    def test_cache_pull_fields_all_modes(self, mock_discover_chunks):
        # We replace the old CSV usecols test with one that mocks discovery and ensures the cache reader requests exact fields
        import numpy as np
        
        # We physically create the chunk to avoid mocking read_csv at all
        self.create_synthetic_chunk(cid=0, include_dff=True)
        self.create_synthetic_phasic_cache(cid=0, include_dff=True)
        self.create_features_csv(peak_count=0) # Must exist for dFF Grid mode
        
        mock_discover_chunks.return_value = [(0, os.path.join(self.traces_dir, 'chunk_0.csv'))]
        
        cases = [
            ("sig_iso_only", ['--write-sig-iso-grid', '--no-write-dff-grid', '--no-write-stacked'], 
             ['time_sec', 'sig_raw', 'uv_raw']),
             
            ("stacked_only", ['--no-write-sig-iso-grid', '--no-write-dff-grid', '--write-stacked'], 
             ['time_sec', 'dff']),
             
            ("dff_grid_only", ['--no-write-sig-iso-grid', '--write-dff-grid', '--no-write-stacked'], 
             ['time_sec', 'dff']),
             
            ("full_mode", ['--write-sig-iso-grid', '--write-dff-grid', '--write-stacked'], 
             ['time_sec', 'sig_raw', 'uv_raw', 'dff'])
        ]
        
        with patch('tools.plot_phasic_dayplot_bundle.resolve_roi', return_value='Region0'):
            for name, flags, expected_fields in cases:
                with patch('tools.plot_phasic_dayplot_bundle.load_cache_chunk_fields') as mock_load:
                    # Provide dummy array tuples back matching requested length
                    mock_load.return_value = tuple([np.array([1, 2])] * len(expected_fields))
                    
                    test_args = [
                        'plot_phasic_dayplot_bundle.py', '--analysis-out', self.analysis_out,
                        '--roi', 'Region0', '--output-dir', self.output_dir, '--sessions-per-hour', '1',
                        '--session-duration-s', '0.1'
                    ] + flags
                    
                    with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
                        bundle.main()
                        
                    self.assertTrue(mock_load.called, f"{name}: load_cache_chunk_fields not called")
                    
                    # Check exact fields requested are strictly equal
                    fields_requested = mock_load.call_args[0][3]
                    self.assertEqual(fields_requested, expected_fields, f"{name}: requested fields mismatch")

    def test_dff_grid_verification_uses_configured_event_signal_and_ignores_display_smoothing(self):
        t = np.arange(0, 600, 0.1)
        fs_hz = 10.0
        rng = np.random.default_rng(7)

        # Detection trace (delta_f): few broad events.
        delta_f = 0.02 * np.sin(0.07 * t)
        for center in (90.0, 260.0, 430.0):
            delta_f += 2.2 * np.exp(-0.5 * ((t - center) / 0.9) ** 2)

        # Display trace (dff): many oscillatory peaks that should NOT drive authoritative count.
        dff = 0.9 * np.sin(2 * np.pi * 1.4 * t) + 0.4 * np.sin(2 * np.pi * 3.2 * t)
        dff += 0.05 * rng.standard_normal(len(t))

        with open(os.path.join(self.analysis_out, 'config_used.yaml'), 'w', encoding='utf-8') as f:
            f.write("target_fs_hz: 10.0\n")
            f.write("lowpass_hz: 1.0\n")
            f.write("filter_order: 3\n")
            f.write("event_signal: 'delta_f'\n")
            f.write("peak_threshold_method: 'mean_std'\n")
            f.write("peak_threshold_k: 1.5\n")
            f.write("peak_min_distance_sec: 0.5\n")
            f.write("peak_pre_filter: 'lowpass'\n")

        cfg = Config.from_yaml(os.path.join(self.analysis_out, 'config_used.yaml'))
        chunk = Chunk(
            chunk_id=0,
            source_file='chunk_0.csv',
            format='rwd',
            time_sec=t,
            uv_raw=np.zeros((len(t), 1), dtype=float),
            sig_raw=np.zeros((len(t), 1), dtype=float),
            fs_hz=fs_hz,
            channel_names=['Region0'],
            dff=dff.reshape(-1, 1),
            delta_f=delta_f.reshape(-1, 1),
        )
        feat_df = extract_features(chunk, cfg)
        expected_count = int(feat_df.iloc[0]['peak_count'])
        self.assertGreater(expected_count, 0)

        self.create_features_csv(peak_count=expected_count)
        self.create_synthetic_phasic_cache(
            cid=0,
            include_dff=True,
            include_sig=True,
            include_delta_f=True,
            dff_data=dff,
            delta_f_data=delta_f,
            sig_data=np.zeros_like(t),
            uv_data=np.zeros_like(t),
            time_data=t,
        )

        for smooth in ('0.2', '20.0'):
            test_args = [
                'plot_phasic_dayplot_bundle.py',
                '--analysis-out', self.analysis_out,
                '--roi', 'Region0',
                '--output-dir', self.output_dir,
                '--sessions-per-hour', '1',
                '--write-dff-grid',
                '--no-write-sig-iso-grid',
                '--write-stacked',
                '--smooth-window-s', smooth,
            ]
            with patch('tools.plot_phasic_dayplot_bundle.sys.argv', test_args):
                bundle.main()

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_dFF_day_000.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'phasic_stacked_day_000.png')))

if __name__ == '__main__':
    unittest.main()
