import os
import tempfile
import unittest
import pytest
import h5py
import numpy as np

from photometry_pipeline.io.hdf5_cache_reader import (
    CacheReadError,
    open_tonic_cache,
    open_phasic_cache,
    list_cache_rois,
    list_cache_chunk_ids,
    resolve_cache_roi,
    load_cache_chunk_fields,
    iter_cache_chunks_for_roi
)

class TestHDF5CacheReader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.valid_tonic_path = os.path.join(self.temp_dir.name, "valid_tonic.h5")
        self.valid_phasic_path = os.path.join(self.temp_dir.name, "valid_phasic.h5")
        
        # Valid Tonic Cache
        with h5py.File(self.valid_tonic_path, 'w') as f:
            meta = f.create_group('meta')
            meta.attrs['mode'] = 'tonic'
            meta.create_dataset('schema_version', data=np.array([1], dtype=int))
            dt_str = h5py.string_dtype(encoding='utf-8')
            meta.create_dataset('rois', data=np.array(['Region0', 'Region1'], dtype=object), dtype=dt_str)
            meta.create_dataset('chunk_ids', data=np.array([7, 2, 9], dtype=int))
            
            for roi in ['Region0', 'Region1']:
                roi_grp = f.create_group(f'roi/{roi}')
                for cid in [7, 2, 9]:
                    c_grp = roi_grp.create_group(f'chunk_{cid}')
                    c_grp.create_dataset('time_sec', data=np.arange(10, dtype=np.float64))
                    c_grp.create_dataset('sig_raw', data=np.ones(10, dtype=np.float64))
                    c_grp.create_dataset('uv_raw', data=np.zeros(10, dtype=np.float64))
                    c_grp.create_dataset('deltaF', data=np.full(10, 2.0, dtype=np.float64))

        # Valid Phasic Cache
        with h5py.File(self.valid_phasic_path, 'w') as f:
            meta = f.create_group('meta')
            meta.attrs['mode'] = 'phasic'
            meta.attrs['schema_version'] = '1.0'
            dt_str = h5py.string_dtype(encoding='utf-8')
            meta.create_dataset('rois', data=np.array(['Region0'], dtype=object), dtype=dt_str)
            meta.create_dataset('chunk_ids', data=np.array([0, 1], dtype=int))
            
            roi_grp = f.create_group('roi/Region0')
            for cid in [0, 1]:
                c_grp = roi_grp.create_group(f'chunk_{cid}')
                c_grp.create_dataset('time_sec', data=np.arange(5, dtype=np.float64))
                c_grp.create_dataset('sig_raw', data=np.ones(5, dtype=np.float64))
                c_grp.create_dataset('uv_raw', data=np.zeros(5, dtype=np.float64))
                c_grp.create_dataset('dff', data=np.full(5, 5.0, dtype=np.float64))
                
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_open_tonic_cache_accepts_valid_cache(self):
        """1. test_open_tonic_cache_accepts_valid_cache"""
        f = open_tonic_cache(self.valid_tonic_path)
        self.assertIsInstance(f, h5py.File)
        self.assertEqual(f['meta'].attrs['mode'], 'tonic')
        f.close()
        
    def test_open_phasic_cache_accepts_valid_cache(self):
        """2. test_open_phasic_cache_accepts_valid_cache"""
        f = open_phasic_cache(self.valid_phasic_path)
        self.assertIsInstance(f, h5py.File)
        self.assertEqual(f['meta'].attrs['mode'], 'phasic')
        f.close()
        
    def test_open_cache_rejects_wrong_mode(self):
        """3. test_open_cache_rejects_wrong_mode"""
        with pytest.raises(CacheReadError):
            open_tonic_cache(self.valid_phasic_path)
        
        with pytest.raises(CacheReadError):
            open_phasic_cache(self.valid_tonic_path)
        
    def test_schema_version_dataset_is_accepted(self):
        """4. test_schema_version_dataset_is_accepted"""
        # self.valid_tonic_path uses dataset for schema_version
        f = open_tonic_cache(self.valid_tonic_path)
        f.close()
        
    def test_schema_version_attr_is_accepted(self):
        """5. test_schema_version_attr_is_accepted"""
        # self.valid_phasic_path uses attr for schema_version
        f = open_phasic_cache(self.valid_phasic_path)
        f.close()

    def test_unsupported_schema_version_fails(self):
        """6. test_unsupported_schema_version_fails"""
        bad_path = os.path.join(self.temp_dir.name, "bad_schema.h5")
        with h5py.File(bad_path, 'w') as f:
            meta = f.create_group('meta')
            meta.attrs['mode'] = 'tonic'
            meta.attrs['schema_version'] = '2.0'
            
        with pytest.raises(CacheReadError):
            open_tonic_cache(bad_path)

    def test_list_cache_rois_returns_strings(self):
        """7. test_list_cache_rois_returns_strings"""
        with open_tonic_cache(self.valid_tonic_path) as f:
            rois = list_cache_rois(f)
            self.assertEqual(rois, ['Region0', 'Region1'])
            self.assertTrue(all(isinstance(r, str) for r in rois))

    def test_list_cache_chunk_ids_preserves_order(self):
        """8. test_list_cache_chunk_ids_preserves_order"""
        with open_tonic_cache(self.valid_tonic_path) as f:
            chunk_ids = list_cache_chunk_ids(f)
            self.assertEqual(chunk_ids, [7, 2, 9])
            self.assertTrue(all(isinstance(c, int) for c in chunk_ids))
            
    def test_resolve_cache_roi_auto_selects_first(self):
        """9. test_resolve_cache_roi_auto_selects_first"""
        with open_tonic_cache(self.valid_tonic_path) as f:
            roi = resolve_cache_roi(f, None)
            self.assertEqual(roi, 'Region0')

    def test_resolve_cache_roi_missing_requested_roi_fails(self):
        """10. test_resolve_cache_roi_missing_requested_roi_fails"""
        with open_tonic_cache(self.valid_tonic_path) as f:
            with pytest.raises(CacheReadError):
                resolve_cache_roi(f, 'MissingRegion')
            
    def test_load_cache_chunk_fields_returns_requested_fields_only(self):
        """11. test_load_cache_chunk_fields_returns_requested_fields_only"""
        with open_tonic_cache(self.valid_tonic_path) as f:
            res = load_cache_chunk_fields(f, 'Region0', 2, ['sig_raw', 'time_sec'])
            self.assertEqual(len(res), 2)
            self.assertTrue(np.array_equal(res[0], np.ones(10)))
            self.assertTrue(np.array_equal(res[1], np.arange(10)))

    def test_load_cache_chunk_fields_missing_field_fails(self):
        """12. test_load_cache_chunk_fields_missing_field_fails"""
        with open_tonic_cache(self.valid_tonic_path) as f:
            with pytest.raises(CacheReadError):
                load_cache_chunk_fields(f, 'Region0', 2, ['time_sec', 'missing_field'])

    def test_open_cache_missing_file_raises_exception_not_system_exit(self):
        with pytest.raises(CacheReadError) as excinfo:
            open_tonic_cache(os.path.join(self.temp_dir.name, "missing_cache.h5"))
        assert "Cache file not found" in str(excinfo.value)

    def test_iter_cache_chunks_for_roi_preserves_chunk_order(self):
        """13. test_iter_cache_chunks_for_roi_preserves_chunk_order"""
        with open_tonic_cache(self.valid_tonic_path) as f:
            generator = iter_cache_chunks_for_roi(f, 'Region1', ['time_sec'])
            chunks = list(generator)
            self.assertEqual(len(chunks), 3)
            # Verify the order of processing aligns with the chunk IDs
            # But the content was same. Let's make the content distinct to test true ordering.
            
        # Create a new cache purely to test the data contents corresponding to 7, 2, 9 are correct.
        order_path = os.path.join(self.temp_dir.name, "order_test.h5")
        with h5py.File(order_path, 'w') as f:
            meta = f.create_group('meta')
            meta.attrs['mode'] = 'tonic'
            meta.attrs['schema_version'] = '1.0'
            dt_str = h5py.string_dtype(encoding='utf-8')
            meta.create_dataset('rois', data=np.array(['R1'], dtype=object), dtype=dt_str)
            meta.create_dataset('chunk_ids', data=np.array([7, 2, 9], dtype=int))
            roi_grp = f.create_group('roi/R1')
            
            roi_grp.create_group('chunk_7').create_dataset('t', data=np.array([7.0]))
            roi_grp.create_group('chunk_2').create_dataset('t', data=np.array([2.0]))
            roi_grp.create_group('chunk_9').create_dataset('t', data=np.array([9.0]))
            
        with open_tonic_cache(order_path) as f:
            generator = iter_cache_chunks_for_roi(f, 'R1', ['t'])
            chunks = list(generator)
            self.assertEqual(chunks[0][0][0], 7.0)
            self.assertEqual(chunks[1][0][0], 2.0)
            self.assertEqual(chunks[2][0][0], 9.0)

    def test_iter_cache_chunks_for_roi_supports_phasic_field_selection(self):
        """14. test_iter_cache_chunks_for_roi_supports_phasic_field_selection"""
        with open_phasic_cache(self.valid_phasic_path) as f:
            generator = iter_cache_chunks_for_roi(f, 'Region0', ['time_sec', 'dff'])
            chunks = list(generator)
            self.assertEqual(len(chunks), 2)
            self.assertTrue(np.array_equal(chunks[0][0], np.arange(5))) # time_sec
            self.assertTrue(np.array_equal(chunks[0][1], np.full(5, 5.0))) # dff

if __name__ == '__main__':
    unittest.main()
