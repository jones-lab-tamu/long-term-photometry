
import unittest
import os
import shutil
import tempfile
from photometry_pipeline.io.adapters import discover_rwd_chunks

class TestRWDDiscovery(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def touch(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write("header\n data")

    def test_standard_structure(self):
        """1) Standard RWD structure returns ordered fluorescence.csv paths"""
        # Create timestamps out of order to test sorting
        ts1 = "2025_11_11-12_55_16"
        ts2 = "2025_11_11-13_25_17" 
        ts3 = "2025_12_01-10_55_17"
        
        # Create directories
        for ts in [ts2, ts3, ts1]: # Created out of order
            p = os.path.join(self.test_dir, ts, "fluorescence.csv")
            self.touch(p)
            
        chunks = discover_rwd_chunks(self.test_dir)
        
        self.assertEqual(len(chunks), 3)
        
        # Expected order (lexicographical by folder name)
        expected_folders = [ts1, ts2, ts3]
        for i, chunk_path in enumerate(chunks):
            # Check suffix
            self.assertTrue(chunk_path.endswith(os.path.join(expected_folders[i], "fluorescence.csv")))
            # Check full path
            expected = os.path.join(self.test_dir, expected_folders[i], "fluorescence.csv")
            # Normalize for OS
            self.assertEqual(os.path.normpath(chunk_path), os.path.normpath(expected))

    def test_ignore_artifacts_and_non_chunks(self):
        """2) Ignore non-chunk folders and ignore irrelevant files"""
        ts = "2025_11_11-12_55_16"
        chunk_dir = os.path.join(self.test_dir, ts)
        
        # Valid file
        self.touch(os.path.join(chunk_dir, "fluorescence.csv"))
        
        # Ignored files in valid chunk dir
        self.touch(os.path.join(chunk_dir, "outputs.csv"))
        self.touch(os.path.join(chunk_dir, "events.csv"))
        self.touch(os.path.join(chunk_dir, "fluorescence-unaligned.csv"))
        
        # Ignored directory
        self.touch(os.path.join(self.test_dir, "random_folder", "stuff.txt"))
        
        chunks = discover_rwd_chunks(self.test_dir)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].endswith("fluorescence.csv"))
        self.assertIn(ts, chunks[0])

    def test_missing_file_skip(self):
        """3) Timestamp-like folder missing fluorescence.csv is skipped"""
        ts_valid = "2025_11_11-12_55_16"
        self.touch(os.path.join(self.test_dir, ts_valid, "fluorescence.csv"))
        
        ts_empty = "2025_11_11-14_55_17"
        os.makedirs(os.path.join(self.test_dir, ts_empty), exist_ok=True)
        
        chunks = discover_rwd_chunks(self.test_dir)
        self.assertEqual(len(chunks), 1)
        self.assertIn(ts_valid, chunks[0])

    def test_empty_or_invalid_root(self):
        """4) Empty/invalid root raises ValueError"""
        # Case A: Root exists but no chunks
        with self.assertRaises(ValueError) as cm:
            discover_rwd_chunks(self.test_dir)
        self.assertIn("No valid RWD chunk directories found", str(cm.exception))
        
        # Case B: Root is strictly not a directory
        fake_file = os.path.join(self.test_dir, "not_a_dir")
        self.touch(fake_file)
        with self.assertRaises(ValueError) as cm:
            discover_rwd_chunks(fake_file)
        self.assertIn("Root path must be a directory", str(cm.exception))

    def test_toplevel_ignored(self):
        """5) Top-level fluorescence.csv is ignored in RWD mode"""
        # Valid chunk
        ts = "2025_11_11-12_55_16"
        self.touch(os.path.join(self.test_dir, ts, "fluorescence.csv"))
        
        # Top level trap
        self.touch(os.path.join(self.test_dir, "fluorescence.csv"))
        
        chunks = discover_rwd_chunks(self.test_dir)
        self.assertEqual(len(chunks), 1)
        self.assertIn(ts, chunks[0])
        self.assertFalse(chunks[0].endswith(os.path.join(self.test_dir, "fluorescence.csv")))

    def test_pipeline_integration(self):
        """6) Pipeline integration check"""
        from photometry_pipeline.pipeline import Pipeline
        from photometry_pipeline.config import Config
        
        # Setup structure
        ts = "2025_11_11-12_55_16"
        p_chunk = os.path.join(self.test_dir, ts, "fluorescence.csv")
        self.touch(p_chunk)
        p_trap = os.path.join(self.test_dir, "fluorescence.csv")
        self.touch(p_trap)
        
        # Minimal config to avoid instantiation errors if any
        cfg = Config()
        pipe = Pipeline(cfg)
        
        # Call discover with force_format='rwd'
        pipe.discover_files(self.test_dir, force_format="rwd")
        
        self.assertEqual(len(pipe.file_list), 1)
        # Should be the chunk file
        self.assertTrue(pipe.file_list[0].endswith(os.path.join(ts, "fluorescence.csv")))
        # Should NOT be the trap file
        self.assertFalse(pipe.file_list[0].endswith(os.path.join(self.test_dir, "fluorescence.csv")))

if __name__ == '__main__':
    unittest.main()
