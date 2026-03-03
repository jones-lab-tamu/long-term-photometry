import json
import os
import shutil
import tempfile
import unittest

from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        
        # Create a mock CSV dataset
        self.input_dir = os.path.join(self.tmp_dir, "dataset")
        os.makedirs(self.input_dir, exist_ok=True)
        
        # NPM format requires FrameCounter, SystemTimestamp, LedState, and ROI cols like RegionA_470G
        # LedState=1 is UV, LedState=2 is Signal
        # Needs at least 2 rows of each to pass the diff>0 monotonicity checks 
        csv_content = (
            "FrameCounter,SystemTimestamp,LedState,RegionA_470G,RegionB_470G\n"
            "1,0.1,1,1.0,1.0\n"
            "2,0.2,2,2.0,2.0\n"
            "3,0.3,1,1.1,1.1\n"
            "4,0.4,2,2.1,2.1\n"
        )
        
        # Files ordered such that natural sort is different from alphabetical
        self.files = ["session_10.csv", "session_2.csv", "session_1.csv"]
        for f in self.files:
            with open(os.path.join(self.input_dir, f), "w") as out:
                out.write(csv_content)
                
        self.config = Config(
            event_signal="dff",
            preview_first_n=2,
            target_fs_hz=1.0,
            chunk_duration_sec=1.0,
            allow_partial_final_chunk=True
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_discover_mode_does_not_write_run_artifacts(self):
        """Discovery MUST NOT create any files or directories in the surrounding environment."""
        # Record initial dir state
        initial_contents = set(os.listdir(self.input_dir))
        tmp_contents = set(os.listdir(self.tmp_dir))
        
        discover_inputs(
            input_dir=self.input_dir,
            config=self.config,
            force_format="auto"
        )
        
        # Assert no new files in dataset dir
        self.assertEqual(set(os.listdir(self.input_dir)), initial_contents)
        # Assert no new files in parent tmp dir (e.g. out_base/run_dirs)
        self.assertEqual(set(os.listdir(self.tmp_dir)), tmp_contents)

    def test_discover_output_schema_and_ordering(self):
        """Assert JSON has correct schema and session ordering is natural."""
        res = discover_inputs(
            input_dir=self.input_dir,
            config=self.config,
            force_format="auto",
            preview_first_n=2
        )
        
        self.assertEqual(res["schema_version"], 1)
        self.assertEqual(res["input_dir"], self.input_dir)
        self.assertEqual(res["resolved_format"], "npm")  # Sniffed from CSV
        self.assertEqual(res["n_total_discovered"], 3)
        self.assertEqual(res["preview_first_n"], 2)
        self.assertEqual(res["n_preview"], 2)
        
        sessions = res["sessions"]
        self.assertEqual(len(sessions), 3)
        
        # Order should be natural sort: 1, 2, 10
        self.assertEqual(sessions[0]["session_id"], "session_1")
        self.assertEqual(sessions[1]["session_id"], "session_2")
        self.assertEqual(sessions[2]["session_id"], "session_10")
        
        # Only the first 2 should be included in preview
        self.assertTrue(sessions[0]["included_in_preview"])
        self.assertTrue(sessions[1]["included_in_preview"])
        self.assertFalse(sessions[2]["included_in_preview"])

    def test_rois_are_canonical(self):
        """Assert ROI ids returned are exactly what the pipeline extracts."""
        res = discover_inputs(
            input_dir=self.input_dir,
            config=self.config,
            force_format="auto"
        )
        
        rois = res["rois"]
        self.assertEqual(len(rois), 2)
        
        # For NPM, the adapter constructs canonical names like Region0, Region1
        roi_ids = [r["roi_id"] for r in rois]
        self.assertIn("Region0", roi_ids)
        self.assertIn("Region1", roi_ids)


if __name__ == '__main__':
    unittest.main()
