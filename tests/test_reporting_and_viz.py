import unittest
import os
import shutil
import json
import yaml
import pandas as pd
import numpy as np
import pathlib
from photometry_pipeline.config import Config
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.core.reporting import generate_run_report
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.viz import plots

class TestReportingAndViz(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.test_dir = tempfile.mkdtemp(prefix="viz_test_")
        self.data_path = os.path.join(self.test_dir, "session.csv")
        self.out_dir = os.path.join(self.test_dir, "output")
        
        # Create Dummy Data (single chunk style)
        t = np.arange(0, 5.0, 1.0/40.0) # 5 seconds
        df = pd.DataFrame({
            "Time(s)": t,
            "Region0-410": np.sin(t),
            "Region0-470": np.cos(t)
        })
        df.to_csv(self.data_path, index=False)
        
        # Create Config
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        with open(self.config_path, "w") as f:
            f.write("target_fs_hz: 40.0\n")
            f.write("chunk_duration_sec: 2.0\n") 
            f.write("rwd_time_col: 'Time(s)'\n")
            f.write("uv_suffix: '-410'\n")
            f.write("sig_suffix: '-470'\n")
            f.write("window_sec: 1.0\n")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_session_metadata_full_schema(self):
        """
        Test A: Session Metadata strictly fully populated.
        """
        config = Config.from_yaml(self.config_path)
        # We need to bypass Pipeline to test load_chunk directly
        chunk = load_chunk(self.data_path, 'rwd', config, 0)
        
        self.assertIn("session_time", chunk.metadata)
        meta = chunk.metadata["session_time"]
        
        expected_keys = [
            "session_id", "session_start_iso", "chunk_index", 
            "zt0_iso", "zt_offset_hours", "notes"
        ]
        for k in expected_keys:
            self.assertIn(k, meta, f"Missing key {k}")
            
        self.assertEqual(meta["session_id"], "session") # filename stem
        self.assertEqual(meta["chunk_index"], 0)
        self.assertTrue(np.isnan(meta["zt_offset_hours"]))
        self.assertEqual(meta["session_start_iso"], "")

    def test_run_report_serialization(self):
        """
        Test B: Run Report Serialization with non-JSON types.
        """
        config = Config.from_yaml(self.config_path)
        # Inject non-primitive types
        config.some_path = pathlib.Path("foo/bar")
        config.some_numpy_int = np.int64(42)
        config.some_numpy_float = np.float64(3.14)
        config.some_numpy_array = np.array([1, 2, 3])
        
        generate_run_report(config, self.out_dir)
        
        report_path = os.path.join(self.out_dir, "run_report.json")
        self.assertTrue(os.path.exists(report_path))
        
        with open(report_path, 'r') as f:
            data = json.load(f)
            
        # Verify specific serializations
        cfg = data["configuration"]
        self.assertEqual(cfg["some_path"], str(pathlib.Path("foo/bar"))) # Windows/Posix aware
        self.assertEqual(cfg["some_numpy_int"], 42)
        self.assertAlmostEqual(cfg["some_numpy_float"], 3.14)
        self.assertEqual(cfg["some_numpy_array"], [1, 2, 3])

    def test_viz_missing_timesec(self):
        """
        Test C: Plot set B/C handles missing time_sec safely (logging/skip).
        """
        # Create a dummy CSV without time_sec
        traces_dir = os.path.join(self.out_dir, "viz_test_traces")
        os.makedirs(traces_dir, exist_ok=True)
        
        bad_df = pd.DataFrame({
            "Region0_uv_raw": [1,2,3],
            "Region0_sig_raw": [4,5,6]
            # No time_sec
        })
        bad_df.to_csv(os.path.join(traces_dir, "trace_0000.csv"), index=False)
        
        viz_dir = os.path.join(self.out_dir, "viz_output")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Capture logs
        with self.assertLogs(level='WARNING') as cm:
            plots.plot_continuous_multiday(traces_dir, "Region0", viz_dir, ["trace_0000.csv"])
            
        self.assertTrue(any("Missing 'time_sec'" in o for o in cm.output))
        # Ensure NO plot created
        self.assertFalse(os.path.exists(os.path.join(viz_dir, "plot_B_continuous_Region0.png")))

    def test_first_success_chunk_logic(self):
        """
        Test D: Chunk 0 fails, Chunk 1 succeeds -> Plots A/D generated.
        Ensures strict file ordering in input.
        """
        # 1. Prepare isolated input directory
        iso_dir = os.path.join(self.test_dir, "input_iso")
        if os.path.exists(iso_dir):
            shutil.rmtree(iso_dir)
        os.makedirs(iso_dir, exist_ok=True)
        
        # 2. Create chunks (RWD requires subdirectories with fluorescence.csv)
        fn0 = os.path.join(iso_dir, "chunk_0000")
        fn1 = os.path.join(iso_dir, "chunk_0001")
        os.makedirs(fn0, exist_ok=True)
        os.makedirs(fn1, exist_ok=True)
        
        file0 = os.path.join(fn0, "fluorescence.csv")
        file1 = os.path.join(fn1, "fluorescence.csv")
        
        with open(file0, 'w') as f:
            f.write("GARBAGE\n")
            
        shutil.copy(self.data_path, file1)
        
        # 3. Configure and Run
        config = Config.from_yaml(self.config_path)
        pipeline = Pipeline(config)
        
        pipeline.run(iso_dir, self.out_dir, force_format='rwd', recursive=False)
        
        # 4. Assertions
        viz_dir = os.path.join(self.out_dir, "viz")
        
        # Check plots A/D exist (from chunk 1)
        self.assertTrue(os.path.exists(os.path.join(viz_dir, "plot_A_raw_traces_Region0.png")))
        self.assertTrue(os.path.exists(os.path.join(viz_dir, "plot_D_correction_impact_Region0.png")))
        
        # Check QC Summary says chunk 0 failed
        qc_path = os.path.join(self.out_dir, "qc", "qc_summary.json")
        with open(qc_path, 'r') as f:
            qc = json.load(f)
            
        failed_files = [x['file'] for x in qc['failed_chunks']]
        self.assertTrue(any("chunk_0000" in f for f in failed_files))
        
if __name__ == '__main__':
    unittest.main()
