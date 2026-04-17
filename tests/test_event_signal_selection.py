import numpy as np
import pandas as pd
import unittest
import os
import tempfile
import yaml
import sys
import subprocess
import json
import shutil
from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.config import Config

class MockChunk:
    def __init__(self, delta_f, dff, time_sec, fs_hz):
        self.delta_f = delta_f
        self.dff = dff
        self.time_sec = time_sec
        self.fs_hz = fs_hz
        self.chunk_id = "test_chunk"
        self.source_file = "test.raw"
        self.channel_names = ["ROI1"]
        self.metadata = {"qc_warnings": []}

class TestEventSignalSelection(unittest.TestCase):
    
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.in_dir = os.path.join(self.tmp_dir, "in")
        os.makedirs(os.path.join(self.in_dir, "chunk_0000"))
        
        # Small deterministic dataset: 15 seconds at 100 Hz (1500 samples)
        # to safely exceed the 1000-sample tonic robustness gate.
        fs = 100.0
        n_samples = 1500
        t = np.arange(n_samples) / fs
        
        # Linearly growing baseline to ensure correlation regression works
        # without degenerate 0 variance noise, varying from 0.1 to 0.2
        uv = np.linspace(0.1, 0.2, n_samples)
        
        # Perfectly correlated base signal
        sig = uv.copy()
        
        # Exactly 5 deterministic spikes of height 0.5
        for i in range(5):
            idx = 200 + i * 200
            # Width is 5 samples to survive filtering/thresholding
            sig[idx: idx + 5] += 0.5 
            
        self.df_in = pd.DataFrame({
            'Timestamp': t,
            'Region0G': uv,
            'Region0R': sig
        })
        self.df_in.to_csv(os.path.join(self.in_dir, "chunk_0000", "fluorescence.csv"), index=False)
        
        # Create a config that relies on 'absolute' thresholding.
        # Spikes of absolute height 0.5 are missed by delta_f (val < 1.0)
        # but massively amplified inside dff logic (0.5 / 0.1 * 100 = 500 > 1.0)
        self.cfg_path = os.path.join(self.tmp_dir, "config.yaml")
        with open(self.cfg_path, 'w') as f:
            yaml.dump({
                'rwd_time_col': 'Timestamp',
                'target_fs_hz': 100.0,
                'chunk_duration_sec': 15.0,
                'window_sec': 2.0,
                'step_sec': 1.0,
                'min_valid_windows': 2,
                'peak_threshold_method': 'absolute',
                'peak_threshold_abs': 1.0,
                'peak_min_distance_sec': 0.2, # tight enough for our 1.5s spaced spikes
                'uv_suffix': 'G',
                'sig_suffix': 'R'
            }, f)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
            
    def test_invalid_event_signal_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'event_signal': 'invalid_signal'}, f)
            temp_path = f.name
            
        try:
            with self.assertRaises(ValueError) as ctx:
                Config.from_yaml(temp_path)
            self.assertIn("Invalid event_signal", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_signal_excursion_polarity_defaults_to_positive(self):
        cfg = Config()
        self.assertEqual(cfg.signal_excursion_polarity, "positive")

    def test_invalid_signal_excursion_polarity_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'signal_excursion_polarity': 'upward_only'}, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                Config.from_yaml(temp_path)
            self.assertIn("Invalid signal_excursion_polarity", str(ctx.exception))
        finally:
            os.remove(temp_path)
            
    def test_peak_routing(self):
        fs = 10.0
        t = np.arange(11) / fs
        # 0.4 < 0.5 (Not a peak)
        # 1.1 >= 0.5 (A peak)
        arr_delta_f = np.array([0, 0, 0, 0.4, 0, 0, 0, 1.1, 0, 0, 0], dtype=float).reshape(-1, 1)
        # Both are peaks if scaled/calculated differently or for dff test
        arr_dff = np.array([0, 0, 0, 1.1, 0, 0, 0, 1.1, 0, 0, 0], dtype=float).reshape(-1, 1)
        chunk = MockChunk(arr_delta_f, arr_dff, t, fs)
        
        config_delta_f = Config()
        config_delta_f.event_signal = 'delta_f'
        config_delta_f.peak_threshold_method = 'absolute'
        config_delta_f.peak_threshold_abs = 0.5
        config_delta_f.peak_min_distance_sec = 0.05
        df_delta_f = extract_features(chunk, config_delta_f)
        self.assertEqual(df_delta_f.loc[0, 'peak_count'], 1)
        
        config_dff = Config()
        config_dff.event_signal = 'dff'
        config_dff.peak_threshold_method = 'absolute'
        config_dff.peak_threshold_abs = 0.5
        config_dff.peak_min_distance_sec = 0.05
        df_dff = extract_features(chunk, config_dff)
        self.assertEqual(df_dff.loc[0, 'peak_count'], 2)

    def test_missing_signal_fails_closed(self):
        fs = 10.0
        t = np.arange(10) / fs
        chunk = MockChunk(None, None, t, fs)
        config = Config()
        config.event_signal = 'dff'
        with self.assertRaisesRegex(ValueError, "chunk.dff was not computed"):
            extract_features(chunk, config)
        config.event_signal = 'delta_f'
        with self.assertRaisesRegex(ValueError, "chunk.delta_f was not computed"):
            extract_features(chunk, config)

    def test_normal_mode_semantics(self):
        out_base = os.path.join(self.tmp_dir, "out_normal")
        events_path = os.path.join(self.tmp_dir, "events_normal.ndjson")
        cmd = [sys.executable, "tools/run_full_pipeline_deliverables.py",
               "--input", self.in_dir, "--out-base", out_base,
               "--config", self.cfg_path, "--format", "rwd",
               "--events", events_path, "--sessions-per-hour", "1"]

        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Runner failed:\n{res.stdout}\n{res.stderr}")
        
        self.assertTrue(os.path.exists(events_path))
        ctx_payload = None
        with open(events_path, "r") as f:
            for line in f:
                d = json.loads(line)
                if d.get("stage") == "engine" and d.get("type") == "context":
                    ctx_payload = d["payload"]
                    break
        self.assertIsNotNone(ctx_payload)
        self.assertIsNone(ctx_payload.get("features_extracted"), "features_extracted MUST be null/None in normal mode pre-analysis")
        self.assertFalse(ctx_payload.get("traces_only"))
        self.assertEqual(ctx_payload.get("event_signal"), "dff")

    def test_traces_only_semantics(self):
        out_base = os.path.join(self.tmp_dir, "out_traces")
        events_path = os.path.join(self.tmp_dir, "events_traces.ndjson")
        cmd = [sys.executable, "tools/run_full_pipeline_deliverables.py",
               "--input", self.in_dir, "--out-base", out_base,
               "--config", self.cfg_path, "--format", "rwd",
               "--events", events_path, "--traces-only", "--sessions-per-hour", "1"]
               
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Runner failed:\n{res.stdout}\n{res.stderr}")
        
        self.assertTrue(os.path.exists(events_path))
        ctx_payload = None
        with open(events_path, "r") as f:
            for line in f:
                d = json.loads(line)
                if d.get("stage") == "engine" and d.get("type") == "context":
                    ctx_payload = d["payload"]
                    break
        self.assertIsNotNone(ctx_payload)
        self.assertFalse(ctx_payload.get("features_extracted"), "features_extracted MUST be False in traces-only mode")
        self.assertTrue(ctx_payload.get("traces_only"))

    def test_run_report_semantics(self):
        out_normal = os.path.join(self.tmp_dir, "out_normal_analyze")
        cmd = [sys.executable, "analyze_photometry.py",
               "--input", self.in_dir, "--out", out_normal,
               "--config", self.cfg_path, "--format", "rwd", "--mode", "phasic", "--recursive"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        
        with open(os.path.join(out_normal, "run_report.json")) as f:
            rr = json.load(f)
        self.assertIsNone(rr["run_context"].get("features_extracted"), "run_report.json features_extracted MUST be None in normal mode")
        
        out_traces = os.path.join(self.tmp_dir, "out_traces_analyze")
        cmd_traces = cmd + ["--traces-only"]
        cmd_traces[cmd_traces.index(out_normal)] = out_traces
        res_traces = subprocess.run(cmd_traces, capture_output=True, text=True)
        self.assertEqual(res_traces.returncode, 0, res_traces.stderr)
        
        with open(os.path.join(out_traces, "run_report.json")) as f:
            rr_t = json.load(f)
        self.assertFalse(rr_t["run_context"].get("features_extracted"), "run_report.json features_extracted MUST be False in traces-only mode")

    def test_cli_override_routes_signal_end_to_end(self):
        out_dff = os.path.join(self.tmp_dir, "out_dff")
        out_delta = os.path.join(self.tmp_dir, "out_delta")
        
        cmd_dff = [sys.executable, "analyze_photometry.py", "--input", self.in_dir, "--out", out_dff, "--config", self.cfg_path, "--format", "rwd", "--mode", "phasic", "--recursive", "--event-signal", "dff"]
        res_dff = subprocess.run(cmd_dff, capture_output=True, text=True)
        self.assertEqual(res_dff.returncode, 0, res_dff.stderr)
        
        cmd_delta = [sys.executable, "analyze_photometry.py", "--input", self.in_dir, "--out", out_delta, "--config", self.cfg_path, "--format", "rwd", "--mode", "phasic", "--recursive", "--event-signal", "delta_f"]
        res_delta = subprocess.run(cmd_delta, capture_output=True, text=True)
        self.assertEqual(res_delta.returncode, 0, res_delta.stderr)
        
        feat_dff_path = os.path.join(out_dff, "features", "features.csv")
        feat_delta_path = os.path.join(out_delta, "features", "features.csv")
        
        feat_dff = pd.read_csv(feat_dff_path)
        feat_delta = pd.read_csv(feat_delta_path)
        
        val_dff = feat_dff.loc[0, 'peak_count']
        val_delta = feat_delta.loc[0, 'peak_count']
        
        # Exact assertions enforcing the determinism
        self.assertEqual(val_delta, 0, f"delta_f peaks must be 0, found {val_delta}.")
        self.assertEqual(val_dff, 5, f"dff peaks must be 5, found {val_dff}.")
        
        with open(os.path.join(out_dff, "run_report.json")) as f:
            rr_dff = json.load(f)
        self.assertEqual(rr_dff['run_context']['event_signal'], 'dff')
        self.assertEqual(rr_dff["run_context"]["signal_excursion_polarity"], "positive")
        
        with open(os.path.join(out_delta, "run_report.json")) as f:
            rr_delta = json.load(f)
        self.assertEqual(rr_delta['run_context']['event_signal'], 'delta_f')
        self.assertEqual(rr_delta["run_context"]["signal_excursion_polarity"], "positive")

if __name__ == '__main__':
    unittest.main()
