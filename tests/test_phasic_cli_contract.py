# python -m unittest tests.test_phasic_cli_contract

import unittest
import os
import shutil
import tempfile
import sys
import subprocess
import pandas as pd
import numpy as np
import yaml

# Removed path injection: relying on absolute path construction via `repo_root` below.

class TestPhasicCLIContract(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config_test.yaml")
        
        # Initial config for generator (safe defaults)
        # Note: rwd_time_col is a placeholder here; we will detect/update it 
        # safely using PyYAML based on actual generated data.
        with open(self.config_path, 'w') as f:
            f.write("""
chunk_duration_sec: 600.0
target_fs_hz: 10.0
baseline_method: uv_raw_percentile_session
baseline_percentile: 10.0
rwd_time_col: TimeStamp # Placeholder
uv_suffix: "-410"
sig_suffix: "-470"
window_sec: 60.0
step_sec: 10.0
peak_threshold_method: percentile
peak_threshold_percentile: 95.0
peak_min_distance_sec: 0.5
qc_max_chunk_fail_fraction: 1.0
""")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_phasic_end_to_end_contract(self):
        """
        Strict Contract:
        1. Generate 'high_phasic' data (frozen generator).
        2. Inspect generated CSVs (Header-based) to detect Time column.
        3. Run 'phasic' pipeline using safely updated config.
        4. Verify outputs exist and contain finite metrics (Strict/Filtered assertions).
        """
        
        # Robust path resolution
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(this_file_dir)
        
        synth_script = os.path.join(repo_root, "tools", "synth_photometry_dataset.py")
        analyze_script = os.path.join(repo_root, "analyze_photometry.py")
        
        # ---------------------------------------------------------
        # 1. Generate Data (Frozen Generator)
        # ---------------------------------------------------------
        
        synth_out = os.path.join(self.test_dir, "synth_data")
        
        # Generator params:
        # - Remove --fs-hz (rely on config/defaults)
        # - Increase total-days to 0.2 for stability
        gen_cmd = [
            sys.executable, synth_script,
            "--out", synth_out,
            "--format", "rwd",
            "--config", self.config_path,
            "--total-days", "0.2", 
            "--recordings-per-hour", "2",
            "--n-rois", "2",
            "--phasic-mode", "high_phasic",
            "--seed", "123"
        ]
        
        print(f"Running Generator: {' '.join(gen_cmd)}")
        result_gen = subprocess.run(gen_cmd, capture_output=True, text=True)
        
        if result_gen.returncode != 0:
            print("GENERATOR STDOUT:", result_gen.stdout)
            print("GENERATOR STDERR:", result_gen.stderr)
            self.fail("Synthetic generator failed (non-zero exit).")
            
        self.assertTrue(os.path.exists(synth_out))
        
        # ---------------------------------------------------------
        # 1b. Dynamic Config Update (Header-Based Discovery)
        # ---------------------------------------------------------
        
        # Load config to get suffixes dynamically (SSOT)
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        uv_suffix = cfg["uv_suffix"]
        sig_suffix = cfg["sig_suffix"]
        
        priority_cols = ["TimeStamp", "Time(s)", "Time", "time", "t_sec", "t"]
        
        found_csv = None
        selected_col = None
        
        inspected_headers = []
        scan_stats = {
            "scanned": 0,
            "has_time": 0,
            "has_uv": 0,
            "has_sig": 0,
            "has_time_uv": 0,
            "has_time_sig": 0,
            "has_time_uv_sig": 0
        }
        
        # Walk and inspect headers
        for root, dirs, files in os.walk(synth_out):
            for f in files:
                if f.endswith(".csv"):
                    fpath = os.path.join(root, f)
                    scan_stats["scanned"] += 1
                    try:
                        # Read header only
                        header = pd.read_csv(fpath, nrows=0).columns.tolist()
                        inspected_headers.append(f"{fpath}: {header}")
                        
                        # Check 1: Time Column
                        time_match = None
                        for c in priority_cols:
                            if c in header:
                                time_match = c
                                break
                        
                        if time_match: scan_stats["has_time"] += 1
                                
                        # Check 2: UV Suffix
                        has_uv = any(uv_suffix in c for c in header)
                        if has_uv: scan_stats["has_uv"] += 1
                        
                        # Check 3: Sig Suffix
                        has_sig = any(sig_suffix in c for c in header)
                        if has_sig: scan_stats["has_sig"] += 1
                        
                        # Derived Stats
                        if time_match and has_uv: scan_stats["has_time_uv"] += 1
                        if time_match and has_sig: scan_stats["has_time_sig"] += 1
                        if time_match and has_uv and has_sig: scan_stats["has_time_uv_sig"] += 1
                        
                        if time_match and has_uv and has_sig:
                            found_csv = fpath
                            selected_col = time_match
                            print(f"Discovery: Selected time column '{selected_col}' from {fpath}. Header: {header}")
                            break
                    except Exception as e:
                        print(f"Skipping file {f} due to error: {e}")
            if found_csv: break
            
        if not found_csv:
            debug_info = "\n".join(inspected_headers[:5]) # Show first 5
            self.fail(f"No valid RWD chunk CSV found matching schema (TimeCol + {uv_suffix} + {sig_suffix}).\nStats: {scan_stats}\nInspected headers:\n{debug_info}")
            
        self.assertIsNotNone(selected_col)
        
        # Safe YAML Update (Deterministic)
        cfg['rwd_time_col'] = selected_col
        
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
            
        # ---------------------------------------------------------
        # 2. Run Pipeline (Phasic Mode)
        # ---------------------------------------------------------
        
        pipeline_out = os.path.join(self.test_dir, "pipeline_out")
        
        pipe_cmd = [
            sys.executable, analyze_script,
            "--input", synth_out,
            "--out", pipeline_out,
            "--config", self.config_path,
            "--mode", "phasic",
            "--overwrite",
            "--recursive"
        ]
        
        print(f"Running Pipeline: {' '.join(pipe_cmd)}")
        result_pipe = subprocess.run(pipe_cmd, capture_output=True, text=True)
        
        if result_pipe.returncode != 0:
            print("PIPELINE STDOUT:", result_pipe.stdout)
            print("PIPELINE STDERR:", result_pipe.stderr)
            self.fail("Pipeline failed (non-zero exit).")
            
        # ---------------------------------------------------------
        # 3. Assert Outputs (Strict Metrics)
        # ---------------------------------------------------------
        
        # A. Metadata
        meta_path = os.path.join(pipeline_out, "run_metadata.json")
        self.assertTrue(os.path.exists(meta_path), "run_metadata.json missing")
        
        # B. Features CSV
        feat_path = os.path.join(pipeline_out, "features", "features.csv")
        self.assertTrue(os.path.exists(feat_path), "features/features.csv missing")
        
        # C. Load and Validate Content
        df = pd.read_csv(feat_path)
        print(f"Features loaded: {len(df)} rows")
        
        # C0. Not empty
        self.assertGreater(len(df), 0, "Features CSV is empty")

        # C1. Required Columns (Must be checked BEFORE coercion to avoid KeyErrors)
        required_cols = ['roi', 'peak_count', 'auc', 'mean', 'std']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        # C2. Coerce Types (Safety)
        for col in ["peak_count", "auc", "mean", "std"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # C3. Strict Metric Checks (Values)
        # 1. At least one row has peak_count >= 3
        peak_max = df["peak_count"].max()
        self.assertGreaterEqual(peak_max, 3, f"Expected peak_count >= 3 in high_phasic mode. Found max={peak_max}")
        
        # 2. At least one row has auc > 1e-6
        auc_max = df["auc"].max()
        self.assertGreater(auc_max, 1e-6, f"Expected AUC > 1e-6 in high_phasic mode. Found max={auc_max}")

        # C4. Tightened Finite Metric Assertion (Event-Positive Filter)
        # Filter for event-positive rows: (peak_count >= 3) AND (auc > 1e-6)
        # Note: We use 1e-6 instead of 0 to be consistent with the strict check above.
        event_positive_mask = (df["peak_count"] >= 3) & (df["auc"] > 1e-6)
        event_positive_rows = df[event_positive_mask]
        
        self.assertGreater(len(event_positive_rows), 0, f"No event-positive rows (peaks>=3 & auc>1e-6) found. Max peaks={peak_max}, Max AUC={auc_max}")
        
        # Check finiteness ONLY within event-positive rows
        # If a row has valid events, its metrics must be finite.
        finite_mask = (
            np.isfinite(event_positive_rows["auc"]) & 
            np.isfinite(event_positive_rows["mean"]) & 
            np.isfinite(event_positive_rows["std"])
        )
        
        # Assert ALL event-positive rows are fully finite
        nonfinite_n = int((~finite_mask).sum())
        self.assertTrue(bool(finite_mask.all()), 
            f"Found {nonfinite_n}/{len(event_positive_rows)} event-positive rows with non-finite auc/mean/std.")

        print("Contract Test Passed: Output exists, features found, metrics robust and finite.")

if __name__ == '__main__':
    unittest.main()
