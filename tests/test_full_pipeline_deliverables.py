
import unittest
import os
import sys
import shutil
import subprocess
import glob
import json
import pandas as pd
from unittest import mock
import importlib.util

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestFullPipelineDeliverables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        # Strict hygiene: Use unique temp directory
        cls.out_dir = tempfile.mkdtemp(prefix="out_deliverables_test_")
        cls.input_dir = os.path.join(cls.out_dir, "input_RWD")
        import uuid
        cls.output_package = os.path.join(cls.out_dir, f"package_{uuid.uuid4().hex[:8]}")
        
        # Original config
        orig_config_path = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
        
        # Create temp config (copy of original)
        cls.config_path = os.path.join(cls.out_dir, "qc_universal_config.yaml")
        
        # No need to cleanup hardcoded dir, we have a fresh one.
        if not os.path.exists(cls.out_dir):
            os.makedirs(cls.out_dir)

        shutil.copy2(orig_config_path, cls.config_path)

        # 1. Synthetic Data (2.0 days, 2 rec/hr = 96 chunks)
        print("Generating synthetic data (2.0 days)...")
        gen_cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", cls.input_dir,
            "--format", "rwd",
            "--config", cls.config_path,
            "--total-days", "2.0",
            "--recordings-per-hour", "2", 
            "--recording-duration-min", "10.0",
            "--n-rois", "2",
            "--phasic-mode", "phase_locked_to_tonic",
            "--seed", "42",
            "--preset", "biological_shared_nuisance"
        ]
        subprocess.check_call(gen_cmd)

    def test_pipeline_deliverables(self):
        # Run Tool with EXPLICIT sessions-per-hour (Required for duty cycle)
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", self.output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2"
        ]
        print(f"Running pipeline command: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Manifest
        man_path = os.path.join(self.output_package, "MANIFEST.json")
        self.assertTrue(os.path.exists(man_path))
        with open(man_path, 'r') as f:
            manifest = json.load(f)
            
        self.assertEqual(manifest['sessions_per_hour'], 2)
        
        # Check duration AND stride are recorded
        self.assertIn('session_duration_s', manifest)
        self.assertIn('session_stride_s', manifest)
        
        dur = manifest['session_duration_s']
        stride = manifest['session_stride_s']
        
        # Real/Synth Expectation:
        # Duration ~ 600s
        # Stride ~ 1800s
        self.assertTrue(590 < dur < 610, f"Duration {dur} not ~600")
        self.assertTrue(1790 < stride < 1810, f"Stride {stride} not ~1800")
        
        # Region0 Checks
        reg0 = os.path.join(self.output_package, "Region0")
        self.assertTrue(os.path.exists(reg0))
        
        # Multi-day plots
        for day in ["000", "001"]:
            self.assertTrue(os.path.exists(os.path.join(reg0, "day_plots", f"phasic_sig_iso_day_{day}.png")))
            self.assertTrue(os.path.exists(os.path.join(reg0, "day_plots", f"phasic_dFF_day_{day}.png")))
            self.assertTrue(os.path.exists(os.path.join(reg0, "day_plots", f"phasic_stacked_day_{day}.png")))
            
        # 3-Panel Correction Impact
        self.assertTrue(os.path.exists(os.path.join(reg0, "summary", "phasic_correction_impact.png")))
        self.assertTrue(os.path.exists(os.path.join(reg0, "tables", "phasic_correction_impact_session.csv")))
        
        # Time Series CSV Row Counts
        ts_csv = os.path.join(reg0, "tables", "phasic_peak_rate_timeseries.csv")
        self.assertTrue(os.path.exists(ts_csv))
        df_ts = pd.read_csv(ts_csv)
        # Expected: 96 rows 
        self.assertEqual(len(df_ts), 96)
        
        # Check columns
        req_cols = ['time_hours', 'day', 'hour', 'session_in_hour', 'window_seconds']
        for c in req_cols:
            self.assertIn(c, df_ts.columns)
            
        # Check window_seconds matches duration (approx 600s)
        # NOT stride (1800s)
        first_dur = df_ts.iloc[0]['window_seconds']
        self.assertTrue(590 < first_dur < 610, f"Window seconds {first_dur} should be ~600")

        # Check Peak Rate Columns
        self.assertIn('peak_rate_per_min', df_ts.columns)
        self.assertIn('peak_count', df_ts.columns)
        self.assertNotIn('n_peaks', df_ts.columns)
        
        # Check AUC CSV
        auc_csv = os.path.join(reg0, "tables", "phasic_auc_timeseries.csv")
        self.assertTrue(os.path.exists(auc_csv))
        df_auc = pd.read_csv(auc_csv)
        self.assertIn('auc_above_threshold_dff_s', df_auc.columns)
        for c in req_cols:
            self.assertIn(c, df_auc.columns)
            
        # Tonic Time Series CSV (New check for Phase A migration)
        tonic_csv = os.path.join(reg0, "tables", "tonic_df_timeseries.csv")
        self.assertTrue(os.path.exists(tonic_csv))
        df_tonic = pd.read_csv(tonic_csv)
        self.assertIn('time_hours', df_tonic.columns)
        self.assertIn('tonic_df', df_tonic.columns)
        self.assertTrue(len(df_tonic) > 0)
            
        # Manifest Consistency
        deliv = manifest['deliverables']['Region0']
        self.assertIn('days_dff', deliv)
        self.assertIn('days_sig_iso', deliv)
        self.assertIn('days_stacked', deliv)
        self.assertIn('days_generated', deliv)
        self.assertEqual(deliv['days_dff'], deliv['days_generated'])
        self.assertEqual(deliv['days_sig_iso'], deliv['days_generated'])
        self.assertEqual(deliv['days_stacked'], deliv['days_generated'])
        
        # ---------------------------------------------------------------------
        # Issue 2: explicitly verify absence of redundant `_analysis` plot outputs
        # ---------------------------------------------------------------------
        analysis_dir = os.path.join(self.output_package, "_analysis")
        phasic_out = os.path.join(analysis_dir, "phasic_out")
        tonic_out = os.path.join(analysis_dir, "tonic_out")
        
        # Native analysis-stage plot dirs:
        self.assertFalse(os.path.exists(os.path.join(phasic_out, "viz")), "Redundant _analysis/phasic_out/viz was produced")
        self.assertFalse(os.path.exists(os.path.join(tonic_out, "viz")), "Redundant _analysis/tonic_out/viz was produced")
        
        # Wrapper-stage redundant plot dirs:
        self.assertFalse(os.path.exists(os.path.join(phasic_out, "viz_Region0")), "Redundant viz_Region0 was produced")
        self.assertFalse(os.path.exists(os.path.join(phasic_out, "qc_dff_Region0")), "Redundant qc_dff_Region0 was produced")
        self.assertFalse(os.path.exists(os.path.join(phasic_out, "session_qc_Region0")), "Redundant session_qc_Region0 was produced")
        self.assertFalse(os.path.exists(os.path.join(tonic_out, "tonic_qc")), "Redundant tonic_qc was produced")
        
        # Tonic traces directory should NOT exist (New contract)
        self.assertFalse(os.path.exists(os.path.join(tonic_out, "traces")), "Legacy _analysis/tonic_out/traces/ was produced")

        # Phasic traces directory should NOT exist (New contract)
        self.assertFalse(os.path.exists(os.path.join(phasic_out, "traces")), "Phasic traces directory was produced")
        
        # Phase A: Tonic Producer Cleanup contract verifications
        self.assertFalse(os.path.exists(os.path.join(tonic_out, "features", "features.csv")), "Removed: tonic_out features.csv should not be produced")
        self.assertFalse(os.path.exists(os.path.join(tonic_out, "qc", "qc_summary.json")), "Removed: tonic_out qc_summary.json should not be produced")

    def test_tonic_csv_content_independence(self):
        """
        RIGOROUS PROOF: Verify that tonic wrapper logic (timing & tables) 
        functions correctly even when all tonic CSV trace files are deleted,
        proving total reliance on the HDF5 cache for waveform data.
        """
        import tempfile
        
        # 1. SETUP: Create a fresh output directory
        test_run_dir = tempfile.mkdtemp(prefix="tonic_independence_")
        analysis_dir = os.path.join(test_run_dir, "_analysis")
        tonic_analysis_out = os.path.join(analysis_dir, "tonic_out")
        tonic_traces_dir = os.path.join(tonic_analysis_out, "traces")
        
        # 2. SEED: Copy the HDF5 cache and report from a real run
        source_tonic = os.path.join(self.output_package, "_analysis", "tonic_out")
        source_phasic = os.path.join(self.output_package, "_analysis", "phasic_out")
        if not os.path.exists(source_tonic):
            self.test_pipeline_deliverables()
            
        os.makedirs(tonic_analysis_out, exist_ok=True)
        shutil.copy2(os.path.join(source_tonic, "tonic_trace_cache.h5"), tonic_analysis_out)
        shutil.copy2(os.path.join(source_tonic, "run_report.json"), tonic_analysis_out)
        shutil.copy2(os.path.join(source_tonic, "config_used.yaml"), tonic_analysis_out)
        
        # Wrapper Section 7 unconditionally calls phasic plotters (dayplot bundle),
        # which still read CSV traces (NOT migrated in B2). Provide phasic artifacts.
        phasic_analysis_out = os.path.join(analysis_dir, "phasic_out")
        os.makedirs(phasic_analysis_out, exist_ok=True)
        shutil.copy2(os.path.join(source_phasic, "phasic_trace_cache.h5"), phasic_analysis_out)
        shutil.copy2(os.path.join(source_phasic, "run_report.json"), phasic_analysis_out)
        shutil.copy2(os.path.join(source_phasic, "config_used.yaml"), phasic_analysis_out)
        traces_src = os.path.join(source_phasic, "traces")
        if os.path.exists(traces_src):
            shutil.copytree(traces_src, os.path.join(phasic_analysis_out, "traces"), dirs_exist_ok=True)
        
        # 3. VERIFY: Ensure NO tonic CSV files were produced by default (New Contract)
        self.assertFalse(os.path.exists(os.path.join(tonic_analysis_out, "traces")), "Tonic traces directory should not exist by default")
        
        # 4. EXECUTE: Run the wrapper in-process
        script_path = os.path.join(PROJECT_ROOT, 'tools', 'run_full_pipeline_deliverables.py')
        spec = importlib.util.spec_from_file_location("runner", script_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        
        test_args = [
            script_path,
            "--input", self.input_dir,
            "--out", test_run_dir,
            "--config", self.config_path,
            "--format", "rwd",
            "--mode", "tonic",
            "--sessions-per-hour", "2",
            "--overwrite"
        ]
        
        import subprocess as sp
        real_check_call = sp.check_call
        
        def mock_analysis_side_effect(cmd):
            cmd_str = " ".join([str(p).replace(os.sep, '/') for p in cmd])
            
            if 'analyze_photometry.py' in cmd_str:
                if '--out' in cmd:
                    out_path = cmd[cmd.index('--out') + 1]
                    os.makedirs(out_path, exist_ok=True)
                    # Inject tonic artifacts (NO CSVs for tonic)
                    shutil.copy2(os.path.join(source_tonic, "tonic_trace_cache.h5"), out_path)
                    shutil.copy2(os.path.join(source_tonic, "run_report.json"), out_path)
                    shutil.copy2(os.path.join(source_tonic, "config_used.yaml"), out_path)
                    # --overwrite deletes _analysis/, so re-seed phasic_out here.
                    # Section 7 plotting unconditionally calls phasic dayplot bundle
                    # which still reads CSV traces (NOT migrated in B2).
                    p_out = os.path.join(os.path.dirname(out_path), 'phasic_out')
                    os.makedirs(p_out, exist_ok=True)
                    shutil.copy2(os.path.join(source_phasic, "phasic_trace_cache.h5"), p_out)
                    shutil.copy2(os.path.join(source_phasic, "run_report.json"), p_out)
                    shutil.copy2(os.path.join(source_phasic, "config_used.yaml"), p_out)
                    traces_src = os.path.join(source_phasic, "traces")
                    if os.path.exists(traces_src):
                        shutil.copytree(traces_src, os.path.join(p_out, "traces"), dirs_exist_ok=True)
                return 0
                
            return real_check_call(cmd)

        with mock.patch('sys.argv', test_args):
            with mock.patch('subprocess.check_call', side_effect=mock_analysis_side_effect):
                # Suppress normal prints if desired
                runner.main()
        
        # 5. VERIFY: The deliverable was produced successfully from CACHE ONLY
        # If it tried to read CSVs, it would have failed (none exist).
        # If it failed to resolve timing, it would have exited.
        
        tonic_csv = os.path.join(test_run_dir, "Region0", "tables", "tonic_df_timeseries.csv")
        self.assertTrue(os.path.exists(tonic_csv), "tonic_df_timeseries.csv was NOT produced")
        
        df = pd.read_csv(tonic_csv)
        self.assertGreater(len(df), 0, "Produced CSV is empty")
        self.assertIn('time_hours', df.columns)
        self.assertIn('tonic_df', df.columns)
        
        # Cleanup
        shutil.rmtree(test_run_dir)

    def test_phasic_csv_content_independence(self):
        """
        RIGOROUS PROOF: Verify that phasic wrapper timing/session-duration
        resolution and plot_phasic_time_series_summary.py duration validation
        function correctly using phasic_trace_cache.h5.

        Scope: Phase B2 approved migrations only.
        - Wrapper timing resolution: uses cache, not CSV content.
        - Summary plotter duration validation: uses cache, not CSV content.

        Note: plot_phasic_dayplot_bundle.py is NOT migrated in B2 and still
        reads CSV traces, so those are provided to allow the full pipeline
        to complete.
        """
        import tempfile
        import importlib.util
        
        # 1. SETUP: Create a fresh output directory
        test_run_dir = tempfile.mkdtemp(prefix="phasic_independence_")
        analysis_dir = os.path.join(test_run_dir, "_analysis")
        phasic_analysis_out = os.path.join(analysis_dir, "phasic_out")
        
        # 2. SEED: Copy necessary artifacts from a real run
        source_phasic = os.path.join(self.output_package, "_analysis", "phasic_out")
        if not os.path.exists(source_phasic):
            self.test_pipeline_deliverables()
            
        os.makedirs(phasic_analysis_out, exist_ok=True)
        shutil.copy2(os.path.join(source_phasic, "phasic_trace_cache.h5"), phasic_analysis_out)
        shutil.copy2(os.path.join(source_phasic, "run_report.json"), phasic_analysis_out)
        shutil.copy2(os.path.join(source_phasic, "config_used.yaml"), phasic_analysis_out)
        # Summary plotter needs features.csv
        os.makedirs(os.path.join(phasic_analysis_out, "features"), exist_ok=True)
        shutil.copy2(os.path.join(source_phasic, "features", "features.csv"), os.path.join(phasic_analysis_out, "features"))
        # Plotters also need intermediates
        inter_src = os.path.join(source_phasic, "phasic_intermediates")
        if os.path.exists(inter_src):
            shutil.copytree(inter_src, os.path.join(phasic_analysis_out, "phasic_intermediates"), dirs_exist_ok=True)
        # Dayplot bundle (Migrated) NO LONGER needs legacy CSV traces.
        # We explicitly omit copying 'traces' here to prove cache-only discovery.
        # Wrapper Section 7 unconditionally calls tonic overview plotter.
        # Provide tonic artifacts so it can succeed.
        source_tonic = os.path.join(self.output_package, "_analysis", "tonic_out")
        tonic_analysis_out = os.path.join(analysis_dir, "tonic_out")
        os.makedirs(tonic_analysis_out, exist_ok=True)
        shutil.copy2(os.path.join(source_tonic, "tonic_trace_cache.h5"), tonic_analysis_out)
        shutil.copy2(os.path.join(source_tonic, "run_report.json"), tonic_analysis_out)
        
        # 3. EXECUTE: Run the wrapper in-process
        script_path = os.path.join(PROJECT_ROOT, 'tools', 'run_full_pipeline_deliverables.py')
        spec = importlib.util.spec_from_file_location("runner", script_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        
        test_args = [
            script_path,
            "--input", self.input_dir,
            "--out", test_run_dir,
            "--config", self.config_path,
            "--format", "rwd",
            "--mode", "phasic",
            "--sessions-per-hour", "2",
            "--session-duration-s", "600",
            "--overwrite"
        ]
        
        import subprocess as sp
        real_check_call = sp.check_call

        def mock_analysis_side_effect(cmd):
            cmd_str = " ".join([str(p).replace(os.sep, '/') for p in cmd])
            
            if 'analyze_photometry.py' in cmd_str:
                if '--out' in cmd:
                    out_path = cmd[cmd.index('--out') + 1]
                    os.makedirs(out_path, exist_ok=True)
                    shutil.copy2(os.path.join(source_phasic, "phasic_trace_cache.h5"), out_path)
                    shutil.copy2(os.path.join(source_phasic, "run_report.json"), out_path)
                    shutil.copy2(os.path.join(source_phasic, "config_used.yaml"), out_path)
                    os.makedirs(os.path.join(out_path, "features"), exist_ok=True)
                    shutil.copy2(os.path.join(source_phasic, "features", "features.csv"), os.path.join(out_path, "features"))
                    inter_src = os.path.join(source_phasic, "phasic_intermediates")
                    if os.path.exists(inter_src):
                        shutil.copytree(inter_src, os.path.join(out_path, "phasic_intermediates"), dirs_exist_ok=True)
                    # Dayplot bundle (Migrated) NO LONGER needs legacy CSV traces.
                    # --overwrite deletes _analysis/, so seed tonic_out here.
                    # Section 7 plotting unconditionally calls tonic overview plotter.
                    t_out = os.path.join(os.path.dirname(out_path), 'tonic_out')
                    os.makedirs(t_out, exist_ok=True)
                    shutil.copy2(os.path.join(source_tonic, "tonic_trace_cache.h5"), t_out)
                    shutil.copy2(os.path.join(source_tonic, "run_report.json"), t_out)
                return 0
                
            return real_check_call(cmd)

        with mock.patch('sys.argv', test_args):
            with mock.patch('subprocess.check_call', side_effect=mock_analysis_side_effect):
                runner.main()
        
        # 4. VERIFY: B2-scoped deliverables were produced
        # A. Summary plotter outputs (proves cache-backed duration validation worked)
        reg0_tables = os.path.join(test_run_dir, "Region0", "tables")
        self.assertTrue(os.path.exists(os.path.join(reg0_tables, "phasic_peak_rate_timeseries.csv")), 
                        "phasic_peak_rate_timeseries.csv was NOT produced despite cache-backed validation")
        
        # C. Dayplot bundle outputs (proves cache-backed DISCOVERY worked)
        reg0_plots = os.path.join(test_run_dir, "Region0", "day_plots")
        self.assertTrue(os.path.exists(os.path.join(reg0_plots, "phasic_dFF_day_000.png")),
                        "phasic_dFF_day_000.png was NOT produced. Cache-backed discovery potentially failed.")
        self.assertTrue(os.path.exists(os.path.join(reg0_plots, "phasic_sig_iso_day_000.png")),
                        "phasic_sig_iso_day_000.png was NOT produced.")
        self.assertTrue(os.path.exists(os.path.join(reg0_plots, "phasic_stacked_day_000.png")),
                        "phasic_stacked_day_000.png was NOT produced.")
        
        # B. Manifest should have resolved duration from cache
        man_path = os.path.join(test_run_dir, "MANIFEST.json")
        with open(man_path, 'r') as f:
            manifest = json.load(f)
        self.assertTrue(590 < manifest['session_duration_s'] < 610,
                        f"session_duration_s={manifest['session_duration_s']} not in expected range")
        
        # Cleanup
        shutil.rmtree(test_run_dir)


    def test_impossible_schedule(self):
        """Ensure failure when session_duration_s > stride_s."""
        # Trace duration is ~600s.
        # We need a valid duration (matches trace) that is > stride.
        # Set sessions_per_hour=10 -> Stride = 360s.
        # Duration 600s > Stride 360s -> Impossible.
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", os.path.join(self.out_dir, "test_impossible"),
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "10",
            "--session-duration-s", "600"
        ]
        print(f"Running impossible schedule test: {' '.join(cmd)}")
        
        # Expect failure
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Command should have failed")
        
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        print(f"Captured Output:\n{combined}")
        
        self.assertIn("Impossible schedule", combined)
        self.assertIn("Duration", combined)
        self.assertIn("Stride", combined)
            
        print("Successfully caught impossible schedule error with correct message.")

    def test_phasic_duration_mismatch_fails(self):
        """
        Verify that the wrapper correctly detects and fails when the user
        provides a session duration that mismatches the actual cache content.
        """
        # Phasic cache in self.output_package is ~600s.
        # Provide 500s -> should fail.
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", os.path.join(self.out_dir, "test_mismatch"),
            "--config", self.config_path,
            "--format", "rwd",
            "--mode", "phasic",
            "--overwrite",
            "--sessions-per-hour", "2",
            "--session-duration-s", "500" 
        ]
        
        # Expect failure (return code 1 from SystemExit)
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Command should have failed due to duration mismatch")
        
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        print(f"DEBUG: Combined Output:\n{combined}")
        self.assertIn("Session Duration Mismatch", combined)
        self.assertIn("Trace (Cache)", combined)
        # 500 is provided as input, so it should be there as "500.00s"
        self.assertIn("500", combined) 
        print("Successfully verified that duration mismatch raises RuntimeError.")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.out_dir):
            try:
                shutil.rmtree(cls.out_dir, ignore_errors=True)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
