import unittest
import os
import sys
import re
import shutil
import subprocess
import glob
import json
import pandas as pd
import numpy as np

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
class TestDeliverablesContract(unittest.TestCase):
    """
    Strictly enforces the contract defined in DELIVERABLES_CONTRACT.md.
    Uses --out-base mode (GUI mode) for the primary run.
    """

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.out_dir):
            try:
                shutil.rmtree(cls.out_dir, ignore_errors=True)
            except OSError:
                pass

    @classmethod
    def setUpClass(cls):
        import tempfile
        # 1. Clean/Create Directories
        cls.out_dir = tempfile.mkdtemp(prefix="out_contract_test_")
        cls.input_dir = os.path.join(cls.out_dir, "input_RWD")
        cls.config_path = os.path.join(cls.out_dir, "config.yaml")

        if not os.path.exists(cls.out_dir):
             os.makedirs(cls.out_dir)

        # 2. Setup Config (same pattern as test_full_pipeline_deliverables)
        orig_config = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
        shutil.copy2(orig_config, cls.config_path)

        # 3. Generate Synthetic Data
        #    Mirrors test_full_pipeline_deliverables.py flags exactly,
        #    except 1 day / 1 ROI to keep the contract test fast.
        print("\n[Contract] Generating synthetic data...")
        gen_cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", cls.input_dir,
            "--format", "rwd",
            "--config", cls.config_path,
            "--total-days", "1.0",
            "--recordings-per-hour", "2",
            "--recording-duration-min", "10.0",
            "--n-rois", "1",
            "--phasic-mode", "phase_locked_to_tonic",
            "--seed", "999",
            "--preset", "biological_shared_nuisance"
        ]
        subprocess.check_call(gen_cmd)

        # 4. Run Pipeline in GUI mode (--out-base + --run-id)
        cls.out_base = os.path.join(cls.out_dir, "out_base")
        cls.run_id = "run_contract_test"
        cls.output_package = os.path.join(cls.out_base, cls.run_id)
        print("[Contract] Running pipeline entrypoint (GUI mode)...")
        run_cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", cls.input_dir,
            "--out-base", cls.out_base,
            "--run-id", cls.run_id,
            "--config", cls.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "2",
            "--events", "auto",
            "--cancel-flag", "auto",
        ]
        subprocess.check_call(run_cmd)

    # ------------------------------------------------------------------
    # Helper: load manifest (each test is independent, no cross-test state)
    # ------------------------------------------------------------------
    def _load_manifest(self):
        man_path = os.path.join(self.output_package, "MANIFEST.json")
        self.assertTrue(os.path.exists(man_path), "MANIFEST.json must exist")
        with open(man_path, 'r') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Test 01: Top-level manifest schema + types + values
    # ------------------------------------------------------------------
    def test_01_manifest_schema(self):
        """Enforce MANIFEST.json required keys, types, and expected values."""
        manifest = self._load_manifest()

        required_keys = [
            "tool", "timestamp", "args", "commands",
            "regions", "deliverables",
            "sessions_per_hour", "session_duration_s", "session_stride_s",
            # New engine fields
            "run_id", "run_dir", "events_path", "cancel_flag_path",
        ]
        for key in required_keys:
            self.assertIn(key, manifest, f"MANIFEST missing required key: {key}")

        # Type checks
        self.assertIsInstance(manifest['tool'], str)
        self.assertIsInstance(manifest['timestamp'], str)
        self.assertIsInstance(manifest['args'], dict)
        self.assertIsInstance(manifest['commands'], list)
        self.assertGreaterEqual(len(manifest['commands']), 1,
                                "commands must have at least one entry")
        for i, cmd in enumerate(manifest['commands']):
            self.assertIsInstance(cmd, list,
                                 f"commands[{i}] must be an argv list, got {type(cmd)}")
            self.assertGreaterEqual(len(cmd), 1,
                                   f"commands[{i}] argv list must not be empty")
            for j, arg in enumerate(cmd):
                self.assertIsInstance(arg, str,
                                     f"commands[{i}][{j}] must be str, got {type(arg)}")
        self.assertIsInstance(manifest['regions'], list)
        self.assertIsInstance(manifest['deliverables'], dict)
        self.assertIsInstance(manifest['sessions_per_hour'], int)
        self.assertIsInstance(manifest['session_duration_s'], (int, float))
        self.assertIsInstance(manifest['session_stride_s'], (int, float))

        # Stable tool identifier
        self.assertEqual(manifest['tool'], 'run_full_pipeline_deliverables')

        # Value checks for 2 sessions/hr synthetic data
        self.assertEqual(manifest['sessions_per_hour'], 2)
        self.assertAlmostEqual(manifest['session_stride_s'], 1800.0, places=1)

        dur = manifest['session_duration_s']
        tol = max(2.0, 0.005 * dur)
        self.assertLess(abs(dur - 600.0), tol,
                        f"session_duration_s {dur} not ~600s")

        # New engine field assertions
        self.assertEqual(manifest['run_id'], self.run_id)
        self.assertEqual(
            os.path.normcase(os.path.normpath(manifest['run_dir'])),
            os.path.normcase(os.path.normpath(os.path.abspath(self.output_package))),
        )
        self.assertIsInstance(manifest['events_path'], str)
        self.assertIsInstance(manifest['cancel_flag_path'], str)

    # ------------------------------------------------------------------
    # Test 02: Per-ROI manifest schema (keys + types)
    # ------------------------------------------------------------------
    def test_02_per_roi_schema(self):
        """Enforce per-ROI deliverables dict has correct keys and types."""
        manifest = self._load_manifest()

        self.assertTrue(len(manifest['regions']) > 0, "No regions in manifest")

        for roi in manifest['regions']:
            self.assertIn(roi, manifest['deliverables'],
                          f"ROI '{roi}' listed in regions but missing from deliverables")
            d = manifest['deliverables'][roi]
            self.assertIsInstance(d, dict, f"deliverables[{roi}] is not a dict")

            # Required keys
            self.assertIn('diagnostic_chunk_id', d,
                          f"deliverables[{roi}] missing 'diagnostic_chunk_id'")
            self.assertIsInstance(d['diagnostic_chunk_id'], int,
                                 f"diagnostic_chunk_id must be int, got {type(d['diagnostic_chunk_id'])}")

            self.assertIn('files', d,
                          f"deliverables[{roi}] missing 'files'")
            self.assertIsInstance(d['files'], list,
                                 f"files must be list, got {type(d['files'])}")
            for f in d['files']:
                self.assertIsInstance(f, str, f"files entry must be str, got {type(f)}")

            self.assertIn('days_generated', d,
                          f"deliverables[{roi}] missing 'days_generated'")
            self.assertIsInstance(d['days_generated'], list,
                                 f"days_generated must be list, got {type(d['days_generated'])}")
            for day in d['days_generated']:
                self.assertIsInstance(day, str, f"days_generated entry must be str, got {type(day)}")

    # ------------------------------------------------------------------
    # Test 03: Required files exist on disk AND in manifest files list
    # ------------------------------------------------------------------
    def test_03_required_files(self):
        """Enforce required deliverable files exist and are recorded."""
        manifest = self._load_manifest()

        required_basenames = [
            "tonic_overview.png",
            "tonic_df_timeseries.csv",
            "phasic_correction_impact.png",
            "phasic_correction_impact_session.csv",
            "phasic_peak_rate_timeseries.png",
            "phasic_peak_rate_timeseries.csv",
            "phasic_auc_timeseries.png",
            "phasic_auc_timeseries.csv"
        ]

        for roi in manifest['regions']:
            roi_path = os.path.join(self.output_package, roi)
            self.assertTrue(os.path.isdir(roi_path), f"ROI folder missing: {roi}")
            files_list = manifest['deliverables'][roi]['files']

            for fname in required_basenames:
                fpath = os.path.join(roi_path, fname)
                self.assertTrue(os.path.exists(fpath),
                                f"Missing deliverable on disk: {fname} in {roi}")
                self.assertIn(fname, files_list,
                              f"File exists on disk but NOT in manifest files list: {fname}")

    # ------------------------------------------------------------------
    # Test 04: Day-set equality (manifest vs actual files)
    # ------------------------------------------------------------------
    def test_04_day_set_consistency(self):
        """Enforce days_generated matches actual day files on disk,
        and all three day-file types exist for every day."""
        manifest = self._load_manifest()

        day_patterns = {
            'sig_iso': 'phasic_sig_iso_day_*.png',
            'dff':     'phasic_dFF_day_*.png',
            'stacked': 'phasic_stacked_day_*.png',
        }

        day_re = {
            'sig_iso': re.compile(r'^phasic_sig_iso_day_(\d+)\.png$'),
            'dff':     re.compile(r'^phasic_dFF_day_(\d+)\.png$'),
            'stacked': re.compile(r'^phasic_stacked_day_(\d+)\.png$'),
        }

        for roi in manifest['regions']:
            roi_path = os.path.join(self.output_package, roi)
            days_manifest = set(manifest['deliverables'][roi]['days_generated'])
            files_list = manifest['deliverables'][roi]['files']

            # Collect days from each pattern on disk
            days_on_disk = set()
            per_type_days = {}
            for ptype, pattern in day_patterns.items():
                found = set()
                for fpath in glob.glob(os.path.join(roi_path, pattern)):
                    m = day_re[ptype].match(os.path.basename(fpath))
                    if m:
                        found.add(m.group(1))
                per_type_days[ptype] = found
                days_on_disk |= found

            # days_generated must equal union of all day files found on disk
            self.assertEqual(days_manifest, days_on_disk,
                             f"ROI {roi}: days_generated {sorted(days_manifest)} "
                             f"!= days on disk {sorted(days_on_disk)}")

            # Every day must have all three file types
            for day in sorted(days_on_disk):
                for ptype, day_set in per_type_days.items():
                    self.assertIn(day, day_set,
                                  f"ROI {roi}: day {day} missing {ptype} file")

            # Every day file must be recorded in manifest files list
            for day in sorted(days_on_disk):
                for ptype, regex in day_re.items():
                    # Reconstruct filename
                    if ptype == 'sig_iso':
                        fname = f"phasic_sig_iso_day_{day}.png"
                    elif ptype == 'dff':
                        fname = f"phasic_dFF_day_{day}.png"
                    elif ptype == 'stacked':
                        fname = f"phasic_stacked_day_{day}.png"
                    self.assertIn(fname, files_list,
                                  f"ROI {roi}: day file {fname} on disk but not in manifest files list")

    # ------------------------------------------------------------------
    # Test 05: CSV schema enforcement
    # ------------------------------------------------------------------
    def test_05_csv_schemas(self):
        """Enforce CSV column contracts for all ROIs."""
        manifest = self._load_manifest()

        for roi in manifest['regions']:
            roi_path = os.path.join(self.output_package, roi)

            # 1. Tonic
            df_tonic = pd.read_csv(os.path.join(roi_path, "tonic_df_timeseries.csv"))
            for c in ["time_hours", "tonic_df"]:
                self.assertIn(c, df_tonic.columns, f"[{roi}] tonic CSV missing '{c}'")

            # 2. Phasic Peak Rate
            df_peak = pd.read_csv(os.path.join(roi_path, "phasic_peak_rate_timeseries.csv"))
            for c in ["time_hours", "day", "hour", "session_in_hour",
                       "window_seconds", "peak_rate_per_min"]:
                self.assertIn(c, df_peak.columns, f"[{roi}] peak CSV missing '{c}'")
            has_count = ("peak_count" in df_peak.columns) or ("n_peaks" in df_peak.columns)
            self.assertTrue(has_count,
                            f"[{roi}] phasic_peak_rate_timeseries.csv missing peak_count/n_peaks")

            # 3. Phasic AUC
            df_auc = pd.read_csv(os.path.join(roi_path, "phasic_auc_timeseries.csv"))
            for c in ["time_hours", "day", "hour", "session_in_hour",
                       "window_seconds", "auc_above_threshold_dff_s"]:
                self.assertIn(c, df_auc.columns, f"[{roi}] AUC CSV missing '{c}'")

    # ------------------------------------------------------------------
    # Test 06: window_seconds == session_duration_s (both CSVs)
    # ------------------------------------------------------------------
    def test_06_window_seconds_consistency(self):
        """Enforce window_seconds matches MANIFEST session_duration_s in both CSVs."""
        manifest = self._load_manifest()
        duration = manifest['session_duration_s']
        tol = max(2.0, 0.005 * duration)

        for roi in manifest['regions']:
            roi_path = os.path.join(self.output_package, roi)

            for csv_name in ["phasic_peak_rate_timeseries.csv",
                             "phasic_auc_timeseries.csv"]:
                df = pd.read_csv(os.path.join(roi_path, csv_name))
                ws = df['window_seconds'].iloc[0]
                diff = abs(ws - duration)
                self.assertLess(diff, tol,
                                f"[{roi}] {csv_name} window_seconds={ws} "
                                f"!= manifest duration={duration} (tol={tol})")

    # ------------------------------------------------------------------
    # Test 07: Impossible schedule rule
    # ------------------------------------------------------------------
    def test_07_impossible_schedule_rule(self):
        """Enforce fatal error when session_duration_s > stride_s."""
        # Trace duration ~600s. 10 sessions/hr -> stride 360s.
        # 600 > 360 -> Impossible schedule.
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", os.path.join(self.out_dir, "impossible_test"),
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "10",
            "--session-duration-s", "600"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0,
                            "Pipeline should fail on impossible schedule")
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        self.assertIn("Impossible schedule", combined,
                      f"Expected 'Impossible schedule' in output, got:\n{combined[:500]}")

    # ------------------------------------------------------------------
    # Test 08: --validate-only success (fast, no analysis) -- legacy mode
    # ------------------------------------------------------------------
    def test_08_validate_only_success(self):
        """--validate-only exits 0 and prints OK without running analysis."""
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", os.path.join(self.out_dir, "validate_only_dummy_out"),
            "--config", self.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "2",
            "--session-duration-s", "600",
            "--validate-only"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"validate-only should exit 0, stderr:\n{result.stderr[:500]}")
        self.assertIn("VALIDATE-ONLY: OK", result.stdout)
        self.assertIn("VALIDATE-ONLY: argv=", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

        # Legacy validate-only must NOT create the output directory
        vo_out = os.path.join(self.out_dir, "validate_only_dummy_out")
        self.assertFalse(os.path.exists(vo_out),
                         f"Legacy validate-only must not create run_dir: {vo_out}")

    # ------------------------------------------------------------------
    # Test 09: --validate-only impossible schedule (fast fail)
    # ------------------------------------------------------------------
    def test_09_validate_only_impossible_schedule(self):
        """--validate-only catches impossible schedule without running analysis."""
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", os.path.join(self.out_dir, "validate_only_impossible"),
            "--config", self.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "10",
            "--session-duration-s", "600",
            "--validate-only"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0,
                            "validate-only should fail on impossible schedule")
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        self.assertIn("Impossible schedule", combined)
        self.assertIn("Duration", combined)
        self.assertIn("Stride", combined)

    # ------------------------------------------------------------------
    # Test 10: Events file exists and is parseable NDJSON
    # ------------------------------------------------------------------
    def test_10_events_file_parseable(self):
        """Verify events.ndjson exists, is valid NDJSON, and has engine start/done."""
        manifest = self._load_manifest()
        events_path = manifest['events_path']
        self.assertTrue(os.path.isfile(events_path),
                        f"Events file does not exist: {events_path}")

        events = []
        with open(events_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON on line {line_num}: {e}\n  Line: {line[:200]}")
                # Check required fields
                for field in ["schema_version", "time_iso", "run_id",
                              "run_dir", "stage", "type", "message"]:
                    self.assertIn(field, obj,
                                  f"Event on line {line_num} missing field '{field}'")
                events.append(obj)

        self.assertGreater(len(events), 0, "Events file is empty")

        # Check engine start event
        engine_starts = [e for e in events
                         if e['stage'] == 'engine' and e['type'] == 'start']
        self.assertGreaterEqual(len(engine_starts), 1,
                                "No engine start event found")

        # Check engine done event
        engine_dones = [e for e in events
                        if e['stage'] == 'engine' and e['type'] == 'done']
        self.assertGreaterEqual(len(engine_dones), 1,
                                "No engine done event found")

    # ------------------------------------------------------------------
    # Test 11: Cancel flag pre-exists causes deterministic cancellation
    # ------------------------------------------------------------------
    def test_11_cancel_flag_deterministic(self):
        """Pre-existing cancel flag causes exit 130 and cancelled manifest."""
        cancel_out_base = os.path.join(self.out_dir, "cancel_out_base")
        cancel_run_id = "run_cancel_test"
        cancel_run_dir = os.path.join(cancel_out_base, cancel_run_id)

        # Pre-create run_dir and cancel flag
        os.makedirs(cancel_run_dir, exist_ok=True)
        cancel_flag = os.path.join(cancel_run_dir, "CANCEL.REQUESTED")
        with open(cancel_flag, "w") as f:
            f.write("cancel\n")

        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out-base", cancel_out_base,
            "--run-id", cancel_run_id,
            "--config", self.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "2",
            "--events", "auto",
            "--cancel-flag", "auto",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 130,
                         f"Cancelled run should exit 130, got {result.returncode}\n"
                         f"stdout:\n{result.stdout[:500]}\nstderr:\n{result.stderr[:500]}")

        # MANIFEST.json must exist
        man_path = os.path.join(cancel_run_dir, "MANIFEST.json")
        self.assertTrue(os.path.exists(man_path),
                        "Cancelled run should produce MANIFEST.json")
        with open(man_path, 'r') as f:
            manifest = json.load(f)
        # self.assertEqual(manifest['status'], 'cancelled') # REMOVED per contract

        # Events file must contain a cancelled event
        events_path = os.path.join(cancel_run_dir, "events.ndjson")
        self.assertTrue(os.path.isfile(events_path),
                        "Cancelled run should produce events.ndjson")
        cancelled_events = []
        with open(events_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get('type') == 'cancelled':
                    cancelled_events.append(obj)
        self.assertGreaterEqual(len(cancelled_events), 1,
                                "No cancelled event found in events.ndjson")
        self.assertEqual(cancelled_events[0].get('error_code'), 'CANCELLED')

    # ------------------------------------------------------------------
    # Test 12: Validate-only in GUI mode (no MANIFEST.json)
    # ------------------------------------------------------------------
    def test_12_validate_only_gui_mode(self):
        """Validate-only in --out-base mode must not create MANIFEST.json."""
        vo_out_base = os.path.join(self.out_dir, "vo_gui_base")
        vo_run_id = "run_validate_only_test"
        vo_run_dir = os.path.join(vo_out_base, vo_run_id)

        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out-base", vo_out_base,
            "--run-id", vo_run_id,
            "--config", self.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "2",
            "--validate-only",
            "--events", "auto",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0,
                         f"validate-only should exit 0, stderr:\n{result.stderr[:500]}")
        self.assertIn("VALIDATE-ONLY: OK", result.stdout)
        self.assertIn(f"VALIDATE-ONLY: run_dir=", result.stdout)
        self.assertIn(f"VALIDATE-ONLY: events_path=", result.stdout)
        self.assertIn(f"VALIDATE-ONLY: cancel_flag_path=", result.stdout)

        # MANIFEST.json must NOT exist
        man_path = os.path.join(vo_run_dir, "MANIFEST.json")
        self.assertFalse(os.path.exists(man_path),
                         "validate-only must not create MANIFEST.json")

        # run_dir may exist (for auto paths), that is fine
        # If events file exists, it must be parseable
        events_path = os.path.join(vo_run_dir, "events.ndjson")
        if os.path.exists(events_path):
            with open(events_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        self.fail(f"Invalid JSON in validate-only events, line {line_num}: {e}")

    # ------------------------------------------------------------------
    # Test 13: Legacy validate-only must not create directories
    # ------------------------------------------------------------------
    def test_13_validate_only_legacy_no_side_effects(self):
        """Legacy --out + --validate-only must not create run_dir or events.

        Case A: --events auto (default).
        Case B: --events pointing inside non-existent out_path.
        """
        # --- Case A: --events auto ---
        out_path_a = os.path.join(self.out_dir, "vo_legacy_case_a")
        if os.path.exists(out_path_a):
            shutil.rmtree(out_path_a)

        cmd_a = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", out_path_a,
            "--config", self.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "2",
            "--validate-only",
        ]
        result_a = subprocess.run(cmd_a, capture_output=True, text=True)

        self.assertEqual(result_a.returncode, 0,
                         f"Case A: exit 0 expected, stderr:\n{result_a.stderr[:500]}")
        self.assertIn("VALIDATE-ONLY: OK", result_a.stdout)
        self.assertFalse(os.path.exists(out_path_a),
                         "Case A: out_path must not be created")
        self.assertIn("events disabled", result_a.stdout,
                      "Case A: must print events disabled message")
        # Message must mention the no-directory guarantee
        self.assertIn("no directories will be created", result_a.stdout,
                      "Case A: message must state no directories will be created")
        # Message must mention the parent-directory requirement
        self.assertIn("existing parent directory", result_a.stdout,
                      "Case A: message must mention existing parent directory")
        self.assertIn("VALIDATE-ONLY: events_path=None", result_a.stdout,
                      "Case A: events_path must be None")

        # --- Case B: explicit --events inside non-existent out_path ---
        out_path_b = os.path.join(self.out_dir, "vo_legacy_case_b")
        if os.path.exists(out_path_b):
            shutil.rmtree(out_path_b)
        explicit_events = os.path.join(out_path_b, "events.ndjson")

        cmd_b = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", out_path_b,
            "--config", self.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "2",
            "--validate-only",
            "--events", explicit_events,
        ]
        result_b = subprocess.run(cmd_b, capture_output=True, text=True)

        self.assertEqual(result_b.returncode, 0,
                         f"Case B: exit 0 expected, stderr:\n{result_b.stderr[:500]}")
        self.assertIn("VALIDATE-ONLY: OK", result_b.stdout)
        self.assertFalse(os.path.exists(out_path_b),
                         "Case B: out_path must not be created")
        self.assertIn("parent directory does not exist", result_b.stdout,
                      "Case B: must state parent directory does not exist")
        self.assertIn("will not be created", result_b.stdout,
                      "Case B: must state will not be created")
        self.assertIn("creates no directories", result_b.stdout,
                      "Case B: must state creates no directories")
        self.assertIn("VALIDATE-ONLY: events_path=None", result_b.stdout,
                      "Case B: events_path must be None")

    # ------------------------------------------------------------------
    # Test 14: Legacy validate-only with explicit events to existing dir
    # ------------------------------------------------------------------
    def test_14_validate_only_legacy_events_existing_dir(self):
        """Legacy validate-only with --events pointing to existing dir writes events."""
        out_path = os.path.join(self.out_dir, "vo_legacy_case_c_out")
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        # Create an existing directory for events (NOT out_path)
        events_dir = os.path.join(self.out_dir, "vo_legacy_case_c_events")
        os.makedirs(events_dir, exist_ok=True)
        events_file = os.path.join(events_dir, "events.ndjson")
        # Remove stale events if present
        if os.path.exists(events_file):
            os.remove(events_file)

        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", out_path,
            "--config", self.config_path,
            "--format", "rwd",
            "--sessions-per-hour", "2",
            "--validate-only",
            "--events", events_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0,
                         f"Case C: exit 0 expected, stderr:\n{result.stderr[:500]}")
        self.assertIn("VALIDATE-ONLY: OK", result.stdout)

        # out_path must NOT be created
        self.assertFalse(os.path.exists(out_path),
                         "Case C: out_path must not be created")

        # events file should have been written (parent existed)
        self.assertTrue(os.path.isfile(events_file),
                        f"Case C: events file should exist at {events_file}")

        # Parse events using context manager
        events = []
        with open(events_file, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as e:
                    self.fail(f"Case C: invalid JSON on line {line_num}: {e}")
                events.append(obj)

        self.assertGreater(len(events), 0, "Case C: events file is empty")
        engine_starts = [e for e in events
                         if e.get("stage") == "engine" and e.get("type") == "start"]
        self.assertGreaterEqual(len(engine_starts), 1,
                                "Case C: missing engine start event")
        engine_dones = [e for e in events
                        if e.get("stage") == "engine" and e.get("type") == "done"]
        self.assertGreaterEqual(len(engine_dones), 1,
                                "Case C: missing engine done event")

        # events_path should be reported in stdout (not None)
        self.assertIn(f"VALIDATE-ONLY: events_path={events_file}", result.stdout,
                      "Case C: events_path should be the explicit path")

    # ------------------------------------------------------------------
    # Test 15: Contract invariants for legacy validate-only
    # ------------------------------------------------------------------
    def test_15_validate_only_contract_invariants(self):
        """Lock legacy validate-only side-effect rules in one place."""

        # --- Invariant 1: legacy, --events auto, no dirs ---
        out_a = os.path.join(self.out_dir, "vo_invariants_a")
        if os.path.exists(out_a):
            shutil.rmtree(out_a)

        r1 = subprocess.run([
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir, "--out", out_a,
            "--config", self.config_path, "--format", "rwd",
            "--sessions-per-hour", "2", "--validate-only",
        ], capture_output=True, text=True)

        self.assertEqual(r1.returncode, 0,
                         f"Inv1: exit 0 expected, stderr:\n{r1.stderr[:500]}")
        self.assertFalse(os.path.exists(out_a),
                         "Inv1: out_path must not be created")
        self.assertIn("events disabled", r1.stdout,
                      "Inv1: must print events disabled")
        self.assertIn("existing parent directory", r1.stdout,
                      "Inv1: must mention existing parent directory")
        self.assertIn("no directories will be created", r1.stdout,
                      "Inv1: must state no directories will be created")
        self.assertIn("VALIDATE-ONLY: events_path=None", r1.stdout,
                      "Inv1: events_path must be None")

        # --- Invariant 2: legacy, explicit events to existing dir ---
        out_b = os.path.join(self.out_dir, "vo_invariants_b")
        if os.path.exists(out_b):
            shutil.rmtree(out_b)

        events_dir = os.path.join(self.out_dir, "vo_invariants_events_dir")
        os.makedirs(events_dir, exist_ok=True)
        events_file = os.path.join(events_dir, "events.ndjson")
        if os.path.exists(events_file):
            os.remove(events_file)

        r2 = subprocess.run([
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir, "--out", out_b,
            "--config", self.config_path, "--format", "rwd",
            "--sessions-per-hour", "2", "--validate-only",
            "--events", events_file,
        ], capture_output=True, text=True)

        self.assertEqual(r2.returncode, 0,
                         f"Inv2: exit 0 expected, stderr:\n{r2.stderr[:500]}")
        self.assertFalse(os.path.exists(out_b),
                         "Inv2: out_path must not be created")
        self.assertTrue(os.path.isfile(events_file),
                        "Inv2: events file must exist")

        events = []
        with open(events_file, "r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    events.append(json.loads(s))
                except json.JSONDecodeError as e:
                    self.fail(f"Inv2: bad JSON line {ln}: {s[:80]}... {e}")
        starts = [e for e in events
                  if e.get("stage") == "engine" and e.get("type") == "start"]
        dones = [e for e in events
                 if e.get("stage") == "engine" and e.get("type") == "done"]
        self.assertGreaterEqual(len(starts), 1, "Inv2: missing engine start")
        self.assertGreaterEqual(len(dones), 1, "Inv2: missing engine done")

        # --- Invariant 3: legacy, explicit events to missing parent ---
        out_c = os.path.join(self.out_dir, "vo_invariants_c")
        if os.path.exists(out_c):
            shutil.rmtree(out_c)
        events_c = os.path.join(out_c, "events.ndjson")

        r3 = subprocess.run([
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir, "--out", out_c,
            "--config", self.config_path, "--format", "rwd",
            "--sessions-per-hour", "2", "--validate-only",
            "--events", events_c,
        ], capture_output=True, text=True)

        self.assertEqual(r3.returncode, 0,
                         f"Inv3: exit 0 expected, stderr:\n{r3.stderr[:500]}")
        self.assertFalse(os.path.exists(out_c),
                         "Inv3: out_path must not be created")
        self.assertFalse(os.path.exists(events_c),
                         "Inv3: events file must not be created")
        self.assertIn("parent directory does not exist", r3.stdout,
                      "Inv3: must state parent dir missing")
        self.assertIn("will not be created", r3.stdout,
                      "Inv3: must state will not be created")
        self.assertIn("creates no directories", r3.stdout,
                      "Inv3: must state creates no directories")
        self.assertIn("VALIDATE-ONLY: events_path=None", r3.stdout,
                      "Inv3: events_path must be None")

    # ------------------------------------------------------------------
    # Test 16: Static guard — no os.makedirs in legacy validate-only path
    # ------------------------------------------------------------------
    def test_16_validate_only_no_makedirs_in_legacy_path(self):
        """Validate-only block must not call os.makedirs outside is_gui_mode."""
        script = os.path.join(PROJECT_ROOT,
                              "tools", "run_full_pipeline_deliverables.py")
        with open(script, "r", encoding="utf-8") as fh:
            source = fh.read()

        # Extract validate-only block
        start_marker = "if args.validate_only:"
        end_marker = "# 1. Setup run directory"
        start_idx = source.find(start_marker)
        self.assertGreater(start_idx, -1,
                           "Could not find 'if args.validate_only:' marker")
        end_idx = source.find(end_marker, start_idx)
        self.assertGreater(end_idx, start_idx,
                           "Could not find '# 1. Setup run directory' marker")

        vo_block = source[start_idx:end_idx]

        if "os.makedirs(" in vo_block:
            # os.makedirs is present; it must be under is_gui_mode
            gui_check_pos = vo_block.find("if is_gui_mode")
            makedirs_pos = vo_block.find("os.makedirs(")
            self.assertGreater(
                gui_check_pos, -1,
                "validate-only block has os.makedirs but no 'if is_gui_mode' "
                "guard — legacy validate-only must not create directories"
            )
            self.assertLess(
                gui_check_pos, makedirs_pos,
                "os.makedirs appears BEFORE 'if is_gui_mode' in validate-only "
                "block — legacy validate-only must not create directories"
            )


if __name__ == '__main__':
    unittest.main()


