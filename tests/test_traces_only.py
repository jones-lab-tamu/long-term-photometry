import unittest
import os
import sys
import shutil
import subprocess
import json
import tempfile
import uuid
import re
import glob

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestTracesOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_dir = tempfile.mkdtemp(prefix="out_traces_only_")
        cls.input_dir = os.path.join(cls.out_dir, "input_RWD")
        cls.config_path = os.path.join(cls.out_dir, "qc_universal_config.yaml")

        orig_config_path = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
        shutil.copy2(orig_config_path, cls.config_path)

        # Synthetic Data (small: 0.5 days, 2 ROIs for speed)
        print("Generating synthetic data ...")
        gen_cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", cls.input_dir,
            "--format", "rwd",
            "--config", cls.config_path,
            "--total-days", "0.5",
            "--recordings-per-hour", "2",
            "--recording-duration-min", "10.0",
            "--n-rois", "2",
            "--seed", "99",
            "--preset", "biological_shared_nuisance"
        ]
        subprocess.check_call(gen_cmd)

    # ----------------------------------------------------------------
    # Test 1: traces-only completes, traces exist, features absent
    # ----------------------------------------------------------------
    def test_traces_only_completes_and_omits_features(self):
        output_package = os.path.join(self.out_dir, f"package_to_{uuid.uuid4().hex[:8]}")
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2",
            "--traces-only"
        ]
        subprocess.check_call(cmd)

        # status.json ends in success
        status = json.load(open(os.path.join(output_package, "status.json")))
        self.assertEqual(status["status"], "success")
        self.assertFalse(status["features_extracted"])
        self.assertTrue(status["traces_only"])

        # Analysis dirs exist
        phasic_out = os.path.join(output_package, "_analysis", "phasic_out")
        tonic_out = os.path.join(output_package, "_analysis", "tonic_out")
        self.assertTrue(os.path.isdir(phasic_out))
        self.assertTrue(os.path.isdir(tonic_out))

        # Traces exist
        phasic_traces = os.path.join(phasic_out, "traces")
        tonic_traces = os.path.join(tonic_out, "traces")
        self.assertTrue(os.path.isdir(phasic_traces))
        self.assertTrue(os.path.isdir(tonic_traces))
        self.assertTrue(len(os.listdir(phasic_traces)) > 0, "phasic traces should exist")
        self.assertTrue(len(os.listdir(tonic_traces)) > 0, "tonic traces should exist")

        # QC exists
        phasic_qc = os.path.join(phasic_out, "qc")
        tonic_qc = os.path.join(tonic_out, "qc")
        self.assertTrue(os.path.isdir(phasic_qc))
        self.assertTrue(os.path.isdir(tonic_qc))

        # Feature outputs are ABSENT
        phasic_features = os.path.join(phasic_out, "features", "features.csv")
        tonic_features = os.path.join(tonic_out, "features", "features.csv")
        self.assertFalse(os.path.exists(phasic_features), "phasic features.csv should NOT exist in traces-only")
        self.assertFalse(os.path.exists(tonic_features), "tonic features.csv should NOT exist in traces-only")

        # Event-detection outputs are ABSENT (in this project, features.csv IS the event detection output;
        # no separate events.csv, detected_events.csv, or bouts.csv exist)
        for lane in [phasic_out, tonic_out]:
            for evt_name in ["events.csv", "detected_events.csv", "event_table.csv", "bouts.csv"]:
                self.assertFalse(
                    os.path.exists(os.path.join(lane, evt_name)),
                    f"{evt_name} should NOT exist in traces-only ({lane})"
                )
            # features directory should either not exist or be empty
            feat_dir = os.path.join(lane, "features")
            if os.path.isdir(feat_dir):
                self.assertEqual(len(os.listdir(feat_dir)), 0,
                                 f"features/ dir should be empty in traces-only ({lane})")

        # Region dirs still produced
        region_dirs = [d for d in os.listdir(output_package)
                       if os.path.isdir(os.path.join(output_package, d))
                       and d not in ("_analysis",) and not d.startswith(".")]
        self.assertTrue(len(region_dirs) > 0, "Region deliverable dirs should exist")

    # ----------------------------------------------------------------
    # Test 2: metadata contract (run_report.json, events.ndjson)
    # ----------------------------------------------------------------
    def test_traces_only_metadata_contract(self):
        output_package = os.path.join(self.out_dir, f"package_meta_{uuid.uuid4().hex[:8]}")
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2",
            "--traces-only"
        ]
        subprocess.check_call(cmd)

        # run_report.json
        phasic_out = os.path.join(output_package, "_analysis", "phasic_out")
        report = json.load(open(os.path.join(phasic_out, "run_report.json")))
        rc = report["run_context"]
        self.assertFalse(rc["features_extracted"], "run_report features_extracted should be false")
        self.assertTrue(rc["traces_only"], "run_report traces_only should be true")
        self.assertEqual(rc["run_type"], "full")

        # events.ndjson
        events_path = os.path.join(output_package, "events.ndjson")
        self.assertTrue(os.path.exists(events_path))
        events = [json.loads(line) for line in open(events_path)]

        # Find the engine:context event
        ctx_events = [e for e in events if e.get("stage") == "engine" and e.get("type") == "context"]
        self.assertTrue(len(ctx_events) >= 1, "engine:context event should exist")
        payload = ctx_events[0].get("payload", {})
        self.assertFalse(payload.get("features_extracted"), "engine:context features_extracted should be false")
        self.assertTrue(payload.get("traces_only"), "engine:context traces_only should be true")

    # ----------------------------------------------------------------
    # Test 3: normal run produces features, features_extracted is null at generation time
    # ----------------------------------------------------------------
    def test_normal_run_features_extracted_null(self):
        output_package = os.path.join(self.out_dir, f"package_full_{uuid.uuid4().hex[:8]}")
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2"
        ]
        subprocess.check_call(cmd)

        # Feature file exists
        phasic_out = os.path.join(output_package, "_analysis", "phasic_out")
        feats = os.path.join(phasic_out, "features", "features.csv")
        self.assertTrue(os.path.exists(feats), "features.csv should exist in full run")

        # run_report.json says features_extracted is null (not true)
        report = json.load(open(os.path.join(phasic_out, "run_report.json")))
        rc = report["run_context"]
        self.assertIsNone(rc["features_extracted"],
                          "run_report features_extracted should be null (None) in normal mode at generation time")
        self.assertFalse(rc["traces_only"], "run_report traces_only should be false in full run")

        # status.json also has features_extracted null
        status = json.load(open(os.path.join(output_package, "status.json")))
        self.assertIsNone(status["features_extracted"],
                          "status.json features_extracted should be null in normal mode")
        self.assertFalse(status["traces_only"])

    def test_underscore_roi_packaging(self):
        """Regression test: ROI names with underscores must survive end-to-end.
        This would FAIL if ROI names were derived by header splitting instead
        of from the adapter's base-name extraction via roi_selection."""
        # 1. Create mutated input: rename Region0 -> Region_0 in column names
        us_input = os.path.join(self.out_dir, "input_underscore")
        if os.path.isdir(us_input):
            shutil.rmtree(us_input)
        shutil.copytree(self.input_dir, us_input)

        rename_map = {"Region0": "Region_0", "Region1": "Region_1"}
        for root, dirs, files in os.walk(us_input):
            for fn in files:
                if fn == "fluorescence.csv":
                    fpath = os.path.join(root, fn)
                    df = pd.read_csv(fpath)
                    new_cols = {}
                    for col in df.columns:
                        new_name = col
                        for old, new in rename_map.items():
                            new_name = new_name.replace(old, new)
                        new_cols[col] = new_name
                    df = df.rename(columns=new_cols)
                    df.to_csv(fpath, index=False)

        # 2. Run traces-only pipeline on mutated input
        output_package = os.path.join(self.out_dir, f"package_us_{uuid.uuid4().hex[:8]}")
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", us_input,
            "--out", output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2",
            "--traces-only"
        ]
        subprocess.check_call(cmd)

        # 3. Assert roi_selection contains underscore names
        phasic_out = os.path.join(output_package, "_analysis", "phasic_out")
        report = json.load(open(os.path.join(phasic_out, "run_report.json")))
        expected_rois = sorted(report["roi_selection"]["selected_rois"])
        self.assertEqual(expected_rois, ["Region_0", "Region_1"],
                         "roi_selection.selected_rois must contain underscore ROI names")

        # 4. Assert region directories use exact underscore names
        region_dirs = sorted([d for d in os.listdir(output_package)
                              if os.path.isdir(os.path.join(output_package, d))
                              and d not in ("_analysis",) and not d.startswith(".")])
        self.assertEqual(region_dirs, ["Region_0", "Region_1"],
                         "Region output dirs must match exact underscore ROI names")

        # 5. Assert MANIFEST.json regions match
        manifest_path = os.path.join(output_package, "MANIFEST.json")
        self.assertTrue(os.path.exists(manifest_path), "MANIFEST.json should exist")
        manifest = json.load(open(manifest_path))
        self.assertEqual(sorted(manifest.get("regions", [])), ["Region_0", "Region_1"],
                         "MANIFEST regions must match underscore ROI names")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.out_dir):
            shutil.rmtree(cls.out_dir, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()
