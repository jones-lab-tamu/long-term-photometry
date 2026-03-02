import unittest
import os
import sys
import shutil
import subprocess
import json
import tempfile
import uuid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestROISelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_dir = tempfile.mkdtemp(prefix="out_roi_selection_")
        cls.input_dir = os.path.join(cls.out_dir, "input_RWD")
        cls.config_path = os.path.join(cls.out_dir, "qc_universal_config.yaml")

        orig_config_path = os.path.join(PROJECT_ROOT, "tests", "qc_universal_config.yaml")
        shutil.copy2(orig_config_path, cls.config_path)

        # 1. Synthetic Data (small dataset: 2.0 days, 3 ROIs)
        print("Generating synthetic data ...")
        gen_cmd = [
            sys.executable, "tools/synth_photometry_dataset.py",
            "--out", cls.input_dir,
            "--format", "rwd",
            "--config", cls.config_path,
            "--total-days", "2.0",
            "--recordings-per-hour", "2", 
            "--recording-duration-min", "10.0",
            "--n-rois", "3",
            "--phasic-mode", "phase_locked_to_tonic",
            "--seed", "42",
            "--preset", "biological_shared_nuisance"
        ]
        subprocess.check_call(gen_cmd)

    def test_include_rois(self):
        output_package = os.path.join(self.out_dir, f"package_incl_{uuid.uuid4().hex[:8]}")
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2",
            "--include-rois", "Region0,Region2"
        ]
        subprocess.check_call(cmd)
        
        # 1. Assert only Region0 and Region2 output directories exist
        reg0 = os.path.join(output_package, "Region0")
        reg1 = os.path.join(output_package, "Region1")
        reg2 = os.path.join(output_package, "Region2")
        self.assertTrue(os.path.exists(reg0))
        self.assertFalse(os.path.exists(reg1))
        self.assertTrue(os.path.exists(reg2))
        
        # 2. Check run_report.json
        report_path = os.path.join(output_package, "_analysis", "phasic_out", "run_report.json")
        with open(report_path, "r") as f:
            report = json.load(f)
            
        roi_sel = report.get("roi_selection")
        self.assertIsNotNone(roi_sel)
        self.assertEqual(len(roi_sel["discovered_rois"]), 3)
        self.assertEqual(roi_sel["include_rois"], ["Region0", "Region2"])
        self.assertEqual(roi_sel["selected_rois"], ["Region0", "Region2"])
        
        # 3. Check inputs:roi_selection event in PARENT events.ndjson
        events_path = os.path.join(output_package, "events.ndjson")
        self.assertTrue(os.path.exists(events_path), "Parent events.ndjson must exist")
        found_event = False
        with open(events_path, "r") as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("stage") == "inputs" and ev.get("type") == "roi_selection":
                    found_event = True
                    self.assertEqual(ev["payload"]["selected_rois"], ["Region0", "Region2"])
                    self.assertEqual(ev["payload"]["discovered_rois"], ["Region0", "Region1", "Region2"])
                    break
        if not found_event:
            print("CONTENTS OF events.ndjson:")
            with open(events_path, "r") as f:
                print(f.read())
        self.assertTrue(found_event, "inputs:roi_selection event not found in parent events.ndjson")
        
        # 4. Regression: pipeline must NOT create events.ndjson under tonic_out or phasic_out
        tonic_events = os.path.join(output_package, "_analysis", "tonic_out", "events.ndjson")
        phasic_events = os.path.join(output_package, "_analysis", "phasic_out", "events.ndjson")
        self.assertFalse(os.path.exists(tonic_events),
                         "Pipeline must not create events.ndjson under tonic_out")
        self.assertFalse(os.path.exists(phasic_events),
                         "Pipeline must not create events.ndjson under phasic_out")

    def test_invalid_include_fails_closed(self):
        output_package = os.path.join(self.out_dir, f"package_inv_{uuid.uuid4().hex[:8]}")
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2",
            "--include-rois", "Region0,ImaginaryROI"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Expected hard failure")
        combined = result.stdout + result.stderr
        self.assertIn("Included ROIs not found in discovered ROIs", combined)
        self.assertIn("ImaginaryROI", combined)

    def test_exclude_ignores_unknown(self):
        output_package = os.path.join(self.out_dir, f"package_exc_{uuid.uuid4().hex[:8]}")
        cmd = [
            sys.executable, "tools/run_full_pipeline_deliverables.py",
            "--input", self.input_dir,
            "--out", output_package,
            "--config", self.config_path,
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2",
            "--exclude-rois", "Region1,FakeROI"
        ]
        subprocess.check_call(cmd)
        
        # Assert succeeded and Filtered Region1 but Warned about FakeROI
        reg0 = os.path.join(output_package, "Region0")
        reg1 = os.path.join(output_package, "Region1")
        reg2 = os.path.join(output_package, "Region2")
        self.assertTrue(os.path.exists(reg0))
        self.assertFalse(os.path.exists(reg1))
        self.assertTrue(os.path.exists(reg2))
        
        report_path = os.path.join(output_package, "_analysis", "phasic_out", "run_report.json")
        with open(report_path, "r") as f:
            report = json.load(f)
            
        roi_sel = report.get("roi_selection")
        self.assertEqual(roi_sel["exclude_rois"], ["Region1", "FakeROI"])
        self.assertEqual(roi_sel["selected_rois"], ["Region0", "Region2"])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.out_dir):
            try:
                shutil.rmtree(cls.out_dir, ignore_errors=True)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
