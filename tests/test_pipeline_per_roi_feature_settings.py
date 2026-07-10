"""Pipeline-level tests for optional per-ROI feature-detection settings (4J16k32b).

Runs the real Pipeline end-to-end (Pass 1 + Pass 2) against a synthetic
two-ROI RWD dataset where both ROIs carry an identical underlying trace, so
any difference in features.csv peak counts can only come from per-ROI
feature-detection settings, not from different input data.
"""

import json
import os
import shutil
import time
import unittest

import numpy as np
import pandas as pd

from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS
from photometry_pipeline.pipeline import Pipeline


class TestPipelinePerRoiFeatureSettings(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_pipeline_per_roi_feature_settings"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.inp = os.path.join(self.test_dir, "input")
        os.makedirs(self.inp, exist_ok=True)

        rng = np.random.default_rng(42)
        fs = 20.0
        n = 400
        t = np.arange(n) / fs
        uv = 100 + 5 * np.sin(0.05 * t) + rng.normal(0, 0.5, n)
        calcium = np.zeros(n)
        for start in (60, 160, 260, 320):
            length = n - start
            t_local = np.arange(length)
            calcium[start:] += 30 * np.exp(-t_local / 40.0)
        sig = 2.0 * uv + calcium + rng.normal(0, 0.5, n)

        # Region1 and Region2 carry byte-identical UV/signal traces: any
        # difference in detected features must come from feature-detection
        # settings, not from different underlying data.
        df = pd.DataFrame({
            "Time(s)": t,
            "Region1-410": uv,
            "Region1-470": sig,
            "Region2-410": uv,
            "Region2-470": sig,
        })
        df.to_csv(os.path.join(self.inp, "file.csv"), index=False)

        self.config = Config()
        self.config.chunk_duration_sec = 20.0
        self.config.target_fs_hz = 20.0
        self.config.allow_partial_final_chunk = True

    def tearDown(self):
        # On Windows a freshly written output file can still be held briefly by
        # an external scanner, making rmtree raise WinError 32. Retry rather
        # than fail an otherwise-passing test on a filesystem race.
        for attempt in range(5):
            if not os.path.exists(self.test_dir):
                return
            try:
                shutil.rmtree(self.test_dir)
                return
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run(self, out_name, **pipeline_kwargs):
        pipe = Pipeline(self.config, **pipeline_kwargs)
        out = os.path.join(self.test_dir, out_name)
        pipe.run(self.inp, out)
        feats = pd.read_csv(os.path.join(out, "features", "features.csv"))
        return out, feats.set_index("roi")

    def test_global_only_path_produces_equal_features_for_identical_rois(self):
        _, feats = self._run("out_global")

        self.assertEqual(
            feats.loc["Region1", "peak_count"], feats.loc["Region2", "peak_count"]
        )

    def test_per_roi_feature_config_changes_only_overridden_roi(self):
        permissive_cfg = Config()
        permissive_cfg.peak_threshold_method = "percentile"
        permissive_cfg.peak_threshold_percentile = 1.0
        permissive_cfg.peak_min_prominence_k = 0.0
        permissive_cfg.peak_min_width_sec = 0.0

        _, feats_global = self._run("out_global")
        _, feats_override = self._run(
            "out_override",
            per_roi_feature_config={"Region1": permissive_cfg},
        )

        # Region2 (no override) must be unaffected by Region1's override.
        self.assertEqual(
            feats_override.loc["Region2", "peak_count"],
            feats_global.loc["Region2", "peak_count"],
        )
        # Region1's permissive threshold must detect more peaks on the same
        # underlying trace than the default global threshold does.
        self.assertGreater(
            feats_override.loc["Region1", "peak_count"],
            feats_global.loc["Region1", "peak_count"],
        )

    def test_default_only_run_still_writes_complete_per_roi_provenance(self):
        """4J16k39b contract change: EVERY feature-extracting run records the
        settings actually consumed for every analyzed ROI, including a
        Default-only run. Previously no file was written, which made absence
        ambiguous between 'Default-only' and 'legacy run' -- the defect this
        task fixes. Absence must now mean only 'legacy'."""
        out, _ = self._run("out_no_provenance")

        provenance_path = os.path.join(out, "features", "feature_event_provenance.json")
        self.assertTrue(os.path.exists(provenance_path))
        with open(provenance_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.assertEqual(payload["schema_version"], "guided_feature_event_provenance.v3")
        by_roi = {entry["roi"]: entry for entry in payload["rois"]}
        self.assertEqual(set(by_roi), {"Region1", "Region2"})
        for entry in by_roi.values():
            self.assertEqual(entry["source"], "default")
            # No fake Custom entries are invented for a Default-only run.
            self.assertEqual(entry["override_config_fields"], {})
            # Complete effective settings, taken from the Config actually used.
            self.assertEqual(
                set(entry["effective_config_fields"]), FEATURE_EVENT_CONFIG_FIELDS
            )
            self.assertEqual(
                entry["effective_config_fields"]["peak_threshold_k"],
                self.config.peak_threshold_k,
            )
            self.assertTrue(entry["effective_config_digest"])

        # The explicit contract signal is stamped into run_report.json, so a
        # consumer never has to infer the contract from file presence.
        with open(os.path.join(out, "run_report.json"), "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(
            report["feature_event_provenance"]["contract_version"],
            "feature_event_provenance.v3",
        )

    def test_provenance_is_generated_from_the_configs_actually_used(self):
        """4J16k39b: effective_config_fields must be read off the Config objects
        Pipeline actually used, NOT copied from the descriptive dict a caller
        passed in. Here the caller's dict deliberately claims settings that
        disagree with the real per-ROI Config; the real Config must win."""
        custom_cfg = Config()
        custom_cfg.peak_threshold_method = "percentile"
        custom_cfg.peak_threshold_percentile = 1.0
        custom_cfg.peak_min_prominence_k = 0.0
        custom_cfg.peak_min_width_sec = 0.0

        out, _ = self._run(
            "out_with_provenance",
            per_roi_feature_config={"Region1": custom_cfg},
            per_roi_feature_provenance={
                "Region1": {
                    "source": "override",
                    "feature_event_profile_id": "custom-region1",
                    "override_config_fields": {"peak_threshold_method": "percentile"},
                    # Deliberately wrong/misleading: must be ignored.
                    "effective_config_fields": {"peak_threshold_k": 999.0},
                },
                "Region2": {
                    "source": "default",
                    "feature_event_profile_id": "feature-profile-1",
                    "override_config_fields": {},
                    "effective_config_fields": {"peak_threshold_k": 999.0},
                },
            },
        )

        provenance_path = os.path.join(out, "features", "feature_event_provenance.json")
        self.assertTrue(os.path.exists(provenance_path))
        with open(provenance_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.assertEqual(payload["schema_version"], "guided_feature_event_provenance.v3")
        by_roi = {entry["roi"]: entry for entry in payload["rois"]}

        # Source is derived from per_roi_feature_config membership, not the dict.
        self.assertEqual(by_roi["Region1"]["source"], "override")
        self.assertEqual(by_roi["Region2"]["source"], "default")
        self.assertEqual(by_roi["Region1"]["feature_event_profile_id"], "custom-region1")

        # The sparse user override is carried through descriptively.
        self.assertEqual(
            by_roi["Region1"]["override_config_fields"],
            {"peak_threshold_method": "percentile"},
        )

        # Effective settings come from the real Config objects, so the caller's
        # bogus peak_threshold_k=999.0 never appears.
        r1 = by_roi["Region1"]["effective_config_fields"]
        r2 = by_roi["Region2"]["effective_config_fields"]
        self.assertEqual(set(r1), FEATURE_EVENT_CONFIG_FIELDS)
        self.assertEqual(r1["peak_threshold_method"], "percentile")
        self.assertEqual(r1["peak_threshold_percentile"], 1.0)
        self.assertNotEqual(r1["peak_threshold_k"], 999.0)
        self.assertEqual(r2["peak_threshold_method"], self.config.peak_threshold_method)
        self.assertNotEqual(r2["peak_threshold_k"], 999.0)

        # Digests are deterministic and reflect the recorded settings.
        from photometry_pipeline.feature_event_provenance import (
            compute_feature_config_digest,
        )

        self.assertEqual(
            by_roi["Region1"]["effective_config_digest"],
            compute_feature_config_digest(r1),
        )
        self.assertNotEqual(
            by_roi["Region1"]["effective_config_digest"],
            by_roi["Region2"]["effective_config_digest"],
        )


if __name__ == "__main__":
    unittest.main()
