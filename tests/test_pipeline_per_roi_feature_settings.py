"""Pipeline-level tests for optional per-ROI feature-detection settings (4J16k32b).

Runs the real Pipeline end-to-end (Pass 1 + Pass 2) against a synthetic
two-ROI RWD dataset where both ROIs carry an identical underlying trace, so
any difference in features.csv peak counts can only come from per-ROI
feature-detection settings, not from different input data.
"""

import json
import os
import shutil
import unittest

import numpy as np
import pandas as pd

from photometry_pipeline.config import Config
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
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

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

    def test_no_provenance_file_written_when_not_supplied(self):
        out, _ = self._run("out_no_provenance")

        provenance_path = os.path.join(out, "features", "feature_event_provenance.json")
        self.assertFalse(os.path.exists(provenance_path))

    def test_provenance_file_written_when_per_roi_provenance_supplied(self):
        out, _ = self._run(
            "out_with_provenance",
            per_roi_feature_provenance={
                "Region1": {
                    "source": "override",
                    "feature_event_profile_id": "custom-region1",
                    # override_config_fields may be sparse (only what the
                    # profile explicitly set); effective_config_fields must
                    # be the complete settings actually used.
                    "override_config_fields": {"peak_threshold_method": "percentile"},
                    "effective_config_fields": {
                        "peak_threshold_method": "percentile",
                        "peak_min_distance_sec": 1.0,
                    },
                },
                "Region2": {
                    "source": "default",
                    "feature_event_profile_id": "feature-profile-1",
                    "override_config_fields": {"peak_threshold_method": "mean_std"},
                    "effective_config_fields": {
                        "peak_threshold_method": "mean_std",
                        "peak_min_distance_sec": 1.0,
                    },
                },
            },
        )

        provenance_path = os.path.join(out, "features", "feature_event_provenance.json")
        self.assertTrue(os.path.exists(provenance_path))
        with open(provenance_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.assertEqual(payload["schema_version"], "guided_feature_event_provenance.v2")
        by_roi = {entry["roi"]: entry for entry in payload["rois"]}
        self.assertEqual(by_roi["Region1"]["source"], "override")
        self.assertEqual(by_roi["Region1"]["feature_event_profile_id"], "custom-region1")
        self.assertEqual(
            by_roi["Region1"]["override_config_fields"],
            {"peak_threshold_method": "percentile"},
        )
        # Provenance must not under-report: effective_config_fields carries
        # the complete settings used, including fields the sparse override
        # never mentioned (peak_min_distance_sec here).
        self.assertEqual(
            by_roi["Region1"]["effective_config_fields"]["peak_min_distance_sec"], 1.0
        )
        self.assertEqual(by_roi["Region2"]["source"], "default")


if __name__ == "__main__":
    unittest.main()
