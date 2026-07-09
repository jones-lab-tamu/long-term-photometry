"""Tests for extract_features' optional per-ROI settings resolution (4J16k32b).

Proves:
- Default/global behavior is unchanged when per_roi_config is omitted.
- A ROI with an entry in per_roi_config uses that ROI's settings only.
- A ROI without an entry falls back to the global config, unaffected by any
  other ROI's override (no cross-contamination).
- Per-ROI results match running extract_features on that ROI alone with its
  own config.
"""

import numpy as np
import unittest

from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.config import Config


class MockChunk:
    def __init__(self, delta_f, dff, time_sec, fs_hz, channel_names):
        self.delta_f = delta_f
        self.dff = dff
        self.time_sec = time_sec
        self.fs_hz = fs_hz
        self.chunk_id = "test_chunk"
        self.source_file = "test.raw"
        self.channel_names = channel_names
        self.metadata = {"qc_warnings": []}


def _make_config(**overrides):
    cfg = Config()
    cfg.event_signal = "dff"
    cfg.peak_threshold_method = "absolute"
    cfg.peak_threshold_abs = 1.0
    cfg.peak_min_distance_sec = 0.1
    # Isolate pure threshold behavior: disable the default prominence/width
    # requirements, which a single-sample test spike would otherwise fail.
    cfg.peak_min_prominence_k = 0.0
    cfg.peak_min_width_sec = 0.0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class TestExtractFeaturesPerRoiConfig(unittest.TestCase):
    def setUp(self):
        fs = 10.0
        n = 11
        t = np.arange(n) / fs
        # A single spike of amplitude 0.6 at index 5 in both ROIs.
        spike = np.zeros(n)
        spike[5] = 0.6
        self.dff = np.stack([spike, spike], axis=1)  # (n, 2): CH1, CH2
        self.delta_f = np.zeros_like(self.dff)
        self.time_sec = t
        self.fs_hz = fs

    def _chunk(self, channel_names):
        return MockChunk(
            self.delta_f, self.dff, self.time_sec, self.fs_hz, channel_names
        )

    def test_global_only_path_unchanged_when_per_roi_config_omitted(self):
        chunk = self._chunk(["CH1", "CH2"])
        global_cfg = _make_config()  # abs threshold 1.0: 0.6 spike is not a peak

        result = extract_features(chunk, global_cfg)

        self.assertEqual(list(result["roi"]), ["CH1", "CH2"])
        self.assertEqual(result.loc[0, "peak_count"], 0)
        self.assertEqual(result.loc[1, "peak_count"], 0)

    def test_per_roi_override_affects_only_that_roi(self):
        chunk = self._chunk(["CH1", "CH2"])
        global_cfg = _make_config()  # abs threshold 1.0: misses the 0.6 spike
        ch1_cfg = _make_config(peak_threshold_abs=0.5)  # catches the 0.6 spike

        result = extract_features(
            chunk, global_cfg, per_roi_config={"CH1": ch1_cfg}
        )

        by_roi = result.set_index("roi")
        self.assertEqual(by_roi.loc["CH1", "peak_count"], 1)
        # CH2 has no override and must fall back to the global config,
        # unaffected by CH1's override (no cross-contamination).
        self.assertEqual(by_roi.loc["CH2", "peak_count"], 0)

    def test_per_roi_result_matches_running_that_roi_alone(self):
        chunk = self._chunk(["CH1", "CH2"])
        global_cfg = _make_config()
        ch1_cfg = _make_config(peak_threshold_abs=0.5)

        combined = extract_features(
            chunk, global_cfg, per_roi_config={"CH1": ch1_cfg}
        )
        combined_by_roi = combined.set_index("roi")

        ch1_alone_chunk = self._chunk(["CH1"])
        ch1_alone_chunk.dff = self.dff[:, :1]
        ch1_alone_chunk.delta_f = self.delta_f[:, :1]
        ch1_alone = extract_features(ch1_alone_chunk, ch1_cfg)

        ch2_alone_chunk = self._chunk(["CH2"])
        ch2_alone_chunk.dff = self.dff[:, 1:]
        ch2_alone_chunk.delta_f = self.delta_f[:, 1:]
        ch2_alone = extract_features(ch2_alone_chunk, global_cfg)

        self.assertEqual(
            combined_by_roi.loc["CH1", "peak_count"],
            ch1_alone.loc[0, "peak_count"],
        )
        self.assertAlmostEqual(
            combined_by_roi.loc["CH1", "auc"], ch1_alone.loc[0, "auc"]
        )
        self.assertEqual(
            combined_by_roi.loc["CH2", "peak_count"],
            ch2_alone.loc[0, "peak_count"],
        )
        self.assertAlmostEqual(
            combined_by_roi.loc["CH2", "auc"], ch2_alone.loc[0, "auc"]
        )

    def test_per_roi_event_signal_override_selects_correct_array(self):
        # CH1 override switches to delta_f, which is all zeros here, so it
        # must report zero peaks even though dff has a detectable spike.
        chunk = self._chunk(["CH1", "CH2"])
        global_cfg = _make_config(peak_threshold_abs=0.5)  # dff spike is a peak
        ch1_cfg = _make_config(peak_threshold_abs=0.5, event_signal="delta_f")

        result = extract_features(
            chunk, global_cfg, per_roi_config={"CH1": ch1_cfg}
        )
        by_roi = result.set_index("roi")

        self.assertEqual(by_roi.loc["CH1", "peak_count"], 0)  # delta_f is all zero
        self.assertEqual(by_roi.loc["CH2", "peak_count"], 1)  # dff spike detected

    def test_empty_per_roi_config_dict_behaves_like_none(self):
        chunk = self._chunk(["CH1", "CH2"])
        global_cfg = _make_config()

        with_none = extract_features(chunk, global_cfg, per_roi_config=None)
        with_empty = extract_features(chunk, global_cfg, per_roi_config={})

        self.assertTrue((with_none["peak_count"] == with_empty["peak_count"]).all())


if __name__ == "__main__":
    unittest.main()
