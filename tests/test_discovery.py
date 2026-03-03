"""
Tests for photometry_pipeline.discovery and runner --discover mode.

Validates:
  C1. Strict JSON schema (all required keys, types, abs paths)
  C2. Side-effect freedom (no files/dirs created)
  C3. Session order matches Pipeline.discover_files() resolution
  C4. ROI ids match adapter output exactly
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs, _session_entry_to_id
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.core.utils import natural_sort_key


# ======================================================================
# Shared helpers
# ======================================================================

def _npm_csv_content() -> str:
    """Minimal valid NPM CSV (4 frames, 2 LEDs, 2 regions).

    LedState 1 = UV, LedState 2 = Signal.
    Must satisfy monotonicity checks on FrameCounter and SystemTimestamp.
    """
    return (
        "FrameCounter,SystemTimestamp,LedState,RegionA_470G,RegionB_470G\n"
        "1,0.1,1,1.0,1.0\n"
        "2,0.2,2,2.0,2.0\n"
        "3,0.3,1,1.1,1.1\n"
        "4,0.4,2,2.1,2.1\n"
    )


def _make_npm_dataset(tmp_dir: str, filenames: list) -> str:
    """Create a temp directory with NPM-format CSV files."""
    input_dir = os.path.join(tmp_dir, "dataset")
    os.makedirs(input_dir, exist_ok=True)
    content = _npm_csv_content()
    for fn in filenames:
        with open(os.path.join(input_dir, fn), "w") as f:
            f.write(content)
    return input_dir


def _make_config(**overrides) -> Config:
    """Create a Config suitable for our tiny mock data."""
    defaults = dict(
        event_signal="dff",
        preview_first_n=2,
        target_fs_hz=1.0,
        chunk_duration_sec=1.0,
        allow_partial_final_chunk=True,
    )
    defaults.update(overrides)
    return Config(**defaults)


# ======================================================================
# Schema keys (source of truth for the contract)
# ======================================================================

_REQUIRED_TOP_KEYS = {
    "schema_version", "input_dir", "resolved_format", "sessions",
    "n_total_discovered", "preview_first_n", "n_preview", "rois",
}
_REQUIRED_SESSION_KEYS = {"index", "session_id", "path", "included_in_preview"}
_REQUIRED_ROI_KEYS = {"roi_id"}


class TestDiscoverOutputSchemaStrict(unittest.TestCase):
    """C1 — Verify every required key, type, and abs-path constraint."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.filenames = ["session_10.csv", "session_2.csv", "session_1.csv"]
        self.input_dir = _make_npm_dataset(self.tmp_dir, self.filenames)
        self.config = _make_config()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_top_level_keys_exact(self):
        res = discover_inputs(self.input_dir, self.config, force_format="auto")
        self.assertEqual(set(res.keys()), _REQUIRED_TOP_KEYS)

    def test_schema_version_is_int_1(self):
        res = discover_inputs(self.input_dir, self.config, force_format="auto")
        self.assertEqual(res["schema_version"], 1)
        self.assertIsInstance(res["schema_version"], int)

    def test_input_dir_is_abs(self):
        res = discover_inputs(self.input_dir, self.config, force_format="auto")
        self.assertTrue(os.path.isabs(res["input_dir"]))

    def test_session_keys_and_types(self):
        res = discover_inputs(self.input_dir, self.config, force_format="auto")
        for sess in res["sessions"]:
            self.assertTrue(
                _REQUIRED_SESSION_KEYS.issubset(set(sess.keys())),
                f"Missing session keys: {_REQUIRED_SESSION_KEYS - set(sess.keys())}",
            )
            self.assertIsInstance(sess["index"], int)
            self.assertIsInstance(sess["session_id"], str)
            self.assertTrue(os.path.isabs(sess["path"]))
            self.assertIsInstance(sess["included_in_preview"], bool)

    def test_session_indices_are_0_to_n(self):
        res = discover_inputs(self.input_dir, self.config, force_format="auto")
        indices = [s["index"] for s in res["sessions"]]
        self.assertEqual(indices, list(range(len(indices))))

    def test_preview_counts(self):
        res = discover_inputs(
            self.input_dir, self.config, force_format="auto"
        )
        n_included = sum(1 for s in res["sessions"] if s["included_in_preview"])
        self.assertEqual(res["n_preview"], n_included)
        self.assertEqual(res["n_total_discovered"], len(res["sessions"]))

    def test_preview_first_n_nullable(self):
        cfg = _make_config(preview_first_n=None)
        res = discover_inputs(self.input_dir, cfg, force_format="auto",
                              preview_first_n=None)
        self.assertIsNone(res["preview_first_n"])
        # When preview_first_n is None, all sessions should be included
        for s in res["sessions"]:
            self.assertTrue(s["included_in_preview"])

    def test_roi_keys(self):
        res = discover_inputs(self.input_dir, self.config, force_format="auto")
        for roi in res["rois"]:
            self.assertTrue(
                _REQUIRED_ROI_KEYS.issubset(set(roi.keys())),
                f"Missing ROI keys: {_REQUIRED_ROI_KEYS - set(roi.keys())}",
            )
            self.assertIsInstance(roi["roi_id"], str)


class TestDiscoverIsSideEffectFree(unittest.TestCase):
    """C2 — Running discovery must not create any files."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.filenames = ["session_1.csv", "session_2.csv"]
        self.input_dir = _make_npm_dataset(self.tmp_dir, self.filenames)
        self.config = _make_config()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_discover_via_cli_creates_no_artifacts(self):
        """Run actual runner --discover subprocess and verify zero new files."""
        config_path = os.path.join(self.tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config.__dict__, f)

        runner_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..",
                         "tools", "run_full_pipeline_deliverables.py")
        )

        # Snapshot what exists before
        before = set(os.listdir(self.tmp_dir))

        argv = [
            sys.executable, runner_script,
            "--input", self.input_dir,
            "--config", config_path,
            "--format", "npm",
            "--discover",
        ]
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"Runner failed.\nSTDOUT:\n{result.stdout}\n"
                         f"STDERR:\n{result.stderr}")

        # Verify stdout is valid JSON
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["schema_version"], 1)

        # No new files/directories created
        after = set(os.listdir(self.tmp_dir))
        self.assertEqual(before, after,
                         f"Discovery created unexpected items: {after - before}")

    def test_discover_lib_creates_no_artifacts(self):
        """Call discover_inputs() directly and verify no new files."""
        before = set(os.listdir(self.tmp_dir))
        discover_inputs(self.input_dir, self.config, force_format="npm")
        after = set(os.listdir(self.tmp_dir))
        self.assertEqual(before, after)


class TestDiscoverOrderMatchesRealRunResolution(unittest.TestCase):
    """C3 — Discovery sessions MUST have the exact same order as Pipeline.

    We verify by calling Pipeline.discover_files() directly (without
    running analysis) and comparing the ordered file lists.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Chosen to trip alphabetical vs natural sort
        self.filenames = ["session_10.csv", "session_2.csv", "session_1.csv"]
        self.input_dir = _make_npm_dataset(self.tmp_dir, self.filenames)
        self.config = _make_config()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_order_matches_pipeline_discover_files(self):
        """Discovery session order == Pipeline.discover_files() order."""
        # Ground truth: Pipeline's own resolver
        pipeline = Pipeline(self.config)
        pipeline.discover_files(self.input_dir, force_format="auto")
        pipeline_paths = [os.path.abspath(f) for f in pipeline.file_list]
        pipeline_ids = [pipeline._session_entry_to_id(f) for f in pipeline.file_list]

        # Discovery
        res = discover_inputs(
            self.input_dir, self.config, force_format="auto"
        )
        discovery_paths = [s["path"] for s in res["sessions"]]
        discovery_ids = [s["session_id"] for s in res["sessions"]]

        self.assertEqual(discovery_paths, pipeline_paths,
                         "Discovery paths differ from Pipeline paths")
        self.assertEqual(discovery_ids, pipeline_ids,
                         "Discovery session_ids differ from Pipeline session_ids")

    def test_session_id_matches_pipeline_helper(self):
        """_session_entry_to_id in discovery matches Pipeline._session_entry_to_id."""
        pipeline = Pipeline(self.config)
        test_paths = [
            os.path.join(self.input_dir, fn) for fn in self.filenames
        ]
        for p in test_paths:
            self.assertEqual(
                _session_entry_to_id(p),
                pipeline._session_entry_to_id(p),
                f"ID mismatch for {p}",
            )


class TestROIIdsMatchAdapterOutput(unittest.TestCase):
    """C4 — ROI ids from discovery MUST equal what load_chunk yields."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.filenames = ["session_1.csv", "session_2.csv"]
        self.input_dir = _make_npm_dataset(self.tmp_dir, self.filenames)
        self.config = _make_config()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_rois_match_load_chunk_channel_names(self):
        """Discovery ROI ids == load_chunk(...).channel_names for valid files."""
        from photometry_pipeline.io.adapters import load_chunk, sniff_format

        first_file = os.path.join(self.input_dir, "session_1.csv")
        fmt = sniff_format(first_file, self.config)
        chunk = load_chunk(first_file, fmt, self.config, chunk_id=0)
        adapter_rois = list(chunk.channel_names)

        res = discover_inputs(
            self.input_dir, self.config, force_format="auto"
        )
        discovery_rois = [r["roi_id"] for r in res["rois"]]

        self.assertEqual(discovery_rois, adapter_rois,
                         "Discovery ROIs differ from adapter channel_names")


if __name__ == "__main__":
    unittest.main()
