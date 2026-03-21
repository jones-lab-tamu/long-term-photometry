import os
import shutil
import tempfile
import unittest
import subprocess
import sys
import glob
import csv

import pandas as pd

from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from photometry_pipeline.io.adapters import load_chunk, sniff_format
from photometry_pipeline.pipeline import Pipeline


def _write_csv(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")


def _vendor_style_lines() -> list[str]:
    return [
        '"{""Meta"":{""Source"":""Vendor""}}",,,,,',
        "TimeStamp,Events,CH1-410,CH1-470,CH2-410,CH2-470",
        "0,,100.0,120.0,200.0,220.0",
        "0.5,,101.0,121.0,201.0,221.0",
        "1.0,,102.0,122.0,202.0,222.0",
        "1.5,,103.0,123.0,203.0,223.0",
    ]


def _synthetic_style_lines(offset: float = 0.0) -> list[str]:
    return [
        "TimeStamp,Region0-470,Region0-410,Region1-470,Region1-410",
        f"0,{120.0 + offset},{100.0 + offset},{220.0 + offset},{200.0 + offset}",
        f"0.5,{121.0 + offset},{101.0 + offset},{221.0 + offset},{201.0 + offset}",
        f"1.0,{122.0 + offset},{102.0 + offset},{222.0 + offset},{202.0 + offset}",
        f"1.5,{123.0 + offset},{103.0 + offset},{223.0 + offset},{203.0 + offset}",
    ]


def _default_test_config() -> Config:
    # Intentionally uses mismatched RWD hints so parser fallback contract is exercised.
    return Config(
        target_fs_hz=2.0,
        chunk_duration_sec=2.0,
        allow_partial_final_chunk=False,
        rwd_time_col="Time(s)",
        uv_suffix="-415",
        sig_suffix="-470",
    )


def _session_paths(discovery_payload: dict) -> list[str]:
    return [s["path"] for s in discovery_payload["sessions"]]


class TestRwdVendorCompat(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_chunked_rwd_root_parity_for_auto_and_explicit_rwd(self):
        root = os.path.join(self.tmp_dir, "synthetic_root")
        first = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        second = os.path.join(root, "2025_01_01-00_30_00", "fluorescence.csv")
        _write_csv(first, _synthetic_style_lines(offset=0.0))
        _write_csv(second, _synthetic_style_lines(offset=10.0))

        cfg = _default_test_config()

        discover_auto = discover_inputs(root, cfg, force_format="auto")
        discover_rwd = discover_inputs(root, cfg, force_format="rwd")
        self.assertEqual(discover_auto["resolved_format"], "RWD")
        self.assertEqual(discover_rwd["resolved_format"], "RWD")

        pipe_auto = Pipeline(cfg)
        pipe_auto.discover_files(root, recursive=False, force_format="auto")
        pipe_rwd = Pipeline(cfg)
        pipe_rwd.discover_files(root, recursive=False, force_format="rwd")

        paths_auto_discovery = _session_paths(discover_auto)
        paths_rwd_discovery = _session_paths(discover_rwd)
        paths_auto_pipeline = [os.path.abspath(p) for p in pipe_auto.file_list]
        paths_rwd_pipeline = [os.path.abspath(p) for p in pipe_rwd.file_list]

        self.assertEqual(paths_auto_discovery, paths_rwd_discovery)
        self.assertEqual(paths_auto_discovery, paths_auto_pipeline)
        self.assertEqual(paths_auto_discovery, paths_rwd_pipeline)

        cli_fmt = pipe_auto._get_format(paths_auto_pipeline[0], force_format="auto")
        gui_fmt = discover_auto["resolved_format"].lower()
        cli_chunk = load_chunk(paths_auto_pipeline[0], cli_fmt, cfg, chunk_id=0)
        gui_chunk = load_chunk(paths_auto_discovery[0], gui_fmt, cfg, chunk_id=0)
        self.assertEqual(cli_chunk.channel_names, gui_chunk.channel_names)
        self.assertEqual(cli_chunk.metadata.get("roi_map"), gui_chunk.metadata.get("roi_map"))
        self.assertEqual(cli_chunk.uv_raw.shape, gui_chunk.uv_raw.shape)
        self.assertEqual(cli_chunk.sig_raw.shape, gui_chunk.sig_raw.shape)

    def test_vendor_style_rwd_with_metadata_and_events_is_supported(self):
        root = os.path.join(self.tmp_dir, "vendor_root")
        chunk_path = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        _write_csv(chunk_path, _vendor_style_lines())

        cfg = _default_test_config()

        self.assertEqual(sniff_format(chunk_path, cfg), "rwd")
        chunk = load_chunk(chunk_path, "rwd", cfg, chunk_id=0)
        self.assertEqual(chunk.channel_names, ["CH1", "CH2"])
        self.assertEqual(chunk.uv_raw.shape[1], 2)
        self.assertEqual(chunk.sig_raw.shape[1], 2)

        discovery_payload = discover_inputs(root, cfg, force_format="auto")
        self.assertEqual(discovery_payload["resolved_format"], "RWD")
        self.assertEqual([r["roi_id"] for r in discovery_payload["rois"]], ["CH1", "CH2"])

    def test_simplified_synthetic_rwd_is_supported(self):
        root = os.path.join(self.tmp_dir, "simple_root")
        chunk_path = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        _write_csv(chunk_path, _synthetic_style_lines())

        cfg = _default_test_config()
        self.assertEqual(sniff_format(chunk_path, cfg), "rwd")
        chunk = load_chunk(chunk_path, "rwd", cfg, chunk_id=0)
        self.assertEqual(chunk.channel_names, ["Region0", "Region1"])

    def test_malformed_rwd_with_plausible_header_reports_pairing_error(self):
        root = os.path.join(self.tmp_dir, "bad_root")
        chunk_path = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        _write_csv(
            chunk_path,
            [
                "TimeStamp,Events,CH1-470",
                "0,,120.0",
                "0.5,,121.0",
            ],
        )

        cfg = _default_test_config()

        with self.assertRaises(ValueError) as cm:
            load_chunk(chunk_path, "rwd", cfg, chunk_id=0)
        msg = str(cm.exception)
        self.assertIn("Recognizable header found", msg)
        self.assertIn("no valid uv/sig channel pairs", msg.lower())
        self.assertNotIn("No recognizable header row", msg)

    def test_natural_channel_order_for_ch_labels(self):
        root = os.path.join(self.tmp_dir, "order_ch_root")
        chunk_path = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        _write_csv(
            chunk_path,
            [
                "TimeStamp,CH10-410,CH10-470,CH2-410,CH2-470,CH1-410,CH1-470",
                "0,10,20,30,40,50,60",
                "0.5,11,21,31,41,51,61",
                "1.0,12,22,32,42,52,62",
                "1.5,13,23,33,43,53,63",
            ],
        )

        cfg = _default_test_config()
        chunk = load_chunk(chunk_path, "rwd", cfg, chunk_id=0)
        self.assertEqual(chunk.channel_names, ["CH1", "CH2", "CH10"])

        discovery_payload = discover_inputs(root, cfg, force_format="auto")
        self.assertEqual([r["roi_id"] for r in discovery_payload["rois"]], ["CH1", "CH2", "CH10"])

    def test_natural_channel_order_for_region_labels(self):
        root = os.path.join(self.tmp_dir, "order_region_root")
        chunk_path = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        _write_csv(
            chunk_path,
            [
                "TimeStamp,Region10-410,Region10-470,Region1-410,Region1-470,Region0-410,Region0-470",
                "0,10,20,30,40,50,60",
                "0.5,11,21,31,41,51,61",
                "1.0,12,22,32,42,52,62",
                "1.5,13,23,33,43,53,63",
            ],
        )

        cfg = _default_test_config()
        chunk = load_chunk(chunk_path, "rwd", cfg, chunk_id=0)
        self.assertEqual(chunk.channel_names, ["Region0", "Region1", "Region10"])

        discovery_payload = discover_inputs(root, cfg, force_format="auto")
        self.assertEqual(
            [r["roi_id"] for r in discovery_payload["rois"]],
            ["Region0", "Region1", "Region10"],
        )

    def test_legacy_generator_emits_vendor_faithful_rwd(self):
        out_dir = os.path.join(self.tmp_dir, "gen_vendor_rwd")
        cfg_path = os.path.join(self.tmp_dir, "gen_vendor_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(
                "chunk_duration_sec: 600\n"
                "target_fs_hz: 20\n"
                "baseline_method: uv_raw_percentile_session\n"
                "baseline_percentile: 10\n"
                "rwd_time_col: TimeStamp\n"
                "uv_suffix: \"-410\"\n"
                "sig_suffix: \"-470\"\n"
                "peak_threshold_method: mean_std\n"
                "window_sec: 20.0\n"
                "step_sec: 5.0\n"
            )

        cmd = [
            sys.executable,
            "tools/synth_photometry_dataset.py",
            "--out",
            out_dir,
            "--format",
            "rwd",
            "--config",
            cfg_path,
            "--total-days",
            str(1.0 / 24.0),
            "--recording-duration-min",
            "10",
            "--recordings-per-hour",
            "2",
            "--n-rois",
            "2",
            "--seed",
            "7",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            f"legacy generator failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

        chunk_dirs = sorted([p for p in glob.glob(os.path.join(out_dir, "*")) if os.path.isdir(p)])
        self.assertEqual(len(chunk_dirs), 2)
        chunk_files = [os.path.join(p, "fluorescence.csv") for p in chunk_dirs]
        for fpath in chunk_files:
            self.assertTrue(os.path.isfile(fpath))

        first_chunk = chunk_files[0]
        with open(first_chunk, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))

        self.assertGreaterEqual(len(rows), 3)
        metadata_row = rows[0]
        header_row = rows[1]
        self.assertIn("Light", metadata_row[0])
        self.assertEqual(header_row[0], "TimeStamp")
        self.assertEqual(header_row[1], "Events")
        self.assertEqual(
            header_row[2:6],
            ["CH1-410", "CH1-470", "CH2-410", "CH2-470"],
        )
        self.assertFalse(any(col.startswith("Region") for col in header_row))

        df = pd.read_csv(first_chunk, header=1)
        self.assertEqual(len(df), 12000)
        self.assertIn("Events", df.columns)
        self.assertTrue(df["Events"].isna().all())

        cfg = Config(
            target_fs_hz=20.0,
            chunk_duration_sec=600.0,
            allow_partial_final_chunk=False,
            rwd_time_col="Time(s)",
            uv_suffix="-415",
            sig_suffix="-470",
        )

        discover_auto = discover_inputs(out_dir, cfg, force_format="auto")
        discover_rwd = discover_inputs(out_dir, cfg, force_format="rwd")
        pipe_auto = Pipeline(cfg)
        pipe_auto.discover_files(out_dir, recursive=False, force_format="auto")
        pipe_rwd = Pipeline(cfg)
        pipe_rwd.discover_files(out_dir, recursive=False, force_format="rwd")

        auto_files = _session_paths(discover_auto)
        rwd_files = _session_paths(discover_rwd)
        self.assertEqual(auto_files, rwd_files)
        self.assertEqual(auto_files, [os.path.abspath(p) for p in pipe_auto.file_list])
        self.assertEqual(auto_files, [os.path.abspath(p) for p in pipe_rwd.file_list])

        parsed = load_chunk(auto_files[0], "rwd", cfg, chunk_id=0)
        self.assertEqual(parsed.channel_names, ["CH1", "CH2"])


if __name__ == "__main__":
    unittest.main()
