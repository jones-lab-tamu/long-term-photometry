import os
import shutil
import tempfile
import unittest

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
    # Intentionally uses a mismatched RWD hint so fallback parsing is exercised.
    return Config(
        target_fs_hz=2.0,
        chunk_duration_sec=2.0,
        allow_partial_final_chunk=False,
        rwd_time_col="Time(s)",
        uv_suffix="-415",
        sig_suffix="-470",
    )


class TestRwdReaderContract(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cli_and_gui_paths_resolve_same_rwd_contract(self):
        root = os.path.join(self.tmp_dir, "synthetic_root")
        first = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        second = os.path.join(root, "2025_01_01-00_30_00", "fluorescence.csv")
        _write_csv(first, _synthetic_style_lines(offset=0.0))
        _write_csv(second, _synthetic_style_lines(offset=10.0))

        cfg = _default_test_config()

        pipeline = Pipeline(cfg)
        pipeline.discover_files(root, recursive=False, force_format="auto")
        cli_files = [os.path.abspath(p) for p in pipeline.file_list]
        self.assertEqual(len(cli_files), 2)

        gui_discovery = discover_inputs(root, cfg, force_format="auto")
        gui_files = [s["path"] for s in gui_discovery["sessions"]]

        self.assertEqual(gui_discovery["resolved_format"], "RWD")
        self.assertEqual(gui_files, cli_files)
        self.assertEqual([r["roi_id"] for r in gui_discovery["rois"]], ["Region0", "Region1"])

        cli_fmt = pipeline._get_format(cli_files[0], force_format="auto")
        self.assertEqual(cli_fmt, "rwd")
        gui_fmt = gui_discovery["resolved_format"].lower()

        cli_chunk = load_chunk(cli_files[0], cli_fmt, cfg, chunk_id=0)
        gui_chunk = load_chunk(gui_files[0], gui_fmt, cfg, chunk_id=0)
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

        disco = discover_inputs(root, cfg, force_format="auto")
        self.assertEqual(disco["resolved_format"], "RWD")
        self.assertEqual([r["roi_id"] for r in disco["rois"]], ["CH1", "CH2"])

    def test_simplified_synthetic_rwd_is_supported(self):
        root = os.path.join(self.tmp_dir, "simple_root")
        chunk_path = os.path.join(root, "2025_01_01-00_00_00", "fluorescence.csv")
        _write_csv(chunk_path, _synthetic_style_lines())

        cfg = _default_test_config()

        self.assertEqual(sniff_format(chunk_path, cfg), "rwd")
        chunk = load_chunk(chunk_path, "rwd", cfg, chunk_id=0)
        self.assertEqual(chunk.channel_names, ["Region0", "Region1"])
        self.assertEqual(chunk.uv_raw.shape[1], 2)
        self.assertEqual(chunk.sig_raw.shape[1], 2)

    def test_malformed_rwd_raises_clear_error(self):
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
        self.assertIn("No recognizable header row", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
