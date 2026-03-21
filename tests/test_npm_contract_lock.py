import csv
import os

import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from photometry_pipeline.io.adapters import load_chunk, sniff_format
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.core.utils import natural_sort_key


def _base_npm_config(**overrides) -> Config:
    cfg = Config(
        allow_partial_final_chunk=True,
        target_fs_hz=1.0,
        chunk_duration_sec=2.0,
        npm_time_axis="system_timestamp",
        npm_frame_col="FrameCounter",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _write_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_valid_npm_csv(path: str, t0: float = 0.0) -> None:
    header = [
        "FrameCounter",
        "SystemTimestamp",
        "ComputerTimestamp",
        "LedState",
        "Region0G",
        "Region1G",
    ]
    rows = [
        [1, t0 + 0.0, t0 + 0.0, 1, 10.0, 11.0],
        [2, t0 + 0.0, t0 + 0.0, 2, 100.0, 110.0],
        [3, t0 + 1.0, t0 + 1.0, 1, 20.0, 21.0],
        [4, t0 + 1.0, t0 + 1.0, 2, 200.0, 210.0],
    ]
    _write_csv(path, header, rows)


def test_npm_flat_directory_discovery_detects_single_run_with_multiple_csvs(tmp_path):
    cfg = _base_npm_config()
    names = [
        "photometryData2025-03-05T15_37_44.csv",
        "photometryData2025-03-05T15_38_01.csv",
        "photometryData2025-03-05T15_38_12.csv",
    ]
    for i, name in enumerate(names):
        _write_valid_npm_csv(str(tmp_path / name), t0=float(i * 10))

    auto = discover_inputs(str(tmp_path), cfg, force_format="auto")
    forced = discover_inputs(str(tmp_path), cfg, force_format="npm")

    assert auto["resolved_format"] == "NPM"
    assert forced["resolved_format"] == "NPM"
    assert auto["n_total_discovered"] == 3
    assert forced["n_total_discovered"] == 3

    auto_names = {os.path.basename(s["path"]) for s in auto["sessions"]}
    forced_names = {os.path.basename(s["path"]) for s in forced["sessions"]}
    assert auto_names == set(names)
    assert forced_names == set(names)


def test_npm_timestamp_filename_order_is_chronological_across_discovery_and_pipeline(tmp_path):
    cfg = _base_npm_config()
    names = [
        "photometryData2025-03-05T15_37_44.csv",
        "photometryData2025-03-05T15_38_01.csv",
        "photometryData2025-03-05T15_38_12.csv",
    ]
    for i, name in enumerate(names):
        _write_valid_npm_csv(str(tmp_path / name), t0=float(i * 10))

    expected_order = sorted(names)

    discover_auto = discover_inputs(str(tmp_path), cfg, force_format="auto")
    discover_forced = discover_inputs(str(tmp_path), cfg, force_format="npm")
    discover_auto_names = [os.path.basename(s["path"]) for s in discover_auto["sessions"]]
    discover_forced_names = [os.path.basename(s["path"]) for s in discover_forced["sessions"]]

    pipeline_auto = Pipeline(cfg)
    pipeline_auto.discover_files(str(tmp_path), force_format="auto")
    pipeline_auto_names = [os.path.basename(p) for p in pipeline_auto.file_list]

    pipeline_forced = Pipeline(cfg)
    pipeline_forced.discover_files(str(tmp_path), force_format="npm")
    pipeline_forced_names = [os.path.basename(p) for p in pipeline_forced.file_list]

    # Contract lock: timestamp-style NPM filenames should process in full-timestamp
    # chronological order, which equals lexicographic order for this pattern.
    assert discover_auto_names == expected_order
    assert discover_forced_names == expected_order
    assert pipeline_auto_names == expected_order
    assert pipeline_forced_names == expected_order


def test_npm_timestamp_order_uses_parsed_datetime_not_prefix_or_suffix(tmp_path):
    cfg = _base_npm_config()
    names = [
        "z_prefix2025-03-05T15_39_01.csv",
        "a_prefix2025-03-05T15_38_59.csv",
        "m_prefix2025-03-05T15_39_00.csv",
    ]
    for i, name in enumerate(names):
        _write_valid_npm_csv(str(tmp_path / name), t0=float(i * 10))

    expected_chronological = [
        "a_prefix2025-03-05T15_38_59.csv",
        "m_prefix2025-03-05T15_39_00.csv",
        "z_prefix2025-03-05T15_39_01.csv",
    ]

    discover = discover_inputs(str(tmp_path), cfg, force_format="npm")
    pipeline = Pipeline(cfg)
    pipeline.discover_files(str(tmp_path), force_format="npm")

    discover_names = [os.path.basename(s["path"]) for s in discover["sessions"]]
    pipeline_names = [os.path.basename(p) for p in pipeline.file_list]

    assert discover_names == expected_chronological
    assert pipeline_names == expected_chronological


def test_npm_timestamp_order_falls_back_when_not_all_files_match_vendor_pattern(tmp_path):
    cfg = _base_npm_config()
    names = [
        "photometryData2025-03-05T15_37_44.csv",
        "photometryData2025-03-05T15_38_01.csv",
        "manual_session_alpha.csv",
    ]
    for i, name in enumerate(names):
        _write_valid_npm_csv(str(tmp_path / name), t0=float(i * 10))

    expected_fallback = sorted(names, key=natural_sort_key)

    discover = discover_inputs(str(tmp_path), cfg, force_format="npm")
    pipeline = Pipeline(cfg)
    pipeline.discover_files(str(tmp_path), force_format="npm")

    discover_names = [os.path.basename(s["path"]) for s in discover["sessions"]]
    pipeline_names = [os.path.basename(p) for p in pipeline.file_list]

    assert discover_names == expected_fallback
    assert pipeline_names == expected_fallback


def test_npm_ledstate_1_maps_to_uv_and_2_maps_to_signal(tmp_path):
    cfg = _base_npm_config(target_fs_hz=1.0, chunk_duration_sec=2.0)
    path = tmp_path / "led_semantics.csv"
    header = ["FrameCounter", "SystemTimestamp", "LedState", "Region0G"]
    rows = [
        [1, 0.0, 1, 10.0],
        [2, 0.0, 2, 100.0],
        [3, 1.0, 1, 20.0],
        [4, 1.0, 2, 200.0],
    ]
    _write_csv(str(path), header, rows)

    chunk = load_chunk(str(path), "npm", cfg, chunk_id=0)
    assert chunk.channel_names == ["Region0"]
    np.testing.assert_allclose(chunk.uv_raw[:, 0], np.array([10.0, 20.0]))
    np.testing.assert_allclose(chunk.sig_raw[:, 0], np.array([100.0, 200.0]))


def test_npm_missing_ledstate_2_fails_with_insufficient_data(tmp_path):
    cfg = _base_npm_config()
    path = tmp_path / "missing_led2.csv"
    header = ["FrameCounter", "SystemTimestamp", "LedState", "Region0G"]
    rows = [
        [1, 0.0, 1, 10.0],
        [2, 1.0, 1, 20.0],
    ]
    _write_csv(str(path), header, rows)

    with pytest.raises(ValueError, match="NPM: Insufficient data"):
        load_chunk(str(path), "npm", cfg, chunk_id=0)


def test_npm_roi_recognition_uses_region_prefix_and_suffix_only(tmp_path):
    cfg = _base_npm_config(target_fs_hz=1.0, chunk_duration_sec=2.0)
    path = tmp_path / "roi_columns.csv"
    header = [
        "FrameCounter",
        "SystemTimestamp",
        "LedState",
        "Region0G",
        "Region1G",
        "OtherG",
        "Region0R",
    ]
    rows = [
        [1, 0.0, 1, 10.0, 11.0, 999.0, 777.0],
        [2, 0.0, 2, 100.0, 110.0, 999.0, 777.0],
        [3, 1.0, 1, 20.0, 21.0, 999.0, 777.0],
        [4, 1.0, 2, 200.0, 210.0, 999.0, 777.0],
    ]
    _write_csv(str(path), header, rows)

    chunk = load_chunk(str(path), "npm", cfg, chunk_id=0)
    assert chunk.channel_names == ["Region0", "Region1"]

    roi_map = chunk.metadata["roi_map"]
    assert roi_map["Region0"]["raw_col"] == "Region0G"
    assert roi_map["Region1"]["raw_col"] == "Region1G"


def test_npm_roi_ordering_is_natural_for_double_digit_indices(tmp_path):
    cfg = _base_npm_config(target_fs_hz=1.0, chunk_duration_sec=2.0)
    path = tmp_path / "roi_natural_order.csv"
    header = [
        "FrameCounter",
        "SystemTimestamp",
        "LedState",
        "Region0G",
        "Region1G",
        "Region10G",
        "Region2G",
    ]
    rows = [
        [1, 0.0, 1, 10.0, 20.0, 110.0, 30.0],
        [2, 0.0, 2, 100.0, 200.0, 1100.0, 300.0],
        [3, 1.0, 1, 11.0, 21.0, 111.0, 31.0],
        [4, 1.0, 2, 101.0, 201.0, 1101.0, 301.0],
    ]
    _write_csv(str(path), header, rows)

    chunk = load_chunk(str(path), "npm", cfg, chunk_id=0)
    roi_map = chunk.metadata["roi_map"]

    # Natural ordering must map Region2 before Region10.
    assert roi_map["Region0"]["raw_col"] == "Region0G"
    assert roi_map["Region1"]["raw_col"] == "Region1G"
    assert roi_map["Region2"]["raw_col"] == "Region2G"
    assert roi_map["Region3"]["raw_col"] == "Region10G"

    # Verify column data mapping is consistent with raw column order.
    np.testing.assert_allclose(chunk.uv_raw[:, 2], np.array([30.0, 31.0]))     # Region2G
    np.testing.assert_allclose(chunk.sig_raw[:, 2], np.array([300.0, 301.0]))  # Region2G
    np.testing.assert_allclose(chunk.uv_raw[:, 3], np.array([110.0, 111.0]))   # Region10G
    np.testing.assert_allclose(chunk.sig_raw[:, 3], np.array([1100.0, 1101.0]))  # Region10G


def test_npm_roi_natural_ordering_preserved_in_pipeline_loaded_chunk(tmp_path):
    cfg = _base_npm_config(target_fs_hz=1.0, chunk_duration_sec=2.0)
    name = "photometryData2025-03-05T15_37_44.csv"
    path = tmp_path / name
    header = [
        "FrameCounter",
        "SystemTimestamp",
        "LedState",
        "Region0G",
        "Region10G",
        "Region2G",
    ]
    rows = [
        [1, 0.0, 1, 10.0, 110.0, 30.0],
        [2, 0.0, 2, 100.0, 1100.0, 300.0],
        [3, 1.0, 1, 11.0, 111.0, 31.0],
        [4, 1.0, 2, 101.0, 1101.0, 301.0],
    ]
    _write_csv(str(path), header, rows)

    pipeline = Pipeline(cfg)
    pipeline.discover_files(str(tmp_path), force_format="npm")
    chunk = load_chunk(pipeline.file_list[0], "npm", cfg, chunk_id=0)
    roi_map = chunk.metadata["roi_map"]

    assert roi_map["Region0"]["raw_col"] == "Region0G"
    assert roi_map["Region1"]["raw_col"] == "Region2G"
    assert roi_map["Region2"]["raw_col"] == "Region10G"


def test_npm_missing_time_column_fails_clearly(tmp_path):
    cfg = _base_npm_config()
    path = tmp_path / "missing_time.csv"
    header = ["FrameCounter", "LedState", "Region0G"]
    rows = [
        [1, 1, 10.0],
        [2, 2, 100.0],
        [3, 1, 20.0],
        [4, 2, 200.0],
    ]
    _write_csv(str(path), header, rows)

    with pytest.raises(ValueError, match="NPM: Missing SystemTimestamp"):
        load_chunk(str(path), "npm", cfg, chunk_id=0)


def test_npm_missing_ledstate_column_fails_clearly(tmp_path):
    cfg = _base_npm_config()
    path = tmp_path / "missing_led_col.csv"
    header = ["FrameCounter", "SystemTimestamp", "Region0G"]
    rows = [
        [1, 0.0, 10.0],
        [2, 0.0, 100.0],
        [3, 1.0, 20.0],
        [4, 1.0, 200.0],
    ]
    _write_csv(str(path), header, rows)

    with pytest.raises(ValueError, match="NPM: Missing LedState"):
        load_chunk(str(path), "npm", cfg, chunk_id=0)


def test_npm_missing_roi_columns_fails_clearly(tmp_path):
    cfg = _base_npm_config()
    path = tmp_path / "missing_rois.csv"
    header = ["FrameCounter", "SystemTimestamp", "LedState", "Amplitude"]
    rows = [
        [1, 0.0, 1, 10.0],
        [2, 0.0, 2, 100.0],
        [3, 1.0, 1, 20.0],
        [4, 1.0, 2, 200.0],
    ]
    _write_csv(str(path), header, rows)

    with pytest.raises(ValueError, match="NPM: No Region columns"):
        load_chunk(str(path), "npm", cfg, chunk_id=0)


def test_npm_auto_sniff_uses_loader_required_columns_not_framecounter(tmp_path):
    cfg = _base_npm_config()
    path = tmp_path / "no_framecounter.csv"
    header = ["SystemTimestamp", "LedState", "Region0G"]
    rows = [
        [0.0, 1, 10.0],
        [0.0, 2, 100.0],
        [1.0, 1, 20.0],
        [1.0, 2, 200.0],
    ]
    _write_csv(str(path), header, rows)

    assert sniff_format(str(path), cfg) == "npm"
