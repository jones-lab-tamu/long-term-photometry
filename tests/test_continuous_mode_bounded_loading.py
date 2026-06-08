from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.io import adapters
from photometry_pipeline.pipeline import Pipeline


def _write_custom_tabular_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    df = pd.DataFrame(
        {
            "time_sec": t,
            "Region0_iso": iso,
            "Region0_sig": sig,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_custom_tabular_two_roi_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso0 = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig0 = 2.0 + 0.9 * iso0 + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    iso1 = 1.2 + 0.015 * np.sin(2.0 * np.pi * 0.03 * t + 0.1)
    sig1 = 1.5 + 1.1 * iso1 + 0.03 * np.sin(2.0 * np.pi * 0.07 * t + 0.4)
    df = pd.DataFrame(
        {
            "time_sec": t,
            "Region0_iso": iso0,
            "Region0_sig": sig0,
            "Region1_iso": iso1,
            "Region1_sig": sig1,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_rwd_csv(path: Path, duration_sec: float, fs_hz: float = 5.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    uv = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.01 * t)
    sig = 0.3 + 1.4 * uv + 0.03 * np.sin(2.0 * np.pi * 0.07 * t + 0.1)
    df = pd.DataFrame(
        {
            "TimeStamp": t,
            "Region0-410": uv,
            "Region0-470": sig,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _continuous_cfg(*, target_fs_hz: float = 10.0) -> Config:
    cfg = Config()
    cfg.acquisition_mode = "continuous"
    cfg.continuous_window_sec = 600.0
    cfg.continuous_step_sec = 600.0
    cfg.allow_partial_final_window = False
    cfg.chunk_duration_sec = 600.0
    cfg.target_fs_hz = float(target_fs_hz)
    cfg.allow_partial_final_chunk = False
    return cfg


def test_continuous_source_cache_stores_metadata_only(tmp_path: Path):
    custom_dir = tmp_path / "custom"
    _write_custom_tabular_csv(custom_dir / "session_000.csv", duration_sec=7200.0, fs_hz=10.0)
    cfg_custom = _continuous_cfg(target_fs_hz=10.0)
    p_custom = Pipeline(cfg_custom, mode="phasic")
    p_custom.discover_files(str(custom_dir), force_format="custom_tabular")
    assert p_custom._continuous_source_cache
    for entry in p_custom._continuous_source_cache.values():
        assert "uv_raw" not in entry
        assert "sig_raw" not in entry
        assert "stacked" not in entry
        assert "t_rel" not in entry
        assert "duration_sec" in entry
        assert "median_dt_sec" in entry
        assert "n_time_samples" in entry
        assert int(entry.get("n_rows", 0)) > 0

    rwd_dir = tmp_path / "rwd"
    _write_rwd_csv(
        rwd_dir / "2025_01_01-00_00_00" / "fluorescence.csv",
        duration_sec=7200.0,
        fs_hz=5.0,
    )
    cfg_rwd = _continuous_cfg(target_fs_hz=5.0)
    p_rwd = Pipeline(cfg_rwd, mode="phasic")
    p_rwd.discover_files(str(rwd_dir), force_format="rwd")
    assert p_rwd._continuous_source_cache
    for entry in p_rwd._continuous_source_cache.values():
        assert "uv_raw" not in entry
        assert "sig_raw" not in entry
        assert "stacked" not in entry
        assert "t_rel" not in entry
        assert "duration_sec" in entry
        assert "median_dt_sec" in entry
        assert "n_time_samples" in entry
        assert int(entry.get("n_rows", 0)) > 0


@pytest.mark.parametrize(
    "fmt,fs_hz,duration_sec,write_fn",
    [
        ("custom_tabular", 10.0, 7200.0, _write_custom_tabular_csv),
        ("rwd", 5.0, 7200.0, _write_rwd_csv),
    ],
)
def test_continuous_window_reads_are_bounded(tmp_path: Path, monkeypatch, fmt, fs_hz, duration_sec, write_fn):
    if fmt == "custom_tabular":
        src_path = tmp_path / "input" / "session_000.csv"
    else:
        src_path = tmp_path / "input" / "2025_01_01-00_00_00" / "fluorescence.csv"
    write_fn(src_path, duration_sec=duration_sec, fs_hz=fs_hz)

    cfg = _continuous_cfg(target_fs_hz=fs_hz)
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src_path),
        fmt,
        cfg,
        source_cache=cache,
    )
    assert len(windows) >= 10

    rows_read = []
    orig = adapters._read_csv_window_rows

    def _spy_read_csv_window_rows(**kwargs):
        rows_read.append(int(kwargs["end_idx"]) - int(kwargs["start_idx"]) + 1)
        return orig(**kwargs)

    monkeypatch.setattr(adapters, "_read_csv_window_rows", _spy_read_csv_window_rows)

    for idx, win in enumerate(windows):
        chunk = adapters.load_chunk(
            str(src_path),
            fmt,
            cfg,
            chunk_id=idx,
            continuous_window=win,
            source_cache=cache,
        )
        expected = int(round(float(win["window_duration_sec"]) * fs_hz))
        assert chunk.sig_raw.shape[0] == expected
        assert chunk.uv_raw.shape[0] == expected

    assert rows_read
    total_rows = int(round(duration_sec * fs_hz))
    assert max(rows_read) < total_rows
    assert max(rows_read) <= int(round(cfg.continuous_window_sec * fs_hz)) + 10


def test_custom_tabular_sequential_iterator_matches_bounded_loader(tmp_path: Path):
    src_path = tmp_path / "input" / "session_000.csv"
    _write_custom_tabular_two_roi_csv(src_path, duration_sec=12.0, fs_hz=10.0)
    cfg = _continuous_cfg(target_fs_hz=10.0)
    cfg.continuous_window_sec = 4.0
    cfg.continuous_step_sec = 4.0
    cfg.chunk_duration_sec = 4.0
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )
    source = adapters.resolve_continuous_source_metadata(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )

    bounded = [
        adapters.load_chunk(
            str(src_path),
            "custom_tabular",
            cfg,
            chunk_id=i,
            continuous_window=win,
            source_cache=cache,
        )
        for i, win in enumerate(windows)
    ]
    sequential = [
        chunk
        for _chunk_id, _win, chunk in adapters.iter_continuous_custom_tabular_chunks(
            source,
            windows,
            cfg,
            read_chunksize=17,
        )
    ]

    assert len(sequential) == len(bounded)
    for expected, actual in zip(bounded, sequential):
        assert actual.chunk_id == expected.chunk_id
        assert actual.channel_names == expected.channel_names
        assert actual.metadata["roi_map"] == expected.metadata["roi_map"]
        assert actual.metadata["window_index"] == expected.metadata["window_index"]
        np.testing.assert_allclose(actual.time_sec, expected.time_sec, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(actual.uv_raw, expected.uv_raw, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(actual.sig_raw, expected.sig_raw, rtol=0.0, atol=1e-12)


def test_custom_tabular_sequential_iterator_rejects_unsorted_windows(tmp_path: Path):
    src_path = tmp_path / "input" / "session_000.csv"
    _write_custom_tabular_two_roi_csv(src_path, duration_sec=12.0, fs_hz=10.0)
    cfg = _continuous_cfg(target_fs_hz=10.0)
    cfg.continuous_window_sec = 4.0
    cfg.continuous_step_sec = 4.0
    cfg.chunk_duration_sec = 4.0
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )
    source = adapters.resolve_continuous_source_metadata(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )

    with pytest.raises(
        ValueError,
        match="requires windows in source row order",
    ):
        list(
            adapters.iter_continuous_custom_tabular_chunks(
                source,
                list(reversed(windows)),
                cfg,
                chunk_ids=list(reversed(range(len(windows)))),
                read_chunksize=17,
            )
        )


def test_custom_tabular_sequential_iterator_rejects_ambiguous_overlap(tmp_path: Path):
    src_path = tmp_path / "input" / "session_000.csv"
    _write_custom_tabular_two_roi_csv(src_path, duration_sec=12.0, fs_hz=10.0)
    cfg = _continuous_cfg(target_fs_hz=10.0)
    cfg.continuous_window_sec = 4.0
    cfg.continuous_step_sec = 4.0
    cfg.chunk_duration_sec = 4.0
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )
    source = adapters.resolve_continuous_source_metadata(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )
    ambiguous = [dict(windows[0]), dict(windows[1])]
    ambiguous[0]["window_end_sec"] = float("nan")
    ambiguous[1]["row_start"] = int(ambiguous[0]["row_stop"]) - 1

    with pytest.raises(
        ValueError,
        match="does not support overlapping windows",
    ):
        list(
            adapters.iter_continuous_custom_tabular_chunks(
                source,
                ambiguous,
                cfg,
                read_chunksize=17,
            )
        )


def test_custom_tabular_pipeline_overlapping_windows_use_bounded_fallback(monkeypatch):
    cfg = _continuous_cfg(target_fs_hz=10.0)
    cfg.continuous_window_sec = 6.0
    cfg.continuous_step_sec = 3.0
    p = Pipeline(cfg, mode="phasic")
    p.file_list = ["source.csv::window_000000", "source.csv::window_000001"]
    p._continuous_window_map = {
        p.file_list[0]: {
            "source_file": "source.csv",
            "format": "custom_tabular",
            "row_start": 0,
            "row_stop": 60,
            "window_index": 0,
        },
        p.file_list[1]: {
            "source_file": "source.csv",
            "format": "custom_tabular",
            "row_start": 30,
            "row_stop": 90,
            "window_index": 1,
        },
    }

    def _fake_load_entry_chunk(_entry, chunk_id, _force_format):
        n = 60
        t = np.arange(n, dtype=float) / 10.0
        return adapters.Chunk(
            chunk_id=chunk_id,
            source_file="source.csv",
            format="custom_tabular",
            time_sec=t,
            uv_raw=np.ones((n, 1), dtype=float),
            sig_raw=np.ones((n, 1), dtype=float) * 2.0,
            fs_hz=10.0,
            channel_names=["Region0"],
            metadata={},
        )

    monkeypatch.setattr(p, "_load_entry_chunk", _fake_load_entry_chunk)

    yielded = list(p._iter_entry_chunks_for_pass(p.file_list, "custom_tabular", "pass1"))

    assert len(yielded) == 2
    assert p._continuous_csv_reading["sequential_csv_reading_used"] is False
    assert p._continuous_csv_reading["bounded_loader_fallback_count"] == 2
    assert p._continuous_csv_reading["phases"]["pass1"]["bounded_loader_fallback_count"] == 2


def test_custom_tabular_pipeline_ambiguous_overlap_uses_bounded_fallback(monkeypatch):
    cfg = _continuous_cfg(target_fs_hz=10.0)
    p = Pipeline(cfg, mode="phasic")
    p.file_list = ["source.csv::window_000000", "source.csv::window_000001"]
    p._continuous_window_map = {
        p.file_list[0]: {
            "source_file": "source.csv",
            "format": "custom_tabular",
            "row_start": 0,
            "row_stop": 60,
            "window_index": 0,
            "window_start_sec": 0.0,
            "window_end_sec": float("nan"),
        },
        p.file_list[1]: {
            "source_file": "source.csv",
            "format": "custom_tabular",
            "row_start": 59,
            "row_stop": 120,
            "window_index": 1,
            "window_start_sec": 60.0,
            "window_end_sec": 120.0,
        },
    }

    def _fake_load_entry_chunk(_entry, chunk_id, _force_format):
        n = 60
        t = np.arange(n, dtype=float) / 10.0
        return adapters.Chunk(
            chunk_id=chunk_id,
            source_file="source.csv",
            format="custom_tabular",
            time_sec=t,
            uv_raw=np.ones((n, 1), dtype=float),
            sig_raw=np.ones((n, 1), dtype=float) * 2.0,
            fs_hz=10.0,
            channel_names=["Region0"],
            metadata={},
        )

    monkeypatch.setattr(p, "_load_entry_chunk", _fake_load_entry_chunk)

    assert p._continuous_entries_support_sequential_custom_tabular(p.file_list) is False
    yielded = list(p._iter_entry_chunks_for_pass(p.file_list, "custom_tabular", "pass1"))

    assert len(yielded) == 2
    assert p._continuous_csv_reading["sequential_csv_reading_used"] is False
    assert p._continuous_csv_reading["bounded_loader_fallback_count"] == 2
    assert p._continuous_csv_reading["phases"]["pass1"]["bounded_loader_fallback_count"] == 2


def test_custom_tabular_sequential_iterator_partial_final_window_matches_bounded_loader(tmp_path: Path):
    src_path = tmp_path / "input" / "session_000.csv"
    _write_custom_tabular_csv(src_path, duration_sec=10.0, fs_hz=10.0)
    cfg = _continuous_cfg(target_fs_hz=10.0)
    cfg.continuous_window_sec = 4.0
    cfg.continuous_step_sec = 4.0
    cfg.chunk_duration_sec = 4.0
    cfg.allow_partial_final_window = False
    cache = {}
    full_only = adapters.plan_continuous_windows_for_source(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )
    assert len(full_only) == 2
    cfg.allow_partial_final_window = True
    cache = {}
    with_partial = adapters.plan_continuous_windows_for_source(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )
    assert len(with_partial) == 3
    source = adapters.resolve_continuous_source_metadata(
        str(src_path),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )
    bounded_last = adapters.load_chunk(
        str(src_path),
        "custom_tabular",
        cfg,
        chunk_id=2,
        continuous_window=with_partial[-1],
        source_cache=cache,
    )
    sequential = list(
        adapters.iter_continuous_custom_tabular_chunks(
            source,
            with_partial,
            cfg,
            read_chunksize=13,
        )
    )
    assert len(sequential) == 3
    sequential_last = sequential[-1][2]
    assert sequential_last.metadata["is_partial_final_window"] is True
    np.testing.assert_allclose(sequential_last.time_sec, bounded_last.time_sec, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sequential_last.uv_raw, bounded_last.uv_raw, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sequential_last.sig_raw, bounded_last.sig_raw, rtol=0.0, atol=1e-12)


def test_custom_tabular_sequential_pipeline_avoids_skiprows_loader_and_records_provenance(
    tmp_path: Path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    _write_custom_tabular_two_roi_csv(input_dir / "session_000.csv", duration_sec=12.0, fs_hz=10.0)
    cfg = _continuous_cfg(target_fs_hz=10.0)
    cfg.continuous_window_sec = 4.0
    cfg.continuous_step_sec = 4.0
    cfg.chunk_duration_sec = 4.0
    cfg.representative_session_index = 0

    def _forbid_skiprows_loader(**_kwargs):
        raise AssertionError("sequential continuous custom_tabular path should not use skiprows loader")

    monkeypatch.setattr(adapters, "_read_csv_window_rows", _forbid_skiprows_loader)
    p = Pipeline(cfg, mode="phasic")
    p.run(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        force_format="custom_tabular",
        recursive=False,
        include_rois=["Region1"],
        traces_only=True,
        sessions_per_hour=None,
    )

    assert p.roi_selection["discovered_rois"] == ["Region0", "Region1"]
    assert p.roi_selection["selected_rois"] == ["Region1"]
    stats = p._continuous_csv_reading
    assert stats["sequential_csv_reading_used"] is True
    assert stats["source_csv_open_read_passes"] >= 2
    assert stats["windows_yielded_sequentially"] >= 6
    assert stats["bounded_loader_fallback_count"] == 0

    report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    provenance = report["derived_settings"]["continuous_csv_reading"]
    assert provenance["sequential_csv_reading_used"] is True
    assert provenance["bounded_loader_fallback_count"] == 0


def test_custom_tabular_full_pipeline_sequential_matches_bounded_fallback(
    tmp_path: Path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    old_out = tmp_path / "old_bounded"
    new_out = tmp_path / "new_sequential"
    _write_custom_tabular_two_roi_csv(input_dir / "session_000.csv", duration_sec=18.0, fs_hz=10.0)

    def _make_cfg() -> Config:
        cfg = _continuous_cfg(target_fs_hz=10.0)
        cfg.continuous_window_sec = 6.0
        cfg.continuous_step_sec = 6.0
        cfg.chunk_duration_sec = 6.0
        cfg.representative_session_index = 0
        cfg.peak_threshold_k = 1.5
        cfg.peak_min_prominence_k = 0.5
        cfg.peak_min_width_sec = 0.1
        cfg.dynamic_fit_mode = "global_linear_regression"
        return cfg

    cfg_old = _make_cfg()
    p_old = Pipeline(cfg_old, mode="phasic")
    monkeypatch.setattr(
        Pipeline,
        "_continuous_entries_support_sequential_custom_tabular",
        lambda self, entries: False,
    )
    p_old.run(
        input_dir=str(input_dir),
        output_dir=str(old_out),
        force_format="custom_tabular",
        recursive=False,
        include_rois=["Region0", "Region1"],
        traces_only=False,
        sessions_per_hour=None,
    )

    monkeypatch.undo()
    cfg_new = _make_cfg()
    p_new = Pipeline(cfg_new, mode="phasic")
    p_new.run(
        input_dir=str(input_dir),
        output_dir=str(new_out),
        force_format="custom_tabular",
        recursive=False,
        include_rois=["Region0", "Region1"],
        traces_only=False,
        sessions_per_hour=None,
    )

    assert p_old.roi_selection["discovered_rois"] == p_new.roi_selection["discovered_rois"]
    assert p_old.roi_selection["selected_rois"] == p_new.roi_selection["selected_rois"]
    assert p_old.n_sessions_resolved == p_new.n_sessions_resolved == 3
    assert p_old.stats.f0_values.keys() == p_new.stats.f0_values.keys()
    for roi in p_old.stats.f0_values:
        assert p_old.stats.f0_values[roi] == pytest.approx(p_new.stats.f0_values[roi], abs=1e-12)

    old_features = pd.read_csv(old_out / "features" / "features.csv")
    new_features = pd.read_csv(new_out / "features" / "features.csv")
    assert len(old_features) == len(new_features) == 6
    assert int(old_features["peak_count"].sum()) == int(new_features["peak_count"].sum())
    pd.testing.assert_frame_equal(
        old_features.sort_index(axis=1),
        new_features.sort_index(axis=1),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )

    old_meta = json.loads((old_out / "run_metadata.json").read_text(encoding="utf-8"))
    new_meta = json.loads((new_out / "run_metadata.json").read_text(encoding="utf-8"))
    for key in [
        "target_fs_hz",
        "seed",
        "allow_partial_final_chunk",
        "roi_map",
        "baseline_method",
        "f0_values",
        "acquisition_mode",
        "continuous_window_sec",
        "continuous_step_sec",
        "allow_partial_final_window",
        "continuous_planned_window_count",
    ]:
        assert old_meta[key] == new_meta[key]

    old_report = json.loads((old_out / "run_report.json").read_text(encoding="utf-8"))
    new_report = json.loads((new_out / "run_report.json").read_text(encoding="utf-8"))
    assert old_report["configuration"] == new_report["configuration"]
    assert old_report["roi_selection"] == new_report["roi_selection"]
    assert old_report["run_context"]["traces_only"] == new_report["run_context"]["traces_only"]
    assert old_report["run_context"]["representative_session_index"] == new_report["run_context"]["representative_session_index"]

    with h5py.File(old_out / "phasic_trace_cache.h5", "r") as old_h5, h5py.File(
        new_out / "phasic_trace_cache.h5", "r"
    ) as new_h5:
        assert int(old_h5["meta"]["n_chunks"][0]) == int(new_h5["meta"]["n_chunks"][0]) == 3
        assert sorted(old_h5["roi"].keys()) == sorted(new_h5["roi"].keys()) == ["Region0", "Region1"]
        for roi in ["Region0", "Region1"]:
            for chunk_id in range(3):
                old_grp = old_h5["roi"][roi][f"chunk_{chunk_id}"]
                new_grp = new_h5["roi"][roi][f"chunk_{chunk_id}"]
                for attr in [
                    "acquisition_mode",
                    "window_index",
                    "window_start_sec",
                    "window_end_sec",
                    "window_duration_sec",
                    "continuous_window_sec",
                    "continuous_step_sec",
                    "is_partial_final_window",
                ]:
                    assert old_grp.attrs[attr] == new_grp.attrs[attr]
                for field in ["time_sec", "sig_raw", "uv_raw", "fit_ref", "delta_f", "dff"]:
                    np.testing.assert_allclose(
                        old_grp[field][...],
                        new_grp[field][...],
                        rtol=1e-12,
                        atol=1e-12,
                    )

    assert old_meta["continuous_csv_reading"]["sequential_csv_reading_used"] is False
    assert new_meta["continuous_csv_reading"]["sequential_csv_reading_used"] is True
