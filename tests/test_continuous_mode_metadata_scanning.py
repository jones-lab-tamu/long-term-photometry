from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.io import adapters


def _write_custom_tabular_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    df = pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_rwd_csv(path: Path, duration_sec: float, fs_hz: float = 5.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    uv = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.01 * t)
    sig = 0.3 + 1.4 * uv + 0.03 * np.sin(2.0 * np.pi * 0.07 * t + 0.1)
    df = pd.DataFrame({"TimeStamp": t, "Region0-410": uv, "Region0-470": sig})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_custom_tabular_times(path: Path, times: list[float]) -> None:
    t = np.asarray(times, dtype=float)
    iso = 1.0 + 0.01 * t
    sig = 2.0 + 0.02 * t
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig}).to_csv(
        path, index=False
    )


def _cfg(*, target_fs_hz: float) -> Config:
    cfg = Config()
    cfg.acquisition_mode = "continuous"
    cfg.continuous_window_sec = 600.0
    cfg.continuous_step_sec = 600.0
    cfg.allow_partial_final_window = False
    cfg.target_fs_hz = float(target_fs_hz)
    cfg.allow_partial_final_chunk = False
    return cfg


def _window_cfg(
    *,
    target_fs_hz: float = 1.0,
    window_sec: float = 600.0,
    allow_partial: bool = False,
) -> Config:
    cfg = _cfg(target_fs_hz=target_fs_hz)
    cfg.continuous_window_sec = float(window_sec)
    cfg.continuous_step_sec = float(window_sec)
    cfg.allow_partial_final_window = bool(allow_partial)
    return cfg


def test_custom_tabular_metadata_scan_reads_header_and_time_only(tmp_path: Path, monkeypatch):
    src = tmp_path / "session_000.csv"
    _write_custom_tabular_csv(src, duration_sec=7200.0, fs_hz=10.0)
    cfg = _cfg(target_fs_hz=10.0)

    calls = []
    orig = adapters.pd.read_csv

    def _spy_read_csv(*args, **kwargs):
        calls.append(dict(kwargs))
        return orig(*args, **kwargs)

    monkeypatch.setattr(adapters.pd, "read_csv", _spy_read_csv)
    adapters.plan_continuous_windows_for_source(str(src), "custom_tabular", cfg, source_cache={})

    assert calls
    time_col = cfg.custom_tabular_time_col
    forbidden_cols = {f"Region0{cfg.custom_tabular_uv_suffix}", f"Region0{cfg.custom_tabular_sig_suffix}"}
    saw_header_only = False
    saw_chunked_time = False
    for kw in calls:
        if kw.get("nrows") == 0:
            saw_header_only = True
            continue
        if "chunksize" in kw:
            saw_chunked_time = True
            assert kw.get("usecols") == [time_col]
            continue
        raise AssertionError(f"Unbounded metadata read detected: {kw}")
        # no coverage pragma
    assert saw_header_only
    assert saw_chunked_time
    for kw in calls:
        usecols = kw.get("usecols")
        if usecols is None:
            continue
        assert not any(col in forbidden_cols for col in usecols)


def test_custom_tabular_metadata_scan_does_not_call_full_time_vector_helper(tmp_path: Path, monkeypatch):
    src = tmp_path / "session_000.csv"
    _write_custom_tabular_csv(src, duration_sec=1200.0, fs_hz=10.0)
    cfg = _cfg(target_fs_hz=10.0)

    def _fail_full_vector_helper(**_kwargs):
        raise AssertionError("continuous planning must not materialize full time vector")

    monkeypatch.setattr(adapters, "_scan_time_column_chunked", _fail_full_vector_helper)
    windows = adapters.plan_continuous_windows_for_source(
        str(src),
        "custom_tabular",
        cfg,
        source_cache={},
    )

    assert len(windows) == 2
    assert all("row_start" in w and "row_stop" in w for w in windows)


def test_rwd_metadata_scan_reads_header_and_time_only(tmp_path: Path, monkeypatch):
    src = tmp_path / "2025_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd_csv(src, duration_sec=7200.0, fs_hz=5.0)
    cfg = _cfg(target_fs_hz=5.0)

    calls = []
    orig = adapters.pd.read_csv

    def _spy_read_csv(*args, **kwargs):
        calls.append(dict(kwargs))
        return orig(*args, **kwargs)

    monkeypatch.setattr(adapters.pd, "read_csv", _spy_read_csv)
    adapters.plan_continuous_windows_for_source(str(src), "rwd", cfg, source_cache={})

    assert calls
    saw_header_only = False
    saw_chunked_time = False
    for kw in calls:
        if kw.get("nrows") == 0:
            saw_header_only = True
            continue
        if "chunksize" in kw:
            saw_chunked_time = True
            assert kw.get("usecols") == ["TimeStamp"]
            continue
        raise AssertionError(f"Unbounded metadata read detected: {kw}")
    assert saw_header_only
    assert saw_chunked_time


def test_rwd_metadata_scan_does_not_call_full_time_vector_helper(tmp_path: Path, monkeypatch):
    src = tmp_path / "2025_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd_csv(src, duration_sec=1200.0, fs_hz=5.0)
    cfg = _cfg(target_fs_hz=5.0)

    def _fail_full_vector_helper(**_kwargs):
        raise AssertionError("continuous planning must not materialize full time vector")

    monkeypatch.setattr(adapters, "_scan_time_column_chunked", _fail_full_vector_helper)
    windows = adapters.plan_continuous_windows_for_source(
        str(src),
        "rwd",
        cfg,
        source_cache={},
    )

    assert len(windows) == 2
    assert all("row_start" in w and "row_stop" in w for w in windows)


def test_continuous_window_row_bounds_match_boundary_semantics(tmp_path: Path):
    src = tmp_path / "session_000.csv"
    _write_custom_tabular_times(src, list(range(0, 1202)))
    cfg = _window_cfg(target_fs_hz=1.0, window_sec=600.0, allow_partial=False)

    windows = adapters.plan_continuous_windows_for_source(
        str(src),
        "custom_tabular",
        cfg,
        source_cache={},
    )

    assert len(windows) == 2
    assert windows[0]["window_start_sec"] == pytest.approx(0.0)
    assert windows[0]["window_end_sec"] == pytest.approx(600.0)
    assert windows[0]["row_start"] == 0
    assert windows[0]["row_stop"] == 601
    assert windows[1]["window_start_sec"] == pytest.approx(600.0)
    assert windows[1]["window_end_sec"] == pytest.approx(1200.0)
    assert windows[1]["row_start"] == 600
    assert windows[1]["row_stop"] == 1201


def test_continuous_partial_final_window_drop_and_keep_semantics(tmp_path: Path):
    src = tmp_path / "session_000.csv"
    _write_custom_tabular_times(src, list(range(0, 1250)))

    cfg_drop = _window_cfg(target_fs_hz=1.0, window_sec=600.0, allow_partial=False)
    dropped = adapters.plan_continuous_windows_for_source(
        str(src),
        "custom_tabular",
        cfg_drop,
        source_cache={},
    )
    assert len(dropped) == 2
    assert all(not w["is_partial_final_window"] for w in dropped)

    cfg_keep = _window_cfg(target_fs_hz=1.0, window_sec=600.0, allow_partial=True)
    kept = adapters.plan_continuous_windows_for_source(
        str(src),
        "custom_tabular",
        cfg_keep,
        source_cache={},
    )
    assert len(kept) == 3
    assert kept[-1]["is_partial_final_window"] is True
    assert kept[-1]["window_start_sec"] == pytest.approx(1200.0)
    assert kept[-1]["window_end_sec"] == pytest.approx(1250.0)
    assert kept[-1]["window_duration_sec"] == pytest.approx(50.0)
    assert kept[-1]["row_start"] == 1200
    assert kept[-1]["row_stop"] == 1250


def test_continuous_time_validation_across_chunk_boundary(tmp_path: Path, monkeypatch):
    src = tmp_path / "session_000.csv"
    _write_custom_tabular_times(src, [0.0, 1.0, 2.0, 2.0, 3.0])
    cfg = _window_cfg(target_fs_hz=1.0, window_sec=2.0, allow_partial=True)
    monkeypatch.setattr(adapters, "_CONTINUOUS_TIME_SCAN_CHUNKSIZE", 3)

    with pytest.raises(ValueError, match="strictly increasing"):
        adapters.plan_continuous_windows_for_source(
            str(src),
            "custom_tabular",
            cfg,
            source_cache={},
        )


def test_continuous_time_validation_rejects_nan_and_nonnumeric(tmp_path: Path):
    src = tmp_path / "session_000.csv"
    src.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time_sec": [0.0, "bad", 2.0],
            "Region0_iso": [1.0, 1.1, 1.2],
            "Region0_sig": [2.0, 2.1, 2.2],
        }
    ).to_csv(src, index=False)
    cfg = _window_cfg(target_fs_hz=1.0, window_sec=2.0, allow_partial=True)

    with pytest.raises(ValueError, match="non-numeric or NaN"):
        adapters.plan_continuous_windows_for_source(
            str(src),
            "custom_tabular",
            cfg,
            source_cache={},
        )


def test_large_time_only_scan_uses_multiple_chunks_and_bounded_metadata(tmp_path: Path, monkeypatch):
    src = tmp_path / "session_000.csv"
    _write_custom_tabular_csv(src, duration_sec=300.0, fs_hz=10.0)
    cfg = _window_cfg(target_fs_hz=10.0, window_sec=60.0, allow_partial=False)
    monkeypatch.setattr(adapters, "_CONTINUOUS_TIME_SCAN_CHUNKSIZE", 128)

    calls = []
    orig = adapters.pd.read_csv

    def _spy_read_csv(*args, **kwargs):
        calls.append(dict(kwargs))
        return orig(*args, **kwargs)

    monkeypatch.setattr(adapters.pd, "read_csv", _spy_read_csv)
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src),
        "custom_tabular",
        cfg,
        source_cache=cache,
    )

    assert len(windows) == 5
    source_meta = next(iter(cache.values()))
    assert "t_rel" not in source_meta
    assert source_meta["n_time_samples"] == 3000
    chunked_time_reads = [
        kw for kw in calls if kw.get("usecols") == ["time_sec"] and "chunksize" in kw
    ]
    assert len(chunked_time_reads) == 2


def test_rwd_metadata_scan_falls_back_to_candidate_time_column(tmp_path: Path, monkeypatch):
    src = tmp_path / "2025_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd_csv(src, duration_sec=1200.0, fs_hz=5.0)
    cfg = _cfg(target_fs_hz=5.0)

    def _fake_detect_header(_path, _config):
        return 0, "missing_time_column"

    monkeypatch.setattr(adapters, "_detect_rwd_header", _fake_detect_header)
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src),
        "rwd",
        cfg,
        source_cache=cache,
    )

    assert windows
    assert cache
    source_meta = next(iter(cache.values()))
    assert source_meta["time_col"] == "TimeStamp"
    assert source_meta["rwd_time_col_resolved"] == "TimeStamp"


def test_custom_tabular_window_load_still_validates_signal_nan(tmp_path: Path):
    src = tmp_path / "session_000.csv"
    _write_custom_tabular_csv(src, duration_sec=1200.0, fs_hz=10.0)
    df = pd.read_csv(src)
    df.loc[6500, "Region0_sig"] = np.nan
    df.to_csv(src, index=False)

    cfg = _cfg(target_fs_hz=10.0)
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src), "custom_tabular", cfg, source_cache=cache
    )
    with pytest.raises(ValueError, match="signal channel columns contain non-numeric/NaN values"):
        adapters.load_chunk(
            str(src),
            "custom_tabular",
            cfg,
            chunk_id=1,
            continuous_window=windows[1],
            source_cache=cache,
        )


def test_rwd_window_load_still_validates_signal_nan(tmp_path: Path):
    src = tmp_path / "2025_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd_csv(src, duration_sec=1200.0, fs_hz=5.0)
    df = pd.read_csv(src)
    df.loc[3500, "Region0-470"] = np.nan
    df.to_csv(src, index=False)

    cfg = _cfg(target_fs_hz=5.0)
    cache = {}
    windows = adapters.plan_continuous_windows_for_source(
        str(src), "rwd", cfg, source_cache=cache
    )
    with pytest.raises(ValueError, match="signal channel columns contain non-numeric/NaN values"):
        adapters.load_chunk(
            str(src),
            "rwd",
            cfg,
            chunk_id=1,
            continuous_window=windows[1],
            source_cache=cache,
        )
