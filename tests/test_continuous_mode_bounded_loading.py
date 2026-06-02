from pathlib import Path

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
        assert "t_rel" in entry
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
        assert "t_rel" in entry
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
