
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from gui.run_report_viewer import RunReportViewer
from photometry_pipeline.continuous_outputs import (
    CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME,
    CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME,
    _sample_elapsed_trace_from_cache,
    generate_continuous_trace_overview_plots,
)
from photometry_pipeline.io.hdf5_cache_reader import open_phasic_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "tools" / "run_full_pipeline_deliverables.py"


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _write_cache(path: Path, *, mode: str, field: str, values_by_chunk: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["mode"] = mode
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("rois", data=np.array(["Region0"], dtype=object), dtype=dt)
        # Deliberately reverse metadata order; reconstruction must sort by window attrs.
        meta.create_dataset("chunk_ids", data=np.array(list(values_by_chunk.keys()), dtype=int))
        meta.create_dataset(
            "source_files",
            data=np.array([f"source_{cid}.csv" for cid in values_by_chunk], dtype=object),
            dtype=dt,
        )
        meta.create_dataset("schema_version", data=np.array([1], dtype=int))
        meta.create_dataset("n_chunks", data=np.array([len(values_by_chunk)], dtype=int))
        roi_group = f.create_group("roi").create_group("Region0")
        for cid, values in values_by_chunk.items():
            grp = roi_group.create_group(f"chunk_{cid}")
            values = np.asarray(values, dtype=float)
            grp.create_dataset("time_sec", data=np.arange(values.size, dtype=float))
            grp.create_dataset("sig_raw", data=values + 10.0)
            grp.create_dataset("uv_raw", data=values + 5.0)
            grp.create_dataset(field, data=values)
            start = 600.0 * cid
            grp.attrs["fs_hz"] = 1.0
            grp.attrs["source_file"] = f"source_{cid}.csv"
            grp.attrs["acquisition_mode"] = "continuous"
            grp.attrs["window_index"] = cid
            grp.attrs["window_start_sec"] = start
            grp.attrs["window_end_sec"] = start + float(values.size)
            grp.attrs["window_duration_sec"] = float(values.size)
            grp.attrs["original_file_duration_sec"] = 1200.0
            grp.attrs["continuous_window_sec"] = 600.0
            grp.attrs["continuous_step_sec"] = 600.0
            grp.attrs["is_partial_final_window"] = False


def _write_custom_tabular_input(path: Path, *, duration_sec: float = 1200.0, fs_hz: float = 10.0) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(duration_sec * fs_hz), dtype=float) / fs_hz
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig}).to_csv(path, index=False)


def _write_continuous_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "target_fs_hz: 10.0",
                "chunk_duration_sec: 600.0",
                "allow_partial_final_chunk: false",
                "acquisition_mode: continuous",
                "continuous_window_sec: 600.0",
                "continuous_step_sec: 600.0",
                "allow_partial_final_window: false",
                "custom_tabular_time_col: time_sec",
                "custom_tabular_uv_suffix: _iso",
                "custom_tabular_sig_suffix: _sig",
                "baseline_method: uv_raw_percentile_session",
                "baseline_percentile: 10",
                "peak_threshold_method: mean_std",
                "peak_threshold_k: 1.0",
                "peak_min_distance_sec: 5.0",
                "window_sec: 20.0",
                "step_sec: 5.0",
                "r_low: -1.0",
                "r_high: 1.0",
                "g_min: 0.0",
                "min_valid_windows: 1",
                "min_samples_per_window: 20",
                "lowpass_hz: 2.0",
                "qc_max_chunk_fail_fraction: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_wrapper(tmp_path: Path, mode: str) -> Path:
    input_dir = tmp_path / f"input_{mode}"
    out_dir = tmp_path / f"out_{mode}"
    cfg_path = tmp_path / f"cfg_{mode}.yaml"
    _write_custom_tabular_input(input_dir / "continuous.csv")
    _write_continuous_config(cfg_path)
    cmd = [
        sys.executable,
        str(WRAPPER),
        "--input",
        str(input_dir),
        "--out",
        str(out_dir),
        "--config",
        str(cfg_path),
        "--format",
        "custom_tabular",
        "--mode",
        mode,
        "--overwrite",
    ]
    env = dict(__import__("os").environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{__import__('os').pathsep}{existing}"
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    return out_dir


def test_sample_elapsed_trace_sorts_by_window_metadata(tmp_path: Path):
    cache_path = tmp_path / "phasic_trace_cache.h5"
    _write_cache(
        cache_path,
        mode="phasic",
        field="dff",
        values_by_chunk={1: np.array([10.0, 11.0]), 0: np.array([1.0, 2.0])},
    )

    with open_phasic_cache(str(cache_path)) as cache:
        elapsed, values, details = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=100,
        )

    np.testing.assert_allclose(elapsed, np.array([0.0, 1.0, 600.0, 601.0]))
    np.testing.assert_allclose(values, np.array([1.0, 2.0, 10.0, 11.0]))
    assert details["chunk_ids"] == [0, 1]
    assert details["n_chunks"] == 2
    assert details["n_samples_seen"] == 4
    assert details["n_points_plotted"] == 4
    assert float(elapsed.max()) > 600.0


def test_large_multichunk_cache_samples_at_most_max_plot_points(tmp_path: Path):
    cache_path = tmp_path / "phasic_trace_cache.h5"
    values_by_chunk = {
        cid: np.linspace(float(cid), float(cid) + 0.5, 100)
        for cid in range(20)
    }
    _write_cache(cache_path, mode="phasic", field="dff", values_by_chunk=values_by_chunk)

    with open_phasic_cache(str(cache_path)) as cache:
        elapsed, values, details = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=101,
        )

    assert len(elapsed) <= 101
    assert len(values) <= 101
    assert details["n_points_plotted"] <= 101
    assert details["n_chunks"] == 20
    assert details["n_samples_seen"] == 2000
    assert details["n_finite_samples"] == 2000
    assert details["elapsed_hour_start"] == pytest.approx(0.0)
    assert details["elapsed_hour_end"] == pytest.approx(((19 * 600.0) + 99.0) / 3600.0)
    assert elapsed[-1] == pytest.approx((19 * 600.0) + 99.0)


def test_chunk_coverage_preserves_first_and_last_chunks(tmp_path: Path):
    cache_path = tmp_path / "phasic_trace_cache.h5"
    values_by_chunk = {
        cid: np.full(50, float(cid))
        for cid in range(5)
    }
    _write_cache(cache_path, mode="phasic", field="dff", values_by_chunk=values_by_chunk)

    with open_phasic_cache(str(cache_path)) as cache:
        elapsed, values, details = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=11,
        )

    assert len(values) <= 11
    assert 0.0 in set(values.tolist())
    assert 4.0 in set(values.tolist())
    assert details["finite_chunk_ids"] == [0, 1, 2, 3, 4]
    assert details["elapsed_hour_start"] == pytest.approx(0.0)
    assert details["elapsed_hour_end"] == pytest.approx(((4 * 600.0) + 49.0) / 3600.0)


def test_sampling_preserves_true_final_point_when_one_point_per_chunk(tmp_path: Path):
    cache_path = tmp_path / "phasic_trace_cache.h5"
    values_by_chunk = {
        cid: np.linspace(float(cid), float(cid) + 0.25, 50)
        for cid in range(5)
    }
    _write_cache(cache_path, mode="phasic", field="dff", values_by_chunk=values_by_chunk)

    with open_phasic_cache(str(cache_path)) as cache:
        elapsed, _values, details = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=5,
        )

    assert details["n_points_plotted"] <= 5
    assert elapsed[0] == pytest.approx(0.0)
    assert elapsed[-1] == pytest.approx((4 * 600.0) + 49.0)
    assert details["elapsed_hour_start"] == pytest.approx(0.0)
    assert details["elapsed_hour_end"] == pytest.approx(((4 * 600.0) + 49.0) / 3600.0)


def test_sampling_preserves_first_and_last_when_finite_chunks_exceed_max_points(tmp_path: Path):
    cache_path = tmp_path / "phasic_trace_cache.h5"
    values_by_chunk = {
        cid: np.linspace(float(cid), float(cid) + 0.25, 30)
        for cid in range(20)
    }
    _write_cache(cache_path, mode="phasic", field="dff", values_by_chunk=values_by_chunk)

    with open_phasic_cache(str(cache_path)) as cache:
        elapsed, _values, details = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=7,
        )

    assert details["n_points_plotted"] <= 7
    assert elapsed[0] == pytest.approx(0.0)
    assert elapsed[-1] == pytest.approx((19 * 600.0) + 29.0)
    assert details["elapsed_hour_start"] == pytest.approx(0.0)
    assert details["elapsed_hour_end"] == pytest.approx(((19 * 600.0) + 29.0) / 3600.0)


def test_sampling_preserves_last_point_with_nan_middle_chunk(tmp_path: Path):
    cache_path = tmp_path / "phasic_trace_cache.h5"
    _write_cache(
        cache_path,
        mode="phasic",
        field="dff",
        values_by_chunk={
            0: np.linspace(1.0, 2.0, 10),
            1: np.full(10, np.nan),
            2: np.linspace(3.0, 4.0, 10),
        },
    )

    with open_phasic_cache(str(cache_path)) as cache:
        elapsed, _values, details = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=3,
        )

    assert details["n_points_plotted"] <= 3
    assert details["finite_chunk_ids"] == [0, 2]
    assert elapsed[0] == pytest.approx(0.0)
    assert elapsed[-1] == pytest.approx(1200.0 + 9.0)
    assert details["elapsed_hour_end"] == pytest.approx((1200.0 + 9.0) / 3600.0)


def test_bounded_sampling_is_deterministic(tmp_path: Path):
    cache_path = tmp_path / "phasic_trace_cache.h5"
    values_by_chunk = {
        cid: np.linspace(float(cid), float(cid) + 1.0, 75)
        for cid in range(7)
    }
    _write_cache(cache_path, mode="phasic", field="dff", values_by_chunk=values_by_chunk)

    with open_phasic_cache(str(cache_path)) as cache:
        elapsed1, values1, details1 = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=31,
        )
    with open_phasic_cache(str(cache_path)) as cache:
        elapsed2, values2, details2 = _sample_elapsed_trace_from_cache(
            cache,
            "Region0",
            "dff",
            max_points=31,
        )

    np.testing.assert_array_equal(elapsed1, elapsed2)
    np.testing.assert_array_equal(values1, values2)
    assert details1 == details2


def test_generate_trace_overview_plots_from_tonic_and_phasic_caches(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_cache(
        run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5",
        mode="tonic",
        field="deltaF",
        values_by_chunk={1: np.array([0.3, 0.4]), 0: np.array([0.1, 0.2])},
    )
    _write_cache(
        run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        mode="phasic",
        field="dff",
        values_by_chunk={1: np.array([3.0, 4.0]), 0: np.array([1.0, 2.0])},
    )

    result = generate_continuous_trace_overview_plots(str(run_dir), mode="both", max_plot_points=3)

    assert result["generated"] is True
    assert f"Region0/summary/{CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME}" in result["plots"]
    assert f"Region0/summary/{CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME}" in result["plots"]
    assert (run_dir / "Region0" / "summary" / CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME).exists()
    assert (run_dir / "Region0" / "summary" / CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME).exists()
    assert result["details"]["tonic"]["Region0"]["n_points_plotted"] <= 3


def test_trace_overview_mode_specific_skips(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_cache(
        run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        mode="phasic",
        field="dff",
        values_by_chunk={0: np.array([1.0, 2.0])},
    )

    result = generate_continuous_trace_overview_plots(str(run_dir), mode="phasic")

    assert f"Region0/summary/{CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME}" in result["plots"]
    assert any(skip["reason"] == "tonic mode not requested" for skip in result["skips"])
    assert not (run_dir / "Region0" / "summary" / CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME).exists()


def test_trace_overview_skip_does_not_create_empty_summary_folder(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_cache(
        run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        mode="phasic",
        field="dff",
        values_by_chunk={0: np.array([np.nan, np.nan])},
    )

    result = generate_continuous_trace_overview_plots(str(run_dir), mode="phasic")

    assert result["plots"] == []
    assert not (run_dir / "Region0" / "summary").exists()
    assert any("No finite values" in skip["reason"] for skip in result["skips"])


def test_nan_only_chunk_does_not_break_other_finite_chunks(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_cache(
        run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        mode="phasic",
        field="dff",
        values_by_chunk={
            0: np.array([1.0, 2.0, 3.0]),
            1: np.array([np.nan, np.nan, np.nan]),
            2: np.array([4.0, 5.0, 6.0]),
        },
    )

    result = generate_continuous_trace_overview_plots(str(run_dir), mode="phasic")

    assert f"Region0/summary/{CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME}" in result["plots"]
    details = result["details"]["phasic"]["Region0"]
    assert details["n_chunks"] == 3
    assert details["n_samples_seen"] == 9
    assert details["n_finite_samples"] == 6
    assert details["finite_chunk_ids"] == [0, 2]
    assert details["elapsed_hour_start"] == pytest.approx(0.0)
    assert details["elapsed_hour_end"] == pytest.approx((1200.0 + 2.0) / 3600.0)


def test_wrapper_records_trace_overview_plots_in_manifest_and_status(tmp_path: Path):
    out_dir = _run_wrapper(tmp_path, "both")

    tonic = f"Region0/summary/{CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME}"
    phasic = f"Region0/summary/{CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME}"
    assert (out_dir / tonic).exists()
    assert (out_dir / phasic).exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    for payload in (manifest["continuous_outputs"], status["continuous_outputs"]):
        assert payload["trace_overview_plots_generated"] is True
        assert tonic in payload["trace_overview_plots"]
        assert phasic in payload["trace_overview_plots"]
        assert "trace_overview_details" in payload
        assert "summary_plots" in payload


def test_gui_viewer_exposes_continuous_trace_tab(tmp_path: Path, qapp):
    out_dir = _run_wrapper(tmp_path, "both")

    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(out_dir)) is True
        tab_map = {
            tab: [Path(path).name for path in paths]
            for tab, paths in viewer._region_tab_images["Region0"].items()
        }
        assert tab_map["Continuous Trace"] == [
            CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME,
            CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME,
        ]
        assert tab_map["Tonic"] == ["tonic_overview.png"]
        assert tab_map["Phasic Summary"] == [
            "phasic_auc_timeseries.png",
            "phasic_peak_rate_timeseries.png",
        ]
    finally:
        viewer.close()
