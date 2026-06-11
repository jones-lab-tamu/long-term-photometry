import os

import h5py
import numpy as np
import pandas as pd
import pytest

from tools import plot_reference_candidate_comparison as plotter


def _write_cache(path, *, include_fit_ref=True, baseline_trace=None):
    t = np.arange(1000, dtype=float) / 10.0
    slow = np.sin(2.0 * np.pi * t / 300.0)
    uv = 1.0 + 0.25 * slow
    sig = 2.0 + 0.5 * slow + 0.03 * np.sin(2.0 * np.pi * t / 12.0)
    fit = 2.0 + 0.5 * slow
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1"
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("rois", data=np.array(["CH3"], dtype=object), dtype=dt)
        meta.create_dataset("chunk_ids", data=np.array([31], dtype=int))
        roi_group = f.create_group("roi").create_group("CH3")
        chunk = roi_group.create_group("chunk_31")
        chunk.create_dataset("time_sec", data=t)
        chunk.create_dataset("sig_raw", data=sig)
        chunk.create_dataset("uv_raw", data=uv)
        if include_fit_ref:
            chunk.create_dataset("fit_ref", data=fit)
        if baseline_trace is not None:
            chunk.create_dataset("baseline_ref_candidate", data=np.asarray(baseline_trace, dtype=float))


def _write_candidate_csv(path, *, roi="CH3", chunk_id=31, window_sec=30.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(
        [
            {
                "roi": roi,
                "chunk_id": chunk_id,
                "dynamic_fit_qc_severity": "context",
                "dynamic_fit_qc_soft_flags": "FITTED_REFERENCE_RESPONSE_SCALE_RICH",
                "dynamic_fit_qc_hard_flags": "",
                "reference_comparison_class": "dynamic_context_baseline_viable",
                "reference_comparison_flags": "DYNAMIC_RESPONSE_SCALE_RICH",
                "baseline_ref_candidate_available": True,
                "baseline_ref_actual_smoothing_window_sec": window_sec,
                "baseline_ref_min_smoothing_window_sec": 10.0,
                "baseline_ref_max_window_fraction_of_chunk": 0.75,
                "baseline_ref_large_window_fraction_warning": 0.50,
                "baseline_ref_smoothing_window_warning": "",
            }
        ]
    ).to_csv(path, index=False)


def _make_analysis_out(tmp_path, *, include_fit_ref=True, baseline_trace=None):
    analysis_out = tmp_path / "_analysis" / "phasic_out"
    analysis_out.mkdir(parents=True)
    _write_cache(
        analysis_out / "phasic_trace_cache.h5",
        include_fit_ref=include_fit_ref,
        baseline_trace=baseline_trace,
    )
    _write_candidate_csv(analysis_out / "qc" / "baseline_reference_candidate_by_chunk.csv")
    return analysis_out


def test_plotter_creates_reference_candidate_png(tmp_path):
    analysis_out = _make_analysis_out(tmp_path)
    out_png = tmp_path / "comparison.png"

    result = plotter.plot_one(
        analysis_out=str(analysis_out),
        roi="CH3",
        chunk_id=31,
        output_path=str(out_png),
    )

    assert result == str(out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plotter_prefers_stored_hdf5_candidate_trace(tmp_path, monkeypatch):
    stored_trace = np.linspace(1.0, 2.0, 1000)
    analysis_out = _make_analysis_out(tmp_path, baseline_trace=stored_trace)
    out_png = tmp_path / "stored.png"

    def _unexpected_recompute(**_kwargs):
        raise AssertionError("plotter should use stored baseline_ref_candidate")

    monkeypatch.setattr(plotter, "compute_baseline_reference_candidate", _unexpected_recompute)

    result = plotter.plot_one(
        analysis_out=str(analysis_out),
        roi="CH3",
        chunk_id=31,
        output_path=str(out_png),
    )

    assert result == str(out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_missing_baseline_candidate_row_has_clear_error(tmp_path):
    analysis_out = _make_analysis_out(tmp_path)
    table = plotter._load_candidate_table(str(analysis_out))

    with pytest.raises(ValueError, match="No baseline candidate row found"):
        plotter._find_candidate_row(table, "CH3", 99)


def test_missing_dynamic_fitted_reference_has_clear_error(tmp_path):
    analysis_out = _make_analysis_out(tmp_path, include_fit_ref=False)

    with pytest.raises(RuntimeError, match="Unable to load ROI 'CH3' chunk 31"):
        plotter.plot_one(
            analysis_out=str(analysis_out),
            roi="CH3",
            chunk_id=31,
            output_path=str(tmp_path / "comparison.png"),
        )


def test_recomputed_candidate_uses_recorded_smoothing_window_metadata(tmp_path, monkeypatch):
    analysis_out = _make_analysis_out(tmp_path)
    table = plotter._load_candidate_table(str(analysis_out))
    row = plotter._find_candidate_row(table, "CH3", 31)
    traces = plotter._load_chunk_traces(str(analysis_out), "CH3", 31)
    captured = {}

    def _fake_candidate(**kwargs):
        captured.update(kwargs)
        return {"baseline_ref_candidate": np.ones_like(kwargs["signal"], dtype=float)}

    monkeypatch.setattr(plotter, "compute_baseline_reference_candidate", _fake_candidate)

    trace, source = plotter._recompute_baseline_candidate_trace(traces, row)

    assert trace.shape == traces["sig_raw"].shape
    assert source == "recomputed_from_metadata"
    assert captured["smoothing_window_sec"] == 30.0
    assert captured["default_smoothing_window_sec"] == 30.0
    assert captured["min_smoothing_window_sec"] == 10.0
    assert captured["max_window_fraction_of_chunk"] == 0.75
    assert captured["large_window_fraction_warning"] == 0.50


def test_stored_candidate_shape_mismatch_has_clear_error(tmp_path):
    analysis_out = _make_analysis_out(tmp_path, baseline_trace=np.arange(10, dtype=float))

    with pytest.raises(RuntimeError, match="Stored baseline_ref_candidate length does not match"):
        plotter._load_chunk_traces(str(analysis_out), "CH3", 31)


def test_default_output_path_uses_qc_reference_candidate_plots():
    path = plotter._default_output_path("analysis", "CH3", 31)

    assert path.endswith(
        os.path.join(
            "analysis",
            "qc",
            "reference_candidate_plots",
            "CH3_chunk_31_reference_candidate_comparison.png",
        )
    )
