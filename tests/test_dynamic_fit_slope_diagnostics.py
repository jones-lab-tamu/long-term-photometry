import json

import h5py
import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core.regression import fit_chunk_dynamic, fit_robust_global_event_reject
from photometry_pipeline.core.reporting import generate_run_report
from photometry_pipeline.core.slope_qc import apply_slope_constraint, summarize_slope
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import load_cache_chunk_attrs
from photometry_pipeline.pipeline import Pipeline, _append_run_report_section
from photometry_pipeline.tuning.cache_correction_retune import _write_correction_inspection


def _make_chunk(uv: np.ndarray, sig: np.ndarray, fs_hz: float = 40.0) -> Chunk:
    t = np.arange(len(uv), dtype=float) / fs_hz
    return Chunk(
        chunk_id=0,
        source_file="synthetic.csv",
        format="npm",
        time_sec=t,
        uv_raw=np.asarray(uv, dtype=float).reshape(-1, 1),
        sig_raw=np.asarray(sig, dtype=float).reshape(-1, 1),
        uv_filt=np.asarray(uv, dtype=float).reshape(-1, 1),
        sig_filt=np.asarray(sig, dtype=float).reshape(-1, 1),
        fs_hz=fs_hz,
        channel_names=["Region0"],
    )


def test_config_accepts_and_rejects_dynamic_fit_slope_constraint(tmp_path):
    cfg_path = tmp_path / "slope_constraint.yaml"
    cfg_path.write_text(
        "dynamic_fit_slope_constraint: nonnegative\n"
        "dynamic_fit_min_slope: 0.1\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.dynamic_fit_slope_constraint == "nonnegative"
    assert cfg.dynamic_fit_min_slope == pytest.approx(0.1)

    bad_mode = tmp_path / "bad_mode.yaml"
    bad_mode.write_text("dynamic_fit_slope_constraint: clampish\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid dynamic_fit_slope_constraint"):
        Config.from_yaml(str(bad_mode))

    bad_min = tmp_path / "bad_min.yaml"
    bad_min.write_text(
        "dynamic_fit_slope_constraint: nonnegative\n"
        "dynamic_fit_min_slope: .inf\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dynamic_fit_min_slope"):
        Config.from_yaml(str(bad_min))

    negative_floor = tmp_path / "negative_floor.yaml"
    negative_floor.write_text(
        "dynamic_fit_slope_constraint: nonnegative\n"
        "dynamic_fit_min_slope: -0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dynamic_fit_min_slope must be >= 0"):
        Config.from_yaml(str(negative_floor))

    unconstrained_negative_floor = tmp_path / "unconstrained_negative_floor.yaml"
    unconstrained_negative_floor.write_text(
        "dynamic_fit_slope_constraint: unconstrained\n"
        "dynamic_fit_min_slope: -0.1\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(str(unconstrained_negative_floor))
    assert cfg.dynamic_fit_slope_constraint == "unconstrained"
    assert cfg.dynamic_fit_min_slope == pytest.approx(-0.1)


def test_summarize_slope_warning_levels_and_spans():
    positive = summarize_slope(np.array([0.5, 1.0, 2.0]), sample_rate_hz=10.0)
    assert positive["warning_level"] == "none"
    assert positive["slope_negative_fraction"] == 0.0

    mixed = summarize_slope(np.array([1.0, -0.5, -0.2, 0.7, -0.1]), sample_rate_hz=10.0)
    assert mixed["n_negative_slope_samples"] == 3
    assert mixed["n_negative_slope_spans"] == 2
    assert mixed["longest_negative_slope_span_samples"] == 2
    assert mixed["longest_negative_slope_span_sec"] == pytest.approx(0.2)
    assert mixed["warning_level"] == "critical"

    scalar = summarize_slope(-1.25)
    assert scalar["n_slope_samples"] == 1
    assert scalar["slope_negative_fraction"] == 1.0
    assert scalar["warning_level"] == "critical"

    all_negative = summarize_slope(np.array([-2.0, -1.0, -0.25]), sample_rate_hz=20.0)
    assert all_negative["n_negative_slope_samples"] == 3
    assert all_negative["n_negative_slope_spans"] == 1
    assert all_negative["longest_negative_slope_span_samples"] == 3
    assert all_negative["longest_negative_slope_span_sec"] == pytest.approx(0.15)
    assert all_negative["warning_level"] == "critical"

    nonfinite = summarize_slope(np.array([np.nan, np.inf, -1.0, 2.0]))
    assert nonfinite["n_nonfinite_slope_samples"] == 2
    assert nonfinite["slope_nonfinite_fraction"] == 0.5
    assert nonfinite["slope_negative_fraction"] == 0.5


def test_apply_slope_constraint_helper_modes_and_spans():
    slopes = np.array([1.0, -0.5, -0.2, np.nan, np.inf, -0.1, 0.4], dtype=float)

    unchanged, unconstrained = apply_slope_constraint(
        slopes,
        constraint_mode="unconstrained",
        min_slope=-0.1,
        sample_rate_hz=10.0,
    )
    np.testing.assert_allclose(unchanged, slopes, equal_nan=True)
    assert unconstrained["slope_constraint_applied"] is False
    assert unconstrained["n_clamped_slope_samples"] == 0

    constrained, summary = apply_slope_constraint(
        slopes,
        constraint_mode="nonnegative",
        min_slope=0.0,
        sample_rate_hz=10.0,
    )
    assert np.all(constrained[np.isfinite(constrained)] >= 0.0)
    assert summary["slope_constraint_applied"] is True
    assert summary["n_clamped_slope_samples"] == 3
    assert summary["n_clamped_slope_spans"] == 2
    assert summary["longest_clamped_slope_span_samples"] == 2
    assert summary["longest_clamped_slope_span_sec"] == pytest.approx(0.2)
    assert summary["unconstrained_slope_summary"]["slope_negative_fraction"] == pytest.approx(0.6)
    assert summary["constrained_slope_summary"]["slope_negative_fraction"] == 0.0

    scalar, scalar_summary = apply_slope_constraint(
        -2.0,
        constraint_mode="nonnegative",
        min_slope=0.25,
    )
    assert scalar == pytest.approx(0.25)
    assert scalar_summary["slope_clamped_fraction"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="dynamic_fit_slope_constraint"):
        apply_slope_constraint(slopes, constraint_mode="invalid", min_slope=0.0)
    with pytest.raises(ValueError, match="dynamic_fit_min_slope"):
        apply_slope_constraint(slopes, constraint_mode="nonnegative", min_slope=float("nan"))
    with pytest.raises(ValueError, match="dynamic_fit_min_slope must be >= 0"):
        apply_slope_constraint(slopes, constraint_mode="nonnegative", min_slope=-0.1)


def test_robust_global_negative_slope_metadata_does_not_change_fit_formula():
    n = 500
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.7 * iso

    result = fit_robust_global_event_reject(
        signal_raw=sig,
        iso_raw=iso,
        max_iters=2,
        residual_z_thresh=50.0,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        min_keep_fraction=0.5,
        sample_rate_hz=fs,
        use_intercept=True,
    )

    summary = result["slope_summary"]
    assert summary["slope_negative_fraction"] == 1.0
    assert summary["warning_level"] == "critical"
    coef = result["final_coef"]
    expected = float(coef["slope"]) * iso + float(coef["intercept"])
    np.testing.assert_allclose(result["iso_fit_signal_units"], expected, rtol=0.0, atol=1e-12)
    assert result["slope_constraint_summary"]["slope_constraint_applied"] is False


def test_robust_global_nonnegative_constraint_clamps_final_fit_only():
    n = 500
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.7 * iso

    result = fit_robust_global_event_reject(
        signal_raw=sig,
        iso_raw=iso,
        max_iters=2,
        residual_z_thresh=50.0,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        min_keep_fraction=0.5,
        sample_rate_hz=fs,
        use_intercept=True,
        slope_constraint="nonnegative",
        min_slope=0.0,
    )

    coef = result["final_coef"]
    assert coef["unconstrained_slope"] < 0.0
    assert coef["slope"] >= 0.0
    expected = float(coef["slope"]) * iso + float(coef["intercept"])
    np.testing.assert_allclose(result["iso_fit_signal_units"], expected, rtol=0.0, atol=1e-12)
    assert result["slope_summary"]["slope_negative_fraction"] == 0.0
    assert result["unconstrained_slope_summary"]["slope_negative_fraction"] == 1.0
    assert result["slope_constraint_summary"]["slope_constraint_applied"] is True


def test_adaptive_negative_local_slope_summary_is_metadata_only():
    n = 2400
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(123)
    uv = 2.0 + 0.45 * np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.sin(2 * np.pi * 0.73 * t)
    sig = 1.4 * uv + 0.6 + 0.01 * rng.normal(size=n)
    neg = (t >= 22.0) & (t <= 36.0)
    sig[neg] = -1.1 * uv[neg] + 5.1 + 0.01 * rng.normal(size=int(np.sum(neg)))

    cfg = Config(
        dynamic_fit_mode="adaptive_event_gated_regression",
        adaptive_event_gate_smooth_window_sec=1.0,
        adaptive_event_gate_local_var_window_sec=1.5,
        adaptive_event_gate_local_var_ratio_thresh=None,
        adaptive_event_gate_residual_z_thresh=50.0,
        adaptive_event_gate_min_trust_fraction=0.1,
        signal_excursion_polarity="both",
    )
    chunk = _make_chunk(uv, sig, fs)

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")

    np.testing.assert_allclose(delta_f[:, 0], sig - uv_fit[:, 0], rtol=0.0, atol=1e-12)
    roi_meta = chunk.metadata["dynamic_fit_adaptive_event_gated"]["Region0"]
    summary = roi_meta["slope_summary"]
    assert summary["slope_negative_fraction"] > 0.0
    assert summary["longest_negative_slope_span_samples"] > 0
    assert summary["warning_level"] != "none"
    assert roi_meta["coef_slope"].shape == sig.shape
    np.testing.assert_allclose(roi_meta["coef_slope_unconstrained"], roi_meta["coef_slope"])
    assert roi_meta["slope_constraint_summary"]["slope_constraint_applied"] is False
    assert any("NEGATIVE_UV_TO_SIGNAL_SLOPE" in str(w) for w in chunk.metadata["qc_warnings"])


def test_adaptive_nonnegative_constraint_clamps_local_slope_trace():
    n = 2400
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(123)
    uv = 2.0 + 0.45 * np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.sin(2 * np.pi * 0.73 * t)
    sig = 1.4 * uv + 0.6 + 0.01 * rng.normal(size=n)
    neg = (t >= 22.0) & (t <= 36.0)
    sig[neg] = -1.1 * uv[neg] + 5.1 + 0.01 * rng.normal(size=int(np.sum(neg)))

    cfg = Config(
        dynamic_fit_mode="adaptive_event_gated_regression",
        adaptive_event_gate_smooth_window_sec=1.0,
        adaptive_event_gate_local_var_window_sec=1.5,
        adaptive_event_gate_local_var_ratio_thresh=None,
        adaptive_event_gate_residual_z_thresh=50.0,
        adaptive_event_gate_min_trust_fraction=0.1,
        signal_excursion_polarity="both",
        dynamic_fit_slope_constraint="nonnegative",
        dynamic_fit_min_slope=0.0,
    )
    chunk = _make_chunk(uv, sig, fs)

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")

    np.testing.assert_allclose(delta_f[:, 0], sig - uv_fit[:, 0], rtol=0.0, atol=1e-12)
    roi_meta = chunk.metadata["dynamic_fit_adaptive_event_gated"]["Region0"]
    finite_slope = roi_meta["coef_slope"][np.isfinite(roi_meta["coef_slope"])]
    assert finite_slope.size > 0
    assert np.min(finite_slope) >= 0.0
    assert roi_meta["unconstrained_slope_summary"]["slope_negative_fraction"] > 0.0
    assert roi_meta["slope_summary"]["slope_negative_fraction"] == 0.0
    assert roi_meta["slope_constraint_summary"]["slope_constraint_applied"] is True
    assert any("DYNAMIC_FIT_SLOPE_CONSTRAINT_APPLIED" in str(w) for w in chunk.metadata["qc_warnings"])


def test_rolling_local_negative_slope_summary_is_metadata_only():
    n = 900
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 3.0 + 0.6 * np.sin(2.0 * np.pi * 0.3 * t)
    sig = 7.0 - 1.4 * uv
    cfg = Config(
        dynamic_fit_mode="rolling_filtered_to_raw",
        window_sec=4.0,
        step_sec=1.0,
        min_samples_per_window=20,
    )
    chunk = _make_chunk(uv, sig, fs)

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")

    np.testing.assert_allclose(delta_f[:, 0], sig - uv_fit[:, 0], rtol=0.0, atol=1e-12)
    roi_meta = chunk.metadata["dynamic_fit_rolling_local"]["Region0"]
    summary = roi_meta["slope_summary"]
    assert summary["slope_negative_fraction"] > 0.9
    assert summary["warning_level"] == "critical"
    assert roi_meta["coef_slope"].shape == sig.shape
    assert roi_meta["slope_constraint_summary"]["slope_constraint_applied"] is False
    assert any("NEGATIVE_UV_TO_SIGNAL_SLOPE" in str(w) for w in chunk.metadata["qc_warnings"])


def test_rolling_local_nonnegative_constraint_clamps_slope_trace():
    n = 900
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 3.0 + 0.6 * np.sin(2.0 * np.pi * 0.3 * t)
    sig = 7.0 - 1.4 * uv
    cfg = Config(
        dynamic_fit_mode="rolling_filtered_to_raw",
        window_sec=4.0,
        step_sec=1.0,
        min_samples_per_window=20,
        dynamic_fit_slope_constraint="nonnegative",
        dynamic_fit_min_slope=0.0,
    )
    chunk = _make_chunk(uv, sig, fs)

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")

    np.testing.assert_allclose(delta_f[:, 0], sig - uv_fit[:, 0], rtol=0.0, atol=1e-12)
    roi_meta = chunk.metadata["dynamic_fit_rolling_local"]["Region0"]
    finite_slope = roi_meta["coef_slope"][np.isfinite(roi_meta["coef_slope"])]
    assert finite_slope.size > 0
    assert np.min(finite_slope) >= 0.0
    assert roi_meta["unconstrained_slope_summary"]["slope_negative_fraction"] > 0.9
    assert roi_meta["slope_summary"]["slope_negative_fraction"] == 0.0
    assert roi_meta["slope_constraint_summary"]["slope_constraint_applied"] is True
    assert any("DYNAMIC_FIT_SLOPE_CONSTRAINT_APPLIED" in str(w) for w in chunk.metadata["qc_warnings"])


def test_slope_summary_persists_to_hdf5_attrs(tmp_path):
    n = 500
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.7 * uv
    cfg = Config(dynamic_fit_mode="global_linear_regression")
    chunk = _make_chunk(uv, sig, fs)
    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    chunk.dff = delta_f.copy()

    cache_path = tmp_path / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        writer.add_chunk(chunk, 0, "synthetic.csv")

    with h5py.File(cache_path, "r") as cache:
        attrs = load_cache_chunk_attrs(cache, "Region0", 0)
        assert "fit_ref" in cache["roi/Region0/chunk_0"]

    assert attrs["dynamic_fit_mode_resolved"] == "global_linear_regression"
    assert attrs["dynamic_fit_slope_warning_level"] == "critical"
    assert attrs["dynamic_fit_slope_slope_negative_fraction"] == pytest.approx(1.0)
    assert attrs["dynamic_fit_slope_n_negative_slope_samples"] == 1
    assert attrs["dynamic_fit_slope_constraint_mode"] == "unconstrained"
    assert attrs["dynamic_fit_slope_constraint_applied"] is False
    assert attrs["dynamic_fit_slope_clamped_fraction"] == pytest.approx(0.0)


def test_slope_constraint_attrs_persist_to_hdf5_attrs(tmp_path):
    n = 500
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.7 * uv
    cfg = Config(
        dynamic_fit_mode="global_linear_regression",
        dynamic_fit_slope_constraint="nonnegative",
        dynamic_fit_min_slope=0.0,
    )
    chunk = _make_chunk(uv, sig, fs)
    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    chunk.dff = delta_f.copy()

    cache_path = tmp_path / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        writer.add_chunk(chunk, 0, "synthetic.csv")

    with h5py.File(cache_path, "r") as cache:
        grp = cache["roi/Region0/chunk_0"]
        attrs = load_cache_chunk_attrs(cache, "Region0", 0)
        for key, value in grp.attrs.items():
            assert not isinstance(value, dict), key

    assert attrs["dynamic_fit_slope_warning_level"] == "none"
    assert attrs["dynamic_fit_slope_constraint_mode"] == "nonnegative"
    assert attrs["dynamic_fit_slope_constraint_applied"] is True
    assert attrs["dynamic_fit_slope_clamped_fraction"] == pytest.approx(1.0)
    assert attrs["dynamic_fit_slope_unconstrained_slope_negative_fraction"] == pytest.approx(1.0)
    assert attrs["dynamic_fit_slope_constrained_slope_negative_fraction"] == pytest.approx(0.0)


def test_older_cache_without_slope_attrs_still_loads(tmp_path):
    cache_path = tmp_path / "legacy_phasic_trace_cache.h5"
    with h5py.File(cache_path, "w") as cache:
        grp = cache.create_group("roi/Region0/chunk_0")
        grp.attrs["chunk_id"] = 0
        grp.attrs["source_file"] = "legacy.csv"

    with h5py.File(cache_path, "r") as cache:
        attrs = load_cache_chunk_attrs(cache, "Region0", 0)

    assert attrs["chunk_id"] == 0
    assert "dynamic_fit_slope_warning_level" not in attrs
    assert "dynamic_fit_slope_slope_negative_fraction" not in attrs
    assert "dynamic_fit_slope_constraint_mode" not in attrs


def test_pipeline_qc_and_run_report_aggregate_negative_slope_warnings(tmp_path):
    n = 500
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.7 * uv
    cfg = Config(dynamic_fit_mode="global_linear_regression")
    chunk = _make_chunk(uv, sig, fs)
    fit_chunk_dynamic(chunk, cfg, mode="phasic")

    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_dynamic_fit_slope_summaries(chunk, 0, "synthetic.csv")

    assert pipeline.qc_summary["dynamic_fit_slope_warning_summary"][
        "roi_chunk_fits_with_any_negative_slope"
    ] == 1
    assert pipeline.qc_summary["dynamic_fit_slope_warning_summary"][
        "roi_chunk_fits_by_warning_level"
    ]["critical"] == 1
    assert pipeline.qc_summary["dynamic_fit_slope_warning_summary"][
        "roi_chunk_fits_with_critical_warnings"
    ] == 1
    assert pipeline.qc_summary["dynamic_fit_slope_warning_summary"][
        "dynamic_fit_modes_affected"
    ] == ["global_linear_regression"]
    assert pipeline.qc_summary["dynamic_fit_slope_warning_summary"]["rois_affected"] == [
        "Region0"
    ]
    assert pipeline.qc_summary["dynamic_fit_slope_warnings"][0]["warning_level"] == "critical"

    generate_run_report(cfg, str(tmp_path))
    _append_run_report_section(
        str(tmp_path),
        "dynamic_fit_slope_warning_summary",
        pipeline.dynamic_fit_slope_warning_summary,
    )
    report = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))
    assert report["derived_settings"]["dynamic_fit_slope_warning_summary"][
        "roi_chunk_fits_with_moderate_high_critical_warnings"
    ] == 1


def test_pipeline_qc_and_run_report_aggregate_slope_constraint_warnings(tmp_path):
    n = 500
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.7 * uv
    cfg = Config(
        dynamic_fit_mode="global_linear_regression",
        dynamic_fit_slope_constraint="nonnegative",
        dynamic_fit_min_slope=0.0,
    )
    chunk = _make_chunk(uv, sig, fs)
    fit_chunk_dynamic(chunk, cfg, mode="phasic")

    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_dynamic_fit_slope_summaries(chunk, 0, "synthetic.csv")

    summary = pipeline.qc_summary["dynamic_fit_slope_constraint_summary"]
    assert summary["roi_chunk_fits_with_slope_constraint_applied"] == 1
    assert summary["roi_chunk_fits_by_constraint_mode"] == {"nonnegative": 1}
    assert summary["roi_chunk_fits_with_any_clamped_slope"] == 1
    assert summary["max_unconstrained_slope_negative_fraction_among_clamped"] == pytest.approx(1.0)
    assert summary["dynamic_fit_modes_with_slope_constraint_applied"] == [
        "global_linear_regression"
    ]
    assert summary["rois_with_slope_constraint_applied"] == ["Region0"]
    assert pipeline.qc_summary["dynamic_fit_slope_constraint_warnings"][0][
        "slope_constraint_applied"
    ] is True

    generate_run_report(cfg, str(tmp_path))
    _append_run_report_section(
        str(tmp_path),
        "dynamic_fit_slope_constraint_summary",
        pipeline.dynamic_fit_slope_constraint_summary,
    )
    report = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))
    assert report["derived_settings"]["dynamic_fit_slope_constraint_summary"][
        "roi_chunk_fits_with_slope_constraint_applied"
    ] == 1


def test_retune_inspection_reads_compact_slope_attrs_and_keeps_dff_key(tmp_path):
    n = 200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.3 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.5 * uv
    fit = -1.5 * uv + 8.0
    chunk = _make_chunk(uv, sig, fs)
    chunk.uv_fit = fit.reshape(-1, 1)
    chunk.delta_f = (sig - fit).reshape(-1, 1)
    chunk.dff = chunk.delta_f.copy()
    chunk.metadata = {
        "dynamic_fit_mode_resolved": "global_linear_regression",
        "dynamic_fit_slope_summary_available": True,
        "dynamic_fit_slope_warning_level": "critical",
        "dynamic_fit_slope_slope_min": -1.5,
        "dynamic_fit_slope_slope_max": -1.5,
        "dynamic_fit_slope_slope_negative_fraction": 1.0,
        "dynamic_fit_slope_longest_negative_slope_span_sec": np.nan,
        "dynamic_fit_slope_constraint_mode": "unavailable",
    }

    artifacts = _write_correction_inspection(str(tmp_path), "Region0", chunk)

    diag = artifacts["retuned_correction_inspection_slope_diagnostics"]
    assert diag["warning_level"] == "critical"
    assert diag["slope_negative_fraction"] == pytest.approx(1.0)
    assert diag["slope_trace_available"] is False
    assert "UV-to-signal slope diagnostics" not in artifacts[
        "retuned_correction_inspection_panel_labels"
    ]
    assert artifacts["retuned_correction_inspection_dff_png"].endswith("_dff.png")
    assert artifacts["retuned_correction_inspection_dff_png"] in artifacts[
        "retuned_correction_inspection_pngs"
    ]
    df = pd.read_csv(artifacts["retuned_correction_session_csv"])
    assert set(
        [
            "dynamic_fit_slope_warning_level",
            "dynamic_fit_slope_negative_fraction",
            "dynamic_fit_slope_min",
            "dynamic_fit_slope_max",
            "dynamic_fit_slope_constraint_mode",
            "dynamic_fit_slope_constraint_applied",
            "dynamic_fit_slope_clamped_fraction",
            "dynamic_fit_slope_unconstrained_negative_fraction",
            "dynamic_fit_slope_constrained_negative_fraction",
        ]
    ).issubset(df.columns)
    assert set(df["dynamic_fit_slope_warning_level"].astype(str)) == {"critical"}
    assert set(df["dynamic_fit_slope_constraint_mode"].astype(str)) == {"unavailable"}


def test_retune_inspection_reads_compact_constraint_attrs(tmp_path):
    n = 200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.3 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 8.0 - 1.5 * uv
    fit = np.full_like(sig, 8.0)
    chunk = _make_chunk(uv, sig, fs)
    chunk.uv_fit = fit.reshape(-1, 1)
    chunk.delta_f = (sig - fit).reshape(-1, 1)
    chunk.dff = chunk.delta_f.copy()
    chunk.metadata = {
        "dynamic_fit_mode_resolved": "global_linear_regression",
        "dynamic_fit_slope_summary_available": True,
        "dynamic_fit_slope_warning_level": "none",
        "dynamic_fit_slope_slope_min": 0.0,
        "dynamic_fit_slope_slope_max": 0.0,
        "dynamic_fit_slope_slope_negative_fraction": 0.0,
        "dynamic_fit_slope_constraint_mode": "nonnegative",
        "dynamic_fit_slope_min_allowed": 0.0,
        "dynamic_fit_slope_constraint_applied": True,
        "dynamic_fit_slope_clamped_fraction": 1.0,
        "dynamic_fit_slope_unconstrained_slope_min": -1.5,
        "dynamic_fit_slope_unconstrained_slope_max": -1.5,
        "dynamic_fit_slope_unconstrained_slope_negative_fraction": 1.0,
        "dynamic_fit_slope_constrained_slope_min": 0.0,
        "dynamic_fit_slope_constrained_slope_max": 0.0,
        "dynamic_fit_slope_constrained_slope_negative_fraction": 0.0,
    }

    artifacts = _write_correction_inspection(str(tmp_path), "Region0", chunk)

    diag = artifacts["retuned_correction_inspection_slope_diagnostics"]
    assert diag["slope_constraint_mode"] == "nonnegative"
    assert diag["slope_constraint_applied"] is True
    assert diag["slope_clamped_fraction"] == pytest.approx(1.0)
    assert diag["unconstrained_slope_negative_fraction"] == pytest.approx(1.0)
    assert diag["constrained_slope_negative_fraction"] == pytest.approx(0.0)
    df = pd.read_csv(artifacts["retuned_correction_session_csv"])
    assert set(df["dynamic_fit_slope_constraint_mode"].astype(str)) == {"nonnegative"}
    assert set(df["dynamic_fit_slope_constraint_applied"].astype(bool)) == {True}


def test_retune_inspection_adds_slope_panel_when_trace_is_available(tmp_path):
    n = 240
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 3.0 + 0.2 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 6.0 - 1.2 * uv
    fit = -1.2 * uv + 6.0
    slope_trace = np.full(n, -1.2, dtype=float)
    chunk = _make_chunk(uv, sig, fs)
    chunk.uv_fit = fit.reshape(-1, 1)
    chunk.delta_f = (sig - fit).reshape(-1, 1)
    chunk.dff = chunk.delta_f.copy()
    chunk.metadata = {
        "dynamic_fit_mode_resolved": "rolling_filtered_to_raw",
        "dynamic_fit_rolling_local": {
            "Region0": {
                "coef_slope": slope_trace,
                "coef_slope_unconstrained": slope_trace.copy(),
                "slope_summary": summarize_slope(slope_trace, sample_rate_hz=fs),
                "slope_constraint_summary": apply_slope_constraint(
                    slope_trace,
                    constraint_mode="unconstrained",
                    min_slope=0.0,
                    sample_rate_hz=fs,
                )[1],
            }
        },
    }

    artifacts = _write_correction_inspection(str(tmp_path), "Region0", chunk)

    assert "UV-to-signal slope diagnostics" in artifacts[
        "retuned_correction_inspection_panel_labels"
    ]
    assert artifacts["retuned_correction_inspection_slope_diagnostics"][
        "slope_trace_available"
    ] is True
    assert artifacts["retuned_correction_inspection_dff_png"].endswith("_dff.png")
