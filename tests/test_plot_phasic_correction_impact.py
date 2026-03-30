import os
from contextlib import contextmanager

import numpy as np
import pytest

import tools.plot_phasic_correction_impact as impact
from photometry_pipeline.viz.display_prep import prepare_centered_common_gain


def test_build_correction_impact_figure_has_four_panels_and_expected_semantics():
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    sig = np.array([57.0, 60.0, 63.0, 60.0], dtype=float)
    iso = np.array([84.5, 85.0, 85.5, 85.0], dtype=float)
    fit = np.array([58.0, 60.5, 62.0, 60.2], dtype=float)
    dff = np.array([-0.02, 0.01, 0.03, 0.00], dtype=float)

    fig, axes = impact.build_correction_impact_figure(
        t=t,
        sig=sig,
        iso=iso,
        fit=fit,
        dff=dff,
        roi="Region0",
        chunk_id=7,
        dynamic_fit_mode="rolling_filtered_to_raw",
        baseline_subtract_before_fit=False,
    )
    try:
        assert len(axes) == 4
        ax1, ax2, ax3, ax4 = axes

        # Panel 1: raw absolute inputs unchanged
        assert np.allclose(ax1.lines[0].get_ydata(), sig)
        assert np.allclose(ax1.lines[1].get_ydata(), iso)
        assert "Raw Inputs (Absolute)" in ax1.get_title()

        # Panel 2: centered common-gain helper semantics
        sig_c_expected, iso_c_expected = prepare_centered_common_gain(sig, iso)
        sig_c = ax2.lines[0].get_ydata()
        iso_c = ax2.lines[1].get_ydata()
        assert np.allclose(sig_c, sig_c_expected)
        assert np.allclose(iso_c, iso_c_expected)
        assert np.isclose(np.nanmedian(sig_c), 0.0)
        assert np.isclose(np.nanmedian(iso_c), 0.0)
        assert np.max(np.abs(sig_c)) > np.max(np.abs(iso_c))
        assert "Common Gain" in ax2.get_title()

        # Panel 3/4: lower-panel semantics unchanged apart from vertical placement
        assert np.allclose(ax3.lines[0].get_ydata(), sig)
        assert np.allclose(ax3.lines[1].get_ydata(), fit)
        assert (
            ax3.get_title()
            == "Dynamic Reference Fitting (Rolling regression (filtered→raw); baseline subtract before fit: off)"
        )
        assert np.allclose(ax4.lines[0].get_ydata(), dff)
        assert ax4.get_title() == "Final Corrected Signal"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_build_correction_impact_figure_global_mode_title():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    sig = np.array([1.0, 2.0, 3.0], dtype=float)
    iso = np.array([2.0, 2.5, 3.0], dtype=float)
    fit = np.array([1.1, 1.9, 3.1], dtype=float)
    dff = np.array([0.0, 0.1, -0.1], dtype=float)

    fig, axes = impact.build_correction_impact_figure(
        t=t,
        sig=sig,
        iso=iso,
        fit=fit,
        dff=dff,
        roi="Region0",
        chunk_id=1,
        dynamic_fit_mode="global_linear_regression",
        baseline_subtract_before_fit=True,
    )
    try:
        assert (
            axes[2].get_title()
            == "Dynamic Reference Fitting (Global linear regression; baseline subtract before fit: inactive)"
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_build_correction_impact_figure_rolling_filtered_to_filtered_with_baseline_on_title():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    sig = np.array([1.0, 2.0, 3.0], dtype=float)
    iso = np.array([2.0, 2.5, 3.0], dtype=float)
    fit = np.array([1.1, 1.9, 3.1], dtype=float)
    dff = np.array([0.0, 0.1, -0.1], dtype=float)

    fig, axes = impact.build_correction_impact_figure(
        t=t,
        sig=sig,
        iso=iso,
        fit=fit,
        dff=dff,
        roi="Region0",
        chunk_id=1,
        dynamic_fit_mode="rolling_filtered_to_filtered",
        baseline_subtract_before_fit=True,
    )
    try:
        assert (
            axes[2].get_title()
            == "Dynamic Reference Fitting (Rolling regression (filtered→filtered); baseline subtract before fit: on)"
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_build_correction_impact_figure_robust_mode_title():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    sig = np.array([1.0, 2.0, 3.0], dtype=float)
    iso = np.array([2.0, 2.5, 3.0], dtype=float)
    fit = np.array([1.1, 1.9, 3.1], dtype=float)
    dff = np.array([0.0, 0.1, -0.1], dtype=float)

    fig, axes = impact.build_correction_impact_figure(
        t=t,
        sig=sig,
        iso=iso,
        fit=fit,
        dff=dff,
        roi="Region0",
        chunk_id=1,
        dynamic_fit_mode="robust_global_event_reject",
        baseline_subtract_before_fit=True,
    )
    try:
        assert (
            axes[2].get_title()
            == "Dynamic Reference Fitting (Robust global fit + event rejection; baseline subtract before fit: inactive)"
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_build_correction_impact_figure_adaptive_mode_title():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    sig = np.array([1.0, 2.0, 3.0], dtype=float)
    iso = np.array([2.0, 2.5, 3.0], dtype=float)
    fit = np.array([1.1, 1.9, 3.1], dtype=float)
    dff = np.array([0.0, 0.1, -0.1], dtype=float)

    fig, axes = impact.build_correction_impact_figure(
        t=t,
        sig=sig,
        iso=iso,
        fit=fit,
        dff=dff,
        roi="Region0",
        chunk_id=1,
        dynamic_fit_mode="adaptive_event_gated_regression",
        baseline_subtract_before_fit=True,
    )
    try:
        assert (
            axes[2].get_title()
            == "Dynamic Reference Fitting (Adaptive event-gated regression; baseline subtract before fit: inactive)"
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_resolve_dynamic_fit_settings_accepts_adaptive_mode(tmp_path):
    analysis_out = tmp_path / "analysis"
    analysis_out.mkdir(parents=True, exist_ok=True)
    (analysis_out / "config_used.yaml").write_text(
        "dynamic_fit_mode: adaptive_event_gated_regression\nbaseline_subtract_before_fit: true\n",
        encoding="utf-8",
    )

    mode, baseline = impact._resolve_dynamic_fit_settings(str(analysis_out))
    assert mode == "adaptive_event_gated_regression"
    assert baseline is True


def test_main_generates_png_with_four_panel_layout(tmp_path, monkeypatch):
    analysis_out = tmp_path / "analysis"
    analysis_out.mkdir(parents=True, exist_ok=True)
    (analysis_out / "config_used.yaml").write_text(
        "dynamic_fit_mode: global_linear_regression\nbaseline_subtract_before_fit: true\n",
        encoding="utf-8",
    )
    # main() validates this file path before reader import/use.
    (analysis_out / "phasic_trace_cache.h5").write_bytes(b"placeholder")
    out_png = tmp_path / "phasic_correction_impact.png"

    t = np.linspace(0.0, 5.0, 101, dtype=float)
    sig = 60.0 + 2.0 * np.sin(t)
    iso = 85.0 + 0.4 * np.sin(t + 0.2)
    fit = 60.0 + 1.7 * np.sin(t + 0.05)
    dff = 0.02 * np.sin(t)

    @contextmanager
    def _fake_open(_cache_path):
        yield object()

    def _fake_load(_cache, _roi, _chunk_id, _fields):
        return t, sig, iso, fit, dff

    import photometry_pipeline.io.hdf5_cache_reader as reader

    monkeypatch.setattr(reader, "open_phasic_cache", _fake_open)
    monkeypatch.setattr(reader, "load_cache_chunk_fields", _fake_load)
    captured = {}
    orig_build = impact.build_correction_impact_figure

    def _spy_build(*args, **kwargs):
        captured["dynamic_fit_mode"] = kwargs.get("dynamic_fit_mode")
        captured["baseline_subtract_before_fit"] = kwargs.get("baseline_subtract_before_fit")
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(impact, "build_correction_impact_figure", _spy_build)
    monkeypatch.setattr(
        impact.sys,
        "argv",
        [
            "plot_phasic_correction_impact.py",
            "--analysis-out",
            str(analysis_out),
            "--roi",
            "Region0",
            "--chunk-id",
            "0",
            "--out",
            str(out_png),
            "--dpi",
            "120",
        ],
    )

    impact.main()

    assert out_png.exists()
    assert os.path.getsize(out_png) > 0
    assert captured.get("dynamic_fit_mode") == "global_linear_regression"
    assert captured.get("baseline_subtract_before_fit") is True


def test_main_defaults_to_rolling_mode_when_config_missing(tmp_path, monkeypatch):
    analysis_out = tmp_path / "analysis_missing_cfg"
    analysis_out.mkdir(parents=True, exist_ok=True)
    (analysis_out / "phasic_trace_cache.h5").write_bytes(b"placeholder")
    out_png = tmp_path / "phasic_correction_impact_fallback.png"

    t = np.linspace(0.0, 1.0, 5, dtype=float)
    sig = np.linspace(1.0, 2.0, 5, dtype=float)
    iso = np.linspace(2.0, 3.0, 5, dtype=float)
    fit = np.linspace(1.2, 2.2, 5, dtype=float)
    dff = np.zeros(5, dtype=float)

    @contextmanager
    def _fake_open(_cache_path):
        yield object()

    def _fake_load(_cache, _roi, _chunk_id, _fields):
        return t, sig, iso, fit, dff

    import photometry_pipeline.io.hdf5_cache_reader as reader

    monkeypatch.setattr(reader, "open_phasic_cache", _fake_open)
    monkeypatch.setattr(reader, "load_cache_chunk_fields", _fake_load)

    captured = {}
    orig_build = impact.build_correction_impact_figure

    def _spy_build(*args, **kwargs):
        captured["dynamic_fit_mode"] = kwargs.get("dynamic_fit_mode")
        captured["baseline_subtract_before_fit"] = kwargs.get("baseline_subtract_before_fit")
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(impact, "build_correction_impact_figure", _spy_build)
    monkeypatch.setattr(
        impact.sys,
        "argv",
        [
            "plot_phasic_correction_impact.py",
            "--analysis-out",
            str(analysis_out),
            "--roi",
            "Region0",
            "--chunk-id",
            "0",
            "--out",
            str(out_png),
            "--dpi",
            "120",
        ],
    )

    impact.main()
    assert out_png.exists()
    assert captured.get("dynamic_fit_mode") == "rolling_filtered_to_raw"
    assert captured.get("baseline_subtract_before_fit") is False


def test_main_keeps_cli_failure_boundary_when_reader_raises_runtime_error(tmp_path, monkeypatch):
    analysis_out = tmp_path / "analysis_cli_error"
    analysis_out.mkdir(parents=True, exist_ok=True)
    (analysis_out / "phasic_trace_cache.h5").write_bytes(b"placeholder")
    out_png = tmp_path / "phasic_correction_impact_error.png"

    def _raise_read_error(_cache_path):
        raise RuntimeError("broken cache")

    import photometry_pipeline.io.hdf5_cache_reader as reader

    monkeypatch.setattr(reader, "open_phasic_cache", _raise_read_error)
    monkeypatch.setattr(
        impact.sys,
        "argv",
        [
            "plot_phasic_correction_impact.py",
            "--analysis-out",
            str(analysis_out),
            "--roi",
            "Region0",
            "--chunk-id",
            "0",
            "--out",
            str(out_png),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        impact.main()
    assert excinfo.value.code == 1
