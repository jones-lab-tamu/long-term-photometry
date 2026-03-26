import os
from contextlib import contextmanager

import numpy as np

import tools.plot_phasic_correction_impact as impact
from photometry_pipeline.viz.display_prep import prepare_centered_common_gain


def test_build_correction_impact_figure_has_four_panels_and_expected_semantics():
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    sig = np.array([57.0, 60.0, 63.0, 60.0], dtype=float)
    iso = np.array([84.5, 85.0, 85.5, 85.0], dtype=float)
    fit = np.array([58.0, 60.5, 62.0, 60.2], dtype=float)
    dff = np.array([-0.02, 0.01, 0.03, 0.00], dtype=float)

    fig, axes = impact.build_correction_impact_figure(
        t=t, sig=sig, iso=iso, fit=fit, dff=dff, roi="Region0", chunk_id=7
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
        assert ax3.get_title() == "Dynamic Reference Fitting (Rolling Local Regression)"
        assert np.allclose(ax4.lines[0].get_ydata(), dff)
        assert ax4.get_title() == "Final Corrected Signal"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_main_generates_png_with_four_panel_layout(tmp_path, monkeypatch):
    analysis_out = tmp_path / "analysis"
    analysis_out.mkdir(parents=True, exist_ok=True)
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
