import numpy as np
import pytest

from photometry_pipeline.viz.display_prep import prepare_centered_common_gain


def test_centered_common_gain_removes_offsets_without_amplitude_equalization():
    sig = np.array([57.0, 60.0, 63.0, 60.0, 57.0], dtype=float)
    iso = np.array([84.5, 85.0, 85.5, 85.0, 84.5], dtype=float)

    sig_c, iso_c = prepare_centered_common_gain(sig, iso)

    assert np.isclose(np.nanmedian(sig_c), 0.0)
    assert np.isclose(np.nanmedian(iso_c), 0.0)

    sig_dyn = float(np.nanmax(np.abs(sig_c)))
    iso_dyn = float(np.nanmax(np.abs(iso_c)))
    assert sig_dyn > iso_dyn
    assert (sig_dyn / iso_dyn) > 4.0


def test_centered_common_gain_median_centering_is_exact_on_deterministic_vectors():
    sig = np.array([1.0, 2.0, 3.0, 100.0], dtype=float)
    iso = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)

    sig_c, iso_c, sig_center, iso_center = prepare_centered_common_gain(
        sig, iso, return_centers=True
    )

    assert sig_center == 2.5
    assert iso_center == 11.5
    assert np.allclose(sig_c, np.array([-1.5, -0.5, 0.5, 97.5], dtype=float))
    assert np.allclose(iso_c, np.array([-1.5, -0.5, 0.5, 1.5], dtype=float))


def test_centered_common_gain_preserves_nan_positions_and_centers_on_finite_values():
    sig = np.array([1.0, 2.0, np.nan, 4.0], dtype=float)
    iso = np.array([10.0, np.nan, 12.0, 14.0], dtype=float)

    sig_c, iso_c = prepare_centered_common_gain(sig, iso)

    assert np.isnan(sig_c[2])
    assert np.isnan(iso_c[1])
    assert np.allclose(sig_c[[0, 1, 3]], np.array([-1.0, 0.0, 2.0], dtype=float))
    assert np.allclose(iso_c[[0, 2, 3]], np.array([-2.0, 0.0, 2.0], dtype=float))


def test_centered_common_gain_constant_traces_become_zeros():
    sig = np.full(5, 60.0, dtype=float)
    iso = np.full(5, 85.0, dtype=float)

    sig_c, iso_c = prepare_centered_common_gain(sig, iso)

    assert np.allclose(sig_c, np.zeros_like(sig))
    assert np.allclose(iso_c, np.zeros_like(iso))


def test_centered_common_gain_rejects_all_nan_trace():
    sig = np.array([np.nan, np.nan], dtype=float)
    iso = np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="no finite values"):
        prepare_centered_common_gain(sig, iso)

