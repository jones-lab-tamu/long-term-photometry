import numpy as np
import pytest
import yaml
from scipy.signal import find_peaks

from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import (
    extract_features,
    get_peak_indices_for_trace,
)
from photometry_pipeline.core.types import Chunk
import tools.plot_phasic_dayplot_bundle as dayplot_bundle


def _old_threshold_distance_only_indices(trace: np.ndarray, fs_hz: float, cfg: Config) -> np.ndarray:
    """
    Reference detector matching pre-hardening semantics:
    threshold + minimum distance only, segmented by finite runs.
    """
    trace_arr = np.asarray(trace, dtype=float)
    is_valid = np.isfinite(trace_arr)
    padded = np.concatenate(([False], is_valid, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    dist_samples = max(1, int(cfg.peak_min_distance_sec * fs_hz))
    all_peaks = []
    for s, e in zip(starts, ends):
        run_y = trace_arr[s:e]
        if len(run_y) < 2:
            continue
        peaks, _ = find_peaks(run_y, height=cfg.peak_threshold_abs, distance=dist_samples)
        if len(peaks):
            all_peaks.append((s + peaks).astype(int))
    if not all_peaks:
        return np.array([], dtype=int)
    return np.concatenate(all_peaks)


def test_backward_compatibility_when_prominence_and_width_disabled():
    fs_hz = 20.0
    trace = np.zeros(220, dtype=float)
    trace[20] = 1.0
    trace[70] = 1.1
    trace[120] = 1.2
    trace[160] = np.nan
    trace[161] = np.nan
    trace[190] = 1.3

    cfg = Config(
        peak_threshold_method="absolute",
        peak_threshold_abs=0.8,
        peak_min_distance_sec=0.2,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
        peak_pre_filter="none",
    )

    expected = _old_threshold_distance_only_indices(trace, fs_hz, cfg)
    actual = get_peak_indices_for_trace(trace, fs_hz, cfg)
    np.testing.assert_array_equal(actual, expected)


def test_noise_only_false_positive_suppression_with_prominence_and_width():
    fs_hz = 40.0
    rng = np.random.default_rng(1)
    trace = 0.08 * rng.standard_normal(2000)

    loose_cfg = Config(
        peak_threshold_method="mean_std",
        peak_threshold_k=0.0,
        peak_min_distance_sec=0.05,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
        peak_pre_filter="none",
    )
    hardened_cfg = Config(
        peak_threshold_method="mean_std",
        peak_threshold_k=0.0,
        peak_min_distance_sec=0.05,
        peak_min_prominence_k=2.0,
        peak_min_width_sec=0.2,
        peak_pre_filter="none",
    )

    loose_peaks = get_peak_indices_for_trace(trace, fs_hz, loose_cfg)
    hardened_peaks = get_peak_indices_for_trace(trace, fs_hz, hardened_cfg)
    assert len(loose_peaks) > 100
    assert len(hardened_peaks) == 0


def test_true_event_retention_with_hardening_enabled():
    fs_hz = 40.0
    n = 2000
    t = np.arange(n) / fs_hz
    rng = np.random.default_rng(2)
    trace = 0.05 * rng.standard_normal(n)
    for center in (10.0, 25.0, 40.0):
        trace += 0.9 * np.exp(-0.5 * ((t - center) / 0.12) ** 2)

    cfg = Config(
        peak_threshold_method="mean_std",
        peak_threshold_k=0.2,
        peak_min_distance_sec=0.5,
        peak_min_prominence_k=2.0,
        peak_min_width_sec=0.2,
        peak_pre_filter="none",
    )
    peaks = get_peak_indices_for_trace(trace, fs_hz, cfg)
    peak_times = t[peaks]
    assert len(peaks) == 3
    np.testing.assert_allclose(peak_times, np.array([10.0, 25.0, 40.0]), atol=0.1, rtol=0.0)


def test_width_specific_rejection_of_narrow_excursions():
    fs_hz = 50.0
    t = np.arange(2000) / fs_hz
    trace = np.zeros_like(t)
    trace[200] = 1.0
    trace[600] = 1.0
    for center in (20.0, 30.0):
        trace += 0.9 * np.exp(-0.5 * ((t - center) / 0.18) ** 2)

    cfg_no_width = Config(
        peak_threshold_method="absolute",
        peak_threshold_abs=0.3,
        peak_min_distance_sec=0.2,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
        peak_pre_filter="none",
    )
    cfg_with_width = Config(
        peak_threshold_method="absolute",
        peak_threshold_abs=0.3,
        peak_min_distance_sec=0.2,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.12,
        peak_pre_filter="none",
    )

    peaks_no_width = get_peak_indices_for_trace(trace, fs_hz, cfg_no_width)
    peaks_with_width = get_peak_indices_for_trace(trace, fs_hz, cfg_with_width)
    assert len(peaks_no_width) == 4
    assert len(peaks_with_width) == 2
    np.testing.assert_allclose(t[peaks_with_width], np.array([20.0, 30.0]), atol=0.05, rtol=0.0)


def test_prominence_specific_rejection_of_low_prominence_fluctuations():
    fs_hz = 40.0
    t = np.arange(2400) / fs_hz
    trace = 0.55 + 0.03 * np.sin(2 * np.pi * 1.8 * t)
    for center in (15.0, 35.0, 50.0):
        trace += 0.35 * np.exp(-0.5 * ((t - center) / 0.10) ** 2)

    cfg_no_prom = Config(
        peak_threshold_method="absolute",
        peak_threshold_abs=0.5,
        peak_min_distance_sec=0.15,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
        peak_pre_filter="none",
    )
    cfg_with_prom = Config(
        peak_threshold_method="absolute",
        peak_threshold_abs=0.5,
        peak_min_distance_sec=0.15,
        peak_min_prominence_k=2.0,
        peak_min_width_sec=0.0,
        peak_pre_filter="none",
    )

    peaks_no_prom = get_peak_indices_for_trace(trace, fs_hz, cfg_no_prom)
    peaks_with_prom = get_peak_indices_for_trace(trace, fs_hz, cfg_with_prom)
    assert len(peaks_no_prom) > 50
    assert len(peaks_with_prom) == 3
    np.testing.assert_allclose(t[peaks_with_prom], np.array([15.0, 35.0, 50.0]), atol=0.05, rtol=0.0)


def test_dayplot_verification_parity_with_hardening_enabled():
    fs_hz = 10.0
    t = np.arange(0.0, 600.0, 1.0 / fs_hz)
    rng = np.random.default_rng(7)

    delta_f = 0.03 * np.sin(0.05 * t) + 0.015 * rng.standard_normal(len(t))
    for center in (90.0, 260.0, 430.0):
        delta_f += 2.0 * np.exp(-0.5 * ((t - center) / 0.9) ** 2)
    dff = 0.8 * np.sin(2 * np.pi * 1.4 * t) + 0.3 * np.sin(2 * np.pi * 3.2 * t)

    cfg = Config(
        event_signal="delta_f",
        target_fs_hz=fs_hz,
        lowpass_hz=1.0,
        peak_pre_filter="lowpass",
        peak_threshold_method="mean_std",
        peak_threshold_k=1.5,
        peak_min_distance_sec=0.5,
        peak_min_prominence_k=1.8,
        peak_min_width_sec=0.12,
    )
    chunk = Chunk(
        chunk_id=0,
        source_file="chunk_0.csv",
        format="rwd",
        time_sec=t,
        uv_raw=np.zeros((len(t), 1), dtype=float),
        sig_raw=np.zeros((len(t), 1), dtype=float),
        fs_hz=fs_hz,
        channel_names=["Region0"],
        dff=dff.reshape(-1, 1),
        delta_f=delta_f.reshape(-1, 1),
    )
    expected_count = int(extract_features(chunk, cfg).iloc[0]["peak_count"])
    assert expected_count > 0

    peaks = dayplot_bundle.verify_peak_count_strict(
        detection_trace=delta_f,
        time_arr=t,
        fs=fs_hz,
        config=cfg,
        expected_count=expected_count,
        roi="Region0",
        cid=0,
        src_file="chunk_0.csv",
    )
    assert len(peaks) == expected_count


def test_config_defaults_and_roundtrip_for_hardening_knobs(tmp_path):
    cfg_default = Config()
    assert cfg_default.peak_min_prominence_k == pytest.approx(0.0)
    assert cfg_default.peak_min_width_sec == pytest.approx(0.0)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "peak_min_prominence_k": 1.75,
                "peak_min_width_sec": 0.25,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cfg_loaded = Config.from_yaml(str(cfg_path))
    assert cfg_loaded.peak_min_prominence_k == pytest.approx(1.75)
    assert cfg_loaded.peak_min_width_sec == pytest.approx(0.25)


def test_config_rejects_negative_hardening_knobs(tmp_path):
    cfg_path = tmp_path / "cfg_bad.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "peak_min_prominence_k": -1.0,
                "peak_min_width_sec": 0.0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="peak_min_prominence_k must be >= 0"):
        Config.from_yaml(str(cfg_path))

