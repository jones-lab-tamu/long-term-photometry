import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import (
    extract_features,
    get_peak_indices_for_trace,
)
from photometry_pipeline.core.types import Chunk
import tools.verification.plot_phasic_qc_grid as qc_grid


def _make_trace_with_true_and_false_candidates(fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, 120.0, 1.0 / fs_hz)
    trace = 0.02 * np.sin(2 * np.pi * 0.15 * t)

    # Narrow, high excursions (should be rejected by width criterion when enabled).
    for sec in (15.0, 35.0, 55.0, 95.0):
        idx = int(round(sec * fs_hz))
        if 0 <= idx < len(trace):
            trace[idx] += 0.9

    # Broad, prominent events (should be retained).
    for center in (25.0, 70.0, 105.0):
        trace += 0.75 * np.exp(-0.5 * ((t - center) / 0.35) ** 2)

    return t, trace


def test_qc_grid_verification_consumer_uses_hardened_authoritative_detector():
    fs_hz = 20.0
    t, trace = _make_trace_with_true_and_false_candidates(fs_hz)

    cfg = Config(
        target_fs_hz=fs_hz,
        event_signal="dff",
        peak_pre_filter="none",
        peak_threshold_method="absolute",
        peak_threshold_abs=0.2,
        peak_min_distance_sec=0.2,
        peak_min_prominence_k=1.8,
        peak_min_width_sec=0.20,
    )

    chunk = Chunk(
        chunk_id=11,
        source_file="synthetic_chunk.csv",
        format="rwd",
        time_sec=t,
        uv_raw=np.zeros((len(t), 1), dtype=float),
        sig_raw=np.zeros((len(t), 1), dtype=float),
        fs_hz=fs_hz,
        channel_names=["Region0"],
        dff=trace.reshape(-1, 1),
    )
    expected_count = int(extract_features(chunk, cfg).iloc[0]["peak_count"])
    assert expected_count == 3

    consumer_indices = qc_grid.verify_peak_count_strict(
        trace_arr=trace,
        time_arr=t,
        fs=fs_hz,
        config=cfg,
        expected_count=expected_count,
        roi="Region0",
        cid=11,
        src_file="synthetic_chunk.csv",
    )
    authoritative_indices = get_peak_indices_for_trace(trace, fs_hz, cfg)

    np.testing.assert_array_equal(consumer_indices, authoritative_indices)
    np.testing.assert_allclose(t[consumer_indices], np.array([25.0, 70.0, 105.0]), atol=0.15, rtol=0.0)


def test_qc_grid_verification_consumer_backward_compat_when_hardening_disabled():
    fs_hz = 20.0
    t, trace = _make_trace_with_true_and_false_candidates(fs_hz)

    cfg = Config(
        target_fs_hz=fs_hz,
        event_signal="dff",
        peak_pre_filter="none",
        peak_threshold_method="absolute",
        peak_threshold_abs=0.2,
        peak_min_distance_sec=0.2,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
    )

    chunk = Chunk(
        chunk_id=12,
        source_file="synthetic_chunk.csv",
        format="rwd",
        time_sec=t,
        uv_raw=np.zeros((len(t), 1), dtype=float),
        sig_raw=np.zeros((len(t), 1), dtype=float),
        fs_hz=fs_hz,
        channel_names=["Region0"],
        dff=trace.reshape(-1, 1),
    )
    expected_count = int(extract_features(chunk, cfg).iloc[0]["peak_count"])

    consumer_indices = qc_grid.verify_peak_count_strict(
        trace_arr=trace,
        time_arr=t,
        fs=fs_hz,
        config=cfg,
        expected_count=expected_count,
        roi="Region0",
        cid=12,
        src_file="synthetic_chunk.csv",
    )
    authoritative_indices = get_peak_indices_for_trace(trace, fs_hz, cfg)

    np.testing.assert_array_equal(consumer_indices, authoritative_indices)
    assert len(consumer_indices) > 3

