import sys

import numpy as np
import pandas as pd
import pytest

import photometry_pipeline.signal_only_f0 as standalone
from photometry_pipeline.core.signal_only_f0_candidate import DEFAULTS as SIGNAL_ONLY_F0_DEFAULTS
from photometry_pipeline.signal_only_f0 import (
    compute_signal_only_f0_dff,
    compute_signal_only_f0_dff_from_csv,
)


def _fake_candidate_with_f0(f0):
    def _fake(signal, time=None, *, config=None, return_uncapped_candidate=False):
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        f0_arr = np.asarray(f0, dtype=float).reshape(-1)
        if f0_arr.size == 1:
            f0_arr = np.full(signal_arr.shape, float(f0_arr[0]))
        assert return_uncapped_candidate is True
        return {
            "signal_only_f0_candidate": np.minimum(f0_arr, signal_arr),
            "signal_only_f0_candidate_uncapped": f0_arr,
            "signal_only_f0_status": "ok",
            "signal_only_f0_candidate_viability": "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    return _fake


def _capturing_fake_candidate(captured, f0=10.0):
    def _fake(signal, time=None, *, config=None, return_uncapped_candidate=False):
        captured["config"] = dict(config or {})
        return _fake_candidate_with_f0(f0)(
            signal,
            time,
            config=config,
            return_uncapped_candidate=return_uncapped_candidate,
        )

    return _fake


def test_array_helper_returns_same_length_outputs_with_core_logic():
    signal = 1.0 + 0.1 * np.sin(np.linspace(0.0, 8.0 * np.pi, 160))

    result = compute_signal_only_f0_dff(signal, sampling_rate_hz=20.0)

    assert result.signal.shape == signal.shape
    assert result.signal_only_f0.shape == signal.shape
    assert result.dff.shape == signal.shape
    assert result.parameters["denominator_source"] == "signal_only_f0_candidate_uncapped"


def test_default_wrapper_config_preserves_core_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _capturing_fake_candidate(captured))

    result = compute_signal_only_f0_dff(np.array([10.0, 11.0, 12.0]))

    cfg = captured["config"]
    assert cfg["signal_only_f0_low_quantile"] == SIGNAL_ONLY_F0_DEFAULTS["signal_only_f0_low_quantile"]
    assert cfg["signal_only_f0_window_sec"] == SIGNAL_ONLY_F0_DEFAULTS["signal_only_f0_window_sec"]
    assert cfg["signal_only_f0_smoothing_window_sec"] == SIGNAL_ONLY_F0_DEFAULTS["signal_only_f0_smoothing_window_sec"]
    assert result.parameters["parameter_sources"] == {
        "percentile": "core_default",
        "baseline_window_sec": "core_default",
        "smoothing_window_sec": "core_default",
    }


def test_user_parameters_override_core_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _capturing_fake_candidate(captured))

    result = compute_signal_only_f0_dff(
        np.array([10.0, 11.0, 12.0]),
        percentile=5.0,
        baseline_window_sec=300.0,
        smoothing_window_sec=30.0,
    )

    cfg = captured["config"]
    assert cfg["signal_only_f0_low_quantile"] == 0.05
    assert cfg["signal_only_f0_window_sec"] == 300.0
    assert cfg["signal_only_f0_smoothing_window_sec"] == 30.0
    assert result.parameters["parameter_sources"] == {
        "percentile": "user_override",
        "baseline_window_sec": "user_override",
        "smoothing_window_sec": "user_override",
    }


def test_array_helper_uses_uncapped_formula(monkeypatch):
    signal = np.array([10.0, 12.0, 8.0, 20.0])
    f0 = np.array([9.0, 9.0, 10.0, 10.0])
    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _fake_candidate_with_f0(f0))

    result = compute_signal_only_f0_dff(signal)

    expected = (signal - f0) / f0
    np.testing.assert_allclose(result.signal_only_f0, f0)
    np.testing.assert_allclose(result.dff, expected)
    assert result.parameters["formula"] == "(signal - signal_only_f0_uncapped_for_dff) / signal_only_f0_uncapped_for_dff"


def test_negative_dff_is_preserved(monkeypatch):
    signal = np.array([8.0, 7.5, 9.0])
    f0 = np.array([10.0, 10.0, 10.0])
    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _fake_candidate_with_f0(f0))

    result = compute_signal_only_f0_dff(signal)

    assert np.all(result.dff < 0.0)
    assert result.parameters["negative_dff_preserved"] is True


def test_huge_peaks_and_locked_high_segments_are_not_clipped(monkeypatch):
    signal = np.array([10.0, 10.5, 1000.0, 1000.0, 1000.0, 11.0])
    f0 = np.full(signal.shape, 10.0)
    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _fake_candidate_with_f0(f0))

    result = compute_signal_only_f0_dff(signal)

    expected = (signal - f0) / f0
    np.testing.assert_allclose(result.dff, expected)
    assert np.nanmax(result.dff) > 90.0
    assert result.parameters["dff_clipped"] is False


def test_csv_helper_preserves_columns_adds_outputs_and_writes_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _fake_candidate_with_f0(10.0))
    input_csv = tmp_path / "student.csv"
    output_csv = tmp_path / "student_signal_only.csv"
    pd.DataFrame({"time": [0.0, 1.0, 2.0], "signal": [10.0, 12.0, 8.0], "roi": ["A", "A", "A"]}).to_csv(
        input_csv,
        index=False,
    )

    no_write = compute_signal_only_f0_dff_from_csv(input_csv, signal_column="signal", time_column="time")

    assert list(no_write.columns) == ["time", "signal", "roi", "signal_only_f0", "signal_only_dff"]
    assert not output_csv.exists()
    assert input_csv.exists()

    written = compute_signal_only_f0_dff_from_csv(
        input_csv,
        signal_column="signal",
        time_column="time",
        output_csv=output_csv,
    )

    assert output_csv.exists()
    loaded = pd.read_csv(output_csv)
    pd.testing.assert_frame_equal(written, loaded)
    assert "signal_only_f0" in loaded.columns
    assert "signal_only_dff" in loaded.columns


def test_csv_helper_rejects_input_overwrite_and_missing_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _fake_candidate_with_f0(10.0))
    input_csv = tmp_path / "student.csv"
    pd.DataFrame({"time": [0.0, 1.0], "signal": [1.0, 2.0]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="signal column not found"):
        compute_signal_only_f0_dff_from_csv(input_csv, signal_column="missing")
    with pytest.raises(ValueError, match="time column not found"):
        compute_signal_only_f0_dff_from_csv(input_csv, signal_column="signal", time_column="missing")
    with pytest.raises(ValueError, match="output_csv must be distinct"):
        compute_signal_only_f0_dff_from_csv(input_csv, signal_column="signal", output_csv=input_csv)


def test_nan_signal_raises_clear_error_before_core_call(monkeypatch):
    signal = np.array([10.0, np.nan, 12.0])
    called = {"value": False}

    def _fake(*args, **kwargs):
        called["value"] = True
        return {}

    monkeypatch.setattr(standalone, "compute_signal_only_f0_candidate", _fake)

    with pytest.raises(ValueError, match="signal contains non-finite values"):
        compute_signal_only_f0_dff(signal)
    assert called["value"] is False


def test_no_gui_imports_required():
    assert "gui.main_window" not in sys.modules
    assert "gui.run_spec" not in sys.modules
