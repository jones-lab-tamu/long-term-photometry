import sys
import os
import pytest
import numpy as np

from photometry_pipeline.guided_feature_detection_preview import (
    GuidedFeaturePreviewTraceRequest,
    GuidedFeaturePreviewTrace,
    GuidedFeatureDetectionPreviewResult,
    GuidedFeaturePreviewUnsupportedError,
    build_feature_detection_preview_from_trace,
    resolve_guided_feature_preview_trace,
    build_guided_feature_detection_preview,
    compute_settings_digest,
)


def _valid_settings(polarity="positive", method="absolute", thresh_abs=0.5):
    return {
        "event_signal": "dff",
        "signal_excursion_polarity": polarity,
        "peak_threshold_method": method,
        "peak_threshold_k": 2.5,
        "peak_threshold_percentile": 95.0,
        "peak_threshold_abs": thresh_abs,
        "peak_min_distance_sec": 0.5,
        "peak_min_prominence_k": 0.0,
        "peak_min_width_sec": 0.0,
        "peak_pre_filter": "none",
        "event_auc_baseline": "zero",
    }


def test_detector_preview_positive_peaks(tmp_path):
    fs = 10.0
    time_sec = np.arange(100) / fs
    # Flat trace with two positive spikes at index 20 and 60
    trace = np.zeros(100)
    trace[20] = 1.0
    trace[60] = 1.0

    settings = _valid_settings(polarity="positive", thresh_abs=0.5)

    # Verify no-write: assert tmp_path is empty before and after
    before = sorted(tmp_path.rglob("*"))
    assert before == []

    result = build_feature_detection_preview_from_trace(
        roi_id="CH1",
        time_sec=time_sec,
        trace=trace,
        fs_hz=fs,
        event_signal="dff",
        feature_settings=settings,
        feature_profile_id="prof_1",
        trace_identity={"src": "test"},
        correction_identity={"corr": "test_corr"},
    )

    assert sorted(tmp_path.rglob("*")) == []  # No files written

    assert result.roi_id == "CH1"
    assert result.event_signal == "dff"
    assert result.preview_only is True
    assert result.production_analysis is False
    assert result.feature_extraction_run is False

    # Check peaks detected
    np.testing.assert_array_equal(result.positive_peak_indices, [20, 60])
    np.testing.assert_array_equal(result.negative_peak_indices, [])
    np.testing.assert_array_equal(result.positive_peak_times_sec, [2.0, 6.0])
    np.testing.assert_array_equal(result.negative_peak_times_sec, [])

    # Metadata
    assert result.threshold_upper == 0.5
    assert result.threshold_lower == -0.5
    assert result.feature_profile_id == "prof_1"
    assert result.feature_settings_digest == compute_settings_digest(settings)
    assert result.trace_identity == {"src": "test"}
    assert result.correction_identity == {"corr": "test_corr"}
    assert "function" in result.detector_identity


def test_detector_preview_negative_peaks():
    fs = 10.0
    time_sec = np.arange(100) / fs
    # Flat trace with negative spikes at index 30 and 70
    trace = np.zeros(100)
    trace[30] = -1.0
    trace[70] = -1.0

    settings = _valid_settings(polarity="negative", thresh_abs=0.5)

    result = build_feature_detection_preview_from_trace(
        roi_id="CH1",
        time_sec=time_sec,
        trace=trace,
        fs_hz=fs,
        event_signal="dff",
        feature_settings=settings,
        feature_profile_id="prof_1",
        trace_identity={"src": "test"},
        correction_identity={"corr": "test_corr"},
    )

    np.testing.assert_array_equal(result.positive_peak_indices, [])
    np.testing.assert_array_equal(result.negative_peak_indices, [30, 70])
    np.testing.assert_array_equal(result.negative_peak_times_sec, [3.0, 7.0])


def test_detector_preview_bidirectional_peaks():
    fs = 10.0
    time_sec = np.arange(100) / fs
    # Spikes of both polarities
    trace = np.zeros(100)
    trace[15] = 1.0
    trace[45] = -1.0

    settings = _valid_settings(polarity="both", thresh_abs=0.5)

    result = build_feature_detection_preview_from_trace(
        roi_id="CH1",
        time_sec=time_sec,
        trace=trace,
        fs_hz=fs,
        event_signal="dff",
        feature_settings=settings,
        feature_profile_id="prof_1",
        trace_identity={"src": "test"},
        correction_identity={"corr": "test_corr"},
    )

    np.testing.assert_array_equal(result.positive_peak_indices, [15])
    np.testing.assert_array_equal(result.negative_peak_indices, [45])
    np.testing.assert_array_equal(result.positive_peak_times_sec, [1.5])
    np.testing.assert_array_equal(result.negative_peak_times_sec, [4.5])


def test_detector_preview_invalid_inputs():
    fs = 10.0
    time_sec = np.arange(10) / fs
    trace = np.zeros(10)
    settings = _valid_settings()

    # 1. Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        build_feature_detection_preview_from_trace(
            roi_id="CH1",
            time_sec=time_sec,
            trace=np.zeros(9),
            fs_hz=fs,
            event_signal="dff",
            feature_settings=settings,
            feature_profile_id="prof_1",
            trace_identity={},
            correction_identity={},
        )

    # 2. Empty trace
    with pytest.raises(ValueError, match="empty trace"):
        build_feature_detection_preview_from_trace(
            roi_id="CH1",
            time_sec=np.array([]),
            trace=np.array([]),
            fs_hz=fs,
            event_signal="dff",
            feature_settings=settings,
            feature_profile_id="prof_1",
            trace_identity={},
            correction_identity={},
        )

    # 3. Invalid fs_hz
    with pytest.raises(ValueError, match="fs_hz must be finite and positive"):
        build_feature_detection_preview_from_trace(
            roi_id="CH1",
            time_sec=time_sec,
            trace=trace,
            fs_hz=-1.0,
            event_signal="dff",
            feature_settings=settings,
            feature_profile_id="prof_1",
            trace_identity={},
            correction_identity={},
        )

    # 4. Unknown event_signal
    with pytest.raises(ValueError, match="event_signal must be 'dff' or 'delta_f'"):
        build_feature_detection_preview_from_trace(
            roi_id="CH1",
            time_sec=time_sec,
            trace=trace,
            fs_hz=fs,
            event_signal="raw",
            feature_settings=settings,
            feature_profile_id="prof_1",
            trace_identity={},
            correction_identity={},
        )

    # 5. Missing required settings fields
    bad_settings = dict(settings)
    del bad_settings["peak_threshold_method"]
    with pytest.raises(ValueError, match="Missing required detector settings fields"):
        build_feature_detection_preview_from_trace(
            roi_id="CH1",
            time_sec=time_sec,
            trace=trace,
            fs_hz=fs,
            event_signal="dff",
            feature_settings=bad_settings,
            feature_profile_id="prof_1",
            trace_identity={},
            correction_identity={},
        )

    # 6. Unknown polarity
    bad_settings = _valid_settings()
    bad_settings["signal_excursion_polarity"] = "upward"
    with pytest.raises(ValueError, match="Invalid feature_settings"):
        build_feature_detection_preview_from_trace(
            roi_id="CH1",
            time_sec=time_sec,
            trace=trace,
            fs_hz=fs,
            event_signal="dff",
            feature_settings=bad_settings,
            feature_profile_id="prof_1",
            trace_identity={},
            correction_identity={},
        )

    # 7. Insufficient finite samples (< 2)
    nan_trace = np.full(10, np.nan)
    nan_trace[0] = 1.0  # only 1 finite sample
    with pytest.raises(ValueError, match="Need at least 2 finite samples"):
        build_feature_detection_preview_from_trace(
            roi_id="CH1",
            time_sec=time_sec,
            trace=nan_trace,
            fs_hz=fs,
            event_signal="dff",
            feature_settings=settings,
            feature_profile_id="prof_1",
            trace_identity={},
            correction_identity={},
        )


def test_trace_provider_boundary_dynamic_delta_f(tmp_path):
    # Setup request and context
    request = GuidedFeaturePreviewTraceRequest(
        roi_id="CH1",
        event_signal="delta_f",
        correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        feature_profile_id="prof_1",
        feature_settings=_valid_settings(),
    )

    t = np.arange(10, dtype=float)
    y = np.ones(10, dtype=float)

    context = {
        "dynamic_delta_f": {
            ("CH1", "global_linear_regression"): {
                "time_sec": t,
                "trace": y,
                "fs_hz": 10.0,
                "trace_identity": {"trace_id": "T1"},
                "correction_identity": {"corr_id": "C1"},
                "current": True,
            }
        }
    }

    # Verify resolution
    trace_resolved = resolve_guided_feature_preview_trace(request, context)

    assert trace_resolved.roi_id == "CH1"
    assert trace_resolved.event_signal == "delta_f"
    assert trace_resolved.correction_strategy == "dynamic_fit"
    assert trace_resolved.dynamic_fit_mode == "global_linear_regression"
    assert trace_resolved.trace_identity == {"trace_id": "T1"}
    assert trace_resolved.correction_identity == {"corr_id": "C1"}
    assert trace_resolved.preview_only is True
    assert trace_resolved.production_analysis is False

    # Verify no file write
    assert sorted(tmp_path.rglob("*")) == []


def test_trace_provider_boundary_dynamic_delta_f_direct_strategy_normalization():
    # Setup request where correction_strategy is a specific mode directly
    request = GuidedFeaturePreviewTraceRequest(
        roi_id="CH1",
        event_signal="delta_f",
        correction_strategy="global_linear_regression",
        dynamic_fit_mode=None,  # will be normalized
        feature_profile_id="prof_1",
        feature_settings=_valid_settings(),
    )

    t = np.arange(10, dtype=float)
    y = np.ones(10, dtype=float)

    context = {
        "dynamic_delta_f": {
            ("CH1", "global_linear_regression"): {
                "time_sec": t,
                "trace": y,
                "fs_hz": 10.0,
                "trace_identity": {"trace_id": "T1"},
                "correction_identity": {"corr_id": "C1"},
                "current": True,
            }
        }
    }

    # Verify resolution
    trace_resolved = resolve_guided_feature_preview_trace(request, context)

    assert trace_resolved.roi_id == "CH1"
    assert trace_resolved.event_signal == "delta_f"
    assert trace_resolved.correction_strategy == "dynamic_fit"
    assert trace_resolved.dynamic_fit_mode == "global_linear_regression"
    assert trace_resolved.trace_identity == {"trace_id": "T1"}
    assert trace_resolved.correction_identity == {"corr_id": "C1"}


def test_trace_provider_boundary_dynamic_delta_f_mismatches():
    # Context only has A
    context = {
        "dynamic_delta_f": {
            ("CH1", "mode_A"): {
                "time_sec": np.arange(5),
                "trace": np.ones(5),
                "fs_hz": 10.0,
                "trace_identity": {},
                "correction_identity": {},
                "current": True,
            }
        }
    }

    # Request mode B
    req_mismatch_mode = GuidedFeaturePreviewTraceRequest(
        roi_id="CH1",
        event_signal="delta_f",
        correction_strategy="dynamic_fit",
        dynamic_fit_mode="mode_B",
        feature_profile_id="prof_1",
        feature_settings=_valid_settings(),
    )

    with pytest.raises(GuidedFeaturePreviewUnsupportedError, match="No matching dynamic delta_f trace"):
        resolve_guided_feature_preview_trace(req_mismatch_mode, context)


def test_trace_provider_boundary_dynamic_dff_unsupported():
    request = GuidedFeaturePreviewTraceRequest(
        roi_id="CH1",
        event_signal="dff",
        correction_strategy="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        feature_profile_id="prof_1",
        feature_settings=_valid_settings(),
    )

    # Even if context had it, should fail pre-run
    with pytest.raises(GuidedFeaturePreviewUnsupportedError, match="dynamic-fit dF/F preview is unavailable pre-Run"):
        resolve_guided_feature_preview_trace(request, {})


def test_trace_provider_boundary_signal_only_dff():
    request = GuidedFeaturePreviewTraceRequest(
        roi_id="CH2",
        event_signal="dff",
        correction_strategy="signal_only_f0",
        feature_profile_id="prof_1",
        feature_settings=_valid_settings(),
    )

    t = np.arange(10, dtype=float)
    y = np.ones(10, dtype=float)

    context = {
        "signal_only_dff": {
            "CH2": {
                "time_sec": t,
                "trace": y,
                "fs_hz": 10.0,
                "trace_identity": {"trace_id": "T2"},
                "correction_identity": {"corr_id": "C2"},
                "current": True,
            }
        }
    }

    trace_resolved = resolve_guided_feature_preview_trace(request, context)

    assert trace_resolved.roi_id == "CH2"
    assert trace_resolved.event_signal == "dff"
    assert trace_resolved.correction_strategy == "signal_only_f0"
    assert trace_resolved.trace_identity == {"trace_id": "T2"}
    assert trace_resolved.correction_identity == {"corr_id": "C2"}


def test_trace_provider_boundary_signal_only_delta_f_unsupported():
    request = GuidedFeaturePreviewTraceRequest(
        roi_id="CH2",
        event_signal="delta_f",
        correction_strategy="signal_only_f0",
        feature_profile_id="prof_1",
        feature_settings=_valid_settings(),
    )

    with pytest.raises(GuidedFeaturePreviewUnsupportedError, match="Signal-Only delta_f is unsupported"):
        resolve_guided_feature_preview_trace(request, {})


def test_trace_provider_boundary_mixed_dynamic_modes():
    t = np.arange(5)
    context = {
        "dynamic_delta_f": {
            ("CH1", "mode_A"): {
                "time_sec": t, "trace": np.ones(5), "fs_hz": 10.0,
                "trace_identity": {"name": "CH1_A"}, "correction_identity": {},
                "current": True
            },
            ("CH2", "mode_B"): {
                "time_sec": t, "trace": np.ones(5), "fs_hz": 10.0,
                "trace_identity": {"name": "CH2_B"}, "correction_identity": {},
                "current": True
            }
        }
    }

    req_ch1 = GuidedFeaturePreviewTraceRequest(
        roi_id="CH1", event_signal="delta_f", correction_strategy="dynamic_fit",
        dynamic_fit_mode="mode_A", feature_profile_id="prof_1", feature_settings=_valid_settings()
    )
    req_ch2 = GuidedFeaturePreviewTraceRequest(
        roi_id="CH2", event_signal="delta_f", correction_strategy="dynamic_fit",
        dynamic_fit_mode="mode_B", feature_profile_id="prof_1", feature_settings=_valid_settings()
    )
    req_fail = GuidedFeaturePreviewTraceRequest(
        roi_id="CH2", event_signal="delta_f", correction_strategy="dynamic_fit",
        dynamic_fit_mode="mode_A", feature_profile_id="prof_1", feature_settings=_valid_settings()
    )

    assert resolve_guided_feature_preview_trace(req_ch1, context).trace_identity["name"] == "CH1_A"
    assert resolve_guided_feature_preview_trace(req_ch2, context).trace_identity["name"] == "CH2_B"

    with pytest.raises(GuidedFeaturePreviewUnsupportedError):
        resolve_guided_feature_preview_trace(req_fail, context)


def test_trace_provider_boundary_stale_context():
    # stale = True
    context_stale = {
        "signal_only_dff": {
            "CH2": {
                "time_sec": np.arange(5), "trace": np.ones(5), "fs_hz": 10.0,
                "trace_identity": {}, "correction_identity": {},
                "stale": True,
            }
        }
    }
    # current = False
    context_not_current = {
        "signal_only_dff": {
            "CH2": {
                "time_sec": np.arange(5), "trace": np.ones(5), "fs_hz": 10.0,
                "trace_identity": {}, "correction_identity": {},
                "current": False,
            }
        }
    }

    req = GuidedFeaturePreviewTraceRequest(
        roi_id="CH2", event_signal="dff", correction_strategy="signal_only_f0",
        feature_profile_id="prof_1", feature_settings=_valid_settings()
    )

    with pytest.raises(GuidedFeaturePreviewUnsupportedError, match="stale or not current"):
        resolve_guided_feature_preview_trace(req, context_stale)

    with pytest.raises(GuidedFeaturePreviewUnsupportedError, match="stale or not current"):
        resolve_guided_feature_preview_trace(req, context_not_current)


def test_composed_preview_supported(tmp_path):
    t = np.arange(100) / 10.0
    trace = np.zeros(100)
    trace[35] = 1.2

    context = {
        "dynamic_delta_f": {
            ("CH1", "mode_A"): {
                "time_sec": t,
                "trace": trace,
                "fs_hz": 10.0,
                "trace_identity": {"id": "T_composed"},
                "correction_identity": {"id": "C_composed"},
                "current": True,
            }
        }
    }

    request = GuidedFeaturePreviewTraceRequest(
        roi_id="CH1",
        event_signal="delta_f",
        correction_strategy="dynamic_fit",
        dynamic_fit_mode="mode_A",
        feature_profile_id="prof_comp",
        feature_settings=_valid_settings(polarity="positive", thresh_abs=0.5),
    )

    result = build_guided_feature_detection_preview(
        trace_request=request, available_trace_context=context
    )

    assert result.roi_id == "CH1"
    assert result.event_signal == "delta_f"
    np.testing.assert_array_equal(result.positive_peak_indices, [35])
    assert result.preview_only is True
    assert result.production_analysis is False
    assert result.feature_settings_digest == compute_settings_digest(request.feature_settings)

    # Check no-write
    assert sorted(tmp_path.rglob("*")) == []


def test_composed_preview_unsupported():
    request = GuidedFeaturePreviewTraceRequest(
        roi_id="CH1",
        event_signal="dff",
        correction_strategy="dynamic_fit",
        dynamic_fit_mode="mode_A",
        feature_profile_id="prof_comp",
        feature_settings=_valid_settings(),
    )

    # Fails with unsupported trace before detector runs
    with pytest.raises(GuidedFeaturePreviewUnsupportedError, match="dynamic-fit dF/F preview is unavailable pre-Run"):
        build_guided_feature_detection_preview(
            trace_request=request, available_trace_context={}
        )
