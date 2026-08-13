from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing, regression
from photometry_pipeline.core.types import Chunk, PerRoiCorrectionSpec
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.preview import correction_preview
from photometry_pipeline.guided_execution_payloads import (
    GUIDED_CONFIG_DEFAULT_OVERRIDES,
    build_guided_execution_startup_mapping_contract,
    build_guided_execution_startup_mapping_contract_for_preprocessing,
)
from photometry_pipeline.guided_feature_detection_preview import (
    build_feature_detection_preview_from_trace,
    compute_settings_digest,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)


def _dynamic_chunk(
    time_sec: np.ndarray,
    *,
    channel_names: tuple[str, ...] = ("ROI0",),
) -> Chunk:
    time = np.asarray(time_sec, dtype=float)
    signal = 2.0 + 0.8 * np.exp(-time / 4.0)
    reference = 1.0 + 0.4 * np.exp(-time / 6.0)
    columns = len(channel_names)
    uv = np.column_stack([reference + 0.03 * index for index in range(columns)])
    sig = np.column_stack([signal + 0.07 * index for index in range(columns)])
    chunk = Chunk(
        chunk_id=0,
        source_file="guided-advanced-preprocessing-synthetic.rwd",
        format="rwd",
        time_sec=time.copy(),
        uv_raw=uv,
        sig_raw=sig,
        fs_hz=10.0,
        channel_names=list(channel_names),
        metadata={},
    )
    config = Config(
        dynamic_fit_mode="global_linear_regression",
        lowpass_hz=1.0,
        min_samples_per_window=8,
    )
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(
        chunk.uv_raw, chunk.fs_hz, config
    )
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(
        chunk.sig_raw, chunk.fs_hz, config
    )
    return chunk


def _dynamic_spec(roi_id: str = "ROI0") -> PerRoiCorrectionSpec:
    return PerRoiCorrectionSpec(
        roi_id=roi_id,
        strategy_family="dynamic_fit",
        selected_strategy="global_linear_regression",
        dynamic_fit_mode="global_linear_regression",
    )


def _single_bleach_context(time_sec: np.ndarray) -> regression.RecordingWideBleachContext:
    sampler = regression.RecordingWideBleachSampler(capacity=64, seed=13)
    signal = 2.0 + 0.8 * np.exp(-time_sec / 4.0)
    reference = 1.0 + 0.4 * np.exp(-time_sec / 6.0)
    sampler.add("ROI0", time_sec, signal, reference)
    return regression.build_recording_wide_bleach_context(
        sampler,
        mode="single_exponential",
        recording_duration_sec=12.0,
        time_basis="recording_relative_acquisition_time",
        sample_rate_hz=10.0,
    )


def test_guided_preprocessing_defaults_validate_and_change_plan_identity():
    plan = GuidedNewAnalysisDraftPlan()
    assert plan.lowpass_hz == 1.0
    assert plan.bleach_correction_mode == "none"

    identity = compute_guided_new_analysis_draft_plan_identity(plan)
    assert compute_guided_new_analysis_draft_plan_identity(
        replace(plan, lowpass_hz=2.0)
    ) != identity
    assert compute_guided_new_analysis_draft_plan_identity(
        replace(plan, bleach_correction_mode="single_exponential")
    ) != identity

    for invalid_cutoff in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="lowpass_hz"):
            replace(plan, lowpass_hz=invalid_cutoff)
    with pytest.raises(ValueError, match="bleach_correction_mode"):
        replace(plan, bleach_correction_mode="polynomial")


def test_guided_startup_contract_carries_only_the_two_advanced_values():
    contract = build_guided_execution_startup_mapping_contract_for_preprocessing(
        lowpass_hz=2.5,
        bleach_correction_mode="double_exponential",
    )
    overrides = {item.name: item.value for item in contract.fixed_config_overrides}
    assert overrides["lowpass_hz"] == 2.5
    assert overrides["bleach_correction_mode"] == "double_exponential"
    canonical = build_guided_execution_startup_mapping_contract()
    canonical_overrides = {
        item.name: item.value for item in canonical.fixed_config_overrides
    }
    assert set(overrides) == set(canonical_overrides) == set(
        GUIDED_CONFIG_DEFAULT_OVERRIDES
    )
    assert all(
        overrides[name] == value
        for name, value in canonical_overrides.items()
        if name not in {"lowpass_hz", "bleach_correction_mode"}
    )


def test_guided_lowpass_at_or_above_nyquist_preserves_identity_bypass():
    data = np.column_stack(
        [np.sin(np.arange(40, dtype=float) / 3.0), np.cos(np.arange(40, dtype=float) / 5.0)]
    )
    filtered, metadata = preprocessing.lowpass_filter_with_meta(
        data,
        10.0,
        Config(lowpass_hz=5.0, filter_order=3),
    )
    assert filtered is data
    assert metadata == {}


def test_recording_wide_double_exponential_dispatches_signal_and_reference_independently():
    time_sec = np.arange(160, dtype=float) / 10.0
    signal = 2.0 + 0.9 * np.exp(-time_sec / 3.0) + 0.4 * np.exp(-time_sec / 9.0)
    reference = 1.0 + 0.5 * np.exp(-time_sec / 2.5) + 0.2 * np.exp(-time_sec / 8.0)
    sampler = regression.RecordingWideBleachSampler(capacity=64, seed=17)
    sampler.add("ROI0", time_sec, signal, reference)
    context = regression.build_recording_wide_bleach_context(
        sampler,
        mode="double_exponential",
        recording_duration_sec=16.0,
        time_basis="recording_relative_acquisition_time",
        sample_rate_hz=10.0,
    )
    assert context.mode_resolved == "double_exponential"
    assert context.per_roi["ROI0"]["signal"]["fit_model"] == "double_exponential"
    assert context.per_roi["ROI0"]["reference"]["fit_model"] == "double_exponential"
    assert context.per_roi["ROI0"]["signal"]["fit_succeeded"] is True
    assert context.per_roi["ROI0"]["reference"]["fit_succeeded"] is True


def test_intermittent_recording_wide_sampling_uses_cumulative_recorded_time():
    local_time = np.arange(20, dtype=float) / 10.0
    sampler = regression.RecordingWideBleachSampler(capacity=64, seed=19)
    sampler.add("ROI0", local_time, 2.0 + local_time, 1.0 + local_time)
    sampler.add("ROI0", 2.0 + local_time, 2.0 + local_time, 1.0 + local_time)
    sampled_time, _values = sampler.sampled_arrays("ROI0", "signal")
    assert sampled_time[0] == 0.0
    assert sampled_time[-1] == 3.9
    assert np.all(np.diff(sampled_time) >= 0.0)


def test_event_preview_identity_records_both_guided_preprocessing_values():
    settings = {
        "event_signal": "dff",
        "signal_excursion_polarity": "positive",
        "peak_threshold_method": "absolute",
        "peak_threshold_k": 2.5,
        "peak_threshold_percentile": 95.0,
        "peak_threshold_abs": 0.5,
        "peak_min_distance_sec": 0.5,
        "peak_min_prominence_k": 0.0,
        "peak_min_width_sec": 0.0,
        "peak_pre_filter": "lowpass",
        "event_auc_baseline": "zero",
    }
    result = build_feature_detection_preview_from_trace(
        roi_id="ROI0",
        time_sec=np.arange(40, dtype=float) / 10.0,
        trace=np.sin(np.arange(40, dtype=float) / 4.0),
        fs_hz=10.0,
        event_signal="dff",
        feature_settings=settings,
        feature_profile_id="guided-preview",
        trace_identity={"trace": "synthetic"},
        correction_identity={"correction": "synthetic"},
        preprocessing_lowpass_hz=2.5,
        preprocessing_bleach_correction_mode="single_exponential",
    )
    assert result.detector_identity["lowpass_hz"] == 2.5
    assert result.detector_identity["guided_bleach_correction_mode"] == (
        "single_exponential"
    )
    digest_settings = dict(settings)
    digest_settings["guided_preprocessing_lowpass_hz"] = 2.5
    digest_settings["guided_preprocessing_bleach_correction_mode"] = (
        "single_exponential"
    )
    assert result.feature_settings_digest == compute_settings_digest(
        digest_settings
    )


def test_recording_wide_sampler_is_deterministic_and_bounded():
    time_sec = np.arange(10_000, dtype=float) / 10.0
    signal = 2.0 + np.sin(time_sec)
    reference = 1.0 + np.cos(time_sec)
    first = regression.RecordingWideBleachSampler(capacity=8, seed=7)
    second = regression.RecordingWideBleachSampler(capacity=8, seed=7)
    for sampler in (first, second):
        sampler.add("ROI0", time_sec[:55], signal[:55], reference[:55])
        sampler.add("ROI0", time_sec[55:], signal[55:], reference[55:])

    first_time, first_signal = first.sampled_arrays("ROI0", "signal")
    second_time, second_signal = second.sampled_arrays("ROI0", "signal")
    assert first_time.shape == first_signal.shape == (8,)
    np.testing.assert_array_equal(first_time, second_time)
    np.testing.assert_array_equal(first_signal, second_signal)
    assert first.channels() == ("ROI0",)
    assert first._reservoirs["ROI0"]["signal"].count == time_sec.size
    assert np.max(first_time) > 500.0


def test_recording_wide_reservoir_first_post_capacity_draw_can_skip_replacement():
    class _HighRandom:
        def random(self, size):
            return np.full(size, 0.9, dtype=float)

    reservoir = regression._RecordingWideBleachReservoir(seed=0, capacity=2)
    reservoir._rng = _HighRandom()
    reservoir.add(
        np.array([0.0, 1.0]),
        np.array([10.0, 20.0]),
    )
    reservoir.add(np.array([2.0]), np.array([30.0]))

    sampled_time, sampled_values = reservoir.arrays()
    np.testing.assert_array_equal(sampled_time, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(sampled_values, np.array([10.0, 20.0]))
    assert reservoir.count == 3


def test_intermittent_bleach_fit_uses_only_the_accepted_pass1_manifest(monkeypatch):
    accepted = _dynamic_chunk(np.arange(120, dtype=float) / 10.0)
    rejected = _dynamic_chunk(np.arange(120, dtype=float) / 10.0)
    pipeline = Pipeline(
        Config(
            acquisition_mode="intermittent",
            baseline_method="uv_raw_percentile_session",
            bleach_correction_mode="single_exponential",
            chunk_duration_sec=99.0,
            target_fs_hz=10.0,
        ),
        per_roi_correction={"ROI0": _dynamic_spec()},
    )
    pipeline.file_list = ["accepted", "rejected"]
    pipeline._guided_reference_dynamic_strategy_map = lambda: {}
    observed_entries = []

    def fake_iter(entries, _force_format, phase_name):
        if phase_name == "pass1":
            assert tuple(entries) == ("accepted", "rejected")
            yield 0, "accepted", accepted, 0.0
        elif phase_name == "guided_bleach_fit":
            observed_entries.append(tuple(entries))
            yield 0, "accepted", accepted, 0.0

    monkeypatch.setattr(pipeline, "_iter_entry_chunks_for_pass", fake_iter)
    pipeline.run_pass_1("rwd")

    assert pipeline._pass1_manifest == ["accepted"]
    assert observed_entries == [("accepted",)]
    assert pipeline._recording_wide_bleach_context is not None
    assert pipeline._recording_wide_bleach_context.per_roi["ROI0"][
        "signal_samples_seen"
    ] == accepted.time_sec.size
    assert rejected is not accepted


def test_intermittent_bleach_time_uses_actual_right_open_chunk_duration():
    pipeline = Pipeline(
        Config(
            acquisition_mode="intermittent",
            bleach_correction_mode="single_exponential",
            chunk_duration_sec=99.0,
            target_fs_hz=10.0,
        ),
        per_roi_correction={"ROI0": _dynamic_spec()},
    )
    first_chunk = _dynamic_chunk(np.arange(100, dtype=float) / 10.0)
    second_chunk = _dynamic_chunk(np.arange(60, dtype=float) / 10.0)
    first_time, first_duration, is_continuous = (
        pipeline._guided_recording_wide_bleach_chunk_times(
            "first", first_chunk, session_offset_sec=0.0
        )
    )
    second_time, second_duration, _ = pipeline._guided_recording_wide_bleach_chunk_times(
        "second", second_chunk, session_offset_sec=first_duration
    )

    assert is_continuous is False
    assert first_duration == 10.0
    assert second_duration == 6.0
    assert first_time[0] == 0.0 and first_time[-1] == 9.9
    assert second_time[0] == 10.0 and second_time[-1] == 15.9
    assert first_duration + second_duration == 16.0

    npm_chunk = _dynamic_chunk(np.arange(60, dtype=float) / 10.0)
    npm_chunk.metadata.update(
        {
            "guided_npm_within_session_start_sec": 0.0,
            "guided_npm_actual_elapsed_sec": 123.0,
        }
    )
    npm_time, npm_duration, _ = pipeline._guided_recording_wide_bleach_chunk_times(
        "npm", npm_chunk, session_offset_sec=16.0
    )
    assert npm_duration == 6.0
    assert npm_time[0] == pytest.approx(16.0)
    assert npm_time[-1] == pytest.approx(21.9)


def test_preview_and_run_use_the_same_actual_intermittent_duration_rule(monkeypatch):
    first_chunk = _dynamic_chunk(np.arange(100, dtype=float) / 10.0)
    second_chunk = _dynamic_chunk(np.arange(60, dtype=float) / 10.0)
    chunks = {"s1": first_chunk, "s2": second_chunk}

    def fake_load_chunk(source, _input_format, _cfg, **_kwargs):
        return chunks[Path(source).name]

    monkeypatch.setattr(correction_preview, "load_chunk", fake_load_chunk)
    cfg = Config(
        acquisition_mode="intermittent",
        bleach_correction_mode="single_exponential",
        chunk_duration_sec=99.0,
        target_fs_hz=10.0,
    )
    context, selected_time = (
        correction_preview._build_guided_preview_recording_wide_bleach_context(
            source_files=("s1", "s2"),
            selected_source_file="s2",
            selected_roi="ROI0",
            input_format="rwd",
            cfg=cfg,
            acquisition_mode="intermittent",
        )
    )
    pipeline = Pipeline(cfg, per_roi_correction={"ROI0": _dynamic_spec()})
    run_time, run_duration, _ = pipeline._guided_recording_wide_bleach_chunk_times(
        "s2", second_chunk, session_offset_sec=10.0
    )

    assert context.recording_duration_sec == 16.0
    assert selected_time is not None
    np.testing.assert_array_equal(selected_time, run_time)
    assert run_duration == 6.0


def test_recording_wide_fit_resolves_sampling_rate_before_each_single_fit(monkeypatch):
    sampler = regression.RecordingWideBleachSampler(capacity=64, seed=23)
    time_sec = np.arange(40, dtype=float) / 10.0
    sampler.add("ROI0", time_sec, 2.0 + time_sec, 1.0 + time_sec)
    calls = []
    original_fit = regression._fit_single_exponential_with_offset_at_times

    def wrapped_fit(trace, sample_time, sample_rate_hz, *, recording_duration_sec):
        calls.append(float(sample_rate_hz))
        return original_fit(
            trace,
            sample_time,
            sample_rate_hz,
            recording_duration_sec=recording_duration_sec,
        )

    monkeypatch.setattr(
        regression, "_fit_single_exponential_with_offset_at_times", wrapped_fit
    )
    regression.build_recording_wide_bleach_context(
        sampler,
        mode="single_exponential",
        recording_duration_sec=4.0,
        time_basis="recording_relative_acquisition_time",
        sample_rate_hz=10.0,
    )

    assert calls == [10.0, 10.0]


def test_recording_wide_fit_does_not_infer_fs_from_random_reservoir_spacing(
    monkeypatch,
):
    sampler = regression.RecordingWideBleachSampler(capacity=8, seed=29)
    time_sec = np.arange(100, dtype=float) / 10.0
    sampler.add("ROI0", time_sec, 2.0 + time_sec, 1.0 + time_sec)
    calls = []
    original_fit = regression._fit_single_exponential_with_offset_at_times

    def wrapped_fit(trace, sample_time, sample_rate_hz, *, recording_duration_sec):
        calls.append(float(sample_rate_hz))
        return original_fit(
            trace,
            sample_time,
            sample_rate_hz,
            recording_duration_sec=recording_duration_sec,
        )

    monkeypatch.setattr(
        regression, "_fit_single_exponential_with_offset_at_times", wrapped_fit
    )
    regression.build_recording_wide_bleach_context(
        sampler,
        mode="single_exponential",
        recording_duration_sec=10.0,
        time_basis="recording_relative_acquisition_time",
        sample_rate_hz=None,
    )

    assert calls == [1.0, 1.0]


def test_recording_wide_context_uses_one_explicit_time_model_for_dynamic_fit():
    time_sec = np.arange(120, dtype=float) / 10.0
    context = _single_bleach_context(time_sec)
    assert context.scope == "recording_wide"
    assert context.time_basis == "recording_relative_acquisition_time"
    assert context.recording_duration_sec == 12.0
    assert context.per_roi["ROI0"]["signal"]["fit_succeeded"] is True

    chunk = _dynamic_chunk(time_sec, channel_names=("ROI0", "ROI1"))
    raw_signal = chunk.sig_raw
    raw_reference = chunk.uv_raw
    config = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="single_exponential",
        lowpass_hz=1.0,
        min_samples_per_window=8,
    )
    uv_fit, delta_f = regression.fit_chunk_dynamic(
        chunk,
        config,
        mode="phasic",
        per_roi_correction={
            "ROI0": _dynamic_spec(),
            "ROI1": PerRoiCorrectionSpec(
                roi_id="ROI1",
                strategy_family="signal_only_f0",
                selected_strategy="signal_only_f0",
            ),
        },
        bleach_correction_context=context,
        bleach_time_sec=time_sec,
    )
    assert uv_fit is not None and delta_f is not None
    assert chunk.sig_raw is raw_signal
    assert chunk.uv_raw is raw_reference
    assert chunk.metadata["bleach_correction_scope"] == "recording_wide"
    assert chunk.metadata["bleach_correction_time_basis"] == (
        "recording_relative_acquisition_time"
    )
    assert chunk.metadata["bleach_correction"]["ROI0"]["signal_applied"] is True
    assert "ROI1" not in chunk.metadata["bleach_correction"]


def test_guided_bleach_off_is_an_exact_preprocessing_noop():
    time_sec = np.arange(120, dtype=float) / 10.0
    chunk = _dynamic_chunk(time_sec)
    raw_signal = chunk.sig_raw
    raw_reference = chunk.uv_raw
    filtered_signal = chunk.sig_filt
    filtered_reference = chunk.uv_filt
    config = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="none",
        lowpass_hz=1.0,
        min_samples_per_window=8,
    )
    regression.fit_chunk_dynamic(
        chunk,
        config,
        mode="phasic",
        per_roi_correction={"ROI0": _dynamic_spec()},
    )
    assert chunk.sig_raw is raw_signal
    assert chunk.uv_raw is raw_reference
    assert chunk.sig_filt is filtered_signal
    assert chunk.uv_filt is filtered_reference
    assert chunk.metadata["bleach_correction_scope"] == "none"
    assert chunk.metadata["bleach_correction_applied"] is False


def test_recording_wide_fit_failure_is_nonfatal_per_roi():
    short_time = np.arange(3, dtype=float) / 10.0
    sampler = regression.RecordingWideBleachSampler(capacity=8, seed=0)
    sampler.add(
        "ROI0",
        short_time,
        np.ones(short_time.size),
        np.ones(short_time.size),
    )
    context = regression.build_recording_wide_bleach_context(
        sampler,
        mode="single_exponential",
        recording_duration_sec=12.0,
        time_basis="cumulative_recorded_acquisition_time",
        sample_rate_hz=10.0,
    )
    chunk = _dynamic_chunk(np.arange(120, dtype=float) / 10.0)
    raw_signal = chunk.sig_raw
    regression.fit_chunk_dynamic(
        chunk,
        Config(
            dynamic_fit_mode="global_linear_regression",
            bleach_correction_mode="single_exponential",
            min_samples_per_window=8,
        ),
        mode="phasic",
        per_roi_correction={"ROI0": _dynamic_spec()},
        bleach_correction_context=context,
        bleach_time_sec=chunk.time_sec,
    )
    assert chunk.sig_raw is raw_signal
    assert chunk.metadata["bleach_correction_applied"] is False
    assert chunk.metadata["bleach_correction"]["ROI0"]["signal_applied"] is False
