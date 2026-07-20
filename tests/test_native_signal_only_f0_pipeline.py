"""Phase 1.2 native phasic Signal-Only F0 dispatch tests."""

from pathlib import Path
import copy
import json
from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest

import photometry_pipeline.pipeline as pipeline_module
from photometry_pipeline.config import Config
from photometry_pipeline.core import regression
from photometry_pipeline.core.signal_only_f0_candidate import (
    compute_signal_only_f0_candidate,
)
from photometry_pipeline.core.types import Chunk, PerRoiCorrectionSpec
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.pipeline import CorrectionProcessingError, Pipeline


def _chunk(*, n=800, fs=20.0, broken_uv=False):
    t = np.arange(n, dtype=float) / fs
    uv0 = 2.0 + 0.15 * np.cos(0.17 * t)
    uv1 = 3.0 + 0.12 * np.sin(0.11 * t + 0.4)
    sig0 = 5.0 + 0.35 * np.sin(0.31 * t) + 0.05 * np.sin(1.7 * t)
    sig1 = 7.0 + 0.30 * np.cos(0.23 * t) + 0.04 * np.sin(1.3 * t)
    uv = np.column_stack([uv0, uv1])
    sig = np.column_stack([sig0, sig1])
    if broken_uv:
        uv[:, 0] = np.nan
    return Chunk(
        chunk_id=4,
        source_file="session_04/fluorescence.csv",
        format="rwd",
        time_sec=t,
        uv_raw=uv,
        sig_raw=sig,
        fs_hz=fs,
        channel_names=["ROI0", "ROI1"],
        metadata={},
    )


def _cfg(fs=20.0):
    return Config(
        target_fs_hz=fs,
        lowpass_hz=2.0,
        filter_order=2,
        window_sec=20.0,
        min_samples_per_window=20,
        signal_only_f0_min_window_samples=21,
        signal_only_f0_min_robust_range=1e-6,
    )


def _signal_only_map(*roi_ids):
    return {
        roi: PerRoiCorrectionSpec(
            roi_id=roi,
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
        )
        for roi in roi_ids
    }


def _mixed_strategy_map():
    return {
        "ROI0": PerRoiCorrectionSpec(
            roi_id="ROI0",
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
        ),
        "ROI1": PerRoiCorrectionSpec(
            roi_id="ROI1",
            strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression",
            dynamic_fit_mode="global_linear_regression",
        ),
    }


def _processed_for_cache(strategy_map):
    pipeline = Pipeline(_cfg(), mode="phasic", per_roi_correction=strategy_map)
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    return pipeline._apply_standard_analysis(_chunk(), 4)


def test_mixed_pipeline_assembles_canonical_trace_and_excludes_signal_only_from_regression(
    monkeypatch,
):
    chunk = _chunk()
    cfg = _cfg()
    strategy_map = {
        "ROI0": PerRoiCorrectionSpec(
            roi_id="ROI0",
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
            parameter_identity="f0-params",
            evidence_identity="f0-evidence",
        ),
        "ROI1": PerRoiCorrectionSpec(
            roi_id="ROI1",
            strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression",
            dynamic_fit_mode="global_linear_regression",
        ),
    }
    seen_groups = []
    original = regression._dispatch_one_dynamic_fit_group

    def capture(sub_chunk, config, mode, resolved_mode):
        seen_groups.append((list(sub_chunk.channel_names), resolved_mode))
        return original(sub_chunk, config, mode, resolved_mode)

    monkeypatch.setattr(regression, "_dispatch_one_dynamic_fit_group", capture)
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=strategy_map)
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    processed = pipeline._apply_standard_analysis(chunk, chunk_id=4)

    assert seen_groups == [(["ROI1"], "global_linear_regression")]
    assert np.isnan(processed.uv_fit[:, 0]).all()
    assert np.isfinite(processed.dff[:, 0]).mean() > 0.99
    assert np.isfinite(processed.dff[:, 1]).mean() > 0.99
    consumed = processed.metadata["correction_strategy_consumed_by_roi"]
    assert consumed["ROI0"]["strategy_family"] == "signal_only_f0"
    assert consumed["ROI0"]["parameter_identity"] == "f0-params"
    assert consumed["ROI1"]["dynamic_fit_mode"] == "global_linear_regression"


def test_all_signal_only_phasic_chunk_invokes_no_dynamic_engine(monkeypatch):
    chunk = _chunk()
    cfg = _cfg()

    def fail(*args, **kwargs):
        raise AssertionError("dynamic-fit engine received an all-Signal-Only chunk")

    monkeypatch.setattr(regression, "_dispatch_one_dynamic_fit_group", fail)
    pipeline = Pipeline(
        cfg,
        mode="phasic",
        per_roi_correction=_signal_only_map("ROI0", "ROI1"),
    )
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    processed = pipeline._apply_standard_analysis(chunk, chunk_id=4)

    assert processed.metadata["dynamic_fit_group_count"] == 0
    assert np.isnan(processed.uv_fit).all()
    assert np.isfinite(processed.dff).all()


def test_broken_isosbestic_channel_does_not_affect_signal_only_canonical_trace():
    chunk = _chunk(broken_uv=True)
    cfg = _cfg()
    pipeline = Pipeline(
        cfg,
        mode="phasic",
        per_roi_correction=_signal_only_map("ROI0", "ROI1"),
    )
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    processed = pipeline._apply_standard_analysis(chunk, chunk_id=4)
    assert np.isfinite(processed.dff[:, 0]).mean() > 0.99
    assert np.isnan(processed.uv_fit[:, 0]).all()


def test_signal_only_candidate_is_computed_once_and_persisted_baseline_is_exact(
    monkeypatch, tmp_path
):
    chunk = _chunk()
    cfg = _cfg()
    strategy_map = _signal_only_map("ROI0", "ROI1")
    calls = []
    original = pipeline_module.compute_signal_only_f0_candidate

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "compute_signal_only_f0_candidate", count)
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=strategy_map)
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    processed = pipeline._apply_standard_analysis(chunk, chunk_id=4)
    pipeline._record_baseline_reference_candidate_metrics(
        processed, chunk_id=4, source_file=processed.source_file
    )
    assert len(calls) == 2

    for roi_index, roi in enumerate(("ROI0", "ROI1")):
        direct = compute_signal_only_f0_candidate(
            processed.sig_raw[:, roi_index],
            processed.time_sec,
            signal_state=processed.metadata["signal_only_f0_production_qc"][roi][
                "signal_state"
            ],
            config=pipeline._signal_only_f0_config(),
            return_uncapped_candidate=True,
        )
        np.testing.assert_allclose(
            processed.metadata["signal_only_f0_production_baseline"][roi],
            direct["signal_only_f0_candidate_uncapped"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            processed.dff[:, roi_index],
            100.0
            * (
                processed.sig_raw[:, roi_index]
                - direct["signal_only_f0_candidate_uncapped"]
            )
            / direct["signal_only_f0_candidate_uncapped"],
            rtol=0.0,
            atol=0.0,
        )

    cache_path = tmp_path / "phasic_trace_cache.h5"
    writer = Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg)
    writer.add_chunk(processed, chunk_id=4, source_file=processed.source_file)
    writer.finalize()
    with h5py.File(cache_path, "r") as handle:
        group = handle["roi/ROI0/chunk_4"]
        assert "signal_only_f0_baseline" in group
        assert "dff" in group
        assert group.attrs["correction_strategy_family"] == "signal_only_f0"
        assert group.attrs["signal_only_f0_production_available"]
        np.testing.assert_allclose(
            group["signal_only_f0_baseline"][()],
            processed.metadata["signal_only_f0_production_baseline"]["ROI0"],
            rtol=0.0,
            atol=0.0,
        )


def test_unusable_signal_fails_with_typed_correction_error():
    chunk = _chunk()
    chunk.sig_raw[:, 0] = 5.0
    pipeline = Pipeline(
        _cfg(), mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1")
    )
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    with pytest.raises(CorrectionProcessingError) as exc_info:
        pipeline._apply_standard_analysis(chunk, chunk_id=4)
    exc = exc_info.value
    assert exc.roi_id == "ROI0"
    assert exc.chunk_id == 4
    assert exc.source_file.endswith("fluorescence.csv")
    assert exc.selected_strategy == "signal_only_f0"


def test_signal_only_coverage_is_measured_against_expected_trace_length():
    cfg = _cfg()

    fully_valid = _chunk()
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1"))
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    assert np.isfinite(pipeline._apply_standard_analysis(fully_valid, 4).dff).all()

    limited_edge_nan = _chunk()
    limited_edge_nan.sig_raw[:20, 0] = np.nan
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1"))
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    assert np.isfinite(pipeline._apply_standard_analysis(limited_edge_nan, 4).dff[:, 0]).mean() > 0.95

    half_valid = _chunk()
    half_valid.sig_raw[:400, 0] = np.nan
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1"))
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    with pytest.raises(CorrectionProcessingError, match="raw signal finite coverage"):
        pipeline._apply_standard_analysis(half_valid, 4)

    ten_valid = _chunk()
    ten_valid.sig_raw[:790, 0] = np.nan
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1"))
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    with pytest.raises(CorrectionProcessingError, match="raw signal finite coverage"):
        pipeline._apply_standard_analysis(ten_valid, 4)


def test_signal_only_f0_and_canonical_dff_coverage_fail_independently(monkeypatch):
    original = pipeline_module.compute_signal_only_f0_candidate

    def low_f0_coverage(signal, time=None, **kwargs):
        result = original(signal, time, **kwargs)
        baseline = np.asarray(result["signal_only_f0_candidate_uncapped"], dtype=float).copy()
        baseline[:200] = np.nan
        result["signal_only_f0_candidate_uncapped"] = baseline
        return result

    monkeypatch.setattr(pipeline_module, "compute_signal_only_f0_candidate", low_f0_coverage)
    pipeline = Pipeline(_cfg(), mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1"))
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    with pytest.raises(CorrectionProcessingError, match="production F0 finite coverage"):
        pipeline._apply_standard_analysis(_chunk(), 4)

    def overflowing_dff(signal, time=None, **kwargs):
        result = original(signal, time, **kwargs)
        result["signal_only_f0_candidate_uncapped"] = np.full(
            np.asarray(signal).shape, np.finfo(float).tiny, dtype=float
        )
        return result

    monkeypatch.setattr(pipeline_module, "compute_signal_only_f0_candidate", overflowing_dff)
    cfg = _cfg()
    cfg.f0_min_value = 0.0
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1"))
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    with pytest.raises(CorrectionProcessingError, match="canonical dF/F"):
        pipeline._apply_standard_analysis(_chunk(), 4)


def test_pipeline_copies_caller_owned_correction_map_before_execution():
    caller_map = {
        "ROI0": PerRoiCorrectionSpec(
            roi_id="ROI0",
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
        ),
        "ROI1": PerRoiCorrectionSpec(
            roi_id="ROI1",
            strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression",
            dynamic_fit_mode="global_linear_regression",
        ),
    }
    pipeline = Pipeline(_cfg(), mode="phasic", per_roi_correction=caller_map)
    caller_map["ROI0"] = caller_map["ROI1"]
    caller_map["ROI1"] = PerRoiCorrectionSpec(
        roi_id="ROI1",
        strategy_family="signal_only_f0",
        selected_strategy="signal_only_f0",
    )
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    processed = pipeline._apply_standard_analysis(_chunk(), 4)
    consumed = processed.metadata["correction_strategy_consumed_by_roi"]
    assert consumed["ROI0"]["strategy_family"] == "signal_only_f0"
    assert consumed["ROI1"]["strategy_family"] == "dynamic_fit"


def test_legacy_pipeline_provenance_records_uniform_synthesized_fit_strategy():
    pipeline = Pipeline(_cfg(), mode="phasic")
    provenance = pipeline._build_requested_correction_provenance(["ROI0", "ROI1"])
    assert provenance["source"] == "legacy_uniform_translation"
    assert provenance["included_roi_ids"] == ["ROI0", "ROI1"]
    assert {entry["strategy_family"] for entry in provenance["requested_by_roi"]} == {
        "dynamic_fit"
    }
    assert all(
        entry["selected_strategy"] == entry["dynamic_fit_mode"]
        for entry in provenance["requested_by_roi"]
    )
    pipeline.config.dynamic_fit_mode = "global_linear_regression"
    resolved, _ = pipeline._resolve_correction_map_for_chunk(["ROI0", "ROI1"])
    assert {spec.selected_strategy for spec in resolved.values()} == {
        provenance["requested_by_roi"][0]["selected_strategy"]
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_baseline_dict",
        "missing_roi_baseline",
        "wrong_baseline_shape",
        "all_nan_baseline",
        "missing_production_qc",
        "production_unavailable",
        "missing_consumed_entry",
        "malformed_consumed_entry",
    ],
)
def test_cache_writer_fails_closed_for_incomplete_signal_only_evidence(tmp_path, mutation):
    cfg = _cfg()
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1"))
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    processed = pipeline._apply_standard_analysis(_chunk(), 4)
    metadata = copy.deepcopy(processed.metadata)
    if mutation == "missing_baseline_dict":
        metadata.pop("signal_only_f0_production_baseline")
    elif mutation == "missing_roi_baseline":
        metadata["signal_only_f0_production_baseline"].pop("ROI0")
    elif mutation == "wrong_baseline_shape":
        metadata["signal_only_f0_production_baseline"]["ROI0"] = np.zeros(5)
    elif mutation == "all_nan_baseline":
        metadata["signal_only_f0_production_baseline"]["ROI0"] = np.full(800, np.nan)
    elif mutation == "missing_production_qc":
        metadata.pop("signal_only_f0_production_qc")
    elif mutation == "production_unavailable":
        metadata["signal_only_f0_production_qc"]["ROI0"][
            "signal_only_f0_production_available"
        ] = False
    elif mutation == "missing_consumed_entry":
        metadata["correction_strategy_consumed_by_roi"] = {}
    elif mutation == "malformed_consumed_entry":
        metadata["correction_strategy_consumed_by_roi"]["ROI0"] = None
    processed.metadata = metadata

    cache_path = tmp_path / f"{mutation}.h5"
    writer = Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg)
    try:
        with pytest.raises(ValueError, match="Native Signal-Only|Native correction"):
            writer.add_chunk(processed, chunk_id=4, source_file=processed.source_file)
    finally:
        writer.abort()
    assert not cache_path.exists()


def _cache_rejects_metadata(tmp_path, processed, metadata, match):
    processed.metadata = metadata
    cache_path = tmp_path / "reject.h5"
    writer = Hdf5TraceCacheWriter(str(cache_path), "phasic", _cfg())
    try:
        with pytest.raises(ValueError, match=match):
            writer.add_chunk(processed, chunk_id=4, source_file=processed.source_file)
    finally:
        writer.abort()
    assert not cache_path.exists()


def test_cache_consumed_strategy_map_requires_exact_complete_roi_coverage(tmp_path):
    processed = _processed_for_cache(_signal_only_map("ROI0", "ROI1"))
    metadata = copy.deepcopy(processed.metadata)
    metadata["correction_strategy_consumed_by_roi"].pop("ROI0")
    metadata["signal_only_f0_production_baseline"].pop("ROI0")
    metadata["signal_only_f0_production_qc"].pop("ROI0")
    _cache_rejects_metadata(tmp_path, processed, metadata, "must cover exactly")

    processed = _processed_for_cache(_mixed_strategy_map())
    metadata = copy.deepcopy(processed.metadata)
    metadata["correction_strategy_consumed_by_roi"].pop("ROI1")
    _cache_rejects_metadata(tmp_path, processed, metadata, "must cover exactly")

    processed = _processed_for_cache(_signal_only_map("ROI0", "ROI1"))
    metadata = copy.deepcopy(processed.metadata)
    metadata["correction_strategy_consumed_by_roi"]["Unknown"] = dict(
        metadata["correction_strategy_consumed_by_roi"]["ROI0"]
    )
    metadata["correction_strategy_consumed_by_roi"]["Unknown"]["roi_id"] = "Unknown"
    _cache_rejects_metadata(tmp_path, processed, metadata, "must cover exactly")


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("unknown_family", "unknown strategy_family"),
        ("dynamic_missing_mode", "unsupported dynamic_fit_mode"),
        ("dynamic_mismatch", "selected_strategy"),
        ("signal_dynamic_mode", "must not carry dynamic_fit_mode"),
        ("orphan_baseline", "non-Signal-Only ROI"),
        ("orphan_qc", "non-Signal-Only ROI"),
    ],
)
def test_cache_validates_every_consumed_entry_and_rejects_orphan_evidence(
    tmp_path, mutation, match
):
    processed = _processed_for_cache(_mixed_strategy_map())
    metadata = copy.deepcopy(processed.metadata)
    consumed = metadata["correction_strategy_consumed_by_roi"]
    if mutation == "unknown_family":
        consumed["ROI1"]["strategy_family"] = "unknown"
    elif mutation == "dynamic_missing_mode":
        consumed["ROI1"].pop("dynamic_fit_mode")
    elif mutation == "dynamic_mismatch":
        consumed["ROI1"]["selected_strategy"] = "robust_global_event_reject"
    elif mutation == "signal_dynamic_mode":
        consumed["ROI0"]["dynamic_fit_mode"] = "global_linear_regression"
    elif mutation == "orphan_baseline":
        metadata["signal_only_f0_production_baseline"]["ROI1"] = np.ones(800)
    elif mutation == "orphan_qc":
        metadata["signal_only_f0_production_qc"]["ROI1"] = {"production_available": True}
    _cache_rejects_metadata(tmp_path, processed, metadata, match)


def test_cache_complete_mixed_map_succeeds_and_legacy_no_map_remains_compatible(tmp_path):
    processed = _processed_for_cache(_mixed_strategy_map())
    complete_path = tmp_path / "complete.h5"
    writer = Hdf5TraceCacheWriter(str(complete_path), "phasic", _cfg())
    writer.add_chunk(processed, chunk_id=4, source_file=processed.source_file)
    writer.finalize()
    assert complete_path.exists()

    legacy = _processed_for_cache(_signal_only_map("ROI0", "ROI1"))
    legacy.metadata.pop("correction_strategy_consumed_by_roi")
    legacy_path = tmp_path / "legacy.h5"
    writer = Hdf5TraceCacheWriter(str(legacy_path), "phasic", _cfg())
    writer.add_chunk(legacy, chunk_id=4, source_file=legacy.source_file)
    writer.finalize()
    assert legacy_path.exists()



@pytest.mark.parametrize("mode", ["tonic", "both"])
def test_signal_only_map_is_structurally_supported_for_tonic_capable_modes(mode):
    pipeline = Pipeline(
        _cfg(), mode=mode, per_roi_correction=_signal_only_map("ROI0")
    )
    assert pipeline.per_roi_correction["ROI0"].selected_strategy == "signal_only_f0"


@pytest.mark.parametrize("strategy_map", [_signal_only_map("ROI0", "ROI1"), _mixed_strategy_map()])
def test_native_tonic_and_phasic_consume_the_same_canonical_trace(strategy_map):
    phasic = Pipeline(_cfg(), mode="phasic", per_roi_correction=strategy_map)
    tonic = Pipeline(_cfg(), mode="tonic", per_roi_correction=strategy_map)
    for pipeline in (phasic, tonic):
        pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    phasic_chunk = phasic._apply_standard_analysis(_chunk(), 4)
    tonic_chunk = tonic._apply_standard_analysis(_chunk(), 4)
    np.testing.assert_allclose(tonic_chunk.delta_f, phasic_chunk.delta_f, equal_nan=True)
    np.testing.assert_allclose(tonic_chunk.dff, phasic_chunk.dff, equal_nan=True)
    assert tonic_chunk.metadata["correction_strategy_consumed_by_roi"] == (
        phasic_chunk.metadata["correction_strategy_consumed_by_roi"]
    )


def _write_rwd(path: Path, *, n=200, fs=10.0):
    t = np.arange(n, dtype=float) / fs
    uv = 2.0 + 0.05 * np.sin(0.2 * t)
    sig = 5.0 + 0.2 * np.sin(0.3 * t) + 0.03 * np.cos(1.1 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"TimeStamp": t, "Region0-410": uv, "Region0-470": sig}
    ).to_csv(path, index=False)


def _write_rwd_four_rois(path: Path, *, n=200, fs=10.0):
    t = np.arange(n, dtype=float) / fs
    values = {"TimeStamp": t}
    for roi_index in range(4):
        uv = 2.0 + 0.04 * np.sin((0.12 + 0.02 * roi_index) * t + roi_index)
        sig = (
            5.0
            + roi_index
            + (0.22 + 0.03 * roi_index) * np.cos((0.20 + 0.01 * roi_index) * t)
            + 0.02 * np.sin(1.1 * t + roi_index)
        )
        values[f"Region{roi_index}-410"] = uv
        values[f"Region{roi_index}-470"] = sig
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(path, index=False)


@pytest.mark.parametrize("mode", ["phasic", "tonic"])
def test_real_pipeline_run_consumes_native_signal_only_map_and_cache(tmp_path, mode):
    source = tmp_path / "input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd(source)
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
    )
    out = tmp_path / "out"
    pipeline = Pipeline(
        cfg,
        mode=mode,
        per_roi_correction=_signal_only_map("Region0"),
    )
    pipeline.run(str(source.parent.parent), str(out), force_format="rwd", recursive=True)
    run_meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    provenance = run_meta["correction_provenance"]
    assert provenance["schema_version"] == "correction_provenance.v1"
    assert provenance["source"] == "explicit_per_roi_map"
    assert provenance["included_roi_ids"] == ["Region0"]
    assert provenance["requested_by_roi"] == [
        {
            "roi_id": "Region0",
            "strategy_family": "signal_only_f0",
            "selected_strategy": "signal_only_f0",
            "dynamic_fit_mode": None,
            "parameter_identity": "",
            "evidence_identity": "",
        }
    ]
    report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    assert report["derived_settings"]["correction_provenance"] == provenance
    assert provenance["analysis_mode"] == mode
    with h5py.File(out / f"{mode}_trace_cache.h5", "r") as handle:
        group = handle["roi/Region0/chunk_0"]
        assert group.attrs["correction_selected_strategy"] == "signal_only_f0"
        assert "signal_only_f0_baseline" in group
        assert np.isfinite(group["dff"][()]).mean() > 0.99


def test_real_pipeline_run_mixed_four_strategy_map_preserves_roi_identity(tmp_path):
    source = tmp_path / "input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd_four_rois(source)
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        window_sec=10.0,
        min_samples_per_window=10,
        signal_only_f0_min_window_samples=21,
    )
    strategies = {
        "Region0": PerRoiCorrectionSpec(
            roi_id="Region0",
            strategy_family="dynamic_fit",
            selected_strategy="robust_global_event_reject",
            dynamic_fit_mode="robust_global_event_reject",
        ),
        "Region1": PerRoiCorrectionSpec(
            roi_id="Region1",
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
        ),
        "Region2": PerRoiCorrectionSpec(
            roi_id="Region2",
            strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression",
            dynamic_fit_mode="global_linear_regression",
        ),
        "Region3": PerRoiCorrectionSpec(
            roi_id="Region3",
            strategy_family="dynamic_fit",
            selected_strategy="adaptive_event_gated_regression",
            dynamic_fit_mode="adaptive_event_gated_regression",
        ),
    }
    out = tmp_path / "out"
    pipeline = Pipeline(cfg, mode="phasic", per_roi_correction=strategies)
    pipeline.run(str(source.parent.parent), str(out), force_format="rwd", recursive=True)
    with h5py.File(out / "phasic_trace_cache.h5", "r") as handle:
        for roi in ("Region0", "Region1", "Region2", "Region3"):
            group = handle[f"roi/{roi}/chunk_0"]
            assert group.attrs["correction_selected_strategy"] == strategies[roi].selected_strategy
            assert np.isfinite(group["dff"][()]).mean() > 0.90
        assert np.isnan(handle["roi/Region1/chunk_0/fit_ref"][()]).all()
        assert np.isfinite(handle["roi/Region0/chunk_0/fit_ref"][()]).mean() > 0.90
        assert np.isfinite(handle["roi/Region2/chunk_0/fit_ref"][()]).mean() > 0.90
        assert np.isfinite(handle["roi/Region3/chunk_0/fit_ref"][()]).mean() > 0.90


def test_real_pipeline_feature_extraction_receives_canonical_selected_traces(
    monkeypatch, tmp_path
):
    source = tmp_path / "input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd(source)
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
    )
    captured_chunks = []
    captured_features = []
    original_extract = pipeline_module.feature_extraction.extract_features

    def capture_extract(chunk, config, per_roi_config=None):
        captured_chunks.append(chunk)
        features = original_extract(chunk, config, per_roi_config=per_roi_config)
        captured_features.append(features.copy())
        return features

    monkeypatch.setattr(
        pipeline_module.feature_extraction,
        "extract_features",
        capture_extract,
    )
    out = tmp_path / "out"
    pipeline = Pipeline(
        cfg,
        mode="phasic",
        per_roi_correction=_signal_only_map("Region0"),
    )
    pipeline.run(str(source.parent.parent), str(out), force_format="rwd", recursive=True)

    assert len(captured_chunks) == 1
    chunk = captured_chunks[0]
    baseline = chunk.metadata["signal_only_f0_production_baseline"]["Region0"]
    expected_dff = 100.0 * (chunk.sig_raw[:, 0] - baseline) / baseline
    np.testing.assert_allclose(chunk.dff[:, 0], expected_dff, rtol=0.0, atol=0.0)
    assert np.isfinite(chunk.dff[:, 0]).mean() > 0.99
    assert np.isnan(chunk.uv_fit[:, 0]).all()
    assert set(captured_features[0]["roi"].astype(str)) == {"Region0"}

    mixed_source = tmp_path / "mixed_input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    _write_rwd_four_rois(mixed_source)
    mixed_cfg = replace(cfg)
    mixed_map = {
        "Region0": PerRoiCorrectionSpec(
            roi_id="Region0",
            strategy_family="dynamic_fit",
            selected_strategy="robust_global_event_reject",
            dynamic_fit_mode="robust_global_event_reject",
        ),
        "Region1": PerRoiCorrectionSpec(
            roi_id="Region1",
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
        ),
        "Region2": PerRoiCorrectionSpec(
            roi_id="Region2",
            strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression",
            dynamic_fit_mode="global_linear_regression",
        ),
        "Region3": PerRoiCorrectionSpec(
            roi_id="Region3",
            strategy_family="dynamic_fit",
            selected_strategy="adaptive_event_gated_regression",
            dynamic_fit_mode="adaptive_event_gated_regression",
        ),
    }
    mixed_capture = []
    mixed_features = []

    def capture_mixed(chunk, config, per_roi_config=None):
        mixed_capture.append(chunk)
        features = original_extract(chunk, config, per_roi_config=per_roi_config)
        mixed_features.append(features.copy())
        return features

    monkeypatch.setattr(
        pipeline_module.feature_extraction,
        "extract_features",
        capture_mixed,
    )
    mixed_out = tmp_path / "mixed_out"
    mixed_pipeline = Pipeline(
        mixed_cfg, mode="phasic", per_roi_correction=mixed_map
    )
    mixed_pipeline.run(
        str(mixed_source.parent.parent),
        str(mixed_out),
        force_format="rwd",
        recursive=True,
    )
    assert len(mixed_capture) == 1
    mixed_chunk = mixed_capture[0]
    assert mixed_chunk.channel_names == ["Region0", "Region1", "Region2", "Region3"]
    assert np.isnan(mixed_chunk.uv_fit[:, 1]).all()
    assert np.isfinite(mixed_chunk.uv_fit[:, 0]).mean() > 0.90
    assert np.isfinite(mixed_chunk.uv_fit[:, 2]).mean() > 0.90
    assert np.isfinite(mixed_chunk.uv_fit[:, 3]).mean() > 0.90
    mixed_baseline = mixed_chunk.metadata["signal_only_f0_production_baseline"]["Region1"]
    np.testing.assert_allclose(
        mixed_chunk.dff[:, 1],
        100.0 * (mixed_chunk.sig_raw[:, 1] - mixed_baseline) / mixed_baseline,
        rtol=0.0,
        atol=0.0,
    )
    assert set(mixed_features[0]["roi"].astype(str)) == set(mixed_chunk.channel_names)
    assert all(
        len(mixed_features[0].loc[mixed_features[0]["roi"].astype(str) == roi]) == 1
        for roi in mixed_chunk.channel_names
    )
def test_real_pipeline_unusable_signal_preserves_typed_failure_and_no_success_outputs(
    tmp_path,
):
    source = tmp_path / "input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    n = 200
    fs = 10.0
    t = np.arange(n, dtype=float) / fs
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "TimeStamp": t,
            "Region0-410": np.full(n, 2.0),
            "Region0-470": np.full(n, 5.0),
        }
    ).to_csv(source, index=False)
    cfg = Config(
        target_fs_hz=fs,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
    )
    out = tmp_path / "failed_out"
    pipeline = Pipeline(
        cfg, mode="phasic", per_roi_correction=_signal_only_map("Region0")
    )
    with pytest.raises(Exception) as exc_info:
        pipeline.run(str(source.parent.parent), str(out), force_format="rwd", recursive=True)

    chain = []
    current = exc_info.value
    while current is not None and len(chain) < 8:
        chain.append(current)
        current = current.__cause__
    typed = next((exc for exc in chain if isinstance(exc, CorrectionProcessingError)), None)
    assert typed is not None
    assert typed.roi_id == "Region0"
    assert typed.selected_strategy == "signal_only_f0"
    assert "signal" in typed.reason.lower() or "candidate" in typed.reason.lower()
    assert not (out / "phasic_trace_cache.h5").exists()
    assert not (out / "phasic_trace_cache.h5.tmp").exists()
    assert not (out / "input_processing_completeness.json").exists()


# ---------------------------------------------------------------------------
# Region3/chunk-22 real-dataset failure shape: a near-flat, quantized raw
# signal whose robust range (p95-p05) sits just either side of
# signal_only_f0_min_robust_range. The real failing recording
# (photometryData2024-09-28T02_04_50.csv, canonical ROI Region3, physical
# column Region3G) has raw signal-channel values clustered almost entirely
# at exactly 1/255 (0.00392156862745...), consistent with an unconnected/
# dead fiber channel reading ADC quantization floor noise rather than real
# photometry signal -- resampled robust range ~=9.92e-7, just under the
# 1e-6 threshold. Two neighboring real chunks from the same ROI measured
# 1.008e-6 and 9.93e-7 respectively: the pass/fail split for this ROI is
# noise-level, not a one-off glitch specific to chunk 22.
# ---------------------------------------------------------------------------


def _near_flat_signal(n, span, *, base=0.00392156862745098):
    """A synthetic near-flat ramp calibrated to straddle
    signal_only_f0_min_robust_range, mirroring the real quantized-flat
    shape's order of magnitude without depending on real research data."""
    return base + np.linspace(0.0, span, n)


def test_signal_only_f0_refuses_near_flat_quantized_signal_matching_real_shape():
    """Reproduces the exact real Region3/chunk-22 failure shape and reason:
    a near-flat raw signal whose p95-p05 robust range falls just under the
    production threshold raises CorrectionProcessingError with reason
    "insufficient_robust_signal_range", exactly as the real dataset did."""
    n = 800
    signal = _near_flat_signal(n, span=1.05e-6)
    p05, p95 = np.percentile(signal, [5.0, 95.0])
    assert p95 - p05 < 1e-6, "fixture must land on the failing side of the threshold"

    chunk = _chunk(n=n)
    chunk.sig_raw[:, 0] = signal
    pipeline = Pipeline(
        _cfg(), mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1")
    )
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    with pytest.raises(CorrectionProcessingError) as exc_info:
        pipeline._apply_standard_analysis(chunk, chunk_id=22)
    exc = exc_info.value
    assert exc.roi_id == "ROI0"
    assert exc.chunk_id == 22
    assert exc.selected_strategy == "signal_only_f0"
    assert "insufficient_robust_signal_range" in exc.reason


def test_signal_only_f0_accepts_signal_just_above_robust_range_threshold():
    """The true boundary: a signal with the same near-flat magnitude but a
    slightly larger span (robust range just over the threshold) must not be
    refused for insufficient_robust_signal_range -- isolating exactly how
    close the real dataset's chunks sit to this pass/fail line."""
    n = 800
    signal = _near_flat_signal(n, span=1.30e-6)
    p05, p95 = np.percentile(signal, [5.0, 95.0])
    assert p95 - p05 >= 1e-6, "fixture must land on the passing side of the threshold"

    chunk = _chunk(n=n)
    chunk.sig_raw[:, 0] = signal
    pipeline = Pipeline(
        _cfg(), mode="phasic", per_roi_correction=_signal_only_map("ROI0", "ROI1")
    )
    pipeline.stats.f0_values = {"ROI0": 1.0, "ROI1": 1.0}
    # Must not raise for the robust-range reason specifically; other
    # coverage/positivity checks downstream are not the boundary under test.
    try:
        pipeline._apply_standard_analysis(chunk, chunk_id=22)
    except CorrectionProcessingError as exc:
        assert "insufficient_robust_signal_range" not in exc.reason
