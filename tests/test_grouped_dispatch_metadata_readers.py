"""Phase 1.2 correction item 2: prove grouped dispatch's metadata survives
through the REAL consumers -- Pipeline QC recording, slope summaries, and
HDF5 cache attributes -- for a genuinely mixed-mode chunk, not just inspected
from regression.py's own return values."""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing, regression
from photometry_pipeline.core.types import Chunk, PerRoiCorrectionSpec
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.pipeline import Pipeline


def _make_chunk(uv_raw, sig_raw, channel_names, fs_hz=40.0) -> Chunk:
    n = int(uv_raw.shape[0])
    return Chunk(
        chunk_id=0, source_file="synth.csv", format="npm",
        time_sec=np.arange(n, dtype=float) / float(fs_hz),
        uv_raw=np.asarray(uv_raw, dtype=float), sig_raw=np.asarray(sig_raw, dtype=float),
        fs_hz=float(fs_hz), channel_names=list(channel_names), metadata={},
    )


def _synth_roi(rng, n, fs, *, slope=1.4, seed_offset=0.0):
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t + seed_offset) + 0.03 * rng.standard_normal(n)
    sig = slope * uv + 2.0 + 0.1 * np.sin(2.0 * np.pi * 0.8 * t + 0.5 + seed_offset)
    return uv, sig


def _mixed_two_roi_chunk_and_config(rng):
    n, fs = 3200, 40.0
    roi_names = ["GlobalLinROI", "RobustROI"]
    uv0, sig0 = _synth_roi(rng, n, fs, slope=1.2, seed_offset=0.0)
    uv1, sig1 = _synth_roi(rng, n, fs, slope=1.8, seed_offset=0.9)
    uv_raw = np.stack([uv0, uv1], axis=1)
    sig_raw = np.stack([sig0, sig1], axis=1)
    chunk = _make_chunk(uv_raw, sig_raw, roi_names, fs)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
    strategy_map = {
        "GlobalLinROI": PerRoiCorrectionSpec(
            roi_id="GlobalLinROI", strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression", dynamic_fit_mode="global_linear_regression",
        ),
        "RobustROI": PerRoiCorrectionSpec(
            roi_id="RobustROI", strategy_family="dynamic_fit",
            selected_strategy="robust_global_event_reject", dynamic_fit_mode="robust_global_event_reject",
        ),
    }
    return chunk, cfg, roi_names, strategy_map


def test_pipeline_qc_and_slope_recording_covers_both_modes_in_a_mixed_chunk():
    """Before the correction item-3 fix, _record_dynamic_fit_slope_summaries
    picked exactly one per-ROI dict based on a single chunk-wide fit_mode,
    silently dropping every other mode's ROI from the recorded output. Both
    ROIs, under their own distinct modes, must be recorded here."""
    rng = np.random.default_rng(13)
    chunk, cfg, roi_names, strategy_map = _mixed_two_roi_chunk_and_config(rng)

    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f

    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_dynamic_fit_validity_metrics(chunk, chunk_id=0, source_file="synth.csv")
    pipeline._record_dynamic_fit_slope_summaries(chunk, chunk_id=0, source_file="synth.csv")

    qc_rois = {rec["roi"] for rec in pipeline.dynamic_fit_qc_records}
    assert qc_rois == set(roi_names), "both ROIs' QC validity records must be present, not just one mode's"
    qc_modes = {rec["roi"]: rec["dynamic_fit_mode"] for rec in pipeline.dynamic_fit_qc_records}
    assert qc_modes["GlobalLinROI"] == "global_linear_regression"
    assert qc_modes["RobustROI"] == "robust_global_event_reject"

    slope_rois = {rec["roi"] for rec in pipeline.dynamic_fit_slope_records}
    assert slope_rois == set(roi_names), "both ROIs' slope summaries must be present, not just one mode's"
    slope_modes = {rec["roi"]: rec["dynamic_fit_mode"] for rec in pipeline.dynamic_fit_slope_records}
    assert slope_modes["GlobalLinROI"] == "global_linear_regression"
    assert slope_modes["RobustROI"] == "robust_global_event_reject"


def test_hdf5_cache_attrs_are_correct_per_roi_for_a_mixed_chunk(tmp_path):
    """io/hdf5_cache.py's add_chunk must write EACH ROI's own resolved mode
    and engine attrs, not one chunk-wide value copied onto every ROI group."""
    rng = np.random.default_rng(17)
    chunk, cfg, roi_names, strategy_map = _mixed_two_roi_chunk_and_config(rng)

    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    chunk.dff = np.zeros_like(delta_f)

    cache_path = str(tmp_path / "phasic_trace_cache.h5")
    writer = Hdf5TraceCacheWriter(cache_path, "phasic", cfg)
    writer.add_chunk(chunk, chunk_id=0, source_file="synth.csv")
    writer.finalize()

    with h5py.File(cache_path, "r") as f:
        roi_group = f["roi"]
        global_lin_attrs = dict(roi_group["GlobalLinROI"]["chunk_0"].attrs)
        robust_attrs = dict(roi_group["RobustROI"]["chunk_0"].attrs)

    assert global_lin_attrs["dynamic_fit_mode_resolved"] == "global_linear_regression"
    assert robust_attrs["dynamic_fit_mode_resolved"] == "robust_global_event_reject"
    assert global_lin_attrs["dynamic_fit_engine"] == "global_linear_ols_v1"
    assert robust_attrs["dynamic_fit_engine"] == "robust_global_event_reject_v1"


def test_hdf5_cache_attrs_correct_for_homogeneous_chunk_unchanged():
    """Sanity guard: a single-mode chunk's HDF5 attrs are unaffected by the
    per-ROI fix -- both ROIs share the one correct mode/engine, exactly as
    before grouped dispatch existed."""
    import tempfile
    import os

    rng = np.random.default_rng(19)
    n, fs = 1600, 40.0
    roi_names = ["A", "B"]
    uv0, sig0 = _synth_roi(rng, n, fs, slope=1.2)
    uv1, sig1 = _synth_roi(rng, n, fs, slope=1.5, seed_offset=0.4)
    chunk = _make_chunk(np.stack([uv0, uv1], axis=1), np.stack([sig0, sig1], axis=1), roi_names, fs)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2,
                 dynamic_fit_mode="adaptive_event_gated_regression")
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    chunk.dff = np.zeros_like(delta_f)

    with tempfile.TemporaryDirectory() as d:
        cache_path = os.path.join(d, "phasic_trace_cache.h5")
        writer = Hdf5TraceCacheWriter(cache_path, "phasic", cfg)
        writer.add_chunk(chunk, chunk_id=0, source_file="synth.csv")
        writer.finalize()
        with h5py.File(cache_path, "r") as f:
            a_attrs = dict(f["roi"]["A"]["chunk_0"].attrs)
            b_attrs = dict(f["roi"]["B"]["chunk_0"].attrs)

    assert a_attrs["dynamic_fit_mode_resolved"] == "adaptive_event_gated_regression"
    assert b_attrs["dynamic_fit_mode_resolved"] == "adaptive_event_gated_regression"
    assert a_attrs["dynamic_fit_engine"] == b_attrs["dynamic_fit_engine"] == "adaptive_event_gated_regression_v1"


# ---------------------------------------------------------------------------
# Correction (final foundation pass): one dynamic-fit ROI must never leak its
# mode/engine onto co-chunked non-dynamic-fit (signal_only_f0 placeholder)
# ROIs, including the specific case that slipped through before -- exactly
# ONE dynamic-fit group, where the flat chunk-wide fallback is a real,
# concrete mode string rather than "mixed" or "none".
# ---------------------------------------------------------------------------


def _one_dynamic_two_placeholder_chunk_and_config(rng):
    n, fs = 3200, 40.0
    roi_names = ["DynamicROI", "PlaceholderA", "PlaceholderB"]
    uv0, sig0 = _synth_roi(rng, n, fs, slope=1.3, seed_offset=0.0)
    uv1, sig1 = _synth_roi(rng, n, fs, slope=1.0, seed_offset=0.5)
    uv2, sig2 = _synth_roi(rng, n, fs, slope=1.0, seed_offset=1.0)
    uv_raw = np.stack([uv0, uv1, uv2], axis=1)
    sig_raw = np.stack([sig0, sig1, sig2], axis=1)
    chunk = _make_chunk(uv_raw, sig_raw, roi_names, fs)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
    strategy_map = {
        "DynamicROI": PerRoiCorrectionSpec(
            roi_id="DynamicROI", strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression", dynamic_fit_mode="global_linear_regression",
        ),
        "PlaceholderA": PerRoiCorrectionSpec(
            roi_id="PlaceholderA", strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0", dynamic_fit_mode=None,
        ),
        "PlaceholderB": PerRoiCorrectionSpec(
            roi_id="PlaceholderB", strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0", dynamic_fit_mode=None,
        ),
    }
    return chunk, cfg, roi_names, strategy_map


def test_hdf5_attrs_do_not_leak_single_dynamic_group_onto_placeholder_rois(tmp_path):
    """The bug this test targets: with exactly one dynamic-fit group, the
    flat dynamic_fit_mode_resolved is a real mode string (not "mixed"), so a
    naive per-ROI-key-truthiness fallback would wrongly attribute it to the
    signal_only_f0 placeholder ROIs too."""
    rng = np.random.default_rng(61)
    chunk, cfg, roi_names, strategy_map = _one_dynamic_two_placeholder_chunk_and_config(rng)

    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    chunk.dff = np.zeros_like(delta_f)
    assert chunk.metadata["dynamic_fit_mode_resolved"] == "global_linear_regression"  # the trap

    cache_path = str(tmp_path / "phasic_trace_cache.h5")
    writer = Hdf5TraceCacheWriter(cache_path, "phasic", cfg)
    writer.add_chunk(chunk, chunk_id=0, source_file="synth.csv")
    writer.finalize()

    with h5py.File(cache_path, "r") as f:
        dynamic_attrs = dict(f["roi"]["DynamicROI"]["chunk_0"].attrs)
        placeholder_a_attrs = dict(f["roi"]["PlaceholderA"]["chunk_0"].attrs)
        placeholder_b_attrs = dict(f["roi"]["PlaceholderB"]["chunk_0"].attrs)

    assert dynamic_attrs["dynamic_fit_mode_resolved"] == "global_linear_regression"
    assert dynamic_attrs["dynamic_fit_engine"] == "global_linear_ols_v1"

    for placeholder_attrs in (placeholder_a_attrs, placeholder_b_attrs):
        assert "dynamic_fit_mode_resolved" not in placeholder_attrs
        assert "dynamic_fit_engine" not in placeholder_attrs


def test_pipeline_records_do_not_leak_single_dynamic_group_onto_placeholder_rois():
    """Same trap, for Pipeline's validity/slope-summary record accumulation."""
    rng = np.random.default_rng(63)
    chunk, cfg, roi_names, strategy_map = _one_dynamic_two_placeholder_chunk_and_config(rng)

    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f

    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_dynamic_fit_validity_metrics(chunk, chunk_id=0, source_file="synth.csv")
    pipeline._record_dynamic_fit_slope_summaries(chunk, chunk_id=0, source_file="synth.csv")

    qc_rois = {rec["roi"] for rec in pipeline.dynamic_fit_qc_records}
    assert qc_rois == {"DynamicROI"}, "no validity record for either placeholder ROI"

    slope_rois = {rec["roi"] for rec in pipeline.dynamic_fit_slope_records}
    assert slope_rois == {"DynamicROI"}, "no slope-summary record for either placeholder ROI"

    for rec in pipeline.dynamic_fit_qc_records + pipeline.dynamic_fit_slope_records:
        if rec["roi"] != "DynamicROI":
            assert rec["dynamic_fit_mode"] != "global_linear_regression"


def test_baseline_reference_candidate_records_do_not_mislabel_placeholder_rois():
    """The third reader (_record_baseline_reference_candidate_metrics) keeps
    a record for every ROI (it always computes the Signal-Only F0 diagnostic
    candidate), but the record's dynamic_fit_mode label must be honest: ""
    for a placeholder ROI, never the dynamic ROI's real mode."""
    rng = np.random.default_rng(65)
    chunk, cfg, roi_names, strategy_map = _one_dynamic_two_placeholder_chunk_and_config(rng)

    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f

    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_dynamic_fit_validity_metrics(chunk, chunk_id=0, source_file="synth.csv")
    pipeline._record_baseline_reference_candidate_metrics(chunk, chunk_id=0, source_file="synth.csv")

    records_by_roi = {rec["roi"]: rec for rec in pipeline.baseline_reference_candidate_records}
    assert set(records_by_roi) == set(roi_names), "every ROI still gets a baseline/Signal-Only-F0 diagnostic record"
    assert records_by_roi["DynamicROI"]["dynamic_fit_mode"] == "global_linear_regression"
    assert records_by_roi["PlaceholderA"]["dynamic_fit_mode"] == ""
    assert records_by_roi["PlaceholderB"]["dynamic_fit_mode"] == ""


def test_all_non_dynamic_chunk_present_empty_map_prevents_flat_fallback(tmp_path):
    """Every ROI is signal_only_f0. dynamic_fit_mode_resolved_by_roi is
    present but empty -- that must still block the chunk-wide fallback
    (which would otherwise be the "none" sentinel key.get(roi, "none") could
    wrongly borrow if truthiness, not presence, were used)."""
    rng = np.random.default_rng(67)
    n, fs = 1600, 40.0
    roi_names = ["OnlyA", "OnlyB"]
    uv0, sig0 = _synth_roi(rng, n, fs, slope=1.1)
    uv1, sig1 = _synth_roi(rng, n, fs, slope=1.4, seed_offset=0.6)
    chunk = _make_chunk(np.stack([uv0, uv1], axis=1), np.stack([sig0, sig1], axis=1), roi_names, fs)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
    strategy_map = {
        roi: PerRoiCorrectionSpec(
            roi_id=roi, strategy_family="signal_only_f0", selected_strategy="signal_only_f0",
            dynamic_fit_mode=None,
        )
        for roi in roi_names
    }
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    chunk.dff = np.zeros_like(delta_f)

    assert chunk.metadata["dynamic_fit_mode_resolved_by_roi"] == {}
    assert chunk.metadata["dynamic_fit_mode_resolved"] == "none"  # the trap: a real, non-empty string

    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_dynamic_fit_validity_metrics(chunk, chunk_id=0, source_file="synth.csv")
    pipeline._record_dynamic_fit_slope_summaries(chunk, chunk_id=0, source_file="synth.csv")
    assert pipeline.dynamic_fit_qc_records == []
    assert pipeline.dynamic_fit_slope_records == []

    cache_path = str(tmp_path / "phasic_trace_cache.h5")
    writer = Hdf5TraceCacheWriter(cache_path, "phasic", cfg)
    writer.add_chunk(chunk, chunk_id=0, source_file="synth.csv")
    writer.finalize()
    with h5py.File(cache_path, "r") as f:
        for roi in roi_names:
            attrs = dict(f["roi"][roi]["chunk_0"].attrs)
            assert "dynamic_fit_mode_resolved" not in attrs
            assert "dynamic_fit_engine" not in attrs


def test_legacy_chunk_with_absent_per_roi_key_still_uses_chunk_wide_fallback():
    """A hand-constructed chunk that never went through fit_chunk_dynamic at
    all (dynamic_fit_mode_resolved_by_roi is genuinely ABSENT, not present-
    and-empty) must still fall back to the flat chunk-wide value for every
    ROI -- the legacy/pre-grouping compatibility path correction item 1
    explicitly requires to be retained."""
    n, fs = 800, 40.0
    t = np.arange(n, dtype=float) / fs
    roi_names = ["LegacyA", "LegacyB"]
    uv = np.ones((n, 2))
    sig = np.ones((n, 2)) * 2.0
    chunk = Chunk(
        chunk_id=0, source_file="legacy.csv", format="npm", time_sec=t,
        uv_raw=uv, sig_raw=sig, fs_hz=fs, channel_names=roi_names,
        uv_fit=np.ones((n, 2)), delta_f=np.ones((n, 2)),
        metadata={
            "dynamic_fit_mode_resolved": "robust_global_event_reject",
            "dynamic_fit_event_reject": {
                "LegacyA": {"slope_summary": {"warning_level": "none", "slope_min": 1.0}},
                "LegacyB": {"slope_summary": {"warning_level": "none", "slope_min": 1.0}},
            },
        },
    )
    assert "dynamic_fit_mode_resolved_by_roi" not in chunk.metadata  # the precondition this test targets

    cfg = Config()
    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_dynamic_fit_slope_summaries(chunk, chunk_id=0, source_file="legacy.csv")

    slope_rois = {rec["roi"]: rec["dynamic_fit_mode"] for rec in pipeline.dynamic_fit_slope_records}
    assert slope_rois == {
        "LegacyA": "robust_global_event_reject",
        "LegacyB": "robust_global_event_reject",
    }


# ---------------------------------------------------------------------------
# Correction (malformed-contract pass): a PRESENT but non-dict
# dynamic_fit_mode_resolved_by_roi is malformed CURRENT metadata, not legacy
# data, and must never silently fall back to the flat dynamic_fit_mode_
# resolved value -- that would mislabel every ROI with a single borrowed
# mode string. Covers both plausible malformed forms: a list and None.
# ---------------------------------------------------------------------------


def _two_roi_chunk_with_malformed_per_roi_key(rng, malformed_value):
    """Two ROIs, real dynamic-fit inputs, but dynamic_fit_mode_resolved_by_roi
    deliberately overwritten to a malformed value AFTER fit_chunk_dynamic ran
    -- simulating corrupted/inconsistent current metadata, not legacy data."""
    n, fs = 1600, 40.0
    roi_names = ["RoiX", "RoiY"]
    uv0, sig0 = _synth_roi(rng, n, fs, slope=1.2, seed_offset=0.0)
    uv1, sig1 = _synth_roi(rng, n, fs, slope=1.6, seed_offset=0.5)
    chunk = _make_chunk(np.stack([uv0, uv1], axis=1), np.stack([sig0, sig1], axis=1), roi_names, fs)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2,
                 dynamic_fit_mode="global_linear_regression")
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")
    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    chunk.dff = np.zeros_like(delta_f)

    assert chunk.metadata["dynamic_fit_mode_resolved"] == "global_linear_regression"  # the trap
    assert chunk.metadata["dynamic_fit_mode_resolved_by_roi"] == {
        "RoiX": "global_linear_regression", "RoiY": "global_linear_regression",
    }
    # Corrupt the per-ROI contract in place, as if something else overwrote it.
    chunk.metadata["dynamic_fit_mode_resolved_by_roi"] = malformed_value
    return chunk, cfg, roi_names


@pytest.mark.parametrize("malformed_value", [["not", "a", "dict"], None])
def test_hdf5_cache_raises_on_malformed_present_per_roi_map(tmp_path, malformed_value):
    rng = np.random.default_rng(81)
    chunk, cfg, roi_names = _two_roi_chunk_with_malformed_per_roi_key(rng, malformed_value)

    cache_path = str(tmp_path / "phasic_trace_cache.h5")
    writer = Hdf5TraceCacheWriter(cache_path, "phasic", cfg)
    with pytest.raises(ValueError, match="not a"):
        writer.add_chunk(chunk, chunk_id=0, source_file="synth.csv")
    writer.abort()
    # Nothing must have been written for this chunk -- no ROI silently
    # labeled global_linear_regression despite the raise.
    assert not os.path.exists(cache_path)


@pytest.mark.parametrize("malformed_value", [["not", "a", "dict"], None])
def test_pipeline_validity_metrics_raises_on_malformed_present_per_roi_map(malformed_value):
    rng = np.random.default_rng(83)
    chunk, cfg, roi_names = _two_roi_chunk_with_malformed_per_roi_key(rng, malformed_value)

    pipeline = Pipeline(cfg, mode="phasic")
    with pytest.raises(RuntimeError, match="not a"):
        pipeline._record_dynamic_fit_validity_metrics(chunk, chunk_id=0, source_file="synth.csv")
    # Fail closed means nothing partial was recorded either.
    assert pipeline.dynamic_fit_qc_records == []


@pytest.mark.parametrize("malformed_value", [["not", "a", "dict"], None])
def test_pipeline_slope_summaries_raises_on_malformed_present_per_roi_map(malformed_value):
    rng = np.random.default_rng(85)
    chunk, cfg, roi_names = _two_roi_chunk_with_malformed_per_roi_key(rng, malformed_value)

    pipeline = Pipeline(cfg, mode="phasic")
    with pytest.raises(RuntimeError, match="not a"):
        pipeline._record_dynamic_fit_slope_summaries(chunk, chunk_id=0, source_file="synth.csv")
    assert pipeline.dynamic_fit_slope_records == []


@pytest.mark.parametrize("malformed_value", [["not", "a", "dict"], None])
def test_baseline_reference_candidate_metrics_fails_closed_explicitly_on_malformed_map(malformed_value):
    """Chosen production policy for this one reader: do not raise (it also
    carries the always-on Signal-Only F0 diagnostic candidate, unaffected by
    dynamic-fit metadata), but never fall back to the flat mode either --
    leave the label empty and record an explicit, inspectable contract
    error."""
    rng = np.random.default_rng(87)
    chunk, cfg, roi_names = _two_roi_chunk_with_malformed_per_roi_key(rng, malformed_value)

    pipeline = Pipeline(cfg, mode="phasic")
    # Must not raise.
    pipeline._record_baseline_reference_candidate_metrics(chunk, chunk_id=0, source_file="synth.csv")

    records_by_roi = {rec["roi"]: rec for rec in pipeline.baseline_reference_candidate_records}
    assert set(records_by_roi) == set(roi_names), "the diagnostic record is still produced for every ROI"
    for roi in roi_names:
        rec = records_by_roi[roi]
        assert rec["dynamic_fit_mode"] == "", f"{roi} must not be mislabeled global_linear_regression"
        assert rec["dynamic_fit_mode_contract_error"] is not None
        assert "not a" in rec["dynamic_fit_mode_contract_error"]


def test_baseline_reference_candidate_metrics_no_contract_error_for_genuinely_absent_key():
    """Sanity guard: genuine key absence (legacy chunk) must NOT populate
    dynamic_fit_mode_contract_error -- only a present-but-malformed value
    does."""
    rng = np.random.default_rng(89)
    n, fs = 1600, 40.0
    roi_names = ["LegacyX", "LegacyY"]
    uv0, sig0 = _synth_roi(rng, n, fs, slope=1.1)
    uv1, sig1 = _synth_roi(rng, n, fs, slope=1.3, seed_offset=0.4)
    chunk = _make_chunk(np.stack([uv0, uv1], axis=1), np.stack([sig0, sig1], axis=1), roi_names, fs)
    chunk.uv_fit = np.ones((n, 2))
    chunk.delta_f = np.ones((n, 2))
    chunk.metadata["dynamic_fit_mode_resolved"] = "adaptive_event_gated_regression"
    assert "dynamic_fit_mode_resolved_by_roi" not in chunk.metadata

    cfg = Config()
    pipeline = Pipeline(cfg, mode="phasic")
    pipeline._record_baseline_reference_candidate_metrics(chunk, chunk_id=0, source_file="legacy.csv")

    records_by_roi = {rec["roi"]: rec for rec in pipeline.baseline_reference_candidate_records}
    for roi in roi_names:
        assert records_by_roi[roi]["dynamic_fit_mode"] == "adaptive_event_gated_regression"
        assert records_by_roi[roi]["dynamic_fit_mode_contract_error"] is None
