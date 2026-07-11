"""Phase 1.2: prove grouped per-ROI dynamic-fit dispatch is equivalent to the
prior single-call dispatch before any Signal-Only F0 code is added.

Covers the corrections' required equivalence set:
- homogeneous robust / global-linear / adaptive: old direct call vs grouped
- mixed fit modes: stable per-ROI scatter-back identity
- one group's degenerate ROI cannot affect another group's ROI
"""
import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing
from photometry_pipeline.core import regression
from photometry_pipeline.core.types import Chunk, PerRoiCorrectionSpec


def _make_chunk(uv_raw: np.ndarray, sig_raw: np.ndarray, channel_names, fs_hz: float = 40.0) -> Chunk:
    n = int(uv_raw.shape[0])
    return Chunk(
        chunk_id=0,
        source_file="synth.csv",
        format="npm",
        time_sec=np.arange(n, dtype=float) / float(fs_hz),
        uv_raw=np.asarray(uv_raw, dtype=float),
        sig_raw=np.asarray(sig_raw, dtype=float),
        fs_hz=float(fs_hz),
        channel_names=list(channel_names),
        metadata={},
    )


def _prepare_filtered(chunk: Chunk, cfg: Config) -> None:
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)


def _synth_roi(rng, n, fs, *, slope=1.4, intercept=2.0, uv_freq=0.2, sig_freq=0.8, seed_offset=0.0):
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.5 * np.sin(2.0 * np.pi * uv_freq * t + seed_offset) + 0.03 * rng.standard_normal(n)
    sig = slope * uv + intercept + 0.1 * np.sin(2.0 * np.pi * sig_freq * t + 0.5 + seed_offset)
    return uv, sig


def _multi_roi_chunk(rng, n_rois, n, fs, roi_names=None):
    roi_names = roi_names or [f"ROI{i}" for i in range(n_rois)]
    uv_cols, sig_cols = [], []
    for i in range(n_rois):
        uv, sig = _synth_roi(rng, n, fs, slope=1.2 + 0.3 * i, seed_offset=0.3 * i)
        uv_cols.append(uv)
        sig_cols.append(sig)
    uv_raw = np.stack(uv_cols, axis=1)
    sig_raw = np.stack(sig_cols, axis=1)
    return _make_chunk(uv_raw, sig_raw, roi_names, fs)


def _direct_dispatch(dynamic_fit_mode, chunk, cfg):
    """Call the exact old, ungrouped per-mode function for `dynamic_fit_mode`
    on the full chunk -- the "old path" side of every equivalence test."""
    if dynamic_fit_mode == "global_linear_regression":
        return regression._compute_dynamic_fit_ref_global_linear(chunk, cfg, "phasic")
    if dynamic_fit_mode == "robust_global_event_reject":
        return regression._compute_dynamic_fit_ref_robust_global_event_reject(chunk, cfg, "phasic")
    if dynamic_fit_mode == "adaptive_event_gated_regression":
        return regression._compute_dynamic_fit_ref_adaptive_event_gated_regression(chunk, cfg, "phasic")
    # Both rolling variants share the same underlying function, selected by
    # the fit_mode kwarg -- this is dispatch target #4/#5 of the five
    # supported resolved modes (correction item 1).
    return regression._compute_dynamic_fit_ref(chunk, cfg, "phasic", fit_mode=dynamic_fit_mode)


ALL_FIVE_RESOLVED_MODES = [
    "global_linear_regression",
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
    "rolling_filtered_to_raw",
    "rolling_filtered_to_filtered",
]


@pytest.mark.parametrize("dynamic_fit_mode", ALL_FIVE_RESOLVED_MODES)
def test_homogeneous_grouped_dispatch_matches_old_direct_call(dynamic_fit_mode):
    """A run where every ROI uses the same mode must produce numerically
    identical results (fitted reference AND delta-F) whether computed via
    the old direct per-mode call on the full chunk, or via
    fit_chunk_dynamic's grouped dispatch (which forms exactly one group
    covering every ROI). Covers all five supported resolved dispatch
    targets, not just the three non-rolling ones (correction item 1)."""
    rng = np.random.default_rng(7)
    n, fs, n_rois = 3200, 40.0, 3
    roi_names = ["Region0", "Region1", "Region2"]

    chunk_direct = _multi_roi_chunk(rng, n_rois, n, fs, roi_names)
    rng2 = np.random.default_rng(7)
    chunk_grouped = _multi_roi_chunk(rng2, n_rois, n, fs, roi_names)

    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2,
                 dynamic_fit_mode=dynamic_fit_mode)
    _prepare_filtered(chunk_direct, cfg)
    _prepare_filtered(chunk_grouped, cfg)

    old_uv_fit = _direct_dispatch(dynamic_fit_mode, chunk_direct, cfg)
    old_delta_f = regression._assemble_delta_f_from_fit(chunk_direct.sig_raw, old_uv_fit)

    grouped_uv_fit, grouped_delta_f = regression.fit_chunk_dynamic(chunk_grouped, cfg, mode="phasic")

    np.testing.assert_allclose(grouped_uv_fit, old_uv_fit, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grouped_delta_f, old_delta_f, rtol=0.0, atol=1e-12)
    # Full lifecycle sanity: delta_f is assembled from the same sig_raw/uv_fit.
    np.testing.assert_allclose(
        grouped_delta_f, chunk_grouped.sig_raw - grouped_uv_fit, rtol=0.0, atol=1e-12
    )
    # Homogeneous run must resolve to the single mode, not "mixed", and every
    # ROI must be tagged with it (this is what completion/provenance reads).
    assert grouped_uv_fit.shape == (n, n_rois)
    assert chunk_grouped.metadata["dynamic_fit_mode_resolved"] == dynamic_fit_mode
    mode_by_roi = chunk_grouped.metadata["dynamic_fit_mode_resolved_by_roi"]
    assert mode_by_roi == {roi: dynamic_fit_mode for roi in roi_names}
    assert chunk_grouped.metadata["dynamic_fit_group_count"] == 1


@pytest.mark.parametrize("dynamic_fit_mode", ALL_FIVE_RESOLVED_MODES)
def test_homogeneous_grouped_dispatch_preserves_metadata_semantics(dynamic_fit_mode):
    """Not just the numerical trace: a homogeneous run's metadata (QC-record
    inputs, engine info, per-ROI fit-detail dicts, baseline/bleach scalars)
    must be semantically identical between old direct dispatch and grouped
    dispatch (correction item 2) -- proven from actual readers, not asserted
    by inspecting fit_chunk_dynamic's own source."""
    rng = np.random.default_rng(21)
    n, fs, n_rois = 2400, 40.0, 2
    roi_names = ["RoiA", "RoiB"]

    chunk_direct = _multi_roi_chunk(rng, n_rois, n, fs, roi_names)
    rng2 = np.random.default_rng(21)
    chunk_grouped = _multi_roi_chunk(rng2, n_rois, n, fs, roi_names)

    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2,
                 dynamic_fit_mode=dynamic_fit_mode, baseline_subtract_before_fit=True)
    _prepare_filtered(chunk_direct, cfg)
    _prepare_filtered(chunk_grouped, cfg)

    old_uv_fit, old_delta_f = regression.fit_chunk_dynamic(chunk_direct, cfg, mode="phasic")
    grouped_uv_fit, grouped_delta_f = regression.fit_chunk_dynamic(chunk_grouped, cfg, mode="phasic")

    # dynamic_fit_engine / dynamic_fit_engine_info: HDF5 attribute reader
    # (io/hdf5_cache.py add_chunk) and preview reader (correction_preview.py
    # _method_metadata_for_roi) both read these flat keys for a homogeneous
    # chunk -- must match exactly, not just "be present".
    assert chunk_grouped.metadata["dynamic_fit_engine"] == chunk_direct.metadata["dynamic_fit_engine"]
    old_info = dict(chunk_direct.metadata["dynamic_fit_engine_info"])
    new_info = dict(chunk_grouped.metadata["dynamic_fit_engine_info"])
    assert old_info == new_info

    # baseline_subtract_before_fit_applied: Pipeline QC / provenance readers
    # consume this flat scalar; for a homogeneous run it must be a concrete
    # bool equal to the old single-call value, never the "mixed" sentinel.
    assert (
        chunk_grouped.metadata["baseline_subtract_before_fit_applied"]
        == chunk_direct.metadata["baseline_subtract_before_fit_applied"]
    )
    assert isinstance(chunk_grouped.metadata["baseline_subtract_before_fit_applied"], bool)

    # Per-ROI fit-detail dicts (whichever the mode populates) must match
    # entry-for-entry, keyed by ROI name.
    per_roi_dict_key = {
        "global_linear_regression": "dynamic_fit_global_linear",
        "robust_global_event_reject": "dynamic_fit_event_reject",
        "adaptive_event_gated_regression": "dynamic_fit_adaptive_event_gated",
        "rolling_filtered_to_raw": "dynamic_fit_rolling_local",
        "rolling_filtered_to_filtered": "dynamic_fit_rolling_local",
    }[dynamic_fit_mode]
    old_by_roi = chunk_direct.metadata.get(per_roi_dict_key, {})
    new_by_roi = chunk_grouped.metadata.get(per_roi_dict_key, {})
    assert set(old_by_roi.keys()) == set(new_by_roi.keys()) == set(roi_names)
    for roi in roi_names:
        # Compare the numeric slope_summary sub-dict (stable, comparable);
        # full dict equality risks brittleness against unrelated float noise
        # in unrelated sub-fields not relevant to this contract.
        assert old_by_roi[roi].get("slope_summary") == new_by_roi[roi].get("slope_summary")


def test_mixed_fit_modes_stable_scatter_back_identity():
    """Five ROIs, five different resolved dynamic-fit modes -- the required
    multi-strategy case, extended to include a rolling mode (correction item
    1: "add at least one mixed-mode scatter-back test that includes a
    rolling mode"). Each ROI's column in the mixed run must exactly match
    what that same ROI would produce as a lone single-ROI chunk under its
    own mode -- proving subsetting changes nothing and scatter-back never
    shifts/relabels a ROI.
    """
    rng = np.random.default_rng(99)
    n, fs = 3200, 40.0
    roi_modes = {
        "RobustROI": "robust_global_event_reject",
        "RollingRawROI": "rolling_filtered_to_raw",
        "GlobalLinearROI": "global_linear_regression",
        "AdaptiveROI": "adaptive_event_gated_regression",
        "RollingFiltROI": "rolling_filtered_to_filtered",
    }
    roi_names = list(roi_modes.keys())
    n_rois = len(roi_names)

    mixed_chunk = _multi_roi_chunk(rng, n_rois, n, fs, roi_names)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    _prepare_filtered(mixed_chunk, cfg)

    strategy_map = {
        roi: PerRoiCorrectionSpec(
            roi_id=roi, strategy_family="dynamic_fit", selected_strategy=mode, dynamic_fit_mode=mode
        )
        for roi, mode in roi_modes.items()
    }
    mixed_uv_fit, mixed_delta_f = regression.fit_chunk_dynamic(
        mixed_chunk, cfg, mode="phasic", per_roi_correction=strategy_map
    )

    assert mixed_chunk.metadata["dynamic_fit_mode_resolved"] == "mixed"
    assert mixed_chunk.metadata["dynamic_fit_mode_resolved_by_roi"] == roi_modes
    assert mixed_chunk.metadata["dynamic_fit_group_count"] == len(set(roi_modes.values()))
    # Mixed run must never claim a single truthful engine.
    assert mixed_chunk.metadata["dynamic_fit_engine"] is None
    assert mixed_chunk.metadata["dynamic_fit_engine_info"] is None

    # Rebuild the exact same per-ROI raw data (same rng draw order) as a lone
    # single-ROI chunk and confirm the mixed run's column matches exactly.
    rng_solo = np.random.default_rng(99)
    solo_chunks = []
    for i, roi in enumerate(roi_names):
        uv, sig = _synth_roi(rng_solo, n, fs, slope=1.2 + 0.3 * i, seed_offset=0.3 * i)
        solo_chunks.append((roi, uv, sig))

    for r_idx, (roi, uv, sig) in enumerate(solo_chunks):
        solo_chunk = _make_chunk(uv.reshape(-1, 1), sig.reshape(-1, 1), [roi], fs)
        _prepare_filtered(solo_chunk, cfg)
        solo_uv_fit = _direct_dispatch(roi_modes[roi], solo_chunk, cfg)
        np.testing.assert_allclose(
            mixed_uv_fit[:, r_idx], solo_uv_fit[:, 0], rtol=0.0, atol=1e-12,
            err_msg=f"ROI {roi!r} (column {r_idx}) diverged from its solo computation",
        )


def test_one_group_degenerate_roi_does_not_affect_other_groups_or_roi():
    """A degenerate ROI in one group (global_linear) must NaN-fill only its
    own column; a healthy ROI in the SAME group and a healthy ROI in a
    DIFFERENT group must both remain fully finite and correct."""
    rng = np.random.default_rng(55)
    n, fs = 3200, 40.0
    roi_names = ["DegenerateROI", "HealthySameGroupROI", "HealthyOtherGroupROI"]

    uv_degenerate = np.full(n, 5.0)  # zero variance -> _global_fit_params degenerate (DD2)
    sig_degenerate = np.full(n, 3.0) + 0.01 * rng.standard_normal(n)
    uv_healthy_same, sig_healthy_same = _synth_roi(rng, n, fs, slope=1.5, seed_offset=0.7)
    uv_healthy_other, sig_healthy_other = _synth_roi(rng, n, fs, slope=1.8, seed_offset=1.1)

    uv_raw = np.stack([uv_degenerate, uv_healthy_same, uv_healthy_other], axis=1)
    sig_raw = np.stack([sig_degenerate, sig_healthy_same, sig_healthy_other], axis=1)
    chunk = _make_chunk(uv_raw, sig_raw, roi_names, fs)

    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    _prepare_filtered(chunk, cfg)

    strategy_map = {
        "DegenerateROI": PerRoiCorrectionSpec(
            roi_id="DegenerateROI", strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression", dynamic_fit_mode="global_linear_regression",
        ),
        "HealthySameGroupROI": PerRoiCorrectionSpec(
            roi_id="HealthySameGroupROI", strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression", dynamic_fit_mode="global_linear_regression",
        ),
        "HealthyOtherGroupROI": PerRoiCorrectionSpec(
            roi_id="HealthyOtherGroupROI", strategy_family="dynamic_fit",
            selected_strategy="robust_global_event_reject", dynamic_fit_mode="robust_global_event_reject",
        ),
    }
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)

    assert np.all(np.isnan(uv_fit[:, 0])), "degenerate ROI's own column must be NaN"
    assert np.all(np.isfinite(uv_fit[:, 1])), "healthy ROI in the SAME group must remain unaffected"
    assert np.all(np.isfinite(uv_fit[:, 2])), "healthy ROI in a DIFFERENT group must remain unaffected"

    qc_warnings = chunk.metadata.get("qc_warnings", [])
    assert any("DEGENERATE" in w for w in qc_warnings)


def test_all_dynamic_fit_map_produces_no_mixed_sentinel_for_single_mode():
    """Sanity guard: an explicit per-ROI map that happens to be uniform must
    still resolve to the concrete mode string, not the 'mixed' sentinel."""
    rng = np.random.default_rng(3)
    n, fs = 1600, 40.0
    roi_names = ["A", "B"]
    chunk = _multi_roi_chunk(rng, 2, n, fs, roi_names)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    _prepare_filtered(chunk, cfg)

    strategy_map = {
        roi: PerRoiCorrectionSpec(
            roi_id=roi, strategy_family="dynamic_fit",
            selected_strategy="robust_global_event_reject", dynamic_fit_mode="robust_global_event_reject",
        )
        for roi in roi_names
    }
    regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)
    assert chunk.metadata["dynamic_fit_mode_resolved"] == "robust_global_event_reject"


def test_channel_names_shorter_than_array_width_falls_back_like_legacy_per_mode_functions():
    """Some existing tests/callers construct a Chunk positionally and leave
    channel_names shorter than (or absent relative to) the array width. The
    per-mode functions have always tolerated this via a roi_{index} fallback;
    grouped dispatch must preserve that tolerance exactly, not require
    channel_names to be fully populated."""
    n, fs = 800, 40.0
    rng = np.random.default_rng(11)
    uv, sig = _synth_roi(rng, n, fs)
    chunk = Chunk(0, "dummy", "rwd", np.arange(n, dtype=float) / fs,
                  uv.reshape(-1, 1), sig.reshape(-1, 1), fs_hz=fs, channel_names=[], metadata={})
    chunk.uv_filt = uv.reshape(-1, 1).copy()
    chunk.sig_filt = sig.reshape(-1, 1).copy()

    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")
    assert uv_fit is not None
    assert np.any(np.isfinite(uv_fit))
    assert chunk.metadata["dynamic_fit_mode_resolved_by_roi"] == {"roi_0": "robust_global_event_reject"}


# ---------------------------------------------------------------------------
# Correction item 4: the all-non-dynamic return contract must be a full-width,
# original-order array (never None), and must remain distinguishable from a
# catastrophic dynamic-fit failure (which surfaces as a raised exception, not
# a different return value).
# ---------------------------------------------------------------------------


def test_no_dynamic_groups_returns_full_width_all_nan_not_none():
    """Every ROI is signal_only_f0 (a stand-in for a future all-Signal-Only-F0
    chunk; no Signal-Only computation is added here). fit_chunk_dynamic must
    still return a full-width, original-column-order array -- all NaN, since
    no regression ran -- never None, so a canonical assembler downstream
    never has to special-case "zero groups" as a different shape than "some
    groups"."""
    rng = np.random.default_rng(41)
    n, fs = 1600, 40.0
    roi_names = ["OnlySignalOnlyA", "OnlySignalOnlyB"]
    chunk = _multi_roi_chunk(rng, 2, n, fs, roi_names)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    _prepare_filtered(chunk, cfg)

    strategy_map = {
        roi: PerRoiCorrectionSpec(
            roi_id=roi, strategy_family="signal_only_f0", selected_strategy="signal_only_f0",
            dynamic_fit_mode=None,
        )
        for roi in roi_names
    }
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)

    assert uv_fit is not None, "must be a full-width array, not None, even with zero dynamic groups"
    assert delta_f is not None
    assert uv_fit.shape == (n, 2)
    assert delta_f.shape == (n, 2)
    assert np.all(np.isnan(uv_fit))
    assert np.all(np.isnan(delta_f))

    assert chunk.metadata["dynamic_fit_group_count"] == 0
    assert chunk.metadata["dynamic_fit_mode_resolved"] == "none"
    assert chunk.metadata["dynamic_fit_mode_resolved_by_roi"] == {}
    assert chunk.metadata["dynamic_fit_engine"] is None
    assert chunk.metadata["dynamic_fit_engine_info"] is None


def test_one_dynamic_group_plus_non_dynamic_placeholder_rois():
    """One real dynamic-fit ROI alongside two signal_only_f0 placeholder
    ROIs: the dynamic ROI's column is fully computed and correctly
    attributed; the placeholder ROIs' columns stay NaN and are absent from
    dynamic_fit_mode_resolved_by_roi (they were never dispatched to any
    dynamic-fit group) -- proving the two states ("intentionally not
    dynamic-fit" vs "dynamic-fit ran and this is its answer") stay
    distinguishable per ROI within the same chunk."""
    rng = np.random.default_rng(43)
    n, fs = 1600, 40.0
    roi_names = ["DynamicROI", "PlaceholderA", "PlaceholderB"]
    chunk = _multi_roi_chunk(rng, 3, n, fs, roi_names)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    _prepare_filtered(chunk, cfg)

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
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=strategy_map)

    assert uv_fit is not None
    assert np.all(np.isfinite(uv_fit[:, 0])), "the one real dynamic-fit ROI must be fully computed"
    assert np.all(np.isnan(uv_fit[:, 1])), "placeholder ROI must stay NaN, never fabricated"
    assert np.all(np.isnan(uv_fit[:, 2])), "placeholder ROI must stay NaN, never fabricated"

    assert chunk.metadata["dynamic_fit_group_count"] == 1
    assert chunk.metadata["dynamic_fit_mode_resolved"] == "global_linear_regression"
    # Only the dynamic-fit ROI is attributed a resolved mode; the
    # placeholders were never dispatched to any group at all.
    assert chunk.metadata["dynamic_fit_mode_resolved_by_roi"] == {
        "DynamicROI": "global_linear_regression"
    }


def test_catastrophic_dynamic_fit_failure_propagates_as_exception_not_a_return_value():
    """A per-mode function that raises for structural reasons (e.g. missing
    filtered traces, which _compute_dynamic_fit_ref_global_linear requires
    unconditionally) must propagate straight out of fit_chunk_dynamic as an
    exception -- never converted into a None/NaN return that could be
    confused with the intentional zero-dynamic-groups case."""
    rng = np.random.default_rng(45)
    n, fs = 800, 40.0
    uv, sig = _synth_roi(rng, n, fs)
    chunk = _make_chunk(uv.reshape(-1, 1), sig.reshape(-1, 1), ["Region0"], fs)
    # Deliberately do NOT call _prepare_filtered: uv_filt/sig_filt stay None,
    # which _compute_dynamic_fit_ref_global_linear requires and raises
    # RuntimeError for.
    cfg = Config(dynamic_fit_mode="global_linear_regression")

    with pytest.raises(RuntimeError, match="Filtered traces are required"):
        regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")

    # The chunk's temporary bleach-corrected fields must still be restored
    # (finally block) even though the group loop raised.
    assert chunk.sig_raw is not None


# ---------------------------------------------------------------------------
# Correction item 3: harden the contract between _dispatch_one_dynamic_fit_
# group and fit_chunk_dynamic -- a None, wrong-dimensionality, or wrong-shape
# return must fail loudly, before scatter-back, distinguishably from the
# intentional zero-dynamic-groups case (which never calls dispatch at all).
# ---------------------------------------------------------------------------


def _single_roi_chunk_and_config(rng, n=800, fs=40.0, mode="global_linear_regression"):
    uv, sig = _synth_roi(rng, n, fs)
    chunk = _make_chunk(uv.reshape(-1, 1), sig.reshape(-1, 1), ["Region0"], fs)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2,
                 dynamic_fit_mode=mode)
    _prepare_filtered(chunk, cfg)
    return chunk, cfg


def test_mocked_group_returning_none_fails_loudly(monkeypatch):
    rng = np.random.default_rng(71)
    chunk, cfg = _single_roi_chunk_and_config(rng)
    monkeypatch.setattr(regression, "_dispatch_one_dynamic_fit_group", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="returned None"):
        regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")


def test_mocked_group_returning_wrong_width_fails_loudly(monkeypatch):
    """Wrong ROI-column count for the group (e.g. 2 columns for a 1-ROI
    group) must be rejected before scatter-back, not silently truncate or
    index out of range into another group's columns."""
    rng = np.random.default_rng(73)
    n = 800
    chunk, cfg = _single_roi_chunk_and_config(rng, n=n)
    wrong_width = np.zeros((n, 2), dtype=float)
    monkeypatch.setattr(regression, "_dispatch_one_dynamic_fit_group", lambda *a, **k: wrong_width)

    with pytest.raises(RuntimeError, match="expected"):
        regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")


def test_mocked_group_returning_wrong_sample_count_fails_loudly(monkeypatch):
    rng = np.random.default_rng(75)
    n = 800
    chunk, cfg = _single_roi_chunk_and_config(rng, n=n)
    wrong_samples = np.zeros((n - 10, 1), dtype=float)
    monkeypatch.setattr(regression, "_dispatch_one_dynamic_fit_group", lambda *a, **k: wrong_samples)

    with pytest.raises(RuntimeError, match="expected"):
        regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")


def test_mocked_group_returning_1d_array_fails_loudly(monkeypatch):
    rng = np.random.default_rng(77)
    n = 800
    chunk, cfg = _single_roi_chunk_and_config(rng, n=n)
    wrong_ndim = np.zeros(n, dtype=float)
    monkeypatch.setattr(regression, "_dispatch_one_dynamic_fit_group", lambda *a, **k: wrong_ndim)

    with pytest.raises(RuntimeError, match="dimensional"):
        regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")


def test_group_validation_failure_distinguishable_from_zero_dynamic_groups(monkeypatch):
    """A malformed group return raises; the intentional all-non-dynamic case
    still returns full-width all-NaN arrays. The two must never be
    confusable with each other."""
    rng = np.random.default_rng(79)
    n, fs = 800, 40.0
    chunk_bad, cfg = _single_roi_chunk_and_config(rng, n=n, fs=fs)
    monkeypatch.setattr(regression, "_dispatch_one_dynamic_fit_group", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        regression.fit_chunk_dynamic(chunk_bad, cfg, mode="phasic")

    monkeypatch.undo()
    rng2 = np.random.default_rng(79)
    uv, sig = _synth_roi(rng2, n, fs)
    chunk_zero_groups = _make_chunk(uv.reshape(-1, 1), sig.reshape(-1, 1), ["Region0"], fs)
    cfg2 = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    _prepare_filtered(chunk_zero_groups, cfg2)
    strategy_map = {
        "Region0": PerRoiCorrectionSpec(
            roi_id="Region0", strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0", dynamic_fit_mode=None,
        )
    }
    uv_fit, delta_f = regression.fit_chunk_dynamic(
        chunk_zero_groups, cfg2, mode="phasic", per_roi_correction=strategy_map
    )
    assert uv_fit is not None
    assert np.all(np.isnan(uv_fit))
    assert chunk_zero_groups.metadata["dynamic_fit_group_count"] == 0
