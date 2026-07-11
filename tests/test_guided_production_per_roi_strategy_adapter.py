"""Phase 1.2 correction item 6: the sole GuidedProductionPerRoiStrategy ->
PerRoiCorrectionSpec conversion, tested against real objects produced by the
real production mapping pipeline -- not hand-built fixtures standing in for
what the mapper would produce."""
from __future__ import annotations

import pytest

import photometry_pipeline.guided_backend_validation_request as contracts
import photometry_pipeline.guided_production_mapping as mapping
from photometry_pipeline.core.types import PerRoiCorrectionSpec
from tests.test_guided_backend_validator import _request as _valid_request
from tests.test_guided_production_mapping import _map, _unchecked


def _real_mapped_entry(**per_roi_changes) -> mapping.GuidedProductionPerRoiStrategy:
    """Build one real GuidedProductionPerRoiStrategy by running an actual
    request through the real production mapping pipeline, exactly as
    tests.test_guided_production_mapping.test_intent_identity_canonicalizes_
    per_roi_production_strategy_map does."""
    values = dict(
        roi_id="ROI0",
        strategy_family="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        selected_strategy="global_linear_regression",
        evidence_source_type="local_correction_preview",
        evidence_reference_json="{}",
        explicit_user_mark=True,
        current_or_stale="current",
    )
    values.update(per_roi_changes)
    request = _valid_request()
    per_roi_entry = contracts.GuidedBackendPerRoiProductionStrategy(**values)
    request = _unchecked(
        request,
        correction=_unchecked(
            request.correction,
            production_strategy_map_version="guided_production_strategy_map.v1",
            per_roi_production_strategy_map=(per_roi_entry,),
        ),
    )
    result = _map(
        request,
        identity=contracts.compute_guided_backend_validation_request_identity(request),
    )
    assert isinstance(result, mapping.GuidedProductionMappingSuccess), result
    entries = result.intent.correction.per_roi_production_strategy_map
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, mapping.GuidedProductionPerRoiStrategy)
    return entry


def test_adapter_converts_real_dynamic_fit_entry():
    entry = _real_mapped_entry()
    spec = mapping.guided_production_per_roi_strategy_to_correction_spec(entry)
    assert isinstance(spec, PerRoiCorrectionSpec)
    assert spec.roi_id == "ROI0"
    assert spec.strategy_family == "dynamic_fit"
    assert spec.selected_strategy == "global_linear_regression"
    assert spec.dynamic_fit_mode == "global_linear_regression"
    assert spec.parameter_identity == ""
    assert spec.evidence_identity == "local_correction_preview::{}"


def test_adapter_converts_real_signal_only_f0_entry():
    entry = _real_mapped_entry(
        strategy_family="signal_only_f0",
        dynamic_fit_mode=None,
        selected_strategy="signal_only_f0",
    )
    spec = mapping.guided_production_per_roi_strategy_to_correction_spec(entry)
    assert spec.strategy_family == "signal_only_f0"
    assert spec.selected_strategy == "signal_only_f0"
    assert spec.dynamic_fit_mode is None


def test_adapter_rejects_non_explicit_mark():
    entry = _real_mapped_entry(explicit_user_mark=False)
    with pytest.raises(ValueError, match="explicit user mark"):
        mapping.guided_production_per_roi_strategy_to_correction_spec(entry)


def test_adapter_rejects_stale_entry():
    entry = _real_mapped_entry(current_or_stale="stale")
    with pytest.raises(ValueError, match="stale"):
        mapping.guided_production_per_roi_strategy_to_correction_spec(entry)


def test_adapter_rejects_unsupported_strategy_family():
    entry = _real_mapped_entry(
        strategy_family="unsupported", dynamic_fit_mode=None, selected_strategy="needs_review"
    )
    with pytest.raises(ValueError, match="unsupported strategy_family"):
        mapping.guided_production_per_roi_strategy_to_correction_spec(entry)


def test_adapter_rejects_mismatched_selected_strategy_and_dynamic_fit_mode():
    """PerRoiCorrectionSpec's own __post_init__ is this adapter's fit-mode
    mapping check; a real entry where the two fields disagree must still
    fail closed even though the production layer itself would not normally
    produce such a mismatch."""
    entry = _real_mapped_entry(
        dynamic_fit_mode="global_linear_regression",
        selected_strategy="robust_global_event_reject",
    )
    with pytest.raises(ValueError):
        mapping.guided_production_per_roi_strategy_to_correction_spec(entry)


def test_adapter_rejects_invalid_dynamic_fit_mode():
    entry = _real_mapped_entry(
        dynamic_fit_mode="not_a_real_mode", selected_strategy="not_a_real_mode"
    )
    with pytest.raises(ValueError):
        mapping.guided_production_per_roi_strategy_to_correction_spec(entry)


def test_map_function_converts_full_real_map_and_feeds_fit_chunk_dynamic():
    """The map-level adapter's output is exactly the shape
    regression.fit_chunk_dynamic's per_roi_correction argument expects."""
    import numpy as np
    from photometry_pipeline.config import Config
    from photometry_pipeline.core import preprocessing, regression
    from photometry_pipeline.core.types import Chunk

    entry = _real_mapped_entry()
    specs = mapping.guided_production_strategy_map_to_correction_specs((entry,))
    assert specs == {"ROI0": mapping.guided_production_per_roi_strategy_to_correction_spec(entry)}

    rng = np.random.default_rng(1)
    n, fs = 1600, 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t) + 0.03 * rng.standard_normal(n)
    sig = 1.4 * uv + 2.0 + 0.1 * rng.standard_normal(n)
    chunk = Chunk(
        chunk_id=0, source_file="x.csv", format="npm", time_sec=t,
        uv_raw=uv.reshape(-1, 1), sig_raw=sig.reshape(-1, 1), fs_hz=fs,
        channel_names=["ROI0"], metadata={},
    )
    cfg = Config()
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic", per_roi_correction=specs)
    assert uv_fit is not None
    assert np.any(np.isfinite(uv_fit))


def test_map_function_rejects_duplicate_roi_id():
    entry_a = _real_mapped_entry()
    entry_b = _real_mapped_entry()
    with pytest.raises(ValueError, match="duplicate roi_id"):
        mapping.guided_production_strategy_map_to_correction_specs((entry_a, entry_b))
