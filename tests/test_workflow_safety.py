import os

import pytest
import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.workflow_safety import (
    CONTINUOUS_AUTO_FORMAT_MESSAGE,
    CONTINUOUS_NPM_UNSUPPORTED_MESSAGE,
    resolve_feature_event_defaults,
    validate_format_mode_compatibility,
    validate_output_write_safety,
)


def test_output_safety_blocks_diagnostic_cache_target_inside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output_base = tmp_path / "output"
    output_base.mkdir()
    target = source / "_guided_diagnostic_cache" / "cache_001"

    result = validate_output_write_safety(
        source_root=source,
        output_base=output_base,
        target_path=target,
        operation_kind="diagnostic_cache",
    )

    assert not result.ok
    assert result.code == "target_inside_source"


def test_output_safety_blocks_source_inside_output_base_for_diagnostic_cache(tmp_path):
    output_base = tmp_path / "workspace"
    source = output_base / "source"
    source.mkdir(parents=True)
    target = output_base / "_guided_diagnostic_cache" / "cache_001"

    result = validate_output_write_safety(
        source_root=source,
        output_base=output_base,
        target_path=target,
        operation_kind="diagnostic_cache",
    )

    assert not result.ok
    assert result.code == "source_inside_output_base"


def test_output_safety_existing_diagnostic_cache_target_fails_by_default(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output_base = tmp_path / "output"
    target = output_base / "_guided_diagnostic_cache" / "cache_001"
    target.mkdir(parents=True)

    result = validate_output_write_safety(
        source_root=source,
        output_base=output_base,
        target_path=target,
        operation_kind="diagnostic_cache",
    )

    assert not result.ok
    assert result.code == "target_exists"


def test_output_safety_existing_output_base_unique_nested_target_allowed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output_base = tmp_path / "output"
    output_base.mkdir()
    target = output_base / "_guided_diagnostic_cache" / "cache_001"

    result = validate_output_write_safety(
        source_root=source,
        output_base=output_base,
        target_path=target,
        operation_kind="diagnostic_cache",
    )

    assert result.ok
    assert result.code == "ok"
    assert any("parent will need to be created" in warning for warning in result.warnings)


def test_output_safety_read_only_review_does_not_require_output_target():
    result = validate_output_write_safety(
        source_root=None,
        output_base=None,
        target_path=None,
        operation_kind="read_only_review",
    )

    assert result.ok
    assert result.code == "read_only_review"


def test_output_safety_blocks_protected_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output_base = tmp_path / "output"
    output_base.mkdir()
    protected = output_base / "completed_run"
    protected.mkdir()
    target = protected / "_guided_diagnostic_cache" / "cache_001"

    result = validate_output_write_safety(
        source_root=source,
        output_base=output_base,
        target_path=target,
        operation_kind="diagnostic_cache",
        protected_roots=[protected],
    )

    assert not result.ok
    assert result.code == "target_inside_protected_root"


@pytest.mark.parametrize("fmt", ["npm", "custom_tabular"])
def test_format_mode_intermittent_supported_formats_allowed(fmt):
    result = validate_format_mode_compatibility(
        input_format=fmt,
        acquisition_mode="intermittent",
    )

    assert result.ok
    assert result.code == "ok"


def test_format_mode_continuous_custom_tabular_allowed():
    result = validate_format_mode_compatibility(
        input_format="custom_tabular",
        acquisition_mode="continuous",
    )

    assert result.ok
    assert result.resolved_format == "custom_tabular"


def test_format_mode_continuous_rwd_matches_backend_support():
    result = validate_format_mode_compatibility(
        input_format="rwd",
        acquisition_mode="continuous",
    )

    assert result.ok
    assert result.resolved_format == "rwd"


def test_format_mode_continuous_npm_blocked_with_stable_message():
    result = validate_format_mode_compatibility(
        input_format="npm",
        acquisition_mode="continuous",
    )

    assert not result.ok
    assert result.code == "continuous_npm_unsupported"
    assert result.message == CONTINUOUS_NPM_UNSUPPORTED_MESSAGE


def test_format_mode_continuous_auto_blocked_as_ambiguous():
    result = validate_format_mode_compatibility(
        input_format="auto",
        acquisition_mode="continuous",
    )

    assert not result.ok
    assert result.code == "continuous_auto_ambiguous"
    assert result.message == CONTINUOUS_AUTO_FORMAT_MESSAGE


def test_feature_event_defaults_from_baseline_config(tmp_path):
    cfg_path = tmp_path / "baseline.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "event_signal": "delta_f",
                "signal_excursion_polarity": "negative",
                "peak_threshold_method": "percentile",
                "peak_threshold_percentile": 90.0,
                "peak_min_distance_sec": 2.0,
                "peak_min_prominence_k": 3.0,
                "peak_min_width_sec": 0.5,
                "peak_pre_filter": "lowpass",
                "event_auc_baseline": "median",
            }
        ),
        encoding="utf-8",
    )

    result = resolve_feature_event_defaults(
        config_source_path=cfg_path,
        baseline_source_kind="custom_config",
        fallback_config=Config(),
    )

    assert result.baseline_source_kind == "custom_config"
    assert os.path.normcase(result.baseline_source_path) == os.path.normcase(str(cfg_path.resolve()))
    assert result.warnings == ()
    assert result.defaults["event_signal"] == "delta_f"
    assert result.defaults["signal_excursion_polarity"] == "negative"
    assert result.defaults["peak_threshold_method"] == "percentile"
    assert result.defaults["peak_threshold_percentile"] == pytest.approx(90.0)
    assert result.defaults["peak_min_distance_sec"] == pytest.approx(2.0)
    assert result.defaults["peak_min_prominence_k"] == pytest.approx(3.0)
    assert result.defaults["peak_min_width_sec"] == pytest.approx(0.5)
    assert result.defaults["peak_pre_filter"] == "lowpass"
    assert result.defaults["event_auc_baseline"] == "median"


def test_feature_event_defaults_fallback_is_explicit(tmp_path):
    missing = tmp_path / "missing.yaml"
    fallback = Config(event_signal="delta_f", peak_threshold_k=7.0)

    result = resolve_feature_event_defaults(
        config_source_path=missing,
        baseline_source_kind="custom_config",
        fallback_config=fallback,
        allow_fallback=True,
    )

    assert result.baseline_source_kind == "fallback"
    assert result.fallback_reason
    assert result.warnings
    assert result.defaults["event_signal"] == "delta_f"
    assert result.defaults["peak_threshold_k"] == pytest.approx(7.0)


def test_feature_event_defaults_missing_baseline_can_fail_without_fallback(tmp_path):
    with pytest.raises(ValueError, match="Baseline config path does not exist"):
        resolve_feature_event_defaults(
            config_source_path=tmp_path / "missing.yaml",
            fallback_config=Config(),
            allow_fallback=False,
        )
