import sys

import pytest

from photometry_pipeline.feature_event_config import (
    FEATURE_EVENT_PEAK_PRE_FILTERS,
    validate_feature_event_config_fields,
)


def _valid_feature_event_config() -> dict:
    return {
        "event_signal": "dff",
        "signal_excursion_polarity": "positive",
        "peak_threshold_method": "mean_std",
        "peak_threshold_k": 2.5,
        "peak_threshold_percentile": 95.0,
        "peak_min_distance_sec": 1.0,
        "peak_min_prominence_k": 2.0,
        "peak_min_width_sec": 0.3,
        "peak_pre_filter": "none",
        "event_auc_baseline": "zero",
    }


def test_shared_feature_event_validator_is_pure_and_non_executing(tmp_path):
    before = sorted(tmp_path.rglob("*"))

    assert validate_feature_event_config_fields(_valid_feature_event_config()) == []

    assert sorted(tmp_path.rglob("*")) == before
    assert "photometry_pipeline.core.feature_extraction" not in sys.modules


def test_shared_feature_event_validator_accepts_valid_config():
    assert validate_feature_event_config_fields(_valid_feature_event_config()) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("event_signal", "raw", "invalid event_signal"),
        ("signal_excursion_polarity", "upward", "invalid signal_excursion_polarity"),
        ("peak_threshold_method", "magic", "invalid peak_threshold_method"),
        ("peak_threshold_k", 0.0, "peak_threshold_k must be > 0"),
        ("peak_threshold_k", "2.5", "peak_threshold_k must be numeric"),
        ("peak_threshold_percentile", -0.1, "peak_threshold_percentile must be >= 0"),
        ("peak_threshold_percentile", 100.1, "peak_threshold_percentile must be <= 100"),
        ("peak_threshold_abs", 0.0, "peak_threshold_abs must be > 0"),
        ("peak_threshold_abs", "1.0", "peak_threshold_abs must be numeric"),
        ("peak_min_distance_sec", -1.0, "peak_min_distance_sec must be >= 0"),
        ("peak_min_prominence_k", -1.0, "peak_min_prominence_k must be >= 0"),
        ("peak_min_width_sec", -1.0, "peak_min_width_sec must be >= 0"),
        ("peak_pre_filter", "smooth", "invalid peak_pre_filter"),
        ("event_auc_baseline", "mean", "invalid event_auc_baseline"),
    ],
)
def test_shared_feature_event_validator_rejects_invalid_values(field, value, message):
    cfg = _valid_feature_event_config()
    cfg[field] = value

    errors = validate_feature_event_config_fields(cfg)

    assert any(message in err for err in errors)


def test_shared_feature_event_validator_rejects_unknown_fields():
    cfg = _valid_feature_event_config()
    cfg["new_detector_mode"] = "not-a-real-config-field"

    errors = validate_feature_event_config_fields(cfg)

    assert any("unknown config fields" in err for err in errors)


def test_peak_pre_filter_values_match_current_gui_registry_semantics():
    assert FEATURE_EVENT_PEAK_PRE_FILTERS == {"none", "lowpass"}

    for mode in ("none", "lowpass"):
        cfg = _valid_feature_event_config()
        cfg["peak_pre_filter"] = mode
        assert validate_feature_event_config_fields(cfg) == []

    cfg = _valid_feature_event_config()
    cfg["peak_pre_filter"] = "smooth"
    assert any("invalid peak_pre_filter" in err for err in validate_feature_event_config_fields(cfg))


def test_absolute_threshold_requires_present_positive_abs():
    cfg = _valid_feature_event_config()
    cfg["peak_threshold_method"] = "absolute"
    cfg.pop("peak_threshold_abs", None)

    errors = validate_feature_event_config_fields(cfg)
    assert any("peak_threshold_abs is required" in err for err in errors)

    cfg["peak_threshold_abs"] = 0.0
    errors = validate_feature_event_config_fields(cfg)
    assert any("peak_threshold_abs must be > 0" in err for err in errors)


def test_inactive_absolute_threshold_is_still_validated_when_stored_in_profile():
    cfg = _valid_feature_event_config()
    cfg["peak_threshold_method"] = "mean_std"
    cfg["peak_threshold_abs"] = 0.0

    errors = validate_feature_event_config_fields(cfg)

    assert any("peak_threshold_abs must be > 0" in err for err in errors)
