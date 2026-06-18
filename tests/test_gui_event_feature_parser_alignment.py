import pytest

from photometry_pipeline.config import Config


def _defaults() -> dict:
    cfg = Config()
    return {
        "event_signal": cfg.event_signal,
        "signal_excursion_polarity": cfg.signal_excursion_polarity,
        "peak_threshold_method": cfg.peak_threshold_method,
        "peak_threshold_k": cfg.peak_threshold_k,
        "peak_threshold_percentile": cfg.peak_threshold_percentile,
        "peak_threshold_abs": cfg.peak_threshold_abs,
        "peak_min_distance_sec": cfg.peak_min_distance_sec,
        "peak_min_prominence_k": cfg.peak_min_prominence_k,
        "peak_min_width_sec": cfg.peak_min_width_sec,
        "peak_pre_filter": cfg.peak_pre_filter,
        "event_auc_baseline": cfg.event_auc_baseline,
    }


def _parse(**overrides):
    from gui.main_window import parse_and_validate_event_feature_knobs

    defaults = _defaults()
    values = {
        "event_signal_text": defaults["event_signal"],
        "peak_method_text": defaults["peak_threshold_method"],
        "peak_k_str": str(defaults["peak_threshold_k"]),
        "peak_pct_str": str(defaults["peak_threshold_percentile"]),
        "peak_abs_str": "1.0",
        "peak_dist_str": str(defaults["peak_min_distance_sec"]),
        "event_auc_text": defaults["event_auc_baseline"],
        "defaults": defaults,
        "peak_pre_filter_text": defaults["peak_pre_filter"],
        "peak_prominence_k_str": str(defaults["peak_min_prominence_k"]),
        "peak_width_sec_str": str(defaults["peak_min_width_sec"]),
        "signal_excursion_polarity_text": defaults["signal_excursion_polarity"],
    }
    values.update(overrides)
    return parse_and_validate_event_feature_knobs(**values)


def test_full_control_parser_accepts_representative_valid_values():
    parsed, err = _parse(
        event_signal_text="delta_f",
        peak_method_text="percentile",
        peak_pct_str="90.0",
        peak_pre_filter_text="lowpass",
        event_auc_text="median",
        signal_excursion_polarity_text="both",
    )

    assert err is None
    assert parsed["event_signal"] == "delta_f"
    assert parsed["peak_threshold_method"] == "percentile"
    assert parsed["peak_threshold_percentile"] == 90.0
    assert parsed["peak_pre_filter"] == "lowpass"
    assert parsed["event_auc_baseline"] == "median"
    assert parsed["signal_excursion_polarity"] == "both"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"event_signal_text": "raw"}, "Invalid Event Signal."),
        ({"signal_excursion_polarity_text": "upward"}, "Invalid Signal Excursion Polarity."),
        ({"peak_method_text": "magic"}, "Invalid Peak Threshold Method."),
        ({"peak_k_str": "0.0"}, "Peak Threshold K must be > 0."),
        (
            {"peak_method_text": "percentile", "peak_pct_str": "101.0"},
            "Peak Threshold Percentile must be between 0 and 100.",
        ),
        (
            {"peak_method_text": "absolute", "peak_abs_str": "0.0"},
            "Peak Threshold Absolute must be > 0.",
        ),
        ({"peak_dist_str": "-0.1"}, "Peak Min Distance (sec) must be >= 0."),
        ({"peak_prominence_k_str": "-0.1"}, "Peak Min Prominence K must be >= 0."),
        ({"peak_width_sec_str": "-0.1"}, "Peak Min Width (s) must be >= 0."),
        ({"peak_pre_filter_text": "smooth"}, "Invalid Peak Pre-Filter."),
        ({"event_auc_text": "mean"}, "Invalid Event AUC Baseline."),
    ],
)
def test_full_control_parser_rejects_representative_invalid_values(override, message):
    parsed, err = _parse(**override)

    assert parsed is None
    assert err == message


def test_full_control_parser_omits_inactive_absolute_threshold():
    parsed, err = _parse(peak_method_text="mean_std", peak_abs_str="0.0")

    assert err is None
    assert "peak_threshold_abs" not in parsed
