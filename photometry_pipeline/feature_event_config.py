"""Pure feature/event config-field validation shared by GUI and plan contracts.

This module intentionally has no GUI, pipeline, RunSpec, feature extraction, or
filesystem dependencies. It validates already-parsed Config field values only.
"""

from __future__ import annotations

from typing import Any


FEATURE_EVENT_CONFIG_FIELDS = {
    "event_signal",
    "signal_excursion_polarity",
    "peak_threshold_method",
    "peak_threshold_k",
    "peak_threshold_percentile",
    "peak_threshold_abs",
    "peak_min_distance_sec",
    "peak_min_prominence_k",
    "peak_min_width_sec",
    "peak_pre_filter",
    "event_auc_baseline",
}
FEATURE_EVENT_EVENT_SIGNALS = {"dff", "delta_f"}
FEATURE_EVENT_POLARITIES = {"positive", "negative", "both"}
FEATURE_EVENT_THRESHOLD_METHODS = {"mean_std", "percentile", "median_mad", "absolute"}
FEATURE_EVENT_PEAK_PRE_FILTERS = {"none", "lowpass"}
FEATURE_EVENT_AUC_BASELINES = {"zero", "median"}


def validate_feature_event_config_fields(config_fields: dict[str, Any]) -> list[str]:
    """Return semantic validation errors for feature/event Config fields.

    The helper expects widget/raw text parsing to have already happened. Any
    present numeric field is validated even when inactive for the selected
    threshold method; this matches the Guided profile contract and keeps stored
    profile payloads internally valid. The Full Control parser omits inactive
    threshold fields before calling this helper.
    """

    errors: list[str] = []
    if not isinstance(config_fields, dict):
        return ["feature/event config_fields must be an object"]

    unknown = sorted(set(config_fields) - FEATURE_EVENT_CONFIG_FIELDS)
    if unknown:
        errors.append(f"unknown config fields: {unknown}")

    event_signal = config_fields.get("event_signal")
    if event_signal is not None and str(event_signal) not in FEATURE_EVENT_EVENT_SIGNALS:
        errors.append(f"invalid event_signal: {event_signal}")

    polarity = config_fields.get("signal_excursion_polarity")
    if polarity is not None and str(polarity) not in FEATURE_EVENT_POLARITIES:
        errors.append(f"invalid signal_excursion_polarity: {polarity}")

    method = config_fields.get("peak_threshold_method")
    method_text = str(method) if method is not None else ""
    if method is not None and method_text not in FEATURE_EVENT_THRESHOLD_METHODS:
        errors.append(f"invalid peak_threshold_method: {method}")

    if "peak_threshold_k" in config_fields:
        _validate_numeric(
            config_fields["peak_threshold_k"],
            "peak_threshold_k",
            errors,
            min_value=0.0,
            inclusive_min=False,
        )
    if "peak_threshold_percentile" in config_fields:
        _validate_numeric(
            config_fields["peak_threshold_percentile"],
            "peak_threshold_percentile",
            errors,
            min_value=0.0,
            max_value=100.0,
        )
    if "peak_threshold_abs" in config_fields:
        _validate_numeric(
            config_fields["peak_threshold_abs"],
            "peak_threshold_abs",
            errors,
            min_value=0.0,
            inclusive_min=False,
        )
    if method_text == "absolute" and "peak_threshold_abs" not in config_fields:
        errors.append("peak_threshold_abs is required when peak_threshold_method is absolute")

    for key in ("peak_min_distance_sec", "peak_min_prominence_k", "peak_min_width_sec"):
        if key in config_fields:
            _validate_numeric(config_fields[key], key, errors, min_value=0.0)

    pre_filter = config_fields.get("peak_pre_filter")
    if pre_filter is not None and str(pre_filter) not in FEATURE_EVENT_PEAK_PRE_FILTERS:
        errors.append(f"invalid peak_pre_filter: {pre_filter}")

    auc_baseline = config_fields.get("event_auc_baseline")
    if auc_baseline is not None and str(auc_baseline) not in FEATURE_EVENT_AUC_BASELINES:
        errors.append(f"invalid event_auc_baseline: {auc_baseline}")

    return errors


def _validate_numeric(
    value: Any,
    label: str,
    errors: list[str],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be numeric")
        return
    numeric = float(value)
    if min_value is not None:
        if inclusive_min and numeric < min_value:
            errors.append(f"{label} must be >= {min_value:g}")
        elif not inclusive_min and numeric <= min_value:
            errors.append(f"{label} must be > {min_value:g}")
    if max_value is not None:
        if inclusive_max and numeric > max_value:
            errors.append(f"{label} must be <= {max_value:g}")
        elif not inclusive_max and numeric >= max_value:
            errors.append(f"{label} must be < {max_value:g}")
