"""Scientist-facing names for values the app stores as internal tokens.

One source of truth, so the Guided setup screens, Review Plan, and the
completed-run Review workspace all use the same words for the same setting.
Stored values, serialized configuration, and backend field names are
unaffected; only what is displayed comes from here.

This lives under ``photometry_pipeline`` rather than in the GUI because the
completed-run reload formatters also render these values, and the pipeline
package must not import the GUI.
"""

from __future__ import annotations

FORMAT_DISPLAY_LABELS = {
    "auto": "Auto",
    "rwd": "RWD",
    "npm": "NPM",
    # Not "one file per session": CSV is now also accepted as one continuous
    # recording file. The recording-structure control states which it is.
    "custom_tabular": "CSV files",
}

# One scientist-facing name per recording structure, so Select data, Recording
# structure, and Review Plan all say the same thing. The stored values stay
# "intermittent"/"continuous".
ACQUISITION_MODE_DISPLAY_LABELS = {
    "intermittent": "Repeated sessions",
    "continuous": "Continuous recording",
}

# Feature Detection settings named as a scientist would read them.
FEATURE_EVENT_SIGNAL_DISPLAY_LABELS = {
    "dff": "dF/F",
}
FEATURE_THRESHOLD_METHOD_DISPLAY_LABELS = {
    "mean_std": "mean + standard-deviation",
    "median_mad": "median + MAD",
    "percentile": "percentile",
    "absolute": "absolute",
}
FEATURE_AUC_BASELINE_DISPLAY_LABELS = {
    "zero": "zero line",
}


def format_display_label(value: str) -> str:
    """The scientist-facing name for an input format."""
    text = str(value or "").strip()
    return FORMAT_DISPLAY_LABELS.get(text.lower(), text or "not set")


def acquisition_mode_display_label(value: str) -> str:
    """The scientist-facing name for a recording structure."""
    text = str(value or "").strip()
    return ACQUISITION_MODE_DISPLAY_LABELS.get(text.lower(), text or "not set")


def feature_event_signal_display_label(value: str) -> str:
    text = str(value or "").strip()
    return FEATURE_EVENT_SIGNAL_DISPLAY_LABELS.get(text.lower(), text)


def feature_threshold_method_display_label(value: str) -> str:
    text = str(value or "").strip()
    return FEATURE_THRESHOLD_METHOD_DISPLAY_LABELS.get(text.lower(), text)


def feature_auc_baseline_display_label(value: str) -> str:
    text = str(value or "").strip()
    return FEATURE_AUC_BASELINE_DISPLAY_LABELS.get(text.lower(), text)
