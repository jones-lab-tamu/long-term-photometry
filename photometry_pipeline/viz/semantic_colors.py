"""Small shared color vocabulary for scientist-facing visualizations."""

from __future__ import annotations


RAW_SIGNAL_COLOR = "#1F77B4"
RAW_REFERENCE_COLOR = "#8545AC"
FITTED_REFERENCE_COLOR = "#EDB120"
DFF_COLOR = "#00B308"
SUMMARY_TRACE_COLOR = "#000000"

# Baselines and non-normalized traces are deliberately neutral so they are not
# mistaken for fitted references or fractional dF/F.
NEUTRAL_BASELINE_COLOR = "#404040"
NEUTRAL_TRACE_COLOR = "#555555"


def color_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert one ``#RRGGBB`` semantic color for lightweight renderers."""
    value = str(color).lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a #RRGGBB color, got {color!r}.")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
