from photometry_pipeline.viz.semantic_colors import (
    DFF_COLOR,
    FITTED_REFERENCE_COLOR,
    RAW_REFERENCE_COLOR,
    RAW_SIGNAL_COLOR,
    SUMMARY_TRACE_COLOR,
    color_to_rgb,
)


def test_scientist_facing_semantic_colors_are_exact():
    assert RAW_SIGNAL_COLOR == "#1F77B4"
    assert color_to_rgb(RAW_SIGNAL_COLOR) == (31, 119, 180)
    assert RAW_REFERENCE_COLOR == "#8545AC"
    assert color_to_rgb(RAW_REFERENCE_COLOR) == (133, 69, 172)
    assert FITTED_REFERENCE_COLOR == "#EDB120"
    assert color_to_rgb(FITTED_REFERENCE_COLOR) == (237, 177, 32)
    assert DFF_COLOR == "#00B308"
    assert color_to_rgb(DFF_COLOR) == (0, 179, 8)
    assert SUMMARY_TRACE_COLOR == "#000000"
    assert color_to_rgb(SUMMARY_TRACE_COLOR) == (0, 0, 0)
