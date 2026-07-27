from types import SimpleNamespace

import pytest

from photometry_pipeline.guided_timeline import (
    GuidedTimelineError,
    accepted_continuous_window_timing,
    map_elapsed_coordinate,
    timeline_provenance,
)


def test_elapsed_coordinate_mapping_covers_clock_and_anchor_boundaries():
    assert map_elapsed_coordinate(0.0, timeline_anchor_mode="elapsed") == (0, 0.0)
    assert map_elapsed_coordinate(
        24 * 3600.0 + 90.0,
        timeline_anchor_mode="elapsed",
    ) == (1, 90.0)
    assert map_elapsed_coordinate(
        0.0,
        timeline_anchor_mode="civil",
        recording_start_clock="11:00",
    ) == (0, 11 * 3600.0)
    assert map_elapsed_coordinate(
        13 * 3600.0,
        timeline_anchor_mode="civil",
        recording_start_clock="11:00",
    ) == (1, 0.0)
    assert map_elapsed_coordinate(
        0.0,
        timeline_anchor_mode="fixed_daily_anchor",
        fixed_daily_anchor_clock="07:00",
        recording_start_clock="11:00",
    ) == (0, 4 * 3600.0)
    assert map_elapsed_coordinate(
        20 * 3600.0,
        timeline_anchor_mode="fixed_daily_anchor",
        fixed_daily_anchor_clock="07:00",
        recording_start_clock="11:00",
    ) == (1, 0.0)


def test_fixed_anchor_before_recording_start_keeps_leading_blank_in_day_zero():
    assert map_elapsed_coordinate(
        0.0,
        timeline_anchor_mode="fixed_daily_anchor",
        fixed_daily_anchor_clock="07:00",
        recording_start_clock="03:00",
    ) == (0, 20 * 3600.0)
    assert map_elapsed_coordinate(
        24 * 3600.0,
        timeline_anchor_mode="fixed_daily_anchor",
        fixed_daily_anchor_clock="07:00",
        recording_start_clock="03:00",
    ) == (1, 20 * 3600.0)


def test_timeline_provenance_is_the_exact_four_field_contract():
    provenance = timeline_provenance(
        timeline_anchor_mode="fixed_daily_anchor",
        fixed_daily_anchor_clock="7:00",
        recording_start_clock="11:00",
        recording_start_clock_source="user_confirmed",
    )
    assert provenance == {
        "timeline_mode": "fixed_daily_anchor",
        "fixed_daily_anchor_clock": "07:00",
        "recording_start_clock": "11:00",
        "recording_start_clock_source": "user_confirmed",
    }
    assert timeline_provenance(
        timeline_anchor_mode="elapsed",
        fixed_daily_anchor_clock=None,
        recording_start_clock=None,
        recording_start_clock_source="not_applicable",
    ) == {
        "timeline_mode": "elapsed",
        "fixed_daily_anchor_clock": None,
        "recording_start_clock": None,
        "recording_start_clock_source": "not_applicable",
    }


def test_civil_and_fixed_placement_require_an_explicit_recording_start_clock():
    with pytest.raises(GuidedTimelineError, match="recording-start clock is required"):
        timeline_provenance(
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock="07:00",
            recording_start_clock=None,
            recording_start_clock_source="not_applicable",
        )
    with pytest.raises(GuidedTimelineError, match="must not carry"):
        timeline_provenance(
            timeline_anchor_mode="elapsed",
            fixed_daily_anchor_clock=None,
            recording_start_clock="11:00",
            recording_start_clock_source="user_confirmed",
        )


def test_window_step_comes_from_accepted_draft_not_window_length_or_segment_duration():
    timing = accepted_continuous_window_timing(
        SimpleNamespace(continuous_window_sec=90.0, continuous_step_sec=37.5),
    )
    assert timing == {
        "window_length_sec": 90.0,
        "window_step_sec": 37.5,
        "window_length_source": "accepted_draft.continuous_window_sec",
        "window_step_source": "accepted_draft.continuous_step_sec",
    }
