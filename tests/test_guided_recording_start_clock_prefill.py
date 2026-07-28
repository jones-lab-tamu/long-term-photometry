"""Guided RWD recording-start clock: prefilled from validated metadata, editable.

Both RWD structures read their own acquisition timestamp from the validated RWD
folder-name convention, prefill the editable field with it, and record whether
the effective value came from that timestamp or from the scientist.  The
effective value drives plotting placement; authoritative session chronology,
spacing and missing-session structure are untouched.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QDeadlineTimer
from PySide6.QtWidgets import QApplication

from gui.main_window import (
    GUIDED_RECORDING_START_CLOCK_PREFILLED_MESSAGE,
    GUIDED_RECORDING_START_CLOCK_USER_ENTERED_MESSAGE,
    GUIDED_STRUCTURE_CHOICE_AUTO,
    GUIDED_TIMELINE_START_CLOCK_REQUIRED_MESSAGE,
    MainWindow,
)
from photometry_pipeline.guided_continuous_saved_artifacts import (
    continuous_plot_coordinates,
)
from photometry_pipeline.guided_timeline import (
    GUIDED_TIMELINE_CLOCK_SOURCES,
    GuidedTimelineError,
    guided_clock_display_offset_sec,
    map_elapsed_coordinate,
    normalize_guided_clock_source,
    timeline_provenance,
)
from photometry_pipeline.viz.phasic_data_prep import compute_day_layout

pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _continuous_window(window, folder: str) -> None:
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("rwd")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )
    window._discovery_cache = {
        "resolved_format": "rwd",
        "acquisition_mode": "continuous",
        "sessions": [],
        "continuous_recording_folder": folder,
    }
    window._sync_guided_recording_visibility()


def _intermittent_discovery(session_paths, *, input_dir=None) -> dict:
    discovery = {
        "resolved_format": "rwd",
        "acquisition_mode": "intermittent",
        "sessions": [{"path": str(path)} for path in session_paths],
    }
    if input_dir is not None:
        discovery["input_dir"] = str(input_dir)
    return discovery


def _intermittent_window(window, session_paths, *, input_dir=None) -> None:
    window._set_guided_workflow_mode("new_analysis")
    window._guided_format_combo.setCurrentText("rwd")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    window._discovery_cache = _intermittent_discovery(
        session_paths, input_dir=input_dir
    )
    window._sync_guided_recording_visibility()


def _set_mode(window, mode: str) -> None:
    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData(mode)
    )


# ======================================================================
# Shared clock authority
# ======================================================================


def test_effective_source_values_stay_simple():
    assert GUIDED_TIMELINE_CLOCK_SOURCES == (
        "validated_metadata",
        "user_entered",
        "not_applicable",
    )


def test_only_the_three_current_source_values_are_accepted():
    for value in GUIDED_TIMELINE_CLOCK_SOURCES:
        assert normalize_guided_clock_source(value) == value
    for rejected in ("user_confirmed", "inferred", "user_override", ""):
        with pytest.raises(GuidedTimelineError):
            normalize_guided_clock_source(rejected)
    with pytest.raises(GuidedTimelineError):
        timeline_provenance(
            timeline_anchor_mode="civil",
            fixed_daily_anchor_clock=None,
            recording_start_clock="12:06:20",
            recording_start_clock_source="user_confirmed",
        )


def test_display_offset_is_effective_minus_validated():
    assert guided_clock_display_offset_sec("12:30:00", "12:27:45") == 135.0
    assert guided_clock_display_offset_sec("12:27:45", "12:27:45") == 0.0


def test_display_offset_resolves_to_the_nearest_clock_occurrence():
    # The field carries a clock, not a date, so a correction across midnight is
    # the small correction the scientist meant.
    assert guided_clock_display_offset_sec("00:00:30", "23:59:30") == 60.0
    assert guided_clock_display_offset_sec("23:59:30", "00:00:30") == -60.0
    # An ordinary same-day correction is unaffected.
    assert guided_clock_display_offset_sec("12:30:00", "12:27:45") == 135.0
    # The resolved offset never exceeds half a day in magnitude.
    for effective, validated in (
        ("00:00:00", "12:00:01"),
        ("12:00:01", "00:00:00"),
        ("06:00:00", "18:00:00"),
    ):
        assert abs(guided_clock_display_offset_sec(effective, validated)) <= 43200.0


def test_validated_folder_seconds_are_not_discarded():
    provenance = timeline_provenance(
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
        recording_start_clock="12:06:20",
        recording_start_clock_source="validated_metadata",
    )
    assert provenance["recording_start_clock"] == "12:06:20"


# ======================================================================
# Continuous RWD
# ======================================================================


def test_continuous_field_autopopulates_from_validated_folder(window, tmp_path):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))

    assert window._guided_recording_start_clock_edit.text() == "12:06:20"
    assert window._guided_recording_start_clock_edit.isEnabled() is True
    assert (
        window._guided_recording_start_clock_help_label.text()
        == GUIDED_RECORDING_START_CLOCK_PREFILLED_MESSAGE
    )
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] == "12:06:20"
    assert values["recording_start_clock_source"] == "validated_metadata"


def test_continuous_user_edit_becomes_user_entered(window, tmp_path):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))

    window._guided_recording_start_clock_edit.setText("12:10:00")
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] == "12:10:00"
    assert values["recording_start_clock_source"] == "user_entered"
    assert (
        window._guided_recording_start_clock_help_label.text()
        == GUIDED_RECORDING_START_CLOCK_USER_ENTERED_MESSAGE
    )


def test_retyping_the_prefilled_value_stays_user_entered(window, tmp_path):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))

    window._guided_recording_start_clock_edit.setText("12:10:00")
    window._guided_recording_start_clock_edit.setText("12:06:20")
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] == "12:06:20"
    assert values["recording_start_clock_source"] == "user_entered"


def test_continuous_fixed_anchor_maps_elapsed_zero_from_inferred_clock(
    window, tmp_path
):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    _set_mode(window, "fixed_daily_anchor")

    values = window._guided_timeline_plan_values()
    day, offset = map_elapsed_coordinate(
        0.0,
        timeline_anchor_mode="fixed_daily_anchor",
        fixed_daily_anchor_clock=values["fixed_daily_anchor_clock"],
        recording_start_clock=values["recording_start_clock"],
    )
    # 12:06:20 - 07:00:00 = 5 h 06 m 20 s after the plotted day begins.
    assert day == 0
    assert offset == 5 * 3600 + 6 * 60 + 20


def test_continuous_fixed_anchor_maps_elapsed_zero_from_edited_clock(
    window, tmp_path
):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    _set_mode(window, "fixed_daily_anchor")
    window._guided_recording_start_clock_edit.setText("12:10:00")

    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock_source"] == "user_entered"
    day, offset = map_elapsed_coordinate(
        0.0,
        timeline_anchor_mode="fixed_daily_anchor",
        fixed_daily_anchor_clock=values["fixed_daily_anchor_clock"],
        recording_start_clock=values["recording_start_clock"],
    )
    assert day == 0
    assert offset == 5 * 3600 + 10 * 60


def test_continuous_civil_maps_first_coordinate_from_midnight(window, tmp_path):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))
    _set_mode(window, "civil")

    values = window._guided_timeline_plan_values()
    day, offset = map_elapsed_coordinate(
        0.0,
        timeline_anchor_mode="civil",
        recording_start_clock=values["recording_start_clock"],
    )
    assert day == 0
    assert offset == 12 * 3600 + 6 * 60 + 20


def test_continuous_elapsed_starts_at_zero_and_keeps_the_field(window, tmp_path):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))
    window._guided_recording_start_clock_edit.setText("12:10:00")
    _set_mode(window, "elapsed")

    # The field is neither cleared nor required, and the accepted plan carries
    # no clock for elapsed placement.
    assert window._guided_recording_start_clock_edit.text() == "12:10:00"
    assert window._guided_recording_start_clock_edit.isHidden() is False
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] is None
    assert values["recording_start_clock_source"] == "not_applicable"
    day, offset = map_elapsed_coordinate(0.0, timeline_anchor_mode="elapsed")
    assert (day, offset) == (0, 0.0)
    assert window._guided_timeline_validation() == (True, "")


# ======================================================================
# Intermittent RWD
# ======================================================================

_FIRST = datetime(2026, 4, 10, 12, 27, 45)
_SESSION_STARTS = [
    datetime(2026, 4, 10, 12, 27, 45),
    datetime(2026, 4, 10, 12, 57, 45),
    datetime(2026, 4, 10, 13, 27, 45),
    datetime(2026, 4, 10, 13, 57, 45),
]


def _session_paths(root: Path) -> list[Path]:
    return [
        root / start.strftime("%Y_%m_%d-%H_%M_%S") / "Fluorescence.csv"
        for start in _SESSION_STARTS
    ]


def _authoritative_sessions(root: Path, *, missing_index: int | None = None):
    paths = _session_paths(root)
    sessions = []
    cache_id = 0
    for index, (start, path) in enumerate(zip(_SESSION_STARTS, paths)):
        missing = index == missing_index
        sessions.append(
            {
                "session_index": index,
                "cache_chunk_id": None if missing else cache_id,
                "source_file": str(path),
                "status": "missing_corrupted" if missing else "valid",
                "expected_start_time": start,
                "expected_duration_sec": 600.0,
                "missing_reason": "",
            }
        )
        if not missing:
            cache_id += 1
    return sessions


def _layout(root: Path, clock, mode="fixed_daily_anchor", missing_index=2):
    sessions = _authoritative_sessions(root, missing_index=missing_index)
    entries = [
        (int(item["cache_chunk_id"]), item["source_file"])
        for item in sessions
        if item["cache_chunk_id"] is not None
    ]
    return compute_day_layout(
        entries,
        None,
        "CH1",
        2,
        timeline_anchor_mode=mode,
        fixed_daily_anchor_clock="07:00" if mode == "fixed_daily_anchor" else None,
        session_index_entries=sessions,
        recording_start_clock=clock,
    )


def _placement_seconds(pds):
    return [
        float(chunk.day_idx) * 86400.0
        + float(chunk.hour_idx) * 3600.0
        + float(chunk.within_hour_offset_sec)
        for chunk in pds.chunks
    ]


def test_intermittent_field_autopopulates_from_first_folder(window, tmp_path):
    _intermittent_window(window, _session_paths(tmp_path))

    assert window._guided_recording_start_clock_edit.text() == "12:27:45"
    assert window._guided_recording_start_clock_edit.isEnabled() is True
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] == "12:27:45"
    assert values["recording_start_clock_source"] == "validated_metadata"


def test_intermittent_unchanged_value_leaves_placement_identical(tmp_path):
    baseline = _placement_seconds(_layout(tmp_path, None))
    prefilled = _placement_seconds(_layout(tmp_path, "12:27:45"))
    assert prefilled == baseline


def test_intermittent_edit_shifts_every_session_by_the_same_offset(tmp_path):
    baseline = _layout(tmp_path, None)
    shifted = _layout(tmp_path, "12:30:00")

    base_seconds = _placement_seconds(baseline)
    shifted_seconds = _placement_seconds(shifted)
    assert [round(b - a, 6) for a, b in zip(base_seconds, shifted_seconds)] == [
        135.0
    ] * len(base_seconds)

    # Spacing between consecutive sessions is unchanged.
    def _gaps(values):
        return [round(b - a, 6) for a, b in zip(values, values[1:])]

    assert _gaps(shifted_seconds) == _gaps(base_seconds)
    assert [c.elapsed_from_start_sec for c in shifted.chunks] == [
        c.elapsed_from_start_sec for c in baseline.chunks
    ]

    # Missing-session identity, index and stored timestamps are untouched.
    assert [c.session_index for c in shifted.chunks] == [0, 1, 2, 3]
    assert [c.status for c in shifted.chunks] == [
        "valid",
        "valid",
        "missing_corrupted",
        "valid",
    ]
    assert [c.expected_start_time for c in shifted.chunks] == _SESSION_STARTS
    assert [c.cache_chunk_id for c in shifted.chunks] == [0, 1, None, 2]

    # The approved missing interval keeps exactly its original width.
    missing_gap = shifted_seconds[2] - shifted_seconds[1]
    assert missing_gap == base_seconds[2] - base_seconds[1] == 1800.0


def _midnight_sessions():
    """Four sessions straddling midnight, first validated start 23:59:30."""
    base = datetime(2026, 4, 10, 23, 59, 30)
    return [base + timedelta(seconds=index * 1800) for index in range(4)]


def _midnight_layout(clock, *, missing_index=2, mode="civil"):
    starts = _midnight_sessions()
    sessions = []
    cache_id = 0
    for index, start in enumerate(starts):
        missing = index == missing_index
        sessions.append(
            {
                "session_index": index,
                "cache_chunk_id": None if missing else cache_id,
                "source_file": "/x/%s/f.csv" % start.strftime("%Y_%m_%d-%H_%M_%S"),
                "status": "missing_corrupted" if missing else "valid",
                "expected_start_time": start,
                "expected_duration_sec": 600.0,
                "missing_reason": "",
            }
        )
        if not missing:
            cache_id += 1
    entries = [
        (int(item["cache_chunk_id"]), item["source_file"])
        for item in sessions
        if item["cache_chunk_id"] is not None
    ]
    return starts, compute_day_layout(
        entries,
        None,
        "CH1",
        2,
        timeline_anchor_mode=mode,
        fixed_daily_anchor_clock="07:00" if mode == "fixed_daily_anchor" else None,
        session_index_entries=sessions,
        recording_start_clock=clock,
    )


def _placement_datetimes(pds, first_placement_dt: datetime):
    """Reconstruct each session's placed wall-clock from its coordinates.

    ``compute_day_layout`` re-bases ``day_idx`` on the day containing the first
    placed session, so the day origin follows the shift.  Reconstructing from
    that origin is what proves the placement, not a raw coordinate delta.
    """
    origin = datetime(
        first_placement_dt.year, first_placement_dt.month, first_placement_dt.day
    )
    return [
        origin
        + timedelta(
            days=int(chunk.day_idx),
            hours=int(chunk.hour_idx),
            seconds=float(chunk.within_hour_offset_sec),
        )
        for chunk in pds.chunks
    ]


def _gaps(values):
    return [round(b - a, 6) for a, b in zip(values, values[1:])]


def test_midnight_override_shifts_by_the_nearest_signed_offset():
    """Correcting 23:59:30 to 00:00:30 is +60 s, not +23 h 59 m."""
    starts, baseline = _midnight_layout(None)
    _starts, shifted = _midnight_layout("00:00:30")
    offset = timedelta(
        seconds=guided_clock_display_offset_sec("00:00:30", "23:59:30")
    )
    assert offset == timedelta(seconds=60)

    # Every session is placed at exactly its validated time plus that offset.
    placed = _placement_datetimes(shifted, starts[0] + offset)
    assert placed == [start + offset for start in starts]

    # Spacing, the internal missing gap and elapsed time are all untouched.
    base_seconds = _placement_seconds(baseline)
    shifted_seconds = _placement_seconds(shifted)
    assert _gaps(shifted_seconds) == _gaps(base_seconds)
    assert shifted_seconds[2] - shifted_seconds[1] == 1800.0
    assert [c.elapsed_from_start_sec for c in shifted.chunks] == [
        c.elapsed_from_start_sec for c in baseline.chunks
    ]
    # One global offset: every coordinate moved by the same amount.
    assert len({round(b - a, 6) for a, b in zip(base_seconds, shifted_seconds)}) == 1

    # Stored authoritative datetimes and identities are unchanged.
    assert [c.expected_start_time for c in shifted.chunks] == starts
    assert [c.session_index for c in shifted.chunks] == [0, 1, 2, 3]
    assert [c.status for c in shifted.chunks] == [
        "valid",
        "valid",
        "missing_corrupted",
        "valid",
    ]


def test_midnight_override_backwards_shifts_negatively():
    """The mirror case: 00:00:30 corrected to 23:59:30 is -60 s."""
    starts = [
        datetime(2026, 4, 11, 0, 0, 30) + timedelta(seconds=i * 1800)
        for i in range(3)
    ]
    sessions = [
        {
            "session_index": i,
            "cache_chunk_id": i,
            "source_file": "/x/%s/f.csv" % s.strftime("%Y_%m_%d-%H_%M_%S"),
            "status": "valid",
            "expected_start_time": s,
            "expected_duration_sec": 600.0,
            "missing_reason": "",
        }
        for i, s in enumerate(starts)
    ]
    entries = [(i, sessions[i]["source_file"]) for i in range(len(starts))]

    def _run(clock):
        return compute_day_layout(
            entries,
            None,
            "CH1",
            2,
            timeline_anchor_mode="civil",
            session_index_entries=sessions,
            recording_start_clock=clock,
        )

    offset = timedelta(
        seconds=guided_clock_display_offset_sec("23:59:30", "00:00:30")
    )
    assert offset == timedelta(seconds=-60)

    baseline = _run(None)
    shifted = _run("23:59:30")
    placed = _placement_datetimes(shifted, starts[0] + offset)
    assert placed == [start + offset for start in starts]
    assert _gaps(_placement_seconds(shifted)) == _gaps(_placement_seconds(baseline))
    assert [c.expected_start_time for c in shifted.chunks] == starts


def test_midnight_override_day_rollover_is_correct():
    """The shifted first session crosses into the next civil day."""
    starts, shifted = _midnight_layout("00:00:30", missing_index=None)
    first = shifted.chunks[0]
    # Validated 2026-04-10 23:59:30 is displayed at 00:00:30 on 2026-04-11,
    # which becomes the plotted-day origin, so it is day 0 hour 0.
    assert (first.day_idx, first.hour_idx, first.within_hour_offset_sec) == (
        0,
        0,
        30.0,
    )
    assert [c.day_idx for c in shifted.chunks] == [0, 0, 0, 0]
    assert [c.hour_idx for c in shifted.chunks] == [0, 0, 1, 1]
    # Without the override the same recording straddles two plotted days.
    _s, baseline = _midnight_layout(None, missing_index=None)
    assert [c.day_idx for c in baseline.chunks] == [0, 1, 1, 1]
    assert [c.expected_start_time for c in shifted.chunks] == starts


def test_midnight_override_under_a_fixed_anchor_places_the_shifted_clocks():
    starts, baseline = _midnight_layout(None, mode="fixed_daily_anchor")
    _s, shifted = _midnight_layout("00:00:30", mode="fixed_daily_anchor")
    offset = timedelta(seconds=60)

    base_seconds = _placement_seconds(baseline)
    shifted_seconds = _placement_seconds(shifted)
    # A 07:00 plotted day contains both 23:59:30 and 00:00:30, so here the
    # anchored coordinate simply advances by the resolved offset.
    assert [round(b - a, 6) for a, b in zip(base_seconds, shifted_seconds)] == [
        60.0
    ] * len(base_seconds)
    assert _gaps(shifted_seconds) == _gaps(base_seconds)
    assert [c.elapsed_from_start_sec for c in shifted.chunks] == [
        c.elapsed_from_start_sec for c in baseline.chunks
    ]
    assert [c.expected_start_time for c in shifted.chunks] == starts
    assert offset == timedelta(seconds=60)


def test_intermittent_fixed_anchor_first_session_uses_effective_clock(tmp_path):
    shifted = _layout(tmp_path, "12:30:00", mode="fixed_daily_anchor")
    first = shifted.chunks[0]
    # 12:30:00 - 07:00:00 = 5 h 30 m after the plotted day begins.
    assert first.day_idx == 0
    assert first.hour_idx == 5
    assert first.within_hour_offset_sec == 1800.0


def test_intermittent_civil_first_session_uses_effective_clock(tmp_path):
    shifted = _layout(tmp_path, "12:30:00", mode="civil")
    first = shifted.chunks[0]
    assert first.day_idx == 0
    assert first.hour_idx == 12
    assert first.within_hour_offset_sec == 1800.0


def test_intermittent_elapsed_keeps_zero_and_authoritative_spacing(tmp_path):
    baseline = _layout(tmp_path, None, mode="elapsed")
    shifted = _layout(tmp_path, "12:30:00", mode="elapsed")
    assert _placement_seconds(shifted) == _placement_seconds(baseline)
    assert shifted.chunks[0].elapsed_from_start_sec == 0.0
    assert [c.elapsed_from_start_sec for c in shifted.chunks] == [
        0.0,
        1800.0,
        3600.0,
        5400.0,
    ]


# ======================================================================
# Intermittent recording identity
# ======================================================================


def _recording_at(root: Path, first_start: datetime, *, count=2) -> list[Path]:
    """Session sources for one recording, first session at ``first_start``."""
    paths = []
    for index in range(count):
        stamp = first_start + timedelta(seconds=index * 1800)
        session = root / stamp.strftime("%Y_%m_%d-%H_%M_%S")
        session.mkdir(parents=True, exist_ok=True)
        paths.append(session / "Fluorescence.csv")
    return paths


def test_intermittent_identity_never_falls_back_to_the_working_directory(
    window, tmp_path
):
    root = tmp_path / "recA"
    paths = _recording_at(root, datetime(2026, 4, 10, 12, 27, 45))
    _intermittent_window(window, paths)

    identity = window._guided_recording_start_clock_recording_identity()
    assert identity != os.path.realpath("")
    assert identity != os.path.realpath(os.getcwd())
    # With no source root reported, the first authoritative session source is
    # the identity.
    assert identity == os.path.realpath(str(paths[0]))

    # When discovery reports the accepted canonical source root, that is used.
    _intermittent_window(window, paths, input_dir=str(root))
    assert window._guided_recording_start_clock_recording_identity() == (
        os.path.realpath(str(root))
    )


def test_intermittent_edit_survives_refresh_of_the_same_recording(
    window, tmp_path
):
    root = tmp_path / "recA"
    paths = _recording_at(root, datetime(2026, 4, 10, 12, 27, 45))
    _intermittent_window(window, paths, input_dir=str(root))
    assert window._guided_recording_start_clock_edit.text() == "12:27:45"

    window._guided_recording_start_clock_edit.setText("12:30:00")
    window._sync_guided_recording_visibility()
    window._refresh_guided_timeline_controls()

    assert window._guided_recording_start_clock_edit.text() == "12:30:00"
    assert window._guided_recording_start_clock_user_edited is True
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] == "12:30:00"
    assert values["recording_start_clock_source"] == "user_entered"


def test_selecting_a_different_intermittent_recording_replaces_the_prefill(
    window, tmp_path
):
    first = _recording_at(tmp_path / "recA", datetime(2026, 4, 10, 12, 27, 45))
    _intermittent_window(window, first, input_dir=str(tmp_path / "recA"))
    window._guided_recording_start_clock_edit.setText("12:30:00")
    assert window._guided_recording_start_clock_user_edited is True

    second = _recording_at(tmp_path / "recB", datetime(2026, 5, 2, 9, 5, 30))
    _intermittent_window(window, second, input_dir=str(tmp_path / "recB"))

    assert window._guided_recording_start_clock_edit.text() == "09:05:30"
    assert window._guided_recording_start_clock_user_edited is False
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock_source"] == "validated_metadata"


@pytest.mark.parametrize("report_source_root", [True, False])
def test_two_recordings_sharing_a_first_session_clock_are_distinguished(
    window, tmp_path, report_source_root
):
    """Same clock, different recordings: the identity must still differ."""
    root_a = tmp_path / "recA"
    root_b = tmp_path / "recB"
    # Different dates, identical first-session clock time.
    paths_a = _recording_at(root_a, datetime(2026, 4, 10, 12, 27, 45))
    paths_b = _recording_at(root_b, datetime(2026, 7, 22, 12, 27, 45))

    _intermittent_window(
        window, paths_a, input_dir=str(root_a) if report_source_root else None
    )
    key_a = window._guided_recording_start_clock_prefill_key()
    window._guided_recording_start_clock_edit.setText("13:00:00")
    assert window._guided_recording_start_clock_user_edited is True

    _intermittent_window(
        window, paths_b, input_dir=str(root_b) if report_source_root else None
    )
    key_b = window._guided_recording_start_clock_prefill_key()

    # Both recordings share the clock component, so only a real recording
    # identity can tell them apart.
    assert key_a is not None and key_b is not None
    assert key_a[1] == key_b[1] == "12:27:45"
    assert key_a[0] != key_b[0]
    assert key_a != key_b
    # The second recording therefore re-prefills and drops the edit.
    assert window._guided_recording_start_clock_edit.text() == "12:27:45"
    assert window._guided_recording_start_clock_user_edited is False


def test_stale_intermittent_discovery_cannot_restore_the_previous_clock(
    window, tmp_path
):
    previous = _recording_at(tmp_path / "recA", datetime(2026, 4, 10, 12, 27, 45))
    _intermittent_window(window, previous, input_dir=str(tmp_path / "recA"))
    assert window._guided_recording_start_clock_edit.text() == "12:27:45"

    newer = _recording_at(tmp_path / "recB", datetime(2026, 5, 2, 9, 5, 30))
    _intermittent_window(window, newer, input_dir=str(tmp_path / "recB"))
    assert window._guided_recording_start_clock_edit.text() == "09:05:30"

    window._guided_discovery_generation += 1
    window._on_guided_roi_discovery_succeeded(
        _intermittent_discovery(previous, input_dir=str(tmp_path / "recA")),
        generation=0,
    )

    assert window._guided_recording_start_clock_edit.text() == "09:05:30"


# ======================================================================
# Source changes
# ======================================================================


def test_selecting_a_new_recording_replaces_the_prefill_and_resets_edited_state(
    window, tmp_path
):
    first = tmp_path / "2026_03_16-12_06_20"
    first.mkdir()
    _continuous_window(window, str(first))
    window._guided_recording_start_clock_edit.setText("12:10:00")
    assert window._guided_recording_start_clock_user_edited is True

    second = tmp_path / "2026_03_17-08_15_42"
    second.mkdir()
    _continuous_window(window, str(second))

    assert window._guided_recording_start_clock_edit.text() == "08:15:42"
    assert window._guided_recording_start_clock_user_edited is False
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock_source"] == "validated_metadata"


def test_an_edit_survives_a_refresh_of_the_same_recording(window, tmp_path):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))
    window._guided_recording_start_clock_edit.setText("12:10:00")

    window._sync_guided_recording_visibility()

    assert window._guided_recording_start_clock_edit.text() == "12:10:00"
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock_source"] == "user_entered"


def test_stale_discovery_cannot_overwrite_a_newer_selection(window, tmp_path):
    newer = tmp_path / "2026_03_17-08_15_42"
    newer.mkdir()
    _continuous_window(window, str(newer))
    assert window._guided_recording_start_clock_edit.text() == "08:15:42"

    stale = {
        "resolved_format": "rwd",
        "acquisition_mode": "continuous",
        "sessions": [],
        "continuous_recording_folder": str(tmp_path / "2026_03_16-12_06_20"),
    }
    window._guided_discovery_generation += 1
    window._on_guided_roi_discovery_succeeded(stale, generation=0)

    assert window._guided_recording_start_clock_edit.text() == "08:15:42"


def test_no_validated_timestamp_leaves_the_field_empty_and_required(
    window, tmp_path
):
    folder = tmp_path / "recording_without_timestamp"
    folder.mkdir()
    _continuous_window(window, str(folder))

    assert window._guided_recording_start_clock_edit.text() == ""
    assert (
        window._guided_recording_start_clock_help_label.text()
        == GUIDED_TIMELINE_START_CLOCK_REQUIRED_MESSAGE
    )
    ok, message = window._guided_timeline_validation()
    assert ok is False
    assert message == GUIDED_TIMELINE_START_CLOCK_REQUIRED_MESSAGE
    values = window._guided_timeline_plan_values()
    assert values["recording_start_clock"] is None
    assert values["recording_start_clock_source"] == "not_applicable"


def test_filesystem_modification_time_is_never_used(window, tmp_path):
    folder = tmp_path / "recording_without_timestamp"
    folder.mkdir()
    # A real, recently modified folder still yields nothing: only the validated
    # RWD naming convention may supply the clock.
    (folder / "Fluorescence.csv").write_text("TimeStamp\n0\n", encoding="utf-8")
    _continuous_window(window, str(folder))

    assert window._guided_authoritative_first_datetime() is None
    assert window._guided_recording_start_clock_edit.text() == ""


# ======================================================================
# Invalidation boundary
# ======================================================================


def test_continuous_saved_artifact_coordinates_follow_the_effective_clock():
    """Saved elapsed samples map through the accepted timeline contract."""
    elapsed = np.array([0.0, 600.0, 1200.0])
    inferred = continuous_plot_coordinates(
        elapsed,
        timeline_provenance(
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock="07:00",
            recording_start_clock="12:06:20",
            recording_start_clock_source="validated_metadata",
        ),
    )
    edited = continuous_plot_coordinates(
        elapsed,
        timeline_provenance(
            timeline_anchor_mode="fixed_daily_anchor",
            fixed_daily_anchor_clock="07:00",
            recording_start_clock="12:10:00",
            recording_start_clock_source="user_entered",
        ),
    )
    assert inferred[0] == 5 * 3600 + 6 * 60 + 20
    assert edited[0] == 5 * 3600 + 10 * 60
    # Only placement moves: the elapsed spacing of the saved samples is intact.
    assert np.allclose(np.diff(inferred), np.diff(elapsed))
    assert np.allclose(np.diff(edited), np.diff(elapsed))


# ======================================================================
# Natural path: real continuous discovery
# ======================================================================


def _continuous_rows(count):
    from tests.test_guided_continuous_rwd_correction_pass_persistence import (
        _values,
    )

    indices = np.arange(count, dtype=float)
    time_s, control, signal = _values(indices)
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(count):
        lines.append(
            "%.4f,%.12f,%.12f,%.12f,%.12f\n"
            % (
                time_s[index],
                control[index, 0],
                signal[index, 0],
                control[index, 1],
                signal[index, 1],
            )
        )
    return lines


def _pump(qapp, predicate, timeout_ms=180_000):
    deadline = QDeadlineTimer(timeout_ms)
    while predicate() and not deadline.hasExpired():
        qapp.processEvents()


def test_real_continuous_discovery_prefills_and_needs_no_manual_entry(
    window, qapp, tmp_path
):
    """Drives the real discovery a scientist triggers with Select ROIs."""
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir(parents=True)
    (folder / "Fluorescence.csv").write_text(
        "".join(_continuous_rows(12_000)), encoding="utf-8", newline=""
    )
    (folder / "Events.csv").write_text(
        "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
    )

    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText("auto")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData(GUIDED_STRUCTURE_CHOICE_AUTO)
    )
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(tmp_path / "output"))
    window._on_guided_discover_rois()
    _pump(qapp, lambda: window._guided_roi_discovery_running)
    _pump(
        qapp,
        lambda: getattr(window, "_guided_roi_discovery_thread", None) is not None,
        20_000,
    )

    assert window._guided_effective_acquisition_mode() == "continuous"
    # The validated recording folder supplied the clock; nothing was typed.
    assert window._guided_recording_start_clock_edit.text() == "12:06:20"
    assert window._guided_recording_structure_readiness() == (
        True,
        "Recording structure is ready.",
    )
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.execution_intent.recording_start_clock == "12:06:20"
    assert plan.execution_intent.recording_start_clock_source == (
        "validated_metadata"
    )
    assert window._guided_timeline_review_lines(plan)[:4] == [
        "Time display: Fixed daily anchor",
        "Start of plotted day: 07:00",
        "Clock time at recording start: 12:06:20",
        "Source: Validated RWD recording timestamp",
    ]


def test_editing_the_clock_refreshes_only_timing_dependent_state(
    window, tmp_path, monkeypatch
):
    folder = tmp_path / "2026_03_16-12_06_20"
    folder.mkdir()
    _continuous_window(window, str(folder))

    # Evidence that must survive an edit to a placement-only field.
    window._guided_preview_last_result = {"status": "ok", "roi": "CH1"}
    window._guided_signal_f0_last_result = {"status": "ok", "roi": "CH1"}
    window._guided_strategy_choices = {"key": {"strategy": "global_linear_regression"}}
    discovery_before = dict(window._discovery_cache)

    calls: list[str] = []
    for name in (
        "_start_guided_roi_discovery",
        "_start_guided_continuous_rwd_recording_check",
        "_refresh_guided_diagnostics_panel",
    ):
        if hasattr(window, name):
            monkeypatch.setattr(
                window,
                name,
                lambda *a, _n=name, **k: calls.append(_n),
            )

    revision_before = window._guided_backend_validation_revision
    window._guided_recording_start_clock_edit.setText("12:10:00")

    # Timing-dependent state refreshed.
    assert window._guided_backend_validation_revision > revision_before
    assert window._guided_backend_validation_stale_reason == (
        "timeline placement changed"
    )

    # Source, correction and detection evidence intact; nothing rescanned.
    assert window._discovery_cache == discovery_before
    assert window._guided_preview_last_result == {"status": "ok", "roi": "CH1"}
    assert window._guided_signal_f0_last_result == {"status": "ok", "roi": "CH1"}
    assert window._guided_strategy_choices == {
        "key": {"strategy": "global_linear_regression"}
    }
    assert calls == []
