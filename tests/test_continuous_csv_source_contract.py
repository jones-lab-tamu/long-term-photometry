"""The v1 continuous generic CSV source contract.

One bounded recording is generated once with the production writer, and each
rejection case is a small mutation of that same source.
"""

from pathlib import Path

import numpy as np
import pytest

from gui.synthetic_demo_generator import (
    GUIDED_CONTINUOUS_DEMO_FILE_NAME,
    GUIDED_CONTINUOUS_DEMO_FS_HZ,
    generate_guided_continuous_demo,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    build_guided_continuous_rwd_target_grid,
)
from photometry_pipeline.io.csv_continuous_source import (
    ContinuousCsvRoiSelection,
    candidate_csv_files,
    inspect_continuous_csv_recording,
)


BOUNDED_DURATION_SEC = 1200.0
ROI_SELECTIONS = [
    ContinuousCsvRoiSelection("ROI1", "ROI1_Signal", "ROI1_Reference"),
    ContinuousCsvRoiSelection("ROI2", "ROI2_Signal", "ROI2_Reference"),
]


@pytest.fixture(scope="module")
def bounded_source(tmp_path_factory) -> Path:
    result = generate_guided_continuous_demo(
        tmp_path_factory.mktemp("csv_contract"), _duration_sec=BOUNDED_DURATION_SEC
    )
    assert result.success, result.message
    return result.input_dir / GUIDED_CONTINUOUS_DEMO_FILE_NAME


def _inspect(path: Path, *, time_column="ElapsedSeconds", time_unit="seconds", rois=None):
    return inspect_continuous_csv_recording(
        path,
        time_column=time_column,
        time_unit=time_unit,
        roi_selections=ROI_SELECTIONS if rois is None else rois,
    )


def _mutate(source: Path, destination: Path, change) -> Path:
    """Write a copy of the bounded source with one small change applied."""
    lines = source.read_text(encoding="utf-8").splitlines()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(change(lines)) + "\n", encoding="utf-8")
    return destination


# --------------------------------------------------------------------------
# Accepted source
# --------------------------------------------------------------------------


def test_seconds_source_is_accepted_with_the_expected_cadence(bounded_source):
    inspection = _inspect(bounded_source)
    assert inspection.status == "completed"
    assert inspection.outcome_category == "inspection_completed"
    assert inspection.parser_facts.header_row_index == 0
    assert inspection.parser_facts.timestamp_unit == "seconds"
    assert inspection.parser_facts.timestamp_scale_to_seconds == 1.0
    cadence = inspection.time_axis.nominal_cadence_seconds
    assert cadence == pytest.approx(1.0 / GUIDED_CONTINUOUS_DEMO_FS_HZ)
    assert inspection.time_axis.duplicate_timestamp_count == 0
    assert inspection.time_axis.backward_timestamp_count == 0
    assert inspection.channels.nonfinite_selected_value_count == 0
    assert [pair.roi_id for pair in inspection.channels.roi_pairs] == ["ROI1", "ROI2"]


def test_milliseconds_source_scales_to_the_same_cadence(bounded_source, tmp_path):
    def to_milliseconds(lines):
        header = lines[0].replace("ElapsedSeconds", "ElapsedMs")
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            rows.append(",".join([f"{float(parts[0]) * 1000.0:.1f}"] + parts[1:]))
        return [header] + rows

    source = _mutate(bounded_source, tmp_path / "ms" / "rec.csv", to_milliseconds)
    inspection = _inspect(source, time_column="ElapsedMs", time_unit="milliseconds")

    assert inspection.status == "completed"
    assert inspection.parser_facts.timestamp_scale_to_seconds == 0.001
    seconds = _inspect(bounded_source)
    assert inspection.time_axis.nominal_cadence_seconds == pytest.approx(
        seconds.time_axis.nominal_cadence_seconds
    )
    assert inspection.time_axis.measured_duration_seconds == pytest.approx(
        seconds.time_axis.measured_duration_seconds
    )


def test_normalized_recording_is_custom_tabular_continuous(bounded_source):
    inspection = _inspect(bounded_source)
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    assert recording.source_format == "custom_tabular"
    assert recording.acquisition_mode == "continuous"
    assert recording.source.header_row_index == 0
    assert recording.source.selected_time_column == "ElapsedSeconds"
    assert tuple(recording.roi.included_roi_ids) == ("ROI1", "ROI2")


def test_real_target_grid_builds_from_the_generic_csv_description(bounded_source):
    inspection = _inspect(bounded_source)
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=bounded_source
    )
    grid = build_guided_continuous_rwd_target_grid(recording, continuity)
    assert grid.target_sample_count == int(
        BOUNDED_DURATION_SEC * GUIDED_CONTINUOUS_DEMO_FS_HZ
    )
    assert grid.cadence_seconds_numerator == 1
    assert grid.cadence_seconds_denominator == GUIDED_CONTINUOUS_DEMO_FS_HZ


def test_candidate_csv_files_counts_only_csv_files(tmp_path):
    assert candidate_csv_files(tmp_path) == ()
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert candidate_csv_files(tmp_path) == ()
    (tmp_path / "continuous_recording.csv").write_text("a\n1\n", encoding="utf-8")
    assert [p.name for p in candidate_csv_files(tmp_path)] == ["continuous_recording.csv"]
    (tmp_path / "second.csv").write_text("a\n1\n", encoding="utf-8")
    assert len(candidate_csv_files(tmp_path)) == 2
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.csv").write_text("a\n1\n", encoding="utf-8")
    # Never a recursive search.
    assert len(candidate_csv_files(tmp_path)) == 2


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------


def test_duplicate_timestamps_are_rejected(bounded_source, tmp_path):
    def duplicate(lines):
        rows = list(lines)
        parts = rows[6].split(",")
        parts[0] = rows[5].split(",")[0]
        rows[6] = ",".join(parts)
        return rows

    inspection = _inspect(_mutate(bounded_source, tmp_path / "dup" / "rec.csv", duplicate))
    assert inspection.status == "failed"
    assert inspection.outcome_category == "duplicate_timestamps_present"


def test_decreasing_timestamps_are_rejected(bounded_source, tmp_path):
    def decreasing(lines):
        rows = list(lines)
        parts = rows[8].split(",")
        parts[0] = "0.125"
        rows[8] = ",".join(parts)
        return rows

    inspection = _inspect(
        _mutate(bounded_source, tmp_path / "back" / "rec.csv", decreasing)
    )
    assert inspection.status == "failed"
    assert inspection.outcome_category == "backward_timestamps_present"


def test_nonfinite_selected_values_are_rejected(bounded_source, tmp_path):
    def nonfinite(lines):
        rows = list(lines)
        parts = rows[10].split(",")
        parts[1] = "nan"
        rows[10] = ",".join(parts)
        return rows

    inspection = _inspect(
        _mutate(bounded_source, tmp_path / "nan" / "rec.csv", nonfinite)
    )
    assert inspection.status == "failed"
    assert inspection.outcome_category == "selected_channel_parse_failure"


def test_nonnumeric_selected_values_are_rejected(bounded_source, tmp_path):
    def nonnumeric(lines):
        rows = list(lines)
        parts = rows[12].split(",")
        parts[2] = "n/a"
        rows[12] = ",".join(parts)
        return rows

    inspection = _inspect(
        _mutate(bounded_source, tmp_path / "text" / "rec.csv", nonnumeric)
    )
    assert inspection.status == "failed"
    assert inspection.outcome_category == "selected_channel_parse_failure"


def test_materially_irregular_cadence_is_rejected_by_the_continuous_authority(
    bounded_source, tmp_path
):
    """A long gap passes inspection but the continuity authority refuses it."""

    def gap(lines):
        rows = [lines[0]]
        for index, line in enumerate(lines[1:]):
            parts = line.split(",")
            shift = 30.0 if index >= 400 else 0.0
            parts[0] = f"{float(parts[0]) + shift:.3f}"
            rows.append(",".join(parts))
        return rows

    source = _mutate(bounded_source, tmp_path / "gap" / "rec.csv", gap)
    inspection = _inspect(source)
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=source
    )
    assert continuity.outcome != "passed"
    assert continuity.material_long_interval_count >= 1


def test_signal_and_reference_cannot_be_the_same_column(bounded_source):
    inspection = _inspect(
        bounded_source,
        rois=[ContinuousCsvRoiSelection("ROI1", "ROI1_Signal", "ROI1_Signal")],
    )
    assert inspection.status == "failed"
    assert inspection.outcome_category == "signal_reference_identical"
    assert "same column" in inspection.scientist_summary


def test_a_column_cannot_be_reused_across_roles(bounded_source):
    inspection = _inspect(
        bounded_source,
        rois=[
            ContinuousCsvRoiSelection("ROI1", "ROI1_Signal", "ROI1_Reference"),
            ContinuousCsvRoiSelection("ROI2", "ROI1_Signal", "ROI2_Reference"),
        ],
    )
    assert inspection.status == "failed"
    assert inspection.outcome_category == "selected_column_reused"


def test_the_time_column_cannot_double_as_a_fluorescence_column(bounded_source):
    inspection = _inspect(
        bounded_source,
        rois=[ContinuousCsvRoiSelection("ROI1", "ElapsedSeconds", "ROI1_Reference")],
    )
    assert inspection.status == "failed"
    assert inspection.outcome_category == "selected_column_reused"


def test_missing_selected_columns_are_rejected(bounded_source):
    missing_time = _inspect(bounded_source, time_column="NotAColumn")
    assert missing_time.status == "failed"
    assert missing_time.outcome_category == "selected_column_missing"

    missing_roi = _inspect(
        bounded_source,
        rois=[ContinuousCsvRoiSelection("ROI1", "Nope_Signal", "ROI1_Reference")],
    )
    assert missing_roi.status == "failed"
    assert missing_roi.outcome_category == "selected_column_missing"


def test_unsupported_time_unit_is_rejected(bounded_source):
    inspection = _inspect(bounded_source, time_unit="minutes")
    assert inspection.status == "failed"
    assert inspection.outcome_category == "unsupported_time_unit"


def test_no_selected_rois_is_rejected(bounded_source):
    inspection = _inspect(bounded_source, rois=[])
    assert inspection.status == "failed"
    assert inspection.outcome_category == "no_selected_rois"
