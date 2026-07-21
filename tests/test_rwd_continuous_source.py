from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from photometry_pipeline.io import rwd_continuous_source as source
from photometry_pipeline.io.rwd_continuous_source import (
    INSPECTION_CONTRACT_VERSION,
    MINIMUM_DURATION_SEC,
    inspect_continuous_rwd_acquisition_folder,
)


def _write_rwd(
    folder: Path,
    *,
    name: str = "Fluorescence.csv",
    columns: tuple[str, ...] = ("TimeStamp", "Events", "CH1-410", "CH1-470"),
    rows: tuple[tuple[object, ...], ...] | None = None,
    preamble: tuple[str, ...] = (),
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = tuple((index * 200.0, "", 1.0 + index, 2.0 + index) for index in range(4))
    path = folder / name
    lines = [*preamble, ",".join(columns)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _inspect(tmp_path: Path, **kwargs):
    folder = tmp_path / "2026_03_16-12_06_20"
    path = _write_rwd(folder, **kwargs)
    return folder, path, inspect_continuous_rwd_acquisition_folder(folder)


def test_valid_direct_acquisition_folder_returns_immutable_facts(tmp_path: Path):
    folder, path, result = _inspect(tmp_path)
    assert result.status == "completed"
    assert result.outcome_category == "inspection_completed"
    assert result.contract_version == INSPECTION_CONTRACT_VERSION
    assert result.source_stable is True
    assert result.full_file_passes == 2
    assert (
        result.time_axis.normalized_last_seconds
        == result.time_axis.measured_duration_seconds
    )
    assert result.source_identity.selected_folder_canonical == str(folder.resolve())
    assert result.source_identity.fluorescence_path_canonical == str(path.resolve())
    with pytest.raises(Exception):
        result.status = "changed"  # type: ignore[misc]


def test_selected_folder_missing(tmp_path: Path):
    result = inspect_continuous_rwd_acquisition_folder(tmp_path / "missing")
    assert result.outcome_category == "selected_path_missing"


def test_selected_path_is_a_file(tmp_path: Path):
    path = tmp_path / "file"
    path.write_text("x", encoding="utf-8")
    result = inspect_continuous_rwd_acquisition_folder(path)
    assert result.outcome_category == "selected_path_not_directory"


def test_missing_fluorescence_csv(tmp_path: Path):
    result = inspect_continuous_rwd_acquisition_folder(tmp_path)
    assert result.outcome_category == "fluorescence_csv_missing"
    assert "directly contains" in result.scientist_summary


@pytest.mark.skipif(source.os.name != "nt", reason="Windows filename rule")
def test_one_case_insensitive_filename_match_preserves_real_path(tmp_path: Path):
    folder, path, result = _inspect(tmp_path, name="fluorescence.CSV")
    assert result.status == "completed"
    assert Path(result.source_identity.fluorescence_path_canonical).name == path.name


def test_multiple_case_variant_candidates_are_ambiguous(tmp_path: Path, monkeypatch):
    first = tmp_path / "Fluorescence.csv"
    second = tmp_path / "FLUORESCENCE.CSV"
    monkeypatch.setattr(source.os, "name", "nt")
    monkeypatch.setattr(source, "_folder_entries", lambda _path: (first, second))
    result = inspect_continuous_rwd_acquisition_folder(tmp_path)
    assert result.outcome_category == "ambiguous_fluorescence_candidates"


def test_no_recursive_child_folder_discovery(tmp_path: Path):
    _write_rwd(tmp_path / "child")
    result = inspect_continuous_rwd_acquisition_folder(tmp_path)
    assert result.outcome_category == "fluorescence_csv_missing"


def test_timestamp_header_and_preamble_row_are_reported(tmp_path: Path):
    rows = tuple((index * 50.0, "", 1 + index, 2 + index) for index in range(4))
    _, _, result = _inspect(
        tmp_path,
        preamble=("Fps:40,Led410Enable:true,Led470Enable:true",),
        rows=rows,
    )
    assert result.parser_facts.header_row_index == 1
    assert result.parser_facts.time_column == "TimeStamp"
    assert result.parser_facts.timestamp_unit == "milliseconds"
    assert result.parser_facts.timestamp_scale_to_seconds == 0.001


def test_supported_time_s_column(tmp_path: Path):
    columns = ("Time(s)", "CH1-410", "CH1-470")
    rows = ((0, 1, 2), (1, 2, 3), (2, 3, 4))
    _, _, result = _inspect(tmp_path, columns=columns, rows=rows)
    assert result.parser_facts.time_column == "Time(s)"
    assert result.parser_facts.timestamp_unit == "seconds"


@pytest.mark.parametrize(
    ("time_column", "timestamps", "expected_unit", "expected_duration"),
    [
        ("Time(s)", (1000.0, 1001.0, 1002.5), "seconds", 2.5),
        ("TimeStamp", (1000.0, 1100.0, 1250.0), "milliseconds", 0.25),
    ],
)
def test_nonzero_raw_origin_normalizes_recording_elapsed_time_to_zero(
    tmp_path: Path,
    time_column: str,
    timestamps: tuple[float, ...],
    expected_unit: str,
    expected_duration: float,
):
    rows = tuple((timestamp, 1, 2) for timestamp in timestamps)
    _, _, result = _inspect(
        tmp_path,
        columns=(time_column, "CH1-410", "CH1-470"),
        rows=rows,
    )

    assert result.parser_facts.timestamp_unit == expected_unit
    assert result.time_axis.raw_first_timestamp == timestamps[0]
    assert result.time_axis.raw_last_timestamp == timestamps[-1]
    assert result.time_axis.normalized_first_seconds == 0.0
    assert result.time_axis.measured_duration_seconds == expected_duration
    assert (
        result.time_axis.normalized_last_seconds
        == result.time_axis.measured_duration_seconds
    )


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (("Elapsed", "CH1-410", "CH1-470"), "no_supported_time_column"),
        (("Time(s)", "TimeStamp", "CH1-410", "CH1-470"), "unsupported_or_ambiguous_header"),
    ],
)
def test_missing_or_ambiguous_time_column(tmp_path: Path, columns, expected):
    rows = tuple(tuple(range(len(columns))) for _ in range(3))
    _, _, result = _inspect(tmp_path, columns=columns, rows=rows)
    assert result.outcome_category == expected


def test_valid_multiple_roi_pairs(tmp_path: Path):
    columns = ("TimeStamp", "CH2-410", "CH2-470", "CH1-410", "CH1-470")
    rows = ((0, 1, 2, 3, 4), (1, 2, 3, 4, 5), (2, 3, 4, 5, 6))
    _, _, result = _inspect(tmp_path, columns=columns, rows=rows)
    assert tuple(pair.roi_id for pair in result.channels.roi_pairs) == ("CH1", "CH2")


def test_incomplete_roi_pair_is_reported(tmp_path: Path):
    columns = ("TimeStamp", "CH1-410", "CH1-470", "CH2-410")
    rows = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
    _, _, result = _inspect(tmp_path, columns=columns, rows=rows)
    assert result.outcome_category == "inconsistent_roi_channel_structure"
    assert result.channels.unmatched_channel_columns == ("CH2-410",)


@pytest.mark.parametrize(
    ("bad_value", "field"),
    [("not-a-number", "nonnumeric_selected_value_count"), ("nan", "nonfinite_selected_value_count")],
)
def test_invalid_selected_channel_value(tmp_path: Path, bad_value: str, field: str):
    rows = ((0, 1, 2), (1, bad_value, 3), (2, 3, 4))
    _, _, result = _inspect(
        tmp_path,
        columns=("TimeStamp", "CH1-410", "CH1-470"),
        rows=rows,
    )
    assert result.outcome_category == "selected_channel_parse_failure"
    assert getattr(result.channels, field) == 1


@pytest.mark.parametrize(
    ("times", "category", "field"),
    [
        ((0, 1, 1, 2), "duplicate_timestamps_present", "duplicate_timestamp_count"),
        ((0, 2, 1, 3), "backward_timestamps_present", "backward_timestamp_count"),
    ],
)
def test_nonmonotonic_timestamps(tmp_path: Path, times, category, field):
    rows = tuple((value, 1, 2) for value in times)
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.outcome_category == category
    assert getattr(result.time_axis, field) == 1


@pytest.mark.parametrize("bad", ["bad", "nan"])
def test_nonnumeric_or_nonfinite_time(tmp_path: Path, bad: str):
    rows = ((0, 1, 2), (bad, 2, 3), (2, 3, 4))
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.outcome_category == "nonnumeric_or_nonfinite_time"


def test_small_realistic_cadence_jitter_is_evidence_not_failure(tmp_path: Path):
    times = (0.0, 0.0501, 0.1000, 0.1502, 0.2001)
    rows = tuple((value, 1, 2) for value in times)
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.status == "completed"
    assert result.time_axis.unusually_long_interval_count == 0
    assert result.time_axis.unusually_short_interval_count == 0


def test_long_and_short_intervals_are_reported_as_evidence(tmp_path: Path):
    times = (0.0, 1.0, 2.0, 2.1, 3.1, 8.1)
    rows = tuple((value, 1, 2) for value in times)
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.status == "completed"
    assert result.time_axis.unusually_long_interval_count == 1
    assert result.time_axis.unusually_short_interval_count == 1
    assert result.time_axis.largest_unusual_intervals[0].estimated_expected_sample_multiple == 5.0


def test_duration_below_product_minimum_is_classified_without_run_readiness(tmp_path: Path):
    rows = ((0, 1, 2), (100, 2, 3), (200, 3, 4))
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.status == "completed"
    assert result.time_axis.minimum_duration_seconds == MINIMUM_DURATION_SEC
    assert result.time_axis.duration_product_classification == "below_product_minimum"
    assert any(item.category == "below_minimum_duration" for item in result.findings)


def test_final_row_is_included(tmp_path: Path):
    rows = ((10, 1, 2), (20, 2, 3), (37, 3, 4))
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.time_axis.total_data_row_count == 3
    assert result.time_axis.raw_last_timestamp == 37


def test_no_data_rows(tmp_path: Path):
    _, _, result = _inspect(tmp_path, rows=())
    assert result.outcome_category == "no_usable_rows"


def test_column_count_drift_is_fatal_and_counted(tmp_path: Path):
    rows = ((0, 1, 2), (1, 2), (2, 3, 4))
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.outcome_category == "inconsistent_roi_channel_structure"
    assert result.channels.malformed_row_count == 1


def test_file_change_during_inspection_fails_closed(tmp_path: Path, monkeypatch):
    folder = tmp_path / "acquisition"
    path = _write_rwd(folder)
    original = source._facts
    first = original(path)
    calls = 0

    def changing_facts(candidate):
        nonlocal calls
        calls += 1
        return first if calls == 1 else replace(first, mtime_ns=first.mtime_ns + 1)

    monkeypatch.setattr(source, "_facts", changing_facts)
    result = inspect_continuous_rwd_acquisition_folder(folder)
    assert result.outcome_category == "source_changed_during_inspection"
    assert result.source_stable is False


def test_result_is_deterministic_for_unchanged_source(tmp_path: Path):
    folder, _, first = _inspect(tmp_path)
    second = inspect_continuous_rwd_acquisition_folder(folder)
    assert first == second


def test_source_sha256_size_and_stable_identity(tmp_path: Path):
    _, path, result = _inspect(tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.source_identity.sha256 == expected
    assert result.source_identity.file_size_bytes == path.stat().st_size
    assert len(result.source_identity.stable_source_identity) == 64


def test_bounded_interval_sample_on_larger_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(source, "DT_SAMPLE_CAPACITY", 32)
    rows = tuple((index * 0.05, 1, 2) for index in range(2000))
    _, _, result = _inspect(tmp_path, columns=("TimeStamp", "CH1-410", "CH1-470"), rows=rows)
    assert result.time_axis.positive_interval_count == 1999
    assert result.time_axis.quantile_sample_count == 32


def test_inspection_creates_no_files_or_caches(tmp_path: Path):
    folder = tmp_path / "acquisition"
    _write_rwd(folder)
    before = sorted(path.relative_to(folder) for path in folder.rglob("*"))
    inspect_continuous_rwd_acquisition_folder(folder)
    after = sorted(path.relative_to(folder) for path in folder.rglob("*"))
    assert after == before == [Path("Fluorescence.csv")]


def test_interrupted_inspection_returns_no_partial_success(tmp_path: Path):
    folder = tmp_path / "acquisition"
    _write_rwd(folder)
    result = inspect_continuous_rwd_acquisition_folder(folder, cancellation_check=lambda: True)
    assert result.outcome_category == "inspection_interrupted"
    assert result.source_identity is None
