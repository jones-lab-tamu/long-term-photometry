import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools import convert_rwd_to_csv_fixture as converter


SESSION_NAMES = (
    "2025_01_01-00_00_00",
    "2025_01_01-00_30_00",
    "2025_01_01-01_00_00",
)


def _write_rwd_session(
    root: Path,
    session_name: str,
    *,
    rois: tuple[str, ...] = ("CH1", "CH2"),
    offset: float = 0.0,
) -> Path:
    session_dir = root / session_name
    session_dir.mkdir(parents=True)
    source = session_dir / "fluorescence.csv"
    columns = ["TimeStamp"]
    for roi in rois:
        columns.extend((f"{roi}-410", f"{roi}-470"))
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for sample_index, time_sec in enumerate((0.0, 0.5, 1.0, 1.5)):
            row = [time_sec]
            for roi_index, _roi in enumerate(rois, start=1):
                reference = offset + (roi_index * 100.0) + sample_index
                signal = offset + (roi_index * 100.0) + 20.0 + sample_index
                row.extend((reference, signal))
            writer.writerow(row)
    return source


def _make_recording(
    tmp_path: Path,
    *,
    inventories: tuple[tuple[str, ...], ...] | None = None,
) -> tuple[Path, list[Path]]:
    root = tmp_path / "rwd_source"
    root.mkdir()
    inventories = inventories or (("CH1", "CH2"),) * len(SESSION_NAMES)
    sources = []
    # Deliberately create sessions out of order; discovery owns chronology.
    for index in (2, 0, 1):
        sources.append(
            _write_rwd_session(
                root,
                SESSION_NAMES[index],
                rois=inventories[index],
                offset=index * 1000.0,
            )
        )
    return root, sources


def _read_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    return header, rows


def _fingerprints(paths: list[Path]) -> dict[Path, tuple[str, int]]:
    return {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in paths
    }


def test_full_conversion_preserves_order_values_source_and_manifest(tmp_path: Path):
    source_root, source_files = _make_recording(tmp_path)
    source_before = _fingerprints(source_files)
    output = tmp_path / "fixture"

    manifest = converter.convert_rwd_to_csv_fixture(
        input_dir=str(source_root),
        output_dir=str(output),
    )

    csv_files = sorted(output.glob("*.csv"))
    assert [path.name for path in csv_files] == [
        "session_0001.csv",
        "session_0002.csv",
        "session_0003.csv",
    ]
    parsed = [_read_csv(path) for path in csv_files]
    assert {tuple(header) for header, _rows in parsed} == {
        (
            "ElapsedSeconds",
            "CH1_signal",
            "CH1_reference",
            "CH2_signal",
            "CH2_reference",
        )
    }
    assert [row[0] for row in parsed[0][1]] == [0.0, 0.5, 1.0, 1.5]
    assert parsed[0][1][0] == [0.0, 120.0, 100.0, 220.0, 200.0]
    assert parsed[1][1][0] == [0.0, 1120.0, 1100.0, 1220.0, 1200.0]
    assert source_before == _fingerprints(source_files)

    manifest_on_disk = json.loads(
        (output / converter.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest_on_disk == manifest
    assert manifest["discovered_source_session_count"] == 3
    assert manifest["converted_session_count"] == 3
    assert manifest["selected_session_limit"] is None
    assert manifest["selected_roi_ids"] == ["CH1", "CH2"]
    assert manifest["ordered_source_session_relative_paths"] == [
        f"{name}/fluorescence.csv" for name in SESSION_NAMES
    ]
    assert manifest["ordered_generated_csv_filenames"] == [
        path.name for path in csv_files
    ]
    assert manifest["output_column_mapping"]["CH1"] == {
        "signal_column": "CH1_signal",
        "reference_column": "CH1_reference",
    }
    assert "manual-validation" in manifest["disclaimer"]
    assert "not modified" in manifest["source_safety"]
    forbidden = ("dff", "correction", "smooth", "event")
    assert not any(
        token in column.lower()
        for column in parsed[0][0]
        for token in forbidden
    )


def test_limit_selects_first_ordered_sessions_and_large_limit_converts_all(
    tmp_path: Path,
):
    source_root, _source_files = _make_recording(tmp_path)
    limited_output = tmp_path / "limited"

    limited = converter.convert_rwd_to_csv_fixture(
        input_dir=str(source_root),
        output_dir=str(limited_output),
        limit=2,
    )

    assert limited["ordered_source_session_relative_paths"] == [
        f"{name}/fluorescence.csv" for name in SESSION_NAMES[:2]
    ]
    assert [path.name for path in sorted(limited_output.glob("*.csv"))] == [
        "session_0001.csv",
        "session_0002.csv",
    ]

    all_output = tmp_path / "all"
    all_available = converter.convert_rwd_to_csv_fixture(
        input_dir=str(source_root),
        output_dir=str(all_output),
        limit=20,
    )
    assert all_available["selected_session_limit"] == 20
    assert all_available["converted_session_count"] == 3
    with pytest.raises(converter.FixtureConversionError, match="positive integer"):
        converter.convert_rwd_to_csv_fixture(
            input_dir=str(source_root),
            output_dir=str(tmp_path / "invalid"),
            limit=0,
        )


def test_milliseconds_are_exact_seconds_scaling_and_one_roi_works(tmp_path: Path):
    source_root, _source_files = _make_recording(tmp_path)
    seconds_output = tmp_path / "seconds"
    milliseconds_output = tmp_path / "milliseconds"

    converter.convert_rwd_to_csv_fixture(
        input_dir=str(source_root),
        output_dir=str(seconds_output),
        limit=1,
        rois=["CH2"],
        time_unit="seconds",
    )
    converter.convert_rwd_to_csv_fixture(
        input_dir=str(source_root),
        output_dir=str(milliseconds_output),
        limit=1,
        rois=["CH2"],
        time_unit="milliseconds",
    )

    seconds_header, seconds_rows = _read_csv(seconds_output / "session_0001.csv")
    milliseconds_header, milliseconds_rows = _read_csv(
        milliseconds_output / "session_0001.csv"
    )
    assert seconds_header == ["ElapsedSeconds", "CH2_signal", "CH2_reference"]
    assert milliseconds_header == [
        "ElapsedMilliseconds",
        "CH2_signal",
        "CH2_reference",
    ]
    assert [row[0] for row in milliseconds_rows] == [
        row[0] * 1000.0 for row in seconds_rows
    ]
    assert [row[1:] for row in milliseconds_rows] == [
        row[1:] for row in seconds_rows
    ]


def test_omitted_rois_selects_only_consistent_pairs(tmp_path: Path):
    source_root, _source_files = _make_recording(
        tmp_path,
        inventories=(("CH1", "CH2"), ("CH1",), ("CH1", "CH2")),
    )

    manifest = converter.convert_rwd_to_csv_fixture(
        input_dir=str(source_root),
        output_dir=str(tmp_path / "fixture"),
    )

    assert manifest["selected_roi_ids"] == ["CH1"]
    header, _rows = _read_csv(tmp_path / "fixture" / "session_0001.csv")
    assert header == ["ElapsedSeconds", "CH1_signal", "CH1_reference"]


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        (["UNKNOWN"], "missing"),
        (["CH2"], "missing"),
    ],
)
def test_requested_roi_must_exist_in_every_session(
    tmp_path: Path, requested: list[str], message: str
):
    source_root, _source_files = _make_recording(
        tmp_path,
        inventories=(("CH1", "CH2"), ("CH1",), ("CH1", "CH2")),
    )
    output = tmp_path / "fixture"

    with pytest.raises(converter.FixtureConversionError, match=message):
        converter.convert_rwd_to_csv_fixture(
            input_dir=str(source_root),
            output_dir=str(output),
            rois=requested,
        )

    assert not output.exists()


def test_nonempty_output_and_all_input_output_overlaps_are_rejected(tmp_path: Path):
    source_root, _source_files = _make_recording(tmp_path)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    sentinel = nonempty / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(converter.FixtureConversionError, match="must not exist"):
        converter.convert_rwd_to_csv_fixture(
            input_dir=str(source_root),
            output_dir=str(nonempty),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    for output in (
        source_root,
        source_root / SESSION_NAMES[0],
        source_root / "new-child",
        source_root.parent,
    ):
        with pytest.raises(
            converter.FixtureConversionError,
            match=r"equal, nested, or parent/child",
        ):
            converter.convert_rwd_to_csv_fixture(
                input_dir=str(source_root),
                output_dir=str(output),
            )


def test_read_failure_removes_only_files_created_by_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root, source_files = _make_recording(tmp_path)
    source_before = _fingerprints(source_files)
    output = tmp_path / "fixture"
    real_load_chunk = converter.load_chunk
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("bounded read failure")
        return real_load_chunk(*args, **kwargs)

    monkeypatch.setattr(converter, "load_chunk", fail_on_second)
    with pytest.raises(
        converter.FixtureConversionError,
        match=r"2025_01_01-00_30_00.*reading.*bounded read failure",
    ):
        converter.convert_rwd_to_csv_fixture(
            input_dir=str(source_root),
            output_dir=str(output),
        )

    assert not output.exists()
    assert source_before == _fingerprints(source_files)


def test_utility_has_only_the_approved_production_reader_dependencies():
    source = Path(converter.__file__).read_text(encoding="utf-8")

    assert "from photometry_pipeline.config import Config" in source
    assert "discover_rwd_chunks" in source
    assert "resolve_continuous_source_metadata" in source
    assert "load_chunk" in source
    for forbidden in (
        "photometry_pipeline.pipeline",
        "photometry_pipeline.gui",
        "guided_",
        "custom_tabular",
        "dff",
        "smooth",
        "event detection",
    ):
        assert forbidden not in source.lower()
