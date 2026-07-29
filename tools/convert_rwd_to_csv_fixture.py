#!/usr/bin/env python3
"""Create disposable one-CSV-per-session fixtures from intermittent RWD data.

This developer utility is intentionally outside the application workflow.  It
reads an existing recording through the production RWD discovery and adapter
functions, then writes ordinary CSV files for manual Guided CSV validation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Sequence


REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from photometry_pipeline.config import Config
from photometry_pipeline.io.adapters import (
    discover_rwd_chunks,
    load_chunk,
    resolve_continuous_source_metadata,
)


UTILITY_NAME = "convert_rwd_to_csv_fixture.py"
MANIFEST_NAME = "fixture_manifest.json"


class FixtureConversionError(ValueError):
    """A clear, expected refusal from the disposable converter."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _resolved_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.is_dir():
        raise FixtureConversionError(
            f"Input must be an existing intermittent RWD directory: {input_path}"
        )
    try:
        common = Path(os.path.commonpath((str(input_path), str(output_path))))
    except ValueError:
        common = None
    if common is not None and common in {input_path, output_path}:
        raise FixtureConversionError(
            "Input and output must not be equal, nested, or parent/child paths."
        )
    if output_path.exists():
        if not output_path.is_dir():
            raise FixtureConversionError(
                f"Output exists and is not a directory: {output_path}"
            )
        try:
            next(output_path.iterdir())
        except StopIteration:
            pass
        else:
            raise FixtureConversionError(
                f"Output directory must not exist or must be empty: {output_path}"
            )
    elif not output_path.parent.is_dir():
        raise FixtureConversionError(
            f"Output parent directory does not exist: {output_path.parent}"
        )


def _safe_roi_header_component(roi_id: str) -> str:
    cleaned = re.sub(r"[,\r\n\x00-\x1f\x7f]+", "_", roi_id).strip()
    if not cleaned:
        raise FixtureConversionError(
            f"ROI ID {roi_id!r} cannot form a clear CSV column name."
        )
    return cleaned


def _metadata_for_sessions(
    session_paths: Sequence[str],
) -> list[dict]:
    metadata: list[dict] = []
    source_cache: dict[str, object] = {}
    inspection_config = Config(acquisition_mode="intermittent")
    for index, session_path in enumerate(session_paths, start=1):
        try:
            resolved = resolve_continuous_source_metadata(
                session_path,
                "rwd",
                inspection_config,
                source_cache=source_cache,
            )
        except Exception as exc:
            relative = Path(session_path).parent.name
            raise FixtureConversionError(
                f"Source session {relative} failed during RWD inspection: {exc}"
            ) from exc
        if not resolved.get("channel_names"):
            raise FixtureConversionError(
                f"Source session {index} has no valid paired RWD ROIs."
            )
        metadata.append(resolved)
    return metadata


def _select_rois(
    metadata: Sequence[dict],
    requested_rois: Sequence[str] | None,
) -> list[str]:
    first_names = [str(value) for value in metadata[0]["channel_names"]]
    first_map = dict(metadata[0]["roi_map"])
    if requested_rois:
        selected = [str(value) for value in requested_rois]
        if len(set(selected)) != len(selected):
            raise FixtureConversionError("Requested ROI IDs must be unique.")
    else:
        selected = [
            roi_id
            for roi_id in first_names
            if all(
                roi_id in item["channel_names"]
                and dict(item["roi_map"]).get(roi_id) == first_map.get(roi_id)
                for item in metadata
            )
        ]
    if not selected:
        raise FixtureConversionError(
            "No consistently available paired ROIs were found."
        )
    for session_index, item in enumerate(metadata, start=1):
        names = set(str(value) for value in item["channel_names"])
        roi_map = dict(item["roi_map"])
        for roi_id in selected:
            if roi_id not in names:
                raise FixtureConversionError(
                    f"Requested ROI {roi_id!r} is missing from selected "
                    f"source session {session_index}."
                )
            if roi_map.get(roi_id) != first_map.get(roi_id):
                raise FixtureConversionError(
                    f"ROI {roi_id!r} does not have the same signal/reference "
                    f"pair in selected source session {session_index}."
                )
    header_parts = [_safe_roi_header_component(roi_id) for roi_id in selected]
    if len(set(header_parts)) != len(header_parts):
        raise FixtureConversionError(
            "Selected ROI IDs collide after CSV-header sanitization."
        )
    return selected


def _output_columns(
    selected_rois: Sequence[str], time_unit: str
) -> tuple[list[str], dict[str, dict[str, str]]]:
    time_column = (
        "ElapsedSeconds"
        if time_unit == "seconds"
        else "ElapsedMilliseconds"
    )
    columns = [time_column]
    mapping: dict[str, dict[str, str]] = {}
    for roi_id in selected_rois:
        component = _safe_roi_header_component(roi_id)
        signal = f"{component}_signal"
        reference = f"{component}_reference"
        columns.extend((signal, reference))
        mapping[roi_id] = {
            "signal_column": signal,
            "reference_column": reference,
        }
    if len(columns) != len(set(columns)):
        raise FixtureConversionError("Generated CSV column names are not unique.")
    return columns, mapping


def _session_config(item: dict) -> Config:
    median_dt_sec = float(item["median_dt_sec"])
    duration_sec = float(item["duration_sec"])
    if (
        not math.isfinite(median_dt_sec)
        or median_dt_sec <= 0
        or not math.isfinite(duration_sec)
        or duration_sec <= 0
    ):
        raise FixtureConversionError("RWD session timing metadata is invalid.")
    return Config(
        target_fs_hz=1.0 / median_dt_sec,
        chunk_duration_sec=duration_sec,
        rwd_time_col=str(item["rwd_time_col_resolved"]),
        acquisition_mode="intermittent",
    )


def _write_session_csv(
    destination: Path,
    *,
    session_path: str,
    metadata: dict,
    chunk_id: int,
    selected_rois: Sequence[str],
    columns: Sequence[str],
    time_unit: str,
) -> None:
    try:
        chunk = load_chunk(
            session_path,
            "rwd",
            _session_config(metadata),
            chunk_id,
        )
    except Exception as exc:
        source_name = Path(session_path).parent.name
        raise FixtureConversionError(
            f"Source session {source_name} failed during RWD reading: {exc}"
        ) from exc

    indices = {roi_id: index for index, roi_id in enumerate(chunk.channel_names)}
    missing = [roi_id for roi_id in selected_rois if roi_id not in indices]
    if missing:
        raise FixtureConversionError(
            f"Source session {Path(session_path).parent.name} changed ROI "
            f"inventory during reading: {missing}"
        )
    time_scale = 1.0 if time_unit == "seconds" else 1000.0
    temp_path = destination.with_suffix(destination.suffix + ".incomplete")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=",")
            writer.writerow(columns)
            for row_index, time_sec in enumerate(chunk.time_sec):
                row: list[float] = [float(time_sec) * time_scale]
                for roi_id in selected_rois:
                    roi_index = indices[roi_id]
                    signal = float(chunk.sig_raw[row_index, roi_index])
                    reference = float(chunk.uv_raw[row_index, roi_index])
                    if not all(
                        math.isfinite(value) for value in (signal, reference)
                    ):
                        raise FixtureConversionError(
                            f"Source session {Path(session_path).parent.name} "
                            f"contains a nonfinite mapped value for ROI {roi_id!r}."
                        )
                    row.extend((signal, reference))
                writer.writerow(row)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def convert_rwd_to_csv_fixture(
    *,
    input_dir: str,
    output_dir: str,
    limit: int | None = None,
    rois: Sequence[str] | None = None,
    time_unit: str = "seconds",
) -> dict:
    """Convert one intermittent RWD recording into disposable CSV fixtures."""
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise FixtureConversionError("Session limit must be a positive integer.")
    if time_unit not in {"seconds", "milliseconds"}:
        raise FixtureConversionError(
            "Time unit must be 'seconds' or 'milliseconds'."
        )

    input_path = _resolved_path(input_dir)
    output_path = _resolved_path(output_dir)
    _validate_paths(input_path, output_path)
    try:
        discovered = discover_rwd_chunks(str(input_path))
    except Exception as exc:
        raise FixtureConversionError(
            f"RWD session discovery failed: {exc}"
        ) from exc
    selected_paths = discovered[:limit] if limit is not None else list(discovered)
    if not selected_paths:
        raise FixtureConversionError("No RWD sessions were selected.")

    print(f"Discovered RWD sessions: {len(discovered)}")
    print(f"Selected sessions: {len(selected_paths)}")
    if limit is not None and limit > len(discovered):
        print(
            f"Requested limit {limit}, but only {len(discovered)} sessions exist; "
            "converting all available sessions."
        )

    metadata = _metadata_for_sessions(selected_paths)
    selected_rois = _select_rois(metadata, rois)
    columns, column_mapping = _output_columns(selected_rois, time_unit)
    print(f"Selected ROI IDs: {', '.join(selected_rois)}")
    print(f"Output time unit: {time_unit}")

    created_output_dir = False
    created_paths: list[Path] = []
    try:
        if not output_path.exists():
            output_path.mkdir()
            created_output_dir = True
        width = max(4, len(str(len(selected_paths))))
        generated_names: list[str] = []
        for index, (session_path, item) in enumerate(
            zip(selected_paths, metadata), start=1
        ):
            filename = f"session_{index:0{width}d}.csv"
            destination = output_path / filename
            created_paths.append(destination)
            print(
                f"Converting session {index}/{len(selected_paths)}: "
                f"{Path(session_path).parent.name} -> {filename}"
            )
            _write_session_csv(
                destination,
                session_path=session_path,
                metadata=item,
                chunk_id=index - 1,
                selected_rois=selected_rois,
                columns=columns,
                time_unit=time_unit,
            )
            generated_names.append(filename)

        manifest = {
            "utility": UTILITY_NAME,
            "source_recording_path": str(input_path),
            "conversion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "discovered_source_session_count": len(discovered),
            "converted_session_count": len(selected_paths),
            "selected_session_limit": limit,
            "selected_roi_ids": list(selected_rois),
            "output_time_unit": time_unit,
            "reader_time_basis": (
                "production RWD adapter relative_seconds_since_session_start "
                "strict interpreted grid"
            ),
            "ordered_source_session_relative_paths": [
                Path(path).relative_to(input_path).as_posix()
                for path in selected_paths
            ],
            "ordered_generated_csv_filenames": generated_names,
            "output_time_column": columns[0],
            "output_column_mapping": column_mapping,
            "disclaimer": (
                "Disposable manual-validation data only; not a supported "
                "scientific export or replacement source."
            ),
            "source_safety": "The original RWD source was opened read-only and not modified.",
        }
        manifest_path = output_path / MANIFEST_NAME
        created_paths.append(manifest_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        for path in reversed(created_paths):
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except OSError:
                pass
        if created_output_dir:
            try:
                output_path.rmdir()
            except OSError:
                pass
        raise

    print(f"Fixture output: {output_path}")
    print(f"CSV files written: {len(selected_paths)}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create disposable one-file-per-session CSV data from one known-good "
            "intermittent RWD recording for manual validation."
        )
    )
    parser.add_argument("--input", required=True, help="Intermittent RWD root.")
    parser.add_argument(
        "--output", required=True, help="New or empty fixture directory."
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Convert only the first N sessions in established RWD order.",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        help="Convert only these ROI IDs, preserving the supplied order.",
    )
    parser.add_argument(
        "--time-unit",
        choices=("seconds", "milliseconds"),
        default="seconds",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        convert_rwd_to_csv_fixture(
            input_dir=args.input,
            output_dir=args.output,
            limit=args.limit,
            rois=args.rois,
            time_unit=args.time_unit,
        )
    except Exception as exc:
        print(f"ERROR: conversion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
