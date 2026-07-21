"""Repair for the NPM output_time_basis producer/consumer key mismatch, and
for the requested-versus-consumed time-basis *contract* mismatch.

Two defects, fixed together:

1. `io/adapters.py::_build_npm_chunk_from_dataframe` used to write the NPM
   time-basis fact under `chunk.metadata["npm_output_time_basis"]`, but
   `io/hdf5_cache.py`'s writer and
   `guided_normalized_recording_consumption.py`'s consumed-evidence builder
   both only ever read `chunk.metadata`/the HDF5 attr named
   `output_time_basis`. So no NPM cache ever carried this attribute, and
   `normalized_recording_completion_error` unconditionally rejected every
   NPM run that reached this check with "has no consumed output time basis
   evidence". Fixed by renaming to the common key.

2. `guided_normalized_recording.py::build_npm_normalized_recording_description`
   authorized `sampling.time_basis` as the *recording-level composed*
   convention (`NPM_GUIDED_COMBINED_OUTPUT_TIME_BASIS`), which only
   `Pipeline._bind_authorized_chunk_chronology` (the superseded "Guided NPM
   authorized worker" route, not reachable from the live shared Guided NPM
   wrapper path) ever produces. `sampling.time_basis` actually denotes the
   per-session canonical cache convention (as RWD's authorizer already
   correctly uses it), so this was authorizing the wrong value for what the
   live production path genuinely produces. Fixed by authorizing
   `NPM_OUTPUT_TIME_BASIS` instead -- the same constant the repaired adapter
   writes.

With both fixed, a genuine Guided-shaped NPM "requested" description,
produced by `build_npm_normalized_recording_description` with no manual
adjustment, reconciles against real ordinary-path execution output.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.guided_normalized_recording import (
    build_npm_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.guided_normalized_recording_consumption import (
    build_npm_consumed_normalized_recording_evidence,
    compare_requested_and_consumed_normalized_recording,
)
from photometry_pipeline.guided_startup_transaction import (
    GUIDED_CANDIDATE_MANIFEST_FILENAME,
    GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME,
    GUIDED_STARTUP_PROVENANCE_FILENAME,
    GUIDED_STARTUP_STATUS_FILENAME,
)
from photometry_pipeline.io.npm_contract import (
    NPM_OUTPUT_TIME_BASIS,
    NpmParserContract,
    inspect_npm_csv,
)
from photometry_pipeline.io.npm_source_snapshot import (
    build_npm_source_candidate_snapshot,
)
from photometry_pipeline.run_completion_contract import (
    normalize_run_mode,
    normalized_recording_completion_error,
    PROFILE_FULL_INTERMITTENT,
)


def _write_npm_source(path: Path) -> None:
    rows = (
        (0, 100.0, 1, 10.0),
        (1, 100.5, 2, 100.0),
        (2, 101.0, 1, 11.0),
        (3, 101.5, 2, 101.0),
        (4, 102.0, 1, 12.0),
        (5, 102.5, 2, 102.0),
    )
    lines = [",".join(("FrameCounter", "Timestamp", "LedState", "Region0G"))]
    for frame_counter, timestamp, led_state, value in rows:
        lines.append(f"{frame_counter},{timestamp},{led_state},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _npm_config() -> Config:
    return Config(
        target_fs_hz=2.0,
        chunk_duration_sec=2.0,
        npm_time_axis="system_timestamp",
        npm_system_ts_col="Timestamp",
        npm_computer_ts_col="Timestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        adapter_value_nan_policy="strict",
        timestamp_cv_max=0.02,
    )


def test_npm_chunk_metadata_uses_common_key_with_npm_specific_value(tmp_path: Path):
    path = tmp_path / "photometryData2025-03-05T15_37_44.csv"
    _write_npm_source(path)
    chunk = load_chunk(str(path), "npm", _npm_config(), 0)
    assert chunk.metadata["output_time_basis"] == NPM_OUTPUT_TIME_BASIS
    assert "npm_output_time_basis" not in chunk.metadata
    # NPM keeps its own distinct value -- never collapsed to RWD's convention.
    assert NPM_OUTPUT_TIME_BASIS != "relative_seconds_since_session_start"


def test_rwd_chunk_metadata_still_uses_session_start_convention(tmp_path: Path):
    path = tmp_path / "fluorescence.csv"
    t = np.arange(20, dtype=float) / 10.0
    pd.DataFrame(
        {
            "TimeStamp": t,
            "Region0-410": 2.0 + 0.1 * np.sin(t),
            "Region0-470": 5.0 + 0.2 * np.cos(t),
        }
    ).to_csv(path, index=False)
    cfg = Config(
        target_fs_hz=10.0, chunk_duration_sec=2.0,
        rwd_time_col="TimeStamp", uv_suffix="-410", sig_suffix="-470",
    )
    chunk = load_chunk(str(path), "rwd", cfg, 0)
    assert chunk.metadata["output_time_basis"] == "relative_seconds_since_session_start"


@pytest.fixture
def real_npm_cache(tmp_path: Path):
    """A real HDF5 phasic trace cache produced by the ordinary (non
    "authorized runtime") production NPM analysis path -- the path the real
    wrapper (tools/run_full_pipeline_deliverables.py -> analyze_photometry.py
    --format npm) actually uses -- plus a genuinely valid "requested"
    NormalizedRecordingDescription built exactly the way real Guided Setup
    check would, with no manual adjustment of any field."""
    root = tmp_path / "npm_input"
    root.mkdir()
    path = root / "photometryData2025-03-05T15_37_44.csv"
    _write_npm_source(path)
    cfg = _npm_config()
    contract = NpmParserContract.from_config(cfg)
    content = contract.content()
    inspection = inspect_npm_csv(str(path), contract)
    snapshot = build_npm_source_candidate_snapshot(str(root))

    requested = build_npm_normalized_recording_description(
        source_snapshot=snapshot,
        session_inspections={
            snapshot.candidates[0].canonical_relative_path: inspection
        },
        parser_contract_content=content,
        session_duration_sec=2.0,
        sessions_per_hour=1,
        discovered_roi_ids=inspection.roi_ids,
        included_roi_ids=inspection.roi_ids,
        target_fs_hz=2.0,
    )

    run_dir = tmp_path / "run"
    out = run_dir / "_analysis" / "phasic_out"
    Pipeline(cfg, mode="phasic").run(
        str(root), str(out), force_format="npm", recursive=True, sessions_per_hour=1
    )
    return run_dir, out / "phasic_trace_cache.h5", requested


def _write_guided_markers(run_dir: Path, requested) -> None:
    """The minimal definitive-Guided marker set `classify_guided_current_native_state`
    requires before `normalized_recording_completion_error` treats a run as
    Guided at all. Only the normalized recording description's content
    matters for this module; the others only need to exist (their own
    content contracts are exercised by the startup test suites)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / GUIDED_CANDIDATE_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (run_dir / GUIDED_STARTUP_PROVENANCE_FILENAME).write_text("{}", encoding="utf-8")
    (run_dir / GUIDED_STARTUP_STATUS_FILENAME).write_text("{}", encoding="utf-8")
    (run_dir / GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME).write_text(
        json.dumps(serialize_normalized_recording_description(requested)),
        encoding="utf-8",
    )


def test_real_npm_hdf5_cache_persists_output_time_basis_attr(real_npm_cache):
    _run_dir, cache_path, _requested = real_npm_cache
    with h5py.File(cache_path, "r") as cache:
        attrs = dict(cache["roi"]["Region0"]["chunk_0"].attrs)
    assert attrs.get("output_time_basis") == NPM_OUTPUT_TIME_BASIS


def test_comparator_reconciles_real_npm_evidence_against_raw_time_basis_contract(
    real_npm_cache,
):
    run_dir, _cache_path, requested = real_npm_cache
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=str(run_dir), analysis_kind="phasic", requested=requested
    )
    assert consumed.sessions[0].output_time_basis == NPM_OUTPUT_TIME_BASIS
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert result == ""


def test_normalized_recording_completion_error_accepts_genuine_npm_run(real_npm_cache):
    """The end-to-end production contract: a genuine Guided-shaped NPM
    "requested" description (built by `build_npm_normalized_recording_description`
    with no manual field adjustment) reconciles against what the real shared
    wrapper path (ordinary `Pipeline.run(force_format="npm")`) actually
    produced, through the same `normalized_recording_completion_error` entry
    point `verify_terminal_set_before_status` calls."""
    run_dir, _cache_path, requested = real_npm_cache
    _write_guided_markers(run_dir, requested)
    mode = normalize_run_mode(
        run_profile="full",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_FULL_INTERMITTENT,
        expected_rois=["Region0"],
        chunked_input_processing=True,
        shared_input_manifest=False,
    )
    error = normalized_recording_completion_error(str(run_dir), mode)
    assert error == ""


def test_comparator_rejects_wrong_npm_time_basis_value(real_npm_cache):
    """Existing coverage (test_guided_normalized_recording_consumption.py::
    test_npm_consumed_output_time_basis_mismatch_caught) already proves this
    for the pure comparator; this repeats it against the real repaired
    producer chain specifically."""
    run_dir, _cache_path, requested = real_npm_cache
    requested = replace(
        requested,
        sampling=replace(requested.sampling, time_basis="some_other_time_basis"),
    )
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=str(run_dir), analysis_kind="phasic", requested=requested
    )
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "time basis" in result
    assert "does not match the authorized time basis" in result


def test_comparator_rejects_missing_output_time_basis_attribute(real_npm_cache):
    """The historical defect this repair fixes: a cache with the attribute
    entirely absent (as every NPM cache was before this repair) must still
    be rejected, not silently accepted."""
    run_dir, cache_path, requested = real_npm_cache
    with h5py.File(cache_path, "r+") as cache:
        del cache["roi"]["Region0"]["chunk_0"].attrs["output_time_basis"]
    consumed = build_npm_consumed_normalized_recording_evidence(
        run_dir=str(run_dir), analysis_kind="phasic", requested=requested
    )
    assert consumed.sessions[0].output_time_basis is None
    result = compare_requested_and_consumed_normalized_recording(requested, consumed)
    assert "no consumed output time basis evidence" in result
