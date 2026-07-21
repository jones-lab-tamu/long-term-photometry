"""End-to-end proof: a genuine, newly generated NPM run through the actual
shared production path (real `build_npm_normalized_recording_description`
authorization, real ordinary `Pipeline.run(force_format="npm")` execution)
whose only current-code problem is a legitimate recording-duration (C8)
mismatch opens through the compact Review path with a persistent warning,
while every success-only control remains unavailable.

This does not use the preserved historical NPM run as acceptance evidence --
that run predates both the output_time_basis key repair and the
sampling.time_basis contract correction and stays byte-for-byte untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from photometry_pipeline.completed_run_review import (
    CompletedRunReviewError,
    load_completed_review_overview,
)
from photometry_pipeline.config import Config
from photometry_pipeline.guided_normalized_recording import (
    build_npm_normalized_recording_description,
    serialize_normalized_recording_description,
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
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    PROFILE_FULL_INTERMITTENT,
    build_manifest_completion_block,
    build_report_completion_block,
    normalize_run_mode,
    normalized_recording_completion_error,
    required_artifacts_for_run_mode,
    review_with_warnings_eligibility,
    verify_terminal_set_before_status,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_npm_session(path: Path, *, start_value: float) -> None:
    rows = (
        (0, 100.0, 1, start_value),
        (1, 100.5, 2, start_value + 90.0),
        (2, 101.0, 1, start_value + 1.0),
        (3, 101.5, 2, start_value + 91.0),
        (4, 102.0, 1, start_value + 2.0),
        (5, 102.5, 2, start_value + 92.0),
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


def _shorten_session(cache_path: Path, roi: str, chunk_id: int, scale: float) -> None:
    """Compress the session's span toward its own (grid-aligned) origin,
    rather than scaling from zero -- NPM's origin is not always exactly
    0.0, so scaling from zero can silently also break the canonical-time
    grid-alignment check this fixture does not intend to exercise."""
    with h5py.File(cache_path, "r+") as handle:
        group = handle[f"roi/{roi}/chunk_{chunk_id}"]
        time_sec = np.asarray(group["time_sec"][()])
        origin = time_sec[0]
        del group["time_sec"]
        group.create_dataset("time_sec", data=origin + (time_sec - origin) * scale)


def _write_guided_markers(run_dir: Path, requested) -> None:
    (run_dir / GUIDED_CANDIDATE_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (run_dir / GUIDED_STARTUP_PROVENANCE_FILENAME).write_text("{}", encoding="utf-8")
    (run_dir / GUIDED_STARTUP_STATUS_FILENAME).write_text("{}", encoding="utf-8")
    (run_dir / GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME).write_text(
        json.dumps(serialize_normalized_recording_description(requested)),
        encoding="utf-8",
    )


def _write_deliverables(run_dir: Path, mode: dict) -> None:
    for rel_path in required_artifacts_for_run_mode(mode):
        full = run_dir / rel_path
        if str(full.resolve()).startswith(str((run_dir / "_analysis").resolve())):
            continue
        full.parent.mkdir(parents=True, exist_ok=True)
        if not full.is_file():
            full.write_bytes(b"fixture-deliverable")


@pytest.fixture
def genuine_npm_c8_only_run(tmp_path: Path):
    """Two real NPM sessions, run through the actual shared production path
    (real Setup-check-shaped authorization + ordinary Pipeline.run), with
    session 1's cache duration shortened after the fact to reproduce the one
    legitimate, currently-known NPM validation problem: a recording that ran
    shorter than its expected session duration."""
    root = tmp_path / "npm_input"
    root.mkdir()
    session_paths = [
        root / "photometryData2025-03-05T15_37_44.csv",
        root / "photometryData2025-03-05T15_38_44.csv",
    ]
    _write_npm_session(session_paths[0], start_value=10.0)
    _write_npm_session(session_paths[1], start_value=20.0)

    cfg = _npm_config()
    contract = NpmParserContract.from_config(cfg)
    content = contract.content()
    snapshot = build_npm_source_candidate_snapshot(str(root))
    inspections = {
        candidate.canonical_relative_path: inspect_npm_csv(
            str(root / candidate.canonical_relative_path), contract
        )
        for candidate in snapshot.candidates
    }
    roi_ids = next(iter(inspections.values())).roi_ids

    requested = build_npm_normalized_recording_description(
        source_snapshot=snapshot,
        session_inspections=inspections,
        parser_contract_content=content,
        session_duration_sec=2.0,
        sessions_per_hour=1,
        discovered_roi_ids=roi_ids,
        included_roi_ids=roi_ids,
        target_fs_hz=2.0,
    )

    run_dir = tmp_path / "run"
    analysis = run_dir / "_analysis" / "phasic_out"
    Pipeline(cfg, mode="phasic").run(
        str(root), str(analysis), force_format="npm", recursive=True, sessions_per_hour=1
    )
    cache_path = analysis / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 0.5)

    mode = normalize_run_mode(
        run_profile="full",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_FULL_INTERMITTENT,
        expected_rois=list(roi_ids),
        chunked_input_processing=True,
        shared_input_manifest=False,
    )
    run_id = "genuine-npm-c8-only-run"
    _write_deliverables(run_dir, mode)
    _write_guided_markers(run_dir, requested)
    (run_dir / "run_report.json").write_text(
        json.dumps(
            {"completion_contract": build_report_completion_block(run_id=run_id)},
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        COMPLETION_KEY: build_manifest_completion_block(
            str(run_dir),
            run_id=run_id,
            run_mode=mode,
            finalized_utc="2026-07-21T00:00:00+00:00",
        )
    }
    (run_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status = {
        "run_id": run_id,
        "run_profile": mode["run_profile"],
        "run_type": mode["run_type"],
        "acquisition_mode": mode["acquisition_mode"],
        "traces_only": mode["traces_only"],
        "phase": "final",
        "status": "error",
        "errors": [
            "TERMINAL_VALIDATION_FAILED: correction evidence is incomplete or "
            "inconsistent: ROI 'Region0' session 1 elapsed time does not match C8"
        ],
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return run_dir, mode, requested, run_id


def test_step1_hdf5_chunk_carries_npm_output_time_basis(genuine_npm_c8_only_run):
    run_dir, _mode, _requested, _run_id = genuine_npm_c8_only_run
    cache_path = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r") as cache:
        attrs = dict(cache["roi"]["Region0"]["chunk_0"].attrs)
    assert attrs.get("output_time_basis") == NPM_OUTPUT_TIME_BASIS


def test_step2_normalized_recording_verification_passes(genuine_npm_c8_only_run):
    run_dir, mode, _requested, _run_id = genuine_npm_c8_only_run
    assert normalized_recording_completion_error(str(run_dir), mode) == ""


def test_step3_strict_verification_fails_on_duration_only(genuine_npm_c8_only_run):
    run_dir, mode, _requested, run_id = genuine_npm_c8_only_run
    error = verify_terminal_set_before_status(str(run_dir), run_id=run_id, run_mode=mode)
    assert "elapsed time does not match C8" in error


def test_step4_warning_review_verifier_collects_only_the_duration_warning(
    genuine_npm_c8_only_run,
):
    run_dir, mode, _requested, run_id = genuine_npm_c8_only_run
    fatal, warnings = review_with_warnings_eligibility(
        str(run_dir), run_id=run_id, run_mode=mode
    )
    assert fatal == ""
    assert len(warnings) >= 1
    assert all(w["roi"] == "Region0" for w in warnings)
    assert {w["session_index"] for w in warnings} == {1}
    for w in warnings:
        assert w["expected_duration_sec"] == 2.0
        assert w["observed_duration_sec"] < w["expected_duration_sec"]
        assert w["duration_difference_sec"] < 0


def test_step5_compact_review_opens_reviewable_with_warning(genuine_npm_c8_only_run):
    run_dir, _mode, _requested, _run_id = genuine_npm_c8_only_run
    overview = load_completed_review_overview(str(run_dir))
    assert overview["review_status"] == "reviewable_with_warning"
    assert overview["terminal_state"] == "failed"
    assert overview["format"] == "npm"
    assert overview["affected_session_count"] == 1
    assert overview["first_affected_session_index"] == 1
    message = overview["validation_warning_message"]
    assert "C8" not in message
    assert "progressively" not in message
    assert "later" not in message.lower()
    # Derived from this fixture's real 2-second expected duration, not a
    # hardcoded "10-minute" figure that would be wrong here.
    assert overview["expected_session_duration_sec"] == 2.0
    assert "2-second" in message
    assert "10-minute" not in message
    assert overview["largest_duration_shortfall_sec"] > 0


def test_step6_success_only_controls_stay_unavailable(qapp, genuine_npm_c8_only_run):
    run_dir, _mode, _requested, _run_id = genuine_npm_c8_only_run
    overview = load_completed_review_overview(str(run_dir))
    assert overview["review_status"] == "reviewable_with_warning"

    window = MainWindow()
    try:
        import os

        resolved = os.path.realpath(str(run_dir))
        window._current_run_dir = resolved
        window._accepted_completed_review_path = resolved
        window._accepted_completed_review_overview = dict(overview)
        window._is_complete_workspace_active = True

        window._refresh_tuning_workspace_availability()
        assert window._tuning_workspace_available is False
        assert "not confirmed as a successful completed run" in (
            window._tuning_availability_label.text()
        )

        ok, reason = window._dff_dayplot_rerender_readiness()
        assert ok is False
        assert "not confirmed as a successful completed run" in reason
    finally:
        window.close()
        window.deleteLater()


def test_missing_dataset_still_rejected_not_masked_by_review_status(
    genuine_npm_c8_only_run,
):
    """Sanity guard: the warning-review path must still fail closed on a
    genuinely structural defect, not just on the duration mismatch."""
    run_dir, _mode, _requested, _run_id = genuine_npm_c8_only_run
    cache_path = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "r+") as cache:
        del cache["roi"]["Region0"]["chunk_0"].attrs["output_time_basis"]
    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(run_dir))
