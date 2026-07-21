"""Compact Review eligibility for a final-validation-failed run whose only
current-code problem is a legitimate recording-duration (C8) mismatch.

Mirrors the real preserved run this feature exists for: analysis completed,
every deliverable was generated, MANIFEST.json was finalized (completion.final
is True) before terminal validation ran, and *then* validation found ROI/
session elapsed-time mismatches. tools/run_full_pipeline_deliverables.py
writes the manifest/report before calling verify_terminal_set_before_status
(see `_finalize_terminal_success`), so status.json="error" with no completion
block is the real shape, not a test artifact.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.completed_run_review import (
    CompletedRunReviewError,
    load_completed_review_overview,
)
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    PROFILE_FULL_INTERMITTENT,
    build_manifest_completion_block,
    build_report_completion_block,
    normalize_run_mode,
    required_artifacts_for_run_mode,
    review_with_warnings_eligibility,
    verify_terminal_set_before_status,
)


def _write_session(path: Path, n: int, *, phase: float) -> None:
    t = np.arange(n, dtype=float) / 10.0
    values = {"TimeStamp": t}
    for index in range(2):
        values[f"Region{index}-410"] = 2.0 + 0.1 * np.sin(0.2 * t + index + phase)
        values[f"Region{index}-470"] = 5.0 + index + 0.2 * np.cos(0.3 * t + index + phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(path, index=False)


def _mixed_map() -> dict[str, PerRoiCorrectionSpec]:
    return {
        "Region0": PerRoiCorrectionSpec(
            "Region0", "dynamic_fit", "global_linear_regression", "global_linear_regression"
        ),
        "Region1": PerRoiCorrectionSpec("Region1", "signal_only_f0", "signal_only_f0"),
    }


@pytest.fixture
def two_session_native_analysis(tmp_path):
    """Two 20-second chunks (sessions) x two ROIs, built with the real
    pipeline so every current-code correction/completeness check genuinely
    passes before any mutation."""
    input_dir = tmp_path / "input"
    _write_session(input_dir / "2024_01_01-00_00_00" / "fluorescence.csv", 200, phase=0.0)
    _write_session(input_dir / "2024_01_01-01_00_00" / "fluorescence.csv", 200, phase=0.5)
    cfg = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        window_sec=10.0,
        min_samples_per_window=10,
        signal_only_f0_min_window_samples=21,
    )
    analysis = tmp_path / "analysis"
    Pipeline(cfg, mode="phasic", per_roi_correction=_mixed_map()).run(
        str(input_dir), str(analysis), force_format="rwd", recursive=True
    )
    mode = normalize_run_mode(
        run_profile="full",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_FULL_INTERMITTENT,
        expected_rois=["Region0", "Region1"],
        chunked_input_processing=True,
        shared_input_manifest=False,
    )
    return analysis, mode


def _build_run(tmp_path: Path, analysis: Path, mode: dict, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(analysis, root / "_analysis" / "phasic_out")
    for rel_path in required_artifacts_for_run_mode(mode):
        full = root / rel_path
        if str(full.resolve()).startswith(str((root / "_analysis").resolve())):
            continue
        full.parent.mkdir(parents=True, exist_ok=True)
        if not full.is_file():
            full.write_bytes(b"fixture-deliverable")
    return root


def _write_failed_terminal_set(
    root: Path, mode: dict, *, run_id: str, terminal_error: str
) -> None:
    """Reproduce the real wrapper's write order: manifest/report are
    finalized first (completion.final=True), *then* terminal validation is
    evaluated and its failure is recorded only in status.json, with no
    status completion block -- exactly the preserved real run's shape."""
    (root / "run_report.json").write_text(
        json.dumps(
            {"completion_contract": build_report_completion_block(run_id=run_id)},
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        COMPLETION_KEY: build_manifest_completion_block(
            str(root),
            run_id=run_id,
            run_mode=mode,
            finalized_utc="2026-07-20T00:00:00+00:00",
        )
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status = {
        "run_id": run_id,
        "run_profile": mode["run_profile"],
        "run_type": mode["run_type"],
        "acquisition_mode": mode["acquisition_mode"],
        "traces_only": mode["traces_only"],
        "phase": "final",
        "status": "error",
        "errors": [f"TERMINAL_VALIDATION_FAILED: {terminal_error}"],
    }
    (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def _shorten_session(cache_path: Path, roi: str, chunk_id: int, scale: float) -> None:
    with h5py.File(cache_path, "r+") as handle:
        group = handle[f"roi/{roi}/chunk_{chunk_id}"]
        time_sec = np.asarray(group["time_sec"][()])
        del group["time_sec"]
        group.create_dataset("time_sec", data=time_sec * scale)


def _break_signal_only_baseline(cache_path: Path, roi: str, chunk_id: int) -> None:
    with h5py.File(cache_path, "r+") as handle:
        del handle[f"roi/{roi}/chunk_{chunk_id}/signal_only_f0_baseline"]


def test_clean_run_has_no_elapsed_time_warnings_and_no_fatal_error(
    two_session_native_analysis, tmp_path
):
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "clean")
    _write_failed_terminal_set(
        root, mode, run_id="clean-run", terminal_error="unused"
    )
    fatal, warnings = review_with_warnings_eligibility(
        str(root), run_id="clean-run", run_mode=mode
    )
    assert fatal == ""
    assert warnings == []
    # The ordinary strict verifier (used by real success classification)
    # remains fail-fast and unaffected by the presence of the collector API.
    assert (
        verify_terminal_set_before_status(str(root), run_id="clean-run", run_mode=mode)
        == ""
    )


def test_c8_only_mismatch_is_reviewable_with_warning(
    two_session_native_analysis, tmp_path
):
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "c8_only")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 0.5)
    _shorten_session(cache_path, "Region1", 1, 0.5)
    _write_failed_terminal_set(
        root,
        mode,
        run_id="c8-run",
        terminal_error=(
            "correction evidence is incomplete or inconsistent: "
            "ROI 'Region0' session 1 elapsed time does not match C8"
        ),
    )

    # The ordinary strict verifier still treats C8 as fatal and may still
    # return on the first failure it finds -- unchanged from today.
    strict_error = verify_terminal_set_before_status(
        str(root), run_id="c8-run", run_mode=mode
    )
    assert "elapsed time does not match C8" in strict_error

    # The warning-collecting verifier continues past every C8 mismatch
    # instead of stopping at the first one, and reports no fatal error.
    fatal, warnings = review_with_warnings_eligibility(
        str(root), run_id="c8-run", run_mode=mode
    )
    assert fatal == ""
    session_indices = sorted({w["session_index"] for w in warnings})
    assert session_indices == [1]
    rois_seen = {w["roi"] for w in warnings}
    assert rois_seen == {"Region0", "Region1"}
    # Real numerical evidence, not just identity -- and every collected
    # entry is a genuine shortfall (observed < expected).
    for w in warnings:
        assert w["expected_duration_sec"] == 20.0
        assert w["observed_duration_sec"] < w["expected_duration_sec"]
        assert w["duration_difference_sec"] == pytest.approx(
            w["observed_duration_sec"] - w["expected_duration_sec"]
        )
        assert w["duration_difference_sec"] < 0

    overview = load_completed_review_overview(str(root))
    assert overview["review_status"] == "reviewable_with_warning"
    assert overview["terminal_state"] == "failed"
    assert overview["affected_session_count"] == 1
    assert overview["first_affected_session_index"] == 1
    message = overview["validation_warning_message"]
    assert "C8" not in message
    assert "terminal validation" not in message.lower()
    # The 20-second fixture duration is derived from persisted evidence, not
    # a hardcoded "10-minute" figure that would be wrong for this fixture.
    assert overview["expected_session_duration_sec"] == 20.0
    assert "20-second" in message
    assert "10-minute" not in message
    assert overview["largest_duration_shortfall_sec"] > 0
    # No claim this implementation does not establish.
    assert "progressively" not in message
    assert "later" not in message.lower()
    assert overview["included_rois"] == [] or set(overview["included_rois"]) <= {
        "Region0",
        "Region1",
    }
    assert overview["analysis_branches"] == ["phasic"]


def test_c8_mismatch_plus_fatal_defect_is_rejected(
    two_session_native_analysis, tmp_path
):
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "c8_plus_fatal")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 0.5)
    # An independent, unrelated structural defect: Region1's Signal-Only F0
    # baseline dataset is simply gone.
    _break_signal_only_baseline(cache_path, "Region1", 0)
    _write_failed_terminal_set(
        root,
        mode,
        run_id="c8-plus-fatal-run",
        terminal_error="correction evidence is incomplete or inconsistent: fixture",
    )

    fatal, warnings = review_with_warnings_eligibility(
        str(root), run_id="c8-plus-fatal-run", run_mode=mode
    )
    assert fatal != ""
    assert "production F0 baseline" in fatal or "Signal-Only" in fatal

    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(root))


def test_unexpectedly_longer_session_remains_rejected(
    two_session_native_analysis, tmp_path
):
    """An observed duration *longer* than expected is a different, unreviewed
    class of problem: it must stay fatal even while collecting shortfall
    warnings elsewhere, never treated as an allowlisted data-quality issue."""
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "overrun")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 1.5)  # grows the span, not shrinks
    _write_failed_terminal_set(
        root, mode, run_id="overrun-run", terminal_error="fixture"
    )

    # Strict successful-completion behavior is unchanged: an overrun is
    # still fatal in the ordinary (non-collecting) verifier, exactly as
    # before this repair.
    strict_error = verify_terminal_set_before_status(
        str(root), run_id="overrun-run", run_mode=mode
    )
    assert "elapsed time does not match C8" in strict_error

    # And it stays fatal in the warning-collecting verifier too -- an
    # overrun is never appended as a warning.
    fatal, warnings = review_with_warnings_eligibility(
        str(root), run_id="overrun-run", run_mode=mode
    )
    assert "elapsed time does not match C8" in fatal
    assert warnings == []

    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(root))


def test_mixed_expected_durations_use_generic_wording(
    two_session_native_analysis, tmp_path
):
    """When affected sessions do not share one expected duration, the
    message must not present any single number as *the* expected duration."""
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "mixed_durations")
    analysis_dir = root / "_analysis" / "phasic_out"
    cache_path = analysis_dir / "phasic_trace_cache.h5"

    completeness_path = analysis_dir / "input_processing_completeness.json"
    completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
    completeness["expected"][0]["expected_duration_sec"] = 25.0
    completeness_path.write_text(json.dumps(completeness, indent=2), encoding="utf-8")

    # Both sessions now shortened relative to their own (different)
    # expected durations: session 0 expects 25s, session 1 expects 20s.
    _shorten_session(cache_path, "Region0", 0, 0.5)
    _shorten_session(cache_path, "Region1", 0, 0.5)
    _shorten_session(cache_path, "Region0", 1, 0.5)
    _shorten_session(cache_path, "Region1", 1, 0.5)
    _write_failed_terminal_set(
        root, mode, run_id="mixed-durations-run", terminal_error="fixture"
    )

    overview = load_completed_review_overview(str(root))
    assert overview["review_status"] == "reviewable_with_warning"
    assert overview["affected_session_count"] == 2
    assert "expected_session_duration_sec" not in overview
    message = overview["validation_warning_message"]
    assert "shorter than expected" in message
    assert "shorter than the expected" not in message
    assert "-second" not in message
    assert "-minute" not in message


def test_stale_status_prose_is_not_the_displayed_reason(
    two_session_native_analysis, tmp_path
):
    """status.json's error prose can be obsolete (e.g. from a bug later
    fixed in verification code); the overview's warning text must reflect
    what current-code verification actually finds, not that stale prose."""
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "stale_prose")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 0.5)
    _shorten_session(cache_path, "Region1", 1, 0.5)
    _write_failed_terminal_set(
        root,
        mode,
        run_id="stale-prose-run",
        terminal_error=(
            "correction evidence is incomplete or inconsistent: ROI 'Region0' "
            "session 0 has invalid canonical time identity"
        ),
    )

    overview = load_completed_review_overview(str(root))
    assert overview["review_status"] == "reviewable_with_warning"
    assert overview["affected_session_count"] == 1
    assert overview["first_affected_session_index"] == 1
    assert "canonical time identity" not in overview["validation_warning_message"]


def test_missing_required_deliverable_rejects(two_session_native_analysis, tmp_path):
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "missing_deliverable")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 0.5)
    _write_failed_terminal_set(
        root, mode, run_id="missing-deliverable-run", terminal_error="fixture"
    )
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest[COMPLETION_KEY]["artifacts"]
    manifest[COMPLETION_KEY]["artifacts"] = [
        entry
        for entry in artifacts
        if "phasic_correction_impact" not in entry["relative_path"]
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(root))


def test_run_id_mismatch_rejects(two_session_native_analysis, tmp_path):
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "run_id_mismatch")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 0.5)
    _write_failed_terminal_set(
        root, mode, run_id="original-run-id", terminal_error="fixture"
    )
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["run_id"] = "a-different-run-id"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(root))


def test_tmp_file_present_rejects(two_session_native_analysis, tmp_path):
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "tmp_leftover")
    cache_path = root / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    _shorten_session(cache_path, "Region0", 1, 0.5)
    _write_failed_terminal_set(
        root, mode, run_id="tmp-leftover-run", terminal_error="fixture"
    )
    (root / "Region0" / "summary").mkdir(parents=True, exist_ok=True)
    (root / "Region0" / "summary" / "leftover.tmp").write_bytes(b"partial")

    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(root))


def test_non_terminal_validation_failure_code_rejects(
    two_session_native_analysis, tmp_path
):
    """An analysis-subprocess failure (or any error that isn't specifically
    TERMINAL_VALIDATION_FAILED) must never reach the warning-review path."""
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "analysis_failure")
    _write_failed_terminal_set(
        root, mode, run_id="analysis-failure-run", terminal_error="fixture"
    )
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["errors"] = ["phasic analysis subprocess exited with code 1"]
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(root))


def test_cancelled_run_rejects(two_session_native_analysis, tmp_path):
    analysis, mode = two_session_native_analysis
    root = _build_run(tmp_path, analysis, mode, "cancelled")
    _write_failed_terminal_set(
        root, mode, run_id="cancelled-run", terminal_error="fixture"
    )
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "cancelled"
    status["errors"] = []
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    with pytest.raises(CompletedRunReviewError):
        load_completed_review_overview(str(root))
