"""Guided Run post-wrapper handoff for a `reviewable_with_warning` result.

Reproduces the real condition: the wrapper process finishes with a nonzero
return code (`TERMINAL_VALIDATION_FAILED: ROI 'Region0' session N elapsed
time does not match C8`), and the run's generated deliverables genuinely
verify as `review_status == "reviewable_with_warning"` through the same
compact Review overview a successful run uses.

Proves three distinct outcomes:

A. Wrapper returncode 0, successful terminal state -- unchanged (not
   re-tested exhaustively here; covered by the existing Guided execution
   wiring/completed-boundary suites).
B. Wrapper returncode nonzero, compact overview verifies
   reviewable_with_warning -- failed run, read-only Results handoff with a
   persistent warning.
C. Wrapper returncode nonzero, compact overview does not verify -- ordinary
   Guided Run failure behavior, unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

import photometry_pipeline.guided_backend_execution as backend
import photometry_pipeline.guided_startup_orchestration as orchestration
from gui.main_window import MainWindow
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
)
from tests.test_guided_gui_run_execution_wiring import (
    _pump_until,
    _run_production_validation_update,
)
from tests.test_guided_startup_allocation import allocation_case


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance._guided_backend_execution_active = False
    thread = getattr(instance, "_guided_run_execution_thread", None)
    if thread is not None and thread.isRunning():
        thread.quit()
        thread.wait(2000)
    instance.close()
    instance.deleteLater()


# ---------------------------------------------------------------------------
# Real Pipeline-produced terminal set, independent of the (synthetic,
# file-less) allocation_case request identity -- the orchestration/wrapper
# boundary treats the runner as an opaque black box, exactly as a real
# subprocess would be.
# ---------------------------------------------------------------------------


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


def _real_two_session_analysis(tmp_root: Path) -> tuple[Path, dict]:
    input_dir = tmp_root / "input"
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
    analysis = tmp_root / "analysis"
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


def _shorten_session(cache_path: Path, roi: str, chunk_id: int, scale: float) -> None:
    with h5py.File(cache_path, "r+") as handle:
        group = handle[f"roi/{roi}/chunk_{chunk_id}"]
        time_sec = np.asarray(group["time_sec"][()])
        del group["time_sec"]
        group.create_dataset("time_sec", data=time_sec * scale)


def _break_signal_only_baseline(cache_path: Path, roi: str, chunk_id: int) -> None:
    with h5py.File(cache_path, "r+") as handle:
        del handle[f"roi/{roi}/chunk_{chunk_id}/signal_only_f0_baseline"]


def _write_deliverables(run_dir: Path, mode: dict) -> None:
    for rel_path in required_artifacts_for_run_mode(mode):
        full = run_dir / rel_path
        if str(full.resolve()).startswith(str((run_dir / "_analysis").resolve())):
            continue
        full.parent.mkdir(parents=True, exist_ok=True)
        if not full.is_file():
            full.write_bytes(b"fixture-deliverable")


def _write_terminal_set(
    run_dir: Path, mode: dict, *, run_id: str, terminal_error: str
) -> None:
    """Mirror tools/run_full_pipeline_deliverables.py's real write order:
    manifest/report finalize (completion.final=True) before terminal
    validation runs, so a validation failure leaves status.json="error"
    with no completion block -- the real preserved run's exact shape."""
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
        "errors": [f"TERMINAL_VALIDATION_FAILED: {terminal_error}"],
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def _parse_out_dir(command) -> Path:
    argv = list(command)
    return Path(argv[argv.index("--out") + 1])


_GUIDED_DEFINITIVE_MARKER_FILENAMES = (
    "guided_candidate_manifest.json",
    "guided_startup_provenance.json",
    "guided_startup_status.json",
    "guided_normalized_recording_description.json",
)


def _strip_guided_definitive_markers(run_dir: Path) -> None:
    """`allocation_case`'s request is a synthetic, file-less identity (no
    real source CSVs behind it), so real materialization's
    guided_normalized_recording_description.json authorizes a session set
    that a genuinely-analyzed 2-session RWD fixture can never reconcile
    against -- that reconciliation is already covered end-to-end elsewhere
    (test_npm_output_time_basis_repair.py,
    test_run_completion_normalized_recording_provenance.py) and is
    orthogonal to what this module proves (the wrapper-returncode -> GUI
    reclassification boundary). Removing the definitive Guided markers here
    makes classify_guided_current_native_state see an ordinary Full
    Control-shaped run, so normalized_recording_completion_error is
    unaffected by this fixture-identity mismatch -- mirroring
    tests/test_completed_run_review_warning.py's fixture, which never
    creates these markers in the first place."""
    for name in _GUIDED_DEFINITIVE_MARKER_FILENAMES:
        path = run_dir / name
        if path.is_file():
            path.unlink()


def _make_runner(tmp_path: Path, *, shape: str):
    """A stand-in for the real subprocess: writes a specific real (Pipeline-
    produced, then targeted-mutated) terminal set into the already-allocated
    run directory, then returns returncode=1 -- exactly like the real
    wrapper does for TERMINAL_VALIDATION_FAILED. `shape` selects which
    negative-control or positive fixture to build."""

    def runner(command):
        run_dir = _parse_out_dir(command)
        run_dir.mkdir(parents=True, exist_ok=True)
        analysis_root = tmp_path / f"src_{shape}"
        analysis, mode = _real_two_session_analysis(analysis_root)
        shutil.copytree(analysis, run_dir / "_analysis" / "phasic_out")
        cache_path = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
        run_id = run_dir.name
        terminal_error = (
            "correction evidence is incomplete or inconsistent: "
            "ROI 'Region0' session 1 elapsed time does not match C8"
        )

        if shape == "structural_defect":
            _break_signal_only_baseline(cache_path, "Region1", 0)
            terminal_error = "correction evidence is incomplete or inconsistent: fixture"
        else:
            _shorten_session(cache_path, "Region0", 1, 0.5)
            _shorten_session(cache_path, "Region1", 1, 0.5)

        # Every deliverable is genuinely present when the manifest is built
        # (build_manifest_completion_block fails closed on a missing
        # mandatory output, exactly like the real wrapper) -- "missing
        # deliverable" is represented the same way
        # tests/test_completed_run_review_warning.py's equivalent negative
        # control does: dropping the artifact's *entry* from the already-
        # finalized MANIFEST.json afterward, not removing the file before
        # finalization.
        _write_deliverables(run_dir, mode)
        _strip_guided_definitive_markers(run_dir)
        _write_terminal_set(run_dir, mode, run_id=run_id, terminal_error=terminal_error)

        if shape == "missing_deliverable":
            manifest_path = run_dir / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = manifest[COMPLETION_KEY]["artifacts"]
            manifest[COMPLETION_KEY]["artifacts"] = [
                entry
                for entry in artifacts
                if "phasic_correction_impact" not in entry["relative_path"]
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        elif shape == "non_terminal_validation_error":
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["errors"] = ["phasic analysis subprocess exited with code 1"]
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        elif shape == "cancelled":
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["status"] = "cancelled"
            status["errors"] = []
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

        return orchestration.GuidedWrapperProcessResult(
            returncode=1,
            stdout="",
            stderr=f"Error: TERMINAL_VALIDATION_FAILED: {terminal_error}",
            command=command,
            started=True,
            completed=True,
        )

    return runner


def _success_runner():
    def runner(command):
        run_dir = _parse_out_dir(command)
        run_dir.mkdir(parents=True, exist_ok=True)
        return orchestration.GuidedWrapperProcessResult(
            returncode=0,
            stdout="wrapper completed",
            stderr="",
            command=command,
            started=True,
            completed=True,
        )

    return runner


# ---------------------------------------------------------------------------
# Tier 1: direct execute_guided_backend_run (orchestration boundary)
# ---------------------------------------------------------------------------


def test_reviewable_with_warning_produces_new_status_and_stays_not_ok(
    allocation_case, tmp_path
):
    request, _plan = allocation_case
    runner = _make_runner(tmp_path, shape="reviewable_with_warning")
    result = backend.execute_guided_backend_run(request=request, runner=runner)

    assert result.status == "wrapper_failed_reviewable_with_warning"
    assert result.ok is False
    assert result.diagnostics.wrapper_returncode == 1
    assert result.user_summary == "Guided analysis finished with a validation warning."
    assert result.completed_run_candidate_path == request.planned_allocated_run_dir
    assert result.requires_completed_run_loader_validation is True

    status_path = Path(request.planned_allocated_run_dir) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["phase"] == "final"
    assert "TERMINAL_VALIDATION_FAILED" in status["errors"][0]


@pytest.mark.parametrize(
    "shape",
    ["missing_deliverable", "structural_defect", "non_terminal_validation_error"],
)
def test_negative_shapes_remain_ordinary_wrapper_failed(
    allocation_case, tmp_path, shape
):
    request, _plan = allocation_case
    runner = _make_runner(tmp_path, shape=shape)
    result = backend.execute_guided_backend_run(request=request, runner=runner)

    assert result.status == "wrapper_failed"
    assert result.ok is False
    assert result.completed_run_candidate_path is None
    assert result.requires_completed_run_loader_validation is False


def test_cancelled_run_never_becomes_reviewable_with_warning(allocation_case, tmp_path):
    request, _plan = allocation_case
    runner = _make_runner(tmp_path, shape="cancelled")
    result = backend.execute_guided_backend_run(request=request, runner=runner)

    assert result.status == "wrapper_failed"
    assert result.completed_run_candidate_path is None


def test_wrapper_success_handoff_is_unchanged(allocation_case):
    request, _plan = allocation_case
    runner = _success_runner()
    result = backend.execute_guided_backend_run(request=request, runner=runner)

    assert result.status == "wrapper_completed_needs_review_loading"
    assert result.ok is True
    assert result.completed_run_candidate_path == request.planned_allocated_run_dir
    assert result.requires_completed_run_loader_validation is True


# ---------------------------------------------------------------------------
# Tier 2: full GUI post-run handoff
# ---------------------------------------------------------------------------


def test_guided_run_opens_results_with_persistent_warning_and_disabled_controls(
    window, allocation_case, monkeypatch, qapp, tmp_path
):
    request, _plan = allocation_case
    _run_production_validation_update(window, request, monkeypatch)
    runner = _make_runner(tmp_path, shape="reviewable_with_warning")
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    _pump_until(
        qapp, lambda: window._guided_run_execution_thread is None, timeout_s=30.0
    )
    result = window._guided_backend_execution_result
    assert result.status == "wrapper_failed_reviewable_with_warning"
    assert result.ok is False
    assert result.user_summary == "Guided analysis finished with a validation warning."

    # Run-page text: no fabricated success, no generic "stopped before
    # results were completed" text, and the new validation-warning wording.
    details_label = window._guided_run_execution_details_label
    details_text = details_label.text()
    assert "stopped before results were completed" not in details_text
    assert "Some recording sessions were shorter than expected" in details_text
    assert "opened in Results" in details_text
    for internal_term in (
        "C8",
        "wrapper",
        "return code",
        "returncode",
        "terminal validation",
        "HDF5",
        "manifest",
        "completion contract",
    ):
        assert internal_term.lower() not in details_text.lower()

    # The auto-triggered load-for-review flow runs on its own worker thread.
    _pump_until(
        qapp,
        lambda: not getattr(window, "_guided_completed_review_loading", False),
        timeout_s=30.0,
    )

    # Results handoff: Review actually opened.
    assert window._guided_workflow_mode == "open_results"
    assert window._current_run_dir == request.planned_allocated_run_dir
    assert window._guided_workflow_stepper.currentRow() == window._guided_step_index(
        "Review"
    )

    # Persistent Results warning is visible.
    viewer = window._guided_report_viewer
    status_text = viewer._status_label.text()
    assert "validation warning" in status_text.lower()
    assert "shorter than expected" in status_text.lower() or "shorter than the expected" in status_text.lower()
    for internal_term in ("C8", "terminal validation", "HDF5", "manifest", "contract"):
        assert internal_term.lower() not in status_text.lower()

    # Success-only controls remain unavailable.
    window._is_complete_workspace_active = True
    window._refresh_tuning_workspace_availability()
    assert window._tuning_workspace_available is False
    assert (
        "not confirmed as a successful completed run"
        in window._tuning_availability_label.text()
    )
    ok, reason = window._dff_dayplot_rerender_readiness()
    assert ok is False
    assert "not confirmed as a successful completed run" in reason

    # The persisted run state never changed.
    status_path = Path(request.planned_allocated_run_dir) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert "completion" not in status


@pytest.mark.parametrize(
    "shape",
    ["missing_deliverable", "structural_defect", "non_terminal_validation_error"],
)
def test_negative_shapes_never_open_results(
    window, allocation_case, monkeypatch, qapp, tmp_path, shape
):
    request, _plan = allocation_case
    _run_production_validation_update(window, request, monkeypatch)
    runner = _make_runner(tmp_path, shape=shape)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    _pump_until(
        qapp, lambda: window._guided_run_execution_thread is None, timeout_s=30.0
    )
    result = window._guided_backend_execution_result
    assert result.status == "wrapper_failed"
    assert window._guided_workflow_mode != "open_results"
    details_text = window._guided_run_execution_details_label.text()
    assert "stopped before results were completed" in details_text
    button = window._guided_load_completed_run_for_review_btn
    assert button.isVisible() is False


def test_cancelled_run_never_opens_warning_review(
    window, allocation_case, monkeypatch, qapp, tmp_path
):
    request, _plan = allocation_case
    _run_production_validation_update(window, request, monkeypatch)
    runner = _make_runner(tmp_path, shape="cancelled")
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    _pump_until(
        qapp, lambda: window._guided_run_execution_thread is None, timeout_s=30.0
    )
    result = window._guided_backend_execution_result
    assert result.status == "wrapper_failed"
    assert window._guided_workflow_mode != "open_results"
