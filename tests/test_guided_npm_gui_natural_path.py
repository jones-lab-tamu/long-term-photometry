"""Phase 4A natural-path regression: drives the real Guided New Analysis
wizard for a supported NPM intermittent recording all the way through
Review Plan, real Check My Setup, and Run -- without bypassing Review Plan
by constructing any request directly.

Proves the two Phase 4A production fixes:
  - gui/main_window.py GUIDED_DATASET_CONTRACT_BLOCKER_CATEGORIES now
    includes the NPM dataset-contract categories, so a plan needing
    dataset-settings confirmation is routed to "Confirm the detected
    dataset settings" instead of the misleading "Guided Run does not yet
    support this configuration" message.
  - gui/main_window.py._refresh_guided_backend_validation_display no
    longer hard-codes "Running NPM analyses is not available yet" for an
    accepted NPM outcome; it reflects the real
    evaluate_guided_npm_run_readiness result (the same predicate that
    already drives the Run button).

The underlying NPM execution architecture (production mapping, execution
authority, startup payload/claim/persistence, worker launch/reconciliation)
is pre-existing and heavily tested elsewhere (guided_npm_* test modules);
this test does not re-implement or re-prove that machinery, only that the
real Guided wizard now reaches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
import photometry_pipeline.guided_npm_run_launch_builder as npm_builder_module
import photometry_pipeline.guided_npm_worker_launch as npm_launch_module
import photometry_pipeline.preview.correction_preview as correction_preview_module
from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.guided_new_analysis_plan import (
    evaluate_guided_new_analysis_execution_subset_readiness,
)
from tests.test_guided_npm_gui_run_affordance import _fake_launched_runtime

NPM_ROIS = ("Region0", "Region1", "Region2")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance._guided_backend_execution_active = False
    thread = getattr(instance, "_guided_npm_run_worker_thread", None)
    if thread is not None and thread.isRunning():
        thread.quit()
        thread.wait(2000)
    instance.close()
    instance.deleteLater()


def _pump_until(qapp, condition, *, timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        qapp.processEvents()


def _write_real_npm_session(path: Path, *, n_seconds: float, rate_hz: float) -> None:
    """Write one structurally real, LedState-interleaved NPM CSV session
    with multiple ROI columns, at roughly `rate_hz` per channel."""
    step = 1.0 / (2.0 * rate_hz)
    rows = ["Timestamp,LedState," + ",".join(f"{roi}G" for roi in NPM_ROIS)]
    t = 0.0
    led = 1
    n_samples = int(n_seconds * 2.0 * rate_hz)
    for i in range(n_samples):
        values = ",".join(
            f"{10.0 + 0.01 * i + roi_index:.4f}"
            for roi_index in range(len(NPM_ROIS))
        )
        rows.append(f"{t:.4f},{led},{values}")
        t += step
        led = 2 if led == 1 else 1
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _configure_npm_new_analysis_setup(window, tmp_path, monkeypatch):
    """Drive Select data -> Recording structure with a real, minimal NPM
    dataset, faked discovery (matching the established RWD test pattern),
    and confirmed dataset-contract settings sourced from the app's own
    baseline NPM defaults (the real production candidate path -- proven
    by test_npm_dataset_contract_candidate_does_not_call_legacy_inference
    to never call legacy RWD-style inference for NPM)."""
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    input_dir = tmp_path / "npm_raw_input"
    input_dir.mkdir()
    session_files = []
    for index in range(2):
        session_path = input_dir / f"photometryData2026-01-0{index + 1}T12_00_00.csv"
        _write_real_npm_session(session_path, n_seconds=3.0, rate_hz=45.0)
        session_files.append(session_path)

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_format_combo.setCurrentText("npm")
    # "both" (combined phasic+tonic in one atomic run) is a genuine,
    # separate, pre-existing "not yet supported" limitation at the
    # production-adapter layer (guided_npm_production_execution_request.py
    # explicitly refuses payload.execution_mode == "both" with
    # "pipeline_combined_mode_unavailable" -- the underlying Pipeline has
    # no atomic combined mode). That is out of scope for this patch; use a
    # genuinely supported single mode so this test proves the real
    # natural path for a supported configuration.
    window._mode_combo.setCurrentText("phasic")

    discovery = {
        "resolved_format": "npm",
        "n_total_discovered": len(session_files),
        "n_preview": len(session_files),
        "sessions": [
            {
                "session_id": source_file.stem,
                "path": str(source_file),
                "included_in_preview": True,
            }
            for source_file in session_files
        ],
        "rois": [{"roi_id": roi} for roi in NPM_ROIS],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    acq_idx = window._guided_acquisition_mode_combo.findData("intermittent")
    assert acq_idx >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(acq_idx)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("3")

    return input_dir, session_files


def _fake_npm_load_chunk(path, input_format, _config, chunk_id, **_kwargs):
    import numpy as np

    time_sec = np.arange(120, dtype=float) / 40.0
    uv = 1.0 + 0.03 * np.sin(time_sec * 0.2)
    sig = 1.2 * uv + 0.05 * np.sin(time_sec * 0.9)
    return Chunk(
        chunk_id=chunk_id,
        source_file=path,
        format=input_format,
        time_sec=time_sec,
        uv_raw=np.column_stack([uv] * len(NPM_ROIS)),
        sig_raw=np.column_stack([sig * (1.0 + 0.01 * i) for i in range(len(NPM_ROIS))]),
        fs_hz=40.0,
        channel_names=list(NPM_ROIS),
        metadata={},
    )


def _confirm_npm_correction_strategies(window, qapp, monkeypatch, *, included_rois):
    monkeypatch.setattr(
        correction_preview_module, "load_chunk", _fake_npm_load_chunk
    )
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    for roi in included_rois:
        roi_idx = window._guided_preview_roi_combo.findData(roi)
        assert roi_idx >= 0
        window._guided_preview_roi_combo.setCurrentIndex(roi_idx)
        assert window._guided_preview_generate_btn.isEnabled()
        window._guided_preview_generate_btn.click()
        _pump_until(
            qapp, lambda: window._guided_correction_preview_thread is None
        )
        result = window._guided_preview_last_result
        assert result["status"] in {"success", "partial"}, result

        window._guided_confirm_roi_combo.setCurrentIndex(
            window._guided_confirm_roi_combo.findData(roi)
        )
        window._guided_confirm_chunk_combo.setCurrentIndex(0)
        strategy_index = window._guided_confirm_strategy_combo.findText(
            "Global Linear Regression"
        )
        assert strategy_index >= 0
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_confirm_ack_cb.setChecked(True)
        assert window._guided_confirm_mark_btn.isEnabled()
        window._guided_confirm_mark_btn.click()


def test_natural_path_npm_reaches_review_plan_check_my_setup_and_run(
    window, tmp_path, monkeypatch, qapp
):
    _configure_npm_new_analysis_setup(window, tmp_path, monkeypatch)

    output_dir = tmp_path / "npm_output"
    output_dir.mkdir()
    window._guided_output_dir_edit.setText(str(output_dir))

    included_rois = NPM_ROIS[:2]  # exclude Region2
    excluded_roi = NPM_ROIS[2]
    for index in range(window._guided_roi_list.count()):
        item = window._guided_roi_list.item(index)
        if item.text() == excluded_roi:
            item.setCheckState(main_window_module.Qt.Unchecked)

    _confirm_npm_correction_strategies(
        window, qapp, monkeypatch, included_rois=included_rois
    )

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    )
    window._guided_feature_event_apply_btn.click()

    # Confirm the detected NPM dataset settings -- the real production
    # step the Phase 4A GUIDED_DATASET_CONTRACT_BLOCKER_CATEGORIES fix
    # correctly now routes an unconfirmed NPM plan toward.
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    plan = window._build_guided_new_analysis_draft_plan()
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    assert any(
        issue.category in ("missing_npm_channel_mapping", "missing_npm_dataset_contract")
        for issue in subset.blocking_issues
    )
    status_text = window._guided_review_plan_status_label.text()
    assert "does not yet support this configuration" not in status_text
    assert "have not been confirmed yet" in status_text
    assert window._guided_review_dataset_contract_action_btn.isHidden() is False

    window._guided_dataset_contract_apply_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.dataset_contract_snapshot.current_applied is True
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    assert subset.first_subset_executable is True

    # Trigger the same refresh the real navigation path uses.
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    )
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    status_text = window._guided_review_plan_status_label.text()
    assert "does not yet support this configuration" not in status_text
    assert "have not been confirmed yet" not in status_text
    assert "This plan is ready" in status_text

    window._guided_review_go_to_run_btn.click()
    window._guided_backend_validate_btn.click()

    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    assert outcome.accepted_for_backend_validation is True

    assert (
        "not available yet" not in window._guided_backend_validation_status_label.text()
    )
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_run_readiness_label.text() == (
        "This NPM recording setup was checked successfully and is ready "
        "to run."
    )

    # Let Run press drive the REAL, unmocked pre-existing NPM launch-
    # builder chain (production mapping -> execution authority ->
    # authorization -> startup payload -> persistence -> claim ->
    # materialization -> prelaunch claim), the same chain the existing
    # guided_npm_* backend suite already proves in isolation -- only wrap
    # it to capture its result. Only the actual OS process launch (the
    # numerical-execution seam) is replaced, so no subprocess starts.
    build_results = []
    real_build_prelaunch_claim = (
        npm_builder_module.build_guided_npm_worker_prelaunch_claim_from_validation
    )

    def _spy_build(**kwargs):
        result = real_build_prelaunch_claim(**kwargs)
        build_results.append(result)
        return result

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _spy_build,
    )
    launch_calls = []
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        lambda claim, **kwargs: (
            launch_calls.append(claim) or _fake_launched_runtime()
        ),
    )

    window._guided_run_btn.click()
    assert window._guided_backend_execution_active is True

    _pump_until(qapp, lambda: window._guided_npm_launch_runtime is not None)

    assert len(build_results) == 1
    build_result = build_results[0]
    assert build_result.ok is True, build_result.blocking_issues
    assert build_result.status == "built"
    claim = build_result.prelaunch_claim
    assert claim.execution_mode == "phasic"
    assert claim.guided_plan_identity
    assert Path(claim.run_directory_path).resolve().parent == output_dir.resolve()

    assert len(launch_calls) == 1
    assert launch_calls[0] is claim
    # No numerical execution: the wrapper subprocess is never spawned in
    # this test (launch_guided_npm_worker_runtime is replaced above), and
    # the allocated run directory must contain no analysis outputs.
    allocated_children = {p.name for p in Path(claim.run_directory_path).iterdir()}
    assert not (allocated_children & {"_analysis", "region_summary.csv"})
