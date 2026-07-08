from __future__ import annotations

import json
import time
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import photometry_pipeline.guided_startup_claim as claim
import photometry_pipeline.guided_startup_orchestration as orchestration
import tools.run_full_pipeline_deliverables as wrapper
from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_guided_gui_run_execution_wiring import (
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
    instance.close()
    instance.deleteLater()


def _write_minimal_successful_phasic_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "w") as cache:
        meta = cache.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1"
        meta.create_dataset("rois", data=np.asarray([b"Region0"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0], dtype=np.int64))
        chunk = cache.create_group("roi/Region0/chunk_0")
        time_sec = np.asarray([0.0, 1.0, 2.0], dtype=float)
        chunk.create_dataset("time_sec", data=time_sec)
        chunk.create_dataset("sig_raw", data=np.asarray([2.0, 2.1, 2.2]))
        chunk.create_dataset("uv_raw", data=np.asarray([1.0, 1.1, 1.2]))
        chunk.create_dataset("fit_ref", data=np.asarray([1.0, 1.0, 1.0]))
        chunk.create_dataset("dff", data=np.asarray([1.0, 1.1, 1.2]))
    (output_dir / "run_report.json").write_text(
        json.dumps(
            {
                "run_context": {"status": "success", "phase": "final"},
                "roi_selection": {
                    "selected_rois": ["Region0"],
                    "discovered_rois": ["Region0"],
                },
            }
        ),
        encoding="utf-8",
    )


def _completion_runner(monkeypatch):
    calls = {
        "prepared_validation": 0,
        "live_verification": 0,
        "input_validation": 0,
        "analysis_stub": 0,
    }
    original_preallocated_validation = (
        wrapper.validate_guided_preallocated_mode_args
    )
    original_config_loader = wrapper.Config.from_yaml

    def runner(command):
        monkeypatch.setattr(wrapper.sys, "argv", [command[1], *command[2:]])

        def validate_preallocated(args):
            calls["prepared_validation"] += 1
            return original_preallocated_validation(args)

        def verify_live(_args):
            calls["live_verification"] += 1
            return object()

        def validate_inputs(_args):
            calls["input_validation"] += 1

        def load_config(path):
            config = original_config_loader(path)
            config.sessions_per_hour = 2
            return config

        def run_cmd(command_argv, roi_label=None):
            if Path(command_argv[1]).name == "analyze_photometry.py":
                calls["analysis_stub"] += 1
                output_dir = Path(
                    command_argv[command_argv.index("--out") + 1]
                )
                _write_minimal_successful_phasic_output(output_dir)
            return {
                "cmd": command_argv,
                "started_utc": "2026-07-02T00:00:00Z",
                "finished_utc": "2026-07-02T00:00:00Z",
                "elapsed_sec": 0.0,
                "returncode": 0,
                "roi_label": roi_label,
            }

        monkeypatch.setattr(
            wrapper, "validate_guided_preallocated_mode_args", validate_preallocated
        )
        monkeypatch.setattr(
            wrapper, "verify_guided_manifest_before_output", verify_live
        )
        monkeypatch.setattr(wrapper, "validate_inputs", validate_inputs)
        monkeypatch.setattr(
            wrapper.Config, "from_yaml", staticmethod(load_config)
        )
        monkeypatch.setattr(wrapper, "run_cmd", run_cmd)
        monkeypatch.setattr(
            wrapper, "_GUIDED_TEST_STOP_AFTER_INITIAL_STATUS", None
        )
        try:
            wrapper.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            return orchestration.GuidedWrapperProcessResult(
                returncode=code,
                stdout="",
                stderr="wrapper exited",
                command=command,
                started=True,
                completed=True,
            )
        return orchestration.GuidedWrapperProcessResult(
            returncode=0,
            stdout="wrapper completed",
            stderr="",
            command=command,
            started=True,
            completed=True,
        )

    return runner, calls


def test_gui_click_produces_loader_accepted_completed_candidate(
    window, allocation_case, monkeypatch
):
    request, _plan = allocation_case
    _run_production_validation_update(window, request, monkeypatch)
    assert window._guided_run_btn.isEnabled() is True
    full_control_before = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda *_args, **_kwargs: pytest.fail("Review auto-loaded"),
    )
    runner, calls = _completion_runner(monkeypatch)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()

    result = window._guided_backend_execution_result
    run_dir = Path(request.planned_allocated_run_dir)
    assert result.status == "wrapper_completed_needs_review_loading"
    assert result.completed_run_candidate_path == str(run_dir)
    assert result.requires_completed_run_loader_validation is True
    assert result.completed_run_claim is False
    assert window._guided_run_readiness_label.text() == (
        "Guided Run finished. Load the completed run for review."
    )
    assert (
        window._guided_load_completed_run_for_review_btn.isHidden() is False
    )
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is True
    assert calls["prepared_validation"] >= 2
    assert calls["live_verification"] == 1
    assert calls["input_validation"] == 1
    assert calls["analysis_stub"] == 1
    assert (
        run_dir / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    ).is_file()
    assert (run_dir / "status.json").is_file()
    assert (run_dir / "MANIFEST.json").is_file()
    assert (run_dir / "run_report.json").is_file()
    assert classify_completed_run_candidate(str(run_dir))[0] is True

    full_control_after = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    assert full_control_after == full_control_before
    visible_text = " ".join(
        (
            window._guided_run_btn.text(),
            window._guided_run_btn.toolTip(),
            window._guided_run_readiness_label.text(),
        )
    ).lower()
    internal_terms = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
        "guided_candidate_manifest",
        "guided_startup",
        "wrapper_claim",
        "backend adapter",
        "orchestration",
        "subprocess",
        "raw command",
    )
    assert not any(term in visible_text for term in internal_terms)

    shutil.rmtree(run_dir / "Region0")
    (run_dir / "status.json").unlink()
    (run_dir / "run_report.json").unlink()
    assert classify_completed_run_candidate(str(run_dir))[0] is False
    review_step_before = window._guided_workflow_stepper.currentRow()
    window._guided_load_completed_run_for_review_btn.click()
    assert window._guided_workflow_stepper.currentRow() == review_step_before
    assert window._guided_run_readiness_label.text() == (
        "The completed run could not be loaded for review. "
        "The output folder may be incomplete."
    )
    assert result.completed_run_claim is False


def _configure_real_analysis_duration_new_analysis_draft(
    window, tmp_path, monkeypatch, *, strategy_by_roi, session_duration_sec=600
):
    """A variant of test_gui_guided_new_analysis_plan.py's
    _configure_complete_guided_new_analysis_draft_without_diagnostic_cache
    whose raw RWD fixture files actually span Guided's fixed
    chunk_duration_sec=600.0 first-subset contract (see 4J16k12:
    guided_execution_payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES), so the real
    analyze_photometry.py subprocess can genuinely complete instead of
    raising "RWD strict: raw_end ... < grid_end ... (End Coverage
    Failure)". The shared 20s fixture is adequate for validate/authorize/
    allocate/materialize (never exercised against real analysis before
    4J16k12), but not for a real completed run. Writing the realistic
    content up front (rather than upgrading it after the shared helper
    returns) avoids invalidating the source/setup signature the shared
    helper's preview step already computed from the original content."""
    from gui.main_window import MainWindow
    from photometry_pipeline.core.types import Chunk
    import photometry_pipeline.preview.correction_preview as correction_preview_module

    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    input_dir = tmp_path / "raw_input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._mode_combo.setCurrentText("both")
    idx = window._format_combo.findText("rwd")
    window._format_combo.setCurrentIndex(idx)

    rois = ("CH1", "CH2", "CH3")
    header = "Time(s)," + ",".join(f"{roi}-410,{roi}-470" for roi in rois)
    fs_hz = 20.0
    n_rows = int(round(session_duration_sec * fs_hz))
    source_files = []
    for index in range(2):
        session_dir = input_dir / f"session-{index}"
        session_dir.mkdir()
        source_file = session_dir / "fluorescence.csv"
        rows = [header]
        rows.extend(
            f"{row_index / fs_hz:.2f}," + ",".join("1.0,2.0" for _ in rois)
            for row_index in range(n_rows)
        )
        source_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
        source_files.append(source_file)
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": len(source_files),
        "n_preview": len(source_files),
        "sessions": [
            {
                "index": index,
                "session_id": f"session-{index}",
                "path": str(source_file),
                "included_in_preview": True,
            }
            for index, source_file in enumerate(source_files)
        ],
        "rois": [{"roi_id": roi} for roi in rois],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        lambda _fmt: {
            "rwd_time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        },
    )
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {
            "csv_path": path,
            "fs_hz": fs_hz,
            "chunk_duration_sec": float(session_duration_sec),
            "time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        },
    )
    time_sec = np.arange(n_rows, dtype=float) / fs_hz
    uv = 1.0 + 0.03 * np.sin(time_sec * 0.15)
    sig = 1.25 * uv + 0.04 * np.sin(time_sec * 0.7)

    def fake_load_chunk(path, input_format, _config, chunk_id):
        return Chunk(
            chunk_id=chunk_id,
            source_file=path,
            format=input_format,
            time_sec=time_sec,
            uv_raw=np.column_stack((uv, uv, uv)),
            sig_raw=np.column_stack((sig, sig * 1.01, sig * 0.99)),
            fs_hz=fs_hz,
            channel_names=list(rois),
            metadata={},
        )

    monkeypatch.setattr(correction_preview_module, "load_chunk", fake_load_chunk)

    acquisition_idx = window._guided_acquisition_mode_combo.findData("intermittent")
    if acquisition_idx >= 0:
        window._guided_acquisition_mode_combo.setCurrentIndex(acquisition_idx)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText(str(session_duration_sec))

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    assert window._guided_diagnostic_cache_record is None

    for roi in rois:
        roi_idx = window._guided_preview_roi_combo.findData(roi)
        assert roi_idx >= 0
        window._guided_preview_roi_combo.setCurrentIndex(roi_idx)
        assert window._guided_preview_generate_btn.isEnabled()
        window._guided_preview_generate_btn.click()
        result = window._guided_preview_last_result
        assert result["status"] in {"success", "partial"}, result
        assert result["source_type"] == "local_raw_segment"
        assert window._guided_diagnostic_cache_record is None

        window._guided_confirm_roi_combo.setCurrentIndex(
            window._guided_confirm_roi_combo.findData(roi)
        )
        window._guided_confirm_chunk_combo.setCurrentIndex(0)
        strategy_text = "Global Linear Regression"
        if strategy_by_roi and roi in strategy_by_roi:
            strategy_text = strategy_by_roi[roi]
        strategy_index = window._guided_confirm_strategy_combo.findText(strategy_text)
        if strategy_index < 0:
            strategy_index = window._guided_confirm_strategy_combo.findData(strategy_text)
        assert strategy_index >= 0
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_confirm_ack_cb.setChecked(True)
        assert window._guided_confirm_mark_btn.isEnabled()
        window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_feature_event_apply_btn.click()

    output_parent = tmp_path / "planned_outputs"
    output_parent.mkdir()
    output_target = output_parent / "future_run_outputs"
    window._guided_output_path_edit.setText(str(output_target))
    window._guided_output_apply_btn.click()
    return output_parent, output_target


def test_real_gui_path_reaches_loadable_completed_run_and_reviews_it(
    window, tmp_path, monkeypatch, qapp
):
    """4J16k13: drives the REAL new_analysis GUI walkthrough (not
    allocation_case's hand-built request) through a genuinely successful
    Guided Run -- real worker thread, real allocation, real materialization,
    real wrapper invocation, real analyze_photometry.py execution, nothing
    mocked below the click -- and then clicks the real "Load completed run
    for review" button to prove the post-success handoff into Review
    actually works end to end, not merely that the contract fields are
    internally consistent (that is already covered by
    test_gui_click_produces_loader_accepted_completed_candidate above,
    using a stubbed analysis subprocess)."""
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    from tests.test_gui_guided_new_analysis_plan import (
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    strategy_by_roi = {
        roi: "Robust Global Event-Reject Fit" for roi in ("CH1", "CH2", "CH3")
    }
    _configure_real_analysis_duration_new_analysis_draft(
        window, tmp_path, monkeypatch, strategy_by_roi=strategy_by_roi
    )
    _confirm_detected_dataset_settings_via_review_plan_button(window, monkeypatch)

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_review_go_to_run_btn.click()

    build_identity = production_mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    monkeypatch.setattr(
        request_builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
    )
    window._guided_backend_validate_btn.click()
    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True

    import gui.main_window as main_window_module

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: None),
    )

    # Window must actually be shown for Qt to report the review button as
    # visible (isVisible() reflects real on-screen visibility, unlike
    # isEnabled(); without .show() every child widget reports not-visible
    # regardless of setVisible(True), independent of app logic).
    window.show()

    window._guided_run_btn.click()
    deadline = time.monotonic() + 120.0
    while (
        window._guided_backend_execution_active
        and time.monotonic() < deadline
    ):
        qapp.processEvents()
        time.sleep(0.02)
    assert window._guided_backend_execution_active is False

    result = window._guided_backend_execution_result
    assert result.status == "wrapper_completed_needs_review_loading"
    assert result.ok is True
    assert result.completed_run_claim is False
    run_dir = Path(result.run_directory)
    assert classify_completed_run_candidate(str(run_dir))[0] is True

    # Post-success GUI state.
    assert window._guided_run_readiness_label.text() == (
        "Guided Run finished. Load the completed run for review."
    )
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_validate_btn.isEnabled() is True
    review_btn = window._guided_load_completed_run_for_review_btn
    assert review_btn.isVisible() is True
    assert review_btn.isEnabled() is True

    # Real click of the real in-app control -- not a direct call into
    # _open_completed_results_dir or the report viewer.
    review_btn.click()
    assert window._guided_run_readiness_label.text() == (
        "Completed run loaded for review."
    )
    assert window._guided_workflow_mode == "open_results"
    assert window._guided_workflow_stepper.currentRow() == (
        window._guided_step_index("Review")
    )
    assert window._current_run_dir == str(run_dir)

    # A subsequent Run press cannot silently reuse/overwrite the just
    # completed run: the retained startup transaction was consumed, and
    # Guided Run stays disabled until a fresh Validate/authorize cycle.
    assert window._guided_startup_transaction_request is None
    assert window._guided_run_btn.isEnabled() is False
