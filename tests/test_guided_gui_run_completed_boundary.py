from __future__ import annotations

import json
import hashlib
import os
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
from gui.main_window import (
    GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE,
    GUIDED_WORKFLOW_STEPS,
    MainWindow,
)
from photometry_pipeline.input_processing_completeness import INPUT_COMPLETENESS_FILENAME
from tests.terminal_run_fixtures import (
    BASE_CONFIG_PATH,
    seed_wrapper_deliverables,
    valid_completeness_record,
    write_current_run,
    write_phasic_feature_outputs,
)
from gui.run_report_parser import (
    classify_completed_run_candidate,
    is_successful_completed_run_dir,
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
    # Defensive cleanup: a failing test must never leave the close guard
    # active and block teardown on a real (unmocked) QMessageBox dialog.
    instance._guided_backend_execution_active = False
    thread = getattr(instance, "_guided_run_execution_thread", None)
    if thread is not None and thread.isRunning():
        thread.quit()
        thread.wait(2000)
    instance.close()
    instance.deleteLater()


def _write_minimal_successful_phasic_output(
    output_dir: Path,
    *,
    candidate=None,
    source_root: str | None = None,
    roi_id: str = "Region0",
    rwd_time_col: str = "TimeStamp",
    uv_suffix: str = "-410",
    sig_suffix: str = "-470",
) -> None:
    """Stub what a real phasic analysis subprocess would have written.

    When ``candidate``/``source_root`` are given (the real authorized RWD
    candidate this test's Guided startup chain materialized
    guided_normalized_recording_description.json for), the C8 ledger,
    ROI id, and per-chunk cache attrs are built to genuinely match that
    candidate's identity and resolved parser/channel/ROI facts -- required
    since B1's terminal normalized-recording verification now reconciles
    this stubbed output against that real authorized description. Without
    a candidate (callers that never reach full terminal classification),
    the ledger falls back to the original disconnected generic fixture.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "w") as cache:
        meta = cache.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1"
        meta.create_dataset("rois", data=np.asarray([roi_id.encode("utf-8")]))
        meta.create_dataset("chunk_ids", data=np.asarray([0], dtype=np.int64))
        chunk = cache.create_group(f"roi/{roi_id}/chunk_0")
        time_sec = np.asarray([0.0, 1.0, 2.0], dtype=float)
        chunk.create_dataset("time_sec", data=time_sec)
        chunk.create_dataset("sig_raw", data=np.asarray([2.0, 2.1, 2.2]))
        chunk.create_dataset("uv_raw", data=np.asarray([1.0, 1.1, 1.2]))
        chunk.create_dataset("fit_ref", data=np.asarray([1.0, 1.0, 1.0]))
        chunk.create_dataset("dff", data=np.asarray([1.0, 1.1, 1.2]))
        if candidate is not None:
            chunk.attrs["fs_hz"] = 40.0
            chunk.attrs["resolved_time_column"] = rwd_time_col
            chunk.attrs["resolved_header_row"] = 0
            chunk.attrs["resolved_timestamp_unit"] = "seconds"
            chunk.attrs["output_time_basis"] = "relative_seconds_since_session_start"
            chunk.attrs["resolved_signal_source"] = f"{roi_id}{sig_suffix}"
            chunk.attrs["resolved_reference_source"] = f"{roi_id}{uv_suffix}"
    (output_dir / "run_report.json").write_text(
        json.dumps(
            {
                "run_context": {"status": "success", "phase": "final"},
                "roi_selection": {
                    "selected_rois": [roi_id],
                    "discovered_rois": [roi_id],
                },
            }
        ),
        encoding="utf-8",
    )
    # A real phasic analysis always snapshots the configuration it ran with and
    # always writes its feature outputs plus the per-ROI settings record beside
    # them. The wrapper will not finalize a run whose mandatory outputs are
    # missing, so the stub must leave behind what a real analysis leaves behind.
    (output_dir / "config_used.yaml").write_text(
        BASE_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_phasic_feature_outputs(output_dir, rois=(roi_id,))
    # A real intermittent analysis accounts for every admitted input chunk.
    if candidate is not None and source_root is not None:
        source_path = os.path.normcase(
            os.path.abspath(
                os.path.normpath(
                    os.path.join(source_root, *candidate.canonical_relative_path.split("/"))
                )
            )
        )
        completeness_record = valid_completeness_record(n_chunks=1)
        completeness_record["expected"][0]["source"] = source_path
        completeness_record["expected"][0]["size_bytes"] = candidate.size_bytes
        completeness_record["expected"][0]["sha256"] = candidate.sha256_content_digest
        completeness_record["processed"][0]["source"] = source_path
    else:
        completeness_record = valid_completeness_record()
    (output_dir / INPUT_COMPLETENESS_FILENAME).write_text(
        json.dumps(completeness_record), encoding="utf-8"
    )

    # The per-ROI plot and table subprocesses are stubbed too, so stand in for
    # the deliverables a real full phasic run would have produced.
    run_dir = output_dir.parents[1]
    seed_wrapper_deliverables(run_dir, [roi_id], tonic=False)


def _completion_runner(monkeypatch, request=None):
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
            return object(), object()

        def verify_normalized_recording_live(_args, _facts, _verified):
            return None

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
                intent = (
                    request.startup_authority.rwd.production_intent
                    if request is not None
                    else None
                )
                candidate_files = (
                    intent.input_source.candidate_files if intent is not None else ()
                )
                included_roi_ids = (
                    intent.roi_scope.included_roi_ids if intent is not None else ()
                )
                _write_minimal_successful_phasic_output(
                    output_dir,
                    candidate=candidate_files[0] if candidate_files else None,
                    source_root=(
                        intent.input_source.source_root_canonical
                        if intent is not None
                        else None
                    ),
                    roi_id=included_roi_ids[0] if included_roi_ids else "Region0",
                    rwd_time_col=(
                        intent.acquisition.rwd_time_col if intent is not None else "TimeStamp"
                    ),
                    uv_suffix=(
                        intent.acquisition.uv_suffix if intent is not None else "-410"
                    ),
                    sig_suffix=(
                        intent.acquisition.sig_suffix if intent is not None else "-470"
                    ),
                )
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
        monkeypatch.setattr(
            wrapper,
            "verify_guided_normalized_recording_description_before_output",
            verify_normalized_recording_live,
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
    window, allocation_case, monkeypatch, qapp
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
    runner, calls = _completion_runner(monkeypatch, request)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()

    # Guided Run executes on a worker thread: control returns immediately
    # with the running guard active, and the final result only reaches the
    # GUI thread once the event loop is pumped. Asserting the result right
    # after the click would race the worker (stale synchronous assumption).
    assert window._guided_backend_execution_active is True
    assert window._guided_run_btn.isEnabled() is False
    _pump_until(
        qapp,
        lambda: window._guided_run_execution_thread is None,
        timeout_s=60.0,
    )
    assert window._guided_backend_execution_active is False

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

    roi_id = request.startup_authority.rwd.production_intent.roi_scope.included_roi_ids[0]
    shutil.rmtree(run_dir / roi_id)
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


def test_guided_review_shows_message_when_run_succeeded_but_no_regions(
    window, allocation_case, monkeypatch, qapp
):
    """4J16k20 negative case: a genuinely successful run (real
    run_report.json/status.json success metadata intact) that happens to
    have zero displayable regions must still navigate to Review -- it is
    not garbage/incomplete, it just has nothing to show -- and the Guided
    Review page must say so clearly, with the run folder path and a
    concise technical reason, rather than appearing blank or being
    rejected identically to a non-existent/failed run."""
    request, _plan = allocation_case
    _run_production_validation_update(window, request, monkeypatch)
    runner, _calls = _completion_runner(monkeypatch, request)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    _pump_until(
        qapp,
        lambda: window._guided_run_execution_thread is None,
        timeout_s=60.0,
    )

    result = window._guided_backend_execution_result
    run_dir = Path(request.planned_allocated_run_dir)
    assert result.status == "wrapper_completed_needs_review_loading"
    assert classify_completed_run_candidate(str(run_dir))[0] is True

    # Rebuild the same folder as a coherent run whose profile promises no
    # per-ROI deliverables. That, not a mutilated full run, is the real case for
    # "successful but nothing to display": a full production run that lost its
    # region outputs is corrupt, and must not reload as successful.
    roi_id = request.startup_authority.rwd.production_intent.roi_scope.included_roi_ids[0]
    shutil.rmtree(run_dir / roi_id)
    # write_current_run below rebuilds the terminal set (C8 ledger, deliverable
    # profile, etc.) independently of the original Guided authorization -- it
    # is no longer the real output of the Guided execution that materialized
    # guided_normalized_recording_description.json, so the Guided startup
    # markers are removed here too. Leaving them would make B1's terminal
    # normalized-recording verification correctly refuse this synthetic
    # rebuild as inconsistent Guided provenance, which is not what this test
    # is exercising (a generic, format-agnostic "successful but nothing to
    # display" terminal classification).
    for guided_marker in (
        "guided_candidate_manifest.json",
        "guided_startup_provenance.json",
            "guided_startup_status.json",
            "guided_startup_wrapper_claim.json",
            "guided_normalized_recording_description.json",
            "guided_per_roi_correction.json",
        ):
        marker_path = run_dir / guided_marker
        if marker_path.exists():
            marker_path.unlink()
    write_current_run(
        run_dir,
        run_id=run_dir.name,
        run_profile="tuning_prep",
        run_type="tuning_prep",
        traces_only=True,
        features=False,
        region="",
    )
    assert is_successful_completed_run_dir(str(run_dir))[0] is True
    assert classify_completed_run_candidate(str(run_dir))[0] is False

    window._guided_load_completed_run_for_review_btn.click()

    # Navigation must proceed -- this is a real completed run, not a
    # rejected candidate.
    assert window._guided_workflow_mode == "open_results"
    assert window._guided_workflow_stepper.currentRow() == (
        window._guided_step_index("Review")
    )
    assert window._workflow_mode_tabs.currentWidget() is (
        window._guided_workflow_tab
    )
    assert window._current_run_dir == str(run_dir)
    assert window._guided_run_readiness_label.text() == (
        "Completed run loaded for review."
    )

    # The Guided Review page itself must clearly explain there is nothing
    # to show, including the run folder path and a concise reason -- not
    # appear blank.
    guided_viewer = window._guided_report_viewer
    assert not guided_viewer._region_paths
    details = guided_viewer._status_label.text()
    assert str(run_dir) in details
    assert "no reviewable outputs" in details.lower()
    assert (
        "no region deliverables" in details.lower()
        or "region deliverables" in details.lower()
    )


def _configure_real_analysis_duration_new_analysis_draft(
    window, tmp_path, monkeypatch, *, strategy_by_roi, session_duration_sec=600,
    analysis_mode="phasic", rois=("CH1", "CH2", "CH3"), session_names=None,
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
    window._mode_combo.setCurrentText(analysis_mode)
    idx = window._format_combo.findText("rwd")
    window._format_combo.setCurrentIndex(idx)

    rois = tuple(rois)
    header = "Time(s)," + ",".join(f"{roi}-410,{roi}-470" for roi in rois)
    fs_hz = 20.0
    n_rows = int(round(session_duration_sec * fs_hz))
    session_names = tuple(session_names or (
        "2025_01_01-00_00_00",
        "2025_01_01-00_10_00",
    ))
    source_files = []
    for index, session_name in enumerate(session_names):
        # Back-to-back at exactly the sessions_per_hour="6" cadence set
        # below, so live recording-timing re-inference (triggered by any
        # discovery resync) agrees with the explicitly configured value
        # instead of silently overwriting it.
        session_dir = input_dir / session_name
        session_dir.mkdir()
        source_file = session_dir / "fluorescence.csv"
        rows = [header]
        rows.extend(
            f"{row_index / fs_hz:.2f},"
            + ",".join(
                f"{1.0 + 0.03 * np.sin((row_index / fs_hz) * 0.15 + roi_index):.8f},"
                f"{1.25 + 0.12 * np.sin((row_index / fs_hz) * 0.7 + roi_index):.8f}"
                for roi_index, _roi in enumerate(rois)
            )
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
                "session_id": session_names[index],
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
            "time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
            "timestamp_unit": "seconds",
            "fs_hz": fs_hz,
            "median_dt": 1.0 / fs_hz,
            "sample_count": n_rows,
            "chunk_duration_sec": float(session_duration_sec),
            "timestamp_duration_sec": float(session_duration_sec),
            "metadata_effective_fs_hz": None,
            "metadata_continuous_time_sec": None,
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
            uv_raw=np.column_stack(
                tuple(uv * (1.0 + 0.002 * index) for index, _roi in enumerate(rois))
            ),
            sig_raw=np.column_stack(
                tuple(sig * (1.0 + 0.005 * index) for index, _roi in enumerate(rois))
            ),
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
        strategy_text = (
            strategy_by_roi.get(roi, "Global Linear Regression")
            if strategy_by_roi else "Global Linear Regression"
        )
        strategy_index = window._guided_confirm_strategy_combo.findText(strategy_text)
        if strategy_index < 0:
            strategy_index = window._guided_confirm_strategy_combo.findData(strategy_text)
        assert strategy_index >= 0
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        assert window._guided_preview_generate_btn.isEnabled()
        window._guided_preview_generate_btn.click()
        result = window._guided_preview_last_result
        assert result["status"] in {"success", "partial"}, result
        assert result["source_type"] == "local_raw_segment"
        assert window._guided_diagnostic_cache_record is None

        row = window._guided_local_preview_confirmation_rows[roi]
        row_combo = row["strategy_combo"]
        strategy_value = (
            "signal_only_f0"
            if strategy_text == "Signal-Only F0"
            else GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE[strategy_text]
        )
        strategy_index = row_combo.findData(strategy_value)
        assert strategy_index >= 0, (roi, strategy_text, strategy_value)
        row_combo.setCurrentIndex(strategy_index)
        candidate = window._guided_local_preview_locked_evidence_for_roi(
            roi, "signal_only_f0"
        ) if strategy_text == "Signal-Only F0" else result
        assert row["action_button"].isEnabled(), {
            key: candidate.get(key) for key in (
                "valid", "selectable", "locked", "current_or_stale",
                "strategy_family", "selected_strategy", "dynamic_fit_mode",
                "issues", "warnings",
            )
        }
        row["action_button"].click()

    # A mixed plan may change the legacy global preview selector while later
    # ROIs are previewed. Reconfirm every persisted row after the full map is
    # visible, matching the scientist's final per-ROI confirmation pass.
    window._rebuild_guided_local_preview_confirmation_rows()
    for roi in rois:
        row = window._guided_local_preview_confirmation_rows[roi]
        if row["action_button"].text() == "Confirmed":
            continue
        label = strategy_by_roi[roi]
        selected = (
            "signal_only_f0"
            if label == "Signal-Only F0"
            else GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE[label]
        )
        index = row["strategy_combo"].findData(selected)
        assert index >= 0
        row["strategy_combo"].setCurrentIndex(index)
        assert row["action_button"].isEnabled()
        row["action_button"].click()

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

    # 4J16k20: the user must stay inside the Guided Workflow tab and
    # actually see the completed run's real CH1/CH2/CH3 outputs on the
    # Guided Review page itself, not merely navigate to a step index while
    # the populated results sit invisibly inside the separate Full
    # Control tab.
    assert window._workflow_mode_tabs.currentWidget() is (
        window._guided_workflow_tab
    )
    guided_viewer = window._guided_report_viewer
    assert guided_viewer._region_paths
    assert set(guided_viewer._region_paths) == {"CH1", "CH2", "CH3"}
    assert guided_viewer._region_combo.count() == 3
    assert any(
        images
        for tabs in guided_viewer._region_tab_images.values()
        for images in tabs.values()
    )

    # A subsequent Run press cannot silently reuse/overwrite the just
    # completed run: the retained startup transaction was consumed, and
    # Guided Run stays disabled until a fresh Validate/authorize cycle.
    assert window._guided_startup_transaction_request is None
    assert window._guided_run_btn.isEnabled() is False


@pytest.mark.extended
@pytest.mark.parametrize(
    ("case_name", "strategy_by_roi", "session_names"),
    [
        ("all_signal_case_a", {"CH1": "Signal-Only F0", "CH2": "Signal-Only F0"}, None),
        ("all_signal_case_b", {"CH1": "Signal-Only F0", "CH2": "Signal-Only F0"}, None),
        ("all_signal_combined", {"CH1": "Signal-Only F0", "CH2": "Signal-Only F0"}, None),
        (
            "mixed_four_combined",
            {
                "CH1": "Robust Global Event-Reject Fit",
                "CH2": "Signal-Only F0",
                "CH3": "Global Linear Regression",
                "CH4": "Adaptive Event-Gated Fit",
            },
            None,
        ),
        (
            "shuffled_discovery",
            {"CH1": "Global Linear Regression"},
            (
                "2025_01_01-00_20_00",
                "2025_01_01-00_00_00",
                "2025_01_01-00_10_00",
            ),
        ),
    ],
)
def test_real_guided_native_correction_lifecycle_matrix(
    tmp_path, monkeypatch, qapp, case_name, strategy_by_roi,
    session_names,
):
    """Real production lifecycle through ordinary production MainWindow
    construction (native Signal-Only F0 is enabled by default; no
    capability injection is needed or possible). Guided Mode exposes no
    phasic-versus-tonic choice: every case authorizes and runs
    execution_mode == "both", producing both output branches -- there is
    no single-mode Guided lifecycle left to parametrize over."""
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    from tests.test_gui_guided_new_analysis_plan import (
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    window = MainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    try:
        raw_enumerations = []
        execution_enumerations = []
        if session_names is not None:
            import photometry_pipeline.io.rwd_source_snapshot as snapshot_module
            import photometry_pipeline.io.adapters as adapters_module

            original_scandir_entries = snapshot_module._scandir_entries

            def reversed_scandir_entries(path):
                entries = original_scandir_entries(path)
                if Path(path) == tmp_path / "raw_input":
                    shuffled = list(reversed(entries))
                    raw_enumerations.append(tuple(entry.name for entry in shuffled))
                    return shuffled
                return entries

            monkeypatch.setattr(
                snapshot_module, "_scandir_entries", reversed_scandir_entries
            )
            original_execution_scandir = adapters_module._scandir_rwd_entries

            def reversed_execution_scandir(path):
                entries = original_execution_scandir(path)
                shuffled = list(reversed(entries))
                execution_enumerations.append(
                    tuple(entry.name for entry in shuffled)
                )
                return shuffled

            monkeypatch.setattr(
                adapters_module, "_scandir_rwd_entries", reversed_execution_scandir
            )
        _configure_real_analysis_duration_new_analysis_draft(
            window,
            tmp_path,
            monkeypatch,
            strategy_by_roi=strategy_by_roi,
            rois=tuple(strategy_by_roi),
            session_names=session_names,
        )
        if session_names is not None:
            # Exercise the exact execution discovery function with its raw
            # enumeration deliberately reversed, independently of the Setup
            # snapshot function patched above.
            discovered_for_execution = adapters_module.discover_rwd_chunks(
                str(tmp_path / "raw_input")
            )
            assert tuple(Path(path).parent.name for path in discovered_for_execution) == tuple(
                sorted(session_names)
            )
            assert execution_enumerations[-1] != tuple(sorted(session_names))
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
        assert window._guided_backend_validation_outcome.status == "validator_accepted", (
            window._guided_backend_validation_outcome.blocking_issues
        )
        assert window._guided_run_btn.isEnabled(), (
            window._guided_run_readiness,
            window._guided_run_readiness_label.text(),
        )
        if session_names is not None:
            expected_session_names = tuple(sorted(session_names))
            frozen_candidates = (
                window._guided_startup_authority.rwd.production_intent
                .input_source.candidate_files
            )
            assert tuple(
                item.canonical_relative_path.split("/", 1)[0]
                for item in frozen_candidates
            ) == expected_session_names
            assert raw_enumerations
            assert raw_enumerations[-1] != expected_session_names

        # Stale-validation invalidation (this task): mutating one
        # analysis-defining choice after a successful Setup check must
        # disable Run and refuse direct guarded invocation, before this
        # representative lifecycle proceeds to a real run. A missing-
        # session approval is analysis-defining but (unlike a Setup-step
        # field) does not also stale the per-ROI local-preview evidence
        # confirmation, so withdrawing it cleanly restores the exact
        # originally-validated plan for a clean revalidate below.
        from photometry_pipeline.guided_new_analysis_plan import (
            GuidedApprovedMissingSession,
        )

        probe_approval = GuidedApprovedMissingSession(
            canonical_relative_path="session-stale-probe/fluorescence.csv",
            size_bytes=1024,
            sha256_content_digest="d" * 64,
            session_index=99,
            expected_start_time="2026-01-01T00:00:00Z",
            expected_duration_sec=600.0,
        )
        window._add_guided_missing_session_approval(probe_approval)
        # Immediate invalidation: Run disables and retained state clears
        # BEFORE the guarded Run handler is ever invoked -- not merely
        # "refuses once clicked".
        assert window._guided_run_btn.isEnabled() is False
        assert window._guided_validated_plan_identity is None
        assert window._guided_startup_authority is None
        assert window._guided_execution_payload_result is None
        assert window._guided_startup_transaction_request is None
        stale_starts = []
        monkeypatch.setattr(
            window,
            "_start_guided_run_execution_worker",
            lambda request: stale_starts.append(request),
        )
        window._on_guided_run_clicked_backend_guarded()
        assert stale_starts == []
        assert window._guided_backend_execution_active is False
        assert window._guided_run_btn.isEnabled() is False
        monkeypatch.undo()
        monkeypatch.setattr(
            request_builder,
            "resolve_application_build_identity",
            lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
        )

        # Restore the visible value (withdraw the approval, reproducing
        # the exact originally-validated plan) -- Run must remain
        # disabled and the direct guard must still refuse: restoring an
        # old visible value must never revive prior validation without a
        # fresh Setup check.
        window._remove_guided_missing_session_approval(
            probe_approval.canonical_relative_path
        )
        window._refresh_guided_run_readiness_display()
        assert window._guided_run_btn.isEnabled() is False
        assert window._guided_validated_plan_identity is None
        assert window._guided_startup_authority is None
        assert window._guided_startup_transaction_request is None
        restore_starts = []
        monkeypatch.setattr(
            window,
            "_start_guided_run_execution_worker",
            lambda request: restore_starts.append(request),
        )
        window._on_guided_run_clicked_backend_guarded()
        assert restore_starts == []
        monkeypatch.undo()
        monkeypatch.setattr(
            request_builder,
            "resolve_application_build_identity",
            lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
        )

        # A fresh Setup check is required to re-enable Run.
        window._guided_backend_validate_btn.click()
        assert window._guided_backend_validation_outcome.status == "validator_accepted", (
            window._guided_backend_validation_outcome.blocking_issues
        )
        assert window._guided_run_btn.isEnabled(), (
            window._guided_run_readiness,
            window._guided_run_readiness_label.text(),
        )

        import gui.main_window as main_window_module
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "information",
            staticmethod(lambda *args, **kwargs: None),
        )
        window.show()
        started = time.monotonic()
        window._guided_run_btn.click()
        deadline = started + 180.0
        while window._guided_backend_execution_active and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        assert not window._guided_backend_execution_active
        result = window._guided_backend_execution_result
        assert result.status == "wrapper_completed_needs_review_loading", (
            tuple(issue.message for issue in result.blocking_issues), result.diagnostics
        )
        run_dir = Path(result.run_directory)
        assert (run_dir / "guided_per_roi_correction.json").is_file()
        assert not (run_dir / "guided_correction_strategy_map.json").exists()
        # Retirement of the obsolete Guided post-hoc applied-dF/F wrapper
        # stage (formerly Section 7.5 in tools/run_full_pipeline_deliverables.py):
        # no applied-dF/F output tree, and no applied-dF/F phase record of
        # any kind, for a real end-to-end native lifecycle.
        assert not (run_dir / "applied_dff").exists()
        for persisted in run_dir.rglob("*"):
            if persisted.is_file() and persisted.suffix.lower() in {
                ".json", ".yaml", ".yml", ".txt", ".ndjson"
            }:
                text = persisted.read_text(encoding="utf-8", errors="replace")
                assert "allow_signal_only_f0_execution" not in text
                assert "applied_dff_orchestration" not in text
                assert "applied_trace_cache" not in text
        startup_provenance = json.loads(
            (run_dir / "guided_startup_provenance.json").read_text(encoding="utf-8")
        )
        from photometry_pipeline.guided_startup_transaction import (
            GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION,
        )
        assert startup_provenance["startup_contract_version"] == (
            GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
        )
        native_payload_path = run_dir / "guided_per_roi_correction.json"
        assert hashlib.sha256(native_payload_path.read_bytes()).hexdigest() == (
            startup_provenance["serialized_native_correction_sha256"]
        )
        from analyze_photometry import load_guided_per_roi_correction
        loaded_specs = load_guided_per_roi_correction(
            str(run_dir / "guided_candidate_manifest.json")
        )
        assert set(loaded_specs) == set(strategy_by_roi)
        authorized_payload = json.loads(native_payload_path.read_text(encoding="utf-8"))
        authorized_records = authorized_payload["per_roi_correction"]
        # Guided Mode exposes no phasic-versus-tonic choice: the wrapper
        # is always invoked with --mode both, and both output branches
        # are always required/present, regardless of the scenario.
        assert (run_dir / "command_invoked.txt").read_text(encoding="utf-8").splitlines()[
            (run_dir / "command_invoked.txt").read_text(encoding="utf-8").splitlines().index("--mode") + 1
        ] == "both"
        expected_branches = {"tonic", "phasic"}
        from photometry_pipeline.run_completion_contract import (
            SUCCESS_STATES,
            classify_run_terminal_state,
        )
        assert classify_run_terminal_state(str(run_dir)).state in SUCCESS_STATES
        selected_by_roi = {
            roi: (
                "signal_only_f0"
                if label == "Signal-Only F0"
                else GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE[label]
            )
            for roi, label in strategy_by_roi.items()
        }
        for branch in expected_branches:
            branch_dir = run_dir / "_analysis" / f"{branch}_out"
            assert (branch_dir / "run_metadata.json").is_file()
            branch_metadata = json.loads(
                (branch_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            assert branch_metadata["correction_provenance"]["requested_by_roi"] == (
                authorized_records
            )
            cache_path = branch_dir / f"{branch}_trace_cache.h5"
            assert cache_path.is_file()
            plot_pattern = "tonic_overview.png" if branch == "tonic" else "phasic_*.png"
            assert list(run_dir.glob(f"*/**/{plot_pattern}"))
            with h5py.File(cache_path, "r") as cache:
                chunk_ids = tuple(int(value) for value in cache["meta/chunk_ids"][()])
                if session_names is not None:
                    assert chunk_ids == (0, 1, 2)
                    persisted_names = tuple(
                        Path(str(cache[f"roi/CH1/chunk_{chunk_id}"].attrs["source_file"]))
                        .parent.name
                        for chunk_id in chunk_ids
                    )
                    assert persisted_names == tuple(sorted(session_names))
                for roi, selected in selected_by_roi.items():
                    for chunk_id in chunk_ids:
                        group = cache[f"roi/{roi}/chunk_{chunk_id}"]
                        assert group.attrs["correction_selected_strategy"] == selected
                        assert "dff" in group
                        if selected == "signal_only_f0":
                            assert "signal_only_f0_baseline" in group
                        else:
                            assert "fit_ref" in group
                def assert_no_capability_attrs(group):
                    for value in group.attrs.values():
                        assert "allow_signal_only_f0_execution" not in str(value)
                    for child in group.values():
                        if isinstance(child, h5py.Group):
                            assert_no_capability_attrs(child)
                assert_no_capability_attrs(cache)
        review_button = window._guided_load_completed_run_for_review_btn
        review_button.click()
        assert window._guided_workflow_mode == "open_results"
        assert window._guided_report_viewer.phasic_review_model is not None
        assert set(
            window._guided_report_viewer.phasic_review_model.analysis_branches
        ) == expected_branches
        expected_review_labels = {
            "Signal-Only F0": "Signal-Only F0",
            "Robust Global Event-Reject Fit": "Robust global fit with event rejection",
            "Global Linear Regression": "Global linear regression",
            "Adaptive Event-Gated Fit": "Adaptive event-gated regression",
        }
        assert {
            roi: window._guided_report_viewer.phasic_review_model.strategy_label_for_roi(roi)
            for roi in strategy_by_roi
        } == {roi: expected_review_labels[label] for roi, label in strategy_by_roi.items()}
        if session_names is not None:
            immediate_sessions = (
                window._guided_report_viewer.phasic_review_model
                .sessions_by_roi["CH1"]
            )
            assert tuple(Path(row.source_file).parent.name for row in immediate_sessions) == (
                tuple(sorted(session_names))
            )

        # Reopening Completed Review must use the normalized run-root
        # contract and execution artifacts, not rediscover the original
        # source tree.  Exercise this boundary for phasic, tonic, combined,
        # mixed-strategy, and shuffled-discovery lifecycles alike.
        source_root = tmp_path / "raw_input"
        if source_root.exists():
            shutil.rmtree(source_root)

        reopened = MainWindow(
            settings=QSettings(
                str(tmp_path / "reopened-settings.ini"), QSettings.IniFormat
            )
        )
        try:
            monkeypatch.setattr(
                main_window_module.QFileDialog,
                "getExistingDirectory",
                staticmethod(lambda *_args, **_kwargs: str(run_dir)),
            )
            reopened._guided_start_open_results_btn.click()
            assert reopened._guided_workflow_mode == "open_results"
            reopened_model = reopened._report_viewer.phasic_review_model
            assert reopened_model is not None
            assert set(reopened_model.analysis_branches) == expected_branches
            assert {
                roi: reopened_model.strategy_label_for_roi(roi)
                for roi in strategy_by_roi
            } == {roi: expected_review_labels[label] for roi, label in strategy_by_roi.items()}
            if session_names is not None:
                assert tuple(
                    Path(row.source_file).parent.name
                    for row in reopened_model.sessions_by_roi["CH1"]
                ) == tuple(sorted(session_names))
        finally:
            reopened.close()
    finally:
        window.close()


def test_direct_run_invocation_succeeds_for_native_signal_only_without_capability_injection(
    tmp_path, monkeypatch, qapp
):
    """Native Signal-Only F0 is enabled by production default: ordinary
    MainWindow() construction, with no test-only capability injection,
    must reach the guarded backend worker for a valid authorized
    Signal-Only plan. The authoritative recheck inside
    _on_guided_run_clicked_backend_guarded (readiness status, retained
    request identity, authorization/outcome agreement) must still gate
    the actual start."""
    from PySide6.QtCore import QSettings
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    from tests.test_gui_guided_new_analysis_plan import (
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    window = MainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    try:
        _configure_real_analysis_duration_new_analysis_draft(
            window,
            tmp_path,
            monkeypatch,
            strategy_by_roi={"CH1": "Signal-Only F0", "CH2": "Signal-Only F0"},
            analysis_mode="phasic",
            rois=("CH1", "CH2"),
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
        retained = window._guided_startup_transaction_request
        output_base = Path(retained.output_base_canonical)
        assert output_base.exists() is False
        starts = []
        monkeypatch.setattr(window, "_start_guided_run_live_status", lambda *_: None)
        monkeypatch.setattr(
            window,
            "_start_guided_run_execution_worker",
            lambda request: starts.append(request),
        )

        window._on_guided_run_clicked_backend_guarded()

        assert starts == [retained]
        assert window._guided_backend_execution_active is True
    finally:
        window._guided_backend_execution_active = False
        window.close()


def test_real_gui_run_refuses_native_payload_mutated_before_wrapper_claim(
    tmp_path, monkeypatch, qapp
):
    from PySide6.QtCore import QSettings
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    import photometry_pipeline.guided_startup_orchestration as startup_orchestration
    from tests.test_gui_guided_new_analysis_plan import (
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    window = MainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    try:
        _configure_real_analysis_duration_new_analysis_draft(
            window,
            tmp_path,
            monkeypatch,
            strategy_by_roi={"CH1": "Signal-Only F0", "CH2": "Signal-Only F0"},
            analysis_mode="phasic",
            rois=("CH1", "CH2"),
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
        assert window._guided_run_btn.isEnabled()

        def mutate_then_launch(argv):
            run_dir = Path(argv[argv.index("--out") + 1])
            payload_path = run_dir / "guided_per_roi_correction.json"
            payload_path.write_bytes(payload_path.read_bytes() + b"\n")
            return startup_orchestration._default_subprocess_runner(argv)

        window._guided_backend_execution_runner = mutate_then_launch
        import gui.main_window as main_window_module
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: None),
        )
        window._guided_run_btn.click()
        _pump_until(
            qapp,
            lambda: window._guided_run_execution_thread is None,
            timeout_s=60,
        )
        result = window._guided_backend_execution_result
        assert result.status == "wrapper_failed"
        assert result.ok is False
        assert result.run_directory
        assert classify_completed_run_candidate(result.run_directory)[0] is False
        assert window._guided_load_completed_run_for_review_btn.isVisible() is False
        assert not (Path(result.run_directory) / "_analysis").exists()
    finally:
        window._guided_backend_execution_active = False
        window.close()


def test_real_gui_hidden_full_control_mode_combo_does_not_affect_guided_authorization(
    tmp_path, monkeypatch, qapp
):
    """Guided Mode exposes no phasic-versus-tonic choice: Full Control's
    hidden `_mode_combo` (never shown by the Guided workflow) must have no
    effect whatsoever on a Guided authorization -- the accepted Guided
    draft always carries `execution_mode == "both"`, the authorization
    identity is unaffected by `_mode_combo`, and no Guided invalidation is
    triggered by that unrelated Full Control widget. This replaces a
    retired test that asserted the opposite (now-incorrect) behavior from
    before the Guided `execution_mode` root-cause repair."""
    from PySide6.QtCore import QSettings
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    from photometry_pipeline.guided_new_analysis_plan import (
        GuidedApprovedMissingSession,
    )
    from tests.test_gui_guided_new_analysis_plan import (
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    window = MainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    try:
        _configure_real_analysis_duration_new_analysis_draft(
            window,
            tmp_path,
            monkeypatch,
            strategy_by_roi={"CH1": "Signal-Only F0", "CH2": "Signal-Only F0"},
            analysis_mode="phasic",
            rois=("CH1", "CH2"),
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
        baseline_request = window._guided_startup_transaction_request
        baseline_identity = (
            window._guided_startup_authority.canonical_authorization_identity
        )
        assert (
            window._build_guided_new_analysis_draft_plan().execution_intent.execution_mode
            == "both"
        )

        for mode in ("phasic", "tonic", "both", "phasic"):
            window._mode_combo.setCurrentText(mode)
            # A hidden Full Control widget the Guided workflow never shows
            # must not invalidate the accepted Guided authorization at all.
            assert window._guided_run_btn.isEnabled() is True
            assert window._guided_startup_transaction_request is baseline_request
            assert (
                window._guided_startup_authority.canonical_authorization_identity
                == baseline_identity
            )
            assert (
                window._build_guided_new_analysis_draft_plan()
                .execution_intent.execution_mode
                == "both"
            )

        # Sensitivity is not broken generally: a genuine, visible
        # Guided-defining change still invalidates the accepted
        # authorization, exactly as before this repair.
        probe_approval = GuidedApprovedMissingSession(
            canonical_relative_path="session-mode-invariance-probe/fluorescence.csv",
            size_bytes=1024,
            sha256_content_digest="e" * 64,
            session_index=99,
            expected_start_time="2026-01-01T00:00:00Z",
            expected_duration_sec=600.0,
        )
        window._add_guided_missing_session_approval(probe_approval)
        assert window._guided_run_btn.isEnabled() is False
        assert window._guided_startup_transaction_request is None
        assert window._guided_startup_authority is None
    finally:
        window.close()


def test_guided_run_refuses_post_setup_canonical_chronology_rename_and_does_not_revive(
    tmp_path, monkeypatch, qapp
):
    """A valid canonical rename after Setup check invalidates the frozen
    source chronology before allocation. Restoring the visible source names
    cannot revive the consumed authorization; a fresh Setup check is required.
    """
    from PySide6.QtCore import QSettings
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    from tests.test_gui_guided_new_analysis_plan import (
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    window = MainWindow(
        settings=QSettings(str(tmp_path / "mutation-settings.ini"), QSettings.IniFormat)
    )
    try:
        session_names = (
            "2025_01_01-00_00_00",
            "2025_01_01-00_10_00",
            "2025_01_01-00_20_00",
        )
        _configure_real_analysis_duration_new_analysis_draft(
            window,
            tmp_path,
            monkeypatch,
            strategy_by_roi={"CH1": "Global Linear Regression"},
            rois=("CH1",),
            session_names=session_names,
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
        assert window._guided_run_btn.isEnabled()
        old_authorization = window._guided_startup_authority
        old_request = window._guided_startup_transaction_request
        planned_run_dir = Path(old_request.planned_allocated_run_dir)

        original = tmp_path / "raw_input" / "2025_01_01-00_10_00"
        moved = tmp_path / "raw_input" / "2024_12_31-23_50_00"
        original.rename(moved)
        assert window._guided_run_btn.isEnabled()  # no GUI field changed

        worker_starts = []
        monkeypatch.setattr(
            window,
            "_start_guided_run_execution_worker",
            lambda request: worker_starts.append(request),
        )
        window._on_guided_run_clicked_backend_guarded()
        assert worker_starts == []
        assert not window._guided_backend_execution_active
        assert not planned_run_dir.exists()
        assert window._guided_startup_authority is None
        assert window._guided_startup_transaction_request is None
        assert window._guided_validated_plan_identity is None
        assert "recording sessions changed" in (
            window._guided_run_readiness_label.text().lower()
        )

        # Recreate the exact superficially valid chronology. The old retained
        # authorization/request objects must not reappear or become executable.
        moved.rename(original)
        window._refresh_guided_run_readiness_display()
        assert window._guided_startup_authority is None
        assert window._guided_startup_transaction_request is None
        assert not window._guided_run_btn.isEnabled()
        window._on_guided_run_clicked_backend_guarded()
        assert worker_starts == []
        assert not planned_run_dir.exists()
        assert old_authorization is not window._guided_startup_authority
        assert old_request is not window._guided_startup_transaction_request

        # Only a fresh Setup check creates new authorization bound to the
        # restored source snapshot and makes Run eligible again.
        window._guided_backend_validate_btn.click()
        assert window._guided_backend_validation_outcome.status == "validator_accepted"
        assert window._guided_run_btn.isEnabled()
        assert window._guided_startup_authority is not old_authorization
        assert window._guided_startup_transaction_request is not old_request
        assert not planned_run_dir.exists()
    finally:
        window.close()


def test_full_run_that_lost_a_region_directory_no_longer_loads_as_successful(
    window, allocation_case, monkeypatch, qapp
):
    """A full production run owes per-ROI deliverables. Losing them is corruption."""
    request, _plan = allocation_case
    _run_production_validation_update(window, request, monkeypatch)
    runner, _calls = _completion_runner(monkeypatch, request)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_run_execution_thread is None, timeout_s=60.0)

    run_dir = Path(request.planned_allocated_run_dir)
    assert is_successful_completed_run_dir(str(run_dir))[0] is True

    roi_id = request.startup_authority.rwd.production_intent.roi_scope.included_roi_ids[0]
    shutil.rmtree(run_dir / roi_id)
    ok, reason = is_successful_completed_run_dir(str(run_dir))
    assert ok is False, reason
    assert classify_completed_run_candidate(str(run_dir))[0] is False
