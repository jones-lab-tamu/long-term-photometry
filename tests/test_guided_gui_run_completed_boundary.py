from __future__ import annotations

import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import photometry_pipeline.guided_startup_claim as claim
import photometry_pipeline.guided_startup_orchestration as orchestration
import tools.run_full_pipeline_deliverables as wrapper
from gui.main_window import MainWindow
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
