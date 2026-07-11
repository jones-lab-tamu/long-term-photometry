"""Focused 4J16k42 tests for Guided missing-session approval state."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton, QMessageBox

from gui.main_window import MainWindow, GUIDED_WORKFLOW_STEPS
from gui.run_report_parser import classify_completed_run_candidate
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedApprovedMissingSession,
    GuidedNewAnalysisDraftPlan,
    evaluate_new_analysis_plan_issues,
)
from photometry_pipeline.guided_backend_execution import (
    GuidedExecutionFailureDetail,
    GuidedBackendExecutionResult,
    execute_guided_backend_run,
)
from photometry_pipeline.guided_backend_validation_materialization import (
    _materialize_approved_missing_candidates,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)
from photometry_pipeline.input_processing_completeness import (
    InputProcessingError,
)
import photometry_pipeline.guided_execution_payloads as payloads
import photometry_pipeline.guided_production_mapping as mapping
import photometry_pipeline.guided_run_authorization as authorization
import pandas as pd
from tests.test_missing_session_backend import _source, NAMES

def _write_valid(chunk_dir: Path, *, seed: int, flat: bool = False, n: int = 6000, fs: float = 10.0):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    if flat:
        uv = np.ones(n)
        sig = np.ones(n) * 1.2
    else:
        uv = 1.0 + 0.03 * rng.normal(0, 1, n)
        sig = 1.2 * uv + 0.1 * rng.normal(0, 1, n)
    pd.DataFrame({"TimeStamp": t, "Region0-470": sig, "Region0-410": uv}).to_csv(
        chunk_dir / "fluorescence.csv", index=False
    )

def _write_corrupted(chunk_dir: Path):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    # Valid header, but corrupt data to cause natural pipeline failure during execution
    (chunk_dir / "fluorescence.csv").write_text(
        "TimeStamp,Region0-470,Region0-410\n0.0,corrupt_signal,1.0\n0.1,corrupt_signal,1.0\n",
        encoding="utf-8"
    )

def _build_input(tmp_path: Path, *, corrupted=(), flat=(), n_sessions=3) -> Path:
    inp = tmp_path / "input"
    for i in range(n_sessions):
        d = inp / NAMES[i]
        if i in corrupted:
            _write_corrupted(d)
        else:
            _write_valid(d, seed=i, flat=(i in flat))
    return inp



@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _approval(relative="2024_01_01-01_00_00/fluorescence.csv", index=1):
    return GuidedApprovedMissingSession(
        canonical_relative_path=relative,
        size_bytes=12,
        sha256_content_digest="a" * 64,
        session_index=index,
        expected_start_time="2024-01-01T01:00:00",
        expected_duration_sec=60.0,
    )


def _pump_until(qapp, condition, *, timeout_s: float = 10.0) -> None:
    import time
    t0 = time.perf_counter()
    while not condition():
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError("Pump condition timed out")
        qapp.processEvents()
        time.sleep(0.01)


def test_plan_allows_multiple_distinct_approvals_without_hidden_limit():
    first = _approval(index=1)
    second = _approval(
        relative="2024_01_01-02_00_00/fluorescence.csv", index=2
    )
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path="C:/recording",
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=1,
        session_duration_sec=60.0,
        approved_missing_sessions=[first, second],
    )
    assert not any(
        issue.category == "duplicate_approved_missing_session"
        for issue in evaluate_new_analysis_plan_issues(plan)
    )
    assert [item.session_index for item in plan.approved_missing_sessions] == [1, 2]


def test_duplicate_plan_approval_is_blocked():
    first = _approval()
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path="C:/recording",
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=1,
        session_duration_sec=60.0,
        approved_missing_sessions=[first, first],
    )
    assert "duplicate_approved_missing_session" in {
        issue.category for issue in evaluate_new_analysis_plan_issues(plan)
    }


def test_snapshot_revalidates_approved_source_identity_and_timestamp(tmp_path: Path):
    input_root = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    snapshot = build_rwd_source_candidate_snapshot(str(input_root))
    candidate = next(
        item for item in snapshot.candidates
        if item.canonical_relative_path.endswith("01_00_00/fluorescence.csv")
    )
    approval = GuidedApprovedMissingSession(
        canonical_relative_path=candidate.canonical_relative_path,
        size_bytes=candidate.size_bytes,
        sha256_content_digest=candidate.sha256_content_digest,
        session_index=1,
        expected_start_time="2024-01-01T01:00:00",
        expected_duration_sec=60.0,
    )
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(input_root),
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=1,
        session_duration_sec=60.0,
        approved_missing_sessions=[approval],
    )
    approved, failure = _materialize_approved_missing_candidates(draft, snapshot)
    assert failure is None
    assert approved and approved[0].canonical_relative_path == candidate.canonical_relative_path

    stale = GuidedApprovedMissingSession(
        canonical_relative_path=candidate.canonical_relative_path,
        size_bytes=candidate.size_bytes + 1,
        sha256_content_digest=candidate.sha256_content_digest,
        session_index=1,
        expected_start_time="2024-01-01T01:00:00",
        expected_duration_sec=60.0,
    )
    stale_draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(input_root),
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=1,
        session_duration_sec=60.0,
        approved_missing_sessions=[stale],
    )
    _approved, failure = _materialize_approved_missing_candidates(stale_draft, snapshot)
    assert failure is not None
    assert failure.blocking_issues[0].detail_code == "source_identity_changed"


def test_review_plan_renders_complete_accumulated_set_with_individual_removal(window):
    first = _approval(index=1)
    second = _approval(
        relative="2024_01_01-02_00_00/fluorescence.csv", index=2
    )
    plan = GuidedNewAnalysisDraftPlan(approved_missing_sessions=[first, second])
    window._refresh_guided_missing_sessions_review(plan)
    assert window._guided_missing_sessions_group.isHidden() is False
    text = "\n".join(
        label.text() for label in window._guided_missing_sessions_group.findChildren(type(window._guided_missing_sessions_explanation_label))
    )
    assert "Session 2" in text and "Session 3" in text
    assert "visible gaps" in text
    assert "missing, not zero" in text
    remove_buttons = [
        button
        for button in window._guided_missing_sessions_group.findChildren(QPushButton)
        if button.text() == "Remove"
    ]
    assert len(remove_buttons) == 2


def test_execution_payload_maps_full_approved_set_to_production_config(monkeypatch):
    """The full accumulated set, not only the latest approval, reaches Config."""
    from dataclasses import replace
    from tests.test_guided_execution_payloads import _unchecked, auth_result as auth_fixture

    try:
        auth = auth_fixture.__wrapped__(monkeypatch)
    finally:
        monkeypatch.undo()
    candidates = auth.production_intent.input_source.candidate_files[:2]
    source = replace(
        auth.production_intent.input_source,
        approved_missing_candidates=tuple(candidates),
    )
    intent = replace(auth.production_intent, input_source=source)
    intent_id = mapping.compute_guided_production_execution_intent_identity(intent)
    provisional = _unchecked(
        auth,
        production_intent=intent,
        production_intent_identity=intent_id,
        canonical_authorization_identity=None,
    )
    auth_id = authorization.compute_guided_run_authorization_identity(provisional)
    updated = _unchecked(provisional, canonical_authorization_identity=auth_id)
    payload = payloads.derive_guided_execution_payloads(
        updated,
        startup_mapping_contract=payloads.build_guided_execution_startup_mapping_contract(),
    ).config_payload
    values = {item.name: item.value for item in payload.values}
    assert values["authorized_missing_sessions"] == [
        str(Path(source.source_root_canonical, *candidate.canonical_relative_path.split("/")))
        for candidate in candidates
    ]


def test_input_processing_error_serialization(tmp_path: Path, monkeypatch):
    import tools.run_full_pipeline_deliverables as deliverables

    sys.argv = [
        "run_full_pipeline_deliverables.py",
        "--input", str(tmp_path),
        "--out", str(tmp_path / "out"),
        "--config", str(tmp_path / "config.yaml"),
        "--format", "rwd",
    ]
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")

    err = InputProcessingError(
        chunk_index=1,
        source=str(tmp_path / "source.csv"),
        phase="pass1",
        category="processing_exception",
        reason="some error",
    )
    def fake_run_cmd(*args, **kwargs):
        raise err

    monkeypatch.setattr(deliverables, "validate_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(deliverables, "run_cmd", fake_run_cmd)

    with pytest.raises(SystemExit) as excinfo:
        deliverables.main()
    assert excinfo.value.code == 1

    status_path = tmp_path / "out" / "status.json"
    assert status_path.is_file()
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    assert len(status_data["failure_details"]) == 1
    detail = status_data["failure_details"][0]
    assert detail["failure_type"] == "input_processing_failure"
    assert detail["category"] == "processing_exception"
    assert detail["phase"] == "pass1"
    assert detail["session_index"] == 1
    assert detail["eligible_for_missing_session_authorization"] is True


def test_input_processing_error_details_deduplicated_across_both_paths(tmp_path: Path, monkeypatch):
    """The wrapper can observe the same failure both via a subprocess-written
    input_processing_error.json and via an in-process InputProcessingError.
    That must collapse to exactly one structured failure detail, not two."""
    import tools.run_full_pipeline_deliverables as deliverables

    out_dir = tmp_path / "out"
    sys.argv = [
        "run_full_pipeline_deliverables.py",
        "--input", str(tmp_path),
        "--out", str(out_dir),
        "--config", str(tmp_path / "config.yaml"),
        "--format", "rwd",
        "--overwrite",
    ]
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")

    source_path = tmp_path / "source.csv"
    err = InputProcessingError(
        chunk_index=1,
        source=str(source_path),
        phase="pass1",
        category="processing_exception",
        reason="some error",
    )

    # Same failure, pre-existing on disk as if a subprocess had already
    # written it -- deliberately using forward slashes (vs the backslash
    # form InputProcessingError.source uses on Windows) to prove the
    # dedup key normalizes source before comparing.
    phasic_out = out_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True, exist_ok=True)
    (phasic_out / "input_processing_error.json").write_text(
        json.dumps({
            "category": "processing_exception",
            "phase": "pass1",
            "source": str(source_path).replace("\\", "/"),
            "session_index": 1,
            "reason": "some error",
        }),
        encoding="utf-8",
    )

    def fake_run_cmd(*args, **kwargs):
        raise err

    monkeypatch.setattr(deliverables, "validate_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(deliverables, "run_cmd", fake_run_cmd)

    with pytest.raises(SystemExit) as excinfo:
        deliverables.main()
    assert excinfo.value.code == 1

    status_path = out_dir / "status.json"
    assert status_path.is_file()
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    # Both collection paths independently observed the identical failure;
    # deduplication must collapse them to exactly one entry.
    assert len(status_data["failure_details"]) == 1
    detail = status_data["failure_details"][0]
    assert detail["category"] == "processing_exception"
    assert detail["phase"] == "pass1"
    assert detail["session_index"] == 1


def test_malformed_structured_detail_hydration(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "failed_run"
    run_dir.mkdir()

    # 1. session_index is boolean (True) - should be ignored
    status_content = {
        "status": "error",
        "phase": "final",
        "failure_details": [
            {
                "failure_type": "input_processing_failure",
                "category": "processing_exception",
                "phase": "pass1",
                "source": "some/source.csv",
                "session_index": True,
                "reason": "unreadable",
                "eligible_for_missing_session_authorization": True
            }
        ]
    }
    (run_dir / "status.json").write_text(json.dumps(status_content), encoding="utf-8")

    internal = SimpleNamespace(
        status="wrapper_failed",
        ok=False,
        wrapper_started=True,
        wrapper_completed=True,
        wrapper_returncode=1,
        allocated_run_dir=str(run_dir),
        startup_transaction_identity=None,
        pure_plan_status=None,
        allocation_status=None,
        materialization_status=None,
        wrapper_command=None,
        blocking_issues=(),
        failure_marker_path=None,
    )

    import photometry_pipeline.guided_backend_execution as backend
    monkeypatch.setattr(backend.orchestration, "run_guided_startup_to_wrapper", lambda **_kwargs: internal)

    result = backend.execute_guided_backend_run(request=None)
    assert result.failure_details == ()


def test_legacy_status_free_text_only(window, tmp_path: Path):
    input_root = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    window._guided_input_dir_edit.setText(str(input_root))
    window._guided_format_combo.setCurrentText("rwd")
    idx = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)

    result = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=()
    )
    assert window._guided_missing_session_approval_from_failed_run(result) is None


def test_gui_verification_boundaries(window, tmp_path: Path):
    input_root = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    source = _source(input_root, 1)

    window._guided_input_dir_edit.setText(str(input_root))
    window._guided_format_combo.setCurrentText("rwd")
    idx = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("60")

    # 1. NPM is rejected
    window._guided_format_combo.setCurrentText("npm")
    detail = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="processing_exception",
        phase="pass1",
        source=source,
        session_index=1,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    result = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=(detail,)
    )
    assert window._guided_missing_session_approval_from_failed_run(result) is None
    window._guided_format_combo.setCurrentText("rwd")

    # 2. Continuous is rejected
    idx_continuous = window._guided_acquisition_mode_combo.findData("continuous")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx_continuous)
    assert window._guided_missing_session_approval_from_failed_run(result) is None
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)

    # 3. Source outside root is rejected
    detail_outside = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="processing_exception",
        phase="pass1",
        source="C:/other_folder/session/fluorescence.csv",
        session_index=1,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    result_outside = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=(detail_outside,)
    )
    assert window._guided_missing_session_approval_from_failed_run(result_outside) is None

    # 4. Source absent from snapshot is rejected
    detail_absent = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="processing_exception",
        phase="pass1",
        source=str(input_root / "2024_01_01-99_99_99/fluorescence.csv"),
        session_index=1,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    result_absent = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=(detail_absent,)
    )
    assert window._guided_missing_session_approval_from_failed_run(result_absent) is None

    # 5. Session index mismatch is rejected
    detail_idx_mismatch = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="processing_exception",
        phase="pass1",
        source=source,
        session_index=2,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    result_idx_mismatch = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=(detail_idx_mismatch,)
    )
    assert window._guided_missing_session_approval_from_failed_run(result_idx_mismatch) is None

    # 6. GUI does not trust the producer's eligibility flag alone: a
    # non-processing category is rejected even when the flag claims eligible.
    detail_wrong_category = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="source_drift",
        phase="pass1",
        source=source,
        session_index=1,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    result_wrong_category = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=(detail_wrong_category,)
    )
    assert window._guided_missing_session_approval_from_failed_run(result_wrong_category) is None

    # 7. GUI does not trust the producer's eligibility flag alone: a
    # non-processing phase (e.g. cancellation) is rejected even when the
    # flag claims eligible.
    detail_wrong_phase = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="processing_exception",
        phase="cancellation",
        source=source,
        session_index=1,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    result_wrong_phase = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=(detail_wrong_phase,)
    )
    assert window._guided_missing_session_approval_from_failed_run(result_wrong_phase) is None


def _setup_guided_recording(window, tmp_path, input_root, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_root))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._mode_combo.setCurrentText("both")
    idx = window._format_combo.findText("rwd")
    window._format_combo.setCurrentIndex(idx)

    rois = ("Region0",)
    source_files = [input_root / name / "fluorescence.csv" for name in ("2024_01_01-00_00_00", "2024_01_01-01_00_00", "2024_01_01-02_00_00")]
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": len(source_files),
        "n_preview": len(source_files),
        "sessions": [
            {
                "index": index,
                "session_id": name,
                "path": str(source_file),
                "included_in_preview": True,
            }
            for index, (name, source_file) in enumerate(zip(("2024_01_01-00_00_00", "2024_01_01-01_00_00", "2024_01_01-02_00_00"), source_files))
        ],
        "rois": [{"roi_id": roi} for roi in rois],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    # Mock _infer_dataset_contract_overrides to return complete set of overrides (with target_fs_hz)
    monkeypatch.setattr(
        window,
        "_infer_dataset_contract_overrides",
        lambda _fmt: {
            "rwd_time_col": "TimeStamp",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
            "target_fs_hz": 10.0,
            "chunk_duration_sec": 600.0,
        },
    )
    monkeypatch.setattr(
        window,
        "_infer_rwd_chunk_contract",
        lambda path: {
            "csv_path": path,
            "time_col": "TimeStamp",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
            "timestamp_unit": "seconds",
            "fs_hz": 10.0,
            "median_dt": 0.1,
            "sample_count": 6000,
            "chunk_duration_sec": 600.0,
            "timestamp_duration_sec": 600.0,
            "metadata_effective_fs_hz": None,
            "metadata_continuous_time_sec": None,
        },
    )

    from photometry_pipeline.core.types import Chunk
    time_sec = np.arange(6000, dtype=float) / 10.0
    uv = np.ones(6000)
    sig = np.ones(6000) * 1.2

    import photometry_pipeline.preview.correction_preview as correction_preview_module
    def fake_load_chunk(path, input_format, _config, chunk_id):
        return Chunk(
            chunk_id=chunk_id,
            source_file=path,
            format=input_format,
            time_sec=time_sec,
            uv_raw=uv.reshape(-1, 1),
            sig_raw=sig.reshape(-1, 1),
            fs_hz=10.0,
            channel_names=list(rois),
            metadata={},
        )
    monkeypatch.setattr(correction_preview_module, "load_chunk", fake_load_chunk)

    acquisition_idx = window._guided_acquisition_mode_combo.findData("intermittent")
    if acquisition_idx >= 0:
        window._guided_acquisition_mode_combo.setCurrentIndex(acquisition_idx)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("600")
    window._guided_dataset_contract_apply_btn.click()

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )

    roi = "Region0"
    roi_idx = window._guided_preview_roi_combo.findData(roi)
    window._guided_preview_roi_combo.setCurrentIndex(roi_idx)
    window._guided_preview_generate_btn.click()

    window._guided_confirm_roi_combo.setCurrentIndex(
        window._guided_confirm_roi_combo.findData(roi)
    )
    window._guided_confirm_chunk_combo.setCurrentIndex(0)
    strategy_index = window._guided_confirm_strategy_combo.findText("Global Linear Regression")
    window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
    window._guided_confirm_ack_cb.setChecked(True)
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

    window._guided_review_go_to_run_btn.click()


def test_non_wrapper_failed_result_is_rejected_even_with_eligible_detail(window, tmp_path: Path):
    input_root = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    source = _source(input_root, 1)

    window._guided_input_dir_edit.setText(str(input_root))
    window._guided_format_combo.setCurrentText("rwd")
    idx = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("60")

    detail = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="processing_exception",
        phase="pass1",
        source=source,
        session_index=1,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    for status in ("wrapper_completed_needs_review_loading", "refused_before_startup", "startup_allocation_failed", ""):
        result = GuidedBackendExecutionResult(
            status=status, ok=False, user_visible_state="failed", user_summary="Error",
            run_directory=str(tmp_path), completed_run_candidate_path=None,
            requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
            blocking_issues=(), diagnostics=None, failure_details=(detail,)
        )
        assert window._guided_missing_session_approval_from_failed_run(result) is None, status


def test_unknown_failure_type_is_rejected_even_with_eligible_fields(window, tmp_path: Path):
    input_root = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    source = _source(input_root, 1)

    window._guided_input_dir_edit.setText(str(input_root))
    window._guided_format_combo.setCurrentText("rwd")
    idx = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("60")

    for bad_failure_type in ("cancellation", "validation_failure", "output_write_failure", ""):
        detail = GuidedExecutionFailureDetail(
            failure_type=bad_failure_type,
            category="processing_exception",
            phase="pass1",
            source=source,
            session_index=1,
            reason="unreadable",
            eligible_for_missing_session_authorization=True,
        )
        result = GuidedBackendExecutionResult(
            status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
            run_directory=str(tmp_path), completed_run_candidate_path=None,
            requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
            blocking_issues=(), diagnostics=None, failure_details=(detail,)
        )
        assert window._guided_missing_session_approval_from_failed_run(result) is None, bad_failure_type


def test_real_wrapper_failed_input_processing_failure_remains_accepted(window, tmp_path: Path):
    input_root = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    source = _source(input_root, 1)

    window._guided_input_dir_edit.setText(str(input_root))
    window._guided_format_combo.setCurrentText("rwd")
    idx = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("60")

    detail = GuidedExecutionFailureDetail(
        failure_type="input_processing_failure",
        category="processing_exception",
        phase="pass1",
        source=source,
        session_index=1,
        reason="unreadable",
        eligible_for_missing_session_authorization=True,
    )
    result = GuidedBackendExecutionResult(
        status="wrapper_failed", ok=False, user_visible_state="failed", user_summary="Error",
        run_directory=str(tmp_path), completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False, wrapper_started=True, wrapper_completed=True,
        blocking_issues=(), diagnostics=None, failure_details=(detail,)
    )
    approval = window._guided_missing_session_approval_from_failed_run(result)
    assert approval is not None
    assert approval.canonical_relative_path == "2024_01_01-01_00_00/fluorescence.csv"
    assert approval.session_index == 1


def test_guided_missing_session_real_gui_rerun_lifecycle(window, tmp_path: Path, monkeypatch, qapp):
    input_root = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    _setup_guided_recording(window, tmp_path, input_root, monkeypatch)

    build_identity = mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    import photometry_pipeline.guided_execution_request_builder as request_builder
    monkeypatch.setattr(
        request_builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
    )

    window._guided_backend_validate_btn.click()
    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True

    # Store initial plan settings for assertions
    initial_plan = window._build_guided_new_analysis_draft_plan()
    initial_rois = [choice.roi_id for choice in initial_plan.per_roi_correction_strategy_choices]
    initial_correction_strategies = [choice.selected_strategy for choice in initial_plan.per_roi_correction_strategy_choices]
    initial_output_path = initial_plan.output_base_path

    # First run fails on Session 2 (index 1)
    window._guided_run_btn.click()
    assert window._guided_backend_execution_active is True

    # QMessageBox automation: choice "Continue with this session missing"
    box_instances = []
    def mock_exec(box_instance):
        box_instances.append(box_instance)
        for btn in box_instance.buttons():
            if "Continue with this session missing" in btn.text():
                box_instance._test_clicked_btn = btn
                break
        return 0

    def mock_clicked_button(box_instance):
        return getattr(box_instance, "_test_clicked_btn", None)

    monkeypatch.setattr(QMessageBox, "exec", mock_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", mock_clicked_button)

    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)

    # First Run assertions
    assert len(box_instances) == 1
    result = window._guided_backend_execution_result
    assert result.status == "wrapper_failed"
    assert result.ok is False
    assert len(result.failure_details) == 1
    fail_detail = result.failure_details[0]
    assert fail_detail.source.replace("\\", "/").endswith("2024_01_01-01_00_00/fluorescence.csv")
    assert fail_detail.category == "processing_exception"
    assert fail_detail.phase in ("pass1", "pass1a", "pass1b", "tonic_pass1c", "pass2")
    assert fail_detail.session_index == 1
    assert classify_completed_run_candidate(result.run_directory)[0] is False

    # Approval assertions
    assert len(window._guided_approved_missing_sessions) == 1
    approved = window._guided_approved_missing_sessions[0]
    assert approved.canonical_relative_path == "2024_01_01-01_00_00/fluorescence.csv"
    assert approved.session_index == 1

    # Verify GUI parameters unchanged
    post_plan = window._build_guided_new_analysis_draft_plan()
    assert [choice.roi_id for choice in post_plan.per_roi_correction_strategy_choices] == initial_rois
    assert [choice.selected_strategy for choice in post_plan.per_roi_correction_strategy_choices] == initial_correction_strategies
    assert post_plan.output_base_path == initial_output_path

    # Verify validation is stale and Run is disabled when on Run step
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Run"))
    assert window._guided_run_btn.isEnabled() is False

    # Revalidation
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_backend_validate_btn.click()
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted", f"Revalidation failed: {outcome.status}, issues: {outcome.blocking_issues}, stale reasons: {window._guided_new_analysis_output_policy_stale_reasons}"
    window._guided_review_go_to_run_btn.click()
    assert window._guided_run_btn.isEnabled() is True

    # Second run completions
    first_run_dir = result.run_directory
    window._guided_run_btn.click()
    assert window._guided_backend_execution_active is True

    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)

    # Second Run assertions
    rerun_result = window._guided_backend_execution_result
    assert rerun_result.status == "wrapper_completed_needs_review_loading"
    assert rerun_result.ok is True
    assert rerun_result.run_directory != first_run_dir

    # Loader verification
    ok, details = classify_completed_run_candidate(rerun_result.run_directory)
    assert ok is True
    # The first failed run remains rejected
    assert classify_completed_run_candidate(first_run_dir)[0] is False

    # Completeness verification
    completeness_path = (
        Path(rerun_result.run_directory) / "_analysis" / "phasic_out" / "input_processing_completeness.json"
    )
    assert completeness_path.is_file()
    completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
    assert len(completeness["missing"]) == 1
    assert completeness["missing"][0]["index"] == 1
    # Check that Session 3 remains chronological session index 2 and retains timestamp/source
    assert completeness["processed"][-1]["index"] == 2
    assert "2024_01_01-02_00_00" in completeness["processed"][-1]["source"]
