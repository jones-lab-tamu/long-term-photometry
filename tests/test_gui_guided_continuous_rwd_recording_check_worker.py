from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gui.main_window import (
    _GuidedContinuousRwdRecordingCheckFailure,
    _GuidedContinuousRwdRecordingCheckRequest,
    _GuidedContinuousRwdRecordingCheckRequestError,
    _GuidedContinuousRwdRecordingCheckSuccess,
    _GuidedContinuousRwdRecordingCheckWorker,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    EVALUATION_INTERRUPTED,
    MATERIAL_LONG_INTERVAL_DETECTED,
    SHORT_AND_LONG_DISCONTINUITIES_DETECTED,
    SHORT_INTERVAL_ANOMALY_DETECTED,
    SOURCE_CHANGED_OR_MISMATCHED,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    ContinuousRwdRecordingAuthorityError,
    GuidedContinuousRwdRecordingDescription,
)


def _request(folder="C:/recording", rois=("CH2", "CH1")):
    return _GuidedContinuousRwdRecordingCheckRequest(folder, rois)


def _signals(worker):
    captured = {"stages": [], "success": [], "failure": [], "cancelled": 0}
    worker.stage_changed.connect(captured["stages"].append)
    worker.succeeded.connect(captured["success"].append)
    worker.failed.connect(captured["failure"].append)
    worker.cancelled.connect(
        lambda: captured.__setitem__("cancelled", captured["cancelled"] + 1)
    )
    return captured


def _terminal_count(captured):
    return len(captured["success"]) + len(captured["failure"]) + captured["cancelled"]


def _patch_pipeline(monkeypatch, *, inspection=None, recording=None, evaluation=None):
    import photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation as b2b
    import photometry_pipeline.guided_continuous_rwd_recording as b1
    import photometry_pipeline.io.rwd_continuous_source as cr1a

    inspection = inspection or SimpleNamespace(
        status="completed",
        outcome_category="inspection_completed",
        scientist_summary="completed",
    )
    recording = recording or SimpleNamespace(
        source=SimpleNamespace(
            fluorescence_path_canonical="C:/recording/Fluorescence.csv"
        )
    )
    evaluation = evaluation or SimpleNamespace(outcome=CONTINUITY_PASSED)
    calls = {"callbacks": [], "b1": 0, "b2b": 0}

    def inspect_source(folder, *, cancellation_check):
        calls["folder"] = folder
        calls["callbacks"].append(cancellation_check)
        return inspection

    def build_recording(value, *, included_roi_ids):
        calls["b1"] += 1
        calls["inspection"] = value
        calls["rois"] = included_roi_ids
        return recording

    def evaluate(value, *, source_path, cancellation_requested):
        calls["b2b"] += 1
        calls["recording"] = value
        calls["source_path"] = source_path
        calls["callbacks"].append(cancellation_requested)
        return evaluation

    monkeypatch.setattr(cr1a, "inspect_continuous_rwd_acquisition_folder", inspect_source)
    monkeypatch.setattr(b1, "build_guided_continuous_rwd_recording_description", build_recording)
    monkeypatch.setattr(b2b, "evaluate_continuous_rwd_timestamp_continuity", evaluate)
    return calls


def test_request_is_frozen_valid_and_preserves_roi_order():
    request = _request()
    assert request.selected_acquisition_folder == "C:/recording"
    assert request.included_roi_ids == ("CH2", "CH1")
    with pytest.raises(FrozenInstanceError):
        request.selected_acquisition_folder = "changed"


@pytest.mark.parametrize("folder", ["", "   ", None, Path("recording")])
def test_request_refuses_invalid_folder(folder):
    with pytest.raises(_GuidedContinuousRwdRecordingCheckRequestError):
        _GuidedContinuousRwdRecordingCheckRequest(folder, ("CH1",))


@pytest.mark.parametrize("rois", [(), [], ("",), ("  ",), (1,), ("CH1", "CH1")])
def test_request_refuses_invalid_roi_tuple(rois):
    with pytest.raises(_GuidedContinuousRwdRecordingCheckRequestError):
        _GuidedContinuousRwdRecordingCheckRequest("C:/recording", rois)


def test_worker_refuses_wrong_request_type():
    with pytest.raises(_GuidedContinuousRwdRecordingCheckRequestError):
        _GuidedContinuousRwdRecordingCheckWorker(object())


def _write_real_source(folder: Path) -> Path:
    folder.mkdir()
    source = folder / "Fluorescence.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("Time(s)", "CH2-410", "CH2-470", "CH1-410", "CH1-470")
        )
        for index in range(6001):
            writer.writerow((index / 10.0, 1.0, 2.0, 3.0, 4.0))
    return source


def test_real_synthetic_source_runs_cr1a_b1_and_b2b(tmp_path):
    folder = tmp_path / "recording"
    source = _write_real_source(folder)
    request = _request(str(folder))
    worker = _GuidedContinuousRwdRecordingCheckWorker(request)
    captured = _signals(worker)

    worker.run()

    assert captured["stages"] == [
        "inspecting_recording",
        "checking_timestamp_continuity",
    ]
    assert _terminal_count(captured) == 1
    assert captured["failure"] == [] and captured["cancelled"] == 0
    success = captured["success"][0]
    assert isinstance(success, _GuidedContinuousRwdRecordingCheckSuccess)
    assert success.request is request
    assert isinstance(success.recording, GuidedContinuousRwdRecordingDescription)
    assert success.recording.roi.included_roi_ids == ("CH1", "CH2")
    assert success.continuity_evaluation.outcome == CONTINUITY_PASSED
    assert success.current_source_path == success.recording.source.fluorescence_path_canonical
    assert Path(success.current_source_path) == source.resolve()


def test_success_uses_exact_request_authorities_path_and_shared_callback(monkeypatch):
    request = _request()
    calls = _patch_pipeline(monkeypatch)
    worker = _GuidedContinuousRwdRecordingCheckWorker(request)
    captured = _signals(worker)

    worker.run()

    assert captured["stages"] == [
        "inspecting_recording",
        "checking_timestamp_continuity",
    ]
    assert _terminal_count(captured) == 1
    success = captured["success"][0]
    assert success.request is request
    assert success.recording is calls["recording"]
    assert success.current_source_path == calls["source_path"]
    assert calls["folder"] == request.selected_acquisition_folder
    assert calls["rois"] is request.included_roi_ids
    assert calls["callbacks"][0] is calls["callbacks"][1]


def test_bounded_cr1a_refusal_is_one_inspection_failure(monkeypatch):
    refusal = SimpleNamespace(
        status="failed",
        outcome_category="fluorescence_csv_missing",
        scientist_summary="Select the folder containing Fluorescence.csv.",
    )
    calls = _patch_pipeline(monkeypatch, inspection=refusal)
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    worker.run()

    assert _terminal_count(captured) == 1 and calls["b1"] == 0
    assert captured["success"] == [] and captured["cancelled"] == 0
    assert captured["failure"] == [
        _GuidedContinuousRwdRecordingCheckFailure(
            "inspection",
            "fluorescence_csv_missing",
            "Select the folder containing Fluorescence.csv.",
        )
    ]
    payload = repr(captured["failure"][0]).lower()
    assert "sha256" not in payload and "identity" not in payload


def test_malformed_cr1a_failure_fields_fall_back_to_bounded_text(monkeypatch):
    refusal = SimpleNamespace(
        status="failed", outcome_category=object(), scientist_summary="x" * 501
    )
    _patch_pipeline(monkeypatch, inspection=refusal)
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    worker.run()
    assert captured["failure"][0] == _GuidedContinuousRwdRecordingCheckFailure(
        "inspection", "inspection_failed", "The recording could not be inspected."
    )


def test_interrupted_cr1a_is_cancellation_not_failure(monkeypatch):
    interrupted = SimpleNamespace(
        status="failed",
        outcome_category="inspection_interrupted",
        scientist_summary="interrupted",
    )
    calls = _patch_pipeline(monkeypatch, inspection=interrupted)
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    worker.run()
    assert captured["cancelled"] == 1 and _terminal_count(captured) == 1
    assert captured["failure"] == [] and calls["b1"] == 0


def test_b1_authority_rejection_is_bounded_and_stops_before_b2b(monkeypatch):
    import photometry_pipeline.guided_continuous_rwd_recording as b1

    calls = _patch_pipeline(monkeypatch)

    def reject(*_args, **_kwargs):
        raise ContinuousRwdRecordingAuthorityError("secret raw authority detail")

    monkeypatch.setattr(b1, "build_guided_continuous_rwd_recording_description", reject)
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request(rois=("UNKNOWN",)))
    captured = _signals(worker)
    worker.run()
    assert _terminal_count(captured) == 1 and calls["b2b"] == 0
    failure = captured["failure"][0]
    assert failure == _GuidedContinuousRwdRecordingCheckFailure(
        "recording_authority",
        "recording_authority_rejected",
        "The inspected recording could not be accepted.",
    )
    assert "secret" not in repr(failure)


@pytest.mark.parametrize(
    ("outcome", "summary_fragment"),
    [
        (SHORT_INTERVAL_ANOMALY_DETECTED, "shorter"),
        (MATERIAL_LONG_INTERVAL_DETECTED, "material timestamp gaps"),
        (SHORT_AND_LONG_DISCONTINUITIES_DETECTED, "both short"),
        (SOURCE_CHANGED_OR_MISMATCHED, "no longer matches"),
    ],
)
def test_b2b_nonpass_outcomes_are_bounded_failures(
    monkeypatch, outcome, summary_fragment
):
    evaluation = SimpleNamespace(outcome=outcome, failure_reason="secret_internal_reason")
    _patch_pipeline(monkeypatch, evaluation=evaluation)
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    worker.run()
    assert _terminal_count(captured) == 1 and captured["success"] == []
    failure = captured["failure"][0]
    assert failure.stage == "continuity" and failure.category == outcome
    assert summary_fragment in failure.scientist_summary
    assert "secret_internal_reason" not in failure.scientist_summary


def test_b2b_interruption_is_cancellation(monkeypatch):
    _patch_pipeline(
        monkeypatch, evaluation=SimpleNamespace(outcome=EVALUATION_INTERRUPTED)
    )
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    worker.run()
    assert captured["cancelled"] == 1 and _terminal_count(captured) == 1
    assert captured["success"] == [] and captured["failure"] == []


def test_cancellation_before_run_is_only_terminal(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    worker.request_cancel()
    worker.run()
    assert captured["cancelled"] == 1 and _terminal_count(captured) == 1
    assert captured["stages"] == [] and "folder" not in calls


@pytest.mark.parametrize(
    "cancel_point",
    ["inspection", "after_inspection", "after_b1", "b2b", "before_success"],
)
def test_cancellation_at_each_scientific_boundary_emits_only_cancelled(
    monkeypatch, cancel_point
):
    import photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation as b2b
    import photometry_pipeline.guided_continuous_rwd_recording as b1
    import photometry_pipeline.io.rwd_continuous_source as cr1a

    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    recording = SimpleNamespace(
        source=SimpleNamespace(fluorescence_path_canonical="C:/recording/Fluorescence.csv")
    )

    def inspect_source(_folder, *, cancellation_check):
        if cancel_point == "inspection":
            worker.request_cancel()
            return SimpleNamespace(outcome_category="inspection_interrupted")
        if cancel_point == "after_inspection":
            worker.request_cancel()
        return SimpleNamespace(status="completed", outcome_category="inspection_completed")

    def build_recording(*_args, **_kwargs):
        if cancel_point == "after_b1":
            worker.request_cancel()
        return recording

    def evaluate(*_args, cancellation_requested, **_kwargs):
        if cancel_point == "b2b":
            worker.request_cancel()
            return SimpleNamespace(outcome=EVALUATION_INTERRUPTED)
        if cancel_point == "before_success":
            worker.request_cancel()
        return SimpleNamespace(outcome=CONTINUITY_PASSED)

    monkeypatch.setattr(cr1a, "inspect_continuous_rwd_acquisition_folder", inspect_source)
    monkeypatch.setattr(b1, "build_guided_continuous_rwd_recording_description", build_recording)
    monkeypatch.setattr(b2b, "evaluate_continuous_rwd_timestamp_continuity", evaluate)
    worker.run()
    assert captured["cancelled"] == 1 and _terminal_count(captured) == 1
    assert captured["success"] == [] and captured["failure"] == []


def test_request_cancel_is_directly_thread_safe_without_worker_event_loop():
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    caller = threading.Thread(target=worker.request_cancel)
    caller.start()
    caller.join(timeout=2)
    assert not caller.is_alive()
    captured = _signals(worker)
    worker.run()
    assert captured["cancelled"] == 1


@pytest.mark.parametrize("stage", ["inspection", "recording_authority", "continuity"])
def test_unexpected_stage_exception_is_contained(monkeypatch, stage):
    import photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation as b2b
    import photometry_pipeline.guided_continuous_rwd_recording as b1
    import photometry_pipeline.io.rwd_continuous_source as cr1a

    _patch_pipeline(monkeypatch)

    def explode(*_args, **_kwargs):
        raise RuntimeError("secret traceback payload")

    target = {
        "inspection": (cr1a, "inspect_continuous_rwd_acquisition_folder"),
        "recording_authority": (b1, "build_guided_continuous_rwd_recording_description"),
        "continuity": (b2b, "evaluate_continuous_rwd_timestamp_continuity"),
    }[stage]
    monkeypatch.setattr(*target, explode)
    worker = _GuidedContinuousRwdRecordingCheckWorker(_request())
    captured = _signals(worker)
    worker.run()
    assert _terminal_count(captured) == 1
    failure = captured["failure"][0]
    assert failure.stage == stage and failure.category == "unexpected_check_failure"
    assert "secret" not in repr(failure)


def test_worker_isolation_and_capability_boundary():
    lowered = inspect.getsource(_GuidedContinuousRwdRecordingCheckWorker).lower()
    assert "mainwindow" not in lowered
    assert "qthread" not in lowered and "movetothread" not in lowered
    assert "review_binding" not in lowered and "b3a" not in lowered
    for forbidden in (
        "qsettings",
        "json",
        "hashlib",
        "serializ",
        "cache",
        "validation",
        "startup",
        "authorization",
        "execution",
        "completion",
        "qwidget",
        "qpushbutton",
    ):
        assert forbidden not in lowered
    assert set(_GuidedContinuousRwdRecordingCheckWorker.__dict__) >= {
        "stage_changed",
        "succeeded",
        "failed",
        "cancelled",
        "run",
        "request_cancel",
    }
