"""Phase 3D regressions: the Guided New Analysis output-destination field
must never silently adopt a completed Guided Run's own folder as an
automatic/remembered default, and must never silently rewrite an explicit
user selection of one either.

gui/main_window.py._set_output_dir_from_automatic_source is the single
authoritative boundary: it is the only way an app-supplied, restored, or
completed-run-derived value reaches the shared output-destination field
(used by both QSettings restore and Open Results), and it is what makes
the resulting Guided-field mirror
(gui/main_window.py._sync_guided_setup_from_full) eligible for completed-
run normalization via
gui/main_window.py._normalize_guided_output_destination_default. An
explicit user edit (typed or browsed directly into either output field)
is never routed through that boundary, so it is never normalized --
instead gui/main_window.py._guided_select_data_readiness blocks Guided
progress with a truthful, scientist-facing message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.main_window import GUIDED_OUTPUT_COMPLETED_RUN_GUIDANCE, GUIDED_WORKFLOW_STEPS, MainWindow
from tests.test_guided_gui_run_execution_wiring import (
    _drive_real_guided_rwd_setup,
    _pump_until,
)
from tests.test_guided_startup_orchestration_real_wrapper_boundary import (
    _real_wrapper_runner,
)


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


def _write_completed_run_status(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )


def _reconfirm_guided_local_preview_strategies(
    window, rois=("CH1", "CH2", "CH3")
) -> None:
    """Re-confirm each ROI's local-preview correction strategy. Output
    destination is part of the pre-existing (Phase 3D-unrelated) local-
    preview setup signature, so any output-destination change -- including
    a Phase 3D normalization -- correctly requires this, exactly as it
    would for a real user changing the output destination by hand."""
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    for roi in rois:
        window._guided_preview_roi_combo.setCurrentIndex(
            window._guided_preview_roi_combo.findData(roi)
        )
        window._guided_preview_generate_btn.click()
        window._guided_confirm_roi_combo.setCurrentIndex(
            window._guided_confirm_roi_combo.findData(roi)
        )
        window._guided_confirm_chunk_combo.setCurrentIndex(0)
        strategy_index = window._guided_confirm_strategy_combo.findText(
            "Global Linear Regression"
        )
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_confirm_ack_cb.setChecked(True)
        window._guided_confirm_mark_btn.click()
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )


# ---------------------------------------------------------------------------
# A. Automatic/remembered default normalization (natural paths)
# ---------------------------------------------------------------------------


def test_qsettings_restored_completed_run_default_normalizes_to_parent_on_startup(
    qapp, tmp_path
):
    """Reproduces Jeff's real defect via the real QSettings-restore path:
    a genuinely completed prior Guided Run's own folder had been
    remembered as the output destination (exactly what Open Results
    persists), and on the next app start it must not silently become the
    default output base for a new analysis. The parent is never set
    directly in this test -- only the completed-run child is supplied,
    through the real settings-restore path the audit identified
    (gui/main_window.py._load_settings_into_widgets)."""
    parent = tmp_path / "study_outputs"
    parent.mkdir()
    completed_run = parent / "guided_run_20260101T000000000000Z_abc123"
    _write_completed_run_status(completed_run)

    settings = QSettings(
        str(tmp_path / "remembered_output.ini"), QSettings.IniFormat
    )
    settings.clear()
    settings.beginGroup("run_config")
    settings.setValue("output_dir", str(completed_run))
    settings.endGroup()
    settings.sync()

    window = MainWindow(settings=settings)
    try:
        assert window._guided_output_dir_edit.text() == str(parent)
        # Full Control's own field is a different, dual-purpose widget (it
        # also identifies a currently-loaded completed run for unrelated
        # Full Control review features) and must keep the raw remembered
        # value -- only the Guided-specific field is normalized.
        assert window._output_dir.text() == str(completed_run)
    finally:
        window.close()
        window.deleteLater()
        settings.clear()


def test_automatic_source_still_syncs_guided_field_when_value_is_unchanged(
    window, tmp_path
):
    """Qt does not emit textChanged when setText's argument already equals
    the widget's current text, so the usual signal-driven sync from
    _output_dir to the Guided field would silently not run for a same-
    value automatic assignment. _set_output_dir_from_automatic_source must
    still reach the Guided field (and still normalize) in that case."""
    parent = tmp_path / "parent"
    parent.mkdir()
    completed_run = parent / "guided_run_prior"
    _write_completed_run_status(completed_run)

    # Establish _output_dir already holding the completed-run child.
    window._output_dir.setText(str(completed_run))

    # A deliberate explicit edit elsewhere in between: proves the upcoming
    # automatic call does not accidentally inherit "explicit" provenance
    # left over from this edit (and, as a side effect of the existing
    # bidirectional Full Control <-> Guided sync, moves _output_dir's text
    # away from completed_run again).
    ordinary_path = tmp_path / "ordinary"
    ordinary_path.mkdir()
    window._guided_output_dir_edit.setText(str(ordinary_path))
    assert window._guided_output_destination_provenance == "explicit"

    # Re-establish the exact precondition under test: _output_dir already
    # holding the same value _set_output_dir_from_automatic_source is about
    # to be called with, so Qt will not emit textChanged for that call.
    window._output_dir.setText(str(completed_run))
    assert window._output_dir.text() == str(completed_run)

    window._set_output_dir_from_automatic_source(str(completed_run))

    assert window._output_dir.text() == str(completed_run)
    assert window._guided_output_dir_edit.text() == str(parent)
    assert window._guided_output_destination_provenance == "automatic"


def test_automatic_completed_run_default_reaches_run_readiness_via_normalized_parent(
    window, tmp_path, monkeypatch, qapp
):
    """Deep natural-path regression: an automatic/remembered/completed-
    run-derived output default (supplied through the real
    _set_output_dir_from_automatic_source boundary used by both QSettings
    restore and Open Results -- never the parent set directly) must
    normalize to its parent, and that corrected parent must be what the
    draft plan, plan identity, authorization, and the retained startup
    request all agree on, reaching real Run readiness and passing beyond
    the pure-plan output-safety refusal on the real production Run path.
    """
    _drive_real_guided_rwd_setup(
        window, tmp_path, monkeypatch, apply_feature_defaults=True
    )
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted"

    remembered_parent = tmp_path / "remembered_parent"
    remembered_parent.mkdir()
    remembered_completed_run = remembered_parent / "guided_run_prior_completed"
    _write_completed_run_status(remembered_completed_run)

    window._set_output_dir_from_automatic_source(str(remembered_completed_run))
    assert window._guided_output_dir_edit.text() == str(remembered_parent)

    # The output-destination change genuinely invalidates the prior
    # authorization -- the same authoritative click-time identity recheck
    # that guards every other Guided setup change (not merely a cosmetic
    # button state) now disagrees with what was previously validated.
    assert window._guided_current_plan_identity_is_validated() is False

    # Output destination is part of the pre-existing local-preview setup
    # signature (unrelated to Phase 3D), so the change also requires
    # re-confirming per-ROI correction strategies -- exactly as it would
    # for a real user changing the output destination by hand.
    _reconfirm_guided_local_preview_strategies(window)
    window._guided_feature_event_apply_btn.click()
    window._guided_backend_validate_btn.click()
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True

    retained_request = window._current_guided_startup_transaction_request()
    assert retained_request is not None
    assert (
        retained_request.filesystem_policy.output_base_is_completed_run_root
        is False
    )
    assert (
        Path(retained_request.output_base_canonical).resolve()
        == remembered_parent.resolve()
    )
    assert (
        Path(retained_request.planned_allocated_run_dir).parent.resolve()
        == remembered_parent.resolve()
    )

    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    runner, calls = _real_wrapper_runner(monkeypatch)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)

    result = window._guided_backend_execution_result
    assert result.status != "refused_before_startup"
    assert result.blocking_issues == ()
    assert calls == {"live_verify": 1, "analysis": 0, "root_makedirs": 0}


# ---------------------------------------------------------------------------
# B. Explicit-selection guard
# ---------------------------------------------------------------------------


def test_explicit_completed_run_output_selection_stays_visible_and_blocks(
    window, tmp_path, monkeypatch
):
    """A user who explicitly browses to or types a completed Guided Run's
    own folder must see that exact selection stay visible (never silently
    rewritten to the parent), get the truthful scientist-facing guidance,
    and be blocked from proceeding -- no output created, no backend
    involved. Simulated through the real widget setText the folder-browse
    result handler (_browse_dir) itself calls -- not a private attribute.
    """
    execution_calls = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: execution_calls.append(request),
    )

    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    input_dir = tmp_path / "raw_input"
    input_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))

    completed_run = tmp_path / "output" / "guided_run_prior"
    _write_completed_run_status(completed_run)

    window._guided_output_dir_edit.setText(str(completed_run))

    assert window._guided_output_dir_edit.text() == str(completed_run)

    ready, reason = window._guided_select_data_readiness()
    assert ready is False
    assert reason == GUIDED_OUTPUT_COMPLETED_RUN_GUIDANCE
    assert (
        "run root" not in reason.lower()
        and "manifest" not in reason.lower()
        and "status.json" not in reason.lower()
    )

    window._refresh_guided_navigation_state()
    assert (
        window._guided_select_data_continue_status.text()
        == GUIDED_OUTPUT_COMPLETED_RUN_GUIDANCE
    )
    assert window._guided_select_data_continue_btn.isEnabled() is False
    assert window._guided_run_btn.isEnabled() is False
    assert window._current_guided_startup_transaction_request() is None
    assert window._guided_backend_execution_result is None
    # Nothing new was created next to the completed run itself.
    assert {p.name for p in completed_run.parent.iterdir()} == {"guided_run_prior"}

    # Direct proof, not merely inferred from disabled buttons: the real
    # worker-start seam Run press would use was never reached. Run is
    # correctly disabled, so it is never pressed here -- this asserts the
    # backend-invocation seam saw zero calls throughout the whole scenario.
    assert execution_calls == []


# ---------------------------------------------------------------------------
# C. Non-completed candidates must never be altered or blocked
# ---------------------------------------------------------------------------


def test_ordinary_output_base_is_not_normalized(window, tmp_path):
    ordinary = tmp_path / "ordinary_output"
    ordinary.mkdir()
    assert window._normalize_guided_output_destination_default(
        str(ordinary)
    ) == str(ordinary)
    assert window._guided_output_destination_is_completed_run_root(
        str(ordinary)
    ) is False


def test_empty_unresolved_output_field_is_not_normalized(window):
    assert window._normalize_guided_output_destination_default("") == ""
    assert window._guided_output_destination_is_completed_run_root("") is False


def test_nonexistent_candidate_with_existing_parent_is_not_normalized(
    window, tmp_path
):
    parent = tmp_path / "existing_parent"
    parent.mkdir()
    candidate = parent / "not_created_yet"
    assert window._normalize_guided_output_destination_default(
        str(candidate)
    ) == str(candidate)
    assert window._guided_output_destination_is_completed_run_root(
        str(candidate)
    ) is False


def test_incomplete_run_directory_is_not_normalized(window, tmp_path):
    run_dir = tmp_path / "guided_run_incomplete"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "materializing", "status": "in_progress"}),
        encoding="utf-8",
    )
    assert window._normalize_guided_output_destination_default(
        str(run_dir)
    ) == str(run_dir)
    assert window._guided_output_destination_is_completed_run_root(
        str(run_dir)
    ) is False


def test_failed_run_directory_is_not_normalized(window, tmp_path):
    run_dir = tmp_path / "guided_run_failed"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "failed"}),
        encoding="utf-8",
    )
    assert window._normalize_guided_output_destination_default(
        str(run_dir)
    ) == str(run_dir)
    assert window._guided_output_destination_is_completed_run_root(
        str(run_dir)
    ) is False


def test_name_only_guided_run_directory_without_completion_contract_is_not_normalized(
    window, tmp_path
):
    """A directory merely named guided_run_... with no status.json /
    run_report.json / MANIFEST.json evidence at all must not be treated
    as a completed run -- name alone proves nothing."""
    run_dir = tmp_path / "guided_run_20260101T000000000000Z_deadbeef"
    run_dir.mkdir()
    assert window._normalize_guided_output_destination_default(
        str(run_dir)
    ) == str(run_dir)
    assert window._guided_output_destination_is_completed_run_root(
        str(run_dir)
    ) is False


def test_normal_directory_containing_old_run_children_is_not_normalized(
    window, tmp_path
):
    """The output base itself (not one of its children) is what gets
    classified -- a normal parent directory that happens to already
    contain completed run children is not itself a completed run root."""
    parent = tmp_path / "parent_with_old_runs"
    parent.mkdir()
    _write_completed_run_status(parent / "guided_run_old_one")
    assert window._normalize_guided_output_destination_default(
        str(parent)
    ) == str(parent)
    assert window._guided_output_destination_is_completed_run_root(
        str(parent)
    ) is False
