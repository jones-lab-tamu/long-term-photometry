"""CR1-F1-G: the Run page setup check on one continuous recording.

A scientist reached the Run page with a complete continuous plan and pressed
"Check my setup". The page answered:

    Guided setup is incomplete or stale.
    Acquisition mode 'continuous' is not supported by the first subset.

while the panel beside it said "Preparing continuous analysis..." and offered a
Stop button.

Both statements came from different authorities. Continuous preparation -- the
thing that actually makes a continuous plan runnable -- now starts only when
the scientist explicitly clicks Check my setup. The setup check uses the same
accepted preparation authority, rather than asking the backend-validation
materializer that begins by refusing anything that is not ``intermittent``.

Structure of this file:

* one central regression drives the complete visible workflow with the
  reported four-ROI mixed-strategy plan;
* the Run-page lifecycle tests reach the same reviewed state through the same
  real path, but with a one-ROI recording, because none of their behavior
  depends on how many previews were generated.

Every test presses the buttons the scientist presses. Nothing installs a
prepared request, forces Run enabled, or bypasses the Run-page handler.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import QDeadlineTimer, Qt
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.main_window import (
    GUIDED_STRUCTURE_CHOICE_AUTO,
    MainWindow,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import _values


pytestmark = pytest.mark.usefixtures("no_real_modals")

CHECK_BUTTON_TEXT = "Check my setup"
FIRST_SUBSET_REFUSAL = "not supported by the first subset"
UNAVAILABLE_TEXT = "Guided Run is not available for this configuration yet."

# The mixed per-ROI plan from the report, in ROI order.
MIXED_STRATEGIES = (
    "robust_global_event_reject",
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
    "signal_only_f0",
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()
    # deleteLater only queues the deletion. Without draining the loop here,
    # every window this file builds stays alive for the whole session.
    for _ in range(5):
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _rows(count, *, rois):
    indices = np.arange(count, dtype=float)
    time_s, control, signal = _values(indices)
    header = ["Time(s)"]
    for roi in rois:
        header.extend(["%s-410" % roi, "%s-470" % roi])
    lines = [",".join(header) + "\n"]
    for index in range(count):
        cells = ["%.4f" % time_s[index]]
        for position in range(len(rois)):
            column = position % control.shape[1]
            cells.append("%.12f" % control[index, column])
            cells.append("%.12f" % signal[index, column])
        lines.append(",".join(cells) + "\n")
    return lines


def _continuous_folder(folder, *, samples=12_000, rois=("CH1", "CH2", "CH3", "CH4")):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Fluorescence.csv").write_text(
        "".join(_rows(samples, rois=rois)), encoding="utf-8", newline=""
    )
    (folder / "Events.csv").write_text(
        "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
    )
    return folder


def _intermittent_folder(folder, *, sessions=2, rois=("CH1",)):
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(sessions):
        session = folder / ("2026_03_1%d-10_00_00" % index)
        session.mkdir()
        (session / "Fluorescence.csv").write_text(
            "".join(_rows(600, rois=rois)), encoding="utf-8", newline=""
        )
    return folder


# ---------------------------------------------------------------------------
# Visible workflow
# ---------------------------------------------------------------------------


def _pump(qapp, predicate, timeout_ms=300_000):
    deadline = QDeadlineTimer(timeout_ms)
    while predicate() and not deadline.hasExpired():
        qapp.processEvents()


def _settle_preparation(window, qapp, timeout_ms=120_000):
    """Wait until preparation has finished *and* reported its result.

    Its terminal handlers are posted to the GUI thread, so the running flag
    can clear a moment before the result is installed and the page restated.
    Waiting only on the flag would read the page mid-transition.
    """
    _pump(qapp, lambda: window._guided_continuous_rwd_preparation_active(), timeout_ms)
    _pump(
        qapp,
        lambda: getattr(window, "_guided_continuous_rwd_prepare_thread", None)
        is not None,
        30_000,
    )
    _pump(
        qapp,
        lambda: "Preparing continuous analysis"
        in window._guided_backend_validation_status_label.text(),
        15_000,
    )


def _drive_to_run_page(
    window,
    qapp,
    folder,
    *,
    fmt="auto",
    structure=GUIDED_STRUCTURE_CHOICE_AUTO,
    continuous=True,
    strategies=MIXED_STRATEGIES,
):
    """Everything a scientist does between New analysis and the Run page."""
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(fmt)
    index = window._guided_acquisition_mode_combo.findData(structure)
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder.parent / "output"))
    if continuous:
        # Civil-time continuous plans require the editable recording-start
        # clock introduced by the current Guided workflow contract.
        window._guided_recording_start_clock_edit.setText("07:00")
    window._on_guided_discover_rois()
    _pump(qapp, lambda: window._guided_roi_discovery_running)
    _pump(
        qapp,
        lambda: getattr(window, "_guided_roi_discovery_thread", None) is not None,
        30_000,
    )
    for row in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(row).setCheckState(Qt.Checked)
    qapp.processEvents()

    window._on_guided_continue_to_recording_structure()
    qapp.processEvents()
    if not continuous:
        window._guided_sessions_per_hour_edit.setText("2")
        window._guided_session_duration_edit.setText("60")
        qapp.processEvents()

    window._on_guided_continue_to_correction_approach()
    if continuous:
        _pump(qapp, lambda: window._guided_continuous_recording_check_running())
        _pump(
            qapp,
            lambda: getattr(window, "_guided_continuous_rwd_check_thread", None)
            is not None,
            30_000,
        )
    qapp.processEvents()

    _generate_previews(window, qapp)
    _confirm_strategies(window, qapp, strategies)

    window._on_guided_continue_to_feature_detection()
    qapp.processEvents()
    window._guided_feature_event_apply_btn.click()
    qapp.processEvents()

    window._on_guided_continue_to_review_plan()
    qapp.processEvents()
    window._guided_review_dataset_contract_action_btn.click()
    qapp.processEvents()

    window._guided_review_go_to_run_btn.click()
    qapp.processEvents()


def _focused_run_page(window, qapp, tmp_path, *, rois=("CH1",)):
    """The smallest valid reviewed continuous plan, on the real Run page.

    Same visible path as ``_drive_to_run_page`` -- discovery, recording check,
    a real correction preview, a real confirmation, Feature Detection, Review
    Plan, Go to Run -- on a recording with one or two ROIs. The Run-page
    lifecycle behavior under test does not depend on how many previews were
    generated, so these tests do not regenerate four of them.
    """
    folder = _continuous_folder(tmp_path / "rec", rois=rois)
    _drive_to_run_page(
        window, qapp, folder, strategies=("robust_global_event_reject",)
    )
    return folder


def _generate_previews(window, qapp):
    for checkbox in window._guided_preview_method_checkboxes.values():
        checkbox.setChecked(True)
    signal_f0 = getattr(window, "_guided_preview_signal_f0_cb", None)
    if signal_f0 is not None:
        signal_f0.setChecked(True)
    for roi_index in range(window._guided_preview_roi_combo.count()):
        window._guided_preview_roi_combo.setCurrentIndex(roi_index)
        window._on_generate_guided_correction_preview()
        _pump(
            qapp,
            lambda: getattr(window, "_guided_correction_preview_running", False),
        )
        _pump(
            qapp,
            lambda: getattr(window, "_guided_correction_preview_thread", None)
            is not None,
            30_000,
        )
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()


def _confirm_strategies(window, qapp, strategies):
    rows = dict(getattr(window, "_guided_local_preview_confirmation_rows", {}))
    for position, roi in enumerate(sorted(rows)):
        row = rows[roi]
        combo = row["strategy_combo"]
        wanted = strategies[position % len(strategies)]
        chosen = None
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            if combo.currentData() == wanted:
                chosen = wanted
                break
        if chosen is None:
            # Fall back to whatever this ROI's evidence supports rather than
            # silently confirming a different strategy than intended.
            for index in range(combo.count()):
                combo.setCurrentIndex(index)
                if combo.currentData():
                    break
        qapp.processEvents()
        row["action_button"].click()
        qapp.processEvents()


def _check_button(window):
    button = window._guided_backend_validate_btn
    assert button.text() == CHECK_BUTTON_TEXT
    return button


def _setup_status(window):
    return window._guided_backend_validation_status_label.text()


def _strategies_by_roi(window):
    return {
        str(choice.get("roi")): str(choice.get("strategy"))
        for choice in (window._guided_strategy_choices or {}).values()
        if choice.get("confirmed")
    }


def _reapply_feature_detection_settings(window, qapp):
    """Re-apply the reviewed Feature Detection settings, through the button.

    Preparation deliberately refuses to repeat itself for a plan it has
    already failed on, so a second attempt needs a real setup action. Apply
    runs the central Guided invalidation hook, which clears that record, and
    leaves the accepted recording and the confirmed dataset contract intact --
    unlike changing the included ROIs, which sends the scientist back through
    the recording check.
    """
    window._guided_feature_event_apply_btn.click()
    qapp.processEvents()


def _assert_lifecycle_is_clean(window, qapp, timeout_ms=30_000):
    """No worker, no thread, no active token, nothing running.

    Fails with the exact remaining state rather than letting a test pass over
    a preparation that never finished.
    """
    deadline = QDeadlineTimer(timeout_ms)
    while not deadline.hasExpired():
        qapp.processEvents()
        if (
            window._guided_continuous_rwd_prepare_worker is None
            and window._guided_continuous_rwd_prepare_thread is None
            and window._guided_continuous_rwd_prepare_active_token is None
            and not window._guided_continuous_rwd_preparation_active()
        ):
            return
    raise AssertionError(
        "continuous preparation did not retire: worker=%r thread=%r "
        "active_token=%r running=%r status=%r"
        % (
            window._guided_continuous_rwd_prepare_worker,
            window._guided_continuous_rwd_prepare_thread,
            window._guided_continuous_rwd_prepare_active_token,
            window._guided_continuous_rwd_preparation_active(),
            window._guided_continuous_rwd_status_message,
        )
    )


# ---------------------------------------------------------------------------
# A. The central visible-path regression
# ---------------------------------------------------------------------------


def test_full_visible_continuous_path_enables_run(
    window, qapp, tmp_path, monkeypatch
):
    """The reported failure, end to end: New analysis through Go to Run with
    the four-ROI mixed-strategy plan, then the real "Check my setup" button.

    This is the wiring proof for the whole patch. It also carries the
    success-side thread boundary, so no separate full workflow is run for it.
    """
    gui_thread = threading.get_ident()
    seen = {}
    real_run = main_window_module._GuidedContinuousRwdPreparationWorker.run

    def run(self):
        seen["worker"] = threading.get_ident()
        real_run(self)

    real_ok = MainWindow._on_guided_continuous_rwd_preparation_succeeded

    def succeeded(self, token, worker, prepared_run):
        seen.setdefault("install", threading.get_ident())
        real_ok(self, token, worker, prepared_run)

    monkeypatch.setattr(
        main_window_module._GuidedContinuousRwdPreparationWorker, "run", run
    )
    monkeypatch.setattr(
        MainWindow, "_on_guided_continuous_rwd_preparation_succeeded", succeeded
    )

    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_run_page(window, qapp, folder)

    assert window._guided_workflow_stepper.currentItem().data(Qt.UserRole) == "Run"
    assert window._guided_continuous_rwd_prepare_worker is None
    assert window._guided_continuous_rwd_prepare_thread is None
    assert window._guided_run_btn.isEnabled() is False
    assert _check_button(window).isEnabled() is True
    assert window._guided_continuous_rwd_cancel_btn.isHidden() is True
    assert _setup_status(window) == "Your Guided setup has not been checked yet."
    assert window._guided_run_readiness_label.text() == (
        "Check your Guided setup before running."
    )

    before = _strategies_by_roi(window)
    assert before == {
        "CH1": "robust_global_event_reject",
        "CH2": "robust_global_event_reject",
        "CH3": "adaptive_event_gated_regression",
        "CH4": "signal_only_f0",
    }

    button = _check_button(window)
    entered = []
    real_click = MainWindow._on_guided_backend_validate_clicked
    monkeypatch.setattr(
        MainWindow,
        "_on_guided_backend_validate_clicked",
        lambda self: (entered.append(1), real_click(self))[1],
    )

    button.click()
    qapp.processEvents()

    # One click, and the intermittent refusal is never the answer.
    assert entered == [1]
    assert FIRST_SUBSET_REFUSAL not in _setup_status(window)
    assert UNAVAILABLE_TEXT not in _setup_status(window)

    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)

    # Run enables, Stop clears, and the page says so plainly.
    assert window._guided_continuous_rwd_prepared_run is not None
    assert window._guided_run_btn.isEnabled() is True
    cancel_btn = window._guided_continuous_rwd_cancel_btn
    assert cancel_btn.isEnabled() is False or cancel_btn.isHidden() is True
    assert "Ready to run" in _setup_status(window)
    assert FIRST_SUBSET_REFUSAL not in _setup_status(window)

    # The plan is exactly what was reviewed.
    draft = window._build_guided_new_analysis_draft_plan()
    assert window._guided_effective_acquisition_mode() == "continuous"
    assert draft.acquisition_mode == "continuous"
    assert draft.execution_intent.execution_mode == "both"
    assert list(draft.included_roi_ids) == ["CH1", "CH2", "CH3", "CH4"]
    assert window._guided_new_analysis_dataset_contract_snapshot.status == "applied"
    assert _strategies_by_roi(window) == before

    # And the prepared request belongs to this plan, carrying every ROI's own
    # confirmed strategy -- nothing collapsed to one global choice.
    from photometry_pipeline.guided_plan_identity import (
        compute_guided_new_analysis_draft_plan_identity,
    )

    prepared = window._guided_continuous_rwd_prepared_run
    assert prepared.plan_identity == (
        compute_guided_new_analysis_draft_plan_identity(draft)
    )
    accepted = prepared.request.accepted_draft
    assert list(accepted.included_roi_ids) == ["CH1", "CH2", "CH3", "CH4"]
    assert {
        str(choice.roi_id): str(choice.selected_strategy)
        for choice in accepted.per_roi_correction_strategy_choices
    } == before

    # Preparation ran off the GUI thread; the result was installed on it.
    assert seen["worker"] != gui_thread, "preparation ran on the GUI thread"
    assert seen["install"] == gui_thread


def test_explicit_continuous_checks_the_same_way(window, qapp, tmp_path):
    """An explicitly chosen Continuous structure takes the same authority as
    an auto-detected one."""
    folder = _continuous_folder(tmp_path / "rec", rois=("CH1",))
    _drive_to_run_page(
        window,
        qapp,
        folder,
        fmt="rwd",
        structure="continuous",
        strategies=("robust_global_event_reject",),
    )

    assert window._guided_selected_acquisition_mode() == "continuous"
    assert window._guided_continuous_rwd_prepare_worker is None
    assert window._guided_continuous_rwd_prepare_thread is None
    assert window._guided_run_execution_worker is None
    assert window._guided_run_status_follower is None
    assert window._guided_run_elapsed_timer is None
    assert window._guided_run_btn.isEnabled() is False
    assert _check_button(window).isEnabled() is True
    assert window._guided_continuous_rwd_cancel_btn.isHidden() is True
    assert _setup_status(window) == "Your Guided setup has not been checked yet."
    assert window._guided_run_readiness_label.text() == (
        "Check your Guided setup before running."
    )
    _check_button(window).click()
    qapp.processEvents()
    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)

    assert FIRST_SUBSET_REFUSAL not in _setup_status(window)
    assert window._guided_run_btn.isEnabled() is True


# ---------------------------------------------------------------------------
# B. Intermittent is untouched
# ---------------------------------------------------------------------------


def test_intermittent_still_uses_the_existing_setup_check(
    window, qapp, tmp_path, monkeypatch
):
    folder = _intermittent_folder(tmp_path / "sessions")
    _drive_to_run_page(
        window,
        qapp,
        folder,
        continuous=False,
        strategies=("robust_global_event_reject",),
    )

    workflow_calls = []
    real = MainWindow._run_guided_backend_validation_workflow

    def counted(self, context):
        workflow_calls.append(1)
        return real(self, context)

    monkeypatch.setattr(
        MainWindow, "_run_guided_backend_validation_workflow", counted
    )

    _check_button(window).click()
    qapp.processEvents()

    # The existing intermittent validator ran, and no continuous
    # preparation was started for it.
    assert workflow_calls == [1]
    assert window._guided_backend_validation_outcome is not None
    assert getattr(window, "_guided_continuous_rwd_prepare_worker", None) is None
    assert window._guided_continuous_rwd_prepared_run is None
    assert "Preparing continuous analysis" not in _setup_status(window)
    # First-subset readiness still governs an intermittent plan, and a valid
    # one still enables Run.
    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True
    assert FIRST_SUBSET_REFUSAL not in _setup_status(window)


def test_unsupported_intermittent_still_shows_the_first_subset_refusal(
    window, qapp, tmp_path
):
    """The first-subset gate is not globally suppressed."""
    from photometry_pipeline.guided_backend_validation_materialization import (
        materialize_guided_backend_validation_facts,
    )
    from tests.test_guided_backend_validation_materialization import (
        _valid_parser_contract,
        _valid_stage2c_draft,
    )
    import dataclasses

    root = tmp_path / "backend"
    root.mkdir()
    draft = _valid_stage2c_draft(root)
    draft = dataclasses.replace(draft, acquisition_mode="continuous")

    result = materialize_guided_backend_validation_facts(
        draft, parser_contract=_valid_parser_contract()
    )

    assert result.status != "materialized"
    assert any(
        FIRST_SUBSET_REFUSAL in str(getattr(issue, "message", ""))
        for issue in result.blocking_issues
    )


# ---------------------------------------------------------------------------
# C-F. Run-page preparation lifecycle, on focused setup
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_preparation(monkeypatch):
    """Hold the real preparation body inside one of its own dependencies.

    The gate sits in a builder the original ``run`` calls, so the worker is
    genuinely mid-preparation. It is released by this fixture no matter how
    the test ends, and the worker's own wait is bounded, so no thread is left
    parked inside a window that is about to be destroyed.
    """
    import photometry_pipeline.guided_continuous_rwd_target_grid as grid_module

    release = threading.Event()
    launches = []
    real_grid = grid_module.build_guided_continuous_rwd_target_grid
    real_init = main_window_module._GuidedContinuousRwdPreparationWorker.__init__

    def counting_init(self, *args, **kwargs):
        launches.append(1)
        return real_init(self, *args, **kwargs)

    def gated_grid(*args, **kwargs):
        # The same target-grid builder is also used by correction previews.
        # Only hold it after the preparation worker has been constructed, so
        # the real reviewed workflow can reach the explicit Check click.
        if launches:
            release.wait(30)
        return real_grid(*args, **kwargs)

    monkeypatch.setattr(
        main_window_module._GuidedContinuousRwdPreparationWorker,
        "__init__",
        counting_init,
    )
    monkeypatch.setattr(
        grid_module, "build_guided_continuous_rwd_target_grid", gated_grid
    )
    try:
        yield launches, release
    finally:
        release.set()


def test_preparation_failure_retry_and_thread_boundary(
    window, qapp, tmp_path, monkeypatch
):
    """A real preparation failure must terminate, say so plainly, leave the
    page usable, and still allow a second attempt.

    The failure is injected at ``build_guided_continuous_rwd_run_config``, a
    dependency the original ``run`` imports inside its own body, so the
    product's own raise/catch/emit path runs. ``run`` itself is never
    replaced: doing so lets the worker thread exit before the terminal
    callback is delivered, which is a property of the substitute rather than
    of the product.
    """
    import photometry_pipeline.guided_continuous_rwd_run_config as config_module

    gui_thread = threading.get_ident()
    seen = {}
    real_builder = config_module.build_guided_continuous_rwd_run_config
    refusing = {"active": True}

    def sometimes_refuse(*args, **kwargs):
        if refusing["active"]:
            seen["raised"] = threading.get_ident()
            raise config_module.GuidedContinuousRwdRunConfigError(
                "The accepted analysis settings for this recording are not "
                "available."
            )
        return real_builder(*args, **kwargs)

    real_failed = MainWindow._on_guided_continuous_rwd_preparation_failed

    def failed(self, token, worker, message):
        seen.setdefault("handled", threading.get_ident())
        real_failed(self, token, worker, message)

    monkeypatch.setattr(
        config_module, "build_guided_continuous_rwd_run_config", sometimes_refuse
    )
    monkeypatch.setattr(
        MainWindow, "_on_guided_continuous_rwd_preparation_failed", failed
    )

    _focused_run_page(window, qapp, tmp_path, rois=("CH1", "CH2"))
    _check_button(window).click()
    qapp.processEvents()
    _settle_preparation(window, qapp)

    # The failure terminated cleanly: token cleared, worker and thread retired.
    _assert_lifecycle_is_clean(window, qapp)
    assert window._guided_continuous_rwd_prepared_run is None
    assert window._guided_run_btn.isEnabled() is False
    cancel_btn = window._guided_continuous_rwd_cancel_btn
    assert cancel_btn.isEnabled() is False or cancel_btn.isHidden() is True

    status = _setup_status(window)
    assert "could not be prepared" in status
    assert FIRST_SUBSET_REFUSAL not in status
    assert UNAVAILABLE_TEXT not in status
    # Plain language only: no exception type or internal marker.
    assert "GuidedContinuousRwdRunConfigError" not in status
    assert "Traceback" not in status
    assert "internal:" not in status
    assert "reason:" not in status

    # It failed on the worker thread and was handled on the GUI thread.
    assert seen["raised"] != gui_thread, "preparation ran on the GUI thread"
    assert seen["handled"] == gui_thread

    # Pressing the button again still reports the failure, not the
    # intermittent refusal, and starts nothing new for the same plan.
    _check_button(window).click()
    qapp.processEvents()
    _assert_lifecycle_is_clean(window, qapp)
    assert FIRST_SUBSET_REFUSAL not in _setup_status(window)
    assert window._guided_run_btn.isEnabled() is False

    # A fresh attempt after the cause is gone prepares and enables Run.
    refusing["active"] = False
    _reapply_feature_detection_settings(window, qapp)
    _check_button(window).click()
    qapp.processEvents()
    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)

    assert window._guided_continuous_rwd_prepared_run is not None
    assert window._guided_run_btn.isEnabled() is True
    assert "Ready to run" in _setup_status(window)


def test_second_click_while_preparing_starts_one_preparation(
    window, qapp, tmp_path, gated_preparation
):
    """The page is passive; one explicit Check click starts preparation and
    the disabled button cannot start another."""
    launches, release = gated_preparation
    _focused_run_page(window, qapp, tmp_path)

    button = _check_button(window)
    button.click()
    qapp.processEvents()
    assert window._guided_continuous_rwd_preparation_active() is True
    assert len(launches) == 1
    assert button.isEnabled() is False
    button.click()
    qapp.processEvents()

    # One worker, one thread, one launch; the second click reports the
    # operation already in flight.
    assert len(launches) == 1
    assert window._guided_continuous_rwd_prepare_worker is not None
    assert window._guided_continuous_rwd_prepare_thread is not None
    assert window._guided_continuous_rwd_preparation_active() is True
    assert "Preparing continuous analysis" in _setup_status(window)
    assert "Preparing continuous analysis" not in (
        window._guided_run_readiness_label.text()
    )
    assert "Check your Guided setup before running" in (
        window._guided_run_readiness_label.text()
    )
    assert FIRST_SUBSET_REFUSAL not in _setup_status(window)

    release.set()
    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)
    assert len(launches) == 1
    assert window._guided_run_btn.isEnabled() is True
    assert "Ready to run" in _setup_status(window)
    assert "Preparing continuous analysis" not in (
        window._guided_run_readiness_label.text()
    )


def test_stop_cancels_the_preparation_started_from_the_button(
    window, qapp, tmp_path, gated_preparation
):
    _launches, release = gated_preparation
    _focused_run_page(window, qapp, tmp_path)
    _check_button(window).click()
    qapp.processEvents()
    assert window._guided_continuous_rwd_preparation_active() is True

    cancel_btn = window._guided_continuous_rwd_cancel_btn
    assert cancel_btn.isHidden() is False
    assert cancel_btn.isEnabled() is True

    cancel_btn.click()
    qapp.processEvents()
    release.set()
    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)

    assert window._guided_continuous_rwd_prepared_run is None
    assert window._guided_run_btn.isEnabled() is False
    assert cancel_btn.isEnabled() is False or cancel_btn.isHidden() is True
    status = _setup_status(window)
    assert FIRST_SUBSET_REFUSAL not in status
    assert status.strip()

    # Cancelling does not poison the plan: it can be prepared again.
    _check_button(window).click()
    qapp.processEvents()
    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)
    assert window._guided_run_btn.isEnabled() is True


def test_changing_the_plan_makes_the_prepared_setup_stale(
    window, qapp, tmp_path
):
    """A prepared setup belongs to the plan that produced it. Touching a
    reviewed input discards it and disables Run until one fresh current
    request is prepared."""
    _focused_run_page(window, qapp, tmp_path, rois=("CH1", "CH2"))
    _check_button(window).click()
    qapp.processEvents()
    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)
    assert window._guided_run_btn.isEnabled() is True
    stale = window._guided_continuous_rwd_prepared_run
    assert stale is not None

    _reapply_feature_detection_settings(window, qapp)
    window._refresh_guided_run_readiness_display()

    assert window._guided_continuous_rwd_prepared_run is None
    assert window._guided_run_btn.isEnabled() is False
    assert FIRST_SUBSET_REFUSAL not in _setup_status(window)
    assert _setup_status(window).strip()

    # The button prepares one fresh request for the plan as it stands now.
    _check_button(window).click()
    qapp.processEvents()
    _settle_preparation(window, qapp)
    _assert_lifecycle_is_clean(window, qapp)

    from photometry_pipeline.guided_plan_identity import (
        compute_guided_new_analysis_draft_plan_identity,
    )

    prepared = window._guided_continuous_rwd_prepared_run
    assert prepared is not None
    assert prepared is not stale
    assert prepared.plan_identity == compute_guided_new_analysis_draft_plan_identity(
        window._build_guided_new_analysis_draft_plan()
    )
    assert window._guided_run_btn.isEnabled() is True
    assert "Ready to run" in _setup_status(window)
