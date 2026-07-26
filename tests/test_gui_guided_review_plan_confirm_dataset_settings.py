"""CR1-F1-F: confirming detected dataset settings from the Review Plan button.

A scientist reached Review Plan on a real continuous recording with everything
shown as complete, clicked "Confirm detected dataset settings", and nothing
appeared to happen. "Go to Run" stayed disabled with no message of any kind.

Two defects, both proven by clicking the real button:

* the dataset settings were inferred by scanning the acquisition folder for
  CSV files, which is the intermittent layout. A continuous recording keeps
  its data in one Fluorescence.csv beside companion files, and the scan read
  whichever sorted first -- Events.csv -- which carries no channel columns, so
  confirmation refused;
* the refusal was written only into the collapsed Technical details section,
  so from the scientist's seat the click did nothing at all.

These tests click the button the scientist clicks. None of them call the
confirmation slot, the candidate builder, or the snapshot installer directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QDeadlineTimer, Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import (
    GUIDED_STRUCTURE_CHOICE_AUTO,
    GUIDED_WORKFLOW_STEPS,
    MainWindow,
)
from photometry_pipeline.guided_new_analysis_plan import (
    build_guided_new_analysis_execution_spec_preview,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import _values


pytestmark = pytest.mark.usefixtures("no_real_modals")

CONFIRM_BUTTON_TEXT = "Confirm detected dataset settings"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _rows(count):
    indices = np.arange(count, dtype=float)
    time_s, control, signal = _values(indices)
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(count):
        lines.append(
            "%.4f,%.12f,%.12f,%.12f,%.12f\n"
            % (
                time_s[index],
                control[index, 0],
                signal[index, 0],
                control[index, 1],
                signal[index, 1],
            )
        )
    return lines


def _continuous_folder(folder, *, samples=12_000):
    """One continuous recording beside the companion CSVs a real RWD
    acquisition folder also carries.

    Events.csv matters: it sorts before Fluorescence.csv, and reading it as
    the fluorescence data is what used to break confirmation.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Fluorescence.csv").write_text(
        "".join(_rows(samples)), encoding="utf-8", newline=""
    )
    (folder / "Events.csv").write_text(
        "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
    )
    (folder / "Outputs.csv").write_text(
        "Time(s),Output\n0.0,0\n", encoding="utf-8", newline=""
    )
    return folder


def _intermittent_folder(folder, *, sessions=2):
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(sessions):
        session = folder / ("2026_03_1%d-10_00_00" % index)
        session.mkdir()
        (session / "Fluorescence.csv").write_text(
            "".join(_rows(600)), encoding="utf-8", newline=""
        )
    return folder


# ---------------------------------------------------------------------------
# Visible workflow
# ---------------------------------------------------------------------------


def _pump(qapp, predicate, timeout_ms=180_000):
    deadline = QDeadlineTimer(timeout_ms)
    while predicate() and not deadline.hasExpired():
        qapp.processEvents()


def _drive_to_review_plan(
    window,
    qapp,
    folder,
    *,
    fmt="auto",
    structure=GUIDED_STRUCTURE_CHOICE_AUTO,
    continuous=True,
    allow_partial_final_window=None,
    continuous_window_sec=None,
):
    """Everything a scientist does between New analysis and Review Plan.

    ``allow_partial_final_window`` and ``continuous_window_sec`` are set on
    Recording structure, which is where a scientist meets them and before any
    correction evidence is confirmed against them.
    """
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(fmt)
    index = window._guided_acquisition_mode_combo.findData(structure)
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder.parent / "output"))
    window._on_guided_discover_rois()
    _pump(qapp, lambda: window._guided_roi_discovery_running)
    _pump(
        qapp,
        lambda: getattr(window, "_guided_roi_discovery_thread", None) is not None,
        20_000,
    )
    for row in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(row).setCheckState(Qt.Checked)
    qapp.processEvents()

    window._on_guided_continue_to_recording_structure()
    qapp.processEvents()
    if continuous_window_sec is not None:
        window._guided_continuous_window_sec_spin.setValue(
            float(continuous_window_sec)
        )
        qapp.processEvents()
    if allow_partial_final_window is not None:
        window._guided_allow_partial_final_window_cb.setChecked(
            bool(allow_partial_final_window)
        )
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
            20_000,
        )
    qapp.processEvents()

    _generate_and_confirm_corrections(window, qapp)

    window._on_guided_continue_to_feature_detection()
    qapp.processEvents()
    window._guided_feature_event_apply_btn.click()
    qapp.processEvents()

    window._on_guided_continue_to_review_plan()
    qapp.processEvents()


def _generate_and_confirm_corrections(window, qapp):
    for checkbox in window._guided_preview_method_checkboxes.values():
        checkbox.setChecked(True)
    for roi_index in range(window._guided_preview_roi_combo.count()):
        window._guided_preview_roi_combo.setCurrentIndex(roi_index)
        window._on_generate_guided_correction_preview()
        _pump(
            qapp,
            lambda: getattr(window, "_guided_correction_preview_running", False),
            300_000,
        )
        _pump(
            qapp,
            lambda: getattr(window, "_guided_correction_preview_thread", None)
            is not None,
            20_000,
        )
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()
    for row in dict(
        getattr(window, "_guided_local_preview_confirmation_rows", {})
    ).values():
        combo = row["strategy_combo"]
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            if combo.currentData():
                break
        qapp.processEvents()
        row["action_button"].click()
        qapp.processEvents()


def _confirm_button(window):
    button = window._guided_review_dataset_contract_action_btn
    assert button.text() == CONFIRM_BUTTON_TEXT
    assert button.isHidden() is False
    return button


def _next_step_text(window):
    return window._guided_review_next_step_label.text()


# ---------------------------------------------------------------------------
# The central visible-path regression
# ---------------------------------------------------------------------------


def test_confirm_button_click_confirms_settings_and_enables_go_to_run(
    window, qapp, tmp_path, monkeypatch
):
    """The reported failure: one click on the real button, on a continuous
    recording whose folder carries companion CSVs."""
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)

    assert window._guided_effective_acquisition_mode() == "continuous"
    assert window._guided_review_go_to_run_btn.isEnabled() is False
    assert "have not been confirmed yet" in (
        window._guided_review_plan_status_label.text()
    )

    button = _confirm_button(window)
    entered = []
    real = MainWindow._on_guided_apply_dataset_contract
    monkeypatch.setattr(
        MainWindow,
        "_on_guided_apply_dataset_contract",
        lambda self: (entered.append(1), real(self))[1],
    )

    button.click()
    qapp.processEvents()

    # One click reached the handler exactly once, and it did its job.
    assert entered == [1]
    snapshot = window._guided_new_analysis_dataset_contract_snapshot
    assert snapshot.status == "applied"
    assert snapshot.explicitly_applied is True
    assert not snapshot.validation_issues
    assert window._guided_review_go_to_run_btn.isEnabled() is True
    assert "This plan is ready" in _next_step_text(window)

    # The plan the scientist reviewed is unchanged by confirming it.
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.acquisition_mode == "continuous"
    assert draft.execution_intent.execution_mode == "both"
    assert list(draft.included_roi_ids) == ["ROI1", "ROI2"]
    assert draft.feature_event_profile_status == "applied"


def test_continuous_settings_come_from_the_fluorescence_file(
    window, qapp, tmp_path
):
    """Not from Events.csv, which sorts first in the acquisition folder."""
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)

    _confirm_button(window).click()
    qapp.processEvents()

    snapshot = window._guided_new_analysis_dataset_contract_snapshot
    assert snapshot.status == "applied"
    # The channel semantics can only have come from the fluorescence file;
    # Events.csv has no channel columns at all.
    values = dict(snapshot.contract_values or {})
    assert values.get("uv_suffix") == "-410"
    assert values.get("sig_suffix") == "-470"
    assert values.get("rwd_time_col") == "Time(s)"


def test_confirming_needs_no_session_settings(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)

    # A continuous recording has no sessions, and none were entered.
    assert window._guided_sessions_per_hour_edit.text().strip() == ""
    assert window._guided_session_duration_edit.text().strip() == ""
    assert list((window._discovery_cache or {}).get("sessions") or []) == []
    assert list(getattr(window, "_guided_approved_missing_sessions", [])) == []

    _confirm_button(window).click()
    qapp.processEvents()

    assert window._guided_new_analysis_dataset_contract_snapshot.status == (
        "applied"
    )
    assert window._guided_review_go_to_run_btn.isEnabled() is True


def test_explicit_continuous_confirms_the_same_way(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(
        window, qapp, folder, fmt="rwd", structure="continuous"
    )

    assert window._guided_selected_acquisition_mode() == "continuous"
    _confirm_button(window).click()
    qapp.processEvents()

    assert window._guided_new_analysis_dataset_contract_snapshot.status == (
        "applied"
    )
    assert window._guided_review_go_to_run_btn.isEnabled() is True


def test_intermittent_confirmation_is_unchanged(window, qapp, tmp_path):
    folder = _intermittent_folder(tmp_path / "sessions")
    _drive_to_review_plan(window, qapp, folder, continuous=False)

    assert window._guided_effective_acquisition_mode() == "intermittent"
    assert getattr(window, "_guided_continuous_rwd_review_binding", None) is None

    _confirm_button(window).click()
    qapp.processEvents()

    snapshot = window._guided_new_analysis_dataset_contract_snapshot
    assert snapshot.status == "applied"
    assert snapshot.acquisition_mode == "intermittent"
    assert window._guided_review_go_to_run_btn.isEnabled() is True


# ---------------------------------------------------------------------------
# Refusal is visible
# ---------------------------------------------------------------------------


def test_refusal_is_explained_where_the_scientist_is_reading(
    window, qapp, tmp_path, monkeypatch
):
    """A refusal must never look like the button did nothing."""
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)

    # Make the settings genuinely unresolvable through the real seam: the
    # accepted recording's reader settings.
    def unresolvable(*args, **kwargs):
        raise RuntimeError("simulated dataset inference failure")

    monkeypatch.setattr(
        MainWindow, "_guided_continuous_recording_reader_overrides", unresolvable
    )

    _confirm_button(window).click()
    qapp.processEvents()

    message = _next_step_text(window)
    assert message.strip()
    assert "could not be confirmed" in message
    # Plain language only: no exception text, paths, or contract vocabulary.
    assert "simulated dataset inference failure" not in message
    assert "RuntimeError" not in message
    assert "suffix" not in message.lower()
    assert str(folder) not in message
    assert "dataset contract" not in message.lower()

    # Nothing was accepted, and the plan is untouched.
    snapshot = window._guided_new_analysis_dataset_contract_snapshot
    assert getattr(snapshot, "status", None) != "applied"
    assert window._guided_review_go_to_run_btn.isEnabled() is False
    assert list(
        window._build_guided_new_analysis_draft_plan().included_roi_ids
    ) == ["ROI1", "ROI2"]

    # The button is usable again and a real retry succeeds.
    button = _confirm_button(window)
    assert button.isEnabled() is True
    monkeypatch.undo()
    button.click()
    qapp.processEvents()
    assert window._guided_new_analysis_dataset_contract_snapshot.status == (
        "applied"
    )
    assert window._guided_review_go_to_run_btn.isEnabled() is True


def test_technical_detail_stays_in_the_technical_section(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)

    monkeypatch.setattr(
        MainWindow,
        "_guided_continuous_recording_reader_overrides",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("channel mapping x")),
    )

    _confirm_button(window).click()
    qapp.processEvents()

    # Exact reason is retained for support, but only in Technical details.
    assert "channel mapping x" in (
        window._guided_dataset_contract_status_label.text()
    )
    assert "channel mapping x" not in _next_step_text(window)


# ---------------------------------------------------------------------------
# Repeat clicks, staleness, same-page refresh
# ---------------------------------------------------------------------------


def test_second_click_after_success_installs_nothing_new(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)
    button = _confirm_button(window)

    button.click()
    qapp.processEvents()
    first = window._guided_new_analysis_dataset_contract_snapshot
    assert first.status == "applied"
    revision = window._guided_backend_validation_revision

    button.click()
    qapp.processEvents()

    # Re-confirming an unchanged plan is a no-op, not a second invalidation.
    assert window._guided_new_analysis_dataset_contract_snapshot is first
    assert window._guided_backend_validation_revision == revision
    assert window._guided_review_go_to_run_btn.isEnabled() is True


def test_a_reentrant_click_cannot_start_a_second_confirmation(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)
    button = _confirm_button(window)

    entries = []
    real_handler = MainWindow._on_guided_apply_dataset_contract

    def counted(self):
        entries.append(len(entries) + 1)
        return real_handler(self)

    monkeypatch.setattr(
        MainWindow, "_on_guided_apply_dataset_contract", counted
    )

    real_candidate = MainWindow._guided_new_analysis_dataset_contract_candidate
    nested = []

    def reentrant(self):
        if not nested:
            nested.append(1)
            # Whatever the scientist does while confirmation is in flight,
            # only one may run.
            button.click()
        return real_candidate(self)

    monkeypatch.setattr(
        MainWindow, "_guided_new_analysis_dataset_contract_candidate", reentrant
    )

    button.click()
    qapp.processEvents()

    assert nested == [1], "the nested click never happened, so nothing was proven"
    assert entries == [1]
    assert window._guided_new_analysis_dataset_contract_snapshot.status == (
        "applied"
    )


def test_changing_the_plan_after_confirming_requires_confirming_again(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)
    _confirm_button(window).click()
    qapp.processEvents()
    assert window._guided_review_go_to_run_btn.isEnabled() is True

    # A reviewed choice changes after the plan was confirmed.
    window._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
    qapp.processEvents()
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_review_go_to_run_btn.isEnabled() is False
    stored = window._guided_new_analysis_dataset_contract_snapshot
    assert not (
        getattr(stored, "status", "") == "applied"
        and getattr(stored, "current_applied", False)
        and not getattr(stored, "stale_reasons", ())
    )


# ---------------------------------------------------------------------------
# Detected recording facts vs. the scientist's analysis-window policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("allow_partial", [True, False])
def test_confirming_preserves_the_partial_final_window_choice(
    window, qapp, tmp_path, allow_partial
):
    """Whether the last short window is reported is the scientist's choice.

    The correction preview restricts itself to whole windows, and that
    restriction must never be mistaken for this production choice.
    """
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(
        window, qapp, folder, allow_partial_final_window=allow_partial
    )

    assert window._guided_allow_partial_final_window_cb.isChecked() is allow_partial
    assert window._build_guided_new_analysis_draft_plan().allow_partial_final_window is (
        allow_partial
    )

    _confirm_button(window).click()
    qapp.processEvents()

    snapshot = window._guided_new_analysis_dataset_contract_snapshot
    assert snapshot.status == "applied"
    assert dict(snapshot.contract_values or {})[
        "allow_partial_final_window"
    ] is allow_partial
    assert snapshot.source_identity.allow_partial_final_window is allow_partial

    # What validation and execution preparation would consume agrees too.
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.allow_partial_final_window is allow_partial
    spec = build_guided_new_analysis_execution_spec_preview(draft)
    contract = spec.dataset_contract
    assert contract["contract_values"]["allow_partial_final_window"] is allow_partial
    assert (
        contract["rwd_normalization"]["structural_values"][
            "allow_partial_final_window"
        ]
        is allow_partial
    )

    # And the preview keeps its own whole-window rule regardless.
    assert window._guided_continuous_preview_config_overrides()[
        "allow_partial_final_window"
    ] is False


def test_confirming_preserves_the_chosen_window_length(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder, continuous_window_sec=300.0)

    _confirm_button(window).click()
    qapp.processEvents()

    snapshot = window._guided_new_analysis_dataset_contract_snapshot
    assert snapshot.status == "applied"
    values = dict(snapshot.contract_values or {})
    assert values["continuous_window_sec"] == pytest.approx(300.0)
    assert values["continuous_step_sec"] == pytest.approx(300.0)
    assert window._build_guided_new_analysis_draft_plan().continuous_window_sec == (
        pytest.approx(300.0)
    )


def test_recording_reader_facts_carry_no_analysis_policy(window, qapp, tmp_path):
    """The shared helper returns detected facts only.

    If window policy ever leaks back into it, both the preview and the
    confirmed settings would start taking their policy from the same place,
    which is exactly what this split exists to prevent.
    """
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(
        window, qapp, folder, allow_partial_final_window=True
    )

    facts = window._guided_continuous_recording_reader_overrides()

    assert set(facts) == {
        "target_fs_hz",
        "rwd_time_col",
        "uv_suffix",
        "sig_suffix",
    }
    assert facts["rwd_time_col"] == "Time(s)"
    assert facts["uv_suffix"] == "-410"
    assert facts["sig_suffix"] == "-470"

    # The preview builds on those facts and adds only its own policy.
    preview = window._guided_continuous_preview_config_overrides()
    for name, value in facts.items():
        assert preview[name] == value
    assert preview["allow_partial_final_window"] is False


def test_go_to_run_enables_without_leaving_review_plan(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _drive_to_review_plan(window, qapp, folder)
    step_before = window._guided_workflow_stepper.currentRow()

    _confirm_button(window).click()
    qapp.processEvents()

    assert window._guided_workflow_stepper.currentRow() == step_before
    assert window._guided_review_go_to_run_btn.isEnabled() is True
