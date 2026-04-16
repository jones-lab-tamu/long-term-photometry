import os
import re
import sys

import pytest
from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QSizePolicy, QToolButton, QLabel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.main_window import MainWindow
from gui.process_runner import RunnerState


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def test_shell_stylesheet_avoids_hardcoded_hex_colors(window):
    sheet = window.styleSheet()
    assert "palette(mid)" in sheet
    assert re.search(r"#[0-9a-fA-F]{3,8}", sheet) is None


def test_left_control_stack_uses_expanding_size_policies(window):
    assert window._controls_stack.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert window._config_panel.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert window._complete_state_panel.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding


def test_setup_fields_are_not_hard_clamped_to_200px(window):
    assert window._sph_edit.maximumWidth() > 10_000
    assert window._duration_edit.maximumWidth() > 10_000
    assert window._smooth_spin.maximumWidth() > 10_000
    assert window._fixed_daily_anchor_time_edit.maximumWidth() > 10_000


def test_run_action_buttons_contained_in_fixed_sidebar(window, qapp):
    window.show()
    window.resize(1280, 900)
    qapp.processEvents()

    left_width = window._left_pane.width()
    assert left_width >= 420

    for btn in (
        window._validate_btn,
        window._run_btn,
        window._cancel_btn,
        window._open_results_btn,
        window._open_folder_btn,
    ):
        assert btn.width() >= btn.minimumSizeHint().width()
        right_x = btn.mapTo(window._left_pane, btn.rect().topRight()).x()
        assert right_x <= left_width + 1


def test_dynamic_fit_note_wraps_inside_sidebar(window, qapp):
    window.show()
    window.resize(1240, 900)
    window._advanced_disclosure_btn.setChecked(True)
    qapp.processEvents()

    note = window._dynamic_fit_mode_note
    assert note.wordWrap()
    assert note.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert note.width() <= window._left_pane.width()
    assert note.property("statusSeverity") is None


def test_form_row_help_icons_are_compact_in_sidebar(window, qapp):
    window.show()
    window.resize(1240, 900)
    window._advanced_disclosure_btn.setChecked(True)
    qapp.processEvents()

    icons = window.findChildren(QToolButton, "formRowHelpIcon")
    assert icons, "Expected visible help icons for primary form rows."
    for icon in icons:
        assert icon.width() <= 16
        assert icon.height() <= 16


def test_help_icon_rollout_emits_no_formlayout_cell_occupied_warning(qapp):
    messages: list[str] = []

    def _handler(_msg_type, _context, message):
        messages.append(message)

    previous = qInstallMessageHandler(_handler)
    try:
        w = MainWindow()
        qapp.processEvents()
        w.close()
        w.deleteLater()
        qapp.processEvents()
    finally:
        qInstallMessageHandler(previous)

    offenders = [m for m in messages if "QFormLayoutPrivate::setItem: Cell" in m]
    assert offenders == []


def test_help_label_rows_are_not_duplicated_after_icon_attachment(window):
    def _count_labels(root, text: str) -> int:
        return len([lbl for lbl in root.findChildren(QLabel) if lbl.text() == text])

    assert _count_labels(window._run_config_group, "Input Directory:") == 1
    assert _count_labels(window._plotting_group, "Timeline Anchor:") == 1
    assert _count_labels(window._advanced_group, "Dynamic Fit Mode:") == 1
    assert _count_labels(window._tuning_controls_container, "Peak Threshold Method:") == 1
    assert _count_labels(window._correction_tuning_controls_container, "Dynamic Fit Mode:") == 1


def test_status_labels_use_centralized_severity_property(window):
    window._set_tuning_workspace_unavailable("Not ready yet")
    assert window._tuning_collapsed_status_label.property("statusSeverity") == "warn"
    assert window._tuning_availability_label.property("statusSeverity") == "warn"

    window._set_tuning_workspace_available("Ready")
    assert window._tuning_collapsed_status_label.property("statusSeverity") == "ready"
    assert window._tuning_availability_label.property("statusSeverity") == "ready"

    window._ui_state = RunnerState.IDLE
    window._validation_passed = False
    window._update_run_reason_label()
    assert window._run_reason_label.property("statusSeverity") in {"warn", "info", "error", "ready"}

    window._use_custom_config_cb.setChecked(False)
    window._update_config_source_ui()
    assert window._active_config_source_label.property("statusSeverity") == "ready"

    window._use_custom_config_cb.setChecked(True)
    window._update_config_source_ui()
    assert window._active_config_source_label.property("statusSeverity") == "info"
