import os
import sys

import pytest
from PySide6.QtWidgets import QApplication

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


def test_mode_sensitive_controls(window):
    window._mode_combo.setCurrentText("tonic")
    window._update_button_states()
    assert not window._sig_iso_render_mode_combo.isEnabled()
    assert not window._dff_render_mode_combo.isEnabled()
    assert not window._stacked_render_mode_combo.isEnabled()
    assert not window._adv_ev_group.isEnabled()
    assert "disabled in tonic mode" in window._mode_context_label.text()

    window._mode_combo.setCurrentText("phasic")
    window._update_button_states()
    assert window._sig_iso_render_mode_combo.isEnabled()
    assert window._dff_render_mode_combo.isEnabled()
    assert window._stacked_render_mode_combo.isEnabled()
    assert window._adv_ev_group.isEnabled()


def test_discovery_dependent_state_signaling(window):
    window._discovery_cache = None
    window._roi_list.clear()
    window._update_button_states()
    assert not window._roi_list.isEnabled()
    assert not window._rep_session_combo.isEnabled()
    assert "Select ROIs..." in window._discovery_controls_hint.text()

    discovered = {
        "n_total_discovered": 3,
        "n_preview": 3,
        "resolved_format": "rwd",
        "sessions": [
            {"session_id": "S0", "included_in_preview": True},
            {"session_id": "S1", "included_in_preview": True},
            {"session_id": "S2", "included_in_preview": True},
        ],
        "rois": [{"roi_id": "Region0"}, {"roi_id": "Region1"}],
    }
    window._discovery_cache = discovered
    window._populate_discovery_ui(discovered)
    assert window._roi_list.isEnabled()
    assert window._rep_session_combo.isEnabled()
    assert "ROI choices loaded" in window._discovery_controls_hint.text()

    window._preview_enabled_cb.setChecked(True)
    window._preview_n_spin.setValue(1)
    window._rep_session_combo.setCurrentIndex(2)  # representative index=1
    window._update_button_states()
    assert "out of range for Preview first N=1" in window._rep_preview_hint.text()


def test_key_artifact_access_buttons(window, tmp_path, monkeypatch):
    run_dir = tmp_path
    for name in [
        "command_invoked.txt",
        "gui_run_spec.json",
        "config_effective.yaml",
        "MANIFEST.json",
        "run_report.json",
    ]:
        (run_dir / name).write_text("{}", encoding="utf-8")

    window._current_run_dir = str(run_dir)
    window._ui_state = RunnerState.SUCCESS
    window._runner.final_status_code = "success"
    window._update_button_states()

    assert window._open_cmd_file_btn.isEnabled()
    assert window._open_spec_file_btn.isEnabled()
    assert window._open_cfg_file_btn.isEnabled()
    assert window._open_manifest_file_btn.isEnabled()
    assert window._open_report_file_btn.isEnabled()

    opened = []

    def _fake_open(path):
        opened.append(path)

    monkeypatch.setattr("gui.main_window._open_file", _fake_open)
    window._on_open_key_artifact("command_invoked.txt")
    assert opened and opened[0].endswith("command_invoked.txt")
