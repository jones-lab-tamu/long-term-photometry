import os
import sys
import tempfile

import pytest
from PySide6.QtCore import Qt
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


def _set_minimally_valid_paths(w: MainWindow):
    w._input_dir.setText("tests/out_manual_complex_5roi_5day_2sph_shared")
    w._config_path.setText("tests/qc_universal_config.yaml")
    w._output_dir.setText(tempfile.mkdtemp(prefix="gui_usability_out_"))


def test_effective_run_summary_updates(window):
    text0 = window._effective_summary_label.text()
    assert "Mode: both" in text0
    assert "Analysis: Full analysis" in text0
    assert "Preview: off" in text0
    assert "Plotting Mode: Standard" in text0

    window._traces_only_cb.setChecked(True)
    window._preview_enabled_cb.setChecked(True)
    window._preview_n_spin.setValue(7)
    window._plotting_mode_combo.setCurrentText("Full")
    text1 = window._effective_summary_label.text()
    assert "Analysis: Traces-only" in text1
    assert "Preview: first N = 7" in text1
    assert "Plotting Mode: Full" in text1

    discovered = {
        "n_total_discovered": 3,
        "n_preview": 3,
        "resolved_format": "rwd",
        "sessions": [
            {"session_id": "S0", "included_in_preview": True},
            {"session_id": "S1", "included_in_preview": True},
            {"session_id": "S2", "included_in_preview": False},
        ],
        "rois": [{"roi_id": "Region0"}, {"roi_id": "Region1"}, {"roi_id": "Region2"}],
    }
    window._discovery_cache = discovered
    window._populate_discovery_ui(discovered)
    window._roi_list.item(0).setCheckState(Qt.Unchecked)  # Include subset
    window._rep_session_combo.setCurrentIndex(2)  # discovery index 1 (S1)
    text2 = window._effective_summary_label.text()
    assert "ROI Filter: Include subset (2/3)" in text2
    assert "Representative Session: Session index 1 (S1)" in text2


def test_run_disabled_reason_text_states(window):
    _set_minimally_valid_paths(window)

    window._validation_passed = False
    window._update_button_states()
    assert "Validation required" in window._run_reason_label.text()

    window._validation_passed = True
    window._update_button_states()
    assert "Ready to run" in window._run_reason_label.text()

    window._sph_edit.setText("0")
    window._validation_passed = True
    window._update_button_states()
    assert "Invalid setting combination:" in window._run_reason_label.text()


def test_fail_closed_terminal_message_is_actionable(window, tmp_path):
    window._current_run_dir = str(tmp_path)
    window._is_validate_only = False
    window._did_finalize_run_ui = False
    window._runner._state = RunnerState.FAIL_CLOSED
    window._runner.fail_closed_code = "MISSING_FILE"
    window._runner.fail_closed_detail = "status.json was never created or is unreadable."
    window._runner.fail_closed_remediation = "Verify run directory is writeable and disk is not full."
    window._runner.final_status_code = "MISSING_FILE"

    window._finalize_run_ui()
    log = window._log_view.toPlainText()
    assert "Run failed (FAIL_CLOSED): MISSING_FILE" in log
    assert "Reason: status.json was never created or is unreadable." in log
    assert "Next step: Verify run directory is writeable and disk is not full." in log
    assert window._last_status_msg.startswith("FAIL_CLOSED:")
