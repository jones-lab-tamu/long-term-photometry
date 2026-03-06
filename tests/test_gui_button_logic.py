import os
import sys
import unittest
from PySide6.QtWidgets import QApplication

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.main_window import MainWindow
from gui.process_runner import RunnerState

class TestButtonLogicReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_run_button_enabled_after_validation_success_real(self):
        """Test that the REAL _run_btn widget is enabled after validation success."""
        window = MainWindow()
        
        # Scenario: Validation just finished successfully
        # Force states instead of running a real process
        window._is_validate_only = True
        window._validation_passed = False
        window._ui_state = RunnerState.SUCCESS
        
        # Logic from _finalize_run_ui
        window._validation_passed = True
        window._is_validate_only = False
        window._update_button_states()
        
        self.assertTrue(window._run_btn.isEnabled(), 
                        f"Run button should be enabled. state={window._ui_state}, val_passed={window._validation_passed}")

    def test_on_config_changed_resets_validation(self):
        window = MainWindow()
        window._validation_passed = True
        window._update_button_states()
        self.assertTrue(window._run_btn.isEnabled())
        
        # Trigger config change
        window._on_config_changed()
        self.assertFalse(window._run_btn.isEnabled(), "Run button should be disabled after config change")
        self.assertFalse(window._validation_passed)

    def test_on_validate_keeps_signal_connectivity(self):
        """Verify that _on_validate does NOT break runner signal connections."""
        import unittest.mock as mock
        window = MainWindow()
        
        # Mock dependencies to avoid filesystem/QBox interaction
        window._validate_gui_inputs = mock.MagicMock(return_value=None)
        window._save_widgets_to_settings = mock.MagicMock()
        window._build_argv = mock.MagicMock(return_value=["--dummy"])
        window._current_run_dir = "/tmp/run"
        window._report_viewer = mock.MagicMock()
        window._log_view = mock.MagicMock()
        
        # Mock _update_button_states to track calls
        window._update_button_states = mock.MagicMock()
        
        # Initial check: emitting state_changed should trigger _update_button_states
        window._runner.state_changed.emit("RUNNING")
        self.assertEqual(window._update_button_states.call_count, 1)
        
        # ACT: Call _on_validate
        # This triggers the bug: self._runner = PipelineRunner()
        with mock.patch("gui.main_window.PipelineRunner.start"): # Don't actually start QProcess
            window._on_validate()
        
        window._update_button_states.reset_mock()
        
        # AFTER: emitting state_changed from the CURRENT runner should still trigger the handler
        window._runner.state_changed.emit("SUCCESS")
        
        # Terminal states like SUCCESS trigger _finalize_run_ui AND then _update_button_states at end of handler
        # So we expect 2 calls.
        self.assertEqual(window._update_button_states.call_count, 2, 
                         "Signal connectivity was broken by _on_validate")

if __name__ == "__main__":
    unittest.main()
