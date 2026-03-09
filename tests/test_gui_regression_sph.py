import unittest
from unittest.mock import MagicMock
import sys
import os
from datetime import datetime
from pathlib import Path

# Bootstrap repo root
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# QApplication instance is required for real widget operations
from PySide6.QtWidgets import QApplication, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QListWidget

class MockConfig:
    """Minimal config object to satisfy MainWindow dependencies."""
    def __init__(self):
        self.event_signal = "dff"
        self.peak_threshold_method = "mean_std"
        self.peak_threshold_k = 2.0
        self.peak_threshold_percentile = 95.0
        self.peak_threshold_abs = 0.0
        self.peak_min_distance_sec = 0.5
        self.event_auc_baseline = "zero"
        self.sessions_per_hour = 2
        self.session_duration_s = 600.0
        self.smooth_window_s = 1.0
        self.window_sec = 60.0
        self.step_sec = 10.0
        self.min_valid_windows = 5
        self.r_low = 0.5
        self.r_high = 0.95
        self.g_min = 1e-9
        self.min_samples_per_window = 100
        self.lowpass_hz = 1.0
        self.baseline_method = "uv_raw_percentile_session"
        self.baseline_percentile = 10.0
        self.f0_min_value = 1e-9

class TestGuiSphRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # QApplication instance is required for widget operations
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def test_sessions_per_hour_propagation(self):
        """
        Verify that Sessions/Hour from GUI correctly propagates to RunSpec and argv,
        without any stale 'sessions_per_hour_source' field.
        """
        from gui.main_window import MainWindow
        from gui.run_spec import RunSpec

        # 1. Create a mocked MainWindow instance that uses real widgets
        # We use MagicMock for the MainWindow itself to avoid full __init__ side effects,
        # but we attach real widgets for the fields that _build_run_spec() consumes.
        win = MagicMock(spec=MainWindow)
        win._default_cfg = MockConfig()
        win._discovery_cache = None # No discovery for this test

        # Core input/output widgets (Real widgets to ensure valid interaction)
        win._input_dir = QLineEdit()
        win._input_dir.setText("C:/data/input")
        
        win._output_dir = QLineEdit()
        win._output_dir.setText("C:/data/out")
        
        win._config_path = QLineEdit()
        win._config_path.setText("C:/configs/test.yaml")
        
        win._format_combo = QComboBox()
        win._format_combo.addItem("auto")
        win._format_combo.addItem("rwd")
        win._format_combo.setCurrentText("rwd")
        
        # SPH widget (Primary test target)
        win._sph_edit = QLineEdit()
        win._sph_edit.setText("2") 
        
        win._duration_edit = QLineEdit()
        win._duration_edit.setText("600")
        
        win._smooth_spin = QDoubleSpinBox()
        win._smooth_spin.setValue(1.0)
        
        win._mode_combo = QComboBox()
        win._mode_combo.addItem("both")
        win._mode_combo.setCurrentText("both")
        
        win._traces_only_cb = QCheckBox()
        win._traces_only_cb.setChecked(False)
        
        # Preview support (Replaces stale _preview_first_n_edit)
        win._preview_enabled_cb = QCheckBox()
        win._preview_enabled_cb.setChecked(False)
        win._preview_n_spin = QSpinBox()
        win._preview_n_spin.setValue(5)
        
        # Session/ROI Selection (Replaces stale _rep_session_idx_edit, _include_roi_edit, etc.)
        win._rep_session_combo = QComboBox()
        win._rep_session_combo.addItem("(auto)")
        
        win._roi_filter_combo = QComboBox()
        win._roi_filter_combo.addItem("Include")
        win._roi_filter_combo.addItem("Exclude")
        
        win._roi_list = QListWidget()

        # Advanced knob widgets (minimal set to avoid crashes in calls within _build_run_spec)
        win._window_sec_edit = QLineEdit("60.0")
        win._step_sec_edit = QLineEdit("10.0")
        win._min_valid_windows_spin = QSpinBox()
        win._min_valid_windows_spin.setValue(5)
        win._r_low_edit = QLineEdit("0.5")
        win._r_high_edit = QLineEdit("0.95")
        win._g_min_edit = QLineEdit("1e-9")
        win._min_samples_per_window_spin = QSpinBox()
        win._min_samples_per_window_spin.setValue(100)
        
        win._lowpass_hz_edit = QLineEdit("1.0")
        win._baseline_method_combo = QComboBox()
        win._baseline_method_combo.addItem("uv_raw_percentile_session")
        win._baseline_percentile_edit = QLineEdit("10.0")
        win._f0_min_value_edit = QLineEdit("1e-9")
        
        win._event_signal_combo = QComboBox()
        win._event_signal_combo.addItem("dff")
        win._peak_method_combo = QComboBox()
        win._peak_method_combo.addItem("mean_std")
        win._peak_k_edit = QLineEdit("2.0")
        win._peak_pct_edit = QLineEdit("95.0")
        win._peak_abs_edit = QLineEdit("0.0")
        win._peak_dist_edit = QLineEdit("0.5")
        win._event_auc_combo = QComboBox()
        win._event_auc_combo.addItem("zero")

        # 2. Attach static methods or helper functions used by _build_run_spec if needed
        # Since win is a MagicMock, we must allow it to skip actual logic for these or use the real ones.
        # However, for a "minimal" test, we just want to see if the resulting Spec is correct.
        
        # 3. Invoke the real method logic manually
        # Note: We pass our mocked 'win' as 'self'.
        spec = MainWindow._build_run_spec(win, validate_only=False)
        
        # 4. Assertions
        self.assertIsInstance(spec, RunSpec)
        
        # Propagated to RunSpec field?
        self.assertEqual(spec.sessions_per_hour, 2)
        
        # Stale field absent?
        self.assertFalse(hasattr(spec, "sessions_per_hour_source"), 
                         "RunSpec should NOT contain sessions_per_hour_source")
        
        # Propagated to argv?
        argv = spec.build_runner_argv()
        self.assertIn("--sessions-per-hour", argv)
        self.assertIn("2", argv)
        
        # Stale flag absent from argv?
        self.assertNotIn("--sessions-per-hour-source", argv)

if __name__ == "__main__":
    unittest.main()
