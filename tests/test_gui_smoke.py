"""
GUI Smoke Test - verifies GUI modules can be imported and constructed
without crashing. Skips if PySide6 is not installed.
"""

import json
import os
import tempfile
import unittest

try:
    import PySide6
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


@unittest.skipUnless(HAS_PYSIDE6, "PySide6 not installed -- skipping GUI smoke test")
class TestGuiSmoke(unittest.TestCase):
    """Import and instantiate GUI classes without showing windows."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        # Reuse existing QApplication if running in an env that already has one
        cls._app = QApplication.instance()
        if cls._app is None:
            cls._app = QApplication([])

    def _make_temp_settings(self):
        """Create a QSettings backed by a temp INI file for isolated testing."""
        from PySide6.QtCore import QSettings
        tmp = tempfile.NamedTemporaryFile(suffix=".ini", delete=False)
        tmp.close()
        self.addCleanup(lambda: os.unlink(tmp.name) if os.path.exists(tmp.name) else None)
        return QSettings(tmp.name, QSettings.IniFormat)

    def test_import_main_window(self):
        from gui.main_window import MainWindow
        settings = self._make_temp_settings()
        w = MainWindow(settings=settings)
        self.assertIsNotNone(w)
        w.close()

    def test_import_manifest_viewer(self):
        from gui.manifest_viewer import ManifestViewer
        v = ManifestViewer()
        self.assertIsNotNone(v)
        v.close()

    def test_import_process_runner(self):
        from gui.process_runner import PipelineRunner
        r = PipelineRunner()
        self.assertIsNotNone(r)
        self.assertFalse(r.is_running())

    def test_open_results_button_exists(self):
        """MainWindow has the Open Results button attribute."""
        from gui.main_window import MainWindow
        settings = self._make_temp_settings()
        w = MainWindow(settings=settings)
        self.assertTrue(hasattr(w, "_open_results_btn"))
        self.assertIsNotNone(w._open_results_btn)
        w.close()

    def test_settings_round_trip(self):
        """QSettings save/load round-trip via injected temp settings."""
        from gui.main_window import MainWindow

        settings = self._make_temp_settings()

        # First window: set a value and save
        test_input = os.path.join(tempfile.gettempdir(), "test_input")
        w1 = MainWindow(settings=settings)
        w1._input_dir.setText(test_input)
        w1._sph_edit.setText("4")
        w1._smooth_spin.setValue(2.5)
        w1._overwrite_cb.setChecked(True)
        w1._save_widgets_to_settings()
        w1.close()

        # Second window with same settings: values should restore
        w2 = MainWindow(settings=settings)
        self.assertEqual(w2._input_dir.text(), test_input)
        self.assertEqual(w2._sph_edit.text(), "4")
        self.assertAlmostEqual(w2._smooth_spin.value(), 2.5, places=1)
        self.assertTrue(w2._overwrite_cb.isChecked())
        w2.close()

    # ------------------------------------------------------------------
    # NEW: Argv contract (token-based, requirement E)
    # ------------------------------------------------------------------
    def test_argv_contract_tokens(self):
        """GUI argv uses --out-base, --events auto, --cancel-flag auto,
        never --out or --overwrite.  Token-based (list) assertions."""
        from gui.main_window import MainWindow

        settings = self._make_temp_settings()
        w = MainWindow(settings=settings)
        w._input_dir.setText(os.path.join(tempfile.gettempdir(), "input"))
        w._output_dir.setText(os.path.join(tempfile.gettempdir(), "output"))
        w._config_path.setText(os.path.join(tempfile.gettempdir(), "cfg.yaml"))
        w._format_combo.setCurrentIndex(0)

        # Normal run argv
        argv = w._build_argv(validate_only=False)

        self.assertNotIn("--out", argv,
                         "GUI must never pass --out (legacy mode)")
        self.assertNotIn("--overwrite", argv,
                         "GUI must never pass --overwrite")
        self.assertIn("--out-base", argv)
        ob_idx = argv.index("--out-base")
        self.assertTrue(len(argv) > ob_idx + 1,
                        "--out-base must be followed by a path")
        self.assertEqual(argv[ob_idx + 1],
                         os.path.join(tempfile.gettempdir(), "output"))

        self.assertIn("--events", argv)
        ev_idx = argv.index("--events")
        self.assertEqual(argv[ev_idx + 1], "auto",
                         "--events must be followed by 'auto'")

        self.assertIn("--cancel-flag", argv)
        cf_idx = argv.index("--cancel-flag")
        self.assertEqual(argv[cf_idx + 1], "auto",
                         "--cancel-flag must be followed by 'auto'")

        self.assertNotIn("--validate-only", argv,
                         "validate_only=False must not add --validate-only")

        # Validate-only argv
        argv_v = w._build_argv(validate_only=True)
        self.assertIn("--validate-only", argv_v,
                       "validate_only=True must add --validate-only")
        # Still has the required auto flags
        self.assertIn("--events", argv_v)
        self.assertIn("--cancel-flag", argv_v)

        w.close()

    # ------------------------------------------------------------------
    # NEW: Events follower robustness (requirement B)
    # ------------------------------------------------------------------
    def test_events_follower_robustness(self):
        """EventsFollower handles valid, blank, malformed, and partial lines."""
        from gui.events_follower import EventsFollower

        tmpdir = tempfile.mkdtemp()
        events_path = os.path.join(tmpdir, "events.ndjson")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))

        # Phase 1: write valid + blank + malformed lines
        with open(events_path, "wb") as fh:
            fh.write(b'{"schema_version":1,"stage":"engine","type":"start","message":"hello"}\n')
            fh.write(b'\n')  # blank line
            fh.write(b'NOT VALID JSON\n')
            # Partial line (no trailing newline)
            fh.write(b'{"schema_version":1,"stage":"engine","type":"do')

        follower = EventsFollower(events_path, poll_ms=100)

        received_events = []
        parse_errors = []
        follower.event_received.connect(lambda e: received_events.append(e))
        follower.parse_error.connect(lambda m: parse_errors.append(m))

        # Poll once
        follower._poll()

        # Should have received exactly 1 valid event
        self.assertEqual(len(received_events), 1,
                         f"Expected 1 event, got {len(received_events)}")
        self.assertEqual(received_events[0]["stage"], "engine")
        self.assertEqual(received_events[0]["type"], "start")

        # Should have 0 parse errors, but 1 dropped line internally
        self.assertEqual(len(parse_errors), 0,
                         f"Expected 0 parse errors (contract B), got {len(parse_errors)}")
        self.assertEqual(follower._dropped_lines_count, 1,
                         "Should have recorded 1 dropped line internally")

        # Partial line must NOT have been emitted
        # (remainder should hold it)
        self.assertEqual(follower._remainder, b'{"schema_version":1,"stage":"engine","type":"do')

        # Phase 2: append the rest of the partial line
        with open(events_path, "ab") as fh:
            fh.write(b'ne","message":"finished"}\n')

        received_events.clear()
        parse_errors.clear()

        # Poll again
        follower._poll()

        # Now the completed line should be emitted
        self.assertEqual(len(received_events), 1,
                         "Completed partial line should produce 1 event")
        self.assertEqual(received_events[0]["type"], "done")
        self.assertEqual(received_events[0]["message"], "finished")
        self.assertEqual(len(parse_errors), 0,
                         "No parse errors expected in phase 2")

        # Remainder should be empty
        self.assertEqual(follower._remainder, b"")

        # Phase 3: drain-safe remainder (requirement A)
        # Write an incomplete JSON fragment as the last bytes
        with open(events_path, "ab") as fh:
            fh.write(b'{"schema_version":1,"stage":"final","type":"incompl')

        received_events.clear()
        parse_errors.clear()

        # Poll to pick up the partial fragment into remainder
        follower._poll()
        self.assertEqual(len(received_events), 0,
                         "Partial fragment must not emit an event")
        self.assertEqual(len(parse_errors), 0,
                         "Partial fragment must not emit parse_error")
        self.assertNotEqual(follower._remainder, b"")

        # Activate the timer so is_active reflects stop() calls.
        # start() resets _offset and _remainder, so save and restore them.
        saved_offset = follower._offset
        saved_remainder = follower._remainder
        follower.start()
        follower._offset = saved_offset
        follower._remainder = saved_remainder
        self.assertTrue(follower.is_active,
                        "Follower must be active before drain")

        # Enter drain mode and poll twice with no new data
        follower.begin_drain()
        follower._poll()  # idle poll 1
        follower._poll()  # idle poll 2 -> triggers drain flush + stop

        # The drain flush must NOT produce parse_error but will emit the stream_warning summary
        self.assertEqual(len(parse_errors), 0,
                         "Drain must not emit parse_error for partial JSON")
        self.assertEqual(len(received_events), 1,
                         "Drain should emit exactly 1 event (the stream_warning summary)")
        self.assertEqual(received_events[0]["type"], "stream_warning")
        self.assertFalse(follower.is_active,
                         "Follower must stop after drain condition")

    # ------------------------------------------------------------------
    # NEW: Events follower handles missing file
    # ------------------------------------------------------------------
    def test_events_follower_missing_file(self):
        """EventsFollower does not crash when file does not exist yet."""
        from gui.events_follower import EventsFollower

        follower = EventsFollower("/nonexistent/path/events.ndjson", poll_ms=100)
        received = []
        follower.event_received.connect(lambda e: received.append(e))

        # Should not raise
        follower._poll()
        follower._poll()
        self.assertEqual(len(received), 0)

    # ------------------------------------------------------------------
    # NEW: Runner state enum exists (requirement)
    # ------------------------------------------------------------------
    def test_runner_state_enum_exists(self):
        """RunnerState enum has all required values."""
        from gui.process_runner import RunnerState

        for name in ("IDLE", "VALIDATING", "RUNNING",
                     "SUCCESS", "FAILED", "FAIL_CLOSED", "CANCELLED"):
            self.assertTrue(hasattr(RunnerState, name),
                            f"RunnerState missing {name}")
            self.assertEqual(getattr(RunnerState, name).value, name)

    # ------------------------------------------------------------------
    # NEW: Open Run Folder button exists
    # ------------------------------------------------------------------
    def test_open_folder_button_exists(self):
        """MainWindow has the Open Run Folder button attribute."""
        from gui.main_window import MainWindow
        settings = self._make_temp_settings()
        w = MainWindow(settings=settings)
        self.assertTrue(hasattr(w, "_open_folder_btn"))
        self.assertIsNotNone(w._open_folder_btn)
        w.close()

    # ------------------------------------------------------------------
    # NEW: Status label exists
    # ------------------------------------------------------------------
    def test_status_label_exists(self):
        """MainWindow has a status label initialized to IDLE."""
        from gui.main_window import MainWindow
        settings = self._make_temp_settings()
        w = MainWindow(settings=settings)
        self.assertTrue(hasattr(w, "_status_label"))
        self.assertIn("IDLE", w._status_label.text())
        w.close()

    # ------------------------------------------------------------------
    # NEW: Validation gating resets on config change
    # ------------------------------------------------------------------
    def test_validation_gating_resets_on_config_change(self):
        """Changing config widgets resets _validation_passed to False."""
        from gui.main_window import MainWindow
        settings = self._make_temp_settings()
        w = MainWindow(settings=settings)

        # Force validation_passed to True
        w._validation_passed = True
        w._update_button_states()
        self.assertTrue(w._run_btn.isEnabled(),
                        "Run should be enabled when validated")

        # Change a config widget
        w._input_dir.setText("/some/new/path")

        # Should have reset
        self.assertFalse(w._validation_passed,
                         "Config change must reset _validation_passed")
        self.assertFalse(w._run_btn.isEnabled(),
                         "Run should be disabled after config change")
        w.close()

    # ------------------------------------------------------------------
    # NEW: Status label composition (requirement C)
    # ------------------------------------------------------------------
    def test_status_label_composition(self):
        """Status label shows BOTH state and last event fields together."""
        from gui.main_window import MainWindow
        settings = self._make_temp_settings()
        w = MainWindow(settings=settings)

        # Simulate state change to RUNNING
        w._on_state_changed("RUNNING")
        label_text = w._status_label.text()
        self.assertIn("State: RUNNING", label_text)

        # Simulate an event arriving (must have schema_version=1)
        w._on_event({"schema_version": 1, "stage": "engine", "type": "start", "message": "hi"})
        label_text = w._status_label.text()
        self.assertIn("State: RUNNING", label_text,
                      "State must survive event update")
        self.assertIn("Stage: engine", label_text) 
        self.assertIn("Type: start", label_text)
        self.assertIn("hi", label_text)

        # Simulate state change to SUCCESS - state updates, event fields remain
        w._on_state_changed("SUCCESS")
        label_text = w._status_label.text()
        self.assertIn("State: SUCCESS", label_text)
        self.assertIn("Stage: engine", label_text,
                      "Last event fields should persist across state changes")
        self.assertIn("Type: start", label_text)

        w.close()


if __name__ == "__main__":
    unittest.main()
