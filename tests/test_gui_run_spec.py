"""
Tests for gui.run_spec: determinism, argv, persistence contract, audit boundary.

No PySide6 required (RunSpec is pure Python).
"""

import json
import os
import shutil
import tempfile
import unittest

import yaml

from gui.run_spec import RunSpec


class TestRunSpec(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Create a valid base config YAML that passes Config.from_yaml
        self.base_config = {
            "target_fs_hz": 40.0,
            "chunk_duration_sec": 600.0,
            "lowpass_hz": 1.0,
            "filter_order": 3,
            "window_sec": 60.0,
            "step_sec": 10.0,
            "min_valid_windows": 5,
            "baseline_method": "uv_raw_percentile_session",
            "baseline_percentile": 10.0,
            "f0_min_value": 1e-9,
            "peak_threshold_method": "mean_std",
            "peak_threshold_k": 2.0,
            "peak_threshold_percentile": 95.0,
            "peak_threshold_abs": 0.0,
            "peak_min_distance_sec": 0.5,
            "allow_partial_final_chunk": False,
            "rwd_time_col": "Time(s)",
            "uv_suffix": "-410",
            "sig_suffix": "-470",
        }
        self.config_path = os.path.join(self.tmp_dir, "base_config.yaml")
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.base_config, f, sort_keys=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ----------------------------------------------------------------
    # Determinism (yaml.safe_dump)
    # ----------------------------------------------------------------
    def test_yaml_determinism_safe_dump(self):
        """Same base + overrides yields byte-identical config_effective.yaml."""
        overrides = {"event_signal": "delta_f", "peak_threshold_k": 3.0}

        run_dir_1 = os.path.join(self.tmp_dir, "run_1")
        run_dir_2 = os.path.join(self.tmp_dir, "run_2")

        spec1 = RunSpec(
            config_source_path=self.config_path,
            config_overrides=overrides.copy(),
        )
        spec2 = RunSpec(
            config_source_path=self.config_path,
            config_overrides=overrides.copy(),
        )

        path1 = spec1.generate_derived_config(run_dir_1)
        path2 = spec2.generate_derived_config(run_dir_2)

        with open(path1, "r") as f:
            content1 = f.read()
        with open(path2, "r") as f:
            content2 = f.read()

        self.assertEqual(content1, content2,
                         "Derived config is not byte-identical across runs")

        loaded = yaml.safe_load(content1)
        self.assertEqual(loaded["event_signal"], "delta_f")
        self.assertEqual(loaded["peak_threshold_k"], 3.0)
        self.assertEqual(loaded["rwd_time_col"], "Time(s)")

    # ----------------------------------------------------------------
    # argv uses explicit --out <run_dir>
    # ----------------------------------------------------------------
    def test_argv_uses_explicit_out_dir(self):
        """argv contains --out <run_dir>, NOT --out-base or --run-id."""
        run_dir = os.path.join(self.tmp_dir, "run_test")

        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            format="rwd",
            config_source_path=self.config_path,
        )

        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()

        self.assertIn("--out", argv)
        out_idx = argv.index("--out")
        self.assertEqual(argv[out_idx + 1], run_dir)

        self.assertNotIn("--out-base", argv)
        self.assertNotIn("--run-id", argv)

        config_idx = argv.index("--config")
        config_val = argv[config_idx + 1]
        self.assertTrue(config_val.endswith("config_effective.yaml"))
        self.assertTrue(os.path.isfile(config_val))

    # ----------------------------------------------------------------
    # command_invoked.txt written
    # ----------------------------------------------------------------
    def test_command_invoked_written(self):
        """write_command_invoked creates file with runner script and --config."""
        run_dir = os.path.join(self.tmp_dir, "run_cmd")

        spec = RunSpec(
            input_dir="/data",
            run_dir=run_dir,
            config_source_path=self.config_path,
        )
        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()

        cmd_path = spec.write_command_invoked(run_dir, argv)

        self.assertTrue(os.path.isfile(cmd_path))
        with open(cmd_path, "r") as f:
            content = f.read()

        self.assertIn("run_full_pipeline_deliverables.py", content)
        self.assertIn("--config", content)
        self.assertIn("config_effective.yaml", content)

    # ----------------------------------------------------------------
    # Preview uses overrides (Config-schema-safe keys)
    # ----------------------------------------------------------------
    def test_preview_uses_overrides(self):
        """get_derived_config_preview includes config_overrides."""
        spec = RunSpec(
            config_source_path=self.config_path,
            config_overrides={"event_signal": "delta_f"},
        )
        preview = spec.get_derived_config_preview()
        loaded = yaml.safe_load(preview)

        self.assertEqual(loaded["event_signal"], "delta_f")
        self.assertEqual(loaded["rwd_time_col"], "Time(s)")

    # ----------------------------------------------------------------
    # B: GUI does not write or emit events (strict substring bans)
    # ----------------------------------------------------------------
    def test_gui_does_not_write_or_emit_events(self):
        """GUI modules must not write audit events or reference EventEmitter.

        run_spec.py: executable code must not contain 'events.ndjson',
            '.emit(', or 'EventEmitter' (docstring documentation is OK).
        main_window.py: must not contain '.emit(' or 'EventEmitter',
            and must not open events.ndjson for write/append.
            (Read-only reference to events.ndjson via EventsFollower is OK.)
        """
        import ast
        import re
        import gui.run_spec as rs_module
        gui_dir = os.path.dirname(rs_module.__file__)

        def _strip_docstring_literals(source: str) -> str:
            """Remove docstring literals from source using AST positions.

            Identifies docstring nodes (first Expr(Constant(str)) in
            Module, ClassDef, FunctionDef, AsyncFunctionDef bodies) and
            removes their exact source segments via ast.get_source_segment.
            """
            tree = ast.parse(source)
            segments_to_remove = []

            def _maybe_docstring(body):
                """If body[0] is a docstring Expr node, return it."""
                if (body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    return body[0]
                return None

            # Module docstring
            doc_node = _maybe_docstring(tree.body)
            if doc_node:
                seg = ast.get_source_segment(source, doc_node)
                if seg:
                    segments_to_remove.append(seg)

            # Class and function docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    doc_node = _maybe_docstring(node.body)
                    if doc_node:
                        seg = ast.get_source_segment(source, doc_node)
                        if seg:
                            segments_to_remove.append(seg)

            # Remove segments (longest first to avoid partial matches)
            result = source
            for seg in sorted(segments_to_remove, key=len, reverse=True):
                result = result.replace(seg, "", 1)

            return result

        def _strip_comments(source: str) -> str:
            """Remove comment tokens from source using tokenize."""
            import io
            import tokenize as tok_mod
            tokens = []
            for tok in tok_mod.generate_tokens(io.StringIO(source).readline):
                if tok.type != tok_mod.COMMENT:
                    tokens.append(tok)
            return tok_mod.untokenize(tokens)

        # --- run_spec.py: ban in executable code (no docstrings, no comments) ---
        rs_path = os.path.join(gui_dir, "run_spec.py")
        with open(rs_path, "r") as f:
            rs_source = f.read()

        rs_code_only = _strip_comments(_strip_docstring_literals(rs_source))

        for banned in ("events.ndjson", ".emit(", "EventEmitter"):
            self.assertNotIn(
                banned, rs_code_only,
                f"run_spec.py executable code contains banned literal {banned!r}"
            )

        # --- main_window.py: ban .emit( and EventEmitter ---
        mw_path = os.path.join(gui_dir, "main_window.py")
        with open(mw_path, "r") as f:
            mw_source = f.read()
        for banned in (".emit(", "EventEmitter"):
            self.assertNotIn(
                banned, mw_source,
                f"main_window.py contains banned literal {banned!r}"
            )

        # --- main_window.py: no WRITE/APPEND to events.ndjson ---
        write_pattern = re.compile(
            r"""open\s*\([^)]*events\.ndjson[^)]*,\s*["'][wa]""",
            re.IGNORECASE,
        )
        match = write_pattern.search(mw_source)
        self.assertIsNone(
            match,
            f"main_window.py opens events.ndjson for write/append: "
            f"{match.group(0) if match else ''}"
        )

    # ----------------------------------------------------------------
    # gui_run_spec.json structure
    # ----------------------------------------------------------------
    def test_gui_run_spec_json_structure(self):
        """gui_run_spec.json has correct fields, no pipeline-resolved data."""
        run_dir = os.path.join(self.tmp_dir, "run_json")

        spec = RunSpec(
            input_dir="/data",
            run_dir=run_dir,
            format="rwd",
            config_source_path=self.config_path,
            config_overrides={"lowpass_hz": 5.0},
            gui_version="1.0.0",
            timestamp_local="2026-03-02T22:00:00",
            user_set_fields=["format"],
        )

        path = spec.write_gui_run_spec(run_dir)
        self.assertTrue(os.path.isfile(path))

        with open(path, "r") as f:
            data = json.load(f)

        self.assertEqual(data["input_dir"], "/data")
        self.assertEqual(data["run_dir"], run_dir)
        self.assertEqual(data["format"], "rwd")
        self.assertEqual(data["gui_version"], "1.0.0")
        self.assertEqual(data["config_overrides"], {"lowpass_hz": 5.0})
        self.assertIn("format", data["user_set_fields"])

        # Must NOT contain runner/pipeline-resolved fields
        for forbidden in ("n_total_discovered", "n_sessions_resolved",
                          "representative_session_id"):
            self.assertNotIn(forbidden, data)

    # ----------------------------------------------------------------
    # User set fields tracking
    # ----------------------------------------------------------------
    def test_user_set_fields_tracking(self):
        """user_set_fields correctly reflects explicit settings."""
        spec = RunSpec(
            input_dir="/data",
            run_dir="/out/run_001",
            format="rwd",
            sessions_per_hour=6,
            config_source_path=self.config_path,
            user_set_fields=["format", "sessions_per_hour"],
        )

        d = spec.to_dict()
        self.assertIn("format", d["user_set_fields"])
        self.assertIn("sessions_per_hour", d["user_set_fields"])
        self.assertNotIn("smooth_window_s", d["user_set_fields"])
        self.assertNotIn("session_duration_s", d["user_set_fields"])

    # ----------------------------------------------------------------
    # Validate effective config
    # ----------------------------------------------------------------
    def test_validate_effective_config_valid(self):
        """Valid derived config passes validation without error."""
        run_dir = os.path.join(self.tmp_dir, "run_valid")
        spec = RunSpec(config_source_path=self.config_path)
        config_path = spec.generate_derived_config(run_dir)
        RunSpec.validate_effective_config(config_path)

    def test_validate_effective_config_invalid(self):
        """Invalid derived config raises ValueError via Config.from_yaml."""
        run_dir = os.path.join(self.tmp_dir, "run_invalid")
        # Write a base config with preview_first_n=0 (invalid per Config
        # validation: must be > 0 or None). This goes in the base YAML,
        # not via config_overrides (which would be rejected by the registry).
        bad_config = self.base_config.copy()
        bad_config["preview_first_n"] = 0
        bad_config_path = os.path.join(self.tmp_dir, "bad_config.yaml")
        with open(bad_config_path, "w") as f:
            yaml.safe_dump(bad_config, f, sort_keys=True)

        spec = RunSpec(config_source_path=bad_config_path)
        config_path = spec.generate_derived_config(run_dir)
        with self.assertRaises(ValueError) as cm:
            RunSpec.validate_effective_config(config_path)
        self.assertIn("invalid", str(cm.exception).lower())

    # ----------------------------------------------------------------
    # C1: argv includes only supported flags (fields that exist today)
    # ----------------------------------------------------------------
    def test_argv_includes_only_supported_flags(self):
        """argv includes core + optional flags for fields that exist."""
        run_dir = os.path.join(self.tmp_dir, "run_flags")

        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            format="rwd",
            config_source_path=self.config_path,
            sessions_per_hour=6,
            session_duration_s=600.0,
            smooth_window_s=2.0,
            validate_only=True,
        )

        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()

        # Core required flags
        self.assertIn("--input", argv)
        self.assertIn("--out", argv)
        self.assertIn("--config", argv)
        self.assertIn("--format", argv)

        # events and cancel-flag always auto
        idx_ev = argv.index("--events")
        self.assertEqual(argv[idx_ev + 1], "auto")
        idx_cf = argv.index("--cancel-flag")
        self.assertEqual(argv[idx_cf + 1], "auto")

        # Optional flags when set
        self.assertIn("--sessions-per-hour", argv)
        self.assertEqual(argv[argv.index("--sessions-per-hour") + 1], "6")
        self.assertIn("--session-duration-s", argv)
        self.assertEqual(argv[argv.index("--session-duration-s") + 1], "600.0")
        self.assertIn("--smooth-window-s", argv)
        self.assertEqual(argv[argv.index("--smooth-window-s") + 1], "2.0")
        self.assertIn("--validate-only", argv)

    # ----------------------------------------------------------------
    # C2: argv omits flags for nonexistent knobs
    # ----------------------------------------------------------------
    def test_argv_omits_flags_for_nonexistent_knobs(self):
        """argv does NOT contain flags for widgets that don't exist in GUI."""
        run_dir = os.path.join(self.tmp_dir, "run_no_phantom")

        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            config_source_path=self.config_path,
        )

        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()

        banned_flags = [
            "--traces-only", "--event-signal", "--preview-first-n",
            "--representative-session-index", "--include-rois",
            "--exclude-rois", "--out-base", "--run-id",
        ]
        for flag in banned_flags:
            self.assertNotIn(flag, argv,
                             f"argv contains banned flag: {flag}")

    # ----------------------------------------------------------------
    # D: GUI writes exactly three files
    # ----------------------------------------------------------------
    def test_gui_writes_only_three_files(self):
        """GUI writes exactly config_effective.yaml, gui_run_spec.json,
        command_invoked.txt to a brand-new empty run_dir."""
        run_dir = os.path.join(self.tmp_dir, "run_files")
        os.makedirs(run_dir)

        spec = RunSpec(
            input_dir="/data",
            run_dir=run_dir,
            config_source_path=self.config_path,
        )

        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()
        spec.write_gui_run_spec(run_dir)
        spec.write_command_invoked(run_dir, argv)

        entries = os.listdir(run_dir)
        expected = {"config_effective.yaml", "gui_run_spec.json",
                    "command_invoked.txt"}

        self.assertEqual(set(entries), expected,
                         f"Expected exactly {expected}, got {set(entries)}")

        # No subdirectories
        for name in entries:
            full = os.path.join(run_dir, name)
            self.assertTrue(os.path.isfile(full),
                            f"Expected file, got directory: {name}")

    # ----------------------------------------------------------------
    # No phantom fields on RunSpec
    # ----------------------------------------------------------------
    def test_no_phantom_fields(self):
        """RunSpec does NOT have fields for widgets that don't exist in GUI."""
        spec = RunSpec()
        d = spec.to_dict()
        phantoms = ("traces_only", "event_signal", "preview_first_n",
                    "representative_session_index", "include_rois",
                    "exclude_rois", "events_mode", "cancel_flag_mode",
                    "out_base", "run_id")
        for name in phantoms:
            self.assertNotIn(name, d,
                             f"RunSpec has phantom field: {name}")

    # ----------------------------------------------------------------
    # argv omits optional flags when set to defaults/None
    # ----------------------------------------------------------------
    def test_argv_omits_optional_flags_when_default(self):
        """argv omits optional flags when set to defaults/None."""
        run_dir = os.path.join(self.tmp_dir, "run_defaults")

        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            config_source_path=self.config_path,
        )

        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()

        self.assertNotIn("--sessions-per-hour", argv)
        self.assertNotIn("--session-duration-s", argv)
        self.assertNotIn("--validate-only", argv)
        # smooth-window-s is always included (has a default value)
        self.assertIn("--smooth-window-s", argv)


if __name__ == "__main__":
    unittest.main()
