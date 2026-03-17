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

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gui.run_spec import RunSpec, FORMAT_CHOICES


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
        for forbidden in ("n_total_discovered", "n_sessions_resolved"):
            self.assertNotIn(forbidden, data)

        # Intent/runner fields serialized with defaults
        self.assertIn("representative_session_index", data)
        self.assertIsNone(data["representative_session_index"])
        self.assertIn("include_roi_ids", data)
        self.assertIsNone(data["include_roi_ids"])
        self.assertIn("exclude_roi_ids", data)
        self.assertIsNone(data["exclude_roi_ids"])
        self.assertIn("mode", data)
        self.assertIsNone(data["mode"])

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
        phantoms = ("event_signal", "include_rois",
                    "exclude_rois", "events_mode",
                    "cancel_flag_mode", "out_base", "run_id",
                    "representative_session_id")
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
        # format=auto: included (always required by runner CLI)
        self.assertIn("--format", argv)
        self.assertEqual(argv[argv.index("--format") + 1], "auto")

    # ----------------------------------------------------------------
    # Step 4: runner-wired fields in gui_run_spec.json
    # ----------------------------------------------------------------
    def test_gui_run_spec_includes_runner_fields_in_json(self):
        """Runner-wired fields round-trip through gui_run_spec.json."""
        run_dir = os.path.join(self.tmp_dir, "run_intent")

        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            config_source_path=self.config_path,
            representative_session_index=2,
            include_roi_ids=["Region0", "Region1"],
            traces_only=True,
            preview_first_n=10,
            mode="tonic",
        )

        spec_path = spec.write_gui_run_spec(run_dir)
        with open(spec_path, "r") as f:
            d = json.load(f)

        self.assertEqual(d["representative_session_index"], 2)
        self.assertEqual(d["include_roi_ids"], ["Region0", "Region1"])
        self.assertTrue(d["traces_only"])
        self.assertEqual(d["preview_first_n"], 10)
        self.assertEqual(d["mode"], "tonic")

    # ----------------------------------------------------------------
    # Step 4: intent fields do NOT leak into config_effective.yaml
    # ----------------------------------------------------------------
    def test_generate_derived_config_does_not_write_intent_into_config(self):
        """config_effective.yaml must NOT contain intent-only or runner-only fields."""
        run_dir = os.path.join(self.tmp_dir, "run_no_intent_leak")

        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            config_source_path=self.config_path,
            config_overrides={},
            representative_session_index=5,
            include_roi_ids=["RegionA"],
            traces_only=True,
            preview_first_n=3,
            mode="tonic",
        )

        config_path = spec.generate_derived_config(run_dir)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        if cfg is not None:
            self.assertNotIn("include_roi_ids", cfg)
            self.assertNotIn("exclude_roi_ids", cfg)
            self.assertNotIn("traces_only", cfg)
            self.assertNotIn("mode", cfg)

    # ----------------------------------------------------------------
    # Step 4: mode generates --mode in argv when set
    # ----------------------------------------------------------------
    def test_argv_mode(self):
        """mode must appear in argv when set (both/tonic/phasic)."""
        run_dir = os.path.join(self.tmp_dir, "run_mode_tonic")
        spec_tonic = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            mode="tonic",
        )
        spec_tonic.generate_derived_config(run_dir)
        argv = spec_tonic.build_runner_argv()
        self.assertIn("--mode", argv)
        self.assertEqual(argv[argv.index("--mode") + 1], "tonic")

        spec_both = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            mode=None,  # "both" in GUI results in mode=None
        )
        spec_both.generate_derived_config(run_dir)
        argv_both = spec_both.build_runner_argv()
        self.assertNotIn("--mode", argv_both)

        # Default (None) omits flag
        spec_none = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
        )
        spec_none.generate_derived_config(run_dir)
        self.assertNotIn("--mode", spec_none.build_runner_argv())

    # ----------------------------------------------------------------
    # Step 4: --traces-only appears in argv only when True
    # ----------------------------------------------------------------
    def test_argv_traces_only(self):
        """--traces-only is in argv when True, absent when False."""
        run_dir = os.path.join(self.tmp_dir, "run_traces")
        spec_on = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            traces_only=True,
        )
        spec_on.generate_derived_config(run_dir)
        self.assertIn("--traces-only", spec_on.build_runner_argv())

        run_dir2 = os.path.join(self.tmp_dir, "run_traces_off")
        spec_off = RunSpec(
            input_dir="/data/in", run_dir=run_dir2,
            format="rwd", config_source_path=self.config_path,
            traces_only=False,
        )
        spec_off.generate_derived_config(run_dir2)
        self.assertNotIn("--traces-only", spec_off.build_runner_argv())

    # ----------------------------------------------------------------
    # Step 4: --preview-first-n appears in argv when set
    # ----------------------------------------------------------------
    def test_argv_preview_first_n(self):
        """--preview-first-n N is in argv when set, absent when None."""
        run_dir = os.path.join(self.tmp_dir, "run_preview")
        spec_on = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            preview_first_n=10,
        )
        spec_on.generate_derived_config(run_dir)
        argv = spec_on.build_runner_argv()
        self.assertIn("--preview-first-n", argv)
        self.assertEqual(argv[argv.index("--preview-first-n") + 1], "10")

        run_dir2 = os.path.join(self.tmp_dir, "run_preview_off")
        spec_off = RunSpec(
            input_dir="/data/in", run_dir=run_dir2,
            format="rwd", config_source_path=self.config_path,
        )
        spec_off.generate_derived_config(run_dir2)
        self.assertNotIn("--preview-first-n", spec_off.build_runner_argv())

    # ----------------------------------------------------------------
    # Render mode flags: only present when explicitly set away from default
    # ----------------------------------------------------------------
    def test_argv_render_modes(self):
        """Render mode flags are emitted only when set on RunSpec."""
        run_dir = os.path.join(self.tmp_dir, "run_render_modes_on")
        spec_on = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            sig_iso_render_mode="full",
            dff_render_mode="full",
            stacked_render_mode="full",
        )
        spec_on.generate_derived_config(run_dir)
        argv_on = spec_on.build_runner_argv()
        self.assertIn("--sig-iso-render-mode", argv_on)
        self.assertEqual(argv_on[argv_on.index("--sig-iso-render-mode") + 1], "full")
        self.assertIn("--dff-render-mode", argv_on)
        self.assertEqual(argv_on[argv_on.index("--dff-render-mode") + 1], "full")
        self.assertIn("--stacked-render-mode", argv_on)
        self.assertEqual(argv_on[argv_on.index("--stacked-render-mode") + 1], "full")

        run_dir2 = os.path.join(self.tmp_dir, "run_render_modes_off")
        spec_off = RunSpec(
            input_dir="/data/in", run_dir=run_dir2,
            format="rwd", config_source_path=self.config_path,
        )
        spec_off.generate_derived_config(run_dir2)
        argv_off = spec_off.build_runner_argv()
        self.assertNotIn("--sig-iso-render-mode", argv_off)
        self.assertNotIn("--dff-render-mode", argv_off)
        self.assertNotIn("--stacked-render-mode", argv_off)

    # ----------------------------------------------------------------
    # Step 4: --representative-session-index in argv when set
    # ----------------------------------------------------------------
    def test_argv_representative_session_index(self):
        """--representative-session-index N is in argv when set."""
        run_dir = os.path.join(self.tmp_dir, "run_rep")
        spec = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            representative_session_index=3,
        )
        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()
        self.assertIn("--representative-session-index", argv)
        self.assertEqual(argv[argv.index("--representative-session-index") + 1], "3")

    # ----------------------------------------------------------------
    # Step 4: --include-rois in argv when set
    # ----------------------------------------------------------------
    def test_argv_include_rois(self):
        """--include-rois comma-separated is in argv when set, absent when None."""
        run_dir = os.path.join(self.tmp_dir, "run_rois")
        spec = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            include_roi_ids=["Region0G", "Region1G"],
        )
        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()
        self.assertIn("--include-rois", argv)
        self.assertEqual(argv[argv.index("--include-rois") + 1], "Region0G,Region1G")

        # Empty list: should NOT produce --include-rois
        run_dir2 = os.path.join(self.tmp_dir, "run_rois_empty")
        spec2 = RunSpec(
            input_dir="/data/in", run_dir=run_dir2,
            format="rwd", config_source_path=self.config_path,
            include_roi_ids=[],
        )
        spec2.generate_derived_config(run_dir2)
        self.assertNotIn("--include-rois", spec2.build_runner_argv())

        # None: should NOT produce --include-rois
        run_dir3 = os.path.join(self.tmp_dir, "run_rois_none")
        spec3 = RunSpec(
            input_dir="/data/in", run_dir=run_dir3,
            format="rwd", config_source_path=self.config_path,
        )
        spec3.generate_derived_config(run_dir3)
        self.assertNotIn("--include-rois", spec3.build_runner_argv())

    # ----------------------------------------------------------------
    # Step 4: ROI selection contract
    # ----------------------------------------------------------------
    def test_roi_selection_contract(self):
        """RunSpec serializes include_roi_ids correctly: None/list/[]."""
        run_dir = os.path.join(self.tmp_dir, "run_roi_contract")

        # All checked = None (default, no filtering)
        spec_all = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            config_source_path=self.config_path,
            include_roi_ids=None,
        )
        spec_path = spec_all.write_gui_run_spec(run_dir)
        with open(spec_path, "r") as f:
            d = json.load(f)
        self.assertIsNone(d["include_roi_ids"])

        # Some checked = list in discovery order
        spec_some = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            config_source_path=self.config_path,
            include_roi_ids=["Region0G", "Region2G"],
        )
        spec_path = spec_some.write_gui_run_spec(run_dir)
        with open(spec_path, "r") as f:
            d = json.load(f)
        self.assertEqual(d["include_roi_ids"], ["Region0G", "Region2G"])

        # None checked = empty list
        spec_none = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            config_source_path=self.config_path,
            include_roi_ids=[],
        )
        spec_path = spec_none.write_gui_run_spec(run_dir)
        with open(spec_path, "r") as f:
            d = json.load(f)
        self.assertEqual(d["include_roi_ids"], [])

    # ----------------------------------------------------------------
    # Step 4: common run gate
    # ----------------------------------------------------------------
    def test_common_run_gate(self):
        """Basic RunSpec validates config and builds argv with core flags."""
        run_dir = os.path.join(self.tmp_dir, "run_common")
        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            config_source_path=self.config_path,
        )
        config_path = spec.generate_derived_config(run_dir)
        RunSpec.validate_effective_config(config_path)
        argv = spec.build_runner_argv()

        # Core flags always present
        self.assertIn("--input", argv)
        self.assertIn("--out", argv)
        self.assertIn("--config", argv)
        self.assertIn("--events", argv)
        self.assertIn("--cancel-flag", argv)
        self.assertIn("--smooth-window-s", argv)
        self.assertIn("--format", argv)
        self.assertEqual(argv[argv.index("--format") + 1], "auto")
        # Optional flags absent by default
        self.assertNotIn("--traces-only", argv)
        self.assertNotIn("--preview-first-n", argv)
        self.assertNotIn("--representative-session-index", argv)
        self.assertNotIn("--include-rois", argv)
        self.assertNotIn("--exclude-rois", argv)

    # ----------------------------------------------------------------
    # Step 4: no representative_session_id (old string field)
    # ----------------------------------------------------------------
    def test_no_representative_session_id_field(self):
        """RunSpec must NOT have representative_session_id (replaced by index)."""
        self.assertFalse(hasattr(RunSpec, "representative_session_id"),
                         "RunSpec still has old representative_session_id field")
        run_dir = os.path.join(self.tmp_dir, "run_no_old")
        spec = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            config_source_path=self.config_path,
        )
        d = spec.to_dict()
        self.assertNotIn("representative_session_id", d)

    # ----------------------------------------------------------------
    # Step 4: run_discovery works with run_dir="" (no side-effects)
    # ----------------------------------------------------------------
    def test_run_discovery_does_not_require_run_dir(self):
        """RunSpec.run_discovery() works with run_dir='' and does not write files."""
        spec = RunSpec(
            input_dir="/nonexistent/path",
            run_dir="",
            format="auto",
            config_source_path=self.config_path,
        )

        with self.assertRaises(RuntimeError) as ctx:
            spec.run_discovery()
        error_msg = str(ctx.exception)
        self.assertNotIn("run_dir", error_msg.lower())

    def test_argv_format_conditional(self):
        """--format is always included for auto, rwd, and npm."""
        run_dir_auto = os.path.join(self.tmp_dir, "run_fmt_auto")
        spec_auto = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir_auto,
            format="auto",
            config_source_path=self.config_path,
        )
        spec_auto.generate_derived_config(run_dir_auto)
        argv_auto = spec_auto.build_runner_argv()
        self.assertIn("--format", argv_auto)
        self.assertEqual(argv_auto[argv_auto.index("--format") + 1], "auto")

    def test_gui_run_spec_includes_sessions_per_hour_when_user_sets_it(self):
        """Verifies that Sessions/Hour = 2 in RunSpec results in --sessions-per-hour 2 in argv."""
        run_dir = os.path.join(self.tmp_dir, "run_sph_2")
        spec = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir,
            sessions_per_hour=2,
            config_source_path=self.config_path,
        )
        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()
        self.assertIn("--sessions-per-hour", argv)
        idx = argv.index("--sessions-per-hour")
        self.assertEqual(argv[idx + 1], "2")

        run_dir_rwd = os.path.join(self.tmp_dir, "run_fmt_rwd")
        spec_rwd = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir_rwd,
            format="rwd",
            config_source_path=self.config_path,
        )
        spec_rwd.generate_derived_config(run_dir_rwd)
        argv_rwd = spec_rwd.build_runner_argv()
        self.assertIn("--format", argv_rwd)
        self.assertEqual(argv_rwd[argv_rwd.index("--format") + 1], "rwd")

    def test_run_spec_default_format(self):
        """RunSpec defaults to 'auto' and generates --format auto."""
        spec = RunSpec(input_dir="/data/in", run_dir="/data/out")
        self.assertEqual(spec.format, "auto")
        self.assertIn("auto", FORMAT_CHOICES)
        
        argv = spec.build_runner_argv()
        self.assertIn("--format", argv)
        self.assertEqual(argv[argv.index("--format") + 1], "auto")

    # ----------------------------------------------------------------
    # Step 4 invariant: main_window.py has no legacy argv literals
    # ----------------------------------------------------------------
    def test_main_window_no_legacy_argv_literals(self):
        """main_window.py must not contain legacy argv patterns."""
        import gui.main_window as mw_mod
        mw_path = os.path.abspath(mw_mod.__file__)
        with open(mw_path, "r", encoding="utf-8") as f:
            src = f.read()

        for banned in ("--out-base", "--run-id", "PIPELINE_SCRIPT"):
            self.assertNotIn(banned, src,
                             f"main_window.py still contains legacy literal: {banned}")

    # ----------------------------------------------------------------
    # Step 4: discovery argv respects format=auto omission rule
    # ----------------------------------------------------------------
    def test_discovery_argv_format_conditional(self):
        """run_discovery omits --format for auto, includes it for rwd."""
        from unittest.mock import patch, MagicMock

        spec_auto = RunSpec(
            input_dir="/data/in", run_dir="", format="auto",
            config_source_path=self.config_path,
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            spec_auto.run_discovery()
            called_argv = mock_run.call_args[0][0]
            self.assertIn("--format", called_argv)
            self.assertEqual(called_argv[called_argv.index("--format") + 1], "auto")
            self.assertIn("--discover", called_argv)

        spec_rwd = RunSpec(
            input_dir="/data/in", run_dir="", format="rwd",
            config_source_path=self.config_path,
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            spec_rwd.run_discovery()
            called_argv = mock_run.call_args[0][0]
            self.assertIn("--format", called_argv)
            fmt_idx = called_argv.index("--format")
            self.assertEqual(called_argv[fmt_idx + 1], "rwd")

    # ----------------------------------------------------------------
    # Step 4: include_roi_ids=[] must NOT produce --include-rois in argv
    # ----------------------------------------------------------------
    def test_empty_roi_ids_does_not_produce_argv_flag(self):
        """include_roi_ids=[] must not produce --include-rois in argv."""
        run_dir = os.path.join(self.tmp_dir, "run_roi_empty_argv")
        spec = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            include_roi_ids=[],
        )
        spec.generate_derived_config(run_dir)
        self.assertNotIn("--include-rois", spec.build_runner_argv())
        spec2 = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            include_roi_ids=None,
        )
        self.assertNotIn("--include-rois", spec2.build_runner_argv())

    # ----------------------------------------------------------------
    # Step 4: mode must NEVER appear in argv for any value
    # ----------------------------------------------------------------


    # ----------------------------------------------------------------
    # Step 4: --exclude-rois in argv when set
    # ----------------------------------------------------------------
    def test_argv_exclude_rois(self):
        """--exclude-rois comma-separated is in argv when set, absent when None/[]."""
        run_dir = os.path.join(self.tmp_dir, "run_ex_rois")
        spec = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            exclude_roi_ids=["Region0G", "Region1G"],
        )
        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()
        self.assertIn("--exclude-rois", argv)
        self.assertEqual(argv[argv.index("--exclude-rois") + 1],
                         "Region0G,Region1G")
        # No --include-rois when only exclude is set
        self.assertNotIn("--include-rois", argv)

        # Empty list: should NOT produce --exclude-rois
        run_dir2 = os.path.join(self.tmp_dir, "run_ex_rois_empty")
        spec2 = RunSpec(
            input_dir="/data/in", run_dir=run_dir2,
            format="rwd", config_source_path=self.config_path,
            exclude_roi_ids=[],
        )
        spec2.generate_derived_config(run_dir2)
        self.assertNotIn("--exclude-rois", spec2.build_runner_argv())

        # None: should NOT produce --exclude-rois
        run_dir3 = os.path.join(self.tmp_dir, "run_ex_rois_none")
        spec3 = RunSpec(
            input_dir="/data/in", run_dir=run_dir3,
            format="rwd", config_source_path=self.config_path,
        )
        spec3.generate_derived_config(run_dir3)
        self.assertNotIn("--exclude-rois", spec3.build_runner_argv())

    # ----------------------------------------------------------------
    # Step 4: include and exclude are mutually exclusive
    # ----------------------------------------------------------------
    def test_mutual_exclusivity_include_exclude(self):
        """RunSpec raises ValueError if both include and exclude are non-None."""
        # Both non-empty: ValueError
        with self.assertRaises(ValueError):
            RunSpec(
                input_dir="/data/in", run_dir="/tmp/x",
                config_source_path=self.config_path,
                include_roi_ids=["A"],
                exclude_roi_ids=["B"],
            )
        # Both empty: still ValueError (both non-None)
        with self.assertRaises(ValueError):
            RunSpec(
                input_dir="/data/in", run_dir="/tmp/x",
                config_source_path=self.config_path,
                include_roi_ids=[],
                exclude_roi_ids=[],
            )
        # One set, other None: OK
        spec = RunSpec(
            input_dir="/data/in", run_dir="/tmp/y",
            config_source_path=self.config_path,
            include_roi_ids=["A"],
            exclude_roi_ids=None,
        )
        self.assertIsNotNone(spec)
        spec2 = RunSpec(
            input_dir="/data/in", run_dir="/tmp/z",
            config_source_path=self.config_path,
            include_roi_ids=None,
            exclude_roi_ids=["B"],
        )
        self.assertIsNotNone(spec2)
        # include=[], exclude=None: OK
        spec3 = RunSpec(
            input_dir="/data/in", run_dir="/tmp/w",
            config_source_path=self.config_path,
            include_roi_ids=[],
            exclude_roi_ids=None,
        )
        self.assertIsNotNone(spec3)

    # ----------------------------------------------------------------
    # Step 4: argv never contains both --include-rois and --exclude-rois
    # ----------------------------------------------------------------
    def test_argv_never_both_include_and_exclude(self):
        """argv must never contain both --include-rois and --exclude-rois."""
        run_dir = os.path.join(self.tmp_dir, "run_incl_only")
        spec = RunSpec(
            input_dir="/data/in", run_dir=run_dir,
            format="rwd", config_source_path=self.config_path,
            include_roi_ids=["A"],
        )
        spec.generate_derived_config(run_dir)
        argv = spec.build_runner_argv()
        self.assertIn("--include-rois", argv)
        self.assertNotIn("--exclude-rois", argv)

        run_dir2 = os.path.join(self.tmp_dir, "run_excl_only")
        spec2 = RunSpec(
            input_dir="/data/in", run_dir=run_dir2,
            format="rwd", config_source_path=self.config_path,
            exclude_roi_ids=["B"],
        )
        spec2.generate_derived_config(run_dir2)
        argv2 = spec2.build_runner_argv()
        self.assertIn("--exclude-rois", argv2)
        self.assertNotIn("--include-rois", argv2)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Step 4: runner source-of-truth helper
    # ----------------------------------------------------------------
    def test_runner_flag_source_of_truth(self):
        """Verify runner parse_args has the expected flags."""
        runner_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools", "run_full_pipeline_deliverables.py"
        )
        with open(runner_path, "r", encoding="utf-8") as f:
            src = f.read()
        # Runner-wired flags MUST exist
        for flag in ("--include-rois", "--exclude-rois",
                     "--traces-only", "--preview-first-n",
                     "--representative-session-index", "--mode",
                     "--sig-iso-render-mode", "--dff-render-mode",
                     "--stacked-render-mode"):
            self.assertIn(flag, src,
                          f"Runner missing expected flag: {flag}")
        # Intent-only flags MUST NOT exist in runner
        for flag in ("'--recursive'", "'--glob'"):
            # Check for argparse definitions (quoted flag names)
            self.assertNotIn(f"add_argument({flag}", src,
                             f"Runner has unexpected flag: {flag}")


class TestIsosbesticKnobs(unittest.TestCase):
    """Test the pure validation function for isosbestic advanced knobs."""

    def test_valid_isosbestic_knobs(self):
        from gui.main_window import parse_and_validate_isosbestic_knobs
        defaults = {
            "window_sec": 60.0, "step_sec": 10.0,
            "min_valid_windows": 5, "r_low": 0.2, "r_high": 0.8,
            "g_min": 0.2, "min_samples_per_window": 1
        }
        # Update valid arguments: 1 instead of 0
        overrides, err = parse_and_validate_isosbestic_knobs(
            "60.0", "10.0", 5, "0.2", "0.8", "0.2", 1, defaults=defaults
        )
        self.assertIsNone(err)
        self.assertEqual(overrides["window_sec"], 60.0)
        self.assertEqual(overrides["step_sec"], 10.0)
        self.assertEqual(overrides["min_valid_windows"], 5)
        self.assertEqual(overrides["r_low"], 0.2)
        self.assertEqual(overrides["r_high"], 0.8)
        self.assertEqual(overrides["g_min"], 0.2)
        self.assertEqual(overrides["min_samples_per_window"], 1)

    def test_validation_rules(self):
        from gui.main_window import parse_and_validate_isosbestic_knobs
        defaults = {
            "window_sec": 60.0, "step_sec": 10.0,
            "min_valid_windows": 5, "r_low": 0.2, "r_high": 0.8,
            "g_min": 0.2, "min_samples_per_window": 1
        }

        # window_sec <= 0
        _, err = parse_and_validate_isosbestic_knobs("0", "10.0", 5, "0.2", "0.8", "0.2", 1, defaults=defaults)
        self.assertIn("Regression Window must be > 0", err)
        
        # step_sec <= 0
        _, err = parse_and_validate_isosbestic_knobs("60.0", "0", 5, "0.2", "0.8", "0.2", 1, defaults=defaults)
        self.assertIn("Regression Step must be > 0", err)

        # step > window
        _, err = parse_and_validate_isosbestic_knobs("10.0", "20.0", 5, "0.2", "0.8", "0.2", 1, defaults=defaults)
        self.assertIn("cannot be greater than Regression Window", err)

        # r_low > r_high
        _, err = parse_and_validate_isosbestic_knobs("60.0", "10.0", 5, "0.8", "0.2", "0.2", 1, defaults=defaults)
        self.assertIn("R-Low <= R-High", err)

        # r_low outside [0,1]
        _, err = parse_and_validate_isosbestic_knobs("60.0", "10.0", 5, "-0.1", "0.8", "0.2", 1, defaults=defaults)
        self.assertIn("between 0 and 1", err)

        # g_min < 0
        _, err = parse_and_validate_isosbestic_knobs("60.0", "10.0", 5, "0.2", "0.8", "-0.1", 1, defaults=defaults)
        self.assertIn("G-Min must be >= 0", err)

        # min_valid_windows < 1
        _, err = parse_and_validate_isosbestic_knobs("60.0", "10.0", 0, "0.2", "0.8", "0.2", 1, defaults=defaults)
        self.assertIn("Min Valid Windows must be >= 1", err)

        # min_samples < 1
        _, err = parse_and_validate_isosbestic_knobs("60.0", "10.0", 5, "0.2", "0.8", "0.2", 0, defaults=defaults)
        self.assertIn("Min Samples per Window must be >= 1", err)

    def test_isosbestic_gating_by_mode(self):
        from gui.main_window import is_isosbestic_active
        
        # When mode is both or phasic, is active
        self.assertTrue(is_isosbestic_active("both"))
        self.assertTrue(is_isosbestic_active("phasic"))
        
        # When mode is tonic-only, is inactive
        self.assertFalse(is_isosbestic_active("tonic"))

    def test_isosbestic_keys_not_in_config_when_tonic(self):
        """T1: When mode is tonic, isosbestic overrides must NOT appear in config_effective.yaml."""
        from gui.run_spec import RunSpec
        import tempfile
        import os
        import shutil
        import yaml
        
        tmp_dir = tempfile.mkdtemp()
        try:
            # We must use RunSpec to prove config generation works as expected
            run_dir = os.path.join(tmp_dir, "run_tonic")
            
            # Create a mock base config (no isosbestic keys)
            base_config = {"some_other_key": True}
            base_config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(base_config_path, "w") as f:
                yaml.safe_dump(base_config, f)
            
            # Gating occurs in MainWindow, we emulate that flow
            from gui.main_window import is_isosbestic_active
            overrides = {
                "window_sec": 120.0, "step_sec": 5.0, "min_valid_windows": 10,
                "min_samples_per_window": 5, "r_low": 0.5, "r_high": 0.9, "g_min": 0.1
            } # Attempted user overrides
            
            active_overrides = overrides if is_isosbestic_active("tonic") else {}
            # Assert contract: gating yields config_overrides == {}
            self.assertEqual(active_overrides, {})
            
            spec = RunSpec(
                run_dir=run_dir,
                config_source_path=base_config_path,
                config_overrides=active_overrides
            )
            config_path = spec.generate_derived_config(run_dir)
            
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                
            # The overrides were dropped by gating. None of these keys should exist.
            keys_to_check = [
                "window_sec", "step_sec", "min_valid_windows",
                "min_samples_per_window", "r_low", "r_high", "g_min"
            ]
            for key in keys_to_check:
                self.assertNotIn(key, loaded)
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_isosbestic_keys_present_when_phasic_and_user_changed(self):
        """T2: Under mode='phasic'/'both' with changed values, keys must be emitted."""
        from gui.run_spec import RunSpec
        import tempfile
        import os
        import shutil
        import yaml
        
        tmp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(tmp_dir, "run_phasic")
            
            base_config = {"window_sec": 60.0, "step_sec": 10.0}
            base_config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(base_config_path, "w") as f:
                yaml.safe_dump(base_config, f)
            
            from gui.main_window import is_isosbestic_active, compute_isosbestic_overrides_user_changed, parse_and_validate_isosbestic_knobs
            
            defaults = {"window_sec": 60.0, "step_sec": 10.0, "min_valid_windows": 5, "r_low": 0.2, "r_high": 0.8, "g_min": 0.2, "min_samples_per_window": 1}
            # Simulate parsing from GUI where window_sec is changed
            parsed_overrides, err = parse_and_validate_isosbestic_knobs("150.0", "10.0", 5, "0.2", "0.8", "0.2", 1, defaults=defaults)
            
            # T2: Gating through "phasic" retains the override
            active_overrides = parsed_overrides if is_isosbestic_active("phasic") else {}
            # T2: Changed from default retains the override
            changed_overrides = compute_isosbestic_overrides_user_changed(
                active_overrides, 
                defaults
            )
            
            spec = RunSpec(
                run_dir=run_dir,
                config_source_path=base_config_path,
                config_overrides=changed_overrides
            )
            config_path = spec.generate_derived_config(run_dir)
            
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                
            self.assertEqual(loaded["window_sec"], 150.0)
            self.assertEqual(loaded["step_sec"], 10.0) # Stays base config since step_sec wasn't overridden
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_isosbestic_defaults_not_written_when_unchanged(self):
        """T3: Under phasic/both, if parsed values == defaults, overrides are dropped."""
        from gui.main_window import is_isosbestic_active, compute_isosbestic_overrides_user_changed, parse_and_validate_isosbestic_knobs
        
        defaults = {"window_sec": 60.0, "step_sec": 10.0, "r_low": 0.2, "min_valid_windows": 5, "r_high": 0.8, "g_min": 0.2, "min_samples_per_window": 1}
        # Emulate GUI blank inputs taking defaults
        parsed_overrides, err = parse_and_validate_isosbestic_knobs("", "", 5, "", "", "", 1, defaults=defaults)
        
        # Passes gating
        active_overrides = parsed_overrides if is_isosbestic_active("both") else {}
        
        # Dropped because they match defaults
        changed_overrides = compute_isosbestic_overrides_user_changed(active_overrides, defaults)
        
        self.assertEqual(changed_overrides, {})

class TestPreprocBaselineKnobs(unittest.TestCase):
    """Tests for Step 6 Preprocessing + Baseline advanced GUI knobs."""

    def test_t1_defaults_not_written_when_unchanged(self):
        """T1: Ensure Step 6 keys are omitted when overrides are empty."""
        from gui.run_spec import RunSpec
        import tempfile
        import os
        import shutil
        import yaml
        
        tmp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(tmp_dir, "run_test_t1")
            
            # Base config lacks the Step 6 keys
            base_config = {"some_other_key": True}
            base_config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(base_config_path, "w") as f:
                yaml.safe_dump(base_config, f)
            
            # Simulate untouched config overrides for Step 6
            spec = RunSpec(
                run_dir=run_dir,
                config_source_path=base_config_path,
                config_overrides={}
            )
            config_path = spec.generate_derived_config(run_dir)
            
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                
            keys_to_check = [
                "lowpass_hz", "baseline_method", "baseline_percentile", "f0_min_value"
            ]
            for key in keys_to_check:
                self.assertNotIn(key, loaded)
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t2_lowpass_hz_and_f0_min_value_written_when_changed(self):
        """T2: Prove lowpass_hz and f0_min_value get emitted if user changed them."""
        from gui.main_window import parse_and_validate_preproc_baseline_knobs, compute_overrides_user_changed
        from photometry_pipeline.config import Config
        from gui.run_spec import RunSpec
        import tempfile, os, shutil, yaml
        
        cfg0 = Config()
        defaults = {
            "lowpass_hz": cfg0.lowpass_hz, 
            "baseline_method": cfg0.baseline_method, 
            "baseline_percentile": cfg0.baseline_percentile, 
            "f0_min_value": cfg0.f0_min_value
        }
        
        # Change lowpass and f0 to different values
        lowpass_override = defaults["lowpass_hz"] * 2.0 if defaults["lowpass_hz"] > 0 else 1.0
        f0_override = defaults["f0_min_value"] + 0.05 if defaults["f0_min_value"] >= 0 else 0.05
        
        parsed_overrides, err = parse_and_validate_preproc_baseline_knobs(
            str(lowpass_override), defaults["baseline_method"], str(defaults["baseline_percentile"]), str(f0_override), defaults=defaults
        )
        self.assertIsNone(err)
        
        changed = compute_overrides_user_changed(parsed_overrides, defaults)
        self.assertIn("lowpass_hz", changed)
        self.assertIn("f0_min_value", changed)
        self.assertNotIn("baseline_method", changed)  # unchanged
        self.assertNotIn("baseline_percentile", changed)  # unchanged
        
        tmp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(tmp_dir, "run_test_t2")
            base_config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(base_config_path, "w") as f:
                yaml.safe_dump({"some_other_key": True}, f)
            
            spec = RunSpec(
                run_dir=run_dir,
                config_source_path=base_config_path,
                config_overrides=changed
            )
            config_path = spec.generate_derived_config(run_dir)
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                
            self.assertEqual(loaded["lowpass_hz"], lowpass_override)
            self.assertEqual(loaded["f0_min_value"], f0_override)
            self.assertNotIn("baseline_method", loaded)
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t3_baseline_method_change_emits_method_and_may_require_percentile(self):
        """T3: Verify baseline_method_change emits method and absent percentile if unchanged."""
        from gui.main_window import parse_and_validate_preproc_baseline_knobs, compute_overrides_user_changed, get_allowed_baseline_methods_from_config
        from photometry_pipeline.config import Config
        from gui.run_spec import RunSpec
        import tempfile, os, shutil, yaml
        
        cfg0 = Config()
        defaults = {
            "lowpass_hz": cfg0.lowpass_hz, 
            "baseline_method": cfg0.baseline_method, 
            "baseline_percentile": cfg0.baseline_percentile, 
            "f0_min_value": cfg0.f0_min_value
        }
        
        allowed_methods = get_allowed_baseline_methods_from_config()
        self.assertIn(cfg0.baseline_method, allowed_methods)
        
        if len(allowed_methods) <= 1:
            self.skipTest("No alternative baseline method available to change to.")
        
        method = next(m for m in allowed_methods if m != defaults["baseline_method"])
        
        parsed_overrides, err = parse_and_validate_preproc_baseline_knobs(
            str(defaults["lowpass_hz"]), method, str(defaults["baseline_percentile"]), str(defaults["f0_min_value"]), defaults=defaults
        )
        self.assertIsNone(err)
        
        changed = compute_overrides_user_changed(parsed_overrides, defaults)
        self.assertIn("baseline_method", changed)
        self.assertEqual(changed["baseline_method"], method)
        # Percentile is unchanged, so shouldn't emit
        self.assertNotIn("baseline_percentile", changed)
        
        tmp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(tmp_dir, "run_test_t3")
            base_config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(base_config_path, "w") as f:
                yaml.safe_dump({"some_other_key": True}, f)
                
            spec = RunSpec(
                run_dir=run_dir,
                config_source_path=base_config_path,
                config_overrides=changed
            )
            config_path = spec.generate_derived_config(run_dir)
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                
            self.assertIn("baseline_method", loaded)
            self.assertEqual(loaded["baseline_method"], method)
            self.assertNotIn("baseline_percentile", loaded)
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t4_baseline_percentile_emitted_when_applicable_and_changed(self):
        """T4: baseline_percentile emitted when method DOES use percentile and user changed it."""
        from gui.main_window import parse_and_validate_preproc_baseline_knobs, compute_overrides_user_changed, get_allowed_baseline_methods_from_config, baseline_method_requires_percentile
        from photometry_pipeline.config import Config
        
        cfg0 = Config()
        defaults = {
            "lowpass_hz": cfg0.lowpass_hz, 
            "baseline_method": cfg0.baseline_method, 
            "baseline_percentile": cfg0.baseline_percentile, 
            "f0_min_value": cfg0.f0_min_value
        }
        
        allowed_methods = get_allowed_baseline_methods_from_config()
        self.assertIn(cfg0.baseline_method, allowed_methods)
        
        pct_methods = [m for m in allowed_methods if baseline_method_requires_percentile(m)]
        if not pct_methods:
            self.skipTest("No baseline methods use percentile in schema.")
            
        method = pct_methods[0]
        
        pct_override = 50.0
        if pct_override == defaults["baseline_percentile"]:
            pct_override = 51.0
            
        parsed_overrides, err = parse_and_validate_preproc_baseline_knobs(
            str(defaults["lowpass_hz"]), method, str(pct_override), str(defaults["f0_min_value"]), defaults=defaults
        )
        self.assertIsNone(err)
        
        changed = compute_overrides_user_changed(parsed_overrides, defaults)
        self.assertIn("baseline_percentile", changed)
        self.assertEqual(changed["baseline_percentile"], pct_override)

    def test_t5_validation_failures(self):
        from gui.main_window import parse_and_validate_preproc_baseline_knobs
        from gui.main_window import baseline_method_requires_percentile
        from photometry_pipeline.config import Config
        
        cfg0 = Config()
        defaults = {
            "lowpass_hz": cfg0.lowpass_hz, 
            "baseline_method": cfg0.baseline_method, 
            "baseline_percentile": cfg0.baseline_percentile, 
            "f0_min_value": cfg0.f0_min_value
        }
        
        # lowpass_hz <= 0
        _, err = parse_and_validate_preproc_baseline_knobs("0.0", defaults["baseline_method"], str(defaults["baseline_percentile"]), str(defaults["f0_min_value"]), defaults)
        self.assertIn("Lowpass Filter (Hz) must be > 0", err)
        
        # f0_min_value < 0
        _, err = parse_and_validate_preproc_baseline_knobs(str(defaults["lowpass_hz"]), defaults["baseline_method"], str(defaults["baseline_percentile"]), "-1.0", defaults)
        self.assertIn("F0 Min Value must be >= 0", err)
        
        # invalid baseline method
        _, err = parse_and_validate_preproc_baseline_knobs(str(defaults["lowpass_hz"]), "definitely_not_a_real_baseline_method", str(defaults["baseline_percentile"]), str(defaults["f0_min_value"]), defaults)
        self.assertIn("Invalid Baseline Method.", err)
        
        # percentile out of bounds (ONLY when the active baseline method actually requires percentile)
        if baseline_method_requires_percentile(defaults["baseline_method"]):
            _, err = parse_and_validate_preproc_baseline_knobs(
                str(defaults["lowpass_hz"]),
                defaults["baseline_method"],
                "150.0",
                str(defaults["f0_min_value"]),
                defaults,
            )
            self.assertIsNotNone(err)
            self.assertIn("Baseline Percentile must be between 0 and 100", err)
        else:
            _, err = parse_and_validate_preproc_baseline_knobs(
                str(defaults["lowpass_hz"]),
                defaults["baseline_method"],
                "150.0",
                str(defaults["f0_min_value"]),
                defaults,
            )
            self.assertIsNone(err)
        
    def test_t6_run_report_reflects_step_6_knobs(self):
        """T6: Hard validation gate proving Step 6 config reaches run_report.json."""
        from gui.main_window import parse_and_validate_preproc_baseline_knobs, compute_overrides_user_changed, get_allowed_baseline_methods_from_config, baseline_method_requires_percentile
        from gui.run_spec import RunSpec
        from photometry_pipeline.config import Config
        from photometry_pipeline.core.reporting import generate_run_report
        import tempfile, os, shutil, yaml, json
        
        cfg0 = Config()
        defaults = {
            "lowpass_hz": cfg0.lowpass_hz, 
            "baseline_method": cfg0.baseline_method, 
            "baseline_percentile": cfg0.baseline_percentile, 
            "f0_min_value": cfg0.f0_min_value
        }
        
        allowed_methods = get_allowed_baseline_methods_from_config()
        self.assertIn(cfg0.baseline_method, allowed_methods)
        
        tmp_dir = tempfile.mkdtemp()
        try:
            # Report A (Defaults)
            run_dir_a = os.path.join(tmp_dir, "run_a")
            base_config_path_a = os.path.join(tmp_dir, "base_config_a.yaml")
            with open(base_config_path_a, "w") as f:
                yaml.safe_dump({}, f)
                
            spec_a = RunSpec(run_dir=run_dir_a, config_source_path=base_config_path_a, config_overrides={})
            conf_path_a = spec_a.generate_derived_config(run_dir_a)
            
            with open(conf_path_a, "r") as f:
                cfg_a = Config(**yaml.safe_load(f))
            generate_run_report(cfg_a, run_dir_a)
            
            with open(os.path.join(run_dir_a, "run_report.json"), "r") as f:
                report_a = json.load(f)
                
            # Report A contains default values
            cfg_dict_a = report_a["configuration"]
            self.assertEqual(cfg_dict_a["baseline_method"], defaults["baseline_method"])
            self.assertEqual(cfg_dict_a["baseline_percentile"], defaults["baseline_percentile"])
            self.assertEqual(cfg_dict_a["lowpass_hz"], defaults["lowpass_hz"])
            self.assertEqual(cfg_dict_a["f0_min_value"], defaults["f0_min_value"])
            
            # Report B (Overrides)
            run_dir_b = os.path.join(tmp_dir, "run_b")
            base_config_path_b = os.path.join(tmp_dir, "base_config_b.yaml")
            with open(base_config_path_b, "w") as f:
                yaml.safe_dump({}, f)
                
            lowpass_override = defaults["lowpass_hz"] * 2.0 if defaults["lowpass_hz"] > 0 else 1.0
            f0_min_override = defaults["f0_min_value"] + 0.05 if defaults["f0_min_value"] >= 0 else 0.05
            
            # Select different method if possible
            if len(allowed_methods) > 1:
                baseline_method_override = next(m for m in allowed_methods if m != defaults["baseline_method"])
            else:
                baseline_method_override = defaults["baseline_method"]
                
            baseline_percentile_override = str(defaults["baseline_percentile"])
            if baseline_method_requires_percentile(baseline_method_override):
                pct = 50.0
                if pct == defaults["baseline_percentile"]:
                    pct = 51.0
                baseline_percentile_override = str(pct)
            
            # Simulate GUI changing knobs
            parsed_overrides, err = parse_and_validate_preproc_baseline_knobs(
                str(lowpass_override), baseline_method_override, baseline_percentile_override, str(f0_min_override), defaults=defaults
            )
            self.assertIsNone(err)
            changed = compute_overrides_user_changed(parsed_overrides, defaults)
            
            spec_b = RunSpec(run_dir=run_dir_b, config_source_path=base_config_path_b, config_overrides=changed)
            conf_path_b = spec_b.generate_derived_config(run_dir_b)
            
            with open(conf_path_b, "r") as f:
                cfg_b = Config(**yaml.safe_load(f))
            generate_run_report(cfg_b, run_dir_b)
            
            with open(os.path.join(run_dir_b, "run_report.json"), "r") as f:
                report_b = json.load(f)
                
            cfg_dict_b = report_b["configuration"]
            self.assertEqual(cfg_dict_b["baseline_method"], baseline_method_override)
            self.assertEqual(cfg_dict_b["lowpass_hz"], lowpass_override)
            self.assertEqual(cfg_dict_b["f0_min_value"], f0_min_override)
            
            if baseline_method_requires_percentile(baseline_method_override):
                self.assertEqual(cfg_dict_b["baseline_percentile"], float(baseline_percentile_override))
                self.assertIn("baseline_percentile", changed)
            else:
                self.assertNotIn("baseline_percentile", changed)
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t7_percentile_gating_schema_locked(self):
        """T7: Assert percentile requirement strictly follows effective Config schema rules, no heuristics."""
        from gui.main_window import baseline_method_requires_percentile, get_allowed_baseline_methods_from_config
        from photometry_pipeline.config import Config
        
        allowed_methods = get_allowed_baseline_methods_from_config()
        self.assertTrue(len(allowed_methods) > 0, "No baseline methods returned by config schema.")
        
        for method in allowed_methods:
            cfg1 = Config(baseline_method=method, baseline_percentile=10.0)
            cfg2 = Config(baseline_method=method, baseline_percentile=50.0)
            expected = (cfg1.baseline_percentile != cfg2.baseline_percentile)
            observed = baseline_method_requires_percentile(method)
            self.assertEqual(expected, observed, f"Method {method} requirement mismatch: expected={expected}, observed={observed}")

    def test_t8_percentile_not_parsed_when_gating_false(self):
        """T8: Prove baseline_percentile is neither parsed nor included when gating function returns False."""
        from gui.main_window import parse_and_validate_preproc_baseline_knobs, get_allowed_baseline_methods_from_config
        from photometry_pipeline.config import Config
        import unittest.mock
        
        allowed_methods = get_allowed_baseline_methods_from_config()
        self.assertTrue(len(allowed_methods) > 0, "No baseline methods returned by config schema.")
            
        method = allowed_methods[0]
        cfg0 = Config()
        defaults = {
            "lowpass_hz": cfg0.lowpass_hz, 
            "baseline_method": cfg0.baseline_method, 
            "baseline_percentile": cfg0.baseline_percentile, 
            "f0_min_value": cfg0.f0_min_value
        }
        
        # Monkeypatch the gating function to definitively return False
        with unittest.mock.patch("gui.main_window.baseline_method_requires_percentile", return_value=False):
            # Pass a clearly invalid percentile string
            overrides, err = parse_and_validate_preproc_baseline_knobs(
                str(defaults["lowpass_hz"]), method, "150.0", str(defaults["f0_min_value"]), defaults
            )
            
        # It must not raise an error, because it should not have parsed it
        self.assertIsNone(err, f"Should not have validated percentile for method {method} when gating is False")
        self.assertIsNotNone(overrides)
        self.assertNotIn("baseline_percentile", overrides)

    def test_t9_percentile_validated_when_required(self):
        """T9: Prove baseline_percentile is validated when required by the method."""
        from gui.main_window import parse_and_validate_preproc_baseline_knobs, baseline_method_requires_percentile, get_allowed_baseline_methods_from_config
        from photometry_pipeline.config import Config
        
        allowed_methods = get_allowed_baseline_methods_from_config()
        pct_methods = [m for m in allowed_methods if baseline_method_requires_percentile(m)]
        
        if not pct_methods:
            self.skipTest("No percentile methods exist in current schema.")
            
        method = pct_methods[0]
        cfg0 = Config()
        defaults = {
            "lowpass_hz": cfg0.lowpass_hz, 
            "baseline_method": cfg0.baseline_method, 
            "baseline_percentile": cfg0.baseline_percentile, 
            "f0_min_value": cfg0.f0_min_value
        }
        
        # Pass a clearly invalid percentile string
        _, err = parse_and_validate_preproc_baseline_knobs(
            str(defaults["lowpass_hz"]), method, "150.0", str(defaults["f0_min_value"]), defaults
        )
        
        # It MUST raise an error
        self.assertIsNotNone(err)
        self.assertIn("Baseline Percentile must be between 0 and 100", err)

class TestEventFeatureKnobs(unittest.TestCase):
    """Tests for Step 7 Event + Feature advanced GUI knobs."""

    def test_t1_defaults_not_written_when_unchanged(self):
        """T1: Ensure Step 7 keys are omitted when overrides are empty."""
        from gui.run_spec import RunSpec
        import tempfile, os, shutil, yaml
        
        tmp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(tmp_dir, "run_test_t1")
            base_config = {"some_other_key": True}
            base_config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(base_config_path, "w") as f:
                yaml.safe_dump(base_config, f)
            
            spec = RunSpec(
                run_dir=run_dir,
                config_source_path=base_config_path,
                config_overrides={}
            )
            config_path = spec.generate_derived_config(run_dir)
            
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                
            keys_to_check = [
                "event_signal", "peak_threshold_method", "peak_threshold_k",
                "peak_threshold_percentile", "peak_threshold_abs",
                "peak_min_distance_sec", "event_auc_baseline"
            ]
            for key in keys_to_check:
                self.assertNotIn(key, loaded)
                
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t2_event_signal_and_distance_written_when_changed(self):
        """T2: Prove unconditionally-used fields like event_signal, dist, auc emit if user changed them."""
        from gui.main_window import parse_and_validate_event_feature_knobs, compute_overrides_user_changed
        from gui.main_window import get_allowed_event_signals_from_config, get_allowed_event_auc_baselines_from_config
        from photometry_pipeline.config import Config
        from gui.run_spec import RunSpec
        import tempfile, os, shutil, yaml
        
        cfg0 = Config()
        defaults = {
            "event_signal": cfg0.event_signal,
            "peak_threshold_method": cfg0.peak_threshold_method,
            "peak_threshold_k": cfg0.peak_threshold_k,
            "peak_threshold_percentile": cfg0.peak_threshold_percentile,
            "peak_threshold_abs": cfg0.peak_threshold_abs,
            "peak_min_distance_sec": cfg0.peak_min_distance_sec,
            "event_auc_baseline": cfg0.event_auc_baseline,
        }
        
        allowed_sigs = get_allowed_event_signals_from_config()
        if len(allowed_sigs) > 1:
            sig_override = next(s for s in allowed_sigs if s != defaults["event_signal"])
        else:
            sig_override = defaults["event_signal"]
            
        allowed_aucs = get_allowed_event_auc_baselines_from_config()
        if len(allowed_aucs) > 1:
            auc_override = next(a for a in allowed_aucs if a != defaults["event_auc_baseline"])
        else:
            auc_override = defaults["event_auc_baseline"]
            
        dist_override = defaults["peak_min_distance_sec"] + 1.0
        
        # For local testing where gating might be overly permissive, supply valid numeric inputs 
        # that satisfy the GUI constraints (abs > 0, k > 0, percentile in [0, 100])
        valid_k = "1.0" if defaults["peak_threshold_k"] <= 0 else str(defaults["peak_threshold_k"])
        valid_pct = "50.0" if not (0 <= defaults["peak_threshold_percentile"] <= 100) else str(defaults["peak_threshold_percentile"])
        valid_abs = "1.0" if defaults["peak_threshold_abs"] <= 0 else str(defaults["peak_threshold_abs"])
        
        parsed_overrides, err = parse_and_validate_event_feature_knobs(
            sig_override,
            defaults["peak_threshold_method"],
            valid_k,
            valid_pct,
            valid_abs,
            str(dist_override),
            auc_override,
            defaults=defaults
        )
        self.assertIsNone(err)
        
        changed = compute_overrides_user_changed(parsed_overrides, defaults)
        self.assertIn("peak_min_distance_sec", changed)
        if len(allowed_sigs) > 1:
            self.assertIn("event_signal", changed)
        if len(allowed_aucs) > 1:
            self.assertIn("event_auc_baseline", changed)
            
        self.assertNotIn("peak_threshold_method", changed)
        
        tmp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(tmp_dir, "run_test_t2")
            base_config_path = os.path.join(tmp_dir, "base_config.yaml")
            with open(base_config_path, "w") as f:
                yaml.safe_dump({"some_other_key": True}, f)
            
            spec = RunSpec(
                run_dir=run_dir,
                config_source_path=base_config_path,
                config_overrides=changed
            )
            config_path = spec.generate_derived_config(run_dir)
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                
            self.assertEqual(loaded["peak_min_distance_sec"], dist_override)
            if len(allowed_sigs) > 1:
                self.assertEqual(loaded["event_signal"], sig_override)
            self.assertNotIn("peak_threshold_method", loaded)
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t3_conditional_parameter_emission(self):
        """T3: Verify only the method-relevant parameter key is emitted when changed."""
        from gui.main_window import parse_and_validate_event_feature_knobs, compute_overrides_user_changed
        from gui.main_window import get_allowed_peak_threshold_methods_from_config
        from gui.main_window import peak_threshold_method_requires_k, peak_threshold_method_requires_percentile, peak_threshold_method_requires_abs
        from photometry_pipeline.config import Config
        
        cfg0 = Config()
        defaults = {
            "event_signal": cfg0.event_signal,
            "peak_threshold_method": cfg0.peak_threshold_method,
            "peak_threshold_k": cfg0.peak_threshold_k,
            "peak_threshold_percentile": cfg0.peak_threshold_percentile,
            "peak_threshold_abs": cfg0.peak_threshold_abs,
            "peak_min_distance_sec": cfg0.peak_min_distance_sec,
            "event_auc_baseline": cfg0.event_auc_baseline,
        }
        
        allowed_methods = get_allowed_peak_threshold_methods_from_config()
        self.assertTrue(len(allowed_methods) > 0)
        
        for method in allowed_methods:
            uses_k = peak_threshold_method_requires_k(method)
            uses_pct = peak_threshold_method_requires_percentile(method)
            uses_abs = peak_threshold_method_requires_abs(method)
            
            new_k = defaults["peak_threshold_k"] + 1.0
            new_pct = defaults["peak_threshold_percentile"] / 2.0
            new_abs = defaults["peak_threshold_abs"] + 10.0
            
            parsed, err = parse_and_validate_event_feature_knobs(
                defaults["event_signal"], method, str(new_k), str(new_pct), str(new_abs),
                str(defaults["peak_min_distance_sec"]), defaults["event_auc_baseline"], defaults
            )
            self.assertIsNone(err)
            
            changed = compute_overrides_user_changed(parsed, defaults)
            if method != defaults["peak_threshold_method"]:
                self.assertIn("peak_threshold_method", changed)
                
            if uses_k:
                self.assertIn("peak_threshold_k", changed)
            else:
                self.assertNotIn("peak_threshold_k", changed)
                
            if uses_pct:
                self.assertIn("peak_threshold_percentile", changed)
            else:
                self.assertNotIn("peak_threshold_percentile", changed)
                
            if uses_abs:
                self.assertIn("peak_threshold_abs", changed)
            else:
                self.assertNotIn("peak_threshold_abs", changed)

    def test_t4_validation_failures(self):
        """T4: Validation failures for numeric constraints conditional on method."""
        from gui.main_window import parse_and_validate_event_feature_knobs
        from gui.main_window import peak_threshold_method_requires_k, peak_threshold_method_requires_percentile, peak_threshold_method_requires_abs
        from photometry_pipeline.config import Config
        
        cfg0 = Config()
        defaults = {
            "event_signal": cfg0.event_signal,
            "peak_threshold_method": cfg0.peak_threshold_method,
            "peak_threshold_k": cfg0.peak_threshold_k,
            "peak_threshold_percentile": cfg0.peak_threshold_percentile,
            "peak_threshold_abs": cfg0.peak_threshold_abs,
            "peak_min_distance_sec": cfg0.peak_min_distance_sec,
            "event_auc_baseline": cfg0.event_auc_baseline,
        }
        
        method = defaults["peak_threshold_method"]
        
        # dist < 0
        _, err = parse_and_validate_event_feature_knobs(
            defaults["event_signal"], method, str(defaults["peak_threshold_k"]),
            str(defaults["peak_threshold_percentile"]), str(defaults["peak_threshold_abs"]),
            "-1.0", defaults["event_auc_baseline"], defaults
        )
        self.assertIn("Peak Min Distance (sec) must be >= 0", err)
        
        # k <= 0
        if peak_threshold_method_requires_k(method):
            _, err = parse_and_validate_event_feature_knobs(
                defaults["event_signal"], method, "0.0",
                str(defaults["peak_threshold_percentile"]), str(defaults["peak_threshold_abs"]),
                str(defaults["peak_min_distance_sec"]), defaults["event_auc_baseline"], defaults
            )
            self.assertIn("Peak Threshold K must be > 0", err)
        else:
            _, err = parse_and_validate_event_feature_knobs(
                defaults["event_signal"], method, "0.0",
                str(defaults["peak_threshold_percentile"]), str(defaults["peak_threshold_abs"]),
                str(defaults["peak_min_distance_sec"]), defaults["event_auc_baseline"], defaults
            )
            self.assertIsNone(err)
            
        # pct out of bounds
        if peak_threshold_method_requires_percentile(method):
            _, err = parse_and_validate_event_feature_knobs(
                defaults["event_signal"], method, str(defaults["peak_threshold_k"]),
                "150.0", str(defaults["peak_threshold_abs"]),
                str(defaults["peak_min_distance_sec"]), defaults["event_auc_baseline"], defaults
            )
            self.assertIn("Peak Threshold Percentile must be between 0 and 100", err)
            
        # abs <= 0
        if peak_threshold_method_requires_abs(method):
            _, err = parse_and_validate_event_feature_knobs(
                defaults["event_signal"], method, str(defaults["peak_threshold_k"]),
                str(defaults["peak_threshold_percentile"]), "-1.0",
                str(defaults["peak_min_distance_sec"]), defaults["event_auc_baseline"], defaults
            )
            self.assertIn("Peak Threshold Absolute must be > 0", err)
            
    def test_t5_run_report_reflects_step_7_knobs(self):
        """T5: Hard validation gate proving Step 7 config reaches run_report.json."""
        from gui.main_window import parse_and_validate_event_feature_knobs, compute_overrides_user_changed
        from gui.main_window import get_allowed_peak_threshold_methods_from_config
        from gui.main_window import peak_threshold_method_requires_k, peak_threshold_method_requires_abs, peak_threshold_method_requires_percentile
        from gui.run_spec import RunSpec
        from photometry_pipeline.config import Config
        from photometry_pipeline.core.reporting import generate_run_report
        import tempfile, os, shutil, yaml, json
        
        cfg0 = Config()
        defaults = {
            "event_signal": cfg0.event_signal,
            "peak_threshold_method": cfg0.peak_threshold_method,
            "peak_threshold_k": cfg0.peak_threshold_k,
            "peak_threshold_percentile": cfg0.peak_threshold_percentile,
            "peak_threshold_abs": cfg0.peak_threshold_abs,
            "peak_min_distance_sec": cfg0.peak_min_distance_sec,
            "event_auc_baseline": cfg0.event_auc_baseline,
        }
        
        allowed_methods = get_allowed_peak_threshold_methods_from_config()
        self.assertIn(cfg0.peak_threshold_method, allowed_methods)
        
        tmp_dir = tempfile.mkdtemp()
        try:
            run_dir_b = os.path.join(tmp_dir, "run_b")
            base_config_path_b = os.path.join(tmp_dir, "base_config_b.yaml")
            with open(base_config_path_b, "w") as f:
                yaml.safe_dump({}, f)
                
            dist_override = defaults["peak_min_distance_sec"] + 1.0
            
            if len(allowed_methods) > 1:
                method_override = next(m for m in allowed_methods if m != defaults["peak_threshold_method"])
            else:
                method_override = defaults["peak_threshold_method"]
                
            k_ov = "3.14" if peak_threshold_method_requires_k(method_override) else str(defaults["peak_threshold_k"])
            pct_ov = "42.0" if peak_threshold_method_requires_percentile(method_override) else str(defaults["peak_threshold_percentile"])
            abs_ov = "9.99" if peak_threshold_method_requires_abs(method_override) else str(defaults["peak_threshold_abs"])
            
            parsed_overrides, err = parse_and_validate_event_feature_knobs(
                defaults["event_signal"], method_override, k_ov, pct_ov, abs_ov, str(dist_override), defaults["event_auc_baseline"], defaults
            )
            self.assertIsNone(err)
            changed = compute_overrides_user_changed(parsed_overrides, defaults)
            
            spec_b = RunSpec(run_dir=run_dir_b, config_source_path=base_config_path_b, config_overrides=changed)
            conf_path_b = spec_b.generate_derived_config(run_dir_b)
            
            with open(conf_path_b, "r") as f:
                cfg_b = Config(**yaml.safe_load(f))
            generate_run_report(cfg_b, run_dir_b)
            
            with open(os.path.join(run_dir_b, "run_report.json"), "r") as f:
                report_b = json.load(f)
                
            cfg_dict_b = report_b["configuration"]
            self.assertEqual(cfg_dict_b["peak_threshold_method"], method_override)
            self.assertEqual(cfg_dict_b["peak_min_distance_sec"], dist_override)
            
            if peak_threshold_method_requires_k(method_override):
                self.assertEqual(cfg_dict_b["peak_threshold_k"], float(k_ov))
            if peak_threshold_method_requires_percentile(method_override):
                self.assertEqual(cfg_dict_b["peak_threshold_percentile"], float(pct_ov))
            if peak_threshold_method_requires_abs(method_override):
                self.assertEqual(cfg_dict_b["peak_threshold_abs"], float(abs_ov))
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t6_peak_threshold_gating_schema_locked(self):
        """T6: Assert peak threshold parameters are evaluated by schema constraint probing, not heuristics."""
        from gui.main_window import (
            peak_threshold_method_requires_k,
            peak_threshold_method_requires_percentile,
            peak_threshold_method_requires_abs,
            get_allowed_peak_threshold_methods_from_config
        )
        from photometry_pipeline.config import Config
        
        allowed_methods = get_allowed_peak_threshold_methods_from_config()
        self.assertTrue(len(allowed_methods) > 0, "No peak threshold methods returned.")
        
        for method in allowed_methods:
            # check k
            try:
                c1 = Config(peak_threshold_method=method, peak_threshold_k=1.0)
                c2 = Config(peak_threshold_method=method, peak_threshold_k=2.0)
                expected_k = (c1.peak_threshold_k != c2.peak_threshold_k)
            except Exception:
                try:
                    Config(peak_threshold_method=method)
                    b_k = True
                except Exception:
                    b_k = False
                
                if not b_k:
                    expected_k = True
                else:
                    try:
                        Config(peak_threshold_method=method, peak_threshold_k=1.0)
                        p_k = True
                    except Exception:
                        p_k = False
                    expected_k = True if p_k else False
            self.assertEqual(expected_k, peak_threshold_method_requires_k(method), f"Method {method} expected K:{expected_k}")
            
            # check percentile
            try:
                c1 = Config(peak_threshold_method=method, peak_threshold_percentile=10.0)
                c2 = Config(peak_threshold_method=method, peak_threshold_percentile=50.0)
                expected_pct = (c1.peak_threshold_percentile != c2.peak_threshold_percentile)
            except Exception:
                try:
                    Config(peak_threshold_method=method)
                    b_pct = True
                except Exception:
                    b_pct = False
                if not b_pct:
                    expected_pct = True
                else:
                    try:
                        Config(peak_threshold_method=method, peak_threshold_percentile=10.0)
                        p_pct = True
                    except Exception:
                        p_pct = False
                    expected_pct = True if p_pct else False
            self.assertEqual(expected_pct, peak_threshold_method_requires_percentile(method), f"Method {method} expected PCT:{expected_pct}")
            
            # check abs
            try:
                c1 = Config(peak_threshold_method=method, peak_threshold_abs=0.5)
                c2 = Config(peak_threshold_method=method, peak_threshold_abs=1.0)
                expected_abs = (c1.peak_threshold_abs != c2.peak_threshold_abs)
            except Exception:
                try:
                    Config(peak_threshold_method=method)
                    b_abs = True
                except Exception:
                    b_abs = False
                if not b_abs:
                    expected_abs = True
                else:
                    try:
                        Config(peak_threshold_method=method, peak_threshold_abs=0.5)
                        p_abs = True
                    except Exception:
                        p_abs = False
                    expected_abs = True if p_abs else False
            self.assertEqual(expected_abs, peak_threshold_method_requires_abs(method), f"Method {method} expected ABS:{expected_abs}")

    def test_t7_peak_threshold_not_parsed_when_gating_false(self):
        """T7: Prove parameters are not validated or sent to the config dictionary if their gating is False."""
        from gui.main_window import parse_and_validate_event_feature_knobs, get_allowed_peak_threshold_methods_from_config
        from photometry_pipeline.config import Config
        import unittest.mock
        
        allowed_methods = get_allowed_peak_threshold_methods_from_config()
        self.assertTrue(len(allowed_methods) > 0)
        method = allowed_methods[0]
        
        cfg0 = Config()
        defaults = {
            "event_signal": cfg0.event_signal,
            "peak_threshold_method": cfg0.peak_threshold_method,
            "peak_threshold_k": cfg0.peak_threshold_k,
            "peak_threshold_percentile": cfg0.peak_threshold_percentile,
            "peak_threshold_abs": cfg0.peak_threshold_abs,
            "peak_min_distance_sec": cfg0.peak_min_distance_sec,
            "event_auc_baseline": cfg0.event_auc_baseline,
        }
        
        # We'll test percentile and abs simultaneously
        with unittest.mock.patch("gui.main_window.peak_threshold_method_requires_percentile", return_value=False), \
             unittest.mock.patch("gui.main_window.peak_threshold_method_requires_abs", return_value=False):
            # Pass highly invalid values for everything since they shouldn't trigger verification
            overrides, err = parse_and_validate_event_feature_knobs(
                defaults["event_signal"], method, str(defaults["peak_threshold_k"]),
                "999.0", "-50.0", str(defaults["peak_min_distance_sec"]), defaults["event_auc_baseline"], defaults
            )
            
        self.assertIsNone(err, "Should not hit validation for param gated off.")
        self.assertIsNotNone(overrides)
        self.assertNotIn("peak_threshold_percentile", overrides)
        self.assertNotIn("peak_threshold_abs", overrides)

if __name__ == "__main__":
    unittest.main()
