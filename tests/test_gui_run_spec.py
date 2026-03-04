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
        # format=auto must OMIT --format entirely (Step 4 rule)
        self.assertNotIn("--format", argv)

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
        # format=auto: omitted
        self.assertNotIn("--format", argv)
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

    # ----------------------------------------------------------------
    # Step 4: format=auto omits --format, format=rwd includes it
    # ----------------------------------------------------------------
    def test_argv_format_conditional(self):
        """--format is omitted for auto, included for explicit formats."""
        run_dir_auto = os.path.join(self.tmp_dir, "run_fmt_auto")
        spec_auto = RunSpec(
            input_dir="/data/in",
            run_dir=run_dir_auto,
            format="auto",
            config_source_path=self.config_path,
        )
        spec_auto.generate_derived_config(run_dir_auto)
        self.assertNotIn("--format", spec_auto.build_runner_argv())

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
            self.assertNotIn("--format", called_argv)
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
                     "--representative-session-index", "--mode"):
            self.assertIn(flag, src,
                          f"Runner missing expected flag: {flag}")
        # Intent-only flags MUST NOT exist in runner
        for flag in ("'--recursive'", "'--glob'"):
            # Check for argparse definitions (quoted flag names)
            self.assertNotIn(f"add_argument({flag}", src,
                             f"Runner has unexpected flag: {flag}")


if __name__ == "__main__":
    unittest.main()
