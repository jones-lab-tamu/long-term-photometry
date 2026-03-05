"""
RunSpec: GUI-side "user intent only" record for pipeline runs.

Maps GUI controls to runner CLI flags and config YAML overrides.
The runner (tools/run_full_pipeline_deliverables.py) remains the sole
execution backend and authoritative audit trail.

Design choices:
  - Uses runner --out <run_dir> so GUI controls the exact run directory.
    Runner uses --out as explicit run_dir (see resolve_output_mode in
    tools/run_full_pipeline_deliverables.py).
  - GUI writes exactly three files into run_dir:
    config_effective.yaml, gui_run_spec.json, command_invoked.txt.
  - GUI does NOT write events.ndjson (audit log) and does NOT emit
    EventEmitter events. Only the runner/pipeline may do so.
  - RunSpec fields correspond 1:1 with GUI widgets that actually exist.
    Fields for widgets not yet in the GUI are intentionally absent.
    Add them to RunSpec when the corresponding widgets are added.

This module has NO PySide6 dependency so it can be unit-tested standalone.
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

FORMAT_CHOICES = ("auto", "rwd", "npm")


def _stable_yaml_dump(data: dict) -> str:
    """Serialize dict to YAML with deterministic key ordering.

    Uses yaml.safe_dump (not yaml.dump) for safety and reproducibility.
    """
    return yaml.safe_dump(
        data, default_flow_style=False, sort_keys=True, allow_unicode=True
    )


@dataclass
class RunSpec:
    """
    GUI-side user intent record. Three sections:

    (A) Runner invocation fields -- only fields with real GUI widgets.
    (B) Config source + overrides.
    (C) Provenance / bookkeeping.

    Fields without corresponding GUI widgets are NOT included.
    When a widget is added to the GUI, add the corresponding field here.
    """

    # --- (A) Runner invocation fields ---
    # Only fields with actual widgets in main_window.py.
    input_dir: str = ""
    run_dir: str = ""               # Explicit run directory (--out)
    format: str = "auto"            # _format_combo widget
    validate_only: bool = False     # Set by _on_validate path
    sessions_per_hour: Optional[int] = None    # _sph_edit widget
    session_duration_s: Optional[float] = None # _duration_edit widget
    smooth_window_s: float = 1.0    # _smooth_spin widget

    # --- (B) Config source + overrides ---
    config_source_path: str = ""
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # --- (C) Provenance ---
    gui_version: str = "1.0.0"
    timestamp_local: str = ""
    user_note: Optional[str] = None

    # --- (D) Intent fields (UI only, NOT config knobs) ---
    # Written into gui_run_spec.json for downstream consumption.
    # MUST NOT appear in config_effective.yaml or build_runner_argv().
    representative_session_index: Optional[int] = None
    include_roi_ids: Optional[List[str]] = None
    exclude_roi_ids: Optional[List[str]] = None
    mode: Optional[str] = None
    traces_only: bool = False
    preview_first_n: Optional[int] = None

    # --- Explicitness tracking ---
    # Records which RunSpec fields were explicitly set by the user
    # (vs defaulted by GUI). Built by MainWindow._build_run_spec.
    user_set_fields: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.include_roi_ids is not None and self.exclude_roi_ids is not None:
            raise ValueError("include_roi_ids and exclude_roi_ids are mutually exclusive")

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return asdict(self)

    def generate_derived_config(self, run_dir: str) -> str:
        """
        Generate config_effective.yaml in run_dir.

        Loads base YAML from config_source_path, applies config_overrides
        (after validating them against the knob registry), and writes
        with stable key ordering for deterministic output.

        Raises ValueError if config_overrides contain unknown or blocked keys.

        Returns the absolute path to the generated file.
        """
        from gui.knobs_registry import filter_config_overrides

        base = {}
        if self.config_source_path and os.path.isfile(self.config_source_path):
            with open(self.config_source_path, "r") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                base = loaded

        # Validate GUI-originated overrides against the knob registry.
        # Base YAML keys bypass this filter (they come from the user's
        # existing config file, not from GUI controls).
        if self.config_overrides:
            filtered = filter_config_overrides(self.config_overrides)
            for key, value in filtered.items():
                base[key] = value

        os.makedirs(run_dir, exist_ok=True)
        config_path = os.path.join(run_dir, "config_effective.yaml")
        with open(config_path, "w") as f:
            f.write(_stable_yaml_dump(base))

        return config_path

    def write_gui_run_spec(self, run_dir: str) -> str:
        """
        Write gui_run_spec.json into run_dir.

        This is the GUI's sole intent record. It MUST NOT include
        any values decided by runner or pipeline (e.g. discovered
        session counts, resolved representative IDs).

        Returns the absolute path to the written file.
        """
        os.makedirs(run_dir, exist_ok=True)
        spec_path = os.path.join(run_dir, "gui_run_spec.json")
        with open(spec_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return spec_path

    def write_command_invoked(self, run_dir: str, argv: list) -> str:
        """
        Write command_invoked.txt into run_dir.

        Contains the exact argv passed to subprocess, one argument per line.

        Returns the absolute path to the written file.
        """
        os.makedirs(run_dir, exist_ok=True)
        cmd_path = os.path.join(run_dir, "command_invoked.txt")
        with open(cmd_path, "w") as f:
            f.write("# Command invoked by GUI\n")
            f.write("# One argument per line\n")
            for arg in argv:
                f.write(arg + "\n")
        return cmd_path

    @staticmethod
    def validate_effective_config(config_path: str) -> None:
        """
        Validate the derived config by loading it through Config.from_yaml.

        Raises ValueError with a clear message if the config is invalid.
        config_path must point to an existing config_effective.yaml.
        """
        if not os.path.isfile(config_path):
            raise ValueError(
                f"Config file not found: {config_path}. "
                "Call generate_derived_config first."
            )
        from photometry_pipeline.config import Config
        try:
            Config.from_yaml(config_path)
        except Exception as e:
            raise ValueError(f"Derived config is invalid: {e}") from e

    def get_derived_config_preview(self) -> str:
        """
        Return the derived config YAML as a string (for UI preview),
        without writing to disk. Uses the same merge + filter logic
        as generate_derived_config.
        """
        from gui.knobs_registry import filter_config_overrides

        base = {}
        if self.config_source_path and os.path.isfile(self.config_source_path):
            with open(self.config_source_path, "r") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                base = loaded

        if self.config_overrides:
            filtered = filter_config_overrides(self.config_overrides)
            for key, value in filtered.items():
                base[key] = value

        return _stable_yaml_dump(base)

    def build_runner_argv(self) -> list:
        """
        Build the argv list for tools/run_full_pipeline_deliverables.py.

        Uses --out <self.run_dir> so the GUI controls the exact run
        directory. Runner uses --out as explicit run_dir (see
        resolve_output_mode in tools/run_full_pipeline_deliverables.py).

        The derived config_effective.yaml must already exist in run_dir.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        runner_script = os.path.join(
            repo_root, "tools", "run_full_pipeline_deliverables.py"
        )

        config_path = os.path.join(self.run_dir, "config_effective.yaml")

        argv = [
            sys.executable,
            runner_script,
            "--input", self.input_dir,
            "--out", self.run_dir,
            "--config", config_path,
            "--events", "auto",
            "--cancel-flag", "auto",
        ]

        argv.extend(["--format", self.format])

        if self.sessions_per_hour is not None:
            argv.extend(["--sessions-per-hour", str(self.sessions_per_hour)])

        if self.session_duration_s is not None:
            argv.extend(["--session-duration-s", str(self.session_duration_s)])

        argv.extend(["--smooth-window-s", str(self.smooth_window_s)])

        if self.traces_only:
            argv.append("--traces-only")
            
        if self.preview_first_n is not None:
            argv.extend(["--preview-first-n", str(self.preview_first_n)])
            
        if self.representative_session_index is not None:
            argv.extend(["--representative-session-index", str(self.representative_session_index)])

        if self.include_roi_ids:
            argv.extend(["--include-rois", ",".join(self.include_roi_ids)])
            
        if self.exclude_roi_ids:
            argv.extend(["--exclude-rois", ",".join(self.exclude_roi_ids)])
            
        if getattr(self, "mode", None) is not None:
            argv.extend(["--mode", self.mode])

        if self.validate_only:
            argv.append("--validate-only")

        return argv

    def run_discovery(self) -> dict:
        """
        Invoke the runner in --discover mode and return the parsed JSON.

        This uses the actual runner backend to discover sessions and ROIs
        without creating any run directories or events.ndjson files.
        """
        import subprocess
        import tempfile

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        runner_script = os.path.join(
            repo_root, "tools", "run_full_pipeline_deliverables.py"
        )

        preview_yaml = self.get_derived_config_preview()
        
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
            tmp.write(preview_yaml)
            tmp.flush()
            tmp_config_path = tmp.name

        try:
            argv = [
                sys.executable,
                runner_script,
                "--input", self.input_dir,
                "--config", tmp_config_path,
                "--discover"
            ]
            argv.extend(["--format", self.format])

            result = subprocess.run(
                argv, 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Discovery failed with code {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse discovery output JSON.\nError: {e}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}") from e
            
        finally:
            if os.path.exists(tmp_config_path):
                os.remove(tmp_config_path)
