"""
Tests for Fix B1v5: Tool-level In-Place Overwrite Cleanup.
Verifies that --overwrite preserves GUI-owned files and log handles.
"""

import os
import shutil
import tempfile
import sys
from pathlib import Path

# Add repo root to sys.path
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tools.run_full_pipeline_deliverables import _cleanup_run_outputs_in_place

def test_overwrite_preserves_gui_owned_files():
    with tempfile.TemporaryDirectory() as run_dir:
        # 1. Create GUI-owned files
        gui_files = [
            "config_effective.yaml",
            "gui_run_spec.json",
            "stdout.txt",
            "stderr.txt"
        ]
        for f in gui_files:
            with open(os.path.join(run_dir, f), "w") as fh:
                fh.write("gui content")

        # 2. Create tool-generated artifacts
        tool_artifacts = [
            "status.json",
            "run_report.json",
            "events.ndjson",
            "MANIFEST.json"
        ]
        for f in tool_artifacts:
            with open(os.path.join(run_dir, f), "w") as fh:
                fh.write("tool content")

        # 3. Create tool-generated directory
        analysis_dir = os.path.join(run_dir, "_analysis")
        os.makedirs(analysis_dir)
        with open(os.path.join(analysis_dir, "plot.png"), "w") as fh:
            fh.write("binary data")

        # 4. Run cleanup
        _cleanup_run_outputs_in_place(run_dir, emitter=None)

        # 5. Assertions
        for f in gui_files:
            assert os.path.exists(os.path.join(run_dir, f)), f"GUI file {f} should be preserved"
        
        for f in tool_artifacts:
            assert not os.path.exists(os.path.join(run_dir, f)), f"Tool artifact {f} should be deleted"
            
        assert not os.path.exists(analysis_dir), "_analysis directory should be deleted"

def test_overwrite_does_not_delete_root_dir():
    with tempfile.TemporaryDirectory() as run_dir:
        _cleanup_run_outputs_in_place(run_dir, emitter=None)
        assert os.path.exists(run_dir), "Root run_dir should NOT be deleted"

def test_overwrite_cleanup_is_idempotent():
    with tempfile.TemporaryDirectory() as run_dir:
        cfg_path = os.path.join(run_dir, "config_effective.yaml")
        with open(cfg_path, "w") as f:
            f.write("preserve me")
            
        # Run twice
        _cleanup_run_outputs_in_place(run_dir, emitter=None)
        _cleanup_run_outputs_in_place(run_dir, emitter=None)
        
        assert os.path.exists(run_dir)
        assert os.path.exists(cfg_path)

def test_overwrite_handles_locked_files_gracefully(capsys):
    """
    On Windows, if a file is open, os.remove might raise PermissionError.
    _cleanup_run_outputs_in_place should catch it and warn instead of crashing.
    """
    with tempfile.TemporaryDirectory() as run_dir:
        status_path = os.path.join(run_dir, "status.json")
        with open(status_path, "w") as f:
            f.write("locked")
        
        # Open the file to simulate a lock (if the OS supports it)
        with open(status_path, "r") as lock_fh:
            # On Windows, this might prevent deletion. 
            # On Linux, deletion usually works even if open.
            _cleanup_run_outputs_in_place(run_dir, emitter=None)
        
        # If it failed to delete, it shouldn't have crashed.
        # We don't assert deletion here because behavior varies by OS,
        # but we assert it survived.
        assert os.path.exists(run_dir)
