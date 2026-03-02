import os
import json
import pytest
from gui.process_runner import _read_final_status, PipelineRunner, RunnerState
from PySide6.QtCore import QCoreApplication

# Ensure a QCoreApplication exists for Signal testing if needed
if not QCoreApplication.instance():
    app = QCoreApplication([])

def test_read_final_status_sentinels(tmp_path):
    """PART 4.1: Prove _read_final_status() classifications."""
    status_path = tmp_path / "status.json"
    
    # 1. Missing file
    code, _ = _read_final_status(str(status_path))
    assert code == "MISSING_FILE"
    
    # 2. Malformed JSON
    status_path.write_text("{invalid")
    code, _ = _read_final_status(str(status_path))
    assert code == "MALFORMED_STATUS"
    
    # 3. Phase != final (even if schema is wrong, Phase check comes first per Requirement B)
    status_path.write_text(json.dumps({"phase": "running", "schema_version": 999}))
    code, _ = _read_final_status(str(status_path))
    assert code == "NOT_FINAL"
    
    # 4. Schema mismatch
    status_path.write_text(json.dumps({"phase": "final", "schema_version": 2}))
    code, _ = _read_final_status(str(status_path))
    assert code == "SCHEMA_MISMATCH"
    
    # 5. Missing status
    status_path.write_text(json.dumps({"phase": "final", "schema_version": 1}))
    code, _ = _read_final_status(str(status_path))
    assert code == "MISSING_STATUS"
    
    # 6. Bad status
    status_path.write_text(json.dumps({"phase": "final", "schema_version": 1, "status": "unknown"}))
    code, _ = _read_final_status(str(status_path))
    assert code == "BAD_STATUS"
    
    # 7. Valid Success
    status_path.write_text(json.dumps({"phase": "final", "schema_version": 1, "status": "success"}))
    code, _ = _read_final_status(str(status_path))
    assert code == "success"

def test_fail_closed_resolution(tmp_path):
    """PART 4.2: Prove exit_code==0 without status.json is FAIL_CLOSED (No Fail-Open)."""
    runner = PipelineRunner()
    # No run_dir set, so status.json is missing
    runner.set_run_dir(str(tmp_path / "nonexistent"))
    
    # Resolve with exit_code 0
    state = runner._resolve_final_state("MISSING_FILE", 0)
    assert state == RunnerState.FAIL_CLOSED
    assert runner.fail_closed_code == "MISSING_FILE"
    assert "was never created or is unreadable" in runner.fail_closed_detail


