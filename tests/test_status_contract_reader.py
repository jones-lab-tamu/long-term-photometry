import os
import json
import pytest
from gui.process_runner import _read_final_status

def test_read_final_status_missing(tmp_path):
    path = str(tmp_path / "missing.json")
    status, errors = _read_final_status(path)
    assert status == "MISSING_FILE"
    assert errors == []

def test_read_final_status_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    status, errors = _read_final_status(str(path))
    assert status == "MALFORMED_STATUS"

def test_read_final_status_not_final(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": "running", "schema_version": 1}))
    status, errors = _read_final_status(str(path))
    assert status == "NOT_FINAL"

def test_read_final_status_missing_schema(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": "final"}))
    status, errors = _read_final_status(str(path))
    assert status == "SCHEMA_MISMATCH"

def test_read_final_status_bad_schema(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": "final", "schema_version": 2}))
    status, errors = _read_final_status(str(path))
    assert status == "SCHEMA_MISMATCH"

def test_read_final_status_missing_status_field(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": "final", "schema_version": 1}))
    status, errors = _read_final_status(str(path))
    assert status == "MISSING_STATUS"

def test_read_final_status_bad_status_value(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": "final", "schema_version": 1, "status": "foo"}))
    status, errors = _read_final_status(str(path))
    assert status == "BAD_STATUS"

def test_read_final_status_success(tmp_path):
    path = tmp_path / "status.json"
    errors_list = ["err1", "err2"]
    path.write_text(json.dumps({
        "phase": "final", 
        "schema_version": 1, 
        "status": "success",
        "errors": errors_list
    }))
    status, errors = _read_final_status(str(path))
    assert status == "success"
    assert errors == errors_list
