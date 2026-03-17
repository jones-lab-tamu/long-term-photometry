"""
Focused robustness tests for status.json atomic write retries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tools import run_full_pipeline_deliverables as wrapper


def test_status_writer_retries_replace_on_transient_lock(monkeypatch, tmp_path):
    status_path = tmp_path / "status.json"
    payload = {"phase": "running", "status": "running", "schema_version": 1}

    real_replace = wrapper.os.replace
    replace_calls = []
    sleep_calls = []

    def flaky_replace(src, dst):
        replace_calls.append((src, dst))
        if len(replace_calls) < 3:
            raise PermissionError(13, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(wrapper.os, "replace", flaky_replace)
    monkeypatch.setattr(wrapper.time, "sleep", lambda sec: sleep_calls.append(sec))

    wrapper._write_status_json(str(status_path), payload)

    assert status_path.exists()
    with open(status_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert out["status"] == "running"
    assert len(replace_calls) == 3
    assert len(sleep_calls) == 2


def test_atomic_writer_raises_after_retry_budget_exhausted(monkeypatch, tmp_path):
    status_path = tmp_path / "status.json"
    payload = {"phase": "final", "status": "success", "schema_version": 1}
    sleep_calls = []

    def always_fail_replace(_src, _dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(wrapper.os, "replace", always_fail_replace)
    monkeypatch.setattr(wrapper.time, "sleep", lambda sec: sleep_calls.append(sec))

    with pytest.raises(PermissionError):
        wrapper._atomic_write_json(
            str(status_path),
            payload,
            replace_retries=2,
            replace_retry_delay_sec=0.01
        )

    assert len(sleep_calls) == 2
