import sys

import pytest

from tools import run_cache_correction_retune as cli


def test_cli_parses_false_override_as_false(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_cache_correction_retune(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "run_cache_correction_retune", _fake_run_cache_correction_retune)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cache_correction_retune.py",
            "--run-dir",
            str(tmp_path),
            "--roi",
            "Region0",
            "--set",
            "baseline_subtract_before_fit=false",
        ],
    )

    rc = cli.main()
    assert rc == 0
    assert captured["overrides"]["baseline_subtract_before_fit"] is False


@pytest.mark.parametrize("raw_value", ["maybe", "truthy", "FALSEY"])
def test_cli_invalid_bool_override_fails_cleanly(monkeypatch, raw_value, capsys):
    def _should_not_run(**kwargs):
        raise AssertionError("run_cache_correction_retune should not be called for invalid bool input")

    monkeypatch.setattr(cli, "run_cache_correction_retune", _should_not_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cache_correction_retune.py",
            "--run-dir",
            "unused",
            "--roi",
            "Region0",
            "--set",
            f"baseline_subtract_before_fit={raw_value}",
        ],
    )

    rc = cli.main()
    captured = capsys.readouterr()
    assert rc == 1
    assert "Invalid boolean override value" in captured.err
