import os
from types import SimpleNamespace

import pytest
import yaml

from gui.run_spec import FORMAT_CHOICES, RunSpec
from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from photometry_pipeline.io.adapters import load_chunk
from tools.run_full_pipeline_deliverables import validate_inputs


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _custom_csv() -> str:
    return (
        "time_sec,Region0_iso,Region0_sig,Region1_iso,Region1_sig\n"
        "0.0,1.0,2.0,1.2,2.2\n"
        "0.1,1.1,2.1,1.3,2.3\n"
        "0.2,1.2,2.2,1.4,2.4\n"
        "0.3,1.3,2.3,1.5,2.5\n"
        "0.4,1.4,2.4,1.6,2.6\n"
        "0.5,1.5,2.5,1.7,2.7\n"
        "0.6,1.6,2.6,1.8,2.8\n"
        "0.7,1.7,2.7,1.9,2.9\n"
        "0.8,1.8,2.8,2.0,3.0\n"
        "0.9,1.9,2.9,2.1,3.1\n"
    )


def _base_cfg() -> Config:
    return Config(
        target_fs_hz=10.0,
        chunk_duration_sec=1.0,
        allow_partial_final_chunk=False,
        custom_tabular_time_col="time_sec",
        custom_tabular_uv_suffix="_iso",
        custom_tabular_sig_suffix="_sig",
    )


def test_custom_tabular_config_validation_rejects_identical_suffixes(tmp_path):
    cfg_path = tmp_path / "bad_custom_tabular.yaml"
    cfg_path.write_text(
        "custom_tabular_uv_suffix: _x\ncustom_tabular_sig_suffix: _x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be different"):
        Config.from_yaml(str(cfg_path))


def test_custom_tabular_load_chunk_valid_contract(tmp_path):
    csv_path = tmp_path / "session_01.csv"
    csv_path.write_text(_custom_csv(), encoding="utf-8")
    cfg = _base_cfg()

    chunk = load_chunk(str(csv_path), "custom_tabular", cfg, chunk_id=0)
    assert chunk.format == "custom_tabular"
    assert chunk.channel_names == ["Region0", "Region1"]
    assert chunk.time_sec.shape[0] == 10
    assert chunk.uv_raw.shape == (10, 2)
    assert chunk.sig_raw.shape == (10, 2)
    contract = chunk.metadata.get("custom_tabular_contract", {})
    assert contract.get("session_model") == "one_csv_per_session"
    assert contract.get("time_col") == "time_sec"
    assert contract.get("uv_suffix") == "_iso"
    assert contract.get("sig_suffix") == "_sig"


def test_custom_tabular_rejects_unpaired_roi_columns(tmp_path):
    csv_path = tmp_path / "bad_pairs.csv"
    csv_path.write_text(
        (
            "time_sec,Region0_iso,Region0_sig,Region1_iso\n"
            "0.0,1.0,2.0,1.2\n"
            "0.1,1.1,2.1,1.3\n"
        ),
        encoding="utf-8",
    )
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="unmatched ROI pairs"):
        load_chunk(str(csv_path), "custom_tabular", cfg, chunk_id=0)


def test_custom_tabular_discovery_requires_explicit_format_and_keeps_session_structure(tmp_path):
    in_dir = tmp_path / "custom_data"
    _write_text(str(in_dir / "session_10.csv"), _custom_csv())
    _write_text(str(in_dir / "session_2.csv"), _custom_csv())
    cfg = _base_cfg()

    result = discover_inputs(str(in_dir), cfg, force_format="custom_tabular")
    assert result["resolved_format"] == "CUSTOM_TABULAR"
    session_ids = [s["session_id"] for s in result["sessions"]]
    assert session_ids == ["session_2", "session_10"]
    assert [r["roi_id"] for r in result["rois"]] == ["Region0", "Region1"]

    with pytest.raises(ValueError, match="Could not automatically detect format"):
        discover_inputs(str(in_dir), cfg, force_format="auto")


def test_run_spec_accepts_custom_tabular_format_and_emits_cli_flag(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(yaml.safe_dump({}), encoding="utf-8")
    run_dir = tmp_path / "run"

    assert "custom_tabular" in FORMAT_CHOICES
    spec = RunSpec(
        input_dir=str(tmp_path),
        run_dir=str(run_dir),
        format="custom_tabular",
        config_source_path=str(base_cfg),
    )
    spec.generate_derived_config(str(run_dir))
    argv = spec.build_runner_argv()
    assert "--format" in argv
    assert argv[argv.index("--format") + 1] == "custom_tabular"


def test_wrapper_validate_inputs_accepts_custom_tabular(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({}), encoding="utf-8")
    args = SimpleNamespace(
        input=str(input_dir),
        config=str(cfg_path),
        format="custom_tabular",
        mode="both",
        run_type="full",
        sessions_per_hour=None,
        session_duration_s=None,
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
    )
    validate_inputs(args)
