from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from tools.run_full_pipeline_deliverables import validate_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "custom_tabular"
EXAMPLE_CFG = EXAMPLES_DIR / "custom_tabular_example_config.yaml"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_cfg(path: Path) -> None:
    payload = {
        "chunk_duration_sec": 1.0,
        "target_fs_hz": 10.0,
        "allow_partial_final_chunk": False,
        "custom_tabular_time_col": "time_sec",
        "custom_tabular_uv_suffix": "_iso",
        "custom_tabular_sig_suffix": "_sig",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _args(input_dir: str, cfg_path: str):
    return SimpleNamespace(
        input=input_dir,
        config=cfg_path,
        format="custom_tabular",
        mode="both",
        run_type="full",
        sessions_per_hour=2,
        session_duration_s=None,
        timeline_anchor_mode="civil",
        fixed_daily_anchor_clock=None,
        validate_only=True,
    )


def test_validate_only_custom_tabular_examples_pass():
    args = _args(str(EXAMPLES_DIR), str(EXAMPLE_CFG))
    validate_inputs(args)


def test_validate_only_custom_tabular_missing_time_column_fails(tmp_path):
    in_dir = tmp_path / "input"
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg)
    _write_text(
        in_dir / "session_001.csv",
        "bad_time,ROI_1_iso,ROI_1_sig\n0.0,1.0,2.0\n0.1,1.1,2.1\n",
    )
    args = _args(str(in_dir), str(cfg))
    with pytest.raises(RuntimeError) as excinfo:
        validate_inputs(args)
    msg = str(excinfo.value).lower()
    assert "custom_tabular" in msg
    assert "session_001.csv" in msg
    assert "time_sec" in msg
    assert "missing required time column" in msg or "required time column" in msg


def test_validate_only_custom_tabular_unpaired_roi_columns_fails(tmp_path):
    in_dir = tmp_path / "input"
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg)
    _write_text(
        in_dir / "session_002.csv",
        "time_sec,ROI_1_sig\n0.0,2.0\n0.1,2.1\n",
    )
    args = _args(str(in_dir), str(cfg))
    with pytest.raises(RuntimeError) as excinfo:
        validate_inputs(args)
    msg = str(excinfo.value).lower()
    assert "session_002.csv" in msg
    assert "unmatched roi pairs" in msg or "missing required paired" in msg
    assert "_iso" in msg or "isosbestic" in msg
    assert "_sig" in msg or "signal" in msg


def test_validate_only_custom_tabular_nonnumeric_or_nan_fails(tmp_path):
    in_dir = tmp_path / "input"
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg)
    _write_text(
        in_dir / "session_003.csv",
        "time_sec,ROI_1_iso,ROI_1_sig\n0.0,1.0,2.0\n0.1,NaN,2.1\n",
    )
    args = _args(str(in_dir), str(cfg))
    with pytest.raises(RuntimeError) as excinfo:
        validate_inputs(args)
    msg = str(excinfo.value).lower()
    assert "session_003.csv" in msg
    assert "non-numeric/nan values" in msg or "contains nan" in msg


def test_validate_only_custom_tabular_nonmonotonic_time_fails(tmp_path):
    in_dir = tmp_path / "input"
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg)
    _write_text(
        in_dir / "session_004.csv",
        "time_sec,ROI_1_iso,ROI_1_sig\n0.0,1.0,2.0\n0.0,1.1,2.1\n",
    )
    args = _args(str(in_dir), str(cfg))
    with pytest.raises(RuntimeError) as excinfo:
        validate_inputs(args)
    msg = str(excinfo.value).lower()
    assert "session_004.csv" in msg
    assert "strictly increasing" in msg
    assert "time_sec" in msg or "timestamps" in msg


def test_validate_only_custom_tabular_all_invalid_reports_file_reasons(tmp_path):
    in_dir = tmp_path / "input"
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg)
    _write_text(
        in_dir / "session_bad_a.csv",
        "bad_time,ROI_1_iso,ROI_1_sig\n0.0,1.0,2.0\n0.1,1.1,2.1\n",
    )
    _write_text(
        in_dir / "session_bad_b.csv",
        "time_sec,ROI_1_sig\n0.0,2.0\n0.1,2.1\n",
    )
    args = _args(str(in_dir), str(cfg))
    with pytest.raises(RuntimeError) as excinfo:
        validate_inputs(args)
    msg = str(excinfo.value).lower()
    assert "no valid custom_tabular files could be parsed" in msg
    assert "first file-level errors" in msg
    assert "session_bad_a.csv" in msg
    assert "session_bad_b.csv" in msg


def test_discovery_custom_tabular_all_invalid_surfaces_file_level_reasons(tmp_path):
    in_dir = tmp_path / "input"
    cfg_path = tmp_path / "cfg.yaml"
    _write_cfg(cfg_path)
    _write_text(
        in_dir / "bad_1.csv",
        "bad_time,ROI_1_iso,ROI_1_sig\n0.0,1.0,2.0\n0.1,1.1,2.1\n",
    )
    _write_text(
        in_dir / "bad_2.csv",
        "time_sec,ROI_1_sig\n0.0,2.0\n0.1,2.1\n",
    )
    cfg = Config.from_yaml(str(cfg_path))
    with pytest.raises(RuntimeError) as excinfo:
        discover_inputs(str(in_dir), cfg, force_format="custom_tabular")
    msg = str(excinfo.value).lower()
    assert "no valid custom_tabular files could be parsed" in msg
    assert "first file-level errors" in msg
    assert "bad_1.csv" in msg
    assert "bad_2.csv" in msg
