from pathlib import Path

import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from photometry_pipeline.io.adapters import load_chunk


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "custom_tabular"
DOC_PATH = REPO_ROOT / "docs" / "custom_tabular_conversion_guide.md"
EXAMPLE_CFG_PATH = EXAMPLES_DIR / "custom_tabular_example_config.yaml"


def _strict_example_cfg() -> Config:
    return Config(
        target_fs_hz=10.0,
        chunk_duration_sec=1.0,
        allow_partial_final_chunk=False,
        custom_tabular_time_col="time_sec",
        custom_tabular_uv_suffix="_iso",
        custom_tabular_sig_suffix="_sig",
    )


def _expected_header_from_defaults() -> list[str]:
    cfg = Config()
    return [
        cfg.custom_tabular_time_col,
        f"ROI_1{cfg.custom_tabular_uv_suffix}",
        f"ROI_1{cfg.custom_tabular_sig_suffix}",
        f"ROI_2{cfg.custom_tabular_uv_suffix}",
        f"ROI_2{cfg.custom_tabular_sig_suffix}",
    ]


def _csv_header(path: Path) -> list[str]:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return [cell.strip() for cell in first_line.split(",")]


def test_custom_tabular_template_loads_with_strict_contract():
    cfg = _strict_example_cfg()
    template_path = EXAMPLES_DIR / "template_single_session.csv"
    chunk = load_chunk(str(template_path), "custom_tabular", cfg, chunk_id=0)

    assert chunk.format == "custom_tabular"
    assert chunk.channel_names == ["ROI_1", "ROI_2"]
    assert chunk.time_sec.shape[0] == 10
    assert chunk.uv_raw.shape == (10, 2)
    assert chunk.sig_raw.shape == (10, 2)
    assert chunk.metadata["custom_tabular_contract"]["time_col"] == "time_sec"
    assert chunk.metadata["custom_tabular_contract"]["uv_suffix"] == "_iso"
    assert chunk.metadata["custom_tabular_contract"]["sig_suffix"] == "_sig"


def test_custom_tabular_demo_files_discover_as_multiple_sessions():
    cfg = _strict_example_cfg()
    result = discover_inputs(str(EXAMPLES_DIR), cfg, force_format="custom_tabular")
    session_ids = [s["session_id"] for s in result["sessions"]]

    assert result["resolved_format"] == "CUSTOM_TABULAR"
    assert "demo_session_001" in session_ids
    assert "demo_session_002" in session_ids
    assert "template_single_session" in session_ids
    assert [r["roi_id"] for r in result["rois"]] == ["ROI_1", "ROI_2"]


def test_custom_tabular_examples_and_docs_match_default_contract():
    expected_header = _expected_header_from_defaults()
    csv_files = sorted(EXAMPLES_DIR.glob("*.csv"))

    assert csv_files, "expected at least one custom_tabular example CSV"
    for csv_path in csv_files:
        assert _csv_header(csv_path) == expected_header

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    doc_text_l = doc_text.lower()
    assert "custom_tabular_time_col" in doc_text
    assert "custom_tabular_uv_suffix" in doc_text
    assert "custom_tabular_sig_suffix" in doc_text
    assert "one csv file represents one session/chunk" in doc_text_l
    assert "conversion-through-csv support, not native vendor parser support" in doc_text_l


def test_custom_tabular_example_config_exists_and_matches_contract():
    assert EXAMPLE_CFG_PATH.exists()

    cfg = Config.from_yaml(str(EXAMPLE_CFG_PATH))
    assert cfg.chunk_duration_sec == 1.0
    assert cfg.target_fs_hz == 10.0
    assert cfg.allow_partial_final_chunk is False
    assert cfg.custom_tabular_time_col == "time_sec"
    assert cfg.custom_tabular_uv_suffix == "_iso"
    assert cfg.custom_tabular_sig_suffix == "_sig"

    cfg_raw = yaml.safe_load(EXAMPLE_CFG_PATH.read_text(encoding="utf-8")) or {}
    for key in (
        "chunk_duration_sec",
        "target_fs_hz",
        "allow_partial_final_chunk",
        "custom_tabular_time_col",
        "custom_tabular_uv_suffix",
        "custom_tabular_sig_suffix",
    ):
        assert key in cfg_raw
