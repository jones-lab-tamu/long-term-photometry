import re
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DATASET = REPO_ROOT / "examples" / "data" / "synthetic_photometry_basic"
BUNDLED_CONFIG = BUNDLED_DATASET / "tutorial_config.yaml"

DOC_PATHS = [
    REPO_ROOT / "docs" / "quickstart_gui_synthetic.md",
    REPO_ROOT / "docs" / "input_formats.md",
    REPO_ROOT / "docs" / "correction_and_dynamic_fit.md",
    REPO_ROOT / "docs" / "event_detection.md",
    REPO_ROOT / "docs" / "continuous_recordings.md",
    REPO_ROOT / "docs" / "batch_processing.md",
]

README_LINKS = [
    "docs/quickstart_gui_synthetic.md",
    "examples/data/synthetic_photometry_basic/",
    "docs/synthetic_dataset_generator_cli.md",
    "docs/synthetic_demo_datasets.md",
    "docs/input_formats.md",
    "docs/custom_tabular_conversion_guide.md",
    "docs/correction_and_dynamic_fit.md",
    "docs/event_detection.md",
    "docs/continuous_recordings.md",
    "docs/batch_processing.md",
    "examples/standalone_dynamic_fit_slope_constraint.py",
]


def test_bundled_synthetic_dataset_assets_exist():
    assert BUNDLED_DATASET.exists()
    assert BUNDLED_CONFIG.exists()
    assert (BUNDLED_DATASET / "generation_manifest.yaml").exists()
    session_csvs = sorted(BUNDLED_DATASET.glob("*/fluorescence.csv"))
    assert len(session_csvs) >= 1

    manifest = yaml.safe_load((BUNDLED_DATASET / "generation_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["format"] == "rwd"
    assert manifest["sessions_generated"] >= 1
    assert manifest["command"]["parsed_args"]["n_rois"] >= 1


def test_bundled_synthetic_tutorial_config_uses_conservative_event_defaults():
    cfg = yaml.safe_load(BUNDLED_CONFIG.read_text(encoding="utf-8"))

    assert cfg["peak_threshold_method"] == "mean_std"
    assert cfg["peak_threshold_k"] == 2.5
    assert cfg["peak_min_distance_sec"] == 1.0
    assert cfg["peak_min_prominence_k"] == 2.0
    assert cfg["peak_min_width_sec"] == 0.3


def test_key_documentation_files_exist_and_reference_bundled_dataset():
    for path in DOC_PATHS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert len(text.strip()) > 200

    quickstart = (REPO_ROOT / "docs" / "quickstart_gui_synthetic.md").read_text(encoding="utf-8")
    assert "examples/data/synthetic_photometry_basic" in quickstart
    assert "VALIDATE-ONLY: OK" in quickstart
    assert "not biological validation" in quickstart.lower()


def test_nonnegative_slope_constraint_docs_are_diagnostic_not_correction_fix():
    doc_text = "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)
    lower = doc_text.lower()

    assert "prevent negative slopes" not in lower
    assert "advanced diagnostic" in lower
    assert "not a general correction improvement" in lower
    assert "unconstrained" in lower


def test_readme_documentation_links_point_to_existing_local_paths():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for rel in README_LINKS:
        assert rel in readme
        assert (REPO_ROOT / rel).exists(), rel


def test_synthetic_generator_cli_docs_clarify_long_demo_wrapper_and_config_contract():
    text = (REPO_ROOT / "docs" / "synthetic_dataset_generator_cli.md").read_text(encoding="utf-8")
    lower = text.lower()

    assert "python examples/generate_long_duration_demo.py" in text
    assert "writes the matching `tutorial_config.yaml`" in lower
    assert "raw generator command assumes that the config file passed to `--config` already exists" in lower


def test_docs_do_not_claim_native_doric_or_tdt_support():
    text = (REPO_ROOT / "docs" / "input_formats.md").read_text(encoding="utf-8").lower()
    assert re.search(r"doric \| not native", text)
    assert re.search(r"tucker davis / tdt \| not native", text)
    assert "conversion-through-csv" in text or "strict `custom_tabular`" in text


def test_bundled_synthetic_dataset_validate_only_smoke(tmp_path):
    out_dir = tmp_path / "validate_only"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_full_pipeline_deliverables.py"),
        "--input",
        str(BUNDLED_DATASET),
        "--out",
        str(out_dir),
        "--config",
        str(BUNDLED_CONFIG),
        "--format",
        "rwd",
        "--mode",
        "both",
        "--sessions-per-hour",
        "2",
        "--validate-only",
        "--overwrite",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALIDATE-ONLY: OK" in result.stdout


def test_bundled_synthetic_dataset_full_run_outputs_match_quickstart(tmp_path):
    out_dir = tmp_path / "full_run"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_full_pipeline_deliverables.py"),
        "--input",
        str(BUNDLED_DATASET),
        "--out",
        str(out_dir),
        "--config",
        str(BUNDLED_CONFIG),
        "--format",
        "rwd",
        "--mode",
        "both",
        "--sessions-per-hour",
        "2",
        "--overwrite",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    for rel in (
        "status.json",
        "MANIFEST.json",
        "run_report.json",
        "events.ndjson",
        "_analysis/phasic_out",
        "_analysis/tonic_out",
        "_analysis/phasic_out/config_used.yaml",
        "_analysis/tonic_out/config_used.yaml",
        "_analysis/phasic_out/features/features.csv",
        "CH1/summary",
        "CH1/day_plots",
        "CH1/tables",
        "CH2/summary",
        "CH2/day_plots",
        "CH2/tables",
    ):
        assert (out_dir / rel).exists(), rel


def test_docs_preserve_scope_and_safety_claims():
    correction = (REPO_ROOT / "docs" / "correction_and_dynamic_fit.md").read_text(encoding="utf-8").lower()
    continuous = (REPO_ROOT / "docs" / "continuous_recordings.md").read_text(encoding="utf-8").lower()
    batch = (REPO_ROOT / "docs" / "batch_processing.md").read_text(encoding="utf-8").lower()
    quickstart = (REPO_ROOT / "docs" / "quickstart_gui_synthetic.md").read_text(encoding="utf-8").lower()

    assert "isosbestic/reference correction is part of the standard phasic preprocessing workflow" in correction
    assert "default behavior is `unconstrained`" in correction
    assert "reported intervention" in correction
    assert "does not prove that the corrected trace is biologically true" in correction
    assert "npm/interleaved continuous input is not currently implemented" in continuous
    assert "do not expect every raw point" in continuous
    assert "immediate subfolders" in batch
    assert "one shared configuration" in batch
    assert "does not perform group statistics" in batch
    assert "group averaging" in batch
    assert "simultaneous multi-recording visualization" in batch
    assert "not biological validation" in quickstart
