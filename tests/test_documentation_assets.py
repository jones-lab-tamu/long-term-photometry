import re
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DATASET = REPO_ROOT / "examples" / "data" / "synthetic_photometry_basic"
BUNDLED_CONFIG = BUNDLED_DATASET / "tutorial_config.yaml"

DOC_PATHS = [
    REPO_ROOT / "docs" / "guided_scientist_guide.md",
    REPO_ROOT / "docs" / "input_formats.md",
    REPO_ROOT / "docs" / "correction_and_dynamic_fit.md",
    REPO_ROOT / "docs" / "event_detection.md",
    REPO_ROOT / "docs" / "continuous_recordings.md",
    REPO_ROOT / "docs" / "batch_processing.md",
]

README_LINKS = [
    "docs/guided_scientist_guide.md",
    "docs/input_formats.md",
    "docs/correction_and_dynamic_fit.md",
    "docs/event_detection.md",
    "docs/continuous_recordings.md",
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


def test_key_documentation_files_exist_and_describe_current_workflow():
    for path in DOC_PATHS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert len(text.strip()) > 200

    guide = (REPO_ROOT / "docs" / "guided_scientist_guide.md").read_text(encoding="utf-8")
    assert "# Guided Scientist Guide" in guide
    assert "software-ready" in guide.lower()
    assert "not biological validation" in guide.lower()
    for stale in (
        "docs/quickstart_gui_synthetic.md",
        "docs/synthetic_demo_datasets.md",
        "docs/tutorial_first_run_with_demo_dataset.md",
        "gui/README.md",
    ):
        assert not (REPO_ROOT / stale).exists(), stale


def test_correction_docs_keep_diagnostic_scope():
    doc_text = "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)
    lower = doc_text.lower()

    assert "prevent negative slopes" not in lower
    assert "advanced diagnostic" in lower
    assert "diagnostic-only" in lower or "diagnostic only" in lower
    assert "unconstrained" in lower


def test_readme_documentation_links_point_to_existing_local_paths():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    lower = readme.lower()

    for rel in README_LINKS:
        assert rel in readme
        assert (REPO_ROOT / rel).exists(), rel

    assert "guided scientist guide" in lower
    assert "guided is the recommended workflow" in lower
    assert re.search(r"full control.*expert.*backward-compatible", lower, re.DOTALL)

    assert "| RWD | Supported | Supported |" in readme
    assert "| Neurophotometrics | Supported | Not currently supported |" in readme
    assert "| CSV files | Supported | Supported |" in readme
    assert re.search(
        r"continuous\s+neurophotometrics\s+input\s+is\s+not\s+currently\s+supported",
        lower,
    )

    assert "requirements_gui.txt" in readme
    assert "python -m gui.app" in readme
    assert "Tools -> Generate Guided Demo Dataset" in readme

    guided_steps = (
        "Start",
        "Select data",
        "Recording structure",
        "Correction approach",
        "Feature detection",
        "Review plan",
        "Run",
        "Review",
    )
    for index, step in enumerate(guided_steps, start=1):
        assert f"{index}. **{step}**" in readme

    assert "custom_tabular" not in lower
    assert "tuning_prep" not in lower
    assert "run_full_pipeline_deliverables.py" not in lower
    assert "--validate-only" not in lower
    assert "--sessions-per-hour" not in lower
    assert "manifest.json" not in lower
    assert not re.search(r"one\s+csv(?:\s+file)?\s*=\s*one\s+session", lower)


def test_guided_scientist_guide_covers_current_visible_workflow():
    guide = (REPO_ROOT / "docs" / "guided_scientist_guide.md").read_text(
        encoding="utf-8"
    )
    lower = guide.lower()

    assert "# Guided Scientist Guide" in guide
    assert "| RWD | Supported | Supported |" in guide
    assert "| Neurophotometrics | Supported | Not currently supported |" in guide
    assert "| CSV files | Supported | Supported |" in guide
    guide_steps = (
        "1. Start",
        "2. Select data",
        "3. Recording structure",
        "4. Correction approach",
        "5. Feature detection",
        "6. Review plan",
        "7. Run",
        "8. Review",
    )
    assert [guide.index(step) for step in guide_steps] == sorted(
        guide.index(step) for step in guide_steps
    )
    assert re.search(
        r"continuous\s+neurophotometrics\s+recordings?\s+are\s+not\s+currently\s+supported",
        lower,
    )
    assert re.search(
        r"do\s+not\s+force\s+a\s+continuous\s+recording\s+through\s+the\s+repeated-session\s+option",
        lower,
    )
    assert "supported repeated-session organization" not in lower
    assert "one csv recording file defaults to a continuous" in lower
    assert "multiple csv recording files default to repeated sessions" in lower
    for label in (
        "Time column",
        "Time units",
        "ROI name",
        "Signal column",
        "Reference column",
        "Add ROI",
        "Select ROIs",
        "Sessions per hour",
        "Session duration (s)",
        "Continuous analysis window (s)",
        "Allow partial final analysis window",
        "Robust Global Event-Reject Fit",
        "Adaptive Event-Gated Fit",
        "Global Linear Regression",
        "Signal-Only F0",
        "Default",
        "Custom",
        "Review plan",
        "Check my setup",
        "Run Guided Analysis",
        "Verification",
        "Phasic dFF",
        "Phasic Summary",
    ):
        assert label in guide

    for control in (
        "Event signal",
        "Signal excursion polarity",
        "Peak threshold method",
        "Peak threshold k",
        "Peak threshold percentile",
        "Peak threshold absolute",
        "Peak min distance (sec)",
        "Peak min prominence k",
        "Peak min width (sec)",
        "Peak pre-filter",
        "Event AUC baseline",
    ):
        assert f"**{control}**" in guide

    assert "Tools -> Generate Guided Demo Dataset" in guide
    assert "preview segment" in lower
    assert re.search(r"complete\s+selected\s+recording\s+set", lower)
    assert re.search(r"explicit\s+per-roi\s+correction\s+strategy", lower)
    assert re.search(r"final\s+analysis\s+recomputes", lower)
    assert re.search(
        r"does\s+not\s+use\s+the\s+reference\s+channel\s+for\s+correction",
        lower,
    )
    assert re.search(r"mixed\s+per-roi\s+plans\s+are\s+supported", lower)
    assert "production route" not in lower
    assert re.search(r"select\s+rois\s+lists\s+the\s+rois\s+found\s+in\s+the\s+recording", lower)
    assert re.search(r"choose\s+the\s+rois\s+you\s+want\s+included\s+in\s+the\s+analysis", lower)
    assert "repeated-session and continuous rwd can prefill" in lower
    assert "for repeated neurophotometrics" in lower
    assert re.search(
        r"csv\s+does\s+not\s+provide\s+an\s+absolute\s+recording-start\s+timestamp",
        lower,
    )
    assert "for repeated-session rwd, neurophotometrics, and csv recordings" in lower
    assert "unusable session is recorded as missing" in lower
    assert "original time position is preserved" in lower
    assert "interval remains blank" in lower
    assert re.search(r"later\s+sessions\s+do\s+not\s+shift\s+earlier", lower)
    assert "no advance approval is required" in lower
    assert "stops instead of guessing" in lower
    assert re.search(r"which\s+session\s+failed", lower)
    assert "correct session order" in lower
    assert re.search(r"timeline\s+position", lower)
    assert re.search(
        r"verification\s+provides\s+a\s+visual\s+review\s+of\s+the\s+completed\s+correction\s+result",
        lower,
    )
    assert re.search(r"not\s+every\s+view\s+appears\s+for\s+every\s+run", lower)
    assert "decision-support" not in lower
    assert "coming later" not in lower
    assert "read-only evidence" not in lower
    for warning_pattern in (
        r"flattens\s+most\s+plausible\s+signal\s+variation",
        r"inverted\s+responses",
        r"exaggerates\s+features\s+that\s+are\s+not\s+apparent\s+in\s+the\s+source\s+channels",
        r"behaves\s+very\s+differently\s+across\s+representative\s+preview\s+segments",
    ):
        assert re.search(warning_pattern, lower)
    assert "if the signal/reference mapping appears wrong" in lower
    assert "before continuing" in lower
    assert "software-ready" in lower
    assert "scientific readiness" in lower
    assert "programming is not required" in lower
    assert "full control remains" in lower
    assert re.search(r"outside\s+the\s+ordinary\s+path", lower)
    assert re.search(
        r"do not continue if the detected source.*does not match the experiment",
        lower,
        re.DOTALL,
    )

    for rel in (
        "input_formats.md",
        "continuous_recordings.md",
        "correction_and_dynamic_fit.md",
        "event_detection.md",
    ):
        assert f"]({rel})" in guide
        assert (REPO_ROOT / "docs" / rel).exists(), rel

    assert re.search(r"Intermittent\s+CSV\s+demo", guide)
    assert re.search(r"Continuous\s+CSV\s+demo", guide)
    for forbidden in (
        "custom_tabular",
        "tuning_prep",
        "run_full_pipeline_deliverables.py",
        "MANIFEST.json",
        "status.json",
        "Validate Only",
        "Run Pipeline",
        "python gui/main.py",
    ):
        assert forbidden.lower() not in lower
    assert "one csv file per recording session" not in lower
    assert not re.search(
        r"one\s+csv(?:\s+file)?\s*(?:=|equals?)\s*one\s+session",
        lower,
    )
    assert "CLI" not in guide


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


def test_bundled_synthetic_dataset_full_run_outputs_match_pipeline_contract(tmp_path):
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
    guide = (REPO_ROOT / "docs" / "guided_scientist_guide.md").read_text(encoding="utf-8").lower()

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
    assert "not biological validation" in guide
