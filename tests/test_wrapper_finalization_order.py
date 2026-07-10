"""Wrapper finalization is ordered and fail-closed (4J16k40b).

A crash anywhere before the success status is written must leave a directory
that cannot be reloaded as a successful run.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import tools.run_full_pipeline_deliverables as wrapper
from gui.run_report_parser import is_successful_completed_run_dir
from photometry_pipeline.run_completion_contract import (
    TERMINAL_FAILED,
    TERMINAL_INTERRUPTED,
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)
from tests.terminal_run_fixtures import (
    BASE_CONFIG_PATH,
    seed_wrapper_analysis_outputs,
    seed_wrapper_deliverables,
)

def _run_wrapper(
    run_dir: Path,
    tmp_path: Path,
    *,
    fail_at: str | None = None,
    seed: bool = True,
    on_checkpoint=None,
):
    """Drive main() with the analysis and plotting subprocesses mocked out.

    `fail_at` raises at a named finalization checkpoint. `on_checkpoint` lets a
    test perturb the run directory at a checkpoint without raising.

    Returns the SystemExit code, or None when main() returned normally.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(BASE_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    if seed:
        seed_wrapper_analysis_outputs(run_dir)
        # The plot/table subprocesses are mocked, so stand in for the per-ROI
        # deliverables a real full run would have written.
        seed_wrapper_deliverables(run_dir, ["Region0"])

    args = [
        "tools/run_full_pipeline_deliverables.py",
        "--input", str(input_dir),
        "--out", str(run_dir),
        "--config", str(config_file),
        "--format", "rwd",
        "--mode", "both",
        "--sessions-per-hour", "1",
        # Cleanup is patched out, so the seeded analysis outputs survive and
        # stand in for what the mocked subprocesses would have written.
        "--overwrite",
    ]

    mock_discovery = {"sessions": [{"id": "chunk_0000"}], "rois": ["Region0"]}

    def hook(name):
        if on_checkpoint is not None:
            on_checkpoint(name)
        if fail_at is not None and name == fail_at:
            raise RuntimeError(f"injected finalization failure at {name}")

    with patch("sys.argv", args), \
         patch("subprocess.check_call"), \
         patch("photometry_pipeline.discovery.discover_inputs", return_value=mock_discovery), \
         patch("tools.run_full_pipeline_deliverables.validate_inputs"), \
         patch("tools.run_full_pipeline_deliverables._cleanup_run_outputs_in_place"), \
         patch.object(wrapper, "_TEST_FINALIZATION_HOOK", hook):
        try:
            wrapper.main()
        except SystemExit as exc:
            return exc.code
    return None


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


# Happy path -------------------------------------------------------------------


def test_completed_run_writes_one_coherent_terminal_set(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    code = _run_wrapper(run_dir, tmp_path)
    assert code in (None, 0)

    classification = classify_run_terminal_state(str(run_dir))
    assert classification.state == TERMINAL_SUCCESS_CURRENT, classification.reason
    assert is_successful_completed_run_dir(str(run_dir))[0] is True

    # The success status pins the manifest, the manifest declares itself final.
    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["completion"]["final"] is True
    assert _status(run_dir)["completion"]["manifest_sha256"]
    assert _status(run_dir)["errors"] == []

    # The mandatory outputs are enumerated, not merely implied.
    required = {
        entry["relative_path"]
        for entry in manifest["completion"]["artifacts"]
        if entry["required"]
    }
    assert "run_report.json" in required
    assert "_analysis/phasic_out/features/feature_event_provenance.json" in required
    assert "_analysis/tonic_out/tonic_trace_cache.h5" in required


def test_finalization_preserves_manifest_timing_phases(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run_wrapper(run_dir, tmp_path)

    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    phases = manifest["timing"]["phases"]
    assert "manifest_write" in phases
    assert "finalize_artifacts" in phases
    assert _status(run_dir)["timing"]["last_completed_phase"] == "finalize_artifacts"


# Failure injection ------------------------------------------------------------


@pytest.mark.parametrize(
    "fail_at",
    ["before_report_finalize", "before_manifest_write", "after_manifest_write", "before_success_status"],
)
def test_injected_finalization_failure_never_yields_success(tmp_path: Path, fail_at: str):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    code = _run_wrapper(run_dir, tmp_path, fail_at=fail_at)
    assert code == 1

    ok, reason = is_successful_completed_run_dir(str(run_dir))
    assert ok is False, f"injected failure at {fail_at} still reloaded as successful: {reason}"

    state = classify_run_terminal_state(str(run_dir)).state
    assert state in (TERMINAL_FAILED, TERMINAL_INTERRUPTED)
    assert _status(run_dir)["status"] != "success"


def test_crash_immediately_after_success_status_is_still_successful(tmp_path: Path):
    """The success status is written only once the terminal set already verified."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _run_wrapper(run_dir, tmp_path, fail_at="after_success_status")

    assert classify_run_terminal_state(str(run_dir)).state == TERMINAL_SUCCESS_CURRENT


def test_missing_feature_provenance_blocks_success(tmp_path: Path):
    """A run whose per-ROI settings record never landed must not finalize as successful."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    seed_wrapper_analysis_outputs(run_dir)
    (run_dir / "_analysis" / "phasic_out" / "features" / "feature_event_provenance.json").unlink()

    code = _run_wrapper(run_dir, tmp_path, seed=False)
    assert code == 1

    assert is_successful_completed_run_dir(str(run_dir))[0] is False
    assert _status(run_dir)["status"] == "error"


def test_missing_mandatory_artifact_blocks_finalization(tmp_path: Path):
    """An output missing only at finalize time still cannot produce a final manifest."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    seed_wrapper_analysis_outputs(run_dir)
    (run_dir / "_analysis" / "tonic_out" / "config_used.yaml").unlink()

    code = _run_wrapper(run_dir, tmp_path, seed=False)
    assert code == 1

    assert is_successful_completed_run_dir(str(run_dir))[0] is False
    status = _status(run_dir)
    assert status["status"] == "error"
    assert any("TERMINAL_VALIDATION_FAILED" in err for err in status["errors"])
    assert not (run_dir / "MANIFEST.json").exists()


def test_terminal_failure_is_not_relabelled_as_cancelled(tmp_path: Path):
    """An error recorded during finalization stays an error, not a cancellation."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _run_wrapper(run_dir, tmp_path, fail_at="before_success_status")
    assert _status(run_dir)["status"] == "error"


# Run mode is execution intent, never an inference from the files it validates --


def test_run_mode_records_execution_intent_not_output_presence(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run_wrapper(run_dir, tmp_path)

    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    run_mode = manifest["completion"]["run_mode"]
    assert run_mode["feature_extraction_ran"] is True
    assert run_mode["deliverable_profile"] == "full_intermittent"
    assert run_mode["expected_rois"] == ["Region0"]
    assert manifest["completion"]["deliverables"]["expected_rois"] == ["Region0"]


@pytest.mark.parametrize(
    "rel_path",
    [
        "_analysis/phasic_out/features/features.csv",
        "_analysis/phasic_out/features/feature_event_provenance.json",
        "Region0/summary/phasic_correction_impact.png",
        "Region0/day_plots/phasic_sig_iso_day_000.png",
    ],
)
def test_output_deleted_just_before_finalization_cannot_excuse_itself(tmp_path: Path, rel_path: str):
    """The expected run mode must not change to make a missing output optional."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _delete_at_finalize(name):
        if name == "before_report_finalize":
            target = run_dir / Path(*rel_path.split("/"))
            if target.exists():
                target.unlink()

    code = _run_wrapper(run_dir, tmp_path, on_checkpoint=_delete_at_finalize)
    assert code == 1

    status = _status(run_dir)
    assert status["status"] == "error"
    assert any("TERMINAL_VALIDATION_FAILED" in err for err in status["errors"])
    assert is_successful_completed_run_dir(str(run_dir))[0] is False
    assert not (run_dir / "MANIFEST.json").exists()


def test_missing_per_roi_deliverable_blocks_success(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    seed_wrapper_analysis_outputs(run_dir)
    seed_wrapper_deliverables(run_dir, ["Region0"])
    (run_dir / "Region0" / "tables" / "phasic_auc_timeseries.csv").unlink()

    code = _run_wrapper(run_dir, tmp_path, seed=False)
    assert code == 1
    assert is_successful_completed_run_dir(str(run_dir))[0] is False
    assert _status(run_dir)["status"] == "error"
