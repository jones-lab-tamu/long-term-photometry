"""Continuous runs are finalized only when their promised window outputs exist (4J16k40).

These drive the real wrapper end-to-end on a small continuous recording: the
analysis subprocesses, `continuous_outputs`, and finalization all run for real.
Only the finalization checkpoint hook is injected, so a deletion happens at the
same moment a crash or a stray cleanup would.
"""

import json
import shutil
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
    FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY,
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)
from tests.test_continuous_mode_completed_run import (
    _write_continuous_config,
    _write_custom_tabular_csv,
)

PHASIC_WINDOW_SUMMARY = "Region0/tables/continuous_phasic_window_summary.csv"
OPTIONAL_PLOT = "Region0/summary/phasic_peak_rate_timeseries.png"


def _run_continuous_wrapper(tmp_path: Path, *, on_checkpoint=None) -> tuple[Path, int | None]:
    """Run the real wrapper on a small continuous recording. Returns (run_dir, exit code)."""
    input_dir = tmp_path / "input"
    _write_custom_tabular_csv(input_dir / "session_000.csv", duration_sec=1200.0, fs_hz=10.0)
    config_path = tmp_path / "continuous.yaml"
    _write_continuous_config(config_path, target_fs_hz=10.0, custom_tabular=True)
    run_dir = tmp_path / "run"

    args = [
        "tools/run_full_pipeline_deliverables.py",
        "--input", str(input_dir),
        "--out", str(run_dir),
        "--config", str(config_path),
        "--format", "custom_tabular",
        "--mode", "phasic",
        "--overwrite",
    ]

    def hook(name):
        if on_checkpoint is not None:
            on_checkpoint(name, run_dir)

    with patch("sys.argv", args), patch.object(wrapper, "_TEST_FINALIZATION_HOOK", hook):
        try:
            wrapper.main()
        except SystemExit as exc:
            return run_dir, exc.code
    return run_dir, None


def _delete_at_finalize(rel_path: str):
    def _hook(name, run_dir: Path):
        if name == "before_report_finalize":
            target = run_dir / Path(*rel_path.split("/"))
            if target.exists():
                target.unlink()

    return _hook


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


# 1. A complete continuous run succeeds ----------------------------------------


def test_continuous_run_with_complete_outputs_is_successful(tmp_path: Path):
    run_dir, code = _run_continuous_wrapper(tmp_path)
    assert code in (None, 0)

    classification = classify_run_terminal_state(str(run_dir))
    assert classification.state == TERMINAL_SUCCESS_CURRENT, classification.reason
    assert is_successful_completed_run_dir(str(run_dir))[0] is True

    completion = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))["completion"]
    run_mode = completion["run_mode"]
    assert run_mode["deliverable_profile"] == "continuous"
    assert run_mode["continuous_outputs_ran"] is True
    assert run_mode["expected_rois"] == ["Region0"]

    deliverables = completion["deliverables"]
    assert PHASIC_WINDOW_SUMMARY in deliverables["required"]

    family = deliverables["continuous_window_index"]["families"][
        FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY
    ]
    assert family["relative_paths"]["Region0"] == PHASIC_WINDOW_SUMMARY
    # One row per analysis window, so the table is the exhaustive window index.
    assert family["window_row_counts"]["Region0"] >= 2

    required = {a["relative_path"] for a in completion["artifacts"] if a["required"]}
    assert PHASIC_WINDOW_SUMMARY in required


# 2/3/7. Losing a promised window output fails, and cannot excuse itself -------


def test_deleting_the_window_summary_before_finalization_fails_closed(tmp_path: Path):
    run_dir, code = _run_continuous_wrapper(
        tmp_path, on_checkpoint=_delete_at_finalize(PHASIC_WINDOW_SUMMARY)
    )
    assert code == 1

    status = _status(run_dir)
    assert status["status"] == "error"
    assert any("TERMINAL_VALIDATION_FAILED" in err for err in status["errors"])
    assert is_successful_completed_run_dir(str(run_dir))[0] is False
    assert not (run_dir / "MANIFEST.json").exists()


def test_deleting_the_window_summary_does_not_change_the_run_mode(tmp_path: Path):
    """The trust boundary: run_mode is execution intent, so it cannot excuse the file.

    Were the expectation inferred from output presence, deleting the table would
    quietly remove it from the required set and the run would finalize.
    """
    run_dir, code = _run_continuous_wrapper(
        tmp_path, on_checkpoint=_delete_at_finalize(PHASIC_WINDOW_SUMMARY)
    )
    assert code == 1

    status = _status(run_dir)
    assert status["status"] == "error"
    # The failure names the output that was owed, rather than silently dropping it.
    assert any("continuous_phasic_window_summary.csv" in err for err in status["errors"])


def test_removing_all_scientist_facing_outputs_before_finalization_fails(tmp_path: Path):
    def _hook(name, run_dir: Path):
        if name == "before_report_finalize":
            shutil.rmtree(run_dir / "Region0")

    run_dir, code = _run_continuous_wrapper(tmp_path, on_checkpoint=_hook)
    assert code == 1

    # The internal analysis outputs are untouched; that is not a completed run.
    assert (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").is_file()
    assert (run_dir / "_analysis" / "phasic_out" / "features" / "features.csv").is_file()
    assert is_successful_completed_run_dir(str(run_dir))[0] is False
    assert _status(run_dir)["status"] == "error"


# 4. Optional continuous outputs may be absent ---------------------------------


def test_deleting_an_optional_continuous_plot_still_succeeds(tmp_path: Path):
    run_dir, code = _run_continuous_wrapper(
        tmp_path, on_checkpoint=_delete_at_finalize(OPTIONAL_PLOT)
    )
    assert code in (None, 0)

    assert not (run_dir / Path(*OPTIONAL_PLOT.split("/"))).exists()
    assert classify_run_terminal_state(str(run_dir)).state == TERMINAL_SUCCESS_CURRENT
