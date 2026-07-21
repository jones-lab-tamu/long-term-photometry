"""A `reviewable_with_warning` compact overview is accepted for read-only
Review only. It must never make a success-only control (tuning, correction
retune, dF/F day-plot rerender) believe the run is a successfully completed
run -- that decision must still fall through to the real terminal
classification, which correctly reports the run as not successful.
"""

from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _accept_warning_overview(window, run_dir) -> None:
    resolved = os.path.realpath(str(run_dir))
    window._current_run_dir = resolved
    window._accepted_completed_review_path = resolved
    window._accepted_completed_review_overview = {
        "review_status": "reviewable_with_warning",
        "terminal_state": "failed",
        "included_rois": ["Region0"],
    }


def test_tuning_workspace_stays_unavailable_for_warning_review_overview(
    window, tmp_path
):
    run_dir = tmp_path / "warning_review_run"
    run_dir.mkdir()
    _accept_warning_overview(window, run_dir)
    window._is_complete_workspace_active = True

    window._refresh_tuning_workspace_availability()

    assert window._tuning_workspace_available is False
    assert "not confirmed as a successful completed run" in (
        window._tuning_availability_label.text()
    )


def test_dff_dayplot_rerender_stays_unavailable_for_warning_review_overview(
    window, tmp_path
):
    run_dir = tmp_path / "warning_review_run"
    run_dir.mkdir()
    _accept_warning_overview(window, run_dir)
    window._is_complete_workspace_active = True

    ok, reason = window._dff_dayplot_rerender_readiness()

    assert ok is False
    assert "not confirmed as a successful completed run" in reason


def test_success_overview_still_permits_tuning_availability_check_to_proceed(
    window, tmp_path
):
    """Control: a "success" review_status must not regress -- it still takes
    the accepted-overview fast path instead of falling through to
    `is_successful_completed_run_dir`."""
    run_dir = tmp_path / "success_review_run"
    run_dir.mkdir()
    resolved = os.path.realpath(str(run_dir))
    window._current_run_dir = resolved
    window._accepted_completed_review_path = resolved
    window._accepted_completed_review_overview = {
        "review_status": "success",
        "terminal_state": "success",
        "included_rois": ["Region0"],
    }
    window._is_complete_workspace_active = True

    window._refresh_tuning_workspace_availability()

    # No real phasic_out/cache exists in this bare directory, so tuning still
    # can't actually proceed -- but it must fail for that reason, not because
    # the run was rejected as unsuccessful.
    assert "not confirmed as a successful completed run" not in (
        window._tuning_availability_label.text()
    )
