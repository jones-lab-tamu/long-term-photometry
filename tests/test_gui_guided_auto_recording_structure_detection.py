"""CR1-F1-B: automatic recording-structure detection under the Guided defaults.

With Format = Auto and Recording structure = Detect automatically, a scientist
should be able to select a supported folder and have the application work out
both the format and -- for RWD -- whether the recording is repeated sessions or
one continuous recording.

Detection is validation, not guessing: the two accepted RWD readers are asked
whether they can read the source, folder names and file counts are never
consulted, and a genuinely ambiguous source is handed back to the scientist.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.main_window import (
    GUIDED_STRUCTURE_CHOICE_AUTO,
    GuidedContinuousRwdRoiDiscoveryError,
    MainWindow,
    _discover_continuous_rwd_rois,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import _values


pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _continuous_folder(folder):
    """One small continuous RWD acquisition folder, with the non-fluorescence
    CSVs a real RWD folder also carries."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(600):
        time_s, control, signal = _values(np.array([index], dtype=float))
        lines.append(
            f"{time_s[0]:.1f},{control[0,0]:.12f},{signal[0,0]:.12f},"
            f"{control[0,1]:.12f},{signal[0,1]:.12f}\n"
        )
    (folder / "Fluorescence.csv").write_text(
        "".join(lines), encoding="utf-8", newline=""
    )
    (folder / "Events.csv").write_text(
        "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
    )
    return folder


def _select_data(window, folder, *, fmt="auto", structure=GUIDED_STRUCTURE_CHOICE_AUTO):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(fmt)
    index = window._guided_acquisition_mode_combo.findData(structure)
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder.parent / "output"))


def _pump_discovery(window, qapp, timeout_ms=180_000):
    from PySide6.QtCore import QDeadlineTimer

    deadline = QDeadlineTimer(timeout_ms)
    while window._guided_roi_discovery_running and not deadline.hasExpired():
        qapp.processEvents()
    deadline = QDeadlineTimer(20_000)
    while (
        getattr(window, "_guided_roi_discovery_thread", None) is not None
        and not deadline.hasExpired()
    ):
        qapp.processEvents()


def _fake_intermittent_result(resolved_format, rois=("CH1", "CH2")):
    """The shape the existing accepted discovery path returns."""
    return {
        "resolved_format": resolved_format,
        "n_total_discovered": 2,
        "n_preview": 2,
        "sessions": [{"session_id": "s1"}, {"session_id": "s2"}],
        "rois": [{"roi_id": roi} for roi in rois],
    }


def _stub_intermittent(window, monkeypatch, result=None, error=None):
    """Replace only the intermittent discovery seam, leaving routing real."""
    calls: list[dict] = []

    def fake_runner(captured, diag=None, phase=None):
        calls.append(dict(captured))
        if error is not None:
            raise error
        return dict(result)

    real_router = window._guided_discovery_runner_for_snapshot

    def router(snapshot):
        chosen = real_router(snapshot)
        if getattr(chosen, "__name__", "") == "run_intermittent":
            return fake_runner
        if getattr(chosen, "__name__", "") == "run_auto_structure":
            def auto(captured, diag=None, phase=None):
                return window._resolve_guided_rwd_structure(
                    captured, fake_runner, diag=diag, phase=phase
                )

            return auto
        return chosen

    monkeypatch.setattr(window, "_guided_discovery_runner_for_snapshot", router)
    return calls


# ---------------------------------------------------------------------------
# Default controls
# ---------------------------------------------------------------------------


def test_defaults_are_auto_format_and_auto_structure(window):
    window._on_guided_start_setup_new_analysis()
    assert window._guided_format_combo.currentText() == "auto"
    assert window._guided_acquisition_mode_combo.currentData() == (
        GUIDED_STRUCTURE_CHOICE_AUTO
    )


def test_auto_format_enables_every_structure_choice(window):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText("auto")
    model = window._guided_acquisition_mode_combo.model()
    for data in (GUIDED_STRUCTURE_CHOICE_AUTO, "intermittent", "continuous"):
        index = window._guided_acquisition_mode_combo.findData(data)
        assert index >= 0
        assert model.item(index).isEnabled() is True, data


def test_explicit_rwd_enables_every_structure_choice(window):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText("rwd")
    model = window._guided_acquisition_mode_combo.model()
    for data in (GUIDED_STRUCTURE_CHOICE_AUTO, "intermittent", "continuous"):
        index = window._guided_acquisition_mode_combo.findData(data)
        assert model.item(index).isEnabled() is True, data


@pytest.mark.parametrize("single_structure_format", ["npm", "custom_tabular"])
def test_single_structure_formats_offer_sessions_only(
    window, single_structure_format
):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(single_structure_format)
    combo = window._guided_acquisition_mode_combo
    model = combo.model()
    assert model.item(combo.findData("intermittent")).isEnabled() is True
    for data in (GUIDED_STRUCTURE_CHOICE_AUTO, "continuous"):
        assert model.item(combo.findData(data)).isEnabled() is False, data
    # And the structure in force is repeated sessions.
    assert window._guided_effective_acquisition_mode() == "intermittent"


@pytest.mark.parametrize("single_structure_format", ["npm", "custom_tabular"])
def test_single_structure_formats_cannot_produce_a_continuous_draft(
    window, tmp_path, single_structure_format
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, fmt="rwd", structure="continuous")
    assert window._guided_effective_acquisition_mode() == "continuous"

    window._guided_format_combo.setCurrentText(single_structure_format)

    assert window._guided_effective_acquisition_mode() == "intermittent"
    assert window._build_guided_new_analysis_draft_plan().acquisition_mode == (
        "intermittent"
    )


# ---------------------------------------------------------------------------
# Explicit routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,structure,expected",
    [
        ("rwd", "continuous", "_discover_continuous_rwd_rois"),
        ("auto", "continuous", "_discover_continuous_rwd_rois"),
        ("rwd", "intermittent", "run_intermittent"),
        ("auto", "intermittent", "run_intermittent"),
        ("rwd", GUIDED_STRUCTURE_CHOICE_AUTO, "run_auto_structure"),
        ("auto", GUIDED_STRUCTURE_CHOICE_AUTO, "run_auto_structure"),
        ("npm", GUIDED_STRUCTURE_CHOICE_AUTO, "run_intermittent"),
        ("custom_tabular", GUIDED_STRUCTURE_CHOICE_AUTO, "run_intermittent"),
    ],
)
def test_decision_table_routes_as_specified(
    window, tmp_path, fmt, structure, expected
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, fmt=fmt, structure=structure)
    snapshot = window._snapshot_guided_discovery_inputs()
    runner = window._guided_discovery_runner_for_snapshot(snapshot)
    assert getattr(runner, "__name__", "") == expected


def test_explicit_continuous_never_tries_other_formats(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, fmt="auto", structure="continuous")
    monkeypatch.setattr(
        window,
        "_build_discovery_spec_from_snapshot",
        lambda *a, **k: pytest.fail(
            "the intermittent/format-detection path ran for an explicit "
            "continuous choice"
        ),
    )

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert window._discovery_cache["resolved_format"] == "rwd"
    assert window._discovery_cache["acquisition_mode"] == "continuous"


# ---------------------------------------------------------------------------
# Automatic RWD structure resolution
# ---------------------------------------------------------------------------


def test_continuous_only_valid_resolves_continuous(window, qapp, tmp_path):
    """The real continuous folder: the intermittent reader genuinely refuses
    it, the continuous reader accepts it."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert window._discovery_cache["resolved_format"] == "rwd"
    assert window._discovery_cache["acquisition_mode"] == "continuous"
    assert window._guided_resolved_acquisition_mode == "continuous"
    assert [
        window._guided_roi_list.item(i).text()
        for i in range(window._guided_roi_list.count())
    ] == ["ROI1", "ROI2"]
    assert window._guided_select_data_ready_to_continue() is True


def test_intermittent_only_valid_resolves_intermittent(
    window, qapp, tmp_path, monkeypatch
):
    folder = tmp_path / "sessions"
    folder.mkdir()
    _select_data(window, folder)
    calls = _stub_intermittent(
        window, monkeypatch, result=_fake_intermittent_result("rwd")
    )

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert calls, "the intermittent validator was not consulted"
    assert calls[0]["acquisition_mode"] == "intermittent"
    assert window._discovery_cache["resolved_format"] == "rwd"
    assert window._discovery_cache["acquisition_mode"] == "intermittent"
    assert window._guided_resolved_acquisition_mode == "intermittent"
    assert window._guided_select_data_ready_to_continue() is True


def test_both_valid_refuses_to_guess(window, qapp, tmp_path):
    """A source both readings accept is handed back to the scientist.

    CR1-F1-C decides this from the bounded structure probes, so the folder
    must genuinely have both shapes: timestamped session subfolders *and* a
    Fluorescence.csv directly inside.
    """
    folder = _continuous_folder(tmp_path / "both")
    session = folder / "2026_03_16-10_00_00"
    session.mkdir()
    (session / "fluorescence.csv").write_text(
        (folder / "Fluorescence.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="",
    )
    _select_data(window, folder)

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert window._discovery_cache is None
    assert window._guided_resolved_acquisition_mode is None
    assert window._guided_roi_list.count() == 0
    assert window._guided_select_data_ready_to_continue() is False
    shown = window._guided_roi_discovery_failure_message(
        "reason: This folder can be read either as repeated sessions or as one "
        "continuous recording. Choose the recording structure, then select "
        "ROIs again."
    )
    assert "Choose the recording structure" in shown


def test_neither_valid_shows_one_plain_source_error(window, qapp, tmp_path):
    folder = tmp_path / "not_a_recording"
    folder.mkdir()
    _select_data(window, folder)

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert window._discovery_cache is None
    assert window._guided_roi_list.count() == 0
    assert window._guided_select_data_ready_to_continue() is False


def test_neither_valid_message_has_no_parser_detail(window, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    snapshot = {
        "input_dir": str(folder),
        "format": "auto",
        "guided_structure_choice": GUIDED_STRUCTURE_CHOICE_AUTO,
        "acquisition_mode": "intermittent",
    }

    def failing_intermittent(captured, diag=None, phase=None):
        raise ValueError(
            "No valid UV/SIG channel suffix pair found in header: X\\Events.csv"
        )

    with pytest.raises(GuidedContinuousRwdRoiDiscoveryError) as excinfo:
        window._resolve_guided_rwd_structure(snapshot, failing_intermittent)

    message = str(excinfo.value)
    # CR1-F1-C widened this to "supported data": the same fall-through now
    # also covers NPM and custom tabular, which are not RWD.
    assert "could not be read as supported data" in message
    for forbidden in ("Traceback", "ValueError", "UV/SIG", "Events.csv"):
        assert forbidden not in message


# ---------------------------------------------------------------------------
# NPM and custom tabular are untouched by structure detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detected_format", ["npm", "custom_tabular"])
def test_auto_detected_single_structure_formats_resolve_intermittent(
    window, qapp, tmp_path, monkeypatch, detected_format
):
    folder = tmp_path / detected_format
    folder.mkdir()
    _select_data(window, folder)
    _stub_intermittent(
        window, monkeypatch, result=_fake_intermittent_result(detected_format)
    )
    # Once the format is truthfully identified, the continuous RWD inspector
    # must not run at all.
    monkeypatch.setattr(
        main_window_module,
        "_discover_continuous_rwd_rois",
        lambda *a, **k: pytest.fail(
            f"continuous RWD inspection ran for {detected_format}"
        ),
    )

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert window._discovery_cache["resolved_format"] == detected_format
    assert window._discovery_cache["acquisition_mode"] == "intermittent"
    assert window._guided_resolved_acquisition_mode == "intermittent"
    assert window._build_guided_new_analysis_draft_plan().acquisition_mode == (
        "intermittent"
    )


# ---------------------------------------------------------------------------
# The draft never carries "auto"
# ---------------------------------------------------------------------------


def test_draft_never_carries_auto_as_acquisition_mode(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)

    # Before discovery the choice is auto, but the draft already resolves to a
    # real structure and Select data is not passable.
    assert window._guided_selected_acquisition_mode() == (
        GUIDED_STRUCTURE_CHOICE_AUTO
    )
    before = window._build_guided_new_analysis_draft_plan()
    assert before.acquisition_mode in {"intermittent", "continuous"}
    assert window._guided_select_data_ready_to_continue() is False

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    after = window._build_guided_new_analysis_draft_plan()
    assert after.acquisition_mode == "continuous"
    assert after.execution_intent.execution_mode == "both"


def test_discovery_without_a_stated_structure_means_repeated_sessions(
    window, qapp, tmp_path, monkeypatch
):
    """Only the continuous route reports "continuous"; every other accepted
    producer describes repeated sessions, so a result that states nothing is
    intermittent -- never left as an unresolved "auto"."""
    folder = tmp_path / "sessions"
    folder.mkdir()
    _select_data(window, folder)
    bare = _fake_intermittent_result("rwd")
    bare.pop("acquisition_mode", None)
    _stub_intermittent(window, monkeypatch, result=bare)

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert window._guided_resolved_acquisition_mode == "intermittent"
    assert window._guided_effective_acquisition_mode() == "intermittent"
    assert window._build_guided_new_analysis_draft_plan().acquisition_mode == (
        "intermittent"
    )
    assert window._guided_select_data_ready_to_continue() is True


def test_changing_folder_clears_the_resolved_structure(
    window, qapp, tmp_path
):
    """A different folder is a different recording, so what the previous one
    resolved to no longer applies, and a discovery still running for it cannot
    install its result afterwards."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)
    assert window._guided_resolved_acquisition_mode == "continuous"
    generation_before = window._guided_discovery_generation

    window._guided_input_dir_edit.setText(str(tmp_path / "somewhere_else"))

    assert window._guided_resolved_acquisition_mode is None
    # The generation moved on, so the previous folder's in-flight result is
    # superseded rather than installed.
    assert window._guided_discovery_generation != generation_before
    window._on_guided_roi_discovery_succeeded(
        {
            "resolved_format": "rwd",
            "acquisition_mode": "continuous",
            "rois": [{"roi_id": "STALE"}],
            "sessions": [],
        },
        generation=generation_before,
    )
    assert window._guided_resolved_acquisition_mode is None
    assert "STALE" not in [
        window._guided_roi_list.item(i).text()
        for i in range(window._guided_roi_list.count())
    ]


def test_explicit_override_after_automatic_resolution_requires_rediscovery(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)
    assert window._guided_resolved_acquisition_mode == "continuous"

    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )

    assert window._guided_resolved_acquisition_mode is None
    assert window._guided_roi_list.count() == 0
    assert window._guided_select_data_ready_to_continue() is False
    assert window._guided_effective_acquisition_mode() == "intermittent"


# ---------------------------------------------------------------------------
# The visible default workflow
# ---------------------------------------------------------------------------


def test_visible_default_workflow_reaches_continuous_settings(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    summary = window._guided_discovery_summary_label.text()
    assert "one continuous recording" in summary
    assert "session" not in summary.lower()
    assert window._guided_select_data_ready_to_continue() is True

    window._on_guided_continue_to_recording_structure()
    assert window._guided_workflow_stack.currentWidget().objectName() == (
        "guidedStepRecordingStructure"
    )
    assert window._guided_continuous_window_sec_spin.isHidden() is False
    assert window._guided_sessions_per_hour_edit.isHidden() is True


def test_visible_default_workflow_reaches_session_settings(
    window, qapp, tmp_path, monkeypatch
):
    folder = tmp_path / "sessions"
    folder.mkdir()
    _select_data(window, folder)
    _stub_intermittent(
        window, monkeypatch, result=_fake_intermittent_result("rwd")
    )

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert window._guided_resolved_acquisition_mode == "intermittent"
    summary = window._guided_discovery_summary_label.text()
    assert "recording session" in summary

    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("120")
    window._on_guided_continue_to_recording_structure()
    assert window._guided_workflow_stack.currentWidget().objectName() == (
        "guidedStepRecordingStructure"
    )
    assert window._guided_sessions_per_hour_edit.isHidden() is False
    assert window._guided_continuous_window_sec_spin.isHidden() is True


def test_auto_structure_detection_runs_off_the_gui_thread(
    window, qapp, tmp_path
):
    import threading

    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    gui_thread = threading.get_ident()
    seen: dict[str, int] = {}
    real = main_window_module._discover_continuous_rwd_rois

    def recording(snapshot, diag=None):
        seen["thread"] = threading.get_ident()
        return real(snapshot, diag)

    import gui.main_window as mw

    original = mw._discover_continuous_rwd_rois
    mw._discover_continuous_rwd_rois = recording
    try:
        window._on_guided_discover_rois()
        _pump_discovery(window, qapp)
    finally:
        mw._discover_continuous_rwd_rois = original

    assert seen.get("thread") is not None
    assert seen["thread"] != gui_thread
