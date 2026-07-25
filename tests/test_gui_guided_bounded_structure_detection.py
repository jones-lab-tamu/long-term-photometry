"""CR1-F1-C: automatic structure detection must be bounded and fast.

CR1-F1-B resolved the structure by running the *full* intermittent discovery
as a probe. That probe's cost is not bounded: its contract inference does
``rows = list(reader)`` -- it pulls whole CSVs into memory (11 s on this
project's 266 MB reference recording) -- and, when the contract does resolve,
it goes on to launch a discovery subprocess. Whether it returned quickly
depended on which CSV the folder happened to enumerate first.

Classification is now done with bounded probes: a directory scan plus a
header-only read for each candidate structure. Only the winning interpretation
does real work.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.main_window import (
    GUIDED_STRUCTURE_CHOICE_AUTO,
    MainWindow,
    _guided_bounded_rwd_header_columns,
    _guided_probe_continuous_rwd_structure,
    _guided_probe_intermittent_rwd_structure,
)
from gui.run_spec import RunSpec

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


def _fluorescence_rows(count):
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(count):
        time_s, control, signal = _values(np.array([index], dtype=float))
        lines.append(
            f"{time_s[0]:.1f},{control[0,0]:.12f},{signal[0,0]:.12f},"
            f"{control[0,1]:.12f},{signal[0,1]:.12f}\n"
        )
    return "".join(lines)


def _continuous_folder(folder, *, rows=600, with_events=True):
    """A continuous acquisition folder: Fluorescence.csv directly inside."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Fluorescence.csv").write_text(
        _fluorescence_rows(rows), encoding="utf-8", newline=""
    )
    if with_events:
        (folder / "Events.csv").write_text(
            "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
        )
    return folder


def _intermittent_folder(folder, *, sessions=2):
    """An intermittent acquisition folder: timestamped session subfolders."""
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(sessions):
        session = folder / f"2026_03_16-1{index}_00_00"
        session.mkdir()
        (session / "fluorescence.csv").write_text(
            _fluorescence_rows(120), encoding="utf-8", newline=""
        )
    return folder


def _select_data(window, folder, *, fmt="auto", structure=GUIDED_STRUCTURE_CHOICE_AUTO):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(fmt)
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData(structure)
    )
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder.parent / "output"))


def _pump(window, qapp, timeout_ms=180_000):
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


def _forbid_full_intermittent(window, monkeypatch, calls=None):
    """Fail the test if any part of the full intermittent path is entered."""
    def forbidden(*_args, **_kwargs):
        pytest.fail("the full intermittent discovery path ran as a probe")

    monkeypatch.setattr(RunSpec, "run_discovery", forbidden)
    monkeypatch.setattr(
        window, "_build_discovery_spec_from_snapshot", forbidden
    )
    monkeypatch.setattr(
        window, "_infer_rwd_dataset_contract_overrides", forbidden
    )


# ---------------------------------------------------------------------------
# The central regression: no full intermittent probe
# ---------------------------------------------------------------------------


def test_continuous_auto_never_runs_the_full_intermittent_path(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    _forbid_full_intermittent(window, monkeypatch)

    window._on_guided_discover_rois()
    _pump(window, qapp)

    assert window._discovery_cache["resolved_format"] == "rwd"
    assert window._discovery_cache["acquisition_mode"] == "continuous"
    assert [
        window._guided_roi_list.item(i).text()
        for i in range(window._guided_roi_list.count())
    ] == ["ROI1", "ROI2"]


def test_continuous_auto_is_unaffected_by_csv_enumeration_order(
    window, qapp, tmp_path, monkeypatch
):
    """The old probe only returned quickly because a tiny non-recording CSV
    happened to sort first. Without it, classification must still be bounded."""
    folder = _continuous_folder(tmp_path / "rec", with_events=False)
    _select_data(window, folder)
    _forbid_full_intermittent(window, monkeypatch)

    window._on_guided_discover_rois()
    _pump(window, qapp)

    assert window._discovery_cache["acquisition_mode"] == "continuous"


def test_bounded_header_read_does_not_consume_the_whole_file(tmp_path):
    """The probe reads a header window, not the recording."""
    path = tmp_path / "big.csv"
    path.write_text(_fluorescence_rows(20_000), encoding="utf-8", newline="")
    reads: list[int] = []

    real_open = open

    class CountingFile:
        def __init__(self, handle):
            self._handle = handle

        def __iter__(self):
            for count, line in enumerate(self._handle, start=1):
                reads.append(count)
                yield line

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    import builtins

    def counting_open(*args, **kwargs):
        return CountingFile(real_open(*args, **kwargs))

    builtins.open = counting_open
    try:
        columns = _guided_bounded_rwd_header_columns(str(path))
    finally:
        builtins.open = real_open

    assert columns is not None
    # Header is row 1; the probe must stop within its bounded window, far
    # short of the 20,000-row file.
    assert max(reads) <= 60


# ---------------------------------------------------------------------------
# Bounded probes classify correctly
# ---------------------------------------------------------------------------


def test_probes_classify_a_continuous_folder(tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    assert _guided_probe_continuous_rwd_structure(str(folder)) is True
    assert _guided_probe_intermittent_rwd_structure(str(folder)) is False


def test_probes_classify_an_intermittent_folder(tmp_path):
    folder = _intermittent_folder(tmp_path / "sessions")
    assert _guided_probe_intermittent_rwd_structure(str(folder)) is True
    assert _guided_probe_continuous_rwd_structure(str(folder)) is False


def test_probes_refuse_an_unsupported_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    assert _guided_probe_intermittent_rwd_structure(str(folder)) is False
    assert _guided_probe_continuous_rwd_structure(str(folder)) is False


def test_probes_do_not_use_folder_names(tmp_path):
    """A folder named "continuous" holding sessions is still sessions."""
    folder = _intermittent_folder(tmp_path / "continuous_recording")
    assert _guided_probe_intermittent_rwd_structure(str(folder)) is True
    assert _guided_probe_continuous_rwd_structure(str(folder)) is False


# ---------------------------------------------------------------------------
# Resolution runs exactly one full discovery
# ---------------------------------------------------------------------------


def test_intermittent_auto_runs_full_discovery_exactly_once(
    window, qapp, tmp_path, monkeypatch
):
    folder = _intermittent_folder(tmp_path / "sessions")
    _select_data(window, folder)

    runs: list[dict] = []

    def fake_runner(captured, diag=None, phase=None):
        runs.append(dict(captured))
        return {
            "resolved_format": "rwd",
            "n_total_discovered": 2,
            "n_preview": 2,
            "sessions": [{"session_id": "s1"}, {"session_id": "s2"}],
            "rois": [{"roi_id": "ROI1"}, {"roi_id": "ROI2"}],
        }

    real_router = window._guided_discovery_runner_for_snapshot

    def router(snapshot):
        chosen = real_router(snapshot)
        assert getattr(chosen, "__name__", "") == "run_auto_structure"

        def auto(captured, diag=None, phase=None):
            return window._resolve_guided_rwd_structure(
                captured, fake_runner, diag=diag, phase=phase
            )

        return auto

    monkeypatch.setattr(window, "_guided_discovery_runner_for_snapshot", router)
    continuous_calls: list[int] = []
    monkeypatch.setattr(
        main_window_module,
        "_discover_continuous_rwd_rois",
        lambda *a, **k: continuous_calls.append(1) or pytest.fail(
            "continuous inspection ran for an intermittent folder"
        ),
    )

    window._on_guided_discover_rois()
    _pump(window, qapp)

    assert len(runs) == 1
    assert runs[0]["acquisition_mode"] == "intermittent"
    assert continuous_calls == []
    assert window._discovery_cache["resolved_format"] == "rwd"
    assert window._discovery_cache["acquisition_mode"] == "intermittent"


def test_continuous_auto_inspects_the_recording_exactly_once(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    calls: list[int] = []
    real = main_window_module._discover_continuous_rwd_rois

    def counted(snapshot, diag=None, phase=None):
        calls.append(1)
        return real(snapshot, diag, phase)

    monkeypatch.setattr(
        main_window_module, "_discover_continuous_rwd_rois", counted
    )

    window._on_guided_discover_rois()
    _pump(window, qapp)

    assert len(calls) == 1
    assert window._discovery_cache["acquisition_mode"] == "continuous"


# ---------------------------------------------------------------------------
# Ambiguity and unsupported sources do no full work
# ---------------------------------------------------------------------------


def test_ambiguous_folder_runs_no_full_discovery_and_asks_the_scientist(
    window, qapp, tmp_path, monkeypatch
):
    """A folder that is both session-shaped and continuous-shaped."""
    folder = _intermittent_folder(tmp_path / "both")
    _continuous_folder(folder)
    assert _guided_probe_intermittent_rwd_structure(str(folder)) is True
    assert _guided_probe_continuous_rwd_structure(str(folder)) is True

    _select_data(window, folder)
    _forbid_full_intermittent(window, monkeypatch)
    monkeypatch.setattr(
        main_window_module,
        "_discover_continuous_rwd_rois",
        lambda *a, **k: pytest.fail("continuous inspection ran for an ambiguous folder"),
    )

    window._on_guided_discover_rois()
    _pump(window, qapp)

    assert window._discovery_cache is None
    assert window._guided_roi_list.count() == 0
    assert window._guided_select_data_ready_to_continue() is False


def test_unsupported_source_reports_once_and_leaves_busy_state(
    window, qapp, tmp_path
):
    folder = tmp_path / "unsupported"
    folder.mkdir()
    (folder / "notes.txt").write_text("not data", encoding="utf-8")
    _select_data(window, folder)

    window._on_guided_discover_rois()
    _pump(window, qapp)

    assert window._guided_roi_discovery_running is False
    assert window._guided_discover_rois_btn.isEnabled() is True
    assert window._discovery_cache is None
    assert window._guided_roi_list.count() == 0
    assert window._guided_select_data_ready_to_continue() is False


# ---------------------------------------------------------------------------
# Truthful phases
# ---------------------------------------------------------------------------


def _capture_status_text(window):
    """Record what the scientist actually sees, at the real GUI mutation.

    Deliberately not by replacing the phase slot: a plain function has no
    QObject receiver, so Qt would connect it directly and the slot would run
    on the worker thread, which is not how production behaves.
    """
    seen: list[tuple[str, int]] = []
    label = window._guided_discovery_summary_label
    real_set_text = label.setText

    def recording(text):
        import threading

        seen.append((str(text), threading.get_ident()))
        return real_set_text(text)

    label.setText = recording
    return seen


def test_discovery_shows_truthful_phases(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    captured = _capture_status_text(window)

    window._on_guided_discover_rois()
    # The busy state names the first phase before any worker output.
    assert window._guided_discovery_summary_label.text() == "Checking data format…"
    _pump(window, qapp)

    seen = [text for text, _thread in captured]
    assert any("data format" in text.lower() for text in seen), seen
    assert any("recording structure" in text.lower() for text in seen), seen
    assert any("continuous recording" in text.lower() for text in seen), seen
    for text in seen:
        for forbidden in ("RunSpec", "parser", "%", "Traceback", "csv"):
            assert forbidden not in text
    # Completion replaces the busy text with the settled summary.
    assert "continuous recording" in (
        window._guided_discovery_summary_label.text().lower()
    )


def test_phase_from_a_superseded_request_cannot_overwrite_settled_text(window):
    window._guided_roi_discovery_running = False
    window._guided_discovery_summary_label.setText("Settled summary.")

    window._on_guided_discovery_phase_changed("Checking recording structure…")

    assert window._guided_discovery_summary_label.text() == "Settled summary."


def test_phase_text_is_applied_on_the_gui_thread(window, qapp, tmp_path):
    """The phase signal is connected as a bound method, so Qt queues it to the
    GUI thread; the status label is never written from the worker."""
    import threading

    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    captured = _capture_status_text(window)

    window._on_guided_discover_rois()
    _pump(window, qapp)

    threads = {thread for _text, thread in captured}
    assert threads
    assert threads == {threading.get_ident()}


# ---------------------------------------------------------------------------
# Explicit choices are unchanged
# ---------------------------------------------------------------------------


def test_explicit_continuous_skips_structure_classification(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, fmt="rwd", structure="continuous")
    monkeypatch.setattr(
        main_window_module,
        "_guided_probe_intermittent_rwd_structure",
        lambda *a, **k: pytest.fail(
            "structure classification ran for an explicit choice"
        ),
    )
    _forbid_full_intermittent(window, monkeypatch)

    window._on_guided_discover_rois()
    _pump(window, qapp)

    assert window._discovery_cache["acquisition_mode"] == "continuous"


def test_explicit_intermittent_still_uses_the_existing_path(
    window, tmp_path
):
    folder = _intermittent_folder(tmp_path / "sessions")
    _select_data(window, folder, fmt="rwd", structure="intermittent")
    snapshot = window._snapshot_guided_discovery_inputs()
    runner = window._guided_discovery_runner_for_snapshot(snapshot)
    assert getattr(runner, "__name__", "") == "run_intermittent"
