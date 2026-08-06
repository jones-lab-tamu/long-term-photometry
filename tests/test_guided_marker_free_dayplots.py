"""Guided Results: marker-free copies of session-based dF/F day plots.

The action reuses the existing display-only marker-off rerender path
(``tools/plot_phasic_dayplot_bundle.py --hide-peak-markers``). It must never
rerun correction or detection, and must never disturb a canonical artifact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.main_window import MainWindow
from photometry_pipeline.config import Config

pytestmark = pytest.mark.usefixtures("no_real_modals")

MARKER_OFF_RELATIVE_DIR = os.path.join(
    "day_plots", "rerendered_display_variants", "dff_peak_markers_off"
)
_ROIS = ("Region0", "Region1")
_CHUNK_IDS = (0, 1, 2)
_FS_HZ = 1.0
_CHUNK_DURATION_SEC = 240.0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _write_png(path, width: int = 700, height: int = 1800) -> None:
    from PySide6.QtGui import QPixmap

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pix = QPixmap(width, height)
    pix.fill()
    assert pix.save(str(path))


def _write_phasic_cache(cache_path: Path) -> None:
    """A completed-run phasic cache carrying a resolvable correction reference."""
    t = np.arange(0.0, _CHUNK_DURATION_SEC, 1.0 / _FS_HZ)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray(list(_ROIS), dtype="S"))
        meta.create_dataset("chunk_ids", data=np.asarray(list(_CHUNK_IDS), dtype=int))
        for roi_index, roi in enumerate(_ROIS):
            for cid in _CHUNK_IDS:
                group = handle.create_group(f"roi/{roi}/chunk_{cid}")
                sig = 1.0 + 0.3 * np.sin(0.07 * t + roi_index) + 0.05 * cid
                uv = 0.4 * np.sin(0.07 * t + 0.3) + 0.2
                fit_ref = 0.9 + 0.001 * t
                group.create_dataset("time_sec", data=t)
                group.create_dataset("sig_raw", data=sig)
                group.create_dataset("uv_raw", data=uv)
                group.create_dataset("fit_ref", data=fit_ref)
                group.create_dataset("delta_f", data=sig - fit_ref)
                group.create_dataset("dff", data=100.0 * (sig - fit_ref) / fit_ref)


def _write_run_report(run_dir: Path, **run_context) -> None:
    context = {
        "run_type": "full",
        "run_profile": "full",
        "status": "success",
        "phase": "final",
    }
    context.update(run_context)
    (run_dir / "run_report.json").write_text(
        json.dumps(
            {
                "status": "success",
                "phase": "final",
                "run_context": context,
                "configuration": {},
                "analytical_contract": {},
            }
        ),
        encoding="utf-8",
    )


def _intermittent_run(tmp_path, *, sessions_per_hour: int | None = 1) -> Path:
    """A completed session-based run the Guided Results viewer can open."""
    run_dir = Path(tmp_path) / "run_complete"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True, exist_ok=True)

    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )
    if sessions_per_hour is None:
        _write_run_report(run_dir)
    else:
        _write_run_report(run_dir, sessions_per_hour=sessions_per_hour)

    config = Config(
        target_fs_hz=_FS_HZ,
        chunk_duration_sec=_CHUNK_DURATION_SEC,
        dynamic_fit_mode="global_linear_regression",
    )
    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=True)
    _write_phasic_cache(phasic_out / "phasic_trace_cache.h5")

    for roi in _ROIS:
        _write_png(run_dir / roi / "summary" / "phasic_correction_impact.png", 900, 420)
        (run_dir / roi / "tables").mkdir(parents=True, exist_ok=True)
        for day in (0, 1):
            _write_png(run_dir / roi / "day_plots" / f"phasic_dFF_day_{day:03d}.png")
            _write_png(run_dir / roi / "day_plots" / f"phasic_sig_iso_day_{day:03d}.png")
    return run_dir


def _load_guided_results(window, run_dir) -> None:
    window._current_run_dir = os.fspath(run_dir)
    assert window._guided_report_viewer.load_report(os.fspath(run_dir)) is True
    window._refresh_guided_dayplot_copy_availability()


def _digest_tree(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


def _capture_command(monkeypatch, *, returncode: int = 0, write_pngs: bool = True):
    captured: dict = {}

    def _fake_run(cmd, capture_output, text, cwd):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        out_dir = cmd[cmd.index("--output-dir") + 1]
        if write_pngs and returncode == 0:
            os.makedirs(out_dir, exist_ok=True)
            _write_png(os.path.join(out_dir, "phasic_dFF_day_000.png"))
            _write_png(os.path.join(out_dir, "phasic_dFF_day_001.png"))
        return type(
            "ProcResult",
            (),
            {"returncode": returncode, "stdout": "ok", "stderr": "boom"},
        )()

    monkeypatch.setattr(main_window_module._subprocess, "run", _fake_run)
    return captured


# ---------------------------------------------------------------------------
# 1. Eligibility
# ---------------------------------------------------------------------------


def test_action_is_enabled_for_a_completed_intermittent_run(window, tmp_path):
    _load_guided_results(window, _intermittent_run(tmp_path))
    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is True
    assert window._guided_marker_free_dayplot_btn.isEnabled()
    assert not window._guided_marker_free_dayplot_group.isHidden()
    assert "presentation copies" in message
    assert "unchanged" in message


def test_action_is_unavailable_when_no_results_are_loaded(window):
    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is False
    assert "Open a completed analysis" in message
    assert window._guided_marker_free_dayplot_group.isHidden()


def test_action_is_unavailable_for_continuous_results(window, tmp_path, monkeypatch):
    _load_guided_results(window, _intermittent_run(tmp_path))
    assert window._guided_marker_free_dayplot_readiness()[0] is True

    monkeypatch.setattr(
        window._guided_report_viewer, "is_continuous_recording_results", lambda: True
    )
    window._refresh_guided_marker_free_dayplot_availability()
    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is False
    assert "only for session-based recordings" in message
    assert not window._guided_marker_free_dayplot_btn.isEnabled()
    assert window._guided_marker_free_dayplot_group.isHidden()


def test_action_is_unavailable_when_no_region_is_selected(window, tmp_path, monkeypatch):
    _load_guided_results(window, _intermittent_run(tmp_path))
    monkeypatch.setattr(window._guided_report_viewer, "selected_region", lambda: "")
    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is False
    assert "Select a region" in message


def test_action_is_unavailable_when_canonical_dff_dayplots_are_missing(window, tmp_path):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    roi = window._guided_report_viewer.selected_region()
    for name in os.listdir(run_dir / roi / "day_plots"):
        if name.lower().startswith("phasic_dff_day_"):
            os.remove(run_dir / roi / "day_plots" / name)

    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is False
    assert f"Region {roi} has no saved dF/F day plots" in message


def test_action_is_unavailable_when_the_phasic_trace_cache_is_missing(window, tmp_path):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    os.remove(run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5")
    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is False
    assert "phasic dF/F trace data" in message


def test_action_is_unavailable_when_config_used_is_missing(window, tmp_path):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    os.remove(run_dir / "_analysis" / "phasic_out" / "config_used.yaml")
    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is False
    assert "analysis settings record" in message


def test_action_is_unavailable_when_sessions_per_hour_cannot_be_resolved(window, tmp_path):
    run_dir = _intermittent_run(tmp_path, sessions_per_hour=None)
    # MANIFEST/status carry no layout either.
    _load_guided_results(window, run_dir)
    ready, message = window._guided_marker_free_dayplot_readiness()
    assert ready is False
    assert "sessions-per-hour" in message


def test_action_is_disabled_while_a_run_is_executing(window, tmp_path, monkeypatch):
    _load_guided_results(window, _intermittent_run(tmp_path))
    assert window._guided_marker_free_dayplot_btn.isEnabled()
    monkeypatch.setattr(window._runner, "is_running", lambda: True)
    window._refresh_guided_marker_free_dayplot_availability()
    assert not window._guided_marker_free_dayplot_btn.isEnabled()


def test_ineligible_action_explains_itself_and_writes_nothing(window, tmp_path, monkeypatch):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    roi = window._guided_report_viewer.selected_region()
    os.remove(run_dir / "_analysis" / "phasic_out" / "config_used.yaml")

    shown: dict = {}
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _p, title, text, *a, **k: shown.update(title=title, text=text),
    )
    monkeypatch.setattr(
        main_window_module._subprocess,
        "run",
        lambda *a, **k: pytest.fail("an ineligible action must not launch the renderer"),
    )
    window._on_guided_create_marker_free_dayplots()
    assert "analysis settings record" in shown["text"]
    assert not os.path.isdir(run_dir / roi / MARKER_OFF_RELATIVE_DIR)


# ---------------------------------------------------------------------------
# 2. Command construction
# ---------------------------------------------------------------------------


def test_command_uses_the_existing_marker_off_bundle_contract(window, tmp_path, monkeypatch):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    roi = window._guided_report_viewer.selected_region()
    captured = _capture_command(monkeypatch)

    window._on_guided_create_marker_free_dayplots()

    cmd = captured["cmd"]
    assert cmd[1].endswith(os.path.join("tools", "plot_phasic_dayplot_bundle.py"))
    assert "--hide-peak-markers" in cmd
    assert "--write-dff-grid" in cmd
    assert "--no-write-sig-iso-grid" in cmd
    assert "--no-write-stacked" in cmd
    assert "--show-peak-markers" not in cmd
    assert cmd[cmd.index("--roi") + 1] == roi
    assert cmd[cmd.index("--analysis-out") + 1] == os.path.join(
        os.fspath(run_dir), "_analysis", "phasic_out"
    )
    out_dir = cmd[cmd.index("--output-dir") + 1]
    assert out_dir == os.path.join(os.fspath(run_dir), roi, MARKER_OFF_RELATIVE_DIR)
    assert "dff_peak_markers_on" not in out_dir
    assert cmd[cmd.index("--sessions-per-hour") + 1] == "1"


def test_command_matches_the_full_control_marker_off_command(window, tmp_path):
    """Both entry points must render the same figure from the same contract."""
    run_dir = _intermittent_run(tmp_path)
    ctx = {
        "sessions_per_hour": 2,
        "session_duration_s": 600.0,
        "timeline_anchor_mode": "fixed_daily_anchor",
        "fixed_daily_anchor_clock": "07:00",
        "run_profile": "full",
        "dff_render_mode": "qc",
    }
    cmd = window._build_marker_off_dff_dayplot_command(
        phasic_out=os.path.join(os.fspath(run_dir), "_analysis", "phasic_out"),
        roi="Region0",
        out_dir=os.path.join(os.fspath(run_dir), "Region0", MARKER_OFF_RELATIVE_DIR),
        ctx=ctx,
    )
    assert "--hide-peak-markers" in cmd
    assert cmd[cmd.index("--session-duration-s") + 1] == "600.0"
    assert cmd[cmd.index("--timeline-anchor-mode") + 1] == "fixed_daily_anchor"
    assert cmd[cmd.index("--fixed-daily-anchor-clock") + 1] == "07:00"
    assert cmd[cmd.index("--dff-render-mode") + 1] == "qc"
    assert cmd[cmd.index("--source-run-profile") + 1] == "full"


# ---------------------------------------------------------------------------
# 3. No analysis rerun / 4. Artifact isolation (real renderer subprocess)
# ---------------------------------------------------------------------------


def test_real_render_never_needs_features_and_leaves_outputs_unchanged(window, tmp_path):
    """Runs the real bundle subprocess.

    ``features.csv`` is deliberately absent: with markers hidden the bundle
    never opens it and never replays detection, so a successful render is
    direct evidence that no detection ran.
    """
    run_dir = _intermittent_run(tmp_path)
    features_path = run_dir / "_analysis" / "phasic_out" / "features" / "features.csv"
    assert not features_path.exists()

    _load_guided_results(window, run_dir)
    roi = window._guided_report_viewer.selected_region()
    variant_rel = f"{roi}/{MARKER_OFF_RELATIVE_DIR}".replace("\\", "/")

    def _outside_variant(tree: dict[str, str]) -> dict[str, str]:
        return {
            name: digest
            for name, digest in tree.items()
            if not name.startswith(variant_rel + "/")
        }

    before = _digest_tree(run_dir)
    assert not any(name.startswith(variant_rel + "/") for name in before)

    window._on_guided_create_marker_free_dayplots()

    status = window._guided_marker_free_dayplot_status_label.text()
    assert "Showing" in status, f"real render did not succeed: {status}"

    after = _digest_tree(run_dir)
    written = [name for name in after if name.startswith(variant_rel + "/")]
    assert written
    pngs = [name for name in written if name.lower().endswith(".png")]
    assert pngs
    assert all(Path(name).name.lower().startswith("phasic_dff_day_") for name in pngs)
    # The renderer's own settings sidecar is the only other file it writes, and
    # it stays inside the isolated variant directory.
    assert set(written) - set(pngs) <= {f"{variant_rel}/dayplot_feature_config.json"}
    # Nothing outside the marker-off variant directory moved.
    assert _outside_variant(after) == _outside_variant(before)
    for guarded in (
        "MANIFEST.json",
        "run_report.json",
        "status.json",
        "_analysis/phasic_out/config_used.yaml",
        "_analysis/phasic_out/phasic_trace_cache.h5",
        f"{roi}/day_plots/phasic_dFF_day_000.png",
        f"{roi}/day_plots/phasic_dFF_day_001.png",
    ):
        assert after[guarded] == before[guarded]
    assert not features_path.exists()


def test_copies_do_not_replace_canonical_dayplots(window, tmp_path, monkeypatch):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    roi = window._guided_report_viewer.selected_region()
    canonical_dir = run_dir / roi / "day_plots"
    before = {
        name: hashlib.sha256((canonical_dir / name).read_bytes()).hexdigest()
        for name in os.listdir(canonical_dir)
        if (canonical_dir / name).is_file()
    }
    _capture_command(monkeypatch)

    window._on_guided_create_marker_free_dayplots()

    after = {
        name: hashlib.sha256((canonical_dir / name).read_bytes()).hexdigest()
        for name in os.listdir(canonical_dir)
        if (canonical_dir / name).is_file()
    }
    assert after == before


# ---------------------------------------------------------------------------
# 5. Viewer behavior
# ---------------------------------------------------------------------------


def test_generated_copies_display_through_the_external_image_sequence(
    window, tmp_path, monkeypatch
):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    roi = window._guided_report_viewer.selected_region()
    _capture_command(monkeypatch)

    sequences: list[list[str]] = []
    original = window._guided_report_viewer.show_external_image_sequence

    def _spy(paths, **kwargs):
        sequences.append(list(paths))
        return original(paths, **kwargs)

    monkeypatch.setattr(
        window._guided_report_viewer, "show_external_image_sequence", _spy
    )
    window._on_guided_create_marker_free_dayplots()

    assert len(sequences) == 1
    variant_dir = os.path.join(os.fspath(run_dir), roi, MARKER_OFF_RELATIVE_DIR)
    assert all(os.path.dirname(path) == variant_dir for path in sequences[0])
    active = window._guided_report_viewer.active_image_path()
    assert os.path.realpath(active).startswith(os.path.realpath(variant_dir))
    status = window._guided_marker_free_dayplot_status_label.text()
    assert f"Showing {len(sequences[0])}" in status
    assert variant_dir in status
    assert "unchanged" in status


def test_canonical_results_display_is_unchanged_when_no_override_is_active(
    window, tmp_path
):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    active = window._guided_report_viewer.active_image_path()
    assert active.endswith(".png")
    assert os.path.isfile(active)
    assert "rerendered_display_variants" not in active


def test_action_does_not_report_showing_when_the_viewer_cannot_switch(
    window, tmp_path, monkeypatch
):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    _capture_command(monkeypatch)
    monkeypatch.setattr(
        window._guided_report_viewer,
        "show_external_image_sequence",
        lambda *a, **k: False,
    )
    window._on_guided_create_marker_free_dayplots()
    status = window._guided_marker_free_dayplot_status_label.text()
    assert "Showing" not in status
    assert "could not switch" in status
    assert "Created" in status


def test_render_failure_surfaces_the_real_error_and_claims_nothing(
    window, tmp_path, monkeypatch
):
    run_dir = _intermittent_run(tmp_path)
    _load_guided_results(window, run_dir)
    _capture_command(monkeypatch, returncode=1, write_pngs=False)
    shown: dict = {}
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        lambda _p, title, text, *a, **k: shown.update(title=title, text=text),
    )
    window._on_guided_create_marker_free_dayplots()
    assert "boom" in shown["text"]
    assert "Showing" not in window._guided_marker_free_dayplot_status_label.text()


def test_full_control_marker_off_rerender_still_uses_the_shared_command(
    window, tmp_path, monkeypatch
):
    """Regression for the extracted command builder.

    The Full Control rerender tests in tests/test_gui_tuning_workspace.py are
    pre-existing failures (their fixture cache predates the correction-reference
    requirement, so ``_on_open_results`` never reaches complete-state), so this
    drives the Full Control marker-off branch directly instead.
    """
    run_dir = _intermittent_run(tmp_path)
    window._current_run_dir = os.fspath(run_dir)
    assert window._report_viewer.load_report(os.fspath(run_dir)) is True
    window._is_complete_workspace_active = True
    window._dff_rerender_show_peak_markers_cb.setChecked(False)
    window._refresh_dff_dayplot_rerender_availability()
    assert window._dff_dayplot_rerender_readiness()[0] is True

    captured = _capture_command(monkeypatch)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        lambda _p, title, text, *a, **k: pytest.fail(f"{title}: {text}"),
    )
    window._on_rerender_dff_day_plots()

    cmd = captured["cmd"]
    roi = window._report_viewer.selected_region()
    assert "--hide-peak-markers" in cmd
    assert "--write-dff-grid" in cmd
    assert "--no-write-sig-iso-grid" in cmd
    assert "--no-write-stacked" in cmd
    assert cmd[cmd.index("--roi") + 1] == roi
    assert cmd[cmd.index("--output-dir") + 1] == os.path.join(
        os.fspath(run_dir), roi, MARKER_OFF_RELATIVE_DIR
    )
    assert captured["cwd"] == window._repo_root_dir()


def test_marker_free_action_is_independent_of_the_continuous_marker_on_action(
    window, tmp_path
):
    """Both Guided day-plot copy actions exist; an intermittent run enables only
    the marker-free one."""
    _load_guided_results(window, _intermittent_run(tmp_path))
    assert window._guided_marker_free_dayplot_btn.isEnabled()
    assert not window._guided_continuous_marker_dayplot_btn.isEnabled()
    continuous_reason = window._guided_continuous_marker_dayplot_readiness()[1]
    assert "only for continuous recordings" in continuous_reason
    # Only the action that applies to this recording type is offered.
    assert not window._guided_marker_free_dayplot_group.isHidden()
    assert window._guided_continuous_marker_dayplot_group.isHidden()


def test_both_dayplot_copy_groups_are_hidden_before_results_are_loaded(window):
    assert window._guided_marker_free_dayplot_group.isHidden()
    assert window._guided_continuous_marker_dayplot_group.isHidden()
