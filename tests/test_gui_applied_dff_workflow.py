import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _make_phasic_out(root: Path, rois: list[str]) -> Path:
    phasic_out = root / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True, exist_ok=True)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([roi.encode("utf-8") for roi in rois]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"chunk0.csv", b"chunk1.csv"]))
        for roi in rois:
            roi_group = h5.create_group(f"roi/{roi}")
            for chunk_id in (0, 1):
                grp = roi_group.create_group(f"chunk_{chunk_id}")
                grp.create_dataset("time_sec", data=np.asarray([0.0, 1.0]))
                grp.create_dataset("dff", data=np.asarray([0.1, 0.2]))
    (phasic_out / "run_metadata.json").write_text("{}", encoding="utf-8")
    qc = phasic_out / "qc"
    qc.mkdir()
    (qc / "qc_summary.json").write_text("{}", encoding="utf-8")
    return phasic_out


def _combo_texts(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _strategy_combo(window, row: int):
    return window._applied_dff_table.cellWidget(row, 4)


def _table_text(window, row: int, col: int) -> str:
    item = window._applied_dff_table.item(row, col)
    return item.text() if item is not None else ""


def test_applied_dff_loads_actual_phasic_cache_rois(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["ROI_A", "Z9"])

    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._on_applied_dff_load_rois()

    assert window._applied_dff_table.rowCount() == 2
    assert [window._applied_dff_table.item(i, 1).text() for i in range(2)] == ["ROI_A", "Z9"]
    assert [window._applied_dff_row_included(i) for i in range(2)] == [True, True]


def test_applied_dff_missing_phasic_cache_refuses(window, tmp_path):
    missing = tmp_path / "not_phasic_out"
    missing.mkdir()
    window._applied_dff_phasic_out_edit.setText(str(missing))

    with pytest.raises(ValueError, match="Missing required phasic cache"):
        window._validate_applied_dff_phasic_out()


def test_applied_dff_manifest_generation_requires_explicit_supported_strategies(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2"])
    output_root = tmp_path / "applied_out"
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()

    first_combo = _strategy_combo(window, 0)
    second_combo = _strategy_combo(window, 1)
    assert "auto" not in _combo_texts(first_combo)
    first_combo.setCurrentText("dynamic_fit")
    second_combo.setCurrentText("signal_only_f0")

    manifest = Path(window._write_applied_dff_manifest())
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))

    assert rows == [
        {"roi": "R1", "strategy": "dynamic_fit"},
        {"roi": "R2", "strategy": "signal_only_f0"},
    ]
    assert manifest.is_relative_to(output_root)


def test_applied_dff_blank_strategy_refuses_before_batch(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1"])
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(tmp_path / "out"))
    window._on_applied_dff_load_rois()

    with pytest.raises(ValueError, match="Select dynamic_fit or signal_only_f0"):
        window._applied_dff_selected_manifest_rows()


def test_applied_dff_manifest_includes_only_checked_rois(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2", "R3"])
    output_root = tmp_path / "out"
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()

    window._set_applied_dff_row_included(0, False)
    _strategy_combo(window, 1).setCurrentText("dynamic_fit")
    _strategy_combo(window, 2).setCurrentText("signal_only_f0")

    manifest = Path(window._write_applied_dff_manifest())
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))

    assert rows == [
        {"roi": "R2", "strategy": "dynamic_fit"},
        {"roi": "R3", "strategy": "signal_only_f0"},
    ]
    assert _table_text(window, 0, 6) == "omitted"


def test_applied_dff_unchecked_rows_do_not_require_strategy(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2"])
    output_root = tmp_path / "out"
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()

    window._set_applied_dff_row_included(0, False)
    _strategy_combo(window, 1).setCurrentText("dynamic_fit")

    rows = window._applied_dff_selected_manifest_rows()

    assert rows == [{"roi": "R2", "strategy": "dynamic_fit"}]
    assert _table_text(window, 0, 6) == "omitted"


def test_applied_dff_checked_row_without_strategy_refuses(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2"])
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(tmp_path / "out"))
    window._on_applied_dff_load_rois()
    window._set_applied_dff_row_included(0, False)

    with pytest.raises(ValueError, match="Select dynamic_fit or signal_only_f0 for ROI R2"):
        window._applied_dff_selected_manifest_rows()


def test_applied_dff_no_included_rows_refuses(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2"])
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(tmp_path / "out"))
    window._on_applied_dff_load_rois()
    window._on_applied_dff_exclude_all()

    with pytest.raises(ValueError, match="Include at least one ROI"):
        window._applied_dff_selected_manifest_rows()


def test_applied_dff_set_all_dynamic_fit_affects_checked_rows_only(window, tmp_path):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2"])
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._on_applied_dff_load_rois()
    window._set_applied_dff_row_included(1, False)

    window._on_applied_dff_set_all_dynamic_fit()

    assert _strategy_combo(window, 0).currentText() == "dynamic_fit"
    assert _strategy_combo(window, 1).currentText() == ""


@pytest.mark.parametrize(
    "relative_output",
    [
        ".",
        "applied_bad",
        "features",
        os.path.join("features", "applied_bad"),
        "phasic_trace_cache.h5",
    ],
)
def test_applied_dff_unsafe_output_root_refuses_before_manifest_write(window, tmp_path, relative_output):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1"])
    (phasic_out / "features").mkdir(exist_ok=True)
    output_root = phasic_out if relative_output == "." else phasic_out / relative_output
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()
    _strategy_combo(window, 0).setCurrentText("dynamic_fit")

    with pytest.raises(ValueError, match="separate from the source phasic_out"):
        window._write_applied_dff_manifest()

    if relative_output in {"applied_bad", os.path.join("features", "applied_bad")}:
        assert not output_root.exists()
    if output_root.is_dir():
        assert not (output_root / "gui_manifest").exists()
        assert not (output_root / "applied_dff_gui_provenance.json").exists()
    assert not (phasic_out / "gui_manifest").exists()
    assert not (phasic_out / "applied_dff_gui_provenance.json").exists()


def test_applied_dff_dry_run_calls_existing_batch_runner(window, tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1"])
    output_root = tmp_path / "out"
    calls = []

    def _fake_batch(phasic, *, manifest, output_root, dry_run=False, **kwargs):
        calls.append(
            {
                "phasic": phasic,
                "manifest": manifest,
                "output_root": output_root,
                "dry_run": dry_run,
                **kwargs,
            }
        )
        assert dry_run is True
        return {
            "dry_run": True,
            "n_manifest_rows": 1,
            "planned_rows": [{"roi": "R1", "strategy": "dynamic_fit"}],
        }

    monkeypatch.setattr("gui.main_window.run_applied_dff_batch", _fake_batch)
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()
    _strategy_combo(window, 0).setCurrentText("dynamic_fit")

    window._on_applied_dff_dry_run()

    assert window._applied_dff_dry_run_ok is True
    assert calls[0]["phasic"] == str(phasic_out.resolve())
    assert calls[0]["output_root"] == str(output_root.resolve())
    assert Path(calls[0]["manifest"]).exists()
    assert (output_root / "gui_manifest" / "explicit_applied_dff_manifest.csv").exists()
    assert (output_root / "applied_dff_gui_provenance.json").exists()
    assert window._applied_dff_table.item(0, 6).text() == "dry-run planned"


def test_applied_dff_dry_run_uses_only_included_rows(window, tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2", "R3"])
    output_root = tmp_path / "out"
    manifests = []

    def _fake_batch(phasic, *, manifest, output_root, dry_run=False, **kwargs):
        manifests.append(list(csv.DictReader(open(manifest, "r", encoding="utf-8"))))
        return {
            "dry_run": True,
            "n_manifest_rows": 2,
            "planned_rows": [
                {"roi": "R1", "strategy": "dynamic_fit"},
                {"roi": "R3", "strategy": "signal_only_f0"},
            ],
        }

    monkeypatch.setattr("gui.main_window.run_applied_dff_batch", _fake_batch)
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()
    _strategy_combo(window, 0).setCurrentText("dynamic_fit")
    window._set_applied_dff_row_included(1, False)
    _strategy_combo(window, 2).setCurrentText("signal_only_f0")

    window._on_applied_dff_dry_run()

    assert manifests == [
        [
            {"roi": "R1", "strategy": "dynamic_fit"},
            {"roi": "R3", "strategy": "signal_only_f0"},
        ]
    ]
    assert _table_text(window, 1, 6) == "omitted"
    assert _table_text(window, 0, 6) == "dry-run planned"
    assert _table_text(window, 2, 6) == "dry-run planned"


def test_applied_dff_batch_calls_existing_runner_and_parses_results(window, tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1"])
    output_root = tmp_path / "out"
    calls = []

    def _write_applied_summary(output_dir: Path):
        applied_dir = output_dir / "applied"
        applied_dir.mkdir(parents=True, exist_ok=True)
        (applied_dir / "applied_correction_summary.json").write_text(
            json.dumps(
                {
                    "applied_trace_warning_level": "none",
                    "applied_trace_review_required": False,
                    "applied_trace_flags": "",
                }
            ),
            encoding="utf-8",
        )

    def _fake_batch(phasic, *, manifest, output_root, dry_run=False, overwrite=False, **kwargs):
        calls.append({"dry_run": dry_run, "overwrite": overwrite})
        if dry_run:
            return {
                "dry_run": True,
                "n_manifest_rows": 1,
                "planned_rows": [{"roi": "R1", "strategy": "dynamic_fit"}],
            }
        output_dir = Path(output_root) / "R1_dynamic_fit"
        _write_applied_summary(output_dir)
        return {
            "dry_run": False,
            "summary_json": str(Path(output_root) / "batch" / "applied_dff_batch_summary.json"),
            "summary": {
                "batch_passed": True,
                "n_rows_completed": 1,
                "n_rows_failed": 0,
                "hdf5_modified_source_phasic_cache": False,
                "legacy_features_modified": False,
                "rows": [
                    {
                        "roi": "R1",
                        "strategy": "dynamic_fit",
                        "output_dir": str(output_dir),
                        "status": "completed",
                        "semantic_status": "pass",
                        "n_chunks_processed": 2,
                    }
                ],
            },
        }

    monkeypatch.setattr("gui.main_window.run_applied_dff_batch", _fake_batch)
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()
    _strategy_combo(window, 0).setCurrentText("dynamic_fit")

    window._on_applied_dff_dry_run()
    window._on_applied_dff_run_batch()

    assert calls == [{"dry_run": True, "overwrite": False}, {"dry_run": False, "overwrite": True}]
    assert window._applied_dff_table.item(0, 5).text() == "dynamic_fit"
    status = window._applied_dff_table.item(0, 6).text()
    assert "semantic=pass" in status
    assert "chunks=2" in status
    assert "warning=none" in status
    assert "source_phasic_cache_unchanged=True" in window._applied_dff_status_label.text()
    assert "legacy_features_unchanged=True" in window._applied_dff_status_label.text()
    assert (output_root / "applied_dff_gui_provenance.json").exists()


def test_applied_dff_strategy_selection_resets_when_phasic_out_changes(window, tmp_path):
    first = _make_phasic_out(tmp_path / "run1", ["R1"])
    second = _make_phasic_out(tmp_path / "run2", ["R2"])
    window._applied_dff_phasic_out_edit.setText(str(first))
    window._on_applied_dff_load_rois()
    _strategy_combo(window, 0).setCurrentText("dynamic_fit")
    assert window._applied_dff_table.rowCount() == 1

    window._applied_dff_phasic_out_edit.setText(str(second))

    assert window._applied_dff_table.rowCount() == 0
    assert window._applied_dff_dry_run_ok is False


def test_applied_dff_include_change_invalidates_successful_dry_run(window, tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path / "run", ["R1", "R2"])
    output_root = tmp_path / "out"

    def _fake_batch(phasic, *, manifest, output_root, dry_run=False, **kwargs):
        return {
            "dry_run": True,
            "n_manifest_rows": 2,
            "planned_rows": [
                {"roi": "R1", "strategy": "dynamic_fit"},
                {"roi": "R2", "strategy": "dynamic_fit"},
            ],
        }

    monkeypatch.setattr("gui.main_window.run_applied_dff_batch", _fake_batch)
    window._applied_dff_phasic_out_edit.setText(str(phasic_out))
    window._applied_dff_output_root_edit.setText(str(output_root))
    window._on_applied_dff_load_rois()
    _strategy_combo(window, 0).setCurrentText("dynamic_fit")
    _strategy_combo(window, 1).setCurrentText("dynamic_fit")
    window._on_applied_dff_dry_run()
    assert window._applied_dff_dry_run_ok is True

    window._set_applied_dff_row_included(1, False)

    assert window._applied_dff_dry_run_ok is False
    assert _table_text(window, 1, 6) == "omitted"
