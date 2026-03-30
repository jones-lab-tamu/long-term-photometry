# GUI for Photometry Pipeline Deliverables

PySide6 desktop frontend for launching and reviewing full photometry runs.
It wraps `tools/run_full_pipeline_deliverables.py`, streams logs, supports cancellation, and loads completed results workspaces.

## Install

From repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_gui.txt
```

`requirements_gui.txt` includes base runtime requirements plus PySide6.

## Run

```bash
python -m gui.app
```

## Core GUI Workflow

1. Set input directory, output base directory, and config source.
2. Validate run settings.
3. Launch run.
4. Review outputs in results tabs (Verification, Tonic, Phasic Sig/Iso, Dynamic Fit, Phasic dFF, Phasic Stacked).
5. Optionally use post-run tuning panels.

## Notes

- Primary backend command is `tools/run_full_pipeline_deliverables.py`.
- Completed-run rendering is driven by run artifacts (`MANIFEST.json`, `run_report.json`, region folders).
- CLI-only users can install with `requirements.txt` instead.
