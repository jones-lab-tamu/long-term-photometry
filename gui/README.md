# GUI for Photometry Pipeline Deliverables

A PySide6 thin wrapper around `tools/run_full_pipeline_deliverables.py`.
The GUI collects run parameters, executes the pipeline via QProcess,
streams logs live, supports cancellation, and renders results from
`MANIFEST.json`.

## Install

```bash
pip install -r requirements_gui.txt
```

This installs PySide6 (optional dependency, not required for CLI usage).

## Run

```bash
python -m gui.app
```

## What the GUI does

1. **Collects inputs** -- input directory, output directory, config YAML, format, sessions/hour, etc.
2. **Validate Only** -- runs `--validate-only` to check inputs without creating output.
3. **Run Pipeline** -- starts the full deliverables pipeline, streams stdout/stderr live.
4. **Cancel** -- terminates the running pipeline (including child processes on Windows).
5. **Open Results...** -- loads a previously-completed output directory via MANIFEST.json.
6. **Results Browser** -- on success, loads `MANIFEST.json` and shows:
   - Summary tab with run metadata
   - Per-ROI tabs with clickable image thumbnails and CSV data tables

## Notes

- The GUI calls **only** `tools/run_full_pipeline_deliverables.py`.
- Results are rendered **only** from `MANIFEST.json`, no filename guessing.
- PySide6 is an optional dependency; CLI users do not need it.
