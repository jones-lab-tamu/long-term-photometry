# Custom Tabular CSV Conversion Guide

## What this format is
- `custom_tabular` is a strict CSV import contract.
- It is intended for users who can export or convert data from another acquisition system into the required table shape.
- It is not native Doric, TDT, or generic TXT support.
- It does not infer arbitrary vendor column names.
- For synthetic RWD/NPM demo datasets intended for GUI walkthroughs and regression testing, see `docs/synthetic_demo_datasets.md`.

## Required file structure
- One CSV file represents one session/chunk.
- Multiple CSV files in the same input folder are treated as multiple sessions.
- One CSV containing multiple sessions is not currently a formal `custom_tabular` session model.
- Session identity is inferred from each CSV filename stem.

## Required columns

| Column pattern | Required? | Meaning | Notes |
|---|---|---|---|
| `time_sec` (or configured `custom_tabular_time_col`) | Yes | Shared sample-time axis in seconds-compatible units | Must be numeric, finite, strictly increasing |
| `<ROI>_iso` (or configured `custom_tabular_uv_suffix`) | Yes (per ROI) | Isosbestic/control channel for ROI | Must have a matching `<ROI>_sig` column |
| `<ROI>_sig` (or configured `custom_tabular_sig_suffix`) | Yes (per ROI) | Signal channel for ROI | Must have a matching `<ROI>_iso` column |
| Any other columns | Optional | Ignored by `custom_tabular` unless they match configured suffixes | Extra columns are not used for ROI/channel mapping |

- Time column name is configurable with `custom_tabular_time_col`.
- Channel suffixes are configurable with `custom_tabular_uv_suffix` and `custom_tabular_sig_suffix`.
- Every ROI base must provide both channels.
- Extra columns are ignored unless they match the configured suffix patterns.

## Time requirements
- Time values must be numeric, finite, and strictly increasing.
- Time may be absolute or relative, but values must be seconds-compatible.
- Loader normalization is relative to first sample (`t_rel = t_raw - t_raw[0]`).
- One shared time column is used for both signal and isosbestic/control channels.
- Per-channel timestamp columns are not part of the current `custom_tabular` contract.

## Signal/control requirements
- Signal and isosbestic/control columns must be numeric and finite.
- NaNs are not allowed in required time/signal/control columns.
- Values can be arbitrary fluorescence units if exported consistently.
- Signal/control channels must align to the shared time column.

## Multiple ROIs
- Each ROI is represented by paired columns.
- Example paired layout: `ROI_1_iso, ROI_1_sig, ROI_2_iso, ROI_2_sig`.
- ROI names are inferred from the shared base text before each suffix.

## Unsupported acquisition systems
- Data from Doric, TDT, or other systems can be used only after conversion into this `custom_tabular` CSV contract.
- This is conversion-through-CSV support, not native vendor parser support.
- Conversion must preserve time, signal, isosbestic/control, ROI/channel identity, and session boundaries.
- Vendor metadata not represented in columns is not interpreted by the importer.

## Minimal example

```csv
time_sec,ROI_1_iso,ROI_1_sig,ROI_2_iso,ROI_2_sig
0.0,1.000,1.550,0.900,1.350
0.1,1.010,1.570,0.915,1.365
0.2,1.025,1.590,0.930,1.385
```

Template/example files are provided in [`examples/custom_tabular/`](../examples/custom_tabular/).
These files are intentionally short contract examples, not realistic photometry datasets.

## Example folder layout

```text
example_custom_data/
  session_001.csv
  session_002.csv
  session_003.csv
```

Each file is one session/chunk.

## Validation and contract check
Use the committed strict short-session config file at `examples/custom_tabular/custom_tabular_example_config.yaml`.

```powershell
python tools/run_full_pipeline_deliverables.py --input examples/custom_tabular --out tutorial_outputs/custom_tabular_validate --config examples/custom_tabular/custom_tabular_example_config.yaml --format custom_tabular --mode both --sessions-per-hour 2 --validate-only --overwrite
```

Loader/contract check used in tests:

```powershell
python -m pytest -q tests/test_custom_tabular_examples.py
```

## Common errors and fixes
- Missing time column:
  - Error indicates missing required `custom_tabular_time_col`; rename the CSV column or set `custom_tabular_time_col`.
- Unpaired ROI columns:
  - Error indicates unmatched ROI pairs; ensure every `<ROI>_iso` has `<ROI>_sig` (and vice versa).
- Nonnumeric values:
  - Convert time/signal/control columns to numeric values before export.
- NaNs in required fields:
  - Fill or remove NaNs in required time/signal/control columns.
- Nonmonotonic timestamps:
  - Sort and clean timestamps so time is strictly increasing.
- Wrong format selected:
  - Use `--format custom_tabular` (auto-detect does not select this mode).
- Multiple sessions in one CSV:
  - Split into one file per session/chunk.
- Wrong suffixes:
  - Rename columns to match `_iso`/`_sig`, or configure `custom_tabular_uv_suffix` and `custom_tabular_sig_suffix`.

## What custom_tabular does not provide
- No native Doric/TDT/TXT parser.
- No automatic heuristic mapping from arbitrary CSV exports.
- No multi-session-in-one-file contract.
- No per-channel timestamp-alignment logic.
- No interpretation of vendor-specific metadata beyond tabular columns.
