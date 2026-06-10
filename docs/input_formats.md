# Supported Input Formats

This page distinguishes native parser support from conversion-through-CSV support.

## Summary

| Format/system | Support status | Notes |
| --- | --- | --- |
| RWD | Native supported path | Expects RWD-style session folders with `fluorescence.csv`; configure time and channel suffixes. |
| NPM / Neurophotometrics | Native supported path for CSV/interleaved session inputs | Requires configured LED/time/channel columns. Continuous NPM is not currently implemented. |
| `custom_tabular` CSV | Strict supported conversion path | One CSV per session/chunk; one time column; paired ROI `_iso`/`_sig` columns by default. |
| Generic CSV | Not heuristic/native | Use only after converting to the strict `custom_tabular` contract. |
| Generic TXT | Not native | Convert to strict `custom_tabular` CSV first. |
| Doric | Not native | Possible only after user export/conversion to strict `custom_tabular` CSV. |
| Tucker Davis / TDT | Not native | Possible only after user export/conversion to strict `custom_tabular` CSV. |

## RWD

RWD input is handled by the RWD adapter and discovery code. A typical dataset contains one folder per recording/session and a `fluorescence.csv` file in each session folder. The config controls the time column and signal/reference suffixes, for example `rwd_time_col`, `uv_suffix`, and `sig_suffix`.

## NPM / Neurophotometrics

NPM input is supported for CSV-style interleaved session recordings with explicit LED/time/channel columns. Configure `npm_frame_col`, `npm_system_ts_col`, `npm_computer_ts_col`, `npm_led_col`, `npm_region_prefix`, `npm_region_suffix`, and `npm_time_axis` as needed.

Continuous NPM/interleaved input is not currently implemented in the continuous pipeline path.

## Strict custom_tabular CSV

`custom_tabular` is a strict conversion contract, not a heuristic CSV importer. It expects:
- one CSV file per session/chunk
- one shared numeric time column, default `time_sec`
- paired ROI columns, default suffixes `_iso` and `_sig`
- finite numeric channel values
- strictly increasing time values

See `docs/custom_tabular_conversion_guide.md` and `examples/custom_tabular/`.

## Sequential files and continuous recordings

Intermittent/session-based runs can use multiple sequential session files discovered from the input folder. Continuous mode uses fixed elapsed-time windows over supported continuous sources. Current continuous support is for RWD and strict `custom_tabular` paths.

## Unsupported native systems

Doric, TDT, generic TXT, and arbitrary CSV files are not parsed as native formats. Researchers can use those systems only if they export time, signal, isosbestic/control signal, ROI/channel identity, and session identity into the strict `custom_tabular` contract.
