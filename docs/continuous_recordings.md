# Continuous Recordings

Continuous recording support is available for supported continuous source formats. Current continuous paths include RWD and strict `custom_tabular`. NPM/interleaved continuous input is not currently implemented.

## Required settings

Key settings:
- `acquisition_mode: continuous`
- `continuous_window_sec`
- `continuous_step_sec`
- `allow_partial_final_window`
- source format: `rwd` or `custom_tabular`

In this version, `continuous_step_sec` must equal `continuous_window_sec`.

## How continuous data are loaded

Continuous sources are planned into fixed elapsed-time windows. Metadata scans are bounded, and window loading reads only the data needed for the current window where supported.

## GUI inspection

Long recordings can contain millions of points. The GUI therefore emphasizes:
- elapsed-time summary plots
- per-window tables
- full-trace overview plots that may be downsampled
- detailed inspection through generated window/cache outputs

Do not expect every raw point in a multi-day recording to be rendered interactively at once.

## Outputs

Typical continuous outputs include:
- continuous phasic and tonic HDF5 caches
- per-ROI continuous window summary tables
- elapsed-time summary plots under `<ROI>/summary/`
- full cached-trace overview plots under `<ROI>/summary/`
- standard `status.json`, `MANIFEST.json`, and `run_report.json`

## Limits

Continuous mode is designed for bounded memory behavior, but runtime still scales with recording duration, sampling rate, ROI count, and selected plotting outputs. Use validate-only planning before long runs.
