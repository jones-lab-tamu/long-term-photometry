# Long-Term Photometry Analysis

Long-Term Photometry Analysis is a desktop application for analyzing
long-duration fiber photometry recordings collected over hours to days. Its
Guided workflow is designed for scientists who may not write code: organize
recordings, compare correction approaches, review feature detection for each
ROI, run the analysis, and inspect the saved results.

Guided is the recommended workflow for new analyses. Full Control remains
available for expert users and backward-compatible workflows, but it is not
required for the ordinary scientist-facing path.

For a complete scientist-facing walkthrough, see the
[Guided Scientist Guide](docs/guided_scientist_guide.md).

## What the application does

- Analyzes repeated session-based recordings or one continuous recording.
- Supports multiple ROIs and places repeated recordings on a timeline.
- Reads RWD, Neurophotometrics, and CSV-file inputs through the Guided setup.
- Previews correction approaches and lets you inspect correction quality by ROI.
- Configures feature detection per ROI with Default or Custom settings.
- Shows a Review plan before the run and saves plots, tables, reports, and
  analysis settings for later inspection.
- Reopens completed results in the Review step.

## Supported data

| Input source | Repeated sessions | Continuous recording |
| --- | --- | --- |
| RWD | Supported | Supported |
| Neurophotometrics | Supported | Not currently supported |
| CSV files | Supported | Supported |

In Guided Select data, CSV input uses the column-mapping controls for the time,
signal, and reference columns for each ROI. It is not a heuristic importer for
arbitrary files.

When format and structure are set to automatic for CSV input, one candidate CSV
file defaults to a continuous recording and multiple candidate CSV files
default to repeated sessions. You can choose the recording structure yourself
when the files do not match that convention. Continuous Neurophotometrics input
is not currently supported.

## Guided workflow at a glance

The current Guided workflow presents these steps in order:

1. **Start** - Begin a new analysis or choose an existing workflow entry.
2. **Select data** - Choose the input and output locations and format, then map
   the data fields needed for the selected source.
3. **Recording structure** - Confirm whether the data are repeated sessions or
   one continuous recording and set the applicable timing information.
4. **Correction approach** - Compare available correction approaches and
   inspect their previews before continuing.
5. **Feature detection** - Review detection settings and previews for each ROI;
   use Default or Custom settings as needed.
6. **Review plan** - Check the complete analysis plan before committing to a
   run.
7. **Run** - Check that the setup is ready, then start the configured analysis.
8. **Review** - Open the saved results and inspect plots, tables, reports, and
   per-ROI outputs.

## Installation and launch on Windows

From the project folder, open PowerShell and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_gui.txt
python -m gui.app
```

`requirements_gui.txt` includes the core analysis dependencies from
`requirements.txt` and the PySide6 GUI dependency, so it is the single
requirements file needed for this application. The commands above were
verified in a fresh Windows virtual environment, including importing and
launching the GUI. This README does not claim an equivalent installation path
for macOS or Linux.

## Try a built-in demo

After launching the app, choose **Tools -> Generate Guided Demo Dataset**. The
dialog provides:

- **Intermittent CSV demo** - a repeated-session example.
- **Continuous CSV demo** - a one-file continuous-recording example.

Choose where to create the demo data, then select the generated folder in
Guided Select data and follow the normal workflow. These datasets are
synthetic demonstrations of the interface and pipeline, not biological data
or biological validation.

## Results

Completed runs save ordinary scientific outputs that can be inspected outside
the app, including:

- corrected traces for each ROI;
- tonic and phasic outputs where applicable;
- per-ROI plots and tables;
- day-level and summary plots;
- feature-event summaries;
- an analysis report and saved settings that document how the run was produced.

Supporting metadata are also saved for reproducibility.

Intermittent runs emphasize session and day-oriented views. Continuous runs
also produce fixed elapsed-time window summaries and full-trace overview plots;
long traces may be downsampled for display. Use **Review** to reopen completed
results after the run.

## Limitations

- Continuous Neurophotometrics recordings are not currently supported.
- The application expects supported input organizations and cannot guarantee
  that arbitrary photometry exports will load correctly.
- Scientists must review whether correction and feature-detection settings are
  appropriate for their data.
- Synthetic demos demonstrate the workflow and are not biological validation
  datasets.

## Further reading

For current format and analysis details:

- [Supported input formats](docs/input_formats.md)
- [Continuous recordings](docs/continuous_recordings.md)
- [Correction and dynamic fitting](docs/correction_and_dynamic_fit.md)
- [Event detection](docs/event_detection.md)
