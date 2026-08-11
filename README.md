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

## Before you install

You need:

- Python 3.10 or newer. The application and its GUI libraries run inside the
  private environment created in the steps below.
- Git if you use the cloning steps. Git is not needed to launch the application
  if you download the repository as a ZIP file instead.
- An internet connection while you download the application and its Python
  libraries.

The GUI dependency, PySide6, and the analysis libraries are installed together
from `requirements_gui.txt`. You do not need to install them separately.

To check the commands before starting:

**Windows Command Prompt**

```cmd
python --version
git --version
```

The first command should report Python 3.10 or newer. The second should report
a Git version. If Windows says that `python` is not recognized but `py --version`
works, use `py -3` in place of `python` in the Windows commands below. If either
command is missing, install Python from [python.org](https://www.python.org/)
and, for the cloning route, Git from [git-scm.com](https://git-scm.com/), then
open a new shell and check again.

**macOS Terminal**

```bash
python3 --version
git --version
```

The first command should report Python 3.10 or newer. The second should report
a Git version. If a command is missing, install the missing software, open a new
Terminal window, and check again.

The instructions below use Git. If you choose **Code -> Download ZIP** on
GitHub instead, unzip the download and use the extracted folder in place of
`long-term-photometry` when you change directories.

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

## Install and run on Windows

The Windows path below uses **Command Prompt**, not PowerShell. Use a new
Command Prompt window so that each command starts from a clear prompt.

1. Open Command Prompt. Press the Windows key, type `Command Prompt`, and
   press Enter. A window with a line ending in `>` should appear.
2. Move to your Documents folder:

   ```cmd
   cd /d "%USERPROFILE%\Documents"
   ```

3. Download the application from GitHub:

   ```cmd
   git clone https://github.com/jones-lab-tamu/long-term-photometry.git
   ```

   Git will print download and progress messages. When it finishes, the
   command should return to the prompt and a new folder named
   `long-term-photometry` should be in Documents.

4. Enter that folder:

   ```cmd
   cd long-term-photometry
   ```

   This is the application folder. The remaining commands must be run here.

5. Create a private Python environment for this application:

   ```cmd
   python -m venv .venv
   ```

   A virtual environment is a small, private set of Python libraries kept in
   this application folder. It prevents this application's libraries from
   interfering with other Python programs on your computer.

6. Activate it:

   ```cmd
   .venv\Scripts\activate
   ```

   Activation succeeded when `(.venv)` appears at the beginning of the
   Command Prompt line. Leave it active for the next commands.

7. Upgrade the installer and install the application libraries:

   ```cmd
   python -m pip install --upgrade pip
   python -m pip install -r requirements_gui.txt
   ```

   Installation can print many lines and may take several minutes. This one
   requirements file includes both the analysis libraries and the PySide6 GUI
   library.

8. Start the application:

   ```cmd
   python -m gui.app
   ```

   A window titled **Long-Term Photometry Analysis** should open. The
   **Guided Workflow** tab is selected by default. On the **Start** step,
   choose **Set up new analysis** to begin a new run. The Command Prompt may
   remain busy while the window is open; that is expected.

If a command ends with an actual error message instead of returning to the
prompt normally, stop and ask your lab's support person before continuing with
later steps.

## Install and run on macOS

The repository has the same Python requirements and GUI entry point on macOS.
The path below is supported by code and dependency inspection, but it has not
been run end-to-end on a Mac by this documentation pass.

1. Open Terminal. Press Command-Space, type `Terminal`, and press Return.
2. Move to your Documents folder:

   ```bash
   cd "$HOME/Documents"
   ```

3. Download the application:

   ```bash
   git clone https://github.com/jones-lab-tamu/long-term-photometry.git
   ```

   Git will print download and progress messages. When it finishes, the
   command should return to the prompt and create a `long-term-photometry`
   folder in Documents.

4. Enter the application folder:

   ```bash
   cd long-term-photometry
   ```

5. Create and activate the private Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   When activation succeeds, `(.venv)` appears at the beginning of the
   Terminal prompt. This environment keeps the application's libraries
   separate from other Python programs.

6. Upgrade the installer and install the GUI libraries:

   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements_gui.txt
   ```

   Installation can print many lines and may take several minutes.

7. Start the application:

   ```bash
   python3 -m gui.app
   ```

   A window titled **Long-Term Photometry Analysis** should open with
   **Guided Workflow** selected. On **Start**, choose **Set up new analysis**.
   The Terminal may remain busy while the application window is open; that is
   expected.

If a command ends with an actual error message instead of returning to the
prompt normally, stop and ask your lab's support person before continuing with
later steps.

On macOS, the application's appearance is under active compatibility testing.
Some Macs may open the application in a dark appearance that is difficult to
read. This guide does not prescribe a system-setting workaround; report the
appearance problem to your lab's support person.

## Run the application again later

Installation is a one-time step. On a later day, open a new shell, return to
the application folder, activate the existing environment, and launch the
application.

**Windows Command Prompt**

```cmd
cd /d "%USERPROFILE%\Documents\long-term-photometry"
.venv\Scripts\activate
python -m gui.app
```

**macOS Terminal**

```bash
cd "$HOME/Documents/long-term-photometry"
source .venv/bin/activate
python3 -m gui.app
```

Seeing `(.venv)` at the beginning of the prompt means the existing environment
is active. Do not repeat the clone, environment-creation, or dependency-install
steps unless your lab's support person asks you to repair the installation.

## First launch and where to go next

There is no separate first-run setup screen. The application opens directly to
the Guided Workflow, with the **Start** step selected. Choose **Set up new
analysis**, then follow the [Guided Scientist Guide](docs/guided_scientist_guide.md)
through data selection, review, running the analysis, and reopening results.
To inspect a completed run instead, choose **Open Results...** on the Start step.

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
- slow-signal and event-signal outputs where applicable;
- per-ROI plots and tables;
- day-level and summary plots;
- detected-event summaries;
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
- This README documents Windows and macOS installation paths. Linux is not
  claimed or documented here.

## Further reading

For current format and analysis details:

- [Supported input formats](docs/input_formats.md)
- [Continuous recordings](docs/continuous_recordings.md)
- [Correction and dynamic fitting](docs/correction_and_dynamic_fit.md)
- [Event detection](docs/event_detection.md)
