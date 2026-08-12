# Guided Scientist Guide

This is the canonical, scientist-facing guide to the Guided workflow in Long-Term
Photometry Analysis. It describes the controls that are visible in the current
application and the decisions that remain with the scientist. This guide
explains how to use Guided to analyze long-duration fiber photometry recordings.

Guided is the recommended workflow for a new analysis. Full Control remains
available for expert users and backward-compatible workflows, but it is outside
the ordinary path described here.

After installation, programming is not required for ordinary Guided use. The
application opens directly to the **Guided Workflow** tab with no separate
first-run setup screen. Start a new analysis from the **Start** step.

## 1. About this guide

Guided takes you from a source recording to a reviewable completed run in a
fixed sequence:

1. Start
2. Select data
3. Recording structure
4. Correction approach
5. Feature detection
6. Review plan
7. Run
8. Review

The sequence is staged. Early steps identify the data and timeline.
Middle steps expose the choices that most affect interpretation. The Review plan
step gathers those choices before analysis starts. Review opens outputs only after a
completed run is available.

Keep three kinds of responsibility separate while you work:

- App determines: the application can detect a format, recording organization,
  available ROIs, and some timing information from the folder you choose.
- You provide: you select the input and output folders, map CSV columns, correct
  any timing or structure choice, choose ROIs, and confirm the settings that
  should be used.
- You judge: you decide whether the recording organization, correction behavior,
  detected events, and final outputs make scientific sense for the experiment.

### Built-in demo

The built-in demo is synthetic, not biological validation. To try it,
choose Tools -> Generate Guided Demo Dataset, choose where to create the data,
then select the generated folder in Select data. Its choices are Intermittent
CSV demo for repeated sessions and Continuous CSV demo for one continuous
recording.

For the built-in Intermittent CSV demo, Guided pre-fills **Sessions per hour**
with 2 and **Session duration (s)** with 600. Review and confirm those values;
they remain editable. Ordinary CSV data do not receive this prefill, so enter
repeated-session timing manually.

A passed setup check means that the selected analysis is software-ready in
Guided. It does not make the data scientifically valid and does not replace
inspection of the recording or the results.

## 2. Supported data and key terms

Guided supports the following combinations:

| Input source | Repeated sessions | Continuous recording |
| --- | --- | --- |
| RWD | Supported | Supported |
| Neurophotometrics | Supported | Not currently supported |
| CSV files | Supported | Supported |

Repeated sessions are recordings saved as separate session units that should be
placed in chronological order. Continuous recording is one uninterrupted
recording that Guided divides into equal analysis windows for the run.

Neurophotometrics is supported for genuinely repeated/session-based recordings.
Continuous Neurophotometrics recordings are not currently supported in Guided.
Do not force a continuous recording through the repeated-session option.

RWD can be used for either repeated sessions or one continuous recording when
the selected folder follows the source organization recognized by the
application. CSV files can also be used for either structure. In Guided, you
identify the time, signal, and reference columns for each ROI.

An ROI is one signal/reference pair analyzed separately. Depending on the
system, it may correspond to a fiber, recording site, region label, or channel.
Guided does not infer anatomical meaning. The signal channel is
the measured channel of interest. The reference or control channel is the
comparison channel used by a reference-based correction. Correction is a
signal-processing step that reduces shared structure; it does not guarantee
that every artifact is removed. A detected feature is a signal excursion that
matches the selected criteria, not automatically a confirmed biological event.
Timeline placement determines where repeated recordings appear across hours and
days.

## 3. What to prepare

Gather the source folder and decide where the results should be saved. Choose a
writable output location with enough space for corrected traces, plots, tables,
reports, and selected analysis outputs. Do not use the source folder as a casual
scratch location for results. Keeping source and results separate makes it
easier to reopen a completed run and preserve the original recording.

Before starting, identify the recording structure, intended ROIs, CSV signal
and reference/control columns, repeated-session timing or continuous window
length, and the correction and event-detection behavior you expect to review.

Do not continue if the detected source, ROI list, CSV mapping, recording
structure, or timing does not match the experiment. Keep original input files
unchanged while analysis runs.

## 4. Start a Guided analysis

Use the README installation and launch instructions. The application opens with
the **Guided Workflow** tab selected, and the **Start** step gives two entry
points:

- **Set up a new analysis**: choose this for a new analysis from raw or input
  data. This is the normal starting point.
- **Open Results...**: choose this to inspect a completed run that was already
  saved. It does not start a new analysis.

Choose **Set up a new analysis** to configure or run an analysis. Choose
**Open Results...** to inspect a finished run. Opening completed results does
not configure a new analysis, check a new setup, or launch another run.

If you begin a new analysis, Guided moves to Select data. If you open completed
results, choose the completed-results folder and wait for Review to load.

You can revisit earlier steps before Run. Changing earlier settings may require
the setup or readiness check again.

## 5. Select data

Select data establishes the source, recording organization, folders, and ROIs.
The visible controls are:

- Format: Auto, RWD, NPM, or CSV files.
- Recording structure: Detect automatically, Intermittent/session-based
  recording, or Continuous/one long recording. This choice describes how the
  files were saved, not how long the experiment lasted.
- Input folder and Browse...
- Output folder and Browse...
- Select ROIs...

The **Continue to Recording Structure** button becomes available after you
have selected an input folder, selected a usable output folder, completed any
required CSV mapping and order confirmation, run **Select ROIs...**, and
included at least one ROI.

### Choose the input folder

Select the folder containing the recordings you want to analyze. For RWD, choose
the folder organized as supported recording/session data. For Neurophotometrics,
choose the folder containing supported repeated-session exports. For CSV, choose
the folder containing the CSV recordings. The application inspects the selected
folder to identify the recording source and available ROIs.

Use **Detect automatically** when you want the application to inspect the
selected folder. If you already know the organization, choose **Intermittent/session-based recording** or **Continuous/one long recording** yourself. The
choice describes how the data were saved, not how long the experiment lasted.

### Choose the output folder

Choose the output folder before the run. Review plan repeats the destination
for verification. Guided creates a new run folder inside the selected output
folder. After a successful run, **Open results folder** opens that run folder.
If the destination is wrong, return to Select data and change it. Input files
are not overwritten.

### CSV column mapping

CSV input can represent either repeated sessions or one continuous recording.
In Guided, you identify the time, signal, and reference columns for each ROI.

When CSV files are selected or detected, Guided shows the Interpret CSV columns
area. Map the fields that the application needs:

- Time column: the column containing elapsed time.
- Time units: seconds or milliseconds.
- ROI name: the label you want to see in the workflow and results.
- Signal column: the measured signal for that ROI.
- Reference column: the isosbestic or control channel used for reference-based
  correction.

Use Add ROI for each additional signal/reference pair. Check every pairing and
do not map a timestamp, status flag, or unrelated numeric column as the signal.

For example, real column names can differ, but the conceptual mapping is:

| Meaning | Example |
| --- | --- |
| Time | Time |
| ROI 1 signal | GCaMP_1 |
| ROI 1 reference | Isosbestic_1 |
| ROI 2 signal | GCaMP_2 |
| ROI 2 reference | Isosbestic_2 |

Automatic CSV structure has a specific current behavior. With format and
structure left automatic, one CSV recording file defaults to a continuous
recording and multiple CSV recording files default to repeated sessions. You
can choose the recording structure yourself when the files do not match that
convention. A continuous CSV selection should identify one recording CSV. A
folder containing multiple CSV session files should normally be treated as
repeated sessions.

For repeated CSV sessions, Guided displays the discovered CSV session order.
Inspect that order and select "I confirm this is the intended recording order."
Chronological order matters: an incorrect order changes the timeline even when
the column mappings are correct.
The mapping is used across selected session files, so confirm their column
layout is compatible.

### ROIs

Select ROIs lists the ROIs found in the recording. Choose the ROIs you want
included in the analysis.

Run **Select ROIs...**, check the ROIs to include, and then continue when the
source, structure, ROI list, output folder, and required CSV order confirmation
match the intended dataset. If the button remains unavailable, read the short
message below it; it names the missing item.

## 6. Confirm recording structure and timing

The recording-structure choice is made on **Select data**. This step keeps the
timing and timeline settings that depend on that choice. Detected timing and
**Use detected timing** are starting points; check them against the acquisition
schedule.

### For repeated sessions

For repeated-session recording, review:

- Sessions per hour
- Session duration (s)

These positive values place sessions on the timeline and must agree with the
recording protocol.

### If a repeated session is unusable

For repeated-session RWD recordings, Guided can offer recovery after one
specific session fails during processing while the remaining sessions are
usable. The warning identifies the failed session. To continue, explicitly choose
**Continue with this session missing**. This approves only that session
for the current recording. The unusable session is recorded as missing; its
original time position is preserved, so the interval remains blank and later
sessions do not shift earlier. The completed analysis reports the missing
session. Then run **Check my setup** again before rerunning. Choosing **Return
to setup** records no approval. This continuation path is not available for
Neurophotometrics or CSV recordings.

Guided can also recognize some readable RWD sessions that are shorter than the
expected recording interval.

Guided stops instead of guessing when it cannot confidently determine which
session failed, the correct session order, or the failed session's timeline
position.

### For continuous recordings

For continuous data, review:

- Continuous analysis window (s)
- Allow partial final analysis window

The window is the length of each analysis unit. Choose a length that supports
the question and leaves enough samples for correction and event checks. Decide
whether a shorter final window is interpretable before enabling it. Guided
treats the source as one continuous trace, not separate acquisition sessions.

### Timeline placement

Timeline placement controls how recordings are displayed in time. The Time
display choices are:

- Fixed daily anchor: place each day relative to the selected circadian-day
  start.
- Civil clock: use actual clock time with midnight as the day boundary.
- Elapsed from first recording: start the plot at the first recording.

When needed, enter Start of plotted day as HH:MM. Gaps, day boundaries, and
session placement should agree with the schedule.

### Clock time at recording start

Repeated-session and continuous RWD can prefill the visible Clock time at
recording start field from a validated recording timestamp. The field remains
editable.

For repeated Neurophotometrics, Guided uses its validated first recording
timestamp in the analysis plan when Civil clock or Fixed daily anchor placement
is used. It does not use the same visible RWD prefill control. CSV does not
provide an absolute recording-start timestamp that Guided can use in the
supported workflow, so Civil clock or Fixed daily anchor placement requires
manual clock entry. Elapsed from first recording does not require an absolute
recording-start clock.

Before continuing, verify the structure, order, recording start, plausible total
duration, and session count or continuous duration against the experimental
record.

Continue when required timing fields and timeline choice are valid. If you use
the repeated-session structure, confirm that the session order and timing
match the experimental record. **Continue to Correction Approach** becomes
available only after the required timing and timeline fields are valid. For
continuous data, the app also checks that the selected recording can be divided
into the requested analysis windows; if it cannot, choose a shorter window or
correct the data choice named in the message.

## 7. Review the correction approach

### What correction means

Correction reduces shared signal/reference structure before event
interpretation. This step presents the available correction methods, a local
preview, and per-ROI strategy confirmation; it does not start the final
analysis.

The available approaches are:

- Robust Global Event-Reject Fit is the recommended starting point. It fits the
  reference relationship while
  excluding event-like periods; compare its local preview with the other
  approaches before confirming a strategy.
- Adaptive Event-Gated Fit is included for recordings where the
  signal-reference relationship may change over time. Compare its local preview
  with the other approaches before confirming a strategy.
- Global Linear Regression is included as a baseline comparison and is generally
  not recommended for long-duration recordings. Compare its local preview with
  the other approaches before confirming a strategy.
- Signal-Only F0 is available for local comparison and can be selected as an
  explicit per-ROI correction strategy in Guided. The local preview is
  diagnostic evidence for comparison. Final analysis
  recomputes the correction across the complete recording and does not reuse
  the preview trace. Signal-Only F0 does not use the reference channel for
  correction. Select it independently for each ROI; mixed per-ROI plans are
  supported.

### Read the preview

The **Preview correction methods** section compares selected methods on one
recording segment. Choose an **ROI**, choose a **Preview segment**, select the
methods to compare, and choose **Generate correction preview**. **Customize**
is available for the Robust and Adaptive methods when you need to change their
preview settings for the selected ROI. For continuous recordings, the segment
choices are analysis windows.

The preview is diagnostic evidence for the selected segment; it does not
modify source data or start the final analysis. If you change the ROI, segment,
or selected methods after generating a preview, generate the preview again
before using it.

### Compare representative windows

Use the Preview segment choice to inspect more than one part of a long
recording when those segments are available. One apparently good segment may
not represent the full recording.

### What to look for

Judge the preview scientifically. Look for shared nuisance structure to be
reduced without the correction following the biological response. Compare it
with the raw signal, reference/control channel, and experimental context. Be
cautious when the correction flattens most plausible signal variation,
introduces inverted responses, exaggerates features that are not apparent in
the source channels, or behaves very differently across representative preview
segments. A finished preview is not by itself evidence that correction is
appropriate.

### Choose an approach for each ROI

After comparing the preview evidence, use the **2. Choose correction
strategy** section to confirm one correction strategy for each included ROI.
Choose the ROI, review its preview segment, choose a **Strategy for this ROI**,
and select **Confirm method**. If the acknowledgement appears, check **I
reviewed the diagnostic evidence...** before confirming. Repeat until the
**Strategies by included ROI** list shows a current choice for every included
ROI.

The method selected for preview is not automatically final for every ROI.
Guided Run recomputes the confirmed correction across the complete selected
recording set, not only the preview segment. Different ROIs may use different
correction strategies. If the signal/reference mapping appears wrong or the
corrected trace remains scientifically implausible, fix the mapping or
reconsider the correction choice before continuing. **Continue to Feature
Detection** remains unavailable until the current preview evidence and
strategy choice have been confirmed for every included ROI.

## 8. Review Feature Detection

### What a detected feature means

Feature Detection controls how events are identified in each ROI. These settings
do not run analysis or write files. Use the preview and recording context to
judge plausibility.

### Adjust the exposed Guided settings

The **Default feature detection settings** section exposes the following
controls. The same fields can be customized for one ROI:

- **Event signal**: Chooses the dF/F or delta-F trace used for peak detection
  and event-derived metrics. Changing this choice can change both event
  locations and scale.
- **Signal excursion polarity**: Chooses positive, negative, or both
  directions for excursions. Use the expected response direction when it is
  known; changing it changes which signed responses are considered.
- **Peak threshold method**: Chooses a mean-and-standard-deviation, median-and-
  MAD, percentile, or absolute cutoff. Use the method that fits the trace's
  noise and scale; it changes how the detection threshold is derived.
- **Peak threshold k**: Sets the multiplier for methods that use k. It is
  relevant to mean-and-standard-deviation and median + MAD thresholding; a
  larger value requires a larger excursion relative to variability, while a
  smaller value admits more.
- **Peak threshold percentile**: Sets the data-derived quantile cutoff for
  percentile thresholding. It is relevant only to that method and changes the
  cutoff as the trace distribution changes.
- **Peak threshold absolute**: Sets a fixed signal-unit cutoff for absolute
  thresholding. Use it when the signal scale is known and stable; changing it
  changes the fixed amplitude requirement.
- **Peak min distance (sec)**: Sets the minimum time between detected peaks.
  Use it when adjacent detections could be the same event; increasing it
  filters or merges more closely spaced detections.
- **Peak min prominence k**: Sets how far a peak must stand out relative to
  robust noise. It is relevant when small fluctuations are common; increasing
  it rejects less-prominent excursions, while decreasing it admits more.
- **Peak min width (sec)**: Sets the minimum event duration. Use it when
  very brief excursions may be artifacts; increasing it rejects narrower
  detections.
- **Peak pre-filter**: Chooses optional preprocessing before peak detection.
  It is relevant when high-frequency noise obscures the trace; changing it
  changes the signal seen by the peak finder and may also soften narrow events.
- **Event AUC baseline**: Chooses the reference baseline for event-area
  summaries. Use it when interpreting event area or polarity; changing it
  changes the area relative to that baseline, not necessarily the event
  boundaries.

Only the threshold field used by the selected method affects that method's
cutoff. None of these settings is automatically better at a higher or lower
value. For technical definitions, see [Event detection](event_detection.md).

### Default versus Custom

The **Feature detection per ROI** table shows **Default** or **Custom** and the
settings used during Run. Begin with Default, then choose **Customize** for an
ROI when its noise, polarity, or response shape needs different treatment.
Custom changes that ROI alone. Other ROIs remain on Default. After a Custom
choice has been made, its button changes to **Edit**; **Reset to default**
removes that ROI's Custom settings.

Expand **Edit Default settings**, make a bounded change, and choose **Use
these as Default settings** when the shared settings should change. Use **Reset
Default settings** to return to the starting profile. If you edit Default
settings but do not apply them, the saved Default settings—not the unsaved
text—will be used during Run.

The **Preview feature detection** panel lets you choose an **ROI**, choose a
**Segment**, and select **Generate Preview**. It previews that ROI with the
settings it will use during Run: its Custom settings if it has them, otherwise
the Default settings. **Show preview details** reveals the numeric preview
summary. The preview helps you judge the settings; it is not the final analysis.

### Read the preview and review every ROI

The question is whether event boundaries, polarity, spacing, and summaries are
defensible for this ROI. Review quiet periods, expected responses, and obvious
artifacts. If the preview does not resemble the signal, adjust settings or
reconsider correction before continuing. **Continue to Review Plan** becomes
available when at least one ROI is included, every included ROI has valid
settings, and any changed Default or Custom settings have been applied.

## 9. Review the analysis plan

Review plan is the last planning checkpoint. It assembles the choices made in
the earlier steps and shows:

- Plan status and any blocking attention items.
- The input and recording-structure summary.
- Included ROIs.
- The correction plan, including the selected strategy by ROI.
- Feature Detection, including Default and Custom assignments.
- The output destination.
- The next step and the **Go to Run** action when the plan is ready.

Confirm that the source, recording structure, timing, selected ROIs, correction
choices, feature-detection settings, and output folder are correct. If the plan
identifies an item that needs attention, return to the indicated step and correct
it before running the analysis.

The plan can report that the setup is complete while your scientific review is
still incomplete. Software readiness and scientific readiness are separate.
The correction and feature-detection previews are evidence for checking your
choices; they are not the final analysis. Choose **Go to Run** only after the
visible plan and your scientific review agree. If the button is unavailable,
read **What needs attention** and correct the named step.

## 10. Run the analysis

Run begins with Check my setup. This check uses the current setup and the
current plan. The Run Guided Analysis button becomes available only after the
check passes for that setup. If you change a source, ROI, timing choice,
correction strategy, feature profile, or output destination, run the check
again.

Read the status and attention text after the check. A blocked run is useful
feedback: return to the step named in the message, correct the specific
choice, and check the setup again. If Run is disabled, correct the named setup
item before checking again.

For repeated-session analysis, the GUI warns that the analysis cannot be
stopped from the GUI once it starts. Do not close the window while it is
running. Plan enough uninterrupted time for the run.

For continuous analysis, the **Stop** button appears while preparation or
analysis is active. Stop requests a stop at the next safe point, so it can take
a moment. It is a cooperative stop, not an immediate termination. Wait for the
status in **Analysis progress** to settle before closing the application.

After a successful run, choose **Open results folder** to view the saved files or
**Load completed run for review** to open them in Guided. If the run fails, the
displayed message identifies the failure, and the incomplete output folder will
not load as a completed run.

## 11. Review completed results

Review summarizes completed-run outputs when results are loaded. Use the
**Region** selector to inspect each included ROI; **Region** is the current
Results label for the ROI being displayed. The current Results viewer also
provides **Run Report**, **Summary**, **Day Plots**, and **Tables** buttons for
opening the saved report and the corresponding saved result areas.

- Verification provides a visual review of the completed correction result.
  It shows representative signal, reference or correction baseline, and dF/F
  traces for the selected ROI, together with the correction method used. When
  the run is loaded, Guided checks that the saved results folder contains the
  expected files and a record of how the run was produced.
- **Slow Signal** shows slow-signal views when that analysis ran.
- **Signal / Reference** shows the signal and reference channels for the
  plotted days.
- **Correction Reference** shows the reference used for correction when that
  view is available.
- **dF/F** shows the reference-corrected event signal for the plotted days.
- **Stacked dF/F** compares plotted days.
- **Event Summary** shows event-activity summaries across plotted days.

The available views depend on which analyses ran and which files were produced.
Older screenshots or saved reports may use the names **Phasic dFF** and
**Phasic Summary**; those are not current Results controls. In the current GUI,
use **dF/F** and **Event Summary**.

These checks confirm that the saved result is internally consistent; they do not
establish biological validity or determine whether the selected correction is
scientifically appropriate.

Continuous Guided results use the same Results viewer described above: choose a
**Region** and inspect the views that were produced, such as **Verification**,
**Slow Signal**, **Signal / Reference**, **Correction Reference**, **dF/F**,
**Stacked dF/F**, and **Event Summary**. Not every view appears for every run.
The continuous overview and window summaries describe the recording duration,
analysis windows, correction state, and whether slow-signal and event analysis
completed. Long traces may be downsampled for display; inspect the overview and
per-window summaries when judging long recordings.

Relate outputs to the experiment. Check timeline placement, correction,
event summaries, and missing sessions against the protocol.
Preserve the output location and settings summary with the experiment record.

## 12. Common verified problems

### Input or structure is not recognized

Read the visible message, return to Select data, and confirm the source and
Recording structure. Automatic detection is a starting point. For CSV, check
for one continuous recording CSV or multiple CSV session files, then inspect
ROIs and timing again.

### The CSV mapping is incomplete or confusing

Confirm the Time column and Time units, then verify every Signal column and
Reference column pairing. If an unintended ROI mapping is present, use its
Remove control or correct the mapping in the Interpret CSV columns area. Leave
the original recording CSV unchanged. Use Add ROI when another valid
signal/reference pair is needed. For repeated sessions, verify that the
displayed CSV session order is chronological.

### Timing does not match the experiment

Compare detected timing with the protocol. Correct Sessions per hour and
Session duration (s), or choose the continuous window and partial-final-window
decision. Review Timeline placement and recording-start clock when times look
wrong.

### Correction preview is unavailable or implausible

Read the visible reason first. Check ROI, Preview segment, and mapping, then
compare the result with raw signal and reference/control. Choose another
approach and regenerate after confirming that the current data and controls are
correct.

### Events are missing or excessive

Inspect the affected ROI's preview. Review polarity, threshold method/value,
minimum distance, prominence, width, and pre-filter. Use Custom only for that
ROI and recheck quiet periods and expected responses.

### Run is disabled

Go to Review plan and read What needs attention. Correct the named step, then
press Check my setup again. Confirm at least one ROI, destination, timing,
correction, and feature settings are complete. Opening completed results does
not launch a new run.

### A completed run does not load in Review

Use Start and choose the completed-results entry point, or use Load completed
run for review after success. Select the actual completed-results folder. If it
is not recognized, read the message and select a completed-results folder rather
than an input or in-progress folder.

### Results do not show a view you expected

Confirm which branches completed and which ROI is selected. Use the continuous
overview or Review status to distinguish "not run" from "not loaded." Use Run
Report, Summary, Day Plots, or Tables when the corresponding output exists. If
the run failed, use the failure message; an incomplete folder will not load as a
completed run.

## 13. Quick checklist for upcoming analyses

Use this short checklist:

- The source format and repeated or continuous structure match the actual files.
- CSV time, signal, reference, and units are mapped correctly.
- Repeated CSV order is confirmed and agrees with chronology.
- The included ROIs are the ROIs intended for analysis.
- Repeated timing or continuous windows match the experimental design.
- Timeline placement and recording-start time are sensible.
- Correction preview behavior is scientifically defensible for each ROI.
- Feature Detection settings and event examples are plausible for each ROI.
- Review plan shows the intended output destination.
- Check my setup passes for the current, unchanged plan.
- Completed Review outputs have been inspected, not merely generated.

The final responsibility remains scientific judgment. Guided makes decisions
visible and keeps the ordinary path orderly; it does not decide whether a
recording supports your claim.

## 14. Further reading

For format details, see [Supported Input Formats](input_formats.md).

For long-recording windows and result presentation, see
[Continuous Recordings](continuous_recordings.md).

For correction concepts and diagnostic interpretation, see
[Correction and Dynamic Fitting](correction_and_dynamic_fit.md).

For event-detection behavior and terminology, see
[Event Detection](event_detection.md).
