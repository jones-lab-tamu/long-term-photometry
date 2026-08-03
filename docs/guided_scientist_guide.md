# Guided Scientist Guide

This is the canonical, scientist-facing guide to the Guided workflow in Long-Term
Photometry Analysis. It describes the controls that are visible in the current
application and the decisions that remain with the scientist. It is written for
someone who wants to analyze long-duration fiber photometry recordings without
learning the application's internal implementation.

Guided is the recommended workflow for a new analysis. Full Control remains
available for expert users and backward-compatible workflows, but it is outside
the ordinary path described here.

After installation, programming is not required for ordinary Guided use.

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

A passed setup check means that the current request is software-ready for the
selected Guided route. It does not make the data scientifically valid and does
not replace inspection of the recording or the results.

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
application. CSV files can also be used for either structure, but they are
mapped explicitly in Guided rather than interpreted as an arbitrary vendor
export.

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

Use the README installation and launch instructions, then open Guided. The Start
step gives two entry points:

- Set up a new analysis from raw or input data.
- Open results from a completed run for review.

Choose the new-analysis path to configure or run analysis. Choose the
completed-results path to inspect a finished run. Opening completed results
does not configure a new analysis, validate a new request, or launch another
run.

If you begin a new analysis, Guided moves to Select data. If you open completed
results, choose the completed-results folder and wait for Review to load.

You can revisit earlier steps before Run. Changing earlier settings may require
the setup or readiness check again.

## 5. Select data

Select data establishes the source, recording organization, folders, and ROIs.
The visible controls are:

- Format: Auto, RWD, NPM, or CSV files.
- Recording structure: Detect automatically, Intermittent/session-based
  recording, or Continuous/one long recording.
- Input folder and Browse...
- Output folder and Browse...
- Select ROIs...

### Choose the input folder

Choose the RWD folder organized as supported recording/session data, the
Neurophotometrics folder containing supported repeated session exports, or the
CSV folder containing candidate files. The application inspects the folder for
source information and ROIs; it does not make arbitrary exports native.

Use Detect automatically when you want the application to inspect the selected
folder. If you already know the organization, choose repeated sessions or one
long continuous recording yourself. The choice describes how the data were
saved, not how long the experiment lasted.

### Choose the output folder

Choose the output folder before the run. Review plan repeats the destination
for verification. If it is wrong, return to Select data and change it. Input
files are not overwritten.

### CSV column mapping

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
structure left automatic, one candidate CSV file defaults to a continuous
recording and multiple candidate CSV files default to repeated sessions. You
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

Select ROIs scans the input folder and lists channels found in the data. Include
only the intended ROIs and confirm their signal/reference pairing, especially
when different files use similar names.

Continue when the source, structure, ROI list, and required CSV order
confirmation match the intended dataset.

## 6. Confirm recording structure and timing

Recording Structure keeps settings that depend on the organization chosen in
Select data. Detected timing and Use detected timing are starting points; check
them against the acquisition schedule.

### For repeated sessions

For intermittent/session-based recording, review:

- Sessions per hour
- Session duration (s)

These positive values place sessions on the timeline and must agree with the
recording protocol.

If the final RWD recording session is shorter than expected, Final recording
session can explicitly exclude that one incomplete session. Use it only for
the last file; earlier incomplete sessions still stop validation. Raw files are
not modified.

### For continuous recordings

For continuous data, review:

- Continuous analysis window (s)
- Allow partial final analysis window

The window is the length of each analysis unit. Choose a length that supports
the question and leaves enough samples for correction and event checks. Decide
whether a shorter final window is interpretable before enabling it. Guided
treats the source as one continuous trace, not separate acquisition sessions.

### Timeline placement

Timeline placement controls how repeated recordings are displayed in time. The
Time display choices are:

- Fixed daily anchor: place each day relative to the selected circadian-day
  start.
- Civil clock: use actual clock time with midnight as the day boundary.
- Elapsed from first recording: start the plot at the first recording.

When needed, enter Start of plotted day as HH:MM. For RWD, Guided can prefill
Clock time at recording start from a validated timestamp. Check or edit it
when metadata or displayed time disagree with the experiment. Gaps, day
boundaries, and session placement should agree with the schedule.

Before continuing, verify the structure, order, recording start, plausible total
duration, and session count or continuous duration against the experimental
record.

Continue when required timing fields and timeline choice are valid and any
incomplete-final-session decision is intentional.

## 7. Review the correction approach

### What correction means

Correction reduces shared signal/reference structure before phasic
interpretation. This step presents cards, a local preview, and per-ROI strategy
confirmation; it does not start final analysis.

The current cards communicate different levels of readiness:

- Robust Global Event-Reject Fit carries the Default label and is the
  recommended starting point. It fits the reference relationship while
  excluding event-like periods.
- Adaptive Event-Gated Fit is a Candidate for recordings where the
  signal-reference relationship may change over time. Inspect its evidence.
- Global Linear Regression is a baseline comparison and is not recommended for
  most long-duration recordings; its label is not an endorsement.
- Signal-Only F0 is available as a diagnostic comparison in the local preview,
  and it can also be an executable per-ROI production choice on the
  repeated/session Neurophotometrics route and the continuous RWD route,
  including supported mixed per-ROI plans. A preview selection by itself is
  not production authorization; the selected route must accept the choice.
- Decision-Support Audit is marked Coming later and currently provides
  read-only evidence. It does not run analysis or choose a strategy.

### Read the preview

The preview compares selected methods on one recording segment. Choose the ROI,
Preview segment, methods, and Generate correction preview. It includes the
reference-based methods and Signal-Only F0 for diagnostic comparison. It is
local evidence and does not modify source data or start final analysis.

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

After comparing evidence, confirm one strategy for each included ROI. A method
selected for preview is not automatically final for every ROI. The final
Guided Run uses the complete selected recording set for the confirmed strategy,
not only the preview segment. Do not continue when the signal/reference mapping
appears wrong or the corrected trace remains scientifically implausible; fix
the mapping or reconsider the correction choice first.

## 8. Review Feature Detection

### What a detected feature means

Feature Detection controls how events are identified in each ROI. These settings
do not run analysis or write files. Use the preview and recording context to
judge plausibility.

### Adjust the exposed Guided settings

The Default Feature Detection settings form exposes the following controls.
The same fields can be customized for one ROI:

- **Event signal**: Chooses the dFF or delta-F trace used for peak detection
  and event-derived metrics. Use the representation that matches the signal
  you are reviewing; changing it can change both event locations and scale.
- **Signal excursion polarity**: Chooses positive, negative, or both
  directions for excursions. Use the expected response direction when it is
  known; changing it changes which signed responses are considered.
- **Peak threshold method**: Chooses a mean-and-standard-deviation, percentile,
  or absolute cutoff. Use the method that fits the trace's noise and scale;
  it changes how the detection threshold is derived.
- **Peak threshold k**: Sets the multiplier for methods that use k. It is
  relevant to mean-and-standard-deviation thresholding; a larger value requires a
  larger excursion relative to variability, while a smaller value admits more.
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

The per-ROI table shows Default or Custom and the settings used during Run.
Begin with Default, then give an ROI Custom settings when its noise, polarity,
or response shape needs different treatment. Custom changes that ROI alone.
Other ROIs remain on Default.

Expand Edit Default settings, make a bounded change, and choose Use these as
Default settings when the shared profile should change. Use Reset Default
settings to return to the starting profile. For a single ROI, use its
customization control in the table. Use Reset to default when the ROI should
again follow the shared profile.

### Read the preview and review every ROI

The question is whether event boundaries, polarity, spacing, and summaries are
defensible for this ROI. Review quiet periods, expected responses, and obvious
artifacts. If the preview does not resemble the signal, adjust settings or
reconsider correction before continuing.

## 9. Review the analysis plan

Review plan is the last planning checkpoint. It assembles the choices made in
the earlier steps and shows:

- Plan status and any blocking attention items.
- The input and recording-structure summary.
- Included ROIs.
- The correction plan, including the selected strategy by ROI.
- Feature Detection, including Default and Custom assignments.
- The output destination.
- The next step and the Go to Run action when the plan is ready.

Read the plan as a scientist, not as a formality. Confirm the source, format,
structure, timing or window length, time placement, ROIs, correction,
feature-detection assignments, and output destination. Follow attention
messages back to the relevant step. Use technical details only when a
maintainer or support person asks for them.

Review plan can report that the setup is complete while your scientific review
is still incomplete. Software readiness and scientific readiness are separate.
Go to Run only when both the visible plan and your own review agree.

## 10. Run the analysis

Run begins with Check my setup. This check uses the current setup and the
current plan. The Run Guided Analysis button becomes available only after the
check passes for that setup. If you change a source, ROI, timing choice,
correction strategy, feature profile, or output destination, run the check
again.

Read the status and attention text after the check. A blocked run is useful
feedback: return to the step named in the message, correct the specific
choice, and check the setup again. Do not work around a disabled Run button by
switching to an unrelated workflow.

For intermittent analysis, the GUI warns that the analysis cannot be stopped
from the GUI once it starts. Do not close the window while it is running. Plan
enough uninterrupted time for the run.

For continuous analysis, the Stop button appears while preparation or analysis
is active. Stop requests a stop at the next safe point, so it can take a
moment. It is a cooperative stop, not an immediate termination. Wait for the
status to settle before closing the application.

After a successful completed run, Guided can show Open results folder and Load
completed run for review. Use the latter to move directly to the Guided Review
step. If the run fails, read the displayed reason and do not treat a partially
created folder as a completed scientific result.

## 11. Review completed results

Review summarizes completed-run outputs when results are loaded. Use the ROI
selector to inspect each included ROI. The normal result views can include:

- Verification, for checking that the completed run has the expected review
  material.
- Tonic, for tonic signal views where tonic analysis ran.
- Phasic Sig/Iso, for signal and reference context.
- Dynamic Fit, for the fitted reference behavior.
- Correction Reference, for the reference used for correction when that view is
  available.
- Phasic dFF, for the reference-corrected phasic trace.
- Phasic Stacked, for comparing plotted days.
- Phasic Summary, for event-activity summaries across plotted days.

Use Run Report to open the saved analysis report. For the selected ROI, Summary,
Day Plots, and Tables open corresponding result areas when they exist. Views
depend on what the completed run produced; do not infer a missing result from
another tab.

Continuous results use a simpler workspace with one ROI selector and Tonic or
Phasic tabs only for analyses that actually ran. The continuous overview
summarizes the recording duration, analysis windows, correction state, and
which analysis branches completed. Long traces may be downsampled for display,
while window summaries provide the more detailed view. Inspect both the
overview and the per-window summaries when judging long recordings.

Relate outputs to the experiment. Check timeline placement, correction,
event summaries, and missing or excluded sessions against the protocol.
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
Remove control or correct the mapping in the Interpret CSV columns area; do
not edit or delete rows in the original recording CSV. Use Add ROI when another
valid signal/reference pair is needed. For repeated sessions, verify that the
displayed CSV session order is chronological.

### Timing does not match the experiment

Compare detected timing with the protocol. Correct Sessions per hour and
Session duration (s), or choose the continuous window and partial-final-window
decision. Review Timeline placement and recording-start clock when times look
wrong.

### Correction preview is unavailable or implausible

Read the visible reason first. Check ROI, Preview segment, and mapping, then
compare the result with raw signal and reference/control. Choose another
candidate and regenerate only when the current data and controls are correct.

### Events are missing or excessive

Inspect the affected ROI's preview. Review polarity, threshold method/value,
minimum distance, prominence, width, and pre-filter. Use Custom only for that
ROI and recheck quiet periods and expected responses.

### Run is disabled

Go to Review plan and read What needs attention. Correct the named step, then
press Check my setup again. Confirm at least one ROI, destination, timing,
correction, and feature settings are complete. Open Results mode cannot launch
a new run.

### A completed run does not load in Review

Use Start and choose the completed-results entry point, or use Load completed
run for review after success. Select the actual completed-results folder. If it
is not recognized, read the message and do not substitute an input or
in-progress folder.

### Results do not show a view you expected

Confirm which branches completed and which ROI is selected. Use the continuous
overview or Review status to distinguish "not run" from "not loaded." Use Run
Report, Summary, Day Plots, or Tables when the corresponding output exists. If
the run failed, use the failure message rather than an incomplete folder.

## 13. Quick checklist for future analyses

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
