# Correction and Dynamic Fitting

Isosbestic/reference correction is part of the standard phasic preprocessing workflow. The advanced controls exist to inspect and tune how correction is performed; they do not mean correction is optional for normal phasic analysis.

## Dynamic fit modes

Configured by `dynamic_fit_mode`:
- `rolling_filtered_to_raw`: rolling local regression fit using filtered inputs and reconstructing into raw signal units.
- `rolling_filtered_to_filtered`: rolling local regression in filtered space.
- `global_linear_regression`: one global linear signal/reference fit per chunk.
- `robust_global_event_reject`: robust global fit with iterative event-dominated sample rejection.
- `adaptive_event_gated_regression`: local/adaptive fit with event gating and coefficient freezing.

`rolling_local_regression` is treated as a legacy alias for `rolling_filtered_to_raw`.

## Robust and event-gated fitting

Robust/event-gated modes are intended to reduce distortion from large phasic events or artifacts that would otherwise dominate the signal/reference fit. They are correction strategies, not standalone motion-artifact detectors.

Artifact handling is primarily through isosbestic/control-channel correction, robust fitting, event-gated fitting, QC warnings, and correction inspection plots. There is no separate general-purpose motion-artifact detector module that classifies every motion event for removal.

## Bleach correction

Configured by `bleach_correction_mode`:
- `none`: no bleach correction.
- `single_exponential`: fit and subtract a single exponential decay component when fitting succeeds.
- `double_exponential`: fit and subtract a two-component exponential decay when fitting succeeds.

Bleach correction metadata and diagnostics are written into run outputs and correction inspection artifacts where available.

## Baseline subtraction before fit

`baseline_subtract_before_fit` applies to rolling fit modes. It can reduce slow baseline influence on local fit estimation. It does not replace baseline/F0 handling used for dF/F calculation.

## Dynamic-fit validity diagnostics

Dynamic-fit validity diagnostics are per-chunk QC metrics that do not change correction behavior. They help identify chunks where the fitted reference is flat, uninformative, negative or mixed-sign in its coupling, or unusually rich in response-scale structure.

Flags indicate chunks needing inspection, not automatic exclusion or automatic correction-mode selection. For sensors or regions with broad slow responses, response-scale-rich fitted references may indicate that the reference fit is modeling biological signal rather than artifact.

The phasic output writes machine-readable diagnostics under `qc/dynamic_fit_qc_by_chunk.csv` and `qc/dynamic_fit_qc_by_chunk.json` when fitted references are available. Later workflows may use these metrics to compare full dynamic reference correction with baseline-only reference correction.

## Retuning and safeguards

Correction retuning is for evaluating correction-sensitive settings on representative traces and writing retuned diagnostic outputs. Downstream event reanalysis changes event-facing thresholds/features after correction without recomputing upstream correction.

Safeguards against post hoc bias include:
- saved `config_effective.yaml` and `config_used.yaml`
- `run_report.json`, `status.json`, and `MANIFEST.json`
- explicit retune request/result files
- fixed exported plots/tables for accepted runs
- separation of correction retuning from downstream event reanalysis
- correction-quality and verification plots

Once selected, settings should be applied consistently across the relevant dataset or reused through a saved configuration.

## Advanced nonnegative slope diagnostic

The nonnegative slope constraint is an advanced diagnostic option. It asks whether the fitted reference remains supported when UV/reference-to-signal coupling is required to be nonnegative.

Default behavior is `unconstrained`. Nonnegative mode is an explicit reported intervention, not a default correction improvement, and does not prove that the corrected trace is biologically true. If enabling this option causes the fitted reference to collapse, flatten, or fall back, the recording should not be interpreted as successfully corrected under the constrained model. Instead, this indicates that isosbestic regression may be unsupported, negative, or mixed-sign for that chunk or dataset.

Users should inspect correction-quality plots and slope diagnostics before using constrained results.
