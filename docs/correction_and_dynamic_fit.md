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

Dynamic-fit validity diagnostics are per-chunk QC metrics that do not change correction behavior. They describe model-assumption concerns, not biological truth. They help identify chunks where the fitted reference is flat, uninformative, negative or mixed-sign in its coupling, or unusually rich in response-scale structure.

Response-scale-rich fitted references are contextual QC flags. They do not automatically indicate a failed correction because legitimate shared artifact or reference structure can occur on similar timescales. They become more concerning when combined with negative or mixed coupling, low fitted-reference range, flat or uninformative fitted references, or visual evidence that the fitted reference follows biological signal rather than reference artifact.

Negative or mixed reference coupling is also a contextual QC flag. It indicates that the reference channel is not acting as a uniformly positive predictor of the signal in that chunk. For some sensors, especially during large biological responses, reference-channel deflections may reflect sensor photophysics or real response structure rather than simple artifact. This flag should prompt inspection and may argue against full response-timescale isosbestic regression, but it does not by itself indicate failed acquisition or invalid biology.

Flat/uninformative fitted references and low fitted-reference range are stronger warning signs because they indicate the fitted reference is not contributing meaningful correction structure. These hard flags mark chunks as needing inspection. Flags indicate chunks needing inspection, not automatic exclusion or automatic correction-mode selection.

The phasic output writes machine-readable diagnostics under `qc/dynamic_fit_qc_by_chunk.csv` and `qc/dynamic_fit_qc_by_chunk.json` when fitted references are available. Later workflows may use these metrics to compare full dynamic reference correction with baseline-only reference correction.

## Baseline-only reference candidate diagnostics

The baseline-only reference candidate is a diagnostic comparison trace computed from configurable ultra-low-pass smoothing of the reference structure. The default requested smoothing window is 300 s, but the requested and actual per-chunk smoothing windows are recorded in `baseline_reference_candidate_by_chunk.csv` and `.json`. If the requested window is too large for a chunk it is adjusted and reported; if the actual window is a large fraction of the chunk, the output should be interpreted cautiously. The `baseline_ref_lowpass_cutoff_hz` field records the baseline-scale boundary used for diagnostics, but this implementation uses reflected-window smoothing rather than an exact frequency-domain low-pass filter. It is intended to test whether the reference channel supports slow baseline correction without allowing the fitted reference to follow response-scale biological events.

This candidate is useful for evaluating chunks where full dynamic isosbestic regression may be questionable, especially chunks with negative or mixed reference coupling, broad sensor responses, or fitted references that contain substantial response-scale structure. In this implementation, the candidate is written for review and comparison only; it does not change the applied correction, dF/F calculation, or event detection.

The phasic output writes candidate metrics under `qc/baseline_reference_candidate_by_chunk.csv` and `qc/baseline_reference_candidate_by_chunk.json` when fitted references are available.

## Dynamic-vs-baseline candidate comparison

The dynamic-vs-baseline comparison class is a diagnostic triage field. It combines existing dynamic-fit QC severity with baseline-candidate QC metrics to indicate whether dynamic fitting, baseline-only reference use, both, or neither appear viable for a chunk. This field does not select or apply a correction mode; it is intended to make review of candidate behavior more systematic.

Contextual flags, such as response-scale-rich fitted references or negative/mixed coupling, are not treated as automatic failures. The comparison outputs are diagnostic only and do not change dF/F calculation, event detection, or applied correction behavior.

A baseline candidate is considered cleanly viable only when it is available, not low/flat, not response-scale-rich, and supported by a positive smoothed reference-to-signal relationship. Negative, weak, mixed/unclear, or unknown smoothed relationships are preserved as diagnostic findings but downgrade the baseline candidate to contextual review status. This affects diagnostic classification only and does not change correction, dF/F calculation, event detection, or baseline-candidate computation.

## Diagnostic correction policy proposals

The correction policy proposal layer converts dynamic-fit QC, baseline-candidate QC, and baseline fit relationship diagnostics into policy-specific triage fields. These fields indicate what the software would propose under conservative, balanced, or liberal review settings. They are provenance outputs only and do not determine the correction used for dF/F, event detection, or feature extraction.

Policy proposals separate correction-mode proposals from review-burden management. `review_required` means mandatory manual review. `review_queue_candidate` means the chunk is useful for representative audit or a review queue but is not necessarily mandatory. `warning_level` separates logged diagnostic severity from manual review burden. This distinction is important for long-duration recordings, where reviewing every contextual chunk would defeat the purpose of automated analysis.

Conservative policy requires review more often when evidence is contextual. Balanced policy accepts clean dynamic-isosbestic cases, logs many contextual cases as warning/audit candidates without mandatory review, and still requires review when no clean defensible reference candidate exists. Liberal policy proceeds more often for screening. None of the policies auto-select negative or inverted baseline candidates.

The legacy baseline-reference candidate remains available as a diagnostic trace, but it is no longer treated as a correction-policy fallback. For sensors that can enter sustained high-output states, reference-derived baseline estimates can remove true signal. Signal-derived F0 diagnostics are used instead when considering fallback proposals.

### Signal-only F0 candidate policy proposals

`signal_only_f0_candidate` is a proposed correction mode only. It is not an applied correction mode and does not change dF/F calculation, event detection, feature extraction, HDF5 applied traces, or plotting outputs.

The proposal can appear when dynamic/reference correction is not clean enough and signal-only F0 diagnostics indicate a sufficiently supported fallback candidate. The candidate uses the signal channel only and never uses the isosbestic/reference channel. High-state, edge-high-state, or partial-high-state context makes the proposal contextual and cautionary; it is not treated as a clean high-confidence fallback. Signal-only F0 candidates that are unavailable, hard-inspect, low-confidence under stricter policies, or insufficiently anchored are not proposed.

The old `baseline_reference_candidate` remains a legacy diagnostic trace and is not reintroduced as a policy fallback.

## Signal-state diagnostics

Signal-state diagnostics describe whether the signal channel contains sustained high-state, mixed dynamic/high-state, edge high-state, or ordinary dynamic candidate behavior. These fields are intended to capture sensor-state regimes that may make reference-based correction conceptually questionable.

The diagnostics use the signal channel only. They report robust distribution, high-state occupancy, edge occupancy, local variability suppression, and step-like transition metrics using configurable relative thresholds and windows rather than one hard-coded sustained-duration rule. They do not prove artifact or biological signal.

These diagnostics are provenance outputs only. They do not alter correction, dF/F calculation, event detection, correction modes, feature extraction, or applied traces. The signal-only F0 candidate diagnostics below may use signal-state findings as context, and the policy proposal layer may use those diagnostics for proposal-only triage.

## Signal-only F0 candidate diagnostics

The signal-only F0 candidate is a diagnostic-only lower-envelope estimate computed from the signal channel alone. It does not use the isosbestic/reference channel and does not replace the applied dynamic isosbestic correction.

The candidate uses configurable rolling lower-quantile and smoothing windows to estimate a conservative baseline-like F0 trace from the signal. Its QC fields report support, lower-state coverage, relationship to the observed signal, above-signal fraction before and after conservative capping, tracking score, robust ranges, viability, confidence, and diagnostic flags. Signal-state diagnostics provide contextual flags for sustained, edge, or partial high-state behavior, but signal-only F0 candidate generation is not restricted to locked-high cases; ordinary chunks with untrustworthy dynamic reference correction can also be evaluated.

This first-pass implementation uses scalar signal-state flags as contextual QC and can apply a contextual cap to avoid letting the diagnostic F0 candidate chase high-state plateaus. It does not perform epoch-level high-state exclusion or downweighting because high-state masks are not yet exported as provenance.

This output is intended to evaluate whether a signal-derived fallback could be appropriate when dynamic isosbestic correction is untrustworthy or conceptually contraindicated. It does not alter correction, dF/F calculation, event detection, feature extraction, or HDF5 applied traces. The correction-policy proposal layer may propose `signal_only_f0_candidate` from these diagnostics, but that proposal still does not apply signal-only correction.

When available during normal phasic analysis, the diagnostic trace is stored in the phasic HDF5 cache at `/roi/<ROI>/chunk_<chunk_id>/signal_only_f0_candidate`. Scalar metrics are written to `qc/baseline_reference_candidate_by_chunk.csv`, `qc/baseline_reference_candidate_by_chunk.json`, and the `signal_only_f0_candidate_summary` block in `qc/qc_summary.json`.

### Recomputing policy proposals without rerunning analysis

When policy rules change after a long run has already completed, diagnostic policy proposal fields can be refreshed without rerunning correction, dF/F calculation, event detection, trace generation, or plotting:

```powershell
python tools/recompute_correction_policy_proposals.py --phasic-out "C:\path\to\_analysis\phasic_out"
```

This updates only `qc/baseline_reference_candidate_by_chunk.csv`, `qc/baseline_reference_candidate_by_chunk.json`, and the `correction_policy_proposal_summary` block in `qc/qc_summary.json`. Existing baseline-reference candidate metrics remain legacy diagnostics.

## Reference candidate comparison plots

Reference candidate comparison plots are diagnostic-only overlays of the raw signal, raw reference, existing dynamic fitted reference, and baseline-only candidate. They separate raw/reference context, candidate reference traces, and residual traces into separate panels. Residual traces are plotted on their own y-axis so differences between dynamic-reference subtraction and baseline-candidate subtraction can be inspected without being compressed by fitted-reference amplitudes.

The metadata box includes QC severity, comparison class, flags, smoothing window, and whether the baseline candidate came from stored HDF5 or recomputation. These plots are intended for review of chunks flagged by dynamic-fit QC or reference-candidate comparison. They do not select or apply a correction mode, and they do not change dF/F calculation or event detection.

When available, baseline-reference candidate traces are stored in the phasic HDF5 cache at `/roi/<ROI>/chunk_<chunk_id>/baseline_ref_candidate` for diagnostic provenance. The comparison plotting tool uses this stored trace when present. For older runs without stored candidate traces, the plotting tool can recompute the candidate from recorded metadata. This storage is diagnostic-only and does not affect correction, dF/F calculation, or event detection.

Example command for multiple chunks:

```powershell
python tools/plot_reference_candidate_comparison.py ^
  --analysis-out C:\path\to\_analysis\phasic_out ^
  --roi CH3 ^
  --chunks 28,29,30,31,32,33
```

### Baseline-candidate fit diagnostics

The optional baseline-candidate fit-diagnostics panels show the smoothed signal and smoothed reference used to fit the baseline candidate. They also show the smoothed signal versus smoothed reference relationship, including slope, correlation, fit stage/status, and residual-excluded points when available. These panels are intended to determine whether the baseline candidate is meaningfully reference-informed, negatively or inversely related to the reference, weakly reference-informed, or mostly behaving like a smoothed signal baseline.

These diagnostics are not correction selection. They do not change dF/F calculation, event detection, or applied correction behavior.

Example command:

```powershell
python tools/plot_reference_candidate_comparison.py ^
  --analysis-out C:\path\to\_analysis\phasic_out ^
  --roi CH3 ^
  --chunks 28,29 ^
  --include-reference-difference ^
  --include-baseline-fit-diagnostics
```

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
