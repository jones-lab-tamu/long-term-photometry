# Recording-Level Correction Strategy

This document describes planned architecture for choosing applied correction strategies at the ROI-recording level. It is design documentation only.

Core rule:

Per-chunk policy outputs are evidence, QC, and proposal labels. They are not final applied correction decisions.

`proposed_correction_mode_<policy>` is a per-chunk proposal/evidence label. It should not be interpreted as "apply this correction mode to this chunk" in the final applied trace. The applied correction strategy should be chosen once per ROI recording. The pipeline should not default to a mosaic of correction modes, such as switching `dynamic_fit` to `signal_only_f0` to `dynamic_fit` across chunks within one continuous ROI recording.

## Terminology

`ROI recording`

One continuous recording for one ROI/channel/sensor stream over a session or analysis unit.

`chunk`

A local analysis window used for QC, dynamic-fit diagnostics, signal-state diagnostics, signal-only F0 diagnostics, and policy proposals.

`per-chunk correction proposal`

A QC/evidence label indicating what correction mode appears most defensible for that chunk considered locally.

`applied correction strategy`

The correction strategy actually used to produce the corrected trace for the ROI recording.

## Requested Strategies

Future user-facing choices should use:

`requested_correction_strategy`

- `dynamic_fit`: user requests dynamic isosbestic correction for the entire ROI recording.
- `signal_only_f0`: user requests signal-only F0 correction for the entire ROI recording.
- `no_correction`: user requests no correction, or raw/normalized signal-only outputs depending on future implementation.
- `auto`: user asks the software to choose one global applied strategy for the ROI recording.

## Provenance Fields

Proposed future provenance fields:

- `requested_correction_strategy`
- `applied_correction_strategy`
- `correction_strategy_selection`
- `auto_selection_confidence`
- `auto_selection_reason`
- `auto_selection_review_required`
- `auto_selection_flags`

Allowed values:

- `requested_correction_strategy`: `dynamic_fit`, `signal_only_f0`, `no_correction`, or `auto`
- `applied_correction_strategy`: `dynamic_fit`, `signal_only_f0`, or `no_correction`
- `correction_strategy_selection`: `manual` or `auto`

Examples:

Manual dynamic fit:

```text
requested_correction_strategy = dynamic_fit
applied_correction_strategy = dynamic_fit
correction_strategy_selection = manual
```

Auto selects dynamic fit:

```text
requested_correction_strategy = auto
applied_correction_strategy = dynamic_fit
correction_strategy_selection = auto
```

Auto selects no correction:

```text
requested_correction_strategy = auto
applied_correction_strategy = no_correction
correction_strategy_selection = auto
```

Do not use pseudo-modes such as:

- `auto_selected_dynamic_fit`
- `auto_selected_signal_only_f0`
- `auto_selected_no_correction`

Auto-selection should be represented by provenance fields, not by creating additional applied correction mode names.

## Current Per-Chunk Proposal Modes

`dynamic_isosbestic`

Dynamic/reference correction appears clean and defensible for this chunk.

`signal_only_f0_candidate`

Dynamic/reference correction is meaningfully suspect or hard-inspect, and signal-only F0 diagnostics support a plausible fallback candidate for this chunk. This is proposal-only. It is not yet an applied correction.

`no_clean_reference_candidate`

No clean automatic correction proposal is available for this chunk, but it may not require mandatory review under the selected policy.

`review_required`

Manual review is required for this chunk.

`baseline_reference_candidate`

Legacy diagnostic only. It is not a policy fallback and not an applied correction mode.

## Two-Tier Flow

Tier 1: Per-chunk evidence/QC

```text
Run dynamic isosbestic diagnostics
Run signal-state diagnostics
Run signal-only F0 candidate diagnostics
Generate per-chunk policy proposals
Output chunk-level warnings, review flags, and candidate evidence
```

Tier 2: Per-ROI recording applied strategy

```text
User selects dynamic_fit, signal_only_f0, no_correction, or auto
If manual, apply the selected strategy globally to the ROI recording
If auto, aggregate chunk-level evidence and select one global applied strategy
Apply one correction strategy consistently across the ROI recording
Preserve chunk-level QC annotations as warnings/review regions
```

Central rule:

Auto does not mean choosing the best correction method independently for each chunk. Auto means choosing one global applied strategy for the ROI recording using chunk-level evidence.

## Auto-Selection Concept

Thresholds and implementation details are not defined yet.

Auto should generally select `dynamic_fit` when:

- most chunks support `dynamic_isosbestic`
- dynamic hard-inspect chunks are rare or isolated
- reference problems are not widespread
- signal-only F0 does not clearly outperform dynamic correction globally

Auto should generally select `signal_only_f0` when:

- dynamic/reference problems are widespread across the ROI recording
- signal-only F0 is consistently available and sufficiently anchored
- signal-only F0 confidence is mostly medium/high
- problematic signal-only F0 chunks are rare or reviewable

Auto should generally select `no_correction` or require manual review when:

- dynamic correction is not defensible globally
- signal-only F0 is also poorly supported, hard-inspect, or inconsistent
- high-risk chunks are widespread
- no single global strategy is defensible

## Practical Best-Available Auto-Selection

Auto is allowed to choose a best available global strategy even when many chunks are not clean. Long-duration recordings often contain mixed local evidence; mixed evidence alone should not force `no_correction` or manual refusal to choose.

Chunk-level QC carries the warning burden. Recording-level `auto_selection_review_required` means the proposed global strategy should be inspected and interpreted with its review regions, not that auto must refuse to propose a strategy.

`signal_only_f0_best_available_dynamic_problem_widespread` means dynamic/reference correction appears broadly problematic while signal-only F0 diagnostics are broadly usable. In that case, the recording-level proposal may select `signal_only_f0` globally with warnings, even if many individual chunks were not strict per-chunk `signal_only_f0_candidate` proposals.

## Chunk-Level QC Under A Chosen Strategy

Chunk-level QC remains useful after a global strategy is chosen.

If applied strategy is `dynamic_fit`:

- chunks with negative/mixed coupling or dynamic hard-inspect should be flagged for review, even though `dynamic_fit` is applied globally

If applied strategy is `signal_only_f0`:

- chunks with insufficient anchors, high extrapolation, large gaps, high-state context, or low confidence should be flagged for review, even though `signal_only_f0` is applied globally

If applied strategy is `no_correction`:

- chunk-level diagnostics still indicate why correction was not applied and where signal/reference behavior was problematic

## Recording-Level Strategy Proposal Utility

`tools/propose_recording_correction_strategy.py` aggregates existing per-chunk evidence into one proposal row per ROI recording. It reads chunk-level QC/proposal records from `_analysis/phasic_out/qc/baseline_reference_candidate_by_chunk.json` and writes:

- `recording_correction_strategy_proposals.csv`
- `recording_correction_strategy_proposals.json`

The utility remains proposal-only. It does not apply correction, recompute dF/F, rerun event detection, modify HDF5 traces, or implement GUI auto mode. Its output is intended to guide future auto-selection design and manual review.

Recording-level grouping may use a derived `recording_key`. Full `source_file` paths can represent individual chunk/session files rather than the broader recording. The proposal utility aggregates by `recording_key` x ROI, not necessarily by full `source_file` x ROI. The `grouping_mode` used for this derivation is recorded in the output provenance, and full input source paths are preserved in `source_files`.

Example:

```powershell
python tools/propose_recording_correction_strategy.py --phasic-out "<phasic_out>" --policy balanced
```

## Compact Strategy Report

`tools/export_recording_correction_strategy_report.py` converts the recording-level proposal table into compact Markdown and CSV reports:

- `recording_correction_strategy_report.md`
- `recording_correction_strategy_compact.csv`

The exporter reads `recording_correction_strategy_proposals.csv` or `recording_correction_strategy_proposals.json`. It is read-only with respect to correction outputs: it does not apply correction, recompute dF/F, rerun event detection, modify HDF5 traces, or edit the source proposal/QC files.

The report is intended for human inspection, student review, and future GUI design. It summarizes one global strategy per ROI recording plus chunk-level review/caution previews. These previews identify chunks requiring inspection under the proposed global strategy; they do not indicate chunkwise switching of correction modes.

Example:

```powershell
python tools/export_recording_correction_strategy_report.py --phasic-out "<phasic_out>"
```

## Interpreting Recording-Level Strategy Outputs

The recording-level selector proposes one global strategy per ROI recording. It does not choose correction modes independently for each chunk.

`applied_correction_strategy_proposed`

The proposed global correction strategy for the ROI recording. Current values include:

- `dynamic_fit`
- `signal_only_f0`
- `no_correction`

`auto_selection_confidence`

Confidence in the recording-level proposal:

- `high`: globally clean or strong evidence
- `medium`: best available strategy with substantial supporting evidence
- `low`: usable but weak/messy evidence, or no confident strategy

`auto_selection_review_required`

A warning/review flag for the proposed global strategy. It does not necessarily mean auto failed. It does not necessarily mean the recording should be excluded. It means the proposed strategy carries enough local warnings that the user should inspect the report, review chunk previews, or inspect plots.

`auto_selection_reason`

The main rule or reason that produced the proposal.

`review_chunk_ids`

Chunks that require closer inspection under the proposed strategy.

`caution_chunk_ids`

Chunks that provide cautionary context but are not necessarily fatal.

Examples:

`dynamic_fit`, high confidence, `review_required = false`

Dynamic isosbestic correction appears globally clean and is proposed as the recording-level strategy. Chunk-level warnings are minimal.

`signal_only_f0`, medium confidence, `review_required = true`, `reason = signal_only_f0_best_available_dynamic_problem_widespread`

Dynamic/reference correction is broadly problematic across the ROI recording. Signal-only F0 diagnostics are broadly usable, so `signal_only_f0` is proposed as the best global strategy. Review is required because the recording is messy, not because auto failed to choose.

`signal_only_f0`, low confidence, `review_required = true`

Signal-only F0 is still the best available global strategy, but the evidence is weaker or more heterogeneous. Inspect review/caution chunks before trusting downstream summaries.

`no_correction`, low confidence, `review_required = true`

No global correction strategy was selected confidently. This should be treated as a high-risk recording-level outcome requiring inspection before proceeding.

Plain-language summary:

`review_required` means "inspect this proposed global strategy," not "stop" and not "switch methods chunk-by-chunk."

Misinterpretation warnings:

- Do not interpret a `signal_only_f0` proposal as evidence that dynamic fitting failed in every chunk.
- Do not interpret `review_required` as evidence that the entire ROI is unusable.
- Do not interpret per-chunk proposals as applied chunkwise correction modes.

## Configuration-Specific Strategy Proposals

A recording-level strategy proposal is valid only for the correction-analysis configuration that generated the per-chunk evidence.

Auto chooses among `dynamic_fit`, `signal_only_f0`, and `no_correction` using the evidence from the current correction-analysis configuration. It does not silently retune dynamic fitting, compare all dynamic-fit modes, or choose settings behind the user's back.

The proposal is conditional on the current run configuration. It is not a permanent statement that an ROI should always use a given correction strategy.

For example:

- If `dynamic_fit_mode = global_linear_regression`, auto may propose `signal_only_f0`.
- If the user reruns the same ROI with `dynamic_fit_mode = adaptive_event_gated_regression`, the dynamic-fit QC may change and auto may propose `dynamic_fit`.
- These are not contradictions. They are configuration-specific proposals.

### Correction-analysis configuration includes

Correction-analysis configuration includes settings and categories that may affect the evidence used by the selector, including but not limited to:

- dynamic fit mode
- rolling/global/robust/adaptive dynamic-fit parameters
- bleach correction mode
- baseline subtraction before fit
- slope constraint settings
- signal-only F0 diagnostic settings
- signal-state diagnostic settings
- correction-policy thresholds
- recording-level strategy selector thresholds
- any preprocessing settings that affect the signal/reference traces used for correction QC

### Staleness rule

If any correction-analysis setting changes, existing per-chunk QC, recording-level proposals, and compact reports should be treated as stale until regenerated.

Examples:

- changing `dynamic_fit_mode` stales the proposal
- changing rolling-window or robust/adaptive fit parameters stales the proposal
- changing bleach correction stales the proposal
- changing baseline subtraction before fit stales the proposal
- changing slope-constraint settings stales the proposal
- changing signal-only F0 settings stales the proposal
- changing policy/selector thresholds stales the proposal

Stale does not necessarily mean wrong. It means the proposal no longer corresponds to the current settings and should not be used as current evidence.

### Future provenance fields

Future implementation should record configuration-specific strategy provenance fields such as:

- `correction_analysis_config_hash`
- `correction_analysis_config_summary`
- `correction_analysis_config_path`
- `strategy_evidence_generated_at`
- `strategy_evidence_source`
- `strategy_evidence_policy`
- `strategy_evidence_grouping_mode`
- `strategy_proposal_stale`
- `strategy_proposal_stale_reason`

`correction_analysis_config_hash` should identify the correction-analysis configuration used to generate the evidence. `correction_analysis_config_summary` should be human-readable. `strategy_proposal_stale` should become true when the current configuration differs from the evidence configuration. `strategy_proposal_stale_reason` should explain which settings changed, if known.

### What auto does not do

- Auto does not silently try every dynamic-fit mode.
- Auto does not retune dynamic-fit parameters behind the user's back.
- Auto does not compare multiple correction-analysis configurations unless the user explicitly runs a future comparison workflow.
- Auto does not switch correction methods chunk-by-chunk.
- Auto selects one global strategy per ROI recording using the evidence from the current configuration.

A future "compare correction configurations" workflow may be useful, but it should be explicit and separately reported, not hidden inside ordinary auto-selection.

### Implications for applied correction

Before writing future `applied_dff` outputs, the pipeline must record:

- `requested_correction_strategy`
- `applied_correction_strategy`
- `correction_strategy_selection`
- `correction_analysis_config_hash`
- `correction_analysis_config_summary`
- `strategy_evidence_policy`
- `strategy_evidence_generated_at`

Applied correction outputs should not be interpreted without knowing both the applied strategy and the configuration that generated the strategy evidence.

## Not Implemented Yet

This document describes planned architecture.

The current codebase has:

- per-chunk dynamic-fit diagnostics
- per-chunk signal-state diagnostics
- per-chunk signal-only F0 candidate diagnostics
- per-chunk correction-policy proposals
- review-set export utilities

The current codebase does not yet implement:

- applied `signal_only_f0` correction
- recording-level auto strategy selection
- GUI controls for `requested_correction_strategy`
- recording-level applied strategy provenance fields
