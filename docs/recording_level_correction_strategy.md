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
