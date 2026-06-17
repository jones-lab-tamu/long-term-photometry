# Applied-dFF Auto Strategy Decision-Contract Audit

This is a read-only design audit for a future auditable `auto` mode. It does
not implement auto-selection, change production code, change GUI behavior, add
strategy-routing behavior, modify source phasic caches, or modify legacy feature
outputs.

## A. Summary Verdict

There is enough existing evidence to design a conservative future auto decision
contract, but not enough validated evidence to implement production auto now.

Recommended next step: add a read-only auto-candidate audit layer that emits
provisional decision labels (`dynamic_fit`, `signal_only_f0`, `needs_review`)
beside full evidence JSON, without running production analysis or writing
manifest strategies automatically.

Key reason: the phasic cache already stores useful dynamic-fit QC attributes and
signal-only F0 candidate diagnostics, but current advisory audit output does not
yet aggregate the strongest dynamic-fit QC fields, and thresholds have not been
validated across enough normal/pathological real datasets.

## B. Evidence Inventory

Existing advisory tool:

```text
tools/audit_applied_dff_strategy_candidates.py
```

Current advisory output fields include:

- `roi`
- `strategy_candidate`
- `strategy_candidate_status`
- `review_required`
- `manual_review_priority`
- `evidence_summary`
- `blocking_issues`
- `cautions`
- `source_phasic_cache_path`
- `source_phasic_cache_sha256`
- chunk coverage fields
- non-finite counts
- value summary statistics
- signal-only F0 viability/confidence/flag counts
- signal-only F0 problem chunk examples
- read-only mutation checks

Current advisory dynamic-fit evidence is limited to:

- `dff` present for each chunk
- `time_sec` present for each chunk
- `dff`/`time_sec` length compatibility
- non-finite source dF/F values
- dF/F summary statistics

Current advisory signal-only F0 evidence includes:

- `sig_raw` and `time_sec` availability
- core `compute_signal_only_f0_candidate(..., return_uncapped_candidate=True)`
- uncapped denominator validity
- computed signal-only dF/F statistics
- negative signal-only dF/F presence
- `signal_only_f0_candidate_viability`
- `signal_only_f0_candidate_confidence`
- `signal_only_f0_flags`
- anchor/extrapolation/large-gap/few-anchor related counts

Existing phasic cache datasets available read-only:

- `/roi/<ROI>/chunk_<chunk_id>/time_sec`
- `/roi/<ROI>/chunk_<chunk_id>/sig_raw`
- `/roi/<ROI>/chunk_<chunk_id>/uv_raw`
- `/roi/<ROI>/chunk_<chunk_id>/fit_ref`
- `/roi/<ROI>/chunk_<chunk_id>/delta_f`
- `/roi/<ROI>/chunk_<chunk_id>/dff`
- `/roi/<ROI>/chunk_<chunk_id>/signal_only_f0_candidate`
- `/roi/<ROI>/chunk_<chunk_id>/baseline_ref_candidate`

Existing phasic cache chunk attributes include strong dynamic-fit QC evidence:

- `dynamic_fit_qc_available`
- `dynamic_fit_qc_severity`
- `dynamic_fit_qc_flags`
- `dynamic_fit_qc_soft_flags`
- `dynamic_fit_qc_hard_flags`
- `dynamic_fit_qc_dynamic_fit_has_hard_flags`
- `dynamic_fit_qc_dynamic_fit_has_soft_flags`
- `dynamic_fit_qc_dynamic_fit_needs_inspection`
- `dynamic_fit_qc_dynamic_fit_negative_or_mixed_coupling`
- `dynamic_fit_qc_dynamic_fit_reference_flat_or_uninformative`
- `dynamic_fit_qc_dynamic_fit_reference_low_range`
- `dynamic_fit_qc_dynamic_fit_response_scale_rich`
- `dynamic_fit_qc_signal_iso_corr`
- `dynamic_fit_qc_signal_fitted_ref_corr`
- `dynamic_fit_qc_iso_fitted_ref_corr`
- `dynamic_fit_qc_fitted_ref_to_signal_range_ratio`
- `dynamic_fit_qc_fitted_ref_to_iso_range_ratio`
- `dynamic_fit_qc_fitted_ref_total_variance`
- `dynamic_fit_qc_fitted_ref_response_scale_fraction`
- `dynamic_fit_qc_fitted_ref_baseline_scale_fraction`
- `dynamic_fit_qc_slope_fraction_negative`
- `dynamic_fit_slope_warning_level`
- `dynamic_fit_slope_slope_min`
- `dynamic_fit_slope_slope_max`
- `dynamic_fit_slope_slope_median`
- `dynamic_fit_slope_slope_negative_fraction`
- `dynamic_fit_slope_clamped_fraction`
- `dynamic_fit_slope_fallback_used`
- `dynamic_fit_slope_constraint_applied`
- `dynamic_fit_slope_nonnegative_support_insufficient`

Existing phasic cache chunk attributes also include signal-only F0 diagnostics:

- `signal_only_f0_candidate_available`
- `signal_only_f0_candidate_viability`
- `signal_only_f0_candidate_confidence`
- `signal_only_f0_flags`
- `signal_only_f0_anchor_count`
- `signal_only_f0_anchor_status`
- `signal_only_f0_low_support_fraction`
- `signal_only_f0_direct_support_fraction`
- `signal_only_f0_extrapolated_fraction`
- `signal_only_f0_edge_extrapolation_fraction`
- `signal_only_f0_max_anchor_gap_fraction_observed`
- `signal_only_f0_max_anchor_gap_sec_observed`
- `signal_only_f0_high_state_context_applied`
- `signal_only_f0_state_aware_used`
- `signal_only_f0_status`
- `signal_only_f0_warning`

Existing applied-dFF production outputs provide downstream proof, after an
explicit manual strategy is run:

- applied cache completeness
- applied trace source
- warning/review propagation
- negative dF/F preservation for `signal_only_f0`
- source phasic cache hash stability
- legacy feature hash stability
- semantic feature verification status

Missing or not yet aggregated in the advisory audit:

- recording-level aggregate dynamic-fit QC status
- segment-level consistency of dynamic-fit QC flags
- explicit signal/control coupling distribution summaries
- explicit correction-reference failure score
- dynamic-fit slope/intercept stability summaries
- robust fitted-reference range/coupling thresholds
- locked-high/prolonged-high state metrics
- huge/prolonged event metrics that distinguish biology from correction failure
- contradiction checks between dynamic-fit evidence and signal-only F0 evidence
- deterministic decision labels and confidence fields
- validated thresholds from enough real datasets

## C. Normal Dataset Findings

Normal phasic_out:

```text
C:\Users\Jeff\Documents\Test_Photometry_Output\Test_Outputs\other_photometry_20260615\run_20260615_200307_c4a6afc8\_analysis\phasic_out
```

Advisory audit output:

```text
scratch\auto_strategy_decision_contract_audit\normal_advisory_audit
```

The advisory audit found 8 ROIs and 571 chunks per ROI. For all ROIs,
`dynamic_fit` was `available`, with complete required source dF/F coverage and
no candidate blockers.

Normal dynamic-fit evidence summary:

| ROI | dynamic_fit status | chunks | median signal/iso corr | dynamic QC notable findings |
| --- | --- | ---: | ---: | --- |
| CH1 | available | 571 | 0.863 | mostly `ok`; 9 context chunks for response scale |
| CH2 | available | 571 | 0.431 | mostly `ok`; 1 inspect chunk; 2 negative/mixed coupling chunks |
| CH3 | available | 571 | 0.431 | mostly `ok`; 4 inspect chunks; 4 negative/mixed coupling chunks |
| CH4 | available | 571 | 0.795 | mostly `ok`; 3 context chunks |
| CH5 | available | 571 | 0.990 | mostly `ok`; 4 context chunks |
| CH7 | available | 571 | 0.986 | all chunks `ok` |
| CH8 | available | 571 | 0.434 | mostly `ok`; 2 inspect chunks; 3 negative/mixed coupling chunks |
| CH9 | available | 571 | 0.980 | mostly `ok`; 3 context chunks |

The existing signal-only F0 candidate is technically computable for all normal
ROIs, but all normal ROIs have signal-only F0 cautions. Common flags include
edge extrapolation, low-support anchored operation, contextual states, and
confidence-capped extrapolation. Therefore, a rule that treats signal-only F0
computability as sufficient would falsely over-select `signal_only_f0`.

The available evidence supports `dynamic_fit` for the normal dataset. The main
false-positive risk is overreacting to a small number of dynamic-fit context or
inspect chunks in otherwise normal ROIs. A conservative future auto contract
should not choose `signal_only_f0` merely because signal-only F0 is computable.

Existing smoke outputs under:

```text
scratch\gui_applied_dff_smoke_20260616\normal_dynamic_fit
```

contain dynamic-fit applied summaries for all 8 normal ROIs, complete at
571/571 chunks with source phasic cache unchanged and no legacy feature
mutation. This root also contains an extra `CH8_signal_only_f0` applied summary
and the current batch summary under this root lists CH8 `signal_only_f0` plus
CH9 `dynamic_fit`; use per-ROI applied summaries, not only the batch file, when
auditing the normal all-dynamic smoke state.

## D. Weird Dataset Findings

Weird/pathological phasic_out:

```text
C:\Users\Jeff\Documents\Test_Photometry_Output\Test_Outputs\student_weird_photometry_after_reset_20260614\run_20260614_165027_3ca29f64\_analysis\phasic_out
```

Advisory audit output:

```text
scratch\auto_strategy_decision_contract_audit\weird_advisory_audit
```

The advisory audit found 8 ROIs and 581 chunks per ROI. For all ROIs,
`dynamic_fit` was technically `available` because source dF/F and time arrays
exist. This confirms that source availability alone is insufficient for auto
selection.

### CH8

Manual accepted strategy: `signal_only_f0`.

Existing production smoke output:

- applied strategy: `signal_only_f0`
- applied trace source: `signal_only_f0_dff`
- complete: 581/581 chunks
- warning/review: `caution`, review required
- negative dF/F preserved
- semantic feature output status: pass
- source phasic cache unchanged
- legacy features unchanged

Dynamic-fit QC evidence for CH8:

- 320/581 chunks with `NEGATIVE_OR_MIXED_REFERENCE_COUPLING`
- 36/581 chunks with hard dynamic-fit QC flags:
  `FITTED_REFERENCE_LOW_RANGE` / `FITTED_REFERENCE_FLAT_OR_UNINFORMATIVE`
- 36/581 chunks with `DYNAMIC_FIT_NEEDS_INSPECTION`
- 320/581 chunks with slope warning level `critical`
- median `dynamic_fit_qc_signal_iso_corr`: -0.062
- median `dynamic_fit_qc_signal_fitted_ref_corr`: 0.277
- median slope negative fraction: 1.0
- median slope median: -0.0795

Signal-only F0 evidence for CH8:

- computed for 581/581 chunks
- viability counts: contextual=309, hard_inspect=6, viable=266
- confidence counts: high=70, low=315, medium=196
- negative signal-only dF/F present
- common flags include available/state-aware/low-support/edge-extrapolated,
  confidence-capped extrapolation, contextual/viable, few anchors, large anchor
  gap, insufficient anchors/low support in production summary

Interpretation:

CH8 has clear correction-reference failure evidence, but signal-only F0 still
carries cautions. A future auto contract could classify this as
`signal_only_f0` only if thresholds explicitly accept this pattern:
strong dynamic-fit failure plus complete signal-only F0 availability and no
unusable denominator. A more conservative first auto implementation could label
CH8 `needs_review` while surfacing it as a high-priority signal-only F0 rescue
candidate.

### CH9

Manual accepted strategy: `dynamic_fit`.

Existing production smoke output:

- applied strategy: `dynamic_fit`
- applied trace source: `dynamic_fit_dff`
- complete: 581/581 chunks
- warning/review: none
- semantic feature output status: pass
- source phasic cache unchanged
- legacy features unchanged

Dynamic-fit QC evidence for CH9:

- 534/581 chunks `ok`
- 44/581 context chunks
- 3/581 inspect chunks
- 9/581 chunks with `NEGATIVE_OR_MIXED_REFERENCE_COUPLING`
- 3/581 chunks with fitted-reference low-range/flat flags
- median `dynamic_fit_qc_signal_iso_corr`: 0.758
- median `dynamic_fit_qc_signal_fitted_ref_corr`: 0.758
- median slope negative fraction: 0.0
- median slope median: 0.621

Signal-only F0 evidence for CH9:

- computed for 581/581 chunks
- viability counts: contextual=467, viable=114
- confidence counts: high=3, low=467, medium=111
- 577 chunks with confidence-capped extrapolation
- 4 chunks with few anchors
- negative signal-only dF/F present

Interpretation:

CH9 has mostly usable dynamic-fit QC and weak/caution-heavy signal-only F0
evidence. This supports `dynamic_fit`.

### Other Weird ROIs

Several other ROIs in the weird dataset show substantial dynamic-fit QC concerns:

- CH1: 376 negative/mixed coupling chunks, 75 inspect chunks
- CH2: 233 negative/mixed coupling chunks, 87 inspect chunks
- CH3: 406 negative/mixed coupling chunks, 14 inspect chunks
- CH4: 237 negative/mixed coupling chunks, 113 inspect chunks
- CH5: 266 negative/mixed coupling chunks, 70 inspect chunks
- CH7: 292 negative/mixed coupling chunks, 140 inspect chunks

Do not assign production strategies from this audit alone. These ROIs should be
treated as ambiguous or potentially pathological until decision thresholds are
validated and manual review confirms whether observed events/artifacts are
biological, correction-reference failure, or mixed.

## E. Proposed Decision Contract

Future auto outputs must be exactly:

```text
dynamic_fit
signal_only_f0
needs_review
```

### `dynamic_fit`

Required evidence:

- Source phasic cache readable.
- ROI present.
- All expected chunks present or explicitly handled by an existing complete
  cache contract.
- `dff`, `time_sec`, `sig_raw`, `uv_raw`, and `fit_ref` present for required
  chunks.
- Dynamic-fit source dF/F finite enough for production requirements.
- Dynamic-fit QC is mostly `ok` or acceptable `context`.
- No recording-level pattern of negative/mixed reference coupling.
- No recording-level pattern of reference flat/uninformative or low range.
- Dynamic-fit slope is finite and stable enough.
- Signal/reference or signal/fitted-reference coupling is within validated
  dynamic-fit-supporting ranges.

Blocking evidence:

- Missing required dynamic-fit datasets.
- Non-finite or partial dynamic-fit source traces beyond policy.
- High fraction of hard dynamic-fit QC flags.
- High fraction of `NEGATIVE_OR_MIXED_REFERENCE_COUPLING`.
- High fraction of fitted-reference flat/uninformative flags.
- Non-finite slope summaries.
- Strong contradiction between dynamic-fit QC and trace statistics.

Warning/caution evidence:

- Small number of context/inspect chunks.
- Isolated negative/mixed coupling chunks.
- High-amplitude dF/F outliers that may be biological or artifacts.
- Low but not failing signal/control coupling.

Confidence/reporting behavior:

- `high` only if dynamic-fit QC is strongly consistent and signal-only F0 is not
  a stronger rescue candidate.
- `medium` if dynamic-fit is usable but has limited caution flags.
- Otherwise `needs_review`.

### `signal_only_f0`

Required evidence:

- Diagnosed correction-reference failure pattern from dynamic-fit QC.
- Signal-only F0 source `sig_raw` and `time_sec` present for all required
  chunks.
- Core uncapped signal-only F0 denominator available and valid for all required
  chunks.
- Signal-only F0 denominator is positive and finite.
- Negative dF/F preservation is possible and observed or explicitly checked.
- Signal-only F0 warnings are within accepted rescue-policy limits.

Blocking evidence:

- Missing signal-only F0 source data.
- Invalid denominator.
- Unusable/nonviable signal-only F0 candidate state.
- Too many hard-inspect chunks.
- Too much extrapolation, too little low support, too many large anchor gaps, or
  too few anchors beyond validated limits.
- Dynamic-fit evidence is usable and not clearly failed.

Warning/caution evidence:

- Contextual viability.
- Low or medium confidence.
- Edge extrapolation.
- Large anchor gaps.
- Few-anchor flags.
- High-state context or rolling fallback.

Confidence/reporting behavior:

- `high` should be rare and require strong correction-reference failure plus
  clean signal-only F0 viability.
- `medium` may be allowed for strong dynamic-fit failure with manageable
  signal-only F0 cautions.
- If signal-only F0 is viable but heavily caution-laden, emit
  `needs_review` unless real-data validation proves the rescue policy.

### `needs_review`

Required evidence:

- Any mixed, contradictory, missing, incomplete, or out-of-distribution evidence.

Blocking evidence:

- None. `needs_review` is the fail-closed decision.

Warning/caution evidence:

- Everything that prevents confident `dynamic_fit` or `signal_only_f0`.

Confidence/reporting behavior:

- Record why automatic routing did not proceed.
- Do not run production applied-dFF analysis automatically.
- Do not populate explicit manifest rows without user confirmation.

## F. Proposed Provenance Schema

These fields are proposed only. They should not be added to production outputs
in this audit task.

- `auto_strategy_decision`: one of `dynamic_fit`, `signal_only_f0`,
  `needs_review`.
- `auto_strategy_confidence`: `high`, `medium`, `low`, or `none`.
- `auto_strategy_decision_status`: `decided`, `blocked`, `needs_review`.
- `auto_strategy_evidence_json`: canonical JSON snapshot of all evidence used.
- `auto_strategy_warning_level`: `none`, `caution`, `severe`.
- `auto_strategy_review_required`: boolean.
- `auto_strategy_flags`: semicolon-delimited decision flags.
- `dynamic_fit_evidence`: structured dynamic-fit QC summary.
- `signal_only_f0_evidence`: structured signal-only F0 viability summary.
- `correction_reference_failure_evidence`: structured coupling/slope/reference
  failure evidence.
- `decision_blockers`: semicolon-delimited hard blockers.
- `decision_cautions`: semicolon-delimited cautions.
- `decision_rationale`: concise human-readable explanation.
- `decision_created_at_utc`: UTC timestamp.
- `decision_tool_name`: future read-only auto audit tool name.
- `decision_tool_version`: tool/schema version.
- `decision_contract_version`: decision contract version.
- `decision_input_rois`: ROI list or ROI.
- `decision_input_chunk_ids`: chunk IDs used.
- `source_phasic_cache_path`: source cache path.
- `source_phasic_cache_sha256`: source cache hash before/after audit.
- `legacy_features_sha256`: optional legacy feature hash before/after audit.
- `hdf5_modified_source_phasic_cache`: must be false.
- `legacy_features_modified`: must be false.
- `no_pipeline_execution`: true for read-only audit.
- `no_feature_routing`: true for read-only audit.

## G. Manual-Review Cases

Force `needs_review` when:

- Both strategies appear viable but evidence is not decisive.
- Dynamic-fit looks imperfect but not clearly invalid.
- Signal-only F0 denominator is weak or heavily extrapolated.
- Signal-only F0 has many low-confidence/contextual/hard-inspect chunks.
- Artifact/event structure may be biological rather than correction failure.
- Dynamic-fit QC and signal-only F0 diagnostics are contradictory.
- Too many chunks are missing, non-finite, or partial.
- Chunk-level evidence is inconsistent across recording segments.
- ROI behavior differs strongly across recording segments.
- Dynamic-fit has a small number of hard flags but no recording-level failure
  pattern.
- Signal-only F0 is viable but dynamic-fit also looks usable.
- Data are outside validated real-data examples.
- Evidence fields required by the contract are missing.
- Source cache schema/version is unknown.

## H. Future Test Plan

Synthetic tests:

- Clean dynamic-fit-supporting case classifies `dynamic_fit`.
- Correction-reference failure case classifies `signal_only_f0` only when
  signal-only F0 is viable.
- Ambiguous mixed-evidence case classifies `needs_review`.
- Missing dynamic-fit evidence classifies `needs_review`.
- Missing signal-only F0 evidence blocks `signal_only_f0` and classifies
  `needs_review`.
- Dynamic-fit available but signal/control decoupled classifies
  `signal_only_f0` or `needs_review`, depending on signal-only F0 viability.
- Signal-only F0 viable but dynamic-fit clean remains `dynamic_fit`.
- Excessive signal-only F0 extrapolation classifies `needs_review`.
- Non-finite chunks classify `needs_review`.
- Decision output is deterministic across repeated runs.
- Read-only audit never modifies source phasic cache or legacy features.
- Auto decision never silently routes feature extraction.

Real-data validation tests:

- Normal dataset should classify all or most relevant ROIs as `dynamic_fit`,
  depending on conservative thresholds.
- Weird CH8 should classify `signal_only_f0` or `needs_review` if thresholds
  are intentionally conservative.
- Weird CH9 should classify `dynamic_fit`.
- Other weird ROIs should remain reviewable until manually validated.
- Output provenance must include evidence JSON, hashes, flags, confidence, and
  rationale.

GUI/manifest tests for later stages:

- Read-only GUI display of proposed auto decision does not mutate manifest.
- User-confirmed "apply proposed strategies to manifest" requires explicit
  confirmation.
- No `auto` executable strategy is accepted by production batch until a separate
  implementation contract exists.

## I. Recommended Implementation Sequence

Stage A:

Add a read-only `tools/audit_applied_dff_auto_strategy_candidates.py` or extend
the existing advisory audit to aggregate dynamic-fit QC attributes, signal-only
F0 diagnostics, and provisional decision labels. It must not run production
analysis or write explicit manifests automatically.

Stage B:

Add synthetic and real-data validation tests. Thresholds should remain
conservative and should prefer `needs_review` over a forced binary decision.

Stage C:

Add GUI display of proposed auto decision as read-only evidence. Do not alter
user-selected strategies.

Stage D:

Add an explicit user-confirmed "apply proposed strategies to manifest" action.
This should still write explicit `dynamic_fit` / `signal_only_f0` manifest rows,
not an executable `auto` strategy.

Stage E:

Only after validation across more datasets, consider true batch auto execution.
That would require a separate production contract and should still be
provenance-recorded and non-silent.

## J. Files Inspected / Commands Run

Inspected:

- `tools/audit_applied_dff_strategy_candidates.py`
- `docs/applied_dff_production_chain_status.md`
- `docs/applied_dff_production_output_contract.md`
- normal source `phasic_trace_cache.h5`
- weird source `phasic_trace_cache.h5`
- existing GUI smoke output summaries under
  `scratch\gui_applied_dff_smoke_20260616`

Commands run:

```text
python tools/audit_applied_dff_strategy_candidates.py --phasic-out "C:\Users\Jeff\Documents\Test_Photometry_Output\Test_Outputs\other_photometry_20260615\run_20260615_200307_c4a6afc8\_analysis\phasic_out" --output-dir "scratch\auto_strategy_decision_contract_audit\normal_advisory_audit" --overwrite
```

```text
python tools/audit_applied_dff_strategy_candidates.py --phasic-out "C:\Users\Jeff\Documents\Test_Photometry_Output\Test_Outputs\student_weird_photometry_after_reset_20260614\run_20260614_165027_3ca29f64\_analysis\phasic_out" --output-dir "scratch\auto_strategy_decision_contract_audit\weird_advisory_audit" --overwrite
```

Additional read-only Python summaries were generated from source HDF5 and audit
outputs.

Created scratch outputs:

```text
scratch\auto_strategy_decision_contract_audit\normal_advisory_audit
scratch\auto_strategy_decision_contract_audit\weird_advisory_audit
scratch\auto_strategy_decision_contract_audit\cache_signal_reference_summary.csv
scratch\auto_strategy_decision_contract_audit\dynamic_fit_qc_attribute_summary.csv
```

## K. Blockers And Cautions

- Do not implement production auto yet.
- The current advisory audit does not aggregate the strongest dynamic-fit QC
  attributes; a future auto audit must add these before any decision logic.
- Signal-only F0 computability alone is not evidence that signal-only F0 should
  be selected.
- Dynamic-fit source availability alone is not evidence that dynamic-fit should
  be selected.
- The normal smoke output root contains both all-dynamic per-ROI summaries and
  an extra CH8 signal-only output; audit scripts should read the intended
  manifest/provenance or per-ROI outputs explicitly.
- Ambiguous/pathological ROIs in the weird dataset should remain manual until
  more real-data validation exists.
- Future auto must remain auditable, deterministic, provenance-rich, and
  fail-closed to `needs_review`.
