# Applied dF/F production output contract

This document is a design contract only. It does not change production code,
HDF5 contents, GUI behavior, strategy selection, feature routing, thresholds, or
preview-tool behavior.

The contract is for a future production implementation that turns an explicit
recording-level correction strategy into an auditable `applied_dff` trace and,
only when requested, routes downstream feature extraction to that trace.

## Current Implementation Status

The current production implementation is explicit-strategy only.

- Production applied-dFF execution supports explicit `dynamic_fit` and explicit
  `signal_only_f0`.
- `dynamic_fit` is the default manual production choice when the correction
  reference is usable.
- `signal_only_f0` is a rescue strategy for diagnosed correction-reference
  failure, not a default replacement for `dynamic_fit`.
- Production `no_correction` outputs are not implemented.
- Executable production `auto` is not implemented.
- `needs_review` is an audit label only, not a production strategy.
- `auto_strategy_decision` is a provisional read-only audit label only, not a
  runnable strategy.
- No current production tool performs silent fallback, automatic strategy
  inference, automatic manifest population, or chunkwise strategy switching.

Current GUI support is manual and explicit. The GUI can load ROIs from
`phasic_trace_cache.h5`, let the user include or exclude ROIs, assign explicit
`dynamic_fit` or `signal_only_f0` strategies to included ROIs, require dry-run
before batch execution, and write a manifest under a validated separate output
root. It does not choose strategies.

Current read-only audit support is evidence-only. Audit tools may write
candidate evidence, provisional labels, and hash provenance under a separate
audit output directory, but they do not write production applied caches, feature
outputs, manifests, or route downstream analysis. Audit output directories must
be separate from source `phasic_out` and legacy features.

## Current Preview Stack

The current preview stack is intentionally non-production.

- `tools/apply_recording_correction_strategy.py` reads the existing phasic cache,
  exports one explicit ROI-level strategy preview, writes CSV/JSON summaries and
  optional plots, and reports `hdf5_modified = false` and
  `feature_detection_input = false`.
- `tools/run_applied_dff_feature_preview.py` consumes an applied preview CSV and
  detects peaks from `applied_dff`; it does not segment full event boundaries,
  writes peak-only rows, and records peak detector/config provenance.
- `tools/verify_applied_dff_preview_stack.py` runs both preview stages as a
  checkpoint verifier, confirms the applied trace is complete before peak
  preview by default, confirms HDF5 and existing feature outputs are unchanged,
  and keeps intermediate preview outputs for inspection.

These tools validate the shape of the future contract, but their outputs remain
preview artifacts under QC/verification directories. They must not be treated as
canonical production feature outputs.

## Existing Cache And Output Structure

The current phasic trace cache is:

```text
<phasic_out>/phasic_trace_cache.h5
```

with the core structure:

```text
/meta/rois
/meta/chunk_ids
/meta/source_files
/meta/schema_version
/meta/n_chunks

/roi/<ROI>/chunk_<chunk_id>/time_sec
/roi/<ROI>/chunk_<chunk_id>/sig_raw
/roi/<ROI>/chunk_<chunk_id>/uv_raw
/roi/<ROI>/chunk_<chunk_id>/dff
/roi/<ROI>/chunk_<chunk_id>/fit_ref
/roi/<ROI>/chunk_<chunk_id>/delta_f
/roi/<ROI>/chunk_<chunk_id>/signal_only_f0_candidate
/roi/<ROI>/chunk_<chunk_id>/baseline_ref_candidate
```

`/roi/<ROI>/chunk_<chunk_id>/dff` currently means the existing dynamic/reference
corrected phasic dF/F produced by the standard pipeline. Downstream feature
extraction currently reads chunk objects and writes one row per chunk/ROI to:

```text
<phasic_out>/features/features.csv
```

with fields such as `chunk_id`, `source_file`, `roi`, `mean`, `median`, `std`,
`mad`, `peak_count`, and `auc`. Existing retune workflows write separate retune
directories and provenance JSON rather than mutating the original cache.

## Proposed Production Storage Location

Production `applied_dff` should be stored in a separate applied cache, not inside
`phasic_trace_cache.h5` for the first production implementation.

Recommended location:

```text
<phasic_out>/applied_dff/applied_trace_cache.h5
```

Rationale:

- The existing `phasic_trace_cache.h5` already has a stable meaning for
  `/roi/<ROI>/chunk_<chunk_id>/dff`.
- Writing `applied_dff` into the same cache risks legacy consumers reading the
  wrong trace or interpreting `dff` as silently redefined.
- A separate applied cache lets production code fail closed until downstream
  readers explicitly opt in to `applied_dff`.
- The applied cache can mirror the existing per-ROI/per-chunk layout while
  carrying a higher schema version and explicit production provenance.
- This follows the repo's retune pattern: derived/candidate outputs live beside
  the original run with their own provenance instead of overwriting original
  artifacts.

The production implementation may later add an opt-in pointer from the phasic
run report to the applied cache, but it should not overwrite or alias the
legacy `dff` dataset without a separate migration contract.

## Dataset And Schema Contract

The applied production cache should use one ROI/chunk group per expected chunk.
Schema versioning must be explicit.

Recommended HDF5 layout:

```text
<phasic_out>/applied_dff/applied_trace_cache.h5

/meta/schema_version
/meta/mode
/meta/source_phasic_cache_path
/meta/source_phasic_cache_sha256
/meta/rois
/meta/chunk_ids
/meta/source_files
/meta/created_at_utc
/meta/tool_name
/meta/tool_version
/meta/contract_name
/meta/contract_version

/recording/<ROI>/summary
/recording/<ROI>/provenance_json

/roi/<ROI>/chunk_<chunk_id>/time_sec
/roi/<ROI>/chunk_<chunk_id>/applied_dff
/roi/<ROI>/chunk_<chunk_id>/applied_trace_source
/roi/<ROI>/chunk_<chunk_id>/source_file
/roi/<ROI>/chunk_<chunk_id>/available
/roi/<ROI>/chunk_<chunk_id>/warning_level
/roi/<ROI>/chunk_<chunk_id>/review_required
/roi/<ROI>/chunk_<chunk_id>/flags

/roi/<ROI>/chunk_<chunk_id>/dynamic_fit_dff
/roi/<ROI>/chunk_<chunk_id>/signal_raw_for_dff
/roi/<ROI>/chunk_<chunk_id>/signal_only_f0_uncapped_for_dff
/roi/<ROI>/chunk_<chunk_id>/signal_only_f0_dff
/roi/<ROI>/chunk_<chunk_id>/denominator_trace
/roi/<ROI>/chunk_<chunk_id>/review_mask
/roi/<ROI>/chunk_<chunk_id>/caution_mask
```

Required dataset semantics:

- `applied_dff` is the trace selected by the one recording-level applied
  strategy for that ROI.
- `time_sec` must be copied from the source chunk and match `applied_dff` length.
- `dynamic_fit_dff` is optional but recommended when the applied strategy is
  `dynamic_fit` or when comparison traces are requested.
- `signal_raw_for_dff` is the numerator source trace used in the dF/F
  calculation. For the current phasic cache this must be copied exactly from
  `/roi/<ROI>/chunk_<chunk_id>/sig_raw` before any applied-dF/F computation. The
  production contract uses the explicit name `signal_raw_for_dff` instead of
  `signal` so consumers do not confuse it with a filtered, normalized, fitted,
  or display-only signal.
- `signal_raw_for_dff` and `signal_only_f0_uncapped_for_dff` are required when
  the applied strategy is `signal_only_f0`.
- `signal_only_f0_dff` is recommended when the applied strategy is
  `signal_only_f0`; it must equal `applied_dff` for available chunks under that
  strategy.
- `signal_only_f0_uncapped_for_dff` is the authoritative denominator dataset for
  `signal_only_f0`. If `denominator_trace` is also written, it must be an exact
  copy or explicitly documented alias of `signal_only_f0_uncapped_for_dff` for
  that strategy. Production readers should prefer the strategy-specific
  `signal_only_f0_uncapped_for_dff` path over the generic `denominator_trace`
  path when both are present.
- `denominator_trace` is optional in the first production schema if the
  strategy-specific denominator path is present. If a future strategy introduces
  a different explicit denominator, that strategy must define its own
  strategy-specific denominator dataset before using the generic alias.
- `review_mask` and `caution_mask` are optional in the first production version
  but, if present, must be boolean arrays with the same length as `applied_dff`.

Do not store `signal_only_f0_candidate` capped to the signal as the denominator
for `signal_only_f0_dff`. The denominator used for dF/F must be the uncapped
core state-aware F0 trace stored at `signal_only_f0_uncapped_for_dff` so negative
dF/F values remain possible.

## CSV Summary Contract

The HDF5 cache should be accompanied by CSV/JSON summaries:

```text
<phasic_out>/applied_dff/applied_correction_summary.csv
<phasic_out>/applied_dff/applied_correction_summary.json
<phasic_out>/applied_dff/applied_correction_chunks.csv
<phasic_out>/applied_dff/applied_correction_chunks.json
```

`applied_correction_summary.csv` should contain one row per ROI and applied
strategy. `applied_correction_chunks.csv` should contain one row per ROI/chunk.

The summary CSV/JSON are not substitutes for trace storage. They are the
human-readable provenance index for the applied cache.

## Required Provenance Fields

Production applied outputs must carry these fields at the recording/ROI summary
level and propagate the relevant subset into downstream feature outputs:

```text
roi
recording_key
requested_correction_strategy
correction_strategy_selection
applied_correction_strategy
applied_trace_source
applied_trace_units
applied_trace_available
applied_trace_complete
reason_if_unavailable
n_chunks
n_chunks_available
n_chunks_unavailable
applied_trace_review_required
applied_trace_warning_level
applied_trace_flags
source_phasic_cache_path
source_phasic_cache_sha256
applied_trace_cache_path
applied_trace_cache_sha256
correction_analysis_config_path
correction_analysis_config_json
correction_analysis_config_hash
correction_analysis_config_summary
strategy_evidence_source
strategy_evidence_policy
strategy_evidence_grouping_mode
strategy_evidence_generated_at
strategy_proposal_stale
strategy_proposal_stale_reason
f0_source_for_signal_only_f0
hdf5_modified_source_phasic_cache
feature_detection_input
created_at_utc
tool_name
tool_version
schema_version
```

Downstream feature/peak outputs that consume `applied_dff` must additionally
record:

```text
feature_detection_input_trace
feature_detection_input_strategy
feature_detection_input_source
feature_detection_input_units
feature_detection_input_applied_trace_complete
feature_detection_input_review_required
feature_detection_input_warning_level
feature_detection_input_warning_flags
feature_detection_preview
peak_detector_name
peak_detector_source_function
peak_detector_module
peak_detector_mode
peak_detection_config_source
peak_detection_config_path
peak_detection_event_signal
peak_detection_sampling_rate_source
peak_detection_config_json
peak_detection_config_hash
peak_detection_config_review_required
```

For production, `feature_detection_preview` must be `false`. Preview tools may
set it to `true`, but production feature tables must not inherit that value.

## Strategy-Specific Behavior

The current production strategy set is limited to explicit `dynamic_fit` and
explicit `signal_only_f0`. `needs_review` is a read-only audit label. `auto` is
not an executable production strategy.

`dynamic_fit`:

- `applied_trace_source = dynamic_fit_dff`.
- `applied_dff` is copied from the existing dynamic/reference corrected phasic
  `dff` trace.
- No signal-only F0 recompute is allowed as fallback.
- Missing or non-finite dynamic dF/F chunks make the applied output incomplete
  or blocked, depending on production policy.

`signal_only_f0`:

- `applied_trace_source = signal_only_f0_dff`.
- `signal_raw_for_dff` is copied exactly from the source phasic cache `sig_raw`
  for the same ROI/chunk and is the numerator trace for dF/F.
- `applied_dff = (signal_raw_for_dff - signal_only_f0_uncapped_for_dff) /
  signal_only_f0_uncapped_for_dff`.
- Negative dF/F values must not be clipped.
- The authoritative denominator trace must be stored at
  `signal_only_f0_uncapped_for_dff`.
- Missing signal, missing denominator, invalid denominator length, non-positive
  denominator, insufficient anchors, insufficient low support, large anchor gaps,
  high extrapolation, low confidence, and hard-inspect states must be propagated
  as chunk and summary warnings.
- No dynamic-fit fallback is allowed.

`no_correction`:

- `applied_trace_source = none`.
- `applied_trace_units = none`.
- `applied_trace_available = false`.
- `applied_trace_complete = false`.
- No `applied_dff` trace is produced.
- Production feature extraction from `applied_dff` must refuse this strategy.

## Completeness And Partial-Output Behavior

Completeness is defined as:

```text
applied_trace_complete =
    n_chunks > 0
    and n_chunks_available == n_chunks
    and n_chunks_unavailable == 0
```

Availability and completeness are different:

- `applied_trace_available = true` means at least one chunk produced applied
  trace samples.
- `applied_trace_complete = true` means every expected chunk produced applied
  trace samples.

If `n_chunks_available > 0` and `n_chunks_unavailable > 0`, production outputs
must add:

```text
APPLIED_TRACE_PARTIAL
```

to `applied_trace_flags`, set `applied_trace_review_required = true`, and raise
`applied_trace_warning_level` to at least `caution` without overriding `severe`.

Production feature extraction should refuse incomplete applied traces by default.
Any future partial-trace override must be explicit, recorded in provenance, and
must propagate `FEATURE_PREVIEW_PARTIAL_APPLIED_TRACE_INPUT` or a production
equivalent warning into downstream outputs.

## Warning And Review Propagation

Warnings are sticky. Downstream production feature detection must not clear,
downgrade, or hide warnings inherited from the applied trace.

Required propagation:

- `applied_trace_review_required` flows into
  `feature_detection_input_review_required`.
- `applied_trace_warning_level` flows into
  `feature_detection_input_warning_level`.
- `applied_trace_flags` flow into `feature_detection_input_warning_flags`.
- Downstream feature warnings may add additional flags, but must not remove
  applied-trace flags.

If downstream detection succeeds on a caution or severe applied trace, the output
is still caution or severe. Successful peak detection is not evidence that the
correction is trustworthy.

## Downstream Peak And Feature Routing Contract

Production feature extraction from applied traces must be an explicit route:

```text
input trace: applied_dff
input source: dynamic_fit_dff | signal_only_f0_dff
input strategy: dynamic_fit | signal_only_f0
input units: dff
```

Recommended production feature location:

```text
<phasic_out>/applied_dff/features/features.csv
<phasic_out>/applied_dff/features/feature_provenance.json
```

The existing legacy output:

```text
<phasic_out>/features/features.csv
```

should remain legacy dynamic/reference feature output until a separate migration
explicitly changes its meaning. Do not replace it silently.

Production feature rows should include the normal feature columns plus input
provenance columns:

```text
chunk_id
source_file
roi
mean
median
std
mad
peak_count
auc
feature_detection_input_trace
feature_detection_input_strategy
feature_detection_input_source
feature_detection_input_units
feature_detection_input_applied_trace_complete
feature_detection_input_review_required
feature_detection_input_warning_level
feature_detection_input_warning_flags
peak_detection_config_hash
applied_trace_cache_sha256
```

If production later emits event-level tables, those rows must carry the same
input provenance. Event boundary fields must only be populated by an event
boundary algorithm with a production contract. Peak-only rows must not invent
zero-crossing event boundaries.

## Relationship To Preview Tools

Preview and production outputs must remain distinguishable.

Preview outputs:

- live under QC, preview, or verification directories
- may write `feature_detection_preview = true`
- may use preview-default peak config provenance
- may write peak-only event rows
- must not modify source HDF5
- must not replace production `features.csv`

Production outputs:

- live under `<phasic_out>/applied_dff/`
- write an applied HDF5 cache and summary contract
- write `feature_detection_preview = false` if feature extraction is routed
- require explicit production config provenance
- may write production features only under the applied output namespace unless a
  migration intentionally changes legacy feature locations

Do not treat the preview default peak config as an inherited production config.
Production must use an explicit run/effective config or explicitly supplied
downstream config and must record its path, JSON snapshot, and hash.

## Dry-Run And Validation-Only Behavior

Dry-run/validation-only modes must not write applied caches, production feature
tables, production summaries, or HDF5 datasets.

They may report planned paths and validation outcomes:

```text
would_write_applied_trace_cache
would_write_applied_summary
would_route_feature_detection
validation_passed
blocking_reasons
```

Validation-only should perform the same refusal checks as production, including
strategy support, source cache readability, ROI/chunk coverage, stale proposal
checks, trace completeness, denominator validity, and downstream config validity.

## Refusal Cases And Blocking Pitfalls

Production implementation must refuse or block these cases:

- unsupported `requested_correction_strategy`
- ambiguous or missing ROI
- missing source phasic cache
- missing expected chunks
- stale strategy proposal used as current evidence without explicit override
- silent fallback from `dynamic_fit` to `signal_only_f0`
- silent fallback from `signal_only_f0` to `dynamic_fit`
- chunkwise strategy switching inside one ROI recording
- overwriting legacy `/roi/<ROI>/chunk_<chunk_id>/dff` without provenance
- overwriting legacy `<phasic_out>/features/features.csv` without migration
- treating preview default peak config as production config
- using capped F0 candidate as the signal-only dF/F denominator
- clipping negative signal-only dF/F values
- setting `feature_detection_input = true` without actually routing feature
  extraction to `applied_dff`
- clearing or downgrading applied warnings after downstream detection
- producing applied-dF/F features for `no_correction`
- mixing multiple ROIs or strategies in one single-ROI trace table without a
  multi-ROI schema
- reporting event start/end/AUC/duration from broad zero-crossing windows as
  production event boundaries

## Migration And Implementation Plan

1. Add production config fields and CLI/API parameters for explicit
   `requested_correction_strategy`.
2. Implement an applied-cache writer for
   `<phasic_out>/applied_dff/applied_trace_cache.h5` without changing
   `phasic_trace_cache.h5`.
3. Implement strict applied-summary and chunk-summary CSV/JSON outputs.
4. Implement dynamic-fit applied trace by copying existing `dff` into
   `applied_dff` with full provenance.
5. Implement signal-only F0 applied trace using uncapped state-aware F0,
   preserving denominator traces and warnings.
6. Implement `no_correction` as an explicit refusal/no-trace outcome.
7. Add production verifier checks for source cache hash, applied cache hash,
   completeness, warning propagation, and no source HDF5 modification.
8. Add opt-in downstream feature routing to the applied cache, writing features
   under `<phasic_out>/applied_dff/features/`.
9. Update run reports/status artifacts to point to applied output artifacts and
   record whether production feature extraction used `applied_dff`.
10. Validate read-only auto-strategy candidate audits across synthetic and
    real datasets before considering any executable auto behavior.
11. Consider read-only GUI display of auto-audit evidence without altering
    user-selected explicit strategies.
12. Consider a later user-confirmed apply-to-manifest workflow that still writes
    explicit `dynamic_fit` or `signal_only_f0` manifest rows, not executable
    `auto` rows.
13. Only after a separate validated production-auto contract exists, consider
    true batch auto execution.
14. Consider a later migration that either updates legacy feature locations or
    makes applied features the default. That migration needs its own contract.

## Open Questions

1. Should production initially allow partial applied traces with an explicit
   override, or hard-block all incomplete traces until a downstream missing-data
   policy exists?
2. Should the applied cache include comparison traces for non-applied strategies
   by default, or only the applied strategy plus denominator/source traces?
3. Should each ROI get its own applied cache, or should one cache hold all ROIs
   with multi-ROI summary rows? The first implementation should prefer one cache
   containing all requested ROIs only if a true multi-ROI schema is implemented.
4. What exact production feature-table name should be used once applied feature
   extraction is no longer preview-only?
5. Should auto be allowed to select `no_correction`, or should `no_correction`
   remain manual-only until uncorrected feature semantics are defined?
6. Should review/caution masks be required in version 1 of the applied cache or
   introduced in version 2 after chunk-level warning propagation is stable?
7. Which config snapshot should be authoritative when correction config and
   downstream peak config are supplied from different files?
