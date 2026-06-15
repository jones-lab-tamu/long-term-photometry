# Applied dF/F Production Chain Status

This checkpoint summarizes the current explicit applied_dff production chain.
It is documentation only and does not change code, tests, schemas, HDF5 outputs,
or runtime behavior.

## 1. Production Applied Cache Writing

Tool:

```text
tools/write_applied_dff_cache.py
```

Supported explicit strategies:

```text
dynamic_fit
signal_only_f0
```

Outputs:

```text
applied_trace_cache.h5
applied_correction_summary.csv
applied_correction_summary.json
applied_correction_chunks.csv
applied_correction_chunks.json
```

Key guarantees:

- Source `phasic_trace_cache.h5` is not modified.
- `dynamic_fit` `applied_dff` equals source phasic `dff`.
- `signal_only_f0` `applied_dff` uses the core uncapped
  `signal_only_f0` denominator.
- Negative dF/F is preserved.
- Auto strategy selection is not implemented.
- Production `no_correction` output is not implemented.
- The writer does not route feature detection.

## 2. Production Applied Cache Verification

Tool:

```text
tools/verify_applied_dff_cache.py
```

Verifies:

- Source phasic cache hash.
- Applied cache hash.
- Strategy-specific datasets.
- `dynamic_fit` formula: `applied_dff == source dff`.
- `signal_only_f0` formula:
  `(signal_raw_for_dff - signal_only_f0_uncapped_for_dff) / signal_only_f0_uncapped_for_dff`.
- Unavailable and partial chunk contract.
- Non-finite applied_dff flag propagation.
- No source phasic cache mutation.
- No legacy feature output mutation.

## 3. Production Applied Feature Running

Tool:

```text
tools/run_applied_dff_features.py
```

Behavior:

- Verifies the applied cache first by default.
- Requires a complete applied trace.
- Reads `applied_dff` from `applied_trace_cache.h5`.
- Writes separate applied-feature outputs under the applied output directory, or
  an explicitly supplied output directory.
- Does not modify legacy `<phasic_out>/features/features.csv`.
- Records feature config provenance.
- Records upstream applied warning and review provenance.
- Supports only explicit `dynamic_fit` and `signal_only_f0`.

## 4. Applied Feature-Output Semantic Verification

Tool:

```text
tools/verify_applied_dff_feature_outputs.py
```

Verifies:

- Feature outputs match direct `extract_features` output from `applied_dff`.
- Feature output granularity is classified.
- One-row-per-chunk output is allowed when it matches the detector contract.
- Required provenance columns are present.
- Feature-row provenance values are correct.
- Source phasic cache, applied cache, legacy feature outputs, and applied feature
  outputs are unchanged by verification.

## 5. Advisory Strategy Candidate Audit

Tool:

```text
tools/audit_applied_dff_strategy_candidates.py
```

Purpose:

- Provides a read-only advisory audit of explicit applied_dff strategy
  candidates.
- Helps the user manually decide whether to run `dynamic_fit` or
  `signal_only_f0` for each ROI.
- Summarizes evidence for manual review.
- Does not choose a strategy.
- Does not run the production pipeline.
- Does not write production applied outputs.
- Does not write feature outputs.

Supported candidate strategies audited:

```text
dynamic_fit
signal_only_f0
```

Outputs:

```text
applied_dff_strategy_candidate_audit.csv
applied_dff_strategy_candidate_audit.json
applied_dff_strategy_candidate_audit_summary.json
applied_dff_strategy_candidate_audit_provenance.json
```

Key guarantees:

- Advisory only.
- No `recommended_strategy`, `chosen_strategy`, `selected_strategy`, or
  `best_strategy` fields.
- Does not modify source `phasic_trace_cache.h5`.
- Does not modify legacy `<phasic_out>/features/features.csv`.
- Refuses unsafe audit output directories that could delete source cache or
  legacy features.
- Does not run `tools/run_applied_dff_pipeline.py`.
- Does not add auto-selection, GUI controls, production `no_correction`, or
  global routing.

Candidate evidence summarized for `dynamic_fit`:

- Required `dff` and `time_sec` coverage.
- Length checks.
- Non-finite counts.
- Summary statistics.
- Blocking issues and cautions.

Candidate evidence summarized for `signal_only_f0`:

- Required `sig_raw` and `time_sec` coverage.
- In-memory core F0 candidate computation.
- Viability counts.
- Confidence counts.
- Flag counts.
- Negative candidate dF/F presence.
- Summary statistics.
- Blocking issues and cautions.

## 6. Explicit End-To-End Orchestrator

Tool:

```text
tools/run_applied_dff_pipeline.py
```

Purpose:

- Provides a one-command explicit production chain for one ROI and one explicit
  strategy.
- Runs the four existing stages in order:

```text
write applied cache
verify applied cache
run applied features
verify feature outputs
```

- Owns sequencing and pipeline-level reporting only.
- The underlying tools still own correction, cache verification, feature
  extraction, and semantic verification.

Supported explicit strategies:

```text
dynamic_fit
signal_only_f0
```

Output layout:

```text
<output-root>/<ROI>_<strategy>/applied/
<output-root>/<ROI>_<strategy>/features/
<output-root>/<ROI>_<strategy>/pipeline/
```

Pipeline summary and provenance:

```text
applied_dff_pipeline_summary.json
applied_dff_pipeline_summary.csv
applied_dff_pipeline_provenance.json
```

Key guarantees:

- Fails immediately if any stage fails.
- Does not continue to downstream stages after upstream failure.
- Does not modify source `phasic_trace_cache.h5`.
- Does not modify legacy `<phasic_out>/features/features.csv`.
- Uses separate output-root outputs.
- Uses only explicit `dynamic_fit` or `signal_only_f0`.
- Rejects unsafe ROI output path components before overwrite deletion.
- Confines overwrite deletion to the resolved pipeline output directory inside
  `output_root`.
- Does not add auto-selection, GUI controls, production `no_correction` output,
  or global routing.

One-ROI manual production entry point:

After a strategy has been manually chosen for one ROI, the one-ROI orchestrator
can be run directly:

```text
python tools/run_applied_dff_pipeline.py --phasic-out ... --roi ... --strategy dynamic_fit|signal_only_f0 --output-root ... --overwrite
```

The four underlying tools remain the contract layers and are still useful for
debugging and stage-specific verification. The advisory audit is the recommended
pre-choice decision-support step. The orchestrator remains the
contract-preserving production unit after the user manually chooses an explicit
one-ROI, one-strategy production run.

## 7. Explicit Manifest Batch Runner

Tool:

```text
tools/run_applied_dff_batch.py
```

Purpose:

- Runs applied_dff production for multiple explicitly specified ROI/strategy
  pairs.
- Reads a user-provided manifest.
- Calls the existing one-ROI orchestrator for each manifest row.
- Writes combined batch summary and provenance.
- Does not infer, recommend, choose, or select strategies.

Supported manifest strategies:

```text
dynamic_fit
signal_only_f0
```

Rejected strategies:

```text
auto
no_correction
```

Manifest format:

```csv
roi,strategy
CH8,signal_only_f0
CH9,dynamic_fit
```

Optional manifest columns:

```text
output_name
feature_config
```

Batch outputs:

```text
<output-root>/batch/applied_dff_batch_summary.csv
<output-root>/batch/applied_dff_batch_summary.json
<output-root>/batch/applied_dff_batch_provenance.json
```

Per-row outputs remain under:

```text
<output-root>/<output_name>/
```

The default `output_name` is:

```text
<ROI>_<strategy>
```

Key guarantees:

- Explicit manifest only.
- No auto-selection.
- No automatic strategy choice from advisory audit.
- No `recommended_strategy`, `chosen_strategy`, `selected_strategy`, or
  `best_strategy` fields.
- Does not duplicate correction, cache writing, feature extraction, or
  verification logic.
- Calls `tools/run_applied_dff_pipeline.py` for each explicit row.
- Does not add GUI controls.
- Does not add global routing.
- Does not implement production `no_correction`.
- Does not replace legacy features.
- Hashes source `phasic_trace_cache.h5` before and after.
- Hashes legacy `<phasic_out>/features/features.csv` before and after when it
  exists.
- Refuses unsafe `output_root` and unsafe `output_name` values that could delete
  source cache or legacy features.
- Supports dry-run.
- Stops on first failed row by default.
- Supports `--continue-on-error` as an explicit option.

Recommended manual production workflow:

1. Run advisory candidate audit.
2. Manually choose `dynamic_fit` or `signal_only_f0` for each ROI.
3. Write an explicit ROI/strategy manifest.
4. Run the explicit batch runner.
5. Let the batch runner call the explicit one-ROI orchestrator for each row.

```text
python tools/audit_applied_dff_strategy_candidates.py --phasic-out ... --output-dir ... --overwrite
```

```text
python tools/run_applied_dff_batch.py --phasic-out ... --manifest ... --output-root ... --overwrite
```

```text
python tools/run_applied_dff_pipeline.py --phasic-out ... --roi ... --strategy dynamic_fit|signal_only_f0 --output-root ... --overwrite
```

The advisory audit is the recommended pre-choice decision-support step. The user
manually chooses strategies. The batch runner is the recommended convenience
layer once ROI/strategy choices are explicit in a manifest. The one-ROI
orchestrator remains the contract-preserving production unit. The four
underlying tools remain the contract layers for debugging and stage-specific
verification.

## 8. Current Real-Data Status

CH8 `signal_only_f0`:

- Applied cache complete: 581/581 chunks.
- Upstream warning/review: caution/review expected.
- Negative dF/F present.
- Applied feature output semantic status: pass.
- Feature granularity: `chunk_summary`.
- Observed feature rows: 581.
- Expected detector rows: 581.
- One row per chunk: true and matches detector.

CH9 `dynamic_fit`:

- Applied cache complete: 581/581 chunks.
- Applied feature output semantic status: pass.
- Feature granularity: `chunk_summary`.
- Observed feature rows: 581.
- Expected detector rows: 581.
- One row per chunk: true and matches detector.

Explicit batch runner manual verification:

- Batch rows: CH8 `signal_only_f0`, CH9 `dynamic_fit`.
- `batch_passed`: true.
- Rows completed: 2.
- Rows failed: 0.
- CH8 applied trace source: `signal_only_f0_dff`.
- CH9 applied trace source: `dynamic_fit_dff`.
- `semantic_status`: pass for both rows.
- `feature_output_granularity`: `chunk_summary` for both rows.
- `n_chunks_processed`: 581 for both rows.
- `n_features`: 581 for both rows.
- `one_feature_row_per_chunk_matches_detector`: true for both rows.
- Source phasic cache unchanged.
- Legacy features unchanged.
- No auto-selection, no strategy chosen, and no inference.

## 9. Intentionally Unsupported

The current chain does not implement:

- Auto strategy selection.
- Automatic strategy choice from the advisory audit.
- Production `no_correction` outputs.
- GUI controls.
- Global pipeline routing.
- Overwriting or replacing legacy feature outputs.
- Chunkwise strategy switching.
- Feature routing from an unverified applied cache.
- Partial-cache feature extraction by default.

## 10. Current Safety Rule

All production downstream applied_dff analysis must be explicit-strategy,
verified-cache, separate-output only.

## 11. Recommended Next Step

Use the advisory-audit plus explicit-manifest batch workflow on representative
recordings before adding any broader routing.

Do not add auto-selection yet. Do not add GUI controls yet. Do not add global
routing yet. Do not allow the advisory audit to choose strategies automatically.

The next implementation step, if needed later, should be limited to improving
reporting or manifest ergonomics, not changing strategy selection behavior.
