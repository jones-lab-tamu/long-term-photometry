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

## 5. Explicit End-To-End Orchestrator

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

Recommended manual production entry point:

```text
python tools/run_applied_dff_pipeline.py --phasic-out ... --roi ... --strategy dynamic_fit|signal_only_f0 --output-root ... --overwrite
```

The four underlying tools remain the contract layers and are still useful for
debugging and stage-specific verification. The orchestrator is the safest manual
entry point for explicit one-ROI, one-strategy production runs.

## 6. Current Real-Data Status

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

## 7. Intentionally Unsupported

The current chain does not implement:

- Auto strategy selection.
- Production `no_correction` outputs.
- GUI controls.
- Global pipeline routing.
- Overwriting or replacing legacy feature outputs.
- Chunkwise strategy switching.
- Feature routing from an unverified applied cache.
- Partial-cache feature extraction by default.

## 8. Current Safety Rule

All production downstream applied_dff analysis must be explicit-strategy,
verified-cache, separate-output only.

## 9. Recommended Next Step

Add a manual strategy decision/audit report to help choose between `dynamic_fit`
and `signal_only_f0` for each ROI.

This should be advisory only. It should not choose or run a strategy
automatically, add GUI controls, or add global routing. Its purpose should be to
help the user decide which explicit strategy to pass to the orchestrator.

Do not add auto-selection, GUI controls, production `no_correction` output, or
global routing before the manual explicit-strategy chain remains stable.
