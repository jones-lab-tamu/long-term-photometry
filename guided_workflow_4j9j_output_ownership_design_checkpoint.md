# 4J9j Output Ownership Design Checkpoint

## 1. Current Output Models

### A. Full Control: GUI-Owned Run Directory

Full Control reads an output base, generates a unique run ID, derives `<output-base>/<run-id>`, and stores that path as the concrete run directory. `RunSpec` builds runner arguments with `--out <run-dir>`.

Before the runner starts, `_build_argv()` creates the run directory and writes:

- `config_effective.yaml`
- `gui_run_spec.json`
- `command_invoked.txt`

Validate-only and execution use separately generated run directories. Successful validation artifacts may later be cleaned up. The run path calls `_build_argv(..., overwrite=True)`, so `--overwrite` is passed for the GUI-created `--out` directory. This is compatible with current Full Control, but it means path creation and provenance writes occur before subprocess execution.

### B. Runner-Owned Output Base

With `--out-base <output-base>`, the runner generates a run ID unless one is explicitly supplied, then resolves `<output-base>/<run-id>`. `--overwrite` is ignored because each run ID is expected to be unique.

The runner can place status, events, cancellation, manifest, and production artifacts under that resolved run directory. The directory is not required during planning. However, current validate-only behavior calls `os.makedirs(run_dir, exist_ok=True)` and writes validation status/events there. Therefore `--out-base --validate-only` is not a no-write operation: it creates or reserves a run directory.

This model supports no-preview-write behavior because preview can retain only the base and run-directory pattern. The concrete final path is not known until a run ID is assigned.

### C. Guided: Current Non-Writing Policy

Guided currently stores only:

- applied output base
- `path_role="output_base"`
- `run_directory_strategy="derive_unique_run_id_under_output_base"`
- `future_run_dir="unresolved_until_execution_start"`
- `overwrite=False`
- no preview-time creation
- no config or command writes
- no validation or execution

`execution_available` remains false.

## 2. Candidate Ownership Models

### Model 1: GUI-Owned Future Run Directory With `--out`

- **Safety:** weaker for Guided because the GUI must create the directory before runner ownership begins.
- **Provenance:** strong under current Full Control because GUI intent/config/command files are colocated before launch.
- **Compatibility:** directly compatible with current `RunSpec`.
- **Validation:** follows existing Full Control behavior but writes before validation starts and uses separate validation/run directories.
- **Risk:** highest risk of hidden pre-execution writes and accidental overwrite semantics.
- **Guided fit:** poor. It conflicts with the accepted `output_base` and no-precreate policy.

### Model 2: Runner-Owned Run Directory With `--out-base`

- **Safety:** strongest production ownership boundary; only the runner creates the final run directory after an explicit action.
- **Provenance:** the runner can record the assigned run ID and actual path as authoritative runtime provenance.
- **Compatibility:** supported by the runner, but current `RunSpec` is `--out`-oriented and must not be reused without a later explicit mapping design.
- **Validation:** current validate-only creates a run directory, so validation cannot be treated as read-only.
- **Preview:** show output base plus a pattern such as `<output-base>/<runner-generated-run-id>`, never a promised concrete path.
- **Guided fit:** best match for the accepted policy.

### Model 3: Separate Validation/Preflight Namespace

- **Safety:** cleanly separates validation artifacts from production outputs.
- **Provenance:** requires a signed or hashed effective-plan link proving validation and execution used identical inputs.
- **Complexity:** substantial; users could confuse preflight and production packages.
- **Decision:** plausible later, but premature before validation semantics and effective-config identity are designed.

### Model 4: Hybrid Or Staged Ownership

Planning can remain base-only, a future mapping preview can show a path pattern, validation can later reserve a runner-owned directory, and execution can reuse or replace it with linked provenance.

This could preserve identity across validation and execution, but reservation, abandonment, retry, cleanup, and stale-validation rules make it too complex for the next stage. It should not be selected until validation itself is designed.

## 3. Contract Risks

- **Planning creates directories:** prohibited; path classification must be non-creating.
- **Pre-action artifact writes:** config and command artifacts must not be written before explicit validation or execution.
- **Overwrite:** Guided must not pass or imply `--overwrite` without a separate explicit user decision; the first subset should keep it disabled.
- **Source/output overlap:** reject equal paths, output inside source, and source inside output.
- **Protected output locations:** reject completed-run, legacy-output, diagnostic-cache, and other protected roots.
- **Validation confusion:** validation artifacts must be visibly non-production and must not satisfy completed-run contracts.
- **Diagnostic/production mixing:** final outputs must not be placed in diagnostic-cache or preview namespaces.
- **Preview/actual mismatch:** preview must display a pattern and ownership rule, not claim a concrete path before run-ID assignment.
- **Reproducibility:** runner-owned creation still requires durable effective-plan/config identity, exact invocation, assigned run ID, and source/plan provenance at explicit action time.

## 4. Required Shared Output Safety Helper

A shared output safety classifier should precede executable mapping. The existing write-safety validator is target-oriented and filesystem-aware; Guided needs a base-oriented, non-creating classification suitable for planning.

The classifier should separate:

1. Pure lexical/path relationships:
   - missing or relative path
   - base equals source
   - base inside source
   - source inside base
   - completed-run, legacy-output, diagnostic-cache, preview, or protected root
   - unsafe overwrite intent
   - planning, validation, or execution context

2. Optional read-only filesystem facts:
   - path exists as file or directory
   - directory empty/non-empty/unknown
   - writable/not writable/unknown
   - creation required

Unknown filesystem facts should remain explicit rather than silently passing. The helper must never create, reserve, delete, or write a path. It should be designed for eventual reuse by Guided and Full Control, but the first integration should be Guided-only so Full Control behavior does not change accidentally.

## 5. Recommended Next Implementation

**4J9k: model-only shared output-base safety and ownership classifier, integrated into Guided preview only.**

It should classify the applied Guided output base, source/protected-root relationships, filesystem facts when explicitly supplied or gathered read-only, write context, overwrite intent, and future owner. Its result should feed the non-writing execution-spec preview and block future mapping when unsafe or unresolved. It should not derive a concrete run ID or instantiate execution objects.

Do not proceed to executable mapping while production ownership, validation side effects, and safety classifications remain implicit.

## 6. No-Go Boundaries

- No `RunSpec` instantiation.
- No argv generation.
- No config, YAML, JSON, or command writing.
- No output directory creation or reservation.
- No validation or pipeline run.
- No final Guided Run.
- No saved-plan restore/adoption.
- No overwrite behavior.
- No Full Control behavior changes unless separately scoped.
- No transient GUI or Full Control state used to fill Guided plan fields.
- No diagnostic-cache, validation, or production artifacts.
- No concrete preview run path presented as guaranteed.
- No treating current `--out-base --validate-only` as read-only.

## Conclusion

Recommended output ownership model:
Runner-owned production run directory under the applied output base, created only after an explicit future execution action.

Why:
It matches Guided's accepted base-only policy and keeps final directory creation inside the authoritative runner, while current validate-only side effects remain explicitly deferred.

Recommended next implementation:
4J9k: model-only shared output-base safety and ownership classifier, integrated into Guided preview only.

Required guardrails:
- Classify source/output/protected-root relationships and overwrite intent without creating or reserving paths.
- Keep preview output as a base plus runner-owned path pattern; record the actual run ID only at future execution time.
- Treat filesystem facts and validation ownership as explicit known/unknown states, never silent assumptions.

Do not implement yet:
- RunSpec/argv/config mapping or production run-directory creation.
- Validation directory reservation, preflight namespace, or validate-to-run reuse.
