# 4J13v: Guided GUI Validate integration-boundary design

## 1. Executive decision

Future Guided Validate should use a thin GUI adapter over a new
backend-neutral application-service function. The service performs exactly:

```text
materialize -> compile -> validate -> return in-memory outcome
```

The GUI must not reimplement materialization, compiler, identity, or validator
rules. It owns only:

- capturing one immutable draft/context snapshot on the GUI thread;
- starting/canceling the operation;
- storing and invalidating the last in-memory outcome;
- presenting status and actionable issues;
- button busy/enabled state.

Backend modules own:

- filesystem fact materialization;
- request construction and identity;
- backend validation;
- deterministic stage-to-outcome translation.

Store the last result only in memory, bound to both canonical request identity
and a GUI draft revision. Any relevant state change invalidates it. An accepted
result does not unlock Run, create a saved plan, or create an artifact.

Missing or stale diagnostic cache state is a materialization refusal with an
actionable message. Validate must not generate the cache. Validate must not
call or mutate Full Control, `RunSpec`, argv/config builders, output allocators,
or runners.

## 2. Audit of current Guided GUI state and entry points

### Guided container and panels

Guided Mode is currently built inside `gui.main_window.MainWindow`, not a
separate controller class. Relevant builders include:

- `_build_guided_start_step`;
- `_build_guided_select_data_step`;
- `_build_guided_recording_structure_step`;
- `_build_guided_correction_approach_step`;
- `_build_guided_diagnostics_step`;
- `_build_guided_confirm_strategy_step`;
- `_build_guided_draft_plan_step`;
- `_build_guided_run_step`;
- `_build_guided_review_step`;
- `_build_guided_setup_summary_panel`.

`_build_guided_run_step` currently contains explanatory labels only. It states
that validation/run controls are a future stage and that the Guided step does
not execute analysis. There is no Guided Validate or Guided Run button to
reuse.

Classification: GUI presentation only and the appropriate future location for
a minimal Validate result panel/button.

### Current single draft builder

```python
MainWindow._build_guided_new_analysis_draft_plan()
```

This is the current single assembly point for
`GuidedNewAnalysisDraftPlan`. It reads Guided widgets and in-memory state for:

- input path, format, acquisition mode, timing, and incomplete-final flags;
- discovered/included/excluded ROIs;
- diagnostic-cache record, paths, identities, and currentness;
- explicit per-ROI strategy marks and evidence chunk;
- correction-preview identity/currentness;
- feature/event profile state;
- dataset contract snapshot;
- output policy state.

It also calls diagnostic-cache resolution helpers while building the draft.
That is read-only but means the method is not a pure state copier. Future
Validate should call it once on the GUI thread, freeze the resulting draft
snapshot, and pass that snapshot to backend orchestration. The backend service
must never call widgets or this method.

Classification: accepted GUI state-to-draft adapter for the first integration,
with a narrow cleanup prerequisite described in section 12.

### Guided state fields

Current in-memory state includes:

- `_guided_diagnostic_cache_record`, `_guided_diagnostic_cache_status`, pending
  request/cache/path fields, and a diagnostic-cache runner;
- `_guided_new_analysis_dataset_contract_snapshot` and candidate snapshot;
- `_guided_strategy_choices`;
- `_guided_preview_last_result`, preview result/currentness fields;
- `_guided_new_analysis_feature_event_profile`, status, errors, stale reasons,
  baseline provenance, and update timestamp;
- `_guided_new_analysis_output_policy_status`, path, validation issues, stale
  reasons, explicit-apply flag, source/cache/signature bindings;
- Guided ROI list and setup widgets;
- imported-plan review fields.

These are sources for draft construction, not validation proof.

### Existing mutation and stale-refresh paths

Relevant handlers include:

- `_sync_guided_setup_from_full` and `_connect_guided_setup_sync`;
- `_on_guided_roi_item_changed`;
- `_on_guided_apply_dataset_contract` /
  `_on_guided_clear_dataset_contract`;
- `_on_guided_mark_strategy_choice`;
- feature/event apply/clear handlers;
- output-policy apply/clear handlers;
- `_refresh_guided_new_analysis_dataset_contract_staleness`;
- `_refresh_guided_new_analysis_output_policy_staleness`;
- diagnostic-cache build/finish/error and preview-generation handlers.

These update separate pieces of state and refresh summaries, but there is no
central backend-validation revision or invalidation signal today.

Classification: state mutation sources that must call one new validation
snapshot invalidator. Existing local stale logic remains intact.

### Local checks and previews

`_refresh_guided_draft_run_plan_preview` calls planning/readiness/preview
helpers. `_guided_new_analysis_draft_plan_summary_text` also builds the older
`GuidedValidationRequest`, calls
`validate_guided_validation_request`, and displays a non-authorizing request
fingerprint and local setup issues.

Setup summaries, readiness evaluators, run previews, diagnostic previews, and
the older local validator are useful UI guidance. They are not backend
validation and must not be relabeled as such.

Classification: obsolete/local-check only for backend acceptance, but retained
as separate readiness presentation.

### Diagnostic cache

`_guided_diagnostic_cache_build_btn` calls
`_on_build_guided_diagnostic_cache`; cache generation allocates/writes cache
state and uses a runner. It is a separate explicit user action.

Classification: unsafe for Guided Validate. A missing/stale cache must refuse;
Validate never triggers this handler.

### Output policy

`_validate_guided_new_analysis_output_policy_path` calls
`validate_output_write_safety` and performs live path checks.
`_known_guided_new_analysis_protected_roots` derives cache/completed-run and
legacy subroots from GUI state.

Applied output policy is an input to the draft. Future Validate must not
reapply or mutate it. For the materializer's backend-neutral
`additional_protected_roots`, the GUI adapter should capture only explicit
root-kind/path pairs needed beyond source/cache facts, particularly a loaded
completed-run root. It should not pass display strings or duplicate legacy
subdirectories when the whole root is protected bidirectionally.

Classification: state source and presentation; write-operation validation
must not be invoked by the validation workflow.

### Full Control Validate and Run

```python
MainWindow._on_validate()
MainWindow._on_run()
MainWindow._build_run_spec(validate_only=False)
MainWindow._build_argv(validate_only=False, overwrite=False)
```

`_on_validate` validates Full Control widgets, calls `_build_argv`, records a
run directory, starts status/log followers, and starts `ProcessRunner`.

`_build_run_spec` calls `_generate_run_id`, calculates a run directory, and
constructs `RunSpec`.

`_build_argv` calls `os.makedirs`, writes effective config, validates it,
builds argv, and writes GUI intent and command records.

Classification: Full Control only and unsafe for Guided Validate.

### Current Guided/Full Control synchronization

`_connect_guided_setup_sync` currently synchronizes setup controls in both
directions between Guided and Full Control widgets. That is pre-existing setup
behavior. The future Validate click must not initiate synchronization or write
Full Control widgets. It snapshots the already-current Guided draft only.

Classification: existing state synchronization, not part of Validate.

## 3. Missing integration context identified by the audit

Materialization requires an explicit `RwdHeaderParsingContract`; no established
Guided GUI validation-context field currently owns that contract.
`RwdHeaderParsingContract` defaults do not provide the required candidate
tuples, so a button handler must not fabricate one from display text.

The validator also requires a `GuidedBackendValidatorContract`, including a
real capability version. The GUI currently has no accepted source/factory for
that runtime contract.

There is also no centralized Guided backend-validation revision or snapshot
invalidator.

Therefore, the safer next checkpoint is a narrow validation-context adapter
before button wiring. It should establish:

- the applied parser contract as immutable backend-neutral state, bound to the
  applied dataset/source context;
- the application-supplied validator contract;
- explicit additional protected roots;
- one monotonically increasing Guided draft revision;
- one invalidation entry point used by all draft-affecting handlers.

It must not infer parser semantics at click time or read Full Control widgets
as an alternate source.

## 4. Backend-neutral integration sequence

Recommended service signature:

```python
validate_current_guided_draft_for_backend(
    draft: GuidedNewAnalysisDraftPlan,
    *,
    parser_contract: RwdHeaderParsingContract,
    additional_protected_roots: tuple[tuple[str, str], ...],
    validator_contract: GuidedBackendValidatorContract,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedGuiBackendValidationOutcome
```

Place this in a new backend-neutral application-service module, for example:

`photometry_pipeline/guided_backend_validation_workflow.py`

Despite the GUI-facing outcome name, this module imports no GUI code.

Exact future sequence:

1. User clicks Guided Validate.
2. GUI verifies new-analysis mode and minimally materializable local state.
3. GUI increments/captures no state; it reads the current revision and creates
   one immutable draft/context snapshot on the GUI thread.
4. GUI passes only that snapshot and backend-neutral contracts to the service.
5. Service checks cancellation.
6. Service calls
   `materialize_guided_backend_validation_facts`.
7. On materialization failure, service returns
   `materialization_failed`.
8. Service checks cancellation.
9. Service calls `compile_guided_backend_validation_request`.
10. On compile failure, service returns `compile_failed`.
11. Service checks cancellation.
12. Service calls `validate_guided_backend_validation_request` with the
    compiled request, canonical identity, and same validator contract.
13. Service returns `validator_refused` or `validator_accepted`.
14. GUI compares the captured revision with the current revision.
15. If they differ, GUI stores/displays the outcome only as stale.
16. GUI presents the in-memory outcome and leaves Run disabled.

The service never calls the draft builder, GUI, Full Control, `RunSpec`, or
runner.

## 5. In-memory outcome model

Define a frozen backend-neutral outcome:

```python
@dataclass(frozen=True)
class GuidedGuiBackendValidationOutcome:
    status: Literal[
        "cancelled",
        "materialization_failed",
        "compile_failed",
        "validator_refused",
        "validator_accepted",
        "internal_error",
    ]
    accepted_for_backend_validation: bool
    run_authorization: bool
    request_identity: str | None
    validation_result: GuidedBackendValidationResult | None
    compile_result: GuidedBackendValidationCompileSuccess | None
    materialization_result:
        GuidedBackendValidationMaterializationSuccess | None
    blocking_issues: tuple[GuidedGuiBackendValidationIssue, ...]
    user_summary: str
    stale: bool = False
    no_files_written: bool = True
    no_directories_created: bool = True
    no_artifacts_created: bool = True
    no_run_id_allocated: bool = True
    no_runner_invoked: bool = True
```

The service outcome itself should not carry GUI widget references or a draft
object. A small normalized issue type should retain stage, category, section,
message, and detail code. Do not flatten issues into display text only.

Invariants:

- only `validator_accepted` has
  `accepted_for_backend_validation=True`;
- every outcome has `run_authorization=False`;
- accepted outcome has matching compile/validator request identity;
- each failed stage exposes only results from completed earlier stages;
- canceled outcome is neither accepted nor refused;
- no outcome contains config path, argv, command text, run directory, run ID,
  artifact path, timestamp, or completed-run marker;
- all outcome objects remain in memory.

Keeping successful materialization/compile objects in the outcome is useful
for immediate diagnostics, but the GUI should not persist them. If memory
weight becomes material, retain only the final validation result, identity,
and normalized issues after presentation.

## 6. GUI state and staleness policy

Add in-memory fields conceptually equivalent to:

```python
self._guided_backend_validation_revision = 0
self._guided_backend_validation_outcome = None
self._guided_backend_validation_outcome_revision = None
```

One method owns invalidation:

```python
def _invalidate_guided_backend_validation(self, reason: str) -> None:
    self._guided_backend_validation_revision += 1
    # Preserve the prior outcome only as explicitly stale, or clear it.
```

Recommended policy: preserve the last outcome for traceability but mark it
stale immediately; never present stale acceptance as current.

Invalidate on at least:

- input path, discovery result, format, acquisition mode, timing, timeline, or
  incomplete-final policy change;
- parser contract change;
- ROI discovery/inclusion/exclusion change;
- diagnostic cache record/root/currentness or any cache identity/signature
  change;
- correction preview identity/currentness change;
- strategy mark add/change/remove, selected strategy/mode, evidence reference,
  or evidence chunk change;
- dynamic-fit parameter contract change;
- dataset contract apply/clear/stale transition;
- feature/event profile/effective-value apply/clear/stale transition;
- output policy/path/currentness or protected-root context change;
- local blocking/unsupported/unresolved state change;
- mode switch, loaded completed-run change, or source context reset.

The request identity remains the backend equality token. The GUI revision
handles edits that occur while materialization is running, before a new request
identity exists.

Current Guided Run remains disabled regardless of outcome freshness.

## 7. Minimal user-facing presentation

Add one small validation panel to the existing Guided Run step; do not redesign
the stepper.

Accepted wording:

> Backend validation accepted the current Guided request. This confirms the
> request is structurally and semantically valid for the first Guided
> validation subset. It does not authorize or start a run.

Materialization failure:

> Guided setup is incomplete or stale. Fix the issue below and validate again.

Compile failure:

> The Guided request could not be compiled. Fix the issue below and validate
> again.

Validator refusal:

> Backend validation refused the current Guided request. Fix the issue below
> and validate again.

Stale:

> Guided validation is stale because the setup changed. Validate again before
> relying on this result.

Show the first issue by default and optionally expandable deterministic
details. Accepted state may show the canonical request identity as a
traceability token.

Never say “ready to run,” “completed,” or “run successful.” Do not show an
output directory, Run ID, config, command, execution progress bar, status-file
indicator, or artifact link.

## 8. Button behavior

The future Guided Validate button:

- exists only in new-analysis mode;
- is enabled when local state is sufficient to attempt materialization, not
  only when local readiness predicts acceptance;
- starts the in-memory workflow;
- enters a busy state while materialization runs;
- may expose Cancel because source snapshot materialization can be slow;
- disables repeated Validate clicks until the operation returns;
- never calls `MainWindow._on_validate`;
- never calls `_build_run_spec`, `_build_argv`, or `ProcessRunner.start`;
- never invokes diagnostic-cache generation;
- never mutates Full Control widgets;
- never enables Run.

The existing Guided Run step should continue to display Run as unavailable.
“Send to Full Control” or existing setup synchronization is not Guided Run and
must not be treated as validation acceptance or execution authorization.

## 9. Error and cancellation handling

Stage failures remain distinct:

- materialization failure: setup incomplete/stale;
- compile failure: request compilation failed;
- validator refusal: backend rejected an identified request;
- unexpected exception: safe `internal_error` summary without traceback or
  object repr.

Identity display rules:

- no identity before compile success;
- identity may be shown after compile success, including validator refusal;
- no validator result unless validator actually returned.

Cancellation checks occur before work and between all three stages.
Materialization already accepts `cancellation_check`; the service also checks
after materialization and after compilation. Cancellation returns `cancelled`,
never validator refusal.

Recommended prior-result policy: leave the previous outcome visible but stale
when a new attempt begins or is canceled. A canceled attempt never restores a
prior outcome to current.

## 10. No-write and no-run boundary

The workflow must not:

- open files in write/append/create modes;
- call `Path.write_text`, `write_bytes`, `touch`, or `mkdir`;
- call `os.mkdir`, `os.makedirs`, rename/replace, or temporary-file creation;
- reserve output, allocate Run ID, instantiate `RunSpec`, build argv/config/
  command text, or create status/report/manifest files;
- call `ProcessRunner`, `PipelineRunner`, `PhotometryPipeline.run`, Full
  Control `_on_validate`, or any production runner;
- mutate Full Control widgets or production output state;
- mark any directory as a completed run.

Important boundary clarification: materialization is intentionally read-only
but necessarily reads and stats source candidates and diagnostic-cache
metadata. Therefore a future end-to-end GUI Validate test cannot forbid all
`open`, `Path.read_*`, `os.stat`, or `os.scandir` calls. It must forbid write
modes and mutation APIs while allowing the audited materializer reads.
Pure compile and validator subtests should continue to forbid all filesystem
I/O.

Future tests should:

- fail on write-capable `open` modes while delegating read modes;
- fail on all directory/output mutation and allocation APIs;
- fail if Full Control validate, `RunSpec`, argv/config builders, process/
  pipeline runners, or artifact writers are called;
- snapshot the filesystem before/after and prove byte-for-byte no changes;
- verify no GUI production state, current run directory, validation reuse
  signature, or Full Control widget value changes;
- verify accepted, refused, canceled, stale, and internal-error outcomes all
  preserve no-write/no-run assertions.

## 11. Relationship to existing local checks

Local readiness remains a fast explanatory layer:

- it may determine whether Validate is minimally attemptable;
- it may highlight missing setup before materialization;
- it remains clearly labeled “local setup checks” or “draft readiness.”

Backend Validate is the first backend-owned acceptance of a compiled,
canonically identified request.

Do not:

- relabel `evaluate_new_analysis_plan_readiness` as backend validation;
- use setup-summary dictionaries as backend inputs;
- use the older `GuidedValidationRequest` fingerprint as canonical identity;
- treat diagnostic previews as validation;
- infer acceptance from “complete for future RunSpec handoff.”

The setup summary is read-only state presentation, not proof.

## 12. Recommended implementation sequence

The audit supports a two-step sequence because parser/validator contract
ownership and centralized stale invalidation are not yet established.

### 4J13w

Implement the narrow backend-neutral workflow/outcome model and the GUI
validation-context adapter:

- establish the immutable parser-contract source;
- establish the application validator-contract source;
- capture explicit protected roots;
- add the central draft revision/invalidation mechanism;
- test materialize/compile/validate orchestration without button wiring,
  artifacts, or Run unlock.

### 4J13x

Wire the minimal Guided Validate button/result panel to that adapter:

- presentation and cancellation only;
- in-memory outcome storage;
- no Full Control Validate path;
- no artifacts;
- Run remains unavailable.

### Later checkpoint

Design Run authorization separately from an accepted, fresh validation
request. Do not combine it with 4J13w or 4J13x.

## 13. Relationship to future Run

Accepted backend validation will be necessary but not sufficient for Run.
A future Run-authorization design must separately decide:

- whether and how the exact canonical request identity is reused;
- how freshness is proven at authorization time;
- how an accepted request maps to a runner/`RunSpec` contract;
- when output allocation and Run ID creation become authorized;
- whether a durable artifact records validated identity;
- how any draft change revokes authorization.

This document does not answer those execution questions. Current validation
always returns `run_authorization=False`, and Guided Run stays disabled.

## 14. Explicit non-goals

This checkpoint does not implement GUI code or tests. Future validation
integration still excludes Guided Run, Run eligibility, `RunSpec`, config/
argv/command generation, production or validation artifacts, directory
creation/reservation, Run ID allocation, Full Control changes, runner changes,
saved-plan adoption, broad UI redesign, 4J12f, build/default identity, full
source manifests, incomplete-final exclusion-enabled behavior, and stricter
ROI identity beyond the current request.

## 15. Tools and commands used

- `Get-Content` on the 4J13v request and audited source regions;
- `rg` for Guided builders, state fields, mutation handlers, stale refresh,
  materializer contracts, Full Control Validate/Run, `RunSpec`, argv/config,
  output allocation, and runners;
- `git status --short`.
