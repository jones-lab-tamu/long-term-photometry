# Stage 4J8f Remaining Execution-Readiness Blocker Audit

## Scope And Contract Check

This is a read-only planning audit of the current post-4J8e Guided `new_analysis` execution-subset readiness state. No code behavior changes are proposed or implemented here.

1. End-user workflow served by the next stage:
   - Move Guided `new_analysis` from "complete plan plus non-executing preview" toward "explicit plan state that can later produce a pure execution/config preview."
   - The immediate user value is removing ambiguous execution-readiness blockers by making remaining execution intent and output/write policy explicit.

2. Workflow classification:
   - Core workflow, not secondary utility. It directly serves input -> plan -> execution -> review.

3. Movement toward input -> plan -> execution -> review:
   - Yes. The remaining blockers are the bridge between an accepted plan and a future execution-spec/config preview.

4. Contract sections touched by the next likely stage:
   - Execution intent fields: mode, run profile, timeline anchor.
   - Output/write boundary: creation timing, overwrite behavior, run directory derivation.
   - Preview/readiness classification: consume only stored plan state and documented fixed defaults.

5. Boundaries at risk:
   - Accidentally implying "planning complete" means "ready to execute."
   - Creating output directories during preview/readiness.
   - Silently choosing scientific timeline semantics without provenance.
   - Collapsing per-ROI choices into executable strategy without explicit first-subset rules.
   - Reusing Full Control widgets/transient state instead of Guided plan state.

## Inspected Code And Tests

- `photometry_pipeline/guided_new_analysis_plan.py`
  - `_execution_field_classifications(plan)`
  - `evaluate_guided_new_analysis_execution_subset_readiness(plan)`
  - `build_guided_new_analysis_run_preview(plan)`
- `gui/main_window.py`
  - `_build_guided_new_analysis_draft_plan`
  - `_guided_new_analysis_run_preview_text`
  - `_guided_new_analysis_execution_subset_text`
  - output policy helpers
  - Full Control `_build_run_spec` and `_build_argv` for comparison only
- `gui/run_spec.py`
  - RunSpec fields, timeline anchor choices, run profile choices, argv mapping, overwrite
- `tools/run_full_pipeline_deliverables.py`
  - runner mode/run-type/timeline/output behavior
- Tests:
  - `tests/test_guided_new_analysis_plan_contract.py`
  - `tests/test_gui_guided_new_analysis_plan.py`
  - `tests/test_gui_guided_workflow.py`

## Best-Case RWD Blocker Inventory

Best-case model plan audited:

- `input_format="rwd"`
- `acquisition_mode="intermittent"`
- valid source/acquisition/ROI setup
- current diagnostic cache
- one included ROI with explicit `global_linear_regression`
- applied feature/event profile
- applied output policy
- consistent current-applied RWD dataset contract snapshot

Actual `evaluate_guided_new_analysis_execution_subset_readiness(plan)` result:

- `planning_complete_for_handoff=True`
- `first_subset_executable=False`
- `execution_available=False`
- `allowed_dynamic_fit_strategy="global_linear_regression"`

Actual blocking issue categories:

| Category | Current field | Classification | Audit classification |
|---|---|---|---|
| `missing_timeline_anchor_mode` | `timeline_anchor_mode` | `required_missing`, blocks subset | Missing explicit plan field; possible fixed-default candidate; scientific/provenance-sensitive |
| `missing_execution_mode` | `mode` | `required_missing`, blocks subset | Missing explicit plan field; execution-spec mapping prerequisite; possible first-subset fixed default if documented |
| `missing_run_profile` | `run_profile` | `required_missing`, blocks subset | Missing explicit plan field; likely fixed-default candidate for first subset |
| `missing_output_creation_policy` | `output_creation_policy` | `required_missing`, blocks subset | Output/write boundary policy; should be explicit plan state or explicit documented fixed policy |

Relevant non-blocking field classifications in the best case:

- `dataset_contract_snapshot`: `present`, not blocking.
- `dataset_contract_overrides`: `present`, not blocking.
- `roi_identity`: `present`, not blocking.
- `acquisition_repair_fields`: `present`, not blocking.
- `fixed_daily_anchor_clock`: `fixed_default`, not blocking.
- `render_modes`: `fixed_default`, not blocking.
- `traces_only`: `fixed_default=False`, not blocking.
- `preview_first_n`: `fixed_default=None`, not blocking.
- `representative_session`: `deferred_full_control`, not blocking.
- `validate_only_behavior`: fixed classification only, not blocking.

Separate non-executing run preview unresolved item:

- `per_roi_correction_execution_contract_unresolved`

This item is emitted by `build_guided_new_analysis_run_preview(plan)` whenever any per-ROI choices exist. After 4J8e, the execution-subset helper already permits a single shared dynamic-fit-family strategy. Therefore this preview unresolved item is now likely an obsolete display/contract blocker for the first subset, or at least needs refinement so it only blocks mixed/unsupported per-ROI execution mappings.

## Fixed-Default Candidates

### `run_profile`

- Candidate value: `full`.
- Existing backend/Full Control equivalence:
  - `RunSpec.run_profile` defaults to `"full"`.
  - runner `--run-type` defaults to `"full"`.
  - Full Control UI default is "Full Run".
- Safe to record as provenance:
  - Yes, as a first-subset fixed default: `run_profile="full"`.
- Scientific meaning:
  - Mostly packaging/workflow intent, not signal-analysis science, but it changes output families and should be visible in provenance.
- Tests:
  - Pure model test that first subset classifies `run_profile` as fixed/default `full`.
  - Preview text shows `run_profile: full (first-subset fixed default)`.
  - Future execution-spec preview would omit `--run-type` or explicitly record equivalent default.

### `timeline_anchor_mode`

- Candidate value: `civil`.
- Existing backend/Full Control equivalence:
  - `RunSpec.timeline_anchor_mode` defaults to `"civil"`.
  - runner `--timeline-anchor-mode` choices are `civil`, `elapsed`, `fixed_daily_anchor`; default is `"civil"`.
  - Full Control UI default is "Civil clock".
- Safe to record as provenance:
  - Technically yes if explicitly documented as `timeline_anchor_mode="civil"` and `fixed_daily_anchor_clock=None`.
- Scientific meaning:
  - Yes. It affects phasic/dayplot hour/day placement and long-duration/circadian interpretation.
- Tests:
  - Pure model test that a documented fixed default records `"civil"` and does not require a fixed anchor clock.
  - Preview test showing timeline anchor provenance.
  - Future argv/spec preview test proving no `--timeline-anchor-mode` flag is needed because runner default is equivalent.
- Audit caution:
  - Although a fixed default is possible, this is scientifically meaningful. If Guided users need circadian alignment decisions, it should become user-visible rather than hidden.

### `execution_mode`

- Candidate values:
  - `phasic` for the first dynamic-fit-only subset, or
  - `both` to match Full Control/runner default.
- Existing backend/Full Control equivalence:
  - Full Control mode widget defaults to `both`.
  - `RunSpec.mode=None` represents `both` and emits no `--mode`.
  - runner `--mode` default is `both`.
- Safe to record as provenance:
  - Yes, but choosing the wrong default changes output scope.
- Scientific/workflow meaning:
  - Workflow significant. `both` runs tonic and phasic families; `phasic` is narrower and more aligned with dynamic-fit-first execution.
- Tests:
  - Pure model test for the chosen first-subset value.
  - Preview test that does not imply unsupported tonic behavior if `both` is chosen.
  - Future execution-spec preview test proving mode maps exactly.
- Audit caution:
  - This is not merely implementation detail. Because the first executable subset was framed as global dynamic-fit-only, `phasic` may be the safer first-subset restriction, but it differs from Full Control default and must be explicit.

## User-Visible Control Candidates

### Timeline anchor

- What the user should see:
  - Timeline anchor selector with Civil clock, Elapsed from first session, Fixed daily anchor.
  - Fixed anchor time field only when fixed mode is selected.
- Where in Guided:
  - Source/setup or Draft Plan execution-intent section, before final execution preview.
- Explicit Apply:
  - Recommended. Timeline semantics should be reviewed/applied as durable Guided plan state.
- Staleness:
  - Changing acquisition mode, source identity, or fixed anchor value should stale the applied execution-intent snapshot.
- Preview:
  - Show selected anchor mode, fixed anchor clock if relevant, provenance, and whether it is first-subset-supported.

### Execution mode

- What the user should see:
  - First-subset execution mode display. If first subset is fixed to `phasic`, show it as a restriction, not a general mode selector.
- Where in Guided:
  - Draft Plan execution-readiness/preview area.
- Explicit Apply:
  - If fixed first-subset default: no separate Apply, but record it as a documented fixed contract default.
  - If selectable: require Apply.
- Staleness:
  - If selectable, changes in correction strategy family or feature/event requirements could stale it.
- Preview:
  - Show `mode: phasic` or `mode: both` plus rationale.

### Run profile

- What the user should see:
  - For first subset, likely "Full production package" as a fixed default or read-only selected profile.
- Where in Guided:
  - Draft Plan execution-readiness/preview area.
- Explicit Apply:
  - Not necessary if fixed to `full`; required if selectable later.
- Staleness:
  - None for a fixed first-subset default.
- Preview:
  - Show `run_profile: full` and that execution remains unavailable until future execution-spec stage.

## Output/Write Boundary Audit

Current state:

- Guided `output_policy_path` records a validated destination path and safety status.
- It does not define creation timing, unique run folder derivation, overwrite behavior, runner `--out` versus `--out-base`, or whether the path is a final run directory or an output base.
- Full Control `_build_run_spec` computes `run_dir = output_base/run_id`.
- Full Control `_build_argv` creates `run_dir`, writes `config_effective.yaml`, writes `gui_run_spec.json`, writes `command_invoked.txt`, validates config, and builds argv.
- Full Control `RunSpec.build_runner_argv()` uses `--out <run_dir>`.
- Runner supports:
  - `--out`: explicit run directory, legacy behavior; existing directory requires `--overwrite`.
  - `--out-base`: runner creates `<out-base>/<run-id>`; `--overwrite` is ignored in this GUI mode.

What `output_creation_policy` means distinct from `output_policy_path`:

- Whether the applied path is treated as:
  - a final run directory, or
  - an output base under which a unique run directory will be derived.
- When the directory is created.
- Whether existing targets are allowed.
- Whether overwrite is allowed.
- Whether GUI or runner owns run-id generation.
- Which files may be written by GUI before runner execution.

Safest first-subset policy:

- Treat `output_policy_path` as an output base or planned destination root, not as permission to write during planning.
- At future execution start only, derive a unique run directory under that root.
- `overwrite=False`.
- fail if derived target already exists.
- no directory creation during readiness or preview.
- no config/YAML/JSON writing until a later explicit execution/validation stage.

Should it be fixed or explicit:

- It should become explicit plan state, even if the first subset only allows one value:
  - `creation_timing="execution_start_only"`
  - `run_directory_strategy="derive_unique_run_id_under_output_policy_path"`
  - `overwrite=False`
  - `precreate_during_preview=False`
- Reason: this is the primary write boundary. It should be reviewed and visible.

Deferred:

- Overwrite support.
- Reusing an existing run directory.
- Saved-plan adoption/import into a runnable output target.
- GUI prewriting `config_effective.yaml`, `gui_run_spec.json`, or command logs.

## Run Profile And Execution Mode Audit

### `missing_execution_mode`

Current backend/Full Control:

- Full Control mode choices: `both`, `phasic`, `tonic`.
- Full Control default: `both`.
- RunSpec encodes `both` as `mode=None`, emitting no `--mode`.
- Runner default: `--mode both`.

Audit:

- If the first executable subset is truly "global dynamic-fit-only," `phasic` is the tighter and safer first-subset restriction.
- If the first execution target is "normal production deliverables," `both` matches backend default but may accidentally include tonic behavior outside the subset.
- Recommendation: do not silently choose `both`. Define a first-subset execution intent field with a documented value, probably `phasic`, and explain that tonic remains deferred.

### `missing_run_profile`

Current backend/Full Control:

- Choices: `full`, `tuning_prep`.
- Full Control default and runner default: `full`.
- `tuning_prep` intentionally skips nonessential outputs.

Audit:

- First Guided execution should likely be `full` if the user is moving from plan to production review.
- This can be a documented fixed default because it matches backend/Full Control default.
- Risk if wrong: choosing `tuning_prep` would make a "successful" Guided run omit expected deliverables.

## Timeline Anchor Audit

Current backend/Full Control:

- Choices: `civil`, `elapsed`, `fixed_daily_anchor`.
- Full Control default: `civil`.
- Runner default: `civil`.
- Fixed daily anchor requires `fixed_daily_anchor_clock`.

Does Guided setup already capture enough?

- No. GuidedNewAnalysisDraftPlan does not currently store `timeline_anchor_mode` or `fixed_daily_anchor_clock`.

Can first subset use a safe fixed default?

- Technically yes: `civil` matches current backend/Full Control default.
- Product/science caveat: timeline anchoring affects long-duration and circadian interpretation, so provenance must be explicit.

Should it be user-visible?

- Recommended eventually. For the immediate first subset, a documented fixed default is acceptable only if displayed clearly and tested.

Needed provenance:

- `timeline_anchor_mode="civil"`
- `fixed_daily_anchor_clock=None`
- source: `first_subset_fixed_default`
- note that users needing elapsed/fixed anchor must use a later Guided stage or Full Control.

## Execution-Spec Mapping Prerequisites

Before designing a pure Guided execution spec object:

- Plan fields or fixed defaults must exist for:
  - execution mode
  - run profile
  - timeline anchor mode
  - fixed daily anchor clock if applicable
  - output creation policy
- Output policy must separate:
  - validated destination/root
  - creation timing
  - unique run directory derivation
  - overwrite policy
- Per-ROI correction rules must be reconciled:
  - same dynamic-fit-family strategy across all included ROIs is first-subset-supported
  - mixed strategies remain planning-valid but execution-preview-blocked
  - Signal-Only F0 remains planning/diagnostic only
  - run preview should stop reporting generic per-ROI mapping unresolved when the first-subset rule is satisfied
- Dataset contract constraints must remain:
  - RWD current-applied consistent snapshot required
  - NPM intermittent requires explicit mapping fields if ever enabled
  - custom tabular requires explicit column mapping fields if ever enabled
  - NPM continuous unsupported
  - auto blocked
- Diagnostic cache relationship:
  - cache identity/provenance must stay current with source/setup and strategy choices
  - future execution spec must consume only current cache identity fields already in plan
- Feature/event state:
  - must remain explicitly applied, valid, current, and stored in plan
- No execution-spec design should call:
  - RunSpec
  - `_build_run_spec`
  - `_build_argv`
  - config writing
  - directory creation
  - validation or run process

## Recommended Next Stage

Recommended staged path:

1. `4J9a`: model-only execution intent and output creation policy contract defaults
   - Add pure model fields/classifications for:
     - timeline anchor fixed default or explicit field
     - execution mode first-subset value
     - run profile first-subset value
     - output creation policy as explicit non-writing plan state
   - Keep GUI out unless a field cannot be safely fixed.
   - No RunSpec, argv, config, directory, validation, or execution.

2. `4J9b`: GUI display/apply for output creation policy if 4J9a decides it cannot be only fixed
   - Focus on write-boundary review.
   - Do not create directories.

3. `4J9c`: readiness consumes the execution intent/output creation policy state
   - Remove `missing_timeline_anchor_mode`, `missing_execution_mode`, `missing_run_profile`, and `missing_output_creation_policy` only when fixed defaults or applied policy state are present.
   - Refine obsolete `per_roi_correction_execution_contract_unresolved` preview blocker for same-strategy dynamic-fit plans.

4. `4J10a`: pure Guided execution-spec design
   - Only after readiness has explicit/fixed state for all execution intent and write-boundary fields.

Why this is core workflow and not a rabbit hole:

- The remaining blockers are exactly the boundary between a complete Guided plan and a future executable preview.
- Output creation policy is the highest-risk write boundary and must be settled before any execution spec.
- Execution mode/run profile/timeline anchor determine what the future execution means; they are not saved-plan adoption, import/export, or unrelated infrastructure.
