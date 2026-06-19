# Guided/Full Control Contract Wiring Audit

## 1. Executive Summary

Guided Workflow is partially aligned with the workflow contract but still substantially completed-run-centered for the decisions that matter most. Full Control is the current executable path for raw NPM/custom_tabular input, acquisition settings, ROI inclusion, feature/event parameters, output selection, validation, and production run execution. Guided can mirror several Full Control raw-input setup widgets and can run the shared ROI discovery path, but its durable `GuidedRunPlan` draft state, correction evidence generation, per-ROI strategy confirmation, output policy planning, export/import, and adoption eligibility are all scoped to a loaded completed run. The largest contract gap is that Guided cannot currently generate correction evidence from raw NPM/custom_tabular input before a production or tuning-prep run exists.

## 2. Current Full Control Capabilities

Full Control is the executable workflow. The GUI builds a `RunSpec`, writes GUI-owned run intent artifacts, then launches `tools/run_full_pipeline_deliverables.py` through `PipelineRunner`.

Input formats:

- `RunSpec.FORMAT_CHOICES` exposes `("auto", "rwd", "npm", "custom_tabular")` in [gui/run_spec.py](gui/run_spec.py:33).
- The Full Control format combo uses those choices in [gui/main_window.py](gui/main_window.py:14918).
- The runner accepts `--format` choices `rwd`, `npm`, `custom_tabular`, and `auto` in [tools/run_full_pipeline_deliverables.py](tools/run_full_pipeline_deliverables.py:577).
- Backend discovery validates `{"auto", "rwd", "npm", "custom_tabular"}` in [photometry_pipeline/discovery.py](photometry_pipeline/discovery.py:56), and pipeline file discovery uses the same set in [photometry_pipeline/pipeline.py](photometry_pipeline/pipeline.py:872).

Acquisition modes and structure:

- Full Control exposes intermittent and continuous acquisition mode in [gui/main_window.py](gui/main_window.py:14980).
- Full Control exposes sessions/hour and session duration in [gui/main_window.py](gui/main_window.py:14947) and [gui/main_window.py](gui/main_window.py:14971).
- Full Control exposes continuous window, continuous step, and allow-partial-final-window controls in [gui/main_window.py](gui/main_window.py:15000) through [gui/main_window.py](gui/main_window.py:15039). The continuous step is disabled and synchronized to the window length.
- `RunSpec` carries `acquisition_mode`, `continuous_window_sec`, `continuous_step_sec`, and `allow_partial_final_window` in [gui/run_spec.py](gui/run_spec.py:70).
- `RunSpec.__post_init__` requires continuous step equal to window in [gui/run_spec.py](gui/run_spec.py:131).
- Runner continuous mode is limited: continuous NPM is explicitly unsupported, and continuous `auto` is considered ambiguous. Evidence: `CONTINUOUS_NPM_UNSUPPORTED_MESSAGE` and `CONTINUOUS_AUTO_FORMAT_MESSAGE` in [tools/run_full_pipeline_deliverables.py](tools/run_full_pipeline_deliverables.py:329), plus pipeline rejection for unsupported continuous formats in [photometry_pipeline/pipeline.py](photometry_pipeline/pipeline.py:995).
- Config has NPM and custom-tabular channel/timebase field names: `npm_*` and `custom_tabular_*` in [photometry_pipeline/config.py](photometry_pipeline/config.py:179). Full Control does not expose those as normal primary widgets; they are controlled through config YAML/default config.

ROI inclusion/exclusion:

- Full Control discovers ROIs through `RunSpec.run_discovery()` in [gui/run_spec.py](gui/run_spec.py:412), which invokes runner `--discover`.
- `_on_discover()` builds a discovery spec and populates ROI UI in [gui/main_window.py](gui/main_window.py:12034).
- `_populate_discovery_ui()` checks all discovered ROIs by default in [gui/main_window.py](gui/main_window.py:12079).
- `_build_run_spec()` converts checked ROI state to `include_roi_ids`; all checked means no include filter, none checked means empty include list, and partial selection becomes explicit include list in [gui/main_window.py](gui/main_window.py:11385).
- `RunSpec.build_runner_argv()` emits `--include-rois` or `--exclude-rois` if present in [gui/run_spec.py](gui/run_spec.py:394).
- Backend pipeline applies include/exclude selections against discovered ROIs in [photometry_pipeline/pipeline.py](photometry_pipeline/pipeline.py:2018).

Feature/event defaults and parameters:

- Full Control uses `Config` defaults for event/feature fields in [photometry_pipeline/config.py](photometry_pipeline/config.py:151).
- GUI parsing and semantic validation are centralized in `parse_and_validate_event_feature_knobs()` in [gui/main_window.py](gui/main_window.py:1021), which calls `validate_feature_event_config_fields()` in [gui/main_window.py](gui/main_window.py:1148).
- Pure feature/event allowed values are in [photometry_pipeline/feature_event_config.py](photometry_pipeline/feature_event_config.py:8).
- `_build_run_spec()` applies changed event/feature overrides into `config_overrides` in [gui/main_window.py](gui/main_window.py:11581).

Correction strategy and preprocessing controls:

- Full Control exposes `dynamic_fit_mode` with allowed Config values including `global_linear_regression`, `robust_global_event_reject`, and `adaptive_event_gated_regression` in [photometry_pipeline/config.py](photometry_pipeline/config.py:50).
- Full Control builds dynamic fit mode controls in [gui/main_window.py](gui/main_window.py:15592) and maps selected controls into `config_overrides` in [gui/main_window.py](gui/main_window.py:11406).
- Signal-Only F0 exists in config/backend support, but Full Control's main production route is still dynamic-fit config driven. Explicit applied-dF/F batch has separate per-ROI strategy choices `dynamic_fit` and `signal_only_f0` in [gui/main_window.py](gui/main_window.py:9435).

Output destination and safety:

- Full Control has an output base directory widget in [gui/main_window.py](gui/main_window.py:14906).
- `_build_run_spec()` creates a unique `run_dir = out_base / run_id` and assigns `_current_run_dir` in [gui/main_window.py](gui/main_window.py:11282).
- `_build_argv()` creates the run directory and writes `config_effective.yaml`, `gui_run_spec.json`, and `command_invoked.txt` before validation/run in [gui/main_window.py](gui/main_window.py:11677).
- `RunSpec.generate_derived_config()`, `write_gui_run_spec()`, and `write_command_invoked()` are the GUI-owned writes in [gui/run_spec.py](gui/run_spec.py:179), [gui/run_spec.py](gui/run_spec.py:230), and [gui/run_spec.py](gui/run_spec.py:246).
- `_on_run()` always passes `overwrite=True` into `_build_argv()` in [gui/main_window.py](gui/main_window.py:12203). Because the GUI run directory is unique per generated run id, this does not normally overwrite a previous GUI run, but the runner receives `--overwrite`.
- Runner legacy `--out` mode refuses existing output directories unless `--overwrite` is supplied in [tools/run_full_pipeline_deliverables.py](tools/run_full_pipeline_deliverables.py:1503).
- I did not find a Full Control guard that blocks choosing an output base inside the input/source directory. This is a contract risk.

Execution boundary:

- Validate Only and Run Pipeline are Full Control actions in [gui/main_window.py](gui/main_window.py:12130) and [gui/main_window.py](gui/main_window.py:12184).
- `_on_run()` requires a current validation signature before starting the pipeline in [gui/main_window.py](gui/main_window.py:12214).

## 3. Current Guided Capabilities

Guided has both a raw-input setup branch and a completed-run branch, but only the completed-run branch has meaningful Guided plan state.

Raw input setup:

- Guided start offers "Set up new analysis" and "Open Results..." in [gui/main_window.py](gui/main_window.py:1689).
- Guided Select Data mirrors Full Control input folder, output folder, and format controls in [gui/main_window.py](gui/main_window.py:1786).
- Guided Recording Structure mirrors Full Control acquisition mode, sessions/hour, session duration, continuous window, partial-final-window, and RWD final-chunk exclusion controls in [gui/main_window.py](gui/main_window.py:1888).
- Guided setup controls directly synchronize into Full Control widgets through `_connect_guided_setup_sync()` in [gui/main_window.py](gui/main_window.py:1982). This means Guided raw setup is reusable by Full Control, but it is not independent durable Guided plan state.
- Guided ROI discovery calls `_on_discover()` and mirrors Full Control ROI checked state in [gui/main_window.py](gui/main_window.py:2365). All ROIs are checked by default because `_populate_discovery_ui()` checks all Full Control ROI items, then Guided mirrors them.
- ROI exclusion is possible before diagnostics through unchecked ROI list items, but this is Full Control shared widget state, not represented in `GuidedRunPlan` for new-analysis execution.

Correction approach:

- Guided correction cards include Robust Global Event-Reject Fit, Adaptive Event-Gated Fit, Global Linear Regression, Signal-Only F0, and Decision-Support Audit in [gui/main_window.py](gui/main_window.py:2418).
- Robust/Adaptive/Global cards update Full Control dynamic-fit mode only, not durable Guided strategy state, through `_select_guided_reference_correction_card()` in [gui/main_window.py](gui/main_window.py:2556).
- Signal-Only F0 records `_guided_correction_intent` only; it does not alter Full Control dynamic fit mode or route analysis in [gui/main_window.py](gui/main_window.py:2568).

Diagnostics and previews:

- Guided Diagnostics is explicitly completed-run based. The preview panel says it compares correction methods from a loaded completed-run cache in [gui/main_window.py](gui/main_window.py:2676).
- Correction preview enablement reads a completed-run phasic cache via `resolve_completed_run_preview_source()` and `open_phasic_cache()` in [gui/main_window.py](gui/main_window.py:3370).
- Signal-Only F0 diagnostic enablement reads a completed-run phasic cache via `resolve_completed_run_signal_only_f0_source()` and `open_phasic_cache()` in [gui/main_window.py](gui/main_window.py:3461).
- `_on_generate_guided_correction_preview()` calls `run_guided_correction_preview_comparison(... source_type="completed_run")` in [gui/main_window.py](gui/main_window.py:3655).
- `_on_generate_guided_signal_only_f0_diagnostic()` calls `run_signal_only_f0_diagnostic_review(run_dir, ...)` in [gui/main_window.py](gui/main_window.py:3604).
- The preview backend supports only `completed_run` and `phasic_cache` source types, not raw input, in [photometry_pipeline/preview/correction_preview.py](photometry_pipeline/preview/correction_preview.py:71).

Per-ROI strategy confirmation:

- Confirm Strategy is completed-run/cache-gated. `_refresh_guided_confirm_strategy_panel()` reads ROI and chunk inventory from completed-run phasic cache in [gui/main_window.py](gui/main_window.py:3830).
- `GUIDED_CONFIRM_STRATEGIES` exposes Robust Global Event-Reject Fit, Adaptive Event-Gated Fit, Global Linear Regression, and Signal-Only F0 in [gui/main_window.py](gui/main_window.py:148).
- `_on_guided_mark_strategy_choice()` stores choices in `_guided_strategy_choices[(run_dir, roi)]` with evidence chunk and evidence summary in [gui/main_window.py](gui/main_window.py:4119). This is in-memory, source-scoped, and explicit; it is not persisted except via draft plan export.
- Forbidden/non-runnable plan strategies are blocked by `GuidedRunPlan` contract helpers: runnable set and forbidden set live in [photometry_pipeline/guided_run_plan.py](photometry_pipeline/guided_run_plan.py:32), and `validate_correction_strategy()` blocks `auto`, `needs_review`, and `no_correction` in [photometry_pipeline/guided_run_plan.py](photometry_pipeline/guided_run_plan.py:111).

Feature/event detection:

- Guided Draft Plan has a feature/event profile editor in [gui/main_window.py](gui/main_window.py:4478).
- Guided defaults come from `Config()` in `_guided_feature_event_editor_defaults()` in [gui/main_window.py](gui/main_window.py:4155). Full Control defaults use the active baseline config and can be custom YAML; therefore Guided currently may drift from Full Control when a custom/default baseline differs from bare `Config()`.
- `_on_guided_apply_feature_event_profile()` validates and writes an in-memory `FeatureEventProfile` for the current completed run in [gui/main_window.py](gui/main_window.py:4286).
- The editor is run-level planning only and does not extract features or write outputs.

Output destination:

- Guided Draft Plan has an output destination editor in [gui/main_window.py](gui/main_window.py:4591).
- `_on_guided_apply_output_policy()` writes only `_guided_draft_output_policy_by_run[run_dir]` and states that no directories/files were created in [gui/main_window.py](gui/main_window.py:4394).
- `_validate_guided_output_path()` blocks output roots equal to or inside the completed run and selected legacy directories in [gui/main_window.py](gui/main_window.py:4339).
- Guided output policy is completed-run-scoped; there is no new-analysis plan output policy yet.

Plan/export/import:

- `_build_guided_draft_run_plan()` only builds `mode="completed_run_planning"` with `source_mode="completed_run"` from completed-run-scoped stores in [gui/main_window.py](gui/main_window.py:3913).
- Draft Plan export writes one user-selected JSON file only when Export is clicked in [gui/main_window.py](gui/main_window.py:4714).
- Imported plan review and future adoption eligibility are read-only and separate from live draft state in [gui/main_window.py](gui/main_window.py:5215) and [gui/main_window.py](gui/main_window.py:5337).
- Guided Run and Review steps are placeholders. `_build_guided_run_step()` states Run is intentionally unavailable in [gui/main_window.py](gui/main_window.py:4937), and `_build_guided_review_step()` only summarizes completed-run outputs in [gui/main_window.py](gui/main_window.py:4968).

## 4. Contract Comparison Table

| Contract requirement | Current implementation | Status | Evidence in code | Notes / risk |
| --- | --- | --- | --- | --- |
| Guided starts from raw/input data for new analysis | Guided has raw setup widgets synced to Full Control | partial | [gui/main_window.py](gui/main_window.py:1786), [gui/main_window.py](gui/main_window.py:1982) | Setup exists, but Guided plan/diagnostics/execution remain completed-run-centered. |
| Guided supports NPM and custom_tabular | Format combo mirrors `FORMAT_CHOICES`; backend supports both | partial | [gui/run_spec.py](gui/run_spec.py:33), [photometry_pipeline/discovery.py](photometry_pipeline/discovery.py:56) | Guided can select/discover via Full Control state, but no Guided raw-input execution plan. |
| Guided supports intermittent and continuous modes | Mirrors Full Control acquisition controls | partial | [gui/main_window.py](gui/main_window.py:1888) | Continuous NPM unsupported; Guided does not own durable acquisition plan state. |
| Guided can repair acquisition structure | Exposes a subset of Full Control structure fields | partial | [gui/main_window.py](gui/main_window.py:1903) | No Guided channel mapping, signal/control assignment, NPM/custom-tabular column repair, or provenance of user repairs. |
| All ROIs included by default | Discovery checks all ROI items by default | matches | [gui/main_window.py](gui/main_window.py:12079) | Applies to shared Full Control discovery state. |
| Guided can exclude ROIs early | Guided ROI list mirrors checkboxes and syncs back | partial | [gui/main_window.py](gui/main_window.py:2388) | Exclusion affects Full Control RunSpec, not a Guided new-analysis plan. |
| Diagnostic evidence before final production run | Only completed-run/phasic-cache preview sources exist | missing | [photometry_pipeline/preview/correction_preview.py](photometry_pipeline/preview/correction_preview.py:71) | Core workflow gap. |
| Preliminary diagnostic/cache stage | Full Control has Tuning Prep Run profile | partial | [gui/main_window.py](gui/main_window.py:15064), [tools/run_full_pipeline_deliverables.py](tools/run_full_pipeline_deliverables.py:337) | This can create completed-run-like artifacts, but it is Full Control execution, not Guided preliminary cache flow. |
| Evidence chunk/window is provenance, not production scope | Guided stores evidence chunk in `EvidenceChunkReview` | partial | [gui/main_window.py](gui/main_window.py:3913) | Correct for completed-run plan; no raw-input evidence model. |
| Per-ROI correction strategy stored in durable plan state | Stored in memory and exportable as `GuidedRunPlan` for completed runs | partial | [gui/main_window.py](gui/main_window.py:4119), [gui/main_window.py](gui/main_window.py:3913) | No durable new-analysis strategy plan; export is optional file. |
| Global Linear Regression is available | Present in cards, preview methods, confirm strategies, Config | matches | [gui/main_window.py](gui/main_window.py:137), [photometry_pipeline/preview/correction_preview.py](photometry_pipeline/preview/correction_preview.py:55), [photometry_pipeline/config.py](photometry_pipeline/config.py:50) | Exposed as baseline/caution. |
| Signal-Only F0 is explicit, not fallback | Guided text and contract treat it as explicit | matches | [gui/main_window.py](gui/main_window.py:2441), [photometry_pipeline/guided_run_plan.py](photometry_pipeline/guided_run_plan.py:38) | Current Guided does not auto-route it. |
| Forbidden strategies blocked | Contract blocks `auto`, `needs_review`, `no_correction` | matches | [photometry_pipeline/guided_run_plan.py](photometry_pipeline/guided_run_plan.py:111) | Applies to plan validation/import eligibility. |
| Feature/event defaults shared with Full Control | Guided uses bare `Config()` defaults | partial | [gui/main_window.py](gui/main_window.py:4155) | Risk of drift from active custom/default baseline used by Full Control. |
| Feature/event settings editable in Guided | Draft Plan editor exposes parameters and applies profile | partial | [gui/main_window.py](gui/main_window.py:4478), [gui/main_window.py](gui/main_window.py:4286) | Completed-run plan only; no execution handoff. |
| Safe new output destination | Full Control creates unique run dir under output base; Guided output policy blocks completed-run unsafe roots | partial | [gui/main_window.py](gui/main_window.py:11282), [gui/main_window.py](gui/main_window.py:4339) | Full Control output-base/source nesting guard not found. |
| Explicit analysis plan before execution | Full Control runs from `RunSpec`; Guided has non-executable `GuidedRunPlan` | partial | [gui/run_spec.py](gui/run_spec.py:50), [photometry_pipeline/guided_run_plan.py](photometry_pipeline/guided_run_plan.py:1) | No Guided plan-to-execution boundary. |
| Guided Run action | Placeholder says unavailable | missing | [gui/main_window.py](gui/main_window.py:4937) | Correctly not faked, but core workflow missing. |
| Completed-run review as secondary | Guided completed-run branch is robust and growing | partial / risk | [gui/main_window.py](gui/main_window.py:1725), [gui/main_window.py](gui/main_window.py:2602) | Secondary utility currently dominates implemented Guided planning. |
| Saved-plan restore/adoption paused | Current code has read-only review and eligibility only | matches | [gui/main_window.py](gui/main_window.py:5215), [gui/main_window.py](gui/main_window.py:5337) | Should remain paused until core Guided path is re-anchored. |

## 5. Workflow Wiring Table

| UI step/action | User-facing purpose | Reads from | Writes to | Mutates durable plan state? | Depends on raw input, completed-run cache, or both? | Can affect production run? | Current status | Intended final status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Control: Select input/output/format | Configure run source and destination | User widgets | Widget state, settings | N/A, builds `RunSpec` later | raw input | yes | wired | expert direct workflow |
| Full Control: Select ROIs | Discover and include/exclude ROIs | Runner `--discover`, input files | `_discovery_cache`, ROI checklist | `RunSpec.include_roi_ids` at run time | raw input | yes | wired | reusable semantics for Guided |
| Full Control: Validate Only | Validate executable config/input | Full Control widgets, config | run_dir artifacts, subprocess validation outputs | writes `RunSpec` artifacts | raw input | yes | wired | remains Full Control; Guided should get own readiness handoff |
| Full Control: Run Pipeline | Execute production analysis | `RunSpec`, runner | production outputs | executable config, not Guided plan | raw input | yes | wired | remains backend execution path |
| Guided Start: Set up new analysis | Enter raw setup branch | none | `_guided_workflow_mode` | no | raw input | indirectly, via synced Full Control widgets | partial | core Guided path |
| Guided Select data | Mirror Full Control input/output/format | Full Control widgets | Full Control widgets | no | raw input | yes indirectly | partial | should write explicit Guided source/acquisition plan |
| Guided Select ROIs | Discover/include ROIs | Full Control discovery | Full Control ROI checklist | no Guided plan state | raw input | yes indirectly | partial | should record inclusion/exclusion in Guided plan |
| Guided Recording structure | Mirror acquisition controls | Full Control widgets | Full Control widgets | no Guided plan state | raw input | yes indirectly | partial | should record detected/user-repaired structure |
| Guided Correction approach cards | Pick correction intent/setup | static cards, Full Control dynamic fit combo | Full Control dynamic fit or `_guided_correction_intent` | no per-ROI plan | raw setup state | yes indirectly for reference cards | partial | should feed per-ROI planning after evidence |
| Guided Diagnostics: preview comparison | Generate correction evidence | completed-run phasic cache | `_guided_workflow/previews/...` | no | completed-run cache | no production run | wired for completed-run only | raw preview or preliminary cache needed |
| Guided Diagnostics: Signal-Only F0 | Generate signal-only diagnostic evidence | completed-run phasic cache | `_guided_workflow/signal_only_f0_diagnostics/...` | no | completed-run cache | no production run | wired for completed-run only | raw preview or preliminary cache needed |
| Guided Confirm Strategy | Mark per-ROI strategy | completed-run phasic cache, evidence labels | `_guided_strategy_choices` | yes, in-memory completed-run plan | completed-run cache | no | wired for completed-run only | should support new-analysis/rerun plan |
| Guided Draft Plan: feature/event profile | Edit feature/event settings | Guided widgets, `Config()` defaults | `_guided_draft_feature_event_profiles_by_run` | yes, completed-run scoped | completed-run source | no | partial | should use Full Control baseline defaults and new-analysis plan |
| Guided Draft Plan: output destination | Plan future output root | user path, current completed run | `_guided_draft_output_policy_by_run` | yes, completed-run scoped | completed-run source | no | partial | should support safe new-analysis output destination |
| Guided Draft Plan: export | Save draft plan JSON | current completed-run Guided plan | one explicit JSON file | no additional live mutation | completed-run plan | no | wired utility | secondary/provenance utility |
| Guided Imported plan review | Review exported plan JSON | selected JSON | candidate state | no | completed-run candidate | no | wired utility | secondary utility |
| Guided adoption eligibility | Show future restore eligibility | candidate + current completed run | candidate status only | no | completed-run candidate/cache | no | wired utility | pause after this stage |
| Guided Run | Future execution | none | none | no | none | no | placeholder | core future implementation |
| Guided Review | Completed output review | loaded completed run | display only | no | completed-run artifacts | no | placeholder/partial | completed-run review branch |

## 6. Raw-Preview Versus Diagnostic-Cache Finding

Finding: A does not currently exist; B exists only as a Full Control execution profile, not as a Guided preliminary diagnostic/cache stage.

A. Lightweight preview/diagnostics directly from raw input:

- I found no Guided correction-preview path that accepts raw NPM/custom_tabular input directly.
- `photometry_pipeline.preview.correction_preview` restricts source types to `completed_run` and `phasic_cache` in [photometry_pipeline/preview/correction_preview.py](photometry_pipeline/preview/correction_preview.py:71).
- The GUI calls preview with `source_type="completed_run"` in [gui/main_window.py](gui/main_window.py:3655).
- Signal-Only F0 diagnostic review starts from a completed run and resolves completed-run/phasic-out artifacts in [photometry_pipeline/signal_only_f0_diagnostics/generate.py](photometry_pipeline/signal_only_f0_diagnostics/generate.py:296).

B. Preliminary diagnostic/cache stage before final production analysis:

- Full Control has a "Tuning Prep Run" profile in [gui/main_window.py](gui/main_window.py:15064).
- The runner defines a tuning-prep artifact contract that guarantees completed-run metadata plus `_analysis/phasic_out/phasic_trace_cache.h5` and `config_used.yaml` while skipping some full outputs in [tools/run_full_pipeline_deliverables.py](tools/run_full_pipeline_deliverables.py:337).
- This could support Guided correction evidence after a clearly labeled preliminary run, but it is currently a Full Control run profile and still executes the runner/writes outputs. It is not wired as a Guided preliminary diagnostic/cache stage.

Current Guided preview/diagnostic tools that assume completed-run caches:

- Correction preview comparison: [gui/main_window.py](gui/main_window.py:3370), [gui/main_window.py](gui/main_window.py:3655), [photometry_pipeline/preview/correction_preview.py](photometry_pipeline/preview/correction_preview.py:513).
- Signal-Only F0 diagnostic review: [gui/main_window.py](gui/main_window.py:3461), [gui/main_window.py](gui/main_window.py:3604), [photometry_pipeline/signal_only_f0_diagnostics/contract.py](photometry_pipeline/signal_only_f0_diagnostics/contract.py:225).
- Confirm Strategy ROI/chunk population: [gui/main_window.py](gui/main_window.py:3830).

## 7. Risks and Contradictions

- Completed-run-centered drift: Most implemented Guided planning, evidence, strategy marking, output policy, import/export, and adoption eligibility require a completed run.
- Raw setup is not durable Guided plan state: Guided raw-input widgets mutate/shared-sync Full Control widgets, but no `GuidedRunPlan(mode="new_analysis")` is built from those inputs.
- Feature/event default drift: Guided Draft Plan uses `Config()` defaults, while Full Control uses the active baseline/custom config. This can produce different defaults for the same effective workflow.
- ROI exclusion is real but not in Guided plan state for new analysis: early ROI exclusion is represented in shared ROI checklist state and eventually `RunSpec`, not a Guided plan.
- Output safety is uneven: Guided completed-run output policy blocks inside completed-run/legacy paths. Full Control creates unique run dirs but I did not find an explicit source-inside-output/output-inside-source guard for the output base.
- Visible widget state risk: Guided raw setup intentionally syncs to Full Control widgets. That makes raw setup potentially executable through Full Control before Guided has an explicit plan boundary.
- Preview artifacts write inside completed-run `_guided_workflow` namespaces. This is separated from production outputs but still requires completed-run context.
- Saved-plan import/adoption branch is safe so far but secondary. Continuing toward restore/adoption before raw-input plan/execution design would deepen product drift.
- Continuous support is partial: continuous RWD/custom_tabular are supported paths; continuous NPM is explicitly unsupported.
- Channel/ROI mapping and signal/control assignment are config-backed but not normal Guided repair controls.

## 8. Recommended Next Implementation Stages

1. Core workflow: Design Guided new-analysis plan state.
   - Define how raw input, acquisition structure, ROI inclusion/exclusion, correction intents, feature/event settings, and output policy become `GuidedRunPlan(mode="new_analysis")` or a successor planning object.

2. Core workflow: Decide raw preview Model A versus preliminary cache Model B.
   - If Model A: design raw-input preview APIs for NPM/custom_tabular without production outputs.
   - If Model B: design a Guided preliminary diagnostic/cache run with clear non-production labeling, output namespace, provenance, and safe cleanup/retention rules.

3. Core workflow: Reconcile Guided feature/event defaults with Full Control.
   - Guided should load defaults from the same active baseline/config path used by Full Control, or explicitly display any divergence.

4. Core workflow: Add Guided acquisition repair provenance.
   - Include current Full Control fields first, then decide whether channel/ROI mapping, signal/control assignment, and NPM/custom-tabular column mapping need Guided controls or a constrained advanced panel.

5. Core workflow: Plan-to-execution handoff design.
   - Decide how Guided plan maps to `RunSpec` or equivalent executable configuration without exposing RunSpec internals to users.

6. Core workflow: Output safety audit/fix design.
   - Confirm Full Control output-base behavior for existing directories and source/output nesting; design shared output safety helpers for Full Control and Guided.

7. Secondary utility: Pause saved-plan adoption.
   - Keep current read-only import/review/eligibility. Do not implement restore/adoption until core Guided workflow is re-anchored.

8. Developer-provenance infrastructure: Add a wiring inventory test/document generator later if useful.
   - This should not replace product tests, but it could guard against accidental addition of adoption/run controls in the wrong stage.

## 9. Open Questions for Jeff

1. Should Guided new-analysis use lightweight raw previews, a preliminary diagnostic/cache run, or both?
2. Is a Tuning Prep Run acceptable as the user-facing preliminary cache concept if renamed and constrained, or should Guided have a separate diagnostic-cache backend?
3. Should Guided expose NPM/custom-tabular channel/time-column mapping as normal workflow controls, or keep them in an advanced/config panel?
4. Should ROI exclusion require a reason/comment in the first Guided implementation?
5. Should Guided output destination require a non-existing folder, or allow an existing empty folder?
6. Should Full Control remain allowed to run with output base inside/near source data, or should shared safety rules block it?
7. What user-facing language should replace "adoption" if saved-plan restore is resumed later?

## Commands Run

- `Get-Content C:\Users\Jeff\Desktop\task67.txt`
- `Get-Content C:\Users\Jeff\Desktop\guided_full_app_workflow_contract_draft.md`
- `git status --short`
- Multiple read-only `rg` and `Get-Content` inspections over `gui/main_window.py`, `gui/run_spec.py`, `tools/run_full_pipeline_deliverables.py`, `photometry_pipeline/guided_run_plan.py`, `photometry_pipeline/feature_event_config.py`, `photometry_pipeline/config.py`, `photometry_pipeline/discovery.py`, `photometry_pipeline/pipeline.py`, `photometry_pipeline/preview/correction_preview.py`, and `photometry_pipeline/signal_only_f0_diagnostics`.

No tests were run because this was audit-only and no runtime code was changed. One malformed `rg` command was rerun correctly; it made no code or file changes.
