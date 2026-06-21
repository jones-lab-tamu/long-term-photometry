# 4J9c Pure Guided Execution-Spec Preview Design Audit

## Scope

This is a read-only design audit for the next Guided `new_analysis` boundary after 4J9b. No code behavior changes are proposed as implemented changes here.

Current best-case RWD intermittent Guided state:

- `planning_complete_for_handoff=True`
- `first_subset_executable=True`
- `execution_available=False`

The future object under audit is a pure, non-writing Guided execution-spec preview. It must not instantiate `gui.run_spec.RunSpec`, build argv, write config/YAML/JSON, create directories, run validation, run the pipeline, read transient GUI widget state, or adopt saved plans.

## 1. Inputs To The Future Execution-Spec Preview

For a best-case first-subset RWD plan, the preview should consume only stored `GuidedNewAnalysisDraftPlan` state and documented fixed defaults.

| Plan source | Stored field(s) | Classification | Future preview use |
|---|---|---:|---|
| Source path | `input_source_path`, `resolved_input_source_path` | required | Authoritative raw input path. Prefer `resolved_input_source_path` when present, keep original as provenance. |
| Input format | `input_format` | required | Must be concrete `rwd` for first executable subset. `auto` remains planning-only unless a current applied dataset contract resolves it and a future rule permits that. |
| Dataset resolved format | `dataset_contract_snapshot.resolved_input_format` | required for contract provenance | Confirms reviewed/applied dataset contract identity. For current best-case RWD this should align with `rwd`. |
| Acquisition mode | `acquisition_mode` | required | First subset best case is `intermittent`; other modes remain classified by existing execution-subset helper. |
| Intermittent timing | `sessions_per_hour`, `session_duration_sec` | required for intermittent | Passed as execution intent values later; preview can display them without validating files. |
| Continuous timing | `continuous_window_sec`, `continuous_step_sec`, `allow_partial_final_window` | deferred/not execution input for best-case RWD intermittent | Preserve in preview as plan state; not consumed by first-subset RWD intermittent execution spec. |
| RWD repair | `exclude_incomplete_final_rwd_chunk` | required/fixed plan value | Existing readiness classifies as acquisition repair field; future preview should include it as a dataset/acquisition contract input. |
| ROI inventory | `discovered_roi_ids` | provenance only | Useful for audit, not the execution filter itself. |
| ROI execution filter | `included_roi_ids`, `excluded_roi_ids` | required | First subset should use included ROI IDs as the future execution filter. Excluded IDs are provenance/UI audit. |
| Diagnostic cache identity | `cache_id`, `cache_root_path`, `artifact_record_path`, `request_json_path`, `provenance_path`, `phasic_trace_cache_path`, `config_used_path`, `source_setup_signature`, `diagnostic_scope_signature`, `build_request_signature`, `stale_or_current`, `stale_reasons` | provenance/validity only | Used to prove strategy choices and dataset contract are current. The future production execution should not consume diagnostic cache artifacts as production inputs. |
| Dataset contract snapshot | `dataset_contract_snapshot` including `contract_values`, `format_specific`, `source_identity`, `current_applied`, validation/stale status | required | Current applied snapshot is required for first-subset RWD contract satisfaction. Preview should show values that would later become dataset/config contract inputs, without generating config. |
| Execution intent | `execution_intent.timeline_anchor_mode`, `fixed_daily_anchor_clock`, `execution_mode`, `run_profile` | fixed default ready | 4J9b consumes `civil`, no fixed clock, `phasic`, `full`. Preview must show these are fixed first-subset model values, not GUI widget reads. |
| Correction choices | `per_roi_correction_strategy_choices` | required | Exactly one explicit current strategy choice per included ROI; all included ROIs must share one first-subset dynamic-fit-family strategy. |
| Correction evidence references | `correction_preview_result_id`, `correction_preview_path`, `correction_preview_status`, `correction_preview_source_cache_id`, `signal_only_f0_*`, `selected_evidence_context` | provenance/not execution input | Keep visible. Do not treat preview or Signal-Only F0 diagnostic artifacts as production execution inputs. |
| Feature/event profile | `feature_event_profile_status`, `feature_event_profile_id`, `feature_event_values`, validation/stale/apply provenance | design risk | Currently required for plan completeness. Whether it is consumed by a phasic-only first execution spec needs resolution before implementation. |
| Output destination policy | `output_policy_status`, `output_policy_path`, validation/stale/apply provenance, `output_policy_safety_summary` | required | Supplies future output base/root only after explicit applied/current state. Must not create the directory. |
| Output creation policy | `output_creation_policy.path_role`, `creation_timing`, `run_directory_strategy`, `overwrite`, `precreate_during_preview`, `config_write_timing`, `gui_preflight_writes_enabled` | fixed/default ready | Must show output path is an output base, future unique run dir is derived later, overwrite false, no preview creation or preflight writes. |
| Plan metadata | `schema_version`, `created_at_utc`, `updated_at_utc`, `production_analysis`, `preliminary_cache` | provenance only | Include in preview provenance. |
| Non-executable flags | `execution_ready`, `executable`, `production_run_enabled` | not execution input | They should remain false and should not drive the future spec. |

## 2. Transient GUI Or Full Control State To Avoid

The future execution-spec preview must not read the following from widgets or Full Control state.

| Tempting source | Current location | Guided representation | Audit decision |
|---|---|---|---|
| Full Control output base/run ID/run dir | `_output_dir`, `_build_run_spec`, `_current_run_dir` | `output_policy_path` plus `output_creation_policy` | Already represented enough for preview as future output base. Actual run ID ownership remains unresolved. |
| Full Control `RunSpec` object | `gui.run_spec.RunSpec` | none by design | Do not instantiate. Future preview should be its own Guided object. |
| CLI argv | `RunSpec.build_runner_argv()` / `MainWindow._build_argv()` | none | Must not generate argv in this stage. |
| Effective config YAML | `RunSpec.generate_derived_config()` | dataset/feature/event values and selected strategy are stored separately | Do not synthesize config yet. Preview may show future config inputs by section only. |
| Dynamic fit widget settings | `_dynamic_fit_mode_combo`, robust/adaptive knobs, slope controls | only selected strategy label is stored in `GuidedPlanCorrectionChoice`; detailed parameter values are not fully stored as execution contract | Needs model field if first executable preview must carry actual dynamic-fit config overrides. Do not read widgets. |
| Bleach correction and preprocessing widgets | `_bleach_correction_mode_combo`, lowpass/baseline controls | not in `GuidedNewAnalysisDraftPlan` except indirectly through dataset/feature/profile concepts | Not needed for first subset only if documented backend defaults are accepted. Otherwise needs explicit model state before executable spec. |
| Event/feature widgets | event signal, peak method, threshold fields, polarity, prefilter, AUC baseline | represented in `feature_event_values` after explicit apply | Already represented when applied; do not read widgets. |
| Tonic output/timeline widgets | `_tonic_output_mode_combo`, `_tonic_timeline_mode_combo` | not represented in Guided plan | Not needed if first subset stays phasic-only. Must remain Full Control-only or get future Guided model fields before `both`/tonic execution. |
| Plot/render widgets | plotting mode and hidden render combos | classified as backend default behavior in execution fields | Not needed for first subset; keep fixed/default or Full Control-only. |
| Representative session widget | `_rep_session_combo` | not represented | Intentionally deferred; not needed for first subset. |
| Preview-first-N widgets | `_preview_enabled_cb`, `_preview_n_spin` | not represented | Intentionally fixed default `None`; not needed for production preview. |
| Custom config source and arbitrary config overrides | `_active_config_source_path()`, `config_overrides` from Full Control | feature/event values and dataset snapshot are stored; broad config source is not a Guided execution contract | Must not be pulled in. Add model fields only for explicitly supported Guided controls. |
| Dataset inference helper | `_infer_dataset_contract_overrides()` | reviewed/applied `dataset_contract_snapshot` | Future preview must consume the snapshot, not re-infer from widgets. |

## 3. RunSpec Comparison Without Using RunSpec

This table compares concepts only. It is not a recommendation to instantiate `RunSpec`.

| Guided plan field | Future execution-spec preview field | Eventual RunSpec/backend concept | Status |
|---|---|---|---|
| `resolved_input_source_path or input_source_path` | `source.input_dir` | `RunSpec.input_dir`, runner `--input` | ready for pure preview |
| `output_policy_path` + `output_creation_policy.path_role=output_base` | `output.output_base` | runner `--out-base` or GUI-derived `--out` parent | needs backend design |
| `output_creation_policy.run_directory_strategy` | `output.future_run_directory_strategy` | run ID/run dir creation | needs backend design |
| `input_format` / snapshot resolved format | `source.format` | `RunSpec.format`, runner `--format` | ready for pure preview for concrete `rwd` |
| `sessions_per_hour` | `acquisition.sessions_per_hour` | `RunSpec.sessions_per_hour`, `--sessions-per-hour` | ready for pure preview |
| `session_duration_sec` | `acquisition.session_duration_s` | `RunSpec.session_duration_s`, `--session-duration-s` | ready for pure preview |
| `acquisition_mode` | `acquisition.mode` | `RunSpec.acquisition_mode`, `--acquisition-mode` | ready for pure preview |
| `continuous_window_sec`, `continuous_step_sec`, `allow_partial_final_window` | `acquisition.continuous_*` | corresponding RunSpec/runner flags | ready for pure preview but not first-subset RWD intermittent input |
| `execution_intent.timeline_anchor_mode` | `execution.timeline_anchor_mode` | `RunSpec.timeline_anchor_mode`, `--timeline-anchor-mode` | fixed default ready (`civil`) |
| `execution_intent.fixed_daily_anchor_clock` | `execution.fixed_daily_anchor_clock` | `RunSpec.fixed_daily_anchor_clock` | fixed default ready (`None`) |
| fixed render defaults | `render_modes` | sig/iso, dFF, stacked render flags | fixed default ready / intentionally deferred |
| `execution_intent.execution_mode` | `execution.mode` | `RunSpec.mode`, runner `--mode` | fixed default ready (`phasic`) but see feature/event risk |
| `execution_intent.run_profile` | `execution.run_profile` | `RunSpec.run_profile`, runner `--run-type` | fixed default ready (`full`) |
| `output_creation_policy.overwrite` | `output.overwrite` | `RunSpec.overwrite`, runner `--overwrite` | fixed default ready (`False`) |
| `included_roi_ids` | `roi_filter.include_roi_ids` | `RunSpec.include_roi_ids`, `--include-rois` | ready for pure preview |
| `excluded_roi_ids` | `roi_filter.excluded_for_provenance` | `RunSpec.exclude_roi_ids`, `--exclude-rois` | provenance only if include list is authoritative |
| `dataset_contract_snapshot.contract_values` | `dataset_contract.values` | `RunSpec.data_contract_overrides`, config schema keys | ready for pure preview, needs later exact config-key mapping |
| `feature_event_values` | `feature_event.values` | `RunSpec.config_overrides`, event/feature CLI/config | design risk |
| selected shared strategy from choices | `correction.global_dynamic_fit_strategy` | likely config `dynamic_fit_mode` and related knobs | needs model field for detailed strategy parameters |
| per-ROI choices | `correction.per_roi_choice_provenance` | no direct first-subset backend per-ROI mapping | provenance only |
| diagnostic cache identity | `provenance.diagnostic_cache_identity` | no production input; current-choice proof | provenance only |
| `representative_session` classification | `preview_controls.representative_session` | `RunSpec.representative_session_index` | intentionally deferred |
| `preview_first_n` classification | `preview_controls.preview_first_n` | `RunSpec.preview_first_n` | fixed default ready (`None`) |
| validation-only behavior | `execution.validation_action` | `RunSpec.validate_only`, runner `--validate-only` | intentionally deferred; preview can state no validation run |
| `config_source_path` / arbitrary Full Control config | none | `RunSpec.config_source_path`, `config_overrides` | not applicable for first subset unless explicit Guided model fields are added |

## 4. Output And Write Boundary

The pure execution-spec preview may describe future output intent, but it must not perform any output operation.

It may include:

- `output_base`: the applied `output_policy_path`.
- `path_role`: `output_base`.
- `future_run_directory_strategy`: `derive_unique_run_id_under_output_base`.
- `future_run_dir`: unresolved/preview-only, not created.
- `overwrite`: `false`.
- `directory_created`: `false`.
- `config_effective_yaml_written`: `false`.
- `gui_run_spec_json_written`: `false`.
- `command_invoked_txt_written`: `false`.
- `events_ndjson_written`: `false`.
- `validation_run`: `false`.
- `execution_started`: `false`.
- `RunSpec_instantiated`: `false`.
- `argv_generated`: `false`.

Future decisions still needed:

1. Whether GUI or runner owns run ID generation.
   - Current Full Control GUI generates `<out_base>/<run_id>` and passes `--out`.
   - Runner also supports `--out-base` and can generate `<out-base>/<run-id>`.
   - Guided output creation policy currently says derive under output base, but not which process owns the run ID.

2. Whether future Guided execution uses `--out` or `--out-base`.
   - `--out-base` better matches the policy language and runner GUI mode, but validate-only can create run dirs.
   - `--out` matches current Full Control GUI prewrite behavior, but requires GUI to create/write files before runner invocation.

3. Where `config_effective.yaml`, `gui_run_spec.json`, and `command_invoked.txt` eventually live.
   - Current Full Control writes all three into `run_dir` before runner starts.
   - Guided preview should only name their future intended locations relative to unresolved future run dir, if at all.

4. Whether validation and execution share one future run directory.
   - Runner validate-only with `--out-base` may create a run dir for status paths.
   - If Guided adds validation before execution, it must decide whether validation reserves the same run dir or remains a separate non-production preflight.

Highest output-boundary rule: the preview object can describe write intent, but cannot resolve it by creating directories, writing config, or generating argv.

## 5. Correction Strategy Limitation

The first execution subset rule should remain:

- all included ROIs have exactly one explicit current correction choice;
- all included ROI choices share one dynamic-fit-family strategy;
- allowed first-subset strategies are `global_linear_regression`, `robust_global_event_reject`, and `adaptive_event_gated_regression`;
- mixed per-ROI strategies are planning-valid but execution-preview-blocked;
- `signal_only_f0` remains planning/diagnostic only until applied-dF/F production routing is explicitly designed;
- `auto`, `needs_review`, and `no_correction` remain forbidden for execution;
- no silent collapse from per-ROI choices into a global strategy.

Future execution-spec preview should show:

- `selected_global_dynamic_fit_strategy`: set only when all included ROI choices share the same allowed dynamic-fit strategy.
- `global_strategy_collapsed`: `false`; use wording such as "first-subset global strategy derived from unanimous per-ROI choices" rather than "collapsed".
- `per_roi_choices`: preserved as provenance, including ROI ID, selected strategy, diagnostic cache identity, explicit mark, and selected timestamp.
- `mixed_strategy_blocker`: present when choices are mixed.
- `signal_only_f0_blocker`: present when any included ROI uses Signal-Only F0, with the reason that production applied-dF/F routing is not designed.

One implementation detail to avoid: do not build `{roi_id: choice}` without duplicate detection. The current readiness helper already detects duplicate included ROI choices for execution subset.

## 6. Feature/Event And Mode Interaction

This is the highest-risk finding.

4J9a/4J9b fixed `execution_mode="phasic"` for the first subset. The backend runner treats `run_phasic_mode = args.mode in ('both', 'phasic')`, and feature-dependent packaging appears under phasic mode when `features.csv` exists and `tuning_prep` is not active. Phasic packaging writes feature-dependent summaries such as peak-rate and AUC time series when `has_features` is true.

Therefore feature/event settings are likely still relevant for phasic-only production if phasic execution includes feature extraction and downstream feature summaries. The current plan/readiness contract requiring an explicitly applied feature/event profile is defensible for a `phasic, full` first subset.

However, there is a design ambiguity:

- "phasic" excludes tonic outputs, but it does not necessarily mean "traces only".
- Current execution fields fix `traces_only=False`, which implies feature extraction remains in scope.
- If the intended first subset is correction/dF/F-only without feature deliverables, then feature/event settings are over-required and `traces_only` should probably be true or feature/event should move out of plan completeness for that subset.
- If the intended first subset is phasic full production deliverables, then feature/event is correctly required, but the future preview must say the first subset includes phasic feature extraction and feature-dependent summaries.
- Switching the first subset to `both` would preserve current Full Control default behavior and tonic outputs, but it would broaden the first executable subset beyond "global dynamic-fit-only/phasic-output subset" and would require tonic output/timeline model fields currently absent from Guided plan state.

Recommendation: keep `phasic, full, traces_only=False` only if the next design explicitly states that first-subset execution-spec preview includes phasic feature extraction and feature-dependent outputs. Do not switch to `both` yet; it would introduce more missing model fields and blur the first-subset boundary. The next implementation should encode feature/event consumption explicitly in the preview so the requirement is no longer implicit.

## 7. Diagnostic Cache Relationship

The stated expectation is correct with one caveat.

The diagnostic cache should not be a production execution input for final analysis. It is evidence/provenance for planning:

- proves correction choices were made against the current source/setup/diagnostic scope;
- supplies identity fields used to detect stale strategy choices;
- records preview/diagnostic artifact provenance;
- helps ensure per-ROI choices are explicit and current.

Final production execution should consume raw input, applied dataset contract, selected correction strategy/config, ROI filter, feature/event settings, and output policy. It should not reuse diagnostic cache HDF5/data as production input.

Caveat: future execution-spec preview may reference diagnostic cache artifact paths as provenance and stale-check evidence. It should explicitly mark `execution_consumes_cache_artifacts=False`, as the current non-executing preview already does.

## 8. Recommended Next Stage

Recommended option: **Option B: `4J9d: resolve feature/event versus phasic-mode contract first`**.

Reason: the output/write boundary is settled enough for a pure preview, and the RunSpec mapping can remain deferred. The feature/event versus `phasic, full, traces_only=False` contract is the design point most likely to change the meaning of a future "executable" Guided preview.

Suggested 4J9d scope:

- model-only/design-only or narrowly model-only;
- define whether first-subset `phasic, full` includes feature extraction and feature-dependent summaries;
- if yes, add/define explicit execution-spec preview fields showing feature/event values are consumed by phasic full production preview;
- if no, revise the first-subset intent before any execution-spec preview dataclass is added;
- no RunSpec, argv, config/YAML/JSON writing, directory creation, validation, or run process.

After that, `4J9e` can safely add a model-only `GuidedExecutionSpecPreview` dataclass and pure builder.
