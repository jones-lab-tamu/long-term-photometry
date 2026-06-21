# 4J9f Guided Execution Mapping Gap Audit

## Scope

This is a read-only design audit of the gap between the current pure `GuidedNewAnalysisExecutionSpecPreview` and any future non-writing executable mapping preview or real execution handoff.

No code behavior changes are proposed or implemented here. The current accepted best-case RWD intermittent state is:

- `spec_preview_available=True`
- `first_subset_executable=True`
- `execution_available=False`
- `backend_mapping_status="preview_only_not_mapped_to_RunSpec"`

The current preview is still non-writing, non-executing, and not mapped to `gui.run_spec.RunSpec`.

## Dynamic-Fit Strategy Mapping

Current Guided preview field:

- `correction.selected_global_dynamic_fit_strategy`

Observed backend/config representation:

- `photometry_pipeline.config.Config.dynamic_fit_mode` accepts `global_linear_regression`, `robust_global_event_reject`, and `adaptive_event_gated_regression`.
- Full Control uses `_dynamic_fit_mode_combo` and related widgets, then maps selected values into `config_overrides`.
- Pipeline code reads resolved dynamic-fit mode from chunk metadata such as `dynamic_fit_mode_resolved`, and records dynamic-fit QC/slope metadata keyed by mode.

Strategy-label mapping:

| Guided strategy | Backend/config key | Mapping status | Blocker? |
|---|---|---:|---:|
| `global_linear_regression` | `dynamic_fit_mode=global_linear_regression` | exact key/value mapping for strategy label | no, for label only |
| `robust_global_event_reject` | `dynamic_fit_mode=robust_global_event_reject` | exact key/value mapping for strategy label | no, for label only |
| `adaptive_event_gated_regression` | `dynamic_fit_mode=adaptive_event_gated_regression` | exact key/value mapping for strategy label | no, for label only |

Parameter gaps:

| Parameter family | Backend/config keys | Current Guided plan state | Mapping status | Blocker? |
|---|---|---:|---:|---:|
| Slope constraint | `dynamic_fit_slope_constraint`, `dynamic_fit_min_slope` | not stored in `GuidedNewAnalysisDraftPlan` | needs fixed-default contract or model fields | yes for executable mapping |
| Robust event rejection | `robust_event_reject_max_iters`, `robust_event_reject_residual_z_thresh`, `robust_event_reject_local_var_window_sec`, `robust_event_reject_local_var_ratio_thresh`, `robust_event_reject_min_keep_fraction` | not stored in `GuidedNewAnalysisDraftPlan` | needs fixed-default contract or model fields | yes for robust mapping |
| Adaptive event gate | `adaptive_event_gate_residual_z_thresh`, `adaptive_event_gate_local_var_window_sec`, `adaptive_event_gate_local_var_ratio_thresh`, `adaptive_event_gate_smooth_window_sec`, `adaptive_event_gate_min_trust_fraction`, `adaptive_event_gate_freeze_interp_method` | not stored in `GuidedNewAnalysisDraftPlan` | needs fixed-default contract or model fields | yes for adaptive mapping |
| Legacy rolling/global regression settings | `window_sec`, `step_sec`, `r_low`, `r_high`, `g_min`, `min_samples_per_window`, `min_valid_windows` | not stored in `GuidedNewAnalysisDraftPlan` | likely backend-default only for first subset, but not proven | potential blocker |

Audit conclusion:

- The selected strategy label is ready for a conceptual mapping to `dynamic_fit_mode`.
- A future executable mapping preview should not map only the label and silently rely on transient Full Control widget values.
- Using backend defaults may be acceptable only if the first-subset contract explicitly records each default value and provenance. Otherwise Guided needs model fields for dynamic-fit parameter snapshots.
- Highest-risk gap: dynamic-fit parameter provenance. It is scientific analysis behavior, not just packaging.

Classification: **missing model fields or explicit fixed-default contract; should block executable mapping until resolved.**

## Feature/Event Values Mapping

Current Guided preview fields:

- `feature_event.values`
- `feature_event.consumption`

Observed backend/config representation:

- `photometry_pipeline.feature_event_config.FEATURE_EVENT_CONFIG_FIELDS` lists allowed feature/event config fields.
- The allowed fields are backend `Config` keys: `event_signal`, `signal_excursion_polarity`, `peak_threshold_method`, `peak_threshold_k`, `peak_threshold_percentile`, `peak_threshold_abs`, `peak_min_distance_sec`, `peak_min_prominence_k`, `peak_min_width_sec`, `peak_pre_filter`, and `event_auc_baseline`.
- Guided applies feature/event profiles by parsing widgets or defaults, validating with `validate_feature_event_config_fields`, and storing the resulting dictionary in `GuidedNewAnalysisDraftPlan.feature_event_values`.

Audit findings:

- `feature_event_values` is already in backend config-key form for the field set it stores.
- It is intentionally narrower than arbitrary `Config`; unknown fields are rejected.
- The profile can include only fields relevant to the selected threshold method. Inactive threshold values may fall back to backend/config defaults unless the applied profile stores them.
- The current 4J9d consumption contract says phasic full with `traces_only=False` consumes feature/event values for phasic feature extraction and feature-dependent summaries.

Remaining gap:

- A future mapping preview needs a deterministic merge policy: applied `feature_event_values` over a documented base config, with any fallback defaults displayed as provenance.
- If the future executable preview cannot show all effective feature/event values, it should block rather than silently rely on hidden defaults.

Classification: **needs translation/merge layer, but the applied values themselves are in backend config-key form.**

## Dataset Contract Mapping

Current Guided preview fields:

- `dataset_contract.contract_values`
- `dataset_contract.format_specific`
- `source_acquisition.exclude_incomplete_final_rwd_chunk`

Observed backend/config representation:

- RWD backend/config keys include `rwd_time_col`, `uv_suffix`, `sig_suffix`, `exclude_incomplete_final_rwd_chunk`, `rwd_excluded_source_files`, and `rwd_contract_validation`.
- Full Control `_infer_dataset_contract_overrides(fmt)` can produce backend config-key overrides including `rwd_time_col`, `uv_suffix`, `sig_suffix`, `exclude_incomplete_final_rwd_chunk`, `rwd_contract_validation`, and sometimes `rwd_excluded_source_files`.
- Current Guided dataset contract snapshots are structural/reviewed plan state. Some candidate snapshots include `input_format`, `resolved_input_format`, acquisition fields, and structural flags, but they do not necessarily contain the full backend RWD override set.

Audit findings:

- The best-case tests may use `rwd_time_col`, `sig_suffix`, and `uv_suffix` in snapshot `contract_values`, but the model does not yet enforce a normalized RWD contract shape for executable mapping.
- `exclude_incomplete_final_rwd_chunk` is represented both in Guided acquisition state and can also be a backend config key. A future mapping must define one canonical source and prove consistency.
- `rwd_contract_validation` and `rwd_excluded_source_files` are backend-oriented provenance/repair outputs and are not guaranteed by current Guided snapshots.

Remaining gap:

- A normalization layer is needed before mapping dataset snapshots to config overrides.
- That layer should classify required RWD keys, copy only backend config-key fields, and record rejected/non-config structural fields separately.

Classification: **needs dataset contract normalization layer before executable mapping.**

## Output Run-Directory Ownership

Current Guided preview fields:

- `output.output_base`
- `output.future_run_directory_strategy="derive_unique_run_id_under_output_base"`
- `overwrite=False`
- `precreate_during_preview=False`
- no config write, no command write, no validation run, no execution

Observed backend/RunSpec behavior:

- `RunSpec` currently uses `--out <run_dir>` so the GUI controls the exact run directory.
- `RunSpec.generate_derived_config`, `write_gui_run_spec`, and `write_command_invoked` write `config_effective.yaml`, `gui_run_spec.json`, and `command_invoked.txt` into `run_dir`.
- Runner supports `--out-base <base>` and can generate `run_dir=<base>/<run_id>`.
- Runner validate-only with `--out-base` may create/reserve a run directory for status/events paths.
- Runner legacy validate-only with `--out` has different side-effect behavior but still has status/event boundary complexity.

Audit options:

| Option | Fit to current Guided policy | Risk |
|---|---:|---|
| Guided GUI derives `run_dir` and eventually passes `--out` | matches current Full Control RunSpec behavior | requires GUI to own run ID and likely write preflight files before runner |
| Runner owns run ID via `--out-base` | best matches Guided `output_base` policy | validate-only can create run dirs; must define reservation semantics |
| Separate validation output root from execution output root | preserves no-production-write boundary | more design work; must prevent confusing validation artifacts with final runs |

Recommendation:

- For Guided, prefer runner-owned `--out-base` in the future because it aligns with `output_policy_path` as an output base and avoids GUI pre-creating a final run directory.
- Before implementation, design whether validation reserves the final run dir or uses a separate validation/preflight namespace.
- Do not map to `--out` until the product explicitly accepts GUI-owned run directory creation and preflight artifact writes.

Classification: **needs output ownership design before executable handoff.**

## RunSpec Compatibility Gap Table

This table compares concepts only. It does not instantiate or recommend instantiating `RunSpec` in the current stage.

| Execution-spec preview field | Eventual RunSpec/backend field | Mapping status | Blocker? |
|---|---|---:|---:|
| `source_acquisition.authoritative_input_source_path` | `RunSpec.input_dir`, runner `--input` | exact mapping ready | no |
| `source_acquisition.input_format` | `RunSpec.format`, runner `--format` | exact mapping ready for concrete `rwd` | no |
| `source_acquisition.sessions_per_hour` | `RunSpec.sessions_per_hour`, runner `--sessions-per-hour` | exact mapping ready | no |
| `source_acquisition.session_duration_sec` | `RunSpec.session_duration_s`, runner `--session-duration-s` | exact mapping ready | no |
| `source_acquisition.acquisition_mode` | `RunSpec.acquisition_mode`, runner `--acquisition-mode` | exact mapping ready | no |
| `continuous_window_sec`, `continuous_step_sec`, `allow_partial_final_window` | matching RunSpec/runner fields | exact mapping ready when relevant | no for intermittent |
| `dataset_contract.contract_values` | `RunSpec.data_contract_overrides`, config keys | needs normalization layer | yes |
| `execution_intent.timeline_anchor_mode=civil` | `RunSpec.timeline_anchor_mode`, runner default or flag | fixed default ready | no |
| `execution_intent.fixed_daily_anchor_clock=None` | `RunSpec.fixed_daily_anchor_clock` | fixed default ready | no |
| `execution_intent.execution_mode=phasic` | `RunSpec.mode`, runner `--mode phasic` | fixed default ready | no |
| `execution_intent.run_profile=full` | `RunSpec.run_profile`, runner default or `--run-type full` | fixed default ready | no |
| `execution_intent.traces_only=False` | `RunSpec.traces_only`, absence of `--traces-only` | fixed default ready | no |
| `roi.included_roi_ids` | `RunSpec.include_roi_ids`, runner `--include-rois` | exact mapping ready | no |
| `correction.selected_global_dynamic_fit_strategy` | config `dynamic_fit_mode` | exact label mapping; parameter contract missing | yes |
| `correction.per_roi_choices` | no first-subset per-ROI backend mapping | provenance only | no if unanimous |
| `feature_event.values` | `RunSpec.config_overrides`, config keys | needs merge/translation provenance | yes until effective values are explicit |
| `output.output_base` | runner `--out-base` or GUI-derived `RunSpec.run_dir`/`--out` | needs output ownership design | yes |
| `output.future_run_dir` | runner/GUI run ID result | intentionally unresolved | yes for real handoff |
| `diagnostic_cache_provenance.*` | no production input | provenance only | no |
| `backend_mapping_status` | none | intentionally preview-only | no |
| `provenance.no_*` flags | audit/provenance only | not applicable | no |

## Safe Preview Boundary

A future non-writing executable mapping preview may display:

- conceptual argv entries as data, if clearly marked not generated and not passed to subprocess;
- future config sections and key/value maps, if represented as in-memory preview data and not serialized as YAML/JSON;
- future output paths relative to an unresolved or reserved run-dir concept, if no directory existence is assumed and no path is created;
- exact mapping statuses and blockers by section.

The line is crossed when the code:

- instantiates `RunSpec`;
- calls `RunSpec.generate_derived_config`, `write_gui_run_spec`, `write_command_invoked`, or `build_runner_argv`;
- writes YAML/JSON/text artifacts;
- creates or reserves a run directory;
- runs validation/discovery/execution;
- reads transient GUI widgets or Full Control state to fill missing Guided plan fields.

## Recommended Next Stage

Recommended next implementation stage: **dynamic-fit mapping contract**, but still model-only.

Reasoning:

- Dynamic-fit is the highest-risk scientific mapping gap. The strategy label maps cleanly to `dynamic_fit_mode`, but the parameter values that define robust/adaptive behavior are not stored in Guided plan state.
- Before dataset normalization or output ownership can produce a useful executable mapping preview, the core correction semantics need explicit defaults or durable model fields.
- The safest next patch should add no execution behavior. It should define a model-only dynamic-fit execution-parameter contract/classification: which parameters are fixed defaults, which are unsupported, and which require future Guided controls.

Suggested follow-up order:

1. Model-only dynamic-fit parameter contract and readiness classification.
2. Feature/event effective-value merge preview with explicit fallback provenance.
3. Dataset contract normalization preview for RWD config keys.
4. Output run-directory ownership design.
5. Only then, a non-writing executable mapping preview.

