# Guided Diagnostic Cache Follow-Up

## 1. Executive Summary

Model B, a Guided-created diagnostic/cache stage, is the realistic near-term architecture. Model A, true raw-input preview, is not currently supported by the code paths I inspected. The existing preview and Signal-Only F0 diagnostic tools consume completed-run or phasic-cache artifacts, especially `_analysis/phasic_out/phasic_trace_cache.h5` and `config_used.yaml`.

Tuning Prep Run is a plausible backend basis for Model B because it already starts from raw input, writes completed-run metadata, writes the phasic trace cache required by current Guided correction tools, and intentionally skips nonessential production deliverables. It is not yet safe as the Guided architecture by itself: it is exposed as a Full Control run profile, writes a completed-run-like folder, and would need a clearly non-production Guided diagnostic namespace plus UI/readiness language that does not imply final production analysis.

Bad Model C should be rejected as the primary Guided workflow. Requiring an existing completed production run before strategy decisions would preserve the current completed-run-centered drift.

## 2. Tuning Prep Run Details

Tuning Prep Run is a runner profile selected by `--run-type tuning_prep`. The GUI exposes it as "Tuning Prep Run" in `MainWindow._build_run_configuration_group()` and describes its contract in `MainWindow._on_run_profile_changed()` in `gui/main_window.py`.

What it computes:

- It runs phasic analysis when mode is `both` or `phasic`.
- It forces traces-only behavior through `effective_traces_only = args.traces_only or args.run_type == "tuning_prep"` in `tools/run_full_pipeline_deliverables.py`.
- It writes a phasic trace cache and config snapshot suitable for downstream tuning and correction retune workflows.
- It can honor ROI include/exclude CLI flags through `RunSpec.build_runner_argv()` and `PhotometryPipeline.run()`.
- It can honor `preview_first_n` because `RunSpec.build_runner_argv()` emits `--preview-first-n` and `PhotometryPipeline.run()` limits discovered sessions to the first N files.

What it skips:

- Tonic analysis/cache generation is skipped when Tuning Prep is selected.
- Feature extraction is skipped through traces-only mode, so feature-dependent summaries are not promised.
- The runner records skipped phases and outputs including tonic cache/output, phasic time-series summaries, and phasic day-plot bundles.

Artifacts it writes or guarantees:

- `status.json` success evidence, root `run_report.json`, and normal completed-run metadata.
- `_analysis/phasic_out/phasic_trace_cache.h5`.
- `_analysis/phasic_out/config_used.yaml`.
- ROI/session entries readable from the phasic cache.
- It does not promise the full production deliverable package.

Current Guided compatibility:

- Correction preview requires a successful completed run or direct `phasic_out` source and checks for `phasic_trace_cache.h5` plus `config_used.yaml` in `photometry_pipeline/preview/correction_preview.py`.
- Signal-Only F0 diagnostic review requires a completed-run source and the same phasic cache/config snapshot in `photometry_pipeline/signal_only_f0_diagnostics/contract.py` and `generate.py`.
- Confirm Strategy reads ROI/chunk inventory from a completed-run phasic cache in `MainWindow._refresh_guided_confirm_strategy_panel()`.

How heavy it is:

- It is lighter than a full production run because it skips tonic cache generation, feature extraction, and several per-ROI summary/day-plot output families.
- It still processes raw input into phasic traces/cache for the selected scope. This is not a small in-memory preview.
- It can be limited by selected ROIs and by `preview_first_n` first-session selection.
- I did not find a current Tuning Prep limit for arbitrary selected chunks/windows or representative time ranges. Making it limitable that way would require a backend scope contract for window/session selection, runner CLI/config fields for that scope, provenance in status/manifest/run_report, and GUI controls/readiness that state the cache is diagnostic-limited rather than production-complete.

Files/functions inspected:

- `tools/run_full_pipeline_deliverables.py`: `RUN_PROFILE_CHOICES`, `TUNING_PREP_ARTIFACT_CONTRACT`, `_skip_plan_for_profile()`, `validate_inputs()`, `_plan_continuous_windows_summary()`, `resolve_run_dir()`, `main()`.
- `gui/main_window.py`: `_build_run_configuration_group()`, `_on_run_profile_changed()`, `_build_run_spec()`, `_build_argv()`, `_validate_gui_inputs()`, Guided preview/diagnostic/confirm refresh methods.
- `gui/run_spec.py`: `RUN_PROFILE_CHOICES`, `RunSpec.generate_derived_config()`, `RunSpec.build_runner_argv()`.
- `photometry_pipeline/pipeline.py`: continuous handling, ROI selection, `preview_first_n`, `traces_only`.
- `photometry_pipeline/preview/correction_preview.py` and `photometry_pipeline/signal_only_f0_diagnostics/*`.

## 3. Model A Versus Model B

| Model | Existing code support | Required new work | User-facing risk | Product-contract fit | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Model A: true raw-input preview | Not found. Current correction preview supports only `completed_run` and `phasic_cache`; Signal-Only F0 public generator accepts completed-run sources only. | New raw-input preview backend/API, raw NPM/custom_tabular loaders, chunk/window selection, config resolution, ROI selection, output namespace, provenance, and tests. | High duplication risk if it reimplements pipeline chunk loading/correction logic separately. | Good if implemented cleanly, but no current path proves feasibility. | Defer until backend design proves it can reuse pipeline semantics without drift. |
| Model B: Guided diagnostic-cache stage | Partially supported by Tuning Prep Run and existing cache-consuming Guided tools. | Guided-owned launch flow from raw setup state, non-production diagnostic-cache namespace, source scoping, output safety, status labels, cache lifecycle, and no final-production wording. | Medium: users may confuse cache generation with final analysis if naming/output location is weak. | Strong fit if cache is created from raw input inside Guided and clearly labeled preliminary. | Recommended next architecture. |
| Bad Model C: completed-run prerequisite | This is close to current implemented Guided behavior. | No new backend work, but it cements product drift. | High: user must run production before choosing strategy. | Poor fit for primary Guided new-analysis workflow. | Reject as primary workflow; keep only as secondary completed-run review/refinement. |

## 4. Continuous Mode Compatibility

Current code support is inherited from the runner and pipeline:

- Intermittent NPM: supported by format choices and pipeline input handling.
- Intermittent custom_tabular: supported by format choices and pipeline input handling.
- Continuous custom_tabular: supported. `_resolve_continuous_format()` accepts `custom_tabular`, `_discover_continuous_sources()` discovers CSV files, and `PhotometryPipeline.discover_files()` plans continuous windows for `custom_tabular`.
- Continuous RWD: supported by the same continuous planning path.
- Continuous NPM: unsupported. The stable message is `Continuous acquisition mode is not yet implemented for NPM/interleaved inputs.`
- Continuous `auto`: conditionally resolves only if sniffing is unambiguous RWD or custom_tabular. If ambiguous, mixed, unknown, or NPM, it fails with `Continuous mode with --format auto is ambiguous for mixed/unknown inputs. Use --format rwd or --format custom_tabular.`

The UI currently does not appear to block continuous NPM in `_validate_gui_inputs()` before launching the runner. The safest block location is shared validation used by Full Control validate/run and future Guided diagnostic-cache readiness. Guided should show the same stable messages before launching a cache generation process.

## 5. Output Safety Finding

Full Control output behavior:

- The user selects an output base directory in `_output_dir`.
- `_build_run_spec()` creates a unique `run_dir = output_base / generated_run_id`.
- GUI RunSpec uses runner `--out <run_dir>`, not `--out-base`, and writes `config_effective.yaml`, `gui_run_spec.json`, and `command_invoked.txt` into that run directory before validate/run.
- The output base may already exist; if it does not exist, `os.makedirs(run_dir, exist_ok=True)` creates the needed path.
- `_validate_gui_inputs()` requires only a nonempty output path; I did not find a guard blocking output base inside input/source or input/source inside output base.
- `_on_run()` passes `overwrite=True` to `_build_argv()`. Because GUI run IDs are unique, normal runs do not overwrite previous output folders. If a run-id collision or explicit existing run directory occurred, runner legacy `--out` behavior can clean in place when `--overwrite` is present.

Recommendation:

Implement output safety as a shared helper before Guided diagnostic-cache work. It should be used by Full Control, Guided diagnostic-cache generation, and Guided final output policy. The helper should explicitly evaluate source/output nesting both directions, existing target behavior, protected completed-run and legacy output directories, and whether the operation is a production run, diagnostic cache, preview, or read-only planning action.

## 6. Feature/Event Defaults Finding

Full Control has an active baseline source:

- `_active_config_source_path()` chooses either the custom YAML path or the lab default config.
- `_active_baseline_config()` loads that path with `Config.from_yaml()` and falls back to `_default_cfg` only if loading fails.
- `_event_feature_defaults_from_active_baseline()` derives event/feature defaults from that active baseline.
- `_validate_gui_inputs()` validates Full Control event/feature widgets against those active defaults.
- `_build_run_spec()` computes config overrides relative to the active baseline and writes `config_effective.yaml`.

Guided currently differs:

- `_guided_feature_event_editor_defaults()` constructs bare `Config()` and uses those values.
- That means Guided Draft Plan defaults can drift from the Full Control effective baseline when the lab default YAML or a user-selected custom YAML differs from bare dataclass defaults.

Recommendation:

Guided should load feature/event defaults from the same active baseline source as Full Control. The safest first step is to reuse `_event_feature_defaults_from_active_baseline()` for Guided editor initialization and reset/sync behavior, then record the config source path or baseline identity in Guided plan provenance so exported plans explain which defaults were used. Do not silently reinterpret existing Guided profiles; apply the shared defaults only when initializing or explicitly resetting the editor.

## 7. Recommended Next Stage

Next stage: design Guided diagnostic-cache generation from raw setup state using Tuning Prep Run semantics, classified as core workflow.

Scope for that stage should remain design-first unless explicitly approved for implementation. It should define:

- A Guided-owned "Build diagnostic cache" action from raw input/setup state.
- A non-production output namespace and labels that cannot be confused with final analysis.
- Shared output safety policy.
- Shared continuous compatibility validation.
- How selected ROIs and optional first-N/session limits are represented.
- The exact artifact contract consumed by correction preview, Signal-Only F0 diagnostic, and Confirm Strategy.
- Provenance that states the cache is preliminary and not the final production analysis.

Secondary utility work, including saved-plan restore/adoption, should remain deferred until this core workflow path is anchored.

## Commands Run

- `Get-Content -Raw C:\Users\Jeff\Desktop\task68.txt`
- `git status --short`
- `Get-Content -Raw guided_full_contract_wiring_audit.md`
- `Get-Content -Raw C:\Users\Jeff\Desktop\guided_full_app_workflow_contract_draft.md`
- Multiple read-only `rg` inspections over `tools`, `gui`, `photometry_pipeline`, and `tests`
- Multiple read-only line-range inspections of `tools/run_full_pipeline_deliverables.py`, `gui/main_window.py`, `gui/run_spec.py`, `photometry_pipeline/pipeline.py`, `photometry_pipeline/preview/correction_preview.py`, and `photometry_pipeline/signal_only_f0_diagnostics`

No runtime code was changed and no tests were run because this was documentation-only.
