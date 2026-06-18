# Guided Workflow long-duration audit

This is a Stage 4D5 audit of the implemented Guided Workflow through Stage
4D4. It is report-only. It does not add Guided features, change backend
analysis behavior, write manifests, create applied-dF/F outputs, run feature
extraction from Guided mode, or change Full Control behavior.

## 1. Executive summary

The application already has substantial long-duration photometry machinery:
raw input discovery, run-spec/config generation, validation/run execution,
feature extraction, event/peak parameter controls, completed-run loading,
read-only output review, correction preview backends, Signal-Only F0 diagnostic
backends, applied-dF/F production tools, and provenance-heavy output writing.
The main gap is not missing backend capability; it is that Guided Workflow is
still a partial, review/planning scaffold and does not yet consume or produce a
reproducible long-duration run plan.

The current Guided implementation mostly respects the long-duration scope
model:

- Open Results mode is completed-run scoped and does not overwrite raw-input
  setup.
- Diagnostics are completed-run scoped and generate preview/diagnostic artifacts
  only under Guided namespaces.
- Correction preview and Signal-Only F0 diagnostic state are treated as
  descriptive evidence, not strategy selection.
- Confirm Strategy is UI-state-only and now stores strategy choices by
  `(completed_run_dir, ROI)`, with evidence chunk stored only as context.
- `auto`, `needs_review`, and `no_correction` are not runnable Guided Confirm
  Strategy choices.
- Run remains skipped in Open Results mode and placeholder-only in new-analysis
  mode.

The main long-duration risks remaining before the next Guided feature are:

- Guided still has no explicit run-plan object. Any future Run implementation
  must not treat the currently visible ROI/chunk as production scope.
- Guided does not expose feature/event detection planning, but feature/event
  detection already exists in backend, Full Control, config schema, standard
  pipeline output, applied-dF/F tools, and tests. It should be reused, not
  reinvented.
- Diagnostics use representative ROI/chunk controls. Current wording mostly
  says preview/evidence, but future work needs stronger tests that the selected
  evidence chunk cannot become production scope.
- Completed-run review in Guided is limited to Diagnostics and pointers; long
  duration output navigation still relies mainly on existing Full Control/Open
  Results viewers.

No acceptance-blocking runtime bugs were found in the current 4D4
implementation. The acceptance-blocking design constraint for future work is to
add a plan layer before any Guided production execution.

## 2. Purpose and scope model

The application is for long-duration fiber photometry analysis: many hours to
many days, many source files/sessions/chunks, and potentially many ROIs. Guided
Workflow should be a safe decision scaffold over that system, not a
chunk-by-chunk approval wizard.

| Concept | Meaning | Legitimate user-decision scope? | Evidence/QC/preview scope? | Current implementation notes |
| --- | --- | --- | --- | --- |
| Raw input folder | Source data for new analysis. | Yes, as source selection for a run. | Yes, via validation/discovery. | Full Control and Guided Start/Select Data share the input directory state. |
| Completed run | Previously generated analysis output used for review, diagnostics, preview, and planning. | Yes, as review/planning source. | Yes. | Open Results uses completed-run validation and populates Diagnostics/Confirm Strategy. |
| ROI | Normal unit for correction-strategy planning. Future actions may target selected ROIs or compatible ROI groups. | Yes. | Yes. | Confirm Strategy stores choices by `(completed_run_dir, ROI)`. |
| Chunk | Internal/source/session/computational unit. | No for routine production strategy decisions. | Yes. | Pipeline and caches are chunk-based internally; Guided uses chunk controls for diagnostics/evidence only. |
| Evidence chunk | Representative chunk used to inspect preview or diagnostic evidence. | No, except as provenance context for a ROI-level choice. | Yes. | Confirm Strategy records `evidence_chunk` with the marked ROI-level choice. |
| Correction strategy | Explicit user-marked ROI-level or selected-ROI-level analysis choice. | Yes. | Can be informed by evidence chunks. | Guided choices are UI-state-only and not written to manifests. |
| Feature/event detection settings | Peak/event detection and summary configuration. | Should be ROI-level, selected-ROI-level, run-level, or profile-level in future Guided mode. | Chunks can be used for preview/QC only. | Existing backend and Full Control support this; Guided does not yet expose it. |
| Run plan | Future explicit reproducible plan across the relevant dataset. | Yes. | May include evidence provenance. | Does not exist yet in Guided mode. |
| Run execution | Future plan-driven execution across the dataset. | Yes, after validation. | No. | Guided Run is placeholder/skipped; Full Control run execution exists. |
| Review | Long-duration output review and refinement. | Yes, at output/ROI/run levels. | Yes. | Full Control/Open Results review exists; Guided Review is placeholder. |

## 3. Current Guided Workflow inventory through Stage 4D4

| Guided element | Current behavior | Wiring/status | Source scope | Writes/calls backend? | Current scope model | Long-duration assessment |
| --- | --- | --- | --- | --- | --- | --- |
| Start | Offers new-analysis setup and Open Results. Shows mode status. | Real UI. | Raw input and completed run. | Open Results calls completed-run loader. | Source selection. | Appropriate. It separates raw input from completed-run review. |
| Open Results mode | Loads a completed run, leaves raw input setup unchanged, navigates to Diagnostics. | Real UI using shared completed-run loading. | Completed run. | Reads completed-run artifacts. | Completed-run scoped. | Appropriate. |
| Select data | Guided new-analysis setup panel synchronized with Full Control raw input state. Hidden/skipped in Open Results. | Real UI, partial scaffold. | Raw input. | No analysis execution. | Source setup. | Appropriate, but not a full Guided discovery/run flow yet. |
| Recording structure | Guided setup panel synchronized with Full Control controls. Hidden/skipped in Open Results. | Real UI, partial scaffold. | Raw input config. | No analysis execution. | Run/setup scope. | Appropriate as setup; future work should reuse discovery/validation. |
| Correction approach | Guided correction-card UI maps to Full Control correction settings. Hidden/skipped in Open Results. | Real UI for setup. | Raw input config. | Updates GUI/config state only. | Run/profile setup, not production strategy routing. | Mostly appropriate. Must remain distinct from applied-dF/F strategy routing. |
| Diagnostics | Shows completed-run artifact status and action cards. | Real UI in Open Results. | Completed run. | Calls preview and Signal-Only F0 diagnostic backends on button click. | Evidence/QC scope. | Appropriate if users understand representative evidence, not exhaustive chunk approval. |
| Correction preview comparison | Selects ROI/evidence chunk and allowed methods; generates preview-only artifacts. | Real UI plus backend. | Completed run. | Calls `run_guided_correction_preview_comparison`. | Evidence chunk. | Appropriate. It must remain preview-only and not strategy routing. |
| Signal-Only F0 diagnostic review | Selects ROI/evidence chunk; generates diagnostic artifacts. | Real UI plus backend. | Completed run. | Calls `run_signal_only_f0_diagnostic_review`. | Evidence chunk. | Appropriate. It must not imply Signal-Only F0 selection. |
| Confirm Strategy | Lets user mark explicit strategy for current ROI with evidence chunk recorded. | Real UI-state-only planning panel. | Completed run. | No backend writes or analysis calls. | ROI-level strategy choice with evidence chunk context. | Appropriate after 4D4 correction. It still lacks persistence/run-plan integration. |
| Run | Placeholder in new-analysis mode; skipped panel in Open Results. | Placeholder/skipped. | None in Guided. | No run execution from Guided. | Future plan execution. | Appropriate until plan model exists. |
| Review | Placeholder. | Placeholder. | Future completed-run outputs. | No review backend. | Future long-duration review. | Needs future design; Full Control has existing review. |
| Current setup summary | Collapsible summary of active setup state. | Real UI. | Raw input/completed run status. | No backend analysis. | Display-only. | Appropriate. |
| Planned stages / not yet wired | Collapsible future-stage notes. | Placeholder. | None. | No backend. | Informational. | Appropriate if kept clear as not wired. |
| Mode banners | Distinguish New analysis from Open Results. | Real UI. | Raw input/completed run. | No backend analysis. | Context display. | Appropriate and important for provenance boundaries. |
| Generated output summaries | Summarize latest preview/diagnostic generated outputs and stale state. | Real UI. | Completed run. | Display-only after generation. | Evidence artifact summary. | Appropriate; should continue using compact paths and stale labels. |

## 4. Implemented-Guided audit findings

### Finding 1

- Finding: Confirm Strategy has been corrected to ROI-level storage with chunk
  retained only as evidence context.
- Evidence from code/tests: `gui/main_window.py` stores
  `self._guided_strategy_choices[(run_dir, roi)]` and includes
  `"evidence_chunk": int(chunk)` in the entry. Tests in
  `tests/test_gui_guided_workflow.py` assert ROI-level keys, evidence chunk
  updates, ROI independence, completed-run scoping, and acknowledgment resets.
- Why it matters for long-duration photometry: Users cannot mark strategy per
  chunk across hundreds of chunks. ROI-level planning matches long-duration
  production strategy scope.
- Severity: no change needed.
- Recommended action: Keep this as a regression-protected design rule. Future
  persistence should preserve ROI-level or selected-ROI-level scope.
- Files/functions involved: `MainWindow._on_guided_mark_strategy_choice`,
  `MainWindow._guided_marked_choice_text`,
  `tests/test_gui_guided_workflow.py`.

### Finding 2

- Finding: Guided still has no durable run-plan object.
- Evidence from code/tests: Confirm Strategy is UI-state-only; tests assert no
  manifest, applied-dF/F output, features, validation, or pipeline run is
  created. Guided Run remains placeholder/skipped.
- Why it matters for long-duration photometry: Future production execution
  needs a reproducible dataset-level or selected-ROI-level plan. Without that
  layer, current GUI selections could accidentally become production scope.
- Severity: acceptance-blocking before next Guided feature.
- Recommended action: Before implementing Guided Run or production routing,
  define an explicit run-plan schema that separates source scope, ROI scope,
  evidence provenance, correction choices, feature settings, and execution
  target.
- Files/functions involved: `gui/main_window.py` Confirm Strategy and Run
  builders; `gui/run_spec.py`; future plan module.

### Finding 3

- Finding: Diagnostics correctly scope preview/diagnostic results to completed
  run, ROI, and evidence chunk, but they remain representative evidence rather
  than exhaustive review.
- Evidence from code/tests: Preview and Signal-Only F0 results store
  `completed_run_dir`, `roi`, and `chunk_index`. Confirm Strategy reports
  previous-run results as not generated for the current completed run and marks
  ROI/chunk changes stale.
- Why it matters for long-duration photometry: Representative evidence is
  useful, but the UI must not imply every chunk must be approved manually.
- Severity: should fix soon.
- Recommended action: Add future wording/tests that explicitly call evidence
  chunks representative QC context. Do not add per-chunk approval state.
- Files/functions involved: `MainWindow._confirm_evidence_status`,
  `MainWindow._guided_confirm_evidence_summary`,
  `tests/test_gui_guided_workflow.py`.

### Finding 4

- Finding: Guided text and tests guard against automatic strategy selection.
- Evidence from code/tests: `GUIDED_CONFIRM_STRATEGIES` contains only
  `robust_global_event_reject`, `adaptive_event_gated_regression`,
  `global_linear_regression`, and `signal_only_f0`. Tests assert `auto`,
  `needs_review`, and `no_correction` are absent, diagnostics do not select a
  strategy, and text does not recommend.
- Why it matters for long-duration photometry: Long-duration strategy routing
  must be explicit and auditable; diagnostic success cannot be treated as a
  decision.
- Severity: no change needed.
- Recommended action: Preserve this guard when adding any decision-support
  audit display.
- Files/functions involved: `GUIDED_CONFIRM_STRATEGIES`,
  `test_guided_confirm_strategy_never_auto_selects_from_loaded_or_generated_evidence`.

### Finding 5

- Finding: Guided preview and Signal-Only F0 diagnostic actions generate files,
  but only in diagnostic/preview namespaces.
- Evidence from code/tests: Correction preview uses
  `photometry_pipeline.preview.correction_preview`; Signal-Only F0 diagnostic
  uses `photometry_pipeline.signal_only_f0_diagnostics`. Contract tests cover
  output safety and no source/legacy mutation. Confirm Strategy tests assert it
  writes no files.
- Why it matters for long-duration photometry: Diagnostic artifacts must be
  revisitable without mutating source caches, production outputs, or legacy
  features.
- Severity: no change needed.
- Recommended action: Future Guided outputs should use the same namespace-safety
  pattern and never write production artifacts from evidence controls.
- Files/functions involved: `photometry_pipeline/preview/correction_preview.py`,
  `photometry_pipeline/signal_only_f0_diagnostics/contract.py`,
  `photometry_pipeline/signal_only_f0_diagnostics/generate.py`,
  related contract/backend tests.

### Finding 6

- Finding: Feature/event detection is absent from Guided mode but present in the
  app.
- Evidence from code/tests: `photometry_pipeline.core.feature_extraction`,
  `Config` peak/event fields, Full Control event controls, `gui/run_spec.py`,
  pipeline `features.csv` writing, applied-dF/F feature tools, and dedicated
  feature/event tests all exist.
- Why it matters for long-duration photometry: Guided should not reinvent peak
  detection or call it missing. It should reuse existing config and backend
  paths when eventually exposing feature/event planning.
- Severity: acceptance-blocking before adding Guided feature/event controls.
- Recommended action: Design Guided feature/event detection as a profile/run/ROI
  scoped wrapper around existing config fields and `extract_features`, with
  chunk previews only for evidence.
- Files/functions involved: `photometry_pipeline/core/feature_extraction.py`,
  `photometry_pipeline/config.py`, `gui/main_window.py`,
  `gui/knobs_registry.py`, `gui/run_spec.py`, `photometry_pipeline/pipeline.py`.

## 5. Existing-functionality inventory and reuse map

| Capability | Exists where | Backend exposure | Full Control GUI exposure | Guided Workflow exposure | Config/schema fields | Output files/artifacts | Tests | Current scope model | Source of truth | Guided reuse recommendation | Risk if reinvented |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Input folder selection | `MainWindow`, `RunSpec`, pipeline CLI | Pipeline `run(input_dir, output_dir, ...)` | Input directory line edit/browse | Start/Select Data share same state | RunSpec fields | Run directory and artifacts | GUI run-spec/usability tests | Raw input folder | Full Control + RunSpec | Reuse shared state and validation | Divergent discovery behavior |
| Completed-run loading | `gui/run_report_parser.py`, `MainWindow._open_completed_results_dir` | Completed-run helper/readers | Open Results | Start/Open Results/Diagnostics/Confirm | N/A | Existing run artifacts | Guided/Open Results tests | Completed run | Completed-run loader | Reuse for all review/diagnostic stages | Cross-run provenance leaks |
| Recording structure controls | `MainWindow`, `RunSpec`, `Config` | Discovery/pipeline arguments | Format, acquisition, timing, ROI discovery controls | Setup panels synchronized/skipped by mode | format, acquisition, sessions, window fields | Config/run report/discovery artifacts | GUI baseline/config/run-spec tests | Run setup | Full Control + RunSpec | Reuse; do not duplicate validation | Mismatched loader assumptions |
| ROI selection | Discovery cache, pipeline include/exclude | `include_rois`, `exclude_rois` | Select ROIs controls | Diagnostics and Confirm Strategy ROI combos | RunSpec include/exclude | Cache/features rows by ROI | GUI run-spec, applied-dF/F GUI tests | ROI or selected ROI set | Discovery cache and RunSpec | Reuse discovery and include/exclude semantics | Current visible ROI could become accidental run scope |
| Correction strategy / dynamic fit modes | `Config.dynamic_fit_mode`, regression backends | `regression.fit_chunk_dynamic` and dynamic fitting modules | Correction controls/cards | Correction approach setup and Confirm Strategy candidates | dynamic fit and correction fields | Phasic cache dF/F/QC | Guided, correction preview, pipeline tests | Config/run or ROI-level planning | Config + regression dispatcher | Use explicit strategy plan; no auto | Silent fallback or per-chunk routing |
| Correction preview comparison | `photometry_pipeline.preview.correction_preview` | Preview backend | Not normal production control | Diagnostics action card | Preview config snapshot | `_guided_workflow/previews/<id>` | Preview contract/backend/guided tests | Evidence chunk | Preview backend | Reuse as representative evidence only | Mistaking preview for production |
| Signal-Only F0 diagnostic review | `photometry_pipeline.signal_only_f0_diagnostics` | Diagnostic backend | Not normal production control | Diagnostics action card | Source config snapshot | `_guided_workflow/signal_only_f0_diagnostics/<id>` | Diagnostic contract/generate/guided tests | Evidence chunk | Diagnostic backend | Reuse as explicit diagnostic evidence | Implicit strategy selection |
| Feature/event detection | `core.feature_extraction.extract_features` | Direct backend and pipeline pass 2 | Full Control event controls and downstream retune controls | Not exposed | `event_signal`, polarity, peak threshold/distance/prominence/width/filter/AUC | `phasic_out/features/features.csv`; applied feature outputs | Feature, event, peak, pipeline, applied feature tests | Chunk-internal, ROI rows, run output | `extract_features` + Config + pipeline | Reuse existing config/backend; chunk preview only | Duplicated detector semantics |
| Feature/event detection parameters | `Config`, `knobs_registry`, parsing helpers | Config dataclass | Event controls, retune controls | Not exposed | peak/event fields | Config effective/run report | GUI run-spec and peak tests | Run/profile currently; could be ROI/profile later | Config schema + GUI parser | Build Guided profiles from existing fields | Incompatible config generation |
| Feature/event detection outputs | Pipeline and applied-dF/F tools | `features.csv/json`, summary/provenance | Open Results viewers and plots | Not directly reviewed in Guided | N/A | `features.csv`, summaries, plots | Many output tests | Run/ROI/chunk rows | Pipeline output contract | Guided Review should summarize/navigate | Chunk-by-chunk manual review burden |
| Run configuration/specification | `gui/run_spec.py` | Derived config and argv | Full Control Validate/Run | Setup summary only | All Config fields + RunSpec fields | `config_effective.yaml`, `gui_run_spec.json`, command files | `test_gui_run_spec.py` | Run | RunSpec | Future Guided Run must generate a plan then RunSpec | Ad hoc execution |
| Validation | GUI build/run spec and runner path | CLI validation paths | Validate Only | Not wired as Guided action | Config/run spec | Validation artifacts | GUI validation tests | Run | Full Control/runner | Reuse exactly | Different pass/fail semantics |
| Run execution | `tools/run_full_pipeline_deliverables.py`, pipeline | Pipeline engine | Full Control Run | Placeholder/skipped | RunSpec/config | Full analysis output | Pipeline/GUI tests | Dataset/run | Runner + pipeline | Future plan-driven execution | Selected chunk execution bug |
| Completed-run review tabs | `gui/run_report_viewer.py`, MainWindow artifact buttons/viewers | Artifact readers | Full Control/Open Results | Diagnostics summary only | N/A | Existing plots/tables/reports | Open Results/viewer tests | Completed run/ROI/output | Existing viewer/parser | Reuse for Guided Review | Rebuilding incomplete reviewer |
| Provenance/output writing | Pipeline, tools, preview/diagnostic/applied tools | Multiple writers | Full Control run artifacts | Preview/diagnostic summaries | Config/provenance fields | run reports, summaries, JSON/CSV | Broad tests | Output contract | Existing writers | Keep namespaces separated | Source/output mutation |
| Applied-dF/F/reference-free workflow | `tools/write_applied_dff_cache.py`, verify/run/batch/audit tools, GUI applied workflow | Explicit applied tools | Applied-dF/F GUI workflow | Not in Guided except future notes | Applied tool arguments/config | Separate applied output dirs | Applied-dF/F tests | ROI/strategy production | Applied-dF/F toolchain | Reuse later after plan design | Silent strategy routing |

## 6. Existing feature/event detection audit

1. Backend modules/functions that perform feature/event detection:
   - `photometry_pipeline.core.feature_extraction.extract_features`
   - `get_peak_indices_for_trace`
   - `compute_detection_threshold_bounds`
   - `compute_auc_above_threshold`
   - `apply_peak_prefilter`
2. Parameters/config fields currently used:
   - `event_signal`
   - `signal_excursion_polarity`
   - `peak_threshold_method`
   - `peak_threshold_k`
   - `peak_threshold_percentile`
   - `peak_threshold_abs`
   - `peak_min_distance_sec`
   - `peak_min_prominence_k`
   - `peak_min_width_sec`
   - `peak_pre_filter`
   - `event_auc_baseline`
   - `traces_only` in RunSpec/CLI controls whether feature extraction is skipped.
3. Full Control GUI controls:
   - Main run event controls around `MainWindow` event feature defaults and
     event/peak widgets.
   - Downstream retune controls for event signal and peak settings.
   - `Skip feature extraction (traces and QC only)` checkbox.
   - GUI knob registry allowlists peak/event config fields.
4. Standard run pipeline integration:
   - `photometry_pipeline.pipeline.Pipeline` calls
     `feature_extraction.extract_features(chunk, self.config)` in pass 2 when
     `traces_only` is false.
5. Output files/tables:
   - Standard phasic output writes `phasic_out/features/features.csv`.
   - Applied-dF/F feature runner writes separate `features.csv`,
     `features.json`, `feature_summary.csv/json`, and provenance.
   - Continuous summaries and plots consume `features.csv`.
6. Tests:
   - `tests/test_feature_extraction_nan_gap_bias.py`
   - `tests/test_event_signal_selection.py`
   - `tests/test_event_polarity_semantics.py`
   - `tests/test_peak_detection_hardening.py`
   - `tests/test_run_applied_dff_features.py`
   - `tests/test_run_applied_dff_feature_preview.py`
   - `tests/test_verify_applied_dff_feature_outputs.py`
   - pipeline and traces-only tests that assert presence/absence of
     `features.csv`.
7. Scope:
   - Internally chunk-level and ROI-row-level.
   - Externally standard run output is run/dataset scoped, with rows by
     chunk/ROI.
   - Full Control settings are run/profile-like, not per-chunk user decisions.
8. Missing from Guided versus missing from the app:
   - Missing from Guided: feature/event planning UI, profile selection, preview
     evidence, and review summaries.
   - Not missing from the app: detector backend, config fields, Full Control
     controls, run output, applied-dF/F feature tooling, and tests.
9. Reusable for Guided:
   - Existing Config fields, parser helpers, knob registry, RunSpec generation,
     `extract_features`, applied feature verification, and output readers.
10. Requires refactor before Guided exposure:
    - A Guided plan/profile representation for run-level/ROI-level feature
      settings.
    - Optional chunk preview API that calls the same detector without writing
      production outputs.
11. Should not be reinvented:
    - Peak detection thresholds, polarity handling, prefiltering, AUC semantics,
      feature output columns, and run-spec config emission.

### Feature/event detection finding

- Finding: Existing feature detection is mature enough to audit and reuse but
  not yet suitable for direct Guided exposure without a plan/profile layer.
- Evidence from code/tests: `extract_features` is the authoritative detector;
  pipeline writes `features.csv`; Full Control exposes peak/event parameters;
  tests cover event signal selection, polarity, peak hardening, NaN gaps,
  traces-only, applied feature outputs, and run-spec config emission.
- Why it matters for long-duration photometry: Feature/event settings are
  analysis decisions that can affect every chunk/ROI. Guided must avoid
  per-chunk tuning and must keep output provenance coherent.
- Severity: acceptance-blocking before adding Guided feature/event controls.
- Recommended action: Add a design step for feature/event profiles and
  representative chunk previews using existing backend functions.
- Files/functions involved: `photometry_pipeline/core/feature_extraction.py`,
  `photometry_pipeline/config.py`, `photometry_pipeline/pipeline.py`,
  `gui/main_window.py`, `gui/run_spec.py`, `gui/knobs_registry.py`.

## 7. Correction and diagnostic audit

1. Current runnable correction strategies exposed by Guided Confirm Strategy:
   - `robust_global_event_reject`
   - `adaptive_event_gated_regression`
   - `global_linear_regression`
   - `signal_only_f0`
2. `auto`, `needs_review`, and `no_correction` are absent from runnable Guided
   strategy choices.
3. Signal-Only F0 is explicit. Diagnostic success does not select Signal-Only
   F0.
4. Correction preview methods exclude Signal-Only F0 and forbidden strategies;
   Signal-Only F0 has a separate diagnostic review action.
5. Current correction preview/diagnostic wording is mostly descriptive. Tests
   assert no recommendation language in the Confirm Strategy evidence summary.
6. Diagnostic actions that generate artifacts:
   - Correction preview comparison writes preview artifacts under Guided preview
     namespace.
   - Signal-Only F0 diagnostic review writes diagnostic artifacts under
     completed-run Guided diagnostic namespace.
7. Diagnostics are completed-run scoped and stale/previous-run guarded.
8. Diagnostics do not mutate source traces or production outputs per contract
   tests.
9. Diagnostic summaries scale as representative evidence summaries. They are not
   yet long-duration aggregate dashboards across all ROIs/chunks.
10. Wording should continue emphasizing representative evidence, not exhaustive
    chunk review.

### Correction/diagnostic finding

- Finding: Correction diagnostics are safe evidence generators but are not a
  long-duration decision engine.
- Evidence from code/tests: Preview and diagnostic backends are output-safe;
  Confirm Strategy does not auto-select; result state is completed-run scoped.
- Why it matters for long-duration photometry: Users need scalable evidence,
  not a manual chunk checklist.
- Severity: should fix soon.
- Recommended action: Add future aggregate diagnostic summaries by ROI and run,
  while preserving representative evidence chunk previews.
- Files/functions involved: `photometry_pipeline/preview/correction_preview.py`,
  `photometry_pipeline/signal_only_f0_diagnostics/generate.py`,
  `gui/main_window.py`, Guided tests.

## 8. Run/Review future-scope audit

1. Current Guided Run behavior:
   - Placeholder in new-analysis mode.
   - Skipped/unavailable in Open Results mode, with navigation back to
     Diagnostics or new-analysis setup.
2. Run is skipped/unavailable in Open Results mode.
3. Current Guided Run text does not execute the currently selected ROI/chunk.
   The larger future risk is architectural: if Run is implemented without a
   plan layer, selected evidence ROI/chunk could be misused as production
   target.
4. Existing run/config machinery to reuse:
   - `gui.run_spec.RunSpec`
   - `RunSpec.generate_derived_config`
   - `RunSpec.build_runner_argv`
   - GUI validation and runner controls
   - `tools/run_full_pipeline_deliverables.py`
   - `photometry_pipeline.pipeline.Pipeline`
5. Current completed-run review:
   - Full Control/Open Results can load completed runs and review artifacts.
   - Guided Diagnostics displays status and can generate preview/diagnostic
     evidence.
   - Guided Review is placeholder only.
6. Missing tests before Guided production:
   - Current selected evidence chunk is never used as Run scope.
   - ROI-level choices are converted to explicit plan entries only by a
     deliberate plan action.
   - Feature/event settings are run/profile/ROI scoped, never evidence-chunk
     scoped.
   - Open Results cannot write production outputs.

### Run/Review finding

- Finding: Guided production execution should remain blocked until a run-plan
  contract exists.
- Evidence from code/tests: Run is placeholder/skipped; Confirm Strategy is
  UI-state-only; no manifest/output writing occurs.
- Why it matters for long-duration photometry: Long-duration execution must be
  reproducible and dataset-wide or selected-ROI-wide, not tied to current GUI
  focus.
- Severity: acceptance-blocking before next Guided production feature.
- Recommended action: Define plan schema and tests before wiring Run.
- Files/functions involved: `gui/main_window.py`, `gui/run_spec.py`,
  `tools/run_full_pipeline_deliverables.py`, `photometry_pipeline/pipeline.py`.

## 9. Test audit

Tests that protect the correct long-duration model:

- Guided tests assert Open Results does not overload raw input setup.
- Guided tests assert page-level layouts are width-resizable and details are
  collapsible.
- Guided tests assert diagnostics are completed-run scoped and stale across
  ROI/chunk/run changes.
- Guided tests assert Confirm Strategy choices are ROI-level by completed run,
  not chunk-keyed.
- Guided tests assert acknowledgment resets on completed-run/ROI/evidence-chunk
  changes.
- Guided tests assert diagnostics do not auto-select strategies.
- Guided tests assert Confirm Strategy writes no manifests, applied-dF/F
  outputs, features, validation, or pipeline outputs.
- Preview/diagnostic contract tests assert namespace safety and source/legacy
  non-mutation.

Tests that may encode a short-session or per-chunk model:

- Preview and diagnostic tests necessarily use small fixtures with two chunks.
  That is acceptable for unit coverage, but not sufficient evidence of
  long-duration scalability.
- Tests focus on current ROI/chunk UI behavior; they should keep labeling chunk
  as evidence context.

Tests that pass while failing to protect the long-duration scope model:

- Existing Guided tests do not yet exercise conversion of multiple ROI-level
  marked choices into a future plan because no plan exists.
- Existing tests do not verify feature/event detection remains profile/run/ROI
  scoped in Guided because Guided has no feature/event controls.
- Existing tests do not verify long-duration review navigation across hundreds
  of chunks/ROIs.

Missing tests before future Guided production work:

- Plan serialization round-trip with multiple ROIs and evidence chunks.
- Run step consumes plan entries, not visible combo-box selection.
- Feature/event plan settings are scoped by profile/run/ROI and use existing
  Config fields.
- Evidence chunk previews cannot create per-chunk production decisions.
- Open Results cannot mutate completed-run production outputs.
- Review summarizes and navigates outputs without requiring manual chunk
  discovery.

Existing feature/event detection tests and coverage:

- `test_feature_extraction_nan_gap_bias.py`: NaN gap behavior in feature
  extraction.
- `test_event_signal_selection.py`: `event_signal` config, traces-only
  semantics, and output differences.
- `test_event_polarity_semantics.py`: signed polarity semantics and threshold
  bounds.
- `test_peak_detection_hardening.py`: detector helper parity, prominence,
  width, defaults, and validation.
- `test_run_applied_dff_features.py`: applied-dF/F feature runner and
  provenance.
- `test_run_applied_dff_feature_preview.py`: applied feature preview/peak
  config behavior.
- `test_verify_applied_dff_feature_outputs.py`: semantic verification of
  applied feature outputs.

## 10. Required corrections before next Guided feature work

### Required correction 1

- Finding: Guided needs an explicit run-plan contract before any production
  execution or manifest writing.
- Evidence from code/tests: Confirm Strategy is UI-state-only and Run is
  placeholder/skipped.
- Why it matters for long-duration photometry: The plan must be dataset/ROI
  scoped and reproducible, not current selection scoped.
- Severity: acceptance-blocking before next Guided production feature.
- Recommended action: Add a report-only/design contract for Guided run plans,
  then tests, before wiring production Run.
- Files/functions involved: future plan module, `gui/main_window.py`,
  `gui/run_spec.py`.

### Required correction 2

- Finding: Feature/event detection must be audited into Guided as reuse of
  existing app capabilities, not as new detector code.
- Evidence from code/tests: Existing detector/config/Full Control/tests are
  present.
- Why it matters for long-duration photometry: Reimplementation would create
  inconsistent peak counts and provenance.
- Severity: acceptance-blocking before adding Guided feature/event work.
- Recommended action: Design Guided feature/event profiles around current
  `Config` and `extract_features` semantics.
- Files/functions involved: `feature_extraction.py`, `config.py`,
  `main_window.py`, `run_spec.py`, feature tests.

### Required correction 3

- Finding: Guided Review needs a long-duration review concept before it becomes
  real UI.
- Evidence from code/tests: Guided Review is placeholder; existing review
  capability is mostly Full Control/Open Results.
- Why it matters for long-duration photometry: Review must summarize and
  navigate many chunks/ROIs without forcing manual per-chunk discovery.
- Severity: should fix soon.
- Recommended action: Reuse existing completed-run viewer/parser and add
  summary/navigation requirements before implementation.
- Files/functions involved: `gui/run_report_viewer.py`,
  `gui/run_report_parser.py`, `gui/main_window.py`.

## 11. Optional future improvements

- Add aggregate per-ROI diagnostic summaries in Guided Diagnostics.
- Add a read-only display of existing feature/event outputs in Guided Review.
- Add a feature/event profile preview that uses one evidence chunk but stores
  run/ROI/profile-level settings.
- Add plan-diff UI showing what will run before execution.
- Add explicit stale-state banners when setup, completed run, ROI, or evidence
  chunk changes after evidence generation.
- Add a run-level provenance dashboard that links source hashes, config,
  correction choices, feature settings, and output files.

## 12. Do not reinvent notes

- Do not reinvent raw data discovery or file loading. Reuse `discovery`,
  adapters, strict validation, and pipeline loader code.
- Do not reinvent config generation. Reuse `RunSpec` and
  `filter_config_overrides`.
- Do not reinvent feature/event detection. Reuse
  `photometry_pipeline.core.feature_extraction`.
- Do not reinvent peak/event parameter parsing. Reuse existing Config fields,
  GUI parsing helpers, and knob registry metadata.
- Do not reinvent correction fitting. Reuse `regression.fit_chunk_dynamic` and
  approved applied-dF/F production tools.
- Do not reinvent completed-run review. Reuse completed-run parser/viewer
  machinery and existing output artifacts.
- Do not reinvent output safety gates. Reuse the preview/diagnostic/applied
  namespace safety patterns already added.
- Do not add executable `auto` strategy selection from Guided diagnostics.

## Files inspected

- `gui/main_window.py`
- `gui/run_spec.py`
- `gui/knobs_registry.py`
- `gui/run_report_parser.py`
- `gui/run_report_viewer.py`
- `photometry_pipeline/config.py`
- `photometry_pipeline/pipeline.py`
- `photometry_pipeline/core/feature_extraction.py`
- `photometry_pipeline/preview/correction_preview.py`
- `photometry_pipeline/signal_only_f0_diagnostics/contract.py`
- `photometry_pipeline/signal_only_f0_diagnostics/generate.py`
- `tools/run_applied_dff_features.py`
- Guided, preview, diagnostic, feature, event, peak, applied-dF/F, and run-spec
  tests under `tests/`.
