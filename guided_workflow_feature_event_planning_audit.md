# Guided Workflow Feature/Event Planning Audit

This Stage 4E document is a design and reuse audit only. It does not add
Guided feature/event controls, run feature extraction, generate previews, write
manifests, write plan files, create applied-dF/F outputs, call validation, call
the pipeline, generate RunSpec output, or change Full Control behavior.

## 1. Executive Summary

Feature/event detection already exists in the application. The missing Guided
Workflow layer is not a detector; it is a planning/profile layer that can
represent feature/event settings in the Guided draft plan without treating the
current ROI/chunk widgets as production scope.

The existing source of truth is:

- `photometry_pipeline.config.Config` for field names, defaults, and config
  validation.
- `gui.knobs_registry` and GUI parser helpers for allowlisted GUI-originated
  config keys and user-facing knob validation.
- Full Control controls in `gui/main_window.py` for current run-level
  feature/event settings.
- `photometry_pipeline.core.feature_extraction.extract_features` and helper
  functions for detection semantics.
- `photometry_pipeline.pipeline.Pipeline` for standard `features/features.csv`
  writing.
- applied-dF/F feature tools for separate applied-output feature extraction and
  verification.

Guided Workflow should eventually expose feature/event settings as explicit
plan profiles: run-level by default, with selected-ROI-group or ROI-specific
overrides only when added by explicit user action. Chunk IDs may be recorded as
preview/QC evidence provenance, but must not define production feature/event
scope.

This stage makes no runtime changes. It documents how a future implementation
should reuse existing machinery and where tests must guard against accidental
per-chunk planning.

## 2. Existing Source-Of-Truth Inventory

| Capability | Existing source of truth | Files/functions/classes | Current scope | How Guided should reuse it | What must not be duplicated |
| --- | --- | --- | --- | --- | --- |
| Config feature/event fields | `Config` dataclass | `photometry_pipeline/config.py` | Run/config level | Store profile `config_fields` using existing Config names | New field names or parallel schema |
| Config validation | `Config.from_yaml`, dataclass construction | `Config.from_yaml`, `Config(**payload)` | Config load/validation | Reuse for profile validation or value validation | Separate enum/range semantics |
| GUI knob allowlist | Knob registry | `gui/knobs_registry.py` | GUI-originated config overrides | Reuse to decide which Config fields Guided can expose | Bypassing allowlist for GUI-edited profile fields |
| Full Control parsing | Event feature parser | `parse_and_validate_event_feature_knobs` in `gui/main_window.py` | Full Control run setup | Reuse parsing rules or factor them into shared helper before Guided controls | New parser with different errors/defaults |
| Full Control feature/event controls | Existing widgets | `_event_signal_combo`, `_signal_excursion_polarity_combo`, `_peak_method_combo`, `_peak_*_edit`, `_peak_pre_filter_combo`, `_event_auc_combo`, `_traces_only_cb` in `gui/main_window.py` | Run-level GUI setup | Use as behavioral reference; Guided may use simpler profiles/presets | Breaking or replacing Full Control controls |
| RunSpec/config generation | Existing RunSpec path | `gui/run_spec.py::RunSpec.generate_derived_config`, `build_runner_argv` | Run execution setup | Future Guided Run should map validated profile fields through RunSpec/config | Direct pipeline calls from widgets |
| Feature extraction backend | Authoritative detector | `photometry_pipeline/core/feature_extraction.py::extract_features`, `get_peak_indices_for_trace`, `compute_detection_threshold_bounds`, `compute_auc_above_threshold` | Per chunk/ROI execution internals | Preview and production should call existing backend | New detector implementation |
| `event_signal` semantics | Backend and Config | `get_event_signal_array`, `Config.event_signal` | Chooses dFF or delta_F trace | Store existing `event_signal` values in profiles | New signal labels |
| Polarity semantics | Backend and Config | `normalize_signal_excursion_polarity`, `get_peak_indices_for_trace`, `compute_auc_above_threshold` | Peak/AUC semantics | Store existing `signal_excursion_polarity` values | Separate signed-event interpretation |
| Threshold semantics | Backend and Config | `compute_detection_threshold_bounds` | Detector threshold | Reuse `mean_std`, `percentile`, `median_mad`, `absolute` | Reimplemented threshold formulas |
| Distance/prominence/width semantics | Backend and Config | `get_peak_indices_for_trace`, `_resolve_prominence_requirement`, `_resolve_width_samples` | Detector constraints | Store existing numeric fields | New peak filtering semantics |
| Pre-filter semantics | Backend and Config | `apply_peak_prefilter`, `Config.peak_pre_filter` | Detection trace preprocessing | Reuse existing modes exposed to Config/GUI | New preview-only smoothing semantics for production |
| AUC baseline semantics | Backend and Config | `extract_features`, `compute_auc_above_threshold`, `Config.event_auc_baseline` | AUC calculation | Store existing `zero`/`median` settings | New AUC definitions |
| Standard pipeline feature outputs | Pipeline pass 2 | `photometry_pipeline/pipeline.py` | Standard phasic output | Future Guided Run should let pipeline write outputs | Direct Guided `features.csv` writer |
| Applied-dF/F feature outputs | Applied toolchain | `tools/run_applied_dff_features.py`, `tools/verify_applied_dff_feature_outputs.py` | Separate applied output dirs | Reuse for applied-dF/F planning/execution later | Writing applied features from Guided widgets |
| Traces-only behavior | Runner/RunSpec/Pipeline | `RunSpec.traces_only`, runner `--traces-only`, `Pipeline.traces_only` | Run-level skip feature extraction | Represent as explicit run/profile policy later | Implicitly skipping features based on incomplete profile |
| Feature/event tests | Existing test suite | `test_event_signal_selection.py`, `test_event_polarity_semantics.py`, `test_peak_detection_hardening.py`, `test_feature_extraction_nan_gap_bias.py`, applied feature tests, pipeline/traces-only tests | Existing app behavior | Keep passing as regression suite | New tests that encode different detector behavior |
| Completed-run review/output loading | Existing Open Results/review paths | `gui/run_report_parser.py`, `gui/run_report_viewer.py`, plot/viewer tools | Completed-run output review | Future Guided Review should read existing feature outputs | New partial feature-output parser unless needed |

## 3. Existing Feature/Event Field Inventory

The following fields are already defined in `photometry_pipeline/config.py`.
Guided profiles should use these exact names in `config_fields`.

| Field | Default | Valid values/type known from code | Used by | Guided exposure recommendation |
| --- | --- | --- | --- | --- |
| `event_signal` | `dff` | `dff`, `delta_f` | `get_event_signal_array`; runner context; applied feature tools force `dff` | Expose directly or through preset; required profile field if user edits event signal |
| `signal_excursion_polarity` | `positive` | `positive`, `negative`, `both` | Peak polarity and signed AUC semantics | Expose directly in advanced/simple profile; important for inhibitory signals |
| `peak_threshold_method` | `mean_std` | `mean_std`, `percentile`, `median_mad`, `absolute` | `compute_detection_threshold_bounds` | Expose directly or via presets; determines which numeric threshold field is active |
| `peak_threshold_k` | `2.5` | float; GUI parser requires `> 0` when active; Config rejects invalid absolute separately | `mean_std`, `median_mad` thresholds | Expose when threshold method uses k |
| `peak_threshold_percentile` | `95.0` | float; GUI parser requires 0 to 100 | percentile threshold | Expose when threshold method is percentile |
| `peak_threshold_abs` | `0.0` | float; Config requires `> 0` when method is `absolute` | absolute threshold | Expose only when method is absolute |
| `peak_min_distance_sec` | `1.0` | float; GUI parser requires `>= 0` | `find_peaks(distance=...)` | Expose in advanced/simple profile |
| `peak_min_prominence_k` | `2.0` | float; Config/parser require `>= 0` | optional prominence requirement from robust noise | Expose in advanced controls; include in presets |
| `peak_min_width_sec` | `0.3` | float; Config/parser require `>= 0` | optional peak width requirement | Expose in advanced controls; include in presets |
| `peak_pre_filter` | `none` | Config allows `none`, `lowpass`; backend also recognizes `smooth` internally, but registry exposes `none`, `lowpass` | `apply_peak_prefilter` | Expose only registry-allowed values unless Config/registry are expanded first |
| `event_auc_baseline` | `zero` | `zero`, `median` | AUC baseline in `extract_features` | Expose directly or preset |
| `traces_only` | `False` on RunSpec/Pipeline | boolean run intent; not a Config field | Runner/RunSpec/Pipeline skip feature extraction and omit features outputs | Represent separately from feature profile as run-output policy, not as detector config |

Additional relevant fields and behavior:

- `export_display_series_csv` is an advanced plotting/export setting, not a
  feature detector parameter.
- `representative_session_index` and `preview_first_n` are Config fields used
  for preview/run selection context, not feature profile detector semantics.
- Applied-dF/F feature tools force `event_signal="dff"` for applied traces.
  A future shared profile contract may need an applied-feature mode that
  constrains or normalizes `event_signal` to `dff`.

## 4. Existing Full Control Behavior

Full Control currently exposes feature/event configuration as run-level setup.
The visible controls include:

- Event Signal combo (`dff`, `delta_f`).
- Signal Excursion Polarity combo (`positive`, `negative`, `both`).
- Peak Threshold Method combo (`mean_std`, `percentile`, `median_mad`,
  `absolute`).
- Numeric entries for threshold k, percentile, absolute threshold, minimum
  distance, minimum prominence, and minimum width.
- Peak Pre-Filter combo.
- Event AUC Baseline combo.
- `Skip feature extraction (traces and QC only)` checkbox.

The controls map through `_event_feature_defaults_from_active_baseline`,
`_sync_event_feature_controls_from_active_baseline`, and
`parse_and_validate_event_feature_knobs`. `_build_run_spec` places parsed
values into RunSpec `config_overrides`, while `traces_only` is a RunSpec intent
field rather than a Config override.

Validation currently occurs before RunSpec/config generation by parsing GUI
values and by `RunSpec.generate_derived_config`, which filters GUI-originated
overrides through `gui.knobs_registry.filter_config_overrides`. The generated
config can then be loaded by `Config.from_yaml`.

Guided should reuse this validation path. If the parser remains GUI-bound, the
next implementation stage should factor the reusable validation into a
non-widget helper rather than duplicating the rules.

## 5. Existing Backend/Output Behavior

The authoritative backend is `photometry_pipeline.core.feature_extraction`.
The main function is `extract_features(chunk, config)`. It returns one row per
ROI for a chunk with columns documented in the function docstring:

- `chunk_id`
- `source_file`
- `roi`
- `mean`
- `median`
- `std`
- `mad`
- `peak_count`
- `auc`

`extract_features` obtains the analysis trace from `event_signal`, computes
threshold bounds, detects peaks with distance/prominence/width constraints,
handles NaN finite runs, and computes polarity-aware AUC.

The standard pipeline calls `feature_extraction.extract_features(chunk,
self.config)` during pass 2 when `traces_only` is false. It concatenates rows
and writes:

```text
<phasic_out>/features/features.csv
```

When `traces_only` is true, feature extraction and feature-dependent outputs
are skipped.

Applied-dF/F feature extraction is separate:

- `tools/run_applied_dff_features.py` verifies/loads an applied trace cache,
  builds chunks with `applied_dff` as `dff`, forces feature config
  `event_signal="dff"`, calls `extract_features`, and writes separate applied
  feature outputs under the applied output directory.
- `tools/verify_applied_dff_feature_outputs.py` re-runs `extract_features`
  against applied chunks and compares expected detector output with
  `features.csv`/`features.json`.

The applied output files include:

- `features.csv`
- `features.json`
- `feature_summary.csv/json`
- `feature_provenance.json`

Existing tests protect these semantics, including event signal routing,
polarity/AUC behavior, peak hardening knobs, NaN gap behavior, traces-only
omission of `features.csv`, applied feature outputs, and semantic verification.

## 6. Proposed Guided Feature/Event Planning Scope Model

Guided feature/event planning should use these scopes:

- Plan-level default feature/event profile: the default profile for planned
  execution.
- Run-level profile: applies to the whole future run unless overridden.
- Selected-ROI-group profile: future explicit action that resolves a selected
  group of ROIs at edit time.
- ROI-specific profile override: applies to one ROI when explicitly edited.
- Evidence preview/QC chunk references: chunks used to inspect detector
  behavior; provenance only.

Required scope rules:

- Chunk IDs may appear only inside evidence/preview provenance.
- Chunk IDs must not define production feature/event profile scope.
- The current visible chunk must not define production feature/event settings.
- The current visible GUI controls must not silently mutate plan profiles.
- A profile changes only through an explicit user action such as "Apply profile
  to plan" or "Update ROI profile".
- Future Guided Run must consume stored plan profiles, not live widget state.

## 7. Proposed Feature/Event Profile Contract

This is design only. Do not implement in this patch.

A future `FeatureEventProfile` extension should remain compatible with the
existing `GuidedRunPlan.feature_event_profiles` list and should use existing
Config field names:

```json
{
  "profile_id": "default",
  "profile_label": "Default feature detection",
  "scope": "run | roi | selected_roi_group",
  "target_rois": ["CH1"],
  "resolved_rois": ["CH1"],
  "config_fields": {
    "event_signal": "dff",
    "signal_excursion_polarity": "positive",
    "peak_threshold_method": "mean_std",
    "peak_threshold_k": 2.5,
    "peak_threshold_percentile": 95.0,
    "peak_threshold_abs": 0.0,
    "peak_min_distance_sec": 1.0,
    "peak_min_prominence_k": 2.0,
    "peak_min_width_sec": 0.3,
    "peak_pre_filter": "none",
    "event_auc_baseline": "zero"
  },
  "evidence_previews": [
    {
      "chunk_id": 0,
      "role": "representative_evidence",
      "summary": "preview only",
      "stale": false
    }
  ],
  "choice_source": "explicit_user_profile_edit",
  "status": "draft | complete | stale | invalid",
  "provenance_references": []
}
```

Contract rules:

- `scope="chunk"` must remain invalid.
- Unknown `config_fields` must be rejected.
- Values should be validated through existing Config/parser semantics.
- `evidence_previews` are not production targets.
- `choice_source` should distinguish explicit user profile edits from defaults
  or imported profile state.

## 8. Recommended Guided UI Design For Future Stage

Add a future Guided step titled `Feature/event settings` or `Feature detection`.
The first implementation should be conservative:

- Start with one run-level default profile.
- Offer simple presets for common cases only if they map exactly to existing
  Config fields.
- Provide an advanced disclosure area for the existing fields.
- Allow an explicit "Apply profile to draft plan" action.
- Show profile status in the existing draft plan checklist.
- Allow representative chunk preview/QC later, but label it as evidence only.

Do not:

- Add per-chunk production tuning.
- Auto-select profiles from diagnostics.
- Change Full Control widgets or behavior.
- Create a second detector.
- Write `features.csv` from profile preview.

## 9. Validation/Checklist Implications

The Stage 4D6d checklist should eventually interpret feature/event settings as:

- No feature/event profile: `not_configured`.
- Valid profile: `pass` for checklist-level completeness, but not execution
  readiness by itself.
- Unknown Config field: `fail`.
- Invalid value under Config/parser rules: `fail`.
- Missing preview chunks: at most `warning`; preview evidence should not be
  required for every plan.
- Chunk-scoped profile: `fail`.
- Execution readiness remains blocked until source, ROI choices, output
  destination, feature/event profile, RunSpec mapping, validation, and output
  safety are all implemented.

## 10. Future Feature/Event Preview Boundary

Feature/event preview should be a separate evidence generator, not production
execution.

Boundary requirements:

- Preview chunks are evidence only.
- Preview must not write `phasic_out/features/features.csv`.
- Preview must not create applied-dF/F features.
- Preview must not make a profile runnable automatically.
- Preview results may be referenced in plan evidence provenance.
- Preview must not retune production settings per chunk unless explicitly
  framed as QC evidence and then applied by a separate profile-edit action.
- Preview output, if later implemented, must use a protected preview namespace
  and must not mutate source caches, completed-run production outputs, or
  legacy feature outputs.

## 11. Acceptance Criteria For Future Implementation

Future feature/event planning implementation must satisfy:

- Feature/event profiles use existing Config field names.
- Unknown Config fields are rejected.
- Invalid Config/parser values are rejected.
- Chunk scope is rejected.
- Feature/event profile round-trip preserves profile fields and evidence
  preview provenance.
- Changing visible preview chunk does not alter profile settings.
- Diagnostics/previews do not auto-select profiles.
- Future Run consumes the stored plan profile, not live widgets.
- Full Control behavior remains unchanged.
- Existing feature extraction tests continue passing.
- Existing applied-dF/F feature tests continue passing.
- `traces_only` or skip-feature-extraction is represented as explicit run
  policy, not confused with an incomplete profile.

## 12. Tests Needed Before Implementation

Required future tests:

- Feature/event profile contract round-trip.
- Unknown profile field rejection.
- Invalid value rejection using existing Config/parser rules.
- `scope="chunk"` rejection.
- Run-level profile maps to all planned ROIs.
- ROI-level override maps only to that ROI.
- Selected-ROI-group profile stores resolved ROI membership from an explicit
  action.
- Evidence preview chunk is provenance only.
- Changing visible chunk does not mutate profile settings.
- Diagnostics/previews do not auto-populate or auto-select profiles.
- Checklist integration: no profile is `not_configured`; valid profile is
  `pass`; invalid profile is `fail`.
- No-output preview tests for any future feature/event preview backend.
- Later RunSpec/config mapping tests proving profile fields go through
  `RunSpec.generate_derived_config`.
- Existing detector tests: event signal, polarity/AUC, threshold semantics,
  prominence/width hardening, NaN gap behavior, traces-only behavior, applied
  feature verification.

## 13. Recommended Next Implementation Stages

1. Stage 4E1: Extend the minimal plan contract/tests for richer
   feature/event profile metadata only. No GUI, no extraction.
2. Stage 4E2: Add in-memory profile display/checklist integration. No
   extraction and no persistence.
3. Stage 4E3: Add Guided feature/event UI controls that reuse existing Config
   fields and parser/registry semantics. No extraction.
4. Stage 4E4: Design or implement optional feature/event preview backend in a
   protected preview namespace. No production outputs.
5. Later: Map validated plan profiles to RunSpec/config generation.
6. Much later: Guided Run wiring, after output safety, validation, RunSpec
   mapping, and provenance are tested.

## 14. Open Questions / Decisions Needed

- Should the first Guided implementation expose direct controls or named
  presets?
- Should the first profile be run-level only?
- Should ROI-level overrides wait until after run-level profile support is
  stable?
- Should selected-ROI-group profiles be stored as resolved ROI lists only, or
  also retain the original selection rule?
- Should standard pipeline and applied-dF/F feature profiles share one contract
  with mode-specific constraints, or separate contracts?
- How should `traces_only` be represented: output policy, feature profile
  status, or run intent?
- What fields define a "complete enough" feature/event profile for checklist
  pass?
- Should preview evidence be recommended but optional for execution?
- Should Guided inherit current Full Control feature settings when creating the
  first draft profile, or require an explicit user apply action?
- Where should reusable parsing move if Guided and Full Control both need it:
  `gui` helper module or a non-GUI config-profile module?

## Files Inspected

- `guided_workflow_long_duration_audit.md`
- `guided_workflow_run_plan_contract.md`
- `photometry_pipeline/guided_run_plan.py`
- `tests/test_guided_run_plan_contract.py`
- `gui/main_window.py`
- `tests/test_gui_guided_workflow.py`
- `gui/knobs_registry.py`
- `gui/run_spec.py`
- `photometry_pipeline/config.py`
- `photometry_pipeline/pipeline.py`
- `photometry_pipeline/core/feature_extraction.py`
- `tools/run_applied_dff_features.py`
- `tools/verify_applied_dff_feature_outputs.py`
- `tests/test_event_signal_selection.py`
- `tests/test_event_polarity_semantics.py`
- `tests/test_peak_detection_hardening.py`
- feature, pipeline, applied-dF/F, and plotting tests located by searching for
  feature/event Config fields and `features.csv`.

