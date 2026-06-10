# Event Detection

Event detection runs on corrected phasic outputs after preprocessing/correction.

## Threshold families

Configured by `peak_threshold_method`:
- `mean_std`: threshold from mean plus/minus `peak_threshold_k` times standard deviation.
- `median_mad`: threshold from median plus/minus a MAD-based robust scale.
- `percentile`: threshold from `peak_threshold_percentile` and its lower-tail counterpart when polarity requires it.
- `absolute`: fixed threshold from `peak_threshold_abs`.

`signal_excursion_polarity` controls whether positive, negative, or both tails are considered.

## Shape and spacing filters

Common parameters:
- `peak_min_distance_sec`: minimum separation between events.
- `peak_min_prominence_k`: prominence multiplier.
- `peak_min_width_sec`: minimum event width.
- `peak_pre_filter`: optional pre-filter mode.
- `event_auc_baseline`: `zero` or `median` AUC baseline.
- `event_signal`: detect events from `dff` or `delta_f`.

Default first-run/tutorial settings are intentionally conservative for synthetic demos:
- `peak_threshold_method: mean_std`
- `peak_threshold_k: 2.5`
- `peak_min_distance_sec: 1.0`
- `peak_min_prominence_k: 2.0`
- `peak_min_width_sec: 0.3`

These defaults are designed to reduce obvious background-noise over-detection in tutorial data. They are not universal sensor-independent truth; weaker events may require lower thresholds or prominence settings after inspection.

## Standardizing settings

Choose event settings before comparing datasets. Reuse a saved config or downstream retune configuration so all relevant datasets use the same event criteria.

Event settings are saved in generated configs and run provenance (`config_used.yaml`, GUI effective configs, run reports/status artifacts where applicable), so selected criteria can be reviewed and reused.

Correction retuning changes upstream correction-sensitive outputs. Downstream event reanalysis changes event detection and event summaries using cached corrected traces.

## Outputs

Event-facing outputs include:
- `_analysis/phasic_out/features/features.csv`
- per-ROI tables under `<run_dir>/<ROI>/tables/`
- phasic summary plots under `<run_dir>/<ROI>/summary/`
- run reports and QC summaries under `_analysis/phasic_out/`

Exact file names vary by run profile and mode.
