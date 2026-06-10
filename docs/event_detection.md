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

## Standardizing settings

Choose event settings before comparing datasets. Reuse a saved config or downstream retune configuration so all relevant datasets use the same event criteria.

Correction retuning changes upstream correction-sensitive outputs. Downstream event reanalysis changes event detection and event summaries using cached corrected traces.

## Outputs

Event-facing outputs include:
- `_analysis/phasic_out/features/features.csv`
- per-ROI tables under `<run_dir>/<ROI>/tables/`
- phasic summary plots under `<run_dir>/<ROI>/summary/`
- run reports and QC summaries under `_analysis/phasic_out/`

Exact file names vary by run profile and mode.
