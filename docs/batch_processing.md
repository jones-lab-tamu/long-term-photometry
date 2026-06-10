# Batch Processing

Batch mode runs the same analysis configuration across multiple independent datasets.

## Dataset discovery

The batch input root is scanned only for immediate subfolders. Each immediate subfolder is treated as one independent dataset. Nested recursive dataset discovery is not the batch contract.

## Shared configuration

Batch mode freezes the current GUI/run settings into one shared configuration and applies it to each dataset row. This is intended for consistent processing across independent animals/runs.

## Outputs

Each dataset receives a normal completed-run output folder under the batch output root, usually under `runs/`.

Batch-level provenance includes:
- `batch_manifest.csv`
- `batch_manifest.json`
- `batch_run_spec.json`
- `batch_config_used.yaml`
- `batch_readme.txt`

Row status values include planned, running, success, failed, skipped, and cancelled depending on the execution state.

## Scope limits

Batch mode does not perform group statistics, group averaging, multi-animal modeling, or simultaneous multi-recording visualization. It is an execution convenience for applying one configuration to independent datasets.
