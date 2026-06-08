"""
Pure batch-planning model for GUI batch processing.

Patch 1 intentionally contains no GUI dialog, subprocess, PipelineRunner, or
analysis execution logic.  Future orchestration should derive each dataset run
as:

    frozen base RunSpec/shared settings
    + BatchDatasetRow.input_path
    + BatchDatasetRow.output_path

and then reuse the existing single-run RunSpec/wrapper execution path.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import yaml


BATCH_DATASET_STATUSES = frozenset(
    {
        "pending",
        "validating",
        "running",
        "success",
        "failed",
        "skipped",
        "cancelled",
    }
)

BATCH_MANIFEST_CSV_FIELDS = [
    "batch_id",
    "dataset_id",
    "dataset_name",
    "input_path",
    "output_path",
    "status",
    "started_at",
    "finished_at",
    "elapsed_sec",
    "mode",
    "format",
    "acquisition_mode",
    "continuous_window_sec",
    "continuous_step_sec",
    "config_path",
    "message",
    "run_dir",
    "status_json_path",
    "run_report_path",
]


@dataclass(frozen=True)
class DiscoveredDataset:
    """One immediate-subfolder dataset discovered under a batch input root."""

    dataset_name: str
    input_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_name", str(self.dataset_name))
        object.__setattr__(self, "input_path", os.path.abspath(os.fspath(self.input_path)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "input_path": self.input_path,
        }


@dataclass(frozen=True)
class BatchDiscoveryResult:
    """Deterministic immediate-subfolder discovery result."""

    batch_input_root: str
    datasets: list[DiscoveredDataset] = field(default_factory=list)
    ignored_root_files: list[str] = field(default_factory=list)
    ignored_hidden_dirs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "batch_input_root", os.path.abspath(os.fspath(self.batch_input_root))
        )
        object.__setattr__(self, "datasets", list(self.datasets))
        object.__setattr__(
            self, "ignored_root_files", [os.path.abspath(os.fspath(p)) for p in self.ignored_root_files]
        )
        object.__setattr__(
            self, "ignored_hidden_dirs", [os.path.abspath(os.fspath(p)) for p in self.ignored_hidden_dirs]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_input_root": self.batch_input_root,
            "datasets": [d.to_dict() for d in self.datasets],
            "ignored_root_files": list(self.ignored_root_files),
            "ignored_root_files_count": len(self.ignored_root_files),
            "ignored_hidden_dirs": list(self.ignored_hidden_dirs),
            "ignored_hidden_dirs_count": len(self.ignored_hidden_dirs),
        }


@dataclass
class BatchDatasetRow:
    """Planning/status row for one independent dataset run."""

    dataset_id: str
    dataset_name: str
    input_path: str
    output_path: str
    status: str = "pending"
    message: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_sec: float | None = None
    run_dir: str | None = None
    status_json_path: str | None = None
    run_report_path: str | None = None

    def __post_init__(self) -> None:
        self.dataset_id = str(self.dataset_id)
        self.dataset_name = str(self.dataset_name)
        self.input_path = os.path.abspath(os.fspath(self.input_path))
        self.output_path = os.path.abspath(os.fspath(self.output_path))
        if self.run_dir is None:
            self.run_dir = self.output_path
        else:
            self.run_dir = os.path.abspath(os.fspath(self.run_dir))
        if self.status_json_path is not None:
            self.status_json_path = os.path.abspath(os.fspath(self.status_json_path))
        if self.run_report_path is not None:
            self.run_report_path = os.path.abspath(os.fspath(self.run_report_path))
        if self.status not in BATCH_DATASET_STATUSES:
            raise ValueError(
                f"Unknown batch dataset status {self.status!r}; "
                f"expected one of {sorted(BATCH_DATASET_STATUSES)}"
            )
        if self.elapsed_sec is not None:
            self.elapsed_sec = float(self.elapsed_sec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRunSpec:
    """
    Frozen, serializable batch intent.

    ``base_run_spec`` and ``shared_settings`` are copied on construction so the
    batch no longer references live GUI widget state.  Patch 2 can use these
    frozen values to construct one normal RunSpec per BatchDatasetRow.
    """

    batch_id: str
    created_at: str
    batch_input_root: str
    batch_output_root: str
    runs_dir: str
    base_run_spec: dict[str, Any] = field(default_factory=dict)
    shared_settings: dict[str, Any] = field(default_factory=dict)
    datasets: list[BatchDatasetRow] = field(default_factory=list)
    overwrite: bool = False
    stop_on_failure: bool = False
    finished_at: str | None = None
    software: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.batch_id = str(self.batch_id)
        self.created_at = str(self.created_at)
        self.batch_input_root = os.path.abspath(os.fspath(self.batch_input_root))
        self.batch_output_root = os.path.abspath(os.fspath(self.batch_output_root))
        self.runs_dir = os.path.abspath(os.fspath(self.runs_dir))
        self.base_run_spec = _json_safe(copy.deepcopy(self.base_run_spec))
        self.shared_settings = _json_safe(copy.deepcopy(self.shared_settings))
        self.datasets = [copy.deepcopy(row) for row in self.datasets]
        self.overwrite = bool(self.overwrite)
        self.stop_on_failure = bool(self.stop_on_failure)
        self.software = _json_safe(copy.deepcopy(self.software))

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "batch_input_root": self.batch_input_root,
            "batch_output_root": self.batch_output_root,
            "runs_dir": self.runs_dir,
            "overwrite": self.overwrite,
            "stop_on_failure": self.stop_on_failure,
            "software": copy.deepcopy(self.software),
            "base_run_spec": copy.deepcopy(self.base_run_spec),
            "shared_settings": copy.deepcopy(self.shared_settings),
            "summary": compute_batch_summary_counts(self.datasets),
            "datasets": [row.to_dict() for row in self.datasets],
        }


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for batch provenance."""
    return datetime.now(timezone.utc).isoformat()


def discover_batch_datasets(batch_input_root: str) -> BatchDiscoveryResult:
    """
    Discover candidate batch datasets using immediate subfolders only.

    Files directly under the batch root are ignored and reported.  Nested
    folders inside a dataset folder are not independently discovered.
    """
    root = os.path.abspath(os.fspath(batch_input_root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Batch input root does not exist: {root}")

    datasets: list[DiscoveredDataset] = []
    ignored_root_files: list[str] = []
    ignored_hidden_dirs: list[str] = []

    for name in sorted(os.listdir(root), key=lambda s: s.lower()):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            if name.startswith("."):
                ignored_hidden_dirs.append(path)
                continue
            datasets.append(DiscoveredDataset(dataset_name=name, input_path=path))
        elif os.path.isfile(path):
            ignored_root_files.append(path)

    return BatchDiscoveryResult(
        batch_input_root=root,
        datasets=datasets,
        ignored_root_files=ignored_root_files,
        ignored_hidden_dirs=ignored_hidden_dirs,
    )


def plan_batch_outputs(
    discovered_datasets: BatchDiscoveryResult | Iterable[DiscoveredDataset],
    batch_output_root: str,
    *,
    overwrite: bool = False,
) -> list[BatchDatasetRow]:
    """Plan deterministic per-dataset output folders under ``<root>/runs``."""
    datasets = (
        discovered_datasets.datasets
        if isinstance(discovered_datasets, BatchDiscoveryResult)
        else list(discovered_datasets)
    )
    output_root = os.path.abspath(os.fspath(batch_output_root))
    runs_dir = os.path.join(output_root, "runs")
    rows: list[BatchDatasetRow] = []

    for index, dataset in enumerate(datasets, start=1):
        slug = sanitize_dataset_slug(dataset.dataset_name)
        dataset_id = f"dataset_{index:03d}"
        output_path = os.path.join(runs_dir, f"{slug}_{index:03d}")
        status = "pending"
        message = ""
        if os.path.exists(output_path) and not overwrite:
            status = "skipped"
            message = f"Output path already exists and overwrite is disabled: {output_path}"
        rows.append(
            BatchDatasetRow(
                dataset_id=dataset_id,
                dataset_name=dataset.dataset_name,
                input_path=dataset.input_path,
                output_path=output_path,
                status=status,
                message=message,
            )
        )

    return rows


def sanitize_dataset_slug(name: str) -> str:
    """Return a deterministic filesystem-safe dataset slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "dataset"


def make_batch_run_spec(
    *,
    batch_id: str,
    batch_input_root: str,
    batch_output_root: str,
    base_run_spec: dict[str, Any] | None = None,
    shared_settings: dict[str, Any] | None = None,
    datasets: list[BatchDatasetRow] | None = None,
    overwrite: bool = False,
    stop_on_failure: bool = False,
    created_at: str | None = None,
    software: dict[str, Any] | None = None,
) -> BatchRunSpec:
    """Convenience constructor that derives the conventional ``runs`` folder."""
    output_root = os.path.abspath(os.fspath(batch_output_root))
    return BatchRunSpec(
        batch_id=batch_id,
        created_at=created_at or utc_now_iso(),
        batch_input_root=batch_input_root,
        batch_output_root=output_root,
        runs_dir=os.path.join(output_root, "runs"),
        base_run_spec=base_run_spec or {},
        shared_settings=shared_settings or {},
        datasets=datasets or [],
        overwrite=overwrite,
        stop_on_failure=stop_on_failure,
        software=software or {},
    )


def compute_batch_summary_counts(rows: Iterable[BatchDatasetRow]) -> dict[str, int]:
    """Count batch rows by supported status."""
    counts = {status: 0 for status in sorted(BATCH_DATASET_STATUSES)}
    total = 0
    for row in rows:
        if row.status not in BATCH_DATASET_STATUSES:
            raise ValueError(f"Unknown batch dataset status: {row.status!r}")
        counts[row.status] += 1
        total += 1
    counts["total"] = total
    return counts


def write_batch_manifest_csv(batch_spec: BatchRunSpec, path: str) -> str:
    """Write deterministic batch manifest CSV and return the written path."""
    out_path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shared = batch_spec.shared_settings
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_MANIFEST_CSV_FIELDS)
        writer.writeheader()
        for row in batch_spec.datasets:
            row_dict = row.to_dict()
            writer.writerow(
                {
                    "batch_id": batch_spec.batch_id,
                    "dataset_id": row.dataset_id,
                    "dataset_name": row.dataset_name,
                    "input_path": row.input_path,
                    "output_path": row.output_path,
                    "status": row.status,
                    "started_at": row.started_at,
                    "finished_at": row.finished_at,
                    "elapsed_sec": row.elapsed_sec,
                    "mode": shared.get("mode", ""),
                    "format": shared.get("format", ""),
                    "acquisition_mode": shared.get("acquisition_mode", ""),
                    "continuous_window_sec": shared.get("continuous_window_sec", ""),
                    "continuous_step_sec": shared.get("continuous_step_sec", ""),
                    "config_path": shared.get("config_path", shared.get("config_source_path", "")),
                    "message": row_dict.get("message", ""),
                    "run_dir": row.run_dir,
                    "status_json_path": row.status_json_path,
                    "run_report_path": row.run_report_path,
                }
            )
    return out_path


def write_batch_manifest_json(batch_spec: BatchRunSpec, path: str) -> str:
    """Write batch manifest JSON and return the written path."""
    out_path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(batch_spec.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def write_batch_run_spec_json(batch_spec: BatchRunSpec, path: str) -> str:
    """Write the frozen batch run spec JSON and return the written path."""
    return write_batch_manifest_json(batch_spec, path)


def write_batch_config_used_yaml(batch_spec: BatchRunSpec, path: str) -> str:
    """Write frozen shared batch settings as YAML and return the written path."""
    out_path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            batch_spec.shared_settings,
            f,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )
    return out_path


def _json_safe(value: Any) -> Any:
    """Deep-convert common Python objects into JSON-serializable primitives."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
