"""Builders for run directories with a coherent terminal set (4J16k40).

Tests that need a *successful* run directory must build one the same way the
wrapper does: mandatory artifacts on disk, a final manifest generated from those
files, and a success status that pins the manifest. Hand-writing a bare
status.json no longer produces a loadable run, which is the point of the
terminal-completion contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    PROFILE_CONTINUOUS,
    PROFILE_FULL_INTERMITTENT,
    PROFILE_TUNING_PREP,
    REPORT_COMPLETION_KEY,
    build_continuous_window_index,
    build_manifest_completion_block,
    build_report_completion_block,
    build_status_completion_block,
    expected_continuous_families,
    normalize_run_mode,
    required_deliverables_for_run_mode,
    sha256_file,
)

DEFAULT_RUN_ID = "run_20260709_000000"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_region_deliverable(run_dir: Path, region: str = "Region0", subfolder: str = "summary") -> None:
    (Path(run_dir) / region / subfolder).mkdir(parents=True, exist_ok=True)


def legacy_run_report(**extra: object) -> dict:
    """A run report with the historical shape that positively identifies a legacy run."""
    report = {
        "run_context": {"run_type": "full"},
        "configuration": {"target_fs_hz": 30.0},
        "derived_settings": {},
        "analytical_contract": {"strict_mode_guarantees": []},
    }
    report.update(extra)
    return report


def write_legacy_run(
    run_dir: Path | str,
    *,
    status: str | None = "success",
    region: str = "Region0",
) -> Path:
    """A positively-identified run from a build that predates the contract."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_report.json", legacy_run_report())
    if status is not None:
        _write_json(
            run_dir / "status.json",
            {"schema_version": 1, "phase": "final", "status": status},
        )
    if region:
        write_region_deliverable(run_dir, region)
    return run_dir


def write_mandatory_artifacts(
    run_dir: Path,
    *,
    run_id: str,
    phasic: bool,
    tonic: bool,
    features: bool,
) -> None:
    """Write the analysis outputs implied by the run mode."""
    if phasic:
        phasic_out = run_dir / "_analysis" / "phasic_out"
        _write_json(phasic_out / "run_report.json", legacy_run_report())
        _write_text(phasic_out / "config_used.yaml", "target_fs_hz: 30.0\n")
        _write_text(phasic_out / "phasic_trace_cache.h5", f"cache-for-{run_id}")
        if features:
            _write_text(phasic_out / "features" / "features.csv", "roi,t_sec\nRegion0,1.0\n")
            _write_json(
                phasic_out / "features" / "feature_event_provenance.json",
                {"schema_version": "guided_feature_event_provenance.v3", "rois": []},
            )
    if tonic:
        tonic_out = run_dir / "_analysis" / "tonic_out"
        _write_json(tonic_out / "run_report.json", legacy_run_report())
        _write_text(tonic_out / "config_used.yaml", "target_fs_hz: 30.0\n")
        _write_text(tonic_out / "tonic_trace_cache.h5", f"tonic-cache-for-{run_id}")


def write_required_deliverables(run_dir: Path, run_mode: dict) -> None:
    """Write exactly the per-ROI deliverables the run mode promises."""
    for rel in required_deliverables_for_run_mode(run_mode):
        _write_text(run_dir / Path(*rel.split("/")), f"deliverable:{rel}")


def full_intermittent_run_mode(
    *,
    expected_rois: list[str],
    phasic: bool = True,
    tonic: bool = True,
    features: bool = True,
    run_profile: str = "full",
    run_type: str = "full",
    traces_only: bool = False,
) -> dict:
    """The run mode a mocked full intermittent wrapper run executes."""
    return normalize_run_mode(
        run_profile=run_profile,
        run_type=run_type,
        acquisition_mode="intermittent",
        traces_only=traces_only,
        phasic_analysis=phasic,
        tonic_analysis=tonic,
        feature_extraction_ran=bool(phasic and features),
        deliverable_profile=PROFILE_FULL_INTERMITTENT,
        expected_rois=expected_rois,
    )


def seed_wrapper_deliverables(
    run_dir: Path | str,
    rois: list[str] | tuple[str, ...] = ("Region0",),
    *,
    tonic: bool = True,
) -> None:
    """Stand in for the per-ROI plot/table subprocesses a mocked wrapper run never launches."""
    write_required_deliverables(
        Path(run_dir), full_intermittent_run_mode(expected_rois=list(rois), tonic=tonic)
    )


def write_current_run(
    run_dir: Path | str,
    *,
    run_id: str = DEFAULT_RUN_ID,
    run_profile: str = "full",
    run_type: str = "full",
    acquisition_mode: str = "intermittent",
    traces_only: bool = False,
    phasic: bool = True,
    tonic: bool = False,
    features: bool = True,
    region: str = "Region0",
    expected_rois: list[str] | None = None,
    deliverable_profile: str | None = None,
    skipped_deliverable_families: list[str] | None = None,
    continuous_outputs_ran: bool | None = None,
    continuous_window_rows: int = 3,
    optional_artifacts: list[str] | None = None,
    write_status: bool = True,
    status_state: str = "success",
    status_phase: str = "final",
    manifest_run_id: str | None = None,
    report_run_id: str | None = None,
    status_run_id: str | None = None,
    manifest_final: bool = True,
) -> Path:
    """Build a run directory that satisfies the current completion contract.

    The keyword overrides exist so tests can corrupt exactly one part of an
    otherwise-coherent terminal set.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_mandatory_artifacts(run_dir, run_id=run_id, phasic=phasic, tonic=tonic, features=features)

    if deliverable_profile is None:
        if acquisition_mode == "continuous":
            deliverable_profile = PROFILE_CONTINUOUS
        elif run_profile == "tuning_prep":
            deliverable_profile = PROFILE_TUNING_PREP
        else:
            deliverable_profile = PROFILE_FULL_INTERMITTENT
    if expected_rois is None:
        expected_rois = [region] if region else []
    if continuous_outputs_ran is None:
        continuous_outputs_ran = deliverable_profile == PROFILE_CONTINUOUS

    report = legacy_run_report()
    report[REPORT_COMPLETION_KEY] = build_report_completion_block(run_id=report_run_id or run_id)
    _write_json(run_dir / "run_report.json", report)

    for rel in optional_artifacts or []:
        _write_text(run_dir / rel, "optional")

    run_mode = normalize_run_mode(
        run_profile=run_profile,
        run_type=run_type,
        acquisition_mode=acquisition_mode,
        traces_only=traces_only,
        phasic_analysis=phasic,
        tonic_analysis=tonic,
        feature_extraction_ran=bool(phasic and features),
        deliverable_profile=deliverable_profile,
        expected_rois=expected_rois,
        skipped_deliverable_families=skipped_deliverable_families or [],
        continuous_outputs_ran=continuous_outputs_ran,
    )
    write_required_deliverables(run_dir, run_mode)
    if region:
        write_region_deliverable(run_dir, region)

    continuous_index = None
    if continuous_outputs_ran:
        continuous_index = build_continuous_window_index(
            str(run_dir),
            run_mode=run_mode,
            row_counts_by_family={
                family: {roi: continuous_window_rows for roi in expected_rois}
                for family in expected_continuous_families(run_mode)
            },
        )
    manifest = {
        "tool": "run_full_pipeline_deliverables",
        "run_id": run_id,
        "run_profile": run_profile,
        "run_type": run_type,
        COMPLETION_KEY: build_manifest_completion_block(
            str(run_dir),
            run_id=manifest_run_id or run_id,
            run_mode=run_mode,
            finalized_utc="2026-07-09T00:00:00+00:00",
            optional_artifacts=optional_artifacts or [],
            continuous_index=continuous_index,
        ),
    }
    if not manifest_final:
        manifest[COMPLETION_KEY]["final"] = False
    _write_json(run_dir / "MANIFEST.json", manifest)

    if write_status:
        status = {
            "schema_version": 1,
            "run_id": run_id,
            "run_profile": run_profile,
            "run_type": run_type,
            "acquisition_mode": acquisition_mode,
            "traces_only": traces_only,
            "phase": status_phase,
            "status": status_state,
            "errors": [],
            "warnings": [],
            COMPLETION_KEY: build_status_completion_block(
                run_id=status_run_id or run_id,
                manifest_sha256=sha256_file(str(run_dir / "MANIFEST.json")),
            ),
        }
        _write_json(run_dir / "status.json", status)

    return run_dir


BASE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "qc_universal_config.yaml"


def _write_trace_cache(path: Path, mode: str) -> None:
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["mode"] = mode
        meta.attrs["schema_version"] = "1"
        meta.create_dataset("rois", data=np.array([b"Region0"]))
        meta.create_dataset("chunk_ids", data=np.array([0], dtype=np.int64))
        meta.create_dataset("source_files", data=np.array([b"chunk_0000.csv"]))
        grp = f.create_group("roi/Region0/chunk_0")
        grp.create_dataset("time_sec", data=np.array([0.0, 3600.0]))
        grp.create_dataset("sig_raw", data=np.array([1.0, 1.1]))
        grp.create_dataset("uv_raw", data=np.array([0.5, 0.6]))
        grp.create_dataset("fit_ref", data=np.array([0.9, 1.0]))
        grp.create_dataset("delta_f", data=np.array([0.1, 0.1]))
        grp.create_dataset("deltaF", data=np.array([0.1, 0.1]))
        grp.create_dataset("dff", data=np.array([0.0, 0.1]))


def write_phasic_feature_outputs(
    phasic_out: Path | str,
    *,
    rois: list[str] | tuple[str, ...] = ("Region0",),
    config_path: Path | str | None = None,
) -> None:
    """Write the feature outputs a real phasic analysis always produces.

    Feature extraction emits one row per ROI per chunk, so features.csv and the
    per-ROI settings record beside it exist whenever it runs -- a recording with
    no detected events yields peak_count 0, never a missing file. The contract
    signal is stamped into the analysis run report, exactly as Pipeline does.
    """
    from photometry_pipeline.config import Config
    from photometry_pipeline.feature_event_provenance import (
        FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
        build_feature_event_provenance_payload,
    )

    phasic_out = Path(phasic_out)
    config_path = Path(config_path) if config_path else BASE_CONFIG_PATH
    cfg = Config.from_yaml(str(config_path))
    rois = list(rois)

    rows = "\n".join(f"{roi},0" for roi in rois)
    _write_text(phasic_out / "features" / "features.csv", f"roi,chunk_id\n{rows}\n")

    payload = build_feature_event_provenance_payload(base_config=cfg, analyzed_rois=rois)
    _write_json(phasic_out / "features" / "feature_event_provenance.json", payload)

    report_path = phasic_out / "run_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else legacy_run_report()
    )
    report["feature_event_provenance"] = {
        "contract_version": FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
        "schema_version": payload.get("schema_version", ""),
        "relative_path": "features/feature_event_provenance.json",
        "global_default_config_digest": payload.get("global_default_config_digest", ""),
        "roi_count": len(payload.get("rois", [])),
    }
    report.setdefault("roi_selection", {"selected_rois": rois, "discovered_rois": rois})
    _write_json(report_path, report)


def seed_wrapper_analysis_outputs(run_dir: Path | str) -> None:
    """Write exactly what the analysis subprocesses produce, for wrapper tests.

    Tests that mock the analysis subprocesses must still leave behind the
    artifacts a real run leaves behind, otherwise they are asserting against a
    run the wrapper would rightly refuse to call successful.
    """
    run_dir = Path(run_dir)
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    phasic_out = run_dir / "_analysis" / "phasic_out"
    (phasic_out / "traces").mkdir(parents=True, exist_ok=True)
    _write_text(
        phasic_out / "traces" / "chunk_0000.csv",
        "time_sec,Region0_deltaF\n0.0,0.0\n3600.0,1.0",
    )
    _write_text(phasic_out / "config_used.yaml", config_text)
    _write_json(phasic_out / "run_report.json", legacy_run_report())
    write_phasic_feature_outputs(phasic_out)
    _write_trace_cache(phasic_out / "phasic_trace_cache.h5", "phasic")

    tonic_out = run_dir / "_analysis" / "tonic_out"
    tonic_out.mkdir(parents=True, exist_ok=True)
    tonic_report = legacy_run_report()
    tonic_report["roi_selection"] = {"selected_rois": ["Region0"]}
    _write_json(tonic_out / "run_report.json", tonic_report)
    _write_text(tonic_out / "config_used.yaml", config_text)
    _write_trace_cache(tonic_out / "tonic_trace_cache.h5", "tonic")


def repin_status_to_manifest(run_dir: Path | str, *, run_id: str = DEFAULT_RUN_ID) -> None:
    """Recompute the status's manifest digest after a test rewrites the manifest."""
    run_dir = Path(run_dir)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    status[COMPLETION_KEY] = build_status_completion_block(
        run_id=run_id, manifest_sha256=sha256_file(str(run_dir / "MANIFEST.json"))
    )
    _write_json(run_dir / "status.json", status)
