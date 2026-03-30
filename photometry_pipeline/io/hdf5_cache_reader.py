"""
Shared helper for pipeline tools to read from the Phase 1 HDF5 trace cache.
Implements the operations required for both tonic and phasic consumers.
"""

import os
import h5py
import numpy as np


SUPPORTED_SCHEMA_VERSIONS = ('1', '1.0')


class CacheReadError(RuntimeError):
    """Raised when a cache file is malformed or cannot be read safely."""


def _raise_cache_error(message: str, *, cause: Exception | None = None) -> None:
    if cause is not None:
        raise CacheReadError(message) from cause
    raise CacheReadError(message)


def _open_cache(cache_path: str, expected_mode: str) -> h5py.File:
    """Internal implementation for opening and validating cache."""
    if not os.path.exists(cache_path):
        _raise_cache_error(f"Cache file not found: {cache_path}")
        
    try:
        f = h5py.File(cache_path, 'r')
    except Exception as e:
        _raise_cache_error(f"Failed to open cache file: {cache_path}", cause=e)
        
    if 'meta' not in f:
        f.close()
        _raise_cache_error(f"Missing /meta in cache: {cache_path}")
        
    meta = f['meta']
    
    if 'mode' not in meta.attrs:
        f.close()
        _raise_cache_error(f"Missing mode attribute in /meta for: {cache_path}")
        
    if meta.attrs['mode'] != expected_mode:
        got_mode = meta.attrs['mode']
        f.close()
        _raise_cache_error(
            f"Cache mode mismatch. Expected {expected_mode}, got {got_mode}"
        )
        
    if 'schema_version' not in meta.attrs and 'schema_version' not in meta:
        f.close()
        _raise_cache_error(f"Missing schema_version in /meta for: {cache_path}")
        
    version = meta.attrs.get('schema_version')
    if version is None and 'schema_version' in meta:
        version_data = meta['schema_version'][()]
        # If it's a zero-dimensional or 1D array of size 1, unwrap it
        if isinstance(version_data, np.ndarray) and version_data.size == 1:
            version_data = version_data.item()
        if isinstance(version_data, bytes):
            version_data = version_data.decode('utf-8')
        version = version_data
    
    if str(version) not in SUPPORTED_SCHEMA_VERSIONS:
        f.close()
        _raise_cache_error(f"Unsupported schema_version '{version}' in cache: {cache_path}")
        
    if 'rois' not in meta:
        f.close()
        _raise_cache_error(f"Missing /meta/rois in cache: {cache_path}")
        
    if 'chunk_ids' not in meta:
        f.close()
        _raise_cache_error(f"Missing /meta/chunk_ids in cache: {cache_path}")
        
    return f


def open_tonic_cache(cache_path: str) -> h5py.File:
    return _open_cache(cache_path, 'tonic')


def open_phasic_cache(cache_path: str) -> h5py.File:
    return _open_cache(cache_path, 'phasic')


def list_cache_rois(cache: h5py.File) -> list[str]:
    """Returns the list of ROIs stored in /meta/rois."""
    try:
        return [roi.decode('utf-8') if isinstance(roi, bytes) else str(roi) for roi in cache['meta']['rois']]
    except Exception as e:
        _raise_cache_error("Failed to read ROIs from cache.", cause=e)


def list_cache_chunk_ids(cache: h5py.File) -> list[int]:
    """Returns the list of chunk IDs stored in /meta/chunk_ids."""
    try:
        return [int(cid) for cid in cache['meta']['chunk_ids']]
    except Exception as e:
        _raise_cache_error("Failed to read chunk_ids from cache.", cause=e)


def list_cache_source_files(cache: h5py.File) -> list[str]:
    """Returns the list of original source files stored in /meta/source_files."""
    try:
        return [sf.decode('utf-8') if isinstance(sf, bytes) else str(sf) for sf in cache['meta']['source_files']]
    except Exception as e:
        _raise_cache_error("Failed to read source_files from cache.", cause=e)


def resolve_cache_roi(cache: h5py.File, requested_roi: str | None = None) -> str:
    """
    Resolve and validate the ROI using the /meta/rois dataset.
    If requested_roi is None, auto-selects the first available ROI.
    """
    rois = list_cache_rois(cache)
    if not rois:
        _raise_cache_error("No ROIs available in cache.")
        
    if requested_roi:
        if requested_roi not in rois:
            _raise_cache_error(f"Requested ROI '{requested_roi}' not found in cache.")
        return requested_roi
    
    roi = rois[0]
    print(f"Auto-selected ROI: {roi}")
    return roi


def load_cache_chunk_fields(cache: h5py.File, roi: str, chunk_id: int, fields: list[str]) -> tuple:
    """
    Loads specific fields for a given ROI and chunk_id as plain numpy arrays.
    """
    roi_group = cache.get(f"roi/{roi}")
    if not roi_group:
        _raise_cache_error(f"Missing dataset group for ROI {roi}")
        
    chunk_group_name = f"chunk_{chunk_id}"
    if chunk_group_name not in roi_group:
        _raise_cache_error(f"Missing {chunk_group_name} in {roi} group.")
        
    grp = roi_group[chunk_group_name]
    
    out = []
    for f in fields:
        if f not in grp:
            _raise_cache_error(f"Missing dataset {f} in {roi}/{chunk_group_name}.")
        # copy to memory as np array
        out.append(grp[f][()])
        
    return tuple(out)


def load_cache_chunk_metadata(cache: h5py.File, roi: str, chunk_id: int, keys: list[str]) -> dict:
    """
    Loads specific metadata attributes for a given ROI and chunk_id.
    """
    roi_group = cache.get(f"roi/{roi}")
    if not roi_group:
        _raise_cache_error(f"Missing dataset group for ROI {roi}")
        
    chunk_group_name = f"chunk_{chunk_id}"
    if chunk_group_name not in roi_group:
        _raise_cache_error(f"Missing {chunk_group_name} in {roi} group.")
        
    grp = roi_group[chunk_group_name]
    
    out = {}
    for k in keys:
        if k not in grp.attrs:
            _raise_cache_error(f"Missing attribute {k} in {roi}/{chunk_group_name}.")
        out[k] = grp.attrs[k]
        
    return out


def iter_cache_chunks_for_roi(cache: h5py.File, roi: str, fields: list[str]):
    """
    Yields tuple of fields for each valid chunk sequentially based on /meta/chunk_ids order.
    """
    chunk_ids = list_cache_chunk_ids(cache)
    
    for chunk_id in chunk_ids:
        yield load_cache_chunk_fields(cache, roi, chunk_id, fields)
