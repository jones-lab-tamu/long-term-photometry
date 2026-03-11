"""
Narrow helper for plotting tools to read from the Phase 1 HDF5 trace cache.
Only implements the operations required for tonic 48h overview plotting.
"""

import os
import sys
import h5py


def open_tonic_cache(cache_path: str) -> h5py.File:
    """
    Open the tonic HDF5 cache in read-only mode and validate mode.
    Hard-fails if the file does not exist or isn't a valid tonic cache.
    """
    if not os.path.exists(cache_path):
        print(f"CRITICAL: Cache file not found: {cache_path}")
        sys.exit(1)
        
    try:
        f = h5py.File(cache_path, 'r')
    except Exception as e:
        print(f"CRITICAL: Failed to open cache file: {e}")
        sys.exit(1)
        
    if 'meta' not in f or f['meta'].attrs.get('mode') != 'tonic':
        print(f"CRITICAL: Invalid or non-tonic cache format: {cache_path}")
        f.close()
        sys.exit(1)
        
    return f


def resolve_tonic_roi(cache: h5py.File, requested_roi: str = None) -> str:
    """
    Resolve and validate the ROI using the /meta/rois dataset.
    If requested_roi is None, auto-selects the first available ROI.
    """
    if 'meta' not in cache or 'rois' not in cache['meta']:
        print("CRITICAL: Missing /meta/rois in cache.")
        sys.exit(1)
        
    rois = [roi.decode('utf-8') if isinstance(roi, bytes) else str(roi) for roi in cache['meta']['rois']]
    if not rois:
        print("CRITICAL: No ROIs available in cache.")
        sys.exit(1)
        
    if requested_roi:
        if requested_roi not in rois:
            print(f"CRITICAL: Requested ROI '{requested_roi}' not found in cache.")
            sys.exit(1)
        return requested_roi
    
    # Auto-select the first ROI
    roi = rois[0]
    print(f"Auto-selected ROI: {roi}")
    return roi


def iter_tonic_chunks_for_roi(cache: h5py.File, roi: str):
    """
    Yields (time_sec, sig_raw, uv_raw, deltaF) for each valid chunk sequentially.
    Hard-fails if any chunk is missing a required dataset.
    """
    if 'meta' not in cache or 'chunk_ids' not in cache['meta']:
        print("CRITICAL: Missing /meta/chunk_ids in cache.")
        sys.exit(1)
        
    chunk_ids = list(cache['meta']['chunk_ids'][()])
    
    # Needs to match the Hdf5TraceCacheWriter layout:
    # /roi/{roi}/chunk_{chunk_id}/{dataset}
    roi_group = cache.get(f"roi/{roi}")
    if not roi_group:
        print(f"CRITICAL: Missing dataset group for ROI {roi}")
        sys.exit(1)
        
    for chunk_id in chunk_ids:
        chunk_group_name = f"chunk_{chunk_id}"
        if chunk_group_name not in roi_group:
            print(f"CRITICAL: Missing chunk_{chunk_id} in {roi} group.")
            sys.exit(1)
            
        grp = roi_group[chunk_group_name]
        
        required_dsets = ('time_sec', 'sig_raw', 'uv_raw', 'deltaF')
        for ds in required_dsets:
            if ds not in grp:
                print(f"CRITICAL: Missing dataset {ds} in {roi}/chunk_{chunk_id}.")
                sys.exit(1)
                
        yield (
            grp['time_sec'][()],
            grp['sig_raw'][()],
            grp['uv_raw'][()],
            grp['deltaF'][()]
        )
