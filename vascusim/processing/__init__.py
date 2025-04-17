"""
Processing module for vascular simulation data.

This module provides utilities for processing vascular geometry data,
including parallel processing capabilities and preprocessing functions.
"""

from .geometry import (
    compute_curvature,
    compute_surface_area,
    compute_volume,
    extract_centerline,
    compute_branch_angles,
)
from .parallel import parallelize, worker_pool, process_batch
from .preprocessing import (
    normalize_geometry,
    resample_geometry,
    filter_noise,
    compute_features,
)

__all__ = [
    # Geometry functions
    "compute_curvature",
    "compute_surface_area",
    "compute_volume",
    "extract_centerline",
    "compute_branch_angles",
    
    # Parallel processing
    "parallelize",
    "worker_pool",
    "process_batch",
    
    # Preprocessing
    "normalize_geometry",
    "resample_geometry",
    "filter_noise",
    "compute_features",
]