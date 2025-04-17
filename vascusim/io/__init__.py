"""
Input/Output module for handling VTU/VTP files and metadata.

This module provides utilities for reading and writing VTU/VTP files,
streaming data from remote sources, and managing cache for efficient
data access during training.
"""

from .cache import CacheManager
from .formats import read_metadata, read_vtp, read_vtu
from .streaming import DataStreamer, HuggingFaceStreamer, NASStreamer
from .vtk_utils import (
    convert_vtk_to_numpy,
    extract_attributes,
    extract_mesh_from_vtu,
    extract_points_from_vtp,
)

__all__ = [
    # File formats
    "read_vtu",
    "read_vtp",
    "read_metadata",
    # Streaming
    "DataStreamer",
    "HuggingFaceStreamer",
    "NASStreamer",
    # Caching
    "CacheManager",
    # VTK utilities
    "extract_mesh_from_vtu",
    "extract_points_from_vtp",
    "extract_attributes",
    "convert_vtk_to_numpy",
]
