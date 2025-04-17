"""
Input/Output module for handling VTU/VTP files and metadata.

This module provides utilities for reading and writing VTU/VTP files,
streaming data from remote sources, and managing cache for efficient
data access during training.
"""

from .formats import read_vtu, read_vtp, read_metadata
from .streaming import DataStreamer, HuggingFaceStreamer, NASStreamer
from .cache import CacheManager
from .vtk_utils import (
    extract_mesh_from_vtu,
    extract_points_from_vtp,
    extract_attributes,
    convert_vtk_to_numpy,
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