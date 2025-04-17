"""
Data handling module for vascular simulation data.

This module provides dataset implementations, conversion utilities,
and transformation functions for working with vascular simulation data
in a PyTorch Geometric compatible format.
"""

from . import transforms
from .conversion import build_graph, vtp_to_pyg, vtu_to_pyg
from .dataset import StreamingVascuDataset, VascuDataset

__all__ = [
    # Datasets
    "VascuDataset",
    "StreamingVascuDataset",
    # Conversion utilities
    "vtu_to_pyg",
    "vtp_to_pyg",
    "build_graph",
    # Transforms
    "transforms",
]
