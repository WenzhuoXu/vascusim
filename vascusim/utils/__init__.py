"""
Utility functions for working with vascular simulation data.

This module provides utilities for visualization and metadata handling
for vascular simulation datasets.
"""

from .metadata import (
    query_metadata,
    filter_by_attribute,
    get_unique_values,
    merge_metadata,
)
from .visualization import (
    plot_geometry,
    plot_flow,
    plot_pressure,
    plot_mesh,
    plot_comparison,
    create_animation,
)

__all__ = [
    # Metadata functions
    "query_metadata",
    "filter_by_attribute",
    "get_unique_values",
    "merge_metadata",
    
    # Visualization
    "plot_geometry",
    "plot_flow",
    "plot_pressure",
    "plot_mesh",
    "plot_comparison",
    "create_animation",
]