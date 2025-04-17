"""
VascuSim - Dataset package for cardiovascular simulations.

This package provides tools for processing, streaming, and analyzing vascular
simulation data stored in VTU/VTP format, with efficient conversion to PyTorch
Geometric data formats for GNN training.
"""

import os
import platform
import sys

import numpy as np

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0.dev0"

# Set the path to the directory where this file is located
package_dir = os.path.dirname(os.path.abspath(__file__))

# Import submodules for easy access
from . import data, io, processing, utils

__all__ = [
    "data",
    "io",
    "processing",
    "utils",
    "__version__",
]
