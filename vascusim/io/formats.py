"""
File format handling for vascular simulation data.

This module provides functions for reading various file formats used in
vascular simulations, including VTU, VTP, and metadata JSON files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import VTK
try:
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    HAS_VTK = True
except ImportError:
    HAS_VTK = False

from .vtk_utils import extract_mesh_from_vtu, extract_points_from_vtp

logger = logging.getLogger(__name__)


def read_vtu(file_path: Union[str, Path]) -> Tuple[Any, Dict, Dict]:
    """
    Read a VTU file and extract the mesh and data.

    Args:
        file_path: Path to the VTU file.

    Returns:
        Tuple of (mesh, cell_data, point_data).

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is invalid.
    """
    if not HAS_VTK:
        raise ImportError(
            "VTK is required for reading VTU files. Install it with 'pip install vtk'."
        )

    file_path = str(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"VTU file not found: {file_path}")

    try:
        return extract_mesh_from_vtu(file_path)
    except Exception as e:
        logger.error(f"Error reading VTU file {file_path}: {e}")
        raise ValueError(f"Invalid VTU file: {e}")


def read_vtp(file_path: Union[str, Path]) -> Tuple[Any, Dict, Dict]:
    """
    Read a VTP file and extract the polydata and data.

    Args:
        file_path: Path to the VTP file.

    Returns:
        Tuple of (polydata, cell_data, point_data).

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is invalid.
    """
    if not HAS_VTK:
        raise ImportError(
            "VTK is required for reading VTP files. Install it with 'pip install vtk'."
        )

    file_path = str(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"VTP file not found: {file_path}")

    try:
        return extract_points_from_vtp(file_path)
    except Exception as e:
        logger.error(f"Error reading VTP file {file_path}: {e}")
        raise ValueError(f"Invalid VTP file: {e}")


def read_metadata(file_path: Union[str, Path]) -> Dict:
    """
    Read a metadata JSON file.

    Args:
        file_path: Path to the metadata file.

    Returns:
        Dictionary containing the metadata.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = str(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    with open(file_path, "r") as f:
        return json.load(f)


def read_vtu_with_metadata(
    vtu_file: Union[str, Path], metadata_file: Optional[Union[str, Path]] = None
) -> Tuple[Any, Dict, Dict, Dict]:
    """
    Read a VTU file and its associated metadata.

    Args:
        vtu_file: Path to the VTU file.
        metadata_file: Path to the metadata file. If None, looks for a .json file
                      with the same base name.

    Returns:
        Tuple of (mesh, cell_data, point_data, metadata).

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If any file doesn't exist.
        ValueError: If any file is invalid.
    """
    vtu_file = str(vtu_file)

    if not os.path.exists(vtu_file):
        raise FileNotFoundError(f"VTU file not found: {vtu_file}")

    # Get mesh and data
    mesh, cell_data, point_data = read_vtu(vtu_file)

    # Determine metadata file if not provided
    if metadata_file is None:
        metadata_file = os.path.splitext(vtu_file)[0] + ".json"

    # Read metadata if available
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            metadata = read_metadata(metadata_file)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid metadata file {metadata_file}: {e}")
    else:
        logger.warning(f"Metadata file not found: {metadata_file}")

    return mesh, cell_data, point_data, metadata


def read_vtp_with_metadata(
    vtp_file: Union[str, Path], metadata_file: Optional[Union[str, Path]] = None
) -> Tuple[Any, Dict, Dict, Dict]:
    """
    Read a VTP file and its associated metadata.

    Args:
        vtp_file: Path to the VTP file.
        metadata_file: Path to the metadata file. If None, looks for a .json file
                      with the same base name.

    Returns:
        Tuple of (polydata, cell_data, point_data, metadata).

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If any file doesn't exist.
        ValueError: If any file is invalid.
    """
    vtp_file = str(vtp_file)

    if not os.path.exists(vtp_file):
        raise FileNotFoundError(f"VTP file not found: {vtp_file}")

    # Get polydata and data
    polydata, cell_data, point_data = read_vtp(vtp_file)

    # Determine metadata file if not provided
    if metadata_file is None:
        metadata_file = os.path.splitext(vtp_file)[0] + ".json"

    # Read metadata if available
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            metadata = read_metadata(metadata_file)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid metadata file {metadata_file}: {e}")
    else:
        logger.warning(f"Metadata file not found: {metadata_file}")

    return polydata, cell_data, point_data, metadata


def read_timestep(
    base_dir: Union[str, Path],
    timestep: Union[int, str],
    file_pattern: str = "{timestep}_{type}.{ext}",
) -> Dict[str, Any]:
    """
    Read all files for a specific timestep.

    Args:
        base_dir: Base directory containing the files.
        timestep: Timestep identifier (number or string).
        file_pattern: Pattern for constructing filenames.
                     Should contain placeholders for {timestep}, {type}, and {ext}.

    Returns:
        Dictionary with keys 'vtu', 'vtp', and 'metadata' containing the data.

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If required files don't exist.
        ValueError: If files are invalid.
    """
    base_dir = Path(base_dir)

    # Prepare result dictionary
    result = {"vtu": None, "vtp": None, "metadata": None, "vtu_data": None, "vtp_data": None}

    # Construct file paths
    vtu_path = base_dir / file_pattern.format(timestep=timestep, type="mesh", ext="vtu")
    vtp_path = base_dir / file_pattern.format(timestep=timestep, type="surface", ext="vtp")
    meta_path = base_dir / file_pattern.format(timestep=timestep, type="meta", ext="json")

    # Read VTU file if it exists
    if vtu_path.exists():
        mesh, cell_data, point_data = read_vtu(vtu_path)
        result["vtu"] = mesh
        result["vtu_data"] = {"cell_data": cell_data, "point_data": point_data}

    # Read VTP file if it exists
    if vtp_path.exists():
        polydata, cell_data, point_data = read_vtp(vtp_path)
        result["vtp"] = polydata
        result["vtp_data"] = {"cell_data": cell_data, "point_data": point_data}

    # Read metadata if it exists
    if meta_path.exists():
        result["metadata"] = read_metadata(meta_path)

    # Ensure at least one file was read
    if all(v is None for k, v in result.items()):
        raise FileNotFoundError(f"No files found for timestep {timestep} in {base_dir}")

    return result


def write_metadata(metadata: Dict, file_path: Union[str, Path], indent: int = 4) -> None:
    """
    Write metadata to a JSON file.

    Args:
        metadata: Dictionary containing the metadata.
        file_path: Path to the output file.
        indent: Indentation level for the JSON file.

    Raises:
        IOError: If the file cannot be written.
    """
    file_path = Path(file_path)

    # Create parent directory if it doesn't exist
    os.makedirs(file_path.parent, exist_ok=True)

    try:
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=indent)
    except IOError as e:
        logger.error(f"Error writing metadata to {file_path}: {e}")
        raise
