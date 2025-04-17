"""
VTK utility functions for handling vascular simulation data.

This module provides utility functions for reading and processing VTK data
formats, including VTU and VTP files commonly used in vascular simulations.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# VTK imports
try:
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    HAS_VTK = True
except ImportError:
    HAS_VTK = False


logger = logging.getLogger(__name__)


def check_vtk_availability() -> bool:
    """
    Check if VTK is available for use.

    Returns:
        Whether VTK is available.
    """
    return HAS_VTK


def extract_mesh_from_vtu(
    vtu_file: str,
) -> Tuple[
    Optional["vtk.vtkUnstructuredGrid"],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
]:
    """
    Extract mesh and data from a VTU file.

    Args:
        vtu_file: Path to the VTU file.

    Returns:
        Tuple of (mesh, cell_data, point_data).

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If the VTU file doesn't exist.
        ValueError: If the VTU file is invalid.
    """
    if not HAS_VTK:
        raise ImportError("VTK is required for VTU extraction. Install it with 'pip install vtk'.")

    if not os.path.exists(vtu_file):
        raise FileNotFoundError(f"VTU file not found: {vtu_file}")

    # Create reader
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file)
    reader.Update()

    # Get mesh
    mesh = reader.GetOutput()

    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise ValueError(f"Invalid VTU file: {vtu_file}")

    # Extract cell data
    cell_data = {}
    for i in range(mesh.GetCellData().GetNumberOfArrays()):
        array_name = mesh.GetCellData().GetArrayName(i)
        array = mesh.GetCellData().GetArray(i)
        cell_data[array_name] = vtk_to_numpy(array)

    # Extract point data
    point_data = {}
    for i in range(mesh.GetPointData().GetNumberOfArrays()):
        array_name = mesh.GetPointData().GetArrayName(i)
        array = mesh.GetPointData().GetArray(i)
        point_data[array_name] = vtk_to_numpy(array)

    return mesh, cell_data, point_data


def extract_points_from_vtp(
    vtp_file: str,
) -> Tuple[
    Optional["vtk.vtkPolyData"], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]
]:
    """
    Extract polydata and data from a VTP file.

    Args:
        vtp_file: Path to the VTP file.

    Returns:
        Tuple of (polydata, cell_data, point_data).

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If the VTP file doesn't exist.
        ValueError: If the VTP file is invalid.
    """
    if not HAS_VTK:
        raise ImportError("VTK is required for VTP extraction. Install it with 'pip install vtk'.")

    if not os.path.exists(vtp_file):
        raise FileNotFoundError(f"VTP file not found: {vtp_file}")

    # Create reader
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()

    # Get polydata
    polydata = reader.GetOutput()

    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"Invalid VTP file: {vtp_file}")

    # Extract cell data
    cell_data = {}
    for i in range(polydata.GetCellData().GetNumberOfArrays()):
        array_name = polydata.GetCellData().GetArrayName(i)
        array = polydata.GetCellData().GetArray(i)
        cell_data[array_name] = vtk_to_numpy(array)

    # Extract point data
    point_data = {}
    for i in range(polydata.GetPointData().GetNumberOfArrays()):
        array_name = polydata.GetPointData().GetArrayName(i)
        array = polydata.GetPointData().GetArray(i)
        point_data[array_name] = vtk_to_numpy(array)

    return polydata, cell_data, point_data


def extract_attributes(
    vtk_data_object: Union["vtk.vtkUnstructuredGrid", "vtk.vtkPolyData"],
    attribute_type: str = "point",
    attributes: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract attributes from a VTK data object.

    Args:
        vtk_data_object: VTK data object to extract from.
        attribute_type: Type of attribute ('point' or 'cell').
        attributes: List of attributes to extract. If None, extract all.

    Returns:
        Dictionary of attribute names and values.

    Raises:
        ValueError: If attribute_type is invalid.
        TypeError: If vtk_data_object is of wrong type.
    """
    if not HAS_VTK:
        raise ImportError(
            "VTK is required for attribute extraction. Install it with 'pip install vtk'."
        )

    # Validate inputs
    if attribute_type not in ["point", "cell"]:
        raise ValueError(f"Invalid attribute type: {attribute_type}. Must be 'point' or 'cell'.")

    if not isinstance(vtk_data_object, (vtk.vtkUnstructuredGrid, vtk.vtkPolyData)):
        raise TypeError("vtk_data_object must be vtkUnstructuredGrid or vtkPolyData")

    # Get the appropriate data
    if attribute_type == "point":
        data = vtk_data_object.GetPointData()
    else:  # cell
        data = vtk_data_object.GetCellData()

    # Extract attributes
    result = {}
    for i in range(data.GetNumberOfArrays()):
        array_name = data.GetArrayName(i)

        if attributes is None or array_name in attributes:
            array = data.GetArray(i)
            result[array_name] = vtk_to_numpy(array)

    return result


def convert_vtk_to_numpy(vtk_array: "vtk.vtkDataArray") -> np.ndarray:
    """
    Convert a VTK data array to a NumPy array.

    Args:
        vtk_array: VTK data array to convert.

    Returns:
        NumPy array.

    Raises:
        ImportError: If VTK is not available.
        TypeError: If vtk_array is of wrong type.
    """
    if not HAS_VTK:
        raise ImportError(
            "VTK is required for array conversion. Install it with 'pip install vtk'."
        )

    if not isinstance(vtk_array, vtk.vtkDataArray):
        raise TypeError("vtk_array must be vtkDataArray")

    return vtk_to_numpy(vtk_array)


def save_as_vtu(
    points: np.ndarray,
    cells: List[Tuple[int, List[int]]],
    point_data: Optional[Dict[str, np.ndarray]] = None,
    cell_data: Optional[Dict[str, np.ndarray]] = None,
    output_file: str = "output.vtu",
) -> bool:
    """
    Save data as a VTU file.

    Args:
        points: Array of point coordinates of shape (num_points, 3).
        cells: List of (cell_type, [point_ids]) tuples.
        point_data: Dictionary of point data arrays.
        cell_data: Dictionary of cell data arrays.
        output_file: Path to the output file.

    Returns:
        Whether the save was successful.

    Raises:
        ImportError: If VTK is not available.
        ValueError: If inputs are invalid.
    """
    if not HAS_VTK:
        raise ImportError(
            "VTK is required for saving VTU files. Install it with 'pip install vtk'."
        )

    # Validate inputs
    if points.ndim != 2 or points.shape[1] not in [2, 3]:
        raise ValueError("points must have shape (num_points, 2) or (num_points, 3)")

    # Create points
    vtk_points = vtk.vtkPoints()

    # If 2D, add third coordinate as 0
    if points.shape[1] == 2:
        points_3d = np.zeros((points.shape[0], 3), dtype=points.dtype)
        points_3d[:, :2] = points
        vtk_points.SetData(numpy_to_vtk(points_3d))
    else:
        vtk_points.SetData(numpy_to_vtk(points))

    # Create cells
    vtk_cells = vtk.vtkCellArray()

    for cell_type, point_ids in cells:
        cell = vtk.vtkIdList()
        for point_id in point_ids:
            cell.InsertNextId(point_id)

        vtk_cells.InsertNextCell(cell)

    # Create grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)

    # Set cells
    for i, (cell_type, _) in enumerate(cells):
        grid.InsertNextCell(cell_type, vtk_cells.GetCell(i))

    # Add point data
    if point_data is not None:
        for name, data in point_data.items():
            vtk_data = numpy_to_vtk(data)
            vtk_data.SetName(name)
            grid.GetPointData().AddArray(vtk_data)

    # Add cell data
    if cell_data is not None:
        for name, data in cell_data.items():
            vtk_data = numpy_to_vtk(data)
            vtk_data.SetName(name)
            grid.GetCellData().AddArray(vtk_data)

    # Write file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(grid)

    try:
        return writer.Write() == 1
    except Exception as e:
        logger.error(f"Error saving VTU file: {e}")
        return False


def save_as_vtp(
    points: np.ndarray,
    polygons: Optional[List[List[int]]] = None,
    lines: Optional[List[List[int]]] = None,
    point_data: Optional[Dict[str, np.ndarray]] = None,
    cell_data: Optional[Dict[str, np.ndarray]] = None,
    output_file: str = "output.vtp",
) -> bool:
    """
    Save data as a VTP file.

    Args:
        points: Array of point coordinates of shape (num_points, 3).
        polygons: List of lists containing point indices for polygons.
        lines: List of lists containing point indices for lines.
        point_data: Dictionary of point data arrays.
        cell_data: Dictionary of cell data arrays.
        output_file: Path to the output file.

    Returns:
        Whether the save was successful.

    Raises:
        ImportError: If VTK is not available.
        ValueError: If inputs are invalid.
    """
    if not HAS_VTK:
        raise ImportError(
            "VTK is required for saving VTP files. Install it with 'pip install vtk'."
        )

    # Validate inputs
    if points.ndim != 2 or points.shape[1] not in [2, 3]:
        raise ValueError("points must have shape (num_points, 2) or (num_points, 3)")

    if polygons is None and lines is None:
        raise ValueError("At least one of polygons or lines must be provided")

    # Create polydata
    polydata = vtk.vtkPolyData()

    # Add points
    vtk_points = vtk.vtkPoints()

    # If 2D, add third coordinate as 0
    if points.shape[1] == 2:
        points_3d = np.zeros((points.shape[0], 3), dtype=points.dtype)
        points_3d[:, :2] = points
        vtk_points.SetData(numpy_to_vtk(points_3d))
    else:
        vtk_points.SetData(numpy_to_vtk(points))

    polydata.SetPoints(vtk_points)

    # Add polygons if provided
    if polygons is not None and len(polygons) > 0:
        vtk_polygons = vtk.vtkCellArray()

        for polygon in polygons:
            vtk_polygon = vtk.vtkPolygon()
            vtk_polygon.GetPointIds().SetNumberOfIds(len(polygon))

            for i, point_id in enumerate(polygon):
                vtk_polygon.GetPointIds().SetId(i, point_id)

            vtk_polygons.InsertNextCell(vtk_polygon)

        polydata.SetPolys(vtk_polygons)

    # Add lines if provided
    if lines is not None and len(lines) > 0:
        vtk_lines = vtk.vtkCellArray()

        for line in lines:
            vtk_line = vtk.vtkLine()
            vtk_line.GetPointIds().SetNumberOfIds(len(line))

            for i, point_id in enumerate(line):
                vtk_line.GetPointIds().SetId(i, point_id)

            vtk_lines.InsertNextCell(vtk_line)

        polydata.SetLines(vtk_lines)

    # Add point data
    if point_data is not None:
        for name, data in point_data.items():
            vtk_data = numpy_to_vtk(data)
            vtk_data.SetName(name)
            polydata.GetPointData().AddArray(vtk_data)

    # Add cell data
    if cell_data is not None:
        for name, data in cell_data.items():
            vtk_data = numpy_to_vtk(data)
            vtk_data.SetName(name)
            polydata.GetCellData().AddArray(vtk_data)

    # Write file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)

    try:
        return writer.Write() == 1
    except Exception as e:
        logger.error(f"Error saving VTP file: {e}")
        return False


def merge_vtu_files(input_files: List[str], output_file: str, append_file_id: bool = True) -> bool:
    """
    Merge multiple VTU files into a single file.

    Args:
        input_files: List of input VTU files.
        output_file: Path to the output file.
        append_file_id: Whether to append file ID to attributes.

    Returns:
        Whether the merge was successful.

    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If any input file doesn't exist.
        ValueError: If no valid input files are provided.
    """
    if not HAS_VTK:
        raise ImportError(
            "VTK is required for merging VTU files. Install it with 'pip install vtk'."
        )

    # Validate inputs
    if not input_files:
        raise ValueError("No input files provided")

    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

    # Create merge filter
    append_filter = vtk.vtkAppendFilter()

    # Add each input file
    for i, file_path in enumerate(input_files):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_path)
        reader.Update()

        grid = reader.GetOutput()

        if grid is None or grid.GetNumberOfPoints() == 0:
            logger.warning(f"Skipping invalid file: {file_path}")
            continue

        # Optionally add file ID to attributes
        if append_file_id:
            file_id = f"file_{i}"
            id_array = vtk.vtkIntArray()
            id_array.SetName("FileID")
            id_array.SetNumberOfComponents(1)
            id_array.SetNumberOfTuples(grid.GetNumberOfCells())
            id_array.Fill(i)
            grid.GetCellData().AddArray(id_array)

        append_filter.AddInputData(grid)

    # Execute the merge
    append_filter.MergePointsOn()
    append_filter.Update()

    merged_grid = append_filter.GetOutput()

    if merged_grid is None or merged_grid.GetNumberOfPoints() == 0:
        logger.error("Failed to merge VTU files")
        return False

    # Write merged file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(merged_grid)

    try:
        return writer.Write() == 1
    except Exception as e:
        logger.error(f"Error saving merged VTU file: {e}")
        return False
