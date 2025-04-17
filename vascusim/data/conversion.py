"""
Conversion utilities for vascular simulation data.

This module provides functions to convert VTU/VTP files to PyTorch Geometric
data format, with support for various attributes and mesh properties.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
from torch_geometric.data import Data

# VTK imports
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

# Import internal modules
from ..io.vtk_utils import (
    extract_mesh_from_vtu,
    extract_points_from_vtp,
    extract_attributes,
    convert_vtk_to_numpy
)


logger = logging.getLogger(__name__)


def vtu_to_pyg(
    vtu_file: str,
    attributes: Optional[List[str]] = None,
    include_cell_data: bool = True,
    include_point_data: bool = True,
    include_mesh: bool = True,
    normalize: bool = False,
) -> Data:
    """
    Convert a VTU file to PyTorch Geometric data format.
    
    Args:
        vtu_file: Path to the VTU file.
        attributes: List of specific attributes to include.
        include_cell_data: Whether to include cell data.
        include_point_data: Whether to include point data.
        include_mesh: Whether to include mesh connectivity.
        normalize: Whether to normalize node positions.
        
    Returns:
        PyTorch Geometric Data object.
        
    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If the VTU file doesn't exist.
        ValueError: If the VTU file is invalid.
    """
    if not HAS_VTK:
        raise ImportError("VTK is required for VTU conversion. Install it with 'pip install vtk'.")
    
    if not os.path.exists(vtu_file):
        raise FileNotFoundError(f"VTU file not found: {vtu_file}")
    
    # Extract mesh and data from VTU file
    mesh, cell_data, point_data = extract_mesh_from_vtu(vtu_file)
    
    if mesh is None:
        raise ValueError(f"Failed to extract mesh from VTU file: {vtu_file}")
    
    # Get node positions
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    points = torch.from_numpy(points).float()
    
    # Normalize positions if requested
    if normalize:
        points_min = points.min(dim=0)[0]
        points_max = points.max(dim=0)[0]
        points = (points - points_min) / (points_max - points_min)
    
    # Create edge connectivity if requested
    edge_index = None
    if include_mesh:
        edge_index = _vtu_to_edge_index(mesh)
    
    # Create PyG data object
    data = Data(pos=points)
    
    if edge_index is not None:
        data.edge_index = edge_index
    
    # Include cell data if requested
    if include_cell_data and cell_data is not None:
        for name, array in cell_data.items():
            if attributes is None or name in attributes:
                if array.ndim == 1:
                    data[f"cell_{name}"] = torch.from_numpy(array).float()
                else:
                    # Handle multi-dimensional arrays
                    for i in range(array.shape[1]):
                        data[f"cell_{name}_{i}"] = torch.from_numpy(array[:, i]).float()
    
    # Include point data if requested
    if include_point_data and point_data is not None:
        for name, array in point_data.items():
            if attributes is None or name in attributes:
                if array.ndim == 1:
                    data[f"node_{name}"] = torch.from_numpy(array).float()
                else:
                    # Handle multi-dimensional arrays
                    for i in range(array.shape[1]):
                        data[f"node_{name}_{i}"] = torch.from_numpy(array[:, i]).float()
    
    return data


def vtp_to_pyg(
    vtp_file: str,
    attributes: Optional[List[str]] = None,
    include_cell_data: bool = True,
    include_point_data: bool = True,
    include_mesh: bool = True,
    normalize: bool = False,
) -> Data:
    """
    Convert a VTP file to PyTorch Geometric data format.
    
    Args:
        vtp_file: Path to the VTP file.
        attributes: List of specific attributes to include.
        include_cell_data: Whether to include cell data.
        include_point_data: Whether to include point data.
        include_mesh: Whether to include mesh connectivity.
        normalize: Whether to normalize node positions.
        
    Returns:
        PyTorch Geometric Data object.
        
    Raises:
        ImportError: If VTK is not available.
        FileNotFoundError: If the VTP file doesn't exist.
        ValueError: If the VTP file is invalid.
    """
    if not HAS_VTK:
        raise ImportError("VTK is required for VTP conversion. Install it with 'pip install vtk'.")
    
    if not os.path.exists(vtp_file):
        raise FileNotFoundError(f"VTP file not found: {vtp_file}")
    
    # Extract polydata and data from VTP file
    polydata, cell_data, point_data = extract_points_from_vtp(vtp_file)
    
    if polydata is None:
        raise ValueError(f"Failed to extract polydata from VTP file: {vtp_file}")
    
    # Get node positions
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    points = torch.from_numpy(points).float()
    
    # Normalize positions if requested
    if normalize:
        points_min = points.min(dim=0)[0]
        points_max = points.max(dim=0)[0]
        points = (points - points_min) / (points_max - points_min)
    
    # Create edge connectivity if requested
    edge_index = None
    if include_mesh:
        edge_index = _vtp_to_edge_index(polydata)
    
    # Create PyG data object
    data = Data(pos=points)
    
    if edge_index is not None:
        data.edge_index = edge_index
    
    # Include cell data if requested
    if include_cell_data and cell_data is not None:
        for name, array in cell_data.items():
            if attributes is None or name in attributes:
                if array.ndim == 1:
                    data[f"cell_{name}"] = torch.from_numpy(array).float()
                else:
                    # Handle multi-dimensional arrays
                    for i in range(array.shape[1]):
                        data[f"cell_{name}_{i}"] = torch.from_numpy(array[:, i]).float()
    
    # Include point data if requested
    if include_point_data and point_data is not None:
        for name, array in point_data.items():
            if attributes is None or name in attributes:
                if array.ndim == 1:
                    data[f"node_{name}"] = torch.from_numpy(array).float()
                else:
                    # Handle multi-dimensional arrays
                    for i in range(array.shape[1]):
                        data[f"node_{name}_{i}"] = torch.from_numpy(array[:, i]).float()
    
    return data


def _vtu_to_edge_index(mesh: 'vtk.vtkUnstructuredGrid') -> torch.Tensor:
    """
    Convert VTK unstructured grid to PyTorch Geometric edge_index.
    
    Args:
        mesh: VTK unstructured grid.
        
    Returns:
        Edge index tensor of shape [2, num_edges].
    """
    num_cells = mesh.GetNumberOfCells()
    edge_list = []
    
    for i in range(num_cells):
        cell = mesh.GetCell(i)
        
        # Get number of points in the cell
        num_points = cell.GetNumberOfPoints()
        
        if num_points > 1:
            # For each cell, create edges between consecutive points
            for j in range(num_points):
                p1 = cell.GetPointId(j)
                p2 = cell.GetPointId((j + 1) % num_points)
                edge_list.append([p1, p2])
    
    # Remove duplicate edges
    edge_set = set(tuple(sorted(edge)) for edge in edge_list)
    edge_list = [list(edge) for edge in edge_set]
    
    # Convert to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return edge_index


def _vtp_to_edge_index(polydata: 'vtk.vtkPolyData') -> torch.Tensor:
    """
    Convert VTK polydata to PyTorch Geometric edge_index.
    
    Args:
        polydata: VTK polydata.
        
    Returns:
        Edge index tensor of shape [2, num_edges].
    """
    edges = []
    
    # Extract edges from polygons
    polygons = polydata.GetPolys()
    if polygons.GetNumberOfCells() > 0:
        polygons.InitTraversal()
        idList = vtk.vtkIdList()
        
        while polygons.GetNextCell(idList):
            num_points = idList.GetNumberOfIds()
            
            if num_points > 1:
                for j in range(num_points):
                    p1 = idList.GetId(j)
                    p2 = idList.GetId((j + 1) % num_points)
                    edges.append((min(p1, p2), max(p1, p2)))
    
    # Extract edges from lines
    lines = polydata.GetLines()
    if lines.GetNumberOfCells() > 0:
        lines.InitTraversal()
        idList = vtk.vtkIdList()
        
        while lines.GetNextCell(idList):
            num_points = idList.GetNumberOfIds()
            
            if num_points > 1:
                for j in range(num_points - 1):
                    p1 = idList.GetId(j)
                    p2 = idList.GetId(j + 1)
                    edges.append((min(p1, p2), max(p1, p2)))
    
    # Remove duplicate edges
    unique_edges = list(set(edges))
    
    # Convert to tensor
    edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
    
    return edge_index


def build_graph(
    nodes: Union[np.ndarray, torch.Tensor],
    edges: Optional[Union[np.ndarray, torch.Tensor]] = None,
    node_features: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
    edge_features: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
    global_features: Optional[Dict[str, Union[np.ndarray, torch.Tensor, float, int]]] = None,
    edge_builder: Optional[Callable] = None,
) -> Data:
    """
    Build a PyTorch Geometric graph from nodes and edges.
    
    Args:
        nodes: Node positions of shape [num_nodes, dim].
        edges: Edge connections of shape [2, num_edges] or [num_edges, 2].
        node_features: Dictionary of node features.
        edge_features: Dictionary of edge features.
        global_features: Dictionary of global features.
        edge_builder: Function to build edges if not provided.
        
    Returns:
        PyTorch Geometric Data object.
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(nodes, np.ndarray):
        nodes = torch.from_numpy(nodes).float()
    
    # Create Data object with node positions
    data = Data(pos=nodes)
    
    # Add edge connections
    if edges is not None:
        if isinstance(edges, np.ndarray):
            edges = torch.from_numpy(edges).long()
        
        # Ensure edge_index has shape [2, num_edges]
        if edges.shape[0] == 2:
            data.edge_index = edges
        else:
            data.edge_index = edges.t().contiguous()
    elif edge_builder is not None:
        # Build edges using the provided function
        data.edge_index = edge_builder(nodes)
    
    # Add node features
    if node_features is not None:
        for name, features in node_features.items():
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            data[name] = features
    
    # Add edge features
    if edge_features is not None:
        for name, features in edge_features.items():
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            data[name] = features
    
    # Add global features
    if global_features is not None:
        for name, value in global_features.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            elif isinstance(value, (int, float)):
                value = torch.tensor([value], dtype=torch.float)
            data[name] = value
    
    return data