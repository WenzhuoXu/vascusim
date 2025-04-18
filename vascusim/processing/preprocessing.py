"""
Preprocessing functions for vascular simulation data.

This module provides utilities for preprocessing vascular simulation data,
including normalization, resampling, filtering, and feature computation.
"""

import logging
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union

# Try to import VTK (optional dependency)
try:
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

# Try to import PyVista (optional dependency)
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# Import local modules
from .geometry import compute_curvature

logger = logging.getLogger(__name__)


def normalize_geometry(
    data: Data, 
    preserve_aspect_ratio: bool = True, 
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> Data:
    """
    Normalize the geometry to a specific range.
    
    Args:
        data: PyTorch Geometric Data object.
        preserve_aspect_ratio: Whether to preserve aspect ratio during normalization.
        target_range: Target range for normalized coordinates.
        
    Returns:
        Normalized PyTorch Geometric Data object.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
    
    # Create a copy of the data
    normalized_data = data.clone()
    
    # Extract node positions
    pos = normalized_data.pos
    
    # Compute min and max coordinates
    min_coords = torch.min(pos, dim=0)[0]
    max_coords = torch.max(pos, dim=0)[0]
    
    # Compute scale factors
    range_size = max_coords - min_coords
    
    if preserve_aspect_ratio:
        # Use the same scale factor for all dimensions
        scale_factor = torch.max(range_size)
        
        # Avoid division by zero
        if scale_factor <= 0:
            scale_factor = 1.0
    else:
        # Use different scale factors for each dimension
        scale_factor = range_size.clone()
        
        # Avoid division by zero
        scale_factor[scale_factor <= 0] = 1.0
    
    # Normalize to [0, 1] range
    normalized_pos = (pos - min_coords) / scale_factor
    
    # Scale to target range
    target_min, target_max = target_range
    normalized_pos = normalized_pos * (target_max - target_min) + target_min
    
    # Update positions
    normalized_data.pos = normalized_pos
    
    return normalized_data


def resample_geometry(
    data: Data, 
    target_points: Optional[int] = None,
    target_density: Optional[float] = None,
    resample_method: str = "uniform"
) -> Data:
    """
    Resample the geometry to a target number of points or density.
    
    Args:
        data: PyTorch Geometric Data object.
        target_points: Target number of points after resampling.
        target_density: Target point density (points per unit length).
        resample_method: Resampling method ("uniform", "curvature", or "importance").
        
    Returns:
        Resampled PyTorch Geometric Data object.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # Check parameters
    if target_points is None and target_density is None:
        raise ValueError("Either target_points or target_density must be specified")
    
    # Use VTK/PyVista if available for more advanced resampling
    if (HAS_VTK or HAS_PYVISTA) and resample_method != "uniform":
        try:
            return _resample_geometry_vtk(data, target_points, target_density, resample_method)
        except Exception as e:
            logger.warning(f"VTK/PyVista resampling failed: {e}. Falling back to basic method.")
    
    # Basic resampling implementation
    # Extract node positions and connectivity
    pos = data.pos
    edge_index = data.edge_index
    
    # Compute current number of points
    num_points = pos.shape[0]
    
    # Compute target number of points if density was specified
    if target_points is None:
        # Compute geometry size
        bbox_size = torch.max(pos, dim=0)[0] - torch.min(pos, dim=0)[0]
        geometry_size = torch.norm(bbox_size)
        
        # Compute target number of points based on density
        target_points = int(geometry_size * target_density)
        
        # Ensure reasonable number of points
        target_points = max(10, min(10000, target_points))
    
    # If target is close to current, return original
    if 0.95 * num_points <= target_points <= 1.05 * num_points:
        return data.clone()
    
    # Prepare resampled data
    if target_points < num_points:
        # Downsample
        return _downsample_geometry(data, target_points, resample_method)
    else:
        # Upsample
        return _upsample_geometry(data, target_points, resample_method)


def _downsample_geometry(data: Data, target_points: int, method: str = "uniform") -> Data:
    """
    Downsample geometry to target number of points.
    
    Args:
        data: PyTorch Geometric Data object.
        target_points: Target number of points.
        method: Downsampling method.
        
    Returns:
        Downsampled Data object.
    """
    pos = data.pos
    edge_index = data.edge_index
    num_points = pos.shape[0]
    
    # Create importance metric based on method
    if method == "curvature":
        # Use curvature as importance metric
        importance = compute_curvature(data)
    elif method == "importance":
        # Use a combination of factors as importance metric
        importance = _compute_importance_metric(data)
    else:  # uniform
        # Uniform importance
        importance = torch.ones(num_points, device=pos.device)
    
    # Convert to probabilities
    if torch.sum(importance) > 0:
        probs = importance / torch.sum(importance)
    else:
        probs = torch.ones(num_points, device=pos.device) / num_points
    
    # Sample indices based on importance
    indices = torch.multinomial(probs, target_points, replacement=False)
    indices, _ = torch.sort(indices)  # Sort for consistency
    
    # Create node mapping from old to new indices
    node_map = -torch.ones(num_points, dtype=torch.long, device=pos.device)
    node_map[indices] = torch.arange(target_points, device=pos.device)
    
    # Create downsampled data object
    downsampled = Data(pos=pos[indices])
    
    # Remap edges
    mask = (node_map[edge_index[0]] >= 0) & (node_map[edge_index[1]] >= 0)
    new_edge_index = node_map[edge_index[:, mask]]
    
    # Handle potential duplicate edges
    edges_set = set()
    new_edges = []
    
    for i in range(new_edge_index.shape[1]):
        edge = (new_edge_index[0, i].item(), new_edge_index[1, i].item())
        if edge[0] != edge[1] and edge not in edges_set:
            edges_set.add(edge)
            edges_set.add((edge[1], edge[0]))  # Add reverse edge
            new_edges.append(edge)
    
    if new_edges:
        downsampled.edge_index = torch.tensor(new_edges, dtype=torch.long, device=pos.device).t()
    else:
        # Fallback: create minimum spanning tree
        downsampled.edge_index = _create_minimum_spanning_tree(downsampled.pos)
    
    # Transfer other node attributes
    for key, value in data:
        if key not in ['pos', 'edge_index'] and isinstance(value, torch.Tensor):
            if value.size(0) == num_points:
                downsampled[key] = value[indices]
    
    return downsampled


def _upsample_geometry(data: Data, target_points: int, method: str = "uniform") -> Data:
    """
    Upsample geometry to target number of points.
    
    Args:
        data: PyTorch Geometric Data object.
        target_points: Target number of points.
        method: Upsampling method.
        
    Returns:
        Upsampled Data object.
    """
    pos = data.pos
    edge_index = data.edge_index
    num_points = pos.shape[0]
    
    # Number of new points to add
    num_new_points = target_points - num_points
    
    # Create a copy of the original data
    upsampled = data.clone()
    
    # Sort edges by importance for adding new points
    if method == "curvature":
        # Use curvature as importance metric
        curvature = compute_curvature(data)
        edge_importance = torch.zeros(edge_index.shape[1], device=pos.device)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_importance[i] = (curvature[src] + curvature[dst]) / 2
    elif method == "importance":
        # Use edge length as importance metric
        edge_importance = torch.zeros(edge_index.shape[1], device=pos.device)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_importance[i] = torch.norm(pos[dst] - pos[src])
    else:  # uniform
        # Uniform importance based on edge length
        edge_importance = torch.zeros(edge_index.shape[1], device=pos.device)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_importance[i] = torch.norm(pos[dst] - pos[src])
    
    # Sort edges by importance
    sorted_indices = torch.argsort(edge_importance, descending=True)
    
    # Add new points by subdividing edges
    new_positions = []
    new_attributes = {}
    
    for key, value in data:
        if key not in ['pos', 'edge_index'] and isinstance(value, torch.Tensor):
            if value.size(0) == num_points:
                new_attributes[key] = []
    
    # Add new points evenly along edges, prioritizing important edges
    points_per_edge = num_new_points // len(sorted_indices) + 1
    remaining_points = num_new_points
    
    for idx in sorted_indices:
        src, dst = edge_index[0, idx], edge_index[1, idx]
        src_pos, dst_pos = pos[src], pos[dst]
        
        # Determine how many points to add to this edge
        points_to_add = min(points_per_edge, remaining_points)
        if points_to_add <= 0:
            break
        
        # Create new points along the edge
        for i in range(1, points_to_add + 1):
            t = i / (points_to_add + 1)
            new_pos = src_pos * (1 - t) + dst_pos * t
            new_positions.append(new_pos)
            
            # Interpolate other attributes
            for key, value in new_attributes.items():
                src_val = data[key][src]
                dst_val = data[key][dst]
                new_val = src_val * (1 - t) + dst_val * t
                value.append(new_val)
        
        remaining_points -= points_to_add
        if remaining_points <= 0:
            break
    
    # Combine original and new points
    upsampled.pos = torch.cat([pos, torch.stack(new_positions)], dim=0)
    
    # Update other attributes
    for key, value in new_attributes.items():
        if value:
            upsampled[key] = torch.cat([data[key], torch.stack(value)], dim=0)
    
    # Create new edge connectivity
    upsampled.edge_index = _create_minimum_spanning_tree(upsampled.pos)
    
    return upsampled


def _compute_importance_metric(data: Data) -> torch.Tensor:
    """
    Compute importance metric for nodes based on multiple factors.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Importance metric tensor.
    """
    num_nodes = data.pos.shape[0]
    importance = torch.ones(num_nodes, device=data.pos.device)
    
    # Add curvature factor
    try:
        curvature = compute_curvature(data)
        # Normalize to [0, 1]
        if torch.max(curvature) > torch.min(curvature):
            curvature = (curvature - torch.min(curvature)) / (torch.max(curvature) - torch.min(curvature))
        importance = importance * (1.0 + 2.0 * curvature)  # Weight curvature higher
    except Exception as e:
        logger.warning(f"Failed to compute curvature for importance: {e}")
    
    # Add degree factor
    try:
        row, col = data.edge_index
        degrees = torch.bincount(row, minlength=num_nodes).float()
        # Normalize to [0, 1]
        if torch.max(degrees) > 0:
            degrees = degrees / torch.max(degrees)
        importance = importance * (1.0 + degrees)
    except Exception as e:
        logger.warning(f"Failed to compute degree for importance: {e}")
    
    # Add additional factors if specific attributes exist
    if hasattr(data, "radius") and data.radius is not None:
        radius = data.radius
        if isinstance(radius, torch.Tensor):
            if radius.dim() == 2 and radius.shape[1] == 1:
                radius = radius.squeeze(1)
            if radius.shape[0] == num_nodes:
                # Normalize to [0, 1]
                if torch.max(radius) > torch.min(radius):
                    norm_radius = (radius - torch.min(radius)) / (torch.max(radius) - torch.min(radius))
                else:
                    norm_radius = torch.zeros_like(radius)
                importance = importance * (1.0 + norm_radius)
    
    # Add flow-related factors if they exist
    for attr in ["velocity", "pressure", "flow"]:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            attr_value = getattr(data, attr)
            if isinstance(attr_value, torch.Tensor):
                if attr_value.dim() > 1:
                    # Vector field - use magnitude
                    magnitude = torch.norm(attr_value, dim=1)
                else:
                    # Scalar field
                    magnitude = attr_value
                
                if magnitude.shape[0] == num_nodes:
                    # Normalize to [0, 1]
                    if torch.max(magnitude) > torch.min(magnitude):
                        norm_magnitude = (magnitude - torch.min(magnitude)) / (torch.max(magnitude) - torch.min(magnitude))
                    else:
                        norm_magnitude = torch.zeros_like(magnitude)
                    
                    # Add gradient of the field
                    try:
                        grad_magnitude = _compute_gradient_magnitude(data, magnitude)
                        if torch.max(grad_magnitude) > torch.min(grad_magnitude):
                            norm_grad = (grad_magnitude - torch.min(grad_magnitude)) / (torch.max(grad_magnitude) - torch.min(grad_magnitude))
                            importance = importance * (1.0 + 2.0 * norm_grad)  # Weight gradient higher
                    except Exception:
                        # Fallback to just using the field value
                        importance = importance * (1.0 + norm_magnitude)
    
    return importance


def _compute_gradient_magnitude(data: Data, field: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient magnitude of a scalar field.
    
    Args:
        data: PyTorch Geometric Data object.
        field: Scalar field tensor.
        
    Returns:
        Gradient magnitude tensor.
    """
    num_nodes = data.pos.shape[0]
    pos = data.pos
    edge_index = data.edge_index
    
    # Initialize gradient magnitude
    gradient_magnitude = torch.zeros_like(field)
    
    # Compute gradient for each node
    for i in range(num_nodes):
        # Find neighbors
        neighbors = edge_index[1, edge_index[0] == i]
        
        if len(neighbors) > 0:
            # Compute vectors to neighbors
            vectors = pos[neighbors] - pos[i].unsqueeze(0)
            
            # Compute field differences
            field_diffs = field[neighbors] - field[i]
            
            # Compute directional derivatives
            derivatives = field_diffs / (torch.norm(vectors, dim=1) + 1e-8)
            
            # Gradient magnitude is the maximum directional derivative
            gradient_magnitude[i] = torch.max(torch.abs(derivatives))
    
    return gradient_magnitude


def _create_minimum_spanning_tree(pos: torch.Tensor) -> torch.Tensor:
    """
    Create minimum spanning tree from point positions.
    
    Args:
        pos: Node position tensor.
        
    Returns:
        Edge index tensor for the minimum spanning tree.
    """
    num_nodes = pos.shape[0]
    device = pos.device
    
    # If only a few nodes, create fully connected graph
    if num_nodes <= 10:
        edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edges.append([i, j])
                edges.append([j, i])  # Add both directions
        
        if edges:
            return torch.tensor(edges, dtype=torch.long, device=device).t()
        else:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # Compute pairwise distances
    pos_i = pos.unsqueeze(1)  # [N, 1, D]
    pos_j = pos.unsqueeze(0)  # [1, N, D]
    dist = torch.norm(pos_i - pos_j, dim=2)  # [N, N]
    
    # Initialize MST algorithm (Prim's algorithm)
    in_tree = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    in_tree[0] = True  # Start with node 0
    
    edges = []
    
    # Run MST algorithm
    for _ in range(num_nodes - 1):
        min_dist = float('inf')
        min_edge = (-1, -1)
        
        # Find the closest node not in tree
        for i in range(num_nodes):
            if in_tree[i]:
                for j in range(num_nodes):
                    if not in_tree[j] and dist[i, j] < min_dist:
                        min_dist = dist[i, j]
                        min_edge = (i, j)
        
        # Add the edge to MST
        i, j = min_edge
        edges.append([i, j])
        edges.append([j, i])  # Add both directions
        in_tree[j] = True
    
    return torch.tensor(edges, dtype=torch.long, device=device).t()


def _resample_geometry_vtk(
    data: Data, 
    target_points: Optional[int] = None,
    target_density: Optional[float] = None,
    resample_method: str = "uniform"
) -> Data:
    """
    Resample geometry using VTK.
    
    Args:
        data: PyTorch Geometric Data object.
        target_points: Target number of points.
        target_density: Target point density.
        resample_method: Resampling method.
        
    Returns:
        Resampled Data object.
    """
    if not HAS_VTK and not HAS_PYVISTA:
        raise ImportError("VTK or PyVista is required for this function")
    
    # Convert PyG data to VTK/PyVista format
    if HAS_PYVISTA:
        # Use PyVista for simpler implementation
        return _resample_geometry_pyvista(data, target_points, target_density, resample_method)
    else:
        # Use VTK directly
        return _resample_geometry_vtk_direct(data, target_points, target_density, resample_method)


def _resample_geometry_pyvista(
    data: Data, 
    target_points: Optional[int] = None,
    target_density: Optional[float] = None,
    resample_method: str = "uniform"
) -> Data:
    """
    Resample geometry using PyVista.
    
    Args:
        data: PyTorch Geometric Data object.
        target_points: Target number of points.
        target_density: Target point density.
        resample_method: Resampling method.
        
    Returns:
        Resampled Data object.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required for this function")
    
    # Extract data
    pos = data.pos.detach().cpu().numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    
    # Create lines for PyVista
    lines = []
    for i in range(edge_index.shape[1]):
        lines.append([2, edge_index[0, i], edge_index[1, i]])
    
    # Create PyVista mesh
    mesh = pv.PolyData(pos, lines=lines)
    
    # Compute target number of points if density was specified
    if target_points is None and target_density is not None:
        # Compute length of geometry
        length = 0.0
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            length += np.linalg.norm(pos[dst] - pos[src])
        
        # Divide by 2 because each edge is counted twice
        length /= 2.0
        
        # Compute target number of points
        target_points = int(length * target_density)
        target_points = max(10, min(10000, target_points))
    
    # Apply different resampling methods
    if resample_method == "uniform":
        # Subdivide based on number of points
        resampled = mesh.subdivide(target_points, subfilter="linear")
    elif resample_method == "curvature":
        # Add curvature field for adaptive resampling
        curvature = compute_curvature(data).detach().cpu().numpy()
        mesh.point_data["curvature"] = curvature
        
        # Adaptive subdivision emphasizing high curvature regions
        resampled = mesh.subdivide_adaptive(max_n_points=target_points, progress_bar=False)
    else:  # "importance"
        # Add importance field for adaptive resampling
        importance = _compute_importance_metric(data).detach().cpu().numpy()
        mesh.point_data["importance"] = importance
        
        # Adaptive subdivision based on importance
        resampled = mesh.subdivide_adaptive(max_n_points=target_points, progress_bar=False)
    
    # Convert back to PyG format
    new_pos = torch.tensor(resampled.points, dtype=data.pos.dtype, device=data.pos.device)
    
    # Extract edges
    new_edges = []
    for i in range(resampled.n_cells):
        cell = resampled.get_cell(i)
        cell_points = cell.point_ids
        
        if len(cell_points) == 2:  # Line cell
            new_edges.append([cell_points[0], cell_points[1]])
            new_edges.append([cell_points[1], cell_points[0]])  # Add both directions
    
    # Create new edge_index
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=data.pos.device).t()
    else:
        # Fallback to minimum spanning tree
        new_edge_index = _create_minimum_spanning_tree(new_pos)
    
    # Create new data object
    resampled_data = Data(pos=new_pos, edge_index=new_edge_index)
    
    # Interpolate attributes
    for key, value in data:
        if key not in ['pos', 'edge_index'] and isinstance(value, torch.Tensor):
            if value.size(0) == data.pos.size(0):
                # Point data - needs interpolation
                if value.dim() > 1:
                    # Vector field
                    n_components = value.size(1)
                    for i in range(n_components):
                        attr_name = f"{key}_{i}"
                        mesh.point_data[attr_name] = value[:, i].detach().cpu().numpy()
                else:
                    # Scalar field
                    mesh.point_data[key] = value.detach().cpu().numpy()
    
    # Interpolate attributes from original to resampled mesh
    for key in mesh.point_data:
        if key in resampled.point_data:
            interp_value = resampled.point_data[key]
            
            # Convert back to torch tensor
            if key.rsplit('_', 1)[0] in data and '_' in key:
                # Component of a vector field
                base_name, idx = key.rsplit('_', 1)
                idx = int(idx)
                
                if not hasattr(resampled_data, base_name):
                    orig_value = data[base_name]
                    if orig_value.dim() > 1:
                        n_components = orig_value.size(1)
                        resampled_data[base_name] = torch.zeros(
                            (resampled.n_points, n_components),
                            dtype=orig_value.dtype,
                            device=data.pos.device
                        )
                
                resampled_data[base_name][:, idx] = torch.tensor(
                    interp_value, dtype=data[base_name].dtype, device=data.pos.device
                )
            else:
                # Scalar field
                resampled_data[key] = torch.tensor(
                    interp_value, dtype=data[key].dtype, device=data.pos.device
                )
    
    return resampled_data


def _resample_geometry_vtk_direct(
    data: Data, 
    target_points: Optional[int] = None,
    target_density: Optional[float] = None,
    resample_method: str = "uniform"
) -> Data:
    """
    Resample geometry using VTK directly.
    
    Args:
        data: PyTorch Geometric Data object.
        target_points: Target number of points.
        target_density: Target point density.
        resample_method: Resampling method.
        
    Returns:
        Resampled Data object.
    """
    if not HAS_VTK:
        raise ImportError("VTK is required for this function")
    
    # Extract data
    pos = data.pos.detach().cpu().numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in pos:
        vtk_points.InsertNextPoint(point)
    
    # Create VTK cells (lines)
    vtk_lines = vtk.vtkCellArray()
    for i in range(edge_index.shape[1]):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge_index[0, i])
        line.GetPointIds().SetId(1, edge_index[1, i])
        vtk_lines.InsertNextCell(line)
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(vtk_lines)
    
    # Compute target number of points if density was specified
    if target_points is None and target_density is not None:
        # Compute length of geometry
        length = 0.0
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            length += np.linalg.norm(pos[dst] - pos[src])
        
        # Divide by 2 because each edge is counted twice
        length /= 2.0
        
        # Compute target number of points
        target_points = int(length * target_density)
        target_points = max(10, min(10000, target_points))
    
    # Add attributes to polydata
    point_data = {}
    for key, value in data:
        if key not in ['pos', 'edge_index'] and isinstance(value, torch.Tensor):
            if value.size(0) == data.pos.size(0):
                # Point data - add to polydata
                if value.dim() > 1:
                    # Vector field
                    n_components = value.size(1)
                    for i in range(n_components):
                        attr_name = f"{key}_{i}"
                        array = numpy_to_vtk(value[:, i].detach().cpu().numpy())
                        array.SetName(attr_name)
                        polydata.GetPointData().AddArray(array)
                        point_data[attr_name] = value[:, i]
                else:
                    # Scalar field
                    array = numpy_to_vtk(value.detach().cpu().numpy())
                    array.SetName(key)
                    polydata.GetPointData().AddArray(array)
                    point_data[key] = value
    
    # Apply different resampling methods
    if resample_method == "uniform":
        # Use subdivision filter
        subdivide = vtk.vtkLinearSubdivisionFilter()
        subdivide.SetInputData(polydata)
        subdivide.SetNumberOfSubdivisions(int(np.log2(target_points / polydata.GetNumberOfPoints())) + 1)
        subdivide.Update()
        resampled = subdivide.GetOutput()
    elif resample_method == "curvature" or resample_method == "importance":
        # Add field for adaptive resampling
        if resample_method == "curvature":
            field_values = compute_curvature(data).detach().cpu().numpy()
            field_name = "curvature"
        else:  # "importance"
            field_values = _compute_importance_metric(data).detach().cpu().numpy()
            field_name = "importance"
        
        importance_array = numpy_to_vtk(field_values)
        importance_array.SetName(field_name)
        polydata.GetPointData().AddArray(importance_array)
        
        # Use adaptive subdivision
        adaptive_subdivide = vtk.vtkAdaptiveSubdivisionFilter()
        adaptive_subdivide.SetInputData(polydata)
        adaptive_subdivide.SetMaximumNumberOfPasses(int(np.log2(target_points / polydata.GetNumberOfPoints())) + 1)
        adaptive_subdivide.SetFieldName(field_name)
        adaptive_subdivide.Update()
        resampled = adaptive_subdivide.GetOutput()
    
    # Extract new points and edges
    new_points = vtk_to_numpy(resampled.GetPoints().GetData())
    new_pos = torch.tensor(new_points, dtype=data.pos.dtype, device=data.pos.device)
    
    # Extract new edges
    new_edges = []
    for i in range(resampled.GetNumberOfCells()):
        cell = resampled.GetCell(i)
        if cell.GetNumberOfPoints() == 2:
            new_edges.append([cell.GetPointId(0), cell.GetPointId(1)])
            new_edges.append([cell.GetPointId(1), cell.GetPointId(0)])  # Add both directions
    
    # Create new edge_index
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=data.pos.device).t()
    else:
        # Fallback to minimum spanning tree
        new_edge_index = _create_minimum_spanning_tree(new_pos)
    
    # Create resampled data object
    resampled_data = Data(pos=new_pos, edge_index=new_edge_index)
    
    # Transfer interpolated attributes
    for key in point_data:
        if resampled.GetPointData().HasArray(key):
            interp_value = vtk_to_numpy(resampled.GetPointData().GetArray(key))
            
            # Convert back to torch tensor
            if key.rsplit('_', 1)[0] in data and '_' in key:
                # Component of a vector field
                base_name, idx = key.rsplit('_', 1)
                idx = int(idx)
                
                if not hasattr(resampled_data, base_name):
                    orig_value = data[base_name]
                    if orig_value.dim() > 1:
                        n_components = orig_value.size(1)
                        resampled_data[base_name] = torch.zeros(
                            (resampled.GetNumberOfPoints(), n_components),
                            dtype=orig_value.dtype,
                            device=data.pos.device
                        )
                
                resampled_data[base_name][:, idx] = torch.tensor(
                    interp_value, dtype=data[base_name].dtype, device=data.pos.device
                )
            else:
                # Scalar field
                resampled_data[key] = torch.tensor(
                    interp_value, dtype=point_data[key].dtype, device=data.pos.device
                )
    
    return resampled_data


def filter_noise(
    data: Data, 
    method: str = "gaussian",
    strength: float = 0.5,
    iterations: int = 1
) -> Data:
    """
    Filter noise from vascular geometry data.
    
    Args:
        data: PyTorch Geometric Data object.
        method: Filtering method ("gaussian", "laplacian", or "median").
        strength: Filter strength (0.0 to 1.0).
        iterations: Number of filter iterations.
        
    Returns:
        Filtered PyTorch Geometric Data object.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # Create a copy of the data
    filtered_data = data.clone()
    
    # Adjust strength parameter to ensure stability
    strength = max(0.0, min(1.0, strength))
    iterations = max(1, iterations)
    
    # Apply the selected filter
    if method == "laplacian":
        filtered_data.pos = _apply_laplacian_filter(filtered_data.pos, filtered_data.edge_index, strength, iterations)
    elif method == "median":
        filtered_data.pos = _apply_median_filter(filtered_data.pos, filtered_data.edge_index, iterations)
    else:  # default to gaussian
        filtered_data.pos = _apply_gaussian_filter(filtered_data.pos, filtered_data.edge_index, strength, iterations)
    
    # Also filter other vector fields if requested
    if hasattr(data, "velocity") and isinstance(data.velocity, torch.Tensor):
        filtered_data.velocity = _apply_gaussian_filter(data.velocity, data.edge_index, strength * 0.5, iterations)
    
    return filtered_data


def _apply_gaussian_filter(
    values: torch.Tensor, 
    edge_index: torch.Tensor, 
    strength: float, 
    iterations: int
) -> torch.Tensor:
    """
    Apply Gaussian filter to a tensor.
    
    Args:
        values: Tensor to filter, shape [num_nodes, feature_dim].
        edge_index: Edge connectivity tensor.
        strength: Filter strength.
        iterations: Number of iterations.
        
    Returns:
        Filtered tensor.
    """
    # Create copy to avoid modifying the input
    filtered_values = values.clone()
    
    # Extract the adjacency information
    row, col = edge_index
    
    # Perform multiple iterations
    for _ in range(iterations):
        # Cache the original values for this iteration
        original_values = filtered_values.clone()
        
        # For each node, compute weighted average with neighbors
        for i in range(values.shape[0]):
            # Find neighbors
            neighbors = col[row == i]
            
            if len(neighbors) > 0:
                # Compute Gaussian weights (all equal in simple version)
                weights = torch.ones(len(neighbors), device=values.device)
                weights = weights / weights.sum()
                
                # Compute weighted average
                neighbor_values = original_values[neighbors]
                avg_value = torch.sum(weights.unsqueeze(1) * neighbor_values, dim=0)
                
                # Apply weighted blend between original and filtered
                filtered_values[i] = (1.0 - strength) * original_values[i] + strength * avg_value
    
    return filtered_values


def _apply_laplacian_filter(
    values: torch.Tensor, 
    edge_index: torch.Tensor, 
    strength: float, 
    iterations: int
) -> torch.Tensor:
    """
    Apply Laplacian filter to a tensor.
    
    Args:
        values: Tensor to filter, shape [num_nodes, feature_dim].
        edge_index: Edge connectivity tensor.
        strength: Filter strength.
        iterations: Number of iterations.
        
    Returns:
        Filtered tensor.
    """
    # Create copy to avoid modifying the input
    filtered_values = values.clone()
    
    # Extract the adjacency information
    row, col = edge_index
    
    # Perform multiple iterations
    for _ in range(iterations):
        # Cache the original values for this iteration
        original_values = filtered_values.clone()
        
        # For each node, compute Laplacian update
        for i in range(values.shape[0]):
            # Find neighbors
            neighbors = col[row == i]
            
            if len(neighbors) > 0:
                # Compute neighborhood average
                neighbor_values = original_values[neighbors]
                avg_value = torch.mean(neighbor_values, dim=0)
                
                # Compute Laplacian (difference between node and average)
                laplacian = avg_value - original_values[i]
                
                # Apply Laplacian update
                filtered_values[i] = original_values[i] + strength * laplacian
    
    return filtered_values


def _apply_median_filter(
    values: torch.Tensor, 
    edge_index: torch.Tensor, 
    iterations: int
) -> torch.Tensor:
    """
    Apply median filter to a tensor.
    
    Args:
        values: Tensor to filter, shape [num_nodes, feature_dim].
        edge_index: Edge connectivity tensor.
        iterations: Number of iterations.
        
    Returns:
        Filtered tensor.
    """
    # Create copy to avoid modifying the input
    filtered_values = values.clone()
    
    # Extract the adjacency information
    row, col = edge_index
    
    # Perform multiple iterations
    for _ in range(iterations):
        # Cache the original values for this iteration
        original_values = filtered_values.clone()
        
        # For each node, compute median of neighborhood
        for i in range(values.shape[0]):
            # Find neighbors including the node itself
            neighbors = torch.cat([torch.tensor([i], device=values.device), col[row == i]])
            
            if len(neighbors) > 0:
                # For each dimension, compute the median
                neighbor_values = original_values[neighbors]
                
                if values.dim() > 1:
                    # For vector values, compute median per component
                    for dim in range(values.shape[1]):
                        filtered_values[i, dim] = torch.median(neighbor_values[:, dim])
                else:
                    # For scalar values
                    filtered_values[i] = torch.median(neighbor_values)
    
    return filtered_values


def compute_features(
    data: Data, 
    feature_types: List[str] = ["curvature", "radius", "angle"],
    normalize: bool = True
) -> Data:
    """
    Compute additional features for vascular geometry.
    
    Args:
        data: PyTorch Geometric Data object.
        feature_types: List of features to compute.
        normalize: Whether to normalize computed features.
        
    Returns:
        Data object with additional features.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # Create a copy of the data
    feature_data = data.clone()
    
    # Compute requested features
    for feature_type in feature_types:
        if feature_type == "curvature":
            # Compute local curvature
            curvature = compute_curvature(data)
            feature_data.curvature = curvature
            
        elif feature_type == "radius":
            # Estimate radius if not already present
            if not hasattr(data, "radius"):
                radius = _estimate_radius(data)
                feature_data.radius = radius
                
        elif feature_type == "angle":
            # Compute branch angles
            angle_dict = _compute_node_angles(data)
            angles = torch.zeros(data.pos.shape[0], dtype=torch.float, device=data.pos.device)
            
            # Fill in the angle values
            for node, angle in angle_dict.items():
                angles[node] = angle
            
            feature_data.angle = angles
            
        elif feature_type == "centrality":
            # Compute graph centrality
            centrality = _compute_centrality(data)
            feature_data.centrality = centrality
            
        elif feature_type == "distance":
            # Compute distance from endpoints
            distance = _compute_endpoint_distance(data)
            feature_data.distance = distance
            
        elif feature_type == "thickness":
            # Compute vessel thickness
            thickness = _compute_thickness(data)
            feature_data.thickness = thickness
    
    # Normalize features if requested
    if normalize:
        for feature_type in feature_types:
            if hasattr(feature_data, feature_type):
                feature = getattr(feature_data, feature_type)
                
                if torch.max(feature) > torch.min(feature):
                    normalized_feature = (feature - torch.min(feature)) / (torch.max(feature) - torch.min(feature))
                    setattr(feature_data, feature_type, normalized_feature)
    
    return feature_data


def _estimate_radius(data: Data) -> torch.Tensor:
    """
    Estimate radius at each node based on local geometry.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Estimated radius tensor.
    """
    num_nodes = data.pos.shape[0]
    pos = data.pos
    edge_index = data.edge_index
    
    # Initialize radius tensor
    radius = torch.zeros(num_nodes, dtype=torch.float, device=pos.device)
    
    # Compute node degrees
    row, col = edge_index
    degrees = torch.bincount(row, minlength=num_nodes).float()
    
    # Compute average length of connected edges for each node
    for i in range(num_nodes):
        # Find neighbors
        neighbors = col[row == i]
        
        if len(neighbors) > 0:
            # Compute distances to neighbors
            neighbor_pos = pos[neighbors]
            distances = torch.norm(neighbor_pos - pos[i].unsqueeze(0), dim=1)
            
            # Average distance as estimate of radius
            radius[i] = torch.mean(distances) * 0.5  # Half the average distance
        else:
            # For isolated nodes, use the average radius
            radius[i] = 0.0
    
    # For zero radius values, use the median of non-zero values
    zero_mask = radius == 0
    if torch.any(zero_mask) and torch.any(~zero_mask):
        median_radius = torch.median(radius[~zero_mask])
        radius[zero_mask] = median_radius
    
    return radius


def _compute_node_angles(data: Data) -> Dict[int, float]:
    """
    Compute the angle at each node based on incoming edges.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Dictionary mapping node indices to angles.
    """
    pos = data.pos
    edge_index = data.edge_index
    
    # Extract adjacency information
    row, col = edge_index
    
    # Dictionary to store angles
    node_angles = {}
    
    # Compute angles for each node
    for i in range(pos.shape[0]):
        # Find neighbors
        neighbors = col[row == i]
        
        if len(neighbors) >= 2:
            # Compute vectors from node to neighbors
            vectors = pos[neighbors] - pos[i].unsqueeze(0)
            
            # Normalize vectors
            norms = torch.norm(vectors, dim=1, keepdim=True)
            vectors = vectors / (norms + 1e-8)
            
            # Compute all pairwise angles
            angles = []
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    cos_angle = torch.dot(vectors[j], vectors[k])
                    # Clamp to valid range for arccos
                    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                    angle = torch.acos(cos_angle) * 180 / np.pi
                    angles.append(angle.item())
            
            # Store the maximum angle (represents the branch angle)
            if angles:
                node_angles[i] = max(angles)
        
        # Special case for endpoints (assign 0 angle)
        elif len(neighbors) == 1:
            node_angles[i] = 0.0
    
    return node_angles


def _compute_centrality(data: Data) -> torch.Tensor:
    """
    Compute node centrality in the graph.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Centrality tensor.
    """
    num_nodes = data.pos.shape[0]
    edge_index = data.edge_index
    
    # Initialize centrality tensor
    centrality = torch.zeros(num_nodes, dtype=torch.float, device=data.pos.device)
    
    # Compute betweenness centrality (simplified approximation)
    # For each node, count how many shortest paths go through it
    for i in range(num_nodes):
        # Skip endpoints
        row, col = edge_index
        degree = torch.sum(row == i).item()
        
        if degree <= 1:
            continue
        
        # Run a simple breadth-first search
        centrality[i] = _compute_node_centrality(i, edge_index, num_nodes)
    
    # Normalize
    if torch.max(centrality) > 0:
        centrality = centrality / torch.max(centrality)
    
    return centrality


def _compute_node_centrality(node: int, edge_index: torch.Tensor, num_nodes: int) -> float:
    """
    Compute centrality for a single node.
    
    Args:
        node: Node index.
        edge_index: Edge connectivity tensor.
        num_nodes: Total number of nodes.
        
    Returns:
        Centrality value.
    """
    # Create adjacency list for faster traversal
    row, col = edge_index
    adj_list = {}
    for i in range(row.size(0)):
        src, dst = row[i].item(), col[i].item()
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append(dst)
    
    # Count shortest paths through this node
    paths_through_node = 0
    
    # Check paths between all pairs of nodes
    for start in range(num_nodes):
        if start == node:
            continue
        
        for end in range(start + 1, num_nodes):
            if end == node:
                continue
            
            # Find if node is on a shortest path from start to end
            if _is_on_shortest_path(start, end, node, adj_list):
                paths_through_node += 1
    
    return float(paths_through_node)


def _is_on_shortest_path(start: int, end: int, node: int, adj_list: Dict[int, List[int]]) -> bool:
    """
    Check if a node is on a shortest path between start and end.
    
    Args:
        start: Start node index.
        end: End node index.
        node: Node to check.
        adj_list: Adjacency list for the graph.
        
    Returns:
        True if the node is on a shortest path.
    """
    # Compute shortest path distance from start to end
    dist_start_end = _shortest_path_distance(start, end, adj_list)
    
    if dist_start_end == float('inf'):
        return False
    
    # Compute shortest path distance from start to node and from node to end
    dist_start_node = _shortest_path_distance(start, node, adj_list)
    dist_node_end = _shortest_path_distance(node, end, adj_list)
    
    # Node is on a shortest path if the sum of distances is equal to the direct distance
    return dist_start_node + dist_node_end == dist_start_end


def _shortest_path_distance(start: int, end: int, adj_list: Dict[int, List[int]]) -> float:
    """
    Compute the shortest path distance between two nodes.
    
    Args:
        start: Start node index.
        end: End node index.
        adj_list: Adjacency list for the graph.
        
    Returns:
        Shortest path distance, or inf if no path exists.
    """
    # Breadth-first search to find shortest path
    queue = [(start, 0)]  # (node, distance)
    visited = {start}
    
    while queue:
        node, distance = queue.pop(0)
        
        if node == end:
            return distance
        
        if node in adj_list:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    return float('inf')  # No path found


def _compute_endpoint_distance(data: Data) -> torch.Tensor:
    """
    Compute distance from each node to the nearest endpoint.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Distance tensor.
    """
    num_nodes = data.pos.shape[0]
    edge_index = data.edge_index
    
    # Identify endpoints (nodes with only one connection)
    row, col = edge_index
    degrees = torch.bincount(row, minlength=num_nodes)
    endpoints = torch.where(degrees == 1)[0]
    
    # If no endpoints, use nodes farthest from center
    if len(endpoints) == 0:
        center = torch.mean(data.pos, dim=0)
        distances = torch.norm(data.pos - center, dim=1)
        _, endpoints = torch.topk(distances, k=2)
    
    # Initialize distances tensor
    distances = torch.full((num_nodes,), float('inf'), dtype=torch.float, device=data.pos.device)
    
    # Compute distances from each endpoint using BFS
    for endpoint in endpoints:
        endpoint_distances = _compute_distances_from_node(endpoint.item(), edge_index, num_nodes)
        
        # Update with minimum distance to any endpoint
        distances = torch.minimum(distances, endpoint_distances)
    
    # Normalize distances
    if torch.max(distances) < float('inf'):
        distances = distances / torch.max(distances)
    else:
        # Handle isolated components
        distances[distances == float('inf')] = 1.0
    
    return distances


def _compute_distances_from_node(
    start_node: int, edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Compute distances from a start node to all other nodes.
    
    Args:
        start_node: Starting node index.
        edge_index: Edge connectivity tensor.
        num_nodes: Total number of nodes.
        
    Returns:
        Distances tensor.
    """
    # Create adjacency list for faster traversal
    row, col = edge_index
    adj_list = {}
    for i in range(row.size(0)):
        src, dst = row[i].item(), col[i].item()
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append(dst)
    
    # Initialize distances tensor
    distances = torch.full((num_nodes,), float('inf'), dtype=torch.float, device=edge_index.device)
    distances[start_node] = 0.0
    
    # Breadth-first search
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        
        if node in adj_list:
            for neighbor in adj_list[node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[node] + 1.0
                    queue.append(neighbor)
    
    return distances


def _compute_thickness(data: Data) -> torch.Tensor:
    """
    Compute vessel thickness at each node.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Thickness tensor.
    """
    # If radius attribute is available, use it
    if hasattr(data, "radius") and data.radius is not None:
        radius = data.radius
        if radius.dim() == 2 and radius.shape[1] == 1:
            radius = radius.squeeze(1)
        
        # Thickness is twice the radius
        return 2.0 * radius
    
    # Estimate thickness based on local geometry
    num_nodes = data.pos.shape[0]
    pos = data.pos
    edge_index = data.edge_index
    
    # Initialize thickness tensor
    thickness = torch.zeros(num_nodes, dtype=torch.float, device=pos.device)
    
    # Create adjacency list for faster lookup
    row, col = edge_index
    adj_list = {}
    for i in range(row.size(0)):
        src, dst = row[i].item(), col[i].item()
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append(dst)
    
    # Compute average distance to neighbors for each node
    for node in range(num_nodes):
        if node in adj_list:
            neighbors = adj_list[node]
            if neighbors:
                # Compute distances to neighbors
                distances = torch.norm(pos[neighbors] - pos[node].unsqueeze(0), dim=1)
                # Average distance as estimate of thickness
                thickness[node] = torch.mean(distances) * 2.0
        
        # If no neighbors or zero thickness, use a default value
        if thickness[node] == 0:
            # Fallback to average value or a default
            thickness[node] = 1.0
    
    # Smooth the thickness values to avoid abrupt changes
    smoothed_thickness = _apply_gaussian_filter(thickness.unsqueeze(1), edge_index, strength=0.5, iterations=2)
    
    return smoothed_thickness.squeeze(1)