"""
Geometry processing functions for vascular simulation data.

This module provides functions for analyzing and processing vascular geometry,
including calculations of geometric properties and extraction of features.
"""

import logging
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union

# Try to import VTK (optional dependency)
try:
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

logger = logging.getLogger(__name__)


def compute_curvature(data: Data) -> torch.Tensor:
    """
    Compute local curvature at each node in the vascular geometry.
    
    Args:
        data: PyTorch Geometric Data object representing vascular geometry.
        
    Returns:
        Tensor of shape [num_nodes] containing curvature values.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # Extract node positions and connectivity
    pos = data.pos
    edge_index = data.edge_index
    
    # Initialize curvature tensor
    num_nodes = pos.shape[0]
    curvature = torch.zeros(num_nodes, dtype=torch.float32, device=pos.device)
    
    # Compute curvature for each node
    for i in range(num_nodes):
        # Find neighbors
        neighbors = edge_index[1, edge_index[0] == i].unique()
        neighbors = neighbors[neighbors != i]  # Exclude self-loops
        
        if len(neighbors) < 2:
            # Not enough neighbors to compute curvature
            continue
            
        # Get neighbor positions
        neighbor_pos = pos[neighbors]
        center_pos = pos[i]
        
        # Compute vectors from center to neighbors
        vectors = neighbor_pos - center_pos
        
        # Normalize vectors
        norms = torch.norm(vectors, dim=1, keepdim=True)
        vectors = vectors / (norms + 1e-8)  # Avoid division by zero
        
        # Compute average direction
        avg_dir = torch.mean(vectors, dim=0)
        avg_dir_norm = torch.norm(avg_dir)
        
        if avg_dir_norm > 1e-8:
            # Normalize average direction
            avg_dir = avg_dir / avg_dir_norm
            
            # Compute deviation from straight line (1 - cos(angle))
            deviations = 1.0 - torch.abs(torch.sum(vectors * avg_dir, dim=1))
            
            # Average deviation is proportional to curvature
            curvature[i] = torch.mean(deviations)
            
    return curvature


def compute_surface_area(data: Data) -> float:
    """
    Compute the total surface area of the vascular geometry.
    
    Args:
        data: PyTorch Geometric Data object representing vascular geometry.
        
    Returns:
        Total surface area value.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # If radius attribute is available, use it for accurate surface area
    if hasattr(data, "radius") and data.radius is not None:
        return _compute_surface_area_with_radius(data)
    else:
        # Fallback to approximation based on edges
        return _compute_surface_area_from_edges(data)


def _compute_surface_area_with_radius(data: Data) -> float:
    """
    Compute surface area using vessel radius.
    
    Args:
        data: PyTorch Geometric Data with radius attribute.
        
    Returns:
        Total surface area value.
    """
    pos = data.pos
    edge_index = data.edge_index
    radius = data.radius
    
    # Ensure radius is a tensor
    if not isinstance(radius, torch.Tensor):
        radius = torch.tensor(radius, dtype=torch.float32, device=pos.device)
    
    # Ensure radius has correct shape
    if radius.dim() == 2 and radius.shape[1] == 1:
        radius = radius.squeeze(1)
    
    total_area = 0.0
    
    # Compute area for each edge
    edges = edge_index.t()
    for i, j in edges:
        i, j = i.item(), j.item()
        
        # Get positions and radii
        p1, p2 = pos[i], pos[j]
        r1, r2 = radius[i], radius[j]
        
        # Compute edge length
        length = torch.norm(p2 - p1)
        
        # Compute lateral surface area of truncated cone
        slant_height = torch.sqrt(length**2 + (r2 - r1)**2)
        area = np.pi * (r1 + r2) * slant_height
        
        total_area += area.item()
    
    # Divide by 2 because each edge is counted twice
    return total_area / 2.0


def _compute_surface_area_from_edges(data: Data) -> float:
    """
    Approximate surface area from edge lengths.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Approximate surface area value.
    """
    pos = data.pos
    edge_index = data.edge_index
    
    # Estimate average radius from geometry
    bbox_size = torch.max(pos, dim=0)[0] - torch.min(pos, dim=0)[0]
    estimated_radius = torch.mean(bbox_size) * 0.05  # Heuristic estimate
    
    total_area = 0.0
    
    # Compute area for each edge
    edges = edge_index.t()
    for i, j in edges:
        i, j = i.item(), j.item()
        
        # Get positions
        p1, p2 = pos[i], pos[j]
        
        # Compute edge length
        length = torch.norm(p2 - p1)
        
        # Approximate area as cylinder
        area = 2 * np.pi * estimated_radius * length
        
        total_area += area.item()
    
    # Divide by 2 because each edge is counted twice
    return total_area / 2.0


def compute_volume(data: Data) -> float:
    """
    Compute the total volume of the vascular geometry.
    
    Args:
        data: PyTorch Geometric Data object representing vascular geometry.
        
    Returns:
        Total volume value.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # If radius attribute is available, use it for accurate volume
    if hasattr(data, "radius") and data.radius is not None:
        return _compute_volume_with_radius(data)
    else:
        # Fallback to approximation based on edges
        return _compute_volume_from_edges(data)


def _compute_volume_with_radius(data: Data) -> float:
    """
    Compute volume using vessel radius.
    
    Args:
        data: PyTorch Geometric Data with radius attribute.
        
    Returns:
        Total volume value.
    """
    pos = data.pos
    edge_index = data.edge_index
    radius = data.radius
    
    # Ensure radius is a tensor
    if not isinstance(radius, torch.Tensor):
        radius = torch.tensor(radius, dtype=torch.float32, device=pos.device)
    
    # Ensure radius has correct shape
    if radius.dim() == 2 and radius.shape[1] == 1:
        radius = radius.squeeze(1)
    
    total_volume = 0.0
    
    # Compute volume for each edge
    edges = edge_index.t()
    for i, j in edges:
        i, j = i.item(), j.item()
        
        # Get positions and radii
        p1, p2 = pos[i], pos[j]
        r1, r2 = radius[i], radius[j]
        
        # Compute edge length
        length = torch.norm(p2 - p1)
        
        # Compute volume of truncated cone
        volume = (1/3) * np.pi * length * (r1**2 + r1*r2 + r2**2)
        
        total_volume += volume.item()
    
    # Divide by 2 because each edge is counted twice
    return total_volume / 2.0


def _compute_volume_from_edges(data: Data) -> float:
    """
    Approximate volume from edge lengths.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Approximate volume value.
    """
    pos = data.pos
    edge_index = data.edge_index
    
    # Estimate average radius from geometry
    bbox_size = torch.max(pos, dim=0)[0] - torch.min(pos, dim=0)[0]
    estimated_radius = torch.mean(bbox_size) * 0.05  # Heuristic estimate
    
    total_volume = 0.0
    
    # Compute volume for each edge
    edges = edge_index.t()
    for i, j in edges:
        i, j = i.item(), j.item()
        
        # Get positions
        p1, p2 = pos[i], pos[j]
        
        # Compute edge length
        length = torch.norm(p2 - p1)
        
        # Approximate volume as cylinder
        volume = np.pi * estimated_radius**2 * length
        
        total_volume += volume.item()
    
    # Divide by 2 because each edge is counted twice
    return total_volume / 2.0


def extract_centerline(
    data: Data, 
    smoothing: float = 0.1, 
    use_radius_weights: bool = True
) -> Data:
    """
    Extract the centerline from a vascular geometry.
    
    Args:
        data: PyTorch Geometric Data object representing vascular geometry.
        smoothing: Smoothing factor for centerline extraction.
        use_radius_weights: Whether to use radius as weights for smoothing.
        
    Returns:
        PyTorch Geometric Data object representing the centerline.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # Extract node positions and connectivity
    pos = data.pos
    edge_index = data.edge_index
    
    # If VTK is available, use it for more accurate centerline extraction
    if HAS_VTK:
        try:
            return _extract_centerline_vtk(data, smoothing)
        except Exception as e:
            logger.warning(f"VTK centerline extraction failed: {e}. Falling back to basic method.")
    
    # Basic centerline extraction using graph connectivity
    
    # Compute node degrees
    row, col = edge_index
    node_degrees = torch.bincount(row, minlength=pos.shape[0])
    
    # Identify branch points (degree > 2) and endpoints (degree == 1)
    branch_points = torch.where(node_degrees > 2)[0]
    endpoints = torch.where(node_degrees == 1)[0]
    
    # If no endpoints found, use points farthest from center as endpoints
    if len(endpoints) == 0:
        center = torch.mean(pos, dim=0)
        distances = torch.norm(pos - center, dim=1)
        _, endpoints = torch.topk(distances, k=2)
    
    # Compute weights for smoothing
    weights = None
    if use_radius_weights and hasattr(data, "radius") and data.radius is not None:
        weights = data.radius
        if weights.dim() == 2 and weights.shape[1] == 1:
            weights = weights.squeeze(1)
    
    # Extract paths between endpoints and branch points
    centerline_edges = []
    visited_nodes = set()
    
    # Connect all endpoints to nearest branch point
    for endpoint in endpoints:
        endpoint = endpoint.item()
        path = _find_path_to_nearest(
            endpoint, branch_points, edge_index, 
            weights=weights, visited=visited_nodes
        )
        
        if path:
            # Add path edges to centerline
            for i in range(len(path) - 1):
                centerline_edges.append([path[i], path[i+1]])
                visited_nodes.add(path[i])
            
            visited_nodes.add(path[-1])
    
    # Connect branch points if not already connected
    for i, bp1 in enumerate(branch_points):
        bp1 = bp1.item()
        for bp2 in branch_points[i+1:]:
            bp2 = bp2.item()
            
            # Check if these branch points need to be connected
            if not _is_connected(bp1, bp2, centerline_edges):
                path = _find_path_to_nearest(
                    bp1, [bp2], edge_index, 
                    weights=weights, visited=visited_nodes
                )
                
                if path:
                    # Add path edges to centerline
                    for j in range(len(path) - 1):
                        centerline_edges.append([path[j], path[j+1]])
    
    # Convert to tensor
    if centerline_edges:
        centerline_edge_index = torch.tensor(centerline_edges, dtype=torch.long).t()
    else:
        # Fallback: use original edges
        centerline_edge_index = edge_index
    
    # Create centerline data object
    centerline_data = Data(
        pos=pos.clone(),
        edge_index=centerline_edge_index
    )
    
    # Smooth centerline if requested
    if smoothing > 0:
        centerline_data.pos = _smooth_positions(
            centerline_data.pos, 
            centerline_data.edge_index,
            iterations=int(3 * smoothing),
            strength=smoothing
        )
    
    # Copy additional attributes
    if hasattr(data, "radius") and data.radius is not None:
        centerline_data.radius = data.radius.clone()
    
    return centerline_data


def _extract_centerline_vtk(data: Data, smoothing: float = 0.1) -> Data:
    """
    Extract centerline using VTK.
    
    Args:
        data: PyTorch Geometric Data object.
        smoothing: Smoothing factor.
        
    Returns:
        Centerline as PyTorch Geometric Data.
    """
    if not HAS_VTK:
        raise ImportError("VTK is required for this function")
    
    # Convert to VTK format
    points = data.pos.detach().cpu().numpy()
    edges = data.edge_index.t().detach().cpu().numpy()
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point[0], point[1], point[2])
    
    # Create VTK lines
    vtk_lines = vtk.vtkCellArray()
    for edge in edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        vtk_lines.InsertNextCell(line)
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(vtk_lines)
    
    # Use VTK's centerline extraction filter
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()
    
    # Extract centerline using VMTK-like approach with VTK's spline filter
    spline_filter = vtk.vtkSplineFilter()
    spline_filter.SetInputConnection(cleaner.GetOutputPort())
    spline_filter.SetSubdivideToLength()
    spline_filter.SetLength(0.5)  # Parameter for spacing
    spline_filter.Update()
    
    # Get result
    centerline = spline_filter.GetOutput()
    
    # Convert back to PyTorch Geometric format
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    
    # Extract lines
    cl_edges = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        if cell.GetNumberOfPoints() > 1:
            for j in range(cell.GetNumberOfPoints() - 1):
                cl_edges.append([
                    cell.GetPointId(j),
                    cell.GetPointId(j + 1)
                ])
    
    # Convert to tensors
    cl_pos = torch.tensor(cl_points, dtype=torch.float32)
    cl_edge_index = torch.tensor(cl_edges, dtype=torch.long).t()
    
    # Create centerline data
    centerline_data = Data(
        pos=cl_pos,
        edge_index=cl_edge_index
    )
    
    return centerline_data


def _find_path_to_nearest(
    start: int, 
    targets: List[int], 
    edge_index: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    visited: Optional[set] = None
) -> List[int]:
    """
    Find shortest path from start node to closest target node.
    
    Args:
        start: Starting node index.
        targets: List of target node indices.
        edge_index: Edge connectivity tensor.
        weights: Optional weights for edges (e.g., radius).
        visited: Optional set of already visited nodes to avoid.
        
    Returns:
        List of node indices forming the path, or empty list if no path found.
    """
    if visited is None:
        visited = set()
    
    targets = set(t for t in targets if t != start)
    if not targets:
        return []
    
    # Initialize
    queue = [(0, start, [start])]  # (distance, node, path)
    seen = {start}
    
    # Extract adjacency list for faster lookup
    adj_list = {}
    row, col = edge_index
    for i in range(row.size(0)):
        src, dst = row[i].item(), col[i].item()
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append(dst)
    
    # Dijkstra's algorithm
    while queue:
        # Get node with smallest distance
        dist, node, path = queue.pop(0)
        
        # Check if we reached a target
        if node in targets:
            return path
        
        # Check neighbors
        if node in adj_list:
            for neighbor in adj_list[node]:
                if neighbor in seen or neighbor in visited:
                    continue
                
                # Compute edge weight
                if weights is not None:
                    weight = 1.0 / (weights[neighbor].item() + 1e-8)  # Larger radius = smaller weight
                else:
                    weight = 1.0
                
                # Add to queue
                queue.append((dist + weight, neighbor, path + [neighbor]))
                seen.add(neighbor)
            
            # Sort queue by distance
            queue.sort(key=lambda x: x[0])
    
    # No path found
    return []


def _is_connected(node1: int, node2: int, edges: List[List[int]]) -> bool:
    """
    Check if two nodes are already connected by a path.
    
    Args:
        node1: First node index.
        node2: Second node index.
        edges: List of edges.
        
    Returns:
        True if connected, False otherwise.
    """
    # Create adjacency list
    adj_list = {}
    for src, dst in edges:
        if src not in adj_list:
            adj_list[src] = []
        if dst not in adj_list:
            adj_list[dst] = []
        adj_list[src].append(dst)
        adj_list[dst].append(src)  # Undirected
    
    # BFS
    queue = [node1]
    visited = {node1}
    
    while queue:
        node = queue.pop(0)
        
        if node == node2:
            return True
        
        if node in adj_list:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
    
    return False


def _smooth_positions(
    pos: torch.Tensor, 
    edge_index: torch.Tensor,
    iterations: int = 3,
    strength: float = 0.1
) -> torch.Tensor:
    """
    Smooth node positions using Laplacian smoothing.
    
    Args:
        pos: Node position tensor.
        edge_index: Edge connectivity tensor.
        iterations: Number of smoothing iterations.
        strength: Smoothing strength (0-1).
        
    Returns:
        Smoothed positions tensor.
    """
    # Create adjacency list for faster lookup
    adj_list = {}
    row, col = edge_index
    for i in range(row.size(0)):
        src, dst = row[i].item(), col[i].item()
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append(dst)
    
    # Apply Laplacian smoothing
    smoothed_pos = pos.clone()
    
    for _ in range(iterations):
        new_pos = smoothed_pos.clone()
        
        for node in range(pos.shape[0]):
            if node in adj_list and adj_list[node]:
                # Compute average position of neighbors
                neighbors = adj_list[node]
                neighbor_pos = smoothed_pos[neighbors]
                avg_pos = torch.mean(neighbor_pos, dim=0)
                
                # Apply weighted average
                new_pos[node] = (1 - strength) * smoothed_pos[node] + strength * avg_pos
        
        smoothed_pos = new_pos
    
    return smoothed_pos


def compute_branch_angles(data: Data) -> Dict[str, float]:
    """
    Compute angles between branches in vascular geometry.
    
    Args:
        data: PyTorch Geometric Data object representing vascular geometry.
        
    Returns:
        Dictionary mapping branch pairs (e.g., "1-2") to angles in degrees.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError("Data object must have valid 'pos' attribute")
        
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object must have valid 'edge_index' attribute")
    
    # Extract node positions and connectivity
    pos = data.pos
    edge_index = data.edge_index
    
    # Compute node degrees
    row, col = edge_index
    node_degrees = torch.bincount(row, minlength=pos.shape[0])
    
    # Identify branch points (nodes with more than 2 connections)
    branch_points = torch.where(node_degrees > 2)[0]
    
    # Dictionary to store branch angles
    angles = {}
    
    # Process each branch point
    for bp in branch_points:
        bp_idx = bp.item()
        
        # Find all connected branches
        branches = []
        
        # Find neighbors
        neighbors = col[row == bp_idx].tolist()
        
        for neighbor in neighbors:
            # Find direction vector from branch point to neighbor
            direction = pos[neighbor] - pos[bp_idx]
            direction = direction / (torch.norm(direction) + 1e-8)
            branches.append((neighbor, direction))
        
        # Compute angles between all pairs of branches
        for i, (branch1, dir1) in enumerate(branches):
            for j, (branch2, dir2) in enumerate(branches):
                if i < j:
                    # Compute angle between direction vectors
                    cos_angle = torch.dot(dir1, dir2)
                    # Clamp to valid range for arccos
                    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                    angle_rad = torch.acos(cos_angle)
                    angle_deg = angle_rad * 180 / np.pi
                    
                    # Store in dictionary
                    branch_key = f"{branch1}-{branch2}"
                    angles[branch_key] = angle_deg.item()
    
    return angles