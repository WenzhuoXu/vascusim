"""
Visualization utilities for vascular simulation data.

This module provides functions for visualizing vascular geometry, flow data,
pressure fields, and other simulation results using PyVista and Matplotlib.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Try to import PyVista for 3D visualization
try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# Try to import VTK
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    HAS_VTK = True
except ImportError:
    HAS_VTK = False

from torch_geometric.data import Data

# Import internal modules
from ..io.vtk_utils import extract_mesh_from_vtu, extract_points_from_vtp

logger = logging.getLogger(__name__)


def check_visualization_libraries() -> Dict[str, bool]:
    """
    Check which visualization libraries are available.

    Returns:
        Dictionary with library availability.
    """
    return {"matplotlib": True, "pyvista": HAS_PYVISTA, "vtk": HAS_VTK}  # Always required


def plot_geometry(
    data: Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData", str, Path],
    ax: Optional[plt.Axes] = None,
    color: str = "blue",
    linewidth: float = 1.0,
    alpha: float = 0.8,
    show_points: bool = False,
    point_size: float = 10,
    title: Optional[str] = None,
    use_pyvista: Optional[bool] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Union[plt.Figure, "pv.Plotter"]:
    """
    Plot vascular geometry.

    Args:
        data: Data to plot. Can be PyTorch Geometric Data, VTK object, or file path.
        ax: Optional matplotlib axes to plot on.
        color: Color for the plot.
        linewidth: Line width for matplotlib plots.
        alpha: Opacity level.
        show_points: Whether to show points.
        point_size: Size of points if shown.
        title: Plot title.
        use_pyvista: Whether to use PyVista (3D) or Matplotlib (2D).
                    If None, uses PyVista if available and 3D data is detected.
        show: Whether to show the plot.
        save_path: Optional path to save the plot.
        figsize: Figure size for matplotlib plots.
        **kwargs: Additional keyword arguments for the plotting functions.

    Returns:
        Figure or Plotter object.

    Raises:
        ImportError: If required libraries are not available.
        ValueError: If the data is invalid or incompatible.
    """
    # Determine whether to use PyVista or Matplotlib
    if use_pyvista is None:
        use_pyvista = HAS_PYVISTA

    # Convert data to appropriate format
    if isinstance(data, (str, Path)):
        # Load from file
        if str(data).endswith(".vtu"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTU files. Install it with 'pip install vtk'."
                )
            mesh, _, _ = extract_mesh_from_vtu(data)
            data = mesh
        elif str(data).endswith(".vtp"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTP files. Install it with 'pip install vtk'."
                )
            polydata, _, _ = extract_points_from_vtp(data)
            data = polydata
        else:
            raise ValueError(f"Unsupported file type: {data}")

    if use_pyvista:
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista is required for 3D visualization. Install it with 'pip install pyvista'."
            )

        return _plot_geometry_pyvista(
            data=data,
            color=color,
            alpha=alpha,
            show_points=show_points,
            point_size=point_size,
            title=title,
            show=show,
            save_path=save_path,
            **kwargs,
        )
    else:
        return _plot_geometry_matplotlib(
            data=data,
            ax=ax,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            show_points=show_points,
            point_size=point_size,
            title=title,
            show=show,
            save_path=save_path,
            figsize=figsize,
            **kwargs,
        )


def _plot_geometry_matplotlib(
    data: Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData"],
    ax: Optional[plt.Axes] = None,
    color: str = "blue",
    linewidth: float = 1.0,
    alpha: float = 0.8,
    show_points: bool = False,
    point_size: float = 10,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> plt.Figure:
    """
    Plot vascular geometry using Matplotlib (internal function).

    Args:
        See plot_geometry for parameter descriptions.

    Returns:
        Matplotlib Figure object.
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Extract points and edges based on data type
    if isinstance(data, Data):
        # PyTorch Geometric Data
        points = data.pos.detach().cpu().numpy()

        if hasattr(data, "edge_index") and data.edge_index is not None:
            edges = data.edge_index.detach().cpu().numpy().T
        else:
            edges = None
    elif HAS_VTK and isinstance(data, vtk.vtkUnstructuredGrid):
        # VTK UnstructuredGrid
        points = vtk_to_numpy(data.GetPoints().GetData())
        edges = _extract_edges_from_vtk(data)
    elif HAS_VTK and isinstance(data, vtk.vtkPolyData):
        # VTK PolyData
        points = vtk_to_numpy(data.GetPoints().GetData())
        edges = _extract_edges_from_polydata(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Plot edges if available
    if edges is not None and len(edges) > 0:
        for i, j in edges:
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                **kwargs,
            )

    # Plot points if requested
    if show_points:
        ax.scatter(points[:, 0], points[:, 1], s=point_size, color=color, alpha=alpha, **kwargs)

    # Set title if provided
    if title is not None:
        ax.set_title(title)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig


def _plot_geometry_pyvista(
    data: Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData"],
    color: str = "blue",
    alpha: float = 0.8,
    show_points: bool = False,
    point_size: float = 10,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> "pv.Plotter":
    """
    Plot vascular geometry using PyVista (internal function).

    Args:
        See plot_geometry for parameter descriptions.

    Returns:
        PyVista Plotter object.
    """
    if not HAS_PYVISTA:
        raise ImportError(
            "PyVista is required for 3D visualization. Install it with 'pip install pyvista'."
        )

    # Create plotter
    plotter = pv.Plotter()

    # Convert data to PyVista mesh
    if isinstance(data, Data):
        # PyTorch Geometric Data
        points = data.pos.detach().cpu().numpy()

        if hasattr(data, "edge_index") and data.edge_index is not None:
            edges = data.edge_index.detach().cpu().numpy().T

            # Create lines for each edge
            lines = []
            for edge in edges:
                lines.append([2, edge[0], edge[1]])

            # Create PyVista mesh
            mesh = pv.PolyData(points, lines=lines)
        else:
            # Just points, no connectivity
            mesh = pv.PolyData(points)
    elif HAS_VTK and isinstance(data, (vtk.vtkUnstructuredGrid, vtk.vtkPolyData)):
        # Convert VTK object to PyVista mesh
        mesh = pv.wrap(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Add mesh to the plotter
    plotter.add_mesh(mesh, color=color, opacity=alpha, show_edges=True, **kwargs)

    # Add points if requested
    if show_points:
        plotter.add_points(mesh.points, color=color, point_size=point_size, opacity=alpha, **kwargs)

    # Set title if provided
    if title is not None:
        plotter.add_title(title)

    # Add axes
    plotter.add_axes()
    plotter.add_bounding_box()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(save_path), transparent_background=True)

    # Show if requested
    if show:
        plotter.show()

    return plotter


def _extract_edges_from_vtk(grid: "vtk.vtkUnstructuredGrid") -> np.ndarray:
    """
    Extract edges from a VTK unstructured grid.

    Args:
        grid: VTK unstructured grid.

    Returns:
        NumPy array of shape (num_edges, 2) containing edge endpoints.
    """
    edges = set()

    for i in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(i)
        num_points = cell.GetNumberOfPoints()

        for j in range(num_points):
            p1 = cell.GetPointId(j)
            p2 = cell.GetPointId((j + 1) % num_points)

            # Store edge with smallest index first for consistency
            edge = (min(p1, p2), max(p1, p2))
            edges.add(edge)

    return np.array(list(edges))


def _extract_edges_from_polydata(polydata: "vtk.vtkPolyData") -> np.ndarray:
    """
    Extract edges from a VTK polydata object.

    Args:
        polydata: VTK polydata object.

    Returns:
        NumPy array of shape (num_edges, 2) containing edge endpoints.
    """
    edges = set()

    # Extract edges from polygons
    polygons = polydata.GetPolys()
    if polygons and polygons.GetNumberOfCells() > 0:
        polygons.InitTraversal()
        id_list = vtk.vtkIdList()

        while polygons.GetNextCell(id_list):
            num_points = id_list.GetNumberOfIds()

            for j in range(num_points):
                p1 = id_list.GetId(j)
                p2 = id_list.GetId((j + 1) % num_points)

                # Store edge with smallest index first for consistency
                edge = (min(p1, p2), max(p1, p2))
                edges.add(edge)

    # Extract edges from lines
    lines = polydata.GetLines()
    if lines and lines.GetNumberOfCells() > 0:
        lines.InitTraversal()
        id_list = vtk.vtkIdList()

        while lines.GetNextCell(id_list):
            num_points = id_list.GetNumberOfIds()

            for j in range(num_points - 1):
                p1 = id_list.GetId(j)
                p2 = id_list.GetId(j + 1)

                # Store edge with smallest index first for consistency
                edge = (min(p1, p2), max(p1, p2))
                edges.add(edge)

    return np.array(list(edges))


def plot_flow(
    data: Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData", str, Path],
    flow_field: str = "velocity",
    scale: float = 0.1,
    density: float = 0.5,
    color_by_magnitude: bool = True,
    cmap: str = "viridis",
    title: Optional[str] = None,
    use_pyvista: Optional[bool] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Union[plt.Figure, "pv.Plotter"]:
    """
    Plot flow fields in vascular geometry.

    Args:
        data: Data to plot. Can be PyTorch Geometric Data, VTK object, or file path.
        flow_field: Name of the flow field attribute.
        scale: Scale factor for arrows.
        density: Density of arrows (0.0 to 1.0).
        color_by_magnitude: Whether to color arrows by magnitude.
        cmap: Colormap for magnitude coloring.
        title: Plot title.
        use_pyvista: Whether to use PyVista (3D) or Matplotlib (2D).
                    If None, uses PyVista if available and 3D data is detected.
        show: Whether to show the plot.
        save_path: Optional path to save the plot.
        figsize: Figure size for matplotlib plots.
        **kwargs: Additional keyword arguments for the plotting functions.

    Returns:
        Figure or Plotter object.

    Raises:
        ImportError: If required libraries are not available.
        ValueError: If the data is invalid or incompatible.
        KeyError: If the flow field is not found in the data.
    """
    # Determine whether to use PyVista or Matplotlib
    if use_pyvista is None:
        use_pyvista = HAS_PYVISTA

    # Convert data to appropriate format
    if isinstance(data, (str, Path)):
        # Load from file
        if str(data).endswith(".vtu"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTU files. Install it with 'pip install vtk'."
                )
            mesh, _, _ = extract_mesh_from_vtu(data)
            data = mesh
        elif str(data).endswith(".vtp"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTP files. Install it with 'pip install vtk'."
                )
            polydata, _, _ = extract_points_from_vtp(data)
            data = polydata
        else:
            raise ValueError(f"Unsupported file type: {data}")

    # Extract flow field
    if isinstance(data, Data):
        # Find the flow field in PyTorch Geometric Data
        flow_field_key = None
        for key in data.keys:
            if key.endswith(flow_field) or key == flow_field:
                flow_field_key = key
                break

        if flow_field_key is None:
            raise KeyError(f"Flow field '{flow_field}' not found in data")

        flow_data = data[flow_field_key].detach().cpu().numpy()
        points = data.pos.detach().cpu().numpy()
    elif HAS_VTK and isinstance(data, (vtk.vtkUnstructuredGrid, vtk.vtkPolyData)):
        # Find the flow field in VTK data
        point_data = data.GetPointData()
        array_idx = point_data.HasArray(flow_field)

        if array_idx == -1:
            raise KeyError(f"Flow field '{flow_field}' not found in data")

        flow_data = vtk_to_numpy(point_data.GetArray(flow_field))
        points = vtk_to_numpy(data.GetPoints().GetData())
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Ensure the flow data has the right shape
    if flow_data.ndim == 1:
        # Scalar field - not suitable for flow visualization
        raise ValueError(f"Flow field '{flow_field}' is scalar, but vector field is required")
    elif flow_data.shape[1] not in [2, 3]:
        raise ValueError(f"Flow field '{flow_field}' has unsupported dimensions: {flow_data.shape}")

    if use_pyvista:
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista is required for 3D visualization. Install it with 'pip install pyvista'."
            )

        # Create PyVista mesh with flow data
        mesh = pv.PolyData(points)
        mesh.point_data[flow_field] = flow_data

        # Create plotter
        plotter = pv.Plotter()

        # Add geometry
        plotter.add_mesh(
            mesh,
            color="lightgray",
            opacity=0.3,
            show_edges=True,
        )

        # Add flow arrows
        plotter.add_arrows(
            mesh.points,
            flow_data,
            scale=scale,
            color="red" if not color_by_magnitude else None,
            scalars="Magnitude" if color_by_magnitude else None,
            cmap=cmap,
            **kwargs,
        )

        # Set title if provided
        if title is not None:
            plotter.add_title(title)

        # Add axes
        plotter.add_axes()
        plotter.add_bounding_box()

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(save_path), transparent_background=True)

        # Show if requested
        if show:
            plotter.show()

        return plotter
    else:
        # Matplotlib quiver plot
        fig, ax = plt.subplots(figsize=figsize)

        # Subsample points based on density
        n_points = points.shape[0]
        n_samples = int(n_points * density)

        if n_samples < n_points:
            indices = np.random.choice(n_points, n_samples, replace=False)
            sample_points = points[indices]
            sample_flow = flow_data[indices]
        else:
            sample_points = points
            sample_flow = flow_data

        # Plot geometry
        ax.scatter(points[:, 0], points[:, 1], s=1, color="lightgray", alpha=0.3)

        # Plot flow vectors
        if color_by_magnitude:
            magnitudes = np.linalg.norm(sample_flow, axis=1)
            quiv = ax.quiver(
                sample_points[:, 0],
                sample_points[:, 1],
                sample_flow[:, 0],
                sample_flow[:, 1],
                magnitudes,
                scale=1 / scale,
                cmap=cmap,
                **kwargs,
            )
            plt.colorbar(quiv, ax=ax, label=f"{flow_field} magnitude")
        else:
            ax.quiver(
                sample_points[:, 0],
                sample_points[:, 1],
                sample_flow[:, 0],
                sample_flow[:, 1],
                color="red",
                scale=1 / scale,
                **kwargs,
            )

        # Set title if provided
        if title is not None:
            ax.set_title(title)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.5)

        # Add labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show if requested
        if show:
            plt.show()

        return fig


def plot_pressure(
    data: Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData", str, Path],
    pressure_field: str = "pressure",
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    use_pyvista: Optional[bool] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Union[plt.Figure, "pv.Plotter"]:
    """
    Plot pressure fields in vascular geometry.

    Args:
        data: Data to plot. Can be PyTorch Geometric Data, VTK object, or file path.
        pressure_field: Name of the pressure field attribute.
        cmap: Colormap for pressure visualization.
        title: Plot title.
        use_pyvista: Whether to use PyVista (3D) or Matplotlib (2D).
                    If None, uses PyVista if available and 3D data is detected.
        show: Whether to show the plot.
        save_path: Optional path to save the plot.
        figsize: Figure size for matplotlib plots.
        **kwargs: Additional keyword arguments for the plotting functions.

    Returns:
        Figure or Plotter object.

    Raises:
        ImportError: If required libraries are not available.
        ValueError: If the data is invalid or incompatible.
        KeyError: If the pressure field is not found in the data.
    """
    # Determine whether to use PyVista or Matplotlib
    if use_pyvista is None:
        use_pyvista = HAS_PYVISTA

    # Convert data to appropriate format
    if isinstance(data, (str, Path)):
        # Load from file
        if str(data).endswith(".vtu"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTU files. Install it with 'pip install vtk'."
                )
            mesh, _, _ = extract_mesh_from_vtu(data)
            data = mesh
        elif str(data).endswith(".vtp"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTP files. Install it with 'pip install vtk'."
                )
            polydata, _, _ = extract_points_from_vtp(data)
            data = polydata
        else:
            raise ValueError(f"Unsupported file type: {data}")

    # Extract pressure field
    if isinstance(data, Data):
        # Find the pressure field in PyTorch Geometric Data
        pressure_field_key = None
        for key in data.keys:
            if key.endswith(pressure_field) or key == pressure_field:
                pressure_field_key = key
                break

        if pressure_field_key is None:
            raise KeyError(f"Pressure field '{pressure_field}' not found in data")

        pressure_data = data[pressure_field_key].detach().cpu().numpy()
        points = data.pos.detach().cpu().numpy()

        # Handle vector data (take magnitude)
        if pressure_data.ndim > 1 and pressure_data.shape[1] > 1:
            pressure_data = np.linalg.norm(pressure_data, axis=1)
    elif HAS_VTK and isinstance(data, (vtk.vtkUnstructuredGrid, vtk.vtkPolyData)):
        # Find the pressure field in VTK data
        point_data = data.GetPointData()
        array_idx = point_data.HasArray(pressure_field)

        if array_idx == -1:
            raise KeyError(f"Pressure field '{pressure_field}' not found in data")

        pressure_array = point_data.GetArray(pressure_field)
        pressure_data = vtk_to_numpy(pressure_array)
        points = vtk_to_numpy(data.GetPoints().GetData())

        # Handle vector data (take magnitude)
        if pressure_array.GetNumberOfComponents() > 1:
            pressure_data = np.linalg.norm(pressure_data, axis=1)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if use_pyvista:
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista is required for 3D visualization. Install it with 'pip install pyvista'."
            )

        # Create PyVista mesh with pressure data
        if isinstance(data, Data):
            # Create from points
            mesh = pv.PolyData(points)
        elif HAS_VTK and isinstance(data, (vtk.vtkUnstructuredGrid, vtk.vtkPolyData)):
            # Convert VTK object to PyVista mesh
            mesh = pv.wrap(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Add pressure data to mesh
        mesh.point_data[pressure_field] = pressure_data

        # Create plotter
        plotter = pv.Plotter()

        # Add mesh with pressure mapping
        plotter.add_mesh(mesh, scalars=pressure_field, cmap=cmap, show_edges=True, **kwargs)

        # Set title if provided
        if title is not None:
            plotter.add_title(title)

        # Add axes
        plotter.add_axes()
        plotter.add_bounding_box()

        # Add scalar bar
        plotter.add_scalar_bar(title=pressure_field, position_x=0.1, position_y=0.1)

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(save_path), transparent_background=True)

        # Show if requested
        if show:
            plotter.show()

        return plotter
    else:
        # Matplotlib scatter plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot pressure as colored scatter points
        scatter = ax.scatter(
            points[:, 0], points[:, 1], c=pressure_data, cmap=cmap, s=10, alpha=0.8, **kwargs
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(pressure_field)

        # Set title if provided
        if title is not None:
            ax.set_title(title)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.5)

        # Add labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show if requested
        if show:
            plt.show()

        return fig


def plot_mesh(
    data: Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData", str, Path],
    color: str = "lightgray",
    edge_color: str = "black",
    show_nodes: bool = True,
    node_size: float = 5,
    title: Optional[str] = None,
    use_pyvista: Optional[bool] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Union[plt.Figure, "pv.Plotter"]:
    """
    Plot the mesh structure of vascular geometry.

    Args:
        data: Data to plot. Can be PyTorch Geometric Data, VTK object, or file path.
        color: Color for faces.
        edge_color: Color for edges.
        show_nodes: Whether to show mesh nodes.
        node_size: Size of nodes if shown.
        title: Plot title.
        use_pyvista: Whether to use PyVista (3D) or Matplotlib (2D).
                    If None, uses PyVista if available and 3D data is detected.
        show: Whether to show the plot.
        save_path: Optional path to save the plot.
        figsize: Figure size for matplotlib plots.
        **kwargs: Additional keyword arguments for the plotting functions.

    Returns:
        Figure or Plotter object.

    Raises:
        ImportError: If required libraries are not available.
        ValueError: If the data is invalid or incompatible.
    """
    # Determine whether to use PyVista or Matplotlib
    if use_pyvista is None:
        use_pyvista = HAS_PYVISTA

    # Convert data to appropriate format
    if isinstance(data, (str, Path)):
        # Load from file
        if str(data).endswith(".vtu"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTU files. Install it with 'pip install vtk'."
                )
            mesh, _, _ = extract_mesh_from_vtu(data)
            data = mesh
        elif str(data).endswith(".vtp"):
            if not HAS_VTK:
                raise ImportError(
                    "VTK is required for reading VTP files. Install it with 'pip install vtk'."
                )
            polydata, _, _ = extract_points_from_vtp(data)
            data = polydata
        else:
            raise ValueError(f"Unsupported file type: {data}")

    if use_pyvista:
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista is required for 3D visualization. Install it with 'pip install pyvista'."
            )

        # Create plotter
        plotter = pv.Plotter()

        # Convert to PyVista mesh
        if isinstance(data, Data):
            # Create from PyTorch Geometric Data
            points = data.pos.detach().cpu().numpy()

            if hasattr(data, "edge_index") and data.edge_index is not None:
                edges = data.edge_index.detach().cpu().numpy().T

                # Create lines for each edge
                lines = []
                for edge in edges:
                    lines.append([2, edge[0], edge[1]])

                # Create PyVista mesh
                mesh = pv.PolyData(points, lines=lines)
            else:
                # Just points, no connectivity
                mesh = pv.PolyData(points)
        elif HAS_VTK and isinstance(data, (vtk.vtkUnstructuredGrid, vtk.vtkPolyData)):
            # Convert VTK object to PyVista mesh
            mesh = pv.wrap(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Add mesh to the plotter
        plotter.add_mesh(
            mesh, color=color, show_edges=True, edge_color=edge_color, style="surface", **kwargs
        )

        # Add points if requested
        if show_nodes:
            plotter.add_points(mesh.points, color="red", point_size=node_size, **kwargs)

        # Set title if provided
        if title is not None:
            plotter.add_title(title)

        # Add axes
        plotter.add_axes()
        plotter.add_bounding_box()

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(save_path), transparent_background=True)

        # Show if requested
        if show:
            plotter.show()

        return plotter
    else:
        # Matplotlib plot
        fig, ax = plt.subplots(figsize=figsize)

        # Extract points and edges
        if isinstance(data, Data):
            # PyTorch Geometric Data
            points = data.pos.detach().cpu().numpy()

            if hasattr(data, "edge_index") and data.edge_index is not None:
                edges = data.edge_index.detach().cpu().numpy().T
            else:
                edges = None
        elif HAS_VTK and isinstance(data, vtk.vtkUnstructuredGrid):
            # VTK UnstructuredGrid
            points = vtk_to_numpy(data.GetPoints().GetData())
            edges = _extract_edges_from_vtk(data)
        elif HAS_VTK and isinstance(data, vtk.vtkPolyData):
            # VTK PolyData
            points = vtk_to_numpy(data.GetPoints().GetData())
            edges = _extract_edges_from_polydata(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Plot edges
        if edges is not None and len(edges) > 0:
            for edge in edges:
                ax.plot(
                    [points[edge[0], 0], points[edge[1], 0]],
                    [points[edge[0], 1], points[edge[1], 1]],
                    color=edge_color,
                    linewidth=1,
                    zorder=1,
                )

        # Plot nodes
        if show_nodes:
            ax.scatter(points[:, 0], points[:, 1], color="red", s=node_size, zorder=2)

        # Set title if provided
        if title is not None:
            ax.set_title(title)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.5)

        # Add labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show if requested
        if show:
            plt.show()

        return fig


def plot_comparison(
    data_list: List[Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData", str, Path]],
    titles: Optional[List[str]] = None,
    scalar_field: Optional[str] = None,
    cmap: str = "viridis",
    use_pyvista: Optional[bool] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
    **kwargs,
) -> Union[plt.Figure, List["pv.Plotter"]]:
    """
    Plot a comparison of multiple vascular geometries.

    Args:
        data_list: List of data objects to compare.
        titles: List of titles for each subplot.
        scalar_field: Optional scalar field to visualize.
        cmap: Colormap for scalar field visualization.
        use_pyvista: Whether to use PyVista (3D) or Matplotlib (2D).
                    If None, uses PyVista if available and 3D data is detected.
        show: Whether to show the plot.
        save_path: Optional path to save the plot.
        figsize: Figure size for matplotlib plots.
        **kwargs: Additional keyword arguments for the plotting functions.

    Returns:
        Figure or list of Plotter objects.

    Raises:
        ImportError: If required libraries are not available.
        ValueError: If inputs are invalid or incompatible.
    """
    n_plots = len(data_list)

    if n_plots == 0:
        raise ValueError("Empty data list")

    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(n_plots)]
    elif len(titles) < n_plots:
        titles.extend([f"Dataset {i+1}" for i in range(len(titles), n_plots)])

    # Determine whether to use PyVista or Matplotlib
    if use_pyvista is None:
        use_pyvista = HAS_PYVISTA

    if use_pyvista:
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista is required for 3D visualization. Install it with 'pip install pyvista'."
            )

        plotters = []

        for i, (data, title) in enumerate(zip(data_list, titles)):
            if scalar_field is not None:
                # Use pressure plot for scalar fields
                plotter = plot_pressure(
                    data=data,
                    pressure_field=scalar_field,
                    cmap=cmap,
                    title=title,
                    use_pyvista=True,
                    show=False,  # Don't show individual plots
                    **kwargs,
                )
            else:
                # Use geometry plot
                plotter = plot_geometry(
                    data=data,
                    title=title,
                    use_pyvista=True,
                    show=False,  # Don't show individual plots
                    **kwargs,
                )

            plotters.append(plotter)

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a combined image using matplotlib
            fig, axes = plt.subplots(1, n_plots, figsize=figsize)

            for i, plotter in enumerate(plotters):
                # Get screenshot from plotter
                img = plotter.screenshot(return_img=True)

                # Display in matplotlib
                if n_plots == 1:
                    ax = axes
                else:
                    ax = axes[i]

                ax.imshow(img)
                ax.set_title(titles[i])
                ax.axis("off")

            plt.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Show if requested
        if show:
            for plotter in plotters:
                plotter.show()

        return plotters
    else:
        # Matplotlib subplot
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        for i, (data, title) in enumerate(zip(data_list, titles)):
            # Get the correct axis
            if n_plots == 1:
                ax = axes
            else:
                ax = axes[i]

            if scalar_field is not None:
                # Use pressure plot for scalar fields (without showing)
                plot_pressure(
                    data=data,
                    pressure_field=scalar_field,
                    cmap=cmap,
                    title=title,
                    ax=ax,
                    use_pyvista=False,
                    show=False,  # Don't show individual plots
                    **kwargs,
                )
            else:
                # Use geometry plot (without showing)
                plot_geometry(
                    data=data,
                    title=title,
                    ax=ax,
                    use_pyvista=False,
                    show=False,  # Don't show individual plots
                    **kwargs,
                )

        plt.tight_layout()

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show if requested
        if show:
            plt.show()

        return fig


def create_animation(
    data_list: List[Union[Data, "vtk.vtkUnstructuredGrid", "vtk.vtkPolyData", str, Path]],
    output_file: Union[str, Path],
    scalar_field: Optional[str] = None,
    cmap: str = "viridis",
    fps: int = 10,
    loop: bool = True,
    title_template: Optional[str] = "Timestep {i}",
    use_pyvista: Optional[bool] = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100,
    **kwargs,
) -> str:
    """
    Create an animation from a sequence of vascular geometries.

    Args:
        data_list: List of data objects for each frame.
        output_file: Path to save the animation (GIF or MP4).
        scalar_field: Optional scalar field to visualize.
        cmap: Colormap for scalar field visualization.
        fps: Frames per second.
        loop: Whether to loop the animation.
        title_template: Template for frame titles.
        use_pyvista: Whether to use PyVista (3D) or Matplotlib (2D).
                    If None, uses PyVista if available and 3D data is detected.
        figsize: Figure size in inches.
        dpi: Resolution in dots per inch.
        **kwargs: Additional keyword arguments for the plotting functions.

    Returns:
        Path to the created animation file.

    Raises:
        ImportError: If required libraries are not available.
        ValueError: If inputs are invalid or incompatible.
    """
    # Check if we have enough data
    n_frames = len(data_list)

    if n_frames == 0:
        raise ValueError("Empty data list")

    # Convert output path to Path object
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    output_format = output_file.suffix.lower()

    if output_format not in [".gif", ".mp4"]:
        raise ValueError(f"Unsupported output format: {output_format}. Use .gif or .mp4")

    # Determine whether to use PyVista or Matplotlib
    if use_pyvista is None:
        use_pyvista = HAS_PYVISTA

    try:
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    except ImportError:
        raise ImportError("Matplotlib animation is required for creating animations.")

    if use_pyvista:
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista is required for 3D visualization. Install it with 'pip install pyvista'."
            )

        # We'll use PyVista for rendering but Matplotlib for animation
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")  # Hide axes

        # Function to update the frame
        def update(frame):
            ax.clear()
            ax.axis("off")

            # Set title if template is provided
            if title_template is not None:
                title = title_template.format(i=frame)
                ax.set_title(title)

            # Create PyVista plotter and render to image
            if scalar_field is not None:
                plotter = plot_pressure(
                    data=data_list[frame],
                    pressure_field=scalar_field,
                    cmap=cmap,
                    title=None,  # Title is handled by matplotlib
                    use_pyvista=True,
                    show=False,
                    **kwargs,
                )
            else:
                plotter = plot_geometry(
                    data=data_list[frame],
                    title=None,  # Title is handled by matplotlib
                    use_pyvista=True,
                    show=False,
                    **kwargs,
                )

            # Get screenshot from plotter
            img = plotter.screenshot(return_img=True)

            # Display in matplotlib
            ax.imshow(img)
            plotter.close()

            return (ax,)

        # Create animation
        anim = FuncAnimation(fig, update, frames=n_frames, blit=True, repeat=loop)

        # Save animation
        if output_format == ".gif":
            writer = PillowWriter(fps=fps)
        else:  # .mp4
            writer = FFMpegWriter(fps=fps)

        anim.save(output_file, writer=writer, dpi=dpi)
        plt.close(fig)

    else:
        # Matplotlib animation
        fig, ax = plt.subplots(figsize=figsize)

        # Function to update the frame
        def update(frame):
            ax.clear()

            # Set title if template is provided
            if title_template is not None:
                title = title_template.format(i=frame)
            else:
                title = None

            # Draw the frame
            if scalar_field is not None:
                plot_pressure(
                    data=data_list[frame],
                    pressure_field=scalar_field,
                    cmap=cmap,
                    title=title,
                    ax=ax,
                    use_pyvista=False,
                    show=False,
                    **kwargs,
                )
            else:
                plot_geometry(
                    data=data_list[frame],
                    title=title,
                    ax=ax,
                    use_pyvista=False,
                    show=False,
                    **kwargs,
                )

            return (ax,)

        # Create animation
        anim = FuncAnimation(fig, update, frames=n_frames, blit=True, repeat=loop)

        # Save animation
        if output_format == ".gif":
            writer = PillowWriter(fps=fps)
        else:  # .mp4
            writer = FFMpegWriter(fps=fps)

        anim.save(output_file, writer=writer, dpi=dpi)
        plt.close(fig)

    return str(output_file)
