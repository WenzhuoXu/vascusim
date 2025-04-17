"""
Tests for the visualization module of vascusim.

This module tests the functionality for visualizing vascular geometry,
flow data, pressure fields, and other simulation results.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from vascusim.utils import visualization


class TestVisualizationUtils:
    """Tests for visualization utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for output files
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create a simple geometry for testing
        self.points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )

        self.edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            dtype=np.int64,
        )

        # Create PyTorch Geometric Data object
        self.data = Data(
            pos=torch.tensor(self.points, dtype=torch.float),
            edge_index=torch.tensor(self.edges.T, dtype=torch.long),
        )

        # Add dummy flow and pressure data
        self.data.velocity = torch.rand((8, 3), dtype=torch.float)
        self.data.pressure = torch.rand(8, dtype=torch.float)

    def teardown_method(self):
        """Tear down test fixtures."""
        # Close all plots
        plt.close("all")

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_check_visualization_libraries(self):
        """Test checking available visualization libraries."""
        libraries = visualization.check_visualization_libraries()

        # Matplotlib should always be available
        assert libraries["matplotlib"] is True

        # Other libraries depend on the environment
        assert "pyvista" in libraries
        assert "vtk" in libraries

    def test_plot_geometry_matplotlib(self):
        """Test plotting geometry with Matplotlib."""
        # Mock matplotlib functions to avoid actual plotting
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.savefig"):

            # Test the function call
            fig = visualization.plot_geometry(
                data=self.data,
                use_pyvista=False,  # Force Matplotlib
                show=False,
                save_path=self.temp_dir / "geometry.png",
            )

            # Check that a figure was created
            assert isinstance(fig, plt.Figure)

    def test_plot_flow_matplotlib(self):
        """Test plotting flow with Matplotlib."""
        # Mock matplotlib functions to avoid actual plotting
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.savefig"):

            # Test the function call
            fig = visualization.plot_flow(
                data=self.data,
                flow_field="velocity",
                use_pyvista=False,  # Force Matplotlib
                show=False,
                save_path=self.temp_dir / "flow.png",
            )

            # Check that a figure was created
            assert isinstance(fig, plt.Figure)

    def test_plot_pressure_matplotlib(self):
        """Test plotting pressure with Matplotlib."""
        # Mock matplotlib functions to avoid actual plotting
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.savefig"):

            # Test the function call
            fig = visualization.plot_pressure(
                data=self.data,
                pressure_field="pressure",
                use_pyvista=False,  # Force Matplotlib
                show=False,
                save_path=self.temp_dir / "pressure.png",
            )

            # Check that a figure was created
            assert isinstance(fig, plt.Figure)

    def test_plot_mesh_matplotlib(self):
        """Test plotting mesh with Matplotlib."""
        # Mock matplotlib functions to avoid actual plotting
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.savefig"):

            # Test the function call
            fig = visualization.plot_mesh(
                data=self.data,
                use_pyvista=False,  # Force Matplotlib
                show=False,
                save_path=self.temp_dir / "mesh.png",
            )

            # Check that a figure was created
            assert isinstance(fig, plt.Figure)

    def test_plot_comparison_matplotlib(self):
        """Test plotting comparison with Matplotlib."""
        # Create another dataset for comparison
        data2 = Data(
            pos=torch.tensor(self.points * 1.5, dtype=torch.float),
            edge_index=torch.tensor(self.edges.T, dtype=torch.long),
            pressure=torch.rand(8, dtype=torch.float),
        )

        # Mock matplotlib functions to avoid actual plotting
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.savefig"):

            # Test the function call
            fig = visualization.plot_comparison(
                data_list=[self.data, data2],
                titles=["Original", "Scaled"],
                use_pyvista=False,  # Force Matplotlib
                show=False,
                save_path=self.temp_dir / "comparison.png",
            )

            # Check that a figure was created
            assert isinstance(fig, plt.Figure)

    @pytest.mark.skipif(not visualization.HAS_PYVISTA, reason="PyVista is not available")
    def test_plot_geometry_pyvista(self):
        """Test plotting geometry with PyVista."""
        # Skip if PyVista is not available
        if not visualization.HAS_PYVISTA:
            pytest.skip("PyVista is not available")

        # Mock PyVista plotter methods to avoid actual plotting
        with mock.patch("pyvista.Plotter.show"), mock.patch("pyvista.Plotter.screenshot"):

            # Test the function call
            plotter = visualization.plot_geometry(
                data=self.data,
                use_pyvista=True,  # Force PyVista
                show=False,
                save_path=self.temp_dir / "geometry_pyvista.png",
            )

            # Check that a plotter was created
            assert hasattr(plotter, "add_mesh")

    @pytest.mark.skipif(not visualization.HAS_PYVISTA, reason="PyVista is not available")
    def test_plot_flow_pyvista(self):
        """Test plotting flow with PyVista."""
        # Skip if PyVista is not available
        if not visualization.HAS_PYVISTA:
            pytest.skip("PyVista is not available")

        # Mock PyVista plotter methods to avoid actual plotting
        with mock.patch("pyvista.Plotter.show"), mock.patch("pyvista.Plotter.screenshot"):

            # Test the function call
            plotter = visualization.plot_flow(
                data=self.data,
                flow_field="velocity",
                use_pyvista=True,  # Force PyVista
                show=False,
                save_path=self.temp_dir / "flow_pyvista.png",
            )

            # Check that a plotter was created
            assert hasattr(plotter, "add_mesh")

    @pytest.mark.skipif(not visualization.HAS_PYVISTA, reason="PyVista is not available")
    def test_plot_pressure_pyvista(self):
        """Test plotting pressure with PyVista."""
        # Skip if PyVista is not available
        if not visualization.HAS_PYVISTA:
            pytest.skip("PyVista is not available")

        # Mock PyVista plotter methods to avoid actual plotting
        with mock.patch("pyvista.Plotter.show"), mock.patch("pyvista.Plotter.screenshot"):

            # Test the function call
            plotter = visualization.plot_pressure(
                data=self.data,
                pressure_field="pressure",
                use_pyvista=True,  # Force PyVista
                show=False,
                save_path=self.temp_dir / "pressure_pyvista.png",
            )

            # Check that a plotter was created
            assert hasattr(plotter, "add_mesh")

    def test_create_animation(self):
        """Test creating an animation from a sequence of geometries."""
        # Create a sequence of geometries (moving cube)
        data_sequence = []
        for i in range(5):
            # Shift the points along the z-axis
            shifted_points = self.points.copy()
            shifted_points[:, 2] += i * 0.2

            # Create data object
            data = Data(
                pos=torch.tensor(shifted_points, dtype=torch.float),
                edge_index=torch.tensor(self.edges.T, dtype=torch.long),
                pressure=torch.rand(8, dtype=torch.float),
            )

            data_sequence.append(data)

        # Mock the animation creation to avoid actual rendering
        with mock.patch("matplotlib.animation.FuncAnimation"), mock.patch(
            "matplotlib.animation.PillowWriter"
        ), mock.patch("matplotlib.animation.FuncAnimation.save"):

            # Test the function call
            output_path = visualization.create_animation(
                data_list=data_sequence,
                output_file=self.temp_dir / "animation.gif",
                use_pyvista=False,  # Force Matplotlib
                fps=10,
                loop=True,
            )

            # Check that an output path was returned
            assert isinstance(output_path, str)
            assert output_path.endswith(".gif")

    def test_extract_edges_from_vtk(self):
        """Test extracting edges from VTK objects."""
        # Skip if VTK is not available
        if not visualization.HAS_VTK:
            pytest.skip("VTK is not available")

        # Create a mock VTK unstructured grid
        with mock.patch("vascusim.utils.visualization._extract_edges_from_vtk") as mock_extract:
            mock_extract.return_value = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

            # Create a dummy VTK object
            mock_vtk_grid = mock.MagicMock()

            # Test the function call
            edges = visualization._extract_edges_from_vtk(mock_vtk_grid)

            # Check the result
            assert isinstance(edges, np.ndarray)
            assert edges.shape == (4, 2)
            mock_extract.assert_called_once()

    def test_unsupported_data_type(self):
        """Test handling of unsupported data types."""
        # Try plotting with an unsupported data type
        with pytest.raises(ValueError):
            visualization.plot_geometry(data=123, use_pyvista=False)  # Not a supported data type


class TestVisualizationIntegration:
    """Integration tests for visualization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for output files
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create a simple mesh for testing
        # A triangular mesh of a pyramid
        self.points = np.array(
            [
                [0, 0, 0],  # Base vertices
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0.5, 0.5, 1],  # Top vertex
            ],
            dtype=np.float32,
        )

        # Triangular faces (0-indexed)
        self.faces = np.array(
            [
                [0, 1, 4],  # Side faces
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
                [0, 2, 1],  # Base triangles
                [0, 3, 2],
            ],
            dtype=np.int64,
        )

        # Create edge connectivity from faces
        self.edges = set()
        for face in self.faces:
            self.edges.add((face[0], face[1]))
            self.edges.add((face[1], face[2]))
            self.edges.add((face[2], face[0]))

        self.edges = np.array(list(self.edges), dtype=np.int64)

        # Create PyTorch Geometric Data object
        self.data = Data(
            pos=torch.tensor(self.points, dtype=torch.float),
            edge_index=torch.tensor(self.edges.T, dtype=torch.long),
        )

        # Add synthetic data attributes
        # Vertex-wise scalar (e.g., pressure)
        self.data.pressure = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.2], dtype=torch.float)

        # Vertex-wise vector (e.g., velocity)
        self.data.velocity = torch.tensor(
            [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.1], [0.0, 0.1, 0.1], [0.0, 0.0, 0.2]],
            dtype=torch.float,
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        # Close all plots
        plt.close("all")

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_visualization_pipeline(self):
        """Test the complete visualization pipeline."""
        # Skip actual rendering to speed up tests
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.savefig"):

            # 1. Basic geometry visualization
            fig1 = visualization.plot_geometry(
                data=self.data,
                use_pyvista=False,
                show=False,
                save_path=self.temp_dir / "1_geometry.png",
            )

            # 2. Pressure field visualization
            fig2 = visualization.plot_pressure(
                data=self.data,
                pressure_field="pressure",
                use_pyvista=False,
                show=False,
                save_path=self.temp_dir / "2_pressure.png",
            )

            # 3. Velocity field visualization
            fig3 = visualization.plot_flow(
                data=self.data,
                flow_field="velocity",
                use_pyvista=False,
                show=False,
                save_path=self.temp_dir / "3_velocity.png",
            )

            # 4. Mesh visualization
            fig4 = visualization.plot_mesh(
                data=self.data,
                use_pyvista=False,
                show=False,
                save_path=self.temp_dir / "4_mesh.png",
            )

            # Check that figures were created
            assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2, fig3, fig4])

    @pytest.mark.skipif(not visualization.HAS_PYVISTA, reason="PyVista is not available")
    def test_comparison_visualization(self):
        """Test comparison visualization of multiple datasets."""
        # Create a modified version of the data for comparison
        data2 = self.data.clone()
        data2.pos = data2.pos * 1.5  # Scale geometry
        data2.pressure = data2.pressure * 0.8  # Reduce pressure

        # Create another dataset with different topology
        points3 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 1]], dtype=np.float32)

        edges3 = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

        data3 = Data(
            pos=torch.tensor(points3, dtype=torch.float),
            edge_index=torch.tensor(edges3.T, dtype=torch.long),
            pressure=torch.tensor([0.2, 0.5, 1.0], dtype=torch.float),
        )

        # Skip actual rendering to speed up tests
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.savefig"):

            # Compare geometries
            fig1 = visualization.plot_comparison(
                data_list=[self.data, data2, data3],
                titles=["Original", "Scaled", "Triangle"],
                use_pyvista=False,
                show=False,
                save_path=self.temp_dir / "comparison_geometry.png",
            )

            # Compare pressure fields
            fig2 = visualization.plot_comparison(
                data_list=[self.data, data2, data3],
                titles=["Original", "Scaled", "Triangle"],
                scalar_field="pressure",
                use_pyvista=False,
                show=False,
                save_path=self.temp_dir / "comparison_pressure.png",
            )

            # Check that figures were created
            assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2])
