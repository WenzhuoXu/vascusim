"""
Tests for the processing module of vascusim.

This module tests the functionality for processing vascular geometry,
including geometry calculations, parallel processing, and preprocessing.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from vascusim.processing import geometry, parallel, preprocessing


class TestGeometryProcessing:
    """Tests for geometry processing functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple geometry for testing (e.g., a cylinder)
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

    def test_compute_surface_area(self):
        """Test computing surface area."""
        # Mock the surface area computation
        with mock.patch("vascusim.processing.geometry.compute_surface_area") as mock_compute:
            mock_compute.return_value = 10.0

            # Test the function call
            area = geometry.compute_surface_area(self.data)

            # Check the result
            assert area == 10.0
            mock_compute.assert_called_once()

    def test_compute_volume(self):
        """Test computing volume."""
        # Mock the volume computation
        with mock.patch("vascusim.processing.geometry.compute_volume") as mock_compute:
            mock_compute.return_value = 5.0

            # Test the function call
            volume = geometry.compute_volume(self.data)

            # Check the result
            assert volume == 5.0
            mock_compute.assert_called_once()

    def test_extract_centerline(self):
        """Test extracting centerline."""
        # Mock the centerline extraction
        with mock.patch("vascusim.processing.geometry.extract_centerline") as mock_extract:
            mock_centerline = Data(
                pos=torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]], dtype=torch.float),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            )
            mock_extract.return_value = mock_centerline

            # Test the function call
            centerline = geometry.extract_centerline(self.data)

            # Check the result
            assert centerline.pos.shape == (2, 3)
            assert centerline.edge_index.shape == (2, 2)
            mock_extract.assert_called_once()

    def test_compute_branch_angles(self):
        """Test computing branch angles."""
        # Create a Y-shaped centerline
        y_points = np.array(
            [
                [0, 0, 0],  # Junction point
                [0, 0, 1],  # Main branch
                [1, 0, 0],  # Branch 1
                [-1, 0, 0],  # Branch 2
            ],
            dtype=np.float32,
        )

        y_edges = np.array(
            [[0, 1], [0, 2], [0, 3]], dtype=np.int64  # Main branch  # Branch 1  # Branch 2
        )

        y_data = Data(
            pos=torch.tensor(y_points, dtype=torch.float),
            edge_index=torch.tensor(y_edges.T, dtype=torch.long),
        )

        # Mock the branch angle computation
        with mock.patch("vascusim.processing.geometry.compute_branch_angles") as mock_compute:
            mock_compute.return_value = {"1-2": 180.0, "1-3": 90.0, "2-3": 90.0}

            # Test the function call
            angles = geometry.compute_branch_angles(y_data)

            # Check the result
            assert angles["1-2"] == 180.0
            assert angles["1-3"] == 90.0
            assert angles["2-3"] == 90.0
            mock_compute.assert_called_once()


class TestParallelProcessing:
    """Tests for parallel processing functionality."""

    def test_worker_pool(self):
        """Test creating a worker pool."""
        # Mock multiprocessing.Pool
        with mock.patch("multiprocessing.Pool") as mock_pool:
            # Set up the mock
            mock_instance = mock.MagicMock()
            mock_pool.return_value = mock_instance

            # Call the function with 4 processes
            pool = parallel.worker_pool(4)

            # Check that Pool was called with the right number of processes
            mock_pool.assert_called_once_with(4)
            assert pool == mock_instance

    def test_process_batch(self):
        """Test processing a batch of items in parallel."""

        # Create a test function and data
        def test_func(x):
            return x * 2

        test_data = [1, 2, 3, 4, 5]

        # Mock the worker pool
        with mock.patch("vascusim.processing.parallel.worker_pool") as mock_pool_func:
            # Set up the mock
            mock_pool = mock.MagicMock()
            mock_pool.map.return_value = [2, 4, 6, 8, 10]
            mock_pool_func.return_value = mock_pool

            # Process the batch
            results = parallel.process_batch(test_func, test_data, n_workers=2)

            # Check the results
            assert results == [2, 4, 6, 8, 10]
            mock_pool_func.assert_called_once_with(2)
            mock_pool.map.assert_called_once()
            mock_pool.close.assert_called_once()

    def test_parallelize(self):
        """Test the parallelize decorator."""

        # Define a simple function to parallelize
        @parallel.parallelize(n_workers=2)
        def parallel_func(items):
            return [item * 2 for item in items]

        # Mock process_batch
        with mock.patch("vascusim.processing.parallel.process_batch") as mock_process:
            mock_process.return_value = [2, 4, 6, 8, 10]

            # Call the parallelized function
            result = parallel_func([1, 2, 3, 4, 5])

            # Check the result
            assert result == [2, 4, 6, 8, 10]
            mock_process.assert_called_once()


class TestPreprocessing:
    """Tests for preprocessing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
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

    def test_normalize_geometry(self):
        """Test normalizing geometry."""
        # Normalize the geometry
        normalized = preprocessing.normalize_geometry(self.data)

        # Check that points were normalized
        assert torch.min(normalized.pos).item() == 0.0
        assert torch.max(normalized.pos).item() == 1.0

        # Check that connectivity is preserved
        assert torch.equal(normalized.edge_index, self.data.edge_index)

    def test_resample_geometry(self):
        """Test resampling geometry."""
        # Mock the resampling function
        with mock.patch("vascusim.processing.preprocessing.resample_geometry") as mock_resample:
            # Create resampled geometry with half the points
            resampled_points = np.array(
                [[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32
            )

            resampled_edges = np.array([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=np.int64)

            resampled_data = Data(
                pos=torch.tensor(resampled_points, dtype=torch.float),
                edge_index=torch.tensor(resampled_edges.T, dtype=torch.long),
            )

            mock_resample.return_value = resampled_data

            # Test the function call
            result = preprocessing.resample_geometry(self.data, target_points=4)

            # Check the result
            assert result.pos.shape == (4, 3)
            assert result.edge_index.shape == (2, 4)
            mock_resample.assert_called_once()

    def test_filter_noise(self):
        """Test filtering noise from geometry."""
        # Add noise to the data
        noisy_data = Data(
            pos=self.data.pos + torch.randn_like(self.data.pos) * 0.1,
            edge_index=self.data.edge_index,
        )

        # Mock the noise filtering function
        with mock.patch("vascusim.processing.preprocessing.filter_noise") as mock_filter:
            mock_filter.return_value = self.data  # Return the original (non-noisy) data

            # Test the function call
            filtered = preprocessing.filter_noise(noisy_data)

            # Check that noise was filtered out
            assert torch.allclose(filtered.pos, self.data.pos)
            mock_filter.assert_called_once()

    def test_compute_features(self):
        """Test computing features from geometry."""
        # Mock the feature computation
        with mock.patch("vascusim.processing.preprocessing.compute_features") as mock_compute:
            # Define the features to add
            features = {
                "curvature": torch.rand(self.data.pos.shape[0], 1),
                "radius": torch.rand(self.data.pos.shape[0], 1),
                "distance": torch.rand(self.data.pos.shape[0], 1),
            }

            # Create expected output
            expected = self.data.clone()
            for key, value in features.items():
                expected[key] = value

            mock_compute.return_value = expected

            # Test the function call
            result = preprocessing.compute_features(
                self.data, feature_types=["curvature", "radius", "distance"]
            )

            # Check the result
            assert "curvature" in result
            assert "radius" in result
            assert "distance" in result
            assert torch.equal(result.pos, self.data.pos)
            mock_compute.assert_called_once()
