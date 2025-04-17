"""
Transformation utilities for vascular simulation data.

This module provides transformation classes for preprocessing and augmenting
vascular simulation data in PyTorch Geometric format.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Normalize(BaseTransform):
    """
    Normalize node positions to the range [0, 1].

    This transform normalizes the node positions in a graph to lie within
    the unit cube, preserving the aspect ratio.

    Attributes:
        attribute (str): The attribute to normalize (default: 'pos').
    """

    def __init__(self, attribute: str = "pos"):
        """
        Initialize the normalizer.

        Args:
            attribute: The attribute to normalize.
        """
        self.attribute = attribute

    def __call__(self, data: Data) -> Data:
        """
        Apply the normalization transform.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Transformed Data object.
        """
        if not hasattr(data, self.attribute):
            return data

        attr = getattr(data, self.attribute)

        if attr is None or attr.nelement() == 0:
            return data

        min_val = attr.min(dim=0)[0]
        max_val = attr.max(dim=0)[0]

        # Handle case where min == max
        scale = max_val - min_val
        scale[scale == 0] = 1.0

        setattr(data, self.attribute, (attr - min_val) / scale)

        return data


class AddNoise(BaseTransform):
    """
    Add Gaussian noise to node features.

    This transform adds random Gaussian noise to specified node features,
    which can be useful for data augmentation or robustness testing.

    Attributes:
        attributes (List[str]): List of attributes to add noise to.
        std (float): Standard deviation of the noise.
        normalize (bool): Whether to normalize after adding noise.
    """

    def __init__(self, attributes: List[str] = ["pos"], std: float = 0.01, normalize: bool = True):
        """
        Initialize the noise transform.

        Args:
            attributes: List of attributes to add noise to.
            std: Standard deviation of the noise.
            normalize: Whether to normalize after adding noise.
        """
        self.attributes = attributes
        self.std = std
        self.normalize = normalize

    def __call__(self, data: Data) -> Data:
        """
        Apply the noise transform.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Transformed Data object.
        """
        for attr_name in self.attributes:
            if hasattr(data, attr_name):
                attr = getattr(data, attr_name)

                if attr is not None and attr.nelement() > 0:
                    # Add Gaussian noise
                    noise = torch.randn_like(attr) * self.std
                    noisy_attr = attr + noise

                    # Normalize if requested
                    if self.normalize and attr_name == "pos":
                        min_val = noisy_attr.min(dim=0)[0]
                        max_val = noisy_attr.max(dim=0)[0]
                        scale = max_val - min_val
                        scale[scale == 0] = 1.0
                        noisy_attr = (noisy_attr - min_val) / scale

                    setattr(data, attr_name, noisy_attr)

        return data


class RemoveFeatures(BaseTransform):
    """
    Remove specified features from the data.

    This transform removes specified features from the Data object,
    which can be useful for ablation studies or reducing memory usage.

    Attributes:
        features (List[str]): List of features to remove.
    """

    def __init__(self, features: List[str]):
        """
        Initialize the feature removal transform.

        Args:
            features: List of features to remove.
        """
        self.features = features

    def __call__(self, data: Data) -> Data:
        """
        Apply the feature removal transform.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Transformed Data object.
        """
        for feature in self.features:
            if hasattr(data, feature):
                delattr(data, feature)

        return data


class SelectFeatures(BaseTransform):
    """
    Select only specified features, removing all others.

    This transform keeps only the specified features in the Data object,
    removing all others. This can be useful for reducing memory usage
    or focusing on specific aspects of the data.

    Attributes:
        features (List[str]): List of features to keep.
    """

    def __init__(self, features: List[str]):
        """
        Initialize the feature selection transform.

        Args:
            features: List of features to keep.
        """
        self.features = features

    def __call__(self, data: Data) -> Data:
        """
        Apply the feature selection transform.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Transformed Data object.
        """
        # Create a new Data object with only the selected features
        new_data = Data()

        for feature in self.features:
            if hasattr(data, feature):
                setattr(new_data, feature, getattr(data, feature))

        # Always include edge_index and batch if present
        for key in ["edge_index", "batch"]:
            if hasattr(data, key):
                setattr(new_data, key, getattr(data, key))

        return new_data


class DownsamplePoints(BaseTransform):
    """
    Downsample the number of points in a graph.

    This transform reduces the number of nodes in a graph by random sampling,
    which can be useful for reducing computational requirements.

    Attributes:
        ratio (float): Ratio of points to keep (0.0 to 1.0).
        min_points (int): Minimum number of points to keep.
        preserve_edges (bool): Whether to update edge indices.
    """

    def __init__(self, ratio: float = 0.5, min_points: int = 10, preserve_edges: bool = True):
        """
        Initialize the downsampling transform.

        Args:
            ratio: Ratio of points to keep (0.0 to 1.0).
            min_points: Minimum number of points to keep.
            preserve_edges: Whether to update edge indices.
        """
        self.ratio = ratio
        self.min_points = min_points
        self.preserve_edges = preserve_edges

    def __call__(self, data: Data) -> Data:
        """
        Apply the downsampling transform.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Transformed Data object.
        """
        if not hasattr(data, "pos") or data.pos is None:
            return data

        num_nodes = data.pos.shape[0]
        num_samples = max(int(num_nodes * self.ratio), self.min_points)
        num_samples = min(num_samples, num_nodes)

        if num_samples == num_nodes:
            return data

        # Random sampling of indices
        idx = torch.randperm(num_nodes)[:num_samples]
        idx, _ = torch.sort(idx)

        # Create mapping for edge indices
        if self.preserve_edges and hasattr(data, "edge_index") and data.edge_index is not None:
            node_map = torch.full((num_nodes,), -1, dtype=torch.long)
            node_map[idx] = torch.arange(num_samples)

        # Update node features
        new_data = Data()
        for key, value in data:
            if key == "pos":
                new_data.pos = data.pos[idx]
            elif key == "edge_index" and self.preserve_edges:
                # Get edges where both endpoints are in the sampled nodes
                mask = torch.isin(data.edge_index[0], idx) & torch.isin(data.edge_index[1], idx)
                new_edge_index = data.edge_index[:, mask]

                # Remap node indices
                new_edge_index = node_map[new_edge_index]
                new_data.edge_index = new_edge_index
            elif key.startswith("node_") or value.shape[0] == num_nodes:
                # Update node features
                new_data[key] = value[idx]
            else:
                # Keep other features unchanged
                new_data[key] = value

        return new_data


class ComputeEdgeFeatures(BaseTransform):
    """
    Compute edge features from node features.

    This transform computes edge features based on the connected nodes,
    which can provide additional information for graph neural networks.

    Attributes:
        node_features (List[str]): List of node features to use.
        edge_features (List[str]): List of edge features to compute.
        aggregations (List[str]): List of aggregation methods.
    """

    def __init__(
        self,
        node_features: List[str] = ["pos"],
        edge_features: Optional[List[str]] = None,
        aggregations: List[str] = ["diff", "mean", "max"],
    ):
        """
        Initialize the edge feature computation.

        Args:
            node_features: List of node features to use.
            edge_features: List of edge features to compute.
            aggregations: List of aggregation methods.
        """
        self.node_features = node_features
        self.edge_features = edge_features or [f"edge_{f}" for f in node_features]
        self.aggregations = aggregations

    def __call__(self, data: Data) -> Data:
        """
        Apply the edge feature computation.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Transformed Data object.
        """
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        edge_index = data.edge_index

        for node_feat, edge_feat in zip(self.node_features, self.edge_features):
            if not hasattr(data, node_feat):
                continue

            node_attr = getattr(data, node_feat)

            if node_attr is None or node_attr.nelement() == 0:
                continue

            # Get features for source and target nodes
            src_features = node_attr[edge_index[0]]
            tgt_features = node_attr[edge_index[1]]

            # Compute aggregations
            for agg in self.aggregations:
                if agg == "diff":
                    # Compute difference (can represent direction)
                    result = tgt_features - src_features
                    setattr(data, f"{edge_feat}_diff", result)
                elif agg == "mean":
                    # Compute mean
                    result = (src_features + tgt_features) / 2
                    setattr(data, f"{edge_feat}_mean", result)
                elif agg == "max":
                    # Compute max
                    result = torch.max(src_features, tgt_features)
                    setattr(data, f"{edge_feat}_max", result)
                elif agg == "min":
                    # Compute min
                    result = torch.min(src_features, tgt_features)
                    setattr(data, f"{edge_feat}_min", result)
                elif agg == "sum":
                    # Compute sum
                    result = src_features + tgt_features
                    setattr(data, f"{edge_feat}_sum", result)

        return data


class RandomRotation(BaseTransform):
    """
    Apply random rotation to node positions.

    This transform applies a random rotation to the node positions,
    which can be useful for data augmentation and rotation invariance.

    Attributes:
        dims (int): Number of dimensions to rotate in (2 or 3).
        degree_range (float): Maximum rotation in degrees.
    """

    def __init__(self, dims: int = 3, degree_range: float = 180.0):
        """
        Initialize the rotation transform.

        Args:
            dims: Number of dimensions to rotate in (2 or 3).
            degree_range: Maximum rotation in degrees.
        """
        self.dims = dims
        self.degree_range = degree_range

    def __call__(self, data: Data) -> Data:
        """
        Apply the rotation transform.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Transformed Data object.
        """
        if not hasattr(data, "pos") or data.pos is None:
            return data

        pos = data.pos

        if pos.shape[1] < self.dims:
            # Not enough dimensions, skip rotation
            return data

        if self.dims == 2:
            # 2D rotation
            theta = torch.rand(1) * 2 * np.pi * (self.degree_range / 360.0)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            rotation_matrix = torch.tensor(
                [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
            ).squeeze()

            # Apply rotation to first two dimensions
            rotated_pos = pos.clone()
            rotated_pos[:, :2] = torch.mm(pos[:, :2], rotation_matrix)
            data.pos = rotated_pos

        elif self.dims == 3:
            # 3D rotation - use Euler angles
            # Convert degree range to radians
            angle_range = np.pi * (self.degree_range / 180.0)

            # Random Euler angles
            alpha = torch.rand(1) * 2 * angle_range - angle_range
            beta = torch.rand(1) * 2 * angle_range - angle_range
            gamma = torch.rand(1) * 2 * angle_range - angle_range

            # Rotation matrices
            Rx = torch.tensor(
                [
                    [1, 0, 0],
                    [0, torch.cos(alpha), -torch.sin(alpha)],
                    [0, torch.sin(alpha), torch.cos(alpha)],
                ]
            ).squeeze()

            Ry = torch.tensor(
                [
                    [torch.cos(beta), 0, torch.sin(beta)],
                    [0, 1, 0],
                    [-torch.sin(beta), 0, torch.cos(beta)],
                ]
            ).squeeze()

            Rz = torch.tensor(
                [
                    [torch.cos(gamma), -torch.sin(gamma), 0],
                    [torch.sin(gamma), torch.cos(gamma), 0],
                    [0, 0, 1],
                ]
            ).squeeze()

            # Combined rotation
            R = torch.mm(torch.mm(Rz, Ry), Rx)

            # Apply rotation
            data.pos = torch.mm(pos, R.t())

        return data
