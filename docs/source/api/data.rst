======================
Data Module Reference
======================

The ``vascusim.data`` module provides dataset implementations and conversion utilities for vascular simulation data.

Dataset Classes
--------------

.. autoclass:: vascusim.data.VascuDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.data.StreamingVascuDataset
   :members:
   :undoc-members:
   :show-inheritance:

Conversion Utilities
-------------------

.. autofunction:: vascusim.data.vtu_to_pyg

.. autofunction:: vascusim.data.vtp_to_pyg

.. autofunction:: vascusim.data.build_graph

Transforms
---------

.. autoclass:: vascusim.data.transforms.Normalize
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.data.transforms.AddNoise
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.data.transforms.RemoveFeatures
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.data.transforms.SelectFeatures
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.data.transforms.DownsamplePoints
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.data.transforms.ComputeEdgeFeatures
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.data.transforms.RandomRotation
   :members:
   :undoc-members:
   :show-inheritance:

Module Contents
--------------

.. automodule:: vascusim.data
   :members:
   :undoc-members:
   :show-inheritance: