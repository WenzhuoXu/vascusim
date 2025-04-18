====================
I/O Module Reference
====================

The ``vascusim.io`` module provides utilities for reading, streaming, and caching vascular simulation data files.

File Formats
-----------

.. autofunction:: vascusim.io.read_vtu

.. autofunction:: vascusim.io.read_vtp

.. autofunction:: vascusim.io.read_metadata

.. autofunction:: vascusim.io.read_vtu_with_metadata

.. autofunction:: vascusim.io.read_vtp_with_metadata

.. autofunction:: vascusim.io.read_timestep

.. autofunction:: vascusim.io.write_metadata

Streaming
--------

.. autoclass:: vascusim.io.DataStreamer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.io.NASStreamer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vascusim.io.HuggingFaceStreamer
   :members:
   :undoc-members:
   :show-inheritance:

Cache Management
---------------

.. autoclass:: vascusim.io.CacheManager
   :members:
   :undoc-members:
   :show-inheritance:

VTK Utilities
------------

.. autofunction:: vascusim.io.extract_mesh_from_vtu

.. autofunction:: vascusim.io.extract_points_from_vtp

.. autofunction:: vascusim.io.extract_attributes

.. autofunction:: vascusim.io.convert_vtk_to_numpy

.. autofunction:: vascusim.io.save_as_vtu

.. autofunction:: vascusim.io.save_as_vtp

.. autofunction:: vascusim.io.merge_vtu_files

Module Contents
-------------

.. automodule:: vascusim.io
   :members:
   :undoc-members:
   :show-inheritance: