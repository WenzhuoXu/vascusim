=============
API Reference
=============

This is the API reference for VascuSim, documenting all modules, classes, and functions
in the package.

.. toctree::
   :maxdepth: 2
   
   data
   io
   processing
   utils

Module Overview
-------------

VascuSim is organized into several modules:

- :doc:`data`: Dataset implementations and data conversion utilities
- :doc:`io`: I/O operations, streaming, and caching
- :doc:`processing`: Geometry processing and analysis
- :doc:`utils`: Visualization and metadata handling

Module Structure
--------------

VascuSim follows a modular structure:

.. code-block:: text

    vascusim/
    ├── __init__.py
    ├── _version.py
    ├── config.py
    ├── data/
    │   ├── __init__.py
    │   ├── conversion.py
    │   ├── dataset.py
    │   └── transforms.py
    ├── io/
    │   ├── __init__.py
    │   ├── cache.py
    │   ├── formats.py
    │   ├── streaming.py
    │   └── vtk_utils.py
    ├── processing/
    │   ├── __init__.py
    │   ├── geometry.py
    │   ├── parallel.py
    │   └── preprocessing.py
    └── utils/
        ├── __init__.py
        ├── metadata.py
        └── visualization.py

Top-Level Functions
-----------------

For convenience, VascuSim re-exports commonly used functions and classes at the top level.

Data Module
~~~~~~~~~~

.. autofunction:: vascusim.data.vtu_to_pyg
.. autofunction:: vascusim.data.vtp_to_pyg
.. autoclass:: vascusim.data.VascuDataset
.. autoclass:: vascusim.data.StreamingVascuDataset

I/O Module
~~~~~~~~~

.. autofunction:: vascusim.io.read_vtu
.. autofunction:: vascusim.io.read_vtp
.. autofunction:: vascusim.io.read_metadata
.. autoclass:: vascusim.io.NASStreamer
.. autoclass:: vascusim.io.HuggingFaceStreamer

Processing Module
~~~~~~~~~~~~~~~

.. autofunction:: vascusim.processing.compute_curvature
.. autofunction:: vascusim.processing.extract_centerline
.. autofunction:: vascusim.processing.normalize_geometry
.. autofunction:: vascusim.processing.filter_noise

Utils Module
~~~~~~~~~~

.. autofunction:: vascusim.utils.plot_geometry
.. autofunction:: vascusim.utils.plot_flow
.. autofunction:: vascusim.utils.plot_pressure
.. autofunction:: vascusim.utils.query_metadata