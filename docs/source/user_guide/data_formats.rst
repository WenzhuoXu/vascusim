=================
Data Formats
=================

VascuSim supports various data formats for vascular simulation data. This guide explains the supported formats,
how they are structured, and how to work with them.

Supported File Formats
---------------------

VascuSim primarily works with the following file formats:

VTU (VTK Unstructured Grid)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The VTU format is used for storing unstructured grid data in VTK's XML format. In vascular simulations,
VTU files typically contain:

- 3D mesh of the vascular structure
- Point data (values at nodes)
- Cell data (values for mesh elements)
- Solution fields like pressure, velocity, wall shear stress, etc.

VTP (VTK Polydata)
~~~~~~~~~~~~~~~~~

The VTP format is used for storing polygonal data in VTK's XML format. In vascular simulations,
VTP files typically contain:

- Surface representations of vessels
- Centerlines
- Point data on the surface
- Cell data for surface elements

JSON Metadata
~~~~~~~~~~~~

VascuSim uses JSON files for storing metadata associated with simulation files. These files
typically have the same base name as the corresponding VTU/VTP file but with a `.json` extension.

Metadata typically includes:

- Simulation parameters
- Patient/model information
- Boundary conditions
- Time step information
- Physical properties (viscosity, density, etc.)

Data Structure in VTU/VTP Files
------------------------------

VTK Unstructured Grid (VTU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A VTU file contains:

- **Points**: 3D coordinates of mesh nodes
- **Cells**: Connectivity information (how points form elements)
- **Point Data**: Scalar or vector fields defined at points
- **Cell Data**: Scalar or vector fields defined on cells

Common point data fields in vascular simulations:

- Velocity vectors
- Pressure
- Wall displacement
- Wall shear stress

VTK Polydata (VTP)
~~~~~~~~~~~~~~~~~

A VTP file contains:

- **Points**: 3D coordinates of surface points
- **Polygons/Lines**: Connectivity information (how points form polygons or lines)
- **Point Data**: Values defined at points
- **Cell Data**: Values defined on polygons/lines

Common point data in vascular surface models:

- Surface normals
- Curvature
- Distance metrics
- Wall thickness

Converting Between Formats
------------------------

VascuSim provides utilities to convert between these formats and PyTorch Geometric Data objects:

VTU to PyTorch Geometric
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vascusim.data import vtu_to_pyg
    
    # Basic conversion
    data = vtu_to_pyg('simulation.vtu')
    
    # With specific attributes and normalization
    data = vtu_to_pyg(
        'simulation.vtu',
        attributes=['velocity', 'pressure'],
        include_cell_data=True,
        include_point_data=True,
        normalize=True
    )

VTP to PyTorch Geometric
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vascusim.data import vtp_to_pyg
    
    # Basic conversion
    data = vtp_to_pyg('surface.vtp')
    
    # With specific attributes and normalization
    data = vtp_to_pyg(
        'surface.vtp',
        attributes=['normals', 'curvature'],
        include_cell_data=False,  # Only include point data
        normalize=True
    )

Understanding the PyTorch Geometric Format
----------------------------------------

When VascuSim converts VTU/VTP files to PyTorch Geometric format, the data is structured as follows:

.. code-block:: python

    # PyTorch Geometric Data object structure
    Data(
        pos=[N, 3],                # Node positions (N nodes, 3D coordinates)
        edge_index=[2, E],         # Edge connectivity (2 x E edges)
        node_{attribute}=[N, D],   # Node attributes (from point data)
        cell_{attribute}=[M, D]    # Cell attributes (from cell data)
    )

Where:
- N is the number of nodes
- E is the number of edges
- M is the number of cells
- D is the dimension of the attribute (1 for scalars, 3 for vectors, etc.)

Node attributes from the original VTU/VTP file are prefixed with `node_` and cell attributes with `cell_`.

Metadata Handling
---------------

VascuSim automatically loads metadata from associated JSON files when available:

.. code-block:: python

    from vascusim.io import read_metadata
    
    # Load metadata directly
    metadata = read_metadata('simulation.json')
    
    # Metadata is automatically loaded when using datasets
    from vascusim.data import VascuDataset
    
    dataset = VascuDataset('path/to/data')
    sample = dataset[0]  # Sample has metadata attributes

Custom Data Loading
-----------------

For custom data handling, VascuSim provides the `build_graph` function:

.. code-block:: python

    from vascusim.data import build_graph
    import torch
    import numpy as np
    
    # Create a graph from custom node positions and edge connections
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T
    
    # Add custom node features
    node_features = {
        'pressure': np.array([1.0, 2.0, 3.0, 4.0]),
        'velocity': np.array([
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0]
        ])
    }
    
    # Build the graph
    data = build_graph(
        nodes=nodes,
        edges=edges,
        node_features=node_features
    )

This approach allows you to build graph data from any source, not just VTU/VTP files.