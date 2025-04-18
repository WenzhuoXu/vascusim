=================
Basic Usage Example
=================

This example demonstrates the basic functionality of VascuSim, including:

- Loading vascular simulation data
- Basic preprocessing
- Extracting geometric features
- Visualization

Loading Sample Data
-----------------

First, let's import the necessary modules and load a sample VTU file:

.. code-block:: python

    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    from vascusim.data import vtu_to_pyg
    from vascusim.processing import (
        normalize_geometry, compute_curvature, 
        extract_centerline, compute_branch_angles
    )
    from vascusim.utils.visualization import (
        plot_geometry, plot_pressure, plot_flow
    )
    
    # Path to sample data
    sample_file = os.path.join("examples", "data", "sample_vessel.vtu")
    
    # Load data and convert to PyTorch Geometric format
    data = vtu_to_pyg(sample_file)
    
    # Print basic information
    print(f"Loaded data: {data}")
    print(f"Number of nodes: {data.pos.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1] // 2}")  # Divide by 2 because edges are bidirectional
    
    # Print available attributes
    print("Available attributes:")
    for key in data.keys:
        if isinstance(data[key], torch.Tensor):
            print(f"  - {key}: {data[key].shape}")
        else:
            print(f"  - {key}: {type(data[key])}")

Basic Preprocessing
-----------------

Now let's perform some basic preprocessing:

.. code-block:: python

    # Normalize geometry to [0, 1] range
    normalized_data = normalize_geometry(data)
    print(f"Normalized data position range: {normalized_data.pos.min().item():.2f} to {normalized_data.pos.max().item():.2f}")
    
    # Visualize the normalized geometry
    plot_geometry(
        normalized_data,
        title="Normalized Geometry",
        show_points=True,
        point_size=10,
        save_path="normalized_geometry.png"
    )

Extracting Geometric Features
---------------------------

Let's compute some geometric features of the vascular model:

.. code-block:: python
    
    # Compute curvature at each node
    curvature = compute_curvature(normalized_data)
    normalized_data.curvature = curvature
    
    print(f"Curvature range: {curvature.min().item():.4f} to {curvature.max().item():.4f}")
    
    # Extract centerline
    centerline = extract_centerline(normalized_data)
    print(f"Centerline extracted: {centerline.pos.shape[0]} nodes, {centerline.edge_index.shape[1] // 2} edges")
    
    # Compute branch angles
    angles = compute_branch_angles(normalized_data)
    print(f"Branch angles: {len(angles)} branches")
    for branch, angle in angles.items():
        print(f"  - Branch {branch}: {angle:.2f} degrees")
    
    # Visualize centerline
    plot_geometry(
        centerline,
        title="Vessel Centerline",
        color="red",
        linewidth=2.0,
        show_points=True,
        point_size=15,
        save_path="centerline.png"
    )
    
    # Visualize curvature on the original geometry
    plot_pressure(
        normalized_data,
        pressure_field="curvature",
        title="Vessel Curvature",
        cmap="viridis",
        save_path="curvature.png"
    )

Visualizing Flow and Pressure
---------------------------

If the data contains flow or pressure fields, we can visualize them:

.. code-block:: python
    
    # Check if velocity field exists
    if hasattr(normalized_data, "node_velocity") or hasattr(normalized_data, "velocity"):
        # Get the correct attribute name
        velocity_field = "node_velocity" if hasattr(normalized_data, "node_velocity") else "velocity"
        
        # Visualize flow field
        plot_flow(
            normalized_data,
            flow_field=velocity_field,
            title="Flow Velocity",
            scale=0.1,
            density=0.5,
            color_by_magnitude=True,
            cmap="coolwarm",
            save_path="flow.png"
        )
    else:
        print("No velocity field found in the data.")
    
    # Check if pressure field exists
    if hasattr(normalized_data, "node_pressure") or hasattr(normalized_data, "pressure"):
        # Get the correct attribute name
        pressure_field = "node_pressure" if hasattr(normalized_data, "node_pressure") else "pressure"
        
        # Visualize pressure field
        plot_pressure(
            normalized_data,
            pressure_field=pressure_field,
            title="Pressure Field",
            cmap="plasma",
            save_path="pressure.png"
        )
    else:
        print("No pressure field found in the data.")

Creating a Side-by-Side Comparison
--------------------------------

Finally, let's create a side-by-side comparison of the original geometry and the centerline:

.. code-block:: python
    
    from vascusim.utils.visualization import plot_comparison
    
    # Create comparison plot
    plot_comparison(
        data_list=[normalized_data, centerline],
        titles=["Full Geometry", "Centerline"],
        scalar_field="curvature",  # Color by curvature
        cmap="viridis",
        save_path="comparison.png"
    )
    
    print("Example completed successfully!")
    print("Generated visualizations:")
    print("  - normalized_geometry.png")
    print("  - centerline.png")
    print("  - curvature.png")
    print("  - flow.png (if velocity field was present)")
    print("  - pressure.png (if pressure field was present)")
    print("  - comparison.png")

Complete Example Code
-------------------

The complete example code is available in the `examples/basic_usage.py` file in the VascuSim package.
You can run it directly with:

.. code-block:: bash

    python -m examples.basic_usage

Next Steps
---------

Now that you understand the basic functionality, you might want to:

- Try the :doc:`advanced_streaming` example to learn about handling larger datasets
- Explore the :doc:`../user_guide/index` for more detailed information
- Check the :doc:`../api/index` for complete API reference