==========
Quickstart
==========

This guide will help you get started with VascuSim quickly. We'll cover basic usage patterns
including loading data, visualization, and basic processing.

Loading VTU/VTP Files
---------------------

VascuSim provides easy functions to load VTU/VTP files and convert them to PyTorch Geometric
format:

.. code-block:: python

    import torch
    from vascusim.data import vtu_to_pyg, vtp_to_pyg
    
    # Load a VTU file
    data = vtu_to_pyg('path/to/simulation.vtu')
    
    # Or load a VTP file
    data = vtp_to_pyg('path/to/surface.vtp')
    
    # Now you have a PyTorch Geometric Data object
    print(data)
    print(f"Number of nodes: {data.pos.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1]//2}")

Using Datasets
-------------

For handling multiple files, you can use VascuSim's dataset classes:

.. code-block:: python
    
    from vascusim.data import VascuDataset, StreamingVascuDataset
    
    # Create a dataset for loading local files
    dataset = VascuDataset(
        source_url='path/to/dataset', 
        streaming_type='auto'  # Automatically detect streaming type
    )
    
    # Access a sample
    data = dataset[0]
    print(data)
    
    # Iterate over all samples
    for i, data in enumerate(dataset):
        print(f"Sample {i}: {data.pos.shape[0]} nodes")

Streaming from Remote Sources
----------------------------

VascuSim supports streaming data from remote sources like NAS storage or Hugging Face:

.. code-block:: python
    
    # Stream from Hugging Face
    hf_dataset = StreamingVascuDataset(
        source_url='huggingface/repo-name',
        streaming_type='hf',
        prefetch=True,  # Enable background prefetching
        prefetch_size=5  # Prefetch 5 samples ahead
    )
    
    # Stream from NAS storage
    nas_dataset = StreamingVascuDataset(
        source_url='192.168.1.100',  # NAS IP address
        username='username',
        password='password',
        streaming_type='nas',
        access_mode='api'  # Use NAS API for access
    )

Visualizing Vascular Geometry
----------------------------

VascuSim provides visualization tools for inspecting vascular geometry:

.. code-block:: python
    
    from vascusim.utils.visualization import plot_geometry, plot_pressure, plot_flow
    
    # Basic geometry visualization
    plot_geometry(data)
    
    # Visualize pressure field
    plot_pressure(data, pressure_field='pressure')
    
    # Visualize flow field
    plot_flow(data, flow_field='velocity')
    
    # Save visualization to file
    plot_geometry(data, save_path='vessel_geometry.png')

Processing Vascular Geometry
---------------------------

VascuSim includes utilities for processing vascular geometry:

.. code-block:: python
    
    from vascusim.processing import normalize_geometry, resample_geometry, filter_noise
    from vascusim.processing import compute_curvature, extract_centerline
    
    # Normalize geometry to [0,1] range
    normalized_data = normalize_geometry(data)
    
    # Resample to target number of points
    resampled_data = resample_geometry(data, target_points=1000)
    
    # Apply noise filtering
    filtered_data = filter_noise(data, method='gaussian', strength=0.5)
    
    # Compute geometric features
    curvature = compute_curvature(data)
    
    # Extract vessel centerline
    centerline = extract_centerline(data)

Working with PyTorch Geometric
-----------------------------

Since VascuSim converts data to PyTorch Geometric format, you can easily use it with PyG's
existing functionality:

.. code-block:: python
    
    from torch_geometric.transforms import NormalizeScale
    from torch_geometric.utils import degree
    from torch_geometric.loader import DataLoader
    
    # Apply PyG transforms
    transform = NormalizeScale()
    transformed_data = transform(data)
    
    # Use PyG utilities
    node_degrees = degree(data.edge_index[0], data.num_nodes)
    
    # Create a dataloader for batch processing
    loader = DataLoader([data1, data2, data3], batch_size=2)
    
    # Iterate over batches
    for batch in loader:
        print(batch)

Next Steps
---------

This quickstart guide covered the basics of using VascuSim. For more detailed information,
check out the following resources:

- :doc:`user_guide/index` - Detailed user guides for each component
- :doc:`api/index` - Complete API reference
- :doc:`examples/index` - Example scripts and tutorials