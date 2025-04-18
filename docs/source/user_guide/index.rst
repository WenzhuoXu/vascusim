==========
User Guide
==========

This user guide provides detailed information on how to use VascuSim for various tasks
related to vascular simulation data processing, analysis, and visualization.

.. toctree::
   :maxdepth: 2
   
   data_formats
   streaming
   visualization

Introduction
-----------

VascuSim is designed to simplify working with vascular simulation data, with a focus on:

1. **Data Loading and Conversion**: Convert VTU/VTP files to PyTorch Geometric format
2. **Efficient Data Streaming**: Access large datasets without excessive memory usage
3. **Geometric Analysis**: Extract useful features from vascular geometries
4. **Interactive Visualization**: Explore simulation results visually

This guide covers these aspects in detail, providing concrete examples and best practices
for each use case.

Workflow Overview
---------------

A typical workflow with VascuSim might look like:

1. **Load Data**: Convert VTU/VTP files to PyTorch Geometric format
2. **Preprocess**: Normalize, filter, and prepare data for analysis
3. **Analyze**: Extract features, compute geometric properties
4. **Visualize**: Explore the data visually
5. **Export**: Save processed data or results

These steps may be performed in a different order or repeated as needed for your specific use case.

Which Guide to Read
-----------------

If you're interested in:

- **Understanding supported data formats**: Start with :doc:`data_formats`
- **Efficiently handling large datasets**: Read :doc:`streaming`
- **Visualizing vascular structures**: Explore :doc:`visualization`

Additional Resources
------------------

- Check out the :doc:`../examples/index` for complete usage examples
- Refer to the :doc:`../api/index` for detailed API documentation
- For development and contributing, see :doc:`../development/index`