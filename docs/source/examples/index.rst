========
Examples
========

This section provides examples of common tasks and workflows using VascuSim.
Each example is complete and runnable, demonstrating how to use different
components of the library effectively.

.. toctree::
   :maxdepth: 1
   
   basic_usage
   advanced_streaming

Basic Usage Example
-----------------

The :doc:`basic_usage` example demonstrates:

- Loading a VTU file and converting it to PyTorch Geometric format
- Basic preprocessing and normalization
- Extracting geometric features
- Visualizing the vessel geometry and flow field

Advanced Streaming Example
------------------------

The :doc:`advanced_streaming` example covers:

- Setting up streaming from different data sources (NAS, Hugging Face)
- Efficient caching and prefetching strategies
- Filtering data based on metadata criteria
- Parallel processing of streamed data

Example Code Structure
--------------------

All example scripts are available in the `examples/` directory of the VascuSim package.
You can run them directly or use them as templates for your own workflows.

The directory structure is:

.. code-block:: text

    examples/
    ├── basic_usage.py                # Basic usage example
    ├── advanced_streaming.py         # Advanced streaming features
    ├── data/                         # Sample data files
    │   ├── sample_vessel.vtu
    │   └── sample_vessel.json
    └── README.md                     # Instructions for running examples

Running the Examples
------------------

To run the examples, first ensure VascuSim is installed:

.. code-block:: bash

    pip install vascusim

Then, run an example script:

.. code-block:: bash

    # Run basic usage example
    python -m examples.basic_usage
    
    # Run advanced streaming example
    python -m examples.advanced_streaming

For examples that require sample data, the necessary files are included in
the `examples/data/` directory.

Example Output
------------

Most examples produce visualizations or console output. For examples that
generate visualizations, the figures are displayed and optionally saved
to files in the current directory.

Adding Your Own Examples
----------------------

If you develop an interesting workflow or application using VascuSim, consider
contributing it to the examples collection. See :doc:`../development/contributing`
for guidelines on contributing to the project.