================================
VascuSim Documentation
================================

.. image:: https://img.shields.io/pypi/v/vascusim.svg
   :target: https://pypi.python.org/pypi/vascusim
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/l/vascusim.svg
   :target: https://github.com/biosimmlab/vascusim/blob/main/LICENSE
   :alt: License

.. image:: https://github.com/biosimmlab/vascusim/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/biosimmlab/vascusim/actions/workflows/tests.yml
   :alt: Tests Status

Dataset package for cardiovascular simulations
----------------------------------------------

VascuSim is a Python library for processing, streaming, and analyzing vascular simulation 
data stored in VTU/VTP format, with efficient conversion to PyTorch Geometric data formats 
for Graph Neural Network training.

Features
--------

* Efficient data loading and conversion from VTU/VTP formats to PyTorch Geometric
* Streaming datasets for handling large simulation collections
* Remote data access from NAS storage and Hugging Face datasets
* Comprehensive geometry processing utilities
* Interactive visualization tools for vascular structures
* Parallel processing capabilities for batch operations

Getting Started
--------------

.. toctree::
   :maxdepth: 1
   
   installation
   quickstart

User Guide
---------

.. toctree::
   :maxdepth: 1
   
   user_guide/index
   user_guide/data_formats
   user_guide/streaming
   user_guide/visualization

API Reference
------------

.. toctree::
   :maxdepth: 1
   
   api/index
   api/data
   api/io
   api/processing
   api/utils

Examples
--------

.. toctree::
   :maxdepth: 1
   
   examples/index
   examples/basic_usage
   examples/advanced_streaming

Development
----------

.. toctree::
   :maxdepth: 1
   
   development/index
   development/contributing
   development/github_workflows

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`