============
Installation
============

VascuSim requires Python 3.8 or newer. It is recommended to install VascuSim in a virtual environment.

Basic Installation
-----------------

You can install VascuSim from PyPI:

.. code-block:: bash

    pip install vascusim

This will install VascuSim with its core dependencies.

Installing with Optional Dependencies
------------------------------------

VascuSim provides optional dependencies for development and documentation:

Development Tools
~~~~~~~~~~~~~~~~

To install development tools (testing, linting, type checking):

.. code-block:: bash

    pip install vascusim[dev]

Documentation
~~~~~~~~~~~~

To install dependencies for building documentation:

.. code-block:: bash

    pip install vascusim[docs]

Installing from Source
---------------------

To install the latest development version from source:

.. code-block:: bash

    git clone https://github.com/biosimmlab/vascusim.git
    cd vascusim
    pip install -e .

Dependencies
-----------

Core Dependencies
~~~~~~~~~~~~~~~~

- numpy>=1.20.0
- torch>=1.10.0
- torch-geometric>=2.0.0
- requests>=2.25.0
- vtk>=9.0.0
- matplotlib>=3.5.0
- tqdm>=4.60.0
- pyvista>=0.34.0
- huggingface_hub>=0.10.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

For NAS streaming:
    - pysmb

For documentation:
    - sphinx>=4.4
    - sphinx-rtd-theme>=1.0
    - sphinx-copybutton>=0.5
    - sphinx-autodoc-typehints>=1.15

For development:
    - pytest>=6.0
    - pytest-cov>=2.12
    - black>=22.0
    - isort>=5.10
    - flake8>=4.0
    - mypy>=0.9

System Requirements
------------------

For visualization features, VTK and PyVista are required. These packages might have additional system dependencies depending on your operating system.

**Linux (Ubuntu/Debian):**

.. code-block:: bash

    sudo apt-get install python3-dev libgl1-mesa-dev xvfb

**macOS:**

VTK and PyVista should work with no additional dependencies.

**Windows:**

VTK and PyVista should work with no additional dependencies.

Verifying Installation
---------------------

After installation, you can verify that VascuSim is working correctly by running:

.. code-block:: python

    import vascusim
    print(vascusim.__version__)

This should print the installed version of VascuSim.