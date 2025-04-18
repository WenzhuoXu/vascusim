==========
Development
==========

This section provides information for developers who want to contribute to VascuSim
or build their own extensions.

.. toctree::
   :maxdepth: 2
   
   contributing
   github_workflows

Development Overview
------------------

VascuSim is designed to be modular and extensible. This guide will help you understand
the overall architecture, development workflow, and best practices for contributing to
the project.

Project Architecture
------------------

VascuSim follows a modular architecture:

- **vascusim.data**: Dataset implementations and data conversion utilities
- **vascusim.io**: I/O operations, streaming, and caching
- **vascusim.processing**: Geometry processing and analysis
- **vascusim.utils**: Visualization and metadata handling

Each module is designed to be as independent as possible, with well-defined interfaces
between them. This approach makes it easier to understand, test, and extend the package.

Development Environment
---------------------

Setting up a development environment for VascuSim:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/biosimmlab/vascusim.git
    cd vascusim
    
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install in development mode with development dependencies
    pip install -e ".[dev]"
    
    # Verify installation
    python -c "import vascusim; print(vascusim.__version__)"

With this setup, changes to the source code will be immediately reflected when you import
the package, without needing to reinstall.

Testing
------

VascuSim uses pytest for testing. To run the tests:

.. code-block:: bash

    # Run all tests
    python -m pytest tests/
    
    # Run tests with coverage
    python -m pytest tests/ --cov=vascusim
    
    # Run specific test
    python -m pytest tests/test_io.py

For more information on testing, see :doc:`contributing`.

Code Quality
----------

VascuSim uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Style guide enforcement
- **mypy**: Type checking

To ensure your code meets the project's quality standards:

.. code-block:: bash

    # Format code with Black
    black vascusim tests
    
    # Sort imports with isort
    isort vascusim tests
    
    # Check style with flake8
    flake8 vascusim tests
    
    # Type checking with mypy
    mypy vascusim

Documentation
------------

VascuSim uses Sphinx for documentation. To build the documentation locally:

.. code-block:: bash

    # Install documentation dependencies
    pip install -e ".[docs]"
    
    # Build the documentation
    cd docs
    make html
    
    # View the documentation
    # Open build/html/index.html in your web browser

For more information on documentation, see :doc:`contributing`.

Release Process
-------------

VascuSim follows a test-driven release process:

1. All tests must pass on all supported platforms
2. Documentation must be up to date
3. Code must pass quality checks
4. A new GitHub release is created with a proper tag
5. The GitHub Action automatically publishes to PyPI

For more information on the release process, see :doc:`github_workflows`.

Getting Help
----------

If you need help with development:

- Check the existing documentation
- Look at the tests for examples
- File an issue on GitHub
- Contact the maintainers

Contributing
----------

Contributions to VascuSim are welcome! Please see :doc:`contributing` for guidelines
on how to contribute effectively.