============
Contributing
============

Thank you for your interest in contributing to VascuSim! This guide will help you
understand how to contribute effectively to the project.

Getting Started
-------------

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a new branch** for your feature or bugfix
4. **Make your changes**
5. **Run tests** to ensure your changes work correctly
6. **Submit a pull request** to the main repository

Setting Up Development Environment
--------------------------------

Set up a development environment with all the necessary dependencies:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/YOUR_USERNAME/vascusim.git
    cd vascusim
    
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install in development mode with development dependencies
    pip install -e ".[dev]"

Code Style
---------

VascuSim follows the Black code style and uses isort for import sorting.
Before submitting a pull request, ensure your code adheres to these standards:

.. code-block:: bash

    # Format code with Black
    black vascusim tests
    
    # Sort imports with isort
    isort vascusim tests
    
    # Check style with flake8
    flake8 vascusim tests
    
    # Type checking with mypy
    mypy vascusim

To make this easier, you can use the provided ``run_test.py`` script:

.. code-block:: bash

    # Run all code quality checks and auto-fix issues
    python run_test.py --fix

Testing
------

VascuSim uses pytest for testing. All new features should include tests,
and all existing tests should pass before submitting a pull request:

.. code-block:: bash

    # Run all tests
    python -m pytest tests/
    
    # Run tests with coverage
    python -m pytest tests/ --cov=vascusim
    
    # Run specific test
    python -m pytest tests/test_io.py

You can also use the ``run_test.py`` script:

.. code-block:: bash

    # Run all tests with coverage
    python run_test.py --coverage

Documentation
------------

Documentation is an essential part of the project. When adding new features or changing
existing functionality, make sure to update the documentation accordingly:

1. **Docstrings**: Every module, class, and function should have a clear docstring
2. **Examples**: Include examples in the docstring when appropriate
3. **User Guide**: Update or add to the user guide if your changes affect user workflows
4. **API Reference**: Ensure new functions and classes are included in the API reference

To build the documentation locally:

.. code-block:: bash

    # Install documentation dependencies
    pip install -e ".[docs]"
    
    # Build the documentation
    cd docs
    make html
    
    # View the documentation
    # Open build/html/index.html in your web browser

Pull Request Process
------------------

1. **Create a new branch** for your feature or bugfix:

   .. code-block:: bash
   
       git checkout -b feature/your-feature-name
       # or
       git checkout -b fix/issue-description

2. **Make your changes** and commit them with clear, descriptive commit messages:

   .. code-block:: bash
   
       git add .
       git commit -m "feat: add new feature X"
       # or
       git commit -m "fix: resolve issue with Y"

   We follow the [Conventional Commits](https://www.conventionalcommits.org/) format
   for commit messages.

3. **Run tests and code quality checks**:

   .. code-block:: bash
   
       python run_test.py --coverage --type-check

4. **Push your changes** to your fork:

   .. code-block:: bash
   
       git push origin feature/your-feature-name

5. **Create a pull request** on GitHub with a clear description of the changes and any related issues

6. **Address review comments** if any are provided by maintainers

7. Once approved, a maintainer will merge your pull request

Types of Contributions
--------------------

There are many ways to contribute to VascuSim:

1. **Bug fixes**: Fixing issues in the existing code
2. **New features**: Adding new functionality to the package
3. **Documentation**: Improving the documentation
4. **Examples**: Adding new examples or improving existing ones
5. **Performance improvements**: Making the code faster or more memory-efficient
6. **Code quality**: Improving code structure, tests, or type annotations

Development Guidelines
--------------------

When developing for VascuSim, keep these guidelines in mind:

1. **Keep it modular**: Each component should have a clear, focused purpose
2. **Write tests**: Aim for high test coverage, especially for complex functionality
3. **Document everything**: Every public API should be well-documented
4. **Backward compatibility**: Avoid breaking changes to the public API
5. **Performance matters**: Be mindful of performance implications, especially for large datasets
6. **Usability first**: Make the API intuitive and consistent

Reporting Issues
--------------

If you encounter an issue with VascuSim, please report it on GitHub with:

1. A clear description of the issue
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Any relevant logs or error messages
6. System information (Python version, operating system, etc.)

Questions and Discussion
----------------------

For questions or discussions about VascuSim development:

1. Use GitHub Discussions for general questions
2. Join the BioSiMMLab community channels (if available)
3. Contact the maintainers directly for specific inquiries

Code of Conduct
-------------

Please be respectful and considerate in all interactions. We are committed to
providing a welcoming and inspiring community for all.

License
------

By contributing to VascuSim, you agree that your contributions will be licensed
under the project's MIT License.