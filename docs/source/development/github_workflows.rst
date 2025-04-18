=================
GitHub Workflows
=================

VascuSim utilizes GitHub Actions for continuous integration and automated package publishing. This document explains the available workflows and how to use them.

Available Workflows
-----------------

The repository includes two primary workflows:

1. **Tests** (`tests.yml`): Runs the test suite and code quality checks
2. **Publish** (`publish.yml`): Publishes the package to PyPI when a new release is created

Tests Workflow
------------

The tests workflow runs automatically on:
- Every push to the `main` branch
- Every pull request targeting the `main` branch
- Manual trigger via GitHub Actions UI

What the Tests Workflow Does
~~~~~~~~~~~~~~~~~~~~~~~~~~

This workflow:
1. Runs tests on multiple operating systems (Linux, Windows, macOS)
2. Tests against multiple Python versions (3.8, 3.9, 3.10, 3.11)
3. Performs code quality checks using:
   - Black (code formatting)
   - isort (import sorting)
   - flake8 (style guide enforcement)
   - mypy (type checking)
4. Runs pytest with coverage reporting
5. Builds the package to verify build integrity
6. Uploads coverage reports to Codecov

How to Use It
~~~~~~~~~~~~

This workflow runs automatically, but you can also trigger it manually:

1. Go to the Actions tab in the repository
2. Select the "Tests" workflow
3. Click "Run workflow"
4. Select the branch to run tests on
5. Click "Run workflow"

Publish Workflow
--------------

The publish workflow runs automatically when:
- A new GitHub release is created
- Manual trigger via GitHub Actions UI

What the Publish Workflow Does