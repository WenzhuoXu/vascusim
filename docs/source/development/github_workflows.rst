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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This workflow:
1. Builds the package (both wheel and source distribution)
2. Publishes the package to TestPyPI
3. Tests the installation from TestPyPI
4. When triggered by a release (not manual trigger), also:
   - Publishes the package to PyPI
   - Builds and deploys the documentation to GitHub Pages

How to Use It
~~~~~~~~~~~~

#### Automatic Release

1. Create a new release in GitHub:
   - Go to the Releases section
   - Click "Draft a new release"
   - Create a new tag (e.g., `v0.1.0`)
   - Add a title and description
   - Click "Publish release"
2. The workflow will run automatically

#### Manual Trigger

1. Go to the Actions tab in the repository
2. Select the "Publish" workflow
3. Click "Run workflow"
4. Optionally provide a version number (if left blank, the version from `vascusim._version` will be used)
5. Click "Run workflow"

When triggered manually, the workflow will publish to TestPyPI only, not to the main PyPI repository.

Workflow Configuration
-------------------

### Tests Workflow Configuration

The tests workflow can be configured by modifying the `.github/workflows/tests.yml` file:
- Change the Python versions in the matrix
- Change the operating systems in the matrix
- Add or remove code quality checks
- Modify the test command

### Publish Workflow Configuration

The publish workflow can be configured by modifying the `.github/workflows/publish.yml` file:
- Change the Python version used for building
- Add additional deployment steps
- Modify the documentation build process

Required Secrets
--------------

For the publish workflow to function correctly, the following GitHub secrets must be configured:

- `PYPI_API_TOKEN`: API token for PyPI (for publishing to the main PyPI repository)
- `TEST_PYPI_API_TOKEN`: API token for TestPyPI (for testing)

These are automatically handled by the GitHub Actions trusted publishing mechanism and don't need to be manually configured if your repository has the proper permissions.

Troubleshooting
-------------

If a workflow fails, check the following:

1. **Test failures**: Look at the specific test that failed and fix the code accordingly
2. **Code quality issues**: Run the quality tools locally to fix issues:
   ```bash
   black vascusim tests
   isort vascusim tests
   flake8 vascusim tests
   mypy vascusim
   ```
3. **Build failures**: Try building the package locally:
   ```bash
   python -m build
   ```
4. **Publishing failures**: Check that your PyPI credentials are correctly configured

Workflow Examples
--------------

### Example of Successful Tests Run

```
✓ Set up Python 3.10
✓ Install dependencies
✓ Check code style with black
✓ Check imports with isort
✓ Run flake8
✓ Type check with mypy
✓ Run tests (23 passed, 100% coverage)
✓ Build package
✓ Upload coverage to Codecov
```

### Example of Successful Publish Run

```
✓ Set up Python 3.10
✓ Install dependencies
✓ Build package
✓ Check wheel and sdist
✓ Display package files
  - vascusim-0.1.0-py3-none.any.whl
  - vascusim-0.1.0.tar.gz
✓ Publish to TestPyPI
✓ Test installation from TestPyPI
✓ Publish to PyPI
✓ Update documentation
✓ Deploy documentation to GitHub Pages
```

Viewing Workflow Results
----------------------

You can view the results of workflow runs in the Actions tab of the repository. Each run will show:

- Which workflow ran
- When it ran
- Which branch or tag it ran on
- Whether it succeeded or failed
- Detailed logs for each step

You can also set up notifications to be alerted when workflows fail, either through GitHub's notification system or through integrations with other services like Slack or email.

Adding New Workflows
-----------------

To add a new workflow:

1. Create a new YAML file in the `.github/workflows/` directory
2. Define the workflow name, triggers, and jobs
3. Commit and push the file to the repository

For more information on GitHub Actions workflow syntax, see the [GitHub Actions documentation](https://docs.github.com/en/actions).