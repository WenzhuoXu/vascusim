# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_rtd_theme

# Try to import vascusim for version info
try:
    import vascusim
    version = vascusim.__version__
except ImportError:
    version = "unknown"

# -- Project information -----------------------------------------------------

project = 'VascuSim'
copyright = '2025, Wenzhuo Xu'
author = 'Wenzhuo Xu'

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx.ext.autodoc.typehints',
]

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Configure napoleon for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Configure intersphinx to link to other docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'pyg': ('https://pytorch-geometric.readthedocs.io/en/latest', None),
}

# Autodoc type hints settings
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'logo_only': False,
    'style_external_links': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'css/custom.css',
]

# HTML logo
# html_logo = "_static/logo.png"

# HTML favicon
# html_favicon = "_static/favicon.ico"

# HTML sidebars
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# -- Extension configuration -------------------------------------------------

# Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Options for viewcode extension ---------------------------------------
viewcode_enable_epub = False

# -- Options for autodoc extension ---------------------------------------
# This value selects what content will be inserted into the main body of an autoclass directive.
autoclass_content = 'both'

# Mock imports for autodoc
autodoc_mock_imports = [
    'numpy',
    'torch',
    'torch_geometric',
    'vtk',
    'vtkmodules',
    'huggingface_hub',
    'pyvista',
    'matplotlib',
    'tqdm',
    'smb',
    'urllib3',
    'requests',
]