# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EPTHenOpt'
copyright = '2025, Maetee Lorprajuksiri'
author = 'Maetee Lorprajuksiri'
release = '0.8.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

# -- Path setup --------------------------------------------------------------
import os
import sys
# Tell Sphinx to look in the root directory for your package
sys.path.insert(0, os.path.abspath('../..'))


# -- General configuration ---------------------------------------------------
# Add the extensions needed to parse docstrings and create the API docs
extensions = [
    'sphinx.ext.autodoc',      # Core library to pull in documentation from docstrings
    'sphinx.ext.autosummary',  # Create summary tables
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx_rtd_theme',        # Use the Read the Docs theme
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
