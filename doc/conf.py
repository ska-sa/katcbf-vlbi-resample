# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# cupy requires CUDA to be installed, which is inconvenient (and possibly not
# even possible on readthedocs). Mock it out.
import sys
from unittest import mock

for module in ["cupy", "cupyx", "cupyx.scipy", "cupyx.scipy.fft", "cupyx.scipy.signal"]:
    sys.modules[module] = mock.MagicMock()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "katcbf-vlbi-resample"
copyright = "2024, National Research Foundation (SARAO)"
author = "Bruce Merry"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "astropy": ("https://docs.astropy.org/en/latest", None),
    "baseband": ("https://baseband.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
