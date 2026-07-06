"""Sphinx configuration for fatqat's API reference and user guide."""

from __future__ import annotations

project = "fatqat"
copyright = "2026, fatqat contributors"
author = "fatqat contributors"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

templates_path: list[str] = []
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "signature"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

html_theme = "pydata_sphinx_theme"
html_static_path: list[str] = []
html_theme_options = {
    "navigation_depth": 2,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "icon_links": [],
}
