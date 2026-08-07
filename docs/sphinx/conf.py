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
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Docstrings mix single- and double-backtick code spans; RST only treats
# double backticks as inline code by default, silently rendering the rest as
# italic title-references instead of erroring. Make both spellings render as
# code so neither style silently looks wrong.
default_role = "py:obj"

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

templates_path: list[str] = []
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    # Backend-author and implementation records are intentionally not part
    # of the end-user documentation site yet.
    "api/implementation.rst",
    "api/job.rst",
    "api/errors.rst",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "signature"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

# The doctest builder compares printed text exactly, and many examples in this
# project print complex arrays. A value that is zero in exact arithmetic can
# come back as a denormal residue - a Kraus sum leaving 1.1e-35j on a density
# matrix, say - and whether it does depends on the machine's BLAS, so an
# example that reads correctly here can fail on a CI runner. `suppress=True`
# prints such a residue as `0.` instead of in scientific notation, which is
# both what the example means and what a reader would write down. Every array
# in these examples is O(1), so nothing meaningful is being hidden; an example
# that genuinely needs to show a small magnitude should print the number rather
# than the array.
doctest_global_setup = "import numpy; numpy.set_printoptions(suppress=True)"

html_theme = "pydata_sphinx_theme"
html_static_path: list[str] = []
html_theme_options = {
    "navigation_depth": 2,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "icon_links": [],
}
