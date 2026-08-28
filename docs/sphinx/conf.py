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
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
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

# Console examples use doctest prompts so Sphinx can execute them. Strip those
# prompts—and the displayed output—when a reader uses the copy button.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

templates_path: list[str] = []
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
# A class reference describes the public API available on that class,
# including methods supplied by its base classes. A page that places member
# documentation manually can opt out with ``:no-members:``.
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# API pages already establish the owning class in their title. Keep complete
# object signatures and link targets, but omit that repeated class name from
# the right-hand page contents.
toc_object_entries_show_parents = "hide"

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

# Tutorial sources stay as small, reviewable Python files outside the Sphinx
# source tree. Sphinx-Gallery executes every file during a clean docs build,
# writes its generated RST, figures, and notebooks below ``docs/sphinx/tutorials``,
# and makes the per-page notebook available as an HTML download. The generated
# tree is a build product and is ignored by Git.
sphinx_gallery_conf = {
    "examples_dirs": "../../tutorials",
    "gallery_dirs": "tutorials",
    "filename_pattern": r"\.py$",
    "abort_on_example_error": True,
    "download_all_examples": False,
    "notebook_images": False,
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["fatqat.css"]
html_theme_options = {
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "icon_links": [],
}

# Guide plots are small executable examples, not committed screenshots. Each
# plot directive can hide its source when the figure is only explanatory, but
# examples show their code by default so readers can reproduce what they see.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 140)]
