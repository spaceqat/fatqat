# Building the docs

Create a virtual environment and install the documentation's own pinned
requirements, then build with warnings-as-errors so a
missing docstring, undocumented public member, or broken cross-reference
fails the build instead of silently vanishing:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W docs/sphinx docs/sphinx/_build/html
```

Open `docs/sphinx/_build/html/index.html`. This matches the existing
`Makefile` and `make.bat` `<build-dir>/<builder>` layout; direct commands and
CI are converging on that convention.

The direct documentation dependencies are declared in
`docs/requirements.in`. `docs/requirements.txt` pins the Python 3.12 runtime
and documentation dependencies consumed by local builds and Read the Docs;
do not edit it by hand. After changing the input file,
regenerate the pin from the repository root with Python 3.12:

```sh
python -m pip install pip-tools==7.6.1
python -m piptools compile --strip-extras --output-file docs/requirements.txt docs/requirements.in
```

This pinned documentation environment is deliberately independent of the
contributor's development and test environment.

## Authoring executable tutorials

Tutorial sources live in the repository-level `tutorials/` directory as
ordinary Python files. They are written in Sphinx-Gallery's notebook style:

- Start the file with a raw triple-quoted reStructuredText docstring containing
  the page title and introduction.
- Separate narrative and executable cells with `# %%`. Narrative cells use
  comment lines, while uncommented lines form the following Python cell.
- Use reStructuredText roles and directives for links, lists, and math. Prefer
  portable reStructuredText when possible so the generated notebook reads well
  outside Sphinx too.
- Seed every source of randomness, avoid network access and credentials, and
  keep execution time appropriate for every public documentation build.
- Let exceptions escape. The gallery is configured to abort the build when any
  tutorial fails.

For example:

```python
r"""
Example title
=============

Inline math uses :math:`x^2`, while display math uses the ``math`` directive.
"""

# %%
# Explain the next executable cell here.
print("This output is included in the rendered page.")
```

Sphinx-Gallery writes generated RST, figures, scripts, and notebooks to
`docs/sphinx/tutorials/`, plus an execution-time report at
`docs/sphinx/sg_execution_times.rst`. Those paths are ignored by Git. Never
add their contents to a commit: only edit and commit the lightweight sources
under the top-level `tutorials/` directory. Each generated page contains its
own Jupyter notebook download; the gallery-wide notebook archive and embedded
base64 notebook images are deliberately disabled to keep artifacts small.

To test from a clean state, remove `docs/sphinx/tutorials/` and the Sphinx
build directory, then run the HTML command above. A clean HTML build executes
every tutorial. The generated page, captured output, figure, and individual
notebook download are all present below `docs/sphinx/_build/html/tutorials/`.

Read the Docs builds and publishes the HTML with warnings treated as errors.
Run the existing documentation doctests locally with the same environment:

```sh
python -m sphinx -b doctest -W docs/sphinx docs/sphinx/_build/doctest
```
