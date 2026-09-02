# Contributing to fatqat documentation

This directory contains the authored pages, executable tutorial sources,
generated-asset sources, and build configuration for fatqat's documentation.
The existing English Markdown source remains under `en/`, while MkDocs
publishes it at the site root.

## Set up the documentation environment

Documentation dependencies are managed through `requirements.in` and its
compiled `requirements.txt`, not through a dependency group in
`pyproject.toml`.

From the repository root, create a dedicated virtual environment and install
the project plus the resolved MkDocs dependency set:

```sh
python -m venv .venv-mkdocs
# Activate .venv-mkdocs for your shell, then:
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r docs/mkdocs/requirements.txt
```

Use Python 3.12 or newer so the environment satisfies both FatQat and the
documentation toolchain.

## Build and preview the site

Start MkDocs from the repository root for a live-reloading preview:

```sh
mkdocs serve
```

Open <http://127.0.0.1:8000/>. The configured build hook renders documentation
figures, executes tutorials when their code cache is missing or stale,
assembles generated tutorial pages and downloads, and validates the content
before MkDocs renders it. Changes to authored pages, tutorial sources, figure
sources, and public docstrings trigger a rebuild.

To force tutorial execution even when the cache is current, run:

```sh
python docs/mkdocs/tools/build_tutorials.py --execute
```

Generated pages, plots, downloads, and runtime cache files are ignored by Git.
For a complete warnings-as-errors build, run:

```sh
mkdocs build --strict
```

## Write guides and API pages

Guide and API pages are native Markdown under `en/`. API pages combine curated
explanations with `mkdocstrings` directives such as `::: fatqat.Program`;
signatures, members, and source docstrings are rendered from `src/fatqat` on
every build.

Add new guide and API pages to `nav` in the repository-root `mkdocs.yml`.
Tutorial pages are discovered automatically and do not need navigation entries.

Follow the writing and public-API standards in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#writing-style-and-api-documentation).
Prefer native Material components such as grids, cards, figures, disclosures,
and content tabs when they make substantive content easier to navigate.

## Add executable tutorials

Tutorials are authored in Markdown under a category folder:

```text
docs/mkdocs/tutorial-sources/en/<category>/<slug>.md
```

The Markdown file is the source of truth. The builder discovers sources
recursively, and the category folder controls gallery grouping. A slug must be
unique across all categories. Tutorials in an existing category require no
registry or navigation change.

Start from a nearby tutorial. Each source begins with YAML front matter:

```yaml
---
title: My tutorial
description: What the tutorial teaches and why it is useful.
figure_alts:
  - Description of the first generated figure.
---
```

The fields shown above are required. `icon` is optional and defaults to
`material-flask-outline`. To choose another, use the
[Material for MkDocs icon search](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/),
select a Material Design icon, and enter its shortcode without the surrounding
colons, for example `icon: material-atom`.

Follow the front matter with one level-one heading. Give the tutorial a clear
learning goal, explain what its results mean, seed stochastic work, and identify
the provenance of external material. It must run without network access or
credentials.

The category order, labels, gallery prose, and generated-panel UI strings live
in `tutorial-sources/gallery.yml`. An unconfigured category appears
automatically with a title derived from its folder name after configured
categories.

Each top-level triple-backtick `python` fence is an executable cell. Cells run
in order and share a namespace. Use `~~~python` for highlighted Python that
should not execute. The builder runs plain Python rather than IPython, so remove
magics and shell commands, print text that should appear, and create figures
explicitly. Create and finish each figure within one cell.

The first generated plot becomes the tutorial card image. Every tutorial must
create at least one Matplotlib figure, with one accessible `figure_alts`
description per generated figure. Choose a representative first figure.

### Convert notebook drafts with Jupytext

[Jupytext](https://jupytext.org/using/cli/) is optional when importing a
notebook or creating a temporary notebook for interactive work:

```sh
python -m pip install jupytext

# Notebook to a Markdown draft
jupytext --to md --output converted.md source.ipynb

# Markdown to a temporary notebook
jupytext --to ipynb --output preview.ipynb docs/mkdocs/tutorial-sources/en/algorithms/my-tutorial.md
```

Use Jupytext's plain `md` format, not `md:myst`. Notebook outputs are not kept
in the Markdown source. Treat conversions as drafts: add or preserve the
required front matter, remove notebook-specific metadata, and check that only
code intended to run remains in top-level triple-backtick `python` fences.
Commit the Markdown source, not the notebook.

### Generate and validate a tutorial

Use the [authoring and preview commands](#build-and-preview-the-site) to
generate and inspect the tutorial.

The generated notebook is written to
`docs/mkdocs/en/downloads/tutorials/<slug>.ipynb`. Generated notebooks, pages,
figures, scripts, and runtime results are ignored and must not be committed.

Before submission, run the
[full documentation build](#validate-before-submission). It performs generation
and validation itself, so it is not necessary to run `generate` first.

## Validate before submission

From the repository root, run the full warnings-as-errors build:

```sh
mkdocs build --strict
```

Before submitting documentation changes:

- read affected public docstrings through Python `help()`;
- inspect the affected rendered pages, including navigation, links, signatures,
  tables, and generated API members; a tutorial is rendered at
  `docs/mkdocs/site/tutorials/<slug>/index.html`;
- verify every documented default, constraint, result, and exception against
  the code or tests; and
- run `git diff --check`.

## Update documentation dependencies

Edit direct dependencies in `requirements.in`, then refresh the compiled
`requirements.txt` with Python 3.12 before committing an upgrade:

```sh
uv pip compile --python-version 3.12 \
  --output-file docs/mkdocs/requirements.txt \
  docs/mkdocs/requirements.in
```

Treat a future MkDocs major-version upgrade as a migration project rather than
a routine dependency refresh.

Commit both `requirements.in` and `requirements.txt` when they change.
