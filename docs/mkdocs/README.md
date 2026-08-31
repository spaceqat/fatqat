# FatQat Material for MkDocs documentation

This directory is the self-contained source for FatQat's English Material
documentation. Markdown pages, configuration, templates, generated-asset
sources, validation, and publishing workflow are maintained here.

`locales.yml` is the single registry of documentation locales. English is the
canonical and currently only active locale. The build utilities deliberately
iterate over that registry so a translated site can be added later without
restoring language-specific branches throughout the toolchain.

## Set up the documentation environment

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

The helper generates ignored publish-tree inputs, removes only
`docs/mkdocs/site`, builds every active locale with warnings treated as errors,
then writes a root locale entry page:

```sh
python docs/mkdocs/manage.py build
python docs/mkdocs/manage.py serve
```

Open <http://127.0.0.1:8000/>. The root currently redirects to `/en/`. The
static server is intended for final review and does not live-reload.
To serve an existing build without rebuilding it, run:

```sh
python docs/mkdocs/manage.py serve --no-build
```

For a live English authoring loop, generate ignored inputs once and then run
MkDocs directly:

```sh
python docs/mkdocs/manage.py generate
python -m mkdocs serve -f docs/mkdocs/mkdocs.en.yml
```

`manage.py generate` renders documentation figures, executes tutorials when
their source-hashed cache is missing or stale, and assembles generated tutorial
pages and downloads. Add `--refresh-tutorials` to force a fresh tutorial run.
Generated pages, plots, downloads, and runtime cache files are intentionally
ignored by Git.

Every build runs `tools/validate_content.py`. It checks Markdown links,
homepage metadata and template destinations, generated assets and downloads,
and executable tutorial code. If another locale is activated, the same
validator also enforces path, code, download, figure, and homepage-link parity
with canonical English content.

During an incomplete migration, `build --no-strict` or `serve --no-strict`
can expose warnings without failing. CI always uses the strict default.

## Author guides and API pages

Guide and API pages are native Markdown under `en/`. API pages combine curated
explanations with `mkdocstrings` directives such as `::: fatqat.Program`;
signatures, members, and source docstrings are rendered from `src/fatqat` on
every build.

The root `en/index.md` contains the homepage content and a small `hero`
front-matter mapping. `overrides/home.html` owns the shared Material landing
structure, while its stylesheet is loaded only by that template. Prefer native
Material grids, cards, figures, disclosures, and content tabs for substantive
layout.

Six guide illustrations come directly from adjacent expandable Python
examples. The three visual guide cards come from
`figure-sources/guide_cards.py`; the atom-pairing illustration comes from
`figure-sources/atom_pairing_lifecycle.py`.

The homepage Grover comparison uses three top-to-bottom executable scripts.
The visible `home_grover_program.py` owns the single logical and fused native
Programs, while `_home_grover_plot.py` keeps presentation details private.
Each execution script therefore focuses on one workflow and does not import
another example. The generator runs every script separately; the homepage
exposes the shared Program in a Material disclosure and the execution scripts
through content tabs in a second disclosure.

Run any comparison script directly from the repository root to display its
labeled figure or figures:

```sh
python docs/mkdocs/figure-sources/home_grover_general.py
python docs/mkdocs/figure-sources/home_grover_google.py
python docs/mkdocs/figure-sources/home_grover_transmon.py
```

## Add executable tutorials

Create one Markdown source under a category folder:

```text
docs/mkdocs/tutorial-sources/en/<category>/<slug>.md
```

The builder discovers sources recursively; there is no tutorial registry or
navigation list to edit. Each source supplies `title`, `description`, `icon`,
and accessible `figure_alts` in YAML front matter. Its top-level `python`
fences are executable cells and define the downloadable script and notebook
code once.

The category order, labels, gallery prose, and generated-panel UI strings live
in `tutorial-sources/gallery.yml`. Tutorials in an existing category require no
registration. An unconfigured category appears automatically with a title
derived from its folder name after configured categories.

The first generated plot becomes the tutorial card image. Every tutorial must
therefore create at least one Matplotlib figure, with one `figure_alts` label
per generated figure. The runtime cache hashes executable cells, so prose-only
edits can reuse captured results while code changes trigger execution.

## Add a future locale

Add a locale only when all of these pieces are ready:

1. Add `mkdocs.<locale>.yml` inheriting `mkdocs.base.yml`.
2. Add the locale's authored page tree under `<locale>/`.
3. Add mirrored tutorial sources under `tutorial-sources/<locale>/`. Keep
   executable Python in canonical English and place
   `<!-- tutorial-code-cell -->` at corresponding translated positions.
4. Add localized gallery and generated-panel strings to `gallery.yml`.
5. Register the locale, label, and config filename in `locales.yml`.
6. Add Material `extra.alternate` entries only after at least two complete
   locales are active.

The locale-aware build, tutorial, figure, and validation utilities will then
include it automatically.

## Configure deployment

Local English builds default to `http://127.0.0.1:8000/en/`. Set a canonical
base URL through the helper when publishing:

```sh
python docs/mkdocs/manage.py build \
  --site-url https://docs.example.org/fatqat
```

Alternatively set `FATQAT_MKDOCS_EN_SITE_URL` for a direct MkDocs invocation.
The repository's `.readthedocs.yaml` uses the helper and supplies both its
output directory and canonical version URL. The publishing destination is not
encoded in CI because the workflow uploads a build artifact rather than
deploying it.

## Keep the toolchain boundary explicit

Edit direct documentation requirements in `requirements.in`, then refresh the
transitive lock with Python 3.12 before committing an upgrade:

```sh
uv pip compile --python-version 3.12 \
  --output-file docs/mkdocs/requirements.txt \
  docs/mkdocs/requirements.in
```

The pins intentionally remain on MkDocs 1.6 and Material for MkDocs 9.7.7.
Material's deprecated Projects and Typeset plugins are not used. Treat a future
MkDocs 2 release as a migration project rather than an automatic dependency
upgrade; review the
[Material compatibility analysis](https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/)
first.

MathJax is loaded from jsDelivr at the exact 3.2.2 release with a SHA-384
Subresource Integrity check. Publishing requires access to that CDN; vendor the
same verified asset and update `overrides/main.html` if the site must work in
an offline or restricted-network environment.
