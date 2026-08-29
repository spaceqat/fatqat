# FatQat Material for MkDocs documentation

This directory is the self-contained source for the English and Simplified
Chinese Material documentation. Its Markdown pages, configuration, templates,
assets, validation, and publishing workflow are maintained here.

Material supports one canonical language per MkDocs project. The two primary
configuration files therefore inherit shared presentation and API settings
from `mkdocs.base.yml`, build into `site/en` and `site/zh`, and link to one
another through the header language selector.

## Set up the documentation environment

From the repository root, create a dedicated virtual environment and install
the project plus the fully resolved MkDocs dependency set:

```sh
python -m venv .venv-mkdocs
# Activate .venv-mkdocs for your shell, then:
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r docs/mkdocs/requirements.txt
```

The MkDocs toolchain requires Python 3.10 or newer. FatQat itself currently
requires Python 3.12 or newer, so use Python 3.12+ when installing both as
shown above.

## Build and review both languages

The helper first generates the ignored publish-tree inputs, removes only
`docs/mkdocs/site`, builds both configurations with warnings treated as errors,
then writes a small root page that chooses Chinese for browsers whose preferred
language starts with `zh` and English otherwise:

```sh
python docs/mkdocs/manage.py build
python docs/mkdocs/manage.py serve
```

`manage.py generate` performs only the generation phase. It renders guide
figures from their tracked examples, executes tutorials when their source-hashed
cache is missing or stale, and assembles both localized tutorial trees. Use
`generate --refresh-tutorials` to force a fresh tutorial run. Generated pages,
plots, downloads, and runtime cache files are intentionally ignored by Git.

The repository's `.readthedocs.yaml` uses the same build command. Read the
Docs supplies its output directory and canonical version URL so the combined
site is published at the version root, with English under `en/` and Simplified
Chinese under `zh/`. This custom HTML job is necessary because the standard
Read the Docs MkDocs builder accepts one MkDocs configuration per build.

Open <http://127.0.0.1:8000/>. The combined server is intended for checking
language switching and final static output; it does not live-reload. For a
single-language authoring loop, use MkDocs directly:

```sh
python -m mkdocs serve -f docs/mkdocs/mkdocs.en.yml
python -m mkdocs serve -f docs/mkdocs/mkdocs.zh.yml --dev-addr 127.0.0.1:8001
```

MkDocs uses clean directory URLs by default: a link such as `/en/guide/`
resolves to `site/en/guide/index.html` through the web server. Do not open the
generated HTML files directly with a `file://` URL; serve the output as shown
above so navigation, search, and language switching behave as they will after
deployment. To review an existing build without rebuilding it, run:

```sh
python docs/mkdocs/manage.py serve --no-build
```

During an incomplete migration, `build --no-strict` or `serve --no-strict`
can expose warnings without failing. CI always uses the strict default.

After generation, each combined build runs `tools/validate_content.py`. It
checks that the
locale trees expose the same Markdown, asset, and download paths; every Chinese
page contains translated text; homepage destinations stay aligned; local
Markdown links resolve; executable tutorial downloads and displayed code remain
byte-for-byte identical; and captured tutorial figures match across locales.
Run it directly after `manage.py generate` for a fast content-only check:

```sh
python docs/mkdocs/tools/validate_content.py
```

## Author guides and API reference pages

English and Chinese guide and API pages are native Markdown under `en/` and
`zh/`. Edit them directly and keep corresponding pages structurally aligned.
API pages combine curated explanations with `mkdocstrings` directives such as
`::: fatqat.Program`; signatures, members, and source docstrings are rendered
from `src/fatqat` during each build.

The two root `index.md` files contain manually maintained, localized content,
while `overrides/home.html` owns their shared Material landing-page structure.
Use native Material grids and cards for substantive sections, and keep the
small `hero` front-matter mappings structurally identical. The homepage
stylesheet is loaded only by the shared template. Seven guide illustrations
come directly from adjacent expandable Python examples; the three visual guide
cards come from `figure-sources/guide_cards.py`. The generator writes identical
ignored PNGs into both locale trees.

## Add and generate executable tutorials

A tutorial needs only one mirrored Markdown pair:

```text
docs/mkdocs/tutorial-sources/
├── en/<category>/<slug>.md
└── zh/<category>/<slug>.md
```

The folder name determines the category. The builder discovers every pair
recursively, so there is no tutorial registry or navigation list to edit. Each
file supplies `title`, `description`, `icon`, and localized `figure_alts` in
YAML front matter. The English body contains ordinary top-level `python`
fences; each is one executable cell. Put one `<!-- tutorial-code-cell -->`
placeholder at the corresponding position in the Chinese body. This keeps the
translated prose in Markdown while defining executable Python only once.

The category order and localized category labels live in
`tutorial-sources/gallery.yml`; rearrange the short `category_order` list to
choose the display order. Tutorials in an existing category still need no
registration. A new, unconfigured category also appears automatically with a
title derived from its folder name, after the configured categories; add it to
`gallery.yml` only when it needs an explicit position or localized
presentation text.

The first generated plot automatically becomes the card image. Every tutorial
must therefore create at least one Matplotlib figure, and `figure_alts` must
contain one accessible label per generated figure. The downloadable Python
script and localized, unexecuted Jupyter notebook are assembled from those
same cells; notebook prose comes from the selected locale's Markdown. Generate
the site inputs with:

```sh
python docs/mkdocs/manage.py generate
```

The ignored runtime cache records the SHA-256 digest of the English Markdown
and every captured PNG. A normal generation reuses it only while all hashes
match; a clean checkout or source change executes the suite before assembling
the localized pages, downloads, figures, and category index. Force all
tutorials to run again with:

```sh
python docs/mkdocs/manage.py generate --refresh-tutorials
```

Generated output is reviewable locally but is never committed. CI and Read the
Docs both start from authored inputs and recreate the complete bilingual site.
When English narrative changes, update the mirrored Chinese Markdown in the
same change; discovery and structural validation check the assembled pages.

## Keep translations aligned

The English and Chinese trees must contain the same relative Markdown paths.
Material uses the locale-root links in `extra.alternate` together with the
generated locale sitemaps to keep readers on the equivalent page. If the
destination locale does not contain that path, Material falls back to its home
page. Every page addition, move, or deletion should update both trees and both
localized `nav` lists in the same change; the content validator enforces path
parity.

The `nav` lists are intentionally repeated. MkDocs configuration inheritance
deep-merges mapping values, but replaces lists, so translated navigation
labels cannot be layered on top of one shared list.

API pages use mkdocstrings with `../../src` as the source path. Set
`locale: zh` for the Chinese build to translate mkdocstrings interface labels;
Python names, signatures, and source docstrings remain authoritative and are
not machine-translated by the build. The small Griffe extension in
`extensions/docstring_roles.py` converts Python cross-reference roles in those
docstrings into native MkDocs links without changing the Python sources.

Edit the direct documentation requirements in `requirements.in`, then
refresh the transitive lock with Python 3.12 before committing an upgrade:

```sh
uv pip compile --python-version 3.12 --output-file docs/mkdocs/requirements.txt docs/mkdocs/requirements.in
```

## Configure a deployment URL

Local defaults use `http://127.0.0.1:8000/en/` and `/zh/`. Before publishing,
set the canonical URLs and language-selector links for the actual host:

```sh
export FATQAT_MKDOCS_EN_SITE_URL=https://docs.example.org/fatqat/en/
export FATQAT_MKDOCS_ZH_SITE_URL=https://docs.example.org/fatqat/zh/
export FATQAT_MKDOCS_EN_LINK=/fatqat/en/
export FATQAT_MKDOCS_ZH_LINK=/fatqat/zh/
```

PowerShell uses `$env:NAME = "value"` for the same variables. Full URLs are
also accepted for the two link variables. The publishing destination is
deliberately not encoded in CI because the workflow only uploads a build
artifact and does not deploy it.

No redirect plugin is needed for the root language entry point: `manage.py`
creates it after both localized builds succeed. Add `mkdocs-redirects` only if
future content moves require per-page legacy redirects.

## Keep the toolchain boundary explicit

The pins intentionally stay on MkDocs 1.6 and Material for MkDocs 9.7.7.
Material 9 is MIT-licensed and is currently maintained rather than receiving
new feature development. The Material maintainers warn that the separately
developed MkDocs 2 removes the existing plugin and theme-override systems, has
no migration path for Material sites, and was not licensed for production use
when they published their assessment. Do not treat a future MkDocs 2 release as
an automatic dependency upgrade; review the
[Material maintainers' compatibility analysis](https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/)
and re-evaluate this experiment first.

The deprecated Material Projects and Typeset plugins are deliberately not
used. Separate per-language projects follow Material's documented i18n model
without depending on those unsupported plugins.

MathJax is loaded from jsDelivr at the exact 3.2.2 release with a SHA-384
Subresource Integrity check. Publishing therefore requires access to that CDN;
vendor the same verified asset and update `overrides/main.html` if the site must
work in an offline or restricted-network environment.
