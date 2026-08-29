# Parallel Material for MkDocs build

This directory contains an English and Simplified Chinese documentation
experiment alongside, and independent from, `docs/sphinx`. It does not alter
the Read the Docs configuration or replace the Sphinx build.

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
shown above. The Sphinx environment remains defined by `docs/requirements.in`
and `docs/requirements.txt`.

## Build and review both languages

The helper removes only `docs/mkdocs/site`, builds both configurations with
warnings treated as errors, then writes a small root page that chooses Chinese
for browsers whose preferred language starts with `zh` and English otherwise:

```sh
python docs/mkdocs/manage.py build
python docs/mkdocs/manage.py serve
```

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

Each combined build first runs `tools/validate_content.py`. It checks that the
locale trees expose the same Markdown, asset, and download paths; every Chinese
page contains translated text; homepage destinations stay aligned; local
Markdown links resolve; executable tutorial downloads and displayed code remain
byte-for-byte identical; and captured tutorial figures match across locales.
Run it directly for a fast content-only check:

```sh
python docs/mkdocs/tools/validate_content.py
```

## Keep generated English pages synchronized

Sphinx API and guide pages, plus the tracked Sphinx-Gallery tutorial scripts,
remain the canonical English sources. Regenerate their Material counterparts
after changing those sources:

```sh
python docs/mkdocs/tools/convert_api.py
python docs/mkdocs/tools/convert_guides.py
python docs/mkdocs/tools/convert_tutorials.py
```

The two root `index.md` files contain manually maintained, localized content,
while `overrides/home.html` owns their shared Material landing-page structure.
Keep their product story aligned with the Sphinx homepage, use native Material
grids and cards for substantive sections, and keep the small `hero` front-matter
mappings structurally identical. The homepage stylesheet is loaded only by the
shared template. `convert_guides.py` intentionally owns only `guide/**` and does
not overwrite either homepage.

Pass `--render` to `convert_guides.py` only when its trusted plot blocks or
figure inputs change and the committed PNGs need refreshing. CI deliberately
does not render figures because graphics output can vary across platforms.

Tutorial pages consume checked-in runtime snapshots from `tutorial-results/`.
Each manifest records the SHA-256 digest of its canonical tutorial source and
every captured PNG, so the default converter fails instead of presenting stale
results. It regenerates both localized tutorial indexes from the bilingual
`TUTORIALS` metadata, using each tutorial's selected captured figure as its card
thumbnail. It also regenerates English tutorial prose and synchronizes only the
generated runtime panels in the manually translated Chinese tutorial pages; it
never executes a tutorial. After reviewing a tutorial source change, refresh
all snapshots in a full Sphinx documentation environment (which includes
scikit-learn) with:

```sh
uv run --group docs python docs/mkdocs/tools/convert_tutorials.py --execute
```

When adding a tracked `tutorials/plot_*.py` file, register it in `TUTORIALS`
with its bilingual card metadata and add bilingual entries to `FIGURE_ALTS`.
The converter rejects unregistered tutorial sources, then generates both index
cards and uses captured figure 1 as the thumbnail by default. Set the
tutorial's `thumbnail_number` only when another reviewed figure is a better
summary of the example.

Tutorial execution is deliberately opt-in because the complete suite is
computationally expensive and graphics can differ with plotting and font
versions. Review changed stdout and figures before committing them. CI runs the
API converter in check mode, regenerates guide and tutorial text from committed
inputs, and fails if either locale changes. Translate any resulting English
narrative changes into the matching Chinese pages before committing.

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

Generated API pages use mkdocstrings with `../../src` as the source path. Set
`locale: zh` for the Chinese build to translate mkdocstrings interface labels;
Python names, signatures, and source docstrings remain authoritative and are
not machine-translated by the build. The small Griffe extension in
`extensions/sphinx_roles.py` converts the Sphinx cross-reference roles that
remain in those docstrings into native MkDocs links without changing the
Python sources.

Edit the four direct documentation requirements in `requirements.in`, then
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
