# Building the docs

Install the `docs` dependency group, then build with warnings-as-errors so a
missing docstring, undocumented public member, or broken cross-reference
fails the build instead of silently vanishing:

```sh
uv sync --group docs
uv run sphinx-build -b html -W docs/sphinx docs/sphinx/_build/html
```

Open `docs/sphinx/_build/html/index.html`. This matches the existing
`Makefile` and `make.bat` `<build-dir>/<builder>` layout; direct commands and
CI are converging on that convention.

The `Build HTML documentation` GitHub Actions workflow also runs the Sphinx
doctest builder. It uploads only `docs/sphinx/_build/html` as the
`fatqat_html_docs` artifact; there is no hosting/publishing step yet.
