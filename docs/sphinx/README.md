# Building the docs

Install the `docs` dependency group, then build with warnings-as-errors so a
missing docstring, undocumented public member, or broken cross-reference
fails the build instead of silently vanishing:

```sh
uv sync --group docs
sphinx-build -b html -W docs/sphinx docs/sphinx/_build
```

Open `docs/sphinx/_build/index.html`. There is no CI job or hosting for this
yet (internal use only) — run the command above locally before trusting a
docs change.
