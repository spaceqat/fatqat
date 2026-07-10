# Building the docs

Install the `docs` dependency group, then build with warnings-as-errors so a
missing docstring, undocumented public member, or broken cross-reference
fails the build instead of silently vanishing:

```sh
uv sync --group docs
sphinx-build -b html -W docs/sphinx docs/sphinx/_build
```

Open `docs/sphinx/_build/index.html`.

The `Build HTML documentation` GitHub Actions workflow builds the same way
on every push/PR to `main` and uploads the result as a workflow artifact
(`fatqat_html_docs`) — there is no hosting/publishing step yet, so download
the artifact from the run to view it.
