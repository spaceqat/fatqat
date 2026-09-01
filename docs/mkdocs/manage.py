"""Build and preview the configured MkDocs documentation locales."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import functools
import html
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
SITE_ROOT = HERE / "site"
LOCALE_REGISTRY = HERE / "locales.yml"
CONTENT_VALIDATOR = HERE / "tools" / "validate_content.py"
GUIDE_FIGURE_RENDERER = HERE / "tools" / "render_guide_figures.py"
TUTORIAL_BUILDER = HERE / "tools" / "build_tutorials.py"


@dataclass(frozen=True)
class _Locale:
    """One active documentation locale."""

    code: str
    label: str
    config: Path


def _load_locales() -> tuple[str, tuple[_Locale, ...]]:
    """Load and validate the shared active-locale registry."""

    payload = yaml.safe_load(LOCALE_REGISTRY.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("locales"), dict):
        raise ValueError(f"{LOCALE_REGISTRY}: expected a locales mapping")
    canonical = payload.get("canonical")
    if not isinstance(canonical, str) or canonical not in payload["locales"]:
        raise ValueError(f"{LOCALE_REGISTRY}: canonical must name an active locale")

    locales: list[_Locale] = []
    for code, raw in payload["locales"].items():
        if not isinstance(code, str) or not isinstance(raw, dict):
            raise ValueError(f"{LOCALE_REGISTRY}: invalid locale entry {code!r}")
        label = raw.get("label")
        config_name = raw.get("config")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{LOCALE_REGISTRY}: {code}.label must be text")
        if not isinstance(config_name, str) or Path(config_name).name != config_name:
            raise ValueError(f"{LOCALE_REGISTRY}: {code}.config must be a filename")
        config = HERE / config_name
        if not config.is_file() or not (HERE / code).is_dir():
            raise ValueError(f"{LOCALE_REGISTRY}: incomplete active locale {code!r}")
        locales.append(_Locale(code, label.strip(), config))
    locales.sort(key=lambda locale: (locale.code != canonical, locale.code))
    return canonical, tuple(locales)


CANONICAL_LOCALE, LOCALES = _load_locales()


def _landing_page() -> str:
    """Return the root locale chooser for the active registry."""

    supported = json.dumps([locale.code for locale in LOCALES])
    default = json.dumps(CANONICAL_LOCALE)
    links = "\n".join(
        f'        <a href="{html.escape(locale.code)}/" '
        f'hreflang="{html.escape(locale.code)}">{html.escape(locale.label)}</a>'
        for locale in LOCALES
    )
    return f"""<!doctype html>
<html lang="{html.escape(CANONICAL_LOCALE)}">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>FatQat documentation</title>
    <script>
      const supported = {supported};
      const preferred = (navigator.languages?.[0] || navigator.language || {default})
        .toLowerCase();
      const target = supported.find(
        locale => preferred === locale || preferred.startsWith(`${{locale}}-`)
      ) || {default};
      window.location.replace(new URL(`${{target}}/`, window.location.href));
    </script>
    <style>
      :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
      body {{ display: grid; min-height: 100vh; margin: 0; place-items: center; }}
      main {{ max-width: 36rem; padding: 2rem; text-align: center; }}
      nav {{ display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; }}
      a {{ border: 1px solid currentColor; border-radius: .5rem; padding: .7rem 1rem; }}
    </style>
  </head>
  <body>
    <main>
      <h1>FatQat documentation</h1>
      <p>Select a language if automatic redirection does not start.</p>
      <nav aria-label="Language selection">
{links}
      </nav>
    </main>
  </body>
</html>
"""


def _run_mkdocs(
    config: Path,
    *,
    site_dir: Path,
    strict: bool,
    environment: Mapping[str, str],
) -> None:
    command = [sys.executable, "-m", "mkdocs", "build", "-f", str(config)]
    command.extend(("--site-dir", str(site_dir)))
    if strict:
        command.append("--strict")
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


class _IdParser(HTMLParser):
    """Collect element IDs from one generated HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: Counter[str] = Counter()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        identifier = dict(attrs).get("id")
        if identifier:
            self.ids[identifier] += 1


def _validate_unique_html_ids(site_root: Path) -> None:
    """Reject ambiguous fragments in generated pages."""

    failures: list[str] = []
    pages = sorted(site_root.rglob("*.html"))
    for page in pages:
        parser = _IdParser()
        parser.feed(page.read_text(encoding="utf-8"))
        duplicates = sorted(
            identifier for identifier, count in parser.ids.items() if count > 1
        )
        if duplicates:
            preview = ", ".join(duplicates[:8])
            suffix = "..." if len(duplicates) > 8 else ""
            failures.append(
                f"{page.relative_to(site_root)}: duplicate IDs {preview}{suffix}"
            )

    if failures:
        raise RuntimeError("duplicate generated HTML IDs:\n" + "\n".join(failures))
    print(f"Validated unique HTML IDs across {len(pages)} generated pages.")


def _remove_readonly(
    function: Callable[[str], object], path: str, error: BaseException
) -> None:
    """Retry removal after clearing a Windows read-only file attribute."""

    del error
    os.chmod(path, stat.S_IWRITE)
    function(path)


def build_site(
    *,
    strict: bool = True,
    site_root: Path = SITE_ROOT,
    site_url: str | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    """Build every active locale and add the root language-selection page."""

    generate_content()
    subprocess.run(
        [sys.executable, str(CONTENT_VALIDATOR)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    if site_root.exists():
        shutil.rmtree(site_root, onexc=_remove_readonly)
    site_root.mkdir(parents=True)

    environment = os.environ.copy()
    if site_url:
        base_url = site_url.rstrip("/")
        for locale in LOCALES:
            prefix = locale.code.upper().replace("-", "_")
            locale_url = f"{base_url}/{locale.code}/"
            environment[f"FATQAT_MKDOCS_{prefix}_SITE_URL"] = locale_url
            environment[f"FATQAT_MKDOCS_{prefix}_LINK"] = locale_url
    if environment_overrides:
        environment.update(environment_overrides)

    for locale in LOCALES:
        _run_mkdocs(
            locale.config,
            site_dir=site_root / locale.code,
            strict=strict,
            environment=environment,
        )

    missing = [
        str(path.relative_to(site_root))
        for path in (site_root / locale.code / "index.html" for locale in LOCALES)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"localized builds did not create: {', '.join(missing)}")

    (site_root / "index.html").write_text(
        _landing_page(), encoding="utf-8", newline="\n"
    )
    (site_root / ".nojekyll").touch()
    _validate_unique_html_ids(site_root)
    print(f"Documentation site written to {site_root}")


def generate_content(*, refresh_tutorials: bool = False) -> None:
    """Create ignored publish-tree pages and assets from tracked sources."""

    subprocess.run(
        [sys.executable, str(GUIDE_FIGURE_RENDERER)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    tutorial_command = [sys.executable, str(TUTORIAL_BUILDER)]
    tutorial_command.append(
        "--execute" if refresh_tutorials else "--execute-if-needed"
    )
    subprocess.run(tutorial_command, cwd=REPOSITORY_ROOT, check=True)


def serve_site(*, host: str, port: int, strict: bool, build: bool) -> None:
    """Build, then serve the static output without live reload."""

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if build:
        base_url = f"http://{display_host}:{port}"
        overrides = {}
        for locale in LOCALES:
            prefix = locale.code.upper().replace("-", "_")
            overrides[f"FATQAT_MKDOCS_{prefix}_SITE_URL"] = (
                f"{base_url}/{locale.code}/"
            )
            overrides[f"FATQAT_MKDOCS_{prefix}_LINK"] = f"/{locale.code}/"
        build_site(
            strict=strict,
            environment_overrides=overrides,
        )
    elif not (SITE_ROOT / "index.html").is_file():
        raise FileNotFoundError("documentation site is missing; run build first")

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(SITE_ROOT))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving {SITE_ROOT} at http://{display_host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build active locales")
    build_parser.add_argument(
        "--no-strict",
        action="store_true",
        help="allow MkDocs warnings during migration",
    )
    build_parser.add_argument(
        "--site-dir",
        type=Path,
        default=SITE_ROOT,
        help="write the documentation site to this directory",
    )
    build_parser.add_argument(
        "--site-url",
        help="canonical base URL containing the localized sites",
    )

    generate_parser = subparsers.add_parser(
        "generate", help="regenerate ignored pages and assets from tracked sources"
    )
    generate_parser.add_argument(
        "--refresh-tutorials",
        action="store_true",
        help="execute tutorials even when the source-hashed cache is current",
    )

    serve_parser = subparsers.add_parser(
        "serve", help="build and serve the configured documentation site"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)
    serve_parser.add_argument(
        "--no-build", action="store_true", help="serve an existing build"
    )
    serve_parser.add_argument(
        "--no-strict",
        action="store_true",
        help="allow MkDocs warnings during migration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            build_site(
                strict=not args.no_strict,
                site_root=args.site_dir.resolve(),
                site_url=args.site_url,
            )
        elif args.command == "serve":
            serve_site(
                host=args.host,
                port=args.port,
                strict=not args.no_strict,
                build=not args.no_build,
            )
        else:
            generate_content(refresh_tutorials=args.refresh_tutorials)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
