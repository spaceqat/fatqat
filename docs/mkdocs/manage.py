"""Build and preview the combined English and Chinese MkDocs site."""

from __future__ import annotations

import argparse
import functools
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
SITE_ROOT = HERE / "site"
CONFIGURATIONS = (HERE / "mkdocs.en.yml", HERE / "mkdocs.zh.yml")
CONTENT_VALIDATOR = HERE / "tools" / "validate_content.py"
LOCALES = ("en", "zh")

LANDING_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="1; url=en/">
    <title>FatQat documentation</title>
    <script>
      const preferred = (navigator.languages?.[0] || navigator.language || "en")
        .toLowerCase();
      const target = preferred.startsWith("zh") ? "zh/" : "en/";
      window.location.replace(new URL(target, window.location.href));
    </script>
    <style>
      :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
      body { display: grid; min-height: 100vh; margin: 0; place-items: center; }
      main { max-width: 36rem; padding: 2rem; text-align: center; }
      nav { display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; }
      a { border: 1px solid currentColor; border-radius: .5rem; padding: .7rem 1rem; }
    </style>
  </head>
  <body>
    <main>
      <h1>FatQat documentation</h1>
      <p>Select a language if automatic redirection does not start.</p>
      <nav aria-label="Language selection">
        <a href="en/" hreflang="en">English</a>
        <a href="zh/" hreflang="zh" lang="zh-Hans">简体中文</a>
      </nav>
    </main>
  </body>
</html>
"""


def _run_mkdocs(config: Path, *, strict: bool, environment: Mapping[str, str]) -> None:
    command = [sys.executable, "-m", "mkdocs", "build", "-f", str(config)]
    if strict:
        command.append("--strict")
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


class _AlternateParser(HTMLParser):
    """Collect language alternates from the document head and header selector."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str, str]] = []
        self._in_head = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "head":
            self._in_head = True
        attributes = dict(attrs)
        language = attributes.get("hreflang")
        href = attributes.get("href")
        classes = (attributes.get("class") or "").split()
        rels = (attributes.get("rel") or "").split()
        if tag == "link" and self._in_head and "alternate" in rels:
            kind = "head"
        elif tag == "a" and "md-select__link" in classes:
            kind = "selector"
        else:
            return
        if language and href:
            self.links.append((kind, language, href))

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._in_head = False


def _page_output_path(locale: str, source: Path) -> Path:
    relative = source.relative_to(HERE / locale).with_suffix("")
    if relative.name == "index":
        return SITE_ROOT / locale / relative.parent / "index.html"
    return SITE_ROOT / locale / relative / "index.html"


def _localized_page_link(root: str, page_url: str) -> str:
    return f"{root.rstrip('/')}/{page_url.lstrip('/')}"


def _validate_alternate_links(environment: Mapping[str, str]) -> None:
    roots = {
        "en": environment.get("FATQAT_MKDOCS_EN_LINK", "/en/"),
        "zh": environment.get("FATQAT_MKDOCS_ZH_LINK", "/zh/"),
    }
    failures: list[str] = []

    for locale in LOCALES:
        for source in sorted((HERE / locale).rglob("*.md")):
            relative = source.relative_to(HERE / locale).with_suffix("")
            if relative.name == "index":
                page_url = (
                    ""
                    if relative.parent == Path(".")
                    else f"{relative.parent.as_posix()}/"
                )
            else:
                page_url = f"{relative.as_posix()}/"
            output = _page_output_path(locale, source)
            parser = _AlternateParser()
            parser.feed(output.read_text(encoding="utf-8"))
            expected_links: list[tuple[str, str, str]] = []
            for language, root in roots.items():
                expected = _localized_page_link(root, page_url)
                expected_links.extend(
                    (("head", language, expected), ("selector", language, expected))
                )
            if sorted(parser.links) != sorted(expected_links):
                failures.append(
                    f"{output.relative_to(HERE)}: expected {expected_links}, "
                    f"found {parser.links}"
                )

    if failures:
        raise RuntimeError("invalid localized alternate links:\n" + "\n".join(failures))
    print("Validated page-preserving language links in both locales.")


def build_site(
    *, strict: bool = True, environment_overrides: Mapping[str, str] | None = None
) -> None:
    """Build both locales and add the root language-selection page."""

    subprocess.run(
        [sys.executable, str(CONTENT_VALIDATOR)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    if SITE_ROOT.exists():
        shutil.rmtree(SITE_ROOT)
    SITE_ROOT.mkdir(parents=True)

    environment = os.environ.copy()
    if environment_overrides:
        environment.update(environment_overrides)

    for config in CONFIGURATIONS:
        _run_mkdocs(config, strict=strict, environment=environment)

    missing = [
        str(path.relative_to(HERE))
        for path in (SITE_ROOT / "en" / "index.html", SITE_ROOT / "zh" / "index.html")
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"localized builds did not create: {', '.join(missing)}")

    _validate_alternate_links(environment)

    (SITE_ROOT / "index.html").write_text(LANDING_PAGE, encoding="utf-8", newline="\n")
    (SITE_ROOT / ".nojekyll").touch()
    print(f"Combined site written to {SITE_ROOT}")


def serve_site(*, host: str, port: int, strict: bool, build: bool) -> None:
    """Build, then serve the combined static output without live reload."""

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if build:
        base_url = f"http://{display_host}:{port}"
        build_site(
            strict=strict,
            environment_overrides={
                "FATQAT_MKDOCS_EN_SITE_URL": f"{base_url}/en/",
                "FATQAT_MKDOCS_ZH_SITE_URL": f"{base_url}/zh/",
                "FATQAT_MKDOCS_EN_LINK": "/en/",
                "FATQAT_MKDOCS_ZH_LINK": "/zh/",
            },
        )
    elif not (SITE_ROOT / "index.html").is_file():
        raise FileNotFoundError("combined site is missing; run the build command first")

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

    build_parser = subparsers.add_parser("build", help="build both localized sites")
    build_parser.add_argument(
        "--no-strict",
        action="store_true",
        help="allow MkDocs warnings during migration",
    )

    serve_parser = subparsers.add_parser(
        "serve", help="build and serve the combined site for cross-language review"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)
    serve_parser.add_argument(
        "--no-build", action="store_true", help="serve an existing combined build"
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
            build_site(strict=not args.no_strict)
        else:
            serve_site(
                host=args.host,
                port=args.port,
                strict=not args.no_strict,
                build=not args.no_build,
            )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
