"""Build and preview the combined English and Chinese MkDocs site."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
import functools
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
SITE_ROOT = HERE / "site"
CONFIGURATIONS = (HERE / "mkdocs.en.yml", HERE / "mkdocs.zh.yml")
CONTENT_VALIDATOR = HERE / "tools" / "validate_content.py"

LANDING_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
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


def _validate_unique_html_ids() -> None:
    """Reject ambiguous fragments in generated pages."""

    failures: list[str] = []
    pages = sorted(SITE_ROOT.rglob("*.html"))
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
                f"{page.relative_to(SITE_ROOT)}: duplicate IDs {preview}{suffix}"
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
    *, strict: bool = True, environment_overrides: Mapping[str, str] | None = None
) -> None:
    """Build both locales and add the root language-selection page."""

    subprocess.run(
        [sys.executable, str(CONTENT_VALIDATOR)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    if SITE_ROOT.exists():
        shutil.rmtree(SITE_ROOT, onexc=_remove_readonly)
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

    (SITE_ROOT / "index.html").write_text(LANDING_PAGE, encoding="utf-8", newline="\n")
    (SITE_ROOT / ".nojekyll").touch()
    _validate_unique_html_ids()
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
