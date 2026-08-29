"""Validate the parallel English and Simplified Chinese documentation trees."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

MKDOCS_ROOT = Path(__file__).resolve().parents[1]
LOCALE_ROOTS = {
    "en": MKDOCS_ROOT / "en",
    "zh": MKDOCS_ROOT / "zh",
}

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
GENERATED_ASSET_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
TUTORIAL_CODE_BLOCK = re.compile(
    r'^```python title="Python (?:cell|单元) (?P<number>\d+)"\n' r"(?P<code>.*?)^```$",
    re.DOTALL | re.MULTILINE,
)


def _relative_files(root: Path, *, suffixes: set[str]) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _describe_paths(paths: set[Path]) -> str:
    return ", ".join(path.as_posix() for path in sorted(paths))


def _markdown_destination(raw_destination: str) -> str:
    """Return a Markdown link destination without an optional title."""

    destination = raw_destination.strip()
    if destination.startswith("<") and ">" in destination:
        return destination[1 : destination.index(">")]
    return re.split(r"\s+[\"']", destination, maxsplit=1)[0]


def _local_target_exists(page: Path, locale_root: Path, destination: str) -> bool:
    split = urlsplit(unquote(destination))
    if split.scheme or split.netloc or not split.path:
        return True

    relative = Path(split.path)
    if relative.is_absolute() or split.path.startswith("/"):
        return True

    target = page.parent / relative
    candidates = [target]
    if not target.suffix:
        candidates.extend((target.with_suffix(".md"), target / "index.md"))
    return any(candidate.is_file() for candidate in candidates)


def _tutorial_code_cells(page: Path) -> list[tuple[int, str]]:
    """Return numbered Python cells while ignoring their localized titles."""

    return [
        (int(match.group("number")), match.group("code"))
        for match in TUTORIAL_CODE_BLOCK.finditer(page.read_text(encoding="utf-8"))
    ]


def _markdown_destinations(page: Path) -> list[str]:
    """Return ordered inline-link and image destinations from one page."""

    return [
        _markdown_destination(match.group(1))
        for match in MARKDOWN_LINK.finditer(page.read_text(encoding="utf-8"))
    ]


def validate() -> list[str]:
    errors: list[str] = []
    english = LOCALE_ROOTS["en"]
    chinese = LOCALE_ROOTS["zh"]

    markdown_paths = {
        locale: _relative_files(root, suffixes={".md"})
        for locale, root in LOCALE_ROOTS.items()
    }
    if missing := markdown_paths["en"] - markdown_paths["zh"]:
        errors.append(f"Chinese Markdown pages are missing: {_describe_paths(missing)}")
    if extra := markdown_paths["zh"] - markdown_paths["en"]:
        errors.append(f"Chinese-only Markdown pages exist: {_describe_paths(extra)}")

    for folder, suffixes in (
        ("assets", GENERATED_ASSET_SUFFIXES),
        ("downloads", {".ipynb", ".py"}),
    ):
        localized = {
            locale: _relative_files(root / folder, suffixes=suffixes)
            for locale, root in LOCALE_ROOTS.items()
        }
        if missing := localized["en"] - localized["zh"]:
            errors.append(f"Chinese {folder} are missing: {_describe_paths(missing)}")
        if extra := localized["zh"] - localized["en"]:
            errors.append(f"Chinese-only {folder} exist: {_describe_paths(extra)}")

    for download in _relative_files(english / "downloads", suffixes={".py"}):
        english_download = english / "downloads" / download
        chinese_download = chinese / "downloads" / download
        if (
            chinese_download.is_file()
            and english_download.read_bytes() != chinese_download.read_bytes()
        ):
            errors.append(
                f"Translated Python download differs from English: {download.as_posix()}"
            )

    tutorial_asset_root = Path("assets/generated/tutorials")
    for asset in _relative_files(
        english / tutorial_asset_root, suffixes=GENERATED_ASSET_SUFFIXES
    ):
        english_asset = english / tutorial_asset_root / asset
        chinese_asset = chinese / tutorial_asset_root / asset
        if (
            chinese_asset.is_file()
            and english_asset.read_bytes() != chinese_asset.read_bytes()
        ):
            errors.append(
                "Captured tutorial figure differs between locales: "
                f"{asset.as_posix()}"
            )

    for relative in sorted(markdown_paths["en"]):
        if relative.parts[:1] != ("tutorials",) or relative.name == "index.md":
            continue
        english_cells = _tutorial_code_cells(english / relative)
        chinese_cells = _tutorial_code_cells(chinese / relative)
        if english_cells != chinese_cells:
            errors.append(
                "Translated tutorial code differs from English: "
                f"{relative.as_posix()}"
            )

    if _markdown_destinations(english / "index.md") != _markdown_destinations(
        chinese / "index.md"
    ):
        errors.append("Homepage link and image destinations differ between locales")

    for locale, root in LOCALE_ROOTS.items():
        for relative in sorted(markdown_paths[locale]):
            page = root / relative
            text = page.read_text(encoding="utf-8")
            if locale == "zh" and not CJK.search(text):
                errors.append(f"Chinese page has no CJK text: {relative.as_posix()}")
            for match in MARKDOWN_LINK.finditer(text):
                destination = _markdown_destination(match.group(1))
                if not _local_target_exists(page, root, destination):
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(
                        f"Broken local link in {locale}/{relative.as_posix()}:{line}: "
                        f"{destination}"
                    )

    if not errors:
        print(
            "Validated locale parity: "
            f"{len(markdown_paths['en'])} Markdown pages, "
            f"{len(_relative_files(english / 'assets', suffixes=GENERATED_ASSET_SUFFIXES))} assets, "
            f"and {len(_relative_files(english / 'downloads', suffixes={'.ipynb', '.py'}))} downloads per locale."
        )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Content validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
