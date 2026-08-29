"""Validate the parallel English and Simplified Chinese documentation trees."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

MKDOCS_ROOT = Path(__file__).resolve().parents[1]
LOCALE_ROOTS = {
    "en": MKDOCS_ROOT / "en",
    "zh": MKDOCS_ROOT / "zh",
}

MARKDOWN_DESTINATION = re.compile(
    r"\[!\[[^\]]*\]\((?P<linked_image>[^)\n]+)\)"
    r"(?:\{[^}\n]*\})?\]\((?P<linked_target>[^)\n]+)\)"
    r"|!?\[[^\]]*\]\((?P<destination>[^)\n]+)\)"
)
FRONT_MATTER = re.compile(r"\A---\r?\n(?P<yaml>.*?)\r?\n---(?:\r?\n|\Z)", re.DOTALL)
HOME_TEMPLATE_DESTINATION = re.compile(
    r'<(?:a|img)\b[^>]*?(?:href|src)="\{\{\s*["\'](?P<destination>[^"\']+)'
    r'["\']\s*\|\s*url\s*\}\}"',
    re.DOTALL,
)
HOME_TEMPLATE_HERO_KEY = re.compile(r"\bhero\.([a-z_][a-z0-9_]*)\b")
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


def _markdown_destination_locations(text: str) -> list[tuple[int, str]]:
    """Return offsets and destinations, including both halves of linked images."""

    destinations: list[tuple[int, str]] = []
    for match in MARKDOWN_DESTINATION.finditer(text):
        if linked_image := match.group("linked_image"):
            destinations.append(
                (match.start(), _markdown_destination(linked_image))
            )
            destinations.append(
                (match.start(), _markdown_destination(match.group("linked_target")))
            )
        else:
            destinations.append(
                (match.start(), _markdown_destination(match.group("destination")))
            )
    return destinations


def _markdown_destinations(page: Path) -> list[str]:
    """Return ordered inline-link and image destinations from one page."""

    return [
        destination
        for _, destination in _markdown_destination_locations(
            page.read_text(encoding="utf-8")
        )
    ]


def _front_matter(page: Path) -> dict[str, object]:
    """Return a page's YAML front matter as a mapping."""

    text = page.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        return {}
    metadata = yaml.safe_load(match.group("yaml")) or {}
    if not isinstance(metadata, dict):
        raise ValueError("front matter must be a mapping")
    return metadata


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

    for relative, label in (
        (Path("index.md"), "Homepage"),
        (Path("tutorials/index.md"), "Tutorial index"),
    ):
        if _markdown_destinations(english / relative) != _markdown_destinations(
            chinese / relative
        ):
            errors.append(f"{label} link and image destinations differ between locales")

    home_template = MKDOCS_ROOT / "overrides" / "home.html"
    template_text = home_template.read_text(encoding="utf-8")
    template_hero_keys = set(HOME_TEMPLATE_HERO_KEY.findall(template_text))
    homepage_heroes: dict[str, dict[str, object]] = {}
    for locale, root in LOCALE_ROOTS.items():
        homepage = root / "index.md"
        try:
            metadata = _front_matter(homepage)
        except (ValueError, yaml.YAMLError) as error:
            errors.append(f"Invalid {locale} homepage front matter: {error}")
            continue
        if metadata.get("template") != "home.html":
            errors.append(f"{locale} homepage must select template: home.html")
        hero = metadata.get("hero")
        if not isinstance(hero, dict):
            errors.append(f"{locale} homepage hero metadata must be a mapping")
            continue
        empty = sorted(
            key
            for key, value in hero.items()
            if not isinstance(value, str) or not value.strip()
        )
        if empty:
            errors.append(
                f"{locale} homepage hero values must be non-empty strings: "
                + ", ".join(empty)
            )
        if set(hero) != template_hero_keys:
            missing = template_hero_keys - set(hero)
            extra = set(hero) - template_hero_keys
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if extra:
                details.append("unused " + ", ".join(sorted(extra)))
            errors.append(
                f"{locale} homepage hero metadata does not match home.html: "
                + "; ".join(details)
            )
        homepage_heroes[locale] = hero

    if set(homepage_heroes) == set(LOCALE_ROOTS) and set(
        homepage_heroes["en"]
    ) != set(homepage_heroes["zh"]):
        errors.append("Homepage hero metadata keys differ between locales")

    template_destinations = HOME_TEMPLATE_DESTINATION.findall(template_text)
    if not template_destinations:
        errors.append("Homepage template exposes no local CTA or image destinations")
    for locale, root in LOCALE_ROOTS.items():
        homepage = root / "index.md"
        for destination in template_destinations:
            if not _local_target_exists(homepage, root, destination):
                errors.append(
                    f"Broken local template destination for {locale}: {destination}"
                )

    for locale, root in LOCALE_ROOTS.items():
        for relative in sorted(markdown_paths[locale]):
            page = root / relative
            text = page.read_text(encoding="utf-8")
            if locale == "zh" and not CJK.search(text):
                errors.append(f"Chinese page has no CJK text: {relative.as_posix()}")
            for offset, destination in _markdown_destination_locations(text):
                if not _local_target_exists(page, root, destination):
                    line = text.count("\n", 0, offset) + 1
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
