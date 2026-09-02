"""Validate the English MkDocs content and its generated files."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

MKDOCS_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = MKDOCS_ROOT / "en"

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
GENERATED_ASSET_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}


def _relative_files(root: Path, *, suffixes: set[str]) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _markdown_destination(raw_destination: str) -> str:
    """Return a Markdown link destination without an optional title."""

    destination = raw_destination.strip()
    if destination.startswith("<") and ">" in destination:
        return destination[1 : destination.index(">")]
    return re.split(r"\s+[\"']", destination, maxsplit=1)[0]


def _local_target_exists(page: Path, destination: str) -> bool:
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


def _markdown_destination_locations(text: str) -> list[tuple[int, str]]:
    """Return offsets and destinations, including both halves of linked images."""

    destinations: list[tuple[int, str]] = []
    for match in MARKDOWN_DESTINATION.finditer(text):
        if linked_image := match.group("linked_image"):
            destinations.append((match.start(), _markdown_destination(linked_image)))
            destinations.append(
                (match.start(), _markdown_destination(match.group("linked_target")))
            )
        else:
            destinations.append(
                (match.start(), _markdown_destination(match.group("destination")))
            )
    return destinations


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
    markdown_paths = _relative_files(DOCS_ROOT, suffixes={".md"})

    home_template = MKDOCS_ROOT / "overrides" / "home.html"
    template_text = home_template.read_text(encoding="utf-8")
    template_hero_keys = set(HOME_TEMPLATE_HERO_KEY.findall(template_text))
    homepage = DOCS_ROOT / "index.md"
    if not homepage.is_file():
        errors.append("en homepage is missing")
    else:
        try:
            metadata = _front_matter(homepage)
        except (ValueError, yaml.YAMLError) as error:
            errors.append(f"Invalid en homepage front matter: {error}")
        else:
            if metadata.get("template") != "home.html":
                errors.append("en homepage must select template: home.html")
            hero = metadata.get("hero")
            if not isinstance(hero, dict):
                errors.append("en homepage hero metadata must be a mapping")
            else:
                empty = sorted(
                    key
                    for key, value in hero.items()
                    if not isinstance(value, str) or not value.strip()
                )
                if empty:
                    errors.append(
                        "en homepage hero values must be non-empty strings: "
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
                        "en homepage hero metadata does not match home.html: "
                        + "; ".join(details)
                    )

    template_destinations = HOME_TEMPLATE_DESTINATION.findall(template_text)
    if not template_destinations:
        errors.append("Homepage template exposes no local CTA or image destinations")
    if homepage.is_file():
        for destination in template_destinations:
            if not _local_target_exists(homepage, destination):
                errors.append(
                    f"Broken local template destination for en: {destination}"
                )

    for relative in sorted(markdown_paths):
        page = DOCS_ROOT / relative
        text = page.read_text(encoding="utf-8")
        for offset, destination in _markdown_destination_locations(text):
            if not _local_target_exists(page, destination):
                line = text.count("\n", 0, offset) + 1
                errors.append(
                    f"Broken local link in en/{relative.as_posix()}:{line}: "
                    f"{destination}"
                )

    if not errors:
        asset_count = len(
            _relative_files(DOCS_ROOT / "assets", suffixes=GENERATED_ASSET_SUFFIXES)
        )
        download_count = len(
            _relative_files(DOCS_ROOT / "downloads", suffixes={".ipynb", ".py"})
        )
        print(
            f"Validated English documentation content: {len(markdown_paths)} "
            "Markdown pages, "
            f"{asset_count} assets, and {download_count} downloads."
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
