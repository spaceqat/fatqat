"""Run documentation generation and validation in the MkDocs lifecycle."""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser

from docs.mkdocs.tools import build_tutorials
from docs.mkdocs.tools import render_guide_figures
from docs.mkdocs.tools import validate_content


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


def on_pre_build(config) -> None:
    """Generate and validate documentation inputs before MkDocs collects them."""

    del config
    render_guide_figures.render_all()
    build_tutorials.build_all(execute_if_needed=True)

    errors = validate_content.validate()
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"Content validation failed:\n{details}")


def on_post_page(output: str, page, config) -> str:
    """Reject duplicate HTML IDs in each rendered page."""

    del config
    parser = _IdParser()
    parser.feed(output)
    duplicates = sorted(
        identifier for identifier, count in parser.ids.items() if count > 1
    )
    if duplicates:
        preview = ", ".join(duplicates[:8])
        suffix = "..." if len(duplicates) > 8 else ""
        raise RuntimeError(
            f"{page.file.src_uri}: duplicate generated HTML IDs {preview}{suffix}"
        )
    return output
