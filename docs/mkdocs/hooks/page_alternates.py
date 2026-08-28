"""Keep Material's language links on the equivalent localized page."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page

_alternate_roots: tuple[dict[str, Any], ...] = ()


def _page_link(root: str, page_url: str) -> str:
    """Append a MkDocs page URL to a locale root without changing its origin."""

    return f"{root.rstrip('/')}/{page_url.lstrip('/')}"


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Remember the configured locale roots before page rendering mutates them."""

    global _alternate_roots
    alternates = config.extra.get("alternate", ())
    _alternate_roots = tuple(dict(alternate) for alternate in alternates)
    return config


def on_page_context(
    context: dict[str, Any],
    *,
    page: Page,
    config: MkDocsConfig,
    nav: Any,
) -> Mapping[str, Any]:
    """Expose page-specific alternate URLs to Material's head and selector."""

    del nav
    config.extra["alternate"] = [
        {**alternate, "link": _page_link(str(alternate["link"]), page.url)}
        for alternate in _alternate_roots
    ]
    return context
