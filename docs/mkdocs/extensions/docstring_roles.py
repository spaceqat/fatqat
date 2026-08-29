"""Render Python cross-reference roles in source docstrings as MkDocs links."""

from __future__ import annotations

import re
from typing import Any

from griffe import Extension, Object

ROLE = re.compile(
    r":(?:py:)?(?P<kind>class|meth|attr|data|exc|func):`(?P<body>[^`]+)`"
)
EXTERNAL_OBJECTS = {
    "pathlib.Path.read_text": (
        "https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text"
    ),
}


def _explicit_target(body: str) -> tuple[str | None, str]:
    match = re.match(r"^(?P<label>.+?)\s*<(?P<target>[^>]+)>$", body.strip())
    if match:
        return match.group("label").strip(), match.group("target").strip()
    return None, body.strip()


def _resolve_target(obj: Object, role: str, target: str) -> str:
    """Resolve a relative cross-reference from its documented object."""

    target = target.lstrip("~")
    if "." in target:
        return target
    if role in {"meth", "attr"} and obj.parent is not None:
        owner = obj.parent if obj.is_function else obj
        return f"{owner.path}.{target}"
    if obj.is_module:
        return f"{obj.path}.{target}"
    if obj.parent is not None:
        return f"{obj.parent.path}.{target}"
    return target


def _convert_role(match: re.Match[str], obj: Object) -> str:
    explicit_label, raw_target = _explicit_target(match.group("body"))
    target = _resolve_target(obj, match.group("kind"), raw_target)
    label = explicit_label or (
        raw_target.lstrip("~").rsplit(".", 1)[-1]
        if raw_target.startswith("~")
        else raw_target
    )
    rendered = f"`{label}`"
    if url := EXTERNAL_OBJECTS.get(target):
        return f"[{rendered}]({url})"
    return f"[{rendered}][{target}]"


class DocstringRolesExtension(Extension):
    """Convert Python object roles before mkdocstrings parses docstrings."""

    def on_object(self, *, obj: Object, **kwargs: Any) -> None:
        del kwargs
        if obj.docstring is None or not ROLE.search(obj.docstring.value):
            return
        obj.docstring.value = ROLE.sub(
            lambda match: _convert_role(match, obj), obj.docstring.value
        )
        obj.docstring.__dict__.pop("parsed", None)
