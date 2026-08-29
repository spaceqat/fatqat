"""Convert the Sphinx API reference from reStructuredText to MkDocs Markdown.

The Sphinx API pages contain substantial hand-written reference material, so
they remain the source of truth while the Material experiment is evaluated.
This converter deliberately supports the closed set of constructs used by
``docs/sphinx/api`` and raises on an unknown directive instead of silently
discarding documentation.

Run from anywhere with::

    python docs/mkdocs/tools/convert_api.py

Use ``--check`` in CI to verify that the committed Markdown matches the RST
sources without rewriting any files.
"""

from __future__ import annotations

import argparse
import html
import os
import posixpath
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
SPHINX_ROOT = REPOSITORY_ROOT / "docs" / "sphinx"
API_SOURCE_ROOT = SPHINX_ROOT / "api"
MKDOCS_LOCALE_ROOT = SCRIPT_PATH.parents[1] / "en"
API_OUTPUT_ROOT = MKDOCS_LOCALE_ROOT / "api"

HEADING_LEVELS = {"=": 1, "-": 2, "~": 3}
AUTODOC_KINDS = {
    "autoclass",
    "autofunction",
    "automethod",
    "autoattribute",
    "autodata",
    "autoexception",
}
MANUAL_PYTHON_KINDS = {
    "module",
    "class",
    "method",
    "attribute",
    "function",
    "exception",
    "type",
    "data",
}
OBJECT_ROLE_KINDS = "class|meth|attr|data|exc|func"
ROLE_RE = re.compile(rf":(?:py:)?(?P<kind>{OBJECT_ROLE_KINDS}):`(?P<body>[^`]+)`")
DOC_RE = re.compile(r":doc:`(?P<body>[^`]+)`")
REF_RE = re.compile(r":ref:`(?P<body>[^`]+)`")
MATH_RE = re.compile(r":math:`(?P<body>[^`]+)`")
EXTERNAL_LINK_RE = re.compile(r"`(?P<label>[^`<>]+)\s*<(?P<url>https?://[^>]+)>`_")
DIRECTIVE_RE = re.compile(r"^\.\. (?P<kind>[A-Za-z0-9_:-]+)::\s*(?P<value>.*)$")
OPTION_RE = re.compile(r"^:(?P<name>[A-Za-z0-9_-]+):\s*(?P<value>.*)$")
LABEL_RE = re.compile(r"^\.\. _(?P<label>[^:]+):\s*$")
ROLE_WITHOUT_CLOSING_TICK_RE = re.compile(
    r":(?:py:)?(?:class|meth|attr|data|exc|func|doc|ref|math):`[^`]*$"
)

BUILTIN_EXCEPTION_URLS = {
    "TypeError": "https://docs.python.org/3/library/exceptions.html#TypeError",
    "ValueError": "https://docs.python.org/3/library/exceptions.html#ValueError",
}

# Public re-exports are the canonical API paths exposed by mkdocstrings.  Some
# Sphinx sources use an implementation-module path, which Sphinx can resolve
# through Python-domain aliases but mkdocstrings deliberately does not emit.
OBJECT_IDENTIFIER_ALIASES = {
    "fatqat.emulator.superconducting.TransmonModel.time_unit": (
        "fatqat.emulator.TransmonModel.time_unit"
    ),
}

# These names are described as class/docstring attributes, but are not emitted
# as standalone mkdocstrings objects (and therefore have no autorefs anchor).
# Keep their visible API spelling as inline code instead of creating a link to
# a nonexistent target.
UNLINKED_OBJECT_REFERENCES = {
    "fatqat.Result.available_data",
    "fatqat.noise.Channel.num_subsystems",
    "fatqat.noise.PauliChannel.terms",
}


@dataclass(frozen=True)
class DocumentTitle:
    source_path: PurePosixPath
    title: str


@dataclass(frozen=True)
class LabelTarget:
    source_path: PurePosixPath
    title: str


@dataclass(frozen=True)
class ManualObjectTarget:
    source_path: PurePosixPath
    anchor: str


def source_relative(path: Path) -> PurePosixPath:
    return PurePosixPath(path.relative_to(SPHINX_ROOT).as_posix())


def output_path_for(source_path: PurePosixPath) -> PurePosixPath:
    return source_path.with_suffix(".md")


def is_heading_underline(line: str) -> bool:
    stripped = line.strip()
    return (
        len(stripped) >= 3 and len(set(stripped)) == 1 and stripped[0] in HEADING_LEVELS
    )


def first_document_title(lines: list[str], fallback: str) -> str:
    for index in range(len(lines) - 1):
        if lines[index].strip() and is_heading_underline(lines[index + 1]):
            return lines[index].strip()
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def coalesce_multiline_roles(lines: Iterable[str]) -> list[str]:
    """Join Sphinx roles and inline literals split across source lines."""

    source = list(lines)
    result: list[str] = []
    index = 0
    while index < len(source):
        line = source[index]
        while ROLE_WITHOUT_CLOSING_TICK_RE.search(line) or line.count("``") % 2:
            index += 1
            if index >= len(source):
                raise ValueError(f"Unterminated Sphinx role: {line!r}")
            line = f"{line.rstrip()} {source[index].strip()}"
        result.append(line)
        index += 1
    return result


def collect_titles() -> dict[PurePosixPath, DocumentTitle]:
    titles: dict[PurePosixPath, DocumentTitle] = {}
    for suffix in ("*.rst", "*.md"):
        for path in sorted(SPHINX_ROOT.rglob(suffix)):
            if "_build" in path.parts or "tutorials" in path.parts:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            relative = source_relative(path)
            title = first_document_title(lines, path.stem.replace("-", " ").title())
            titles[relative] = DocumentTitle(relative, title)
    return titles


def heading_after(lines: list[str], start: int, fallback: str) -> str:
    for index in range(start, min(len(lines) - 1, start + 8)):
        if not lines[index].strip():
            continue
        if is_heading_underline(lines[index + 1]):
            return lines[index].strip()
        break
    return fallback.replace("-", " ").title()


def collect_labels() -> dict[str, LabelTarget]:
    labels: dict[str, LabelTarget] = {}
    for path in sorted(API_SOURCE_ROOT.rglob("*.rst")):
        lines = path.read_text(encoding="utf-8").splitlines()
        relative = source_relative(path)
        for index, line in enumerate(lines):
            match = LABEL_RE.match(line)
            if not match:
                continue
            label = match.group("label")
            labels[label] = LabelTarget(
                relative,
                heading_after(lines, index + 1, label),
            )
    return labels


def strip_signature(value: str) -> str:
    return value.split("(", 1)[0].strip()


def qualify_object(
    value: str,
    *,
    current_module: str | None,
    parent_scope: str | None = None,
    member: bool = False,
) -> str:
    name = strip_signature(value).lstrip("~")
    if name.startswith("fatqat.") or name == "fatqat":
        return name
    if member and parent_scope:
        if name.startswith(parent_scope + "."):
            return name
        if "." not in name:
            return f"{parent_scope}.{name}"
    if current_module:
        if name.startswith(current_module + "."):
            return name
        return f"{current_module}.{name}"
    return name


def following_options(lines: list[str], start: int, indent: int) -> dict[str, str]:
    options: dict[str, str] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= indent:
            break
        option = OPTION_RE.match(line.strip())
        if not option:
            break
        options[option.group("name")] = option.group("value")
        index += 1
    return options


def collect_manual_objects() -> dict[str, ManualObjectTarget]:
    objects: dict[str, ManualObjectTarget] = {}
    member_kinds = {"method", "attribute"}
    for path in sorted(API_SOURCE_ROOT.rglob("*.rst")):
        lines = coalesce_multiline_roles(path.read_text(encoding="utf-8").splitlines())
        relative = source_relative(path)
        current_module: str | None = None
        scopes: list[tuple[int, str]] = []
        for index, line in enumerate(lines):
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            current = re.match(r"^\.\. currentmodule::\s*(\S+)\s*$", stripped)
            if current and indent == 0:
                current_module = current.group(1)
                continue
            manual = re.match(
                r"^\.\. py:(module|class|method|attribute|function|exception|type|data)::\s*(.+)$",
                stripped,
            )
            if not manual:
                continue
            kind, value = manual.groups()
            while scopes and scopes[-1][0] >= indent:
                scopes.pop()
            if kind == "module":
                current_module = strip_signature(value)
                objects[current_module] = ManualObjectTarget(relative, current_module)
                continue
            parent = scopes[-1][1] if scopes else None
            identifier = qualify_object(
                value,
                current_module=current_module,
                parent_scope=parent,
                member=kind in member_kinds,
            )
            objects[identifier] = ManualObjectTarget(relative, identifier)
            options = following_options(lines, index + 1, indent)
            canonical = options.get("canonical")
            if canonical:
                objects[canonical] = ManualObjectTarget(relative, canonical)
            if kind == "class":
                scopes.append((indent, identifier))
    return objects


def parse_explicit_target(body: str) -> tuple[str | None, str]:
    match = re.match(r"^(?P<label>.+?)\s*<(?P<target>[^>]+)>$", body.strip())
    if match:
        return match.group("label").strip(), match.group("target").strip()
    return None, body.strip()


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class RstApiConverter:
    def __init__(
        self,
        source_path: PurePosixPath,
        titles: dict[PurePosixPath, DocumentTitle],
        labels: dict[str, LabelTarget],
        manual_objects: dict[str, ManualObjectTarget],
    ) -> None:
        self.source_path = source_path
        self.output_path = output_path_for(source_path)
        self.titles = titles
        self.labels = labels
        self.manual_objects = manual_objects
        self.current_module: str | None = None
        self.stats: Counter[str] = Counter()

    def relative_output_link(
        self,
        target_source: PurePosixPath,
        *,
        anchor: str | None = None,
    ) -> str:
        target_output = output_path_for(target_source)
        if target_output == self.output_path:
            link = ""
        else:
            link = posixpath.relpath(
                str(target_output),
                str(self.output_path.parent),
            )
        if anchor:
            return f"{link}#{anchor}" if link else f"#{anchor}"
        return link or self.output_path.name

    def resolve_document(self, target: str) -> tuple[PurePosixPath, str]:
        clean = target.strip()
        if clean.startswith("/"):
            resolved = PurePosixPath(clean.lstrip("/"))
        else:
            resolved = PurePosixPath(
                posixpath.normpath(str(self.source_path.parent / clean))
            )
        candidates: list[PurePosixPath]
        if resolved.suffix:
            candidates = [resolved]
        else:
            candidates = [resolved.with_suffix(".rst"), resolved.with_suffix(".md")]
        for candidate in candidates:
            if candidate in self.titles:
                return candidate, self.titles[candidate].title
        fallback = resolved.name.replace("-", " ").replace("_", " ").title()
        suffix = resolved.suffix or ".md"
        return resolved.with_suffix(suffix), fallback

    def resolve_object_identifier(
        self,
        target: str,
        *,
        scope: str | None,
    ) -> str:
        clean = target.lstrip("~")
        clean = OBJECT_IDENTIFIER_ALIASES.get(clean, clean)
        if clean in BUILTIN_EXCEPTION_URLS:
            return clean
        if clean.startswith("fatqat.") or clean == "fatqat":
            return clean
        if scope and "." not in clean:
            scoped = f"{scope}.{clean}"
            if scoped in self.manual_objects:
                return scoped
        if self.current_module:
            return f"{self.current_module}.{clean}"
        return clean

    def convert_object_role(
        self,
        match: re.Match[str],
        *,
        scope: str | None,
    ) -> str:
        body = match.group("body")
        explicit_label, target = parse_explicit_target(body)
        shorten = target.startswith("~")
        identifier = self.resolve_object_identifier(target, scope=scope)
        if explicit_label is not None:
            label = explicit_label
        elif shorten:
            label = target.lstrip("~").rsplit(".", 1)[-1]
        else:
            label = target
        rendered_label = f"`{label}`"
        if identifier in BUILTIN_EXCEPTION_URLS:
            return f"[{rendered_label}]({BUILTIN_EXCEPTION_URLS[identifier]})"
        if identifier in UNLINKED_OBJECT_REFERENCES:
            return rendered_label
        manual = self.manual_objects.get(identifier)
        if manual:
            link = self.relative_output_link(manual.source_path, anchor=manual.anchor)
            return f"[{rendered_label}]({link})"
        return f"[{rendered_label}][{identifier}]"

    def convert_doc_role(self, match: re.Match[str]) -> str:
        explicit_label, target = parse_explicit_target(match.group("body"))
        target_path, target_title = self.resolve_document(target)
        label = explicit_label or target_title
        link = self.relative_output_link(target_path)
        return f"[{label}]({link})"

    def convert_ref_role(self, match: re.Match[str]) -> str:
        explicit_label, label = parse_explicit_target(match.group("body"))
        target = self.labels.get(label)
        if target is None:
            raise ValueError(
                f"Unknown Sphinx reference label {label!r} in {self.source_path}"
            )
        text = explicit_label or target.title
        link = self.relative_output_link(target.source_path, anchor=label)
        return f"[{text}]({link})"

    def inline(self, text: str, *, scope: str | None = None) -> str:
        text = DOC_RE.sub(self.convert_doc_role, text)
        text = REF_RE.sub(self.convert_ref_role, text)
        text = ROLE_RE.sub(
            lambda match: self.convert_object_role(match, scope=scope), text
        )
        text = MATH_RE.sub(lambda match: f"${match.group('body')}$", text)
        text = EXTERNAL_LINK_RE.sub(
            lambda match: f"[{match.group('label').strip()}]({match.group('url')})",
            text,
        )
        text = re.sub(r"``([^`]+)``", r"`\1`", text)
        return text

    @staticmethod
    def consume_indented(lines: list[str], index: int) -> tuple[list[str], int]:
        base_indent = len(lines[index]) - len(lines[index].lstrip())
        block: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            line = lines[cursor]
            if not line.strip():
                block.append(line)
                cursor += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                break
            block.append(line)
            cursor += 1
        return textwrap.dedent("\n".join(block)).splitlines(), cursor

    @staticmethod
    def split_options(block: list[str]) -> tuple[dict[str, str], list[str]]:
        options: dict[str, str] = {}
        cursor = 0
        while cursor < len(block) and not block[cursor].strip():
            cursor += 1
        while cursor < len(block):
            match = OPTION_RE.match(block[cursor])
            if not match:
                break
            options[match.group("name")] = match.group("value")
            cursor += 1
        while cursor < len(block) and not block[cursor].strip():
            cursor += 1
        return options, block[cursor:]

    def convert_autodoc(
        self, kind: str, value: str, options: dict[str, str]
    ) -> list[str]:
        identifier = qualify_object(value, current_module=self.current_module)
        output = [f"::: {identifier}"]
        rendered_options: list[tuple[str, bool | str | list[str]]] = []

        class_like = kind in {"autoclass", "autoexception"}
        if class_like:
            if "no-members" in options:
                rendered_options.append(("members", False))
            elif "members" in options and options["members"].strip():
                members = [part.strip() for part in options["members"].split(",")]
                rendered_options.append(("members", members))

            # Sphinx's conf.py enables public members for every class, but
            # mkdocstrings' ``members: true`` deliberately bypasses filters.
            # Leave members unset unless the source explicitly selects a list,
            # then reproduce Sphinx's private/special/excluded-member policy
            # with ordered filters.
            filters: list[str] = []
            if "no-members" not in options and not (
                "members" in options and options["members"].strip()
            ):
                filters.append("!^_")

            inherited = "no-inherited-members" not in options
            rendered_options.append(("inherited_members", inherited))
            rendered_options.append(("show_bases", "show-inheritance" in options))
            rendered_options.append(
                ("merge_init_into_class", options.get("class-doc-from") == "both")
            )

            excluded = [
                part.strip()
                for part in options.get("exclude-members", "").split(",")
                if part.strip()
            ]
            special = [
                part.strip()
                for part in options.get("special-members", "").split(",")
                if part.strip()
            ]
            if excluded or special:
                if special:
                    allowed = "|".join(re.escape(item) for item in special)
                    filters.append(rf"^(?:{allowed})$")
                if excluded:
                    names = "|".join(re.escape(item) for item in excluded)
                    filters.append(rf"!^(?:{names})$")
            if filters:
                rendered_options.append(("filters", filters))

        if kind == "autodata" and "no-value" in options:
            rendered_options.append(("show_attribute_values", False))

        if rendered_options:
            output.append("    options:")
            for name, option_value in rendered_options:
                if isinstance(option_value, bool):
                    scalar = "true" if option_value else "false"
                    output.append(f"      {name}: {scalar}")
                elif isinstance(option_value, list):
                    output.append(f"      {name}:")
                    for item in option_value:
                        output.append(f"        - {yaml_scalar(item)}")
                else:
                    output.append(f"      {name}: {yaml_scalar(option_value)}")
        self.stats[kind] += 1
        return output

    def table_cell(self, lines: list[str], *, scope: str | None) -> str:
        pieces: list[str] = []
        pending_break = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                pending_break = True
                continue
            if pending_break and pieces:
                pieces.append("<br><br>")
            elif pieces:
                pieces.append(" ")
            pending_break = False
            if stripped.startswith(("* ", "- ", "+ ")):
                if pieces and pieces[-1] == " ":
                    pieces[-1] = "<br>"
                pieces.append("• " + stripped[2:])
            else:
                pieces.append(stripped)
        value = self.inline("".join(pieces), scope=scope)
        # Python-Markdown's tables extension removes this escape before inline
        # parsing, including inside code and math spans. An HTML entity would
        # be displayed literally inside a Markdown code span.
        return value.replace("|", r"\|")

    def convert_list_table(
        self,
        title: str,
        options: dict[str, str],
        body: list[str],
        *,
        scope: str | None,
    ) -> list[str]:
        rows: list[list[list[str]]] = []
        current_row: list[list[str]] | None = None
        current_cell: list[str] | None = None
        for line in body:
            if line.startswith("* - ") or line == "* -":
                current_row = []
                rows.append(current_row)
                current_cell = [line[4:]]
                current_row.append(current_cell)
            elif line.startswith("  - ") or line == "  -":
                if current_row is None:
                    raise ValueError(f"Table cell without row in {self.source_path}")
                current_cell = [line[4:]]
                current_row.append(current_cell)
            elif current_cell is not None:
                current_cell.append(line[4:] if line.startswith("    ") else line)
            elif line.strip():
                raise ValueError(
                    f"Unsupported list-table line in {self.source_path}: {line!r}"
                )

        if not rows:
            raise ValueError(f"Empty list-table in {self.source_path}")
        width = max(len(row) for row in rows)
        if any(len(row) != width for row in rows):
            raise ValueError(f"Ragged list-table in {self.source_path}")

        rendered = [
            [self.table_cell(cell, scope=scope) for cell in row] for row in rows
        ]
        header_rows = int(options.get("header-rows", "0") or "0")
        output: list[str] = []
        if title:
            output.extend([f"**{self.inline(title, scope=scope)}**", ""])
        if header_rows == 1:
            header, data = rendered[0], rendered[1:]
        elif header_rows == 0:
            header = ["" for _ in range(width)]
            data = rendered
        else:
            raise ValueError(
                f"Only zero or one header row is supported in {self.source_path}"
            )
        output.append("| " + " | ".join(header) + " |")
        output.append("| " + " | ".join("---" for _ in header) + " |")
        output.extend("| " + " | ".join(row) + " |" for row in data)
        self.stats["list-table"] += 1
        return output

    def convert_toctree(self, options: dict[str, str], body: list[str]) -> list[str]:
        entries = [line.strip() for line in body if line.strip()]
        output: list[str] = []
        caption = options.get("caption")
        if caption:
            output.extend([f"## {self.inline(caption)}", ""])
        for entry in entries:
            explicit_label, target = parse_explicit_target(entry)
            target_path, title = self.resolve_document(target)
            link = self.relative_output_link(target_path)
            output.append(f"- [{explicit_label or title}]({link})")
        self.stats["toctree"] += 1
        return output

    def convert_manual_python(
        self,
        kind: str,
        value: str,
        options: dict[str, str],
        body: list[str],
        *,
        scope: str | None,
        object_depth: int,
    ) -> list[str]:
        if kind == "module":
            identifier = strip_signature(value)
            self.current_module = identifier
            output = [f'<a id="{html.escape(identifier, quote=True)}"></a>']
            if body:
                output.extend(
                    self.render_lines(body, scope=scope, object_depth=object_depth)
                )
            self.stats["py:module"] += 1
            return output

        identifier = qualify_object(
            value,
            current_module=self.current_module,
            parent_scope=scope,
            member=kind in {"method", "attribute"},
        )
        anchors = [identifier]
        canonical = options.get("canonical")
        if canonical and canonical not in anchors:
            anchors.append(canonical)
        # MkDocs registers heading IDs with autorefs.  A raw HTML anchor is
        # sufficient for a direct fragment link, but it is invisible to
        # autorefs' cross-page identifier inventory, so put the primary Python
        # identifier on the heading and reserve raw anchors for aliases.
        output = [
            f'<a id="{html.escape(anchor, quote=True)}"></a>' for anchor in anchors[1:]
        ]
        heading_level = min(6, 3 + object_depth)
        output.extend(
            [
                (f"{'#' * heading_level} {kind} `{value}` " f"{{ #{identifier} }}"),
                "",
            ]
        )
        declared_type = options.get("type")
        if declared_type:
            output.extend([f"**Type:** `{declared_type}`", ""])
        if body:
            child_scope = identifier if kind == "class" else scope
            output.extend(
                self.render_lines(
                    body,
                    scope=child_scope,
                    object_depth=object_depth + (1 if kind == "class" else 0),
                )
            )
        self.stats[f"py:{kind}"] += 1
        return output

    def convert_definition_list(
        self,
        term: str,
        body: list[str],
        *,
        scope: str | None,
        object_depth: int,
    ) -> list[str]:
        output = [f"**{self.inline(term, scope=scope)}**", ""]
        output.extend(self.render_lines(body, scope=scope, object_depth=object_depth))
        return output

    def render_lines(
        self,
        lines: list[str],
        *,
        scope: str | None = None,
        object_depth: int = 0,
    ) -> list[str]:
        output: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]

            if index + 1 < len(lines) and is_heading_underline(lines[index + 1]):
                marker = lines[index + 1].strip()[0]
                output.extend(
                    [
                        f"{'#' * HEADING_LEVELS[marker]} {self.inline(line.strip(), scope=scope)}",
                        "",
                    ]
                )
                index += 2
                continue

            label = LABEL_RE.match(line)
            if label:
                output.extend(
                    [
                        f'<a id="{html.escape(label.group("label"), quote=True)}"></a>',
                        "",
                    ]
                )
                self.stats["label"] += 1
                index += 1
                continue

            directive = DIRECTIVE_RE.match(line)
            if directive:
                kind = directive.group("kind")
                value = directive.group("value")
                block, next_index = self.consume_indented(lines, index)
                options, body = self.split_options(block)

                if kind == "currentmodule":
                    self.current_module = value.strip()
                    self.stats[kind] += 1
                elif kind in AUTODOC_KINDS:
                    output.extend(self.convert_autodoc(kind, value, options))
                elif kind == "list-table":
                    output.extend(
                        self.convert_list_table(value, options, body, scope=scope)
                    )
                elif kind == "math":
                    output.extend(["$$", *body, "$$"])
                    self.stats[kind] += 1
                elif kind == "code-block":
                    language = value.strip() or "text"
                    output.extend([f"```{language}", *body, "```"])
                    self.stats[kind] += 1
                elif kind == "toctree":
                    output.extend(self.convert_toctree(options, body))
                elif kind.startswith("py:") and kind[3:] in MANUAL_PYTHON_KINDS:
                    output.extend(
                        self.convert_manual_python(
                            kind[3:],
                            value,
                            options,
                            body,
                            scope=scope,
                            object_depth=object_depth,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported directive {kind!r} in {self.source_path}"
                    )
                output.append("")
                index = next_index
                continue

            # RST definition lists otherwise become accidental code blocks in
            # Markdown. There are five such model-property entries today.
            if (
                line.strip()
                and not line.startswith((" ", "* ", "- ", "+ ", "#."))
                and index + 1 < len(lines)
                and lines[index + 1].startswith("   ")
            ):
                body, next_index = self.consume_indented(lines, index)
                output.extend(
                    self.convert_definition_list(
                        line.strip(),
                        body,
                        scope=scope,
                        object_depth=object_depth,
                    )
                )
                output.append("")
                index = next_index
                continue

            output.append(self.inline(line, scope=scope))
            index += 1

        # More than two consecutive blank lines add no meaning and make the
        # generated files harder to inspect.
        compact: list[str] = []
        for line in output:
            if not line and len(compact) >= 2 and compact[-1] == compact[-2] == "":
                continue
            compact.append(line.rstrip())
        return compact

    def convert(self, source: str) -> str:
        lines = coalesce_multiline_roles(source.splitlines())
        rendered = self.render_lines(lines)
        title = self.titles[self.source_path].title
        header = [
            "---",
            f"title: {yaml_scalar(title)}",
            "---",
            "<!-- Generated by docs/mkdocs/tools/convert_api.py; edit docs/sphinx/api instead. -->",
            "",
        ]
        result = "\n".join(header + rendered).rstrip() + "\n"
        leftovers = {
            "directive": re.search(r"^\.\. [A-Za-z0-9_:-]+::", result, re.MULTILINE),
            "object role": re.search(rf":(?:py:)?(?:{OBJECT_ROLE_KINDS}):`", result),
            "document role": re.search(r":(?:doc|ref|math):`", result),
            "RST literal": re.search(r"(?<!`)``(?!`)[^`]+(?<!`)``(?!`)", result),
        }
        remaining = [name for name, match in leftovers.items() if match]
        if remaining:
            raise ValueError(
                f"Unconverted constructs in {self.source_path}: {', '.join(remaining)}"
            )
        return result


def generate() -> tuple[dict[Path, str], Counter[str]]:
    if not API_SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing Sphinx API tree: {API_SOURCE_ROOT}")
    titles = collect_titles()
    labels = collect_labels()
    manual_objects = collect_manual_objects()
    generated: dict[Path, str] = {}
    totals: Counter[str] = Counter()
    for source_path in sorted(API_SOURCE_ROOT.rglob("*.rst")):
        relative = source_relative(source_path)
        converter = RstApiConverter(relative, titles, labels, manual_objects)
        result = converter.convert(source_path.read_text(encoding="utf-8"))
        destination = MKDOCS_LOCALE_ROOT / Path(*output_path_for(relative).parts)
        generated[destination] = result
        totals.update(converter.stats)
    return generated, totals


def check_or_write(generated: dict[Path, str], *, check: bool) -> int:
    stale: list[Path] = []
    for path, expected in generated.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8", newline="\n")
    if stale:
        for path in stale:
            print(f"out of date: {path.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated Markdown differs from the current files",
    )
    arguments = parser.parse_args()
    generated, stats = generate()
    status = check_or_write(generated, check=arguments.check)
    action = "checked" if arguments.check else "generated"
    print(f"{action} {len(generated)} API pages")
    print(
        "converted constructs: "
        + ", ".join(f"{name}={count}" for name, count in sorted(stats.items()))
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
