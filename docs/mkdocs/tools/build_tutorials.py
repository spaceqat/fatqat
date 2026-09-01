"""Discover, execute, and assemble English Markdown tutorials for MkDocs."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import nbformat
import yaml

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
MKDOCS_ROOT = SCRIPT_PATH.parents[1]
TUTORIAL_ROOT = MKDOCS_ROOT / "tutorial-sources"
SOURCE_ROOT = TUTORIAL_ROOT / "en"
GALLERY = yaml.safe_load((TUTORIAL_ROOT / "gallery.yml").read_text(encoding="utf-8"))
GALLERY_UI = GALLERY["ui"]["en"]
GALLERY_INDEX = GALLERY["index"]["en"]
PAGE_ROOT = MKDOCS_ROOT / "en" / "tutorials"
DOWNLOAD_ROOT = MKDOCS_ROOT / "en" / "downloads" / "tutorials"
ASSET_ROOT = MKDOCS_ROOT / "en" / "assets" / "generated" / "tutorials"
RESULT_ROOT = MKDOCS_ROOT / "tutorial-results"

FRONT_MATTER = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n", re.DOTALL)
PYTHON_CELL = re.compile(r"(?ms)^```python(?:[^\n]*)\n(?P<code>.*?)^```(?=\n|$)")


@dataclass(frozen=True)
class TutorialSource:
    """Authored metadata and body for one tutorial."""

    path: Path
    title: str
    description: str
    icon: str
    figure_alts: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class Tutorial:
    """One tutorial and its executable Python cells."""

    category: str
    slug: str
    source: TutorialSource
    code_cells: tuple[str, ...]


def _write_text(path: Path, content: str) -> None:
    content = content.rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha256(tutorial: Tutorial) -> str:
    """Hash only executable cells so prose edits can reuse runtime results."""

    payload = "\n\n# %%\n\n".join(tutorial.code_cells).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_source(path: Path) -> TutorialSource:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        raise ValueError(f"{path}: expected YAML front matter")
    metadata = yaml.safe_load(match.group("yaml"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: front matter must be a mapping")

    def required_text(name: str) -> str:
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {name} must be non-empty text")
        return value.strip()

    icon = metadata.get("icon", "material-flask-outline")
    if not isinstance(icon, str) or not re.fullmatch(r"material-[a-z0-9-]+", icon):
        raise ValueError(f"{path}: icon must be a Material icon name")
    raw_alts = metadata.get("figure_alts")
    if (
        not isinstance(raw_alts, list)
        or not raw_alts
        or not all(isinstance(alt, str) and alt.strip() for alt in raw_alts)
    ):
        raise ValueError(f"{path}: figure_alts must contain at least one label")
    return TutorialSource(
        path=path,
        title=required_text("title"),
        description=required_text("description"),
        icon=icon,
        figure_alts=tuple(alt.strip() for alt in raw_alts),
        body=text[match.end() :].strip(),
    )


def discover_tutorials() -> tuple[Tutorial, ...]:
    """Discover English sources; their parent folder is the category."""

    inventory = {
        path.relative_to(SOURCE_ROOT): path
        for path in SOURCE_ROOT.rglob("*.md")
        if not path.name.startswith("_")
    }

    tutorials: list[Tutorial] = []
    slugs: set[str] = set()
    for relative in sorted(inventory):
        if relative.parent == Path("."):
            raise ValueError(f"{relative}: put each tutorial in a category subfolder")
        slug = relative.stem
        if slug in slugs:
            raise ValueError(f"tutorial slug {slug!r} occurs in more than one category")
        slugs.add(slug)
        source = _parse_source(inventory[relative])
        code_cells = tuple(
            match.group("code").rstrip() for match in PYTHON_CELL.finditer(source.body)
        )
        if not code_cells:
            raise ValueError(
                f"{source.path}: expected at least one top-level Python block"
            )
        tutorials.append(
            Tutorial(
                category=relative.parent.as_posix(),
                slug=slug,
                source=source,
                code_cells=code_cells,
            )
        )
    if not tutorials:
        raise ValueError(f"no tutorials found under {SOURCE_ROOT}")
    return tuple(tutorials)


def _result_path(tutorial: Tutorial) -> Path:
    return RESULT_ROOT / f"{tutorial.slug}.json"


def _execute_tutorial(tutorial: Tutorial, output: Path) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None
    namespace: dict[str, object] = {"__name__": "__main__", "__package__": None}
    figure_number = 0
    captured: dict[str, dict[str, object]] = {}
    for cell_number, code in enumerate(tutorial.code_cells, start=1):
        plt.close("all")
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(
                    compile(
                        code + "\n",
                        f"{tutorial.source.path}#cell-{cell_number}",
                        "exec",
                    ),
                    namespace,
                )
        except Exception as error:
            raise RuntimeError(
                f"{tutorial.source.path}: execution failed in "
                f"Python cell {cell_number}"
            ) from error

        figures: list[dict[str, str]] = []
        for matplotlib_number in plt.get_fignums():
            figure_number += 1
            name = f"{tutorial.slug}-{figure_number:02d}.png"
            asset = output / name
            plt.figure(matplotlib_number).savefig(
                asset,
                dpi=144,
                bbox_inches="tight",
                facecolor="white",
                metadata={"Software": "fatqat Markdown tutorial builder"},
            )
            figures.append({"name": name, "sha256": _sha256(asset)})
        plt.close("all")
        captured[str(cell_number)] = {
            "stdout": stdout.getvalue().replace("\r\n", "\n").rstrip(),
            "figures": figures,
        }

    if figure_number != len(tutorial.source.figure_alts):
        raise ValueError(
            f"{tutorial.source.path}: generated {figure_number} figures "
            f"but front matter defines "
            f"{len(tutorial.source.figure_alts)} figure_alts"
        )
    return {
        "schema": 2,
        "source": str(tutorial.source.path.relative_to(MKDOCS_ROOT)).replace("\\", "/"),
        "code_sha256": _code_sha256(tutorial),
        "cells": captured,
    }


def capture_all(tutorials: tuple[Tutorial, ...]) -> None:
    """Execute all discovered tutorials before atomically refreshing the cache."""

    source_path = str(REPOSITORY_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    original_directory = Path.cwd()
    manifests: dict[str, dict[str, object]] = {}
    try:
        os.chdir(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory(prefix="fatqat-mkdocs-tutorials-") as temp:
            temporary_assets = Path(temp)
            for tutorial in tutorials:
                print(
                    "Executing " f"{tutorial.source.path.relative_to(REPOSITORY_ROOT)}"
                )
                manifests[tutorial.slug] = _execute_tutorial(tutorial, temporary_assets)

            expected = {
                figure["name"]
                for manifest in manifests.values()
                for cell in manifest["cells"].values()
                for figure in cell["figures"]
            }
            ASSET_ROOT.mkdir(parents=True, exist_ok=True)
            for stale in ASSET_ROOT.glob("*.png"):
                if stale.name not in expected:
                    stale.unlink()
            for name in sorted(expected):
                shutil.copyfile(temporary_assets / name, ASSET_ROOT / name)

            RESULT_ROOT.mkdir(parents=True, exist_ok=True)
            for stale in RESULT_ROOT.glob("*.json"):
                if stale.stem not in manifests:
                    stale.unlink()
            for slug, manifest in manifests.items():
                _write_text(
                    RESULT_ROOT / f"{slug}.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                )
    finally:
        os.chdir(original_directory)


def _load_results(tutorial: Tutorial) -> dict[int, dict[str, object]]:
    result_path = _result_path(tutorial)
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("schema") != 2 or payload.get("code_sha256") != _code_sha256(
        tutorial
    ):
        raise ValueError(f"{result_path}: runtime cache is stale")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, dict) or len(raw_cells) != len(tutorial.code_cells):
        raise ValueError(f"{result_path}: invalid cell inventory")
    results: dict[int, dict[str, object]] = {}
    figure_count = 0
    for number in range(1, len(tutorial.code_cells) + 1):
        raw = raw_cells.get(str(number))
        if not isinstance(raw, dict) or not isinstance(raw.get("stdout"), str):
            raise ValueError(f"{result_path}: invalid cell {number}")
        figures = raw.get("figures")
        if not isinstance(figures, list):
            raise ValueError(f"{result_path}: invalid figures for cell {number}")
        for figure in figures:
            figure_count += 1
            name = figure.get("name") if isinstance(figure, dict) else None
            digest = figure.get("sha256") if isinstance(figure, dict) else None
            if name != f"{tutorial.slug}-{figure_count:02d}.png" or not isinstance(
                digest, str
            ):
                raise ValueError(f"{result_path}: invalid figure {figure_count}")
            asset = ASSET_ROOT / name
            if not asset.is_file() or _sha256(asset) != digest:
                raise ValueError(f"{result_path}: missing or modified {asset}")
        results[number] = raw
    if figure_count != len(tutorial.source.figure_alts):
        raise ValueError(f"{result_path}: figure count does not match source metadata")
    return results


def cache_is_current(tutorials: tuple[Tutorial, ...]) -> bool:
    try:
        for tutorial in tutorials:
            _load_results(tutorial)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return True


def _result_block(
    tutorial: Tutorial,
    cell_number: int,
    result: dict[str, object],
    first_figure: int,
) -> tuple[str, int]:
    lines = [f'!!! example "{GALLERY_UI["runtime_result"]}"', ""]
    stdout = result["stdout"]
    if stdout:
        lines.append("    ```text")
        lines.extend(f"    {line}" if line else "" for line in stdout.splitlines())
        lines.extend(("    ```", ""))
    figure_number = first_figure
    for figure in result["figures"]:
        name = figure["name"]
        alt = tutorial.source.figure_alts[figure_number - 1]
        lines.extend((f"    ![{alt}](../assets/generated/tutorials/{name})", ""))
        figure_number += 1
    return "\n".join(lines).rstrip(), figure_number


def _render_body(
    tutorial: Tutorial,
    results: dict[int, dict[str, object]],
) -> str:
    figure_number = 1
    cell_number = 0

    def code_block(code: str) -> str:
        nonlocal cell_number, figure_number
        cell_number += 1
        label = GALLERY_UI["python_cell"]
        rendered = f'```python title="{label} {cell_number}"\n{code}\n```'
        result_block, figure_number = _result_block(
            tutorial, cell_number, results[cell_number], figure_number
        )
        result = results[cell_number]
        if (
            result_block.endswith('"')
            and not result["stdout"]
            and not result["figures"]
        ):
            return rendered
        return rendered + "\n\n" + result_block

    return PYTHON_CELL.sub(
        lambda match: code_block(match.group("code").rstrip()), tutorial.source.body
    )


def _category_label(category: str) -> tuple[str, str]:
    if category in GALLERY["categories"]:
        content = GALLERY["categories"][category]["en"]
        return content["title"], content["description"]
    title = category.rsplit("/", maxsplit=1)[-1].replace("-", " ").title()
    description = GALLERY_UI["default_category_description"].format(title=title)
    return title, description


def _insert_page_header(tutorial: Tutorial, body: str) -> str:
    category_title, _ = _category_label(tutorial.category)
    labels = (
        GALLERY_UI["track"],
        GALLERY_UI["downloads"],
        GALLERY_UI["python_script"],
        GALLERY_UI["notebook"],
    )
    download_root = f"../downloads/tutorials/{tutorial.slug}"
    preamble = "\n".join(
        (
            '<div class="grid cards" markdown>',
            "",
            f"-   :material-map-marker-path: **{labels[0]}**",
            "",
            f"    {category_title}",
            "",
            f"-   :material-download: **{labels[1]}**",
            "",
            f"    [{labels[2]} `.py`]({download_root}.py)"
            f'{{ download="{tutorial.slug}.py" }} \u00b7 '
            f"[{labels[3]} `.ipynb`]({download_root}.ipynb)"
            f'{{ download="{tutorial.slug}.ipynb" }}',
            "",
            "</div>",
        )
    )
    heading = re.search(r"(?m)^# .+$", body)
    if not heading:
        raise ValueError(f"{tutorial.source.path}: expected one level-one heading")
    return body[: heading.end()] + "\n\n" + preamble + body[heading.end() :]


def _render_page(
    tutorial: Tutorial,
    results: dict[int, dict[str, object]],
) -> str:
    body = _insert_page_header(tutorial, _render_body(tutorial, results))
    return "\n".join(
        (
            "---",
            f"title: {json.dumps(tutorial.source.title, ensure_ascii=False)}",
            f"description: {json.dumps(tutorial.source.description, ensure_ascii=False)}",
            "---",
            "<!-- Generated from docs/mkdocs/tutorial-sources; do not edit here. -->",
            "",
            body,
        )
    )


def _render_download(tutorial: Tutorial) -> str:
    lines = [
        f'"""{tutorial.source.title}\n\n{tutorial.source.description}\n"""',
        "",
    ]
    for code in tutorial.code_cells:
        lines.extend(("# %%", code, ""))
    return "\n".join(lines)


def _render_notebook(tutorial: Tutorial) -> str:
    """Create an unexecuted notebook from the authored Markdown cells."""

    source = tutorial.source
    cells = []
    cursor = 0

    def cell_id(kind: str) -> str:
        identity = f"{tutorial.category}/{tutorial.slug}:en:{len(cells)}:{kind}"
        return hashlib.sha256(identity.encode()).hexdigest()[:12]

    for match in PYTHON_CELL.finditer(source.body):
        if prose := source.body[cursor : match.start()].strip():
            cells.append(
                nbformat.v4.new_markdown_cell(
                    prose,
                    id=cell_id("markdown"),
                )
            )
        code = match.group("code").rstrip()
        cells.append(
            nbformat.v4.new_code_cell(
                code,
                id=cell_id("code"),
            )
        )
        cursor = match.end()
    if prose := source.body[cursor:].strip():
        cells.append(
            nbformat.v4.new_markdown_cell(
                prose,
                id=cell_id("markdown"),
            )
        )

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    nbformat.validate(notebook)
    return nbformat.writes(notebook, version=4)


def _render_index(tutorials: tuple[Tutorial, ...]) -> str:
    content = GALLERY_INDEX
    lines = [
        "---",
        f"title: {json.dumps(content['title'], ensure_ascii=False)}",
        f"description: {json.dumps(content['description'], ensure_ascii=False)}",
        "---",
        "<!-- Generated from tutorial category folders; do not edit here. -->",
        "",
        f"# {content['title']}",
        "",
        content["introduction"],
        "",
        f'!!! tip "{content["tip_title"]}"',
        "",
        f"    {content['tip']}",
        "",
    ]
    discovered = {tutorial.category for tutorial in tutorials}
    configured = tuple(GALLERY["category_order"])
    unknown = set(configured) - discovered
    if len(configured) != len(set(configured)) or unknown:
        raise ValueError(
            "gallery category_order must contain unique source folders; "
            f"unknown={sorted(unknown)}"
        )
    categories = sorted(
        discovered,
        key=lambda category: (
            configured.index(category) if category in configured else len(configured),
            category,
        ),
    )
    for category in categories:
        title, description = _category_label(category)
        lines.extend(
            (
                f"## {title}",
                "",
                description,
                "",
                '<div class="grid cards" markdown>',
                "",
            )
        )
        for tutorial in (item for item in tutorials if item.category == category):
            source = tutorial.source
            thumbnail = f"{tutorial.slug}-01.png"
            lines.extend(
                (
                    f"-   [![{source.figure_alts[0]}](../assets/generated/tutorials/{thumbnail})"
                    f"{{ loading=lazy }}]({tutorial.slug}.md)",
                    "",
                    f"    :{source.icon}:{{ .lg .middle }} **{source.title}**",
                    "",
                    "    ---",
                    "",
                    f"    {source.description}",
                    "",
                    f"    [:material-arrow-right: {content['open']}]({tutorial.slug}.md)",
                    "",
                )
            )
        lines.extend(("</div>", ""))
    return "\n".join(lines)


def assemble_all(tutorials: tuple[Tutorial, ...]) -> None:
    expected_pages = {"index.md", *(f"{tutorial.slug}.md" for tutorial in tutorials)}
    expected_downloads = {
        f"{tutorial.slug}.{suffix}"
        for tutorial in tutorials
        for suffix in ("ipynb", "py")
    }
    for root in (PAGE_ROOT, DOWNLOAD_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    _write_text(PAGE_ROOT / "index.md", _render_index(tutorials))
    for stale in PAGE_ROOT.glob("*.md"):
        if stale.name not in expected_pages:
            stale.unlink()
    for tutorial in tutorials:
        results = _load_results(tutorial)
        _write_text(
            PAGE_ROOT / f"{tutorial.slug}.md",
            _render_page(tutorial, results),
        )
        _write_text(
            DOWNLOAD_ROOT / f"{tutorial.slug}.py",
            _render_download(tutorial),
        )
        _write_text(
            DOWNLOAD_ROOT / f"{tutorial.slug}.ipynb",
            _render_notebook(tutorial),
        )
    for pattern in ("*.ipynb", "*.py"):
        for stale in DOWNLOAD_ROOT.glob(pattern):
            if stale.name not in expected_downloads:
                stale.unlink()
    print(f"Assembled {len(tutorials)} English tutorials from category folders.")


def build_all(
    *,
    force_execution: bool = False,
    execute_if_needed: bool = True,
) -> None:
    """Refresh runtime results when needed and assemble all tutorials."""

    tutorials = discover_tutorials()
    if force_execution or (execute_if_needed and not cache_is_current(tutorials)):
        capture_all(tutorials)
    assemble_all(tutorials)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute", action="store_true", help="force tutorial execution"
    )
    execution.add_argument(
        "--execute-if-needed",
        action="store_true",
        help="execute when the code-hashed build cache is absent or stale",
    )
    args = parser.parse_args()
    build_all(
        force_execution=args.execute,
        execute_if_needed=args.execute_if_needed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
