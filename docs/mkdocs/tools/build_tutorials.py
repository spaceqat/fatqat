"""Discover, execute, and assemble bilingual Markdown tutorials for MkDocs."""

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

import yaml


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
MKDOCS_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = MKDOCS_ROOT / "tutorial-sources"
GALLERY = yaml.safe_load((SOURCE_ROOT / "gallery.yml").read_text(encoding="utf-8"))
PAGE_ROOTS = {locale: MKDOCS_ROOT / locale / "tutorials" for locale in ("en", "zh")}
DOWNLOAD_ROOTS = {
    locale: MKDOCS_ROOT / locale / "downloads" / "tutorials"
    for locale in ("en", "zh")
}
ASSET_ROOTS = {
    locale: MKDOCS_ROOT / locale / "assets" / "generated" / "tutorials"
    for locale in ("en", "zh")
}
RESULT_ROOT = MKDOCS_ROOT / "tutorial-results"

FRONT_MATTER = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n", re.DOTALL)
PYTHON_CELL = re.compile(
    r"(?ms)^```python(?:[^\n]*)\n(?P<code>.*?)^```(?=\n|$)"
)
CODE_PLACEHOLDER = re.compile(r"<!-- tutorial-code-cell -->")

@dataclass(frozen=True)
class LocalizedSource:
    """Authored metadata and body for one locale."""

    path: Path
    title: str
    description: str
    icon: str
    figure_alts: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class Tutorial:
    """One automatically discovered bilingual tutorial pair."""

    category: str
    slug: str
    en: LocalizedSource
    zh: LocalizedSource
    code_cells: tuple[str, ...]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha256(tutorial: Tutorial) -> str:
    """Hash only executable cells so prose edits can reuse runtime results."""

    payload = "\n\n# %%\n\n".join(tutorial.code_cells).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_source(path: Path) -> LocalizedSource:
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
    if not isinstance(raw_alts, list) or not raw_alts or not all(
        isinstance(alt, str) and alt.strip() for alt in raw_alts
    ):
        raise ValueError(f"{path}: figure_alts must contain at least one label")
    return LocalizedSource(
        path=path,
        title=required_text("title"),
        description=required_text("description"),
        icon=icon,
        figure_alts=tuple(alt.strip() for alt in raw_alts),
        body=text[match.end() :].strip(),
    )


def discover_tutorials() -> tuple[Tutorial, ...]:
    """Discover mirrored locale files; their parent folder is the category."""

    roots = {locale: SOURCE_ROOT / locale for locale in ("en", "zh")}
    inventories = {
        locale: {
            path.relative_to(root): path
            for path in root.rglob("*.md")
            if not path.name.startswith("_")
        }
        for locale, root in roots.items()
    }
    if inventories["en"].keys() != inventories["zh"].keys():
        raise ValueError(
            "tutorial translations must mirror English source paths: "
            f"missing_zh={sorted(map(str, inventories['en'].keys() - inventories['zh'].keys()))}, "
            f"missing_en={sorted(map(str, inventories['zh'].keys() - inventories['en'].keys()))}"
        )

    tutorials: list[Tutorial] = []
    slugs: set[str] = set()
    for relative in sorted(inventories["en"]):
        if relative.parent == Path("."):
            raise ValueError(f"{relative}: put each tutorial in a category subfolder")
        slug = relative.stem
        if slug in slugs:
            raise ValueError(f"tutorial slug {slug!r} occurs in more than one category")
        slugs.add(slug)
        en = _parse_source(inventories["en"][relative])
        zh = _parse_source(inventories["zh"][relative])
        code_cells = tuple(match.group("code").rstrip() for match in PYTHON_CELL.finditer(en.body))
        if not code_cells:
            raise ValueError(f"{en.path}: expected at least one top-level Python block")
        placeholder_count = len(CODE_PLACEHOLDER.findall(zh.body))
        if placeholder_count != len(code_cells):
            raise ValueError(
                f"{zh.path}: expected {len(code_cells)} tutorial-code-cell placeholders, "
                f"found {placeholder_count}"
            )
        if en.icon != zh.icon:
            raise ValueError(f"{relative}: icon differs between locales")
        if len(en.figure_alts) != len(zh.figure_alts):
            raise ValueError(f"{relative}: figure_alts length differs between locales")
        tutorials.append(
            Tutorial(
                category=relative.parent.as_posix(),
                slug=slug,
                en=en,
                zh=zh,
                code_cells=code_cells,
            )
        )
    if not tutorials:
        raise ValueError(f"no tutorial pairs found under {SOURCE_ROOT}")
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
                    compile(code + "\n", f"{tutorial.en.path}#cell-{cell_number}", "exec"),
                    namespace,
                )
        except Exception as error:
            raise RuntimeError(
                f"{tutorial.en.path}: execution failed in Python cell {cell_number}"
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

    if figure_number != len(tutorial.en.figure_alts):
        raise ValueError(
            f"{tutorial.en.path}: generated {figure_number} figures but front matter "
            f"defines {len(tutorial.en.figure_alts)} figure_alts"
        )
    return {
        "schema": 2,
        "source": str(tutorial.en.path.relative_to(MKDOCS_ROOT)).replace("\\", "/"),
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
                print(f"Executing {tutorial.en.path.relative_to(REPOSITORY_ROOT)}")
                manifests[tutorial.slug] = _execute_tutorial(tutorial, temporary_assets)

            expected = {
                figure["name"]
                for manifest in manifests.values()
                for cell in manifest["cells"].values()
                for figure in cell["figures"]
            }
            for asset_root in ASSET_ROOTS.values():
                asset_root.mkdir(parents=True, exist_ok=True)
                for stale in asset_root.glob("*.png"):
                    if stale.name not in expected:
                        stale.unlink()
                for name in sorted(expected):
                    shutil.copyfile(temporary_assets / name, asset_root / name)

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
    if payload.get("schema") != 2 or payload.get("code_sha256") != _code_sha256(tutorial):
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
            if name != f"{tutorial.slug}-{figure_count:02d}.png" or not isinstance(digest, str):
                raise ValueError(f"{result_path}: invalid figure {figure_count}")
            for asset_root in ASSET_ROOTS.values():
                asset = asset_root / name
                if not asset.is_file() or _sha256(asset) != digest:
                    raise ValueError(f"{result_path}: missing or modified {asset}")
        results[number] = raw
    if figure_count != len(tutorial.en.figure_alts):
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
    locale: str,
    first_figure: int,
) -> tuple[str, int]:
    lines = [f'!!! example "{"Runtime result" if locale == "en" else "运行结果"}"', ""]
    stdout = result["stdout"]
    if stdout:
        lines.append("    ```text")
        lines.extend(f"    {line}" if line else "" for line in stdout.splitlines())
        lines.extend(("    ```", ""))
    figure_number = first_figure
    source = tutorial.en if locale == "en" else tutorial.zh
    for figure in result["figures"]:
        name = figure["name"]
        alt = source.figure_alts[figure_number - 1]
        lines.extend((f"    ![{alt}](../assets/generated/tutorials/{name})", ""))
        figure_number += 1
    return "\n".join(lines).rstrip(), figure_number


def _render_body(tutorial: Tutorial, results: dict[int, dict[str, object]], locale: str) -> str:
    figure_number = 1
    cell_number = 0

    def code_block(code: str) -> str:
        nonlocal cell_number, figure_number
        cell_number += 1
        label = "Python cell" if locale == "en" else "Python 单元"
        rendered = f'```python title="{label} {cell_number}"\n{code}\n```'
        result_block, figure_number = _result_block(
            tutorial, cell_number, results[cell_number], locale, figure_number
        )
        if result_block.endswith('"') and not results[cell_number]["stdout"] and not results[cell_number]["figures"]:
            return rendered
        return rendered + "\n\n" + result_block

    if locale == "en":
        return PYTHON_CELL.sub(lambda match: code_block(match.group("code").rstrip()), tutorial.en.body)
    codes = iter(tutorial.code_cells)
    return CODE_PLACEHOLDER.sub(lambda _match: code_block(next(codes)), tutorial.zh.body)


def _category_label(category: str, locale: str) -> tuple[str, str]:
    if category in GALLERY["categories"]:
        content = GALLERY["categories"][category][locale]
        return content["title"], content["description"]
    title = category.rsplit("/", maxsplit=1)[-1].replace("-", " ").title()
    description = (
        f"Tutorials collected from the {title} source folder."
        if locale == "en"
        else f"从 {title} 源文件夹自动收集的教程。"
    )
    return title, description


def _insert_page_header(tutorial: Tutorial, locale: str, body: str) -> str:
    source = tutorial.en if locale == "en" else tutorial.zh
    category_title, _ = _category_label(tutorial.category, locale)
    download = f"../downloads/tutorials/{tutorial.slug}.py"
    labels = (
        ("Track", "Executable source", "Download", "Source-backed tutorial")
        if locale == "en"
        else ("学习路径", "可执行源码", "下载", "基于源码的教程")
    )
    preamble = "\n".join(
        (
            '<div class="grid cards" markdown>',
            "",
            f"-   :material-map-marker-path: **{labels[0]}**",
            "",
            f"    {category_title}",
            "",
            f"-   :material-language-python: **{labels[1]}**",
            "",
            f"    [{labels[2]} `{tutorial.slug}.py`]({download}){{ download }}",
            "",
            "</div>",
            "",
            f'!!! info "{labels[3]}"',
            "",
            (
                "    This page, its download, runtime output, and plots are assembled "
                "from the authored Markdown source."
                if locale == "en"
                else "    本页、下载文件、运行输出与图形均由英文 Markdown 源文件在构建时组装。"
            ),
        )
    )
    heading = re.search(r"(?m)^# .+$", body)
    if not heading:
        raise ValueError(f"{source.path}: expected one level-one heading")
    return body[: heading.end()] + "\n\n" + preamble + body[heading.end() :]


def _render_page(tutorial: Tutorial, results: dict[int, dict[str, object]], locale: str) -> str:
    source = tutorial.en if locale == "en" else tutorial.zh
    body = _insert_page_header(tutorial, locale, _render_body(tutorial, results, locale))
    return "\n".join(
        (
            "---",
            f"title: {json.dumps(source.title, ensure_ascii=False)}",
            f"description: {json.dumps(source.description, ensure_ascii=False)}",
            "---",
            "<!-- Generated from docs/mkdocs/tutorial-sources; do not edit here. -->",
            "",
            body,
        )
    )


def _render_download(tutorial: Tutorial) -> str:
    lines = [
        f'"""{tutorial.en.title}\n\n{tutorial.en.description}\n"""',
        "",
    ]
    for code in tutorial.code_cells:
        lines.extend(("# %%", code, ""))
    return "\n".join(lines)


def _render_index(tutorials: tuple[Tutorial, ...], locale: str) -> str:
    content = GALLERY["index"][locale]
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
        title, description = _category_label(category, locale)
        lines.extend((f"## {title}", "", description, "", '<div class="grid cards" markdown>', ""))
        for tutorial in (item for item in tutorials if item.category == category):
            source = tutorial.en if locale == "en" else tutorial.zh
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
    expected_downloads = {f"{tutorial.slug}.py" for tutorial in tutorials}
    for root in (*PAGE_ROOTS.values(), *DOWNLOAD_ROOTS.values()):
        root.mkdir(parents=True, exist_ok=True)
    for locale, page_root in PAGE_ROOTS.items():
        _write_text(page_root / "index.md", _render_index(tutorials, locale))
        for stale in page_root.glob("*.md"):
            if stale.name not in expected_pages:
                stale.unlink()
    for tutorial in tutorials:
        results = _load_results(tutorial)
        for locale in ("en", "zh"):
            _write_text(PAGE_ROOTS[locale] / f"{tutorial.slug}.md", _render_page(tutorial, results, locale))
            _write_text(DOWNLOAD_ROOTS[locale] / f"{tutorial.slug}.py", _render_download(tutorial))
    for root in DOWNLOAD_ROOTS.values():
        for stale in root.glob("*.py"):
            if stale.name not in expected_downloads:
                stale.unlink()
    print(f"Assembled {len(tutorials)} bilingual tutorials from category folders.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="force tutorial execution")
    execution.add_argument(
        "--execute-if-needed",
        action="store_true",
        help="execute when the source-hashed build cache is absent or stale",
    )
    args = parser.parse_args()
    tutorials = discover_tutorials()
    if args.execute or (args.execute_if_needed and not cache_is_current(tutorials)):
        capture_all(tutorials)
    assemble_all(tutorials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
