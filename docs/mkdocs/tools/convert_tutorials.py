"""Convert and optionally execute Sphinx-Gallery tutorials for MkDocs.

The top-level ``tutorials/*.py`` files remain the canonical executable sources.
This script performs a deliberately small, deterministic conversion tailored to
their current Sphinx-Gallery notebook structure:

* the module docstring and leading comment blocks become Markdown narrative;
* Python blocks become fenced notebook-style cells;
* common reStructuredText math, links, roles, and code blocks are translated;
* Sphinx-Gallery validation-only spans are hidden from the rendered page;
* checked-in runtime snapshots restore the figures and stdout shown by the
  Sphinx-Gallery build; and
* the original Python file is copied byte-for-byte into the download tree.

Run this file from any working directory. It writes only below
``docs/mkdocs``. The normal mode is deterministic and does not execute the
tutorials. Pass ``--execute`` in the full documentation environment to refresh
the source-hashed runtime snapshots and localized copies of their figures.
"""

from __future__ import annotations

import argparse
import ast
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
import textwrap


@dataclass(frozen=True)
class Tutorial:
    """Metadata that determines the public tutorial organization."""

    source_name: str
    slug: str
    title: str
    title_zh: str
    category: str
    summary: str
    summary_zh: str
    icon: str
    thumbnail_number: int = 1


@dataclass(frozen=True)
class CapturedFigure:
    """One source-hashed figure in a captured tutorial result."""

    name: str
    sha256: str


@dataclass(frozen=True)
class CellResult:
    """Captured public output belonging to one displayed Python cell."""

    stdout: str
    figures: tuple[CapturedFigure, ...]


TUTORIALS = (
    Tutorial(
        source_name="plot_bell_state.py",
        slug="bell-state",
        title="Prepare and measure a Bell state",
        title_zh="制备并测量贝尔态",
        category="Foundations",
        summary=(
            "Follow a two-qubit Bell state from exact amplitudes to seeded "
            "measurement counts and a comparison with the ideal distribution."
        ),
        summary_zh=(
            "从精确振幅出发，得到固定随机种子下的测量计数，并将其与理想分布比较，"
            "完整追踪一个两比特贝尔态。"
        ),
        icon="material-set-split",
    ),
    Tutorial(
        source_name="plot_vqe_h2.py",
        slug="vqe-h2",
        title="Find the ground-state energy of H₂ with VQE",
        title_zh="使用 VQE 求解 H₂ 的基态能量",
        category="Algorithms",
        summary=(
            "Run exact, finite-shot, and noisy VQE loops for molecular hydrogen "
            "and make the variational bound and sampling uncertainty explicit."
        ),
        summary_zh=(
            "对氢分子运行精确、有限采样和含噪声的 VQE 循环，明确展示变分上界与"
            "采样不确定性。"
        ),
        icon="material-chart-bell-curve-cumulative",
    ),
    Tutorial(
        source_name="plot_qnn_digits.py",
        slug="qnn-digits",
        title="Recognize handwritten digits with a quantum neural network",
        title_zh="使用量子神经网络识别手写数字",
        category="Algorithms",
        summary=(
            "Train a data-reuploading circuit to distinguish handwritten 3s and "
            "6s while evaluating a whole parameter batch with one sweep."
        ),
        summary_zh=(
            "训练一个数据重上传电路来区分手写数字 3 和 6，同时通过一次扫描评估"
            "整批参数。"
        ),
        icon="material-brain",
    ),
    Tutorial(
        source_name="plot_atom_array_ghz8.py",
        slug="atom-array-ghz8",
        title="Entangle eight atoms into a GHZ state",
        title_zh="将八个原子纠缠为 GHZ 态",
        category="Neutral-atom physics",
        summary=(
            "Use dynamic Pair and Unpair operations to build an eight-atom GHZ "
            "state, then test both its correlations and coherent phase."
        ),
        summary_zh=(
            "利用动态 `Pair` 和 `Unpair` 操作构建八原子 GHZ 态，然后同时检验其"
            "关联与相干相位。"
        ),
        icon="material-atom",
    ),
    Tutorial(
        source_name="plot_atom2level_antiferromagnetic_chain.py",
        slug="antiferromagnetic-chain",
        title="Build antiferromagnetic correlations in a Rydberg chain",
        title_zh="在里德伯原子链中建立反铁磁关联",
        category="Neutral-atom physics",
        summary=(
            "Design a three-stage Rydberg pulse from physical units and watch "
            "short-range antiferromagnetic order emerge in a ten-site chain."
        ),
        summary_zh=(
            "从实际物理单位出发设计三阶段里德伯脉冲，观察短程反铁磁序如何在十个"
            "格点的原子链中出现。"
        ),
        icon="material-sine-wave",
    ),
    Tutorial(
        source_name="plot_pxp_z2_revival.py",
        slug="pxp-z2-revival",
        title="Revivals and entanglement growth in an open PXP chain",
        title_zh="开放 PXP 链中的复苏与纠缠增长",
        category="Neutral-atom physics",
        summary=(
            "Trotterize the constrained PXP Hamiltonian and compare many-body "
            "revivals and half-chain entropy with an independent exact solve."
        ),
        summary_zh=(
            "对受约束的 PXP 哈密顿量进行 Trotter 分解，并将多体复苏与半链纠缠熵"
            "同独立的精确求解结果比较。"
        ),
        icon="material-waveform",
    ),
)

CATEGORY_CONTENT = {
    "Foundations": {
        "en": (
            "Foundations",
            "Start with a compact circuit whose exact state, samples, and visual "
            "interpretation can all be checked by hand.",
        ),
        "zh": (
            "基础",
            "从一个紧凑的量子电路入手：其精确态、采样结果与可视化解读都可以手工验证。",
        ),
    },
    "Algorithms": {
        "en": (
            "Algorithms",
            "Build parameterized programs once, then use optimizers, sweeps, and "
            "estimators to answer chemistry and machine-learning questions.",
        ),
        "zh": (
            "算法",
            "一次构建参数化程序，再通过优化器、扫描与估计器回答量子化学和机器学习问题。",
        ),
    },
    "Neutral-atom physics": {
        "en": (
            "Neutral-atom physics",
            "Move from programmable connectivity to continuous-time Rydberg dynamics "
            "and constrained many-body evolution.",
        ),
        "zh": (
            "中性原子物理",
            "从可编程连接逐步走向连续时间里德伯动力学与受约束的多体演化。",
        ),
    },
}

INDEX_CONTENT = {
    "en": {
        "title": "Tutorials",
        "description": (
            "Executable fatqat case studies, from first circuits to many-body dynamics."
        ),
        "introduction": (
            "Go beyond the task-focused user guide with complete, deterministic case",
            "studies. Each page alternates explanation and executable Python cells, and",
            "includes the original source file for local exploration.",
        ),
        "tip_title": "Choose a path",
        "tip": (
            "Begin with the Bell state if you are new to fatqat. The algorithms",
            "reuse the same parameter and execution model, while the neutral-atom",
            "track moves progressively closer to many-body hardware physics.",
        ),
        "open_tutorial": "Open tutorial",
    },
    "zh": {
        "title": "教程",
        "description": "从入门量子电路到多体动力学的可执行 fatqat 案例研究。",
        "introduction": (
            "在面向具体任务的用户指南之外，这些完整且可复现的案例将带你进一步理解 fatqat。",
            "每个页面都交替展示原理说明和可执行的 Python 单元，并附有原始源文件，便于在本地继续探索。",
        ),
        "tip_title": "选择学习路径",
        "tip": (
            "如果刚接触 fatqat，建议从贝尔态开始。算法篇会复用相同的参数与执行模型；",
            "中性原子篇则循序渐进，逐步贴近多体硬件的真实物理。",
        ),
        "open_tutorial": "打开教程",
    },
}

FIGURE_ALTS = {
    "bell-state": (
        (
            "Seeded Bell-state measurement frequencies compared with the ideal distribution",
            "固定随机种子的贝尔态测量频率与理想分布对比",
        ),
    ),
    "vqe-h2": (
        ("Exact VQE convergence trace", "精确 VQE 收敛曲线"),
        (
            "Finite-shot VQE traces with statistical uncertainty",
            "带统计不确定度的有限采样 VQE 曲线",
        ),
        (
            "Noiseless and depolarizing-noise VQE energy traces",
            "无噪声与退极化噪声下的 VQE 能量曲线",
        ),
    ),
    "qnn-digits": (
        ("Average-pooled handwritten digit inputs", "平均池化后的手写数字输入"),
        ("COBYLA training-loss trace", "COBYLA 训练损失曲线"),
        (
            "Held-out handwritten digit predictions after training",
            "训练后对留出手写数字样本的预测",
        ),
    ),
    "atom-array-ghz8": (
        (
            "Eight-atom GHZ measurement frequencies",
            "八原子 GHZ 态的测量频率",
        ),
    ),
    "antiferromagnetic-chain": (
        ("Three-stage Rydberg pulse schedule", "三阶段里德伯脉冲时序"),
        (
            "Rydberg density, connected correlations, and ordering summary",
            "里德伯密度、连通关联与有序性汇总",
        ),
    ),
    "pxp-z2-revival": (
        (
            "PXP revival fidelities, entanglement entropy, and site occupations",
            "PXP 复苏保真度、纠缠熵与位点占据",
        ),
    ),
}

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
MKDOCS_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "tutorials"
PAGE_ROOT = MKDOCS_ROOT / "en" / "tutorials"
DOWNLOAD_ROOT = MKDOCS_ROOT / "en" / "downloads" / "tutorials"
ZH_PAGE_ROOT = MKDOCS_ROOT / "zh" / "tutorials"
EN_ASSET_ROOT = MKDOCS_ROOT / "en" / "assets" / "generated" / "tutorials"
ZH_ASSET_ROOT = MKDOCS_ROOT / "zh" / "assets" / "generated" / "tutorials"
RESULT_ROOT = MKDOCS_ROOT / "tutorial-results"

CELL_SEPARATOR = re.compile(r"^#\s*%%\s*$")
HIDDEN_START = "# sphinx_gallery_start_ignore"
HIDDEN_END = "# sphinx_gallery_end_ignore"
RST_HEADING = re.compile(r"^[=\-~^\"]{3,}\s*$")
RST_ROLE = re.compile(
    r":(?:py:)?(?:attr|class|const|data|exc|func|meth|mod|obj):`([^`]+)`"
)
RST_DOC_ROLE = re.compile(r":doc:`([^`]+)`")
RST_REF_ROLE = re.compile(r":ref:`([^`]+)`")
RST_INLINE_MATH = re.compile(r":math:`([^`]+)`", re.DOTALL)
RST_EXTERNAL_LINK = re.compile(r"`([^`<]+?)\s*<([^>]+)>`_", re.DOTALL)

DOC_TITLES = {
    "guide/ideal-and-noisy": "ideal and noisy execution",
    "guide/interpret-results": "interpreting results",
    "guide/program": "the Program guide",
    "guide/simulation": "the simulation guide",
}

RESULT_BLOCK = re.compile(
    r"\n*<!-- tutorial-result-start:cell-\d+ -->.*?"
    r"<!-- tutorial-result-end:cell-\d+ -->",
    re.DOTALL,
)
ZH_CODE_CELL = re.compile(
    r'(?ms)(^```python title="Python 单元 (?P<number>\d+)"\n.*?^```)(?=\n|$)'
)


def _write_text(path: Path, content: str) -> None:
    """Write normalized UTF-8 text with a single trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content.rstrip() + "\n")


def _module_docstring_and_body(source: Path) -> tuple[str, list[str]]:
    """Return the module docstring and exact source lines following it."""

    source_text = source.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source))
    if not tree.body or not isinstance(tree.body[0], ast.Expr):
        raise ValueError(f"{source}: expected a module docstring")

    docstring_node = tree.body[0]
    if not isinstance(docstring_node.value, ast.Constant) or not isinstance(
        docstring_node.value.value, str
    ):
        raise ValueError(f"{source}: expected a module docstring")
    if docstring_node.end_lineno is None:
        raise ValueError(f"{source}: parser did not report the docstring end")

    docstring = ast.get_docstring(tree, clean=False)
    if docstring is None:
        raise ValueError(f"{source}: expected a module docstring")
    body_lines = source_text.splitlines()[docstring_node.end_lineno :]
    return docstring, body_lines


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_document_title(text: str, expected_title: str) -> str:
    """Remove the leading RST title, checking it against public metadata."""

    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) < 2 or not RST_HEADING.fullmatch(lines[1].strip()):
        raise ValueError(f"expected an RST title for {expected_title!r}")
    title = lines[0].strip()
    if title != expected_title:
        raise ValueError(
            f"tutorial metadata title {expected_title!r} does not match {title!r}"
        )
    return "\n".join(lines[2:]).lstrip("\n")


def _strip_hidden_validation(lines: list[str], source: Path) -> list[str]:
    """Remove Sphinx-Gallery validation spans from displayed notebook cells."""

    visible: list[str] = []
    hidden = False
    for line in lines:
        marker = line.strip()
        if marker == HIDDEN_START:
            if hidden:
                raise ValueError(f"{source}: nested {HIDDEN_START}")
            hidden = True
            continue
        if marker == HIDDEN_END:
            if not hidden:
                raise ValueError(f"{source}: unmatched {HIDDEN_END}")
            hidden = False
            continue
        if not hidden:
            visible.append(line)
    if hidden:
        raise ValueError(f"{source}: missing {HIDDEN_END}")
    return visible


def _split_cells(lines: list[str]) -> list[list[str]]:
    """Split source lines at Sphinx-Gallery notebook separators."""

    cells: list[list[str]] = []
    current: list[str] = []
    seen_separator = False
    for line in lines:
        if CELL_SEPARATOR.fullmatch(line.strip()):
            if seen_separator and any(part.strip() for part in current):
                cells.append(current)
            current = []
            seen_separator = True
            continue
        if seen_separator:
            current.append(line)
        elif line.strip():
            raise ValueError("unexpected executable content before the first # %%")
    if seen_separator and any(part.strip() for part in current):
        cells.append(current)
    return cells


def _split_narrative_and_code(cell: list[str]) -> tuple[str, str]:
    """Separate a cell's leading prose comments from its executable code."""

    start = 0
    while start < len(cell) and not cell[start].strip():
        start += 1

    narrative_lines: list[str] = []
    cursor = start
    while cursor < len(cell):
        line = cell[cursor]
        stripped = line.lstrip()
        if not line.strip():
            narrative_lines.append("")
        elif stripped == "#":
            narrative_lines.append("")
        elif stripped.startswith("# "):
            narrative_lines.append(stripped[2:])
        elif stripped.startswith("#"):
            narrative_lines.append(stripped[1:])
        else:
            break
        cursor += 1

    code_lines = cell[cursor:]
    while code_lines and not code_lines[0].strip():
        code_lines.pop(0)
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    narrative = "\n".join(narrative_lines).strip()
    code = "\n".join(code_lines).rstrip()
    return narrative, code


def _collect_indented_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect the indented body after an RST directive."""

    cursor = start
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    block_start = cursor
    while cursor < len(lines):
        line = lines[cursor]
        if line.strip() and len(line) == len(line.lstrip()):
            break
        cursor += 1
    block = lines[block_start:cursor]
    while block and not block[-1].strip():
        block.pop()
    return textwrap.dedent("\n".join(block)).splitlines(), cursor


def _convert_rst_blocks(text: str) -> str:
    """Convert the block-level RST constructs used by the tutorials."""

    source_lines = text.splitlines()
    output: list[str] = []
    cursor = 0
    while cursor < len(source_lines):
        line = source_lines[cursor]
        stripped = line.strip()

        if cursor + 1 < len(source_lines) and RST_HEADING.fullmatch(
            source_lines[cursor + 1].strip()
        ):
            marker = source_lines[cursor + 1].strip()[0]
            level = 2 if marker in "=-" else 3
            output.extend((f"{'#' * level} {stripped}", ""))
            cursor += 2
            continue

        if stripped == ".. math::":
            block, cursor = _collect_indented_block(source_lines, cursor + 1)
            output.extend(("$$", *block, "$$", ""))
            continue

        code_match = re.fullmatch(r"\.\. code-block::\s*([\w+-]*)", stripped)
        if code_match:
            language = code_match.group(1) or "text"
            block, cursor = _collect_indented_block(source_lines, cursor + 1)
            output.extend((f"```{language}", *block, "```", ""))
            continue

        admonition_match = re.fullmatch(
            r"\.\. (note|tip|warning|important)::\s*(.*)", stripped
        )
        if admonition_match:
            kind, title = admonition_match.groups()
            block, cursor = _collect_indented_block(source_lines, cursor + 1)
            title_suffix = f' "{title}"' if title else ""
            output.append(f"!!! {kind}{title_suffix}")
            output.extend(f"    {part}" if part else "" for part in block)
            output.append("")
            continue

        output.append(line)
        cursor += 1

    return "\n".join(output)


def _normalize_inline_whitespace(value: str) -> str:
    """Join a wrapped inline RST value without changing its meaning."""

    return re.sub(r"\s+", " ", value).strip()


def _convert_external_link(match: re.Match[str]) -> str:
    label, target = match.groups()
    return f"[{_normalize_inline_whitespace(label)}]({target.strip()})"


def _convert_object_role(match: re.Match[str]) -> str:
    value = _normalize_inline_whitespace(match.group(1))
    explicit = re.fullmatch(r"(.+?)\s*<([^>]+)>", value)
    if explicit:
        label, target = explicit.groups()
    else:
        target = value
        shortened = target.startswith("~")
        target = target.lstrip("~")
        label = target.rsplit(".", maxsplit=1)[-1] if shortened else target
    return f"[`{label}`][{target}]"


def _convert_doc_role(match: re.Match[str]) -> str:
    value = _normalize_inline_whitespace(match.group(1))
    explicit = re.fullmatch(r"(.+?)\s*<([^>]+)>", value)
    if explicit:
        label, target = explicit.groups()
    else:
        target = value
        normalized = target.lstrip("/")
        label = DOC_TITLES.get(normalized, normalized.rsplit("/", maxsplit=1)[-1])

    normalized_target = target.lstrip("/")
    if normalized_target.startswith("guide/"):
        href = f"../{normalized_target}.md"
    elif normalized_target.startswith("api/"):
        href = f"../{normalized_target}.md"
    else:
        href = f"{normalized_target}.md"
    return f"[{label}]({href})"


def _convert_ref_role(match: re.Match[str]) -> str:
    value = _normalize_inline_whitespace(match.group(1))
    explicit = re.fullmatch(r"(.+?)\s*<([^>]+)>", value)
    label = explicit.group(1) if explicit else value
    return f"`{label}`"


def _convert_rst(text: str) -> str:
    """Translate the tutorial subset of RST into Material-friendly Markdown."""

    converted = RST_EXTERNAL_LINK.sub(_convert_external_link, text)
    converted = re.sub(r"``([^`]+)``", r"`\1`", converted)
    converted = _convert_rst_blocks(converted)
    converted = RST_INLINE_MATH.sub(
        lambda match: f"${_normalize_inline_whitespace(match.group(1))}$",
        converted,
    )
    converted = RST_ROLE.sub(_convert_object_role, converted)
    converted = RST_DOC_ROLE.sub(_convert_doc_role, converted)
    converted = RST_REF_ROLE.sub(_convert_ref_role, converted)
    converted = converted.replace(
        "[`fatqat.operations`][fatqat.operations]",
        "[`fatqat.operations`](../api/operations.md)",
    )
    converted = converted.replace("Sphinx-Gallery", "the docs build")
    converted = converted.replace("generated notebook", "downloadable Python source")
    converted = converted.replace("downloadable notebook", "downloadable Python source")
    converted = re.sub(
        r"Matplotlib supplies the figure captured by\s+the docs build\.",
        "Matplotlib supplies the captured runtime figure.",
        converted,
    )
    converted = converted.replace(
        "The tutorial has three acts, all executed when this page is built:",
        "The tutorial source contains three executable acts:",
    )
    converted = re.sub(
        r"The documentation build also checks that expectation\. The validation-only\s+"
        r"lines are executed by the docs build but omitted from the public page and\s+"
        r"downloadable Python source\.",
        "The canonical source also checks that expectation with validation-only "
        "lines; this page omits those checks from displayed code.",
        converted,
    )
    converted = converted.replace(
        "public output stable across clean documentation builds",
        "printed output stable across repeated tutorial runs",
    )
    converted = converted.replace(
        "across documentation builds", "across repeated tutorial runs"
    )
    converted = converted.replace(
        "quick enough for an executable documentation build",
        "quick enough for a local tutorial run",
    )
    converted = re.sub(
        r"Because this page and its\s+downloadable Python source come from the same "
        r"executable source,",
        "Because the downloadable Python source is canonical,",
        converted,
    )
    converted = re.sub(
        r"This page\s+and its downloadable Python source come from the same "
        r"executable source",
        "The downloadable Python file is the canonical executable source",
        converted,
    )

    if re.search(r":\w+(?::\w+)?:`", converted):
        raise ValueError(f"unconverted RST role in narrative:\n{converted}")
    if re.search(r"(?m)^\s*\.\. \w", converted):
        raise ValueError(f"unconverted RST directive in narrative:\n{converted}")

    converted = re.sub(r"\n{3,}", "\n\n", converted)
    return converted.strip()


def _result_block(
    tutorial: Tutorial,
    cell_number: int,
    result: CellResult,
    *,
    locale: str,
) -> str:
    """Render captured stdout and figures immediately after their source cell."""

    if locale not in {"en", "zh"}:
        raise ValueError(f"unsupported result locale: {locale}")
    title = "Runtime result" if locale == "en" else "运行结果"
    lines = [
        f"<!-- tutorial-result-start:cell-{cell_number} -->",
        f'!!! example "{title}"',
        "",
    ]
    if result.stdout:
        lines.extend(("    ```text",))
        lines.extend(
            f"    {line}" if line else "" for line in result.stdout.splitlines()
        )
        lines.extend(("    ```", ""))

    alt_index = 0 if locale == "en" else 1
    figure_alts = FIGURE_ALTS[tutorial.slug]
    for figure in result.figures:
        figure_number = int(figure.name.rsplit("-", maxsplit=1)[1].removesuffix(".png"))
        alt = figure_alts[figure_number - 1][alt_index]
        lines.extend(
            (
                f"    ![{alt}](../assets/generated/tutorials/{figure.name})",
                "",
            )
        )

    lines.append(f"<!-- tutorial-result-end:cell-{cell_number} -->")
    return "\n".join(lines)


def _load_results(
    tutorial: Tutorial, source: Path
) -> tuple[int, dict[int, CellResult]]:
    """Load and validate the checked-in runtime snapshot for one tutorial."""

    result_path = RESULT_ROOT / f"{tutorial.slug}.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"missing {result_path}; refresh snapshots with "
            "convert_tutorials.py --execute"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError(f"{result_path}: unsupported result schema")
    if payload.get("source") != tutorial.source_name:
        raise ValueError(f"{result_path}: source name does not match tutorial metadata")
    source_digest = _sha256(source)
    if payload.get("source_sha256") != source_digest:
        raise ValueError(
            f"{result_path}: snapshot is stale for {tutorial.source_name}; "
            "refresh it with convert_tutorials.py --execute"
        )

    code_cells = payload.get("code_cells")
    if not isinstance(code_cells, int) or code_cells < 1:
        raise ValueError(f"{result_path}: invalid code_cells value")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, dict):
        raise ValueError(f"{result_path}: cells must be an object")

    results: dict[int, CellResult] = {}
    captured_figure_names: list[str] = []
    for raw_number, raw_result in raw_cells.items():
        try:
            cell_number = int(raw_number)
        except ValueError as error:
            raise ValueError(
                f"{result_path}: invalid cell number {raw_number!r}"
            ) from error
        if not 1 <= cell_number <= code_cells or str(cell_number) != raw_number:
            raise ValueError(f"{result_path}: invalid cell number {raw_number!r}")
        if not isinstance(raw_result, dict):
            raise ValueError(f"{result_path}: cell {cell_number} must be an object")
        stdout = raw_result.get("stdout", "")
        if not isinstance(stdout, str):
            raise ValueError(f"{result_path}: cell {cell_number} stdout must be text")

        figures: list[CapturedFigure] = []
        raw_figures = raw_result.get("figures", [])
        if not isinstance(raw_figures, list):
            raise ValueError(
                f"{result_path}: cell {cell_number} figures must be a list"
            )
        for raw_figure in raw_figures:
            if not isinstance(raw_figure, dict):
                raise ValueError(
                    f"{result_path}: cell {cell_number} has invalid figure metadata"
                )
            name = raw_figure.get("name")
            digest = raw_figure.get("sha256")
            if (
                not isinstance(name, str)
                or not re.fullmatch(rf"{re.escape(tutorial.slug)}-\d{{2}}\.png", name)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError(
                    f"{result_path}: cell {cell_number} has invalid figure metadata"
                )
            for asset_root in (EN_ASSET_ROOT, ZH_ASSET_ROOT):
                asset = asset_root / name
                if not asset.is_file() or _sha256(asset) != digest:
                    raise ValueError(
                        f"{result_path}: missing or modified captured figure {asset}"
                    )
            figures.append(CapturedFigure(name=name, sha256=digest))
            captured_figure_names.append(name)
        if stdout or figures:
            results[cell_number] = CellResult(stdout=stdout, figures=tuple(figures))

    expected_figure_names = [
        f"{tutorial.slug}-{number:02d}.png"
        for number in range(1, len(FIGURE_ALTS[tutorial.slug]) + 1)
    ]
    if captured_figure_names != expected_figure_names:
        raise ValueError(
            f"{result_path}: expected figures {expected_figure_names}, "
            f"found {captured_figure_names}"
        )
    return code_cells, results


def _render_page(
    tutorial: Tutorial,
    source: Path,
    code_cells: int,
    results: dict[int, CellResult],
) -> str:
    """Render one tutorial as alternating Markdown and Python notebook cells."""

    docstring, body_lines = _module_docstring_and_body(source)
    introduction_rst = _remove_document_title(docstring, tutorial.title)
    introduction = _convert_rst(introduction_rst)
    visible_lines = _strip_hidden_validation(body_lines, source)
    cells = _split_cells(visible_lines)
    if not cells:
        raise ValueError(f"{source}: no notebook cells found")

    rendered_cells: list[str] = []
    code_cell_number = 0
    for cell in cells:
        narrative_rst, code = _split_narrative_and_code(cell)
        if narrative_rst:
            rendered_cells.append(_convert_rst(narrative_rst))
        if code:
            code_cell_number += 1
            rendered_cells.append(
                f'```python title="Python cell {code_cell_number}"\n{code}\n```'
            )
            if result := results.get(code_cell_number):
                rendered_cells.append(
                    _result_block(tutorial, code_cell_number, result, locale="en")
                )

    if code_cell_number != code_cells:
        raise ValueError(
            f"{source}: snapshot records {code_cells} code cells, "
            f"converter found {code_cell_number}"
        )

    download_path = f"../downloads/tutorials/{tutorial.source_name}"
    header = "\n".join(
        (
            "---",
            f"title: {json.dumps(tutorial.title, ensure_ascii=False)}",
            f"description: {json.dumps(tutorial.summary, ensure_ascii=False)}",
            "---",
            "<!-- Generated by docs/mkdocs/tools/convert_tutorials.py. -->",
            "",
            f"# {tutorial.title}",
            "",
            '<div class="grid cards" markdown>',
            "",
            "-   :material-map-marker-path: **Track**",
            "",
            f"    {tutorial.category}",
            "",
            "-   :material-language-python: **Executable source**",
            "",
            f"    [Download `{tutorial.source_name}`]({download_path}){{ download }}",
            "",
            "</div>",
            "",
            introduction,
            "",
            '!!! info "Source-backed tutorial"',
            "",
            "    The narrative and executable cells come from the tracked tutorial",
            "    source. Validation-only Sphinx-Gallery spans are not displayed.",
            "    Runtime panels contain checked-in snapshots captured from that same",
            "    source. Run the download directly to reproduce its plots and stdout.",
            "",
        )
    )
    return header + "\n" + "\n\n".join(rendered_cells)


def _sync_chinese_results(
    tutorial: Tutorial,
    page: Path,
    code_cells: int,
    results: dict[int, CellResult],
) -> None:
    """Synchronize generated result panels while preserving translated prose."""

    if not page.is_file():
        raise FileNotFoundError(f"missing Chinese tutorial page: {page}")
    text = RESULT_BLOCK.sub("", page.read_text(encoding="utf-8"))
    found_cells: list[int] = []

    def inject(match: re.Match[str]) -> str:
        cell_number = int(match.group("number"))
        found_cells.append(cell_number)
        result = results.get(cell_number)
        if result is None:
            return match.group(1)
        return (
            match.group(1)
            + "\n\n"
            + _result_block(tutorial, cell_number, result, locale="zh")
        )

    text = ZH_CODE_CELL.sub(inject, text)
    expected_cells = list(range(1, code_cells + 1))
    if found_cells != expected_cells:
        raise ValueError(
            f"{page}: expected Chinese Python cells {expected_cells}, "
            f"found {found_cells}"
        )
    _write_text(page, text)


def _render_index(locale: str) -> str:
    """Render one localized visual, category-based tutorial landing page."""

    if locale not in INDEX_CONTENT:
        raise ValueError(f"unsupported tutorial-index locale: {locale}")
    content = INDEX_CONTENT[locale]
    title = content["title"]
    lines = [
        "---",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"description: {json.dumps(content['description'], ensure_ascii=False)}",
        "---",
        "<!-- Generated by docs/mkdocs/tools/convert_tutorials.py. -->",
        "",
        f"# {title}",
        "",
        *content["introduction"],
        "",
        f'!!! tip "{content["tip_title"]}"',
        "",
        *(f"    {line}" for line in content["tip"]),
        "",
    ]

    alt_index = 0 if locale == "en" else 1
    for category, localized_content in CATEGORY_CONTENT.items():
        category_title, introduction = localized_content[locale]
        lines.extend(
            (
                f"## {category_title}",
                "",
                introduction,
                "",
                '<div class="grid cards" markdown>',
                "",
            )
        )
        for tutorial in TUTORIALS:
            if tutorial.category != category:
                continue
            thumbnail_index = tutorial.thumbnail_number - 1
            figure_alts = FIGURE_ALTS.get(tutorial.slug)
            if not figure_alts:
                raise ValueError(
                    f"{tutorial.slug}: add localized FIGURE_ALTS before publishing "
                    "its tutorial card"
                )
            if thumbnail_index not in range(len(figure_alts)):
                raise ValueError(
                    f"{tutorial.slug}: thumbnail {tutorial.thumbnail_number} does not "
                    f"match {len(figure_alts)} captured figures"
                )
            thumbnail_name = f"{tutorial.slug}-{tutorial.thumbnail_number:02d}.png"
            thumbnail_alt = figure_alts[thumbnail_index][alt_index]
            tutorial_title = tutorial.title if locale == "en" else tutorial.title_zh
            tutorial_summary = (
                tutorial.summary if locale == "en" else tutorial.summary_zh
            )
            lines.extend(
                (
                    f"-   [![{thumbnail_alt}]"
                    f"(../assets/generated/tutorials/{thumbnail_name})"
                    f"{{ loading=lazy }}]({tutorial.slug}.md)",
                    "",
                    f"    :{tutorial.icon}:{{ .lg .middle }} **{tutorial_title}**",
                    "",
                    "    ---",
                    "",
                    f"    {tutorial_summary}",
                    "",
                    f"    [:material-arrow-right: {content['open_tutorial']}]"
                    f"({tutorial.slug}.md)",
                    "",
                )
            )
        lines.extend(("</div>", ""))

    return "\n".join(lines)


def _execute_tutorial(
    tutorial: Tutorial,
    source: Path,
    asset_root: Path,
) -> dict[str, object]:
    """Execute notebook cells and capture their stdout and Matplotlib figures."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "tutorial execution requires the project's full documentation "
            "environment"
        ) from error

    # The source calls ``show`` to give interactive runs their natural end
    # point. Capturing the still-open pyplot figures below is the headless
    # equivalent and avoids the Agg backend's non-interactive warning.
    plt.show = lambda *args, **kwargs: None

    _, body_lines = _module_docstring_and_body(source)
    cells = _split_cells(body_lines)
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__package__": None,
    }
    code_cell_number = 0
    figure_number = 0
    captured_cells: dict[str, dict[str, object]] = {}

    plt.close("all")
    for source_cell_number, cell in enumerate(cells, start=1):
        _, executable_code = _split_narrative_and_code(cell.copy())
        visible_cell = _strip_hidden_validation(cell.copy(), source)
        _, visible_code = _split_narrative_and_code(visible_cell)
        if visible_code:
            code_cell_number += 1
        if not executable_code:
            continue

        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(
                    compile(
                        executable_code + "\n",
                        f"{source}#cell-{source_cell_number}",
                        "exec",
                    ),
                    namespace,
                )
        except Exception as error:
            raise RuntimeError(
                f"{source}: execution failed in source cell {source_cell_number}"
            ) from error

        figures: list[dict[str, str]] = []
        for matplotlib_number in plt.get_fignums():
            figure_number += 1
            name = f"{tutorial.slug}-{figure_number:02d}.png"
            asset = asset_root / name
            asset.parent.mkdir(parents=True, exist_ok=True)
            figure = plt.figure(matplotlib_number)
            figure.savefig(
                asset,
                dpi=144,
                bbox_inches="tight",
                facecolor="white",
                metadata={"Software": "fatqat tutorial result capture"},
            )
            figures.append({"name": name, "sha256": _sha256(asset)})
        plt.close("all")

        output = stdout.getvalue().replace("\r\n", "\n").rstrip()
        if visible_code and (output or figures):
            captured_cells[str(code_cell_number)] = {
                "stdout": output,
                "figures": figures,
            }
        elif output or figures:
            raise ValueError(
                f"{source}: hidden-only cell {source_cell_number} produced public output"
            )

    expected_figures = len(FIGURE_ALTS[tutorial.slug])
    if figure_number != expected_figures:
        raise ValueError(
            f"{source}: captured {figure_number} figures, expected {expected_figures}; "
            "update FIGURE_ALTS after reviewing the new output"
        )
    return {
        "schema": 1,
        "source": tutorial.source_name,
        "source_sha256": _sha256(source),
        "code_cells": code_cell_number,
        "cells": captured_cells,
    }


def capture_all() -> None:
    """Execute every tutorial, then refresh snapshots after all runs succeed."""

    source_path = str(REPOSITORY_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    original_directory = Path.cwd()
    manifests: dict[str, dict[str, object]] = {}
    try:
        os.chdir(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory(prefix="fatqat-mkdocs-tutorials-") as temp:
            temporary_assets = Path(temp) / "assets"
            for tutorial in TUTORIALS:
                source = SOURCE_ROOT / tutorial.source_name
                print(f"Executing {source.relative_to(REPOSITORY_ROOT)}")
                manifests[tutorial.slug] = _execute_tutorial(
                    tutorial, source, temporary_assets
                )

            expected_assets = {
                raw_figure["name"]
                for manifest in manifests.values()
                for raw_cell in manifest["cells"].values()
                for raw_figure in raw_cell["figures"]
            }
            for asset_root in (EN_ASSET_ROOT, ZH_ASSET_ROOT):
                asset_root.mkdir(parents=True, exist_ok=True)
                for stale in asset_root.glob("*.png"):
                    if stale.name not in expected_assets:
                        stale.unlink()
                for name in sorted(expected_assets):
                    shutil.copyfile(temporary_assets / name, asset_root / name)

            RESULT_ROOT.mkdir(parents=True, exist_ok=True)
            for slug, manifest in manifests.items():
                _write_text(
                    RESULT_ROOT / f"{slug}.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                )
    finally:
        os.chdir(original_directory)


def convert_all() -> None:
    """Convert tutorials using checked-in results and copy exact source files."""

    unknown_categories = sorted(
        {tutorial.category for tutorial in TUTORIALS} - set(CATEGORY_CONTENT)
    )
    if unknown_categories:
        raise ValueError(
            "tutorial categories need localized index content: "
            + ", ".join(unknown_categories)
        )

    expected_sources = {tutorial.source_name for tutorial in TUTORIALS}
    actual_sources = {path.name for path in SOURCE_ROOT.glob("plot_*.py")}
    if actual_sources != expected_sources:
        missing = sorted(actual_sources - expected_sources)
        stale = sorted(expected_sources - actual_sources)
        raise ValueError(
            "tutorial inventory changed; update TUTORIALS "
            f"(unlisted={missing}, missing={stale})"
        )

    PAGE_ROOT.mkdir(parents=True, exist_ok=True)
    ZH_PAGE_ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    _write_text(PAGE_ROOT / "index.md", _render_index("en"))
    _write_text(ZH_PAGE_ROOT / "index.md", _render_index("zh"))

    for tutorial in TUTORIALS:
        source = SOURCE_ROOT / tutorial.source_name
        page = PAGE_ROOT / f"{tutorial.slug}.md"
        download = DOWNLOAD_ROOT / tutorial.source_name
        code_cells, results = _load_results(tutorial, source)
        _write_text(page, _render_page(tutorial, source, code_cells, results))
        _sync_chinese_results(
            tutorial,
            ZH_PAGE_ROOT / f"{tutorial.slug}.md",
            code_cells,
            results,
        )
        shutil.copyfile(source, download)
        if source.read_bytes() != download.read_bytes():
            raise RuntimeError(f"download copy differs from {source}")
        print(page.relative_to(REPOSITORY_ROOT))
        print(download.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute every tutorial and refresh source-hashed runtime snapshots",
    )
    arguments = parser.parse_args()
    if arguments.execute:
        capture_all()
    convert_all()
