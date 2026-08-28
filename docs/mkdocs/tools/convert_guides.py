"""Port the canonical Sphinx/MyST guide pages to Material-compatible Markdown.

The Sphinx documentation remains authoritative and untouched.  This importer is
deliberately narrow: it knows the MyST constructs used by FatQat's current guide
and renders the ten ``plot`` directives to committed PNG assets.  Re-running it
therefore gives reviewers a useful drift check without pretending that arbitrary
Sphinx documents can be converted losslessly.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPHINX_ROOT = REPO_ROOT / "docs" / "sphinx"
MKDOCS_ROOT = REPO_ROOT / "docs" / "mkdocs"
EN_ROOT = MKDOCS_ROOT / "en"
GUIDE_ASSET_ROOT = EN_ROOT / "assets" / "generated" / "guide"

GUIDE_PAGES = (
    "quickstart",
    "program",
    "execution-models",
    "simulation",
    "interpret-results",
    "ideal-and-noisy",
    "performance",
    "hardware-profile-simulation",
    "hamiltonian-emulation",
    "transmon-emulation",
    "neutral-atom-emulation",
    "interoperability",
    "troubleshooting",
)

ICON_MAP = {
    "play": "material-play-circle-outline",
    "workflow": "material-transit-connection-variant",
    "git-branch": "material-source-branch",
    "pulse": "material-sine-wave",
    "beaker": "material-flask-outline",
    "list-unordered": "material-format-list-bulleted",
}

REFERENCE_TARGETS = {
    "noise-backend-support": "../api/noise/backend-support.md#noise-backend-support",
    "noise-simulator-support": "../api/noise/backend-support.md#noise-simulator-support",
    "noise-emulator-support": "../api/noise/backend-support.md#noise-emulator-support",
}

TUTORIAL_TARGETS = {
    "../tutorials/plot_pxp_z2_revival": "../tutorials/pxp-z2-revival",
    "../tutorials/plot_atom2level_antiferromagnetic_chain": (
        "../tutorials/antiferromagnetic-chain"
    ),
    "../tutorials/plot_atom_array_ghz8": "../tutorials/atom-array-ghz8",
}


def _object_role(match: re.Match[str]) -> str:
    payload = match.group("payload").strip()
    if "<" in payload and payload.endswith(">"):
        label, target = payload.rsplit("<", 1)
        label = label.strip()
        target = target[:-1].strip()
    else:
        target = payload
        label = target.lstrip("~").rsplit(".", 1)[-1]
    target = target.lstrip("~")
    return f"[`{label}`][{target}]"


def _doc_role(match: re.Match[str]) -> str:
    payload = match.group("payload").strip()
    if "<" in payload and payload.endswith(">"):
        label, target = payload.rsplit("<", 1)
        label = label.strip()
        target = target[:-1].strip()
    else:
        target = payload
        label = target.rsplit("/", 1)[-1].replace("-", " ").title()
    target = TUTORIAL_TARGETS.get(target, target)
    if not re.search(r"\.[A-Za-z0-9]+$", target):
        target += ".md"
    return f"[{label}]({target})"


def convert_inline_markup(text: str) -> str:
    text = re.sub(
        r"\{py:(?:class|meth|func|attr|data|exc|mod)\}`(?P<payload>[^`]+)`",
        _object_role,
        text,
    )
    text = re.sub(r"\{doc\}`(?P<payload>[^`]+)`", _doc_role, text)
    text = re.sub(
        r"\{ref\}`(?:(?P<label>[^`<>]+)\s*<)?(?P<target>[\w-]+)>?`",
        lambda match: (
            f'[{(match.group("label") or match.group("target")).strip()}]'
            f'({REFERENCE_TARGETS.get(match.group("target"), "#" + match.group("target"))})'
        ),
        text,
    )
    text = re.sub(r"(?<=\()([^()\s]+)\.rst(?=[)#])", r"\1.md", text)
    return text


def _find_directive_end(lines: list[str], start: int, fence: str) -> int:
    for index in range(start + 1, len(lines)):
        if lines[index].strip() == fence:
            return index
    raise ValueError(f"Unclosed MyST directive beginning on line {start + 1}")


def _indent_block(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in text.splitlines())


def _convert_grid(body: str) -> str:
    lines = body.splitlines()
    cards: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(
            r"^(?P<fence>:{3,})\{grid-item-card\}(?:\s+(?P<title>.*))?$",
            lines[index].strip(),
        )
        if not match:
            index += 1
            continue
        end = _find_directive_end(lines, index, match.group("fence"))
        title = (match.group("title") or "").strip()
        title = re.sub(
            r"\{octicon\}`([^`]+)`",
            lambda item: f":{ICON_MAP.get(item.group(1), 'material-circle-small')}: ",
            title,
        )
        content_lines = lines[index + 1 : end]
        link = None
        cleaned: list[str] = []
        for line in content_lines:
            stripped = line.strip()
            if stripped.startswith(":link:"):
                link = stripped.split(":link:", 1)[1].strip()
                if not re.search(r"\.[A-Za-z0-9]+$", link):
                    link += ".md"
            elif stripped.startswith(":link-type:"):
                continue
            else:
                cleaned.append(line)
        heading = f"**{title}**"
        if link:
            heading = f"[{heading}]({link})"
        content = convert_colon_directives("\n".join(cleaned).strip())
        card = f"-   {heading}"
        if content:
            card += "\n\n" + _indent_block(content, 4)
        cards.append(card)
        index = end + 1
    return '<div class="grid cards" markdown>\n\n' + "\n\n".join(cards) + "\n\n</div>"


def _convert_tab_set(body: str) -> str:
    lines = body.splitlines()
    tabs: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(
            r"^(?P<fence>:{3,})\{tab-item\}\s+(?P<title>.+)$",
            lines[index].strip(),
        )
        if not match:
            index += 1
            continue
        end = _find_directive_end(lines, index, match.group("fence"))
        content = convert_colon_directives("\n".join(lines[index + 1 : end]).strip())
        tabs.append(f'=== "{match.group("title").strip()}"\n\n{_indent_block(content)}')
        index = end + 1
    return "\n\n".join(tabs)


def convert_colon_directives(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(
            r"^(?P<fence>:{3,})\{(?P<kind>[\w-]+)\}(?:\s+(?P<title>.*))?$",
            lines[index].strip(),
        )
        if not match:
            output.append(lines[index])
            index += 1
            continue
        fence = match.group("fence")
        end = _find_directive_end(lines, index, fence)
        kind = match.group("kind")
        title = (match.group("title") or "").strip()
        body = "\n".join(lines[index + 1 : end]).strip("\n")
        if kind == "grid":
            replacement = _convert_grid(body)
        elif kind == "tab-set":
            replacement = _convert_tab_set(body)
        elif kind in {"tip", "note", "warning", "important", "info"}:
            label = f' "{title}"' if title else ""
            replacement = (
                f"!!! {kind}{label}\n\n{_indent_block(convert_colon_directives(body))}"
            )
        elif kind == "dropdown":
            body_lines = body.splitlines()
            expanded = False
            while body_lines and body_lines[0].strip().startswith(":"):
                if body_lines[0].strip() == ":open:":
                    expanded = True
                body_lines.pop(0)
            body = "\n".join(body_lines).lstrip()
            marker = "???+" if expanded else "???"
            replacement = f'{marker} question "{title}"\n\n' + _indent_block(
                convert_colon_directives(body)
            )
        elif kind == "container":
            replacement = convert_colon_directives(body)
        else:
            replacement = convert_colon_directives(body)
        output.extend(replacement.splitlines())
        index = end + 1
    return "\n".join(output)


def _parse_plot(block: str, fallback_prefix: str) -> tuple[str, str, str]:
    lines = block.splitlines()
    if not lines or lines[0].strip() != ".. plot::":
        raise ValueError("eval-rst block is not a plot directive")
    alt = "Generated FatQat guide figure"
    prefix = fallback_prefix
    index = 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            break
        if stripped.startswith(":alt:"):
            alt = stripped.split(":alt:", 1)[1].strip()
        elif stripped.startswith(":filename-prefix:"):
            prefix = stripped.split(":filename-prefix:", 1)[1].strip()
        index += 1
    code = textwrap.dedent("\n".join(lines[index:])).strip()
    return alt, prefix, code


def _render_plot(prefix: str, code: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GUIDE_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    namespace = {"__name__": "__main__", "__file__": str(SPHINX_ROOT)}
    exec(compile(code, f"<guide-plot:{prefix}>", "exec"), namespace)  # noqa: S102
    figure_numbers = plt.get_fignums()
    if not figure_numbers:
        raise RuntimeError(f"Plot {prefix!r} did not create a Matplotlib figure")
    written: list[Path] = []
    for number, figure_number in enumerate(figure_numbers, start=1):
        suffix = "" if len(figure_numbers) == 1 else f"-{number}"
        destination = GUIDE_ASSET_ROOT / f"{prefix}{suffix}.png"
        plt.figure(figure_number).savefig(destination, dpi=150, bbox_inches="tight")
        written.append(destination)
    plt.close("all")
    return written


def convert_plot_blocks(text: str, page_name: str, render: bool) -> str:
    pattern = re.compile(r"```\{eval-rst\}\n(?P<body>.*?)\n```", re.DOTALL)
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        body = match.group("body")
        if ".. plot::" not in body:
            return ""
        alt, prefix, code = _parse_plot(body, f"{page_name}-{counter}")
        image_paths: list[Path]
        if render:
            image_paths = _render_plot(prefix, code)
        else:
            existing = sorted(GUIDE_ASSET_ROOT.glob(f"{prefix}*.png"))
            image_paths = existing
        images = []
        for path in image_paths:
            relative = path.relative_to(EN_ROOT).as_posix()
            images.append(f"![{alt}](../{relative})")
        source = textwrap.indent(code, "    ")
        details = (
            '??? example "Reproduce this figure"\n\n'
            "    ```python\n"
            f"{source}\n"
            "    ```"
        )
        return "\n\n".join(images + [details])

    return pattern.sub(replace, text)


def strip_toctrees(text: str) -> str:
    return re.sub(r"\n?```\{toctree\}.*?\n```\n?", "\n", text, flags=re.DOTALL)


def convert_page(text: str, page_name: str, render: bool) -> str:
    text = convert_plot_blocks(text, page_name, render)
    text = strip_toctrees(text)
    text = re.sub(r"```\{doctest\}", "```pycon", text)
    text = convert_colon_directives(text)
    text = convert_inline_markup(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def home_page(source: str) -> str:
    intro = source.split("::::{grid}", 1)[0].rstrip()
    cards = """
<div class="grid cards" markdown>

-   :material-play-circle-outline: **[Run your first Program](guide/quickstart.md)**

    Build and draw the Bell circuit, run it, and turn its counts into a figure.

-   :material-transit-connection-variant: **[Learn the Program](guide/program.md)**

    Add registers, measurements, conditions, parameters, qudits, and mixed local
    dimensions without changing authoring models.

-   :material-source-branch: **[Choose the modeling level](guide/execution-models.md)**

    Compare general simulation, a hardware profile, and Hamiltonian-level
    emulation using one Program.

-   :material-sine-wave: **[Study hardware behavior](guide/hardware-profile-simulation.md)**

    Work with native gates, layouts, connectivity, pulse controls, leakage, and
    physical models.

-   :material-flask-outline: **[Work through tutorials](tutorials/index.md)**

    Continue into complete algorithm and physics case studies with downloadable
    sources.

-   :material-format-list-bulleted: **[Look up the API](api/index.md)**

    Find exact signatures, supported operations, shapes, units, and validation
    rules.

</div>
"""
    return convert_inline_markup(intro) + "\n\n" + cards.strip() + "\n"


def guide_index(source: str, render: bool) -> str:
    # Render the three original illustrations before replacing the Sphinx grid.
    convert_plot_blocks(source, "guide-index", render)
    tail_marker = "## Begin with a working program"
    tail = source.split(tail_marker, 1)[1]
    return f"""# User guide

Choose the level of detail that answers your question. Every path starts from
the same backend-independent [`Program`][fatqat.Program], so moving from an
algorithm study to a hardware or physics study does not require a second
authoring model.

<div class="grid cards" markdown>

-   :material-chart-bell-curve-cumulative: **Explore the algorithm**

    Start with ideal circuit behavior, inspect states and measurements, then add
    controlled noise and measure performance.

    [Start with simulation :material-arrow-right:](simulation.md)

-   :material-chip: **Test hardware constraints**

    Add topology, native operations, placement, occupancy, movement, and
    reference noise without changing the logical workload.

    [Open hardware-profile simulation :material-arrow-right:](hardware-profile-simulation.md)

-   :material-atom: **Follow the physics**

    Resolve calibrated gates and direct pulse controls into continuous dynamics
    for transmons and neutral atoms.

    [Open Hamiltonian emulation :material-arrow-right:](hamiltonian-emulation.md)

</div>

<div class="grid" markdown>

![Five-qubit variational ansatz](../assets/generated/guide/guide-path-algorithm.png)

![Hardware topology with supported and unsupported couplings](../assets/generated/guide/guide-path-hardware.png)

![Driven-atom spectroscopy heatmap](../assets/generated/guide/guide-path-physics.png)

</div>

## One Program, three levels

| Question | Execution target | Typical answer |
| --- | --- | --- |
| What does the algorithm do? | General simulator | Counts, states, expectations, or a unitary |
| Does it fit this device profile? | Hardware-profile simulator | Native-operation, layout, and noise behavior |
| What dynamics produce it? | Hamiltonian emulator | Time evolution, leakage, occupancy, and pulse effects |

{tail_marker}
{tail}
"""


def write_guides(render: bool) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    EN_ROOT.mkdir(parents=True, exist_ok=True)
    (EN_ROOT / "guide").mkdir(parents=True, exist_ok=True)

    source_home = (SPHINX_ROOT / "index.md").read_text(encoding="utf-8")
    (EN_ROOT / "index.md").write_text(home_page(source_home), encoding="utf-8")

    source_index = (SPHINX_ROOT / "guide" / "index.md").read_text(encoding="utf-8")
    converted_index = convert_page(
        guide_index(source_index, render), "guide-index", False
    )
    (EN_ROOT / "guide" / "index.md").write_text(converted_index, encoding="utf-8")

    for page_name in GUIDE_PAGES:
        source_path = SPHINX_ROOT / "guide" / f"{page_name}.md"
        destination = EN_ROOT / "guide" / f"{page_name}.md"
        converted = convert_page(
            source_path.read_text(encoding="utf-8"), page_name, render
        )
        destination.write_text(converted, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="execute the ten trusted guide plot blocks and refresh PNG assets",
    )
    arguments = parser.parse_args()
    write_guides(arguments.render)


if __name__ == "__main__":
    main()
