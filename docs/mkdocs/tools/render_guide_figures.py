"""Render documentation figures from their tracked, executable sources."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import runpy
import shutil
import sys
import tempfile
import textwrap

SCRIPT_PATH = Path(__file__).resolve()
MKDOCS_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
GUIDE_ROOT = MKDOCS_ROOT / "en" / "guide"
CARD_SOURCE = MKDOCS_ROOT / "figure-sources" / "guide_cards.py"
FIGURE_SOURCE_ROOT = MKDOCS_ROOT / "figure-sources"
GUIDE_OUTPUT = MKDOCS_ROOT / "en" / "assets" / "generated" / "guide"
HOME_OUTPUT = MKDOCS_ROOT / "en" / "assets" / "generated" / "home"
DEFAULT_FIGURE_DPI = 144
GUIDE_SOURCES = (
    (
        FIGURE_SOURCE_ROOT / "atom_pairing_lifecycle.py",
        ("atom-pairing-lifecycle.svg",),
    ),
    (
        FIGURE_SOURCE_ROOT / "atom_loss_lifecycle.py",
        ("atom-loss-lifecycle.svg",),
    ),
)
HOME_SOURCES = (
    (
        FIGURE_SOURCE_ROOT / "home_grover_general.py",
        ("grover-circuit.png", "grover-general.png"),
    ),
    (
        FIGURE_SOURCE_ROOT / "home_grover_sc.py",
        ("grover-sc-profile.png",),
    ),
    (
        FIGURE_SOURCE_ROOT / "home_grover_transmon.py",
        ("grover-transmon.png",),
    ),
)
HOME_FIGURES = tuple(
    name for _source, expected_names in HOME_SOURCES for name in expected_names
)

IMAGE = re.compile(
    r"!\[[^]]*]\(\.\./assets/generated/guide/" r"(?P<name>[a-z0-9-]+\.(?:png|svg))\)"
)
REPRODUCTION = re.compile(
    r'(?ms)^\?\?\? example "Reproduce this figure"\s*\n\s*'
    r"^    ```python\s*\n(?P<code>.*?)^    ```"
)


def _load_card_renderer():
    """Load the three guide-card illustrations from their source module."""

    spec = importlib.util.spec_from_file_location("fatqat_guide_cards", CARD_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CARD_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render


def _figure_metadata(name: str) -> dict[str, object]:
    """Return metadata supported by the selected Matplotlib output format."""

    creator = "fatqat documentation figure renderer"
    if Path(name).suffix == ".svg":
        return {"Creator": creator, "Date": None}
    return {"Software": creator}


def _page_examples(page: Path) -> list[tuple[str, str]]:
    """Pair each inline reproduction block with the preceding guide image."""

    text = page.read_text(encoding="utf-8")
    examples: list[tuple[str, str]] = []
    previous_end = 0
    for match in REPRODUCTION.finditer(text):
        prefix = text[previous_end : match.start()]
        images = list(IMAGE.finditer(prefix))
        if not images:
            raise ValueError(f"{page}: reproduction block has no preceding figure")
        name = images[-1].group("name")
        code = textwrap.dedent(match.group("code"))
        examples.append((name, code))
        previous_end = match.end()
    return examples


def _render_inline_example(name: str, code: str, output: Path) -> None:
    """Execute one documented example and save its single Matplotlib figure."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.close("all")
    namespace = {"__name__": "__main__", "__package__": None}
    exec(compile(code, f"<documentation figure {name}>", "exec"), namespace)
    figure_numbers = plt.get_fignums()
    if len(figure_numbers) != 1:
        raise ValueError(
            f"{name}: reproduction example created {len(figure_numbers)} figures"
        )
    plt.figure(figure_numbers[0]).savefig(
        output / name,
        dpi=DEFAULT_FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata=_figure_metadata(name),
    )
    plt.close("all")


def _render_labeled_sources(
    sources: tuple[tuple[Path, tuple[str, ...]], ...],
    output: Path,
    *,
    group: str,
) -> set[str]:
    """Run figure scripts and save their contract-labeled Matplotlib figures."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rendered_names: set[str] = set()
    expected_all = {
        name for _source, expected_names in sources for name in expected_names
    }
    for source, expected_names in sources:
        plt.close("all")
        print(f"Rendering {group} figures from {source.relative_to(MKDOCS_ROOT)}")
        runpy.run_path(str(source))

        figures = {}
        for number in plt.get_fignums():
            figure = plt.figure(number)
            name = figure.get_label()
            if not name:
                raise ValueError(f"{source}: every figure needs a filename label")
            if name in figures:
                raise ValueError(f"{source}: duplicate figure label {name!r}")
            figures[name] = figure

        names = set(figures)
        expected = set(expected_names)
        if names != expected:
            raise ValueError(
                f"{source}: figure outputs do not match its contract: "
                f"missing={sorted(expected - names)}, "
                f"unexpected={sorted(names - expected)}"
            )
        duplicates = names & rendered_names
        if duplicates:
            raise ValueError(f"duplicate {group} figures: {sorted(duplicates)}")

        for name in expected_names:
            figure = figures[name]
            figure.savefig(
                output / name,
                dpi=DEFAULT_FIGURE_DPI,
                bbox_inches="tight",
                facecolor="white",
                metadata=_figure_metadata(name),
            )
        rendered_names.update(names)

    if rendered_names != expected_all:
        raise ValueError(f"{group} figure source contract is incomplete")
    plt.close("all")
    return rendered_names


def _sync_rendered(
    temporary_output: Path,
    output: Path,
    expected: set[str],
    *,
    suffixes: set[str],
) -> None:
    """Copy changed outputs and remove stale generated files."""

    output.mkdir(parents=True, exist_ok=True)
    for stale in output.iterdir():
        if stale.suffix in suffixes and stale.name not in expected:
            stale.unlink()
    for name in sorted(expected):
        temporary = temporary_output / name
        destination = output / name
        if destination.is_file() and destination.read_bytes() == temporary.read_bytes():
            continue
        shutil.copyfile(temporary, destination)


def render_all() -> None:
    """Regenerate every English guide and homepage figure."""

    import matplotlib

    # Select the non-interactive backend before importing any figure source.
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = "fatqat-documentation"

    source_path = str(REPOSITORY_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    figure_source_path = str(FIGURE_SOURCE_ROOT)
    if figure_source_path not in sys.path:
        sys.path.insert(0, figure_source_path)
    with tempfile.TemporaryDirectory(prefix="fatqat-mkdocs-figures-") as temp:
        temporary_root = Path(temp)
        temporary_guide = temporary_root / "guide"
        temporary_home = temporary_root / "home"
        temporary_guide.mkdir()
        temporary_home.mkdir()

        rendered = set(_load_card_renderer()(temporary_guide))
        sourced = _render_labeled_sources(
            GUIDE_SOURCES,
            temporary_guide,
            group="guide",
        )
        duplicates = rendered & sourced
        if duplicates:
            raise ValueError(f"duplicate guide figure source for {sorted(duplicates)}")
        rendered.update(sourced)
        for page in sorted(GUIDE_ROOT.glob("*.md")):
            for name, code in _page_examples(page):
                if name in rendered:
                    raise ValueError(f"duplicate guide figure source for {name}")
                print(f"Rendering {name} from {page.relative_to(MKDOCS_ROOT)}")
                _render_inline_example(name, code, temporary_guide)
                rendered.add(name)

        referenced = {
            match.group("name")
            for page in GUIDE_ROOT.glob("*.md")
            for match in IMAGE.finditer(page.read_text(encoding="utf-8"))
        }
        if rendered != referenced:
            raise ValueError(
                "guide figure sources do not match references: "
                f"missing={sorted(referenced - rendered)}, "
                f"unreferenced={sorted(rendered - referenced)}"
            )

        home_figures = _render_labeled_sources(
            HOME_SOURCES,
            temporary_home,
            group="homepage",
        )
        if home_figures != set(HOME_FIGURES):
            raise ValueError("homepage figure source contract is incomplete")

        _sync_rendered(
            temporary_guide,
            GUIDE_OUTPUT,
            rendered,
            suffixes={".png", ".svg"},
        )
        _sync_rendered(
            temporary_home,
            HOME_OUTPUT,
            home_figures,
            suffixes={".png"},
        )

    print(f"Rendered {len(rendered)} English guide figures.")
    print(f"Rendered {len(HOME_FIGURES)} English homepage figures.")


if __name__ == "__main__":
    render_all()
