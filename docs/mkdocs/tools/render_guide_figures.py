"""Render guide figures from their tracked, executable source examples."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shutil
import sys
import textwrap


SCRIPT_PATH = Path(__file__).resolve()
MKDOCS_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
EN_GUIDE_ROOT = MKDOCS_ROOT / "en" / "guide"
CARD_SOURCE = MKDOCS_ROOT / "figure-sources" / "guide_cards.py"
EN_OUTPUT = MKDOCS_ROOT / "en" / "assets" / "generated" / "guide"
ZH_OUTPUT = MKDOCS_ROOT / "zh" / "assets" / "generated" / "guide"

IMAGE = re.compile(
    r"!\[[^]]*]\(\.\./assets/generated/guide/(?P<name>[a-z0-9-]+\.png)\)"
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
    exec(compile(code, f"<guide figure {name}>", "exec"), namespace)
    figure_numbers = plt.get_fignums()
    if len(figure_numbers) != 1:
        raise ValueError(
            f"{name}: reproduction example created {len(figure_numbers)} figures"
        )
    plt.figure(figure_numbers[0]).savefig(
        output / name,
        dpi=144,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "fatqat guide figure renderer"},
    )
    plt.close("all")


def render_all() -> None:
    """Regenerate every guide figure referenced by English source Markdown."""

    source_path = str(REPOSITORY_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    EN_OUTPUT.mkdir(parents=True, exist_ok=True)

    rendered = set(_load_card_renderer()(EN_OUTPUT))
    for page in sorted(EN_GUIDE_ROOT.glob("*.md")):
        for name, code in _page_examples(page):
            if name in rendered:
                raise ValueError(f"duplicate guide figure source for {name}")
            print(f"Rendering {name} from {page.relative_to(MKDOCS_ROOT)}")
            _render_inline_example(name, code, EN_OUTPUT)
            rendered.add(name)

    referenced = {
        match.group("name")
        for page in EN_GUIDE_ROOT.glob("*.md")
        for match in IMAGE.finditer(page.read_text(encoding="utf-8"))
    }
    if rendered != referenced:
        raise ValueError(
            "guide figure sources do not match references: "
            f"missing={sorted(referenced - rendered)}, "
            f"unreferenced={sorted(rendered - referenced)}"
        )
    for stale in EN_OUTPUT.glob("*.png"):
        if stale.name not in rendered:
            stale.unlink()

    ZH_OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in ZH_OUTPUT.glob("*.png"):
        if stale.name not in rendered:
            stale.unlink()
    for name in sorted(rendered):
        shutil.copyfile(EN_OUTPUT / name, ZH_OUTPUT / name)
    print(f"Rendered {len(rendered)} guide figures for both locales.")


if __name__ == "__main__":
    render_all()
