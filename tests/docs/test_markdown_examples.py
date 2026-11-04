"""Execute the code examples in the Markdown user guides.

Each guide page under ``docs/mkdocs/en/guide`` becomes one test. The test runs
the page's fenced code blocks from top to bottom in a single namespace, the way
a reader working through the page would:

- ``pycon`` blocks are doctests: each ``>>>`` example runs and its printed
  output must match the text shown on the page;
- ``python`` blocks only have to run without raising.

Two kinds of blocks are left out. The "Reproduce this figure/output" blocks
are already executed by the documentation build (``render_guide_figures``) and
by ``test_guide_cards.py``. A block whose first line is ``# guide-test: skip``
is an intentional fragment, or needs something CI does not have (Qiskit,
FFmpeg); the documentation build removes that marker line before rendering, so
readers never see it.

The tests run in the normal ``python -m pytest`` session. Run a single page
with, for example::

    python -m pytest tests/docs/test_markdown_examples.py -k program
"""

import doctest
import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import fatqat as fq

MKDOCS = Path(__file__).parents[2] / "docs" / "mkdocs"
GUIDES = MKDOCS / "en" / "guide"
# A fenced pycon or python block. A block may be indented, for example inside an
# admonition or a content tab; its closing fence uses the same indentation.
CODE_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<language>pycon|python)[ \t]*\n"
    r"(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
# A block whose first line is this comment is not run. Keep the spelling in
# sync with on_page_markdown in docs/mkdocs/hooks.py, which hides it.
SKIP_MARKER = "# guide-test: skip"
# "..." in a shown output matches anything, and runs of whitespace compare equal.
OPTIONS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


def _load_figure_renderer():
    """Import the documentation figure tool, which is not an installed module."""

    source = MKDOCS / "tools" / "render_guide_figures.py"
    spec = importlib.util.spec_from_file_location("fatqat_figure_renderer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The documentation build already runs the "Reproduce this ..." blocks; reuse
# its pattern so both sides agree on which blocks those are.
REPRODUCTION = _load_figure_renderer().REPRODUCTION


def _code_blocks(page: Path) -> list[tuple[int, str, str]]:
    """Return ``(line, language, code)`` for each candidate block, in page order.

    ``line`` is the page line of the block's first code line, used in error
    locations. ``code`` has the block's indentation removed. Skip-marked blocks
    are still returned; the test decides to skip them.
    """

    text = page.read_text(encoding="utf-8")
    reproduced = [match.span() for match in REPRODUCTION.finditer(text)]
    blocks = []
    for match in CODE_BLOCK.finditer(text):
        if any(start <= match.start() < end for start, end in reproduced):
            continue
        indent = match.group("indent")
        code = "\n".join(
            line.removeprefix(indent) for line in match.group("body").splitlines()
        )
        line = text.count("\n", 0, match.start()) + 2
        blocks.append((line, match.group("language"), code))
    return blocks


# Pages without any runnable block (such as the guide index) get no test.
PAGES = sorted(page for page in GUIDES.glob("*.md") if _code_blocks(page))


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.stem)
def test_guide_examples_run_and_match_their_output(page, tmp_path, monkeypatch):
    """Run one guide page and fail at the first block that raises or mismatches."""

    # Examples may write files (for example ``figure.savefig``); keep them out
    # of the repository.
    monkeypatch.chdir(tmp_path)
    # One namespace per page, shared by every block, so later blocks can use
    # names defined earlier on the same page.
    namespace = {"__name__": "__main__"}
    # Guides may change process-wide defaults, such as simulation.md calling
    # set_default_runtime; restore them so later tests are unaffected.
    default_runtime = fq.simulator.get_default_runtime()
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=OPTIONS)
    try:
        for line, language, code in _code_blocks(page):
            if code.partition("\n")[0].strip() == SKIP_MARKER:
                continue
            location = f"{page.name}:{line}"
            if language == "python":
                compiled = compile(code, location, "exec")
                exec(compiled, namespace)  # pylint: disable=exec-used
                continue
            test = parser.get_doctest(code, namespace, location, str(page), line - 1)
            # get_doctest copies the globals; share the page namespace instead.
            test.globs = namespace
            failures = []
            runner.run(test, out=failures.append, clear_globs=False)
            assert not failures, "".join(failures)
    finally:
        fq.simulator.set_default_runtime(default_runtime)
        # Matplotlib keeps figures open globally; do not leak them into the next
        # page.
        plt.close("all")
