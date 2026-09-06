"""Regression coverage for executable documentation figure sources."""

import importlib.util
from pathlib import Path


def _load_figure_renderer():
    source = (
        Path(__file__).parents[2]
        / "docs"
        / "mkdocs"
        / "tools"
        / "render_guide_figures.py"
    )
    spec = importlib.util.spec_from_file_location("fatqat_figure_renderer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figure_renderer = _load_figure_renderer()


def test_inline_text_output_is_captured_verbatim(tmp_path):
    figure_renderer._render_inline_example(
        "output",
        "example.txt",
        'print("generated output", end="")',
        tmp_path,
    )

    assert (tmp_path / "example.txt").read_text(encoding="utf-8") == "generated output"


def test_guide_card_renderer_runs_against_current_simulator_api(tmp_path):
    rendered = figure_renderer._load_card_renderer()(tmp_path)

    assert rendered == (
        "guide-path-algorithm.png",
        "guide-path-hardware.png",
        "guide-path-physics.png",
    )
    assert {path.name for path in tmp_path.iterdir()} == set(rendered)


def test_all_documentation_figures_render_against_current_api(tmp_path, monkeypatch):
    guide_output = tmp_path / "guide"
    home_output = tmp_path / "home"
    monkeypatch.setattr(figure_renderer, "GUIDE_OUTPUT", guide_output)
    monkeypatch.setattr(figure_renderer, "HOME_OUTPUT", home_output)

    figure_renderer.render_all()

    assert {path.name for path in home_output.iterdir()} == set(
        figure_renderer.HOME_FIGURES
    )
