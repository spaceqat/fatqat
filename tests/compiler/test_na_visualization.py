from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

import fatqat as fq
import fatqat.compiler as compiler
from fatqat.compiler import (
    ValidationError,
    create_na_animation as public_create_na_animation,
)
from fatqat.compiler.dialects import NAMeasure
from fatqat.compiler.dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    ScheduledGate,
    TransferEvent,
    ZonedPlan,
)
from fatqat.compiler.visualization import create_na_animation, save_na_animation
from fatqat.compiler.visualization.na_zoned import _build_frame_spans


@pytest.fixture(name="architecture")
def fixture_architecture():
    def zone(location, *, rows=1, columns=1, separation=(6, 6)):
        return {
            "slms": [
                {
                    "location": list(location),
                    "site_seperation": list(separation),
                    "r": rows,
                    "c": columns,
                }
            ]
        }

    return {
        "storage_zones": [zone((0, 0), columns=4)],
        "entanglement_zones": [zone((-24, 52)), zone((-20, 52))],
    }


@pytest.fixture(name="plan")
def fixture_plan():
    atoms = fq.QuantumRegister(4, name="atoms")
    clbits = fq.ClassicalRegister(1, name="bits")
    atom0, atom1, atom2, atom3 = atoms
    return ZonedPlan(
        atoms=tuple(atoms),
        clbits=tuple(clbits),
        initial_placement=(
            (atom0, (0.0, 0.0)),
            (atom1, (6.0, 0.0)),
            (atom2, (12.0, 0.0)),
            (atom3, (18.0, 0.0)),
        ),
        events=(
            TransferEvent(
                "activate",
                (atom0, atom1),
                ((0.0, 0.0), (6.0, 0.0)),
                (15.0, 15.0),
            ),
            MoveEvent(
                "big_move",
                (atom0, atom1),
                ((0.0, 0.0), (6.0, 0.0)),
                ((-24.0, 52.0), (-20.0, 52.0)),
                (57.3, 55.2),
                (5.0, 10.0),
            ),
            GateBatch(
                3,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.0",),
                        fq.operations.RX(0.25),
                        (atom2,),
                        ((12.0, 0.0),),
                    ),
                    ScheduledGate(
                        "na.1",
                        ("logical.1",),
                        fq.operations.CZ,
                        (atom0, atom1),
                        ((-24.0, 52.0), (-20.0, 52.0)),
                    ),
                ),
                52.0,
            ),
            CrosstalkEvent((atom0,), ((-24.0, 52.0),), (5.0,)),
            TransferEvent(
                "deactivate",
                (atom0, atom1),
                ((-24.0, 52.0), (-20.0, 52.0)),
                (15.0, 15.0),
            ),
        ),
        terminal_measurements=(NAMeasure("na.2", ("logical.2",), atom3, clbits[0]),),
    )


@pytest.fixture(name="animation")
def fixture_animation(plan, architecture):
    animation = create_na_animation(plan, architecture)
    yield animation
    _close_animation(animation)


def test_create_animation_is_read_only_and_covers_every_zoned_event(plan, architecture):
    before_plan = plan
    before_architecture = deepcopy(architecture)

    animation = create_na_animation(plan, architecture, fps=20)

    try:
        assert isinstance(animation, FuncAnimation)
        assert plan == before_plan
        assert architecture == before_architecture
        spans = _build_frame_spans(plan, 20)
        assert tuple(span.kind for span in spans) == (
            "initial",
            "activate",
            "big_move",
            "gate_batch",
            "crosstalk",
            "deactivate",
        )
        assert spans[0].end - spans[0].begin == 4
        assert spans[1].end - spans[1].begin >= 1
        assert spans[2].end - spans[2].begin == 4
        assert spans[3].end - spans[3].begin >= 8
        assert spans[4].end - spans[4].begin == 2
    finally:
        _close_animation(animation)


def test_canvas_draws_both_trap_grids_and_initial_atom_labels(plan, architecture):
    animation = create_na_animation(plan, architecture)

    try:
        animator = _animator(animation)
        trap_centers = {
            tuple(patch.center)
            for patch in animator.ax.patches
            if isinstance(patch, Circle) and patch.radius == 1
        }
        assert trap_centers == {
            (0, 0),
            (6, 0),
            (12, 0),
            (18, 0),
            (-24, 52),
            (-20, 52),
        }
        assert _offsets(animator) == [
            (0.0, 0.0),
            (6.0, 0.0),
            (12.0, 0.0),
            (18.0, 0.0),
        ]
        assert [label.get_text() for label in animator.atom_labels] == [
            "0",
            "1",
            "2",
            "3",
        ]
        assert [label.get_position() for label in animator.atom_labels] == [
            (-1.0, 1.0),
            (5.0, 1.0),
            (11.0, 1.0),
            (17.0, 1.0),
        ]
    finally:
        _close_animation(animation)


def test_activate_and_deactivate_toggle_aod_row_and_column_indicators(
    plan, architecture
):
    animation = create_na_animation(plan, architecture)

    try:
        animator = _animator(animation)
        spans = _build_frame_spans(plan, 30)
        activate = spans[1]
        deactivate = spans[-1]

        animation._func(activate.begin)
        assert animator.title.get_text() == "Activate"
        assert animator.column_guides[0].get_alpha() == 0.5
        assert animator.row_guides[0].get_alpha() == 0.5
        assert tuple(animator.column_guides[0].get_xdata()) == (0.0, 0.0)
        assert tuple(animator.row_guides[1].get_ydata()) == (0.0, 0.0)
        assert animator.column_guides[2].get_alpha() == 0.0

        animation._func(deactivate.begin)
        assert animator.title.get_text() == "Deactivate"
        assert animator.column_guides[0].get_alpha() == 0.0
        assert animator.row_guides[1].get_alpha() == 0.0
    finally:
        _close_animation(animation)


def test_big_move_cubic_interpolation_reaches_exact_start_and_end(plan, architecture):
    animation = create_na_animation(plan, architecture, fps=30)

    try:
        animator = _animator(animation)
        move = _build_frame_spans(plan, 30)[2]

        animation._func(move.begin)
        assert _offsets(animator)[:2] == [(0.0, 0.0), (6.0, 0.0)]

        animation._func(move.end - 1)
        assert _offsets(animator)[:2] == [(-24.0, 52.0), (-20.0, 52.0)]
        assert tuple(animator.column_guides[0].get_xdata()) == (-24.0, -24.0)
        assert tuple(animator.row_guides[1].get_ydata()) == (52.0, 52.0)
    finally:
        _close_animation(animation)


def test_mixed_gate_batch_highlights_each_location_and_global_rydberg_region(
    plan, architecture
):
    animation = create_na_animation(plan, architecture)

    try:
        animator = _animator(animation)
        gate_batch = _build_frame_spans(plan, 30)[3]

        animation._func(gate_batch.begin)

        assert animator.title.get_text() == "Stage 3\nGate Batch"
        assert {tuple(circle.center) for circle in animator.gate_highlights} == {
            (12.0, 0.0),
            (-24.0, 52.0),
            (-20.0, 52.0),
        }
        assert isinstance(animator.global_rydberg_overlay, Rectangle)
        assert animator.global_rydberg_overlay in animator.ax.patches
    finally:
        _close_animation(animation)


def test_crosstalk_has_a_visible_frame_without_changing_physical_state(
    plan, architecture
):
    animation = create_na_animation(plan, architecture)

    try:
        animator = _animator(animation)
        crosstalk = _build_frame_spans(plan, 30)[4]

        animation._func(crosstalk.begin)

        assert crosstalk.end > crosstalk.begin
        assert animator.title.get_text() == "Crosstalk"
        assert _offsets(animator)[:2] == [(-24.0, 52.0), (-20.0, 52.0)]
    finally:
        _close_animation(animation)


def test_valid_event_free_plan_still_has_an_initial_frame(architecture):
    atom = fq.QuantumRegister(1, name="atoms")[0]
    empty_plan = ZonedPlan(
        atoms=(atom,),
        clbits=(),
        initial_placement=((atom, (0.0, 0.0)),),
        events=(),
        terminal_measurements=(),
    )

    animation = create_na_animation(empty_plan, architecture, fps=1)

    try:
        spans = _build_frame_spans(empty_plan, 1)
        assert len(spans) == 1
        assert spans[0].kind == "initial"
        assert spans[0].begin == 0
        assert spans[0].end == 1
        assert tuple(animation.new_frame_seq()) == (0,)
    finally:
        _close_animation(animation)


@pytest.mark.parametrize("fps", (True, 0, -1, 2.5))
def test_create_animation_rejects_invalid_fps(plan, architecture, fps):
    with pytest.raises(ValueError, match="fps must be a positive integer"):
        create_na_animation(plan, architecture, fps=fps)


def test_create_animation_rejects_an_invalid_plan(architecture):
    with pytest.raises(ValidationError, match="expected ZonedPlan"):
        create_na_animation(object(), architecture)


def test_animation_construction_creates_no_files_and_keeps_ffmpeg_setting(
    tmp_path, monkeypatch, plan, architecture
):
    monkeypatch.chdir(tmp_path)
    previous_ffmpeg_path = mpl.rcParams["animation.ffmpeg_path"]

    animation = create_na_animation(plan, architecture)

    try:
        assert list(tmp_path.iterdir()) == []
        assert mpl.rcParams["animation.ffmpeg_path"] == previous_ffmpeg_path
    finally:
        _close_animation(animation)


def test_save_animation_requires_available_ffmpeg(monkeypatch, animation, tmp_path):
    monkeypatch.setattr(mpl.animation.writers, "is_available", lambda _: False)

    with pytest.raises(compiler.UnsupportedFeatureError, match="FFmpeg"):
        save_na_animation(animation, tmp_path / "schedule.mp4")


def test_save_animation_uses_only_the_requested_path(monkeypatch, animation, tmp_path):
    output = tmp_path / "schedule.mp4"
    seen = {}
    monkeypatch.setattr(mpl.animation.writers, "is_available", lambda _: True)
    monkeypatch.setattr(
        animation,
        "save",
        lambda path, **kwargs: seen.update(path=path, kwargs=kwargs),
    )

    save_na_animation(animation, output)

    assert seen == {"path": output, "kwargs": {"writer": "ffmpeg"}}
    assert tuple(tmp_path.iterdir()) == ()


def test_save_animation_rejects_non_mp4_output(animation, tmp_path):
    with pytest.raises(ValueError, match=r"\.mp4"):
        save_na_animation(animation, tmp_path / "schedule.gif")


def test_save_animation_does_not_create_a_missing_parent(animation, tmp_path):
    missing_parent = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        save_na_animation(animation, missing_parent / "schedule.mp4")

    assert not missing_parent.exists()


def test_importing_fatqat_compiler_does_not_check_ffmpeg():
    source_root = Path(__file__).parents[2] / "src"
    check_import = """
import matplotlib.animation

def fail_if_checked(_):
    raise AssertionError("compiler import must not check FFmpeg availability")

matplotlib.animation.writers.is_available = fail_if_checked
import fatqat.compiler
"""

    result = subprocess.run(
        [sys.executable, "-c", check_import],
        capture_output=True,
        check=False,
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(source_root)},
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_compiler_exports_animation_helpers():
    assert public_create_na_animation is create_na_animation
    assert compiler.save_na_animation is save_na_animation


def _animator(animation):
    return animation._func.__self__


def _offsets(animator):
    return [tuple(position) for position in animator.atom_scatter.get_offsets()]


def _close_animation(animation):
    animation._draw_was_started = True
    plt.close(animation._fig)
