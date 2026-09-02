"""Tests for Result counts visualization."""

import matplotlib
import pytest
from cycler import cycler
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from fatqat.errors import ResultFieldUnavailableError
from fatqat.result import Result


def _result(counts: dict[str, int]) -> Result:
    width = len(next(iter(counts))) if counts else 1
    tuple_counts = {
        tuple(int(digit) for digit in label): count for label, count in counts.items()
    }
    return Result(
        counts=tuple_counts,
        classical_dims=(3,) * width,
        available=frozenset({"counts"}),
    )


def _tick_labels(figure) -> list[str]:
    return [label.get_text() for label in figure.axes[0].get_xticklabels()]


def _bar_heights(figure) -> list[float]:
    return [patch.get_height() for patch in figure.axes[0].patches]


def test_result_draw_defaults_to_counts_view():
    figure = _result({"0": 3, "1": 1}).draw()

    assert isinstance(figure, matplotlib.figure.Figure)
    assert figure.axes[0].get_ylabel() == "Counts"
    assert _bar_heights(figure) == [3, 1]


def test_frequencies_stat_divides_counts_by_total():
    figure = _result({"0": 3, "1": 1}).draw(stat="frequencies")

    assert _bar_heights(figure) == pytest.approx([0.75, 0.25])
    assert figure.axes[0].get_ylabel() == "Frequency"
    assert figure.axes[0].get_ylim() == pytest.approx((0, 1))


def test_counts_sort_by_key():
    figure = _result({"10": 2, "00": 7, "01": 5}).draw(sort="key")

    assert _tick_labels(figure) == ["00", "01", "10"]
    assert _bar_heights(figure) == [7, 5, 2]


def test_counts_sort_by_descending_count_with_key_tiebreaker():
    figure = _result({"10": 2, "00": 5, "01": 2}).draw(sort="count")

    assert _tick_labels(figure) == ["00", "01", "10"]
    assert _bar_heights(figure) == [5, 2, 2]


def test_number_to_keep_aggregates_less_frequent_outcomes():
    figure = _result({"00": 7, "01": 2, "10": 5}).draw(
        number_to_keep=2,
        sort="key",
    )

    assert _tick_labels(figure) == ["00", "10", "other"]
    assert _bar_heights(figure) == [7, 5, 2]


def test_frequencies_include_aggregated_other_in_unit_total():
    figure = _result({"00": 7, "01": 2, "10": 5}).draw(
        stat="frequencies",
        number_to_keep=1,
    )

    assert _tick_labels(figure) == ["00", "other"]
    assert sum(_bar_heights(figure)) == pytest.approx(1)
    assert _bar_heights(figure) == pytest.approx([0.5, 0.5])


def test_counts_draw_preserves_erasure_digit_two():
    figure = _result({"0": 2, "2": 3}).draw()

    assert _tick_labels(figure) == ["0", "2"]


def test_counts_draw_inherits_matplotlib_colors():
    with matplotlib.rc_context(
        {
            "figure.facecolor": "#fef3c7",
            "axes.facecolor": "#ecfdf5",
            "axes.prop_cycle": cycler(color=["#db2777"]),
        }
    ):
        figure = _result({"0": 2}).draw()

    assert to_hex(figure.get_facecolor()) == "#fef3c7"
    assert to_hex(figure.axes[0].get_facecolor()) == "#ecfdf5"
    assert to_hex(figure.axes[0].patches[0].get_facecolor()) == "#db2777"


def test_count_bars_have_no_visible_edge():
    with matplotlib.rc_context(
        {"patch.force_edgecolor": True, "patch.edgecolor": "#db2777"}
    ):
        figure = _result({"0": 2}).draw()
    edge = figure.axes[0].patches[0].get_edgecolor()

    assert edge[-1] == 0


def test_counts_draw_embeds_in_existing_axis():
    host = Figure()
    FigureCanvasAgg(host)
    axis = host.add_subplot(111)

    returned = _result({"0": 2}).draw(ax=axis)

    assert returned is host
    assert len(axis.patches) == 1


def test_counts_draw_rejects_figsize_with_existing_axis():
    host = Figure()
    FigureCanvasAgg(host)
    axis = host.add_subplot(111)

    with pytest.raises(ValueError, match="figsize cannot"):
        _result({"0": 2}).draw(ax=axis, figsize=(5, 3))


def test_counts_draw_unavailable_raises():
    with pytest.raises(ResultFieldUnavailableError, match="counts not available"):
        Result().draw()


def test_counts_draw_rejects_empty_counts():
    result = Result(
        counts={},
        classical_dims=(2,),
        available=frozenset({"counts"}),
    )

    with pytest.raises(ValueError, match="counts are empty"):
        result.draw()


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    (
        ({"view": "state"}, ValueError, "unsupported Result view"),
        ({"stat": "probabilities"}, ValueError, "stat must be"),
        ({"stat": 1}, TypeError, "stat must be a string"),
        ({"sort": "alphabetical"}, ValueError, "sort must be"),
        ({"sort": 1}, TypeError, "sort must be a string"),
        ({"number_to_keep": 0}, ValueError, "must be positive"),
        ({"number_to_keep": True}, TypeError, "positive int or None"),
    ),
)
def test_counts_draw_rejects_invalid_options(kwargs, exception, match):
    with pytest.raises(exception, match=match):
        _result({"0": 2}).draw(**kwargs)
