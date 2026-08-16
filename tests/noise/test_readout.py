"""ReadoutConfusion value tests."""

import numpy as np
import pytest

from fatqat.noise import ReadoutConfusion


def test_readout_confusion_copies_and_freezes_matrix():
    source = np.array([[0.9, 0.2], [0.1, 0.8]])
    confusion = ReadoutConfusion(source)

    source[0, 0] = 0.0
    assert np.allclose(confusion.matrix, [[0.9, 0.2], [0.1, 0.8]])
    assert confusion.matrix.flags.writeable is False
    with pytest.raises(ValueError):
        confusion.matrix[0, 0] = 0.0


def test_readout_confusion_has_content_value_semantics():
    first = ReadoutConfusion([[0.9, 0.2], [0.1, 0.8]])
    second = ReadoutConfusion([[0.9, 0.2], [0.1, 0.8]])

    assert first == second
    assert hash(first) == hash(second)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(1),
        np.ones((2, 3)),
        [[0.8, 0.2], [0.1, 0.7]],
        [[1.1, 0.0], [-0.1, 1.0]],
        [[float("nan"), 0.0], [0.0, 1.0]],
    ],
)
def test_readout_confusion_rejects_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        ReadoutConfusion(matrix)
