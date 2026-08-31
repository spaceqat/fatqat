"""Tests for model-independent sampled waveform authoring."""

import math

import pytest

from fatqat.emulator import SampledWaveform


def test_sampled_waveform_copies_sequences_to_immutable_tuples():
    times = [0, 0.5, 1]
    values = [1, -2, 3]
    waveform = SampledWaveform(times, values)
    times[1] = 9
    values[1] = 9

    assert waveform.times == (0.0, 0.5, 1.0)
    assert waveform.values == (1.0, -2.0, 3.0)
    assert waveform.duration == 1.0
    assert hash(waveform) == hash(SampledWaveform((0, 0.5, 1), (1, -2, 3)))
    with pytest.raises(AttributeError):
        waveform.times = (0.0, 1.0)


@pytest.mark.parametrize(
    ("times", "values", "error"),
    [
        ((0,), (1,), ValueError),
        ((0, 1), (1,), ValueError),
        ((0.1, 1), (1, 2), ValueError),
        ((0, 0), (1, 2), ValueError),
        ((0, -1), (1, 2), ValueError),
        ((0, True), (1, 2), TypeError),
        ((0, 1), (False, 2), TypeError),
        ((0, math.inf), (1, 2), ValueError),
        ((0, 1), (1, math.nan), ValueError),
        ((0, 1), (1 + math.inf * 1j, 2), ValueError),
        ((0, 1), (1 + math.nan * 1j, 2), ValueError),
        (((0, 1), (2, 3)), (1, 2), TypeError),
    ],
)
def test_sampled_waveform_rejects_invalid_data(times, values, error):
    with pytest.raises(error):
        SampledWaveform(times, values)


def test_sampled_waveform_supports_finite_complex_and_mixed_samples():
    complex_waveform = SampledWaveform((0, 1), (1j, -2 + 0.5j))
    mixed_waveform = SampledWaveform((0, 1), (1, 2j))

    assert complex_waveform.values == (1j, -2 + 0.5j)
    assert mixed_waveform.values == (1.0, 2j)
    assert isinstance(mixed_waveform.values[0], float)
    assert hash(complex_waveform) == hash(SampledWaveform((0.0, 1.0), (1j, -2 + 0.5j)))


def test_sampled_waveform_has_no_interpolation_or_channel_policy():
    waveform = SampledWaveform((0, 2), (-1, 1))

    assert not hasattr(waveform, "interpolation")
    assert not hasattr(waveform, "channel")


def test_waveform_base_is_abstract():
    from fatqat._waveforms import Waveform

    with pytest.raises(TypeError, match="abstract"):
        Waveform()  # pylint: disable=abstract-class-instantiated
