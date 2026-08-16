"""Carrier-loss declaration tests."""

import pytest

from fatqat.noise import Channel, Loss


def test_loss_is_width_agnostic_and_not_a_channel():
    loss = Loss(p=0.25)

    assert loss.p == 0.25
    assert not isinstance(loss, Channel)
    assert not hasattr(loss, "num_subsystems")


@pytest.mark.parametrize("bad_p", [-0.1, 1.1, True, "0.1"])
def test_loss_validates_probability(bad_p):
    with pytest.raises(ValueError):
        Loss(p=bad_p)
