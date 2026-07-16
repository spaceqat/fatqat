"""Channel registry (`ChannelImplementationMap`) and CPTP validation behavior."""

import numpy as np
import pytest

from fatqat.errors import BackendValidationError
from fatqat.noise import (
    Channel,
    ChannelImplementationMap,
    default_channel_implementation_map,
)
from fatqat.noise.base import _validate_cptp
from fatqat.noise.catalog import Depolarizing


class _CustomChannel(Channel):
    pass


def _identity_rule(channel, *, targets):
    return (np.eye(2, dtype=complex),)


def test_register_and_get_roundtrip():
    channel_map = ChannelImplementationMap()
    channel_map.register(_CustomChannel, _identity_rule)

    assert channel_map.get(_CustomChannel) is _identity_rule
    assert channel_map.get(Depolarizing) is None
    assert channel_map.supported_channels() == frozenset({_CustomChannel})


def test_register_rejects_non_channel_type():
    channel_map = ChannelImplementationMap()
    with pytest.raises(TypeError):
        channel_map.register(int, _identity_rule)


def test_register_rejects_non_callable_rule():
    channel_map = ChannelImplementationMap()
    with pytest.raises(TypeError):
        channel_map.register(_CustomChannel, "not a rule")


def test_copy_is_independent():
    channel_map = ChannelImplementationMap()
    channel_map.register(_CustomChannel, _identity_rule)
    clone = channel_map.copy()
    clone.register(Depolarizing, _identity_rule)

    assert channel_map.get(Depolarizing) is None
    assert clone.get(_CustomChannel) is _identity_rule


def test_default_map_covers_catalog():
    channel_map = default_channel_implementation_map()
    names = {c.__name__ for c in channel_map.supported_channels()}
    assert names == {"Depolarizing", "AmplitudeDamping", "PhaseDamping"}


def test_validate_cptp_accepts_complete_kraus_set():
    gamma = 0.3
    k0 = np.diag([1.0, np.sqrt(1 - gamma)]).astype(complex)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    _validate_cptp((k0, k1), 2, "test")  # must not raise


def test_validate_cptp_rejects_incomplete_set():
    with pytest.raises(BackendValidationError, match="not trace-preserving"):
        _validate_cptp((0.5 * np.eye(2, dtype=complex),), 2, "test")


def test_validate_cptp_rejects_wrong_shape():
    with pytest.raises(BackendValidationError, match="shape"):
        _validate_cptp((np.eye(3, dtype=complex),), 2, "test")


def test_validate_cptp_rejects_empty_tuple():
    with pytest.raises(BackendValidationError, match="empty"):
        _validate_cptp((), 2, "test")
