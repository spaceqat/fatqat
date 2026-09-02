"""Channel registry (`ChannelImplementationMap`) and Kraus shape validation."""

import numpy as np
import pytest

from fatqat.errors import BackendValidationError
from fatqat.noise import (
    Channel,
    ChannelImplementationMap,
    default_channel_implementation_map,
)
from fatqat.noise.base import _validate_kraus_shapes
from fatqat.noise.catalog import Depolarizing


class _CustomChannel(Channel):
    pass


def test_num_subsystems_is_the_only_public_channel_width_name():
    class TwoSubsystemChannel(Channel):
        num_subsystems = 2

    assert Channel.num_subsystems is None
    assert _CustomChannel().num_subsystems is None
    assert TwoSubsystemChannel.num_subsystems == 2
    assert TwoSubsystemChannel().num_subsystems == 2
    assert not hasattr(Channel, "arity")
    assert not hasattr(TwoSubsystemChannel(), "arity")
    assert not hasattr(Channel, "_num_subsystems")
    assert not hasattr(TwoSubsystemChannel(), "_num_subsystems")


def test_custom_channel_may_derive_num_subsystems_from_instance_data():
    class DynamicChannel(Channel):
        def __init__(self, width):
            self.width = width

        @property
        def num_subsystems(self):
            return self.width

    assert DynamicChannel(3).num_subsystems == 3


def test_custom_channel_rejects_retired_private_width_declaration():
    with pytest.raises(TypeError, match="num_subsystems"):
        type("LegacySubsystemCount", (Channel,), {"_num_subsystems": 2})


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_custom_channel_rejects_invalid_num_subsystems(bad):
    with pytest.raises(ValueError, match="num_subsystems"):
        type("BadSubsystemCount", (Channel,), {"num_subsystems": bad})


def _identity_rule(channel, *, targets):
    return (np.eye(2, dtype=complex),)


def test_add_is_the_only_public_registration_verb_and_roundtrips():
    channel_map = ChannelImplementationMap()
    assert channel_map.add(_CustomChannel, _identity_rule) is None

    assert channel_map.get(_CustomChannel) is _identity_rule
    assert channel_map.get(Depolarizing) is None
    assert channel_map.supported_channels() == frozenset({_CustomChannel})
    assert not hasattr(channel_map, "register")


def test_add_rejects_non_channel_type():
    channel_map = ChannelImplementationMap()
    with pytest.raises(TypeError):
        channel_map.add(int, _identity_rule)


def test_add_rejects_non_callable_rule():
    channel_map = ChannelImplementationMap()
    with pytest.raises(TypeError):
        channel_map.add(_CustomChannel, "not a rule")


def test_copy_is_independent():
    channel_map = ChannelImplementationMap()
    channel_map.add(_CustomChannel, _identity_rule)
    clone = channel_map.copy()
    clone.add(Depolarizing, _identity_rule)

    assert channel_map.get(Depolarizing) is None
    assert clone.get(_CustomChannel) is _identity_rule


def test_default_map_covers_catalog():
    channel_map = default_channel_implementation_map()
    names = {c.__name__ for c in channel_map.supported_channels()}
    assert names == {
        "Depolarizing",
        "AmplitudeDamping",
        "TransitionRelaxation",
        "PhaseDamping",
        "PauliChannel",
    }


def test_validate_shapes_accepts_complete_kraus_set():
    gamma = 0.3
    k0 = np.diag([1.0, np.sqrt(1 - gamma)]).astype(complex)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    _validate_kraus_shapes((k0, k1), 2, "test")  # must not raise


def test_validate_shapes_accepts_non_cptp_set():
    # Deliberate: trace preservation is not enforced at runtime, the same
    # posture as gate matrices never being checked for unitarity. The
    # built-in catalog's CPTP property is covered by its own tests.
    _validate_kraus_shapes((0.5 * np.eye(2, dtype=complex),), 2, "test")


def test_validate_shapes_rejects_wrong_shape():
    with pytest.raises(BackendValidationError, match="shape"):
        _validate_kraus_shapes((np.eye(3, dtype=complex),), 2, "test")


def test_validate_shapes_rejects_empty_tuple():
    with pytest.raises(BackendValidationError, match="empty"):
        _validate_kraus_shapes((), 2, "test")
