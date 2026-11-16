import subprocess
import sys

import numpy as np
import pytest

import fatqat as fq
from fatqat import _sc_target
from fatqat.compiler import SCTarget
from fatqat.simulator import SCQubitSimulator
from fatqat.simulator.fake_superconducting import (
    DEFAULT_COUPLINGS,
    DEFAULT_NUM_QUBITS,
    _SCQubitRotationSimulator,
)

# Every constructor below shares one validation implementation, so each of
# them must reject the same inputs with the same message.
_SC_CONSTRUCTORS = (SCTarget, SCQubitSimulator, _SCQubitRotationSimulator)

_REJECTED_CONSTRUCTOR_INPUTS = (
    ({"num_qubits": 3.0, "couplings": ()}, TypeError, "num_qubits must be an integer"),
    (
        {"num_qubits": 0, "couplings": ()},
        ValueError,
        "num_qubits must be a positive integer",
    ),
    (
        {"num_qubits": 3, "couplings": ([0, 1],)},
        TypeError,
        "couplings must contain two-integer tuples",
    ),
    (
        {"num_qubits": 3, "couplings": ((0, "1"),)},
        TypeError,
        "coupling endpoints must be integers",
    ),
    (
        {"num_qubits": 3, "couplings": ((0, 3),)},
        ValueError,
        "coupling endpoint is outside device_sites",
    ),
    (
        {"num_qubits": 3, "couplings": ((1, 1),)},
        ValueError,
        "coupling endpoints must be distinct",
    ),
)


def test_sc_target_is_public_on_the_compiler_namespace():
    assert fq.compiler.SCTarget is SCTarget
    assert "SCTarget" in fq.compiler.__all__


def test_sc_target_is_defined_in_the_top_level_private_module():
    # Placement is a hard constraint, not a style choice: defining the value
    # object under fatqat.compiler and importing it back from the simulator
    # closes a real import cycle in both directions.
    assert SCTarget is _sc_target.SCTarget
    assert SCTarget.__module__ == "fatqat._sc_target"


def test_sc_target_is_not_aliased_on_the_top_level_namespace():
    with pytest.raises(AttributeError):
        fq.SCTarget


@pytest.mark.parametrize("first", ["fatqat.simulator", "fatqat.compiler"])
def test_both_import_orders_succeed_in_a_fresh_interpreter(first):
    source = f"import {first}\nimport fatqat.compiler\nimport fatqat.simulator\n"
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_construction_rejects_positional_arguments():
    with pytest.raises(TypeError):
        # pylint: disable-next=missing-kwoa
        SCTarget(3, ())


@pytest.mark.parametrize("kwargs", [{}, {"num_qubits": 3}, {"couplings": ()}])
def test_construction_requires_both_fields(kwargs):
    with pytest.raises(TypeError):
        SCTarget(**kwargs)


def test_construction_rejects_unknown_keywords():
    with pytest.raises(TypeError):
        SCTarget(num_qubits=3, couplings=(), basis=("x",))


@pytest.mark.parametrize("num_qubits", [True, 3.0, "3", None, np.int64(3)])
def test_rejects_a_site_count_that_is_not_a_strict_int(num_qubits):
    with pytest.raises(TypeError) as excinfo:
        SCTarget(num_qubits=num_qubits, couplings=())

    assert str(excinfo.value) == "num_qubits must be an integer"


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_rejects_a_non_positive_site_count(num_qubits):
    with pytest.raises(ValueError) as excinfo:
        SCTarget(num_qubits=num_qubits, couplings=())

    assert str(excinfo.value) == "num_qubits must be a positive integer"


@pytest.mark.parametrize("couplings", [([0, 1],), ((0, 1, 2),), ((0,),), (0,), ("01",)])
def test_rejects_couplings_that_are_not_two_element_tuples(couplings):
    with pytest.raises(TypeError) as excinfo:
        SCTarget(num_qubits=3, couplings=couplings)

    assert str(excinfo.value) == "couplings must contain two-integer tuples"


def test_rejects_couplings_that_are_not_iterable():
    with pytest.raises(TypeError):
        SCTarget(num_qubits=3, couplings=None)


@pytest.mark.parametrize("edge", [(0, "1"), (0, 1.0), (True, 1), (np.int64(0), 1)])
def test_rejects_coupling_endpoints_that_are_not_strict_ints(edge):
    with pytest.raises(TypeError) as excinfo:
        SCTarget(num_qubits=3, couplings=(edge,))

    assert str(excinfo.value) == "coupling endpoints must be integers"


@pytest.mark.parametrize("edge", [(0, 3), (3, 0), (-1, 0), (0, -1)])
def test_rejects_coupling_endpoints_outside_the_site_range(edge):
    with pytest.raises(ValueError) as excinfo:
        SCTarget(num_qubits=3, couplings=(edge,))

    assert str(excinfo.value) == "coupling endpoint is outside device_sites"


def test_rejects_self_couplings():
    with pytest.raises(ValueError) as excinfo:
        SCTarget(num_qubits=3, couplings=((1, 1),))

    assert str(excinfo.value) == "coupling endpoints must be distinct"


def test_normalizes_each_edge_to_ascending_endpoints():
    assert SCTarget(num_qubits=3, couplings=((2, 0),)).couplings == ((0, 2),)


def test_merges_repeated_and_reversed_edges_keeping_first_canonical_order():
    target = SCTarget(num_qubits=3, couplings=((2, 1), (0, 1), (1, 2), (1, 0)))

    assert target.couplings == ((1, 2), (0, 1))


def test_accepts_a_single_site_without_couplings():
    target = SCTarget(num_qubits=1, couplings=())

    assert (target.num_qubits, target.couplings) == (1, ())


def test_accepts_several_sites_without_couplings():
    target = SCTarget(num_qubits=4, couplings=[])

    assert (target.num_qubits, target.couplings) == (4, ())


def test_accepts_a_three_site_chain():
    target = SCTarget(num_qubits=3, couplings=((0, 1), (1, 2)))

    assert target.couplings == ((0, 1), (1, 2))


def test_accepts_a_fully_connected_three_site_target():
    target = SCTarget(num_qubits=3, couplings=((0, 1), (0, 2), (1, 2)))

    assert target.couplings == ((0, 1), (0, 2), (1, 2))


def test_reads_the_coupling_input_once_and_ignores_later_mutation():
    edges = [(0, 1), (1, 2)]

    target = SCTarget(num_qubits=3, couplings=edges)
    edges.clear()
    edges.append((0, 2))

    assert target.couplings == ((0, 1), (1, 2))


def test_accepts_a_one_shot_generator():
    target = SCTarget(num_qubits=3, couplings=(edge for edge in [(1, 0), (2, 1)]))

    assert target.couplings == ((0, 1), (1, 2))


@pytest.mark.parametrize("attribute, value", [("num_qubits", 5), ("couplings", ())])
def test_attributes_are_read_only(attribute, value):
    target = SCTarget(num_qubits=3, couplings=((0, 1),))

    with pytest.raises(AttributeError):
        setattr(target, attribute, value)


@pytest.mark.parametrize("constructor", _SC_CONSTRUCTORS)
@pytest.mark.parametrize("kwargs, error, message", _REJECTED_CONSTRUCTOR_INPUTS)
def test_target_and_both_simulators_share_one_validation_source(
    constructor, kwargs, error, message
):
    with pytest.raises(error) as excinfo:
        constructor(**kwargs)

    assert str(excinfo.value) == message


def test_get_compiler_target_returns_an_sc_target():
    backend = SCQubitSimulator(num_qubits=3, couplings=((0, 1),))

    assert isinstance(backend.get_compiler_target(), SCTarget)


def test_get_compiler_target_returns_the_same_cached_snapshot():
    backend = SCQubitSimulator(num_qubits=3, couplings=((0, 1),))

    assert backend.get_compiler_target() is backend.get_compiler_target()


def test_get_compiler_target_matches_the_normalized_constructor_input():
    backend = SCQubitSimulator(num_qubits=3, couplings=[(2, 1), (0, 1), (1, 2)])

    target = backend.get_compiler_target()

    assert (target.num_qubits, target.couplings) == (3, ((1, 2), (0, 1)))


def test_get_compiler_target_covers_every_device_site():
    backend = SCQubitSimulator(num_qubits=4, couplings=((0, 1),))

    assert backend.get_compiler_target().num_qubits == len(backend.device_sites)


def test_get_compiler_target_carries_the_default_grid():
    target = SCQubitSimulator().get_compiler_target()

    assert (target.num_qubits, target.couplings) == (
        DEFAULT_NUM_QUBITS,
        DEFAULT_COUPLINGS,
    )


@pytest.mark.parametrize(
    "attribute",
    ["implementation_for", "implementation_map", "run", "device_sites"],
)
def test_compiler_target_carries_construction_constraints_only(attribute):
    # Gate rules stay on the simulator; the target is capacity plus CZ edges.
    target = SCQubitSimulator(num_qubits=3, couplings=((0, 1),)).get_compiler_target()

    assert not hasattr(target, attribute)


def test_rotation_profile_exposes_no_compiler_target():
    backend = _SCQubitRotationSimulator(num_qubits=3, couplings=((0, 1),))

    assert not hasattr(backend, "get_compiler_target")
