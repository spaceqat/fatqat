import pytest

import fatqat as fq
from fatqat.compiler.algorithms import ExecuteNode, RouteSwap, SabreResult
from fatqat.compiler.dialects import SCNode


def _qrefs(count: int):
    register = fq.QuantumRegister(count, name="q")
    return tuple(register[index] for index in range(count))


def test_semantic_swap_node_and_physical_route_swap_are_distinct_values():
    q0, q1 = _qrefs(2)
    semantic = SCNode("sc.0", ("logical.0",), fq.operations.Swap, (q0, q1))
    routed = RouteSwap("route.swap.0", (2, 3))

    assert semantic.qubits == (q0, q1)
    assert semantic.origin_ids == ("logical.0",)
    assert routed.sites == (2, 3)
    assert not hasattr(routed, "origin_ids")


def test_sabre_result_keeps_ordered_events_and_layout_snapshots():
    q0, q1 = _qrefs(2)
    result = SabreResult(
        events=(ExecuteNode(0, (0, 2)), RouteSwap("route.swap.0", (1, 2))),
        initial_layout=((q0, 0), (q1, 2)),
        final_layout=((q0, 0), (q1, 1)),
    )

    assert result.events[1].swap_id == "route.swap.0"
    assert result.initial_layout != result.final_layout


def test_route_swap_requires_two_distinct_sites_and_nonempty_id():
    with pytest.raises(ValueError, match="distinct"):
        RouteSwap("route.swap.0", (2, 2))
    with pytest.raises(ValueError, match="non-empty"):
        RouteSwap("", (1, 2))


def test_route_swap_id_does_not_enforce_a_naming_policy():
    event = RouteSwap("custom-router-swap-0", (1, 2))

    assert event.swap_id == "custom-router-swap-0"


def test_sabre_layout_is_injective():
    q0, q1 = _qrefs(2)

    with pytest.raises(ValueError, match="site appears more than once"):
        SabreResult((), ((q0, 0), (q1, 0)), ((q0, 0), (q1, 1)))


def test_sabre_layout_rejects_classical_refs():
    classical = fq.ClassicalRegister(1, name="c")[0]

    with pytest.raises(TypeError, match="qubits"):
        SabreResult((), ((classical, 0),), ((classical, 0),))


def test_sabre_layouts_must_contain_the_same_refs():
    q0, q1, q2 = _qrefs(3)

    with pytest.raises(ValueError, match="same refs"):
        SabreResult((), ((q0, 0), (q1, 1)), ((q0, 0), (q2, 1)))
