"""Tests SimulatorBackend method selection and the Barrier no-op contract."""

import numpy as np
import pytest

from fatqat.backends import SimulatorBackend
from fatqat.errors import BackendValidationError
from fatqat import operations as ops
from fatqat.program import Program


def _bell(with_barriers: bool = False) -> Program:
    p = Program(2, 2)
    p.add(ops.H, 0)
    if with_barriers:
        p.add(ops.Barrier, (0, 1))
    p.add(ops.CX, (0, 1))
    if with_barriers:
        p.add(ops.Barrier, 0)
    p.add_measurement((0, 1), (0, 1))
    return p


# --- method selection --------------------------------------------------------


def test_default_method_is_statevector():
    backend = SimulatorBackend()
    assert backend._state_field == "statevector"


@pytest.mark.parametrize("alias", ["density_matrix", "DM", "dm", "Dm"])
def test_density_matrix_aliases(alias):
    assert SimulatorBackend(method=alias)._state_field == "density_matrix"


@pytest.mark.parametrize("alias", ["statevector", "SV", "sv"])
def test_statevector_aliases(alias):
    assert SimulatorBackend(method=alias)._state_field == "statevector"


def test_unknown_method_rejected():
    with pytest.raises(BackendValidationError, match="unsupported method"):
        SimulatorBackend(method="mps")


def test_alias_selects_identical_behavior():
    p = _bell()
    a = SimulatorBackend(method="SV").run(p, shots=64, seed=7).result()
    b = SimulatorBackend(method="statevector").run(p, shots=64, seed=7).result()
    assert a.get_counts() == b.get_counts()
    c = SimulatorBackend(method="DM").run(p, shots=64, seed=7).result()
    d = SimulatorBackend(method="density_matrix").run(p, shots=64, seed=7).result()
    assert c.get_counts() == d.get_counts()


def test_method_selects_native_state_field():
    p = Program(1)
    p.add(ops.H, 0)
    sv = (
        SimulatorBackend(method="SV")
        .run(p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )
    rho = (
        SimulatorBackend(method="DM")
        .run(p, result_config={"counts": False, "density_matrix": True})
        .result()
        .get_density_matrix()
    )
    assert np.allclose(rho, np.outer(sv, sv.conj()))


def test_metadata_records_method_and_backend_name():
    p = _bell()
    result = SimulatorBackend(method="DM").run(p, shots=5, seed=0).result()
    assert result.metadata["backend_name"] == "SimulatorBackend"
    assert result.metadata["method"] == "density_matrix"


# --- barrier: no simulation semantics ----------------------------------------


def test_barrier_is_skipped_in_lowering():
    backend = SimulatorBackend()
    p = _bell(with_barriers=True)
    plan, facts = backend._lower_program(p)
    p_ref = _bell()
    plan_ref, _ = backend._lower_program(p_ref)
    assert len(plan) == len(plan_ref)  # barriers emit no steps
    assert facts.has_measurement is True and facts.has_reset is False


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_barrier_does_not_change_counts(method):
    with_b = (
        SimulatorBackend(method=method)
        .run(_bell(with_barriers=True), shots=128, seed=11)
        .result()
    )
    without = SimulatorBackend(method=method).run(_bell(), shots=128, seed=11).result()
    assert with_b.get_counts() == without.get_counts()


def test_barrier_does_not_change_state():
    p = Program(2)
    p.add(ops.H, 0)
    p.add(ops.Barrier, (0, 1))
    p.add(ops.CX, (0, 1))
    sv = (
        SimulatorBackend()
        .run(p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(sv, np.array([1, 0, 0, 1]) / np.sqrt(2))


def test_barrier_is_preserved_in_program_operations():
    # The compiler-facing contract: the frontend keeps barriers verbatim.
    p = _bell(with_barriers=True)
    barriers = [
        step
        for step in p.operations
        if hasattr(step, "operation") and isinstance(step.operation, ops.BarrierGate)
    ]
    assert len(barriers) == 2
    assert barriers[0].targets != barriers[1].targets


# --- resource-layout / engine-allocation hook split --------------------------


def test_no_resolve_layout_method_remains():
    # resolve_layout used to carry both a physical and an engine-index
    # meaning at once; it must be gone, replaced by the two distinct hooks.
    assert not hasattr(SimulatorBackend, "resolve_layout")


def test_run_resolves_resource_layout_and_engine_index_allocation_exactly_once(
    monkeypatch,
):
    backend = SimulatorBackend()
    calls = {"resource_layout": 0, "engine": 0}
    original_resource_layout = backend._resolve_resource_layout
    original_allocate_engine_indices = backend._allocate_engine_indices

    def counting_resource_layout(program):
        calls["resource_layout"] += 1
        return original_resource_layout(program)

    def counting_allocate_engine_indices(program):
        calls["engine"] += 1
        return original_allocate_engine_indices(program)

    monkeypatch.setattr(backend, "_resolve_resource_layout", counting_resource_layout)
    monkeypatch.setattr(
        backend, "_allocate_engine_indices", counting_allocate_engine_indices
    )

    p = Program(1)
    p.add(ops.H, 0)
    backend.run(p, result_config={"counts": False, "statevector": True})

    assert calls == {"resource_layout": 1, "engine": 1}


def test_resource_layout_failure_raises_directly_not_as_a_failed_job():
    # A validation failure in _resolve_resource_layout must propagate
    # directly from run(), never be captured into Job.failed().
    class _ExplodingResourceLayoutBackend(SimulatorBackend):
        def _resolve_resource_layout(self, program):
            raise BackendValidationError("resource layout boom")

    backend = _ExplodingResourceLayoutBackend()
    p = Program(1)
    p.add(ops.H, 0)
    with pytest.raises(BackendValidationError, match="resource layout boom"):
        backend.run(p)


def test_engine_index_allocation_failure_raises_directly_not_as_a_failed_job(
    monkeypatch,
):
    # Same guarantee for _allocate_engine_indices. Injected via monkeypatch,
    # not a subclass override: no real backend overrides this hook (unlike
    # _resolve_resource_layout above), so a subclass would misrepresent it
    # as a supported extension point.
    def exploding(program):
        raise BackendValidationError("engine allocation boom")

    backend = SimulatorBackend()
    monkeypatch.setattr(backend, "_allocate_engine_indices", exploding)
    p = Program(1)
    p.add(ops.H, 0)
    with pytest.raises(BackendValidationError, match="engine allocation boom"):
        backend.run(p)
