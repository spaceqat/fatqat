"""Starting a run from a given state instead of the computational zero.

The interesting failures here are silent ones: a state that reaches the first
shot but not the rest, or the parent process but not a worker, produces wrong
numbers rather than an error. Those paths get their own tests.
"""

import importlib.util

import numpy as np
import pytest

import fatqat as fq
from fatqat import operations as ops
from fatqat.errors import BackendValidationError
from fatqat.simulator import Simulator

_STATE_ONLY = {"counts": False, "final_state": True}

# numba is optional, so the parametrization carries whichever runtimes exist.
# It matters here more than in most places: the two take different routes to a
# per-shot buffer, and a divergence between them is a wrong number, not an
# error.
_RUNTIMES = ["numpy"]
if importlib.util.find_spec("numba") is not None:
    _RUNTIMES.append("numba")


def _cx_program():
    """``CX`` with qubit 0 controlling qubit 1, so |01> becomes |11>."""
    program = fq.Program(2)
    program.add(ops.CX, (0, 1))
    return program


def _dynamic_program():
    """Measurement plus feedforward, which forces per-shot replay."""
    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))
    program.measure(1, 1)
    return program


# --- the state is used ---------------------------------------------------


def test_statevector_run_starts_from_the_given_state():
    result = (
        Simulator(method="SV")
        .run(
            _cx_program(),
            shots=0,
            initial_state=[0, 1, 0, 0],
            result_config=_STATE_ONLY,
        )
        .result()
    )

    assert result.get_statevector() == pytest.approx([0, 0, 0, 1], abs=1e-12)


def test_density_matrix_run_accepts_a_ket_as_the_pure_state():
    # The same array that starts a statevector run starts a density-matrix one,
    # so comparing the two representations needs no second form.
    result = (
        Simulator(method="DM")
        .run(
            _cx_program(),
            shots=0,
            initial_state=[0, 1, 0, 0],
            result_config=_STATE_ONLY,
        )
        .result()
    )

    assert result.get_density_matrix().diagonal() == pytest.approx(
        [0, 0, 0, 1], abs=1e-12
    )


def test_density_matrix_run_accepts_a_mixed_state():
    rho = np.diag([0.0, 0.3, 0.7, 0.0])

    result = (
        Simulator(method="DM")
        .run(_cx_program(), shots=0, initial_state=rho, result_config=_STATE_ONLY)
        .result()
    )

    # CX maps |01> -> |11> and leaves |10> alone, so the weights swap places.
    assert result.get_density_matrix().diagonal() == pytest.approx(
        [0.0, 0.0, 0.7, 0.3], abs=1e-12
    )


def test_a_second_run_does_not_inherit_the_first_one_s_state():
    """The engine is reused across runs of the same shape, so it must be reset.

    Held on the engine and never cleared, a state given to one run would still
    be there for the next - which would look like a correct run of the wrong
    experiment.
    """
    backend = Simulator(method="SV")

    backend.run(
        _cx_program(), shots=0, result_config=_STATE_ONLY, initial_state=[0, 1, 0, 0]
    ).result()
    second = backend.run(_cx_program(), shots=0, result_config=_STATE_ONLY).result()

    assert second.get_statevector() == pytest.approx([1, 0, 0, 0], abs=1e-12)


def test_qudit_registers_are_supported():
    # The state spans the product of the subsystem dimensions, so nothing about
    # this is qubit-specific.
    program = fq.Program([fq.QuantumRegister(2, dim=3)])
    program.add(ops.Sum, (0, 1))
    start = np.zeros(9)
    start[1] = 1.0

    result = (
        Simulator(method="SV")
        .run(program, shots=0, initial_state=start, result_config=_STATE_ONLY)
        .result()
    )

    assert result.get_statevector().shape == (9,)
    assert result.get_statevector().sum() == pytest.approx(1.0, abs=1e-12)


# --- only the shape is checked -------------------------------------------


def test_an_unnormalized_state_is_accepted_and_stays_unnormalized():
    # Deliberately not rejected: nothing downstream relies on normalization,
    # and the exported state should report what the arithmetic actually gave.
    result = (
        Simulator(method="SV")
        .run(
            _cx_program(),
            shots=0,
            initial_state=[0, 2, 0, 0],
            result_config=_STATE_ONLY,
        )
        .result()
    )

    assert result.get_statevector() == pytest.approx([0, 0, 0, 2], abs=1e-12)


def test_counts_still_come_from_a_normalized_distribution():
    # The other half of the same decision: sampling normalizes defensively, so
    # an unnormalized start yields a valid distribution rather than an error.
    program = fq.Program(2, 2)
    program.measure((0, 1), (0, 1))

    counts = (
        Simulator(method="SV")
        .run(
            program,
            shots=100,
            initial_state=[0, 3, 0, 0],
            simulation_config={"seed": 1},
        )
        .result()
        .get_counts()
    )

    assert counts == {"01": 100}


def test_a_non_hermitian_matrix_is_accepted():
    # There is no Hermitian-only optimization here, so refusing one would only
    # block arithmetic we have no reason to object to.
    rho = np.array(
        [[0, 0, 0, 0], [0, 0.5, 0.2j, 0], [0, 0.9j, 0.5, 0], [0, 0, 0, 0]],
        dtype=complex,
    )

    result = (
        Simulator(method="DM")
        .run(_cx_program(), shots=0, initial_state=rho, result_config=_STATE_ONLY)
        .result()
    )

    assert result.get_density_matrix().shape == (4, 4)


@pytest.mark.parametrize(
    "method, bad",
    [
        ("SV", [1, 0, 0]),
        ("SV", np.eye(4)),
        ("DM", np.eye(3)),
        ("DM", [1, 0, 0]),
    ],
)
def test_a_wrong_shape_is_rejected(method, bad):
    with pytest.raises(BackendValidationError, match="initial_state has shape"):
        Simulator(method=method).run(
            _cx_program(), shots=0, initial_state=bad, result_config=_STATE_ONLY
        )


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_operator_methods_reject_an_initial_state(method):
    # They compute the program's map rather than a state evolving under it, so
    # there is nothing for a starting state to be.
    with pytest.raises(BackendValidationError, match="not meaningful"):
        Simulator(method=method).run(
            _cx_program(),
            shots=0,
            initial_state=[1, 0, 0, 0],
            result_config=_STATE_ONLY,
        )


# --- the paths where a mistake would be silent ---------------------------


@pytest.mark.parametrize("runtime", _RUNTIMES)
def test_every_shot_of_a_dynamic_run_starts_from_the_state(runtime):
    # Per-shot replay re-initializes between shots. If the state were applied
    # once rather than held, only the first shot would start from it and the
    # counts would quietly be a mixture of two different experiments.
    #
    # Both runtimes, because they take different routes: numba compiles the
    # whole trajectory into one kernel that builds its own per-shot buffer, so
    # a state honoured on the NumPy path can be ignored on that one - which is
    # a wrong answer, not an error.
    counts = (
        Simulator(method="SV", runtime=runtime)
        .run(
            _dynamic_program(),
            shots=400,
            initial_state=[0, 0, 1, 0],
            simulation_config={"seed": 7, "parallel_mode": "serial"},
        )
        .result()
        .get_counts()
    )

    # Qubit 1 starts set; the feedforward X clears it exactly when clbit 0 read
    # 1. Starting from zero instead would give '00'/'11'.
    assert set(counts) == {"10", "01"}


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("parallel_mode", ["multiprocessing", "loky"])
def test_parallel_workers_agree_with_serial(parallel_mode, runtime, monkeypatch):
    # A worker builds its own engine. Without the state travelling with the
    # work it would start from zero while the serial path did not, and the two
    # would disagree silently.
    start = [0, 0, 1, 0]
    program = _dynamic_program()
    if runtime == "numba":
        # This program is normally handled by Numba's whole-shot kernel and
        # therefore never creates workers. Force the supported fallback so the
        # Numba engine class, plan, and initial state are actually serialized
        # through the worker path this test promises to cover.
        from fatqat.simulator._engine import nb

        monkeypatch.setattr(nb, "_plan_compilable", lambda plan: False)

    serial = (
        Simulator(method="SV", runtime=runtime)
        .run(
            program,
            shots=400,
            initial_state=start,
            simulation_config={"seed": 7, "max_workers": 1},
        )
        .result()
        .get_counts()
    )
    parallel = (
        Simulator(method="SV", runtime=runtime)
        .run(
            program,
            shots=400,
            initial_state=start,
            simulation_config={
                "seed": 7,
                "max_workers": 2,
                "parallel_mode": parallel_mode,
            },
        )
        .result()
        .get_counts()
    )

    assert parallel == serial


def test_the_worker_batch_is_handed_the_state():
    # Pin the wiring rather than the effect. Intercepting the executor - rather
    # than the worker function, which has to stay picklable - captures exactly
    # what would have been shipped, so dropping the state from the dispatch
    # fails here instead of only surfacing as a numeric disagreement.
    from fatqat.simulator._engine import parallel

    seen = {}

    class _RecordingExecutor:
        def __init__(self, max_workers=None, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, function, *iterables):
            # zip stops at the finite `batches` iterable, so the repeat()s
            # bound themselves.
            calls = list(zip(*iterables))
            seen["last_argument"] = calls[0][-1]
            return [function(*arguments) for arguments in calls]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(parallel, "ProcessPoolExecutor", _RecordingExecutor)
        Simulator(method="SV", runtime="numpy").run(
            _dynamic_program(),
            shots=40,
            initial_state=[0, 0, 1, 0],
            simulation_config={
                "seed": 3,
                "max_workers": 2,
                "parallel_mode": "multiprocessing",
            },
        ).result().get_counts()

    assert seen["last_argument"] == pytest.approx([0, 0, 1, 0])


def test_a_later_run_does_not_inherit_an_earlier_initial_state():
    # The engine is reused whenever the system shape repeats, so a state bound
    # to that cache would leak into the next run of the same shape.
    backend = Simulator(method="SV")
    program = _cx_program()

    backend.run(
        program, shots=0, initial_state=[0, 1, 0, 0], result_config=_STATE_ONLY
    ).result()
    second = backend.run(program, shots=0, result_config=_STATE_ONLY).result()

    assert second.get_statevector() == pytest.approx([1, 0, 0, 0], abs=1e-12)


def test_the_callers_array_is_not_evolved_in_place():
    start = np.array([0, 1, 0, 0], dtype=complex)
    untouched = start.copy()

    Simulator(method="SV").run(
        _cx_program(), shots=0, initial_state=start, result_config=_STATE_ONLY
    ).result()

    assert np.array_equal(start, untouched)


def test_the_engine_releases_the_initial_state_after_the_eager_run():
    backend = Simulator(method="SV")
    backend.run(
        _cx_program(), shots=0, initial_state=[0, 1, 0, 0], result_config=_STATE_ONLY
    ).result()

    assert backend._engine.initial_state is None
