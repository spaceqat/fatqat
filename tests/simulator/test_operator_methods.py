"""Tests the operator simulation methods: ``unitary`` and ``superop``.

These methods compute the program's *map* rather than a state under it, so
the assertions here fall into three groups:

- against an independent reference: a full-space embedding built from
  ``numpy.kron`` alone, with no engine machinery in the loop.
- against the state methods, which is the strongest available check and needs
  no external convention: column 0 of the unitary is the statevector, and the
  super-operator applied to ``vec(rho)`` is the density matrix.
- across runtimes: ``numpy`` and ``numba`` must agree elementwise.

Every numeric test runs on both runtimes through the ``runtime`` fixture;
the ``numba`` axis exercises the package's default runtime dependency.
"""

import numpy as np
import pytest

import fatqat as fq
from fatqat import operations as ops
from fatqat.errors import BackendValidationError, ResultFieldUnavailableError
from fatqat.program import Program
from fatqat.simulator import Simulator

_ATOL = 1e-12


@pytest.fixture(params=["numpy", "numba"], name="runtime")
def _runtime(request):
    """Return each supported execution runtime."""
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


# --- reference embedding (numpy.kron only, no engine machinery) ---


def _embed(matrix: np.ndarray, targets: tuple[int, ...], dims: tuple[int, ...]):
    """Embed a local matrix into the full space, little-endian.

    Built by permuting a ``kron`` of the local matrix with identities, which
    shares no code with either engine: ``targets[0]`` is the local matrix's
    most-significant index digit and subsystem 0 is the flat index's least
    significant digit.
    """
    n = len(dims)
    size = int(np.prod(dims))
    rest = [q for q in range(n) if q not in targets]
    # kron order is most- to least-significant, i.e. reversed subsystem order.
    order = list(targets) + rest
    kron = matrix
    for q in rest:
        kron = np.kron(kron, np.eye(dims[q]))
    # `kron`'s index digits run in `order`; rebuild the flat index in
    # descending subsystem order (n-1 ... 0), which is the flat convention.
    axes = [order.index(q) for q in range(n - 1, -1, -1)]
    tensor = kron.reshape(tuple(dims[q] for q in order) * 2)
    tensor = np.transpose(tensor, axes + [n + a for a in axes])
    return tensor.reshape(size, size)


def _reference_unitary(steps, dims):
    """Product of embedded local matrices, applied left to right."""
    total = np.eye(int(np.prod(dims)), dtype=complex)
    for matrix, targets in steps:
        total = _embed(np.asarray(matrix, dtype=complex), targets, dims) @ total
    return total


# --- programs used across the tests ---


def _ghz_program(n: int = 3) -> Program:
    program = Program(n)
    program.add(ops.H, 0)
    for q in range(n - 1):
        program.add(ops.CX, (q, q + 1))
    return program


def _mixed_program() -> Program:
    """A program covering rotations, a diagonal, and a reversed-target CX."""
    program = Program(3)
    program.add(ops.H, 0)
    program.add(ops.RX(0.7), 1)
    program.add(ops.T, 2)
    program.add(ops.CX, (2, 0))
    program.add(ops.CPhase(0.3), (1, 2))
    return program


def _qutrit_program() -> Program:
    register = fq.QuantumRegister(2, dim=3)
    program = Program([register])
    program.add(ops.Fourier, 0)
    program.add(ops.Shift(1), 1)
    program.add(ops.Sum, (0, 1))
    return program


def _noisy_program() -> Program:
    program = Program(2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    return program


def _depolarizing_noise() -> fq.NoiseModel:
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.15), operation=ops.H)
    return noise


def _unitary_of(program, runtime="numpy", **kwargs):
    return Simulator("unitary", runtime=runtime, **kwargs).run(program).result()


def _superop_of(program, runtime="numpy", **kwargs):
    return Simulator("superop", runtime=runtime, **kwargs).run(program).result()


# --- unitary: shape, defaults, and the independent reference ---


def test_unitary_is_returned_by_default_and_is_unitary(runtime):
    unitary = _unitary_of(_ghz_program(), runtime).get_unitary()
    assert unitary.shape == (8, 8)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(8), atol=_ATOL)


def test_unitary_matches_an_independent_kron_embedding(runtime):
    """Every reference matrix is spelled out here, so nothing is shared with
    the implementation map the run itself used."""
    unitary = _unitary_of(_mixed_program(), runtime).get_unitary()
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    rx = np.array(
        [
            [np.cos(0.35), -1j * np.sin(0.35)],
            [-1j * np.sin(0.35), np.cos(0.35)],
        ],
        dtype=complex,
    )
    t = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
    cx = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    cphase = np.diag([1, 1, 1, np.exp(1j * 0.3)]).astype(complex)
    expected = _reference_unitary(
        [
            (hadamard, (0,)),
            (rx, (1,)),
            (t, (2,)),
            (cx, (2, 0)),
            (cphase, (1, 2)),
        ],
        (2, 2, 2),
    )
    assert np.allclose(unitary, expected, atol=_ATOL)


def test_unitary_first_column_is_the_statevector(runtime):
    program = _mixed_program()
    unitary = _unitary_of(program, runtime).get_unitary()
    statevector = (
        Simulator("SV", runtime=runtime)
        .run(program, result_config={"final_state": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(unitary[:, 0], statevector, atol=_ATOL)


def test_unitary_handles_qutrit_registers(runtime):
    unitary = _unitary_of(_qutrit_program(), runtime).get_unitary()
    assert unitary.shape == (9, 9)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(9), atol=_ATOL)
    statevector = (
        Simulator("SV", runtime=runtime)
        .run(_qutrit_program(), result_config={"final_state": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(unitary[:, 0], statevector, atol=_ATOL)


def test_unitary_skips_barriers(runtime):
    with_barrier = _ghz_program()
    with_barrier.add(ops.Barrier, (0, 1, 2))
    assert np.allclose(
        _unitary_of(with_barrier, runtime).get_unitary(),
        _unitary_of(_ghz_program(), runtime).get_unitary(),
        atol=_ATOL,
    )


# --- superop: the vectorization contract and the density-matrix cross-check ---


def test_superop_of_a_unitary_program_is_kron_u_conj_u(runtime):
    program = _mixed_program()
    unitary = _unitary_of(program, runtime).get_unitary()
    superop = _superop_of(program, runtime).get_superop()
    assert superop.shape == (64, 64)
    assert np.allclose(superop, np.kron(unitary, unitary.conj()), atol=_ATOL)


@pytest.mark.parametrize("with_reset", [False, True])
def test_superop_reproduces_the_density_matrix_run(runtime, with_reset):
    program = _noisy_program()
    if with_reset:
        program.add(ops.Reset, 0)
    noise = _depolarizing_noise()
    superop = _superop_of(program, runtime, noise=noise).get_superop()
    density_matrix = (
        Simulator("DM", runtime=runtime, noise=noise)
        .run(program, result_config={"final_state": True})
        .result()
        .get_density_matrix()
    )
    rho_zero = np.zeros((4, 4), dtype=complex)
    rho_zero[0, 0] = 1.0
    evolved = (superop @ rho_zero.reshape(-1)).reshape(4, 4)
    assert np.allclose(evolved, density_matrix, atol=_ATOL)


def test_superop_acts_correctly_on_an_arbitrary_input_state(runtime):
    """The super-operator is the whole map, not just its action on |0..0>."""
    program = _noisy_program()
    program.add(ops.Reset, 1)
    superop = _superop_of(program, runtime, noise=_depolarizing_noise()).get_superop()

    generator = np.random.default_rng(11)
    root = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
    rho = root @ root.conj().T
    rho /= np.trace(rho).real
    evolved = (superop @ rho.reshape(-1)).reshape(4, 4)

    assert np.allclose(evolved, evolved.conj().T, atol=_ATOL)  # still Hermitian
    assert np.isclose(np.trace(evolved).real, 1.0, atol=_ATOL)  # still a state
    assert np.min(np.linalg.eigvalsh(evolved)) > -_ATOL  # still positive
    # Linearity pins the value: rho decomposes over the basis matrices whose
    # images are the super-operator's own columns.
    columns = sum(
        rho[i, j] * superop[:, i * 4 + j].reshape(4, 4)
        for i in range(4)
        for j in range(4)
    )
    assert np.allclose(evolved, columns, atol=_ATOL)


def test_superop_reset_is_the_partial_trace_channel(runtime):
    program = Program(2)
    program.add(ops.Reset, 0)
    superop = _superop_of(program, runtime).get_superop()

    generator = np.random.default_rng(5)
    root = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
    rho = root @ root.conj().T
    rho /= np.trace(rho).real
    evolved = (superop @ rho.reshape(-1)).reshape(4, 4)

    # Subsystem 0 is the least-significant digit, so tracing it out leaves the
    # 2x2 block indexed by subsystem 1, re-prepared in |0>.
    traced = rho.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)
    expected = np.zeros((4, 4), dtype=complex)
    expected[np.ix_([0, 2], [0, 2])] = traced
    assert np.allclose(evolved, expected, atol=_ATOL)


def test_superop_handles_qutrit_registers(runtime):
    program = _qutrit_program()
    superop = _superop_of(program, runtime).get_superop()
    unitary = _unitary_of(program, runtime).get_unitary()
    assert superop.shape == (81, 81)
    assert np.allclose(superop, np.kron(unitary, unitary.conj()), atol=_ATOL)


# --- runtime parity ---


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_numpy_and_numba_runtimes_agree(method):
    pytest.importorskip("numba")
    program = _mixed_program()
    program.add(ops.Swap, (0, 2))
    noise = _depolarizing_noise() if method == "superop" else None
    field = f"get_{method}"

    def run(runtime):
        backend = Simulator(method, runtime=runtime, noise=noise)
        return getattr(backend.run(program).result(), field)()

    assert np.allclose(run("numpy"), run("numba"), atol=_ATOL)


# --- the fused Numba path ---


def _numba_engines():
    pytest.importorskip("numba")
    from fatqat.simulator._engine import nb

    return nb


@pytest.fixture(name="no_fusion")
def _no_fusion(monkeypatch):
    """Suppress gate fusion, leaving the column kernel to run the plan as lowered."""
    nb = _numba_engines()
    monkeypatch.setattr(nb, "_fuse_operator_payloads", lambda payloads, dims: payloads)
    return nb


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_column_kernel_matches_the_per_gate_kernels_bit_for_bit(method, no_fusion):
    """Unfused, the column kernel keeps the per-step accumulation order exactly."""
    nb = no_fusion
    program = _mixed_program()
    program.add(ops.Swap, (0, 2))
    noise = _depolarizing_noise() if method == "superop" else None
    backend = Simulator(method, runtime="numba", noise=noise)
    fused = getattr(backend.run(program).result(), f"get_{method}")()

    # Replay the same plan through the single-application fallback.
    engine = backend._engine
    plan, _ = backend._lower_program(program)
    system_dims = tuple(
        register.dim
        for register in program.quantum_registers
        for _ in range(register.size)
    )
    n_clbits = sum(register.size for register in program.classical_registers)
    engine.initialize(system_dims, n_clbits)
    generator = np.random.default_rng(0)
    for step in nb._fuse_gate_channels(plan) if method == "superop" else plan:
        if isinstance(step, nb.ApplyMatrixStep):
            engine.apply(step)
        elif isinstance(step, nb.ApplyChannelStep):
            engine.apply_channel(step, generator)
        else:
            engine.reset_subsystems(step.reset_indices, generator)

    assert np.array_equal(fused, engine.state)


# --- gate fusion ---


def _fusion_programs():
    """Programs whose structure exercises each fusion path."""
    phases = Program(4)  # QFT-style phase cascade: all diagonal
    for target in range(1, 4):
        phases.add(ops.CPhase(0.3 * target), (target, 0))

    permutations = Program(4)  # CX ladder: all permutation
    for target in range(3):
        permutations.add(ops.CX, (target, target + 1))

    rotations = Program(3)  # dense+diagonal pairs per qubit
    for target in range(3):
        rotations.add(ops.RY(0.2 * target + 0.1), target)
        rotations.add(ops.RZ(0.3 * target + 0.2), target)

    return {"phases": phases, "permutations": permutations, "rotations": rotations}


@pytest.fixture(name="force_fusion")
def _force_fusion(monkeypatch):
    """Fuse regardless of operator size, which every fast test program is below."""
    nb = _numba_engines()
    monkeypatch.setattr(nb, "_MIN_SIZE_TO_FUSE", 0)
    return nb


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_gate_fusion_preserves_the_operator(method, force_fusion):
    """Fusion preserves the public operator result within numeric tolerance."""
    del force_fusion  # fixture lowers the threshold so this small program fuses
    program = _fusion_programs()["rotations"]
    fused = getattr(
        Simulator(method, runtime="numba")
        .run(program, simulation_config={"fusion": True})
        .result(),
        f"get_{method}",
    )()
    unfused = getattr(
        Simulator(method, runtime="numba")
        .run(program, simulation_config={"fusion": False})
        .result(),
        f"get_{method}",
    )()
    assert np.allclose(fused, unfused, atol=_ATOL)


@pytest.mark.parametrize(
    "name, expected_steps",
    [("phases", 1), ("permutations", 1), ("rotations", 1)],
)
def test_gate_fusion_collapses_whole_runs(name, expected_steps):
    """Without this the equivalence tests above could pass by fusing nothing."""
    nb = _numba_engines()
    backend = Simulator("unitary", runtime="numba")
    program = _fusion_programs()[name]
    plan, _ = backend._lower_program(program)
    backend.run(program).result()  # sizes the engine and fills its caches
    engine = backend._engine
    payloads = engine._operator_payloads(plan)
    fused = nb._fuse_operator_payloads(payloads, engine._operator_row_dims())
    assert len(payloads) > len(fused)
    assert len(fused) == expected_steps


def test_gate_fusion_keeps_a_diagonal_run_diagonal():
    """A run of phase gates stays on the in-place diagonal kernel, at any width."""
    nb = _numba_engines()
    backend = Simulator("unitary", runtime="numba")
    program = Program(4)
    for target in range(4):
        program.add(ops.RZ(0.2 * target + 0.1), target)
    plan, _ = backend._lower_program(program)
    backend.run(program).result()
    engine = backend._engine
    fused = nb._fuse_operator_payloads(
        engine._operator_payloads(plan), engine._operator_row_dims()
    )
    assert len(fused) == 1
    assert fused[0][1] == nb._DIAGONAL
    assert fused[0][5] == (0, 1, 2, 3)  # one pass over all four subsystems


def test_gate_fusion_stops_widening_a_dense_run():
    """The cost model caps dense growth."""
    nb = _numba_engines()
    backend = Simulator("unitary", runtime="numba")
    program = Program(8)
    for target in range(8):
        program.add(ops.RY(0.2 * target + 0.1), target)  # dense, all distinct targets
    plan, _ = backend._lower_program(program)
    backend.run(program).result()
    engine = backend._engine
    fused = nb._fuse_operator_payloads(
        engine._operator_payloads(plan), engine._operator_row_dims()
    )
    assert len(fused) > 1, "8 distinct dense targets must not collapse to one pass"
    for payload in fused:
        assert len(payload[5]) <= 4, "dense blocks stayed within the cost model"


def test_small_operators_skip_fusion_entirely():
    """Below the size gate the plan reaches the kernel exactly as lowered."""
    nb = _numba_engines()
    backend = Simulator("unitary", runtime="numba")
    program = Program(3)
    for target in range(3):
        program.add(ops.RZ(0.2 * target + 0.1), target)
    plan, _ = backend._lower_program(program)
    backend.run(program).result()
    engine = backend._engine
    assert engine.state.size < nb._MIN_SIZE_TO_FUSE
    # Fusion would have collapsed this run, so the gate is what leaves it alone.
    payloads = engine._operator_payloads(plan)
    assert len(nb._fuse_operator_payloads(payloads, engine._operator_row_dims())) == 1
    assert len(payloads) == 3


def test_gate_fusion_leaves_a_sparse_step_alone():
    """A CSR super-operator keeps its sparse walk instead of joining a dense block."""
    nb = _numba_engines()
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.2), operation=ops.H)
    backend = Simulator("superop", runtime="numba", noise=noise)
    program = Program(2)
    program.add(ops.RZ(0.4), 0)
    program.add(ops.X, 1)
    program.add(ops.H, 0)
    program.add(ops.Reset, 1)
    plan, _ = backend._lower_program(program)
    backend.run(program).result()
    engine = backend._engine
    payloads = engine._operator_payloads(plan)
    assert any(payload[4] is not None for payload in payloads), "expected a CSR step"
    fused = nb._fuse_operator_payloads(payloads, engine._operator_row_dims())
    assert any(payload[4] is not None for payload in fused)


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_serial_and_threaded_numba_give_identical_operators(method):
    _numba_engines()
    program = _mixed_program()
    backend = Simulator(method, runtime="numba")
    read = f"get_{method}"

    def run(kernel_parallelism):
        return getattr(
            backend.run(
                program,
                simulation_config={
                    "shot_parallelism": "serial",
                    "kernel_parallelism": kernel_parallelism,
                    "max_workers": 1 if kernel_parallelism == "serial" else 2,
                },
            ).result(),
            read,
        )()

    assert np.array_equal(run("threads"), run("serial"))


@pytest.mark.parametrize("runtime", ["numpy", "numba"], indirect=True)
@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_empty_program_returns_the_identity(method, runtime):
    """No steps at all: the fused path has nothing to pack and must not run."""
    size = 4
    operator = getattr(
        Simulator(method, runtime=runtime).run(Program(2)).result(), f"get_{method}"
    )()
    expected = size if method == "unitary" else size * size
    assert np.array_equal(operator, np.eye(expected, dtype=complex))


def test_fused_superop_covers_every_structure_code():
    """Diagonal, permutation, dense, and CSR-sparse steps in one packed plan."""
    nb = _numba_engines()
    program = Program(2)
    program.add(ops.RZ(0.4), 0)  # diagonal super-operator
    program.add(ops.X, 1)  # permutation super-operator
    program.add(ops.H, 0)  # dense super-operator
    program.add(ops.Reset, 1)  # Kraus channel
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.2), operation=ops.H)  # sparse CSR

    backend = Simulator("superop", runtime="numba", noise=noise)
    plan, _ = backend._lower_program(program)
    backend.run(program).result()  # populates the engine's resolution caches
    codes = {
        payload[1] if payload[4] is None else nb._SPARSE
        for payload in backend._engine._operator_payloads(plan)
    }
    assert {nb._DENSE, nb._DIAGONAL, nb._PERMUTATION, nb._SPARSE} <= codes

    assert np.allclose(
        backend.run(program).result().get_superop(),
        Simulator("superop", noise=noise).run(program).result().get_superop(),
        atol=_ATOL,
    )


# --- result plumbing ---


@pytest.mark.parametrize(
    "method, produced, absent",
    [
        ("unitary", "unitary", ["superop", "statevector", "density_matrix"]),
        ("superop", "superop", ["unitary", "statevector", "density_matrix"]),
    ],
)
def test_only_the_method_native_field_is_available(method, produced, absent):
    result = Simulator(method).run(_ghz_program(2)).result()
    assert result.available_data == frozenset({produced})
    for name in absent:
        with pytest.raises(ResultFieldUnavailableError):
            getattr(result, f"get_{name}")()
    with pytest.raises(ResultFieldUnavailableError):
        result.get_counts()


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_metadata_echoes_the_operator_method(method):
    result = Simulator(method).run(_ghz_program(2)).result()
    assert result.metadata["method"] == method
    assert result.metadata["runtime"] == "numba"


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_final_state_false_suppresses_the_operator(method):
    result = (
        Simulator(method).run(_ghz_program(2), result_config={"final_state": False})
    ).result()
    assert result.available_data == frozenset()


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_shots_are_ignored(method):
    """An operator is one deterministic pass, so shots cannot change it."""
    program = _ghz_program(2)
    read = f"get_{method}"
    once = getattr(Simulator(method).run(program, shots=1).result(), read)()
    many = getattr(Simulator(method).run(program, shots=4096).result(), read)()
    assert np.allclose(once, many, atol=_ATOL)


@pytest.mark.parametrize("runtime", ["numpy", "numba"], indirect=True)
def test_backend_instance_is_reusable_across_system_sizes(runtime):
    backend = Simulator("unitary", runtime=runtime)
    read = "get_unitary"
    two = getattr(backend.run(_ghz_program(2)).result(), read)()
    three = getattr(backend.run(_ghz_program(3)).result(), read)()
    again = getattr(backend.run(_ghz_program(2)).result(), read)()
    assert two.shape != three.shape
    assert np.allclose(two, again, atol=_ATOL)


# --- validation: what the operator methods refuse ---


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_measurement_is_rejected(method):
    program = Program(2, 2)
    program.add(ops.H, 0)
    program.measure((0, 1), (0, 1))
    with pytest.raises(BackendValidationError, match="cannot execute a measurement"):
        Simulator(method).run(program)


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_counts_request_is_rejected(method):
    with pytest.raises(BackendValidationError, match="cannot produce counts"):
        Simulator(method).run(_ghz_program(2), result_config={"counts": True})


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_feedforward_condition_is_rejected(method):
    program = Program(1, 1)
    program.add(ops.X, 0, condition=(program.classical_registers[0][0], 0))
    with pytest.raises(BackendValidationError, match="feedforward condition"):
        Simulator(method).run(program)


def test_unitary_rejects_reset():
    program = _ghz_program(2)
    program.add(ops.Reset, 0)
    with pytest.raises(BackendValidationError, match="cannot execute reset"):
        Simulator("unitary").run(program)


def test_unitary_rejects_channel_noise():
    with pytest.raises(BackendValidationError, match="cannot execute channel noise"):
        Simulator("unitary", noise=_depolarizing_noise()).run(_noisy_program())


def test_unitary_accepts_a_noise_model_that_never_fires():
    """Support is decided by the lowered plan, not by the model being present."""
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.1), operation=ops.Y)
    unitary = _unitary_of(_noisy_program(), noise=noise).get_unitary()
    assert np.allclose(unitary, _unitary_of(_noisy_program()).get_unitary(), atol=_ATOL)


def test_validation_errors_raise_from_run_not_from_the_job():
    program = Program(1, 1)
    program.measure(0, 0)
    with pytest.raises(BackendValidationError):
        Simulator("unitary").run(program)


@pytest.mark.parametrize("method", ["UNITARY", "SuperOp"])
def test_method_names_are_case_insensitive(method):
    assert Simulator(method)._state_field == method.lower()
