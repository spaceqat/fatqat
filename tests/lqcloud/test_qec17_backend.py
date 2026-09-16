"""Focused, network-free tests for the optional LQCloud QEC17 adapter."""

from __future__ import annotations

import sys
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.lqcloud import LQCloudQEC17Backend, program_to_qec17_circuit

_FIXED_NATIVE_CASES = [
    (ops.I, "id"),
    (ops.H, "h"),
    (ops.HY, "hy"),
    (ops.X, "x"),
    (ops.MX, "mx"),
    (ops.Y, "y"),
    (ops.MY, "my"),
    (ops.Z, "z"),
    (ops.MZ, "mz"),
    (ops.XHalf, "xhalf"),
    (ops.MXHalf, "mxhalf"),
    (ops.YHalf, "yhalf"),
    (ops.MYHalf, "myhalf"),
    (ops.XYHalf, "xyhalf"),
    (ops.MXYHalf, "mxyhalf"),
    (ops.MXMYHalf, "mxmyhalf"),
    (ops.XMYHalf, "xmyhalf"),
    (ops.S, "s"),
    (ops.Sdg, "sdg"),
    (ops.T, "t"),
]


class _RecordingCircuit:
    """Small stand-in exposing exactly the SDK methods used by the adapter."""

    def __init__(self, n_qubits):
        self.num_qubits = n_qubits
        self.calls = []

    def _record(self, name, *args):
        self.calls.append((name, args))

    def __getattr__(self, name):
        return lambda *args: self._record(name, *args)


class _RecordingCloudBackend:
    def __init__(self):
        self.calls = []
        self.job = object()

    def run(self, circuit, *, shots, **options):
        self.calls.append((circuit, shots, options))
        return self.job


class _RecordingProvider:
    instances = []

    def __init__(self, **options):
        self.options = options
        self.get_backend_calls = []
        self.backend = _RecordingCloudBackend()
        type(self).instances.append(self)

    def get_backend(self, name, **options):
        self.get_backend_calls.append((name, options))
        return self.backend


@pytest.fixture(name="fake_lqcloud")
def _fake_lqcloud(monkeypatch):
    _RecordingProvider.instances.clear()
    module = SimpleNamespace(
        QuantumCircuit=_RecordingCircuit,
        LQCloudProvider=_RecordingProvider,
    )
    monkeypatch.setitem(sys.modules, "lqcloud", module)
    return module


@pytest.mark.parametrize(
    ("operation", "method"),
    _FIXED_NATIVE_CASES,
)
def test_converter_translates_each_fixed_native_gate(fake_lqcloud, operation, method):
    program = fq.Program(1)
    program.add(operation, 0)

    circuit = program_to_qec17_circuit(program)

    assert circuit.num_qubits == 1
    assert circuit.calls == [
        (method, (0,)),
        ("barrier", ()),
        ("measure_all", ()),
    ]


def test_converter_translates_parametric_native_gates_and_cz(fake_lqcloud):
    program = fq.Program(2)
    matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    program.add(ops.RZ(0.25), 0)
    program.add(ops.SU2(matrix), 1)
    program.add(ops.CZ, (0, 1))

    circuit = program_to_qec17_circuit(program)

    assert circuit.calls[0] == ("rz", (0.25, 0))
    assert circuit.calls[1][0] == "su2"
    assert circuit.calls[1][1][0] == ops.SU2(matrix).matrix
    assert circuit.calls[1][1][1] == 1
    assert circuit.calls[2:] == [
        ("cz", (0, 1)),
        ("barrier", ()),
        ("measure_all", ()),
    ]


def test_converter_flattens_registers_and_preserves_barriers(fake_lqcloud):
    first = fq.QuantumRegister(2, name="first")
    second = fq.QuantumRegister(1, name="second")
    program = fq.Program([first, second])
    program.add(ops.H, first.all())
    program.add(ops.X, second[0])
    program.add(ops.Barrier, (first[1], second[0]))

    circuit = program_to_qec17_circuit(program)

    assert circuit.calls == [
        ("h", (0,)),
        ("h", (1,)),
        ("x", (2,)),
        ("barrier", (1, 2)),
        ("barrier", ()),
        ("measure_all", ()),
    ]


@pytest.mark.parametrize("operation", [ops.RX(0.5), ops.RY(0.5), ops.SX])
def test_converter_rejects_non_native_or_out_of_scope_operations(
    fake_lqcloud, operation
):
    program = fq.Program(1)
    program.add(operation, 0)

    with pytest.raises(ValueError):
        program_to_qec17_circuit(program)


def test_converter_rejects_reset_because_platform_owns_state_preparation(
    fake_lqcloud,
):
    program = fq.Program(1)
    program.add(ops.Reset, 0)

    with pytest.raises(ValueError, match="state preparation.*LQCloud platform"):
        program_to_qec17_circuit(program)


def test_converter_rejects_explicit_measurement(fake_lqcloud):
    program = fq.Program(1, 1)
    program.measure(0, 0)

    with pytest.raises(ValueError, match="Measurement"):
        program_to_qec17_circuit(program)


def test_converter_rejects_feedforward(fake_lqcloud):
    program = fq.Program(1, 1)
    program.add(ops.X, 0, condition=(0, 1))

    with pytest.raises(ValueError, match="dynamic circuits"):
        program_to_qec17_circuit(program)


def test_converter_rejects_unbound_rz(fake_lqcloud):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(ops.RZ(theta), 0)

    with pytest.raises(ValueError, match="unbound or non-real parameter"):
        program_to_qec17_circuit(program)


@pytest.mark.parametrize(
    "theta",
    [float("nan"), float("inf"), float("-inf"), 10**10_000],
    ids=["nan", "positive-infinity", "negative-infinity", "overflow"],
)
def test_converter_rejects_non_finite_rz(fake_lqcloud, theta):
    program = fq.Program(1)
    program.add(ops.RZ(theta), 0)

    with pytest.raises(ValueError, match="must be finite"):
        program_to_qec17_circuit(program)


@pytest.mark.parametrize(
    "program",
    [
        fq.Program(0),
        fq.Program(18),
        fq.Program([fq.QuantumRegister(1, dim=3)]),
    ],
)
def test_converter_rejects_unsupported_program_shapes(fake_lqcloud, program):
    with pytest.raises(ValueError):
        program_to_qec17_circuit(program)


def test_missing_sdk_has_installation_hint(monkeypatch):
    import fatqat.lqcloud.converter as converter

    real_import_module = converter.import_module

    def missing_sdk(name):
        if name == "lqcloud":
            raise ModuleNotFoundError("No module named 'lqcloud'", name="lqcloud")
        return real_import_module(name)

    monkeypatch.setattr(converter, "import_module", missing_sdk)
    with pytest.raises(ImportError, match="pip install lqcloud==0.4.2"):
        program_to_qec17_circuit(fq.Program(1))


def test_namespace_import_does_not_eagerly_require_sdk():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['lqcloud'] = None; "
            "import fatqat; import fatqat.lqcloud",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_conversion_preserves_program_and_does_not_insert_idle_gates(fake_lqcloud):
    theta = fq.Parameter("theta")
    program = fq.Program(2)
    program.add(ops.RZ(theta), 0)
    program.add(ops.X, 1)
    program.add(ops.I, 0)
    bound = program.assign_parameters({theta: 0.25})
    original = bound._instructions

    first = program_to_qec17_circuit(bound)
    second = program_to_qec17_circuit(bound)

    assert bound._instructions == original
    assert first is not second
    assert (
        first.calls
        == second.calls
        == [
            ("rz", (0.25, 0)),
            ("x", (1,)),
            ("id", (0,)),
            ("barrier", ()),
            ("measure_all", ()),
        ]
    )


def test_real_sdk_serializes_complete_native_program_offline():
    pytest.importorskip("lqcloud")
    serialization = pytest.importorskip("lqcloud.backend.serialization")
    program = fq.Program(2)
    for operation, _method in _FIXED_NATIVE_CASES:
        program.add(operation, 0)
    program.add(ops.RZ(0.25), 0)
    program.add(ops.SU2(np.eye(2)), 0)
    program.add(ops.CZ, (0, 1))

    circuit = program_to_qec17_circuit(program)
    payload = serialization.serialize_circuit(circuit, initial_layout=[6, 11])

    instructions = payload["instructions"]
    expected_names = [
        "reset",
        "reset",
        *(method for _operation, method in _FIXED_NATIVE_CASES),
        "rz",
        "su2",
        "cz",
        "barrier",
        "measure",
        "measure",
    ]
    assert payload["n_qubits"] == payload["n_clbits"] == 2
    assert payload["initial_layout"] == [6, 11]
    assert [instruction["name"] for instruction in instructions] == expected_names
    assert instructions[-2]["qubits"] == [6]
    assert instructions[-2]["clbits"] == [0]
    assert instructions[-1]["qubits"] == [11]
    assert instructions[-1]["clbits"] == [1]


def test_real_sdk_serializes_full_width_register_offline():
    serialization = pytest.importorskip("lqcloud.backend.serialization")
    program = fq.Program(17)
    program.add(ops.H, program.quantum_registers[0].all())

    payload = serialization.serialize_circuit(program_to_qec17_circuit(program))

    assert payload["n_qubits"] == payload["n_clbits"] == 17
    measurements = [
        instruction
        for instruction in payload["instructions"]
        if instruction["name"] == "measure"
    ]
    assert [instruction["qubits"] for instruction in measurements] == [
        [index] for index in range(17)
    ]


def test_ghz_example_prepares_expected_state_and_maps_physical_path(fake_lqcloud):
    example = runpy.run_path(
        str(Path(__file__).with_name("run_qec17_four_qubit_ghz.py"))
    )
    program = example["build_four_qubit_ghz_program"]()
    state = (
        fq.simulator.Simulator()
        .run(program, result_config={"final_state": True, "counts": False})
        .result()
        .get_statevector()
    )
    expected = np.zeros(16, dtype=complex)
    expected[[0, 15]] = 1 / np.sqrt(2)
    np.testing.assert_allclose(state, expected, atol=1e-14)

    backend = LQCloudQEC17Backend("test-key")
    backend.run(program, initial_layout=example["INITIAL_LAYOUT"])
    circuit, _, options = _RecordingProvider.instances[-1].backend.calls[-1]
    assert options["initial_layout"] == [6, 11, 4, 12]
    assert [args for name, args in circuit.calls if name == "cz"] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_backend_constructs_fixed_target_and_returns_native_job(fake_lqcloud):
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    backend = LQCloudQEC17Backend("secret", url="https://cloud.example")

    job = backend.run(
        program,
        shots=25,
        initial_layout=[0, 9],
        result_format="memory",
    )

    provider = _RecordingProvider.instances[-1]
    assert backend.name == "QZ01-surface_code"
    assert backend.num_qubits == 17
    assert provider.options == {
        "api_key": "secret",
        "interactive": False,
        "url": "https://cloud.example",
    }
    assert provider.get_backend_calls == [("QZ01-surface_code", {"verify": False})]
    assert job is provider.backend.job
    circuit, shots, options = provider.backend.calls[-1]
    assert shots == 25
    assert options == {"initial_layout": [0, 9], "result_format": "memory"}
    assert circuit.calls[-2:] == [("barrier", ()), ("measure_all", ())]


def test_backend_omits_identity_layout_from_sdk_call(fake_lqcloud):
    backend = LQCloudQEC17Backend("secret")

    backend.run(fq.Program(1))

    provider = _RecordingProvider.instances[-1]
    _, shots, options = provider.backend.calls[-1]
    assert shots == 1024
    assert options == {}


@pytest.mark.parametrize("shots", [True, 1.5, "10"])
def test_backend_rejects_non_integer_shots(fake_lqcloud, shots):
    backend = LQCloudQEC17Backend("secret")
    with pytest.raises(TypeError, match="shots must be an integer"):
        backend.run(fq.Program(1), shots=shots)


@pytest.mark.parametrize("shots", [0, -1, 50_001])
def test_backend_rejects_out_of_range_shots(fake_lqcloud, shots):
    backend = LQCloudQEC17Backend("secret")
    with pytest.raises(ValueError, match="1..50000"):
        backend.run(fq.Program(1), shots=shots)


@pytest.mark.parametrize(
    ("layout", "error", "message"),
    [
        ({0: 1}, TypeError, "list/tuple"),
        ([0], ValueError, "length"),
        ([0, True], ValueError, "must be an int"),
        ([0, 0], ValueError, "duplicate"),
        ([0, 17], ValueError, "outside"),
    ],
)
def test_backend_rejects_invalid_initial_layout(fake_lqcloud, layout, error, message):
    backend = LQCloudQEC17Backend("secret")
    with pytest.raises(error, match=message):
        backend.run(fq.Program(2), initial_layout=layout)


def test_backend_rejects_non_adjacent_mapped_cz_before_submission(fake_lqcloud):
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    backend = LQCloudQEC17Backend("secret")

    with pytest.raises(ValueError, match="non-adjacent"):
        backend.run(program, initial_layout=[0, 1])

    provider = _RecordingProvider.instances[-1]
    assert provider.backend.calls == []


def test_backend_supports_reversed_cz_edges_in_grouped_registers(fake_lqcloud):
    first = fq.QuantumRegister(2)
    second = fq.QuantumRegister(2)
    program = fq.Program([first, second])
    program.add(ops.CZ, (first.all(), second.all()))

    backend = LQCloudQEC17Backend("test-key")
    backend.run(program, initial_layout=(11, 12, 6, 4))

    circuit, _, options = _RecordingProvider.instances[-1].backend.calls[-1]
    assert options["initial_layout"] == [11, 12, 6, 4]
    assert circuit.calls[:2] == [("cz", (0, 2)), ("cz", (1, 3))]


@pytest.mark.parametrize("operation", [ops.Reset, ops.CX])
def test_backend_does_not_submit_unsupported_program(fake_lqcloud, operation):
    program = fq.Program(2)
    target = 0 if operation is ops.Reset else (0, 1)
    program.add(operation, target)
    backend = LQCloudQEC17Backend("test-key")

    with pytest.raises(ValueError):
        backend.run(program)

    assert _RecordingProvider.instances[-1].backend.calls == []


def test_backend_propagates_sdk_submission_errors(fake_lqcloud, monkeypatch):
    backend = LQCloudQEC17Backend("test-key")
    failure = RuntimeError("submission failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(_RecordingProvider.instances[-1].backend, "run", fail)
    with pytest.raises(RuntimeError) as error:
        backend.run(fq.Program(1))
    assert error.value is failure


@pytest.mark.parametrize(("api_key", "error"), [(None, TypeError), ("", ValueError)])
def test_backend_validates_api_key(fake_lqcloud, api_key, error):
    with pytest.raises(error):
        LQCloudQEC17Backend(api_key)
