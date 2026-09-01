"""Tests for fatqat.qasm -- OpenQASM <-> fatqat.Program, both directions.

Run with:  PYTHONPATH=<path-to-fatqat-src> python3 -m pytest test_qasm.py -v
(or just: python3 test_qasm.py, it also runs standalone)
"""

from __future__ import annotations

import math

import pytest

import fatqat as fc
from fatqat.emulator import ControlChannel, PulseControl
from fatqat.errors import FatqatError
import fatqat.operations as ops
from fatqat.operations import Measurement, PulseOperation
from fatqat.qasm import (
    QASMTranspileError,
    QasmExportError,
    from_qasm,
    program_to_qasm,
    qasm_to_program,
)
from fatqat.emulator import SampledWaveform


def test_qasm_exceptions_share_public_fatqat_error_hierarchy():
    assert issubclass(QASMTranspileError, FatqatError)
    assert issubclass(QasmExportError, FatqatError)


def test_qasm_transpile_error_remains_value_error_compatible():
    assert issubclass(QASMTranspileError, ValueError)


# ===========================================================================
# Export direction: fatqat.Program -> OpenQASM
# ===========================================================================


def test_bell_state_v3():
    p = fc.Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.measure_all()
    out = program_to_qasm(p, version=3)
    assert "OPENQASM 3.0;" in out
    assert "qubit[2] q;" in out
    assert "h q[0];" in out
    assert "cx q[0], q[1];" in out
    assert "c[0] = measure q[0];" in out
    assert "c[1] = measure q[1];" in out


def test_bell_state_v2():
    p = fc.Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.measure_all()
    out = program_to_qasm(p, version=2)
    assert "OPENQASM 2.0;" in out
    assert "qreg q[2];" in out
    assert "measure q[0] -> c[0];" in out


def test_all_fixed_gates():
    p = fc.Program(3)
    for g in (
        ops.H,
        ops.I,
        ops.S,
        ops.Sdg,
        ops.T,
        ops.Tdg,
        ops.X,
        ops.Y,
        ops.Z,
    ):
        p.add(g, 0)
    p.add(ops.CX, (0, 1))
    p.add(ops.CZ, (0, 1))
    p.add(ops.Swap, (0, 1))
    p.add(ops.CY, (0, 1))
    p.add(ops.CCX, (0, 1, 2))
    p.add(ops.CSwap, (0, 1, 2))
    out = program_to_qasm(p, version=3)
    for expected in (
        "h q[0];",
        "id q[0];",
        "s q[0];",
        "sdg q[0];",
        "t q[0];",
        "tdg q[0];",
        "x q[0];",
        "y q[0];",
        "z q[0];",
        "cx q[0], q[1];",
        "cz q[0], q[1];",
        "swap q[0], q[1];",
        "cy q[0], q[1];",
        "ccx q[0], q[1], q[2];",
        "cswap q[0], q[1], q[2];",
    ):
        assert expected in out


def test_cs_maps_to_cp_pi_over_2():
    p = fc.Program(2)
    p.add(ops.CS, (0, 1))
    out = program_to_qasm(p, version=3)
    assert "cp(pi/2) q[0], q[1];" in out


def test_iswap_emits_custom_gate_and_call():
    p = fc.Program(2)
    p.add(ops.iSwap, (0, 1))
    out = program_to_qasm(p, version=3)
    assert "gate iswap a, b {" in out
    assert "iswap q[0], q[1];" in out


def test_parametric_gates():
    p = fc.Program(2)
    p.add(ops.RX(0.3), 0)
    p.add(ops.RY(0.3), 0)
    p.add(ops.RZ(0.3), 0)
    p.add(ops.Phase(0.3), 0)
    p.add(ops.CPhase(0.3), (0, 1))
    out = program_to_qasm(p, version=3)
    assert "rx(0.3) q[0];" in out
    assert "ry(0.3) q[0];" in out
    assert "rz(0.3) q[0];" in out
    assert "p(0.3) q[0];" in out
    assert "cp(0.3) q[0], q[1];" in out


def test_reset():
    p = fc.Program(2)
    p.add(ops.Reset, (0, 1))
    out = program_to_qasm(p, version=3)
    assert "reset q[0];" in out
    assert "reset q[1];" in out


def test_condition_v3_multi_term():
    p = fc.Program(2, 3)
    p.add(
        ops.X,
        1,
        condition=((p.classical_registers[0][0], 1), (p.classical_registers[0][2], 0)),
    )
    out = program_to_qasm(p, version=3)
    assert "if (c[0] == 1 && c[2] == 0) { x q[1]; }" in out


def test_condition_v2_full_register_ok():
    p = fc.Program(2, 1)
    p.add(ops.X, 1, condition=(p.classical_registers[0][0], 1))
    out = program_to_qasm(p, version=2)
    assert "if (c == 1) x q[1];" in out


def test_condition_v2_partial_register_rejected():
    p = fc.Program(2, 3)
    p.add(ops.X, 1, condition=(p.classical_registers[0][0], 1))
    with pytest.raises(QasmExportError):
        program_to_qasm(p, version=2)


def test_condition_v2_across_two_registers_rejected():
    a = fc.ClassicalRegister(1, name="a")
    b = fc.ClassicalRegister(1, name="b")
    p = fc.Program(2, [a, b])
    p.add(ops.X, 1, condition=((a[0], 1), (b[0], 1)))
    with pytest.raises(QasmExportError, match="multiple classical registers"):
        program_to_qasm(p, version=2)


def test_condition_v2_across_lookalike_registers_rejected():
    a = fc.ClassicalRegister(1, name="c")
    b = fc.ClassicalRegister(1, name="c")
    p = fc.Program(2, [a, b])
    p.add(ops.X, 1, condition=((a[0], 1), (b[0], 1)))
    with pytest.raises(QasmExportError, match="multiple classical registers"):
        program_to_qasm(p, version=2)


def test_qudit_dim2_reductions():
    p = fc.Program(2)
    p.add(ops.Shift(1), 0)  # -> x
    p.add(ops.Shift(2), 0)  # -> elided
    p.add(ops.Clock(1), 0)  # -> z
    p.add(ops.Sum, (0, 1))  # -> cx
    p.add(ops.SwapLevels(0, 1), 0)  # -> x
    p.add(ops.Fourier, 0)  # -> h
    p.add(ops.InverseFourier, 0)  # -> h
    p.add(ops.SubspaceRX(0.4, (0, 1)), 0)  # -> rx(0.4)
    p.add(ops.SubspaceRY(0.4, (1, 0)), 0)  # -> ry(-0.4)
    p.add(ops.SubspaceRZ(0.4, (1, 0)), 0)  # -> rz(-0.4)
    p.add(ops.CClock(1), (0, 1))  # -> cz
    out = program_to_qasm(p, version=3)
    assert "x q[0];" in out
    assert "// elided: Shift(power=2) is identity at dim=2" in out
    assert "z q[0];" in out
    assert "cx q[0], q[1];" in out
    assert "h q[0];" in out
    assert "rx(0.4) q[0];" in out
    assert "ry(-0.4) q[0];" in out
    assert "rz(-0.4) q[0];" in out
    assert "cz q[0], q[1];" in out


def test_qudit_dim_gt_2_rejected():
    qreg = fc.QuantumRegister(2, dim=3)
    p = fc.Program([qreg])
    p.add(ops.Clock(1), 0)
    with pytest.raises(QasmExportError):
        program_to_qasm(p, version=3)


def test_view_bearing_program_rejected_before_scalar_ref_formatting():
    # RegisterView-bearing programs are not QASM-exportable yet; the guard
    # must fire before to_qasm() ever tries to format a scalar ref, not
    # crash later with a missing-attribute error.
    qubits = fc.GridRegister(1, 2, name="qubits")
    p = fc.Program([qubits])
    p.add(ops.RX(0.3), qubits.row(0))
    with pytest.raises(QasmExportError, match="view"):
        program_to_qasm(p, version=3)


def test_export_same_named_qreg_and_creg_do_not_collide():
    # Regression test: a quantum register and a classical register that
    # happen to share a name (or sanitize to the same identifier) used to
    # be tracked in separate "taken names" sets on export, so both could
    # silently render as the same QASM identifier -- e.g. `qubit[2] r;`
    # and `bit[2] r;`, which is invalid QASM (a single identifier
    # namespace is shared between quantum and classical declarations) and
    # which this module's own importer rejects on round-trip.
    qreg = fc.QuantumRegister(2, name="r")
    creg = fc.ClassicalRegister(2, name="r")
    p = fc.Program([qreg], [creg])
    p.add(ops.H, 0)
    p.measure(0, 0)

    out = program_to_qasm(p, version=3)

    declared_names = set()
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("qubit[") or line.startswith("bit["):
            name = line.split("]", 1)[1].strip().rstrip(";").strip()
            assert name not in declared_names, (
                f"duplicate identifier {name!r} in emitted QASM:\n{out}"
            )
            declared_names.add(name)

    # And the result must actually round-trip back through from_qasm.
    program = from_qasm(out)
    assert len(program.quantum_registers) == 1 and len(program.classical_registers) == 1


# ===========================================================================
# Import direction: OpenQASM -> fatqat.Program
# ===========================================================================


def test_from_qasm_builds_bell_program():
    program = from_qasm("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """)

    assert [op.operation.name for op in program._instructions[:2]] == ["H", "CX"]
    measurement = program._instructions[2]
    assert isinstance(measurement, Measurement)
    assert measurement.targets == (
        program.quantum_registers[0][0],
        program.quantum_registers[0][1],
    )
    assert measurement.outputs == (
        program.classical_registers[0][0],
        program.classical_registers[0][1],
    )


def test_from_qasm_preserves_multiple_register_names():
    program = qasm_to_program("""
        OPENQASM 2.0;
        qreg qa[1];
        qreg qb[1];
        creg ca[1];
        x qb[0];
        measure qb[0] -> ca[0];
        """)

    assert [reg.name for reg in program.quantum_registers] == ["qa", "qb"]
    assert program._instructions[0].targets == (program.quantum_registers[1][0],)


def test_from_qasm_expands_whole_register_single_qubit_gate():
    program = from_qasm("""
        OPENQASM 2.0;
        qreg q[3];
        x q;
        """)

    assert [op.operation.name for op in program._instructions] == ["X", "X", "X"]
    assert [op.targets[0].index for op in program._instructions] == [0, 1, 2]


def test_from_qasm_parses_parameter_expressions():
    program = from_qasm("""
        OPENQASM 2.0;
        qreg q[1];
        rz(pi / 2) q[0];
        u2(0, pi) q[0];
        """)

    assert math.isclose(program._instructions[0].operation.theta, math.pi / 2)
    assert [op.operation.name for op in program._instructions[1:]] == ["U2"]
    assert program._instructions[1].operation.phi == 0
    assert math.isclose(program._instructions[1].operation.lam, math.pi)


def test_from_qasm_supports_classical_conditions():
    program = from_qasm("""
        OPENQASM 2.0;
        qreg q[1];
        creg c[2];
        if(c==2) x q[0];
        """)

    assert program._instructions[0].condition == (
        (program.classical_registers[0][0], 0),
        (program.classical_registers[0][1], 1),
    )


def test_from_qasm_rejects_unsupported_gate():
    with pytest.raises(QASMTranspileError, match="unsupported gate"):
        from_qasm("""
            OPENQASM 2.0;
            qreg q[2];
            rzz(0.5) q[0], q[1];
            """)


def test_from_qasm_expands_custom_gate_definition():
    program = from_qasm("""
        OPENQASM 2.0;
        gate myx a { x a; }
        qreg q[1];
        myx q[0];
        """)

    assert [op.operation.name for op in program._instructions] == ["X"]
    assert program._instructions[0].targets == (program.quantum_registers[0][0],)


def test_from_qasm_expands_iswap_custom_gate_matching_forward_tool_output():
    # This is exactly the custom gate block fatqat_to_qasm.py emits for
    # iSwap -- if this doesn't parse, the two tools cannot round-trip.
    program = from_qasm("""
        OPENQASM 3.0;
        include "stdgates.inc";

        gate iswap a, b {
            s a;
            s b;
            h a;
            cx a, b;
            cx b, a;
            h b;
        }

        qubit[2] q;

        iswap q[0], q[1];
        """)

    assert [op.operation.name for op in program._instructions] == [
        "S",
        "S",
        "H",
        "CX",
        "CX",
        "H",
    ]
    # cx a,b then cx b,a -- control/target must swap between the two CX calls.
    cx_ops = [op for op in program._instructions if op.operation.name == "CX"]
    assert cx_ops[0].targets == (
        program.quantum_registers[0][0],
        program.quantum_registers[0][1],
    )
    assert cx_ops[1].targets == (
        program.quantum_registers[0][1],
        program.quantum_registers[0][0],
    )


def test_from_qasm_custom_gate_with_parameter_expression():
    program = from_qasm("""
        OPENQASM 2.0;
        gate my_crz(theta) a, b {
            rz(theta/2) b;
            cx a, b;
            rz(-theta/2) b;
            cx a, b;
        }
        qreg q[2];
        my_crz(pi/2) q[0], q[1];
        """)

    assert [op.operation.name for op in program._instructions] == [
        "RZ",
        "CX",
        "RZ",
        "CX",
    ]
    assert math.isclose(program._instructions[0].operation.theta, math.pi / 4)
    assert math.isclose(program._instructions[2].operation.theta, -math.pi / 4)


def test_from_qasm_rejects_opaque_declaration():
    with pytest.raises(QASMTranspileError, match="opaque"):
        from_qasm("""
            OPENQASM 2.0;
            opaque myx a;
            qreg q[1];
            """)


def test_from_qasm3_bit_level_and_conditions():
    # This is exactly the form fatqat_to_qasm.py emits for QASM3
    # conditions -- if this doesn't parse, the two tools cannot round-trip.
    program = from_qasm("""
        OPENQASM 3.0;
        qubit[2] q;
        bit[3] c;
        if (c[0] == 1 && c[2] == 0) { x q[1]; }
        """)

    assert program._instructions[0].operation.name == "X"
    assert program._instructions[0].condition == (
        (program.classical_registers[0][0], 1),
        (program.classical_registers[0][2], 0),
    )


def test_from_qasm_u3_gate_matches_exact_matrix_up_to_global_phase():
    import numpy as np

    from fatqat.simulator import Simulator

    theta, phi, lam = 0.9, 0.4, -0.6
    program = from_qasm(f"""
        OPENQASM 2.0;
        qreg q[1];
        u3({theta}, {phi}, {lam}) q[0];
        """)
    job = Simulator("SV").run(
        program, result_config={"counts": False, "final_state": True}, shots=1
    )
    sv = job.result().get_statevector()

    u3 = np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ]
    )
    target = u3 @ np.array([1, 0], dtype=complex)
    # Compare via density matrices since the two are only equal up to a
    # global phase (fatqat has no global-phase primitive).
    rho_a = np.outer(sv, sv.conj())
    rho_b = np.outer(target, target.conj())
    assert np.allclose(rho_a, rho_b, atol=1e-8)


def test_from_qasm_maps_supported_gate_names():
    program = from_qasm("""
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        ccx q[0], q[1], q[2];
        cp(pi) q[0], q[2];
        reset q[0];
        barrier q;
        """)

    assert [op.operation.name for op in program._instructions] == [
        "H",
        "CCX",
        "CPhase",
        "Reset",
    ]
    assert isinstance(program._instructions[0].operation, type(ops.H))


def test_from_qasm3_builds_bell_program():
    program = from_qasm("""
        OPENQASM 3;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cnot q[0], q[1];
        c = measure q;
        """)

    assert program.metadata["source"] == "openqasm3.0"
    assert [op.operation.name for op in program._instructions[:2]] == ["H", "CX"]
    measurement = program._instructions[2]
    assert isinstance(measurement, Measurement)
    assert measurement.targets == (
        program.quantum_registers[0][0],
        program.quantum_registers[0][1],
    )
    assert measurement.outputs == (
        program.classical_registers[0][0],
        program.classical_registers[0][1],
    )


def test_from_qasm3_single_qubit_and_bit_declarations():
    program = from_qasm("""
        OPENQASM 3.0;
        qubit q;
        bit c;
        x q;
        c = measure q;
        """)

    assert program.quantum_registers[0].size == 1
    assert program.classical_registers[0].size == 1
    assert [op.operation.name for op in program._instructions[:1]] == ["X"]


def test_from_qasm3_if_block_and_gate_aliases():
    program = from_qasm("""
        OPENQASM 3;
        qubit[3] q;
        bit[2] c;
        if (c == 1) { toffoli q[0], q[1], q[2]; }
        """)

    assert program._instructions[0].operation.name == "CCX"
    assert program._instructions[0].condition == (
        (program.classical_registers[0][0], 1),
        (program.classical_registers[0][1], 0),
    )


def test_qasm_export_rejects_pulse_operation_explicitly():
    program = fc.Program(1)
    program.add(
        PulseOperation(
            1.0,
            (
                PulseControl(
                    ControlChannel(),
                    SampledWaveform((0.0, 1.0), (1.0, 1.0)),
                ),
            ),
        )
    )

    with pytest.raises(QasmExportError, match="PulseOperation is not supported"):
        program_to_qasm(program)


def test_qasm_export_rejects_unbound_parameters_descriptively():
    theta = fc.Parameter("theta")
    program = fc.Program(1)
    program.add(ops.RX(theta), 0)

    with pytest.raises(
        fc.errors.BackendValidationError,
        match=r"^program has unbound parameters: theta$",
    ):
        program_to_qasm(program)


def test_export_sx_and_u_family():
    import numpy as np

    from fatqat.simulator import Simulator

    program = fc.Program(1)
    program.add(ops.SX, 0)
    program.add(ops.U(0.7, 0.4, 1.1), 0)
    program.add(ops.U1(0.3), 0)
    program.add(ops.U2(0.2, 0.5), 0)
    program.add(ops.U3(0.9, -0.4, 0.25), 0)

    qasm3 = program_to_qasm(program, version=3)
    assert "sx q[0];" in qasm3
    assert "U(0.7, 0.4, 1.1) q[0];" in qasm3
    assert "p(0.3) q[0];" in qasm3
    assert "U(pi/2, 0.2, 0.5) q[0];" in qasm3

    qasm2 = program_to_qasm(program, version=2)
    assert "gate sx a" in qasm2
    assert "u3(0.7, 0.4, 1.1) q[0];" in qasm2
    assert "u1(0.3) q[0];" in qasm2

    def statevector(p):
        job = Simulator("SV").run(
            p, result_config={"counts": False, "final_state": True}, shots=1
        )
        return np.asarray(job.result().get_statevector())

    reference = statevector(program)
    reimported = statevector(from_qasm(qasm2))
    k = int(np.argmax(np.abs(reference)))
    phase = reimported[k] / reference[k]
    assert np.isclose(abs(phase), 1.0)
    assert np.allclose(reimported, reference * phase)


def test_export_qasm2_sx_definition_emitted_once_and_parses():
    qiskit_qasm2 = pytest.importorskip("qiskit.qasm2")

    program = fc.Program(1)
    program.add(ops.SX, 0)
    program.add(ops.SX, 0)

    qasm2 = program_to_qasm(program, version=2)
    assert qasm2.count("gate sx a") == 1
    circuit = qiskit_qasm2.loads(qasm2)
    assert [instruction.operation.name for instruction in circuit.data] == ["sx", "sx"]


def test_from_qasm_u_family_matches_qiskit_converter_identity():
    program = from_qasm("""
        OPENQASM 2.0;
        qreg q[1];
        u1(0.3) q[0];
        u3(0.9, 0.4, -0.6) q[0];
        sx q[0];
        """)

    names = [step.operation.name for step in program._instructions]
    assert names == ["U1", "U3", "SX"]
    u3 = program._instructions[1].operation
    assert (u3.theta, u3.phi, u3.lam) == (0.9, 0.4, -0.6)


def test_export_barrier_as_native_statement():
    p = fc.Program(2, 1)
    p.add(ops.H, 0)
    p.add(ops.Barrier, (0, 1))
    p.add(ops.X, 1)
    out3 = program_to_qasm(p, version=3)
    assert "barrier q[0], q[1];" in out3
    out2 = program_to_qasm(p, version=2)
    assert "barrier q[0], q[1];" in out2

    # a conditioned barrier exports unconditioned (it is a no-op either way)
    p2 = fc.Program(1, 1)
    p2.add(ops.Barrier, 0, condition=(0, 1))
    out = program_to_qasm(p2, version=2)
    assert "barrier q[0];" in out
    assert "if" not in out


def test_from_qasm_scalar_register_broadcast():
    program = from_qasm("""
        OPENQASM 2.0;
        qreg q[1];
        qreg r[3];
        cx q[0], r;
        """)
    q = program.quantum_registers[0]
    r = program.quantum_registers[1]
    applied = [step.targets for step in program._instructions]
    assert applied == [(q[0], r[0]), (q[0], r[1]), (q[0], r[2])]


def test_from_qasm_mismatched_register_widths_still_rejected():
    with pytest.raises(QASMTranspileError, match="equal size"):
        from_qasm("""
            OPENQASM 2.0;
            qreg a[2];
            qreg b[3];
            cx a, b;
            """)


def test_from_qasm_u0_accepts_ignored_parameter():
    program = from_qasm("""
        OPENQASM 2.0;
        qreg q[1];
        u0(1) q[0];
        u0 q[0];
        """)
    assert [step.operation.name for step in program._instructions] == ["I", "I"]


def test_export_register_named_after_gate_is_renamed():
    qiskit_qasm2 = pytest.importorskip("qiskit.qasm2")

    sx_reg = fc.QuantumRegister(1, name="sx")
    iswap_reg = fc.QuantumRegister(2, name="iswap")
    p = fc.Program([sx_reg, iswap_reg])
    p.add(ops.SX, sx_reg[0])
    p.add(ops.iSwap, (iswap_reg[0], iswap_reg[1]))

    out = program_to_qasm(p, version=2)
    assert "qreg sx_[1];" in out
    assert "qreg iswap_[2];" in out
    assert "sx sx_[0];" in out
    circuit = qiskit_qasm2.loads(out)
    assert circuit.num_qubits == 3
