from typing import get_type_hints

import pytest

from fatqat.compiler import (
    compile_qasm_to_sc,
    to_sc_simulator_program,
)
from fatqat.compiler.dialects import NativeMeasure, SCNativeProgram
from fatqat.compiler.dialects.sc_native import _RotationNativeProgram
from fatqat.compiler.pipelines import _compile_qasm_to_sc_rotation
from fatqat.operations import Measurement
from fatqat.simulator import SCQubitSimulator
from fatqat.simulator.fake_superconducting import _SCQubitRotationSimulator

TRIANGLE_QASM = """
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
x q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[0], q[2];
c = measure q;
"""

LINE = ((0, 1), (1, 2))
FIVE_SITE_LINE = ((0, 1), (1, 2), (2, 3), (3, 4))


@pytest.mark.parametrize(
    ("backend", "compile_qasm", "program_type", "pass_name"),
    (
        (
            SCQubitSimulator(num_qubits=3, couplings=LINE, runtime="numpy"),
            compile_qasm_to_sc,
            SCNativeProgram,
            "lower-sc-to-native",
        ),
        (
            _SCQubitRotationSimulator(num_qubits=3, couplings=LINE, runtime="numpy"),
            _compile_qasm_to_sc_rotation,
            _RotationNativeProgram,
            "lower-sc-to-rotation",
        ),
    ),
)
def test_qasm_target_pipeline_preserves_public_order_through_routing(
    backend, compile_qasm, program_type, pass_name
):
    result = compile_qasm(TRIANGLE_QASM, backend, seed=5)

    assert type(result.output) is program_type
    assert result.route == ("parse-qasm", "normalize-sc", pass_name)

    program, resource_layout = to_sc_simulator_program(result.output)
    counts = (
        backend.run(
            program,
            shots=256,
            resource_layout=resource_layout,
            simulation_config={"seed": 9},
        )
        .result()
        .get_counts()
    )

    assert counts == {"110": 256}


def test_simulator_bridge_uses_fixed_physical_refs_and_original_clbits():
    backend = SCQubitSimulator(num_qubits=5, couplings=FIVE_SITE_LINE, runtime="numpy")
    native = compile_qasm_to_sc(TRIANGLE_QASM, backend, seed=2).output

    program, resource_layout = to_sc_simulator_program(native)

    used_sites = {site for _qubit, site in native.initial_layout + native.final_layout}
    for instruction in native.operations:
        if hasattr(instruction, "sites"):
            used_sites.update(instruction.sites)
        elif hasattr(instruction, "site"):
            used_sites.add(instruction.site)
    physical_refs = {
        register[index]
        for register in program.quantum_registers
        for index in range(register.size)
    }
    assert resource_layout.refs == physical_refs
    assert resource_layout.device_labels == used_sites

    native_clbits = tuple(
        instruction.clbit
        for instruction in native.operations
        if isinstance(instruction, NativeMeasure)
    )
    simulator_clbits = tuple(
        output
        for instruction in program._instructions
        if isinstance(instruction, Measurement)
        for output in instruction.outputs
    )
    assert simulator_clbits == native_clbits


def test_simulator_bridge_exposes_only_the_canonical_native_type():
    assert get_type_hints(to_sc_simulator_program)["native"] is SCNativeProgram
    with pytest.raises(TypeError, match="native must be SCNativeProgram"):
        to_sc_simulator_program(object())
