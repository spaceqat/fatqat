import pytest

from fatqat.compiler import (
    compile_qasm_to_google,
    compile_qasm_to_ibm,
    to_sc_simulator_program,
)
from fatqat.compiler.dialects import GoogleProgram, IBMProgram, NativeMeasure
from fatqat.operations import Measurement
from fatqat.simulator import SCQubitGoogleSimulator, SCQubitIBMSimulator

TRIANGLE_QASM = """
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
h q[0];
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
            SCQubitIBMSimulator(num_qubits=3, couplings=LINE, runtime="numpy"),
            compile_qasm_to_ibm,
            IBMProgram,
            "lower-sc-to-ibm",
        ),
        (
            SCQubitGoogleSimulator(num_qubits=3, couplings=LINE, runtime="numpy"),
            compile_qasm_to_google,
            GoogleProgram,
            "lower-sc-to-google",
        ),
    ),
)
def test_qasm_target_pipeline_executes_on_corresponding_simulator(
    backend, compile_qasm, program_type, pass_name
):
    result = compile_qasm(TRIANGLE_QASM, backend, seed=5)

    assert isinstance(result.output, program_type)
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

    assert sum(counts.values()) == 256


def test_simulator_bridge_uses_fixed_physical_refs_and_original_clbits():
    backend = SCQubitIBMSimulator(
        num_qubits=5, couplings=FIVE_SITE_LINE, runtime="numpy"
    )
    native = compile_qasm_to_ibm(TRIANGLE_QASM, backend, seed=2).output

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
