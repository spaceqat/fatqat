import fatqat as fq
from fatqat.compiler.passes import snapshot_program
from fatqat.resource_layout import ResourceLayout
from fatqat.simulator._connectivity import _AtomConnectivity


def test_compiler_refs_bind_directly_to_simulator_sites():
    register = fq.QuantumRegister(2, name="atoms")
    logical = snapshot_program(fq.Program([register]))
    layout = ResourceLayout({logical.qubits[0]: 5, logical.qubits[1]: 2})
    assert layout.device_label(register[0]) == 5
    assert layout.device_label(register[1]) == 2


def test_connectivity_is_over_program_refs_not_site_labels():
    register = fq.QuantumRegister(2, name="atoms")
    logical = snapshot_program(fq.Program([register]))
    connectivity = _AtomConnectivity().pair(*logical.qubits)
    assert connectivity.are_paired(register[0], register[1])


def test_same_named_registers_keep_distinct_refs_across_layout_and_connectivity():
    first_register = fq.QuantumRegister(1, name="atoms")
    second_register = fq.QuantumRegister(1, name="atoms")
    logical = snapshot_program(fq.Program([first_register, second_register]))
    first_ref, second_ref = logical.qubits

    assert first_ref != second_ref

    layout = ResourceLayout({first_ref: 5, second_ref: 2})
    assert layout.device_label(first_register[0]) == 5
    assert layout.device_label(second_register[0]) == 2

    connectivity = _AtomConnectivity().pair(first_ref, second_ref)
    assert connectivity.are_paired(first_register[0], second_register[0])
    assert not connectivity.unpair(first_ref, second_ref).are_paired(
        first_register[0], second_register[0]
    )


def test_arrangement_describes_sites_without_owning_atoms():
    arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
    register = fq.QuantumRegister(2, name="atoms")
    layout = ResourceLayout({register[0]: 0, register[1]: 1})
    assert arrangement.coordinates == ((0.0, 0.0, 0.0), (6.0, 0.0, 0.0))
    assert layout.device_labels == frozenset((0, 1))
