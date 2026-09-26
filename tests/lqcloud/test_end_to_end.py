"""Compiler semantics on public device topology snapshots (not live QPU tests)."""

import json
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq

from fatqat.lqcloud import LQCloudBackend
from fatqat.compiler.dialects.sc_native import NativeGate, NativeMeasure

pytest.importorskip("lqcloud")

DEVICES = json.loads((Path(__file__).parent / "fixtures/devices.json").read_text())


def _logical(case):
    program = fq.LogicalProgram(3, 5)
    ops = fq.operations
    sequences = {
        "asymmetric": [(ops.X, 0)],
        "bell": [(ops.H, 0), (ops.CX, (0, 2))],
        "ghz": [(ops.H, 0), (ops.CX, (0, 1)), (ops.CX, (1, 2))],
        "toffoli": [
            (ops.H, 0),
            (ops.H, 1),
            (ops.RY(0.37), 2),
            (ops.CCX, (1, 0, 2)),
            (ops.H, 2),
        ],
        "iswap": [
            (ops.H, 0),
            (ops.H, 2),
            (ops.iSwap, (0, 2)),
            (ops.RX(0.6), 0),
            (ops.H, 2),
        ],
        "routing": [
            (ops.H, 0),
            (ops.H, 1),
            (ops.H, 2),
            (ops.CZ, (0, 1)),
            (ops.CZ, (1, 2)),
            (ops.CZ, (0, 2)),
            (ops.RY(0.4), 0),
            (ops.RX(-0.7), 2),
        ],
    }
    for op, targets in sequences[case]:
        program.add(op, targets)
    return program


def _probabilities(program, measurements):
    state = (
        fq.simulator.Simulator("SV", runtime="numpy")
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    size = sum(reg.size for reg in program.quantum_registers)
    result = np.zeros(32)
    for index, probability in enumerate(np.abs(state) ** 2):
        classical = 0
        for qubit, clbit in measurements:
            bit = (index >> (size - 1 - qubit)) & 1
            classical |= bit << (4 - clbit)
        result[classical] += probability
    return result


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device["name"])
@pytest.mark.parametrize(
    "case", ["asymmetric", "bell", "ghz", "toffoli", "iswap", "routing"]
)
@pytest.mark.parametrize("qasm", [False, True])
def test_device_pipeline_preserves_probabilities(cloud, device, case, qasm):
    cloud["configs"] = [device]
    backend = LQCloudBackend(device["name"])
    source = _logical(case)
    expected = _probabilities(source, [(2, 0), (0, 3)])
    source.measure(2, 0)
    source.measure(0, 3)
    if qasm:
        compiled = fq.compiler.compile_qasm_to_sc(
            fq.qasm.to_qasm(source), backend, seed=7
        )
    else:
        compiled = fq.compiler.compile_to_sc(source, backend, seed=7)
    sites = compiled.physical_layout
    indices = {site: index for index, site in enumerate(sites)}
    physical = fq.Program(len(sites))
    measurements = []
    edges = {frozenset(edge) for edge in device["topology"]["coupling_map"]}
    for instruction in compiled.output.operations:
        if isinstance(instruction, NativeGate):
            physical.add(
                instruction.operation,
                tuple(indices[site] for site in instruction.sites),
            )
            if instruction.operation is fq.operations.CZ:
                assert frozenset(instruction.sites) in edges
        elif isinstance(instruction, NativeMeasure):
            measurements.append((indices[instruction.site], instruction.clbit.index))
    actual = _probabilities(physical, measurements)
    assert np.allclose(actual, expected, atol=1e-10)
    assert cloud["submitted"] == []


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device["name"])
def test_device_snapshot_reaches_sdk_payload(cloud, device):
    cloud["configs"] = [device]
    backend = LQCloudBackend(device["name"])
    program = _logical("ghz")
    program.measure(2, 0)
    program.measure(0, 3)
    compiled = fq.compiler.compile_to_sc(program, backend, seed=7)
    backend.run(compiled, shots=10)
    payload = cloud["submitted"][0]["command"]["circuit"]
    assert payload["n_clbits"] == 5
    assert {
        instruction["clbits"][0]
        for instruction in payload["instructions"]
        if instruction["name"] == "measure"
    } == {0, 3}
    assert payload["initial_layout"] == list(compiled.physical_layout)
