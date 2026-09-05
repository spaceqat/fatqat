"""Define Grover gate data and Programs for the homepage examples."""

from collections import Counter
from math import pi

import fatqat as fq
import fatqat.operations as ops

TARGET = "101"
TARGET_INDEX = int(TARGET, 2)

# Each rotation angle is expressed in quarter turns (pi / 4). This is the
# exact, fused nearest-neighbour realization of two Grover iterations for 101.
FUSED_GATES = (
    ("RY", 0, -2), ("RX", 1, -1), ("CZ", 0, 1), ("RY", 1, 2),
    ("RX", 2, 1), ("CZ", 1, 2), ("RZ", 1, 3), ("RY", 1, 2),
    ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, 1), ("CZ", 1, 2),
    ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, -1),
    ("CZ", 1, 2), ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2),
    ("RX", 2, -1), ("CZ", 1, 2), ("RZ", 0, -3), ("RY", 0, 2),
    ("RX", 1, -3), ("CZ", 0, 1), ("RY", 1, -2), ("RZ", 2, -1),
    ("RY", 2, -2), ("CZ", 1, 2), ("RZ", 1, -1), ("RY", 1, 2),
    ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, 1), ("CZ", 1, 2),
    ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, -1),
    ("CZ", 1, 2), ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2),
    ("RX", 2, -1), ("CZ", 1, 2), ("RZ", 0, 1), ("RY", 0, -2),
    ("RX", 1, -3), ("CZ", 0, 1), ("RY", 1, -2), ("RZ", 2, 1),
    ("RY", 2, 2), ("CZ", 1, 2), ("RZ", 1, -1), ("RY", 1, 2),
    ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, 1), ("CZ", 1, 2),
    ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, -1),
    ("CZ", 1, 2), ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2),
    ("RX", 2, -1), ("CZ", 1, 2), ("RZ", 0, 1), ("RY", 0, 2),
    ("RX", 1, -3), ("CZ", 0, 1), ("RY", 1, -2), ("RZ", 2, -1),
    ("RY", 2, -2), ("CZ", 1, 2), ("RZ", 1, -1), ("RY", 1, 2),
    ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, 1), ("CZ", 1, 2),
    ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2), ("RX", 2, -1),
    ("CZ", 1, 2), ("RY", 1, 2), ("CZ", 0, 1), ("RY", 1, -2),
    ("RX", 2, -1), ("CZ", 1, 2), ("RZ", 0, 1), ("RY", 0, -2),
    ("RY", 1, 2), ("RZ", 1, -4), ("RZ", 2, -4),
)


def build_native_program():
    """Build the fused rotation Program shared by two execution examples."""
    program = fq.Program(3)
    rotations = {"RX": ops.RX, "RY": ops.RY, "RZ": ops.RZ}
    for gate in FUSED_GATES:
        if gate[0] == "CZ":
            program.add(ops.CZ, gate[1:])
        else:
            name, target, quarter_turns = gate
            program.add(rotations[name](quarter_turns * pi / 4), target)

    counts = Counter(gate[0] for gate in FUSED_GATES)
    assert counts == {"RX": 17, "RY": 37, "RZ": 13, "CZ": 32}
    assert all(
        gate[0] != "CZ" or abs(gate[1] - gate[2]) == 1
        for gate in FUSED_GATES
    )
    return program


def build_logical_program():
    """Build the equivalent compact Program used for the circuit drawing."""
    program = fq.Program(3)

    def fused_layer(*rotations, rz_targets=()):
        for target, theta in rotations:
            program.add(ops.RY(theta), target)
        for target in rz_targets:
            program.add(ops.RZ(pi), target)

    program.add(ops.H, 0)
    fused_layer((1, pi / 2))
    program.add(ops.CCX, (0, 1, 2))
    fused_layer((0, pi / 2), (1, pi / 2), (2, -pi / 2), rz_targets=(1,))
    program.add(ops.CCX, (0, 1, 2))
    program.add(ops.Barrier, (0, 1, 2))

    fused_layer((0, -pi / 2), (1, pi / 2), (2, pi / 2), rz_targets=(1,))
    program.add(ops.CCX, (0, 1, 2))
    fused_layer((0, pi / 2), (1, pi / 2), (2, -pi / 2), rz_targets=(1,))
    program.add(ops.CCX, (0, 1, 2))
    fused_layer((0, -pi / 2), (1, -pi / 2))
    program.add(ops.Z, 2)
    program.add(ops.Barrier, (0, 1, 2))
    return program
