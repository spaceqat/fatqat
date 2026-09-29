"""Define the shared Grover source and result ordering for the homepage."""

from math import pi

import fatqat as fq
import fatqat.operations as ops

TARGET = "101"
TARGET_INDEX = int(TARGET, 2)
COUPLINGS = ((0, 1), (1, 2))
COMPILER_SEED = 7


def build_logical_program():
    """Build two compact Grover iterations marking 101."""
    program = fq.LogicalProgram(3)

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

    fused_layer((0, -pi / 2), (1, pi / 2), (2, pi / 2), rz_targets=(1,))
    program.add(ops.CCX, (0, 1, 2))
    fused_layer((0, pi / 2), (1, pi / 2), (2, -pi / 2), rz_targets=(1,))
    program.add(ops.CCX, (0, 1, 2))
    fused_layer((0, -pi / 2), (1, -pi / 2))
    program.add(ops.Z, 2)
    return program


def logical_axis_order(compiled, result):
    """Read a routed final state in the source Program's qubit order."""
    refs_by_site = {
        compiled.resource_layout.device_label(ref): ref
        for ref in compiled.resource_layout.refs
    }
    axes_by_ref = {
        axis["register_ref"]: index
        for index, axis in enumerate(result.metadata["state_axes"])
    }
    return tuple(
        axes_by_ref[refs_by_site[site]]
        for _logical, site in compiled.output.final_layout
    )
