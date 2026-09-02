import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator


def _teleport_program(prep):
    # q0 = payload, q1/q2 = Bell pair; teleport payload onto q2.
    p = fq.Program(3, 2)
    for gate in prep:
        p.add(gate, 0)
    p.add(ops.H, 1)
    p.add(ops.CX, (1, 2))
    p.add(ops.CX, (0, 1))
    p.add(ops.H, 0)
    p.measure(0, 0)
    p.measure(1, 1)
    p.add(ops.X, 2, condition=(1, 1))
    p.add(ops.Z, 2, condition=(0, 1))
    return p


def test_teleportation_moves_one_state_to_target():
    p = _teleport_program(prep=[ops.X])
    p.measure(2, 0)
    counts = (
        Simulator("SV")
        .run(p, shots=64, simulation_config={"seed": 1})
        .result()
        .get_counts()
    )
    assert all(key[0] == "1" for key in counts)


def test_teleportation_moves_plus_state_to_target():
    p = _teleport_program(prep=[ops.H])
    p.add(ops.H, 2)
    p.measure(2, 0)
    counts = (
        Simulator("SV")
        .run(p, shots=64, simulation_config={"seed": 3})
        .result()
        .get_counts()
    )
    assert all(key[0] == "0" for key in counts)


def test_bit_flip_code_corrects_single_x_error():
    p = fq.Program(5, 5)
    p.add(ops.X, 1)
    p.add(ops.CX, (0, 3))
    p.add(ops.CX, (1, 3))
    p.add(ops.CX, (1, 4))
    p.add(ops.CX, (2, 4))
    p.measure(3, 0)
    p.measure(4, 1)
    p.add(ops.X, 1, condition=((0, 1), (1, 1)))
    p.measure(0, 2)
    p.measure(1, 3)
    p.measure(2, 4)
    counts = (
        Simulator("SV")
        .run(p, shots=32, simulation_config={"seed": 2})
        .result()
        .get_counts()
    )
    assert all(key[2:] == "000" for key in counts)


def test_reset_seed_independence_matches_born_rule():
    def program():
        p = fq.Program(2, 1)
        p.add(ops.H, 0)
        p.add(ops.CX, (0, 1))
        p.add(ops.Reset, 0)
        p.measure(1, 0)
        return p

    ones = 0
    n = 1000
    backend = Simulator("SV")
    for s in range(n):
        counts = (
            backend.run(program(), shots=1, simulation_config={"seed": s})
            .result()
            .get_counts()
        )
        ones += counts.get("1", 0)
    frac = ones / n
    assert 0.44 < frac < 0.56
