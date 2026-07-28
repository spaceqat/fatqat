"""Render a fatqat :py:class:`~fatqat.Program` as a circuit diagram via QuTiP-QIP.

This is an optional feature: it requires the ``qutip-qip`` package (the
``draw`` dependency group). The dependency is imported lazily, inside the
functions that need it, so importing :py:mod:`fatqat.draw` - and the rest of
fatqat - never requires ``qutip-qip`` to be installed.

Two entry points:

- :py:func:`to_qubit_circuit` translates a program into a
  ``qutip_qip.circuit.QubitCircuit`` (the reusable seam - call QuTiP's own
  ``.draw(...)`` on the result if you want its full set of options).
- :py:func:`draw` is a thin convenience wrapper that renders the circuit as a
  matplotlib ``Figure`` (save it yourself with ``fig.savefig("circuit.png")``)
  or as a text-diagram ``str`` for the terminal.

Translation is one-directional (fatqat -> QuTiP) and for drawing only: the
resulting circuit is never executed, so an operation that QuTiP cannot
simulate is still fine to *draw*. Any gate without a native QuTiP equivalent -
including user-defined custom operations - is drawn as a labeled box carrying
the operation's own ``name``.

QuTiP-QIP draws qubit circuits only, so a program that declares any qudit
(``dim != 2``) register is rejected with a clear error rather than mis-drawn.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

from .operations import BarrierGate, Measurement, ResetGate
from .program import Program
from .registers import RegisterRef, RegisterView, _view_members

# Map fatqat operation names to their native QuTiP-QIP gate. Each value is
# ``(qutip_name, n_controls, param_attr)``:
#   - qutip_name : the gate name QuTiP-QIP draws (its spellings differ from
#     fatqat's - e.g. Hadamard is "SNOT", sqrt-X is "SQRTNOT").
#   - n_controls : how many of the operation's *leading* operands are controls.
#     fatqat's operand convention puts control(s) first and target(s) last
#     (see ``implementation/base.py``), so ``wires[:n_controls]`` are the
#     controls and ``wires[n_controls:]`` the targets.
#   - param_attr : the operation attribute to pass to QuTiP as ``arg_value``
#     (the rotation/phase angle), or ``None`` for a non-parametric gate.
# Any operation *not* listed here is drawn as a labeled box (see _add_operation).
_NATIVE_GATES: dict[str, tuple[str, int, str | None]] = {
    "H": ("SNOT", 0, None),
    "X": ("X", 0, None),
    "Y": ("Y", 0, None),
    "Z": ("Z", 0, None),
    "S": ("S", 0, None),
    "T": ("T", 0, None),
    "SX": ("SQRTNOT", 0, None),
    "RX": ("RX", 0, "theta"),
    "RY": ("RY", 0, "theta"),
    "RZ": ("RZ", 0, "theta"),
    "Phase": ("PHASEGATE", 0, "theta"),
    "CX": ("CNOT", 1, None),
    "CZ": ("CZ", 1, None),
    "CPhase": ("CPHASE", 1, "theta"),
    "Swap": ("SWAP", 0, None),
    "iSwap": ("ISWAP", 0, None),
    "CCX": ("TOFFOLI", 2, None),
    "CSwap": ("FREDKIN", 1, None),
}


def _require_qutip():
    """Import QuTiP-QIP's ``QubitCircuit``, or raise a clear install hint.

    Imported lazily so that ``import fatqat.draw`` works without ``qutip-qip``;
    only actually *drawing* a circuit needs the dependency.
    """
    try:
        from qutip_qip.circuit import QubitCircuit
    except ImportError as exc:  # pragma: no cover - only hit without qutip-qip
        raise ImportError(
            "fatqat.draw requires the optional 'qutip-qip' dependency; "
            "install it with `uv sync --group draw` (or `pip install qutip-qip`)."
        ) from exc
    return QubitCircuit


def _wire_maps(program: Program) -> tuple[dict, dict]:
    """Assign consecutive global wire indices to every qubit and clbit ref.

    QuTiP's ``QubitCircuit(N)`` numbers wires ``0..N-1``, so each fatqat
    ``RegisterRef`` needs a single global index. Registers are numbered in
    declaration order (concatenated), which matches the flat ordering the
    simulator itself uses - so wire ``k`` in the diagram is subsystem ``k`` in
    a run. Refs are keyed by ``(id(register), slot)`` because two distinct
    registers can share a name.
    """
    qubit_index: dict[tuple[int, int], int] = {}
    for register in program.quantum_registers:
        for slot in range(register.size):
            qubit_index[(id(register), slot)] = len(qubit_index)
    clbit_index: dict[tuple[int, int], int] = {}
    for register in program.classical_registers:
        for slot in range(register.size):
            clbit_index[(id(register), slot)] = len(clbit_index)
    return qubit_index, clbit_index


def _wire(ref: RegisterRef, index_map: dict) -> int:
    """Resolve one ref to its global wire index."""
    return index_map[(id(ref.register), ref.index)]


def _expand_targets(targets: tuple) -> tuple[tuple[RegisterRef, ...], ...]:
    """Expand a possibly-grouped target tuple into scalar operand tuples.

    A fatqat operation may target a whole ``RegisterView`` (a broadcast group);
    a circuit diagram needs one gate per scalar operand, so a view target is
    expanded the same way the backend expands it for execution - via the
    library's own ``_view_members``, which defines the deterministic member
    order. The common case (all targets are plain refs) returns unchanged.
    """
    if not any(isinstance(target, RegisterView) for target in targets):
        return (targets,)
    members = tuple(
        _view_members(target) if isinstance(target, RegisterView) else (target,)
        for target in targets
    )
    # Arity 1: one gate per member. Arity >= 2: zip members position-wise (the
    # views are guaranteed equal-length by the frontend that built the group).
    if len(members) == 1:
        return tuple((member,) for member in members[0])
    return tuple(zip(*members))


def to_qubit_circuit(program: Program):
    """Translate a program into a QuTiP-QIP ``QubitCircuit`` for drawing.

    Args:
        program: The program to translate. Must use only qubit (``dim == 2``)
            registers.

    Returns:
        A ``qutip_qip.circuit.QubitCircuit`` mirroring the program's gates,
        measurements, resets, barriers, and feedforward conditions.

    Raises:
        ValueError: If any quantum register has ``dim != 2``.
        ImportError: If ``qutip-qip`` is not installed.
    """
    qubit_circuit_cls = _require_qutip()

    # QuTiP-QIP has no qudit support; reject rather than silently mis-draw.
    for register in program.quantum_registers:
        if register.dim != 2:
            raise ValueError(
                "fatqat.draw renders qubit circuits only; register "
                f"{register.name!r} has dim={register.dim}"
            )

    qubit_index, clbit_index = _wire_maps(program)
    circuit = qubit_circuit_cls(len(qubit_index), num_cbits=len(clbit_index))

    for step in program.operations:
        # Measurement is a distinct instruction type (not an AppliedOperation);
        # emit one QuTiP measurement per (qubit -> clbit) pair.
        if isinstance(step, Measurement):
            for target, output in zip(step.targets, step.outputs):
                circuit.add_measurement(
                    "M",
                    targets=_wire(target, qubit_index),
                    classical_store=_wire(output, clbit_index),
                )
            continue
        # Applied gate/reset/barrier - expand any grouped target first so each
        # emitted element acts on scalar wires.
        for operands in _expand_targets(step.targets):
            _add_operation(circuit, step, operands, qubit_index, clbit_index)
    return circuit


def _add_operation(circuit, step, operands, qubit_index, clbit_index):
    """Add one scalar operation occurrence to the QuTiP circuit."""
    operation = step.operation
    wires = [_wire(ref, qubit_index) for ref in operands]

    # A feedforward condition draws as a classical control wire. QuTiP's
    # ``classical_controls`` records only *which* clbits gate the operation,
    # not the value they must equal, so a ``(clbit, 0)`` term or a multi-term
    # AND is drawn as an ordinary classical control - the diagram cannot show
    # the exact predicate, only that the operation is classically conditioned.
    classical_controls = (
        [_wire(ref, clbit_index) for ref, _ in step.condition]
        if step.condition
        else None
    )

    # Barrier: QuTiP-QIP has no barrier primitive, so draw it as a labeled box
    # spanning its wires. (A dashed separator would require reaching into
    # QuTiP's private layout and has no text-renderer equivalent, so a box is
    # used for a uniform result across both renderers.)
    if isinstance(operation, BarrierGate):
        circuit.add_gate("barrier", targets=wires)
        return

    # Reset: also no native primitive; draw a small ``|0>`` box per target.
    if isinstance(operation, ResetGate):
        for wire in wires:
            circuit.add_gate("|0>", targets=wire)
        return

    native = _NATIVE_GATES.get(operation.name)
    if native is not None:
        qutip_name, n_controls, param_attr = native
        arg_value = getattr(operation, param_attr) if param_attr else None
        circuit.add_gate(
            qutip_name,
            controls=wires[:n_controls] or None,  # [] -> None for uncontrolled
            targets=wires[n_controls:],
            arg_value=arg_value,
            classical_controls=classical_controls,
        )
        return

    # Any other operation - a user-defined custom gate, a gate QuTiP lacks
    # (e.g. Sdg, Tdg, CY, CS), or a qudit gate that slipped through - is drawn
    # as a labeled box carrying its own name. This is what keeps drawing total:
    # an unknown gate produces a named box instead of an error.
    circuit.add_gate(
        operation.name,
        targets=wires,
        classical_controls=classical_controls,
    )


def draw(program: Program, renderer: str = "matplotlib", **kwargs: Any):
    """Render a program's circuit diagram.

    Args:
        program: The program to draw.
        renderer: ``"matplotlib"`` (default) returns a matplotlib ``Figure`` -
            save it yourself with ``fig.savefig("circuit.png")``; ``"text"``
            returns the terminal diagram as a ``str``. Any other renderer name
            (e.g. ``"latex"``) is forwarded to QuTiP-QIP unchanged.
        **kwargs: Forwarded to the QuTiP renderer (e.g. ``dpi``, ``theme``,
            ``title`` for matplotlib).

    Returns:
        A matplotlib ``Figure`` for ``"matplotlib"``, a ``str`` for ``"text"``,
        or whatever QuTiP-QIP's ``draw`` returns for any other renderer.
    """
    circuit = to_qubit_circuit(program)

    if renderer == "text":
        # QuTiP's text renderer prints to stdout and returns None; capture it
        # so the caller gets the diagram as a string to print or save.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            circuit.draw("text", **kwargs)
        return buffer.getvalue()

    if renderer == "matplotlib":
        # ``QubitCircuit.draw("matplotlib")`` calls ``plt.show()`` and returns
        # None, which is no good for "return a Figure the caller saves". Use
        # the underlying ``MatRenderer`` (which exposes ``.fig``) and suppress
        # its ``plt.show()`` so nothing pops up and the Figure is returned.
        from qutip_qip.circuit.mat_renderer import MatRenderer
        import matplotlib.pyplot as plt

        original_show = plt.show
        plt.show = lambda *args, **kw: None
        try:
            mat_renderer = MatRenderer(circuit, **kwargs)
            mat_renderer.canvas_plot()
        finally:
            plt.show = original_show
        return mat_renderer.fig

    return circuit.draw(renderer, **kwargs)
