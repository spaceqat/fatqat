"""Draw FATQAT programs through QuTiP-QIP.

:meth:`fatqat.Program.draw` is the normal entry point. Use
:func:`to_qubit_circuit` when you need the intermediate QuTiP-QIP circuit and
its renderer options. Conversion is one-way and intended only for drawing;
custom-operation placeholders in the returned circuit are not executable.

QuTiP-QIP is imported only when a drawing function is called, so importing
FATQAT does not load the rendering stack.
"""

from __future__ import annotations

import contextlib
import functools
import io
from typing import TYPE_CHECKING, Any

from .. import operations as ops
from ..errors import UnsupportedOperationError
from ..operations import Measurement, PulseOperation
from ..registers import RegisterRef, RegisterView, _view_members

if TYPE_CHECKING:
    from ..program import Program

# Map exact fatqat built-in operation types to their native QuTiP-QIP gate.
# Using types instead of public ``Operation.name`` strings prevents a custom
# operation that happens to reuse a built-in display name from being silently
# reinterpreted as that built-in.
#
# Each value is
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
_NATIVE_GATES: dict[type[ops.Operation], tuple[str, int, str | None]] = {
    type(ops.H): ("SNOT", 0, None),
    type(ops.X): ("X", 0, None),
    type(ops.Y): ("Y", 0, None),
    type(ops.Z): ("Z", 0, None),
    type(ops.S): ("S", 0, None),
    type(ops.T): ("T", 0, None),
    type(ops.SX): ("SQRTNOT", 0, None),
    ops.RX: ("RX", 0, "theta"),
    ops.RY: ("RY", 0, "theta"),
    ops.RZ: ("RZ", 0, "theta"),
    ops.Phase: ("PHASEGATE", 0, "theta"),
    type(ops.CX): ("CNOT", 1, None),
    type(ops.CY): ("CY", 1, None),
    type(ops.CZ): ("CZ", 1, None),
    type(ops.CS): ("CS", 1, None),
    ops.CPhase: ("CPHASE", 1, "theta"),
    type(ops.Swap): ("SWAP", 0, None),
    type(ops.iSwap): ("ISWAP", 0, None),
    type(ops.CCX): ("TOFFOLI", 2, None),
    type(ops.CSwap): ("FREDKIN", 1, None),
}


def _require_qutip():
    """Import QuTiP-QIP's ``QubitCircuit``, or raise a clear repair hint.

    Imported lazily so that importing fatqat does not load the drawing stack.
    """
    try:
        from qutip_qip.circuit import QubitCircuit
    except ImportError as exc:  # pragma: no cover - only hit without qutip-qip
        raise ImportError(
            "fatqat requires 'qutip-qip' for circuit drawing; "
            "reinstall fatqat to restore its required dependencies."
        ) from exc
    return QubitCircuit


# --- QuTiP-QIP API compatibility ---------------------------------------------
#
# QuTiP-QIP is mid-refactor: the released package and its master branch want
# gates described in two different ways, and the difference is not cosmetic.
#
#   released (<= 0.4.x)  add_gate("SNOT", targets=[0], arg_value=0.3)
#                        - any string is accepted, including names QuTiP has
#                          never heard of, which is how custom gates, barriers
#                          and resets are drawn as labeled boxes.
#
#   master (0.5.0.dev)   add_gate(H, targets=[0]) / add_gate(RZ(0.3), ...)
#                        - gate *classes* for non-parametric gates, *instances*
#                          for parametric ones. Strings still work for names
#                          QuTiP knows (with a DeprecationWarning), but an
#                          unknown string now raises ValueError - so the
#                          string-based custom-gate boxes break outright.
#                        - `arg_value=` and a string measurement name are
#                          likewise deprecated.
#
# The two are reconciled below by resolving one small "emit" vocabulary per
# installed version. Detection is by capability (does the class registry
# import?), never by version number, so a release that ships the new API is
# picked up automatically.
_API_STRING = "string"  # released QuTiP-QIP: gates named by string
_API_CLASS = "class"  # master QuTiP-QIP: gates given as classes/instances
_BARRIER_RENDER_LABEL = "__fatqat_barrier__"
_BARRIER_STYLE_KEY = "_fatqat_barrier"


@functools.lru_cache(maxsize=1)
def _gate_api() -> str:
    """Return which QuTiP-QIP gate API is installed, resolved once.

    Both versions ship ``qutip_qip.operations.gates``; only the class-based
    one populates it with the old-name -> new-class registry, so the attribute
    (not the module, and not a version number) is the capability probe.
    """
    from qutip_qip.operations import gates

    return _API_CLASS if hasattr(gates, "GATE_CLASS_MAP") else _API_STRING


@functools.lru_cache(maxsize=None)
def _native_gate(qutip_name: str):
    """Resolve one native gate name to the class-based API's gate class.

    ``GATE_CLASS_MAP`` maps exactly the legacy spellings this module already
    uses (``"SNOT"`` -> ``H``, ``"CNOT"`` -> ``CX``, ``"SQRTNOT"`` -> ``SQRTX``
    ...), so `_NATIVE_GATES` needs no second table - only this lookup. The one
    gap is ``"PHASEGATE"``, whose map entry is ``None``; it is resolved to the
    ``PHASE`` class directly.
    """
    from qutip_qip.operations import gates

    gate_class = gates.GATE_CLASS_MAP.get(qutip_name)
    if gate_class is None:
        gate_class = getattr(
            gates, "PHASE" if qutip_name == "PHASEGATE" else qutip_name
        )
    return gate_class


@functools.lru_cache(maxsize=None)
def _box_gate(label: str, num_wires: int):
    """Build a drawing-only gate class for a labeled box (class-based API).

    The class-based API has no way to name a box that is not a registered gate:
    ``add_gate`` rejects unknown strings, and the official escape hatch
    (``get_unitary_gate``) demands - and validates the unitarity of - an actual
    matrix, which a drawing tool does not have and should not need.

    So a minimal concrete ``Gate`` subclass is synthesized instead. ``Gate`` is
    an abstract base whose ``get_qobj`` must be overridden for the class to be
    instantiable at all, but *no QuTiP renderer ever calls it* - a matrix is
    needed to simulate a gate, never to draw one. The override therefore raises
    if anything ever does try to simulate this box, which is the honest
    behavior: it is a picture of a gate, not the gate.

    Cached so repeated occurrences of one gate reuse a single class.
    """
    from qutip_qip.operations import Gate

    class _DrawingBox(Gate):
        __slots__ = ()
        name = label
        num_qubits = num_wires

        # Signature matches the class-based API's abstract `get_qobj(dtype)`.
        # pylint checks against whichever version is installed, and the
        # released base class declares a different one - hence the disable.
        @staticmethod
        def get_qobj(dtype: str = "dense"):  # pylint: disable=arguments-differ
            raise NotImplementedError(
                f"{label!r} is a drawing-only placeholder and has no matrix"
            )

    return _DrawingBox


def _optional(**kwargs) -> dict:
    """Drop empty ``controls``/``classical_controls`` entries.

    The two APIs disagree on how "no controls" is spelled (the released one
    takes ``None``, the class-based one requires an int or sequence and
    rejects ``None``), so an empty value is omitted entirely and each version
    applies its own default.
    """
    return {key: value for key, value in kwargs.items() if value}


def _add_box(
    circuit,
    label: str,
    wires: list[int],
    classical_controls,
    *,
    arg_label: str | None = None,
    classical_control_value: int | None = None,
    style: dict[str, Any] | None = None,
) -> None:
    """Add a labeled box spanning ``wires``, on either API."""
    if _gate_api() == _API_STRING:
        # Passing a label string through ``QubitCircuit.add_gate`` asks QuTiP
        # to resolve it in its native registry. A custom gate named ``CY``,
        # ``CS`` or any other registered spelling would then be reinterpreted
        # (and can fail arity validation). A generic Gate instance bypasses
        # that lookup and guarantees this remains a plain drawing box.
        from qutip_qip.operations import Gate

        gate = Gate(
            name=label,
            targets=wires,
            arg_label=arg_label,
            classical_control_value=classical_control_value,
            style=style,
            **_optional(classical_controls=classical_controls),
        )
        circuit.add_gate(gate)
        return
    # Class-API gate instances retain their own label rather than an
    # ``arg_label=`` passed to ``add_gate``. Put a conditional box's complete
    # display label on the drawing-only class itself.
    gate = _box_gate(arg_label or label, len(wires))
    optional = _optional(classical_controls=classical_controls)
    if classical_control_value is not None:
        optional["classical_control_value"] = classical_control_value
    if style is not None:
        optional["style"] = style
    circuit.add_gate(
        gate,
        targets=wires,
        **optional,
    )


def _add_native(
    circuit,
    qutip_name: str,
    controls,
    targets,
    arg_value,
    classical_controls,
    *,
    arg_label: str | None = None,
    classical_control_value: int | None = None,
) -> None:
    """Add a gate QuTiP implements natively, on either API."""
    optional = _optional(controls=controls, classical_controls=classical_controls)
    if classical_control_value is not None:
        optional["classical_control_value"] = classical_control_value
    if _gate_api() == _API_STRING:
        circuit.add_gate(
            qutip_name,
            targets=targets,
            arg_value=arg_value,
            arg_label=arg_label,
            **optional,
        )
        return
    # Class-based API: parametric gates are passed as instances carrying their
    # angle (`arg_value=` is deprecated), non-parametric ones as the class.
    gate_class = _native_gate(qutip_name)
    gate = gate_class(arg_value) if arg_value is not None else gate_class
    circuit.add_gate(gate, targets=targets, **optional)


def _add_measurement(circuit, target: int, classical_store: int) -> None:
    """Add a Z-basis measurement, on either API."""
    if _gate_api() == _API_STRING:
        measurement = "M"
    else:
        # A string name is deprecated here too; `Mz` is the Z-basis class the
        # string used to be silently interpreted as. Resolved by attribute
        # because it does not exist in the released package.
        from qutip_qip.operations import measurement as qutip_measurement

        measurement = qutip_measurement.Mz
    circuit.add_measurement(
        measurement, targets=target, classical_store=classical_store
    )


def _mat_renderer_cls():
    """Import ``MatRenderer`` from whichever module path this version uses."""
    try:
        from qutip_qip.circuit.draw.mat_renderer import MatRenderer  # master
    except ImportError:
        from qutip_qip.circuit.mat_renderer import MatRenderer  # released
    return MatRenderer


def _text_renderer_cls():
    """Import ``TextRenderer`` from whichever module path this version uses."""
    try:
        from qutip_qip.circuit.draw.text_renderer import TextRenderer  # master
    except ImportError:
        from qutip_qip.circuit.text_renderer import TextRenderer  # released
    return TextRenderer


def _is_barrier_marker(gate) -> bool:
    """Return whether a QuTiP drawing box represents fatqat's Barrier."""
    return bool((getattr(gate, "style", None) or {}).get(_BARRIER_STYLE_KEY))


def _adapt_legacy_condition_controls(circuit) -> None:
    """Expose 0.4 classical controls to renderers as drawing-only controls.

    QuTiP-QIP 0.4 stores ``classical_controls`` but neither renderer reads the
    field. Negative control indices are safe on its drawing-only ``Gate`` and
    map each cbit into the renderer's combined qbit/cbit layer arrays:
    ``c0 -> -num_cbits``, ..., ``cN -> -1``. The replacement remains local to
    the fresh circuit created by :meth:`fatqat.Program.draw`;
    :func:`to_qubit_circuit` keeps returning its normal semantic translation.
    """
    from qutip_qip.operations import Gate

    for index, gate in enumerate(circuit.gates):
        classical_controls = getattr(gate, "classical_controls", None)
        if not classical_controls:
            continue
        virtual_controls = [
            control - circuit.num_cbits for control in classical_controls
        ]
        circuit.gates[index] = Gate(
            name="_fatqat_conditioned",
            targets=list(gate.targets),
            controls=list(gate.controls or ()) + virtual_controls,
            arg_label=getattr(gate, "arg_label", None) or gate.name,
        )


def _render_matplotlib(circuit, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    axis = kwargs.pop("ax", None)
    owns_figure = axis is None
    if owns_figure:
        figure, axis = plt.subplots()
    else:
        figure = axis.get_figure()

    renderer_base = _mat_renderer_cls()

    class _FatqatMatRenderer(renderer_base):
        """Render fatqat barriers without exposing a QuTiP box primitive."""

        def _draw_barrier(self, gate, layer):
            targets = sorted(gate.targets)
            wires = list(range(targets[0], targets[-1] + 1))
            xskip = self._get_xskip(wires, layer)
            width = max(2 * self.style.gate_pad, self._min_gate_width / 2)
            center = xskip + self.style.gate_margin + width / 2
            extension = self.style.wire_sep / 4
            barrier = Line2D(
                [center, center],
                [
                    (targets[0] + self._cwires) * self.style.wire_sep - extension,
                    (targets[-1] + self._cwires) * self.style.wire_sep + extension,
                ],
                color=self.style.wire_color,
                linestyle="--",
                linewidth=1,
                zorder=self._zorder["gate"],
            )
            self._ax.add_line(barrier)
            self._manage_layers(width, wires, layer, xskip)

        def _draw_singleq_gate(self, gate, layer):
            if _is_barrier_marker(gate):
                self._draw_barrier(gate, layer)
                return
            super()._draw_singleq_gate(gate, layer)

        def _draw_multiq_gate(self, gate, layer):
            if _is_barrier_marker(gate):
                self._draw_barrier(gate, layer)
                return
            super()._draw_multiq_gate(gate, layer)

    renderer = _FatqatMatRenderer(circuit, ax=axis, **kwargs)
    if "end_wire_ext" not in kwargs:
        # QuTiP measures the trailing extension in multiples of layer_sep but
        # uses an absolute start pad. Match those physical lengths so the idle
        # wire around the first and last gates is visually symmetric. Keep an
        # explicit end_wire_ext override available to callers.
        renderer.style.end_wire_ext = renderer._start_pad / renderer.style.layer_sep
    if _gate_api() == _API_STRING:
        renderer._layer_list.update(
            {
                cbit - renderer._cwires: [renderer._start_pad]
                for cbit in range(renderer._cwires)
            }
        )
    if owns_figure:
        figure.set_dpi(renderer.style.dpi)

    # QuTiP hard-codes pyplot layout/show calls. Keep them figure-local and
    # non-interactive while its renderer runs, then restore them immediately.
    original_tight_layout = plt.tight_layout
    original_show = plt.show
    plt.tight_layout = figure.tight_layout
    plt.show = lambda *args, **kw: None
    try:
        renderer.canvas_plot()
    finally:
        plt.tight_layout = original_tight_layout
        plt.show = original_show
    return renderer.fig


def _render_text(circuit, **kwargs) -> str:
    renderer_base = _text_renderer_cls()

    class _FatqatTextRenderer(renderer_base):
        """Render a compact vertical dashed separator for fatqat barriers."""

        _barrier_parts = (" ┊ ", "─┊─", " ┊ ")
        _barrier_width = len(_barrier_parts[0])

        def _draw_singleq_gate(self, gate_name):
            if gate_name == _BARRIER_RENDER_LABEL:
                return self._barrier_parts, self._barrier_width
            return super()._draw_singleq_gate(gate_name)

        def _draw_multiq_gate(self, gate, gate_text):
            if _is_barrier_marker(gate):
                top, middle, bottom = self._barrier_parts
                return (
                    top,
                    middle,
                    middle,
                    middle,
                    bottom,
                ), self._barrier_width
            return super()._draw_multiq_gate(gate, gate_text)

        def _update_target_multiq(self, gate, wire_list, parts):
            if not _is_barrier_marker(gate):
                super()._update_target_multiq(gate, wire_list, parts)
                return
            top, middle, bottom = self._barrier_parts
            for wire in wire_list:
                self._render_strs["top_frame"][wire] += top
                self._render_strs["mid_frame"][wire] += middle
                self._render_strs["bot_frame"][wire] += bottom

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _FatqatTextRenderer(circuit, **kwargs).layout()
    return buffer.getvalue()


def _wire_maps(program: Program) -> tuple[dict, dict]:
    """Assign consecutive global wire indices to every qubit and clbit ref.

    QuTiP's ``QubitCircuit(N)`` numbers wires ``0..N-1``, so each fatqat ``RegisterRef`` needs a single global index. Registers are numbered in declaration order (concatenated), which matches the flat ordering the simulator itself uses - so wire ``k`` in the diagram is subsystem ``k`` in a run. Refs are keyed by ``(id(register), slot)`` because two distinct registers can share a name.
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

    A fatqat operation may target a whole ``RegisterView`` (a broadcast group); a circuit diagram needs one gate per scalar operand, so a view target is expanded the same way the backend expands it for execution - via the library's own ``_view_members``, which defines the deterministic member order. The common case (all targets are plain refs) returns unchanged.
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


def to_qubit_circuit(program: Program, *, _barrier_markers: bool = False):
    """Convert a program to a QuTiP-QIP circuit for drawing.

    Quantum and classical slots become wires in register declaration order.
    Native gates use QuTiP-QIP symbols; gates it cannot draw natively,
    including custom and qudit operations, become boxes labeled with the
    operation name. Measurements, reset, barriers, and classical conditions
    are retained; a barrier appears as a box labeled ``barrier`` (fatqat's
    own `fatqat.Program.draw` renders it as a dashed separator instead), and
    a condition attached to a barrier is not depicted. Register dimensions
    are not shown.

    Use the returned circuit for drawing only. Placeholder gates for non-native operations cannot be simulated with QuTiP-QIP.

    Args:
        program: Program to translate. Qudit registers are accepted, but their dimensions are not visible in the diagram.

    Returns:
        A ``qutip_qip.circuit.QubitCircuit`` ready for QuTiP-QIP's drawing methods.

    Raises:
        ImportError: If QuTiP-QIP is unavailable.
        UnsupportedOperationError: If ``program`` contains a :class:`~fatqat.operations.PulseOperation`.
    """
    qubit_circuit_cls = _require_qutip()
    qubit_index, clbit_index = _wire_maps(program)
    circuit = qubit_circuit_cls(len(qubit_index), num_cbits=len(clbit_index))

    for step in program._instructions:
        # Measurement is distinct from the private applied-operation record;
        # emit one QuTiP measurement per (qubit -> clbit) pair.
        if isinstance(step, Measurement):
            for target, output in zip(step.targets, step.outputs):
                _add_measurement(
                    circuit,
                    _wire(target, qubit_index),
                    _wire(output, clbit_index),
                )
            continue
        if isinstance(step.operation, PulseOperation):
            raise UnsupportedOperationError(
                "PulseOperation is not supported by circuit drawing"
            )
        # Applied gate/reset/barrier - expand any grouped target first so each
        # emitted element acts on scalar wires.
        for operands in _expand_targets(step.targets):
            _add_operation(
                circuit,
                step,
                operands,
                qubit_index,
                clbit_index,
                barrier_markers=_barrier_markers,
            )
    return circuit


def _add_operation(
    circuit, step, operands, qubit_index, clbit_index, *, barrier_markers=False
):
    """Add one scalar operation occurrence to the QuTiP circuit."""
    operation = step.operation
    wires = [_wire(ref, qubit_index) for ref in operands]

    # Preserve both the controlled cbits and their exact predicate. Released
    # QuTiP stores these fields but needs the small renderer adapter above to
    # show them; the newer class API renders its cbit instructions directly.
    condition_terms = (
        [(_wire(ref, clbit_index), value) for ref, value in step.condition]
        if step.condition
        else []
    )
    classical_controls = [wire for wire, _ in condition_terms] or None
    # QuTiP represents binary control literals as an integer whose highest bit
    # corresponds to the first listed cbit. Preserve that semantic value when
    # possible. Higher-dimensional classical values remain explicit in the
    # visible predicate label even though QuTiP has no mixed-radix field.
    classical_control_value = (
        sum(
            value << (len(condition_terms) - index - 1)
            for index, (_, value) in enumerate(condition_terms)
        )
        if condition_terms and all(value in (0, 1) for _, value in condition_terms)
        else None
    )

    def conditioned_label(label: str) -> str | None:
        if not condition_terms:
            return None
        predicate = " & ".join(f"c{wire}={value}" for wire, value in condition_terms)
        return f"{label} if {predicate}"

    # QuTiP-QIP has no barrier primitive. For fatqat's own renderers, add a
    # drawing-only box carrying a private style marker that the adapters
    # replace with a dashed vertical separator. Circuits handed to any other
    # renderer (the public to_qubit_circuit contract, or a forwarded renderer
    # name) get a plain box labeled "barrier" instead, so the private marker
    # label never leaks into user-visible output.
    if isinstance(operation, type(ops.Barrier)):
        if barrier_markers:
            _add_box(
                circuit,
                "barrier",
                wires,
                None,
                arg_label=_BARRIER_RENDER_LABEL,
                style={_BARRIER_STYLE_KEY: True},
            )
        else:
            _add_box(circuit, "barrier", wires, None)
        return

    # Reset: also no native primitive; draw a small ``|0>`` box per target.
    if isinstance(operation, type(ops.Reset)):
        for wire in wires:
            _add_box(
                circuit,
                "|0>",
                [wire],
                classical_controls,
                arg_label=conditioned_label("|0>"),
                classical_control_value=classical_control_value,
            )
        return

    native = _NATIVE_GATES.get(type(operation))
    if native is not None:
        qutip_name, n_controls, param_attr = native
        arg_label = conditioned_label(operation.name)
        # QuTiP-QIP 0.4 requires the legacy semantic name ``SNOT`` for a
        # Hadamard gate and otherwise exposes that spelling in its renderers.
        # Keep the compatible gate name while presenting the conventional H.
        # Remove this override once fatqat requires QuTiP-QIP 0.5 or later,
        # whose class-based gate API names and renders the gate as H directly.
        if arg_label is None and qutip_name == "SNOT":
            arg_label = "H"
        _add_native(
            circuit,
            qutip_name,
            controls=wires[:n_controls] or None,  # [] -> None for uncontrolled
            targets=wires[n_controls:],
            arg_value=getattr(operation, param_attr) if param_attr else None,
            classical_controls=classical_controls,
            arg_label=arg_label,
            classical_control_value=classical_control_value,
        )
        return

    # Any other operation - a user-defined custom gate, a gate QuTiP lacks
    # (e.g. Sdg or Tdg), or a qudit gate (Shift, Clock, Sum, ...) - is
    # drawn as a labeled box carrying its own name. This is what keeps drawing
    # total: an unknown gate produces a named box instead of an error.
    _add_box(
        circuit,
        operation.name,
        wires,
        classical_controls,
        arg_label=conditioned_label(operation.name),
        classical_control_value=classical_control_value,
    )


def _draw_program(program: Program, renderer: str = "matplotlib", **kwargs: Any):
    """Implement :meth:`fatqat.Program.draw` without loading QuTiP eagerly.

    Args:
        program: The program to draw.
        renderer: ``"matplotlib"`` (default) returns a matplotlib ``Figure`` -
            save it yourself with ``fig.savefig("circuit.png")``;
            ``"text"`` returns the terminal diagram as a ``str``. Any other
            renderer name (e.g. ``"latex"``) is forwarded to QuTiP-QIP
            unchanged.
        **kwargs: Forwarded to the QuTiP renderer (e.g. ``dpi``, ``theme``,
            ``title`` for matplotlib).

    Returns:
        A matplotlib ``Figure`` for ``"matplotlib"``, a ``str`` for ``"text"``,
        or whatever QuTiP-QIP's ``draw`` returns for any other renderer.
    """
    if renderer in {"text", "matplotlib"}:
        circuit = to_qubit_circuit(program, _barrier_markers=True)
        if _gate_api() == _API_STRING:
            _adapt_legacy_condition_controls(circuit)

        if renderer == "text":
            return _render_text(circuit, **kwargs)
        return _render_matplotlib(circuit, **kwargs)

    # Foreign renderers receive marker-free circuits; see _add_operation.
    circuit = to_qubit_circuit(program)
    return circuit.draw(renderer, **kwargs)
