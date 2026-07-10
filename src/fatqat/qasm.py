"""OpenQASM <-> fatqat conversion, in both directions.

The parsers/renderers in this module are intentionally independent of
Qiskit or any other quantum SDK -- everything here is a hand-written,
from-scratch translation between OpenQASM 2.0/3.0 text and fatqat's own
circuit language (`fatqat.Program`).

Import direction (OpenQASM -> Program): `from_qasm` / `qasm_to_program`.
Export direction (Program -> OpenQASM): `to_qasm` / `program_to_qasm`.

Every non-trivial gate decomposition used below (u/u2/u3 on the import
side; the qudit-gate dim=2 reductions and the iSwap custom-gate
definition on the export side) was checked by direct matrix
multiplication against the textbook target unitary before being written
down here -- see `tests/test_qasm.py`.

See the `from_qasm` and `to_qasm` docstrings below for the full list of
what each direction supports and does not support.
"""

from __future__ import annotations

import ast
import keyword
import math
import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import operations as ops
from .program import AppliedOperation, Program
from .registers import ClassicalRegister, QuantumRegister, Register, RegisterRef


class QASMTranspileError(ValueError):
    """Raised when OpenQASM input cannot be converted to a fatqat program."""


@dataclass
class _GateDef:
    """A parsed `gate name(params) qargs { body }` definition.

    ``body`` is a list of ``(call_name, param_expr_texts, qarg_names)``
    tuples: the *unevaluated* text of each parameter expression (since it
    may reference this gate's own formal parameters) and the formal qubit
    argument names used at each call site (resolved against this gate's own
    ``qargs`` when the macro is expanded).
    """

    params: list[str]
    qargs: list[str]
    body: list[tuple[str, list[str], list[str]]]


_QASM2_REGISTER_DECL_RE = re.compile(
    r"^(qreg|creg)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$"
)
_QASM3_REGISTER_DECL_RE = re.compile(
    r"^(qubit|bit)\s*(?:\[\s*(\d+)\s*\])?\s+([A-Za-z_]\w*)$"
)
_MEASURE_RE = re.compile(r"^measure\s+(.+?)\s*->\s*(.+)$")
_ASSIGN_MEASURE_RE = re.compile(r"^(.+?)\s*=\s*measure\s+(.+)$")
# Condition text is captured whole (non-greedy up to the first matching ')')
# and parsed separately by `_parse_condition_text`, which supports both the
# QASM2 whole-register form (`c == N`) and the QASM3 bit-level AND form
# (`c[0] == 1 && c[2] == 0`) -- see that function.
_IF_RE = re.compile(r"^if\s*\((.*?)\)\s*(.+)$", re.DOTALL)
_IF_BLOCK_RE = re.compile(r"^if\s*\((.*?)\)\s*\{\s*(.*)\s*\}$", re.DOTALL)
_GATE_RE = re.compile(r"^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*(.*)$")
_REF_RE = re.compile(r"^([A-Za-z_]\w*)(?:\[\s*(\d+)\s*\])?$")
_COND_TERM_RE = re.compile(
    r"^([A-Za-z_]\w*)(?:\[\s*(\d+)\s*\])?\s*(==|!=)\s*(\d+)$"
)
_GATE_DEF_HEADER_RE = re.compile(
    r"^gate\s+([A-Za-z_]\w*)\s*(?:\((.*?)\))?\s*([A-Za-z_][\w\s,]*)$"
)

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MATH_FUNCS: dict[str, Callable[..., float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "sqrt": math.sqrt,
}


def from_qasm(source: str) -> Program:
    """Convert an OpenQASM 2.0 or 3.0 string into a fatqat ``Program``.

    Supported:
        * Register declarations: ``qreg``/``creg`` (QASM2) and
          ``qubit``/``bit`` (QASM3), including whole-register broadcast
          (``h q;``, ``reset q;``).
        * Measurements (``measure q -> c;`` and ``c = measure q;``),
          ``reset``, ``barrier`` (accepted and ignored -- it is a
          scheduling hint with no fatqat equivalent).
        * Classical conditions: the whole-register form
          ``if (creg == integer)`` and the bit-level AND form
          ``if (c[0] == 1 && c[2] == 0)``. A single bare ``if`` (no
          ``else``) is supported.
        * User-defined ``gate name(params) qargs { ... }`` macros, expanded
          inline and recursively at every call site.
        * Built-in gates: ``id``/``x``/``y``/``z``/``h``/``s``/``sdg``/
          ``t``/``tdg``, ``rx``/``ry``/``rz``, ``p``/``phase``/``u1``,
          ``cx``/``cnot``/``cy``/``cz``/``swap``/``ccx``/``toffoli``/
          ``cswap``/``fredkin``, ``cp``/``cu1``, and ``u``/``u2``/``u3``
          (decomposed into ``rz``/``ry``/``rz``, exact up to a global
          phase -- fatqat has no global-phase primitive, and this never
          affects measurement probabilities).

    Not supported (raises ``QASMTranspileError``):
        * Gates with no built-in mapping above and no local ``gate``
          definition -- e.g. ``crz``/``cry``/``crx``, ``ch``, ``cu``/
          ``cu3``, ``sx``/``sxdg``, ``rxx``/``ryy``/``rzz``/``rzx``, and
          multi-controlled gates (``c3x``/``c4x``/``mcx``). These *do*
          work if the QASM source itself provides a local ``gate``
          definition for them built from supported primitives. This
          matters most for QASM exported from real hardware or other
          toolchains, which commonly use ``sx`` and controlled-rotation
          gates as basis gates.
        * ``opaque`` declarations (no body to expand).
        * Classical control flow: ``for``/``while`` loops, the ``else``
          branch of ``if``, subroutines (``def``), and gate modifiers
          (``ctrl @`` / ``inv @`` / ``pow(n) @``).
        * ``||`` (OR) and whole-register ``!=`` inside conditions.

    All unsupported constructs fail loudly rather than silently producing
    an incorrect translation, though a few (``if``/``else``, ``for``, gate
    modifiers) currently surface as a generic parse error rather than a
    message naming the specific construct.
    """

    builder = _QASMBuilder(source)
    return builder.build()


def from_qasm_file(path: str | Path, *, encoding: str = "utf-8") -> Program:
    """Read an OpenQASM 2.0 or 3.0 file and convert it into a fatqat ``Program``."""

    return from_qasm(Path(path).read_text(encoding=encoding))


qasm_to_program = from_qasm


class _QASMBuilder:
    def __init__(self, source: str) -> None:
        self._statements = _split_statements(_strip_comments(source))
        self._qregs: list[QuantumRegister] = []
        self._cregs: list[ClassicalRegister] = []
        self._qreg_by_name: dict[str, QuantumRegister] = {}
        self._creg_by_name: dict[str, ClassicalRegister] = {}
        self._gate_defs: dict[str, _GateDef] = {}
        self._pending: list[tuple[str, Any]] = []
        self._program: Program | None = None
        self._version = "unknown"

    def build(self) -> Program:
        for statement in self._statements:
            self._collect_or_defer(statement)

        self._program = Program(self._qregs, self._cregs)
        self._program.metadata["source"] = f"openqasm{self._version}"

        for kind, payload in self._pending:
            if kind == "measure":
                self._add_measurement(*payload)
            elif kind == "instruction":
                self._add_instruction(payload)
            else:  # pragma: no cover - internal guard
                raise AssertionError(kind)
        return self._program

    @property
    def program(self) -> Program:
        if self._program is None:
            raise AssertionError("program has not been created")
        return self._program

    def _collect_or_defer(self, statement: str) -> None:
        if not statement:
            return
        lowered = statement.lower()
        if lowered.startswith("openqasm "):
            version = statement.split(None, 1)[1].strip()
            if version == "3":
                version = "3.0"
            if version not in {"2.0", "3.0"}:
                raise QASMTranspileError(
                    f"only OPENQASM 2.0 and 3.0 are supported, got {version!r}"
                )
            self._version = version
            return
        if lowered.startswith("include "):
            return
        if lowered.startswith("opaque "):
            raise QASMTranspileError(
                "'opaque' gate declarations have no body and cannot be translated"
            )
        if lowered.startswith("gate "):
            self._add_gate_def(statement)
            return
        if lowered.startswith(("defcal ", "extern ", "input ", "output ", "let ")):
            raise QASMTranspileError(f"unsupported OpenQASM declaration {statement!r}")

        decl = _QASM2_REGISTER_DECL_RE.match(statement)
        if decl:
            self._add_register_decl(*decl.groups())
            return
        decl3 = _QASM3_REGISTER_DECL_RE.match(statement)
        if decl3:
            kind, size_text, name = decl3.groups()
            qasm2_kind = "qreg" if kind == "qubit" else "creg"
            self._add_register_decl(qasm2_kind, name, size_text or "1")
            return

        measure = _MEASURE_RE.match(statement)
        if measure:
            self._pending.append(("measure", measure.groups()))
            return
        assigned_measure = _ASSIGN_MEASURE_RE.match(statement)
        if assigned_measure:
            c_operand, q_operand = assigned_measure.groups()
            self._pending.append(("measure", (q_operand, c_operand)))
            return

        self._pending.append(("instruction", statement))

    def _add_register_decl(self, kind: str, name: str, size_text: str) -> None:
        if name in self._qreg_by_name or name in self._creg_by_name:
            raise QASMTranspileError(f"duplicate register name {name!r}")
        size = int(size_text)
        if size <= 0:
            raise QASMTranspileError(f"register {name!r} must have positive size")
        if kind == "qreg":
            reg = QuantumRegister(size, name=name)
            self._qregs.append(reg)
            self._qreg_by_name[name] = reg
        else:
            reg = ClassicalRegister(size, name=name)
            self._cregs.append(reg)
            self._creg_by_name[name] = reg

    def _add_gate_def(self, statement: str) -> None:
        if "{" not in statement or not statement.rstrip().endswith("}"):
            raise QASMTranspileError(f"malformed gate definition: {statement!r}")
        header, body_text = statement.split("{", 1)
        body_text = body_text.rstrip()[:-1]  # drop the trailing '}'
        m = _GATE_DEF_HEADER_RE.match(header.strip())
        if not m:
            raise QASMTranspileError(f"cannot parse gate definition header {header.strip()!r}")
        name, params_text, qargs_text = m.groups()
        name = name.lower()
        if name in self._gate_defs:
            raise QASMTranspileError(f"gate {name!r} redefined")
        param_names = [p.strip() for p in _split_top_level(params_text or "", ",") if p.strip()]
        qarg_names = [q.strip() for q in _split_top_level(qargs_text, ",") if q.strip()]
        if not qarg_names:
            raise QASMTranspileError(f"gate {name!r} declares no qubit parameters")
        body: list[tuple[str, list[str], list[str]]] = []
        for inner in _split_statements(body_text):
            call = _GATE_RE.match(inner)
            if not call:
                raise QASMTranspileError(f"cannot parse statement {inner!r} inside gate {name!r}")
            call_name, call_params_text, call_qargs_text = call.groups()
            call_param_exprs = (
                [p.strip() for p in _split_top_level(call_params_text, ",") if p.strip()]
                if call_params_text
                else []
            )
            call_qarg_names = [q.strip() for q in _split_top_level(call_qargs_text, ",") if q.strip()]
            body.append((call_name.lower(), call_param_exprs, call_qarg_names))
        self._gate_defs[name] = _GateDef(param_names, qarg_names, body)

    def _add_measurement(self, q_operand: str, c_operand: str) -> None:
        q_refs = self._resolve_operand(q_operand, self._qreg_by_name, "quantum")
        c_refs = self._resolve_operand(c_operand, self._creg_by_name, "classical")
        if len(q_refs) != len(c_refs):
            raise QASMTranspileError("measurement operands must have the same size")
        self.program.add_measurement(_as_operand(q_refs), _as_operand(c_refs))

    def _add_instruction(self, statement: str) -> None:
        condition = None
        conditional = _IF_BLOCK_RE.match(statement)
        if conditional:
            cond_text, block = conditional.groups()
            nested = _split_statements(block)
            if len(nested) != 1:
                raise QASMTranspileError("if blocks must contain exactly one supported instruction")
            statement = nested[0]
            condition = self._parse_condition_text(cond_text)

        conditional = _IF_RE.match(statement) if condition is None else None
        if conditional:
            cond_text, statement = conditional.groups()
            condition = self._parse_condition_text(cond_text)

        gate = _GATE_RE.match(statement)
        if not gate:
            raise QASMTranspileError(f"cannot parse instruction {statement!r}")
        name, params_text, operands_text = gate.groups()
        name = name.lower()

        if name == "barrier":
            return

        params = _parse_params(params_text)
        operand_groups = [
            self._resolve_operand(part, self._qreg_by_name, "quantum")
            for part in _split_top_level(operands_text, ",")
            if part.strip()
        ]
        if not operand_groups:
            raise QASMTranspileError(f"instruction {name!r} has no operands")

        if name == "reset":
            self._apply_gate(ops.Reset, operand_groups, condition=condition)
            return

        width = len(operand_groups[0])
        if any(len(group) != width for group in operand_groups):
            raise QASMTranspileError(f"{name!r} register operands must have equal size")

        expanded = self._expand_gate(name, params, len(operand_groups))
        for operands in zip(*operand_groups):
            for op, positions in expanded:
                targets = tuple(operands[p] for p in positions)
                self.program.add(
                    op, targets[0] if len(targets) == 1 else targets, condition=condition
                )

    def _expand_gate(
        self, name: str, params: tuple[float, ...], n_operands: int, _depth: int = 0
    ) -> list[tuple[Any, tuple[int, ...]]]:
        """Return ``[(operation, operand_positions), ...]``, where each
        ``operand_positions`` tuple indexes into the *caller's* operand list
        (0-based, in the order the gate was called with)."""

        if _depth > 64:
            raise QASMTranspileError(
                f"gate {name!r} is defined recursively (or nested too deeply) and cannot be expanded"
            )

        if name in self._gate_defs:
            gdef = self._gate_defs[name]
            if len(params) != len(gdef.params):
                raise QASMTranspileError(
                    f"gate {name!r} expects {len(gdef.params)} parameter(s), got {len(params)}"
                )
            if n_operands != len(gdef.qargs):
                raise QASMTranspileError(
                    f"gate {name!r} expects {len(gdef.qargs)} qubit(s), got {n_operands}"
                )
            param_env = dict(zip(gdef.params, params))
            qarg_pos = {qname: i for i, qname in enumerate(gdef.qargs)}
            out: list[tuple[Any, tuple[int, ...]]] = []
            for call_name, call_param_exprs, call_qarg_names in gdef.body:
                sub_params = tuple(_eval_angle(expr, param_env) for expr in call_param_exprs)
                try:
                    positions = tuple(qarg_pos[q] for q in call_qarg_names)
                except KeyError as exc:
                    raise QASMTranspileError(
                        f"{exc.args[0]!r} is not a qubit parameter of gate {name!r}"
                    ) from None
                for sub_op, sub_positions in self._expand_gate(
                    call_name, sub_params, len(call_qarg_names), _depth + 1
                ):
                    out.append((sub_op, tuple(positions[p] for p in sub_positions)))
            return out

        return [(op, tuple(range(n_operands))) for op in self._builtin_gate(name, params, n_operands)]

    def _builtin_gate(self, name: str, params: tuple[float, ...], n_operands: int) -> tuple[Any, ...]:
        fixed = {
            "id": ops.I,
            "u0": ops.I,
            "h": ops.H,
            "x": ops.X,
            "y": ops.Y,
            "z": ops.Z,
            "s": ops.S,
            "sdg": ops.Sdg,
            "t": ops.T,
            "tdg": ops.Tdg,
            "cx": ops.CX,
            "cnot": ops.CX,
            "cz": ops.CZ,
            "cy": ops.CY,
            "swap": ops.Swap,
            "ccx": ops.CCX,
            "toffoli": ops.CCX,
            "cswap": ops.CSwap,
            "fredkin": ops.CSwap,
        }
        if name in fixed:
            _require_param_count(name, params, 0)
            op = fixed[name]
            _require_operand_count(name, op.num_subsystems, n_operands)
            return (op,)

        parametric = {
            "rx": (1, ops.RX),
            "ry": (1, ops.RY),
            "rz": (1, ops.RZ),
            "p": (1, ops.Phase),
            "phase": (1, ops.Phase),
            "u1": (1, ops.Phase),
            "cp": (1, ops.CPhase),
            "cu1": (1, ops.CPhase),
        }
        if name in parametric:
            count, factory = parametric[name]
            _require_param_count(name, params, count)
            op = factory(*params)
            _require_operand_count(name, op.num_subsystems, n_operands)
            return (op,)

        if name in {"u", "u3"}:
            _require_param_count(name, params, 3)
            _require_operand_count(name, 1, n_operands)
            theta, phi, lam = params
            # U3(theta,phi,lam) == RZ(phi) . RY(theta) . RZ(lam) as a matrix
            # product, which means RZ(lam) must be applied FIRST and RZ(phi)
            # LAST (gates are applied in time order, matrices compose in the
            # opposite order). The previous version of this code applied
            # RZ(phi) first and RZ(lam) last -- i.e. phi and lam were swapped
            # -- which is only invisible when phi == lam or one of them is 0.
            # Exact up to a global phase e^{i(phi+lam)/2}; fatqat has no
            # global-phase primitive to restore it with, which never affects
            # measurement probabilities.
            return (ops.RZ(lam), ops.RY(theta), ops.RZ(phi))
        if name == "u2":
            _require_param_count(name, params, 2)
            _require_operand_count(name, 1, n_operands)
            phi, lam = params
            return (ops.RZ(lam), ops.RY(math.pi / 2), ops.RZ(phi))

        raise QASMTranspileError(
            f"unsupported gate {name!r} (no built-in mapping and no local 'gate' "
            "definition); define it as a local 'gate' block built from "
            "supported primitives if you need it"
        )

    def _apply_gate(
        self,
        op: Any,
        operand_groups: Sequence[tuple[RegisterRef, ...]],
        *,
        condition: tuple[tuple[RegisterRef, int], ...] | None,
    ) -> None:
        arity = op.num_subsystems
        if arity is None:
            if len(operand_groups) != 1:
                raise QASMTranspileError(f"{op.name} expects one register operand")
            targets: tuple[int | RegisterRef, ...] = tuple(operand_groups[0])
            self.program.add(op, targets, condition=condition)
            return

        if len(operand_groups) != arity:
            raise QASMTranspileError(
                f"{op.name} expects {arity} operand(s), got {len(operand_groups)}"
            )
        width = len(operand_groups[0])
        if any(len(group) != width for group in operand_groups):
            raise QASMTranspileError(f"{op.name} register operands must have equal size")

        for operands in zip(*operand_groups):
            self.program.add(op, tuple(operands), condition=condition)

    def _resolve_operand(
        self,
        text: str,
        registers: dict[str, QuantumRegister] | dict[str, ClassicalRegister],
        kind: str,
    ) -> tuple[RegisterRef, ...]:
        match = _REF_RE.match(text.strip())
        if not match:
            raise QASMTranspileError(f"invalid {kind} operand {text!r}")
        name, index_text = match.groups()
        if name not in registers:
            raise QASMTranspileError(f"unknown {kind} register {name!r}")
        reg = registers[name]
        if index_text is None:
            return tuple(reg[i] for i in range(reg.size))
        return (reg[int(index_text)],)

    def _parse_condition_text(self, cond_text: str) -> tuple[tuple[RegisterRef, int], ...]:
        """Parse an `if (...)` condition into fatqat's AND-of-equalities
        form. Supports the QASM2 whole-register form (`c == N`, decoded into
        every bit of `c`) and the QASM3 bit-level form (`c[0] == 1`), with
        any number of `&&`-joined terms mixing both styles. `||` and `!=` on
        a whole register have no fatqat equivalent and are rejected.
        """
        if "||" in cond_text:
            raise QASMTranspileError("'||' (OR) conditions have no fatqat equivalent")

        resolved: dict[tuple[str, int], int] = {}
        for term_text in cond_text.split("&&"):
            term_text = term_text.strip()
            m = _COND_TERM_RE.match(term_text)
            if not m:
                raise QASMTranspileError(f"unsupported condition term {term_text!r}")
            reg_name, idx_text, op, value_text = m.groups()
            if reg_name not in self._creg_by_name:
                raise QASMTranspileError(f"unknown classical register {reg_name!r}")
            reg = self._creg_by_name[reg_name]
            value = int(value_text)

            if idx_text is None:
                if op == "!=":
                    raise QASMTranspileError(
                        "'!=' on a whole classical register has no fatqat "
                        "AND-condition equivalent (it is an OR over every "
                        "other bit pattern); use per-bit conditions instead"
                    )
                if not 0 <= value < 2**reg.size:
                    raise QASMTranspileError(
                        f"condition value {value} does not fit register {reg_name!r}"
                    )
                for i in range(reg.size):
                    self._merge_condition_term(resolved, reg_name, i, (value >> i) & 1)
            else:
                idx = int(idx_text)
                if not 0 <= idx < reg.size:
                    raise QASMTranspileError(
                        f"bit index {idx} out of range for register {reg_name!r} (size {reg.size})"
                    )
                if value not in (0, 1):
                    raise QASMTranspileError(
                        f"condition on a single bit must compare to 0 or 1, got {value}"
                    )
                bit = (1 - value) if op == "!=" else value
                self._merge_condition_term(resolved, reg_name, idx, bit)

        return tuple((self._creg_by_name[name][i], v) for (name, i), v in resolved.items())

    @staticmethod
    def _merge_condition_term(
        resolved: dict[tuple[str, int], int], reg_name: str, index: int, value: int
    ) -> None:
        key = (reg_name, index)
        if key in resolved and resolved[key] != value:
            raise QASMTranspileError(f"contradictory condition on {reg_name}[{index}]")
        resolved[key] = value


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _split_statements(source: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    depth = 0
    for char in source:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if char == "}" and depth == 0:
                current.append(char)
                statement = "".join(current).strip()
                if statement:
                    statements.append(statement)
                current = []
                continue
        if char == ";" and depth == 0:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(char)
    trailing = "".join(current).strip()
    if trailing:
        raise QASMTranspileError(f"missing semicolon after {trailing!r}")
    return statements


def _parse_params(params_text: str | None) -> tuple[float, ...]:
    if params_text is None:
        return ()
    params_text = params_text.strip()
    if not params_text:
        return ()
    return tuple(_eval_angle(part) for part in _split_top_level(params_text, ","))


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == separator and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _eval_angle(expr: str, env: dict[str, float] | None = None) -> float:
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise QASMTranspileError(f"invalid angle expression {expr!r}") from exc
    return float(_eval_angle_node(tree.body, env or {}))


def _eval_angle_node(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id in env:
            return env[node.id]
        raise QASMTranspileError(f"unknown identifier {node.id!r} in expression")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _eval_angle_node(node.left, env), _eval_angle_node(node.right, env)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_angle_node(node.operand, env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _MATH_FUNCS:
            raise QASMTranspileError(f"unsupported function {node.func.id!r}")
        if node.keywords:
            raise QASMTranspileError("angle functions do not accept keyword arguments")
        return float(_MATH_FUNCS[node.func.id](*(_eval_angle_node(arg, env) for arg in node.args)))
    raise QASMTranspileError(f"unsupported angle expression {ast.unparse(node)!r}")


def _as_operand(refs: tuple[RegisterRef, ...]) -> RegisterRef | tuple[RegisterRef, ...]:
    return refs[0] if len(refs) == 1 else refs


def _require_param_count(name: str, params: tuple[float, ...], expected: int) -> None:
    if len(params) != expected:
        raise QASMTranspileError(
            f"gate {name!r} expects {expected} parameter(s), got {len(params)}"
        )


def _require_operand_count(name: str, expected: int | None, actual: int) -> None:
    if expected is not None and expected != actual:
        raise QASMTranspileError(f"gate {name!r} expects {expected} qubit(s), got {actual}")
# ===========================================================================
# Export direction: fatqat.Program -> OpenQASM source text
# ===========================================================================


class QasmExportError(Exception):
    """Raised when a fatqat program cannot be faithfully represented in QASM."""


# ---------------------------------------------------------------------------
# Identifier sanitising
# ---------------------------------------------------------------------------

# Reserved words in both OpenQASM 2.0 and 3.0 that must not collide with a
# register identifier we generate. Not exhaustive of every future keyword,
# but covers the practically-relevant ones.
_QASM_RESERVED = {
    "openqasm", "include", "qreg", "creg", "qubit", "bit", "gate", "opaque",
    "if", "else", "for", "while", "measure", "reset", "barrier", "u", "cx",
    "pi", "true", "false", "const", "let", "def", "return", "input",
    "output", "extern", "box", "duration", "stretch", "delay", "reset",
    "in", "int", "uint", "float", "angle", "bool", "complex", "array",
}


def _sanitize_identifier(raw: str | None, fallback: str, taken: set[str]) -> str:
    """Turn a user-supplied register name into a safe, unique QASM identifier."""
    name = raw if raw else fallback
    # Replace anything that isn't [A-Za-z0-9_] with underscore.
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"_{cleaned}" if cleaned else fallback
    if cleaned[0] == "_":
        # QASM identifiers conventionally start with a letter; prefix safely.
        cleaned = f"r{cleaned}"
    if keyword.iskeyword(cleaned) or cleaned.lower() in _QASM_RESERVED:
        cleaned = f"{cleaned}_"
    base = cleaned
    suffix = 0
    while cleaned in taken:
        suffix += 1
        cleaned = f"{base}_{suffix}"
    taken.add(cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Register layout: map every fatqat Register to a QASM array name + check dim
# ---------------------------------------------------------------------------

@dataclass
class _RegInfo:
    qasm_name: str
    size: int


class _Layout:
    """Maps fatqat registers to QASM identifiers, and refs to `name[idx]`."""

    def __init__(self, program: Program) -> None:
        self.q_info: dict[int, _RegInfo] = {}
        self.c_info: dict[int, _RegInfo] = {}
        # OpenQASM has a single flat identifier namespace shared by quantum
        # and classical declarations -- `qubit[2] r; bit[2] r;` redeclares
        # `r` and is invalid, and this module's own importer (`from_qasm`)
        # correctly rejects that. Use ONE shared `taken` set here too, or a
        # same-named qreg/creg pair would silently sanitize to the same
        # identifier and produce QASM that is invalid (and that this same
        # module's importer would then reject on round-trip).
        taken: set[str] = set()

        for i, reg in enumerate(program.qreg):
            self._check_dim(reg, "quantum")
            name = _sanitize_identifier(reg.name, f"q{i}", taken)
            self.q_info[id(reg)] = _RegInfo(name, reg.size)

        for i, reg in enumerate(program.creg):
            self._check_dim(reg, "classical")
            name = _sanitize_identifier(reg.name, f"c{i}", taken)
            self.c_info[id(reg)] = _RegInfo(name, reg.size)

    @staticmethod
    def _check_dim(reg: Register, kind: str) -> None:
        if reg.dim != 2:
            label = reg.name if reg.name else "(unnamed)"
            raise QasmExportError(
                f"{kind} register {label!r} (size={reg.size}) has dim={reg.dim}; "
                "OpenQASM has no representation for qudits (dim != 2). Only "
                "qubit/bit programs (dim=2 everywhere) can be exported to QASM."
            )

    def qref(self, ref: RegisterRef) -> str:
        info = self.q_info[id(ref.register)]
        return f"{info.qasm_name}[{ref.index}]"

    def cref(self, ref: RegisterRef) -> str:
        info = self.c_info[id(ref.register)]
        return f"{info.qasm_name}[{ref.index}]"

    def q_declarations(self, version: int) -> list[str]:
        if version == 3:
            return [f"qubit[{i.size}] {i.qasm_name};" for i in self.q_info.values()]
        return [f"qreg {i.qasm_name}[{i.size}];" for i in self.q_info.values()]

    def c_declarations(self, version: int) -> list[str]:
        if version == 3:
            return [f"bit[{i.size}] {i.qasm_name};" for i in self.c_info.values()]
        return [f"creg {i.qasm_name}[{i.size}];" for i in self.c_info.values()]


# ---------------------------------------------------------------------------
# Numeric formatting
# ---------------------------------------------------------------------------

def _fmt(theta: float) -> str:
    """Render a float angle as a compact QASM numeric literal."""
    theta = float(theta)
    # Recognise a few common exact multiples of pi for readability; falls
    # back to a plain float literal otherwise. Purely cosmetic -- either form
    # evaluates to the same value in QASM.
    if theta == 0:
        return "0"
    ratio = theta / math.pi
    if abs(ratio) <= 8:
        from fractions import Fraction
        frac = Fraction(ratio).limit_denominator(64)
        # Only use the cosmetic pi/N form if it reconstructs theta to within
        # float round-off; otherwise this would silently change the angle.
        if abs(theta - float(frac) * math.pi) < 1e-12 * max(1.0, abs(theta)):
            if frac.numerator == 0:
                return "0"
            num, den = frac.numerator, frac.denominator
            core = "pi" if abs(num) == 1 else f"{abs(num)}*pi"
            sign = "-" if num < 0 else ""
            return f"{sign}{core}/{den}" if den != 1 else f"{sign}{core}"
    return repr(theta)


# ---------------------------------------------------------------------------
# Gate lowering: fatqat Operation -> (gate_name, params, needs_custom_gate)
# ---------------------------------------------------------------------------

# A lowering returns either:
#   ("gate", qasm_gate_name, [param_strs])   -- one ordinary gate call
#   ("skip", reason_comment)                 -- operation reduces to identity
# Multi-line/decomposed gates (iSwap) are registered as custom gates instead
# and simply returned as a normal ("gate", "iswap", []) call referencing that
# definition.

_FIXED_GATE_MAP = {
    "H": "h", "I": "id", "S": "s", "Sdg": "sdg", "T": "t", "Tdg": "tdg",
    "X": "x", "Y": "y", "Z": "z",
    "CX": "cx", "CZ": "cz", "Swap": "swap", "CY": "cy",
    "CCX": "ccx", "CSwap": "cswap",
}


def _lower(op: ops.Operation, dim: int) -> tuple[str, ...]:
    """Return a lowering tuple for a single fatqat Operation (dim already
    checked == 2 by the caller)."""
    name = type(op).__name__

    if name in ("HGate", "IGate", "SGate", "SdgGate", "TGate", "TdgGate",
                "XGate", "YGate", "ZGate", "CXGate", "CZGate", "SwapGate",
                "CYGate", "CCXGate", "CSwapGate"):
        return ("gate", _FIXED_GATE_MAP[op.name], [])

    if name == "CSGate":
        # CS = diag(1,1,1,i) == controlled-phase(pi/2).
        return ("gate", "cp", [_fmt(math.pi / 2)])

    if name == "iSwapGate":
        return ("gate", "iswap", [])

    if name == "RX":
        return ("gate", "rx", [_fmt(op.theta)])
    if name == "RY":
        return ("gate", "ry", [_fmt(op.theta)])
    if name == "RZ":
        return ("gate", "rz", [_fmt(op.theta)])
    if name == "Phase":
        return ("gate", "p", [_fmt(op.theta)])
    if name == "CPhase":
        return ("gate", "cp", [_fmt(op.theta)])

    # --- qudit-family gates, valid here only because dim == 2 was verified
    # by the caller; each reduces to a fixed qubit gate per its docstring. ---
    if name == "Shift":
        power = op.power % 2
        return ("gate", "x", []) if power == 1 else ("skip", f"Shift(power={op.power}) is identity at dim=2")
    if name == "Clock":
        power = op.power % 2
        return ("gate", "z", []) if power == 1 else ("skip", f"Clock(power={op.power}) is identity at dim=2")
    if name == "SumGate":
        return ("gate", "cx", [])
    if name == "SwapLevels":
        # Only (0,1)/(1,0) are possible at dim=2; both reduce to X.
        return ("gate", "x", [])
    if name == "FourierGate":
        return ("gate", "h", [])
    if name == "FourierdgGate":
        return ("gate", "h", [])
    if name == "SubspaceRX":
        # Symmetric in (j,k) at dim=2 -- see derivation in accompanying notes.
        return ("gate", "rx", [_fmt(op.theta)])
    if name == "SubspaceRY":
        j, k = op.subspace
        theta = op.theta if (j, k) == (0, 1) else -op.theta
        return ("gate", "ry", [_fmt(theta)])
    if name == "SubspaceRZ":
        j, k = op.subspace
        theta = op.theta if (j, k) == (0, 1) else -op.theta
        return ("gate", "rz", [_fmt(theta)])
    if name == "CClock":
        power = op.power % 2
        return ("gate", "cz", []) if power == 1 else ("skip", f"CClock(power={op.power}) is identity at dim=2")
    # (CClock's class name has no "Gate" suffix, matching Shift/Clock/SwapLevels
    # above -- fatqat's naming convention here is inconsistent with e.g.
    # SumGate/FourierGate, so this matcher is intentionally exhaustive rather
    # than pattern-based.)

    raise QasmExportError(f"no QASM lowering is defined for operation {op.name!r} ({name})")


def _qasm2_lower_name(qasm3_name: str) -> str:
    """A handful of stdgates.inc names differ from qelib1.inc names."""
    return {"cp": "cu1", "p": "u1"}.get(qasm3_name, qasm3_name)


# ---------------------------------------------------------------------------
# Custom gate definitions (emitted once, only if actually used)
# ---------------------------------------------------------------------------

_ISWAP_DEF_QASM3 = (
    "gate iswap a, b {\n"
    "    s a;\n"
    "    s b;\n"
    "    h a;\n"
    "    cx a, b;\n"
    "    cx b, a;\n"
    "    h b;\n"
    "}"
)
_ISWAP_DEF_QASM2 = (
    "gate iswap a, b {\n"
    "    s a;\n"
    "    s b;\n"
    "    h a;\n"
    "    cx a, b;\n"
    "    cx b, a;\n"
    "    h b;\n"
    "}"
)


# ---------------------------------------------------------------------------
# Condition rendering
# ---------------------------------------------------------------------------

def _condition_terms_qasm3(condition, layout: _Layout) -> str:
    parts = [f"{layout.cref(ref)} == {value}" for ref, value in condition]
    return " && ".join(parts)


def _condition_value_qasm2(condition, layout: _Layout):
    """Return (register_qasm_name, integer_value) if `condition` can be
    losslessly expressed as a QASM2 whole-register `if`, else raise.

    Requires every term to reference the *same* classical register and every
    bit of that register to be pinned down by exactly one term.
    """
    # Register is a frozen dataclass with a dict `metadata` field, so it is
    # not hashable -- de-duplicate by `id()` instead of putting it in a set.
    regs_by_id = {id(ref.register): ref.register for ref, _ in condition}
    if len(regs_by_id) != 1:
        raise QasmExportError(
            "QASM 2 cannot express a condition spanning multiple classical "
            "registers; use --version 3 (OpenQASM 3) instead"
        )
    (reg,) = regs_by_id.values()
    seen = {}
    for ref, value in condition:
        if ref.index in seen:
            raise QasmExportError(
                f"condition names clbit {ref.index} of register {reg.name!r} "
                "more than once"
            )
        seen[ref.index] = value
    if len(seen) != reg.size:
        raise QasmExportError(
            f"QASM 2 can only express a condition on register {reg.name!r} "
            f"if it pins down all {reg.size} of its bits (only {len(seen)} "
            "given); use --version 3 (OpenQASM 3) instead, which supports "
            "arbitrary bit-level conditions"
        )
    int_value = sum(seen[i] << i for i in range(reg.size))
    return reg, int_value


# ---------------------------------------------------------------------------
# Main translation
# ---------------------------------------------------------------------------

def to_qasm(program: Program, version: int = 3) -> str:
    """Translate a fatqat `Program` into OpenQASM source text.

    Supported:
        * Fixed gates (H/X/Y/Z/S/Sdg/T/Tdg/I, CX/CY/CZ/Swap/CCX/CSwap),
          parametric gates (RX/RY/RZ/Phase/CPhase), CS (emitted as
          `cp(pi/2)`), and iSwap (emitted as a self-contained custom `gate
          iswap a, b {...}` definition, numerically verified equal to
          fatqat's iSwap).
        * Reset, measurement, and conditions -- QASM 3 supports arbitrary
          bit-level AND conditions; QASM 2 only when the condition pins
          down every bit of a single classical register (see `version`
          below).
        * fatqat's qudit-family gates (Shift/Clock/Sum/SwapLevels/
          Fourier(dg)/SubspaceRX/RY/RZ/CClock) *only* when every register
          involved has `dim == 2` -- each reduces to a standard qubit gate
          in that case (e.g. `Sum` -> `cx`), verified against fatqat's own
          matrix implementations.

    Not supported (raises `QasmExportError`):
        * Any register with `dim != 2` (a qudit) -- OpenQASM has no
          representation for qudits at all, so this is a hard limitation
          of the target format, not just of this function.
        * Any operation with no QASM lowering defined above.
        * QASM 2 export of a condition that does not pin down every bit of
          exactly one classical register (use `version=3` instead, which
          has no such restriction).

    Args:
        program: The fatqat program to translate.
        version: `3` for OpenQASM 3.0 (default, recommended -- supports
            arbitrary bit-level classical conditions), or `2` for OpenQASM
            2.0 (qelib1.inc; whole-register conditions only).

    Returns:
        Complete OpenQASM source text, ready to write to a `.qasm` file.

    Raises:
        QasmExportError: If the program uses a qudit register (dim != 2), an
            operation with no QASM equivalent, or (QASM 2 only) a classical
            condition that cannot be expressed as a whole-register equality.
    """
    if version not in (2, 3):
        raise ValueError(f"version must be 2 or 3, got {version!r}")

    layout = _Layout(program)
    body: list[str] = []
    uses_iswap = False

    for step in program.operations:
        if isinstance(step, ops.Measurement):
            for qref, cref in zip(step.qreg, step.clreg):
                if version == 3:
                    body.append(f"{layout.cref(cref)} = measure {layout.qref(qref)};")
                else:
                    body.append(f"measure {layout.qref(qref)} -> {layout.cref(cref)};")
            continue

        assert isinstance(step, AppliedOperation)
        op = step.operation

        if type(op).__name__ == "ResetGate":
            lines = [f"reset {layout.qref(t)};" for t in step.targets]
        else:
            kind, *rest = _lower(op, dim=2)
            if kind == "skip":
                (reason,) = rest
                body.append(f"// elided: {reason}")
                continue
            _, gate_name, params = (kind, *rest)
            if gate_name == "iswap":
                uses_iswap = True
            qasm_gate = gate_name if version == 3 else _qasm2_lower_name(gate_name)
            arg_list = ", ".join(layout.qref(t) for t in step.targets)
            if params:
                param_list = ", ".join(params)
                lines = [f"{qasm_gate}({param_list}) {arg_list};"]
            else:
                lines = [f"{qasm_gate} {arg_list};"]

        if step.condition is None:
            body.extend(lines)
            continue

        if version == 3:
            cond = _condition_terms_qasm3(step.condition, layout)
            if len(lines) == 1:
                body.append(f"if ({cond}) {{ {lines[0]} }}")
            else:
                body.append(f"if ({cond}) {{")
                body.extend(f"    {ln}" for ln in lines)
                body.append("}")
        else:
            reg, int_value = _condition_value_qasm2(step.condition, layout)
            reg_name = layout.c_info[id(reg)].qasm_name
            for ln in lines:
                body.append(f"if ({reg_name} == {int_value}) {ln}")

    header: list[str]
    if version == 3:
        header = ['OPENQASM 3.0;', 'include "stdgates.inc";']
        if uses_iswap:
            header += ["", _ISWAP_DEF_QASM3]
    else:
        header = ['OPENQASM 2.0;', 'include "qelib1.inc";']
        if uses_iswap:
            header += ["", _ISWAP_DEF_QASM2]

    declarations = layout.q_declarations(version) + layout.c_declarations(version)

    lines_out = header + [""] + declarations + [""] + body
    return "\n".join(lines_out).rstrip() + "\n"


program_to_qasm = to_qasm  # backward-compatible alias
