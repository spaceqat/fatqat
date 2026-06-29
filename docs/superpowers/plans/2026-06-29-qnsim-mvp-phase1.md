# qnsim MVP Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build qnsim MVP Phase 1 — a frontend `Program` object model plus a qubit `StateVectorBackend` that runs the minimal workflow and produces `counts` / `statevector`.

**Architecture:** Two stages. Stage 1 is the backend-free frontend (`registers`, `operations`, `program`) locking the `Program` / `Operation` / `AppliedOperation` / `Measurement` contracts. Stage 2 is `StateVectorBackend.run() -> Job` over `errors`, `layout`, `implementation` (class-keyed matrix rules), a numpy `engine`, and a `result` layer. The frontend stores normalized data only; all execution-support validation lives at the backend entry.

**Tech Stack:** Python 3.13, numpy (only runtime dep), pytest (dev), uv (env + runner), src layout.

## Global Constraints

- Python `>=3.13`; runtime dependency limited to `numpy`; dev dependency `pytest`.
- src layout: package at `src/qnsim/`, tests at `tests/`. Test runner: `uv run pytest`.
- TDD: every task writes the failing test first, runs it red, implements minimally, runs it green, commits.
- Gate `operation.name` is **uppercase** (`"H"`, `"CX"`, `"RX"`).
- Bare `int` operands = **global flat index** across registers in declaration order.
- statevector basis = **little-endian**: amplitude index bit `q` is the value of flat qubit `q` (qubit 0 = least-significant bit).
- counts key = little-endian over clbits: clbit 0 is the **rightmost** char; unwritten clbits stay `0`; same slot → last write wins.
- Multi-qubit gate matrix convention: operand 0 is the **most-significant** bit of the gate's local index (e.g. `CX` = `(control, target)` with control as MSB).
- `git commit` messages: plain, **no `Co-Authored-By` / AI attribution trailer**.
- Errors: `QnsimError → BackendValidationError → UnsupportedOperationError`, plus `ResultFieldUnavailableError`; warning `NoMeasurementWarning(UserWarning)`.
- Out of scope (do NOT implement): `Operation.on()`, feedforward execution, `Reset`, mid-circuit measurement, noise, density matrix, parameter system, `Program.validate()`, `qs.simulate`, `Experiment`.

---

## File Structure

Stage 1 (frontend):
- `src/qnsim/registers.py` — `Register` / `QuantumRegister` / `ClassicalRegister` / `RegisterRef`
- `src/qnsim/operations.py` — `Operation` base, gate classes/instances, `num_qubits`
- `src/qnsim/program.py` — `AppliedOperation`, `Measurement`, `Program`
- `src/qnsim/__init__.py` — public `qs.*` surface

Stage 2 (backend):
- `src/qnsim/errors.py` — exception hierarchy + `NoMeasurementWarning`
- `src/qnsim/layout.py` — `ResourceLayout`
- `src/qnsim/implementation.py` — `MatrixImplementation`, `MatrixRule`, `MatrixImplementationMap`, gate rules, default map
- `src/qnsim/engine.py` — statevector apply + measurement/sampling
- `src/qnsim/result.py` — `ResultConfig`, `Result`, counts builder
- `src/qnsim/job.py` — `Job`
- `src/qnsim/backends.py` — `StateVectorBackend`

Tests mirror each unit under `tests/`.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/qnsim/__init__.py` (placeholder)
- Create: `tests/__init__.py` (empty)
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "qnsim"
version = "0.0.1"
description = "Quantum noisy simulator (MVP Phase 1)"
requires-python = ">=3.13"
dependencies = ["numpy>=2.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/qnsim"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create placeholder package + test package files**

`src/qnsim/__init__.py`:

```python
"""qnsim — quantum noisy simulator (MVP Phase 1)."""

__version__ = "0.0.1"
```

`tests/__init__.py`: empty file.

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
dist/
*.egg-info/
```

- [ ] **Step 3: Sync environment and verify pytest runs**

Run: `uv sync`
Then: `uv run pytest -q`
Expected: uv creates `.venv` with numpy + pytest installed; pytest exits 0 with "no tests ran" (collected 0 items).

> Optional (only if PyPI is slow in CN): set `UV_DEFAULT_INDEX` to a mirror before `uv sync`, e.g. `export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`. Not required for the plan.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/qnsim/__init__.py tests/__init__.py .gitignore
git commit -m "chore: scaffold qnsim package (uv + src layout + pytest)"
```

---

## Task 1: Registers

**Files:**
- Create: `src/qnsim/registers.py`
- Test: `tests/test_registers.py`

**Interfaces:**
- Produces:
  - `Register(size: int, name: str | None = None, metadata: Mapping = {})` frozen; `reg[i] -> RegisterRef`; rejects `size <= 0`.
  - `QuantumRegister(Register)`, `ClassicalRegister(Register)` (subclasses, same fields).
  - `RegisterRef(register: Register, index: int)` frozen.

- [ ] **Step 1: Write the failing test**

`tests/test_registers.py`:

```python
import pytest

from qnsim.registers import (
    Register,
    QuantumRegister,
    ClassicalRegister,
    RegisterRef,
)


def test_getitem_returns_registerref():
    qr = QuantumRegister(3, name="q")
    ref = qr[1]
    assert isinstance(ref, RegisterRef)
    assert ref.register is qr
    assert ref.index == 1


def test_getitem_out_of_range_raises_indexerror():
    qr = QuantumRegister(2)
    with pytest.raises(IndexError):
        qr[2]
    with pytest.raises(IndexError):
        qr[-1]


def test_size_first_construction_and_keyword_name():
    cr = ClassicalRegister(4, name="c")
    assert cr.size == 4
    assert cr.name == "c"


def test_non_positive_size_rejected():
    with pytest.raises(ValueError):
        QuantumRegister(0)
    with pytest.raises(ValueError):
        ClassicalRegister(-1)


def test_registers_are_frozen():
    qr = QuantumRegister(1)
    with pytest.raises(Exception):
        qr.size = 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.registers'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/registers.py`:

```python
"""Register / RegisterRef value objects (frozen)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Register:
    size: int
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise TypeError(f"register size must be int, got {type(self.size)!r}")
        if self.size <= 0:
            raise ValueError(f"register size must be positive, got {self.size}")

    def __getitem__(self, index: int) -> "RegisterRef":
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"register index must be int, got {type(index)!r}")
        if not 0 <= index < self.size:
            raise IndexError(index)
        return RegisterRef(register=self, index=index)


@dataclass(frozen=True)
class QuantumRegister(Register):
    pass


@dataclass(frozen=True)
class ClassicalRegister(Register):
    pass


@dataclass(frozen=True)
class RegisterRef:
    register: Register
    index: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registers.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/registers.py tests/test_registers.py
git commit -m "feat: registers (Register/Quantum/Classical/RegisterRef)"
```

---

## Task 2: Operations

**Files:**
- Create: `src/qnsim/operations.py`
- Test: `tests/test_operations.py`

**Interfaces:**
- Produces:
  - `Operation` base, frozen dataclass; instance property `num_qubits -> int`; class attrs `name: str`, `_num_qubits: int`.
  - Fixed-gate classes `HGate, TGate, XGate, YGate, ZGate, CXGate, CZGate` and pre-built instances `H, T, X, Y, Z, CX, CZ`.
  - Parametric gate classes `RX, RY, RZ` each with field `theta: float`.
  - Names are uppercase; `CX`/`CZ` are 2-qubit, rest 1-qubit.

- [ ] **Step 1: Write the failing test**

`tests/test_operations.py`:

```python
import pytest

from qnsim import operations as ops
from qnsim.operations import Operation


def test_fixed_gate_instances_exist_with_uppercase_names():
    assert ops.H.name == "H"
    assert ops.X.name == "X"
    assert ops.Y.name == "Y"
    assert ops.Z.name == "Z"
    assert ops.T.name == "T"
    assert ops.CX.name == "CX"
    assert ops.CZ.name == "CZ"


def test_num_qubits():
    assert ops.H.num_qubits == 1
    assert ops.X.num_qubits == 1
    assert ops.CX.num_qubits == 2
    assert ops.CZ.num_qubits == 2


def test_parametric_gate_is_class_storing_theta():
    g = ops.RX(0.2)
    assert isinstance(g, Operation)
    assert g.name == "RX"
    assert g.theta == 0.2
    assert g.num_qubits == 1
    assert ops.RY(0.3).name == "RY"
    assert ops.RZ(0.4).name == "RZ"


def test_gates_distinguished_by_class():
    assert type(ops.X) is not type(ops.H)
    assert isinstance(ops.X, Operation)


def test_operations_are_frozen():
    with pytest.raises(Exception):
        ops.RX(0.1).theta = 9.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_operations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.operations'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/operations.py`:

```python
"""Operation base and the Phase 1 gate set, exposed as the `qs.ops` namespace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Operation:
    name: ClassVar[str] = "OP"
    _num_qubits: ClassVar[int] = 1

    def __post_init__(self) -> None:
        n = type(self)._num_qubits
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"_num_qubits must be a positive int, got {n!r}")

    @property
    def num_qubits(self) -> int:
        return type(self)._num_qubits


@dataclass(frozen=True)
class HGate(Operation):
    name: ClassVar[str] = "H"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class TGate(Operation):
    name: ClassVar[str] = "T"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class XGate(Operation):
    name: ClassVar[str] = "X"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class YGate(Operation):
    name: ClassVar[str] = "Y"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class ZGate(Operation):
    name: ClassVar[str] = "Z"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class CXGate(Operation):
    name: ClassVar[str] = "CX"
    _num_qubits: ClassVar[int] = 2


@dataclass(frozen=True)
class CZGate(Operation):
    name: ClassVar[str] = "CZ"
    _num_qubits: ClassVar[int] = 2


@dataclass(frozen=True)
class RX(Operation):
    theta: float
    name: ClassVar[str] = "RX"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class RY(Operation):
    theta: float
    name: ClassVar[str] = "RY"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class RZ(Operation):
    theta: float
    name: ClassVar[str] = "RZ"
    _num_qubits: ClassVar[int] = 1


# Pre-built fixed-gate instances (parametric gates are used as classes: RX(theta)).
H = HGate()
T = TGate()
X = XGate()
Y = YGate()
Z = ZGate()
CX = CXGate()
CZ = CZGate()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_operations.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/operations.py tests/test_operations.py
git commit -m "feat: Operation base and Phase 1 gate set"
```

---

## Task 3: AppliedOperation & Measurement value objects

**Files:**
- Create: `src/qnsim/program.py` (value objects only; `Program` added in Task 4)
- Test: `tests/test_applied_operation.py`

**Interfaces:**
- Produces:
  - `ConditionTerm = tuple[RegisterRef, int]`; `Condition = tuple[ConditionTerm, ...] | None`.
  - `AppliedOperation(operation, targets: tuple[RegisterRef, ...], condition: Condition = None)` frozen; `__post_init__` enforces: `targets` is a tuple, `len(targets) == operation.num_qubits`, every target is a `RegisterRef` to a `QuantumRegister`.
  - `Measurement(qreg: RegisterRef, clreg: RegisterRef, metadata: Mapping = {})` frozen.

- [ ] **Step 1: Write the failing test**

`tests/test_applied_operation.py`:

```python
import pytest

from qnsim.registers import QuantumRegister, ClassicalRegister
from qnsim import operations as ops
from qnsim.program import AppliedOperation, Measurement


def test_applied_operation_accepts_correct_arity():
    qr = QuantumRegister(2)
    ao = AppliedOperation(operation=ops.CX, targets=(qr[0], qr[1]))
    assert ao.operation is ops.CX
    assert ao.targets == (qr[0], qr[1])
    assert ao.condition is None


def test_applied_operation_wrong_arity_raises():
    qr = QuantumRegister(2)
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.X, targets=(qr[0], qr[1]))  # X is 1-qubit
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.CX, targets=(qr[0],))  # CX is 2-qubit


def test_applied_operation_targets_must_be_quantum():
    cr = ClassicalRegister(1)
    with pytest.raises(TypeError):
        AppliedOperation(operation=ops.X, targets=(cr[0],))


def test_applied_operation_targets_must_be_tuple():
    qr = QuantumRegister(1)
    with pytest.raises(TypeError):
        AppliedOperation(operation=ops.X, targets=[qr[0]])  # list, not tuple


def test_measurement_fields():
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    m = Measurement(qreg=qr[0], clreg=cr[0])
    assert m.qreg == qr[0]
    assert m.clreg == cr[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_applied_operation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.program'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/program.py`:

```python
"""Program container plus AppliedOperation / Measurement value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .operations import Operation
from .registers import QuantumRegister, RegisterRef

ConditionTerm = tuple[RegisterRef, int]
Condition = tuple[ConditionTerm, ...] | None


@dataclass(frozen=True)
class AppliedOperation:
    operation: Operation
    targets: tuple[RegisterRef, ...]
    condition: Condition = None

    def __post_init__(self) -> None:
        if not isinstance(self.targets, tuple):
            raise TypeError("targets must be a tuple of RegisterRef")
        expected = self.operation.num_qubits
        if len(self.targets) != expected:
            raise ValueError(
                f"{self.operation.name} expects {expected} target(s), "
                f"got {len(self.targets)}"
            )
        for t in self.targets:
            if not isinstance(t, RegisterRef):
                raise TypeError(f"target must be RegisterRef, got {type(t)!r}")
            if not isinstance(t.register, QuantumRegister):
                raise TypeError("operation targets must reference a QuantumRegister")


@dataclass(frozen=True)
class Measurement:
    qreg: RegisterRef
    clreg: RegisterRef
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_applied_operation.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/program.py tests/test_applied_operation.py
git commit -m "feat: AppliedOperation and Measurement value objects"
```

---

## Task 4: Program construction & flat-index resolution

**Files:**
- Modify: `src/qnsim/program.py` (add `Program` class)
- Test: `tests/test_program_construction.py`

**Interfaces:**
- Produces:
  - `Program(qreg: int | list[QuantumRegister], clreg: int | list[ClassicalRegister] = 0, *, metadata=None)`.
    - `int` expands to one register of that size; `int == 0` → no register (empty list).
  - `Program.registers(*, qreg, clreg=None, metadata=None)` classmethod.
  - attrs: `program.qreg: list[QuantumRegister]`, `program.creg: list[ClassicalRegister]`, `program.operations: list`, `program.metadata: dict`.
  - internal resolvers `_resolve_qubit(operand) -> RegisterRef`, `_resolve_clbit(operand) -> RegisterRef` (consumed by Tasks 5–7).

- [ ] **Step 1: Write the failing test**

`tests/test_program_construction.py`:

```python
import pytest

from qnsim.program import Program
from qnsim.registers import QuantumRegister, ClassicalRegister


def test_int_construction_creates_default_registers():
    p = Program(2, 2)
    assert len(p.qreg) == 1 and p.qreg[0].size == 2
    assert len(p.creg) == 1 and p.creg[0].size == 2
    assert p.operations == []


def test_zero_classical_means_no_classical_register():
    p = Program(2)
    assert len(p.qreg) == 1
    assert p.creg == []


def test_registers_classmethod_with_explicit_registers():
    qr = QuantumRegister(3, name="data")
    cr = ClassicalRegister(2, name="ro")
    p = Program.registers(qreg=[qr], clreg=[cr])
    assert p.qreg == [qr]
    assert p.creg == [cr]


def test_flat_qubit_resolution_across_registers():
    qr0 = QuantumRegister(2, name="a")
    qr1 = QuantumRegister(2, name="b")
    p = Program.registers(qreg=[qr0, qr1])
    assert p._resolve_qubit(0) == qr0[0]
    assert p._resolve_qubit(1) == qr0[1]
    assert p._resolve_qubit(2) == qr1[0]
    assert p._resolve_qubit(3) == qr1[1]


def test_flat_qubit_resolution_out_of_range_raises():
    p = Program(2)
    with pytest.raises(IndexError):
        p._resolve_qubit(2)


def test_resolve_qubit_rejects_foreign_ref():
    p = Program(2)
    foreign = QuantumRegister(2, name="other")
    with pytest.raises(ValueError):
        p._resolve_qubit(foreign[0])


def test_metadata_defaults_to_empty_dict():
    p = Program(1)
    assert p.metadata == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_program_construction.py -v`
Expected: FAIL with `ImportError: cannot import name 'Program'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/qnsim/program.py`:

```python
from .registers import ClassicalRegister, Register


class Program:
    def __init__(
        self,
        qreg: int | list[QuantumRegister],
        clreg: int | list[ClassicalRegister] = 0,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.qreg: list[QuantumRegister] = self._coerce_registers(
            qreg, QuantumRegister, "q"
        )
        self.creg: list[ClassicalRegister] = self._coerce_registers(
            clreg, ClassicalRegister, "c"
        )
        self.operations: list[AppliedOperation | Measurement] = []
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

    @classmethod
    def registers(
        cls,
        *,
        qreg: list[QuantumRegister],
        clreg: list[ClassicalRegister] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Program":
        p = cls.__new__(cls)
        p.qreg = list(qreg)
        p.creg = list(clreg) if clreg is not None else []
        p.operations = []
        p.metadata = dict(metadata) if metadata else {}
        return p

    @staticmethod
    def _coerce_registers(spec, cls, default_name):
        if isinstance(spec, int) and not isinstance(spec, bool):
            if spec < 0:
                raise ValueError(f"register count must be >= 0, got {spec}")
            return [cls(spec, name=default_name)] if spec > 0 else []
        return list(spec)

    @staticmethod
    def _flat_from_int(i: int, regs: list[Register]) -> RegisterRef:
        if not isinstance(i, int) or isinstance(i, bool):
            raise TypeError(f"operand must be int or RegisterRef, got {type(i)!r}")
        if i < 0:
            raise IndexError(i)
        remaining = i
        for reg in regs:
            if remaining < reg.size:
                return reg[remaining]
            remaining -= reg.size
        raise IndexError(i)

    def _resolve_ref(self, operand, regs, kind_cls, kind_name) -> RegisterRef:
        if isinstance(operand, RegisterRef):
            if not isinstance(operand.register, kind_cls):
                raise TypeError(f"expected a {kind_name} ref")
            if not any(operand.register is r for r in regs):
                raise ValueError(f"ref does not belong to this program's {kind_name}s")
            return operand
        return self._flat_from_int(operand, regs)

    def _resolve_qubit(self, operand) -> RegisterRef:
        return self._resolve_ref(operand, self.qreg, QuantumRegister, "quantum register")

    def _resolve_clbit(self, operand) -> RegisterRef:
        return self._resolve_ref(
            operand, self.creg, ClassicalRegister, "classical register"
        )
```

> Note: keep these new imports merged with the existing import block at the top of the file (don't create a duplicate `from .registers import` line — extend the existing one to include `ClassicalRegister, Register`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_program_construction.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/program.py tests/test_program_construction.py
git commit -m "feat: Program construction and flat-index resolution"
```

---

## Task 5: Program.add

**Files:**
- Modify: `src/qnsim/program.py` (add `Program.add`)
- Test: `tests/test_program_add.py`

**Interfaces:**
- Produces: `Program.add(op: Operation, qreg: int | RegisterRef | tuple[int | RegisterRef, ...], *, condition=None) -> None`. Single operand may be bare `int`/`RegisterRef`; multiple operands must be a `tuple`. In-place append of an `AppliedOperation`. (condition is accepted but normalized in Task 7 — until then pass it straight through as `None` only.)

- [ ] **Step 1: Write the failing test**

`tests/test_program_add.py`:

```python
import pytest

from qnsim.program import Program, AppliedOperation
from qnsim import operations as ops


def test_add_single_operand_int():
    p = Program(2)
    p.add(ops.H, 0)
    assert len(p.operations) == 1
    ao = p.operations[0]
    assert isinstance(ao, AppliedOperation)
    assert ao.operation is ops.H
    assert ao.targets == (p.qreg[0][0],)


def test_add_returns_none_and_mutates_in_place():
    p = Program(1)
    assert p.add(ops.X, 0) is None
    assert len(p.operations) == 1


def test_add_multi_operand_tuple():
    p = Program(2)
    p.add(ops.CZ, (0, 1))
    ao = p.operations[0]
    assert ao.targets == (p.qreg[0][0], p.qreg[0][1])


def test_add_parametric_gate():
    p = Program(1)
    p.add(ops.RX(0.2), 0)
    assert p.operations[0].operation.theta == 0.2


def test_add_rejects_variadic_positional():
    p = Program(2)
    with pytest.raises(TypeError):
        p.add(ops.CZ, 0, 1)  # variadic not supported


def test_add_wrong_arity_raises():
    p = Program(2)
    with pytest.raises(ValueError):
        p.add(ops.CZ, 0)  # CZ needs 2 targets


def test_add_rejects_non_operation():
    p = Program(1)
    with pytest.raises(TypeError):
        p.add(ops.RX, 0)  # passed the class, not an instance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_program_add.py -v`
Expected: FAIL with `AttributeError: 'Program' object has no attribute 'add'`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Program` class in `src/qnsim/program.py`:

```python
    def add(
        self,
        op: Operation,
        qreg: int | RegisterRef | tuple[int | RegisterRef, ...],
        *,
        condition=None,
    ) -> None:
        if not isinstance(op, Operation):
            raise TypeError(
                f"op must be an Operation instance, got {type(op)!r} "
                "(did you forget to call a parametric gate, e.g. ops.RX(0.2)?)"
            )
        operands = qreg if isinstance(qreg, tuple) else (qreg,)
        targets = tuple(self._resolve_qubit(o) for o in operands)
        normalized = self._normalize_condition(condition)
        self.operations.append(
            AppliedOperation(operation=op, targets=targets, condition=normalized)
        )

    def _normalize_condition(self, condition):
        # Full normalization arrives in Task 7; until then only None is supported.
        if condition is None:
            return None
        raise NotImplementedError("condition normalization not yet implemented")
```

> `test_add_rejects_variadic_positional` passes because `add` has no `*args`; the extra positional `1` raises `TypeError` from Python's call machinery.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_program_add.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/program.py tests/test_program_add.py
git commit -m "feat: Program.add with target normalization and arity check"
```

---

## Task 6: Program.add_measurement

**Files:**
- Modify: `src/qnsim/program.py`
- Test: `tests/test_program_measurement.py`

**Interfaces:**
- Produces: `Program.add_measurement(qreg: int | RegisterRef, clreg: int | RegisterRef, *, metadata=None) -> None`. In-place append of a `Measurement` with resolved quantum/classical refs.

- [ ] **Step 1: Write the failing test**

`tests/test_program_measurement.py`:

```python
import pytest

from qnsim.program import Program, Measurement
from qnsim import operations as ops


def test_add_measurement_appends_measurement():
    p = Program(2, 2)
    p.add_measurement(0, 0)
    assert len(p.operations) == 1
    m = p.operations[0]
    assert isinstance(m, Measurement)
    assert m.qreg == p.qreg[0][0]
    assert m.clreg == p.creg[0][0]


def test_add_measurement_returns_none():
    p = Program(1, 1)
    assert p.add_measurement(0, 0) is None


def test_operations_preserve_order_and_type_mix():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    assert len(p.operations) == 4
    assert p.operations[0].operation.name == "H"
    assert isinstance(p.operations[2], Measurement)


def test_add_measurement_rejects_quantum_ref_as_clreg():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.add_measurement(0, p.qreg[0][1])  # quantum ref as classical slot
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_program_measurement.py -v`
Expected: FAIL with `AttributeError: 'Program' object has no attribute 'add_measurement'`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Program` class:

```python
    def add_measurement(
        self,
        qreg: int | RegisterRef,
        clreg: int | RegisterRef,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        q = self._resolve_qubit(qreg)
        c = self._resolve_clbit(clreg)
        self.operations.append(
            Measurement(qreg=q, clreg=c, metadata=dict(metadata) if metadata else {})
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_program_measurement.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/program.py tests/test_program_measurement.py
git commit -m "feat: Program.add_measurement"
```

---

## Task 7: condition normalization

**Files:**
- Modify: `src/qnsim/program.py` (replace `_normalize_condition`, add `_resolve_classical_slot`)
- Test: `tests/test_program_condition.py`

**Interfaces:**
- Produces: `Program._normalize_condition(condition) -> tuple[ConditionTerm, ...] | None`.
  - Public sugar: single `(slot, lit)` or a sequence `((slot, lit), ...)`; discriminator = `condition[0]` is a tuple.
  - Stored canonical form: `tuple[(RegisterRef, int), ...]`, where each slot is resolved to a `ClassicalRegister` ref. Multiple terms = logical AND. Bare-int slot → global classical flat ref; non-classical ref → error.

- [ ] **Step 1: Write the failing test**

`tests/test_program_condition.py`:

```python
import pytest

from qnsim.program import Program
from qnsim import operations as ops


def test_single_condition_normalized_to_and_list():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=(0, 1))
    cond = p.operations[0].condition
    assert cond == ((p.creg[0][0], 1),)


def test_single_condition_with_ref():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=(p.creg[0][1], 0))
    assert p.operations[0].condition == ((p.creg[0][1], 0),)


def test_multiple_conditions_are_conjunction():
    p = Program(2, 2)
    p.add(ops.X, 0, condition=((0, 1), (1, 0)))
    cond = p.operations[0].condition
    assert cond == ((p.creg[0][0], 1), (p.creg[0][1], 0))


def test_no_condition_is_none():
    p = Program(2, 2)
    p.add(ops.X, 0)
    assert p.operations[0].condition is None


def test_condition_rejects_quantum_ref_slot():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.add(ops.X, 0, condition=(p.qreg[0][1], 1))


def test_condition_int_slot_resolves_global_classical_flat():
    cr = []
    from qnsim.registers import QuantumRegister, ClassicalRegister

    p = Program.registers(
        qreg=[QuantumRegister(1)],
        clreg=[ClassicalRegister(2, name="a"), ClassicalRegister(2, name="b")],
    )
    p.add(ops.X, 0, condition=(2, 1))  # flat clbit 2 -> creg[1][0]
    assert p.operations[0].condition == ((p.creg[1][0], 1),)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_program_condition.py -v`
Expected: FAIL — `NotImplementedError: condition normalization not yet implemented`.

- [ ] **Step 3: Write minimal implementation**

In `src/qnsim/program.py`, replace the placeholder `_normalize_condition` with:

```python
    def _normalize_condition(self, condition):
        if condition is None:
            return None
        # Discriminate single term `(slot, lit)` from a sequence of terms.
        terms = condition if isinstance(condition[0], tuple) else (condition,)
        return tuple(
            (self._resolve_classical_slot(slot), int(literal))
            for slot, literal in terms
        )

    def _resolve_classical_slot(self, slot) -> RegisterRef:
        if isinstance(slot, RegisterRef):
            if not isinstance(slot.register, ClassicalRegister):
                raise TypeError("condition slot ref must reference a ClassicalRegister")
            if not any(slot.register is r for r in self.creg):
                raise ValueError("condition slot ref not in this program")
            return slot
        return self._resolve_clbit(slot)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_program_condition.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/program.py tests/test_program_condition.py
git commit -m "feat: condition normalization to canonical AND form"
```

---

## Task 8: Program.copy

**Files:**
- Modify: `src/qnsim/program.py`
- Test: `tests/test_program_copy.py`

**Interfaces:**
- Produces: `Program.copy() -> Program`. Copies `operations`, `qreg`, `creg` lists and the `metadata` dict; value objects shared (immutable).

- [ ] **Step 1: Write the failing test**

`tests/test_program_copy.py`:

```python
from qnsim.program import Program
from qnsim import operations as ops


def test_copy_is_independent_for_operations():
    p = Program(2, 2)
    p.add(ops.H, 0)
    q = p.copy()
    q.add(ops.X, 1)
    assert len(p.operations) == 1
    assert len(q.operations) == 2


def test_copy_isolates_metadata():
    p = Program(1, metadata={"src": "orig"})
    q = p.copy()
    q.metadata["src"] = "changed"
    assert p.metadata["src"] == "orig"


def test_copy_isolates_register_lists():
    p = Program(1)
    q = p.copy()
    assert q.qreg == p.qreg
    assert q.qreg is not p.qreg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_program_copy.py -v`
Expected: FAIL with `AttributeError: 'Program' object has no attribute 'copy'`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Program` class:

```python
    def copy(self) -> "Program":
        new = Program.__new__(Program)
        new.qreg = list(self.qreg)
        new.creg = list(self.creg)
        new.operations = list(self.operations)
        new.metadata = dict(self.metadata)
        return new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_program_copy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/program.py tests/test_program_copy.py
git commit -m "feat: Program.copy with container isolation"
```

---

## Task 9: Frontend public API surface

**Files:**
- Modify: `src/qnsim/__init__.py`
- Test: `tests/test_package_api.py`

**Interfaces:**
- Produces: top-level `qs.Program`, `qs.QuantumRegister`, `qs.ClassicalRegister`, `qs.RegisterRef`, `qs.Measurement`, and `qs.ops` namespace.

- [ ] **Step 1: Write the failing test**

`tests/test_package_api.py`:

```python
import qnsim as qs


def test_top_level_frontend_surface():
    program = qs.Program(2, 2)
    program.add(qs.ops.H, 0)
    program.add(qs.ops.CZ, (0, 1))
    program.add(qs.ops.RX(0.1), 0)
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    assert len(program.operations) == 5
    assert program.operations[0].operation.name == "H"
    assert isinstance(program.operations[3], qs.Measurement)


def test_register_types_exposed():
    qr = qs.QuantumRegister(2, name="q")
    cr = qs.ClassicalRegister(2, name="c")
    p = qs.Program.registers(qreg=[qr], clreg=[cr])
    assert isinstance(qr[0], qs.RegisterRef)
    assert p.qreg == [qr]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_package_api.py -v`
Expected: FAIL with `AttributeError: module 'qnsim' has no attribute 'Program'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/qnsim/__init__.py`:

```python
"""qnsim — quantum noisy simulator (MVP Phase 1)."""

from . import operations as ops
from .program import AppliedOperation, Measurement, Program
from .registers import (
    ClassicalRegister,
    QuantumRegister,
    Register,
    RegisterRef,
)

__version__ = "0.0.1"

__all__ = [
    "ops",
    "Program",
    "AppliedOperation",
    "Measurement",
    "Register",
    "QuantumRegister",
    "ClassicalRegister",
    "RegisterRef",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -v`
Expected: PASS — all Stage 1 tests green (entire suite).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/__init__.py tests/test_package_api.py
git commit -m "feat: expose frontend public API (Stage 1 complete)"
```

> **Stage 1 review checkpoint.** Entire frontend is testable with zero numpy/backend dependency. Confirm the full suite is green before starting Stage 2.

---

## Task 10: Error hierarchy

**Files:**
- Create: `src/qnsim/errors.py`
- Test: `tests/test_errors.py`

**Interfaces:**
- Produces: `QnsimError(Exception)`, `BackendValidationError(QnsimError)`, `UnsupportedOperationError(BackendValidationError)`, `ResultFieldUnavailableError(QnsimError)`, `NoMeasurementWarning(UserWarning)`.

- [ ] **Step 1: Write the failing test**

`tests/test_errors.py`:

```python
from qnsim.errors import (
    QnsimError,
    BackendValidationError,
    UnsupportedOperationError,
    ResultFieldUnavailableError,
    NoMeasurementWarning,
)


def test_hierarchy():
    assert issubclass(BackendValidationError, QnsimError)
    assert issubclass(UnsupportedOperationError, BackendValidationError)
    assert issubclass(ResultFieldUnavailableError, QnsimError)
    assert issubclass(NoMeasurementWarning, UserWarning)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.errors'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/errors.py`:

```python
"""qnsim exception hierarchy and warnings."""

from __future__ import annotations


class QnsimError(Exception):
    """Base class for all qnsim errors."""


class BackendValidationError(QnsimError):
    """Raised at backend entry when a program/request is not acceptable."""


class UnsupportedOperationError(BackendValidationError):
    """Raised when the backend does not support an operation or feature."""


class ResultFieldUnavailableError(QnsimError):
    """Raised when a Result field was not produced by this run."""


class NoMeasurementWarning(UserWarning):
    """Warned when counts contain only never-written clbits and no state is delivered."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_errors.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/errors.py tests/test_errors.py
git commit -m "feat: error hierarchy and NoMeasurementWarning"
```

---

## Task 11: ResourceLayout

**Files:**
- Create: `src/qnsim/layout.py`
- Test: `tests/test_layout.py`

**Interfaces:**
- Produces:
  - `ResourceLayout` with `system_dims: tuple[int, ...]`, `qubit_index(ref) -> int`, `clbit_index(ref) -> int`, `n_qubits` / `n_clbits` properties.
  - `ResourceLayout.from_program(program) -> ResourceLayout` — quantum/classical refs flattened in declaration order; `system_dims = (2,) * n_qubits`.

- [ ] **Step 1: Write the failing test**

`tests/test_layout.py`:

```python
import pytest

from qnsim.layout import ResourceLayout
from qnsim.program import Program
from qnsim.registers import QuantumRegister, ClassicalRegister


def test_single_register_layout():
    p = Program(3, 2)
    layout = ResourceLayout.from_program(p)
    assert layout.system_dims == (2, 2, 2)
    assert layout.n_qubits == 3
    assert layout.n_clbits == 2
    assert layout.qubit_index(p.qreg[0][0]) == 0
    assert layout.qubit_index(p.qreg[0][2]) == 2
    assert layout.clbit_index(p.creg[0][1]) == 1


def test_multi_register_flat_concatenation():
    qa = QuantumRegister(2, name="a")
    qb = QuantumRegister(2, name="b")
    p = Program.registers(qreg=[qa, qb])
    layout = ResourceLayout.from_program(p)
    assert layout.qubit_index(qa[0]) == 0
    assert layout.qubit_index(qa[1]) == 1
    assert layout.qubit_index(qb[0]) == 2
    assert layout.qubit_index(qb[1]) == 3
    assert layout.system_dims == (2, 2, 2, 2)


def test_unknown_ref_raises():
    p = Program(1)
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout.from_program(p)
    with pytest.raises(KeyError):
        layout.qubit_index(foreign[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.layout'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/layout.py`:

```python
"""ResourceLayout: the single source of truth for flat qubit/clbit indices."""

from __future__ import annotations

from .registers import RegisterRef


class ResourceLayout:
    def __init__(self, system_dims, q_offsets, c_offsets, n_clbits):
        self.system_dims: tuple[int, ...] = system_dims
        self._q_offsets = q_offsets  # id(register) -> base flat index
        self._c_offsets = c_offsets
        self._n_clbits = n_clbits

    @property
    def n_qubits(self) -> int:
        return len(self.system_dims)

    @property
    def n_clbits(self) -> int:
        return self._n_clbits

    def qubit_index(self, ref: RegisterRef) -> int:
        try:
            base = self._q_offsets[id(ref.register)]
        except KeyError:
            raise KeyError("qubit ref not part of this layout") from None
        return base + ref.index

    def clbit_index(self, ref: RegisterRef) -> int:
        try:
            base = self._c_offsets[id(ref.register)]
        except KeyError:
            raise KeyError("clbit ref not part of this layout") from None
        return base + ref.index

    @classmethod
    def from_program(cls, program) -> "ResourceLayout":
        q_offsets: dict[int, int] = {}
        offset = 0
        for reg in program.qreg:
            q_offsets[id(reg)] = offset
            offset += reg.size
        n_qubits = offset

        c_offsets: dict[int, int] = {}
        coffset = 0
        for reg in program.creg:
            c_offsets[id(reg)] = coffset
            coffset += reg.size

        return cls(
            system_dims=(2,) * n_qubits,
            q_offsets=q_offsets,
            c_offsets=c_offsets,
            n_clbits=coffset,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layout.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/layout.py tests/test_layout.py
git commit -m "feat: ResourceLayout flat-index mapping"
```

---

## Task 12: Implementation map (gate matrices + rules)

**Files:**
- Create: `src/qnsim/implementation.py`
- Test: `tests/test_implementation.py`

**Interfaces:**
- Produces:
  - `MatrixImplementation(matrix: np.ndarray, target_indices: tuple[int, ...])` frozen.
  - `MatrixRule = Callable[[AppliedOperation], np.ndarray]`.
  - `MatrixImplementationMap` with `register(op_cls, rule)`, `get(op_cls) -> MatrixRule | None`.
  - `default_implementation_map() -> MatrixImplementationMap` registering all Phase 1 gate classes.
  - Convention: 2-qubit matrices use operand 0 as MSB (`CX` = control-MSB).

- [ ] **Step 1: Write the failing test**

`tests/test_implementation.py`:

```python
import numpy as np
import pytest

from qnsim import operations as ops
from qnsim.implementation import (
    MatrixImplementation,
    MatrixImplementationMap,
    default_implementation_map,
)
from qnsim.program import AppliedOperation
from qnsim.registers import QuantumRegister


def _applied(op, n=2):
    qr = QuantumRegister(n)
    targets = tuple(qr[i] for i in range(op.num_qubits))
    return AppliedOperation(operation=op, targets=targets)


def test_fixed_gate_matrices():
    m = default_implementation_map()
    x = m.get(type(ops.X))(_applied(ops.X))
    assert np.allclose(x, [[0, 1], [1, 0]])
    cz = m.get(type(ops.CZ))(_applied(ops.CZ))
    assert np.allclose(cz, np.diag([1, 1, 1, -1]))
    cx = m.get(type(ops.CX))(_applied(ops.CX))
    assert np.allclose(cx, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def test_h_matrix_is_unitary_and_correct():
    m = default_implementation_map()
    h = m.get(type(ops.H))(_applied(ops.H))
    assert np.allclose(h, np.array([[1, 1], [1, -1]]) / np.sqrt(2))


def test_parametric_rx_reads_theta():
    m = default_implementation_map()
    theta = 0.5
    rx = m.get(ops.RX)(_applied(ops.RX(theta)))
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    assert np.allclose(rx, [[c, -1j * s], [-1j * s, c]])


def test_unregistered_class_returns_none():
    m = MatrixImplementationMap()
    assert m.get(type(ops.X)) is None


def test_matrix_implementation_holds_target_indices():
    impl = MatrixImplementation(matrix=np.eye(2), target_indices=(3,))
    assert impl.target_indices == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_implementation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.implementation'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/implementation.py`:

```python
"""Class-keyed matrix rules. Rules return only the local matrix; indices come from layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import operations as ops
from .operations import Operation
from .program import AppliedOperation

MatrixRule = Callable[[AppliedOperation], np.ndarray]


@dataclass(frozen=True)
class MatrixImplementation:
    matrix: np.ndarray
    target_indices: tuple[int, ...]


# Module-level constant matrices (reused; do not rebuild per call).
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
# 2-qubit, operand 0 = MSB (control), operand 1 = LSB (target).
_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _rx(applied: AppliedOperation) -> np.ndarray:
    theta = applied.operation.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(applied: AppliedOperation) -> np.ndarray:
    theta = applied.operation.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(applied: AppliedOperation) -> np.ndarray:
    theta = applied.operation.theta
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


class MatrixImplementationMap:
    def __init__(self) -> None:
        self._rules: dict[type[Operation], MatrixRule] = {}

    def register(self, op_cls: type[Operation], rule: MatrixRule) -> None:
        self._rules[op_cls] = rule

    def get(self, op_cls: type[Operation]) -> MatrixRule | None:
        return self._rules.get(op_cls)


def default_implementation_map() -> MatrixImplementationMap:
    m = MatrixImplementationMap()
    m.register(ops.XGate, lambda _ao: _X)
    m.register(ops.YGate, lambda _ao: _Y)
    m.register(ops.ZGate, lambda _ao: _Z)
    m.register(ops.HGate, lambda _ao: _H)
    m.register(ops.TGate, lambda _ao: _T)
    m.register(ops.CXGate, lambda _ao: _CX)
    m.register(ops.CZGate, lambda _ao: _CZ)
    m.register(ops.RX, _rx)
    m.register(ops.RY, _ry)
    m.register(ops.RZ, _rz)
    return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_implementation.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/implementation.py tests/test_implementation.py
git commit -m "feat: class-keyed matrix implementation map"
```

---

## Task 13: Engine — statevector apply

**Files:**
- Create: `src/qnsim/engine.py`
- Test: `tests/test_engine_apply.py`

**Interfaces:**
- Produces:
  - `zero_state(n_qubits: int) -> np.ndarray` — `|0...0>` length `2**n`.
  - `apply(state, matrix, targets, n_qubits) -> np.ndarray` — applies a `2**k` matrix to flat `targets` (operand 0 = MSB of the matrix local index), little-endian state (qubit q = bit q).

- [ ] **Step 1: Write the failing test**

`tests/test_engine_apply.py`:

```python
import numpy as np

from qnsim.engine import zero_state, apply


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def test_zero_state():
    assert np.allclose(zero_state(2), [1, 0, 0, 0])


def test_x_on_qubit0():
    s = apply(zero_state(1), _X, (0,), 1)
    assert np.allclose(s, [0, 1])


def test_x_on_qubit0_of_two():
    # little-endian: qubit0 is bit0, so |00> -> |01> at index 1
    s = apply(zero_state(2), _X, (0,), 2)
    assert np.allclose(s, [0, 1, 0, 0])


def test_x_on_qubit1_of_two():
    # qubit1 is bit1, so |00> -> index 2
    s = apply(zero_state(2), _X, (1,), 2)
    assert np.allclose(s, [0, 0, 1, 0])


def test_bell_state_h_then_cx():
    s = zero_state(2)
    s = apply(s, _H, (0,), 2)
    s = apply(s, _CX, (0, 1), 2)  # control=qubit0, target=qubit1
    assert np.allclose(s, [1, 0, 0, 1] / np.sqrt(2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_apply.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.engine'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/engine.py`:

```python
"""Statevector engine: matrix application and measurement sampling."""

from __future__ import annotations

import numpy as np


def zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def apply(state, matrix, targets, n_qubits) -> np.ndarray:
    """Apply a 2**k matrix to flat `targets`.

    Conventions:
    - state is little-endian: amplitude index bit q = value of qubit q.
    - matrix local index: operand 0 is the MSB, operand k-1 the LSB.
    """
    k = len(targets)
    psi = state.reshape((2,) * n_qubits)  # axis p corresponds to qubit (n_qubits-1-p)
    target_axes = [n_qubits - 1 - q for q in targets]
    m = np.asarray(matrix, dtype=complex).reshape((2,) * (2 * k))
    # m axes: [out_0..out_{k-1}, in_0..in_{k-1}]; contract inputs with target axes.
    psi = np.tensordot(m, psi, axes=(list(range(k, 2 * k)), target_axes))
    # Result axes: [out_0..out_{k-1}] + remaining state axes (original relative order).
    remaining = [ax for ax in range(n_qubits) if ax not in target_axes]
    perm = [0] * n_qubits
    for j, ax in enumerate(target_axes):
        perm[ax] = j
    for idx, ax in enumerate(remaining):
        perm[ax] = k + idx
    psi = np.transpose(psi, perm)
    return psi.reshape(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine_apply.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/engine.py tests/test_engine_apply.py
git commit -m "feat: statevector engine apply"
```

---

## Task 14: Engine — measurement sampling & collapse

**Files:**
- Modify: `src/qnsim/engine.py`
- Test: `tests/test_engine_measure.py`

**Interfaces:**
- Produces:
  - `probabilities(state) -> np.ndarray`.
  - `sample_indices(state, shots, rng) -> np.ndarray[int]` — multinomial sample of basis indices.
  - `collapse(state, n_qubits, measured_qubits, rng) -> tuple[np.ndarray, dict[int, int]]` — sample one outcome on `measured_qubits`, project + renormalize, return `(collapsed_state, {qubit: bit})`.

- [ ] **Step 1: Write the failing test**

`tests/test_engine_measure.py`:

```python
import numpy as np

from qnsim.engine import zero_state, apply, probabilities, sample_indices, collapse


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def test_probabilities():
    s = apply(zero_state(1), _H, (0,), 1)
    assert np.allclose(probabilities(s), [0.5, 0.5])


def test_sample_indices_deterministic_state():
    s = zero_state(2)  # always |00> -> index 0
    rng = np.random.default_rng(0)
    idx = sample_indices(s, 100, rng)
    assert idx.shape == (100,)
    assert np.all(idx == 0)


def test_sample_indices_balanced_with_seed():
    s = apply(zero_state(1), _H, (0,), 1)
    rng = np.random.default_rng(42)
    idx = sample_indices(s, 2000, rng)
    frac_one = np.mean(idx == 1)
    assert 0.45 < frac_one < 0.55


def test_collapse_projects_to_basis_state():
    s = apply(zero_state(1), _H, (0,), 1)
    rng = np.random.default_rng(1)
    collapsed, bits = collapse(s, 1, [0], rng)
    outcome = bits[0]
    expected = np.zeros(2, dtype=complex)
    expected[outcome] = 1.0
    assert np.allclose(collapsed, expected)
    assert np.isclose(np.linalg.norm(collapsed), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_measure.py -v`
Expected: FAIL with `ImportError: cannot import name 'probabilities'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/qnsim/engine.py`:

```python
def probabilities(state) -> np.ndarray:
    p = np.abs(state) ** 2
    total = p.sum()
    return p / total if total > 0 else p


def sample_indices(state, shots, rng) -> np.ndarray:
    p = probabilities(state)
    return rng.choice(len(state), size=shots, p=p)


def collapse(state, n_qubits, measured_qubits, rng):
    p = probabilities(state)
    idx = int(rng.choice(len(state), p=p))
    bits = {q: (idx >> q) & 1 for q in measured_qubits}
    arange = np.arange(len(state))
    keep = np.ones(len(state), dtype=bool)
    for q, b in bits.items():
        keep &= (((arange >> q) & 1) == b)
    new = np.where(keep, state, 0.0).astype(complex)
    norm = np.linalg.norm(new)
    if norm > 0:
        new = new / norm
    return new, bits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine_measure.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/engine.py tests/test_engine_measure.py
git commit -m "feat: engine measurement sampling and collapse"
```

---

## Task 15: Result layer (ResultConfig, Result, counts builder)

**Files:**
- Create: `src/qnsim/result.py`
- Test: `tests/test_result.py`

**Interfaces:**
- Produces:
  - `ResultConfig(counts: bool | None = None, statevector: bool | None = None)` frozen; `None` means backend default.
  - `build_counts(indices, n_clbits, measurements) -> dict[str, int]` — `measurements` is a list of `(qubit_flat, clbit_flat)` in program order (later writes override earlier); key is little-endian over clbits, unwritten clbits `0`.
  - `Result(counts=None, statevector=None, available=frozenset())` with `get_counts()` / `get_statevector()` raising `ResultFieldUnavailableError` when unavailable.

- [ ] **Step 1: Write the failing test**

`tests/test_result.py`:

```python
import numpy as np
import pytest

from qnsim.result import ResultConfig, Result, build_counts
from qnsim.errors import ResultFieldUnavailableError


def test_resultconfig_defaults():
    rc = ResultConfig()
    assert rc.counts is None
    assert rc.statevector is None


def test_build_counts_little_endian_key():
    # 2 clbits; measurement maps qubit0->clbit0, qubit1->clbit1
    # index 1 = qubit0=1, qubit1=0 -> clbit0=1,clbit1=0 -> key "01"
    indices = [1, 1, 0]
    counts = build_counts(indices, n_clbits=2, measurements=[(0, 0), (1, 1)])
    assert counts == {"01": 2, "00": 1}


def test_build_counts_unwritten_clbit_stays_zero():
    # 2 clbits but only clbit0 written from qubit0
    indices = [1, 1]
    counts = build_counts(indices, n_clbits=2, measurements=[(0, 0)])
    assert counts == {"01": 2}


def test_build_counts_last_write_wins():
    # both measurements target clbit0; second uses qubit1
    # index 2 = qubit1=1, qubit0=0 -> clbit0 set by qubit1 -> 1 -> key "01"
    counts = build_counts([2], n_clbits=2, measurements=[(0, 0), (1, 0)])
    assert counts == {"01": 1}


def test_result_get_counts_available():
    r = Result(counts={"00": 5}, available=frozenset({"counts"}))
    assert r.get_counts() == {"00": 5}


def test_result_get_counts_unavailable_raises():
    r = Result(available=frozenset())
    with pytest.raises(ResultFieldUnavailableError):
        r.get_counts()


def test_result_get_statevector_unavailable_raises():
    r = Result(available=frozenset({"counts"}))
    with pytest.raises(ResultFieldUnavailableError):
        r.get_statevector()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_result.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.result'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/result.py`:

```python
"""Result and ResultConfig, plus the counts assembly helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ResultFieldUnavailableError


@dataclass(frozen=True)
class ResultConfig:
    counts: bool | None = None
    statevector: bool | None = None


def build_counts(indices, n_clbits, measurements) -> dict[str, int]:
    """Tally counts from sampled basis indices.

    measurements: list of (qubit_flat, clbit_flat) in program order; later writes
    to the same clbit override earlier ones. Key is little-endian (clbit 0 rightmost),
    unwritten clbits stay 0.
    """
    counts: dict[str, int] = {}
    for idx in indices:
        idx = int(idx)
        clbits = [0] * n_clbits
        for q, c in measurements:
            clbits[c] = (idx >> q) & 1
        key = "".join(str(clbits[c]) for c in range(n_clbits - 1, -1, -1))
        counts[key] = counts.get(key, 0) + 1
    return counts


class Result:
    def __init__(self, counts=None, statevector=None, available=frozenset()):
        self._counts = counts
        self._statevector = statevector
        self.available_data = frozenset(available)

    def get_counts(self) -> dict[str, int]:
        if "counts" not in self.available_data:
            raise ResultFieldUnavailableError("counts not available in this result")
        return self._counts

    def get_statevector(self) -> np.ndarray:
        if "statevector" not in self.available_data:
            raise ResultFieldUnavailableError(
                "statevector not available in this result"
            )
        return self._statevector
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_result.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/result.py tests/test_result.py
git commit -m "feat: Result, ResultConfig, counts builder"
```

---

## Task 16: Job

**Files:**
- Create: `src/qnsim/job.py`
- Test: `tests/test_job.py`

**Interfaces:**
- Produces: `Job` with `status` in `{"DONE", "ERROR"}`; `Job.done(result)` / `Job.failed(exc)` constructors; `result()` returns the `Result` when DONE or re-raises the stored exception when ERROR.

- [ ] **Step 1: Write the failing test**

`tests/test_job.py`:

```python
import pytest

from qnsim.job import Job


def test_done_job_returns_result():
    job = Job.done("RESULT")
    assert job.status == "DONE"
    assert job.result() == "RESULT"


def test_error_job_reraises():
    job = Job.failed(ValueError("boom"))
    assert job.status == "ERROR"
    with pytest.raises(ValueError, match="boom"):
        job.result()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_job.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.job'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/job.py`:

```python
"""Eager Job handle: DONE carries a Result, ERROR re-raises on result()."""

from __future__ import annotations


class Job:
    def __init__(self, status: str, result=None, error: BaseException | None = None):
        self.status = status
        self._result = result
        self._error = error

    @classmethod
    def done(cls, result) -> "Job":
        return cls(status="DONE", result=result)

    @classmethod
    def failed(cls, error: BaseException) -> "Job":
        return cls(status="ERROR", error=error)

    def result(self):
        if self.status == "ERROR":
            raise self._error
        return self._result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_job.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/job.py tests/test_job.py
git commit -m "feat: eager Job handle"
```

---

## Task 17: StateVectorBackend — validation + counts execution

**Files:**
- Create: `src/qnsim/backends.py`
- Test: `tests/test_backend.py`

**Interfaces:**
- Consumes: `ResourceLayout.from_program`, `default_implementation_map`, `engine.zero_state/apply/sample_indices`, `result.build_counts/Result/ResultConfig`, `Job`, errors.
- Produces:
  - `StateVectorBackend(*, seed=None)` with `resolve_layout(program)` and `run(program, *, shots=1024, result_config=None) -> Job`.
  - Validation raised directly from `run()`: unsupported op → `UnsupportedOperationError`; any `condition` → `UnsupportedOperationError`; gate after a measurement (mid-circuit) → `UnsupportedOperationError`; sampling with `shots <= 0` → `BackendValidationError`.
  - This task covers the counts path (statevector defaults handled in Task 18).

- [ ] **Step 1: Write the failing test**

`tests/test_backend.py`:

```python
import pytest

import qnsim as qs
from qnsim.backends import StateVectorBackend
from qnsim.errors import BackendValidationError, UnsupportedOperationError
from qnsim import operations as ops
from qnsim.program import Program


def _h_cz_program():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    return p


def test_counts_happy_path_keys():
    backend = StateVectorBackend(seed=123)
    job = backend.run(_h_cz_program(), shots=500, result_config=qs.ResultConfig(counts=True))
    counts = job.result().get_counts()
    assert sum(counts.values()) == 500
    # H on q0 then CZ (no effect here): q1 always 0, q0 ~ 50/50 -> keys "00"/"01"
    assert set(counts) <= {"00", "01"}


def test_unsupported_operation_raises():
    class FooGate(ops.Operation):
        name = "FOO"
        _num_qubits = 1

    p = Program(1, 1)
    p.add(FooGate(), 0)
    p.add_measurement(0, 0)
    with pytest.raises(UnsupportedOperationError):
        StateVectorBackend().run(p, shots=10)


def test_condition_rejected():
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 1))
    p.add_measurement(0, 0)
    with pytest.raises(UnsupportedOperationError):
        StateVectorBackend().run(p, shots=10)


def test_mid_circuit_measurement_rejected():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    p.add(ops.X, 1)  # gate after a measurement
    p.add_measurement(1, 1)
    with pytest.raises(UnsupportedOperationError):
        StateVectorBackend().run(p, shots=10)


def test_nonpositive_shots_with_counts_raises():
    with pytest.raises(BackendValidationError):
        StateVectorBackend().run(_h_cz_program(), shots=0)


def test_deterministic_with_seed():
    a = StateVectorBackend(seed=7).run(_h_cz_program(), shots=300).result().get_counts()
    b = StateVectorBackend(seed=7).run(_h_cz_program(), shots=300).result().get_counts()
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qnsim.backends'`.

- [ ] **Step 3: Write minimal implementation**

`src/qnsim/backends.py`:

```python
"""Qubit statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import numpy as np

from . import engine
from .errors import BackendValidationError, UnsupportedOperationError
from .implementation import default_implementation_map
from .job import Job
from .layout import ResourceLayout
from .program import AppliedOperation, Measurement, Program
from .result import Result, ResultConfig, build_counts


class StateVectorBackend:
    def __init__(self, *, seed=None):
        self._seed = seed
        self._impl_map = default_implementation_map()

    def resolve_layout(self, program: Program) -> ResourceLayout:
        return ResourceLayout.from_program(program)

    def run(self, program, *, shots: int = 1024, result_config=None) -> Job:
        config = result_config if result_config is not None else ResultConfig()
        layout = self.resolve_layout(program)
        self._validate(program, config, shots, layout)
        try:
            return Job.done(self._execute(program, config, shots, layout))
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(self, program, config, shots, layout) -> None:
        seen_measurement = False
        has_measurement = False
        for step in program.operations:
            if isinstance(step, Measurement):
                seen_measurement = True
                has_measurement = True
                continue
            if isinstance(step, AppliedOperation):
                if seen_measurement:
                    raise UnsupportedOperationError(
                        "mid-circuit measurement is not supported in Phase 1 "
                        "(a gate appears after a measurement)"
                    )
                if step.condition is not None:
                    raise UnsupportedOperationError(
                        "conditional (feedforward) operations are not supported in Phase 1"
                    )
                if self._impl_map.get(type(step.operation)) is None:
                    raise UnsupportedOperationError(type(step.operation).__name__)
        effective_counts = config.counts if config.counts is not None else has_measurement
        if effective_counts and shots <= 0:
            raise BackendValidationError(
                f"counts require shots > 0, got shots={shots}"
            )

    # --- execution ---
    def _execute(self, program, config, shots, layout) -> Result:
        state = self._evolve(program, layout)
        measurements = self._measurement_map(program, layout)
        has_measurement = len(measurements) > 0
        effective_counts = config.counts if config.counts is not None else has_measurement

        counts = None
        available = set()
        if effective_counts:
            rng = np.random.default_rng(self._seed)
            if has_measurement:
                indices = engine.sample_indices(state, shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)  # nothing measured -> all-zero clbits
            counts = build_counts(indices, layout.n_clbits, measurements)
            available.add("counts")

        return Result(counts=counts, available=frozenset(available))

    def _evolve(self, program, layout) -> np.ndarray:
        state = engine.zero_state(layout.n_qubits)
        for step in program.operations:
            if isinstance(step, AppliedOperation):
                rule = self._impl_map.get(type(step.operation))
                matrix = rule(step)
                targets = tuple(layout.qubit_index(t) for t in step.targets)
                state = engine.apply(state, matrix, targets, layout.n_qubits)
        return state

    @staticmethod
    def _measurement_map(program, layout):
        out = []
        for step in program.operations:
            if isinstance(step, Measurement):
                out.append(
                    (layout.qubit_index(step.qreg), layout.clbit_index(step.clreg))
                )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backend.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/backends.py tests/test_backend.py
git commit -m "feat: StateVectorBackend validation and counts execution"
```

---

## Task 18: StateVectorBackend — statevector return & NoMeasurementWarning

**Files:**
- Modify: `src/qnsim/backends.py` (extend `_execute`)
- Test: `tests/test_backend_statevector.py`

**Interfaces:**
- Produces (behavior on existing `run`):
  - statevector default rule: `config.statevector is None` → attach iff no measurement.
  - explicit `statevector=True` with measurement: only `shots == 1` (projected/collapsed state); `shots > 1` → `BackendValidationError`.
  - `NoMeasurementWarning` when counts produced, some clbit never written, and no state delivered.

- [ ] **Step 1: Write the failing test**

`tests/test_backend_statevector.py`:

```python
import warnings

import numpy as np
import pytest

import qnsim as qs
from qnsim.backends import StateVectorBackend
from qnsim.errors import BackendValidationError, NoMeasurementWarning
from qnsim import operations as ops
from qnsim.program import Program


def test_statevector_default_attached_when_no_measurement():
    p = Program(1)
    p.add(ops.H, 0)
    job = StateVectorBackend().run(p, result_config=qs.ResultConfig(counts=False))
    sv = job.result().get_statevector()
    assert np.allclose(sv, np.array([1, 1]) / np.sqrt(2))


def test_statevector_not_attached_by_default_with_measurement():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    result = StateVectorBackend(seed=0).run(p, shots=10).result()
    with pytest.raises(qs.ResultFieldUnavailableError):
        result.get_statevector()


def test_projected_statevector_shots_one():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    sv = (
        StateVectorBackend(seed=0)
        .run(p, shots=1, result_config=qs.ResultConfig(counts=True, statevector=True))
        .result()
        .get_statevector()
    )
    # collapsed to a basis state
    assert np.isclose(np.linalg.norm(sv), 1.0)
    assert np.count_nonzero(np.round(np.abs(sv), 6)) == 1


def test_statevector_with_measurement_and_many_shots_rejected():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    with pytest.raises(BackendValidationError):
        StateVectorBackend().run(
            p, shots=10, result_config=qs.ResultConfig(counts=True, statevector=True)
        )


def test_no_measurement_warning_when_counts_only_and_no_state():
    p = Program(1, 1)  # has a clbit, never measured
    p.add(ops.H, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        StateVectorBackend(seed=0).run(
            p, shots=10, result_config=qs.ResultConfig(counts=True, statevector=False)
        ).result()
    assert any(issubclass(w.category, NoMeasurementWarning) for w in caught)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backend_statevector.py -v`
Expected: FAIL — e.g. `test_statevector_default_attached_when_no_measurement` raises `ResultFieldUnavailableError` (statevector never attached yet).

- [ ] **Step 3: Write minimal implementation**

In `src/qnsim/backends.py`: add the imports and validation for the `shots>1 + statevector + measurement` combination, then extend `_execute`. Update the top imports and `_validate`, and replace `_execute`:

```python
import warnings

from .errors import NoMeasurementWarning
```

Add to the END of `_validate` (after the shots check):

```python
        if (
            config.statevector is True
            and any(isinstance(s, Measurement) for s in program.operations)
            and shots > 1
        ):
            raise BackendValidationError(
                "statevector with measurement is only supported for shots == 1 "
                "in Phase 1"
            )
```

Replace `_execute` with:

```python
    def _execute(self, program, config, shots, layout) -> Result:
        state = self._evolve(program, layout)
        measurements = self._measurement_map(program, layout)
        has_measurement = len(measurements) > 0
        rng = np.random.default_rng(self._seed)

        counts = None
        statevector = None
        available = set()

        # Decide statevector delivery.
        want_sv = config.statevector
        if want_sv is None:
            want_sv = not has_measurement

        collapsed_state = None
        collapsed_index = None
        if want_sv and has_measurement:
            # Only reached for shots == 1 (validated). Collapse on measured qubits.
            measured_qubits = [q for q, _c in measurements]
            collapsed_state, bits = engine.collapse(
                state, layout.n_qubits, measured_qubits, rng
            )
            collapsed_index = 0
            for q, b in bits.items():
                collapsed_index |= b << q

        # Counts.
        effective_counts = config.counts if config.counts is not None else has_measurement
        if effective_counts:
            if has_measurement:
                if collapsed_index is not None:
                    indices = np.array([collapsed_index], dtype=int)
                else:
                    indices = engine.sample_indices(state, shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)
            counts = build_counts(indices, layout.n_clbits, measurements)
            available.add("counts")

        # Statevector.
        if want_sv:
            statevector = collapsed_state if has_measurement else state
            available.add("statevector")

        # NoMeasurementWarning: counts produced, some clbit never written, no state.
        if effective_counts and "statevector" not in available:
            written = {c for _q, c in measurements}
            if any(c not in written for c in range(layout.n_clbits)):
                warnings.warn(
                    "counts contain clbits that were never measured; "
                    "returning zero-filled counts",
                    NoMeasurementWarning,
                    stacklevel=2,
                )

        return Result(
            counts=counts, statevector=statevector, available=frozenset(available)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backend_statevector.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/backends.py tests/test_backend_statevector.py
git commit -m "feat: statevector return semantics and NoMeasurementWarning"
```

---

## Task 19: Public backend API + end-to-end workflow

**Files:**
- Modify: `src/qnsim/__init__.py`
- Test: `tests/test_e2e.py`

**Interfaces:**
- Produces: top-level `qs.StateVectorBackend`, `qs.ResultConfig`, `qs.Result`, `qs.Job`, error/warning classes, and `qs.backends` module alias (`qs.backends.StateVectorBackend`).

- [ ] **Step 1: Write the failing test**

`tests/test_e2e.py`:

```python
import qnsim as qs


def test_minimal_workflow_from_spec():
    program = qs.Program(2, 2)
    program.add(qs.ops.H, 0)
    program.add(qs.ops.CZ, (0, 1))
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    backend = qs.StateVectorBackend(seed=2024)
    job = backend.run(program, shots=1000, result_config=qs.ResultConfig(counts=True))
    result = job.result()
    counts = result.get_counts()

    assert sum(counts.values()) == 1000
    assert set(counts) <= {"00", "01"}
    # roughly balanced between the two reachable outcomes
    assert all(150 < v < 850 for v in counts.values())


def test_backends_module_alias():
    assert qs.backends.StateVectorBackend is qs.StateVectorBackend


def test_error_and_warning_classes_exposed():
    assert issubclass(qs.UnsupportedOperationError, qs.BackendValidationError)
    assert issubclass(qs.BackendValidationError, qs.QnsimError)
    assert issubclass(qs.NoMeasurementWarning, UserWarning)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: FAIL with `AttributeError: module 'qnsim' has no attribute 'StateVectorBackend'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/qnsim/__init__.py`:

```python
"""qnsim — quantum noisy simulator (MVP Phase 1)."""

from . import backends
from . import operations as ops
from .backends import StateVectorBackend
from .errors import (
    BackendValidationError,
    NoMeasurementWarning,
    QnsimError,
    ResultFieldUnavailableError,
    UnsupportedOperationError,
)
from .job import Job
from .program import AppliedOperation, Measurement, Program
from .registers import (
    ClassicalRegister,
    QuantumRegister,
    Register,
    RegisterRef,
)
from .result import Result, ResultConfig

__version__ = "0.0.1"

__all__ = [
    "ops",
    "backends",
    "Program",
    "AppliedOperation",
    "Measurement",
    "Register",
    "QuantumRegister",
    "ClassicalRegister",
    "RegisterRef",
    "StateVectorBackend",
    "Job",
    "Result",
    "ResultConfig",
    "QnsimError",
    "BackendValidationError",
    "UnsupportedOperationError",
    "ResultFieldUnavailableError",
    "NoMeasurementWarning",
]
```

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `uv run pytest -v`
Expected: PASS — entire Stage 1 + Stage 2 suite green.

- [ ] **Step 5: Commit**

```bash
git add src/qnsim/__init__.py tests/test_e2e.py
git commit -m "feat: expose backend public API; end-to-end workflow green (Phase 1 complete)"
```

---

## Plan Self-Review

**Spec coverage:**
- §0 target workflow → Task 19 e2e. ✅
- §0 gate surface (X Y Z H T CX CZ + RX RY RZ) → Task 2 + Task 12. ✅
- §1 TDD / uv+src+pytest → Task 0 + every task. ✅
- §1 `StateVectorBackend` naming, top-level primary, `qs.backends` alias → Task 17 + Task 19. ✅
- §1 conditional execution rejected → Task 17 `test_condition_rejected`. ✅
- §1 seed on backend → Task 17 `test_deterministic_with_seed`. ✅
- §3.1 registers/operations/program/__init__ units → Tasks 1,2,3–8,9. ✅
- §3.2 arity via `_num_qubits`/`num_qubits` + `AppliedOperation.__post_init__` → Task 2 + Task 3. ✅
- §3.2 construction, flat int, add, add_measurement (no basis/observable), condition norm, copy → Tasks 4,5,6,7,8. ✅
- §3.4 minimal-example assertions → Task 9. ✅
- §4.1 errors/layout/impl/engine/result/backend+job units → Tasks 10–18. ✅
- §4.2 counts little-endian / unwritten zero / last-write → Task 15. ✅
- §4.2 statevector little-endian bit order → Task 13 engine convention + tests. ✅
- §4.2 statevector default/projected/shots>1 reject → Task 18. ✅
- §4.2 ResultConfig defaults / result_config=None normalize → Task 15 + Task 17 (`config = result_config or ResultConfig()`, then effective counts defaults from program shape). ✅
- §4.2 get_counts/get_statevector raise when unavailable → Task 15. ✅
- §4.2 shots default 1024 / >0 only when sampling → Task 17 (`shots: int = 1024`, check gated on effective counts). ✅
- §4.2 NoMeasurementWarning condition → Task 18. ✅
- §4.2 Job DONE/ERROR, validation raises in run, execution → ERROR re-raised → Task 16 + Task 17 (`run` raises in `_validate`, wraps `_execute`). ✅
- §4.2 resolve_layout default + rule-not-index → Task 11 + Task 12 + Task 17 `_evolve`. ✅
- §4.2 counts sampling multinomial (shots>1), collapse only for shots==1 projected → Task 14 + Task 18. ✅
- §4.3 validation list → Task 17 `_validate`. ✅
- §6 out-of-scope items → none implemented. ✅

**Placeholder scan:** Task 5 intentionally ships a temporary `_normalize_condition` that raises `NotImplementedError`, replaced in Task 7 (each is independently green); no other placeholders.

**Type consistency:** `build_counts(indices, n_clbits, measurements)`, `measurements = list[(qubit_flat, clbit_flat)]`, `engine.apply(state, matrix, targets, n_qubits)`, `engine.collapse(...) -> (state, {qubit: bit})`, `Result(counts, statevector, available)`, `Job.done/failed/result`, `ResourceLayout.from_program / qubit_index / clbit_index` — all consistent across producing and consuming tasks.
