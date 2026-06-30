# Restrict Integer Register Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a bare-`int` operand valid only when the relevant register kind has exactly one register; require an explicit `RegisterRef` once a program has multiple (or zero) quantum/classical registers.

**Architecture:** All integer operand paths — gate targets, condition slots, and measurement qreg/clreg — funnel through `Program._flat_from_int`, so the behavior change lives in that one method. Explicit `RegisterRef` operands keep flowing through `_resolve_ref` untouched, so multi-register programs stay fully usable via explicit refs. This is a frontend operand-resolution change only; the backend `ResourceLayout` still concatenates registers into flat indices (that flattening is unchanged).

**Tech Stack:** Python `>=3.13`, numpy, pytest, uv (runner: `uv run pytest`).

## Global Constraints

- Python `>=3.13`; runtime dependency limited to `numpy`; dev dependency `pytest`.
- src layout: package at `src/qnsim/`, tests at `tests/`. The git repo root is the `qnsim/` directory, so all paths and `git add` arguments below are repo-relative (`src/qnsim/...`, `tests/...`). Test runner: `uv run pytest`.
- TDD: write the failing tests first, run them red, implement minimally, run green, commit. Commits keep the suite green — never commit a knowingly-red test in isolation from its implementation.
- New resolution rule: a bare `int` operand resolves **only** when the relevant register list has exactly one register; with multiple or zero registers it raises `TypeError` whose message contains `explicit RegisterRef` and points the caller at `reg[0]`-style refs. Explicit `RegisterRef` operands are always accepted (subject to the existing belongs-to-program check).
- Out of scope: changing `ResourceLayout` flattening, the counts/statevector flat-index conventions, or any backend code.
- `git commit` messages: plain, **no `Co-Authored-By` / AI attribution trailer**.

---

## File Structure

- Modify: `src/qnsim/program.py` — restrict `Program._flat_from_int` to the single-register case.
- Modify: `tests/test_program_construction.py` — replace the cross-register flat test with a rejection test; add an explicit-ref test.
- Modify: `tests/test_program_condition.py` — replace the cross-register flat condition test with a rejection test; add an explicit-ref test.
- Modify: `tests/test_program_measurement.py` — add rejection + explicit-ref tests for both register kinds.
- Modify: `docs/superpowers/specs/2026-06-29-qnsim-mvp-phase1-working-plan-design.md` and `docs/superpowers/plans/2026-06-29-qnsim-mvp-phase1.md` — update the operand-resolution rule so docs match code.

Note: `tests/test_program_construction.py::test_flat_qubit_resolution_out_of_range_raises` (uses `Program(2)`, single register) and `tests/test_layout.py::test_multi_register_flat_concatenation` (tests `ResourceLayout`, not operand resolution) are intentionally left unchanged — both stay green under the new rule.

---

## Task 1: Restrict integer operand resolution to single-register programs

This is one atomic task: all three test files change together with the single-function implementation, so the suite is red after the tests and green after the implementation — one green commit, no red commit on `main`.

**Files:**
- Modify: `src/qnsim/program.py` (method `_flat_from_int`)
- Test: `tests/test_program_construction.py`, `tests/test_program_condition.py`, `tests/test_program_measurement.py`

**Interfaces:**
- Consumes: `Program.registers(*, qreg, clreg)`, `Program._resolve_qubit`, `Program.add`, `Program.add_measurement`, `QuantumRegister`, `ClassicalRegister`, `RegisterRef`, `reg[i]` indexing.
- Produces: `Program._flat_from_int(i: int, regs: list[Register]) -> RegisterRef` raises `TypeError` (message contains `explicit RegisterRef`) when `len(regs) != 1`; otherwise returns `regs[0][i]`, delegating bounds/negative checks to `Register.__getitem__` (`IndexError`).

- [ ] **Step 1: Rewrite the construction tests**

In `tests/test_program_construction.py`, replace:

```python
def test_flat_qubit_resolution_across_registers():
    qr0 = QuantumRegister(2, name="a")
    qr1 = QuantumRegister(2, name="b")
    p = Program.registers(qreg=[qr0, qr1])
    assert p._resolve_qubit(0) == qr0[0]
    assert p._resolve_qubit(1) == qr0[1]
    assert p._resolve_qubit(2) == qr1[0]
    assert p._resolve_qubit(3) == qr1[1]
```

with:

```python
def test_int_qubit_resolution_rejects_multiple_registers():
    qr0 = QuantumRegister(2, name="a")
    qr1 = QuantumRegister(2, name="b")
    p = Program.registers(qreg=[qr0, qr1])

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p._resolve_qubit(0)


def test_explicit_qubit_refs_work_across_multiple_registers():
    qr0 = QuantumRegister(2, name="a")
    qr1 = QuantumRegister(2, name="b")
    p = Program.registers(qreg=[qr0, qr1])

    assert p._resolve_qubit(qr0[1]) == qr0[1]
    assert p._resolve_qubit(qr1[0]) == qr1[0]
```

(`test_program_construction.py` already imports `pytest`, `QuantumRegister`, and `ClassicalRegister` — no import change needed.)

- [ ] **Step 2: Rewrite the condition tests**

In `tests/test_program_condition.py`, add the register import below the existing imports:

```python
from qnsim.registers import QuantumRegister, ClassicalRegister
```

Then replace:

```python
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

with:

```python
def test_condition_int_slot_rejects_multiple_classical_registers():
    p = Program.registers(
        qreg=[QuantumRegister(1)],
        clreg=[ClassicalRegister(2, name="a"), ClassicalRegister(2, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add(ops.X, 0, condition=(0, 1))


def test_condition_explicit_slot_refs_work_across_multiple_classical_registers():
    p = Program.registers(
        qreg=[QuantumRegister(1)],
        clreg=[ClassicalRegister(2, name="a"), ClassicalRegister(2, name="b")],
    )

    p.add(ops.X, 0, condition=(p.creg[1][0], 1))

    assert p.operations[0].condition == ((p.creg[1][0], 1),)
```

- [ ] **Step 3: Add the measurement tests**

In `tests/test_program_measurement.py`, add the register import below the existing imports:

```python
from qnsim.registers import QuantumRegister, ClassicalRegister
```

Then append these four tests to the end of the file:

```python
def test_add_measurement_int_qreg_rejects_multiple_quantum_registers():
    p = Program.registers(
        qreg=[QuantumRegister(1, name="a"), QuantumRegister(1, name="b")],
        clreg=[ClassicalRegister(2, name="c")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add_measurement(0, 0)


def test_add_measurement_explicit_qreg_ref_works_with_multiple_quantum_registers():
    p = Program.registers(
        qreg=[QuantumRegister(1, name="a"), QuantumRegister(1, name="b")],
        clreg=[ClassicalRegister(2, name="c")],
    )

    p.add_measurement(p.qreg[1][0], 1)

    assert p.operations[0].qreg == p.qreg[1][0]
    assert p.operations[0].clreg == p.creg[0][1]


def test_add_measurement_int_clreg_rejects_multiple_classical_registers():
    p = Program.registers(
        qreg=[QuantumRegister(2, name="q")],
        clreg=[ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add_measurement(0, 0)


def test_add_measurement_explicit_clreg_ref_works_with_multiple_classical_registers():
    p = Program.registers(
        qreg=[QuantumRegister(2, name="q")],
        clreg=[ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    p.add_measurement(1, p.creg[1][0])

    assert p.operations[0].qreg == p.qreg[0][1]
    assert p.operations[0].clreg == p.creg[1][0]
```

- [ ] **Step 4: Run the new tests and verify they fail**

Run: `uv run pytest tests/test_program_construction.py tests/test_program_condition.py tests/test_program_measurement.py -v`
Expected: the two multi-register rejection tests for integers fail — `_resolve_qubit(0)` / `add(..., condition=(0, 1))` / `add_measurement(0, 0)` still resolve through the flattened register list instead of raising `TypeError`. (The explicit-ref tests already pass; they exercise the unchanged `RegisterRef` path.)

- [ ] **Step 5: Restrict `_flat_from_int`**

In `src/qnsim/program.py`, replace the `_flat_from_int` method:

```python
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
```

with:

```python
    @staticmethod
    def _flat_from_int(i: int, regs: list[Register]) -> RegisterRef:
        if not isinstance(i, int) or isinstance(i, bool):
            raise TypeError(f"operand must be int or RegisterRef, got {type(i)!r}")
        if len(regs) != 1:
            raise TypeError(
                "integer operands are only allowed when there is exactly one "
                "register of the relevant kind; pass an explicit RegisterRef "
                "(e.g. qreg[0] or creg[0]) instead"
            )
        # Bounds and negative-index checks are delegated to Register.__getitem__,
        # which raises IndexError.
        return regs[0][i]
```

Note: the original `if i < 0: raise IndexError(i)` is dropped — `regs[0][i]` delegates to `Register.__getitem__`, which already raises `IndexError` for negative and out-of-range indices, so the single-register out-of-range test still passes.

- [ ] **Step 6: Run the focused tests and the full suite**

Run: `uv run pytest tests/test_program_construction.py tests/test_program_condition.py tests/test_program_measurement.py -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS — full suite green (existing tests minus the 2 replaced cross-register tests, plus the new tests).

- [ ] **Step 7: Commit**

```bash
git add src/qnsim/program.py tests/test_program_construction.py tests/test_program_condition.py tests/test_program_measurement.py
git commit -m "fix: require explicit refs for multi-register integer operands"
```

---

## Task 2: Update the design spec and plan to the new rule

The new behavior contradicts the operand-resolution rule recorded in the spec and the Phase 1 plan. Update both so docs match code. Leave the layout/counts/statevector flattening text (spec lines ~149–157) unchanged — that flattening is unaffected.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-29-qnsim-mvp-phase1-working-plan-design.md`
- Modify: `docs/superpowers/plans/2026-06-29-qnsim-mvp-phase1.md`

**Interfaces:** documentation only; no code or tests.

- [ ] **Step 1: Update the qubit-operand rule in the design spec**

In `docs/superpowers/specs/2026-06-29-qnsim-mvp-phase1-working-plan-design.md`, replace:

```markdown
  - 裸 `int` = **全局 flat 索引**（按 register 声明顺序拼接解析）。
```

with:

```markdown
  - 裸 `int` operand 仅在该 program 的 quantum register 恰好只有一个时有效（解析为该 register 的 `RegisterRef`）；存在多个或零个 register 时抛 `TypeError`，要求改用显式 `RegisterRef`（如 `qreg[0]`）。
```

- [ ] **Step 2: Update the measurement rule in the design spec**

Replace:

```markdown
  - in-place，追加 `Measurement`；`clreg` 接受 `int | RegisterRef`，裸 int 按全局 classical flat 解析。
```

with:

```markdown
  - in-place，追加 `Measurement`；`clreg` 接受 `int | RegisterRef`，裸 int 仅在 classical register 恰好只有一个时有效，否则要求显式 `RegisterRef`。
```

- [ ] **Step 3: Update the condition-slot rule in the design spec**

Replace:

```markdown
  - 裸 int slot 解析成 `ClassicalRegister` 上的 `RegisterRef`；非 classical ref 拒绝。
```

with:

```markdown
  - 裸 int slot 仅在 classical register 恰好只有一个时解析成该 register 的 `RegisterRef`，否则要求显式 `RegisterRef`；非 classical ref 拒绝。
```

- [ ] **Step 4: Update the test-points line in the design spec**

Replace:

```markdown
- 裸 int → 全局 flat 索引解析（多 register 拼接）
```

with:

```markdown
- 裸 int 仅在单 register 时解析；多/零 register 时抛 `TypeError`，需显式 `RegisterRef`
```

- [ ] **Step 5: Update the global constraint in the Phase 1 plan**

In `docs/superpowers/plans/2026-06-29-qnsim-mvp-phase1.md`, replace:

```markdown
- Bare `int` operands = **global flat index** across registers in declaration order.
```

with:

```markdown
- Bare `int` operands resolve only when the relevant register kind has exactly one register; with multiple (or zero) registers they raise `TypeError` directing the caller to an explicit `RegisterRef`. (The backend `ResourceLayout` still concatenates registers for flat indices — that flattening is unchanged.)
```

- [ ] **Step 6: Verify the suite is still green and commit**

Run: `uv run pytest -q`
Expected: PASS (docs-only change; nothing should move).

```bash
git add docs/superpowers/specs/2026-06-29-qnsim-mvp-phase1-working-plan-design.md docs/superpowers/plans/2026-06-29-qnsim-mvp-phase1.md
git commit -m "docs: align operand-resolution rule with single-register integer restriction"
```

---

## Self-Review

**Spec coverage:** The change is centralized in `_flat_from_int`, which every integer operand path uses (gate targets via `_resolve_qubit`, condition slots via `_resolve_classical_slot` → `_resolve_clbit`, measurement qreg/clreg via `_resolve_qubit`/`_resolve_clbit`). Task 1 covers all three with rejection + explicit-ref tests for both register kinds. Task 2 updates every doc location that stated the old rule (spec lines 99/104/109/125, plan line 17), and explicitly leaves the layout flattening text intact.

**Placeholder scan:** Concrete repo-relative paths, exact before/after code and prose, exact `uv run pytest` commands, expected red/green outcomes, and commit commands. No TBD/TODO.

**Type consistency:** `_flat_from_int(i: int, regs: list[Register]) -> RegisterRef` signature is unchanged; only its body changes. Test error-match string `"explicit RegisterRef"` is a substring of the implemented message. `Program.registers`, `_resolve_qubit`, `add`, `add_measurement`, `QuantumRegister`, `ClassicalRegister`, and `reg[i]` are all used with their existing signatures.

**Green-commit discipline:** Task 1 lands all tests and the implementation in a single commit (red only within the task, never committed red). Task 2 is docs-only and keeps the suite green.
