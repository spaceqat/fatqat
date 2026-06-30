# qnsim Engine Ownership Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `StateVectorBackend` from receiving a live engine across a method boundary; make the backend own the engine and drive it in place, while the engine keeps holding the evolution context and returns only inert, private values.

**Architecture:** The engine stays stateful — it owns the state vector and (eventually) compiled kernels. The backend constructs the engine once in `__init__` and re-initializes it per run, driving it through `evolve → readout` calls instead of being handed a live session. Hardening: `export_state()` returns a copy so `Result` never aliases engine internals, and `collapse()` returns the flat outcome index so the backend does no bit arithmetic on engine output.

**Tech Stack:** Python 3.13, numpy, pytest, uv (runner: `uv run pytest`).

## Global Constraints

- Python `>=3.13`; runtime dependency limited to `numpy`; dev dependency `pytest`.
- src layout: package at `src/qnsim/`, tests at `tests/`. Test runner: `uv run pytest`.
- TDD: every task writes the failing test first (where applicable), runs it red, implements minimally, runs it green, commits.
- statevector basis = little-endian: amplitude index bit `q` is the value of flat qubit `q` (qubit 0 = least-significant bit).
- counts key = little-endian over clbits: clbit 0 is the rightmost char.
- The engine remains stateful and owns the evolution context. The engine class and any state it returns must stay private (not exported from `__init__.py`); only numerical results (sample indices, statevector arrays, outcome ints) cross to the backend.
- The backend performs no array numerics. It resolves gates into `MatrixImplementation` (dict lookup + index arithmetic only) and routes engine outputs into `Result`.
- Seeded determinism is a hard invariant: `StateVectorBackend(seed=...)` must produce identical counts before and after this refactor. The existing `tests/test_backend.py::test_deterministic_with_seed` and `tests/test_e2e.py::test_minimal_workflow_from_spec` must stay green at every task boundary.
- `git commit` messages: plain, **no `Co-Authored-By` / AI attribution trailer**.

## Design Decision: engine construction site

The engine is constructed once in `StateVectorBackend.__init__` (`self._engine = StateVectorEngine()`) and re-initialized at the start of every run via `self._engine.initialize(n_qubits)`. This honors "engine init kept in the backend so compiled code can be reused" and reuses one instance across runs.

Accepted MVP limitation: because the backend holds per-run state on `self._engine`, a single `StateVectorBackend` instance is **not** safe for concurrent `run()` calls. This is fine for the single-threaded MVP. Task 2 adds a guard test (`test_backend_run_is_repeatable`) ensuring sequential reuse re-initializes correctly.

---

## File Structure

Files touched (all exist):
- `src/qnsim/engine.py` — `StateVectorEngine`: `export_state()` copy (Task 1), `collapse()` returns flat index (Task 3)
- `src/qnsim/backends.py` — own + drive engine in place (Task 2), consume flat collapse index (Task 3)
- `tests/test_engine_apply.py` — copy-semantics test (Task 1)
- `tests/test_engine_measure.py` — collapse-returns-index test (Task 3)
- `tests/test_backend_reuse.py` — new, repeatability guard (Task 2)

No new source files. No public API changes.

---

## Task 1: Engine `export_state()` returns an independent copy

**Files:**
- Modify: `src/qnsim/engine.py` (method `export_state`)
- Test: `tests/test_engine_apply.py`

**Interfaces:**
- Consumes: `StateVectorEngine.initialize(n_qubits)`, `StateVectorEngine.export_state() -> np.ndarray` (existing).
- Produces: `export_state()` returns a fresh array each call; mutating the returned array does not affect the engine's internal state or later exports.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engine_apply.py`:

```python
def test_export_state_returns_independent_copy():
    eng = _engine(1)
    first = eng.export_state()
    first[0] = 999.0
    second = eng.export_state()
    assert second[0] != 999.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_apply.py::test_export_state_returns_independent_copy -v`
Expected: FAIL — `export_state` currently returns `self._state` by reference, so `second[0]` is `999.0` and the assertion fails.

- [ ] **Step 3: Write minimal implementation**

In `src/qnsim/engine.py`, change `export_state`:

```python
    def export_state(self) -> np.ndarray:
        self._require_state()
        return self._state.copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine_apply.py -v`
Expected: PASS (all engine-apply tests, including the new one).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — 86 tests (85 prior + 1 new). The copy does not affect any consumer; `Result.get_statevector()` still returns equal values.

- [ ] **Step 6: Commit**

```bash
git add src/qnsim/engine.py tests/test_engine_apply.py
git commit -m "refactor: export_state returns a copy so Result never aliases engine state"
```

---

## Task 2: Backend owns the engine and drives it in place

**Files:**
- Modify: `src/qnsim/backends.py` (`__init__`, `_execute`, `_evolve`)
- Test: `tests/test_backend_reuse.py` (create)

**Interfaces:**
- Consumes: `StateVectorEngine()` (no-arg construction), `engine.initialize(n_qubits)`, `engine.apply(MatrixImplementation(...))`, `engine.collapse(measured_qubits, rng) -> dict[int, int]` (unchanged in this task), `engine.sample_indices(shots, rng)`, `engine.export_state()`.
- Produces: `_evolve(self, program, layout) -> None` (drives `self._engine`, no return). `StateVectorBackend.__init__` sets `self._engine`. `run()` is repeatable on one backend instance: each call re-initializes the engine, so sequential runs are independent.

- [ ] **Step 1: Write the guard test**

This is a refactor; the test guards the reused-instance decision. It passes against current code and must stay green through the refactor — it goes red only if the engine is reused without re-initialization.

Create `tests/test_backend_reuse.py`:

```python
import qnsim as qs


def test_backend_run_is_repeatable():
    program = qs.Program(1, 1)
    program.add(qs.ops.X, 0)
    program.add_measurement(0, 0)

    backend = qs.StateVectorBackend(seed=0)
    first = backend.run(program, shots=10).result().get_counts()
    second = backend.run(program, shots=10).result().get_counts()

    # X|0> = |1>, so every shot reads 1 on both runs; the second run must NOT
    # start from the leftover |1> state (which X would flip back to |0>).
    assert first == {"1": 10}
    assert second == {"1": 10}
```

- [ ] **Step 2: Run the guard test against current code (baseline)**

Run: `uv run pytest tests/test_backend_reuse.py -v`
Expected: PASS — current `_evolve` builds a fresh engine each run, so reuse is already correct. This locks the behavior before the refactor.

- [ ] **Step 3: Add the engine as a backend-owned attribute**

In `src/qnsim/backends.py`, update `__init__`:

```python
    def __init__(self, *, seed=None):
        self._seed = seed
        self._impl_map = default_implementation_map()
        self._engine = StateVectorEngine()
```

- [ ] **Step 4: Change `_evolve` to drive the owned engine and return None**

Replace the `_evolve` method body in `src/qnsim/backends.py`:

```python
    def _evolve(self, program, layout) -> None:
        """Reset the owned engine and apply each gate as a MatrixImplementation."""
        engine = self._engine
        engine.initialize(layout.n_qubits)
        for step in program.operations:
            if isinstance(step, AppliedOperation):
                rule = self._impl_map.get(type(step.operation))
                matrix = rule(step)
                target_indices = tuple(layout.qubit_index(t) for t in step.targets)
                engine.apply(
                    MatrixImplementation(matrix=matrix, target_indices=target_indices)
                )
```

- [ ] **Step 5: Update `_execute` to drive in place instead of receiving the engine**

In `src/qnsim/backends.py`, change the first two lines of `_execute` from:

```python
        engine = self._evolve(program, layout)
        measurements = self._measurement_map(program, layout)
```

to:

```python
        self._evolve(program, layout)
        engine = self._engine
        measurements = self._measurement_map(program, layout)
```

Leave the rest of `_execute` unchanged.

- [ ] **Step 6: Run the guard test and the full suite**

Run: `uv run pytest -q`
Expected: PASS — 86 tests. The guard test stays green (re-init per run), seeded determinism unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/qnsim/backends.py tests/test_backend_reuse.py
git commit -m "refactor: backend owns the engine and drives it in place"
```

---

## Task 3: `collapse()` returns the flat outcome index; backend drops bit reconstruction

**Files:**
- Modify: `src/qnsim/engine.py` (method `collapse`)
- Modify: `src/qnsim/backends.py` (`_execute` collapse branch)
- Test: `tests/test_engine_measure.py`

**Interfaces:**
- Consumes: `StateVectorEngine.initialize`, `apply`, `export_state`, `probabilities`.
- Produces: `StateVectorEngine.collapse(measured_qubits, rng) -> int` — samples one basis index, projects the internal state onto the measured-qubit values, renormalizes, and returns the flat sampled index. The backend assigns this int directly to `collapsed_index` (no per-qubit reconstruction).

- [ ] **Step 1: Update the engine collapse test**

In `tests/test_engine_measure.py`, replace `test_collapse_projects_internal_state_to_basis` with:

```python
def test_collapse_returns_flat_index_and_projects():
    eng = _h_engine(1, 0)
    rng = np.random.default_rng(1)
    idx = eng.collapse([0], rng)
    assert idx in (0, 1)
    expected = np.zeros(2, dtype=complex)
    expected[idx] = 1.0
    assert np.allclose(eng.export_state(), expected)
    assert np.isclose(np.linalg.norm(eng.export_state()), 1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_engine_measure.py::test_collapse_returns_flat_index_and_projects -v`
Expected: FAIL — `collapse` currently returns a `dict[int, int]`, so `idx in (0, 1)` is `False` (a dict is not in that tuple) and the assertion fails.

- [ ] **Step 3: Change `collapse` to return the flat index**

In `src/qnsim/engine.py`, replace the `collapse` method:

```python
    def collapse(self, measured_qubits, rng) -> int:
        """Sample one outcome, project the internal state, return the flat index."""
        self._require_state()
        idx = int(rng.choice(len(self._state), p=self.probabilities()))
        arange = np.arange(len(self._state))
        keep = np.ones(len(self._state), dtype=bool)
        for q in measured_qubits:
            bit = (idx >> q) & 1
            keep &= ((arange >> q) & 1) == bit
        new = np.where(keep, self._state, 0.0).astype(complex)
        norm = np.linalg.norm(new)
        if norm > 0:
            new = new / norm
        self._state = new
        return idx
```

- [ ] **Step 4: Run the engine measure tests**

Run: `uv run pytest tests/test_engine_measure.py -v`
Expected: PASS.

- [ ] **Step 5: Update the backend to consume the flat index**

In `src/qnsim/backends.py`, replace the collapse branch in `_execute`:

```python
        collapsed_index = None
        if want_sv and has_measurement:
            # Only reached for shots == 1 (validated). Collapse on measured qubits;
            # the engine's internal state becomes the projected statevector.
            measured_qubits = [q for q, _c in measurements]
            bits = engine.collapse(measured_qubits, rng)
            collapsed_index = 0
            for q, b in bits.items():
                collapsed_index |= b << q
```

with:

```python
        collapsed_index = None
        if want_sv and has_measurement:
            # Only reached for shots == 1 (validated). Collapse on measured qubits;
            # the engine's internal state becomes the projected statevector and the
            # flat outcome index feeds counts directly.
            measured_qubits = [q for q, _c in measurements]
            collapsed_index = engine.collapse(measured_qubits, rng)
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — 86 tests. `build_counts` reads `(collapsed_index >> q) & 1` for each measured qubit, identical to the previous reconstruction; seeded determinism (`test_deterministic_with_seed`, `test_minimal_workflow_from_spec`) unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/qnsim/engine.py src/qnsim/backends.py tests/test_engine_measure.py
git commit -m "refactor: collapse returns flat outcome index; backend drops bit reconstruction"
```

---

## Plan Self-Review

**Spec coverage (against the agreed design):**
- "Don't hand a live engine across a boundary" → Task 2 makes `_evolve` return `None` and the backend own/drive `self._engine`. ✅
- "Engine stays stateful, holds evolution context" → engine retains `self._state`; only construction site moved. ✅
- "Nothing mutable crosses the boundary / Result must not alias engine internals" → Task 1 `export_state()` copy. ✅
- "Backend does no numerics on engine output" → Task 3 moves the `idx |= b << q` bit arithmetic into the engine; backend assigns the int directly. ✅
- "Engine + returned state stay private" → no `__init__.py` changes; engine class and `collapse`'s int/array returns are not exported. ✅
- "Seeded determinism invariant" → asserted at every task's full-suite step; rng call sequence (one `rng.choice` in collapse, one in `sample_indices`) is unchanged. ✅

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✅

**Type consistency:**
- `_evolve` return type: `StateVectorEngine` → `None` (Task 2), consistently consumed (`self._evolve(...)` discards result, reads `self._engine`). ✅
- `collapse` return type: `dict[int, int]` → `int` (Task 3); both the engine test and the backend branch updated in the same task, so no task boundary sees a mismatch. ✅
- `export_state` signature unchanged (`-> np.ndarray`), only the returned object's provenance (copy) changes. ✅

**Cross-task suite health:** Task 1 ends at 86 green; Task 2 keeps `collapse` returning the dict (backend reconstruction intact) → 86 green; Task 3 flips `collapse` and its sole backend consumer together → 86 green. No intermediate red suite. ✅
