# qnsim MVP Phase 1 — Final Code Review

**Date:** 2026-06-30
**Branch:** main
**Commit range:** f60c189..7599ca1 (21 commits)
**Test result:** 85/85 pass

## Verdict: READY TO MERGE

No CRITICAL or MAJOR issues found. The statevector logic, counts assembly, error hierarchy, and validation order all match the spec. All 9 findings below are MINOR; they are clean-up candidates before Phase 2.

---

## Findings

### 1. Mutable array in frozen dataclass
**File:** `src/qnsim/implementation.py:17-20`
**Class:** `MatrixImplementation`
Frozen dataclass stores a `np.ndarray`; Python cannot freeze array contents, so the matrix data remains mutable after construction.
**Fix:** Store a copy and make it read-only: `object.__setattr__(self, 'matrix', np.array(matrix)); self.matrix.flags.writeable = False` in `__post_init__`.

### 2. Stray TODO comments in engine.py
**File:** `src/qnsim/engine.py:7-8,14`
Three TODO comments were added by a task implementer outside their scope. They describe legitimate future refactoring (engine as class, `zero_state` moved to backend) but clutter the module.
**Fix:** Remove the comments; record the refactoring intent in a Phase 2 task.

### 3. Unused `pytest` import in test file
**File:** `tests/test_implementation.py:2`
`import pytest` is unused — no `pytest.raises` or fixtures in the file.
**Fix:** Remove the import.

### 4. Stray misspelled TODO comment in test file
**File:** `tests/test_implementation.py:13`
Comment reads `# TODO: This test is a bit unessenary` (misspelled). Remove it.

### 5. Stray TODO comment with cryptic bool fallthrough
**File:** `src/qnsim/program.py:78`
`_coerce_registers` contains `# TODO: why bool?`. The existing `isinstance(spec, int) and not isinstance(spec, bool)` check was meant to reject bools, but a bool that slips past produces a cryptic `TypeError: 'bool' object is not iterable`.
**Fix:** Remove the TODO; add an explicit `if isinstance(spec, bool): raise TypeError(...)` guard.

### 6. Inaccurate `NoMeasurementWarning` docstring
**File:** `src/qnsim/errors.py:22-23`
Docstring says "counts contain *only* never-written clbits" but the warning fires when *any* clbit was never written (partial-measurement programs also trigger it).
**Fix:** Change to "one or more clbits were never written by any measurement."

### 7. Mapping metadata stored by reference
**Files:** `src/qnsim/registers.py:13`, `src/qnsim/program.py:163`
`Register.metadata` and `Measurement.metadata` accept a `Mapping` and store it by reference. A caller passing a mutable `dict` retains a handle that can mutate the "frozen" object's metadata post-construction.
**Fix:** Store `dict(metadata)` (or `types.MappingProxyType(dict(metadata))`) via `object.__setattr__` in `__post_init__`.

### 8. Empty condition tuple causes `IndexError`
**File:** `src/qnsim/program.py:134-142`
`_normalize_condition(())` crashes with `IndexError: tuple index out of range` because the function reads `condition[0]` unconditionally.
**Fix:** Treat `()` as `None` (no condition) or raise `ValueError("condition must be non-empty if provided")`.

### 9. Duplicate qubit targets not caught at construction
**File:** `src/qnsim/program.py:127`, `AppliedOperation.__post_init__`
`p.add(ops.CX, (0, 0))` is accepted silently and only fails at execution time with a cryptic numpy `ValueError`.
**Fix:** Add a duplicate-target check in `AppliedOperation.__post_init__` and raise `ValueError("operation targets must be distinct qubits")`.

### 10. `MatrixImplementation.target_indices` is a dead field
**File:** `src/qnsim/implementation.py:17-20`
`target_indices` is never populated by any map rule and is not used in the execution path (the backend computes target indices from `layout.qubit_index()` directly). It is scaffolding for a potential compiled-circuit design.
**Fix:** Remove for MVP clarity, or add a comment explaining the future design intent.

### 11. `Job.result()` silently returns `None` for non-terminal status
**File:** `src/qnsim/job.py:21-23`
If `Job` is constructed manually with a status other than `"DONE"` or `"ERROR"`, `result()` falls through all conditions and returns `None` with no indication of the problem.
**Fix:** Add `else: raise RuntimeError(f"job is not complete (status={self.status!r})")`.

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | MINOR | `implementation.py:17` | Mutable ndarray in frozen dataclass |
| 2 | MINOR | `engine.py:7-14` | Stray TODO comments |
| 3 | MINOR | `test_implementation.py:2` | Unused `pytest` import |
| 4 | MINOR | `test_implementation.py:13` | Stray misspelled TODO comment |
| 5 | MINOR | `program.py:78` | Stray TODO + cryptic bool fallthrough |
| 6 | MINOR | `errors.py:22` | Inaccurate `NoMeasurementWarning` docstring |
| 7 | MINOR | `registers.py:13`, `program.py:163` | Metadata stored by reference |
| 8 | MINOR | `program.py:134` | Empty condition tuple → `IndexError` |
| 9 | MINOR | `program.py:127` | Duplicate qubit targets not caught at construction |
| 10 | MINOR | `implementation.py:17` | `MatrixImplementation.target_indices` unused |
| 11 | MINOR | `job.py:21` | `Job.result()` returns `None` for non-terminal status |
