# Compile a logical circuit

!!! warning "Compiler under development"

    The compiler module is under active development. Its interfaces and
    supported behavior may change between releases. Pin an exact FatQat
    version when reproducibility matters.

Build a [`LogicalProgram`][fatqat.LogicalProgram] when you want FatQat to choose
a hardware-family instruction set, map logical qubits, and route two-qubit
gates. It uses the same authoring interface as [`Program`][fatqat.Program] for
direct simulation:

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2)
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
```

Compilation snapshots the LogicalProgram without editing it, so the same source can
still be simulated directly or compiled for another target. The compiler
accepts the static, numeric gate subset described below; direct simulation
continues to support conditions on LogicalProgram and the broader Program
operation set. The Python compiler entry points accept only an exact
LogicalProgram; an ordinary Program remains an execution representation and is
not converted implicitly.

## Compile and run on an SC profile

The public superconducting route targets
[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator]. Its coupling graph is
the compiler's routing target; it may be any valid graph and does not need to
be a grid.

```python
sc_backend = fq.simulator.SCQubitSimulator(
    num_qubits=3,
    couplings=((0, 1), (1, 2)),
    runtime="numpy",
)

compiled = fq.compiler.compile_to_sc(circuit, sc_backend, seed=7)

counts = (
    sc_backend.run(
        compiled,
        shots=100,
        simulation_config={"seed": 7},
    )
    .result()
    .get_counts()
)
```

The route freezes the editable circuit, normalizes it to the common SC gate
IR, performs SABRE mapping and routing, and lowers it to the public
X/SX/RZ/CZ native profile. The rotation/iSWAP profile is currently private and
is not part of the public compiler contract.

## Compile for a neutral-atom architecture

The neutral-atom route uses the bundled ZAP scheduler:

```python
from fatqat.compiler.algorithms.zap import load_architecture

architecture = load_architecture("default")
compiled = fq.compiler.compile_to_na(circuit, architecture)
na_backend = fq.simulator.AtomArraySimulator(runtime="numpy")

counts = na_backend.run(compiled, shots=100).result().get_counts()
```

Both target compilers return an executable result at their default final
boundary. Pass that result directly to the matching simulator. The final
compiler IR remains available as `compiled.output`: an `SCNativeProgram` for
SC or a `ZonedPlan` for NA. `compiled.route` records the passes that ran.

An explicit intermediate `emit` returns an inspectable
[`CompilationResult`][fatqat.compiler.CompilationResult] rather than an
executable result. The low-level
[`to_sc_simulator_program`][fatqat.compiler.to_sc_simulator_program] and
[`to_na_simulator_program`][fatqat.compiler.to_na_simulator_program] functions
remain available when advanced callers construct or modify final IR directly.

## Supported source behavior

The v0.3 compiler accepts static, numeric gate circuits. Measurements must be
terminal. Bind symbolic parameters with `assign_parameters()` before
compilation; mid-circuit
measurement, conditions, feed-forward, and general control flow are not yet
compiler inputs.

```python
theta = fq.Parameter("theta")
template = fq.LogicalProgram(1)
template.add(fq.operations.RX(theta), 0)
bound = template.assign_parameters({theta: 0.25})
```

Each classical slot may be written at most once. Register views are expanded
into scalar gate occurrences when the source is frozen, preserving register
identity and operand order.

`LogicalProgram` rejects device operations and custom operation classes while
the circuit is authored. Built-in logical operations are checked for static
compiler and target support at later boundaries. Passing an ordinary `Program`
to a Python compiler entry point raises `ValidationError` instead of silently
changing its meaning or operation set.

Target normalization reports unsupported combinations: in particular, the
current NA route rejects `SX` and `Reset`. Such a failure is reported as a
[`PassError`][fatqat.compiler.PassError] naming the pass and the underlying
unsupported operation.

OpenQASM remains an equal frontend through
[`compile_qasm_to_sc`][fatqat.compiler.compile_qasm_to_sc] and
[`compile_qasm_to_na`][fatqat.compiler.compile_qasm_to_na]. Python and QASM
inputs converge at the same immutable logical IR, so all later lowering is
shared. The OpenQASM `U`, `u1`, `u2`, and `u3` families are normalized into
the existing target rotations; they are not added to the SC or NA hardware
dialects. Their default results use the same direct execution API:

```python
qasm_source = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

sc_compiled = fq.compiler.compile_qasm_to_sc(qasm_source, sc_backend)
sc_result = sc_backend.run(sc_compiled).result()

na_compiled = fq.compiler.compile_qasm_to_na(qasm_source, architecture)
na_result = na_backend.run(na_compiled).result()
```
