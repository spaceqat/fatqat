# Compile a logical circuit

Use [`LogicalProgram`][fatqat.LogicalProgram] when you want FatQat to choose a
hardware-family instruction set, map logical qubits, and route two-qubit gates.
It is a small, editable circuit frontend with familiar gate methods:

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
```

`LogicalProgram` is a restricted [`Program`][fatqat.Program] subclass. It
uses the same quantum registers, classical registers, references, and
instruction storage, so a general simulator can run it directly. Its authoring
interface rejects device operations such as atom placement, pairing, and
direct pulse controls. Its generic `add()` method accepts FatQat's built-in
device-independent operations; custom operations remain available through the
broader `Program` interface. The compiler accepts `LogicalProgram`, not the
broader `Program`.

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
compilation; mid-circuit measurement, conditions, feed-forward, and general
control flow are not yet compiler inputs. A `LogicalProgram` may still contain
conditions and run directly on the general simulator; compilation reports that
unsupported boundary explicitly.

```python
theta = fq.Parameter("theta")
template = fq.LogicalProgram(1).rx(theta, 0)
bound = template.assign_parameters({theta: 0.25})
```

The frontend offers the union of the current logical gate methods. Target
normalization reports unsupported combinations: in particular, the current NA
route rejects `sx()` and `reset()`. Such a failure is reported as a
[`PassError`][fatqat.compiler.PassError] naming the pass and the underlying
unsupported operation.

Register views are a construction convenience inherited from `Program`.
Freezing a logical program expands a view into scalar gate occurrences before
creating immutable logical IR; compiler IR continues to use the same
`RegisterRef` identities rather than introducing separate IR qubit objects.

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
