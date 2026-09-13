# Compile and run a logical circuit

FatQat can prepare the same logical circuit for a superconducting simulator or
a neutral-atom simulator. The normal workflow has three steps:

1. Build a [`LogicalProgram`][fatqat.LogicalProgram].
2. Compile it for a target.
3. Run the compiled result on the matching simulator.

## Build a circuit

This Bell-state circuit will be used for both targets:

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2)
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
```

`LogicalProgram` uses the same circuit-building interface as
[`Program`][fatqat.Program]. You can simulate it directly while developing an
algorithm, then compile it when you are ready to run against a hardware
profile. Compilation does not modify the circuit, so it can be reused with
more than one target.

## Run on a superconducting profile

Create an [`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] with the
number of physical qubits and their allowed two-qubit connections. The
compiler maps the logical qubits to this topology and adds routing operations
where they are needed.

```python
sc_backend = fq.simulator.SCQubitSimulator(
    num_qubits=3,
    couplings=((0, 1), (1, 2)),
    runtime="numpy",
)

sc_compiled = fq.compiler.compile_to_sc(circuit, sc_backend, seed=7)
result = sc_backend.run(
    sc_compiled,
    shots=100,
    simulation_config={"seed": 7},
).result()

counts = result.get_counts()
```

`couplings` can describe any valid graph; the topology does not need to be a
grid. The current public SC profile uses X, SX, RZ, and CZ as its native gate
set.

## Run on a neutral-atom architecture

For neutral atoms, load an architecture and compile the same circuit with
`compile_to_na()`:

```python
from fatqat.compiler.algorithms import load_architecture

architecture = load_architecture("default")
na_compiled = fq.compiler.compile_to_na(circuit, architecture)

na_backend = fq.simulator.AtomArraySimulator(runtime="numpy")
result = na_backend.run(na_compiled, shots=100).result()
counts = result.get_counts()
```

The architecture describes the zones and movement constraints used to create
the neutral-atom schedule. FatQat also includes `scale_to_100` and
`scale_to_500` profiles for larger studies.

## Compile OpenQASM

OpenQASM 2 and OpenQASM 3 can be compiled without first creating a
`LogicalProgram`. Choose the function that matches the target:

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

OpenQASM `U`, `u1`, `u2`, and `u3` gates are converted to rotations supported
by the target compilation route.

## Inspect a compilation

The default compile functions return a result that is ready for the matching
simulator. It also keeps two useful views of the compilation:

- `compiled.output` is the final compiler representation: an
  `SCNativeProgram` for SC or a `ZonedPlan` for NA.
- `compiled.route` lists the compiler stages that ran.

Compiled programs keep classical registers in source declaration order,
including unused registers and unwritten slots. Measurement order does not
change count-key positions. Counts follow the usual
[ordering and zero-fill rules](../api/result.md#ordering-and-mutable-values).

To inspect an earlier representation, pass its IR identifier with `emit`. In
that case the compiler returns a
[`CompilationResult`][fatqat.compiler.CompilationResult] for inspection rather
than a simulator-ready result.

For a neutral-atom compilation, you can animate the resulting schedule:

```python
animation = fq.compiler.create_na_animation(na_compiled.output, architecture)
fq.compiler.save_na_animation(animation, "na-schedule.mp4")
```

Creating the animation requires Matplotlib. Saving it as MP4 also requires
FFmpeg on `PATH`.

## Bind circuit parameters

Keep symbolic parameters while building a reusable circuit, then bind numeric
values before compiling:

```python
theta = fq.Parameter("theta")

template = fq.LogicalProgram(1)
template.add(fq.operations.RX(theta), 0)

circuit = template.assign_parameters({theta: 0.25})
compiled = fq.compiler.compile_to_sc(circuit, sc_backend)
```

`assign_parameters()` returns a new `LogicalProgram`; the template remains
available for another set of values.

## Current compiler support

The compiler currently handles static circuits with numeric parameters. Both
targets support I, H, X, Y, Z, S, Sdg, T, Tdg, RX, RY, RZ, Phase, U, U1, U2,
U3, CX, CZ, and Swap. The SC route also supports SX and Reset. Measurements
belong at the end of the circuit, and each classical slot can receive one
measurement result.

Other operations available to `LogicalProgram`, including classical
conditions, feed-forward, mid-circuit measurement, multi-controlled gates,
and qudit gates, can still be used in direct simulation but are not compiled
yet. If a circuit uses an operation unavailable on its chosen target,
compilation raises a [`PassError`][fatqat.compiler.PassError] that identifies
the compiler stage and operation.
