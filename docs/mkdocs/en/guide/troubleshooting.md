# Diagnose a Program workflow

Most first-run problems occur at one of four boundaries:

```text
write the Program
       |
       v
match the target
       |
       v
configure the run
       |
       v
read the Result
```

Start with the symptom below. You should not need to inspect FatQat's internal
execution engine.

## Writing the Program

???+ question "An integer target is ambiguous"

    Bare integers are convenient for `Program(2)`, where there is only one
    quantum register. Once a Program contains several registers, use the register
    references you created:

    ```python
    qubit = fq.QuantumRegister(1, name="qubit")
    qutrit = fq.QuantumRegister(1, dim=3, name="qutrit")
    program = fq.Program([qubit, qutrit])

    program.add(ops.X, qubit[0])
    program.add(ops.Shift(1), qutrit[0])
    ```

    This also makes each target's local dimension unambiguous. See
    [Write quantum computations with Program](program.md).

??? question "Measurement reports a dimension mismatch"

    A measured quantum resource and its classical destination must have the same
    local dimension. A qutrit needs a classical trit, not an ordinary bit:

    ```python
    qutrit = fq.QuantumRegister(1, dim=3)
    trit = fq.ClassicalRegister(1, dim=3)
    program = fq.Program([qutrit], [trit])
    program.measure(qutrit[0], trit[0])
    ```

??? question "The Program still contains an unbound parameter"

    Binding returns a new Program; it does not modify the template in place.
    Keep the returned value and submit that one:

    ```python
    bound_program = template.assign_parameters({theta: 0.4})
    backend.run(bound_program)
    ```

    For a batch of bindings, use the sweep workflow in [Simulate a quantum
    program](simulation.md).

## Matching the execution target

???+ question "An operation works on Simulator but is unsupported here"

    That usually means the selected hardware profile or emulator cannot realize
    the instruction as written. The general simulator accepts a wider logical
    operation set; constrained targets intentionally do not.

    Check the target's supported operations and ask which detail you meant to
    study:

    - return to the general simulator for unrestricted algorithm behavior;
    - rewrite the Program in the profile's native operations when hardware
      constraints are the point; or
    - supply an implementation/calibration only if you are deliberately extending
      the target.

    FatQat does not silently transpile or route the Program. See [Choose how much
    physics to model](execution-models.md).

??? question "A two-resource gate fails because of placement"

    The operation can be native while its two physical locations are not
    connected. Inspect or choose a [`ResourceLayout`][fatqat.ResourceLayout] and keep the
    ordered logical targets distinct from their device labels. The worked
    failure-and-fix is in [Test a Program against a hardware
    profile](hardware-profile-simulation.md).

??? question "A direct control is rejected by an emulator"

    Each pulse channel comes from a particular physical model and identifies one
    of that model's resources. Build the control from the same model used by the
    emulator. Then check that the waveform uses the model's time and control units,
    that its values are allowed, and that the operation duration covers every
    control. [Follow a Program into physical
    dynamics](hamiltonian-emulation.md) builds this chain one piece at a time.

## Configuring the run

??? question "Counts change each time"

    Counts are sampled. Increase `shots` when you want a steadier estimate, and
    set a seed when you need to reproduce a particular sample:

    ```python
    result = backend.run(
        program,
        shots=1000,
        simulation_config={"seed": 7},
    ).result()
    ```

    A fixed seed aids debugging; it does not make a small sample more accurate.

??? question "A statevector is unavailable for a noisy or dynamic run"

    After measurement, reset, or trajectory noise, one statevector is only one
    sampled history. Request it with `shots=1` when that single history is what you
    want. Use density-matrix simulation for an exact mixed state.

??? question "The first Numba run is unexpectedly slow"

    Numba compiles kernels lazily. Warm the workload before measuring repeated
    runtime, or choose NumPy when predictable one-off latency matters more. The
    benchmark pattern is in [Performance and scaling](performance.md).

## Reading the Result

???+ question "A Result accessor says its data is unavailable"

    Runs contain only the data that the selected execution method produced and
    that you requested. Inspect the result before choosing an accessor:

    ```python
    print(result.available_data)
    if "statevector" in result.available_data:
        state = result.get_statevector()
    ```

    [Ask questions of a run](interpret-results.md) connects each output to the
    question it answers.

??? question "Count strings appear reversed"

    Displayed count strings put classical slot 0 on the right. When local
    dimensions differ—or when you simply want an unambiguous program-order
    representation—use `result.get_counts_as_tuples()` instead. Tuple position 0
    is classical slot 0.

## Reduce the problem without losing the clue

If none of those matches, make a copy of the failing workflow and reduce it:

1. keep one register and one operation;
2. run it with the general `Simulator`;
3. add the requested result field;
4. switch to the intended profile or emulator; and
5. restore operations one at a time.

The step where it changes tells you which boundary to inspect. Preserve the
small reproducer—it is also the most useful starting point for a bug report.
