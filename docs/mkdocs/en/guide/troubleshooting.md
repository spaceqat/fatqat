# Troubleshoot common problems { #diagnose-a-program-workflow }

Find the symptom that matches what you see. Start with the suggested check or
fix, then follow the linked example if you need more context.

## Writing the Program

???+ question "An integer target is ambiguous"

    If your Program has multiple quantum registers, specify the register as
    well as the index. For example, use `qubit[0]` to pick slot 0 in `qubit`.

    <!-- guide-test: skip -->
    ```python
    qubit = fq.QuantumRegister(1, name="qubit")
    qutrit = fq.QuantumRegister(1, dim=3, name="qutrit")
    program = fq.Program([qubit, qutrit])

    program.add(ops.X, qubit[0])
    program.add(ops.Shift(1), qutrit[0])
    ```

    See [Program & Register](program.md) for more examples of register references.

??? question "Measurement reports a dimension mismatch"

    Check that the classical destination has the same dimension as the resource
    you measure. For a qutrit, use a classical register with `dim=3`.

    <!-- guide-test: skip -->
    ```python
    qutrit = fq.QuantumRegister(1, dim=3)
    trit = fq.ClassicalRegister(1, dim=3)
    program = fq.Program([qutrit], [trit])
    program.measure(qutrit[0], trit[0])
    ```

??? question "The Program still contains an unbound parameter"

    Run the Program returned by `assign_parameters()`. That Program contains the
    bound values, and the original is available for binding another set.

    <!-- guide-test: skip -->
    ```python
    bound_program = template.assign_parameters({theta: 0.4})
    backend.run(bound_program)
    ```

    For a batch of bindings, use the sweep workflow in [Simulate a quantum
    program](simulation.md).

## Matching the execution target

???+ question "An operation works on Simulator but is unsupported here"

    Check the device or emulator's supported operations first. Use the general
    `Simulator` to check the algorithm itself. To model the selected device,
    write the Program using its native operations or
    [compile a supported logical circuit](compiler.md) for that target.

??? question "A two-qubit gate fails on the selected positions"

    Look up the two physical positions in the
    [`ResourceLayout`][fatqat.ResourceLayout] and check that the device connects
    them. Use a pair that shares a connection, as in
    [the superconducting device example](superconducting-simulation.md).

??? question "A direct control is rejected by an emulator"

    First check that the channel is available on the model passed to the
    emulator. Then check the waveform's units and allowed values, and make sure
    the pulse duration covers every control. See
    [Write and run pulse programs](hamiltonian-emulation.md) for a worked example.

## Configuring the run

??? question "Counts change each time"

    Increase `shots` for a steadier estimate. Set a seed when you need the
    same sample to reproduce a result.

    <!-- guide-test: skip -->
    ```python
    result = backend.run(
        program,
        shots=1000,
        execution_config={"seed": 7},
    ).result()
    ```

??? question "A statevector is unavailable for a noisy or dynamic run"

    To inspect one trajectory after measurement, reset, or trajectory noise,
    run with `shots=1` and request the final state. Use density-matrix simulation
    when you need the mixed state averaged over trajectories.

??? question "The first Numba run is unexpectedly slow"

    Run the same workload once before timing it. Numba compiles kernels the first
    time they run. To time a one-off run, try NumPy. See
    [Performance and scaling](performance.md) for the benchmark setup.

## Reading the Result

???+ question "A requested field is missing from the Result"

    Inspect `result.available_data` to see which fields the run produced. If
    the field you need is missing, check the requested outputs and simulation method.

    <!-- guide-test: skip -->
    ```python
    print(result.available_data)
    if "statevector" in result.available_data:
        state = result.get_statevector()
    ```

    The [Result reference](../api/result.md) lists the available fields, their
    accessors, and which runs produce them.

??? question "How are count strings ordered?"

    Read a FatQat count string from classical slot 0 on the left. Tuple keys
    follow the same order and are easier to read when slots have different
    dimensions. When any classical dimension is at least 10, the strings
    separate entries with commas.

## Make a small example that reproduces the problem { #reduce-the-problem-without-losing-the-clue }

If none of those matches, make a copy of the failing workflow and reduce it:

1. keep one register and one operation;
2. run it with the general `Simulator`;
3. add the requested result field;
4. switch to the intended profile or emulator; and
5. restore operations one at a time.

The step where it changes tells you which boundary to inspect. Preserve the
small reproducer—it is also the most useful starting point for a bug report.
