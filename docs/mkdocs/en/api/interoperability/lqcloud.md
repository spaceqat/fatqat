# LogicalQubit Cloud

`LQCloudBackend` connects the SC compiler to AGate-100, QZ01-surface_code and
MQ02. It uses the official optional SDK for authentication, submission and
polling. See the [cloud hardware guide](../../guide/lqcloud.md) for installation
and a complete example.

## Backend and compilation

```python
from fatqat.lqcloud import LQCloudBackend
import fatqat as fq

backend = LQCloudBackend("QZ01-surface_code")
program = fq.LogicalProgram(2, 2)
program.add(fq.operations.H, 0)
program.add(fq.operations.CX, (0, 1))
program.measure_all()
compiled = fq.compiler.compile_to_sc(program, backend)
```

Construction fetches device configuration once. Compilation uses that snapshot,
does not submit tasks and does not mutate the source. `run()` accepts a final
cloud compilation for the same device and topology, not a bare Program or an
earlier `emit` result. The shot range is 1–50,000.

The compiler defaults to `sc.lqcloud.native.v1` for this backend. Select an
earlier boundary with its source/IR type's `IR_ID`; that result is for
inspection, not execution.

::: fatqat.lqcloud.LQCloudBackend

::: fatqat.lqcloud.LQCloudCompilationResult

## Jobs and results

Submission returns an asynchronous `LQCloudJob`. Its status method queries
the SDK and returns an uppercase cloud status, distinct from the completed-only
`fatqat.Job` returned by current simulators.

`result()` returns `fatqat.Result` with only `counts` available. Metadata
contains `job_id` and `backend`. Classical slots retain source declaration
order, including unused slots. Invalid response widths or shot totals are
errors, not silently repaired results. Network, authentication and remote
execution errors use SDK exception types.

Treat the SDK circuit inside a compiled result as read-only. Compile again
when changing the logical circuit or device, rather than editing physical
wires or overriding the transport mapping. Use separate backend handles when
submitting concurrently.

::: fatqat.lqcloud.LQCloudJob
