---
title: "Backend"
---

# Backend

Use [`Backend`][fatqat.Backend] as a type annotation when a function can run a
[`Program`][fatqat.Program] on different execution backends. The shared call is
`run(program)`, which returns a [`Job`][fatqat.Job] whose result is a
[`Result`][fatqat.Result]. Any implementation with that interface satisfies the
protocol.

The concrete backend defines any extra run options and which result fields it
produces.

```python
from fatqat import Backend, Program, Result

def execute(backend: Backend, program: Program) -> Result:
    return backend.run(program).result()
```

## API reference

::: fatqat.Backend
    options:
      members:
        - run
      show_bases: false
