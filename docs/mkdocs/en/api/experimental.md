---
title: "Experimental APIs"
---

# Experimental APIs


The APIs listed here are public, but their signatures and behavior may change
between releases. Use them only when the stable backend workflow does not
cover your integration.

## Compiler

The [compiler](compiler.md) is under development; its interfaces and supported
behavior may change. Its [LogicalProgram](logical-program.md) frontend accepts
built-in device-independent operations and is required by the Python compiler
entry points.

## Direct construction


Backends return [`Job`][fatqat.Job] and [`Result`][fatqat.Result];
applications normally should not construct either class. When writing a
backend integration, create a successful job as `Job("DONE", result=value)`
or a failed job as `Job("ERROR", error=exc)`, and translate the backend's
output into a `Result`. See [Job](job.md) and [Result](result.md) for the interfaces.

## Extension points


The following extension points are also evolving:

- [Matrix implementations](implementation.md) for custom matrices and device-specific gate rules
- [Gate realization](pulse-control/gate-realization.md) for custom pulse definitions

Built-in pulse maps are listed under [Emulators](emulators/index.md).
