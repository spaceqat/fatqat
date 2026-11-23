---
title: "Experimental APIs"
---

# Experimental APIs

These APIs are available for use while their interfaces are under development.
Their signatures and behavior may change between releases.

## Compiler

Start with [LogicalProgram](logical-program.md) if you write the logical gates in
Python. If you already have OpenQASM source, choose the OpenQASM entry point for
your target. [Compiler pipelines](compiler-pipelines.md) lists the inputs each
entry point accepts.

## Custom backends { #direct-construction }

An implementation of [`Backend`][fatqat.Backend] puts its output fields in a
[`Result`][fatqat.Result] and returns a [`Job`][fatqat.Job]. Use `Job("DONE",
result=value)` to return the result. To store an execution failure for the caller,
use `Job("ERROR", error=exc)`.

## Extend a backend { #extension-points }

- [Matrix implementation maps](implementation.md) let a simulator use your own
  gate matrices or pick a rule for a specific device location.

- [Custom gate pulses](pulse-control/gate-realization.md) let you specify the
  physical controls that implement a gate. The [emulator
  pages](emulators/index.md) describe the built-in gate-to-pulse maps.
