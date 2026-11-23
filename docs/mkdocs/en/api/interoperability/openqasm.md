---
title: "OpenQASM"
---

# OpenQASM

<a id="fatqat.qasm"></a>

OpenQASM 2 and 3 both use the same conversion functions. Read source text with
[`from_qasm()`][fatqat.qasm.from_qasm] or a file with
[`from_qasm_file()`][fatqat.qasm.from_qasm_file]. These return a
[`Program`][fatqat.Program]. To turn a program back into OpenQASM text, use
[`to_qasm()`][fatqat.qasm.to_qasm]. All three functions work without Qiskit.

`qasm_to_program` and `program_to_qasm` are aliases for the text conversion
functions.

## Import a circuit { #import-support }

Imported quantum and classical registers have dimension two. Along with the
supported built-in gates, you can import measurements, resets, and local `gate`
definitions. A gate can target a single register member or act element by element
on equal-length registers. If a call combines a single member with a whole
register, FatQat reuses that member for each application. Import skips `barrier`
statements.

You can condition a gate or reset on a whole register equaling a value, or on
several bit comparisons that must all hold. The importer skips `include`
directives, so the source must define any gate that FatQat does not provide. See
[`from_qasm()`][fatqat.qasm.from_qasm] for the supported expressions and gates.

## Export a program { #export-support }

[`to_qasm()`][fatqat.qasm.to_qasm] writes OpenQASM 3 by default. Use `version=2`
when the receiving tool needs OpenQASM 2 and each condition fixes every bit of one
classical register. OpenQASM 3 can represent combinations of individual bit
comparisons.

A program must have bound parameters, dimension-two registers, and targets
specified as individual register members. `Barrier` exports as a native `barrier`
statement. Direct pulse controls,
[`RegisterView`](../registers.md#fatqat.RegisterView) targets, and operations
without an OpenQASM representation are rejected. Export preserves operand order;
register names may change and program metadata is omitted.

In a whole-register condition, bit `i` contributes `2**i` to the integer. For
example, `(c0, c1) = (1, 0)` becomes `if (c == 1)`. FatQat still puts `c0` on the
left when displaying counts.

## Handle conversion errors { #errors }

[`QASMTranspileError`][fatqat.qasm.QASMTranspileError] reports source that FatQat
cannot import. [`QasmExportError`][fatqat.qasm.QasmExportError] reports a program
that cannot be written in the requested version. Catch
[`FatqatError`][fatqat.errors.FatqatError] to handle both. Existing import
handlers can also catch
[`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError).

If [`from_qasm_file()`][fatqat.qasm.from_qasm_file] cannot read or decode the
file, it raises the corresponding Python exception.

## API reference { #reference }

::: fatqat.qasm.from_qasm

::: fatqat.qasm.from_qasm_file

::: fatqat.qasm.to_qasm

::: fatqat.qasm.QASMTranspileError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.qasm.QasmExportError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
