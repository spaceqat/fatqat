OpenQASM
========

.. py:module:: fatqat.qasm

Use :func:`from_qasm` or :func:`from_qasm_file` to import OpenQASM 2 or 3 into
a :class:`~fatqat.Program`, and :func:`to_qasm` to export a program. These
functions do not require Qiskit. Existing code can continue to use
``qasm_to_program`` and ``program_to_qasm``.

.. code-block:: python

   from fatqat.qasm import from_qasm, to_qasm

   program = from_qasm(
       "OPENQASM 3.0; "
       'include "stdgates.inc"; '
       "qubit[2] q; h q[0]; cx q[0], q[1];"
   )
   source = to_qasm(program)  # OpenQASM 3.0 by default

Import support
--------------

Imported quantum and classical registers have dimension 2. FATQAT supports
scalar operands, equal-sized whole-register operations, measurement, reset,
local ``gate`` definitions, and the built-ins listed by :func:`from_qasm`.
Barriers are ignored. A condition can guard a gate or reset using
whole-register equality or an AND of bit comparisons.

``include`` statements do not load files. A gate normally supplied by an
include must already be built in or defined in a local ``gate`` block. Other
control flow, gate modifiers, declarations, and classical conditions raise
:exc:`QASMTranspileError`.

Export support
--------------

:func:`to_qasm` emits OpenQASM 3 by default. The program must be fully bound.
Use ``version=2`` only when each condition compares every bit in a single
classical register.

Export requires ``dim == 2`` for every register and scalar operation targets.
Supported gates and dimension-2 reductions are listed by :func:`to_qasm`.
Register names may change to avoid conflicts, and program metadata is omitted.

Barrier, direct pulse controls, :class:`~fatqat.RegisterView` targets, and
operations without a QASM representation raise :exc:`QasmExportError`.

Errors
------

Catch :exc:`fatqat.errors.FatqatError` to handle either conversion error. Catch
:exc:`QASMTranspileError` to handle translator rejection separately from
:exc:`QasmExportError`, which reports a program that cannot be represented.
For compatibility, :exc:`QASMTranspileError` is also a :exc:`ValueError`.

File I/O and text-decoding failures from :func:`from_qasm_file` use the
underlying Python exceptions.

Reference
---------

.. autofunction:: from_qasm

.. autofunction:: from_qasm_file

.. autofunction:: to_qasm

.. autoexception:: QASMTranspileError
   :no-members:
   :no-inherited-members:

.. autoexception:: QasmExportError
   :no-members:
   :no-inherited-members:
