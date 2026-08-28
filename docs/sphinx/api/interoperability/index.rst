Interoperability
================

Draw programs, import or export supported OpenQASM, and run compatible Qiskit
circuits with FATQAT.

.. toctree::
   :maxdepth: 1

   openqasm
   qiskit

Circuit drawing
---------------

:meth:`fatqat.Program.draw` is the usual way to get a matplotlib figure or a
terminal diagram. Its options and return values are documented with
:doc:`../program`.

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   program = fq.Program(2)
   program.add(ops.H, 0)
   program.add(ops.CX, (0, 1))

   figure = program.draw()
   figure.savefig("bell-circuit.png")
   print(program.draw("text"))

Use :func:`fatqat.draw.to_qubit_circuit` only when integrating with QuTiP-QIP's
drawing tools. The returned circuit is for rendering, not simulation. Qudit
dimensions are not shown, pulse operations are unsupported, and custom
operations may appear only as labeled boxes.

.. autofunction:: fatqat.draw.to_qubit_circuit
