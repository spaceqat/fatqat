"""Visualization support for FATQAT programs and results.

Use domain-object methods such as :meth:`fatqat.Program.draw` for normal
drawing. The public helpers exported here are intended for integrations that
need access to intermediate visualization objects.
"""

from .qutip_circuit import to_qubit_circuit

__all__ = ["to_qubit_circuit"]
