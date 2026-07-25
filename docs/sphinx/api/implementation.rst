Implementation (``fq.implementation``)
=========================================

Matrix-implementation rules and the implementation map that backends use to
resolve an operation to its local matrix. Most users only need
``default_matrix_implementation_map``; the rest of this namespace is for
building device-specific maps (see ``SCQubitIBMSimulator`` and
``SCQubitGoogleSimulator``).

.. autofunction:: fatqat.implementation.default_matrix_implementation_map

.. autoclass:: fatqat.implementation.ImplementationMap
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.MatrixImplementation
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.FixedMatrix
   :members:
   :show-inheritance:

``DeviceOperands`` is a type alias, ``tuple[Hashable, ...]``, for the
key identifying a device-specific target (e.g. a qubit pair for a
nearest-neighbor two-qubit gate).
