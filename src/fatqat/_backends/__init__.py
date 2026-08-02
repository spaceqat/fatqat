"""Private infrastructure shared by the simulator and emulator backend families.

This package holds everything the two families have in common and nothing
specific to either: the lowered execution plan (`steps`), the backend/engine
run contract (`engine_contract`), the lowering helpers and config normalizer
(`backend_utils`), program view normalization, and terminal-measurement
analysis.

It is private. Users reach simulation through :mod:`fatqat.simulator` and
:mod:`fatqat.emulator`; nothing here is part of the public API. Keeping it
separate is what lets `simulator` and `emulator` share this code without either
importing the other - both depend on this package, and it depends on neither.

Modules are imported directly (``from fatqat._backends.steps import ...``);
this package deliberately re-exports nothing.
"""
