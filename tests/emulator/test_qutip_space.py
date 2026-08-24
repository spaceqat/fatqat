"""Canonical FATQAT axes at the QuTiP construction boundary."""

import numpy as np
from qutip import basis, ket2dm

from fatqat._index_allocation import _EngineAllocation
from fatqat.emulator._qutip_space import _QutipTensorSpace


def test_qutip_space_preserves_canonical_mixed_radix_flat_order():
    canonical = _EngineAllocation(("qubit", "qutrit"), (2, 3))
    space = _QutipTensorSpace(canonical)

    state = space.full_tensor((basis(2, 1), basis(3, 1)))
    axis_zero_projector = space.expand_local(0, ket2dm(basis(2, 1)))

    assert space.dims == (3, 2)
    assert space.targets((0, 1)) == (1, 0)
    assert state.full().reshape(-1).nonzero()[0].tolist() == [3]
    assert np.flatnonzero(axis_zero_projector.diag()).tolist() == [1, 3, 5]
