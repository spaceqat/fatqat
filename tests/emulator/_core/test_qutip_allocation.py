"""Canonical-axis to QuTiP-factor translation contract."""

from qutip import basis, tensor

from fatqat._index_allocation import _EngineAllocation
from fatqat.emulator._core.qutip_allocation import _QutipEngineAllocation


def test_qutip_view_preserves_canonical_mixed_radix_flat_order():
    canonical = _EngineAllocation(("qubit", "qutrit"), (2, 3))
    allocation = _QutipEngineAllocation(canonical)

    canonical_digits = (1, 1)
    state = tensor(
        *(
            basis(dimension, digit)
            for dimension, digit in zip(
                allocation.qutip_dims,
                allocation.factor_order(canonical_digits),
            )
        )
    )

    assert allocation.qutip_dims == (3, 2)
    assert allocation.factor_indices((0, 1)) == (1, 0)
    assert state.full().reshape(-1).nonzero()[0].tolist() == [3]
