"""Thin cloud submission wrapper for LQCloud's QEC17 backend."""

from __future__ import annotations

from typing import Any

from .. import operations as ops
from .._backends.view_normalization import _break_grouped_operations
from ..program import Program, _AppliedOperation
from ._qec17 import CZ_EDGES, NUM_QUBITS
from .converter import (
    _flatten_quantum_refs,
    _load_lqcloud,
    program_to_qec17_circuit,
)


def _normalize_initial_layout(
    n_qubits: int, initial_layout: list[int] | tuple[int, ...] | None
) -> tuple[tuple[int, ...], list[int] | None]:
    """Validate a logical-to-physical QEC17 layout."""
    if initial_layout is None:
        identity = tuple(range(n_qubits))
        return identity, None
    if not isinstance(initial_layout, (list, tuple)):
        raise TypeError(
            "initial_layout must be a list/tuple of physical qubit indices, "
            f"got {type(initial_layout).__name__}"
        )
    if len(initial_layout) != n_qubits:
        raise ValueError(
            f"initial_layout length ({len(initial_layout)}) must equal the "
            f"number of program qubits ({n_qubits})"
        )

    normalized: list[int] = []
    for index, physical in enumerate(initial_layout):
        if isinstance(physical, bool) or not isinstance(physical, int):
            raise ValueError(
                f"initial_layout[{index}] must be an int physical qubit index, "
                f"got {type(physical).__name__}: {physical!r}"
            )
        if not 0 <= physical < NUM_QUBITS:
            raise ValueError(
                f"initial_layout[{index}]={physical} is outside the QEC17 "
                f"physical range 0..{NUM_QUBITS - 1}"
            )
        normalized.append(physical)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "initial_layout must not contain duplicate physical indices, "
            f"got {normalized}"
        )
    return tuple(normalized), normalized


def _validate_mapped_cz_edges(
    program: Program, refs: dict[Any, int], layout: tuple[int, ...]
) -> None:
    """Reject every CZ whose physical endpoints are not QEC17 neighbors."""
    for step in _break_grouped_operations(program._instructions):
        if not isinstance(step, _AppliedOperation) or step.operation is not ops.CZ:
            continue
        logical = tuple(refs[target] for target in step.targets)
        physical = (layout[logical[0]], layout[logical[1]])
        if frozenset(physical) not in CZ_EDGES:
            raise ValueError(
                f"CZ on logical qubits {logical} maps to non-adjacent QEC17 "
                f"physical qubits {physical}"
            )


class LQCloudQEC17Backend:
    """Submit native FATQAT programs to LQCloud ``QZ01-surface_code``.

    This adapter deliberately remains thin: it translates a single FATQAT
    `Program`, validates fixed QEC17 constraints, and returns the LQCloud SDK's
    native asynchronous ``Job`` unchanged.

    Args:
        api_key: LQCloud API key.
        url: Optional LQCloud service URL override. ``None`` uses the SDK's
            default service URL.

    Attributes:
        name: Fixed backend name, ``"QZ01-surface_code"``.
        num_qubits: Physical capacity, 17 qubits.

    Raises:
        ImportError: If the optional LQCloud SDK is not installed.
        TypeError: If ``api_key`` is not a string.
        ValueError: If ``api_key`` is empty.
    """

    name = "QZ01-surface_code"
    num_qubits = NUM_QUBITS

    def __init__(self, api_key: str, *, url: str | None = None) -> None:
        if not isinstance(api_key, str):
            raise TypeError(f"api_key must be a string, got {type(api_key).__name__}")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")

        lqcloud_sdk = _load_lqcloud()
        provider_options: dict[str, Any] = {
            "api_key": api_key,
            "interactive": False,
        }
        if url is not None:
            provider_options["url"] = url
        self._provider = lqcloud_sdk.LQCloudProvider(**provider_options)
        # The target name is fixed by this adapter, so a discovery request is
        # unnecessary. Submission remains fully delegated to the SDK.
        self._cloud_backend = self._provider.get_backend(self.name, verify=False)

    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        initial_layout: list[int] | tuple[int, ...] | None = None,
        **run_options: Any,
    ) -> Any:
        """Translate and submit one program, returning the native LQCloud Job.

        ``initial_layout[k]`` maps flattened logical qubit ``k`` to one of the
        physical QEC17 indices 0 through 16. Additional keyword options are
        passed through to ``lqcloud`` (for example ``result_format`` or
        ``readout_correction``).

        Args:
            program: One static, bound Program using QEC17 native gates and
                no explicit resets or measurements. The adapter appends a
                full barrier and measurement of every logical qubit.
            shots: Number of repetitions, from 1 through 50,000. Default 1024.
            initial_layout: List or tuple of distinct physical indices, one
                per logical qubit. ``None`` uses the identity mapping.
            **run_options: Additional options passed unchanged to the SDK.
                Their defaults and validation are defined by LQCloud.

        Returns:
            Any: The LQCloud SDK's asynchronous Job, unchanged. Its result
                and bit ordering follow LQCloud, not ``fatqat.Result``.

        Raises:
            TypeError: If program, shots, or layout has an unsupported type.
            ValueError: If shots, layout, topology, or program operations
                violate the QEC17 constraints. Invalid programs are not submitted.

        SDK submission errors propagate to the caller; later execution errors
        are reported through the returned SDK job.
        """
        if type(shots) is not int:
            raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
        if not 1 <= shots <= 50_000:
            raise ValueError(f"shots must be in the range 1..50000, got {shots}")

        refs, n_qubits = _flatten_quantum_refs(program)
        layout, sdk_layout = _normalize_initial_layout(n_qubits, initial_layout)
        _validate_mapped_cz_edges(program, refs, layout)
        circuit = program_to_qec17_circuit(program)

        if sdk_layout is not None:
            run_options["initial_layout"] = sdk_layout
        return self._cloud_backend.run(circuit, shots=shots, **run_options)


__all__ = ["LQCloudQEC17Backend"]
