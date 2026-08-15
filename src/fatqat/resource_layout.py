"""ResourceLayout: the public mapping from program resources to device labels.

`ResourceLayout` maps a program's scalar `RegisterRef`s to opaque, backend-
defined device resource labels (a physical site, coordinate, or any other
hashable backend identity). It carries no dimensions, no classical-slot
positions, and no engine subsystem indices - those belong to the private
engine allocation in `fatqat._index_allocation`. See
docs/superpowers/specs/2026-07-22-fatqat-resource-layout-and-noise-selector-design.md.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from .registers import RegisterRef

type DeviceOperand = Hashable


class ResourceLayout:
    """Maps scalar `RegisterRef`s to opaque device resource labels.

    Refs key this mapping by identity, not by field values: a register is an
    entity, so a lookalike built with the same size and name is a different
    register and is deliberately not found here.
    """

    def __init__(self, labels: Mapping[RegisterRef, DeviceOperand]) -> None:
        """Create a resource layout from explicit ref-to-label pairs.

        Args:
            labels: Mapping from each covered `RegisterRef` to its device
                resource label.
        """
        self._labels: dict[RegisterRef, DeviceOperand] = dict(labels)
        self._refs_by_label: dict[DeviceOperand, RegisterRef | None] = {}
        for ref, label in self._labels.items():
            self._refs_by_label[label] = None if label in self._refs_by_label else ref

    def device_label(self, ref: RegisterRef) -> DeviceOperand:
        """Return the device resource label mapped to ``ref``.

        Raises:
            KeyError: If ``ref`` is not part of this layout.
        """
        try:
            return self._labels[ref]
        except KeyError:
            raise KeyError("ref not part of this resource layout") from None

    @property
    def device_labels(self) -> frozenset[DeviceOperand]:
        """Return the set of every device resource label in this layout."""
        return frozenset(self._labels.values())

    @property
    def refs(self) -> frozenset[RegisterRef]:
        """Return the set of every RegisterRef in this layout."""
        return frozenset(self._labels)

    def device_labels_for(
        self, refs: tuple[RegisterRef, ...]
    ) -> tuple[DeviceOperand, ...]:
        """Return the device resource labels for ``refs``, in operand order.

        Raises:
            KeyError: If any ref in ``refs`` is not part of this layout.
        """
        return tuple(self.device_label(ref) for ref in refs)

    def _ref_for_label(self, label: DeviceOperand) -> RegisterRef:
        """Return the unique program ref for a backend-owned device label.

        Raises:
            KeyError: If the label is absent or maps from multiple refs. Public
                layouts need not be injective; family backends using this
                private reverse seam must provide an injective layout.
        """
        ref = self._refs_by_label.get(label)
        if ref is None:
            raise KeyError("device label must identify exactly one program ref")
        return ref
