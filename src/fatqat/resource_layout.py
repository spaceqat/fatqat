"""Placement labels for program quantum references."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from .registers import RegisterRef

type DeviceOperand = Hashable


class ResourceLayout:
    """Associate scalar quantum refs with backend device labels.

    Labels are opaque hashable values such as site numbers, coordinates, or
    strings. They identify device resources rather than simulator tensor axes.

    Most programs can use their backend's default layout. Supply an explicit
    layout when you need a particular supported placement. Each backend defines
    the labels it accepts and checks program coverage, uniqueness, dimensions,
    placement, and connectivity when the program runs.

    Examples:
        >>> import fatqat as fq
        >>> qubits = fq.QuantumRegister(2, name="q")
        >>> layout = fq.ResourceLayout({qubits[0]: "left", qubits[1]: "right"})
        >>> layout.device_labels_for((qubits[1], qubits[0]))
        ('right', 'left')
    """

    def __init__(self, labels: Mapping[RegisterRef, DeviceOperand]) -> None:
        """Create a resource layout from explicit ref-to-label pairs.

        Args:
            labels: Mapping from quantum `RegisterRef` objects to opaque,
                hashable labels defined by the backend. FATQAT defines no
                universal label type or reserved value. The mapping is
                shallow-copied.

        Raises:
            TypeError: If ``labels`` cannot be copied into a dictionary or a
                device label is not hashable.
        """
        self._labels: dict[RegisterRef, DeviceOperand] = dict(labels)
        self._refs_by_label: dict[DeviceOperand, RegisterRef | None] = {}
        for ref, label in self._labels.items():
            self._refs_by_label[label] = None if label in self._refs_by_label else ref

    def device_label(self, ref: RegisterRef) -> DeviceOperand:
        """Return the device resource label mapped to ``ref``.

        Args:
            ref: Scalar quantum register ref supplied as a key when the layout
                was constructed.

        Returns:
            The mapped backend-defined device label.

        Raises:
            KeyError: If ``ref`` is not part of this layout.
        """
        try:
            return self._labels[ref]
        except KeyError:
            raise KeyError("ref not part of this resource layout") from None

    @property
    def device_labels(self) -> frozenset[DeviceOperand]:
        """Return the device labels as an immutable set."""
        return frozenset(self._labels.values())

    @property
    def refs(self) -> frozenset[RegisterRef]:
        """Return every mapped `RegisterRef` as an immutable set."""
        return frozenset(self._labels)

    def device_labels_for(
        self, refs: tuple[RegisterRef, ...]
    ) -> tuple[DeviceOperand, ...]:
        """Return the device resource labels for ``refs``, in operand order.

        Args:
            refs: Tuple of scalar quantum refs.

        Returns:
            Device labels in the same order.

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
