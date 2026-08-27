"""Public mapping from program quantum references to backend device labels."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from .registers import RegisterRef

type DeviceOperand = Hashable


class ResourceLayout:
    """Read-only program-reference to device-label lookup object.

    Device labels are opaque hashable values defined by a backend, such as a
    site number, coordinate, or string identifier. They are not simulator
    tensor-axis indices. Public lookups map register refs to device labels.

    Construction shallow-copies the input mapping, so later top-level changes
    to that mapping are not observed. The ref keys and label objects are
    retained. Use immutable labels whose equality and hashes remain stable for
    the layout's lifetime. Backend calls do not mutate a layout, so it can be
    reused with the same register objects and a compatible backend.

    The selected backend validates program coverage, label uniqueness and
    type, subsystem dimensions, placement, and connectivity when the layout is
    used.

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
            labels: Mapping from quantum :class:`~fatqat.RegisterRef` objects
                to opaque, hashable device labels defined by the backend.
                FATQAT defines no universal label type or reserved value. The
                mapping is shallow-copied.

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
            ref: Stored mapping key to look up. Under the supported contract,
                this is a scalar quantum register reference whose register
                identity matches the original key.

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
        """Return every mapped :class:`~fatqat.RegisterRef` as an immutable set."""
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
