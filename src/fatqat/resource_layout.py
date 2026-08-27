"""Public mapping from program quantum references to backend device labels."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from .registers import RegisterRef

type DeviceOperand = Hashable


class ResourceLayout:
    """Read-only program-reference to device-label lookup object.

    Device labels are opaque hashable values defined by a backend, such as a
    site number, coordinate, or string identifier. They are not simulator
    tensor-axis indices. The mapping is intentionally one-way: public methods
    look up labels from refs but do not reverse-map a label to a ref.

    Construction shallow-copies the input mapping. Later additions or
    replacements in that mapping are not observed; the ``RegisterRef`` keys
    and label objects themselves are retained. Label equality and hashes must
    remain stable for the layout's lifetime, so immutable labels are strongly
    preferred. Refs distinguish their owning registers by identity, so a ref
    from a separately constructed lookalike register is not present. The
    layout object itself also compares by identity and does not implement the
    general ``Mapping`` interface. Backend calls do not mutate it, so it may be
    reused with the same register objects and a compatible backend.

    A standalone layout may be partial, include foreign or non-quantum keys, or
    assign one label to more than one ref because key types and ownership are
    not checked by the constructor. Current backends require complete coverage
    and distinct exclusive labels. Other label, program-dimension, placement,
    and connectivity checks are backend-specific and can occur during binding,
    preparation, or lowering.

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
            labels: Mapping from each covered quantum ``RegisterRef`` to an
                opaque hashable device label. This is the supported typed
                contract; the constructor copies with ``dict()`` and does not
                enforce the input container or key type. FATQAT defines no
                fixed label type or reserved values.

        Raises:
            TypeError: If ``labels`` cannot be copied into a dictionary or a
                device label is not hashable.
            ValueError: If an iterable cannot be converted to key/value pairs.
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
        """Return the unique device labels as an immutable set.

        Repeated values collapse to one set member. Such repeats are accepted
        by this value object but rejected by current backends when the layout
        is used for a run.
        """
        return frozenset(self._labels.values())

    @property
    def refs(self) -> frozenset[RegisterRef]:
        """Return every mapped ``RegisterRef`` as an immutable set."""
        return frozenset(self._labels)

    def device_labels_for(
        self, refs: tuple[RegisterRef, ...]
    ) -> tuple[DeviceOperand, ...]:
        """Return the device resource labels for ``refs``, in operand order.

        Args:
            refs: Tuple of scalar quantum refs. Runtime lookup iterates any
                supplied iterable; an empty iterable is accepted.

        Returns:
            Device labels in the same order, including repeated input refs.

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
