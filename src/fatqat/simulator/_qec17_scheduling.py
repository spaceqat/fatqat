"""Private tick scheduling for QEC17 gate-level simulation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral

from .. import operations as ops
from .._backends.view_normalization import ProgramInstruction
from ..errors import BackendValidationError
from ..operations import Measurement, Operation
from ..program import _AppliedOperation
from ..registers import RegisterRef

_QEC17_TICK_DURATION_SECONDS = 20e-9


def _schedule_qec17_asap(
    instructions: Iterable[ProgramInstruction],
    all_qubits: Iterable[RegisterRef],
    operation_duration_ticks: Callable[[Operation], int | None],
) -> tuple[ProgramInstruction, ...]:
    """Insert simulator-only idle ticks for a scalar QEC17 instruction stream.

    The scheduler tracks one integer availability time per qubit. Operations
    sharing a qubit retain source order, while disjoint operations may occupy
    the same ticks. A multi-qubit operation starts when its latest operand is
    available; earlier operands receive one internal ``I`` instruction per
    20 ns waiting tick. Barriers similarly synchronize only their targets and
    are not emitted. A final synchronization covers every supplied qubit.

    Measurements and resets form conservative full-width scheduling
    boundaries. If any executable operation is conditional, or if the duration
    callback does not recognize an operation, scheduling raises
    BackendValidationError: a static ASAP rewrite cannot in general know which
    branch or extension operation advances time. Barrier conditions retain the generic
    simulator contract and are ignored.

    ``instructions`` must already be scalar-expanded. The returned tuple may
    contain new `_AppliedOperation` records, but neither the input records nor
    their owning `Program` are mutated.
    """
    source = tuple(instructions)
    if any(
        isinstance(step, _AppliedOperation)
        and not isinstance(step.operation, type(ops.Barrier))
        and step.condition is not None
        for step in source
    ):
        raise BackendValidationError(
            "QEC17 idle scheduling does not support executable classical conditions"
        )

    qubits = tuple(all_qubits)
    if len(set(qubits)) != len(qubits):
        raise ValueError("QEC17 scheduling qubits must be distinct")

    durations: dict[int, int] = {}
    for position, step in enumerate(source):
        if not isinstance(step, _AppliedOperation) or isinstance(
            step.operation, (type(ops.Barrier), type(ops.Reset))
        ):
            continue
        duration = operation_duration_ticks(step.operation)
        if duration is None:
            raise BackendValidationError(
                f"QEC17 idle scheduling has no duration for {step.operation.name}"
            )
        if isinstance(duration, bool) or not isinstance(duration, Integral):
            raise TypeError(
                "QEC17 operation duration must be a non-negative integer tick count"
            )
        duration = int(duration)
        if duration < 0:
            raise ValueError(
                "QEC17 operation duration must be a non-negative integer tick count"
            )
        durations[position] = duration

    available_at = {qubit: 0 for qubit in qubits}
    scheduled: list[ProgramInstruction] = []

    def scalar_targets(
        step: _AppliedOperation | Measurement,
    ) -> tuple[RegisterRef, ...]:
        targets: list[RegisterRef] = []
        for target in step.targets:
            if not isinstance(target, RegisterRef):
                raise TypeError(
                    "QEC17 scheduling requires scalar-expanded instruction targets"
                )
            if target not in available_at:
                raise ValueError(
                    "QEC17 instruction target is absent from the supplied qubits"
                )
            targets.append(target)
        return tuple(targets)

    def pad_to(targets: tuple[RegisterRef, ...], tick: int) -> None:
        for target in targets:
            for _ in range(tick - available_at[target]):
                scheduled.append(_AppliedOperation(operation=ops.I, targets=(target,)))
            available_at[target] = tick

    def synchronize(targets: tuple[RegisterRef, ...]) -> None:
        if targets:
            pad_to(targets, max(available_at[target] for target in targets))

    for position, step in enumerate(source):
        if isinstance(step, Measurement):
            scalar_targets(step)
            synchronize(qubits)
            scheduled.append(step)
            continue

        if not isinstance(step, _AppliedOperation):  # pragma: no cover - type boundary
            raise TypeError(f"unsupported QEC17 instruction {step!r}")

        targets = scalar_targets(step)
        if isinstance(step.operation, type(ops.Barrier)):
            synchronize(targets)
            continue
        if isinstance(step.operation, type(ops.Reset)):
            synchronize(qubits)
            scheduled.append(step)
            continue

        duration = durations[position]

        start = max((available_at[target] for target in targets), default=0)
        pad_to(targets, start)
        scheduled.append(step)
        for target in targets:
            available_at[target] = start + duration

    synchronize(qubits)
    return tuple(scheduled)


__all__ = ["_QEC17_TICK_DURATION_SECONDS", "_schedule_qec17_asap"]
