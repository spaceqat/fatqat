"""Shared private traversal and replacement for program parameters."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from numbers import Real
from typing import Any, Callable, TypeVar

import numpy as np

from .errors import BackendValidationError
from .parameters import Parameter, ParameterVector

ValueT = TypeVar("ValueT")


def _discover_parameters(instructions: Sequence[Any]) -> tuple[Parameter, ...]:
    """Find direct operation-field parameters in first-appearance order.

    Discovery intentionally inspects only top-level fields of dataclass
    operations. It de-duplicates by parameter identity and is the common
    ordering source for binding, sweep rows, and unbound diagnostics.
    """
    discovered: list[Parameter] = []
    seen: set[Parameter] = set()
    for instruction in instructions:
        operation = getattr(instruction, "operation", None)
        if operation is None or not is_dataclass(operation):
            continue
        for field_info in fields(operation):
            value = getattr(operation, field_info.name)
            if isinstance(value, Parameter) and value not in seen:
                seen.add(value)
                discovered.append(value)
    return tuple(discovered)


def _validate_parameter_scalar(value: object) -> Real:
    """Enforce the shared scalar policy without coercing the accepted value.

    Keeping built-in ``int``/``float`` and NumPy integer/floating scalars
    unchanged avoids introducing a second conversion policy between
    single-point and batch binding.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("parameter values must be int, float, or NumPy real scalars")
    return value


def _materialize_vector_value(value: object) -> list[object]:
    """Materialize one vector assignment and require a one-dimensional shape.

    Iterables are consumed exactly once. Element scalar validation remains the
    caller's responsibility so this helper owns only container shape.
    """
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("parameter vector values must be one-dimensional sequences")
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("parameter vector values must be one-dimensional")
        return list(value)
    try:
        materialized = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "parameter vector values must be one-dimensional sequences"
        ) from exc
    try:
        array = np.asarray(materialized, dtype=object)
    except ValueError as exc:
        raise ValueError(
            "parameter vector values must form a rectangular container"
        ) from exc
    if array.ndim != 1:
        raise ValueError("parameter vector values must be one-dimensional")
    return materialized


def _expand_parameter_bindings(
    instructions: Sequence[Any],
    values: Mapping[Parameter | ParameterVector, object],
    *,
    normalize_parameter: Callable[[object], ValueT],
    split_vector: Callable[[ParameterVector, object], Sequence[ValueT]],
) -> tuple[tuple[Parameter, ...], dict[Parameter, ValueT]]:
    """Own key validation and vector expansion for every binding path.

    The callbacks provide path-specific scalar or column normalization. This
    function alone checks key types, program membership, and duplicate scalar
    assignments, preventing the public and sweep paths from drifting.
    """
    if not isinstance(values, Mapping):
        raise TypeError("parameter values must be a mapping")

    discovered = _discover_parameters(instructions)
    present = set(discovered)
    expanded: dict[Parameter, ValueT] = {}

    def add(parameter: Parameter, normalized_value: ValueT) -> None:
        if parameter in expanded:
            raise ValueError(f"parameter {parameter.name!r} is assigned more than once")
        expanded[parameter] = normalized_value

    for key, raw_value in values.items():
        if isinstance(key, Parameter):
            if key not in present:
                raise ValueError(
                    f"parameter {key.name!r} is not present in the program"
                )
            add(key, normalize_parameter(raw_value))
            continue
        if isinstance(key, ParameterVector):
            parameters = tuple(key)
            if not parameters:
                raise ValueError("a zero-length parameter vector cannot be bound")
            if any(parameter not in present for parameter in parameters):
                raise ValueError(
                    f"parameter vector {key.name!r} is not fully present in the program"
                )
            vector_values = split_vector(key, raw_value)
            if len(vector_values) != len(parameters):
                raise ValueError(
                    f"parameter vector {key.name!r} expects {len(parameters)} values, "
                    f"got {len(vector_values)}"
                )
            for parameter, element in zip(parameters, vector_values, strict=True):
                add(parameter, element)
            continue
        raise TypeError("parameter mapping keys must be Parameter or ParameterVector")

    return discovered, expanded


def _normalize_parameter_mapping(
    instructions: Sequence[Any],
    values: Mapping[Parameter | ParameterVector, object],
) -> dict[Parameter, Real]:
    """Normalize one partial public mapping into stable scalar-key order.

    Missing parameters are valid here because ``Program.assign_parameters``
    supports partial binding. All supplied keys and values are fully checked.
    """

    def normalize_vector(_vector: ParameterVector, raw_value: object) -> Sequence[Real]:
        return tuple(
            _validate_parameter_scalar(value)
            for value in _materialize_vector_value(raw_value)
        )

    discovered, expanded = _expand_parameter_bindings(
        instructions,
        values,
        normalize_parameter=_validate_parameter_scalar,
        split_vector=normalize_vector,
    )

    return {
        parameter: expanded[parameter]
        for parameter in discovered
        if parameter in expanded
    }


def _materialize_batch_array(value: object, *, expected_ndim: int) -> np.ndarray:
    """Materialize one batch value and enforce its rank and positive length.

    NumPy integer and floating arrays are kept as they are; anything else
    becomes an object array so the shared scalar validator can check each
    element. Width and cross-column length checks belong to the batch
    normalizer, which has the necessary parameter context.
    """
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("parameter batch values must be array-like containers")
    if isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
        array = value
    elif isinstance(value, np.ndarray) or np.isscalar(value):
        array = np.asarray(value, dtype=object)
    else:
        try:
            materialized = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "parameter batch values must be array-like containers"
            ) from exc
        try:
            array = np.asarray(materialized, dtype=object)
        except ValueError as exc:
            raise ValueError(
                "parameter batch values must form a rectangular container"
            ) from exc
    if array.ndim != expected_ndim:
        raise ValueError(
            f"parameter batch value must have rank {expected_ndim}, got {array.ndim}"
        )
    if array.shape[0] == 0:
        raise ValueError("parameter batch length must be greater than zero")
    return array


def _normalize_parameter_batch(
    instructions: Sequence[Any],
    bindings: Mapping[Parameter | ParameterVector, object],
) -> dict[Parameter, np.ndarray]:
    """Validate a complete binding batch and return one column per parameter.

    This is the sole batch-validation boundary. The result covers every
    discovered parameter in discovery order; each column is a read-only
    one-dimensional array of the batch length. `_parameter_batch_rows` turns
    it into trusted rows for ``_assign_normalized_parameters``.
    """

    def normalize_parameter(value: object) -> np.ndarray:
        return _materialize_batch_array(value, expected_ndim=1)

    def split_vector(vector: ParameterVector, value: object) -> Sequence[np.ndarray]:
        array = _materialize_batch_array(value, expected_ndim=2)
        if array.shape[1] != len(vector):
            raise ValueError(
                f"parameter vector {vector.name!r} batch expects width "
                f"{len(vector)}, got {array.shape[1]}"
            )
        return tuple(array[:, index] for index in range(array.shape[1]))

    discovered, columns = _expand_parameter_bindings(
        instructions,
        bindings,
        normalize_parameter=normalize_parameter,
        split_vector=split_vector,
    )
    if not discovered:
        raise ValueError("parameter sweep requires a parameterized program")
    if not columns:
        raise ValueError("parameter sweep requires at least one assignment")

    missing = [parameter.name for parameter in discovered if parameter not in columns]
    if missing:
        raise ValueError(
            "parameter sweep is missing assignments: " + ", ".join(missing)
        )

    lengths = {len(column) for column in columns.values()}
    if len(lengths) != 1:
        raise ValueError("parameter batch values must share one leading length")

    return {
        parameter: _validate_batch_column(columns[parameter])
        for parameter in discovered
    }


def _validate_batch_column(column: np.ndarray) -> np.ndarray:
    """Return a read-only copy of one column that passed the scalar policy.

    NumPy integer and floating columns satisfy the policy by dtype. Other
    columns are checked element by element and converted once, so a column of
    Python floats becomes float64.
    """
    if column.dtype.kind in "iuf":
        validated = np.array(column)
    else:
        validated = np.array([_validate_parameter_scalar(value) for value in column])
    validated.flags.writeable = False
    return validated


def _parameter_batch_rows(
    columns: Mapping[Parameter, np.ndarray],
) -> tuple[dict[Parameter, Real], ...]:
    """Split validated batch columns into one parameter mapping per row."""
    length = len(next(iter(columns.values())))
    return tuple(
        {parameter: column[index] for parameter, column in columns.items()}
        for index in range(length)
    )


def _replace_parameterized_instructions(
    instructions: Sequence[Any],
    normalized_values: Mapping[Parameter, object],
) -> tuple[Any, ...]:
    """Copy instructions while replacing fields from a trusted scalar mapping.

    This structural helper deliberately performs no binding validation. It
    preserves instruction targets and conditions and creates a new operation
    only when at least one direct field is replaced.
    """
    replaced_instructions: list[Any] = []
    for instruction in instructions:
        operation = getattr(instruction, "operation", None)
        if operation is None or not is_dataclass(operation):
            replaced_instructions.append(instruction)
            continue
        replacements = {
            field_info.name: normalized_values[value]
            for field_info in fields(operation)
            if isinstance((value := getattr(operation, field_info.name)), Parameter)
            and value in normalized_values
        }
        if not replacements:
            replaced_instructions.append(instruction)
            continue
        new_operation = replace(operation, **replacements)
        replaced_instructions.append(replace(instruction, operation=new_operation))
    return tuple(replaced_instructions)


def _format_unbound_parameters(parameters: Sequence[Parameter]) -> str:
    """Format diagnostic labels and disambiguate same-named identities.

    Ordinal suffixes are local to the message and follow discovery order; they
    are never accepted as binding keys.
    """
    counts = Counter(parameter.name for parameter in parameters)
    ordinals: defaultdict[str, int] = defaultdict(int)
    labels: list[str] = []
    for parameter in parameters:
        name = parameter.name
        if counts[name] == 1:
            labels.append(name)
            continue
        ordinals[name] += 1
        labels.append(f"{name}#{ordinals[name]}")
    return ", ".join(labels)


def _raise_for_unbound_parameters(instructions: Sequence[Any]) -> None:
    """Reject parameters before a backend realizes numeric operations.

    Simulator, Estimator, pulse, and QASM entry points share this guard so an
    unbound placeholder never leaks into numeric realization or export code.
    """
    parameters = _discover_parameters(instructions)
    if parameters:
        raise BackendValidationError(
            "program has unbound parameters: " + _format_unbound_parameters(parameters)
        )
