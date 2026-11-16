"""Process-wide defaults for matrix simulator construction."""

from ..errors import BackendValidationError

_default_runtime = "numba"


def _validate_runtime(runtime: str) -> str:
    normalized = str(runtime).lower()
    if normalized not in ("numpy", "numba"):
        raise BackendValidationError(
            f"unsupported runtime={runtime!r}; expected 'numpy' or 'numba'"
        )
    return normalized


def get_default_runtime() -> str:
    """Return the runtime used by new simulators when none is specified.

    Returns:
        The canonical name, ``"numpy"`` or ``"numba"``.
    """
    return _default_runtime


def set_default_runtime(runtime: str) -> None:
    """Set the default runtime for new simulators.

    Simulators use this default when ``runtime`` is omitted or ``None``.

    Args:
        runtime: ``"numpy"`` or ``"numba"``, case-insensitive. Set ``"numba"``
            to restore the initial default.

    Raises:
        BackendValidationError: If ``runtime`` is invalid. The previous
            setting is preserved.
    """
    global _default_runtime  # pylint: disable=global-statement
    _default_runtime = _validate_runtime(runtime)
