"""Numba thread-mask lifecycle and execution worker limits.

Engines enter the scope around an execution. Numerical planning reads the
resolved worker limit; compiled workflows and arithmetic do not change the mask.
"""

from contextlib import contextmanager

from numba import get_num_threads, set_num_threads

from ._execution_policy import _ExecutionPolicy as ExecutionPolicy


@contextmanager
def _thread_scope(policy: ExecutionPolicy):
    """Apply one resolved Numba thread ceiling and restore the caller's mask."""
    previous = get_num_threads()
    target = _execution_thread_workers(policy)
    if target != previous:
        set_num_threads(target)
    try:
        yield
    finally:
        if target != previous:
            set_num_threads(previous)


def _execution_thread_workers(policy: ExecutionPolicy) -> int:
    """Resolve the Numba mask owned by the selected execution axis."""
    uses_threads = policy.shot_strategy == "threads" or policy.kernel_strategy in {
        "adaptive",
        "threads",
    }
    if not uses_threads:
        return 1
    if policy.worker_limit is None:
        return get_num_threads()
    return policy.worker_limit


def _kernel_thread_workers(policy: ExecutionPolicy) -> int:
    """Resolve only the capacity available to one evolution's kernels."""
    if policy.kernel_strategy == "serial":
        return 1
    if policy.worker_limit is None:
        return get_num_threads()
    return policy.worker_limit
