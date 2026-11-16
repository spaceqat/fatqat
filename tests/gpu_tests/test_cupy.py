"""CuPy smoke tests; skip this module when CuPy or CUDA is unavailable."""

import pytest

try:
    import cupy as cp
except ImportError as exc:
    pytest.skip(f"CuPy is unavailable: {exc}", allow_module_level=True)

# Only availability checks may skip. Computation failures below must fail.
try:
    _device_count = cp.cuda.runtime.getDeviceCount()
except (cp.cuda.runtime.CUDARuntimeError, OSError) as exc:
    pytest.skip(f"CUDA is unavailable: {exc}", allow_module_level=True)

if _device_count < 1:
    pytest.skip("No CUDA GPU detected", allow_module_level=True)


def test_cupy_sum_of_squares() -> None:
    """Compute a sum of squares on the GPU and verify the result."""
    assert cp.cuda.runtime.getDeviceCount() > 0, "No CUDA GPU detected"

    with cp.cuda.Device(0):
        x = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)
        result_gpu = cp.sum(x * x)

        cp.cuda.get_current_stream().synchronize()
        result = float(result_gpu.get())

    assert result == 14.0
