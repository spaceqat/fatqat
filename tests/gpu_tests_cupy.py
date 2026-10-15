"""Smoke tests that require a working CUDA GPU."""

import cupy as cp


def test_cupy_sum_of_squares() -> None:
    """Compute a sum of squares on the GPU and verify the result."""
    assert cp.cuda.runtime.getDeviceCount() > 0, "No CUDA GPU detected"

    with cp.cuda.Device(0):
        x = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)
        result_gpu = cp.sum(x * x)

        cp.cuda.get_current_stream().synchronize()
        result = float(result_gpu.get())

    assert result == 14.0
