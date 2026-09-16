"""Prepare and run a four-qubit GHZ state on LQCloud QEC17.

Run this file directly from an IDE, or from PowerShell with::

    $env:LQCLOUD_API_KEY = "your real API key"
    python tests/lqcloud/run_qec17_four_qubit_ghz.py

The FatQAT LQCloud adapter appends the terminal global barrier and
``measure_all()``. Initial reset is owned by the LQCloud platform, so this
program deliberately contains neither ``Reset`` nor explicit measurement.
"""

from __future__ import annotations

import os

import fatqat as fq
import fatqat.operations as ops
from fatqat.lqcloud import LQCloudQEC17Backend

LQCLOUD_URL = "https://cloud.logicalqubit.com"
SHOTS = 1024
TIMEOUT_SECONDS = 600

# logical q0, q1, q2, q3 -> physical Q6, Q11, Q4, Q12. Consecutive
# physical qubits form a connected QEC17 path for the three CZ operations.
INITIAL_LAYOUT = [6, 11, 4, 12]


def build_four_qubit_ghz_program() -> fq.Program:
    """Return a native-gate FATQAT program preparing (|0000> + |1111>)/sqrt(2)."""

    program = fq.Program(4)
    program.add(ops.H, 0)

    # Native decomposition of CX(control, target):
    # MYHalf(target) -> CZ(control, target) -> YHalf(target).
    for control, target in ((0, 1), (1, 2), (2, 3)):
        program.add(ops.MYHalf, target)
        program.add(ops.CZ, (control, target))
        program.add(ops.YHalf, target)

    return program


def main() -> None:
    """Submit the GHZ circuit, wait for completion, and print raw counts."""

    api_key = os.environ.get("LQCLOUD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Set the LQCLOUD_API_KEY environment variable before submission."
        )

    try:
        from lqcloud.exceptions import JobTimeoutError
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install the optional SDK first: python -m pip install lqcloud==0.4.2"
        ) from error

    program = build_four_qubit_ghz_program()
    backend = LQCloudQEC17Backend(
        api_key=api_key,
        url=LQCLOUD_URL,
    )

    print(
        f"Submitting 4-qubit GHZ to {backend.name}: "
        f"logical [0, 1, 2, 3] -> physical {INITIAL_LAYOUT}"
    )
    job = backend.run(
        program,
        shots=SHOTS,
        initial_layout=INITIAL_LAYOUT,
    )
    print(f"Job ID: {job.job_id}")

    try:
        result = job.result(timeout=TIMEOUT_SECONDS)
    except JobTimeoutError:
        job.cancel()
        raise

    print("Counts:")
    print(result.get_counts())
    print("An ideal GHZ result is concentrated on 0000 and 1111.")


if __name__ == "__main__":
    main()
