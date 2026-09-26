"""LogicalQubit Cloud devices and asynchronous job handles."""

from ..compiler.dialects.sc_native import NativeMeasure
from ..compiler.targets import _SCTarget
from ..result import Result
from .converter import LQCloudCompilationResult, _load_sdk


class LQCloudBackend:
    """Compile for and submit to one LogicalQubit Cloud device.

    Construction discovers the selected device and snapshots its coupling
    graph. Use with ``fatqat.compiler.compile_to_sc`` or
    ``compile_qasm_to_sc``, then pass the result to ``run``. Requires Python
    3.12 and the optional ``fatqat[lqcloud]`` dependency. No task is submitted
    during construction or compilation.

    Args:
        name: Exact backend name: ``AGate-100``, ``QZ01-surface_code`` or
            ``MQ02``. No default device is selected.
        api_key: Optional authentication key. When omitted, the SDK reads
            ``LQCLOUD_API_KEY`` or its saved account configuration. FatQat
            does not persist credentials or enable SDK task storage.

    Raises:
        ValueError: If the name or returned device configuration is invalid.
        ImportError: If the optional SDK is unavailable.
        lqcloud.LQCloudError: If authentication or discovery fails.
    """

    def __init__(self, name: str, *, api_key: str | None = None) -> None:
        if name not in ("AGate-100", "QZ01-surface_code", "MQ02"):
            raise ValueError("name must be AGate-100, QZ01-surface_code or MQ02")
        sdk = _load_sdk()
        provider = sdk.LQCloudProvider(api_key=api_key, interactive=False, store=False)
        self._backend = provider.get_backend(name)
        config = self._backend.config
        count = config["qubits"]
        if type(count) is not int or count < 1:
            raise ValueError("device qubits must be a positive integer")
        couplings = set()
        for edge in config["topology"]["coupling_map"]:
            if (
                len(edge) != 2
                or any(type(q) is not int or not 0 <= q < count for q in edge)
                or edge[0] == edge[1]
            ):
                raise ValueError("invalid device coupling_map")
            couplings.add(tuple(sorted(edge)))
        gates = config.get("native_gates")
        if gates is not None and not {"h", "rz", "cz"} <= set(gates):
            raise ValueError("device must support H, RZ and CZ")
        self._target = _SCTarget(
            name, tuple(range(count)), frozenset(couplings), "lqcloud"
        )

    @property
    def name(self) -> str:
        """Selected cloud backend name."""
        return self._target.name

    @property
    def target(self) -> _SCTarget:
        """Read-only topology snapshot used by the SC compiler."""
        return self._target

    def run(
        self, compiled: LQCloudCompilationResult, *, shots: int = 1024
    ) -> "LQCloudJob":
        """Submit a prepared compilation without re-routing or decomposing it.

        Args:
            compiled: Final compilation for the same device/configuration.
                At least one terminal measurement is required. Earlier emit
                results and uncompiled programs are not executable here.
            shots: Number of repetitions, 1 through 50,000; default 1024.

        Returns:
            An asynchronous LQCloudJob; execution may incur platform fees.

        Raises:
            TypeError: If compiled is not a cloud compilation or shots is not
                an integer.
            ValueError: If shots, target or measurements are invalid.
            lqcloud.LQCloudError: If SDK validation or submission fails.
        """
        if not isinstance(compiled, LQCloudCompilationResult):
            raise TypeError("run expects an LQCloudCompilationResult")
        if type(shots) is not int:
            raise TypeError("shots must be an integer")
        if not 1 <= shots <= 50_000:
            raise ValueError("shots must be in 1..50000")
        if compiled.target != self.target:
            raise ValueError("compilation target does not match this backend")
        if not any(isinstance(op, NativeMeasure) for op in compiled.output.operations):
            raise ValueError("cloud submission requires at least one measurement")
        job = self._backend.run(
            compiled.program,
            shots=shots,
            initial_layout=list(compiled.physical_layout),
            readout_correction=False,
            dynamic_decoupling=False,
        )
        return LQCloudJob(job, self.name, compiled.classical_dims, shots)


class LQCloudJob:
    """Asynchronous cloud job with FatQat count results.

    Obtain this handle from LQCloudBackend.run(). SDK exceptions propagate;
    a result timeout neither cancels nor resubmits the task. Reuse the handle
    to query it again. Cloud states are not the completed-only fatqat.Job
    states. No simulator-only statevector is produced.
    """

    def __init__(
        self, job, backend_name: str, classical_dims: tuple[int, ...], shots: int
    ) -> None:
        self._job = job
        self._backend_name = backend_name
        self._classical_dims = classical_dims
        self._shots = shots

    @property
    def job_id(self) -> str:
        """Remote task identifier, available immediately after submission."""
        return self._job.job_id

    def status(self) -> str:
        """Query PENDING, QUEUED, RUNNING, COMPLETED, FAILED or CANCELLED."""
        return self._job.status().name

    def cancel(self) -> bool:
        """Request cancellation; return whether the SDK reports success."""
        return self._job.cancel()

    def result(self, timeout: int | None = None) -> Result:
        """Wait for counts, preserving source classical slot order.

        Count validation applies after SDK normalization.

        Args:
            timeout: Positive integer maximum wait in seconds, or None for
                unlimited waiting.

        Returns:
            FatQat Result with SDK-normalized counts and job_id/backend metadata.

        Raises:
            lqcloud.JobTimeoutError: If the wait expires; the job remains live.
            lqcloud.JobError: If execution fails or is cancelled.
            ValueError: If the response cannot represent valid shot counts.
        """
        if timeout is not None and (type(timeout) is not int or timeout <= 0):
            raise ValueError("timeout must be a positive integer or None")
        result = self._job.result(timeout=timeout, verbose=False)
        if (
            not isinstance(result, _load_sdk().Result)
            or not result.success
            or result.data.get("success") is False
        ):
            raise ValueError("cloud job did not return successful counts")
        counts = result.get_counts()
        width = len(self._classical_dims)
        if (
            not isinstance(counts, dict)
            or any(
                not isinstance(key, str)
                or len(key) != width
                or set(key) - {"0", "1"}
                or type(count) is not int
                or count < 0
                for key, count in counts.items()
            )
            or sum(counts.values()) != self._shots
        ):
            raise ValueError("cloud counts have invalid width, bits or shot totals")
        return Result(
            counts={
                tuple(int(bit) for bit in key): count for key, count in counts.items()
            },
            available=frozenset({"counts"}),
            classical_dims=self._classical_dims,
            metadata={"job_id": self.job_id, "backend": self._backend_name},
        )
