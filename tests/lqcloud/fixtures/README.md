# Device topology snapshots

`devices.json` contains public configuration fields obtained on 2026-09-24
from https://cloud.logicalqubit.com/api/v1/qpus. Only backend names, capacities,
coupling maps, native-gate declarations and readout capabilities are retained.
No credentials, connection configuration or calibration records are included.

Tests use these snapshots offline with the real LQCloud 0.5.0 serializer.
They establish compilation and submission-protocol behavior, not live QPU
execution or device fidelity. Production reads current configuration instead
of importing these fixtures.
