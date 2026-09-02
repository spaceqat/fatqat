"""Deterministic dense-atom scheduling used by the internal ZAP router.

Derived from the bundled ZAP scheduler under the MIT License; see LICENSE in
this package.
"""

from __future__ import annotations


class Scheduler:
    """Build per-stage gate groups from a flat gate list ``g_q``."""

    def __init__(
        self,
        g_q: list[tuple[int, int]],
        results_code: dict[str, object],
        operation_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Create a scheduler that mutates ``results_code`` with stage data."""
        self.g_q = g_q
        if operation_ids is None:
            operation_ids = tuple(f"input.{index}" for index in range(len(g_q)))
        if len(operation_ids) != len(g_q):
            raise ValueError("operation_ids must have the same length as g_q")
        self.operation_ids = operation_ids
        self.results_code = results_code
        self.results_code["stages"] = {
            "num_stage": 0,
            "stage": {qubit: {} for qubit in range(self.results_code["n_q"])},
            "qs_status": {},
        }
        self.list_scheduling: list[list[int]] = []

    def asap_joint(self) -> None:
        """Schedule the full gate stream as soon as both atoms are free."""
        list_qubit_stage = [0 for _ in range(self.results_code["n_q"])]
        for index, gate in enumerate(self.g_q):
            stage = max(list_qubit_stage[gate[0]], list_qubit_stage[gate[1]])
            if stage >= len(self.list_scheduling):
                self.list_scheduling.append([])
            self.list_scheduling[stage].append(index)

            stage += 1
            list_qubit_stage[gate[0]] = stage
            list_qubit_stage[gate[1]] = stage
        self.save_results()

    def asap_separate(self) -> None:
        """Schedule 2q layers first, then place 1q gates without reordering them."""
        list_qubit_stage = [0 for _ in range(self.results_code["n_q"])]

        two_qubit_gates = []
        single_qubit_gates = []

        for index, gate in enumerate(self.g_q):
            if gate[0] == gate[1]:
                single_qubit_gates.append((index, gate))
            else:
                two_qubit_gates.append((index, gate))

        for index, gate in two_qubit_gates:
            stage = max(list_qubit_stage[gate[0]], list_qubit_stage[gate[1]])
            if stage >= len(self.list_scheduling):
                self.list_scheduling.append([])
            self.list_scheduling[stage].append(index)

            stage += 1
            list_qubit_stage[gate[0]] = stage
            list_qubit_stage[gate[1]] = stage

        prev_2q_stage = {}
        for index, gate in enumerate(self.g_q):
            if index in [item for stage in self.list_scheduling for item in stage]:
                prev_2q_stage[gate[0]] = next(
                    stage_index
                    for stage_index, stage in enumerate(self.list_scheduling)
                    if index in stage
                )
                prev_2q_stage[gate[1]] = next(
                    stage_index
                    for stage_index, stage in enumerate(self.list_scheduling)
                    if index in stage
                )
            else:
                stage = (
                    max(
                        prev_2q_stage.get(gate[0], -1),
                        prev_2q_stage.get(gate[1], -1),
                    )
                    + 1
                )

                while stage >= len(self.list_scheduling):
                    self.list_scheduling.append([])

                existing_two_qubit_gates = any(
                    self.g_q[item][0] != self.g_q[item][1]
                    for item in self.list_scheduling[stage]
                )

                if existing_two_qubit_gates:
                    self.list_scheduling.insert(stage, [index])
                else:
                    self.list_scheduling[stage].append(index)

        self.save_results()

    def save_results(self) -> None:
        """Write ``list_scheduling`` into ``results_code['stages']``."""
        stage_dict = {}
        for stage, gates in enumerate(self.list_scheduling):
            if all(self.g_q[gate][0] == self.g_q[gate][1] for gate in gates):
                stage_type = "1qGate"
            elif all(self.g_q[gate][0] != self.g_q[gate][1] for gate in gates):
                stage_type = "2qGate"
            else:
                stage_type = "mGate"
            stage_dict[stage] = {
                "type": stage_type,
                "idx": gates,
                "gates": [self.g_q[gate] for gate in gates],
                "operation_ids": [self.operation_ids[gate] for gate in gates],
            }

        qs_status = {
            qubit: [
                {"stage": stage_id, "status": None}
                for stage_id in range(len(self.list_scheduling))
            ]
            for qubit in range(self.results_code["n_q"])
        }
        for stage, gates in enumerate(self.list_scheduling):
            for gate in gates:
                q0, q1 = self.g_q[gate]
                if q0 == q1:
                    qs_status[q0][stage]["status"] = "1qGate"
                else:
                    qs_status[q0][stage]["status"] = "2qGate"
                    qs_status[q1][stage]["status"] = "2qGate"

        self.results_code["stages"]["qs_status"] = qs_status
        self.results_code["stages"]["num_stage"] = len(self.list_scheduling)
        self.results_code["stages"]["stage"] = stage_dict
