"""Frozen ZAP atom placement algorithm.

Derived from the bundled ZAP placer under the MIT License; see LICENSE in this
package.
"""

# Preserve the frozen companion control flow and mutation layout verbatim.
# pylint: disable=attribute-defined-outside-init,cell-var-from-loop
# pylint: disable=consider-using-enumerate,consider-using-in,no-else-break,no-else-return

import math
import numpy as np

R_B = 5  # Rydberg interaction radius


class Placer:
    """
    Class to determine a qubit layout based on minimal distances between qubits
    and available interaction sites within a specified Rydberg interaction radius.
    """

    def __init__(
        self,
        slm_sites: list,
        results_code: dict,
        list_full_gates: list,
        qubit_mapping: list,
        routing_strategy: str,
        architecture: dict = None,
    ):
        """
        Args:
            slm_sites: ``[storage_traps, entanglement_traps]``.
            results_code: Must include ``n_q``, ``stages`` from the scheduler.
            list_full_gates: Gates grouped per schedule stage.
            qubit_mapping: User-specified traps, or empty for automatic init.
            routing_strategy: Idle handling policy (``baseline``, ``lookahead``, ``always_move``, ``always_stay``).
            architecture: Optional fidelity/T2/routing weights for cost heuristics.
        """
        self.stg_slm_sites, self.ent_slm_sites = slm_sites
        self.results_code = results_code
        self.stages = results_code["stages"]["stage"]
        self.n_q = results_code["n_q"]
        self.list_full_gates = list_full_gates
        self.routing_strategy = routing_strategy
        self.qs_status = results_code["stages"]["qs_status"]

        architecture = architecture or {}
        operation_fidelity = architecture.get("operation_fidelity", {})
        qubit_spec = architecture.get("qubit_spec", {})
        routing_cfg = architecture.get("routing", {})

        two_q_fid = operation_fidelity.get("two_qubit_gate", 0.999)
        self.fidelity_2q_idle = 1 - (1 - two_q_fid) / 2

        self.fidelity_atom_transfer = operation_fidelity.get("atom_transfer", 0.999)

        self.time_coherence = qubit_spec.get("T2", 1.5e6)
        self.parallel_priority_weight = float(
            routing_cfg.get("parallel_priority_weight", 1000.0)
        )
        self.initial_mapping_parallel_lookahead = int(
            routing_cfg.get("initial_mapping_parallel_lookahead", 3)
        )
        # ``routing_strategy``: idle policy — ``baseline``/``lookahead`` compare crosstalk vs
        # transfer+movement decoherence; ``always_move`` / ``always_stay`` force one side.
        self.idle_cost_alpha = float(routing_cfg.get("idle_cost_alpha", 1.0))

        if len(qubit_mapping) > 0:
            self.current_mapping = qubit_mapping
        else:
            self.current_mapping = [(-1, -1) for _ in range(self.n_q)]
            self.init_slm_sites()
            self.init_mapping()

        from copy import deepcopy

        # Home traps for idle-vs-storage heuristics and final return-to-storage bias.
        self.initial_mapping = deepcopy(self.current_mapping)

        self.init_pairs()

    def movement_duration(self, d: float) -> float:
        """
        Approximate movement duration (us) for a given distance (um).
        This mirrors Router.movement_duration so that the placer can
        roughly assess decoherence cost caused by qubit movement.
        """
        if d <= 0:
            return 0.0
        return 200 * ((d / 110) ** 0.5)

    def init_slm_sites(self):
        n_cols = [1 for _ in range(len(self.stg_slm_sites))]
        n_rows = [1 for _ in range(len(self.stg_slm_sites))]
        for i in range(len(self.stg_slm_sites)):
            for j in range(i + 1, len(self.stg_slm_sites)):
                if self.stg_slm_sites[i][0] == self.stg_slm_sites[j][0]:
                    n_rows[i] += 1
                if self.stg_slm_sites[i][1] == self.stg_slm_sites[j][1]:
                    n_cols[i] += 1
        n_col = max(n_cols)
        n_row = max(n_rows)
        stg_n_col = min(int(np.sqrt(self.n_q)) + 4, n_col)
        stg_n_row = min(int(np.sqrt(self.n_q)) + 2, n_row)
        while True:
            if stg_n_col * stg_n_row >= self.n_q:
                break
            else:
                stg_n_col = min(stg_n_col + 1, n_col)
                stg_n_row = min(stg_n_row + 1, n_row)

        self.stg_slm_sites = sorted(
            self.stg_slm_sites, key=lambda item: (item[0], -item[1])
        )
        tmp = []
        for i in range(stg_n_col):
            for j in range(stg_n_row):
                tmp.append(self.stg_slm_sites[i * n_row + j])
        self.stg_slm_sites = tmp

        n_cols = [1 for _ in range(len(self.ent_slm_sites))]
        n_rows = [1 for _ in range(len(self.ent_slm_sites))]
        for i in range(len(self.ent_slm_sites)):
            for j in range(i + 1, len(self.ent_slm_sites)):
                if self.ent_slm_sites[i][0] == self.ent_slm_sites[j][0]:
                    n_rows[i] += 1
                if self.ent_slm_sites[i][1] == self.ent_slm_sites[j][1]:
                    n_cols[i] += 1
        n_col = max(n_cols)
        n_row = max(n_rows)
        ent_n_col = min(stg_n_col + 4, n_col)
        ent_n_row = min(stg_n_row, n_row)
        self.ent_slm_sites = sorted(
            self.ent_slm_sites, key=lambda item: (item[0], item[1])
        )
        tmp = []
        for i in range(ent_n_col):
            for j in range(ent_n_row):
                tmp.append(self.ent_slm_sites[i * n_row + j])
        self.ent_slm_sites = tmp

    def init_mapping(self):
        """
        Initialize the qubit mapping based on minimal distances.
        """
        # Sort storage qubits by their distance to the closest entanglement zone location
        self.stg_slm_sites = sorted(
            self.stg_slm_sites,
            key=lambda coord: float(
                np.mean([math.dist(coord, ez_coord) for ez_coord in self.ent_slm_sites])
            ),
        )

        # Assign weights to qubits based on the number of gates they participate in
        self.qubit_weight = np.zeros(self.n_q)
        for stage, gates in enumerate(self.list_full_gates):
            weight = 1 / (stage + 1)
            for i in range(len(gates)):
                self.qubit_weight[gates[i][0]] += weight
                self.qubit_weight[gates[i][1]] += weight

        # Sort qubits based on their weights
        qubit_priority = sorted(
            range(self.n_q), key=lambda i: self.qubit_weight[i], reverse=True
        )

        # Parallel-aware assignment for early movements:
        # estimate first move as storage -> nearest entanglement site and avoid
        # producing mutually incompatible vectors (same rule as router.compatible_2D).
        assigned_sites = []
        planned_init_vectors = []

        nearest_ent_site = {}
        nearest_ent_dist = {}
        for site in self.stg_slm_sites:
            best_ent = min(self.ent_slm_sites, key=lambda ez: math.dist(site, ez))
            nearest_ent_site[site] = best_ent
            nearest_ent_dist[site] = math.dist(site, best_ent)

        first_2q_stage = [None for _ in range(self.n_q)]
        for stage_idx, gates in enumerate(self.list_full_gates):
            for q0, q1 in gates:
                if q0 != q1:
                    if first_2q_stage[q0] is None:
                        first_2q_stage[q0] = stage_idx
                    if first_2q_stage[q1] is None:
                        first_2q_stage[q1] = stage_idx

        for q in qubit_priority:
            available_sites = [
                site for site in self.stg_slm_sites if site not in assigned_sites
            ]
            if not available_sites:
                break

            best_site = None
            best_rank = None
            for site in available_sites:
                vector = None
                stage_idx = first_2q_stage[q]
                if (
                    stage_idx is not None
                    and stage_idx <= self.initial_mapping_parallel_lookahead
                ):
                    target = nearest_ent_site[site]
                    vector = (site[0], target[0], site[1], target[1])

                conflicts = 0
                if vector is not None:
                    for planned in planned_init_vectors:
                        if not self._compatible_2d(vector, planned):
                            conflicts += 1

                # Weighted score: parallel priority weight + nearest entanglement site distance.
                score = (
                    self.parallel_priority_weight * conflicts + nearest_ent_dist[site]
                )
                rank = (score, conflicts, nearest_ent_dist[site])
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_site = site

            self.current_mapping[q] = best_site
            assigned_sites.append(best_site)
            stage_idx = first_2q_stage[q]
            if (
                stage_idx is not None
                and stage_idx <= self.initial_mapping_parallel_lookahead
            ):
                target = nearest_ent_site[best_site]
                planned_init_vectors.append(
                    (best_site[0], target[0], best_site[1], target[1])
                )

    def init_params(self):
        """
        Initialize parameters used during placement operations.
        """
        self.execute_qubits_1q = []
        self.execute_qubits_2q = []
        self.gate_2q = []
        self.list_gate = self.list_full_gates[self.index_list_gate]
        # Planned movement vectors in this stage; used to favor parallel moves.
        self._planned_vectors = []

    def _vector_for_move(self, q, target_site):
        sx, sy = self.current_mapping[q]
        tx, ty = target_site
        if (sx, sy) == (tx, ty):
            return None
        return (sx, tx, sy, ty)

    def _compatible_2d(self, a, b):
        """Mirror Router.compatible_2D for placement-time parallelism scoring."""
        if a[0] == b[0] and a[1] != b[1]:
            return False
        if a[1] == b[1] and a[0] != b[0]:
            return False
        if a[0] < b[0] and a[1] >= b[1]:
            return False
        if a[0] > b[0] and a[1] <= b[1]:
            return False
        if a[2] == b[2] and a[3] != b[3]:
            return False
        if a[3] == b[3] and a[2] != b[2]:
            return False
        if a[2] < b[2] and a[3] >= b[3]:
            return False
        if a[2] > b[2] and a[3] <= b[3]:
            return False
        return True

    def _vector_conflicts(self, vector, extra_vectors=None):
        if vector is None:
            return 0
        extra_vectors = extra_vectors or []
        conflicts = 0
        for planned in self._planned_vectors:
            if not self._compatible_2d(vector, planned):
                conflicts += 1
        for planned in extra_vectors:
            if not self._compatible_2d(vector, planned):
                conflicts += 1
        return conflicts

    def _commit_move(self, q, target_site):
        vector = self._vector_for_move(q, target_site)
        if vector is not None:
            self._planned_vectors.append(vector)
        self.current_mapping[q] = target_site

    def _single_move_rank(self, q, target_site, prefer_site=None):
        vector = self._vector_for_move(q, target_site)
        conflicts = self._vector_conflicts(vector)
        distance = math.dist(self.current_mapping[q], target_site)
        prefer_penalty = (
            0.0 if prefer_site is not None and target_site == prefer_site else 1.0
        )
        score = (
            conflicts * self.parallel_priority_weight + distance + 1e-6 * prefer_penalty
        )
        return score, conflicts, distance

    def _pair_move_rank(self, q0, q1, p0, p1):
        v0 = self._vector_for_move(q0, p0)
        v1 = self._vector_for_move(q1, p1)
        conflicts = self._vector_conflicts(v0) + self._vector_conflicts(
            v1, extra_vectors=[v0] if v0 else []
        )
        distance = math.dist(self.current_mapping[q0], p0) + math.dist(
            self.current_mapping[q1], p1
        )
        score = conflicts * self.parallel_priority_weight + distance
        return score, conflicts, distance

    def init_pairs(self):
        """
        Initialize all valid entanglement pairs based on distance constraints.
        """
        self.all_pairs = [
            (site1, site2)
            for i, site1 in enumerate(self.ent_slm_sites)
            for j, site2 in enumerate(self.ent_slm_sites)
            if i < j and math.dist(site1, site2) < R_B
        ]

    def placing(self, index_list_gate):
        """
        Perform qubit placement based on minimal distances.

        Args:
            index_list_gate (int): Index of the gate list to process.
        """
        self.index_list_gate = index_list_gate
        self.init_params()
        # Classify qubits into 1-qubit and 2-qubit gates
        self.assign_qubits()

        self.available_pairs = [
            (p0, p1)
            for p0, p1 in self.all_pairs
            if p0 not in self.current_mapping and p1 not in self.current_mapping
        ]

        if self.stages[index_list_gate]["type"] != "1qGate":
            self._move_unentangled_qubits()
            self._process_two_qubit_gates()

        return self.current_mapping

    def _move_unentangled_qubits(self):
        """Move unentangled qubits to appropriate storage sites."""

        # Helper: check if a site is available (not occupied by other qubits)
        def is_site_available_for(site, qubit_id):
            for i, pos in enumerate(self.current_mapping):
                if i != qubit_id and pos == site:
                    return False
            return True

        num_stages = self.results_code["stages"]["num_stage"]
        for q in range(self.n_q):
            # Only consider qubits that are currently in the entanglement zone
            # and are not participating in a two-qubit gate at this stage.
            if (
                self.current_mapping[q] in self.ent_slm_sites
                and self.qs_status[q][self.index_list_gate]["status"] != "2qGate"
            ):

                # Find the next stage where this qubit participates in a 2q gate.
                next_2q_index = None
                for i in range(self.index_list_gate + 1, num_stages):
                    if self.qs_status[q][i]["status"] == "2qGate":
                        next_2q_index = i
                        break

                # Determine the range of stages during which this qubit would
                # experience crosstalk if it stays in the entanglement zone.
                if next_2q_index is None:
                    # No future 2q gates for this qubit: consider until the end.
                    idle_end = num_stages
                    num_transfer_events = 2
                else:
                    idle_end = next_2q_index
                    num_transfer_events = 4

                # Count how many stages in [current, idle_end) contain 2q or mGate gates.
                crosstalk_stage_count = max(0, idle_end - self.index_list_gate)

                # --- Estimate crosstalk cost (log-fidelity loss) ---
                # Each affected stage applies fidelity_2q_idle once to this qubit.
                crosstalk_cost = crosstalk_stage_count * (
                    -math.log(self.fidelity_2q_idle)
                )

                # --- Estimate movement and decoherence cost if we move to storage ---
                # Atom-transfer cost (estimated).
                transfer_cost = num_transfer_events * (
                    -math.log(self.fidelity_atom_transfer)
                )

                # Choose the best candidate storage site (without actually moving yet).
                original_trap = self.initial_mapping[q]
                candidate_sites = []
                if original_trap in self.stg_slm_sites and is_site_available_for(
                    original_trap, q
                ):
                    candidate_sites.append(original_trap)
                else:
                    candidate_sites.extend(
                        site
                        for site in self.stg_slm_sites
                        if is_site_available_for(site, q)
                    )

                if not candidate_sites:
                    # No available storage site: cannot move this qubit now.
                    continue

                # Approximate movement duration using the nearest available storage site.
                candidate_sites.sort(
                    key=lambda site: math.dist(site, self.current_mapping[q])
                )
                best_storage_site = candidate_sites[0]
                distance_out = math.dist(self.current_mapping[q], best_storage_site)

                # Assume we will later move back over a similar distance.
                approx_movement_time = (
                    0.5 * num_transfer_events * self.movement_duration(distance_out)
                )

                # Decoherence penalty: movement introduces additional time during which
                # some qubits are idle. For a small segment dt, -log(exp(-dt/T2)) ≈ dt/T2.
                decoherence_cost = (
                    (self.n_q - 1) * approx_movement_time / self.time_coherence
                )

                alpha = self.idle_cost_alpha
                move_cost = transfer_cost + alpha * decoherence_cost

                # If the expected crosstalk loss dominates the movement+decoherence loss,
                # we move this qubit to storage; otherwise keep it in the entanglement zone.
                if self.routing_strategy == "always_move":
                    self._move_to_target_zone(q, "storage")
                elif self.routing_strategy == "always_stay":
                    pass
                elif (
                    self.routing_strategy == "lookahead"
                    or self.routing_strategy == "baseline"
                ):
                    if crosstalk_cost > move_cost:
                        self._move_to_target_zone(q, "storage")
                else:
                    raise RuntimeError(
                        f"unhandled routing_strategy {self.routing_strategy!r}"
                    )

            # Also check if qubit is in storage zone but not at original trap position,
            # and will not be involved in future 2q gates: we can safely return it home.
            elif (
                self.current_mapping[q] in self.stg_slm_sites
                and self.current_mapping[q] != self.initial_mapping[q]
                and self.qs_status[q][self.index_list_gate] != "2qGate"
            ):
                if self.index_list_gate < num_stages - 1:
                    has_future_2q = any(
                        self.qs_status[q][i] == "2qGate"
                        for i in range(self.index_list_gate + 1, num_stages)
                    )
                    if not has_future_2q:
                        self._move_to_target_zone(q, "storage")
                else:
                    # Last stage, return to original trap
                    self._move_to_target_zone(q, "storage")

    def _process_two_qubit_gates(self):
        for q0, q1 in self.gate_2q:
            # Check if qubits are already mapped to the same site
            # If true, no further action is needed
            # If not, find the closest pair of sites for the two qubits
            if not (
                (self.current_mapping[q0], self.current_mapping[q1]) in self.all_pairs
                or (self.current_mapping[q1], self.current_mapping[q0])
                in self.all_pairs
            ):
                if (
                    self.pair_status(self.current_mapping[q0]) == 1
                    and self.pair_status(self.current_mapping[q1]) == 1
                ):
                    other_site = self.find_other_pair_site(self.current_mapping[q0])
                    if other_site is not None:
                        self._commit_move(q1, other_site)
                    else:
                        raise ValueError(f"No valid pair site found for qubit {q1}")
                elif self.pair_status(self.current_mapping[q0]) == 1:
                    other_site = self.find_other_pair_site(self.current_mapping[q0])
                    if other_site is not None:
                        self._commit_move(q1, other_site)
                    else:
                        raise ValueError(f"No valid pair site found for qubit {q1}")
                elif self.pair_status(self.current_mapping[q1]) == 1:
                    other_site = self.find_other_pair_site(self.current_mapping[q1])
                    if other_site is not None:
                        self._commit_move(q0, other_site)
                    else:
                        raise ValueError(f"No valid pair site found for qubit {q0}")
                else:
                    # If both qubits are mapped to entanglement sites,
                    # and both pairs are not available,
                    # find the closest pair of sites
                    target_q0, target_q1 = self._find_closest_pair(q0, q1)
                    self._commit_move(q0, target_q0)
                    self._commit_move(q1, target_q1)

    def assign_qubits(self):
        """
        Classify gates into 1-qubit and 2-qubit gates.
        """
        self.execute_qubits = sorted({q for gate in self.list_gate for q in gate})
        for gate in self.list_gate:
            q0, q1 = gate
            if q0 == q1:
                self.execute_qubits_1q.append(q0)
            else:
                self.gate_2q.append((q0, q1))
                self.execute_qubits_2q += [q0, q1]

    def find_other_pair_site(self, site):
        """
        Find the other site of a pair given one site.

        Args:
            site (tuple): The site coordinates.

        Returns:
            tuple: The other site of the pair.
        """
        for p0, p1 in self.all_pairs:
            if site in (p0, p1):
                return p0 if site == p1 else p1
        return None

    def pair_status(self, site):
        """
        Check the pairing status of a given site.

        Args:
            site (tuple): The site coordinates.

        Returns:
            int: Pairing status (0 = unentangled, 1 = partially mapped, 2 = fully mapped).
        """
        for p0, p1 in self.all_pairs:
            if site in (p0, p1):
                mapped0 = p0 in self.current_mapping
                mapped1 = p1 in self.current_mapping

                if mapped0 and mapped1:
                    return 2  # Both mapped
                else:
                    return 1  # Partially mapped
        return -1  # Not in any pair

    def _move_to_target_zone(self, q, zone_type, compulsory=True):
        """
        Move qubit q to a specified zone based on zone type (nearest, storage, entanglement).
        When moving to storage zone, prioritizes returning to the original trap position.

        Args:
            q (int): Index of the qubit to move.
            zone_type (str): Target zone type ('nearest', 'storage', or 'entanglement').
            compulsory (bool): Whether the move to the specified zone type is mandatory.

        Raises:
            ValueError: If no available site is found in the specified zone.
        """

        # Helper function to check if a site is available (not occupied by other qubits)
        def is_site_available(site):
            # Check if site is occupied by any qubit other than q
            for i, pos in enumerate(self.current_mapping):
                if i != q and pos == site:
                    return False
            return True

        # Select target zone based on type and availability
        if zone_type == "storage":
            # First, check if the original trap position is available
            original_trap = self.initial_mapping[q]
            # Original trap available is a preference, not a hard constraint.
            available_sites = [
                site for site in self.stg_slm_sites if is_site_available(site)
            ]
            if not available_sites and not compulsory:
                zone_type = "nearest"

        if zone_type == "entanglement":
            available_sites = [
                site for site in self.ent_slm_sites if self.pair_status(site) == 0
            ]
            if not available_sites and not compulsory:
                zone_type = "nearest"

        if zone_type == "nearest":
            available_sites = [
                site for site in self.stg_slm_sites if is_site_available(site)
            ]
            available_sites += [
                site for site in self.ent_slm_sites if self.pair_status(site) == 0
            ]

        original_trap = self.initial_mapping[q]
        prefer_site = (
            original_trap
            if (
                original_trap in self.stg_slm_sites and original_trap in available_sites
            )
            else None
        )
        available_sites = sorted(
            available_sites,
            key=lambda site: self._single_move_rank(q, site, prefer_site=prefer_site),
        )

        # Update mapping or raise error if no valid site is found
        if available_sites:
            self._commit_move(q, available_sites[0])
        else:
            raise ValueError(f"No available sites in {zone_type} zone for qubit {q}")

    def _find_closest_pair(self, q0, q1):
        """
        Find the closest available pair of sites for two qubits.

        Args:
            q0 (int): Index of the first qubit.
            q1 (int): Index of the second qubit.

        Returns:
            tuple: Closest pair of sites.
        """
        available_pairs = [
            (p0, p1)
            for p0, p1 in self.available_pairs
            if p0 not in self.current_mapping and p1 not in self.current_mapping
        ]

        best_rank = None
        closest_pair = None
        for p0, p1 in available_pairs:
            distance_1 = math.dist(p0, self.current_mapping[q0]) + math.dist(
                p1, self.current_mapping[q1]
            )
            distance_2 = math.dist(p1, self.current_mapping[q0]) + math.dist(
                p0, self.current_mapping[q1]
            )

            if distance_1 < distance_2:
                oriented = (p0, p1)
            else:
                oriented = (p1, p0)

            rank = self._pair_move_rank(q0, q1, oriented[0], oriented[1])
            if best_rank is None or rank < best_rank:
                best_rank = rank
                closest_pair = oriented

        if closest_pair:
            return closest_pair
        else:
            raise ValueError("No valid pair found")

    def _return_all_to_original_traps(self):
        """
        Move all qubits back to their original trap positions.
        This is called after the last gate stage to ensure all qubits return home.
        """

        # Helper function to check if a site is available for a specific qubit
        def is_site_available_for(site, qubit_id):
            for i, pos in enumerate(self.current_mapping):
                if i != qubit_id and pos == site:
                    return False
            return True

        # First pass: move qubits whose original traps are available
        for q in range(self.n_q):
            if self.current_mapping[q] != self.initial_mapping[q]:
                original_trap = self.initial_mapping[q]
                if original_trap in self.stg_slm_sites and is_site_available_for(
                    original_trap, q
                ):
                    self.current_mapping[q] = original_trap

        # Second pass: handle qubits whose original traps are occupied
        # Try to swap positions to get everyone home
        for q in range(self.n_q):
            if self.current_mapping[q] != self.initial_mapping[q]:
                original_trap = self.initial_mapping[q]
                # Find which qubit is at the original trap
                for other_q in range(self.n_q):
                    if other_q != q and self.current_mapping[other_q] == original_trap:
                        # Check if we can swap: other_q's original trap should be available or at current position
                        other_original = self.initial_mapping[other_q]
                        current_pos = self.current_mapping[q]

                        # If other qubit's original is the current position, perfect swap
                        if other_original == current_pos:
                            self.current_mapping[other_q] = current_pos
                            self.current_mapping[q] = original_trap
                            break
                        # If other qubit's original is available, we can move it there and take its place
                        elif is_site_available_for(other_original, other_q):
                            self.current_mapping[other_q] = other_original
                            self.current_mapping[q] = original_trap
                            break

                # If still not at original position, find nearest available storage site
                if self.current_mapping[q] != self.initial_mapping[q]:
                    available_sites = [
                        site
                        for site in self.stg_slm_sites
                        if is_site_available_for(site, q)
                    ]
                    if available_sites:
                        # Prefer original trap if it becomes available
                        if self.initial_mapping[q] in available_sites:
                            self.current_mapping[q] = self.initial_mapping[q]
                        else:
                            # Find nearest available site
                            available_sites = sorted(
                                available_sites,
                                key=lambda site: math.dist(
                                    site, self.current_mapping[q]
                                ),
                            )
                            self.current_mapping[q] = available_sites[0]
