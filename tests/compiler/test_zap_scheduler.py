import pytest

from fatqat.compiler.algorithms.zap.scheduler import Scheduler


def test_asap_joint_preserves_dependency_order_and_operation_ids():
    results = {"n_q": 3}
    scheduler = Scheduler(
        g_q=[(0, 1), (2, 2), (1, 2)],
        operation_ids=("g0", "g1", "g2"),
        results_code=results,
    )

    scheduler.asap_joint()

    assert scheduler.list_scheduling == [[0, 1], [2]]
    assert results["stages"]["stage"][0]["operation_ids"] == ["g0", "g1"]
    assert results["stages"]["stage"][1]["operation_ids"] == ["g2"]


def test_asap_joint_assigns_deterministic_input_ids_when_ids_are_omitted():
    results = {"n_q": 2}
    scheduler = Scheduler(g_q=[(0, 0), (0, 1)], results_code=results)

    scheduler.asap_joint()

    assert results["stages"]["stage"][0]["operation_ids"] == ["input.0"]
    assert results["stages"]["stage"][1]["operation_ids"] == ["input.1"]


def test_scheduler_rejects_operation_ids_with_a_mismatched_length():
    with pytest.raises(ValueError, match="same length"):
        Scheduler(
            g_q=[(0, 0), (1, 1)],
            operation_ids=("g0",),
            results_code={"n_q": 2},
        )


@pytest.mark.parametrize(
    ("gates", "expected_type"),
    (
        ([(0, 0), (1, 1)], "1qGate"),
        ([(0, 1), (2, 3)], "2qGate"),
        ([(0, 0), (1, 2)], "mGate"),
    ),
)
def test_save_results_classifies_single_stage_gate_types(gates, expected_type):
    results = {"n_q": 4}
    scheduler = Scheduler(g_q=gates, results_code=results)

    scheduler.asap_joint()

    assert scheduler.list_scheduling == [[0, 1]]
    assert results["stages"]["stage"][0]["type"] == expected_type


def test_asap_joint_produces_equal_results_for_identical_inputs():
    gates = [(0, 1), (2, 2), (1, 2), (0, 0)]
    first_results = {"n_q": 3}
    second_results = {"n_q": 3}
    first = Scheduler(gates, first_results)
    second = Scheduler(gates, second_results)

    first.asap_joint()
    second.asap_joint()

    assert first.list_scheduling == [[0, 1], [2, 3]]
    assert first.list_scheduling == second.list_scheduling
    assert first_results["stages"] == second_results["stages"]


def test_asap_separate_keeps_two_qubit_layers_and_inserts_one_qubit_gates():
    results = {"n_q": 3}
    scheduler = Scheduler(g_q=[(0, 1), (2, 2), (1, 2)], results_code=results)

    scheduler.asap_separate()

    assert scheduler.list_scheduling == [[1], [0], [2]]
    assert [results["stages"]["stage"][index]["type"] for index in range(3)] == [
        "1qGate",
        "2qGate",
        "2qGate",
    ]
