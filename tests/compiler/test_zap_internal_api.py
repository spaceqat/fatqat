import pytest

from fatqat.compiler.algorithms.zap import (
    ZapInteraction,
    ZapTrace,
    architecture_sites,
    load_architecture,
)
from fatqat.compiler.algorithms import (
    ZapInteraction as ExportedZapInteraction,
    ZapTrace as ExportedZapTrace,
    architecture_sites as exported_architecture_sites,
    load_architecture as exported_load_architecture,
)


def test_internal_zap_values_and_packaged_default_architecture():
    interaction = ZapInteraction("na.0", (0, 1))
    trace = ZapTrace(2, ({"type": "Init"},))
    first = load_architecture("default")
    second = load_architecture("default")

    assert interaction.operation_id == "na.0"
    assert interaction.atoms == (0, 1)
    assert trace.atom_count == 2
    assert trace.instructions == ({"type": "Init"},)
    assert first == second
    assert first is not second
    assert architecture_sites(first, "storage_zones")
    assert architecture_sites(first, "entanglement_zones")
    assert ExportedZapInteraction is ZapInteraction
    assert ExportedZapTrace is ZapTrace
    assert exported_architecture_sites is architecture_sites
    assert exported_load_architecture is load_architecture


@pytest.mark.parametrize(
    ("operation_id", "atoms", "message"),
    (
        ("", (0,), "operation_id"),
        (1, (0,), "operation_id"),
        ("na.0", (), "one or two"),
        ("na.0", (0, 1, 2), "one or two"),
        ("na.0", (-1,), "non-negative integers"),
        ("na.0", (True,), "non-negative integers"),
        ("na.0", (0.5,), "non-negative integers"),
        ("na.0", (1, 1), "distinct"),
    ),
    ids=(
        "empty-operation-id",
        "non-string-operation-id",
        "no-atoms",
        "three-atoms",
        "negative-atom",
        "boolean-atom",
        "fractional-atom",
        "duplicate-endpoint",
    ),
)
def test_internal_zap_interaction_rejects_invalid_values(operation_id, atoms, message):
    with pytest.raises(ValueError, match=message):
        ZapInteraction(operation_id, atoms)


@pytest.mark.parametrize("name", ("default", "scale_to_100", "scale_to_500"))
def test_each_packaged_architecture_loads_as_an_independent_value(name):
    first = load_architecture(name)
    second = load_architecture(name)

    first["storage_zones"][0]["slms"][0]["location"][0] = -1

    assert second["storage_zones"][0]["slms"][0]["location"][0] == 0


def test_internal_architecture_loader_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unknown ZAP architecture"):
        load_architecture("missing")


def test_architecture_sites_expands_rows_and_columns_without_duplicate_sites():
    architecture = {
        "storage_zones": [
            {
                "slms": [
                    {
                        "location": [2, 3],
                        "site_seperation": [4, 5],
                        "r": 2,
                        "c": 2,
                    },
                    {
                        "location": [2, 3],
                        "site_seperation": [4, 5],
                        "r": 1,
                        "c": 1,
                    },
                ]
            }
        ]
    }

    assert architecture_sites(architecture, "storage_zones") == [
        (2, 3),
        (6, 3),
        (2, 8),
        (6, 8),
    ]
