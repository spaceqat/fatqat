"""Tests for public identity-based parameter values."""

from dataclasses import FrozenInstanceError

import pytest

import fatqat as fq


def test_parameter_uses_identity_for_equality_and_hashing():
    first = fq.Parameter("theta")
    second = fq.Parameter("theta")

    assert first is not second
    assert first != second
    assert len({first, second}) == 2
    assert {first: 1, second: 2}[first] == 1


@pytest.mark.parametrize("value", [None, 1, True])
def test_parameter_rejects_non_string_names(value):
    with pytest.raises(TypeError, match="name must be a string"):
        fq.Parameter(value)


def test_parameter_rejects_empty_name():
    with pytest.raises(ValueError, match="must not be empty"):
        fq.Parameter("")


def test_parameter_vector_retains_elements_in_declaration_order():
    angles = fq.ParameterVector("angles", 3)

    assert len(angles) == 3
    assert tuple(parameter.name for parameter in angles) == (
        "angles[0]",
        "angles[1]",
        "angles[2]",
    )
    assert angles[1] is angles[1]
    assert angles[-1] is angles[2]
    assert len(set(angles)) == 3


def test_same_named_parameter_vectors_are_distinct():
    first = fq.ParameterVector("angles", 2)
    second = fq.ParameterVector("angles", 2)

    assert first != second
    assert first[0] != second[0]
    assert len({first, second}) == 2


def test_parameter_vector_accepts_zero_length():
    vector = fq.ParameterVector("empty", 0)

    assert len(vector) == 0
    assert tuple(vector) == ()


@pytest.mark.parametrize("value", [True, 1.0, "2", None])
def test_parameter_vector_rejects_non_integer_length(value):
    with pytest.raises(TypeError, match="length must be an integer"):
        fq.ParameterVector("angles", value)


def test_parameter_vector_rejects_negative_length():
    with pytest.raises(ValueError, match="must be non-negative"):
        fq.ParameterVector("angles", -1)


@pytest.mark.parametrize("index", [True, 0.0, "0", slice(None)])
def test_parameter_vector_rejects_non_integer_indices(index):
    vector = fq.ParameterVector("angles", 2)

    with pytest.raises(TypeError, match="indices must be integers"):
        vector[index]


def test_parameter_vector_preserves_tuple_index_errors():
    vector = fq.ParameterVector("angles", 2)

    with pytest.raises(IndexError):
        vector[2]


def test_parameter_values_are_immutable():
    parameter = fq.Parameter("theta")
    vector = fq.ParameterVector("angles", 2)

    with pytest.raises(FrozenInstanceError):
        parameter.name = "phi"
    with pytest.raises(FrozenInstanceError):
        vector.name = "other"
    with pytest.raises(FrozenInstanceError):
        vector.length = 3
    with pytest.raises(FrozenInstanceError):
        vector._parameters = ()


def test_parameter_values_are_explicit_top_level_exports():
    assert fq.Parameter.__module__ == "fatqat.parameters"
    assert fq.ParameterVector.__module__ == "fatqat.parameters"
    assert "Parameter" in fq.__all__
    assert "ParameterVector" in fq.__all__
