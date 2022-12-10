"""Tests for sentence negation."""

import pytest
from negate import Negator
from .data import (aux_root_affirmative, aux_root_negative,
                   aux_root_children_affirmative, aux_root_children_negative,
                   general_verbs_affirmative, general_verbs_negative,
                   misc, failing)


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    aux_root_affirmative
)
def test_aux_root_affirmative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    aux_root_negative
)
def test_aux_root_negative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    aux_root_children_affirmative
)
def test_aux_root_children_affirmative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    aux_root_children_negative
)
def test_aux_root_children_negative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    general_verbs_affirmative
)
def test_general_verbs_affirmative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    general_verbs_negative
)
def test_general_verbs_negative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    misc
)
def test_misc(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence

@pytest.mark.xfail
@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    failing
)
def test_failing(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool
):
    assert negator.negate_sentence(
        input_sentence, prefer_contractions) == output_sentence
