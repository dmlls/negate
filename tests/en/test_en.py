"""Tests for sentence negation."""

from pathlib import Path

import pytest
from negate import Negator

from ..utils import set_up_negator
from .data import (
    aux_root_affirmative,
    aux_root_children_affirmative,
    aux_root_children_negative,
    aux_root_negative,
    failing,
    general_verbs_affirmative,
    general_verbs_negative,
    inversions_affirmative,
    inversions_negative,
    misc,
)

negator_model: Negator = None


@pytest.fixture
def negator(request) -> Negator:
    # Initialize negator only once.
    global negator_model
    if negator_model is None:
        negator_model = set_up_negator(
            request=request, language=Path(__file__).parent.name
        )
    return negator_model


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", aux_root_affirmative
)
def test_aux_root_affirmative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", aux_root_negative
)
def test_aux_root_negative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions",
    aux_root_children_affirmative,
)
def test_aux_root_children_affirmative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", aux_root_children_negative
)
def test_aux_root_children_negative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", general_verbs_affirmative
)
def test_general_verbs_affirmative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", general_verbs_negative
)
def test_general_verbs_negative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", inversions_affirmative
)
def test_inversions_affirmative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", inversions_negative
)
def test_inversions_negative(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.parametrize("input_sentence, output_sentence, prefer_contractions", misc)
def test_misc(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )


@pytest.mark.xfail
@pytest.mark.parametrize(
    "input_sentence, output_sentence, prefer_contractions", failing
)
def test_failing(
    negator: Negator,  # pylint: disable=redefined-outer-name
    input_sentence: str,
    output_sentence: str,
    prefer_contractions: bool,
):
    assert (
        negator.negate_sentence(input_sentence, prefer_contractions=prefer_contractions)
        == output_sentence
    )
