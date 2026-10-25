"""Characterization tests: freeze metrics and predictions against reference values.

The reference values were computed with the dependency versions the pickled models were trained with
(see ``_provenance`` in the fixture). Any drift means a dependency upgrade changed the features fed to the model.
"""

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from text_complexity_computer import TextComplexityComputer

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "reference_scores.json").read_text(encoding="utf-8"))
SCALERS = list(REFERENCE["results"])
TEXT_NAMES = list(REFERENCE["texts"])


@pytest.fixture(scope="module", params=SCALERS)
def tcc_and_scaler(request: pytest.FixtureRequest) -> tuple[TextComplexityComputer, str]:
    scaler = None if request.param == "None" else request.param
    with pytest.warns(UserWarning) if scaler is None else nullcontext():
        tcc = TextComplexityComputer(scaler=scaler)
    return tcc, request.param



@pytest.mark.parametrize("text_name", TEXT_NAMES)
def test_metrics_scores_match_reference(tcc_and_scaler: tuple[TextComplexityComputer, str], text_name: str) -> None:
    tcc, scaler_key = tcc_and_scaler
    expected = REFERENCE["results"][scaler_key][text_name]["metrics"]

    scores = tcc.get_metrics_scores(REFERENCE["texts"][text_name])

    assert list(scores.columns) == list(expected)
    actual = {name: float(value) for name, value in scores.iloc[0].items()}
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("text_name", TEXT_NAMES)
def test_compute_matches_reference(tcc_and_scaler: tuple[TextComplexityComputer, str], text_name: str) -> None:
    tcc, scaler_key = tcc_and_scaler
    expected = REFERENCE["results"][scaler_key][text_name]["compute"]

    prediction = tcc.model.predict(tcc.get_metrics_scores(REFERENCE["texts"][text_name]))

    assert [int(value) for value in prediction] == expected
