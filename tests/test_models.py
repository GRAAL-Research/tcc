import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from text_complexity_computer.models import RandomForestClassifier, Scaler
from text_complexity_computer.tools import resource_path

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "reference_scores.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model() -> RandomForestClassifier:
    return RandomForestClassifier(resource_path("fr", "fr_model.npz"))


def _reference_features(scaler: str) -> pd.DataFrame:
    return pd.DataFrame([REFERENCE["results"][scaler][name]["metrics"] for name in REFERENCE["texts"]])


@pytest.mark.parametrize("scaler", ["MinMaxScaler", "StandardScaler"])
def test_predict_matches_reference_on_reference_features(model: RandomForestClassifier, scaler: str) -> None:
    expected = [REFERENCE["results"][scaler][name]["compute"][0] for name in REFERENCE["texts"]]

    assert model.predict(_reference_features(scaler)).tolist() == expected


def test_predict_proba_is_a_distribution(model: RandomForestClassifier) -> None:
    proba = model.predict_proba(_reference_features("StandardScaler"))

    assert proba.shape == (len(REFERENCE["texts"]), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_predict_matches_features_by_name_not_position(model: RandomForestClassifier) -> None:
    features = _reference_features("MinMaxScaler")

    shuffled = features[features.columns[::-1]]

    assert model.predict(shuffled).tolist() == model.predict(features).tolist()


def test_predict_raises_on_missing_feature(model: RandomForestClassifier) -> None:
    features = _reference_features("MinMaxScaler").drop(columns=["mls"])

    with pytest.raises(ValueError, match="mls"):
        model.predict(features)


@pytest.mark.parametrize("scaler", ["MinMaxScaler", "StandardScaler"])
def test_scaler_transforms_raw_reference_metrics(scaler: str) -> None:
    raw = _reference_features("None")
    expected = _reference_features(scaler)

    scaled = Scaler(resource_path("fr", f"fr_{scaler}.npz")).transform(raw)

    np.testing.assert_allclose(scaled, expected[raw.columns].to_numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_predict_raises_on_non_finite_feature(model: RandomForestClassifier, bad_value: float) -> None:
    features = _reference_features("MinMaxScaler")
    features.loc[0, "mls"] = bad_value

    with pytest.raises(ValueError, match="NaN or infinity"):
        model.predict(features)
