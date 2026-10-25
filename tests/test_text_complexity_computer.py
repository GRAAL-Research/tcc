import numpy as np
import pytest
import spacy
from spacy.tokens import Doc

from text_complexity_computer import METRICS, TextComplexityComputer, clean_text
from text_complexity_computer import text_complexity_computer as tcc_module
from text_complexity_computer.calculation_functions import biberpy

TEXT = "Le chat dort sur le tapis. Il fait beau aujourd'hui."


class TestInit:
    def test_unknown_scaler_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scaler"):
            TextComplexityComputer(scaler="RobustScaler")  # type: ignore[arg-type]

    def test_without_scaler_warns(self) -> None:
        with pytest.warns(UserWarning, match="without scaler"):
            tcc = TextComplexityComputer(scaler=None)

        assert tcc.scaler is None

    @pytest.mark.parametrize("falsy_scaler", [False, ""])
    def test_falsy_scaler_means_no_scaler(self, falsy_scaler: object) -> None:
        with pytest.warns(UserWarning, match="without scaler"):
            tcc = TextComplexityComputer(scaler=falsy_scaler)  # type: ignore[arg-type]

        assert tcc.scaler is None

    def test_missing_spacy_model_without_pip_gives_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing_model(name: str) -> None:
            raise OSError(f"[E050] Can't find model '{name}'.")

        def download_without_pip(name: str) -> None:
            raise SystemExit(1)

        monkeypatch.setattr(spacy, "load", missing_model)
        monkeypatch.setattr(tcc_module, "download", download_without_pip)

        with pytest.raises(OSError, match="python -m spacy download fr_core_news_sm"):
            TextComplexityComputer()


class TestGetMetricsScores:
    def test_all_metrics_returns_every_metric_sorted(self, tcc: TextComplexityComputer) -> None:
        scores = tcc.get_metrics_scores(TEXT)

        expected_columns = sorted([*METRICS, *biberpy.dimnames.values()])
        assert list(scores.columns) == expected_columns
        assert scores.shape == (1, len(expected_columns))

    def test_single_metric_name(self, tcc: TextComplexityComputer) -> None:
        scores = tcc.get_metrics_scores(TEXT, metrics="mls", with_biberpy=False)

        assert list(scores.columns) == ["mls"]

    def test_metric_subset_matches_full_computation(self, tcc: TextComplexityComputer) -> None:
        full = tcc.get_metrics_scores(TEXT)

        subset = tcc.get_metrics_scores(TEXT, metrics=["nlm", "fk_ease"])

        assert subset["nlm"].iloc[0] == pytest.approx(full["nlm"].iloc[0])
        assert subset["fk_ease"].iloc[0] == pytest.approx(full["fk_ease"].iloc[0])

    def test_no_metric_without_biberpy_returns_empty_frame(self, tcc: TextComplexityComputer) -> None:
        scores = tcc.get_metrics_scores(TEXT, metrics=None, with_biberpy=False)

        assert scores.shape == (1, 0)

    def test_metrics_as_numpy_array(self, tcc: TextComplexityComputer) -> None:
        scores = tcc.get_metrics_scores(TEXT, metrics=np.array(["nlm", "mls"]), with_biberpy=False)

        assert list(scores.columns) == ["mls", "nlm"]

    def test_mattr_and_mtld_share_the_same_value(self, tcc: TextComplexityComputer) -> None:
        scores = tcc.get_metrics_scores(TEXT * 20, metrics=["mattr", "mtld"], with_biberpy=False)

        assert scores["mattr"].iloc[0] == scores["mtld"].iloc[0]

    def test_unknown_metric_raises(self, tcc: TextComplexityComputer) -> None:
        with pytest.raises(ValueError, match="not_a_metric"):
            tcc.get_metrics_scores(TEXT, metrics=["mls", "not_a_metric"])


def test_clean_text_is_exported() -> None:
    assert clean_text("Il  fait   beau.") == "Il fait beau."


class TestCompute:
    def test_returns_one_known_class(self, tcc: TextComplexityComputer) -> None:
        prediction = tcc.compute(TEXT)

        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (1,)
        assert prediction[0] in tcc.model.classes_


class TestGetSpObject:
    def test_matches_spacy_pipeline_on_clean_text(self, tcc: TextComplexityComputer) -> None:
        sp_object_test = spacy.load("fr_core_news_sm")("Je suis une tortue.")

        sp_object = tcc.get_sp_object("Je suis une tortue.")

        assert isinstance(sp_object, Doc)
        assert sp_object.text == sp_object_test.text
        for token, token_test in zip(sp_object, sp_object_test, strict=True):
            assert (token.text, token.pos_, token.lemma_, token.dep_) == (
                token_test.text,
                token_test.pos_,
                token_test.lemma_,
                token_test.dep_,
            )
