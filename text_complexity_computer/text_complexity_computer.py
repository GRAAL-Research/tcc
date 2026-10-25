import subprocess
import warnings
from collections.abc import Callable, Iterable
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import spacy
from spacy.cli.download import download
from spacy.language import Language
from spacy.tokens import Doc

from .calculation_functions import (
    biberpy,
    lexical_diversity,
    readability,
    syntactic_complexity,
    vocabulary_complexity,
)
from .models import RandomForestClassifier, Scaler
from .tools import clean_text, read_num_list, read_word_lists, resource_path

__all__ = ["METRICS", "SCALERS", "TextComplexityComputer"]

LANGUAGE = "fr"
SPACY_MODEL_NAME = "fr_core_news_sm"
SCALERS = ("MinMaxScaler", "StandardScaler")

ScalerName = Literal["MinMaxScaler", "StandardScaler"]

METRICS: dict[str, Callable[[Doc], float]] = {
    "mls": syntactic_complexity.mean_length_sentences,
    "ps_30": syntactic_complexity.ps_30,
    "nws_90": syntactic_complexity.nws_90,
    "mlt": syntactic_complexity.mean_length_tunit,
    "tu_s": syntactic_complexity.tu_s,
    "ctu_tu": syntactic_complexity.ctu_tu,
    "dc_c": syntactic_complexity.dc_c,
    "c_s": syntactic_complexity.c_s,
    "c_tu": syntactic_complexity.c_tu,
    "cp_c": syntactic_complexity.cp_c,
    "cp_tu": syntactic_complexity.cp_tu,
    "pa": partial(vocabulary_complexity.pa, language=LANGUAGE),
    "nlm": vocabulary_complexity.nlm,
    "uni_gram_lem": partial(vocabulary_complexity.uni_gram_lem, language=LANGUAGE),
    "msttr": lexical_diversity.mean_sequential_ttr,
    # The bundled model was trained with the MTLD value in the "mattr" feature: keep it to preserve predictions.
    "mattr": lexical_diversity.measure_textual_lexical_diversity,
    "mtld": lexical_diversity.measure_textual_lexical_diversity,
    "fk_ease": readability.fk_ease,
    "bingui": readability.bingui,
    "km_score": readability.km_formula,
}


def _load_spacy_model(model_name: str) -> Language:
    try:
        return spacy.load(model_name)
    except OSError:
        pass
    try:
        download(model_name)
        return spacy.load(model_name)
    except (OSError, SystemExit, subprocess.CalledProcessError) as error:
        # spacy.cli.download relies on pip, which environments created by uv do not ship.
        raise OSError(
            f"The spaCy model {model_name!r} is not installed and could not be downloaded. "
            f"Install it with `python -m spacy download {model_name}` or `uv pip install {model_name}` "
            "(see https://spacy.io/models/fr)."
        ) from error


def _normalize_metrics(metrics: Iterable[str] | str | None) -> list[str]:
    if metrics is None:
        return []
    if isinstance(metrics, str) and metrics == "all":  # isinstance first: an array compares element-wise
        return list(METRICS)
    selected = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = sorted(set(selected) - set(METRICS))
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available metrics: {sorted(METRICS)}.")
    return selected


class TextComplexityComputer:
    """Assess the complexity of a French text.

    Args:
        scaler: Scaler applied to the metrics before classification: ``"MinMaxScaler"`` (default),
            ``"StandardScaler"`` or ``None`` to use raw metrics.
        verbosity: Verbosity level when reading the Biber word lists (0 is silent).

    Example:
        >>> tcc = TextComplexityComputer()
        >>> tcc.compute("Alibaba et les 40 voleurs.")
        array([1])
    """

    def __init__(self, scaler: ScalerName | None = "MinMaxScaler", verbosity: int = 1) -> None:
        if not scaler:  # None, or a falsy value such as False or "" accepted by earlier versions
            scaler = None
        elif scaler not in SCALERS:
            raise ValueError(f"Unknown scaler {scaler!r}. Choose one of {SCALERS} or None.")

        self.tagger = _load_spacy_model(SPACY_MODEL_NAME)
        self.tagger.max_length = 5_000_000

        self.word_lists, self.mwe_list = read_word_lists(resource_path(LANGUAGE, "fr.properties"), verbosity=verbosity)
        self.tag_list = read_num_list(resource_path(LANGUAGE, "fr.tag.num"))
        biberpy.configure(language=LANGUAGE, word_lists=self.word_lists, mwe_list=self.mwe_list, tag_list=self.tag_list)

        self.scaler: Scaler | None = None
        if scaler is None:
            warnings.warn("You are running TextComplexityComputer without scaler (scaler=None).", stacklevel=2)
        else:
            self.scaler = Scaler(resource_path(LANGUAGE, f"fr_{scaler}.npz"))

        self.model = RandomForestClassifier(resource_path(LANGUAGE, "fr_model.npz"))

    def get_metrics_scores(
        self,
        text: str,
        metrics: Iterable[str] | str | None = "all",
        with_biberpy: bool = True,
    ) -> pd.DataFrame:
        """Compute the metrics scores of a text.

        Available metrics (outside biberpy):

        - ``mls``: mean length of sentences
        - ``ps_30``: percentage of sentences longer than 30 words
        - ``nws_90``: length of the 90th percentile sentence
        - ``mlt``: mean length of T-units
        - ``tu_s``: T-units per sentence
        - ``ctu_tu``: complex T-units per T-unit
        - ``dc_c``: dependent clauses per clause
        - ``c_s``: clauses per sentence
        - ``c_tu``: clauses per T-unit
        - ``cp_c``: coordinate phrases per clause
        - ``cp_tu``: coordinate phrases per T-unit
        - ``pa``: percentage of words not in a reference list (Gougenheim list)
        - ``nlm``: number of letters per word
        - ``uni_gram_lem``: uni-gram model with Lexique3 on lemmas
        - ``msttr``: mean sequential type-token ratio
        - ``mattr``: alias of ``mtld`` kept for compatibility with the bundled model
        - ``mtld``: measure of textual lexical diversity
        - ``fk_ease``: Flesch-Kincaid reading ease
        - ``bingui``: punctuation marks per sentence
        - ``km_score``: Kandel and Moles formula

        The biberpy dimensions are the values of ``biberpy.dimnames``.

        Args:
            text: Text to compute.
            metrics: Metrics to compute (outside biberpy): ``"all"`` (default), a metric name, a list of metric
                names or ``None``.
            with_biberpy: Whether to compute the biberpy dimensions.

        Returns:
            A one-row DataFrame of the selected metrics, columns sorted by name. Values are scaled when a scaler
            is set; scaling uses every metric, so unselected ones count as 0.

        Raises:
            ValueError: If a metric name is unknown.
        """
        selected_metrics = _normalize_metrics(metrics)
        text = clean_text(text)

        sp_object = self.tagger(text)
        scores: dict[str, float] = dict.fromkeys(METRICS, 0.0)
        computed: dict[Callable[[Doc], float], float] = {}  # "mattr" and "mtld" share the same function
        for metric in selected_metrics:
            function = METRICS[metric]
            if function not in computed:
                computed[function] = function(sp_object)
            scores[metric] = computed[function]

        biber_names = list(biberpy.dimnames.values())
        if with_biberpy:
            scores.update(biberpy.getbiberdims(text))
            selected_metrics = [*selected_metrics, *biber_names]
        else:
            scores.update(dict.fromkeys(biber_names, 0.0))

        columns = sorted(scores)
        metrics_scores = pd.DataFrame([[scores[column] for column in columns]], columns=columns, dtype=np.float64)
        if self.scaler is not None:
            metrics_scores = pd.DataFrame(self.scaler.transform(metrics_scores), columns=columns)
        return metrics_scores[sorted(selected_metrics)]

    def get_sp_object(self, text: str) -> Doc:
        """Return the spaCy document of the cleaned text."""
        return self.tagger(clean_text(text))

    def compute(self, text: str) -> np.ndarray:
        """Estimate the difficulty level of a text.

        Args:
            text: Text to compute.

        Returns:
            A one-element array with the estimated difficulty level.
        """
        return self.model.predict(self.get_metrics_scores(text))
