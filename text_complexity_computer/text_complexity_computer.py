import os
import pickle
import warnings
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import spacy
from spacy.cli import download
from spacy.tokens import Doc

from .calculation_functions import biberpy, vocabulary_complexity
from .calculation_functions import (
    readability,
    syntactic_complexity,
    lexical_diversity,
)
from .tools import clean_text, read_word_lists, read_num_list


class TextComplexityComputer:
    """
    TextComplexityComputer class

    Attributes:
        scaler (Union[str, None]): chose the scaler between MinMaxScaler (by default), StandardScaler and none.

    Methods:
        get_metrics_scores(text: str, metrics: Union[list, str, None] = 'all', with_biberpy: bool = True): Getter
    of the metrics scores
        get_sp_object(text: str): Getter of the spaCy object
        compute(text: str): Compute the text and evaluate the global difficulty level
    """

    def __init__(self, language: str = "fr", scaler: Union[str, None] = "MinMaxScaler", verbosity: int = 1):
        """
        Constructor of TextComplexityComputer

        Args:
            language (str): The language to use, either `'fr'` or `'en'`. By default, `'fr'`.
            scaler (Union[str, None]): chose the scaler between StandardScaler (by default), MinMaxScaler and none.
        """
        self.language = language
        if self.language == "fr":
            model_name = "fr_core_news_sm"
            try:
                self.tagger = spacy.load(model_name)
            except OSError:
                print(download(model_name))
                self.tagger = spacy.load(model_name)

        elif self.language == "en":
            model_name = "en_core_news_md"
            try:
                self.tagger = spacy.load(model_name)
            except OSError:
                print(download(model_name))
                self.tagger = spacy.load(model_name)
        else:
            raise ValueError(f"language can be 'fr' or 'en', not {self.language}")

        biberpy.language = self.language

        self.word_lists, self.mwe_list = read_word_lists(
            os.path.join(os.path.dirname(__file__), 'resources', self.language, f"{self.language}.properties"),
            verbosity=verbosity,
        )
        self.tag_list = read_num_list(
            os.path.join(os.path.dirname(__file__), 'resources', self.language, f"{self.language}.tag.num"),
        )

        biberpy.word_lists = self.word_lists
        biberpy.mwe_list = self.mwe_list

        biberpy.tag_list = self.tag_list

        self.tagger.max_length = 5000000

        if scaler:
            with open(
                os.path.join(os.path.dirname(__file__), 'resources', self.language, f"{self.language}_{scaler}.pickle"),
                "rb",
            ) as file:
                self.scaler = pickle.load(file)
        else:
            warnings.warn("You are running TextComplexityComputer without scaler (scaler=None).")
            self.scaler = None

        # Get Classifier
        with open(
            os.path.join(os.path.dirname(__file__), 'resources', self.language, f"{self.language}_model.pickle"), "rb"
        ) as rb:
            self.model = pickle.load(rb)

    def get_metrics_scores(
        self,
        text: str,
        metrics: Union[list, str, None] = "all",
        with_biberpy: bool = True,
    ) -> pd.DataFrame:
        """
        Getter of the metrics scores
            Details of metrics:
                - mls: Mean Length of Sentences
                - ps_30: Percentage of sentences longer than 30 words
                - nws_90: Length of the 90th percentile sentence
                - mlt: Mean Length of T-Unit
                - tu_s: T-Unit per Sentences
                - ctu_tu: Complex T-Unit per T-Unit
                - dc_c: Dependent clause per Clause
                - c_s: Clauses per Sentences
                - c_tu: Clauses per T-Unit
                - cp_c: Coordinate phrase per Clause
                - cp_tu: Coordinate phrase per T-Unit
                - pa: Percentage of words not in a reference list (Gougenheim list)
                - nlm: Number of letter per word
                - uni_gram_lem: Uni-gram model with Lexique3 on lemmas
                - msttr: Mean Sequential Type-Token Ration
                - mtld: Measure of Textual Lexical Diversity
                - mtld_ma_bid: MTLD-Moving Average bidirectionnal
                - fk_ease: Flesch-Kincaid reading ease ('homemade')
                - textstat\'s fk_ease: Flesch-Kincaid reading ease (from textstat's library)
                - Bingui: Ratio comma / sentence extended to other punctuations
                - km_score: Kandel and Moles formula
                - and others from biberpy module (see biberpy.py documentation for more).

        Args:
            text (string): text that will be computed
            metrics (Union[list, str, None], optional): list of metrics that will be computed (outside biberpy). By
            default, it will process them all.
            with_biberpy (bool, optional): process the biberpy's metrics (default at True).

        Returns:
            pd.DataFrame: the selected metrics scores
        """
        text = clean_text(text)

        metrics_call = {
            "mls": syntactic_complexity.mls,
            "ps_30": syntactic_complexity.ps_30,
            "nws_90": syntactic_complexity.nws_90,
            "mlt": syntactic_complexity.mlt,
            "tu_s": syntactic_complexity.tu_s,
            "ctu_tu": syntactic_complexity.ctu_tu,
            "dc_c": syntactic_complexity.dc_c,
            "c_s": syntactic_complexity.c_s,
            "c_tu": syntactic_complexity.c_tu,
            "cp_c": syntactic_complexity.cp_c,
            "cp_tu": syntactic_complexity.cp_tu,
            "pa": partial(vocabulary_complexity.pa, language=self.language),
            "nlm": vocabulary_complexity.nlm,
            "uni_gram_lem": partial(vocabulary_complexity.uni_gram_lem, language=self.language),
            "msttr": lexical_diversity.msttr,
            "mattr": lexical_diversity.mtld,
            "mtld": lexical_diversity.mtld,
            "fk_ease": readability.fk_ease,
            "bingui": readability.bingui,
            "km_score": readability.km_formula,
        }

        all_metrics = list(metrics_call.keys())

        if metrics == "all":
            metrics = all_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif metrics is None:
            metrics = []

        sp_object = self.tagger(text)
        metrics_scores = pd.DataFrame(np.zeros((1, len(all_metrics))), columns=all_metrics)
        for metric in metrics:
            metrics_scores[metric][0] = metrics_call[metric](sp_object)

        if with_biberpy:
            biber = biberpy.getbiberdims(text)
            metrics_scores = metrics_scores.assign(**biber)
            metrics = [*metrics, *list(biberpy.dimnames.values())]
        else:
            for metric in biberpy.dimnames.values():
                metrics_scores[metric] = 0

        if self.scaler:
            scaled_metrics_scores = pd.DataFrame(
                self.scaler.transform(metrics_scores[sorted(metrics_scores.columns)]),
                columns=sorted(metrics_scores.columns),
            )
            metrics_scores = scaled_metrics_scores
        return metrics_scores[sorted(metrics)]

    def get_sp_object(self, text: str) -> Doc:
        """
        Getter of the spaCy object
        Args:
            text (string): text that will be computed

        Returns:
            spacy.tokens.doc.Doc: the spaCy object
        """
        return self.tagger(clean_text(text))

    def compute(self, text: str):
        """
        Compute the text and evaluate the global difficulty level
        Args:
            text (string): text that will be computed

        Returns:
            int: estimation of the level of difficulty
        """
        return self.model.predict(self.get_metrics_scores(text))
