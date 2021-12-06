import pandas as pd

from .calculation_functions import readability as r, syntactic_complexity as sc, vocabulary_complexity as vc, \
    lexical_diversity as ld, biberpy as b
import fr_core_news_md
from typing import Union
import regex as re
from spacy.tokens import Doc
import os
import pickle
import numpy as np


def _clean_text(text: str) -> str:
    """Text cleaner
    Below is a comprehensive list of changes:
        - remove the bullet lists marker
        - special characters
        - add space between points
        - remove space multiplications

    Args:
        text (string): text that will be computed

    Returns:
        string: cleaned text
    """
    text = text.replace('_', ' ')
    text = text.replace('•', ' ')
    text = text.replace('○', ' ')
    text = re.sub(r'\[…\]', ' ', text)
    text = text.replace('…', ' . ')
    text = re.sub(r'- \d+ -', ' ', text)
    text = re.sub(r'\d+\)?\. ', ' ', text)
    text = re.sub(r'\(?\d+(\.\d+)*\)?\.? ', ' ', text)
    text = re.sub(r'\(?\w+\) ', ' ', text)
    text = re.sub(r'\.+(\w+)', '. \g<1>', text)
    text = re.sub(r'\s+', ' ', text)
    return text


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

    def __init__(self, scaler: Union[str, None] = "MinMaxScaler"):
        """
        Contstructor of TextComplexityComputer

        Args:
            scaler (Union[str, None]): chose the scaler between StandardScaler (by default), MinMaxScaler and none.
        """
        self.tagger = fr_core_news_md.load()
        self.tagger.max_length = 5000000

        path = os.path.abspath(os.path.dirname(__file__))
        # Get scaler
        if scaler:
            with open(os.path.join(path, scaler + ".pickle"), "rb") as rb:
                self.scaler = pickle.load(rb)
        else:
            print("!!WARNING!! You are running TextComplexityComputer without scaler (scaler=None). !!WARNING!!")
            self.scaler = None

        # Get Classifier
        with open(os.path.join(path, "model.pickle"), "rb") as rb:
            self.model = pickle.load(rb)

    def get_metrics_scores(self, text: str, metrics: Union[list, str, None] = 'all',
                           with_biberpy: bool = True) -> pd.DataFrame:
        """
        Getter of the metrics scores
            Details of metrics:
                - mls: Mean Lenght of Sentences
                - ps_30: Percentage of sentences longer than 30 words
                - nws_90: Lenght of the 90th percentile sentence
                - mlt: Mean Lenght of T-Unit
                - tu_s: T-Unit per Sentences
                - ctu_tu: Complex T-Unit per T-Unit
                - dc_c: Dependent clause per Clause
                - c_s: Clauses per Sentences
                - c_tu: Clauses per T\-Unit
                - cp_c: Coordinate phrase per Clause
                - cp_tu: Coordinate phrase per T\-Unit
                - pa: Percentage of words not in a reference list (Gougenheim list)
                - nlm: Nombre de Lettre par Mot
                - uni_gram_lem: Uni-gram model with Lexique3 on lemmas
                - msttr: Mean Sequential Type-Token Ration
                - mtld: Mesure of Textual Lexical Diversity
                - mtld_ma_bid: MTLD-Moving Average bidirectionnal
                - fk_ease: Flesch-Kincaid reading ease ('homemade')
                - textstat\'s fk_ease: Flesch-Kincaid reading ease (from textstat's librarie)
                - bingui: Ratio comma / sentence extended to other punctuations
                - km_score: Kandel and Moles formula
                - and others from biberpy module (see biberpy.py documentation for more)

        Args:
            text (string): text that will be computed
            metrics (Union[list, str, None], optional): list of metrics that will be computed (outside biberpy). By
            default, it will process them all.
            with_biberpy (bool, optional): process the biberpy's metrics (default at True).

        Returns:
            pd.DataFrame: the selected metrics scores
        """
        text = _clean_text(text)

        metrics_call = {'mls': sc.mls, 'ps_30': sc.ps_30, 'nws_90': sc.nws_90, 'mlt': sc.mlt, 'tu_s': sc.tu_s,
                        'ctu_tu': sc.ctu_tu, 'dc_c': sc.dc_c, 'c_s': sc.c_s, 'c_tu': sc.c_tu, 'cp_c': sc.cp_c,
                        'cp_tu': sc.cp_tu, 'pa': vc.pa, 'nlm': vc.nlm, 'uni_gram_lem': vc.uni_gram_lem,
                        'msttr': ld.msttr, 'mattr': ld.mtld, 'mtld': ld.mtld, 'fk_ease': r.fk_ease, 'bingui': r.bingui,
                        'km_score': r.km_formula}

        all_metrics = list(metrics_call.keys())

        if metrics is 'all':
            metrics = all_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif metrics is None:
            metrics = list()

        sp_object = self.tagger(text)
        metrics_scores = pd.DataFrame(np.zeros((1, len(all_metrics))), columns=all_metrics)
        for metric in metrics:
            metrics_scores[metric][0] = metrics_call[metric](sp_object)

        if with_biberpy:
            biber = b.getbiberdims(text)
            metrics_scores = metrics_scores.assign(**biber)
            metrics = [*metrics, *list(b.dimnames.values())]
        else:
            for metric in b.dimnames.values():
                metrics_scores[metric] = 0

        if self.scaler:
            scaled_metrics_scores = pd.DataFrame(self.scaler.transform(metrics_scores[sorted(metrics_scores.columns)]),
                                                 columns=sorted(metrics_scores.columns))
            return scaled_metrics_scores[sorted(metrics)]
        else:
            return metrics_scores[sorted(metrics)]

    def get_sp_object(self, text: str) -> Doc:
        """
        Getter of the spaCy object
        Args:
            text (string): text that will be computed

        Returns:
            spacy.tokens.doc.Doc: the spaCy object
        """
        return self.tagger(_clean_text(text))

    def compute(self, text: str):
        """
        Compute the text and evaluate the global difficulty level
        Args:
            text (string): text that will be computed

        Returns:
            int: estimation of the level of difficulty
        """
        return self.model.predict(self.get_metrics_scores(text))
