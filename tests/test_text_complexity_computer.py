from unittest import TestCase, main

import spacy
from spacy.cli import download
from spacy.tokens import Doc
import pandas as pd

from text_complexity_computer import TextComplexityComputer


class TestTCC(TestCase):
    def test_init(self):
        tcc = TextComplexityComputer()

    def test_with_minmax(self):
        tcc = TextComplexityComputer(scaler="MinMaxScaler")

    def test_get_sp_object(self):
        try:
            tagger = spacy.load("fr_core_news_sm")
        except OSError:
            print(download('fr_core_news_sm'))
            tagger = spacy.load("fr_core_news_sm")
        sp_object_test = tagger("Je suis une tortue.")

        tcc = TextComplexityComputer()
        sp_object = tcc.get_sp_object("Je suis une tortue.")

        self.assertEqual(type(sp_object), Doc)
        self.assertEqual(sp_object.text, sp_object_test.text)
        for token, token_test in zip(sp_object, sp_object_test):
            self.assertEqual(token.text, token_test.text)
            self.assertEqual(token.pos_, token_test.pos_)
            self.assertEqual(token.lemma_, token_test.lemma_)
            self.assertEqual(token.dep_, token_test.dep_)


if __name__ == "__main__":
    main()
