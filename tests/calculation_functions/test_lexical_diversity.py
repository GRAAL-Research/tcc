from unittest import TestCase, main

from text_complexity_computer import TextComplexityComputer
from text_complexity_computer.calculation_functions import lexical_diversity as ld

tcc = TextComplexityComputer()


class TestTTR(TestCase):
    def test_givenText_thenGetMetric(self):
        # 5 words with 3 stopwords ("Ce", "est", "un") and a doubled word ("matin"): ttr = 1/2 = 0.5
        text = "Ce matin est un matin"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.ttr(sp_object), 0.5)

    def test_givenTextWithPunctuations_thenGetMetric(self):
        # 5 words with 3 stopwords ("Ce", "est", "un") and a doubled word ("matin"): ttr = 1/2 = 0.5
        # (without punctuations)
        text = "Ce matin, est: un (matin)."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.ttr(sp_object), 0.5)

    def test_givenTextWithQuotes_thenGetMetric(self):
        # 5 words with 3 stopwords ("Ce", "est", "un") and a doubled word ("matin"): ttr = 1/2 = 0.5 (without quotes)
        text = "Ce \"matin\" est un 'matin'"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.ttr(sp_object), 0.5)


class TestMSTTR(TestCase):
    # The segment size has been reduced to 3 to avoid having to compute and design long texts (at least >100 words).
    # The tests remain correct despite this modification.
    def test_givenText_thenGetMetric(self):
        # 6 words (without stopwords) with 1 in each segment that is repeated: msttr = (2/3+2/3)/2 = 2 / 3
        text = "une journée journée est forte plaisante lorsque le matin est matin."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.msttr(sp_object, segment_size=3), 2 / 3)

    def test_givenNotDivisibleText_thenGetMetric(self):
        # 7 words (without stopwords) forming at most 2 segments of 3 words (the last word is out) with 1 in each
        # segment that is repeated: msttr = (2/3+2/3)/2 = 2/3
        text = "une journée journée est forte plaisante lorsque le matin est matin erreur."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.msttr(sp_object, segment_size=3), 2 / 3)

    def test_givenTextWithPunctuations_thenGetMetric(self):
        # 6 words (without stopwords and punctuations) with 1 in each segment that is repeated:
        # msttr = (2/3+2/3)/2 = 2 / 3
        text = "une journée, journée: est forte plaisante! lorsque le matin; est matin."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.msttr(sp_object, segment_size=3), 2 / 3)

    def test_givenTextWithQuotes_thenGetMetric(self):
        # 6 words (without stopwords and quotes) with 1 in each segment that is repeated: msttr = (2/3+2/3)/2 = 2 / 3
        text = "une journée 'journée' est forte plaisante lorsque le \"matin\" est matin."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.msttr(sp_object, segment_size=3), 2 / 3)


class TestMATTR(TestCase):
    # The window size has been reduced to 3 to avoid having to compute long texts (at least >100 words).
    # The tests remain correct despite this modification.
    def test_givenText_thenGetMetric(self):
        # 6 words (without stopwords) with 2 repeated: mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée journée est forte plaisante lorsque le matin est matin."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.mattr(sp_object, window_size=3), 0.8333333333333333)

    def test_givenNotDivisibleText_thenGetMetric(self):
        # 7 words (without stopwords) with 2 repeated words and window size is not a multiple of 3:
        # mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée journée est forte plaisante lorsque le matin est matin erreur."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.mattr(sp_object, window_size=3), 0.8333333333333333)

    def test_givenTextWithPunctuations_thenGetMetric(self):
        # 6 words (without stopwords and punctuations) with 2 repeated: mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée, journée: est forte plaisante! lorsque le matin; est matin."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.mattr(sp_object, window_size=3), 0.8333333333333333)

    def test_givenTextWithQuotes_thenGetMetric(self):
        # # 6 words (without stopwords and quotes) with 2 repeated: mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée 'journée' est forte plaisante lorsque le \"matin\" est matin."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.mattr(sp_object, window_size=3), 0.8333333333333333)


class TestMTLD(TestCase):
    def test_givenText_thenGetMetric(self):
        text = (
            "Je ronge les lapins. En outre le poivre en addition au sel permet de relever un je ne sais quoi lors "
            "de la ronge du rongeur qui agrémente une raideur certaine venant probablement de la rigidité du "
            "dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(ld.mtld(sp_object), 63.00000000000002)


if __name__ == "__main__":
    main()
