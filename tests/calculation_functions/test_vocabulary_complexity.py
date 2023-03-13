from unittest import TestCase, main
from text_complexity_computer.calculation_functions import vocabulary_complexity as vc
from text_complexity_computer import TextComplexityComputer

tcc = TextComplexityComputer()


class TestPa(TestCase):
    def test_givenText_thenGetMetric(self):
        # 4 words one of which is not in Gougenheim list (sandale): pa = 1/4 = 0.25
        text = "je mange une sandale"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.pa(sp_object, language="fr"), 0.25)

    def test_givenTextWithCapital_thenGetMetric(self):
        # 4 words one of which is not in Gougenheim list (sandale): pa = 1/4 = 0.25
        text = "Je Mange une sandale"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.pa(sp_object, language="fr"), 0.25)


class TestNlm(TestCase):
    def test_givenText_thenGetMetric(self):
        # 3 words for 9 letters: nlm = 9/3 l= 3
        text = "mot non sot"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.nlm(sp_object), 3.0)

    def test_givenTextWithPunctuations_thenGetMetric(self):
        # 5 words for 15 letters (except punctuations): nlm = 15/5 = 3
        text = "mot, non sot, est bof!"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.nlm(sp_object), 3.0)

    def test_givenTextWithQuotes_thenGetMetric(self):
        # 5 words for 15 letters (except quotes): nlm = 15/5 = 3
        text = "mot non sot 'est' bof"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.nlm(sp_object), 3.0)


class TestNmm8(TestCase):
    def test_givenText_thenGetMetric(self):
        # 4 words, one of which has more than 8 letters: nmm_8 = 1/4 = 0.25
        text = "je mange une antilope"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.nmm_8(sp_object), 0.25)

    def test_givenTextWithPunctuations_thenGetMetric(self):
        # 4 words, one of which has more than 8 letters (except punctuations): nmm_8 = 1/4 = 0.25
        text = "Je, mange, une antilope!"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.nmm_8(sp_object), 0.25)

    def test_givenTextWithQuotes_thenGetMetric(self):
        # 4 words, one of which has more than 8 letters (except quotes): nmm_8 = 1/4 = 0.25
        text = "Je 'mange' une antilope"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.nmm_8(sp_object), 0.25)


class TestUniGramLem(TestCase):
    def test_givenText_thenGetMetric(self):
        # see vocabulary_complexity.unigram_lem for calculation
        text = "il mange une chaussure"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.uni_gram_lem(sp_object, language="fr"), -8.32673949170022)

    def test_givenTextWithZeroProbabilityWord_thenGetMetric(self):
        # 'abaisse' has a probability of 0.0
        text = "il abaisse une chaussure"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.uni_gram_lem(sp_object, language="fr"), -9.454065176893348)

    def test_givenTextWithUnknownWord_thenGetMetric(self):
        # 'sheeeesh' will not be reconised as a French word
        text = "sheeeesh c'est malade"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(vc.uni_gram_lem(sp_object, language="fr"), -7.500609988400598)


if __name__ == "__main__":
    main()
