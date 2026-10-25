from text_complexity_computer import TextComplexityComputer
from text_complexity_computer.calculation_functions import vocabulary_complexity as vc


class TestPa:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 4 words one of which is not in Gougenheim list (sandale): pa = 1/4 = 0.25
        text = "je mange une sandale"
        sp_object = tcc.get_sp_object(text)
        assert vc.pa(sp_object, language="fr") == 0.25

    def test_given_text_with_capital_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 4 words one of which is not in Gougenheim list (sandale): pa = 1/4 = 0.25
        text = "Je Mange une sandale"
        sp_object = tcc.get_sp_object(text)
        assert vc.pa(sp_object, language="fr") == 0.25


class TestNlm:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 3 words for 9 letters: nlm = 9/3 l= 3
        text = "mot non sot"
        sp_object = tcc.get_sp_object(text)
        assert vc.nlm(sp_object) == 3.0

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 5 words for 15 letters (except punctuations): nlm = 15/5 = 3
        text = "mot, non sot, est bof!"
        sp_object = tcc.get_sp_object(text)
        assert vc.nlm(sp_object) == 3.0

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 5 words for 15 letters (except quotes): nlm = 15/5 = 3
        text = "mot non sot 'est' bof"
        sp_object = tcc.get_sp_object(text)
        assert vc.nlm(sp_object) == 3.0


class TestNmm8:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 4 words, one of which has more than 8 letters: nmm_8 = 1/4 = 0.25
        text = "je mange une antilope"
        sp_object = tcc.get_sp_object(text)
        assert vc.nmm_8(sp_object) == 0.25

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 4 words, one of which has more than 8 letters (except punctuations): nmm_8 = 1/4 = 0.25
        text = "Je, mange, une antilope!"
        sp_object = tcc.get_sp_object(text)
        assert vc.nmm_8(sp_object) == 0.25

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 4 words, one of which has more than 8 letters (except quotes): nmm_8 = 1/4 = 0.25
        text = "Je 'mange' une antilope"
        sp_object = tcc.get_sp_object(text)
        assert vc.nmm_8(sp_object) == 0.25


class TestUniGramLem:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # see vocabulary_complexity.unigram_lem for calculation
        text = "il mange une chaussure"
        sp_object = tcc.get_sp_object(text)
        assert vc.uni_gram_lem(sp_object, language="fr") == -8.32673949170022

    def test_given_text_with_zero_probability_word_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 'abaisse' has a probability of 0.0
        text = "il abaisse une chaussure"
        sp_object = tcc.get_sp_object(text)
        assert vc.uni_gram_lem(sp_object, language="fr") == -9.454065176893348

    def test_given_text_with_unknown_word_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 'sheeeesh' will not be reconised as a French word
        text = "sheeeesh c'est malade"
        sp_object = tcc.get_sp_object(text)
        assert vc.uni_gram_lem(sp_object, language="fr") == -7.500609988400598
