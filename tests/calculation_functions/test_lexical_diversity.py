from text_complexity_computer import TextComplexityComputer
from text_complexity_computer.calculation_functions import lexical_diversity as ld


class TestTTR:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 5 words with 3 stopwords ("Ce", "est", "un") and a doubled word ("matin"): ttr = 1/2 = 0.5
        text = "Ce matin est un matin"
        sp_object = tcc.get_sp_object(text)
        assert ld.type_token_ratio(sp_object) == 0.5

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 5 words with 3 stopwords ("Ce", "est", "un") and a doubled word ("matin"): ttr = 1/2 = 0.5
        # (without punctuations)
        text = "Ce matin, est: un (matin)."
        sp_object = tcc.get_sp_object(text)
        assert ld.type_token_ratio(sp_object) == 0.5

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 5 words with 3 stopwords ("Ce", "est", "un") and a doubled word ("matin"): ttr = 1/2 = 0.5 (without quotes)
        text = "Ce \"matin\" est un 'matin'"
        sp_object = tcc.get_sp_object(text)
        assert ld.type_token_ratio(sp_object) == 0.5


class TestMSTTR:
    # The segment size has been reduced to 3 to avoid having to compute and design long texts (at least >100 words).
    # The tests remain correct despite this modification.
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 6 words (without stopwords) with 1 in each segment that is repeated: msttr = (2/3+2/3)/2 = 2 / 3
        text = "une journée journée est forte plaisante lorsque le matin est matin."
        sp_object = tcc.get_sp_object(text)
        assert ld.mean_sequential_ttr(sp_object, segment_size=3) == 2 / 3

    def test_given_not_divisible_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 7 words (without stopwords) forming at most 2 segments of 3 words (the last word is out) with 1 in each
        # segment that is repeated: msttr = (2/3+2/3)/2 = 2/3
        text = "une journée journée est forte plaisante lorsque le matin est matin erreur."
        sp_object = tcc.get_sp_object(text)
        assert ld.mean_sequential_ttr(sp_object, segment_size=3) == 2 / 3

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 6 words (without stopwords and punctuations) with 1 in each segment that is repeated:
        # msttr = (2/3+2/3)/2 = 2 / 3
        text = "une journée, journée: est forte plaisante! lorsque le matin; est matin."
        sp_object = tcc.get_sp_object(text)
        assert ld.mean_sequential_ttr(sp_object, segment_size=3) == 2 / 3

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 6 words (without stopwords and quotes) with 1 in each segment that is repeated: msttr = (2/3+2/3)/2 = 2 / 3
        text = "une journée 'journée' est forte plaisante lorsque le \"matin\" est matin."
        sp_object = tcc.get_sp_object(text)
        assert ld.mean_sequential_ttr(sp_object, segment_size=3) == 2 / 3


class TestMATTR:
    # The window size has been reduced to 3 to avoid having to compute long texts (at least >100 words).
    # The tests remain correct despite this modification.
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 6 words (without stopwords) with 2 repeated: mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée journée est forte plaisante lorsque le matin est matin."
        sp_object = tcc.get_sp_object(text)
        assert ld.moving_average_ttr(sp_object, window_size=3) == 0.8333333333333333

    def test_given_not_divisible_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 7 words (without stopwords) with 2 repeated words and window size is not a multiple of 3:
        # mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée journée est forte plaisante lorsque le matin est matin erreur."
        sp_object = tcc.get_sp_object(text)
        assert ld.moving_average_ttr(sp_object, window_size=3) == 0.8333333333333333

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 6 words (without stopwords and punctuations) with 2 repeated: mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée, journée: est forte plaisante! lorsque le matin; est matin."
        sp_object = tcc.get_sp_object(text)
        assert ld.moving_average_ttr(sp_object, window_size=3) == 0.8333333333333333

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # # 6 words (without stopwords and quotes) with 2 repeated: mattr = (2/3+1+1+2/3)/4 = 5/6
        text = "une journée 'journée' est forte plaisante lorsque le \"matin\" est matin."
        sp_object = tcc.get_sp_object(text)
        assert ld.moving_average_ttr(sp_object, window_size=3) == 0.8333333333333333


class TestMTLD:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        text = (
            "Je ronge les lapins. En outre le poivre en addition au sel permet de relever un je ne sais quoi lors "
            "de la ronge du rongeur qui agrémente une raideur certaine venant probablement de la rigidité du "
            "dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        assert ld.measure_textual_lexical_diversity(sp_object) == 63.00000000000002
