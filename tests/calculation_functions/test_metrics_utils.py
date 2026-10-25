from text_complexity_computer import TextComplexityComputer
from text_complexity_computer.calculation_functions import metrics_utils as mu


class TestGetNumWords:
    def test_given_text_then_get_num_words(self, tcc: TextComplexityComputer) -> None:
        # 8 words
        text = "Le cheval de feu montre le plus fort"
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_words(sp_object) == 8

    def test_given_text_with_punctuations_then_get_num_words(self, tcc: TextComplexityComputer) -> None:
        # 11 words without punctuations
        text = "Le cheval, de feu, montre: le plus fort; le plus beau."
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_words(sp_object) == 11

    def test_given_text_with_quotes_then_get_num_words(self, tcc: TextComplexityComputer) -> None:
        # 8 words without quotes
        text = "Le cheval de \"feu\" montre le plus 'fort'"
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_words(sp_object) == 8

    def test_given_text_when_min_size_is_set_then_get_num_words(self, tcc: TextComplexityComputer) -> None:
        # 4 words of more than 4 letters: 'cheval', 'montre', 'plus' and 'fort'
        text = "Le cheval de feu montre le plus fort."
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_words(sp_object, min_size=4) == 4

    def test_given_text_when_get_size_is_true_then_get_num_words(self, tcc: TextComplexityComputer) -> None:
        # 8 words for a total of 29 letters
        text = "Le cheval de feu montre le plus fort."
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_words_and_letters(sp_object) == (8, 29)

    def test_given_text_with_recurrences_when_get_size_is_true_then_get_num_words(
        self, tcc: TextComplexityComputer
    ) -> None:
        text = "Le lapin suis un autre lapin."
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_words_and_unique_words(sp_object) == (6, 5)


class TestGetNumSyllables:
    def test_given_text_then_get_num_syllabe(self, tcc: TextComplexityComputer) -> None:
        # 10 syllables
        text = "Le cheval de feu montre le plus fort"
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_syllables(sp_object) == 10

    def test_given_text_with_all_accents_then_get_num_syllabe(self, tcc: TextComplexityComputer) -> None:
        # 100 syllables (right on the money!)
        # source : https://10fastfingers.com/text/52387-Phrases-avec-lettres-avec-lettres-accents-et-ligatures
        text = (
            "Portez ce vieux whisky au juge blond qui fume sur son île intérieure, à côté de l'alcôve ovoïde, où "
            "les bûches se consument dans l'âtre, ce qui lui permet de penser à la caenogénèse de l'être dont "
            "il est question dans la cause ambiguë entendue à Moÿ, dans un capharnaüm qui, pense-t-il, diminue çà "
            "et là la qualité de son œuvre."
        )
        sp_object = tcc.get_sp_object(text)
        assert mu.get_num_syllables(sp_object) == 100


class TestSafeDivide:
    def test_given_ints_then_get_safe_divide(self) -> None:
        assert mu.safe_divide(1, 2) == 1 / 2

    def test_given_ints_when_division_by_zero_then_get_safe_divide(self) -> None:
        assert mu.safe_divide(1, 0) == 0.0

    def test_given_ints_when_returned_value_is_set_then_get_safe_divide(self) -> None:
        assert mu.safe_divide(1, 0, returned_value=1.0) == 1.0
