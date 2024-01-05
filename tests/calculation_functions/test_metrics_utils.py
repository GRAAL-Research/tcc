from unittest import TestCase, main

from text_complexity_computer.calculation_functions import metrics_utils as mu
from text_complexity_computer import TextComplexityComputer

tcc = TextComplexityComputer()


class TestGetNumWords(TestCase):
    def test_givenText_thenGetNumWords(self):
        # 8 words
        text = "Le cheval de feu montre le plus fort"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_words(sp_object), 8)

    def test_givenTextWithPunctuations_thenGetNumWords(self):
        # 11 words without punctuations
        text = "Le cheval, de feu, montre: le plus fort; le plus beau."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_words(sp_object), 11)

    def test_givenTextWithQuotes_thenGetNumWords(self):
        # 8 words without quotes
        text = "Le cheval de \"feu\" montre le plus 'fort'"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_words(sp_object), 8)

    def test_givenText_whenMinSizeIsSet_thenGetNumWords(self):
        # 4 words of more than 4 letters: 'cheval', 'montre', 'plus' and 'fort'
        text = "Le cheval de feu montre le plus fort."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_words(sp_object, min_size=4), 4)

    def test_givenText_whenGetSizeIsTrue_thenGetNumWords(self):
        # 8 words for a total of 29 letters
        text = "Le cheval de feu montre le plus fort."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_words(sp_object, get_size=True), (8, 29))

    def test_givenTextWithRecurrences_whenGetSizeIsTrue_thenGetNumWords(self):
        text = "Le lapin suis un autre lapin."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_words(sp_object, get_unique=True), (6, 5))


class TestGetNumSyllables(TestCase):
    def test_givenText_thenGetNumSyllabe(self):
        # 10 syllables
        text = "Le cheval de feu montre le plus fort"
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_syllables(sp_object), 10)

    def test_givenTextWithAllAccents_thenGetNumSyllabe(self):
        # 100 syllables (right on the money!)
        # source : https://10fastfingers.com/text/52387-Phrases-avec-lettres-avec-lettres-accents-et-ligatures
        text = (
            "Portez ce vieux whisky au juge blond qui fume sur son île intérieure, à côté de l'alcôve ovoïde, où "
            "les bûches se consument dans l'âtre, ce qui lui permet de penser à la caenogénèse de l'être dont "
            "il est question dans la cause ambiguë entendue à Moÿ, dans un capharnaüm qui, pense-t-il, diminue çà "
            "et là la qualité de son œuvre."
        )
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(mu.get_num_syllables(sp_object), 100)


class TestSafeDivide(TestCase):
    def test_givenInts_thenGetSafeDivide(self):
        self.assertEqual(mu.safe_divide(1, 2), 1 / 2)

    def test_givenInts_whenDivisionByZero_thenGetSafeDivide(self):
        self.assertEqual(mu.safe_divide(1, 0), 0.0)

    def test_givenInts_whenReturnedValueIsSet_thenGetSafeDivide(self):
        self.assertEqual(mu.safe_divide(1, 0, returned_value=1.0), 1.0)


if __name__ == "__main__":
    main()
