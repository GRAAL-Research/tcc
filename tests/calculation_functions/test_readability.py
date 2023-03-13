from unittest import TestCase, main

from text_complexity_computer.calculation_functions import readability as r
from text_complexity_computer import TextComplexityComputer

tcc = TextComplexityComputer()


class TestBingui(TestCase):
    def test_givenText_thenGetMetric(self):
        text = "Je, le cheval de feu, montre: le plus fort le plus brave (vous)."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(r.bingui(sp_object), 4)

    def test_givenTextWithSemicolon_thenGetMetric(self):
        text = "Je, le cheval de feu montre: le plus fort; le plus brave (vous)."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(r.bingui(sp_object), 4)


class TestFkEase(TestCase):
    def test_givenText_thenGetMetric(self):
        text = "Allons manger chez le voisin. Il sait faire de bons petits plats."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(r.fk_ease(sp_object), 94.995)

    def test_givenHardText_thenGetMetric(self):
        text = (
            "Je ronge les lapins. En outre le poivre en addition au sel permet de relever un je ne sais quoi lors "
            "de la ronge du rongeur qui agrémente une raideur certaine venant probablement de la rigidité du "
            "dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(r.fk_ease(sp_object), 48.2117307692308)


class TestKmFormula(TestCase):
    def test_givenText_thenGetMetric(self):
        text = "Allons manger chez le voisin. Il sait faire de bons petits plats."
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(r.km_formula(sp_object), 189.87)

    def test_givenHardText_thenGetMetric(self):
        text = (
            "Je ronge les lapins. En outre le poivre en addition au sel permet de relever un je ne sais quoi lors "
            "de la ronge du rongeur qui agrémente une raideur certaine venant probablement de la rigidité du "
            "dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        self.assertEqual(r.km_formula(sp_object), 140.1035)


if __name__ == "__main__":
    main()
