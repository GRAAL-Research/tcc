from text_complexity_computer import TextComplexityComputer
from text_complexity_computer.calculation_functions import readability as r


class TestFkEase:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        text = "Allons manger chez le voisin. Il sait faire de bons petits plats."
        sp_object = tcc.get_sp_object(text)
        assert r.fk_ease(sp_object) == 94.995

    def test_given_hard_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        text = (
            "Je ronge les lapins. En outre le poivre en addition au sel permet de relever un je ne sais quoi lors "
            "de la ronge du rongeur qui agrémente une raideur certaine venant probablement de la rigidité du "
            "dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        assert r.fk_ease(sp_object) == 48.2117307692308


class TestKmFormula:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        text = "Allons manger chez le voisin. Il sait faire de bons petits plats."
        sp_object = tcc.get_sp_object(text)
        assert r.km_formula(sp_object) == 189.87

    def test_given_hard_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        text = (
            "Je ronge les lapins. En outre le poivre en addition au sel permet de relever un je ne sais quoi lors "
            "de la ronge du rongeur qui agrémente une raideur certaine venant probablement de la rigidité du "
            "dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        assert r.km_formula(sp_object) == 140.1035
