from text_complexity_computer import TextComplexityComputer
from text_complexity_computer.calculation_functions import syntactic_complexity as sc


class TestMls:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 2 sentences of 4 words each: mls = 8/2 = 4
        text = "Je mange un arbre. Il fait vraiment beau."
        sp_object = tcc.get_sp_object(text)
        assert sc.mean_length_sentences(sp_object) == 4.0

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 2 sentences of 4 words each (except punctuations): mls = 8/2 = 4
        text = "Je mange, un arbre. Il fait, vraiment: beau."
        sp_object = tcc.get_sp_object(text)
        assert sc.mean_length_sentences(sp_object) == 4.0

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 4 words, one of which has more than 8 letters (except quotes): nmm_8 = 1/4 = 0.25
        text = "Je mange un \"arbre\". Il fait 'vraiment' beau."
        sp_object = tcc.get_sp_object(text)
        assert sc.mean_length_sentences(sp_object) == 4.0


class TestNws90:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # Here, the 90th percentile sentence is the third one. It has 5 words so: nws_90 = 5
        text = "Il ne peut aimer le rouge pourpre. Je mange un arbre vert. Il fait vraiment beau."
        sp_object = tcc.get_sp_object(text)
        assert sc.nws_90(sp_object) == 5.0

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # Here, the 90th percentile sentence is the third one. It has 5 words so: nws_90 = 5
        text = "Il ne peut, aimer, le rouge pourpre. Je mange un arbre vert! Il fait vraiment: beau."
        sp_object = tcc.get_sp_object(text)
        assert sc.nws_90(sp_object) == 5.0

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # Here, the 90th percentile sentence is the third one. It has 5 words so: nws_90 = 5
        text = "Il ne peut 'aimer' le rouge pourpre. Je mange un arbre vert. Il fait vraiment beau."
        sp_object = tcc.get_sp_object(text)
        assert sc.nws_90(sp_object) == 5.0


class TestPs30:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # The first sentence has 4 words (<30 words) and the second has 34 words (>30 words): ps_30 = 1/2 = 0.5
        text = (
            "Je ronge les lapins. En outre le poivre en addition au sel permet de relever un je ne sais quoi lors "
            "de la ronge du rongeur qui agrémente une raideur certaine venant probablement de la rigidité du "
            "dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        assert sc.ps_30(sp_object) == 0.5

    def test_given_text_with_punctuations_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # Both of the sentences have less than 30 words without the punctuations, but not with!
        text = (
            "Je ronge les lapins. Le poivre, en addition, au sel permet de: relever un je ne sais quoi, lors "
            "de la ronge, qui agrémente une raideur venant de la rigidité du dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        assert sc.ps_30(sp_object) == 0.0

    def test_given_text_with_quotes_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # Both of the sentences have less than 30 words without the quotes, but not with!
        text = (
            "Je ronge les lapins. Le poivre en \"addition\" au sel permet de 'relever' un je ne sais quoi lors "
            "de la ronge qui \"agrémente\" une 'raideur' venant de la rigidité du dit-lapin."
        )
        sp_object = tcc.get_sp_object(text)
        assert sc.ps_30(sp_object) == 0.0


# ==================== I - T-UNITS ==================== #
class TestGetTUnits:
    def test_given_text_then_get_t_units(self, tcc: TextComplexityComputer) -> None:
        # 2 coordinates -> 2 T-Units
        text = "Il joue au foot et il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_t_units(sp_object) == 2

    def test_given_text_with_double_nesting_then_get_t_units(self, tcc: TextComplexityComputer) -> None:
        # 1 matrix plus subordinate clause -> 1 T-Unit
        text = "Il joue au foot parce qu'il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_t_units(sp_object) == 1

    def test_given_text_with_triple_nesting_then_get_t_units(self, tcc: TextComplexityComputer) -> None:
        # 1 matrix plus subordinate clause + 1 coordinate -> 2 T-Unit
        text = "Il joue au foot parce qu'il aime le beau sport car il est lui-même beau."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_t_units(sp_object) == 2

    def test_given_text_with_nominal_conj_coor_then_get_t_units(self, tcc: TextComplexityComputer) -> None:
        # 2 nominal clauses -> 2 T-Units
        text = "Bonjour à toutes et à tous."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_t_units(sp_object) == 2

    def test_given_text_when_complex_count_is_true_then_get_complex_and_base_t_units(
        self, tcc: TextComplexityComputer
    ) -> None:
        # 2 coordinates -> 2 T-Units, 0 Complex
        text = "Il joue au foot et il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc._count_t_units(sp_object) == (2, 0)

    def test_given_text_with_triple_nesting_when_complex_count_is_true_then_get_complex_and_base_t_units(
        self, tcc: TextComplexityComputer
    ) -> None:
        # 1 matrix plus subordinate clause (complex) + 1 coordinate -> 2 T-Unit, 1 Complex
        text = "Il joue au foot parce qu'il aime le beau sport et qu'il est beau."
        sp_object = tcc.get_sp_object(text)
        assert sc._count_t_units(sp_object) == (2, 1)

    def test_given_text_with_dependent_clause_when_complex_count_is_true_then_get_complex_and_base_t_units(
        self, tcc: TextComplexityComputer
    ) -> None:
        # 1 matrix plus subordinate clause (complex) -> 1 T-Unit, 1 Complex
        text = "Il dit que tu aimes nager."
        sp_object = tcc.get_sp_object(text)
        assert sc._count_t_units(sp_object) == (1, 1)


class TestMlt:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 10 words for 2 T-Units : mlt = 10/2 = 5
        text = "Il joue au foot et il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc.mean_length_tunit(sp_object) == 5.0

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 12 words for 2 T-Units (without punctuations) : mlt = 12/2 = 6
        text = "Il joue: au foot, au tenis, et il aime le beau sport"
        sp_object = tcc.get_sp_object(text)
        assert sc.mean_length_tunit(sp_object) == 6.0


class TestTuS:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 2 sentences for 3 T-Units : tu_s = 3/2
        text = "Il joue au foot et il aime le beau sport. Quel joueur !"
        sp_object = tcc.get_sp_object(text)
        assert sc.tu_s(sp_object) == 3 / 2

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 2 sentences for 3 T-Units : tu_s = 3/2
        text = 'Il joue: au foot, au tenis, et il "aime" le beau sport. Quel (joueur) !'
        sp_object = tcc.get_sp_object(text)
        assert sc.tu_s(sp_object) == 3 / 2


class TestCtuTu:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 1 Complex T-Units for 2 T-Units : ctu_tu = 1/2
        text = "Il joue au foot parce qu'il aime le beau sport. Quel joueur !"
        sp_object = tcc.get_sp_object(text)
        assert sc.ctu_tu(sp_object) == 1 / 2

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 1 Complex T-Units for 2 T-Units : ctu_tu = 1/2
        text = 'Il joue: au foot, au tenis, parce qu\'il "aime" le beau sport. Quel (joueur) !'
        sp_object = tcc.get_sp_object(text)
        assert sc.ctu_tu(sp_object) == 1 / 2


# ====================================================== #


# ==================== II - CLAUSES ==================== #
class TestGetClauses:
    def test_given_text_then_get_clauses(self, tcc: TextComplexityComputer) -> None:
        # 1 matrix plus 1 subordinate clause -> 2 Clauses
        text = "Il joue au foot parce qu'il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_clauses(sp_object) == 2

    def test_given_text_with_double_nesting_then_get_clauses(self, tcc: TextComplexityComputer) -> None:
        # 1 matrix plus 1 subordinate clause + 1 coordinate -> 3 Clauses
        text = "Il joue au foot parce qu'il aime le beau sport et les sport c'est sa passion."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_clauses(sp_object) == 3

    def test_given_text_when_dependent_count_is_true_then_get_dependent_and_base_clauses(
        self, tcc: TextComplexityComputer
    ) -> None:
        # 1 matrix plus 1 subordinate clause -> 2 Clauses, 1 Dependent
        text = "Il joue au foot et il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc._count_clauses(sp_object) == (2, 1)


class TestDcC:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 1 matrix plus 1 subordinate clause, so 2 clauses one of which is dependent : dc_c = 1/2
        text = "Il joue au foot parce qu'il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc.dc_c(sp_object) == 1 / 2

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 3 clauses one of which is dependent : dc_c = 1/3
        text = 'Il joue: au foot, au hockey, parce qu\'il "aime" le beau sport. Quel (joueur) !'
        sp_object = tcc.get_sp_object(text)
        assert sc.dc_c(sp_object) == 1 / 3


class TestCS:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 3 clauses for 2 sentences : c_s = 3/2
        text = "Il joue au foot parce qu'il aime le beau sport. Quel joueur !"
        sp_object = tcc.get_sp_object(text)
        assert sc.c_s(sp_object) == 3 / 2

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 3 clauses for 2 sentences : c_s = 3/2
        text = 'Il joue: au foot, au hockey, parce qu\'il "aime" le beau sport. Quel (joueur) !'
        sp_object = tcc.get_sp_object(text)
        assert sc.c_s(sp_object) == 3 / 2


class TestCTu:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 2 clauses for 1 T-units : c_tu = 2/1 = 2
        text = "Il joue au foot parce qu'il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc.c_tu(sp_object) == 2

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 3 clauses for 2 T-units : c_tu = 3/2
        text = 'Il joue: au foot, au hockey, parce qu\'il "aime" le beau sport. Quel (joueur) !'
        sp_object = tcc.get_sp_object(text)
        assert sc.c_tu(sp_object) == 3 / 2


# ====================================================== #


# ==================== III - COORDINATE PHRASES ==================== #
class TestGetCoordinate:
    def test_given_text_then_get_coordinate(self, tcc: TextComplexityComputer) -> None:
        # 1 Coordinate
        text = "Il joue au foot car il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_coordinate(sp_object) == 1

    def test_given_text_with_double_nesting_then_get_coordinate(self, tcc: TextComplexityComputer) -> None:
        # 1 Coordinate
        text = "Il joue au foot parce qu'il aime le beau sport et les sport c'est sa passion."
        sp_object = tcc.get_sp_object(text)
        assert sc._get_coordinate(sp_object) == 1


class TestCpC:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 2 clauses for 1 coordinate : cp_c = 1/2
        text = "Il joue au foot car il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc.cp_c(sp_object) == 1 / 2

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 3 clauses for 1 coordinate : cp_c = 1/3
        text = 'Il joue: au foot, au hockey, car il "aime" le beau sport. Quel (joueur) !'
        sp_object = tcc.get_sp_object(text)
        assert sc.cp_c(sp_object) == 1 / 3


class TestCpTu:
    def test_given_text_then_get_metric(self, tcc: TextComplexityComputer) -> None:
        # 2 T-Units for 1 coordinate : cp_tu = 1/2
        text = "Il joue au foot car il aime le beau sport."
        sp_object = tcc.get_sp_object(text)
        assert sc.cp_c(sp_object) == 1 / 2

    def test_given_text_with_punctuations_then_get_metrics(self, tcc: TextComplexityComputer) -> None:
        # 3 T-Units for 1 coordinate : cp_tu = 1/3
        text = 'Il joue: au foot, au hockey, car il "aime" le beau sport. Quel (joueur) !'
        sp_object = tcc.get_sp_object(text)
        assert sc.cp_c(sp_object) == 1 / 3


# ====================================================== #
