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

    def test_get_metrics_scores(self):
        tcc = TextComplexityComputer()
        metrics_scores = tcc.get_metrics_scores("Je mange. Tu mange du pain. Il mange le poivre du facteur.")
        metrics_scores_test = pd.DataFrame(
            [
                [
                    -0.48674180443361265,
                    -0.26782901938580217,
                    11.324103789197943,
                    -2.1759589353140956,
                    -2.2065514148367624,
                    1.4373265773969601,
                    -1.7569960303243464,
                    -0.648418455994531,
                    -1.0889428603387208,
                    -2.3466365707651153,
                    -1.3421827813898417,
                    -0.7380330751884534,
                    -0.7398272776796808,
                    -2.088560044333416,
                    -0.4068659006582878,
                    -0.866394764441893,
                    -1.8636901995829833,
                    -0.7610047765922021,
                    -2.4071973070152897,
                    -2.420369179619322,
                    -3.1162982640444303,
                    -4.695277358851864,
                    -0.4202732279920149,
                    0.0,
                    -1.2088880485580447,
                    -0.5337148036456381,
                    1.4759803583847844,
                    -0.5303527261238863,
                    14.396828326620568,
                    -1.3905868882930263,
                    2.6650142381702637,
                    -0.9255749532357956,
                    -0.6988758071638023,
                    -2.0837736126627466,
                    -7.494633118752427,
                    -0.9255749532357956,
                    -0.7480616156684424,
                    -3.932549139551214,
                    -1.2011055151607553,
                    -0.8374346295677407,
                    -0.7644231038182376,
                    -3.3878422337695384,
                    -2.120166734671064,
                    -0.27336636492089206,
                    -0.562462316707151,
                    -1.1491897003860712,
                    -0.7006732526531749,
                    -5.163517444035459,
                    8.149634790519626,
                    -1.2441060017747305,
                    -0.7880258865483534,
                    -0.8032414147671776,
                    -0.47223761623031313,
                    -0.5418174952958014,
                    -0.33513695960812295,
                    -0.7926236172260314,
                    -0.47684984367111133,
                    -1.5045530361560102,
                    -0.540867926164624,
                    -0.6863526604325139,
                    -1.8305418105622941,
                    -3.83514144277809,
                ]
            ],
            columns=[
                "1persProns",
                "2persProns",
                "3persProns",
                "ADV",
                "Nouns",
                "TTR",
                "WHclauses",
                "amplifiers",
                "analNegn",
                "attrAdj",
                "beAsMain",
                "bingui",
                "c_s",
                "c_tu",
                "causative",
                "conditional",
                "conjuncts",
                "contractions",
                "cp_c",
                "cp_tu",
                "ctu_tu",
                "dc_c",
                "demonstrProns",
                "discoursePart",
                "doAsProVerb",
                "downtoners",
                "fk_ease",
                "generalEmphatics",
                "impersProns",
                "indefProns",
                "km_score",
                "mattr",
                "mls",
                "mlt",
                "msttr",
                "mtld",
                "necessModals",
                "nlm",
                "nominalizations",
                "nws_90",
                "otherSubord",
                "pa",
                "pastVerbs",
                "piedPiping",
                "placeAdverbials",
                "possibModals",
                "predicModals",
                "preposn",
                "presVerbs",
                "privateVerbs",
                "ps_30",
                "publicVerbs",
                "seemappear",
                "sncRelatives",
                "strandedPrep",
                "suasiveVerbs",
                "syntNegn",
                "thatDeletion",
                "timeAdverbials",
                "tu_s",
                "whQuestions",
                "wordLength",
            ],
        )

        for metrics, metrics_test in zip(metrics_scores.to_numpy()[0], metrics_scores_test.to_numpy()[0]):
            self.assertEqual(metrics_test, metrics)
        self.assertEqual(metrics_scores_test.columns.to_list(), metrics_scores.columns.to_list())
        self.assertTrue(metrics_scores.equals(metrics_scores_test))

    def test_get_metrics_scores_empty(self):
        tcc = TextComplexityComputer()
        metrics_scores = tcc.get_metrics_scores("")
        metrics_scores_test = pd.DataFrame(
            [
                [
                    -0.48674180443361265,
                    -0.26782901938580217,
                    -0.9838425778606268,
                    -2.1759589353140956,
                    -5.159486711753425,
                    -5.637458198642641,
                    -1.7569960303243464,
                    -0.648418455994531,
                    -1.0889428603387208,
                    -2.3466365707651153,
                    -1.3421827813898417,
                    -0.7380330751884534,
                    -0.9957210433373782,
                    -5.360725045487416,
                    -0.4068659006582878,
                    -0.866394764441893,
                    -1.8636901995829833,
                    -0.7610047765922021,
                    -2.4071973070152897,
                    -2.420369179619322,
                    -3.1162982640444303,
                    -4.695277358851864,
                    -0.4202732279920149,
                    0.0,
                    -1.2088880485580447,
                    -0.5337148036456381,
                    -0.8741893974300999,
                    -0.5303527261238863,
                    -0.9872279147821489,
                    -1.3905868882930263,
                    3.0368630912997157,
                    -0.9392576655130533,
                    -0.8131231085916424,
                    -3.1897745675031635,
                    -7.494633118752427,
                    -0.9392576655130533,
                    -0.7480616156684424,
                    -17.484353814981173,
                    -1.2011055151607553,
                    -1.028478199000304,
                    -0.7644231038182376,
                    -6.328675181838428,
                    -2.120166734671064,
                    -0.27336636492089206,
                    -0.562462316707151,
                    -1.1491897003860712,
                    -0.7006732526531749,
                    -5.163517444035459,
                    -1.9089414151388668,
                    -1.2441060017747305,
                    -0.7880258865483534,
                    -0.8032414147671776,
                    -0.47223761623031313,
                    -0.5418174952958014,
                    -0.33513695960812295,
                    -0.7926236172260314,
                    -0.47684984367111133,
                    -1.5045530361560102,
                    -0.540867926164624,
                    -1.2209407542328983,
                    -1.8305418105622941,
                    -16.732855879043168,
                ]
            ],
            columns=[
                "1persProns",
                "2persProns",
                "3persProns",
                "ADV",
                "Nouns",
                "TTR",
                "WHclauses",
                "amplifiers",
                "analNegn",
                "attrAdj",
                "beAsMain",
                "bingui",
                "c_s",
                "c_tu",
                "causative",
                "conditional",
                "conjuncts",
                "contractions",
                "cp_c",
                "cp_tu",
                "ctu_tu",
                "dc_c",
                "demonstrProns",
                "discoursePart",
                "doAsProVerb",
                "downtoners",
                "fk_ease",
                "generalEmphatics",
                "impersProns",
                "indefProns",
                "km_score",
                "mattr",
                "mls",
                "mlt",
                "msttr",
                "mtld",
                "necessModals",
                "nlm",
                "nominalizations",
                "nws_90",
                "otherSubord",
                "pa",
                "pastVerbs",
                "piedPiping",
                "placeAdverbials",
                "possibModals",
                "predicModals",
                "preposn",
                "presVerbs",
                "privateVerbs",
                "ps_30",
                "publicVerbs",
                "seemappear",
                "sncRelatives",
                "strandedPrep",
                "suasiveVerbs",
                "syntNegn",
                "thatDeletion",
                "timeAdverbials",
                "tu_s",
                "whQuestions",
                "wordLength",
            ],
        )

        for metrics, metrics_test in zip(metrics_scores.to_numpy()[0], metrics_scores_test.to_numpy()[0]):
            self.assertEqual(metrics_test, metrics)
        self.assertEqual(metrics_scores_test.columns.to_list(), metrics_scores.columns.to_list())
        self.assertTrue(metrics_scores.equals(metrics_scores_test))

    def test_without_biberpy(self):
        tcc = TextComplexityComputer()
        metrics_scores = tcc.get_metrics_scores(
            "Je mange. Tu mange du pain. Il mange le poivre du facteur.",
            with_biberpy=False,
        )
        metrics_scores_test = pd.DataFrame(
            [
                [
                    -0.7380330751884534,
                    -0.7398272776796808,
                    -2.088560044333416,
                    -2.4071973070152897,
                    -2.420369179619322,
                    -3.1162982640444303,
                    -4.695277358851864,
                    1.4759803583847844,
                    2.6650142381702637,
                    -0.9255749532357956,
                    -0.6988758071638023,
                    -2.0837736126627466,
                    -7.494633118752427,
                    -0.9255749532357956,
                    -3.932549139551214,
                    -0.8374346295677407,
                    -3.3878422337695384,
                    -0.7880258865483534,
                    -0.6863526604325139,
                ]
            ],
            columns=[
                "bingui",
                "c_s",
                "c_tu",
                "cp_c",
                "cp_tu",
                "ctu_tu",
                "dc_c",
                "fk_ease",
                "km_score",
                "mattr",
                "mls",
                "mlt",
                "msttr",
                "mtld",
                "nlm",
                "nws_90",
                "pa",
                "ps_30",
                "tu_s",
            ],
        )
        for metrics, metrics_test in zip(metrics_scores.to_numpy()[0], metrics_scores_test.to_numpy()[0]):
            self.assertEqual(metrics_test, metrics)
        self.assertEqual(metrics_scores_test.columns.to_list(), metrics_scores.columns.to_list())
        self.assertTrue(metrics_scores.equals(metrics_scores_test))

    def test_get_one_metric(self):
        tcc = TextComplexityComputer()
        metrics_scores = tcc.get_metrics_scores("", metrics="mls", with_biberpy=False)
        metrics_scores_test = pd.DataFrame([[-0.8131231085916424]], columns=["mls"])

        for metrics, metrics_test in zip(metrics_scores.to_numpy()[0], metrics_scores_test.to_numpy()[0]):
            self.assertEqual(metrics_test, metrics)
        self.assertEqual(metrics_scores_test.columns.to_list(), metrics_scores.columns.to_list())
        self.assertTrue(metrics_scores.equals(metrics_scores_test))

    def test_get_multiple_metrics(self):
        tcc = TextComplexityComputer()
        metrics_scores = tcc.get_metrics_scores("", metrics=["mls", "mattr", "pa"], with_biberpy=False)
        metrics_scores_test = pd.DataFrame(
            [[-0.9392576655130533, -0.8131231085916424, -6.328675181838428]],
            columns=["mattr", "mls", "pa"],
        )

        for metrics, metrics_test in zip(metrics_scores.to_numpy()[0], metrics_scores_test.to_numpy()[0]):
            self.assertEqual(metrics_test, metrics)
        self.assertEqual(metrics_scores_test.columns.to_list(), metrics_scores.columns.to_list())
        self.assertTrue(metrics_scores.equals(metrics_scores_test))

    def test_get_only_biberpy(self):
        tcc = TextComplexityComputer()
        metrics_scores = tcc.get_metrics_scores(
            "Je mange. Tu mange du pain. Il mange le poivre du facteur.", metrics=None
        )
        metrics_scores_test = pd.DataFrame(
            [
                [
                    -0.48674180443361265,
                    -0.26782901938580217,
                    11.324103789197943,
                    -2.1759589353140956,
                    -2.2065514148367624,
                    1.4373265773969601,
                    -1.7569960303243464,
                    -0.648418455994531,
                    -1.0889428603387208,
                    -2.3466365707651153,
                    -1.3421827813898417,
                    -0.4068659006582878,
                    -0.866394764441893,
                    -1.8636901995829833,
                    -0.7610047765922021,
                    -0.4202732279920149,
                    0.0,
                    -1.2088880485580447,
                    -0.5337148036456381,
                    -0.5303527261238863,
                    14.396828326620568,
                    -1.3905868882930263,
                    -0.7480616156684424,
                    -1.2011055151607553,
                    -0.7644231038182376,
                    -2.120166734671064,
                    -0.27336636492089206,
                    -0.562462316707151,
                    -1.1491897003860712,
                    -0.7006732526531749,
                    -5.163517444035459,
                    8.149634790519626,
                    -1.2441060017747305,
                    -0.8032414147671776,
                    -0.47223761623031313,
                    -0.5418174952958014,
                    -0.33513695960812295,
                    -0.7926236172260314,
                    -0.47684984367111133,
                    -1.5045530361560102,
                    -0.540867926164624,
                    -1.8305418105622941,
                    -3.83514144277809,
                ]
            ],
            columns=[
                "1persProns",
                "2persProns",
                "3persProns",
                "ADV",
                "Nouns",
                "TTR",
                "WHclauses",
                "amplifiers",
                "analNegn",
                "attrAdj",
                "beAsMain",
                "causative",
                "conditional",
                "conjuncts",
                "contractions",
                "demonstrProns",
                "discoursePart",
                "doAsProVerb",
                "downtoners",
                "generalEmphatics",
                "impersProns",
                "indefProns",
                "necessModals",
                "nominalizations",
                "otherSubord",
                "pastVerbs",
                "piedPiping",
                "placeAdverbials",
                "possibModals",
                "predicModals",
                "preposn",
                "presVerbs",
                "privateVerbs",
                "publicVerbs",
                "seemappear",
                "sncRelatives",
                "strandedPrep",
                "suasiveVerbs",
                "syntNegn",
                "thatDeletion",
                "timeAdverbials",
                "whQuestions",
                "wordLength",
            ],
        )

        for metrics, metrics_test in zip(metrics_scores.to_numpy()[0], metrics_scores_test.to_numpy()[0]):
            self.assertEqual(metrics_test, metrics)
        self.assertEqual(metrics_scores_test.columns.to_list(), metrics_scores.columns.to_list())
        self.assertTrue(metrics_scores.equals(metrics_scores_test))

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

    def test_compute(self):
        tcc = TextComplexityComputer()
        result = tcc.compute(
            """Des roseaux de cristal bruissent sur la rive. Du bout des doigts, Lisa disperse les bleus. Sa paume 
            ouverte fond les pastels, les étend dans le ciel puis les laisse glisser en taches confuses, dans 
            l'immensité du lac. Des verts audacieux pour le haut des talus et des vagues diaprées là où l'herbe se 
            couche. Ses ongles démêlent l'espace. Sur le côté, un amandier par petites touches crémeuses. Un chemin de 
            traverse baigne sa main d'une lumière orange. Autour du lac, le soleil ensemence déjà les prés. On dirait un 
            champ de blé dont les gerbes fauchées plissent le paysage. Puis, un rayon céleste renverse le cours de 
            l'onde. mÀ petits coups précis, du bout de l'index ou de l'auriculaire replié, Lisa invite le brun à se 
            fondre sous les arbres. Déjà, un buisson s'affirme dans la transparence de l'eau. Une fois encore, elle 
            écoute l'étrange musique qui conduit ses rêves ; cymbales, flûtes et tambourins multiplient les tons. 
            Le chant d'un violon effile les dernières brumes, et le jour s'épanouit. Alors, sur un coin de ciel, Lisa 
            promène une dernière fois ses doigts, pour que s'y découpe, encore incertaine, la voile blanche d'un cerf 
            volant. """
        )
        self.assertEqual(result, [3])

    def test_compute_hard(self):
        tcc = TextComplexityComputer()
        text = (
            "Cet avenant prévoit les asd agerfaserg asrg ar ag rea ger are ag er aer AE RGA ER GAER "
            "GAERGAEDRSG ARE GA ga ardg szdfgsdrgzdfgzdrgzdfgzdsrgzdfgzdrgzdfgzdrgzdfg conditions et "
            "restrictions imposées par l’Autorité des marchés financiers (l’« Autorité ») en vertu de "
            "l’article de la Loi concernant le transport rémunéré de personnes par automobile, RLRQ, "
            "c. T- Le « répondant » s’entend du répondant d’un système de transport autorisé conformément "
            "au chapitre III de la Loi concernant le transport rémunéré de personnes par automobile. "
            "L’expression « chauffeur inscrit » s’entend de tout chauffeur inscrit auprès du répondant "
            "au sens de la Loi concernant le transport rémunéré de personnes par automobile. "
            "L’expression « police d’assurance personnelle » fait référence au contrat d’assurance "
            "de responsabilité, en vertu de l’article de la Loi sur l’assurance automobile, "
            "qui assure le véhicule utilisé par un chauffeur inscrit en dehors de la période "
            "de couverture du présent contrat d’assurance. Le contrat d’assurance s’applique à "
            "partir du moment où un chauffeur inscrit se rend disponible pour effectuer du "
            "transport rémunéré de personnes dans le cadre du système de transport du répondant "
            "jusqu’à ce qu’il cesse d’être ainsi disponible (la « période de couverture »). "
            "octobre Par exemple, le contrat d’assurance s’applique à partir du moment où un chauffeur "
            "inscrit se connecte au moyen technologique utilisé par le répondant pour répartir les demandes"
            " de courses, tel qu’une application mobile, jusqu’à ce que le chauffeur inscrit se déconnecte."
            " L’assuré désigné est : le répondant, chaque chauffeur inscrit et, dans le cas où un chauffeur "
            "inscrit utilise un véhicule dont il n’est pas propriétaire pour effectuer du transport rémunéré "
            "de personnes dans le cadre du système de transport du répondant, le propriétaire de ce véhicule. "
            "Caractéristiques du véhicule désigné : les automobiles utilisées par les chauffeurs inscrits pour "
            "effectuer du transport rémunéré de personnes. Créancier qui a droit aux indemnités du chapitre B, "
            "selon son intérêt : le créancier qui, au jour du sinistre, a droit aux indemnités du chapitre B en "
            "vertu de la police d’assurance personnelle assurant le véhicule utilisé par le chauffeur inscrit. "
            "Conformément à l’article de la Loi concernant le transport rémunéré de personnes par automobile, "
            "les dispositions du titre III de la Loi sur l’assurance automobile qui visent le propriétaire "
            "s’appliquent au répondant avec les adaptations nécessaires. Cette règle a pour effet, entre autres, "
            "de faire intervenir le présent contrat d’assurance en priorité pendant la période de couverture. "
            "Le contrat d’assurance doit prévoir les garanties minimales suivantes : Le chapitre A - "
            "Un seul montant d’assurance est prévu au chapitre A et ce montant est d’au moins million $. - "
            "Le montant d’assurance prévu au chapitre A est applicable pendant toute la durée de la période "
            "de couverture. Le chapitre B, incluant les deux protections suivantes : - la Protection 2; Pour "
            "que les garanties de la Protection s’appliquent, la condition suivante doit être respectée : "
            "La police d’assurance personnelle qui assure le véhicule utilisé par le chauffeur inscrit doit "
            "inclure, au jour du sinistre, la Protection ou la Protection La franchise pour la Protection est "
            "la même que celle inscrite à la police d’assurance personnelle qui assure le véhicule utilisé par "
            "le chauffeur inscrit pour la Protection ou la Protection 2, selon le cas. Les pièces justificatives "
            "permettant d’établir la protection et la franchise prévues à la police d’assurance personnelle"
            " doivent être fournies à l’assureur. - la Protection 3; Pour que les garanties de la Protection"
            " s’appliquent, la condition suivante doit être respectée : La police d’assurance personnelle"
            " qui assure le véhicule utilisé par le chauffeur inscrit doit inclure, au jour du sinistre, "
            "la Protection 1, la Protection ou la Protection Cependant, si la police d’assurance personnelle"
            " qui assure le véhicule utilisé par le chauffeur inscrit prévoit la Protection 4, les garanties"
            " de la présente protection ne s’appliquent qu’advenant la réalisation d’un risque couvert par"
            " la Protection La franchise pour la Protection est la même que celle inscrite à la police "
            "d’assurance personnelle qui assure le véhicule utilisé par le chauffeur inscrit pour la Protection "
            "1, la Protection ou la Protection 4, selon le cas. Les pièces justificatives permettant d’établir"
            " la protection et la franchise prévues à la police d’assurance personnelle doivent être fournies"
            " à l’assureur. Pour que les garanties du FAQ No s’appliquent, la condition suivante doit être"
            " respectée : La police d’assurance personnelle qui assure le véhicule utilisé par le chauffeur"
            " inscrit doit inclure, au jour du sinistre, un avenant FAQ No20, F. A. Q No20a, F. A. Q. No20b"
            " ou F. A. Q. No20c, et les pièces justificatives permettant de le démontrer doivent être fourn"
            "ies à l’assureur. Pour que les garanties du FAQ No s’appliquent, l’une ou l’autre des conditio"
            "ns suivantes doit être respectée : La police d’assurance personnelle qui assure le véhicule ut"
            "ilisé par le chauffeur inscrit doit inclure, au jour du sinistre, un avenant FAQ No43, et les "
            "pièces justificatives permettant de le démontrer doivent être fournies à l’assureur. Dans un t"
            "el cas, les garanties applicables sont les mêmes que celles prévues à la police d’assurance pe"
            "rsonnelle. Le véhicule utilisé par le chauffeur inscrit doit être couvert, au jour du sinistre"
            ", par un F. P. Q. No – Formulaire d’assurance complémentaire pour les dommages occasionnés au "
            "véhicule assuré (assurance de (le « FPQ No »), et les pièces justificatives permettant de le d"
            "émontrer doivent être fournies à l’assureur. Dans un tel cas, les garanties du FAQ No applicab"
            "les sont les suivantes, selon le cas : o Option 43A – Perte partielle – Pièces neuves; o Optio"
            "n 43E – Perte totale – Indemnisation selon la valeur de remplacement du véhicule. Il est enten"
            "du que la valeur des dommages établie selon l’Option 43E ne peut excéder le montant de l’indem"
            "nité calculée conformément aux articles et du FPQ No5, selon le cas. Le présent avenant retire"
            " de l’exclusion E. du chapitre A et de l’exclusion I. du chapitre B les utilisations du véhicu"
            "le comme taxi ou véhicule fourni avec chauffeur afin de permettre l’utilisation des véhicules "
            "assurés pour effectuer du transport rémunéré de personnes. Toutes les autres conditions du"
            " contrat d’assurance restent les mêmes."
        )
        result = tcc.compute(text)
        self.assertEqual(result, [6])


if __name__ == "__main__":
    main()
