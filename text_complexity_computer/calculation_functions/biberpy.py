# -*- coding: utf-8 -*-
# Code adaptated and taken from the Python version for doug Biber's MDA
# https://github.com/ssharoff/biberpy
# Copyright (C) 2017-2021  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
"""
A script for collecting Biber-like features from one-line text collections and a dictionary.

Expanded from experiments in Intellitext

Note: The whole script was obtained via https://github.com/ssharoff/biberpy and made by ssharoff.
It has been adapted by le-smog for use in the TextComplexityComputer framework.
All these modifications are preceded by a comment tagged with #*.
"""

dimnames = {
    "A01": "pastVerbs",
    "A03": "presVerbs",
    "B04": "placeAdverbials",
    "B05": "timeAdverbials",
    "C06": "1persProns",
    "C07": "2persProns",
    "C08": "3persProns",
    "C09": "impersProns",
    "C10": "demonstrProns",
    "C11": "indefProns",
    "C12": "doAsProVerb",
    "D13": "whQuestions",
    "E14": "nominalizations",
    "E16": "Nouns",
    "G19": "beAsMain",
    "H23": "WHclauses",
    "H33": "piedPiping",
    "H34": "sncRelatives",
    "H35": "causative",
    "H37": "conditional",
    "H38": "otherSubord",
    "I39": "preposn",
    "I40": "attrAdj",
    "I42": "ADV",
    "J43": "TTR",
    "J44": "wordLength",
    "K45": "conjuncts",
    "K46": "downtoners",
    "K48": "amplifiers",
    "K49": "generalEmphatics",
    "K50": "discoursePart",
    "K55": "publicVerbs",
    "K56": "privateVerbs",
    "K57": "suasiveVerbs",
    "K58": "seemappear",
    "L52": "possibModals",
    "L53": "necessModals",
    "L54": "predicModals",
    "N59": "contractions",
    "N60": "thatDeletion",
    "N61": "strandedPrep",
    "P66": "syntNegn",
    "P67": "analNegn",
}

language = None
tag_list = None
word_lists = None
mwe_list = None


def word_at(w):
    if isinstance(w, str):
        return w
    elif isinstance(w, list):
        return w[0].lower()


def lemma_at(w):
    if isinstance(w, str):
        try:
            out = tag_list[w][1]
        except:
            out = w
    elif isinstance(w, list):
        out = w[1]
    return out


def pos_at(w):
    if isinstance(w, str):
        try:
            out = tag_list[w][2]
        except:
            out = "PROPN"
    elif isinstance(w, list):
        out = w[2]
    return out


def fine_position_at(w):
    if isinstance(w, str):
        try:
            out = tag_list[w][3]
        except:
            out = "_"
    elif isinstance(w, list):
        out = w[3]
    return out


def is_word_set(w, word_type):
    return w in word_lists[word_type]


def find_lemma_in_sentence(doc, pos, ftclass, get_location=False):
    """
    The last parameter is for providing locations
    """
    count = 0
    out = []
    for i, w in enumerate(doc):
        if (pos == "" or pos_at(w) == pos) and lemma_at(w) in word_lists[ftclass]:
            # found in single words
            count += 1
            if get_location:
                out.append(i)
    if ftclass in mwe_list:
        A = mwe_list[ftclass]
        for end_index, (_, _) in A.iter(str(doc)):
            count += 1

    return count, out


def pos_with_lemma_filter(doc, pos, ftclass):
    count, _ = find_lemma_in_sentence(doc, pos, ftclass)
    return count


def simplePartsOfSpeech(doc, pos, finepos="", getloc=False):
    count = 0
    out = []
    for i, w in enumerate(doc):
        mainposTrue = not pos or pos_at(w) == pos
        fineposTrue = not finepos or (fine_position_at(w).find(finepos) >= 0)
        if mainposTrue and fineposTrue:
            count += 1
            if getloc:
                out.append(i)
    return count, out


def isDemonstrativePronoun(doc, l):
    try:
        nextPos = pos_at(doc[l + 1])
        nextWord = word_at(doc[l + 1])
    except:
        nextPos = nextWord = ""
    if (
        is_word_set(nextWord, "modalVerbs")
        or nextPos == "PRON"
        or is_word_set(nextWord, "clausePunctuation")
        or nextWord == "and"
    ):
        return False
    else:
        return True


def thatDeletion(doc):
    count, positions = find_lemma_in_sentence(doc=doc, pos="VERB", ftclass="specialVerbs", get_location=True)
    thatcount = 0
    for l in positions:
        try:
            nextWord = word_at(doc[l + 1])
            # First Biber prescription
            if isDemonstrativePronoun(doc, l + 1) or is_word_set(nextWord, "subjectPronouns"):
                thatcount += 1
            else:
                nextPos = pos_at(doc[l + 1])
                posAfter = pos_at(doc[l + 2])
                lCounter = 2
                # Second Biber prescription
                if nextPos == "PRON" or nextPos == "NOUN" and is_word_set(word_at(doc[l + lCounter]), "modalVerbs"):
                    thatcount += 1
                else:
                    # Third and most complicated Biber prescription
                    if nextPos in ["ADJ", "ADV", "DET", "PRON"]:
                        # This is the optional adjective
                        if posAfter == "ADJ":
                            lCounter += 1
                            posAfter = pos_at(doc[l + lCounter])
                        # Then a noun
                        if posAfter == "NOUN":
                            lCounter += 1
                            # Then if there's an auxiliary we accept this one
                            if is_word_set(word_at(doc[l + lCounter]), "modalVerbs"):
                                thatcount += 1
        except:
            # the index is out of the doc length
            count += -1

        # Only get to here if none of the prescriptions fit so
        # discount this one
        count += -1
    return thatcount


def contractions(doc):
    count = 0
    for w in doc:
        if word_at(w).find("'") >= 0:
            count += 1
    return count


def demonstrativePronouns(doc):
    count, positions = find_lemma_in_sentence(doc=doc, pos="", ftclass="demonstrativePronouns", get_location=True)
    for l in positions:
        if not isDemonstrativePronoun(doc, l):
            count += -1
    return count


def doAsProVerb(doc):
    # As usual we operate using BFI. First check if there are any DOs in the
    # sentence.
    doCount, doPositions = find_lemma_in_sentence(doc=doc, pos="", ftclass="doVerb", get_location=True)

    for doPosition in doPositions:
        try:
            # If the DO is followed by an adverb then a verb
            # or directly by a verb then it is NOT one we count
            # Also this condition should take into account the sentence
            # boundaries
            if pos_at(doc[doPosition + 1]) == "VERB" or (
                pos_at(doc[doPosition + 1]) in ["ADV", "PART"] and pos_at(doc[doPosition + 2]) == "VERB"
            ):
                doCount += -1
            else:
                # Biber's other specification seems wrong, he says:
                # 'punctuation WHP DO' implies a question but his WHP
                # only includes who, whose and which. I think it should be
                # those four PLUS the whQuestions. Need to put in the prior
                # punctuation
                previousWord = word_at(doc[doPosition - 1])
                if is_word_set(previousWord, "whQuestions") or is_word_set(previousWord, "whMarkers"):
                    doCount += -1
        except:
            doCount += -1
    return doCount


def beAsMainVerb(doc):
    # As usual we operate using BFI. First check if there are any BEs in the
    # sentence.
    beCount, bePositions = find_lemma_in_sentence(doc=doc, pos="", ftclass="beVerb", get_location=True)
    for loc in bePositions:
        # Biber's prescription for this is just that the BE is
        # followed by determiner, possessive pronoun, preposition,
        # title or adjective. We have to leave out title for now.
        # We also discount case where the BE is at the end of the sentence
        # (the SENT is at $end so the last word is at $end - 1).
        try:
            pos = pos_at(doc[loc + 1])
            if not pos in ["DET", "ADJ", "PRON", "ADP"]:
                beCount += -1
        except:
            beCount += -1
    return beCount


def strandedPrepositions(doc):
    prepCount, prepPositions = simplePartsOfSpeech(doc, "ADP", "", True)
    for loc in prepPositions:
        try:
            nextWord = word_at(doc[loc + 1])
            if not is_word_set(nextWord, "clausePunctuation"):
                prepCount += -1
        except:
            prepCount += -1
    return prepCount


def nominalizations(doc):
    nounCount, nounPositions = simplePartsOfSpeech(doc, "NOUN", "", True)
    nomCount = 0
    for loc in nounPositions:
        lemma = lemma_at(doc[loc])
        if (
            (language == "en" and lemma[-3:] in ["ion", "ent", "ess", "ism"])
            or (language == "es" and lemma[-3:] in ["ión", "nto", "leo", "cia", "dad"])
            or (language == "fr" and lemma[-3:] in ["ion", "ent", "ité", "eté", "nce", "loi"])
            or (language == "ru" and lemma[-3:] in ["ция", "сть", "ние", "тие", "тво"])
        ):
            nomCount += 1
    return nomCount


def conjuncts(doc):
    singleWords = pos_with_lemma_filter(doc, "", "conjunctsSingle")
    MWEs = 0  # 'conjunctsMWE'
    pCount = 0
    # punctuation+else, punctuation+altogether, punctuation+rather
    pCount, pPositions = find_lemma_in_sentence(doc=doc, pos="", ftclass="beVerb", get_location=True)
    for loc in pPositions:
        try:
            prevPos = pos_at(doc[loc - 1])
            if not prevPos == "PUNCT":
                pCount += 1
        except:
            pCount += 1

    return singleWords + MWEs + pCount


def BYpassives(doc):  # very simple at the moment, only the next word counts
    vCount, vPositions = simplePartsOfSpeech(doc, "", "Voice=Pass", True)
    passCount = 0
    for loc in vPositions:
        try:
            nextWord = lemma_at(doc[loc + 1])
            if (
                (language == "en" and nextWord == "by")
                or (language == "es" and nextWord == "por")
                or (language == "fr" and nextWord == "par")
            ):
                passCount += 1
            elif language == "ru":  # the instrumental case in a window of 2 words
                found = False
                for locIns in range(loc - 2, loc + 2):
                    finepos = fine_position_at(doc[locIns])
                    if finepos.find("Case=Ins") >= 0:
                        found = True
                        break
                if found:
                    passCount += 1
        except:  # do nothing
            passCount += 1
    return passCount


def syntheticNegation(doc):
    noCount, noPositions = find_lemma_in_sentence(doc=doc, pos="", ftclass="notWord", get_location=True)
    for l in noPositions:
        try:
            pos = pos_at(doc[l + 1])
            nextWord = word_at(doc[l + 1])
            if not (pos in ["ADJ", "NOUN"] and is_word_set(nextWord, "quantifiers")):
                noCount += -1
        except:
            noCount += -1
    nNCount, _ = find_lemma_in_sentence(doc=doc, pos="", ftclass="neitherWord", get_location=False)
    noCount += nNCount
    return noCount


def osubordinators(doc):
    dCount, _ = find_lemma_in_sentence(doc=doc, pos="", ftclass="osubordinators", get_location=False)
    return dCount


def discourseParticles(doc):
    dCount, dPositions = find_lemma_in_sentence(doc=doc, pos="", ftclass="discourseParticles", get_location=True)
    for l in dPositions:
        try:
            prevPos = pos_at(doc[l - 1])
            if not prevPos == "PUNCT":
                dCount += -1
        except:  # do nothing, we're at the begging of a document
            dCount += 0
    return dCount


def presentParticipialClauses(doc):
    ppQuery = "VerbForm=Con" if language == "ru" else "VerbForm=Ger"
    ppCount, ppPositions = simplePartsOfSpeech(doc, "VERB", ppQuery, True)
    for l in ppPositions:
        try:
            w0 = word_at(doc[l - 1])
            if is_word_set(w0, "clausePunctuation"):
                ppCount += -1
        except:
            ppCount += -1
    return ppCount


def predicativeAdjectives(doc):
    adjCount, adjPositions = simplePartsOfSpeech(doc, "ADJ", "", True)
    for l in adjPositions:
        try:
            previousLemma = lemma_at(doc[l - 1])
            nextPos = pos_at(doc[l + 1])
            if not (is_word_set(previousLemma, "beVerb") and nextPos in ["ADJ", "NOUN"]):
                adjCount += -1
        except:
            adjCount += -1
    return adjCount


def piedPiping(doc):
    whCount, whPositions = find_lemma_in_sentence(doc=doc, pos="", ftclass="piedPiping", get_location=True)
    for l in whPositions:
        try:
            prevPos = pos_at(doc[l - 1])
            if not prevPos == "ADP":
                whCount += -1
        except:
            whCount += -1
    return whCount


def wordLength(doc):
    totLength = 0
    wordCount = 0
    for w in doc:
        word = word_at(w)
        if not is_word_set(word, "clausePunctuation"):
            totLength += len(word)
            wordCount += 1
    return totLength / (wordCount + 0.000001)


def typeTokenRatio(doc):
    # Biber only looks at the first 400 words of each document 'to avoid
    # skewing the results to give larger values for smaller documents'.
    biberLength = min(400, len(doc))
    seenBefore = {}
    tokenCounter = 0
    for w in doc[:biberLength]:
        if not is_word_set(word_at(w), "clausePunctuation"):
            tokenCounter += 1
            if not lemma_at(w) in seenBefore:
                seenBefore[lemma_at(w)] = 1
    return len(seenBefore) / (tokenCounter + 0.000001)


def getbiberdims(doc):
    """
    processes each document as a list of tokenised words

    Note: Comments #* removes irrelevant metrics for the application
    """

    doc = doc.strip().lower().split()

    dimlist = {}
    normalise = len(doc) + 0.000001
    dimlist[dimnames["A01"]] = simplePartsOfSpeech(doc, "VERB", "Tense=Past")[0] / normalise
    dimlist[dimnames["A03"]] = simplePartsOfSpeech(doc, "VERB", "Tense=Pres")[0] / normalise

    dimlist[dimnames["B04"]] = pos_with_lemma_filter(doc, "", "placeAdverbials") / normalise
    dimlist[dimnames["B05"]] = pos_with_lemma_filter(doc, "", "timeAdverbials") / normalise

    dimlist[dimnames["C06"]] = pos_with_lemma_filter(doc, "", "firstPersonPronouns") / normalise
    dimlist[dimnames["C07"]] = pos_with_lemma_filter(doc, "PRON", "secondPersonPronouns") / normalise
    dimlist[dimnames["C08"]] = pos_with_lemma_filter(doc, "", "thirdPersonPronouns") / normalise
    dimlist[dimnames["C09"]] = pos_with_lemma_filter(doc, "", "itWord") / normalise  # impersonal
    dimlist[dimnames["C10"]] = demonstrativePronouns(doc) / normalise
    dimlist[dimnames["C11"]] = pos_with_lemma_filter(doc, "", "indefinitePronouns") / normalise
    dimlist[dimnames["C12"]] = doAsProVerb(doc) / normalise

    dimlist[dimnames["D13"]] = pos_with_lemma_filter(doc, "", "whQuestions") / normalise
    dimlist[dimnames["E14"]] = nominalizations(doc) / normalise
    dimlist[dimnames["E16"]] = (simplePartsOfSpeech(doc, "NOUN")[0] / normalise) - dimlist[
        dimnames["E14"]
    ]  # we substract nominalizations

    # * dimlist['F18'] =  BYpassives(doc)/normalise # to debug

    dimlist[dimnames["G19"]] = beAsMainVerb(doc) / normalise
    dimlist[dimnames["H23"]] = pos_with_lemma_filter(doc, "", "whMarkers") / normalise
    # * dimlist['H25'] = presentParticipialClauses(doc)/normalise
    dimlist[dimnames["H33"]] = piedPiping(doc) / normalise  # the manner in which he was told
    dimlist[dimnames["H34"]] = (
        pos_with_lemma_filter(doc, "", "sentenceRelatives") / normalise
    )  # Bob likes fried mangoes, which is the most disgusting
    dimlist[dimnames["H35"]] = pos_with_lemma_filter(doc, "", "becauseWord") / normalise
    # * dimlist['H36'] = posWithLemmaFilter(doc,'','concessives')/normalise
    dimlist[dimnames["H37"]] = pos_with_lemma_filter(doc, "", "conditionalSubordination") / normalise
    dimlist[dimnames["H38"]] = osubordinators(doc) / normalise

    dimlist[dimnames["I39"]] = simplePartsOfSpeech(doc, "ADP")[0] / normalise
    # * dimlist['I41'] = predicativeAdjectives(doc)/normalise
    dimlist[dimnames["I40"]] = (simplePartsOfSpeech(doc, "ADJ")[0] / normalise) - predicativeAdjectives(doc) / normalise
    dimlist[dimnames["I42"]] = simplePartsOfSpeech(doc, "ADV")[0] / normalise

    dimlist[dimnames["J43"]] = typeTokenRatio(doc)
    dimlist[dimnames["J44"]] = wordLength(doc)

    dimlist[dimnames["K45"]] = conjuncts(doc) / normalise
    dimlist[dimnames["K46"]] = pos_with_lemma_filter(doc, "", "downtopers") / normalise
    # * dimlist['K47'] = posWithLemmaFilter(doc,'','generalHedges')/normalise
    dimlist[dimnames["K48"]] = pos_with_lemma_filter(doc, "", "amplifiers") / normalise
    dimlist[dimnames["K49"]] = pos_with_lemma_filter(doc, "", "generalEmphatics") / normalise
    dimlist[dimnames["K50"]] = discourseParticles(doc) / normalise

    dimlist[dimnames["L52"]] = pos_with_lemma_filter(doc, "", "possibilityModals") / normalise
    dimlist[dimnames["L53"]] = pos_with_lemma_filter(doc, "", "necessityModals") / normalise
    dimlist[dimnames["L54"]] = pos_with_lemma_filter(doc, "", "predictionModals") / normalise

    dimlist[dimnames["K55"]] = pos_with_lemma_filter(doc, "VERB", "publicVerbs") / normalise
    dimlist[dimnames["K56"]] = pos_with_lemma_filter(doc, "VERB", "privateVerbs") / normalise
    dimlist[dimnames["K57"]] = pos_with_lemma_filter(doc, "", "suasiveVerbs") / normalise
    dimlist[dimnames["K58"]] = pos_with_lemma_filter(doc, "", "seemappear") / normalise

    dimlist[dimnames["N59"]] = contractions(doc) / normalise
    dimlist[dimnames["N60"]] = thatDeletion(doc) / normalise

    dimlist[dimnames["N61"]] = strandedPrepositions(doc) / normalise

    dimlist[dimnames["P66"]] = syntheticNegation(doc) / normalise
    dimlist[dimnames["P67"]] = pos_with_lemma_filter(doc, "", "notWord") / normalise

    return dimlist
