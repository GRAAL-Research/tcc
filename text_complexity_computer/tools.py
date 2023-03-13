import re
import sys
from typing import Dict, Tuple

import ahocorasick


def clean_text(text: str) -> str:
    """Text cleaner
    Below is a comprehensive list of changes:
        - remove the bullet lists marker
        - special characters
        - add space between points
        - remove space multiplications

    Args:
        text (string): text that will be computed

    Returns:
        string: cleaned text
    """
    text = text.replace("_", " ")
    text = text.replace("•", " ")
    text = text.replace("○", " ")
    text = re.sub(r"\[…\]", " ", text)
    text = text.replace("…", " . ")
    text = re.sub(r"- \d+ -", " ", text)
    text = re.sub(r"\d+\)?\. ", " ", text)
    text = re.sub(r"\(?\d+(\.\d+)*\)?\.? ", " ", text)
    text = re.sub(r"\(?\w+\) ", " ", text)
    text = re.sub(r"\.+(\w+)", r". \g<1>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def read_word_lists(file_path: str, verbosity: int) -> Tuple[Dict, Dict]:
    """
    reads lists in the format
    firstPersonPronouns = I,we,me,us,my,our,myself,ourselves
    """
    word_lists = {}
    mwe_list = {}
    with open(file_path, "r", encoding="utf8") as file:
        for i, line in enumerate(file.readlines()):
            x = line.strip().split(" = ")
            if len(x) == 2:
                values = x[1].split(",")
                word_lists[x[0]] = set([w.strip() for w in values])
                A = ahocorasick.Automaton()  # separately add an automaton structure for finding MWEs
                mwecur = []
                for mwe in values:
                    mwe = mwe.strip()
                    if mwe.find(" ") > 0:
                        mwecur.append(mwe.lower())
                if len(mwecur) > 0:
                    for idx, key in enumerate(set(mwecur)):
                        A.add_word(key, (idx, key))
                    A.make_automaton()
                    mwe_list[x[0]] = A
            else:
                if verbosity > 0 and len(line) > 1:
                    print("Error in line %i, %s" % (i, line), file=sys.stderr)
    if verbosity > 1:
        print("Read total %d classes" % len(word_lists), file=sys.stderr)
        print(sorted(word_lists.keys()), file=sys.stderr)
    word_lists["specialVerbs"] = set.union(
        word_lists["publicVerbs"], word_lists["privateVerbs"], word_lists["suasiveVerbs"]
    )
    word_lists["modalVerbs"] = set.union(
        word_lists["possibilityModals"], word_lists["necessityModals"], word_lists["predictionModals"]
    )
    return word_lists, mwe_list


def read_num_list(file_path: str) -> Dict:
    """
    reads a numfile in the format
    1625260 years year NOUN Number=Plur
    This produces a "most frequent tag" substitute for tagging
    """
    tag_list = {}
    with open(file_path, "r", encoding="utf8") as file:
        for line in file.readlines():
            x = line.rstrip().split()
            if len(x) == 5 and not x[1] in tag_list:
                tag_list[x[1].lower()] = (x[0], x[2], x[3], x[4])
    return tag_list
