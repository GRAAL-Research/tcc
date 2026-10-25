import re
import sys
from importlib.resources.abc import Traversable
from importlib.resources import files

import ahocorasick

WordLists = dict[str, set[str]]
MweList = dict[str, ahocorasick.Automaton]
TagList = dict[str, tuple[str, str, str, str]]


def resource_path(language: str, file_name: str) -> Traversable:
    """Return the path of a resource file bundled with the package."""
    return files("text_complexity_computer").joinpath("resources", language, file_name)


def clean_text(text: str) -> str:
    """Clean a text before computing its metrics.

    Changes made:

    - remove the bullet list markers and some special characters
    - add a space after points
    - collapse repeated whitespaces

    Args:
        text: Text to clean.

    Returns:
        The cleaned text.
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


def read_word_lists(file_path: Traversable, verbosity: int) -> tuple[WordLists, MweList]:
    """Read word lists in the format ``firstPersonPronouns = I,we,me,us,my,our,myself,ourselves``.

    Returns:
        The word lists and, for the lists holding multi-word expressions, an Aho-Corasick automaton to find them.
    """
    word_lists: WordLists = {}
    mwe_list: MweList = {}
    lines = file_path.read_text(encoding="utf8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        x = line.strip().split(" = ")
        if len(x) == 2:
            values = x[1].split(",")
            word_lists[x[0]] = {w.strip() for w in values}
            mwecur = [mwe.strip().lower() for mwe in values if mwe.strip().find(" ") > 0]
            if mwecur:
                automaton = ahocorasick.Automaton()
                for idx, key in enumerate(set(mwecur)):
                    automaton.add_word(key, (idx, key))
                automaton.make_automaton()
                mwe_list[x[0]] = automaton
        elif verbosity > 0 and len(line) > 1:
            print(f"Error in line {i}, {line}", file=sys.stderr)
    if verbosity > 1:
        print(f"Read total {len(word_lists)} classes", file=sys.stderr)
        print(sorted(word_lists.keys()), file=sys.stderr)
    word_lists["specialVerbs"] = set.union(
        word_lists["publicVerbs"], word_lists["privateVerbs"], word_lists["suasiveVerbs"]
    )
    word_lists["modalVerbs"] = set.union(
        word_lists["possibilityModals"], word_lists["necessityModals"], word_lists["predictionModals"]
    )
    return word_lists, mwe_list


def read_num_list(file_path: Traversable) -> TagList:
    """Read a num file in the format ``1625260 years year NOUN Number=Plur``.

    This produces a "most frequent tag" substitute for tagging.
    """
    tag_list: TagList = {}
    for line in file_path.read_text(encoding="utf8").splitlines():
        x = line.rstrip().split()
        if len(x) == 5 and x[1] not in tag_list:
            tag_list[x[1].lower()] = (x[0], x[2], x[3], x[4])
    return tag_list
