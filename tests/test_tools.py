from pathlib import Path

from text_complexity_computer.tools import read_num_list, read_word_lists

LINE_SEPARATOR = chr(0x2028)


def test_read_num_list_keeps_the_first_entry_regardless_of_case(tmp_path: Path) -> None:
    num_file = tmp_path / "tags.num"
    num_file.write_text("500 paris paris PROPN _\n20 Paris Paris NOUN _\n", encoding="utf8")

    assert read_num_list(num_file) == {"paris": ("500", "paris", "PROPN", "_")}


def test_read_word_lists_does_not_split_on_unicode_line_separators(tmp_path: Path) -> None:
    lists = {
        "publicVerbs": "dire",
        "privateVerbs": "penser",
        "suasiveVerbs": "exiger",
        "possibilityModals": "pouvoir",
        "necessityModals": "devoir",
        "predictionModals": "aller",
        "amplifiers": "très,vraiment" + LINE_SEPARATOR + "bien",
    }
    properties = tmp_path / "fr.properties"
    properties.write_text("".join(f"{name} = {words}\n" for name, words in lists.items()), encoding="utf8")

    word_lists, _ = read_word_lists(properties, verbosity=0)

    assert word_lists["amplifiers"] == {"très", "vraiment" + LINE_SEPARATOR + "bien"}
    assert word_lists["specialVerbs"] == {"dire", "penser", "exiger"}
