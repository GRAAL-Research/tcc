import re


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
