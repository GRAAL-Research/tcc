import pytest

from text_complexity_computer import TextComplexityComputer


@pytest.fixture(scope="session")
def tcc() -> TextComplexityComputer:
    return TextComplexityComputer()
