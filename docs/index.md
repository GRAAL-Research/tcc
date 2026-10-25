# TextComplexityComputer

TextComplexityComputer (TCC) assesses the complexity of a French text. It computes about sixty linguistic metrics
(syntactic complexity, vocabulary, lexical diversity, readability formulas and Biber dimensions) and feeds them to a
classifier that estimates a difficulty level.

The approach is described in
[Quantifying French Document Complexity](https://caiac.pubpub.org/pub/iaeeogod) (Canadian AI 2022).

## Installation

TCC requires Python 3.11 or later.

```sh
pip install text_complexity_computer
```

or, with [uv](https://docs.astral.sh/uv/):

```sh
uv add text_complexity_computer
```

TCC uses the spaCy model `fr_core_news_sm`. It is downloaded on first use when missing; to install it beforehand
(for instance in a container or an environment without `pip`):

```sh
python -m spacy download fr_core_news_sm
```

## Quickstart

```python
from text_complexity_computer import TextComplexityComputer

tcc = TextComplexityComputer()

tcc.compute("Alibaba et les 40 voleurs.")
# array([1])

tcc.get_metrics_scores("Le chat dort sur le tapis. Il fait beau.", metrics=["mls", "fk_ease"], with_biberpy=False)
#    fk_ease       mls
# 0  1.15424 -0.015608
```

`compute` returns the estimated difficulty level, an integer from 0 to 7. `get_metrics_scores` returns the metrics in
a one-row `pandas.DataFrame`. See [Usage](usage.md) for the available metrics and scalers.

!!! tip
    Creating a `TextComplexityComputer` loads the spaCy pipeline and the word lists, which takes a few seconds. Create
    it once and reuse it for all your texts.

## Cite

```bibtex
@article{Primpied2022Quantifying,
    author = {Primpied, Vincent and Beauchemin, David and Khoury, Richard},
    journal = {Proceedings of the Canadian Conference on Artificial Intelligence},
    year = {2022},
    month = {may 27},
    note = {https://caiac.pubpub.org/pub/iaeeogod},
    publisher = {Canadian Artificial Intelligence Association (CAIAC)},
    title = {Quantifying {French} {Document} {Complexity}},
}
```

## License

TCC is licensed under the [GPLv3](https://github.com/GRAAL-Research/tcc/blob/main/LICENSE.txt). The Biber
dimensions module is adapted from [biberpy](https://github.com/ssharoff/biberpy) by Serge Sharoff, also under the GPLv3.
