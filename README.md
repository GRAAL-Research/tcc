<div align="center">

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/text_complexity_computer)](https://pypi.org/project/text_complexity_computer)
[![PyPI Status](https://badge.fury.io/py/text_complexity_computer.svg)](https://badge.fury.io/py/text_complexity_computer)
[![Downloads](https://static.pepy.tech/badge/text_complexity_computer)](https://pepy.tech/project/text_complexity_computer)
[![Downloads/month](https://static.pepy.tech/badge/text_complexity_computer/month)](https://pepy.tech/project/text_complexity_computer)
[![CI](https://github.com/GRAAL-Research/tcc/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GRAAL-Research/tcc/actions/workflows/ci.yml)
[![Documentation](https://github.com/GRAAL-Research/tcc/actions/workflows/docs.yml/badge.svg?branch=main)](https://graal-research.github.io/tcc/)
[![PR welcome](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg)](https://github.com/GRAAL-Research/tcc/blob/main/.github/CONTRIBUTING.md)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/GRAAL-Research/tcc/blob/main/LICENSE.txt)

</div>

# TextComplexityComputer

TextComplexityComputer (TCC) assesses the complexity of a French text. It computes about sixty linguistic metrics
(syntactic complexity, vocabulary, lexical diversity, readability formulas and Biber dimensions) and feeds them to a
classifier that estimates a difficulty level from 0 to 7.

**Documentation: <https://graal-research.github.io/tcc/>**

## Installation

TCC requires Python 3.11 or later.

```sh
pip install text_complexity_computer
# or
uv add text_complexity_computer
```

The spaCy model `fr_core_news_sm` is downloaded on first use. To install it beforehand:

```sh
python -m spacy download fr_core_news_sm
```

## Getting started

```python
from text_complexity_computer import TextComplexityComputer

tcc = TextComplexityComputer()

tcc.compute("Alibaba et les 40 voleurs.")
# array([1])

tcc.get_metrics_scores("Le chat dort sur le tapis. Il fait beau.", metrics=["mls", "fk_ease"], with_biberpy=False)
#    fk_ease       mls
# 0  1.15424 -0.015608
```

See the [usage guide](https://graal-research.github.io/tcc/usage/) for the metrics, the scalers and the bundled model.

## Cite

Use the following to cite this package and our article:

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

## Contributing

We welcome bug reports and feature proposals. See the
[contributing guidelines](https://github.com/GRAAL-Research/tcc/blob/main/.github/CONTRIBUTING.md) to set up the
development environment with uv.

## License

TCC is licensed under the GPLv3, as found in the [LICENSE file](https://github.com/GRAAL-Research/tcc/blob/main/LICENSE.txt).
The Biber dimensions module is adapted from [biberpy](https://github.com/ssharoff/biberpy) by Serge Sharoff, also under the GPLv3.
