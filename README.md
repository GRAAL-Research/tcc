<div align="center">

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tcc)](https://pypi.org/project/text_complexity_computer)
[![PyPI Status](https://badge.fury.io/py/text_complexity_computer.svg)](https://badge.fury.io/py/text_complexity_computer)
[![PyPI Status](https://pepy.tech/badge/text_complexity_computer)](https://pepy.tech/project/text_complexity_computer)
[![Downloads](https://pepy.tech/badge/text_complexity_computer/month)](https://pepy.tech/project/text_complexity_computer)

[![Formatting](https://github.com/GRAAL-Research/tcc/actions/workflows/formatting.yml/badge.svg?branch=main)](https://github.com/GRAAL-Research/tcc/actions/workflows/formatting.yml)
[![Linting](https://github.com/GRAAL-Research/tcc/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/GRAAL-Research/tcc/actions/workflows/linting.yml)
[![Tests](https://github.com/GRAAL-Research/tcc/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/GRAAL-Research/tcc/actions/workflows/tests.yml)
[![pr welcome](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)

</div>

## Here is TextComplexityComputer.

TextComplexityComputer is a state-of-the-art library for assessing complexity of a text in French. Use
it to assess the complexity of a text in French. it is compatible with the __latest version of Scikit-Learn__ and
__Python >= 3.8__.

## Getting Started:

```python
from text_complexity_computer import TextComplexityComputer

tcc = TextComplexityComputer()
print(tcc.get_metrics_scores("Alibaba et les 40 voleurs."))
print(tcc.compute("Alibaba et les 40 voleurs."))
```

------------------

## Installation

- **Install the stable version of TextComplexityComputer:**

```sh
pip install text_complexity_computer
```

- **Install the latest development version of TextComplexityComputer:**

```sh
pip install -U git+https://github.com/GRAAL-Research/tcc.git@dev
```

------------------

## Cite

Use the following to cite this package and our article;

```
@article{Primpied2022Quantifying,
	author = {Primpied, Vincent and Beauchemin, David and Khoury, Richard},
	journal = {Proceedings of the Canadian Conference on Artificial Intelligence},
	year = {2022},
	month = {may 27},
	note = {https://caiac.pubpub.org/pub/iaeeogod},
	publisher = {Canadian Artificial Intelligence Association (CAIAC)},
	title = {Quantifying {French} {Document} {Complexity} },
}
```

------------------

## Contributing to TextComplexityComputer

We welcome user input, whether it is regarding bugs found in the library or feature propositions ! Make sure to have a
look at our [contributing guidelines](https://github.com/GRAAL-Research/tcc/blob/main/.github/CONTRIBUTING.md)
for more details on this matter.

## License

TextComplexityComputer is LGPLv3 licensed, as found in
the [LICENSE file](https://github.com/GRAAL-Research/tcc/blob/main/LICENSE).

------------------
