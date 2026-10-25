# Usage

## Difficulty level

```python
from text_complexity_computer import TextComplexityComputer

tcc = TextComplexityComputer()
level = tcc.compute(text)[0]  # integer from 0 to 7
```

`tcc.model.predict_proba(tcc.get_metrics_scores(text))` returns the probability of each level, in the order of
`tcc.model.classes_`.

## Scalers

The classifier was trained on scaled metrics. The `scaler` argument selects the scaling applied by
`get_metrics_scores` and `compute`:

| `scaler`                   | Effect                                                             |
|----------------------------|--------------------------------------------------------------------|
| `"MinMaxScaler"` (default) | Min-max scaling fitted on the training corpus.                     |
| `"StandardScaler"`         | Standardization (zero mean, unit variance) fitted on the corpus.   |
| `None`                     | Raw metrics. A warning is emitted since `compute` expects scaling. |

Use `scaler=None` to read the metrics in their natural units:

```python
tcc = TextComplexityComputer(scaler=None)
tcc.get_metrics_scores("Le chat dort sur le tapis. Il fait beau.", metrics=["mls", "fk_ease"], with_biberpy=False)
#    fk_ease  mls
# 0   119.19  3.0
```

## Metrics

`get_metrics_scores(text, metrics="all", with_biberpy=True)` accepts `"all"`, a metric name, a list of metric names
or `None`. An unknown name raises a `ValueError`. Columns are sorted by name.

| Name           | Metric                                                         |
|----------------|----------------------------------------------------------------|
| `mls`          | Mean length of sentences                                       |
| `ps_30`        | Percentage of sentences of 30 words or more                    |
| `nws_90`       | Length of the 90th percentile sentence                         |
| `mlt`          | Mean length of T-units                                         |
| `tu_s`         | T-units per sentence                                           |
| `ctu_tu`       | Complex T-units per T-unit                                     |
| `dc_c`         | Dependent clauses per clause                                   |
| `c_s`          | Clauses per sentence                                           |
| `c_tu`         | Clauses per T-unit                                             |
| `cp_c`         | Coordinate phrases per clause                                  |
| `cp_tu`        | Coordinate phrases per T-unit                                  |
| `pa`           | Share of words not in the Gougenheim list of easy words        |
| `nlm`          | Mean number of letters per word                                |
| `uni_gram_lem` | Mean log frequency of the lemmas (Lexique 3.83)                |
| `msttr`        | Mean sequential type-token ratio                               |
| `mtld`         | Measure of textual lexical diversity                           |
| `mattr`        | Same value as `mtld`, kept as is because the model uses it     |
| `fk_ease`      | Flesch-Kincaid reading ease                                    |
| `bingui`       | Punctuation marks per sentence                                 |
| `km_score`     | Kandel and Moles formula (Flesch adaptation for French)        |

With `with_biberpy=True`, the 43 Biber dimensions (pronouns, verb tenses, modals, subordination, etc.) are added; their
names are the values of `text_complexity_computer.calculation_functions.biberpy.dimnames`.

When a scaler is set, scaling uses every metric: the metrics you did not select count as 0 before scaling.

## Bundled model

The classifier is a random forest trained with scikit-learn 1.3.2. Since version 0.3.0 it ships as plain numpy arrays
and runs without scikit-learn, which avoids the silent errors caused by loading an old pickle with a newer
scikit-learn. The test suite checks the metrics and predictions against reference values computed with the original
environment.
