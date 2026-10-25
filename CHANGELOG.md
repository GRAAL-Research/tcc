## dev

## 0.3.0

- Fix wrong metrics with pandas 3: every metric outside biberpy stayed at 0, so `compute` predicted from wrong features.
- Fix wrong predictions with scikit-learn 1.4 or later, which misreads the trees pickled with scikit-learn 1.3.2.
- Ship the classifier and the scalers as numpy arrays (`.npz`) run without scikit-learn, which is no longer a
  dependency. Predictions match scikit-learn 1.3.2 exactly. `model` and `scaler` are now
  `text_complexity_computer.models` objects exposing `predict`, `predict_proba` and `transform`.
- Require Python 3.11 or later and support Python 3.11 to 3.14.
- Raise a `ValueError` on an unknown metric or scaler name.
- Split `get_num_words(get_size=True)` and `get_num_words(get_unique=True)` into `get_num_words_and_letters` and
  `get_num_words_and_unique_words`.
- Exclude preshed 3.0.14, which breaks spaCy, and require spaCy 3.8.15 or later.
- Move the packaging to `pyproject.toml` (hatchling, uv), replace black and pylint by ruff, add mypy and `py.typed`.
- Add reference tests that freeze the metrics and predictions.
- New documentation site built with MkDocs.
- Declare the GPLv3 license, consistent with `LICENSE.txt` and the biberpy module (previously declared as LGPLv3).

## 0.2

- Removed en.
- Fix model size mismatch.
- Improve codebase.
- Improve setup.
- Add Python 3.11 support.

## Beta (0.1.7)

- Initial release of the library.
- Fixed missing data package.
- Fixed Biperby.
- Add missing en model.
- Fix error in SpaCy model.
- Improve code and fix `get_num_sentences` and `nws_90`.
- Improve code cleanest.
