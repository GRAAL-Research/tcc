# Contributing to TextComplexityComputer
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, you **must** ensure it is properly tested.
3. If you've changed APIs, update the documentation.
4. Ensure the CI passes (formatting, linting, type checking and tests).
5. Add an entry to the `dev` section of `CHANGELOG.md`.
6. Submit that pull request!

## Any contributions you make will be under the GPLv3 Software License
In short, when you submit code changes, your submissions are understood to be under the same [GPLv3 License](https://choosealicense.com/licenses/gpl-3.0/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Write bug reports with detail, background, and sample code

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/GRAAL-Research/tcc/issues). You should use one of our [proposed templates](https://github.com/GRAAL-Research/tcc/tree/main/.github/ISSUE_TEMPLATE) when appropriate; they are integrated with GitHub and do most of the formatting for you. It's that easy!

**Great Bug Reports** tend to have:

- A quick and clear summary and/or background
- Steps to reproduce
  - Be specific and clear!
  - Give sample code if you can. Try to reduce the bug to the minimum amount of code needed to reproduce: it will help in our troubleshooting procedure.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)
Feel free to include any print screen or other file you feel may further clarify your point.

## Do you have a suggestion for an enhancement?

We use GitHub issues to track enhancement requests.  Before you create an enhancement request:

* Make sure you have a clear idea of the enhancement you would like.  If you have a vague idea, consider discussing
it first on the users list.

* Check the documentation to make sure your feature does not already exist.

* Do a [quick search](https://github.com/GRAAL-Research/tcc/issues) to see whether your enhancement has already been suggested.

When creating your enhancement request, please:

* Provide a clear title and description.

* Explain why the enhancement would be useful.  It may be helpful to highlight the feature in other libraries.

* Include code examples to demonstrate how the enhancement would be used.


## Development environment

The project uses [uv](https://docs.astral.sh/uv/). The following command creates the virtual environment with the
library, the development tools and the spaCy model:

```shell
uv sync
```

## Coding style

The code is formatted and linted with [ruff](https://docs.astral.sh/ruff/) and type checked with
[mypy](https://mypy.readthedocs.io) in strict mode, as configured in `pyproject.toml`:

```shell
uv run ruff format .
uv run ruff check .
uv run mypy
```

## Tests

If your pull request introduces a new feature, deliver it with tests that ensure the correct behavior. The tests are
located under the `tests` folder. **All** the tests must pass:

```shell
uv run pytest
```

The CI runs them on Linux, Windows and macOS with Python 3.11 to 3.14, and on Python 3.11 with the lowest supported
dependency versions.

`tests/test_reference_scores.py` compares the metrics and predictions with reference values computed in the
environment the model was trained in. A change in these values means the features fed to the model changed: it must
be deliberate and explained in the pull request.

## Bundled model

The classifier and the scalers are stored as numpy arrays in `text_complexity_computer/resources/fr/*.npz`. They were
exported from the original scikit-learn 1.3.2 pickles with `scripts/export_models.py`; the inference code lives in
`text_complexity_computer/models.py`.

## Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org) from the `docs` folder and the docstrings
(Google style). To preview it:

```shell
uv run --group docs mkdocs serve
```

It is deployed to GitHub Pages on every push to `main`.

## License
By contributing, you agree that your contributions will be licensed under its GPLv3 License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).
