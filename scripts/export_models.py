"""Export the pickled scikit-learn estimators to version-independent ``.npz`` files.

The original estimators were serialized with scikit-learn 1.3.2. Since scikit-learn 1.4 changed how trees store
their leaf values, unpickling them in newer versions silently yields wrong probabilities. This script must therefore
run with scikit-learn 1.3.2 (Python 3.11 or 3.12), e.g.::

    uv run --python 3.11 --with "scikit-learn==1.3.2" --with "numpy<2" python scripts/export_models.py <pickles_dir>

It writes ``fr_model.npz``, ``fr_MinMaxScaler.npz`` and ``fr_StandardScaler.npz`` in
``text_complexity_computer/resources/fr``.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import sklearn

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "text_complexity_computer" / "resources" / "fr"


def _load(path: Path) -> object:
    with path.open("rb") as file:
        return pickle.load(file)


def export_forest(pickle_path: Path, output_path: Path) -> None:
    forest = _load(pickle_path)
    trees = [estimator.tree_ for estimator in forest.estimators_]
    offsets = np.cumsum([0] + [tree.node_count for tree in trees[:-1]])

    def shifted_children(children: np.ndarray, offset: int) -> np.ndarray:
        return np.where(children == -1, -1, children + offset)

    values = np.concatenate([tree.value[:, 0, :] for tree in trees])  # class counts per node (single output)
    np.savez_compressed(
        output_path,
        feature_names=np.asarray(forest.feature_names_in_, dtype=str),
        classes=forest.classes_,
        roots=offsets,
        children_left=np.concatenate(
            [shifted_children(tree.children_left, offset) for tree, offset in zip(trees, offsets, strict=True)]
        ),
        children_right=np.concatenate(
            [shifted_children(tree.children_right, offset) for tree, offset in zip(trees, offsets, strict=True)]
        ),
        feature=np.concatenate([tree.feature for tree in trees]),
        threshold=np.concatenate([tree.threshold for tree in trees]),
        leaf_proba=values / values.sum(axis=1, keepdims=True),
        max_depth=np.asarray(max(tree.max_depth for tree in trees)),
    )


def export_scaler(pickle_path: Path, output_path: Path) -> None:
    scaler = _load(pickle_path)
    feature_names = np.asarray(scaler.feature_names_in_, dtype=str)
    if type(scaler).__name__ == "MinMaxScaler":
        np.savez_compressed(
            output_path, kind="minmax", feature_names=feature_names, scale=scaler.scale_, min=scaler.min_
        )
    else:
        np.savez_compressed(
            output_path, kind="standard", feature_names=feature_names, mean=scaler.mean_, scale=scaler.scale_
        )


def main() -> None:
    if sklearn.__version__ != "1.3.2":
        sys.exit(f"scikit-learn 1.3.2 is required to read the original pickles, found {sklearn.__version__}.")
    pickles_dir = Path(sys.argv[1])
    export_forest(pickles_dir / "fr_model.pickle", OUTPUT_DIR / "fr_model.npz")
    for name in ("MinMaxScaler", "StandardScaler"):
        export_scaler(pickles_dir / f"fr_{name}.pickle", OUTPUT_DIR / f"fr_{name}.npz")


if __name__ == "__main__":
    main()
