"""Numpy inference for the bundled scaler and random forest.

The estimators were trained with scikit-learn 1.3.2 and exported to ``.npz`` arrays by ``scripts/export_models.py``.
Inference reproduces scikit-learn's arithmetic, so predictions do not depend on the installed scikit-learn version.
"""

from importlib.resources.abc import Traversable

import numpy as np
import pandas as pd


def _load_npz(path: Traversable) -> dict[str, np.ndarray]:
    with path.open("rb") as file, np.load(file, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _ordered_features(X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    missing = sorted(set(feature_names) - set(X.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}.")
    return X[feature_names].to_numpy(dtype=np.float64)


class Scaler:
    """Min-max or standard scaler, equivalent to scikit-learn's ``MinMaxScaler`` and ``StandardScaler``."""

    def __init__(self, path: Traversable) -> None:
        arrays = _load_npz(path)
        self.kind = str(arrays["kind"])
        self.feature_names_in_ = arrays["feature_names"]
        self.scale_ = arrays["scale"]
        self.offset_ = arrays["min"] if self.kind == "minmax" else arrays["mean"]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scale the features of ``X``, matched by column name."""
        values = _ordered_features(X, list(self.feature_names_in_))
        if self.kind == "minmax":
            return values * self.scale_ + self.offset_
        return (values - self.offset_) / self.scale_


class RandomForestClassifier:
    """Random forest inference equivalent to scikit-learn's ``RandomForestClassifier.predict``.

    All the trees are stored as one flat node array; ``roots`` holds the index of each tree root.
    """

    def __init__(self, path: Traversable) -> None:
        arrays = _load_npz(path)
        self.feature_names_in_ = arrays["feature_names"]
        self.classes_ = arrays["classes"]
        self._roots = arrays["roots"]
        self._children_left = arrays["children_left"]
        self._children_right = arrays["children_right"]
        self._feature = arrays["feature"]
        self._threshold = arrays["threshold"]
        self._leaf_proba = arrays["leaf_proba"]
        self._max_depth = int(arrays["max_depth"])

    def _leaves(self, X: pd.DataFrame) -> np.ndarray:
        # scikit-learn casts the features to float32 before comparing them with the float64 thresholds.
        values = _ordered_features(X, list(self.feature_names_in_)).astype(np.float32).astype(np.float64)
        nodes = np.broadcast_to(self._roots, (values.shape[0], self._roots.shape[0])).copy()
        rows = np.arange(values.shape[0])[:, np.newaxis]
        for _ in range(self._max_depth):
            is_leaf = self._children_left[nodes] == -1
            if is_leaf.all():
                break
            go_left = values[rows, self._feature[nodes]] <= self._threshold[nodes]
            next_nodes = np.where(go_left, self._children_left[nodes], self._children_right[nodes])
            nodes = np.where(is_leaf, nodes, next_nodes)
        return nodes

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the class probabilities, averaged over the trees."""
        return self._leaf_proba[self._leaves(X)].sum(axis=1) / self._roots.shape[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the most probable class of each row of ``X``."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
