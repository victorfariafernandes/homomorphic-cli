"""Iris dataset preparer: 4 features, 3 classes -> 3 One-vs-Rest sub-problems."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from lssvm.preprocessors.base import split_scale, to_ovr_splits


def prepare_binary(
    class_idx: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load Iris, split, scale, and return OvR binary sub-problems.

    Parameters
    ----------
    class_idx : If given, return only that class-vs-rest problem.
                If None, return all three OvR problems.
    test_size : Fraction of data held out for testing.
    random_state : Seed for reproducible splits.

    Returns
    -------
    List of (X_train, X_test, y_train, y_test, class_name) tuples.
    y values are +1 (target class) or -1 (rest).
    """
    iris = load_iris()
    X_train, X_test, y_train_raw, y_test_raw = split_scale(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )
    return to_ovr_splits(
        X_train, X_test, y_train_raw, y_test_raw, iris.target_names, class_idx
    )


def raw_test_labels(
    test_size: float = 0.2, random_state: int = 42
) -> np.ndarray:
    """Return the raw (integer) multiclass test labels for OvR accuracy scoring."""
    iris = load_iris()
    _, _, _, y_test_raw = split_scale(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )
    return y_test_raw
