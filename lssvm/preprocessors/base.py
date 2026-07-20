"""Shared dataset-preparation helpers for the LSSVM preprocessors package.

Each dataset module (``iris``, ``breast_cancer``) loads its raw sklearn dataset
and then reuses these helpers to split, scale, and binarize into the standard
LSSVM sub-problem shape:

    (X_train, X_test, y_train, y_test, class_name)

where ``y`` values are +1 (target class) or -1 (rest).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split with a StandardScaler fit on the train split.

    Returns ``(X_train, X_test, y_train_raw, y_test_raw)`` where the X arrays are
    standardized and the y arrays are the original integer class labels.
    """
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train_raw, y_test_raw


def to_ovr_splits(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    class_names,
    class_idx: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Build One-vs-Rest binary sub-problems (one per class, or a single class).

    y values are +1 for the target class, -1 for the rest.
    """
    indices = [class_idx] if class_idx is not None else range(len(class_names))
    results = []
    for c in indices:
        y_tr = np.where(y_train_raw == c, 1.0, -1.0)
        y_te = np.where(y_test_raw == c, 1.0, -1.0)
        results.append((X_train, X_test, y_tr, y_te, class_names[c]))
    return results


def to_binary_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    positive_class: int,
    class_name: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Build a single binary sub-problem (+1 for ``positive_class``, -1 otherwise)."""
    y_tr = np.where(y_train_raw == positive_class, 1.0, -1.0)
    y_te = np.where(y_test_raw == positive_class, 1.0, -1.0)
    return [(X_train, X_test, y_tr, y_te, class_name)]
