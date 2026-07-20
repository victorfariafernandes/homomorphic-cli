"""Breast-cancer dataset preparer: 30 features, 2 classes -> 1 binary sub-problem.

sklearn's ``load_breast_cancer`` encodes target 0 = malignant, 1 = benign. We
build a single binary problem with y = +1 for malignant (target 0) and -1 for
benign (target 1).

Malignant is the positive class for two reasons: (1) it is the clinical
convention (positive = malignant detection); (2) this solver's KKT variant
produces an orientation-independent weight vector that ranks the malignant class
higher, so assigning it +1 keeps sign(score) aligned with the decision direction
(labelling benign +1 inverts the classifier).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer

from lssvm.preprocessors.base import split_scale, to_binary_split

# +1 label maps to this raw class; malignant (0) is the positive class.
POSITIVE_CLASS = 0
CLASS_NAME = "malignant_vs_benign"


def prepare_binary(
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load breast cancer, split, scale, and return a single binary sub-problem.

    Returns
    -------
    A one-element list of (X_train, X_test, y_train, y_test, class_name).
    y values are +1 (malignant) or -1 (benign).
    """
    data = load_breast_cancer()
    X_train, X_test, y_train_raw, y_test_raw = split_scale(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return to_binary_split(
        X_train, X_test, y_train_raw, y_test_raw, POSITIVE_CLASS, CLASS_NAME
    )


def raw_test_labels(
    test_size: float = 0.2, random_state: int = 42
) -> np.ndarray:
    """Return the raw (integer) test labels for binary accuracy scoring."""
    data = load_breast_cancer()
    _, _, _, y_test_raw = split_scale(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return y_test_raw
