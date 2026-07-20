"""Dataset preparers for LSSVM training/inference.

Each supported dataset is a sub-package exposing ``prepare_binary(...)`` (and
``raw_test_labels(...)``) returning the standard sub-problem shape:

    (X_train, X_test, y_train, y_test, class_name)

with y values in {+1, -1}. Entry points resolve the ``--dataset`` CLI flag
through :data:`SUPPORTED_DATASETS` and dispatch via :func:`prepare_binary`.
"""
from __future__ import annotations

from lssvm.preprocessors import breast_cancer, iris

SUPPORTED_DATASETS = {"iris", "breast_cancer"}
DEFAULT_DATASET = "iris"

_PREPARERS = {
    "iris": iris.prepare_binary,
    "breast_cancer": breast_cancer.prepare_binary,
}

_RAW_LABELS = {
    "iris": iris.raw_test_labels,
    "breast_cancer": breast_cancer.raw_test_labels,
}

# Per-dataset kernel selection, keyed by class index. Shared by every entry
# point so plaintext / train / infer stay in lockstep. Class indices not listed
# fall back to the linear kernel downstream.
DATASET_KERNEL_SELECTION = {
    "iris": {0: "linear", 1: "poly", 2: "poly"},
    "breast_cancer": {0: "linear"},
}


def _check_dataset(dataset: str) -> None:
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported: {supported}"
        )


def prepare_binary(dataset: str, **kwargs):
    """Dispatch to the named dataset's ``prepare_binary``."""
    _check_dataset(dataset)
    return _PREPARERS[dataset](**kwargs)


def raw_test_labels(dataset: str, **kwargs):
    """Dispatch to the named dataset's raw multiclass/binary test labels."""
    _check_dataset(dataset)
    return _RAW_LABELS[dataset](**kwargs)


def kernel_selection(dataset: str) -> dict[int, str]:
    """Return the class-index -> kernel-name map for the given dataset."""
    _check_dataset(dataset)
    return DATASET_KERNEL_SELECTION[dataset]
