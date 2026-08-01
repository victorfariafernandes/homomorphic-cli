"""Load serialized federated FHE models and evaluate on the held-out test set
(--dataset=iris or breast_cancer).

Reports per-class and OvR multiclass accuracy, precision, recall, and F1.
"""

from __future__ import annotations

import sys
from config.parallel import bootstrap as _init_parallel

_init_parallel()

import numpy as np

from federated_lssvm.solver_selection import (
    DEFAULT_SOLVER_NAME,
    parse_dataset_name,
    parse_solver_name,
    parse_models_root,
    resolve_solver_module,
)
from lssvm.preprocessing import (
    linear_kernel,
    polynomial_kernel,
    homogeneous_poly_kernel,
    poly_feature_map,
    homogeneous_poly_feature_map,
    preprocess_features,
)
from lssvm.preprocessors import (
    DEFAULT_DATASET,
    kernel_selection,
    prepare_binary,
    raw_test_labels,
)
from lssvm.solvers.utils import SPARSE_BOOTSTRAP_SLOTS
from config.metrics import precision, recall, f1_score, confusion_matrix

solv = None

_KERNEL_REGISTRY = {
    "linear": (linear_kernel, None, "primal:linear"),
    "poly": (polynomial_kernel, poly_feature_map, "primal:poly:degree=2:c=1.0"),
    "homo_poly": (
        homogeneous_poly_kernel,
        homogeneous_poly_feature_map,
        "primal:homo_poly:degree=2",
    ),
}


def build_class_kernels(dataset: str) -> dict:
    """Per-dataset class-index -> (name, kernel, feature_map, mode_str). Must match
    federated_lssvm.train exactly, or inference applies a different feature map."""
    return {
        idx: (name,) + _KERNEL_REGISTRY[name]
        for idx, name in kernel_selection(dataset).items()
    }


def _model_dir(k: int, class_idx: int, model_root: str | None = None) -> str:
    root = model_root if model_root is not None else f"models/k={k}"
    return f"{root}/class_{class_idx}"


def main(
    k: int = 20,
    solver_name: str | None = None,
    dataset: str = DEFAULT_DATASET,
    model_root: str | None = None,
) -> None:
    global solv
    solv = resolve_solver_module(solver_name or DEFAULT_SOLVER_NAME)

    class_kernels = build_class_kernels(dataset)
    splits = prepare_binary(dataset)
    n_test = len(splits[0][1])
    label = "OvR" if len(splits) > 1 else "binary"

    print(f"=== Federated FHE Inference  ({dataset} {label}, k={k}, n_test={n_test}) ===\n")

    y_test_raw = raw_test_labels(dataset)
    classifiers = []

    for class_idx, (_, X_te, _, y_te, name) in enumerate(splits):
        kernel_name, _, feature_map, _ = class_kernels.get(
            class_idx, ("linear", linear_kernel, None, "primal:linear")
        )
        model_dir = _model_dir(k, class_idx, model_root)
        phi_mean = np.load(f"{model_dir}/phi_mean.npy")
        # Class-mean-midpoint threshold saved at train time (0.0 back-compat default).
        try:
            threshold = float(np.load(f"{model_dir}/threshold.npy"))
        except FileNotFoundError:
            threshold = 0.0
        X_te_feat, _ = preprocess_features(X_te, feature_map, phi_mean=phi_mean)
        d = X_te_feat.shape[1]
        print(f"--- Class {class_idx} ({name}) ---")
        print(f"  Loading model from {model_dir}/ ...")

        cc, keys, b_ct, w_ct, mode_str, security = solv.load_global_checkpoint(
            model_dir, d=d, n_test=n_test
        )
        print(f"  Model loaded  [mode={mode_str}, security={security}]")

        slots = SPARSE_BOOTSTRAP_SLOTS if security == "128" else None
        # predict_cipher batches internally and returns plaintext scores.
        # Recalibrate by subtracting the saved threshold (zero FHE cost).
        scores = np.array(solv.predict_cipher(cc, keys, b_ct, w_ct, X_te_feat, slots=slots)) - threshold
        preds = np.sign(scores)
        preds[preds == 0] = 1.0

        acc = float(np.mean(preds == y_te) * 100)
        prec = precision(preds, y_te)
        rec = recall(preds, y_te)
        f1 = f1_score(preds, y_te)
        tp, fp, fn, tn = confusion_matrix(preds, y_te)

        print(f"  Accuracy : {acc:.2f}%")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")

        classifiers.append({"class_idx": class_idx, "scores": scores, "preds": preds, "y_te": y_te})

    # Binary (sign) or OvR multiclass (argmax) accuracy, on recalibrated scores
    if len(classifiers) == 1:
        c = classifiers[0]
        acc = float(np.mean(c["preds"] == c["y_te"]) * 100)
        print(f"Binary Accuracy: {acc:.2f}%")
    else:
        score_matrix = np.column_stack([c["scores"] for c in classifiers])
        class_indices = np.array([c["class_idx"] for c in classifiers])
        ovr_preds = class_indices[score_matrix.argmax(axis=1)]
        ovr_acc = float(np.mean(ovr_preds == y_test_raw) * 100)
        print(f"OvR Multiclass Accuracy: {ovr_acc:.2f}%")


if __name__ == "__main__":
    k = 20
    args = [a for a in sys.argv[1:] if a.lstrip("-").isdigit()]
    if args:
        k = int(args[0])
    solver_name = parse_solver_name(sys.argv[1:])
    dataset = parse_dataset_name(sys.argv[1:])
    models_root = parse_models_root(sys.argv[1:])
    model_root = f"{models_root}/k={k}" if models_root != "models" else None
    main(k=k, solver_name=solver_name, dataset=dataset, model_root=model_root)
