"""Load serialized federated FHE models and evaluate on the Iris test set.

Reports per-class and OvR multiclass accuracy, precision, recall, and F1.
"""

from __future__ import annotations

import sys
from config.parallel import bootstrap as _init_parallel

_init_parallel()

import numpy as np

from federated_lssvm.solver_selection import (
    DEFAULT_SOLVER_NAME,
    parse_solver_name,
    resolve_solver_module,
)
from lssvm.preprocessing import (
    prepare_iris_binary,
    linear_kernel,
    polynomial_kernel,
    homogeneous_poly_kernel,
    poly_feature_map,
    homogeneous_poly_feature_map,
    preprocess_features,
)
from config.metrics import precision, recall, f1_score, confusion_matrix

solv = None

# Must match federated_lssvm.train.CLASS_KERNEL_SELECTION exactly, otherwise
# inference applies a different feature map than training.
CLASS_KERNEL_SELECTION = {0: "linear", 1: "poly", 2: "poly"}
_KERNEL_REGISTRY = {
    "linear": (linear_kernel, None, "primal:linear"),
    "poly": (polynomial_kernel, poly_feature_map, "primal:poly:degree=2:c=1.0"),
    "homo_poly": (
        homogeneous_poly_kernel,
        homogeneous_poly_feature_map,
        "primal:homo_poly:degree=2",
    ),
}
CLASS_KERNELS = {
    idx: (name,) + _KERNEL_REGISTRY[name]
    for idx, name in CLASS_KERNEL_SELECTION.items()
}


def main(k: int = 20, solver_name: str | None = None) -> None:
    global solv
    solv = resolve_solver_module(solver_name or DEFAULT_SOLVER_NAME)

    splits = prepare_iris_binary()
    n_test = len(splits[0][1])

    print(f"=== Federated FHE Inference  (k={k}, n_test={n_test}) ===\n")

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split as tts

    iris = load_iris()
    _, _, _, y_test_raw = tts(
        iris.data,
        iris.target,
        test_size=0.2,
        stratify=iris.target,
        random_state=42,
    )

    all_scores = []

    for class_idx, (_, X_te, _, y_te, name) in enumerate(splits):
        kernel_name, _, feature_map, _ = CLASS_KERNELS.get(
            class_idx, ("linear", linear_kernel, None, "primal:linear")
        )
        model_dir = f"models/k={k}/class_{class_idx}"
        phi_mean = np.load(f"{model_dir}/phi_mean.npy")
        X_te_feat, _ = preprocess_features(X_te, feature_map, phi_mean=phi_mean)
        d = X_te_feat.shape[1]
        print(f"--- Class {class_idx} ({name} vs rest) ---")
        print(f"  Loading model from {model_dir}/ ...")

        cc, keys, b_ct, w_ct, mode_str = solv.load_global_checkpoint(
            model_dir, d=d, n_test=n_test
        )
        print(f"  Model loaded  [mode={mode_str}]")

        scores_ct = solv.predict_cipher(cc, keys, b_ct, w_ct, X_te_feat)
        scores = np.array(solv.decrypt_vector(cc, keys, scores_ct, n_test))
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

        all_scores.append(scores)

    # OvR multiclass
    score_matrix = np.column_stack(all_scores)
    ovr_preds = score_matrix.argmax(axis=1)
    ovr_acc = float(np.mean(ovr_preds == y_test_raw) * 100)
    print(f"OvR Multiclass Accuracy: {ovr_acc:.2f}%")


if __name__ == "__main__":
    k = 20
    args = [a for a in sys.argv[1:] if a.lstrip("-").isdigit()]
    if args:
        k = int(args[0])
    solver_name = parse_solver_name(sys.argv[1:])
    main(k=k, solver_name=solver_name)
