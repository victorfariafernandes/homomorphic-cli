"""Plaintext FedAvg runner for federated LSSVM (no FHE).

Usage:
    python -m federated_lssvm.plain_run [k]

Defaults to k=3 clients.
"""
from __future__ import annotations

import sys
import time
import numpy as np
from lssvm.preprocessing import (
    prepare_iris_binary,
    build_lssvm_matrix,
    linear_kernel,
    polynomial_kernel,
    homogeneous_poly_kernel,
    poly_feature_map,
    homogeneous_poly_feature_map,
    preprocess_features,
    gcv_gamma,
)
from lssvm.plain import predict_lssvm

# setosa is linearly separable — linear kernel gives 100% binary accuracy at γ=1.1.
# versicolor/virginica overlap; poly kernel + GCV-tuned γ gives 90% each → 96.67% OvR.
CLASS_KERNEL_SELECTION = {0: "linear", 1: "poly", 2: "poly"}

# Fallback gamma values (GCV overrides at run-time for poly classes).
# For linear/setosa: γ=1.1 is manually chosen — GCV minimizes regression loss, not
# classification accuracy, and finds a suboptimal γ for perfectly separable data.
KERNEL_GAMMA = {
    "linear":    1.1,
    "poly":      1.27,
    "homo_poly": 1.27,
}

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


def partition_all(X: np.ndarray, y: np.ndarray, k: int, base_seed: int = 42):
    rng = np.random.default_rng(base_seed)
    pos_idx = np.where(y == 1.0)[0].copy()
    neg_idx = np.where(y == -1.0)[0].copy()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_chunks = np.array_split(pos_idx, k)
    neg_chunks = np.array_split(neg_idx, k)
    partitions = []
    for i in range(k):
        indices = np.sort(np.concatenate([pos_chunks[i], neg_chunks[i]]))
        partitions.append((X[indices], y[indices]))
    return partitions


def plaintext_federated_reference(partitions_feat, X_te_feat, gamma: float):
    """Returns (preds, w_avg, b_avg, client_times_s, inference_time_s)."""
    w_sum = None
    b_sum = 0.0
    client_times = []
    for X_c, y_c in partitions_feat:
        t0 = time.perf_counter()
        H, rhs = build_lssvm_matrix(X_c, y_c, gamma)
        try:
            sol = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(H, rhs, rcond=None)[0]
        client_times.append(time.perf_counter() - t0)
        b_i = sol[0]
        alpha_i = sol[1:]
        w_i = X_c.T @ (alpha_i * y_c)
        b_sum += b_i
        w_sum = w_i if w_sum is None else w_sum + w_i

    k = len(partitions_feat)
    w_avg = w_sum / k
    b_avg = b_sum / k

    t_inf = time.perf_counter()
    scores = X_te_feat @ w_avg + b_avg
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    inference_time = time.perf_counter() - t_inf

    return preds, w_avg, b_avg, client_times, inference_time


def main(k: int = 3):
    splits = prepare_iris_binary()
    n_test = len(splits[0][1])
    print(f"Plaintext Federated LSSVM (Iris OvR, k={k})")
    classifiers_plain_fed = []
    classifiers_full = []

    # Timing accumulators
    t_train_total_start = time.perf_counter()
    class_train_times: list[float] = []
    class_infer_times: list[float] = []
    all_client_times: list[list[float]] = []

    for class_idx, (X_tr, X_te, y_tr, y_te, name) in enumerate(splits):
        kernel_name, _, feature_map, mode_str = CLASS_KERNELS.get(
            class_idx, ("linear", linear_kernel, None, "primal:linear")
        )

        if feature_map is not None:
            best_gamma, gcv_val = gcv_gamma(X_tr, y_tr, feature_map)
            print(
                f"Class {class_idx} ({name}) — GCV: γ={best_gamma:.6f}  GCV={gcv_val:.6f}"
            )
        else:
            best_gamma = KERNEL_GAMMA[kernel_name]
            print(
                f"Class {class_idx} ({name}) — linear kernel: using fixed γ={best_gamma:.4f}"
            )
        gamma = best_gamma
        # Single per-class preprocessing path shared with the FHE solver:
        # linear → raw features; poly → featuremap + center + unit-L2 norm.
        X_tr_feat, phi_mean = preprocess_features(X_tr, feature_map)
        X_te_feat, _        = preprocess_features(X_te, feature_map, phi_mean=phi_mean)

        t_class_start = time.perf_counter()

        # Full-data plaintext reference
        H_full, rhs_full = build_lssvm_matrix(X_tr_feat, y_tr, gamma)
        print(f"  Full-data H={H_full.shape}, cond={np.linalg.cond(H_full):.2f} ...")
        try:
            sol_full = np.linalg.solve(H_full, rhs_full)
        except np.linalg.LinAlgError:
            sol_full = np.linalg.lstsq(H_full, rhs_full, rcond=None)[0]
        alpha_full = sol_full[1:]
        preds_plain_full, scores_full = predict_lssvm(
            X_te_feat, X_tr_feat, alpha_full, y_tr, sol_full[0]
        )

        # Partition and per-client plaintext solves
        parts = partition_all(X_tr_feat, y_tr, k)
        parts_feat = [(Xc, yc) for (Xc, yc) in parts]
        for client_id, (Xc, yc) in enumerate(parts_feat):
            H_c, rhs_c = build_lssvm_matrix(Xc, yc, gamma)
            print(f"  [client {client_id}] H={H_c.shape}, cond={np.linalg.cond(H_c):.2f} ...")

        preds_plain_fed, w_plain_fed, _, client_times, infer_time = (
            plaintext_federated_reference(parts_feat, X_te_feat, gamma)
        )

        t_class_end = time.perf_counter()
        class_train_times.append(t_class_end - t_class_start)
        class_infer_times.append(infer_time)
        all_client_times.append(client_times)

        classifiers_plain_fed.append({"class_idx": class_idx, "scores": X_te_feat @ w_plain_fed})
        classifiers_full.append({"class_idx": class_idx, "scores": scores_full})

        def acc(p: np.ndarray) -> float:
            return float(np.mean(p == y_te) * 100)

        def precision(p: np.ndarray) -> float:
            tp = float(np.sum((p == 1.0) & (y_te == 1.0)))
            fp = float(np.sum((p == 1.0) & (y_te != 1.0)))
            if tp + fp == 0:
                return 0.0
            return float(tp / (tp + fp) * 100)

        def f1(p: np.ndarray) -> float:
            tp = float(np.sum((p == 1.0) & (y_te == 1.0)))
            fp = float(np.sum((p == 1.0) & (y_te != 1.0)))
            fn = float(np.sum((p != 1.0) & (y_te == 1.0)))
            if tp == 0:
                return 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec + rec == 0:
                return 0.0
            return float(2 * prec * rec / (prec + rec) * 100)

        print(f"Class {class_idx} ({name}):")
        print(f"  {'Approach':<36} | Accuracy | Precision |   F1  ")
        print(f"  {'-'*36}-+---------+-----------+-------")
        print(
            f"  {'Plain FedAvg':<36} | {acc(preds_plain_fed):7.2f}% | {precision(preds_plain_fed):9.2f}% | {f1(preds_plain_fed):6.2f}%"
        )
        print(
            f"  {'Full-data (reference)':<36} | {acc(preds_plain_full):7.2f}% | {precision(preds_plain_full):9.2f}% | {f1(preds_plain_full):6.2f}%"
        )

        # Per-client timing for this class
        print(f"  Timing — class {class_idx} training: {class_train_times[-1]*1e3:.3f} ms  "
              f"inference: {infer_time*1e6:.1f} µs")
        for cid, ct in enumerate(client_times):
            print(f"    [client {cid}] solve: {ct*1e6:.1f} µs")

    t_train_total = time.perf_counter() - t_train_total_start

    # OvR multiclass accuracy using score matrices
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

    def _ovr_acc(classifiers):
        score_matrix = np.column_stack([c["scores"] for c in classifiers])
        class_indices = np.array([c["class_idx"] for c in classifiers])
        predicted = class_indices[score_matrix.argmax(axis=1)]
        return np.mean(predicted == y_test_raw) * 100

    print(f"OvR Multiclass Accuracy (Federated plaintext, k={k}): {_ovr_acc(classifiers_plain_fed):.2f}%")

    # ── Timing summary ──────────────────────────────────────────────
    print(f"\n=== Timing Summary (k={k}) ===")
    print(f"  Total training wall time : {t_train_total*1e3:.3f} ms")
    for ci, (ct, it) in enumerate(zip(class_train_times, class_infer_times)):
        print(f"  Class {ci} — training: {ct*1e3:.3f} ms  |  inference: {it*1e6:.1f} µs")
        times = all_client_times[ci]
        avg_ct = sum(times) / len(times)
        print(f"    clients: min={min(times)*1e6:.1f} µs  "
              f"avg={avg_ct*1e6:.1f} µs  max={max(times)*1e6:.1f} µs  "
              f"total={sum(times)*1e3:.3f} ms")


if __name__ == "__main__":
    args = sys.argv[1:]
    numeric_args = [a for a in args if a.lstrip("-").isdigit()]
    k = int(numeric_args[0]) if numeric_args else 3
    main(k=k)
