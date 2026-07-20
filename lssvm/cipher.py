"""LSSVM solver over CKKS-encrypted data using Householder QR.

Pipeline:
  1. Subsample Iris to keep H small enough for FHE.
  2. Apply feature map for the selected kernel (polynomial kernels expand features).
  3. Call solver: encrypt H, run Householder QR + Q^T@rhs + back-sub + primal weights → (b_ct, w_ct).
  4. Predict on the full test set using encrypted primal weights (no training data needed).
  5. Serialize model (crypto context, keys, b_ct, w_ct, mode) to disk.
"""

from __future__ import annotations

import sys
import time
import importlib
import numpy as np
from config.parallel import bootstrap as _init_parallel

_init_parallel()


from lssvm.solvers.utils import depth_for_size, SPARSE_BOOTSTRAP_SLOTS
from lssvm.preprocessing import (
    prepare_iris_binary,
    build_lssvm_matrix,
    linear_kernel,
    polynomial_kernel,
    homogeneous_poly_kernel,
    poly_feature_map,
    homogeneous_poly_feature_map,
    center_features,
    normalize_features,
)
from lssvm.plain import predict_lssvm
from config.metrics import print_class_report

solver_name = sys.argv[1] if len(sys.argv) > 1 else "qr_householder_cipher_row"
solv = importlib.import_module(f"lssvm.solvers.{solver_name}")

# cg_cipher does not support the security="128" bootstrapping path (out of scope for that
# solver); only the Householder QR solvers do. Fall back to "notset" (its only mode) there.
_SUPPORTS_BOOTSTRAP_SECURITY = hasattr(solv, "SPARSE_BOOTSTRAP_SLOTS") or solver_name in (
    "qr_householder_cipher_row",
    "qr_householder_cipher_col",
)

# ── configuration ──────────────────────────────────────────────────
N_PER_CLASS = 2  # samples per binary class → H is (2*N+1) x (2*N+1)
D_SQRT = 2  # Chebyshev degree for sqrt (QR step)
D_INV = 2  # Chebyshev degree for 1/t (QR step — vtv reciprocal)
D_INV_BACKSUB = 8  # Chebyshev degree for 1/t in back-sub
DEPTH_SAFETY = 1.15  # calibrated FLEXIBLEAUTO safety factor
DEPTH_OVERRIDE = None  # set int to bypass estimator during experimentation
N_OVERRIDE = None  # set int to force ring dimension (None = auto-scale from depth)
GAMMA = 1.0  # regularisation
SECURITY = "128" if _SUPPORTS_BOOTSTRAP_SECURITY else "notset"

# ── Kernel selection per OvR class ────────────────────────────────
# Available options:
#   "linear"     → linear kernel,                   no feature expansion
#   "poly"       → polynomial kernel (x·y + c)^d,   feature dim expands to C(d+deg, deg)
#   "homo_poly"  → homogeneous polynomial (x·y)^d,  feature dim expands to C(d+deg-1, deg)
#
# Change the value for each class index to switch kernels.
CLASS_KERNEL_SELECTION = {
    0: "linear",
    1: "linear",
    2: "linear",
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


def subsample_for_fhe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select n_per_class samples from each binary class (+1 / -1)."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_train == 1.0)[0]
    idx_neg = np.where(y_train == -1.0)[0]

    n_pos = min(n_per_class, len(idx_pos))
    n_neg = min(n_per_class, len(idx_neg))
    if n_pos < n_per_class or n_neg < n_per_class:
        print(f"  WARNING: not enough samples, using {n_pos} pos / {n_neg} neg")

    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    sel = np.sort(np.concatenate([sel_pos, sel_neg]))

    return X_train[sel], y_train[sel]


def main():
    splits = prepare_iris_binary()
    classifiers = []

    # All sub-problems have the same raw H size before feature expansion
    n_raw = 2 * N_PER_CLASS + 1
    depth = depth_for_size(
        n_raw,
        n_raw,
        D_SQRT,
        D_INV,
        D_INV_BACKSUB,
        safety_factor=DEPTH_SAFETY,
        depth_override=DEPTH_OVERRIDE,
    )
    n_test = len(splits[0][1]) if splits else n_raw

    # Compute max feature dimension across all kernels (needed for rotation key generation)
    sample_X = splits[0][0][:1]  # one sample for feature map sizing
    max_feat_dim = n_raw  # fallback
    for _, _, feature_map_fn, _ in CLASS_KERNELS.values():
        if feature_map_fn is not None:
            d = feature_map_fn(sample_X).shape[1]
        else:
            d = sample_X.shape[1]
        max_feat_dim = max(max_feat_dim, d)

    print("=== LSSVM FHE Solver (Iris OvR) ===")
    print(
        f"Gamma={GAMMA}  N_per_class={N_PER_CLASS}  H size={n_raw}x{n_raw}  depth={depth}"
    )
    print(
        f"Chebyshev degrees: sqrt={D_SQRT}, inv_qr={D_INV}, inv_backsub={D_INV_BACKSUB}"
    )
    print(f"Max feature dim (after kernel map): {max_feat_dim}\n")

    print(f"Setting up crypto context (security={SECURITY}) ...")
    t_ctx = time.perf_counter()
    security_kwargs = {"security": SECURITY} if _SUPPORTS_BOOTSTRAP_SECURITY else {}
    cc, keys = solv.setup_crypto_context(
        depth, matrix_size=n_raw, n_test=n_test, feature_dim=max_feat_dim, N=N_OVERRIDE,
        **security_kwargs,
    )
    slots = SPARSE_BOOTSTRAP_SLOTS if SECURITY == "128" else None
    slot_count = solv.get_slot_count(cc)
    print(
        f"Context ready in {time.perf_counter() - t_ctx:.1f}s  (slots={slot_count})\n"
    )

    for class_idx, (X_tr, X_te, y_tr, y_te, name) in enumerate(splits):
        kernel_name, _, feature_map, mode_str = CLASS_KERNELS.get(
            class_idx, ("linear", linear_kernel, None, "primal:linear")
        )
        print(f"--- Class {class_idx} ({name} vs rest) [kernel={kernel_name}] ---")

        t_sub = time.perf_counter()
        X_sub, y_sub = subsample_for_fhe(X_tr, y_tr, N_PER_CLASS)
        print(f"  Subsampled {len(y_sub)} points in {time.perf_counter() - t_sub:.3f}s")

        # Apply feature map if needed (polynomial kernels expand to higher-dim space)
        if feature_map is not None:
            X_sub_feat = feature_map(X_sub)
            X_te_feat = feature_map(X_te)
        else:
            X_sub_feat = X_sub
            X_te_feat = X_te
        X_sub_feat, phi_mean = center_features(X_sub_feat)
        X_te_feat,  _        = center_features(X_te_feat, phi_mean=phi_mean)
        X_sub_feat = normalize_features(X_sub_feat)
        X_te_feat  = normalize_features(X_te_feat)

        t_mat = time.perf_counter()
        # Build H using linear kernel on mapped features (equivalent to kernel on raw)
        H_np, rhs_np = build_lssvm_matrix(X_sub_feat, y_sub, GAMMA)
        n = H_np.shape[0]
        print(
            f"  Built H ({H_np.shape[0]}x{H_np.shape[1]}), cond={np.linalg.cond(H_np):.1f}"
            f"in {time.perf_counter() - t_mat:.3f}s"
        )
        H_list = H_np.tolist()
        rhs_list = rhs_np.tolist()

        # ── FHE solve + primal weight computation ──
        print(f"  Starting FHE QR solve ({n}x{n}, depth={depth}) ...")
        t0 = time.perf_counter()
        b_ct, w_ct, _ = solv.solver(
            cc,
            keys,
            H_list,
            rhs_list,
            X_sub_feat,
            y_sub,
            D_sqrt=D_SQRT,
            D_inv=D_INV,
            D_inv_backsub=D_INV_BACKSUB,
            **security_kwargs,
        )
        elapsed = time.perf_counter() - t0
        print(f"  FHE QR solve + primal weights: {elapsed:.1f}s")

        # ── plaintext reference — same subsample ──
        sol_plain = np.linalg.solve(H_np, rhs_np)
        b_plain = sol_plain[0]
        alpha_plain = sol_plain[1:]
        w_plain_sub = alpha_plain @ (y_sub[:, None] * X_sub_feat)
        preds_plain, _ = predict_lssvm(
            X_te_feat, X_sub_feat, alpha_plain, y_sub, b_plain
        )

        # ── plaintext reference — full training set ──
        X_tr_feat = feature_map(X_tr) if feature_map is not None else X_tr
        X_tr_feat, phi_mean_full = center_features(X_tr_feat)
        X_tr_feat = normalize_features(X_tr_feat)
        H_full_np, rhs_full_np = build_lssvm_matrix(X_tr_feat, y_tr, GAMMA)
        sol_full = np.linalg.solve(H_full_np, rhs_full_np)
        alpha_full = sol_full[1:]
        w_plain_full = alpha_full @ (y_tr[:, None] * X_tr_feat)

        # ── cipher prediction (primal, no training data needed) ──
        t_cp = time.perf_counter()
        predict_kwargs = {"slots": slots} if _SUPPORTS_BOOTSTRAP_SECURITY else {}
        # predict_cipher batches internally and returns plaintext scores.
        scores_cipher = np.array(
            solv.predict_cipher(cc, keys, b_ct, w_ct, X_te_feat, **predict_kwargs)
        )
        print(f"  Cipher predict (primal): {time.perf_counter() - t_cp:.4f}s")
        preds_cipher = np.sign(scores_cipher)
        preds_cipher[preds_cipher == 0] = 1.0

        # ── decrypt primal weights ──
        d = X_sub_feat.shape[1]
        w_fhe = np.array(solv.decrypt_vector(cc, keys, w_ct, d))

        print_class_report(
            class_idx,
            name,
            preds_cipher,
            preds_plain,
            y_te,
            w_fhe,
            w_plain_sub,
            w_plain_full,
        )

        # ── serialize model ──
        out_dir = f"models/class_{class_idx}"
        solv.serialize_model(cc, keys, b_ct, w_ct, out_dir, mode_str=mode_str, **security_kwargs)
        np.save(f"{out_dir}/phi_mean.npy", phi_mean)
        print(f"  Model serialized to {out_dir}/  [{mode_str}]")
        print()

        classifiers.append(
            {
                "class_idx": class_idx,
                "scores": scores_cipher,
            }
        )

    # ── OvR multiclass accuracy ──
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    _, _, _, y_test_raw = train_test_split(
        iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
    )
    score_matrix = np.column_stack([c["scores"] for c in classifiers])
    class_indices = np.array([c["class_idx"] for c in classifiers])
    predicted = class_indices[score_matrix.argmax(axis=1)]
    ovr_acc = np.mean(predicted == y_test_raw) * 100
    print(f"OvR Multiclass Accuracy (FHE): {ovr_acc:.2f}%")


if __name__ == "__main__":
    main()
