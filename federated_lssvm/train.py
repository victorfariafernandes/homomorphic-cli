"""Federated LSSVM training via One-Shot FedAvg over CKKS-encrypted weights.

Pipeline:
  1. Load Iris (120 train / 30 test, stratified split).
  2. Partition all 120 training samples disjointly across k clients.
  3. Each client trains a local FHE LS-SVM on their partition.
  4. Server aggregates encrypted weight vectors: w_global = (1/k) * sum(w_ct_i).
  5. Evaluate global model on the 30-sample test set.
  6. Serialize to models/k={k}/class_{i}/.
"""

from __future__ import annotations

import sys
from config.parallel import bootstrap as _init_parallel

_init_parallel()

import os
import time
import numpy as np
from openfhe import (
    SerializeToFile,
    DeserializeCiphertext,
    DeserializeCryptoContext,
    DeserializePrivateKey,
    DeserializePublicKey,
    BINARY,
)

from federated_lssvm.solver_selection import (
    DEFAULT_SOLVER_NAME,
    DEFAULT_SECURITY_LEVEL,
    parse_solver_name,
    parse_security_level,
    resolve_solver_module,
)
from lssvm.solvers.utils import (
    depth_for_size,
    SPARSE_BOOTSTRAP_SLOTS,
    write_security_marker,
    read_security_marker,
)
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
from config.metrics import weight_relative_error

# ── solver backend ─────────────────────────────────────────────────
solv = None

# ── configuration ──────────────────────────────────────────────────
D_SQRT = 2
D_INV = 2
D_INV_BACKSUB = 8
DEPTH_SAFETY = 1.20
DEPTH_OVERRIDE = None
N_OVERRIDE = None
KERNEL_GAMMA = {
    "linear":    1.1,   # fixed — GCV suboptimal for linearly separable setosa
    "poly":      1.27,  # GCV overrides at run-time; fallback only
    "homo_poly": 1.27,
}
N_PER_CLASS_BASELINE = (
    2  # used only for the single-client FHE baseline (matches lssvm_cipher.py)
)

# ── kernel registry (local copy — avoids lssvm_cipher sys.argv side effect) ──
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


# ── per-client checkpoint helpers ──────────────────────────────────

CLASS_CHECKPOINT_SCHEMA_VERSION = 1


def _class_checkpoint_marker(class_dir: str) -> str:
    return f"{class_dir}/checkpoint.json"


def _read_checkpoint_depth(class_dir: str) -> int | None:
    marker = _class_checkpoint_marker(class_dir)
    if not os.path.exists(marker):
        return None
    with open(marker, encoding="utf-8") as f:
        for line in f:
            if line.startswith("depth="):
                try:
                    return int(line.strip().split("=", 1)[1])
                except ValueError:
                    return None
    return None


def _class_context_exists(class_dir: str, required_depth: int) -> bool:
    stored = _read_checkpoint_depth(class_dir)
    return stored is not None and stored >= required_depth


def _cts_exist(out_dir: str) -> bool:
    return os.path.exists(f"{out_dir}/bias.bin") and os.path.exists(
        f"{out_dir}/weights.bin"
    )


def _save_cts(out_dir: str, cc, keys, b_ct, w_ct) -> None:
    os.makedirs(out_dir, exist_ok=True)
    assert SerializeToFile(
        f"{out_dir}/bias.bin", b_ct, BINARY
    ), f"Failed to serialize bias to {out_dir}"
    assert SerializeToFile(
        f"{out_dir}/weights.bin", w_ct, BINARY
    ), f"Failed to serialize weights to {out_dir}"


def _save_class_context(out_dir: str, cc, keys, depth: int, security: str = "128") -> None:
    os.makedirs(out_dir, exist_ok=True)
    assert SerializeToFile(
        f"{out_dir}/cryptocontext.bin", cc, BINARY
    ), f"Failed to serialize crypto context to {out_dir}"
    assert SerializeToFile(
        f"{out_dir}/public_key.bin", keys.publicKey, BINARY
    ), f"Failed to serialize public key to {out_dir}"
    assert SerializeToFile(
        f"{out_dir}/secret_key.bin", keys.secretKey, BINARY
    ), f"Failed to serialize secret key to {out_dir}"
    with open(_class_checkpoint_marker(out_dir), "w", encoding="utf-8") as f:
        f.write(f"schema_version={CLASS_CHECKPOINT_SCHEMA_VERSION}\n")
        f.write(f"depth={depth}\n")
    write_security_marker(out_dir, security)


def _load_class_context(out_dir: str, matrix_size: int, n_test: int, feature_dim: int):
    cc, ok = DeserializeCryptoContext(f"{out_dir}/cryptocontext.bin", BINARY)
    assert ok, f"Failed to deserialize crypto context from {out_dir}"

    pk, ok = DeserializePublicKey(f"{out_dir}/public_key.bin", BINARY)
    assert ok, f"Failed to deserialize public key from {out_dir}"

    sk, ok = DeserializePrivateKey(f"{out_dir}/secret_key.bin", BINARY)
    assert ok, f"Failed to deserialize secret key from {out_dir}"

    cc.EvalMultKeyGen(sk)
    rot_indices_fn = getattr(solv, "_rotation_indices", None)
    if rot_indices_fn is None:
        raise AttributeError(
            f"Solver module '{solv.__name__}' does not expose rotation key indices"
        )
    rot_indices = rot_indices_fn(matrix_size, n_test, feature_dim=feature_dim)
    cc.EvalRotateKeyGen(sk, rot_indices)

    class _Keys:
        pass

    keys = _Keys()
    keys.publicKey = pk
    keys.secretKey = sk
    return cc, keys


def _load_cts(out_dir: str):
    b_ct, ok = DeserializeCiphertext(f"{out_dir}/bias.bin", BINARY)
    assert ok, f"Failed to deserialize bias from {out_dir}"
    w_ct, ok = DeserializeCiphertext(f"{out_dir}/weights.bin", BINARY)
    assert ok, f"Failed to deserialize weights from {out_dir}"
    return b_ct, w_ct


def _cts_are_finite(cc, keys, b_ct, w_ct, d: int) -> bool:
    try:
        b_val = solv.decrypt_vector(cc, keys, b_ct, 1)[0]
        w_vals = np.array(solv.decrypt_vector(cc, keys, w_ct, d))
    except Exception:
        return False
    return np.isfinite(b_val) and np.all(np.isfinite(w_vals))


# ── subsample helper (inline copy from lssvm_cipher.py) ────────────
def _subsample_for_fhe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_train == 1.0)[0]
    idx_neg = np.where(y_train == -1.0)[0]
    n_pos = min(n_per_class, len(idx_pos))
    n_neg = min(n_per_class, len(idx_neg))
    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    sel = np.sort(np.concatenate([sel_pos, sel_neg]))
    return X_train[sel], y_train[sel]


# ── core federated functions ────────────────────────────────────────


def partition_all(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    base_seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Distribute all training samples disjointly across k clients.

    Uses np.array_split so remainders are spread evenly (some clients get
    one extra sample). Each client's partition preserves the pos/neg ratio.
    """
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


def fhe_aggregate(cc, b_cts: list, w_cts: list) -> tuple:
    """Average encrypted client models via FedAvg inside CKKS.

    Only uses EvalAdd (noise-cheap) and one EvalMult by plaintext scalar 1/k.
    No decryption is performed.
    """
    k = len(b_cts)

    b_sum = b_cts[0]
    for ct in b_cts[1:]:
        b_sum = cc.EvalAdd(b_sum, ct)
    b_global = cc.EvalMult(b_sum, 1.0 / k)

    w_sum = w_cts[0]
    for ct in w_cts[1:]:
        w_sum = cc.EvalAdd(w_sum, ct)
    w_global = cc.EvalMult(w_sum, 1.0 / k)

    return b_global, w_global


def _report_depth(
    stage: str,
    budget: int,
    ciphertexts: dict[str, object],
    *,
    indent: int = 2,
) -> None:
    """Print a one-line depth-consumption report for each named ciphertext.

    stage    : label printed before the table (e.g. "client 0 solve", "aggregation")
    budget   : multiplicative depth the crypto context was created with
    ciphertexts : {label: ciphertext} — each ct must expose .GetLevel()
    indent   : leading spaces
    """
    pad = " " * indent
    rows = []
    for label, ct in ciphertexts.items():
        consumed = ct.GetLevel()
        remaining = budget - consumed
        pct = consumed / budget * 100 if budget > 0 else 0.0
        rows.append((label, consumed, remaining, pct))

    header = f"{pad}[depth] {stage}  (budget={budget})"
    col_w = max(len(r[0]) for r in rows) + 2
    print(header)
    for label, consumed, remaining, pct in rows:
        bar_len = 20
        filled = round(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"{pad}  {label:<{col_w}} consumed={consumed:3d}  remaining={remaining:3d}"
            f"  [{bar}] {pct:5.1f}%"
        )


def plaintext_federated_reference(
    partitions_feat: list[tuple[np.ndarray, np.ndarray]],
    X_te_feat: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute FedAvg in plaintext numpy for validation.

    Returns (predictions, w_avg, b_avg).
    """
    w_sum = None
    b_sum = 0.0
    for X_c, y_c in partitions_feat:
        H, rhs = build_lssvm_matrix(X_c, y_c, gamma)
        try:
            sol = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(H, rhs, rcond=None)[0]
        b_i = sol[0]
        alpha_i = sol[1:]
        w_i = X_c.T @ (alpha_i * y_c)
        b_sum += b_i
        w_sum = w_i if w_sum is None else w_sum + w_i

    k = len(partitions_feat)
    w_avg = w_sum / k
    b_avg = b_sum / k
    scores = X_te_feat @ w_avg + b_avg
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    return preds, w_avg, b_avg


def smoke_test_fedavg(security: str = DEFAULT_SECURITY_LEVEL) -> None:
    """Validate FedAvg aggregation on two synthetic 7x7 LSSVM systems.

    Uses only 7x7 matrices (6 training samples → 7x7 H) to stay within the
    memory budget. Builds LSSVM-structured matrices via build_lssvm_matrix so
    the FHE solver's sign=+1 Householder assumption holds. No Iris data needed.
    Callable via --smoke-test flag.
    """
    print("=== smoke_test_fedavg: two synthetic 7x7 LSSVM systems ===")

    n_per_class = 3  # 3 pos + 3 neg = 6 training samples → 7x7 H
    n_feat = 4  # same dimensionality as Iris linear features

    # Synthetic client datasets with fixed seeds for reproducibility
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(10)

    X_c1 = rng0.random((2 * n_per_class, n_feat))
    y_c1 = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    gamma = KERNEL_GAMMA["linear"]
    H1_np, rhs1_np = build_lssvm_matrix(X_c1, y_c1, gamma)

    X_c2 = rng1.random((2 * n_per_class, n_feat))
    y_c2 = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    H2_np, rhs2_np = build_lssvm_matrix(X_c2, y_c2, gamma)

    # Plaintext expected averages of FedAvg
    sol1 = np.linalg.solve(H1_np, rhs1_np)
    sol2 = np.linalg.solve(H2_np, rhs2_np)

    b1, alpha1 = sol1[0], sol1[1:]
    b2, alpha2 = sol2[0], sol2[1:]
    w1 = X_c1.T @ (alpha1 * y_c1)
    w2 = X_c2.T @ (alpha2 * y_c2)

    expected_b_avg = (b1 + b2) / 2.0
    expected_w_avg = (w1 + w2) / 2.0  # shape: (n_feat,)

    # Set up 7x7 crypto context with feature_dim=n_feat for rotation keys
    depth = depth_for_size(
        7,
        7,
        D_SQRT,
        D_INV,
        D_INV_BACKSUB,
        safety_factor=DEPTH_SAFETY,
        depth_override=DEPTH_OVERRIDE,
    )
    if security == "128":
        # The bootstrapping path uses its own fixed budget (see setup_crypto_context).
        depth = solv._BOOTSTRAP_USABLE_DEPTH + solv._BOOTSTRAP_SKIP_BELOW_LEVEL
    print(f"  Context depth={depth} for 7x7 matrix (security={security}) ...")
    t0 = time.perf_counter()
    cc, keys = solv.setup_crypto_context(
        depth, matrix_size=7, n_test=1, feature_dim=n_feat, N=N_OVERRIDE, security=security
    )
    print(f"  Context ready in {time.perf_counter() - t0:.1f}s")

    # FHE solve for each synthetic client
    print("  FHE solve client 1 ...")
    t1 = time.perf_counter()
    b_ct1, w_ct1, _ = solv.solver(
        cc,
        keys,
        H1_np.tolist(),
        rhs1_np.tolist(),
        X_c1,
        y_c1,
        D_sqrt=D_SQRT,
        D_inv=D_INV,
        D_inv_backsub=D_INV_BACKSUB,
        security=security,
    )
    print(f"  Client 1 done in {time.perf_counter() - t1:.1f}s")

    print("  FHE solve client 2 ...")
    t2 = time.perf_counter()
    b_ct2, w_ct2, _ = solv.solver(
        cc,
        keys,
        H2_np.tolist(),
        rhs2_np.tolist(),
        X_c2,
        y_c2,
        D_sqrt=D_SQRT,
        D_inv=D_INV,
        D_inv_backsub=D_INV_BACKSUB,
        security=security,
    )
    print(f"  Client 2 done in {time.perf_counter() - t2:.1f}s")

    # FedAvg aggregation (the function under test)
    b_global, w_global = fhe_aggregate(cc, [b_ct1, b_ct2], [w_ct1, w_ct2])

    # Decrypt and compare against plaintext average
    w_fhe_avg = np.array(solv.decrypt_vector(cc, keys, w_global, n_feat))
    b_fhe_avg = solv.decrypt_vector(cc, keys, b_global, 1)[0]

    w_err = float(
        np.linalg.norm(w_fhe_avg - expected_w_avg)
        / (np.linalg.norm(expected_w_avg) + 1e-15)
    )
    # Combined [b, w] relative error avoids inflated ratios when b ≈ 0
    fhe_full = np.concatenate([[b_fhe_avg], w_fhe_avg])
    expected_full = np.concatenate([[expected_b_avg], expected_w_avg])
    full_err = float(
        np.linalg.norm(fhe_full - expected_full)
        / (np.linalg.norm(expected_full) + 1e-15)
    )

    print(f"  bias  (expected={expected_b_avg:.4f}, fhe={b_fhe_avg:.4f})")
    print(f"  weight relative error: {w_err:.2e}")
    print(f"  full [b,w] relative error: {full_err:.2e}")

    assert w_err < 0.05, f"Smoke test FAILED: weight error {w_err:.2e} > 5%"
    assert full_err < 0.05, f"Smoke test FAILED: full [b,w] error {full_err:.2e} > 5%"
    print("=== smoke_test_fedavg PASSED ===\n")


def _print_comparison_table(
    class_idx: int,
    name: str,
    y_te: np.ndarray,
    preds_single: np.ndarray,
    preds_fed_fhe: np.ndarray,
    preds_fed_plain: np.ndarray,
    preds_full_plain: np.ndarray,
    w_fhe_fed: np.ndarray,
    w_plain_fed: np.ndarray,
    k: int,
) -> None:
    def acc(p: np.ndarray) -> float:
        return float(np.mean(p == y_te) * 100)

    def precision(p: np.ndarray) -> float:
        # Precision for positive class (label == 1.0)
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

    print(f"  --- Class {class_idx} ({name} vs rest) comparison (k={k}) ---")
    print(f"  {'Approach':<40} | Accuracy  | Precision | F1")
    print(f"  {'-'*40}-+----------+-----------+------")
    print(
        f"  {'Single-client FHE (N_per_class=2, seed=42)':<40} | {acc(preds_single):6.2f}% | {precision(preds_single):8.2f}% | {f1(preds_single):5.2f}%"
    )
    print(
        f"  {f'Federated FHE  ({k} clients avg.)':<40} | {acc(preds_fed_fhe):6.2f}% | {precision(preds_fed_fhe):8.2f}% | {f1(preds_fed_fhe):5.2f}%"
    )
    print(
        f"  {'Federated plaintext reference':<40} | {acc(preds_fed_plain):6.2f}% | {precision(preds_fed_plain):8.2f}% | {f1(preds_fed_plain):5.2f}%"
    )
    print(
        f"  {'Full-data plaintext reference':<40} | {acc(preds_full_plain):6.2f}% | {precision(preds_full_plain):8.2f}% | {f1(preds_full_plain):5.2f}%"
    )
    w_err = weight_relative_error(w_fhe_fed, w_plain_fed)
    print(f"  FHE fed weights vs plaintext fed weights: {w_err:.4e}")
    print()


def main(
    k: int = 3,
    serialize: bool = True,
    n_per_class: int | None = None,
    solver_name: str | None = None,
    security: str = DEFAULT_SECURITY_LEVEL,
) -> None:
    global solv
    solv = resolve_solver_module(solver_name or DEFAULT_SOLVER_NAME)
    slots = SPARSE_BOOTSTRAP_SLOTS if security == "128" else None

    splits = prepare_iris_binary()
    n_test = len(splits[0][1])  # 30

    print(f"=== Federated FHE LS-SVM (Iris OvR, k={k} clients) ===")
    print(f"Dataset: 120 train / {n_test} test  |  KERNEL_GAMMA={KERNEL_GAMMA}")
    print(f"Chebyshev: sqrt={D_SQRT}, inv_qr={D_INV}, inv_backsub={D_INV_BACKSUB}\n")

    # Step 1: Compute all partitions (plaintext, no FHE) to determine context size
    all_partitions = {}
    max_client_n = 1
    for class_idx, (X_tr, _, y_tr, _, _) in enumerate(splits):
        parts = partition_all(X_tr, y_tr, k)
        if n_per_class is not None:
            parts = [
                _subsample_for_fhe(Xc, yc, n_per_class, seed=42 + class_idx * 1000 + i)
                for i, (Xc, yc) in enumerate(parts)
            ]
        all_partitions[class_idx] = parts
        max_n = max(len(y_c) for _, y_c in parts) + 1
        max_client_n = max(max_client_n, max_n)
    # Single-client baseline uses N_PER_CLASS_BASELINE samples per class
    max_client_n = max(max_client_n, N_PER_CLASS_BASELINE * 2 + 1)
    print(f"Max client H size: {max_client_n}x{max_client_n}  (k={k})\n")

    # Step 2: Feature dimension for rotation key generation
    sample_X = splits[0][0][:1]
    max_feat_dim = max_client_n
    for _, _, feature_map_fn, _ in CLASS_KERNELS.values():
        d = feature_map_fn(sample_X).shape[1] if feature_map_fn else sample_X.shape[1]
        max_feat_dim = max(max_feat_dim, d)

    # Step 3: One shared crypto context for all clients and all classes
    depth = (
        depth_for_size(
            max_client_n,
            max_client_n,
            D_SQRT,
            D_INV,
            D_INV_BACKSUB,
            safety_factor=DEPTH_SAFETY,
            depth_override=DEPTH_OVERRIDE,
        )
        # fhe_aggregate EvalMult(1) + predict_cipher 2×EvalMult(2)
        # + implicit ModDown(1) + decrypt margin(2)
        + 6
    )
    if security == "128":
        # The bootstrapping path ignores the no-bootstrap depth estimate above and uses
        # its own fixed per-round budget; report/track the depth the context really has.
        depth = solv._BOOTSTRAP_USABLE_DEPTH + solv._BOOTSTRAP_SKIP_BELOW_LEVEL
    class_dirs = [f"models/k={k}/class_{class_idx}" for class_idx in range(len(splits))]
    existing_ctx_dir = next(
        (
            class_dir
            for class_dir in class_dirs
            if os.path.exists(f"{class_dir}/cryptocontext.bin")
            and os.path.exists(f"{class_dir}/public_key.bin")
            and os.path.exists(f"{class_dir}/secret_key.bin")
            and _class_context_exists(class_dir, depth)
            and read_security_marker(class_dir) == security
        ),
        None,
    )

    if existing_ctx_dir is not None:
        print(f"Loading shared crypto context from {existing_ctx_dir} ...")
        t_ctx = time.perf_counter()
        cc, keys = _load_class_context(
            existing_ctx_dir, max_client_n, n_test, max_feat_dim
        )
    else:
        print(f"Setting up shared crypto context (depth={depth}, security={security}) ...")
        t_ctx = time.perf_counter()
        cc, keys = solv.setup_crypto_context(
            depth,
            matrix_size=max_client_n,
            n_test=n_test,
            feature_dim=max_feat_dim,
            N=N_OVERRIDE,
            security=security,
        )
    slot_count = solv.get_slot_count(cc)
    print(
        f"Context ready in {time.perf_counter() - t_ctx:.1f}s  (slots={slot_count})\n"
    )

    # Extract and print crypto context parameters
    try:
        ring_dim = cc.GetRingDimension()
    except Exception:
        ring_dim = "N/A"
    
    # Compute modulus bit length from solver configuration
    # First modulus: 60 bits, additional levels: 50 bits each
    first_mod_bits = 60
    additional_mod_bits = depth * 50
    total_modulus_bits = first_mod_bits + additional_mod_bits
    
    # OpenFHE CKKS defaults from solver setup (fixed across all solvers)
    security_level = "HEStd_128_classic" if security == "128" else "HEStd_NotSet"
    scaling_factor_bits = 50  # SetScalingModSize(50) in all solvers

    print(f"Crypto Context Parameters:")
    print(f"  Security Level: {security_level}")
    print(f"  Ring Dimension: {ring_dim}")
    print(f"  Modulus: {total_modulus_bits} bits (60 + {depth}×50)")
    print(f"  Scaling Factor Bits per Level: {scaling_factor_bits}")
    print(f"  Multiplicative Depth: {depth}\n")

    classifiers_fed = []
    classifiers_single = []

    for class_idx, (X_tr, X_te, y_tr, y_te, name) in enumerate(splits):
        kernel_name, _, feature_map, mode_str = CLASS_KERNELS.get(
            class_idx, ("linear", linear_kernel, None, "primal:linear")
        )
        print(f"--- Class {class_idx} ({name} vs rest) [kernel={kernel_name}] ---")

        # Single per-class preprocessing path shared with the plaintext
        # reference and inference: linear → raw; poly → map+center+unit-L2 norm.
        X_tr_feat, phi_mean = preprocess_features(X_tr, feature_map)
        X_te_feat, _        = preprocess_features(X_te, feature_map, phi_mean=phi_mean)

        # Per-class gamma: GCV for non-linear kernels, fixed fallback for linear
        if feature_map is not None:
            gamma, gcv_val = gcv_gamma(X_tr, y_tr, feature_map)
            print(f"  GCV: γ={gamma:.6f}  GCV={gcv_val:.6f}")
        else:
            gamma = KERNEL_GAMMA[kernel_name]
            print(f"  Linear kernel: fixed γ={gamma:.4f}")

        # Full-data plaintext reference
        H_full, rhs_full = build_lssvm_matrix(X_tr_feat, y_tr, gamma)
        print(f"  Full-data H={H_full.shape}, cond={np.linalg.cond(H_full):.2f} ...")
        try:
            sol_full = np.linalg.solve(H_full, rhs_full)
        except np.linalg.LinAlgError:
            sol_full = np.linalg.lstsq(H_full, rhs_full, rcond=None)[0]
        alpha_full = sol_full[1:]
        preds_plain_full, _ = predict_lssvm(
            X_te_feat, X_tr_feat, alpha_full, y_tr, sol_full[0]
        )

        # Per-client FHE training (with checkpointing)
        parts = all_partitions[class_idx]
        b_cts, w_cts = [], []
        parts_feat = []
        class_dir = f"models/k={k}/class_{class_idx}"
        if not _class_context_exists(class_dir, depth):
            _save_class_context(class_dir, cc, keys, depth, security=security)
        for client_id, (X_c, y_c) in enumerate(parts):
            # Each client centers by the shared training mean (phi_mean); unit-L2
            # normalization is per-row, so client rows match the full-data frame.
            X_c_feat, _ = preprocess_features(X_c, feature_map, phi_mean=phi_mean)
            parts_feat.append((X_c_feat, y_c))
            ckpt_dir = f"models/k={k}/class_{class_idx}/client_{client_id}"
            if _cts_exist(ckpt_dir):
                print(f"  [client {client_id}] Resuming from checkpoint {ckpt_dir}")
                b_ct_i, w_ct_i = _load_cts(ckpt_dir)
                _d = X_c_feat.shape[1]
                if not _cts_are_finite(cc, keys, b_ct_i, w_ct_i, _d):
                    print(
                        f"  [client {client_id}] Checkpoint invalid or stale, recomputing"
                    )
                    H_c, rhs_c = build_lssvm_matrix(X_c_feat, y_c, gamma)
                    print(
                        f"  [client {client_id}] H={H_c.shape}, cond={np.linalg.cond(H_c):.1f} ..."
                    )
                    t0 = time.perf_counter()
                    b_ct_i, w_ct_i, _ = solv.solver(
                        cc,
                        keys,
                        H_c.tolist(),
                        rhs_c.tolist(),
                        X_c_feat,
                        y_c,
                        D_sqrt=D_SQRT,
                        D_inv=D_INV,
                        D_inv_backsub=D_INV_BACKSUB,
                        security=security,
                    )
                    print(
                        f"  [client {client_id}] FHE solve: {time.perf_counter() - t0:.1f}s"
                    )
                    _save_cts(ckpt_dir, cc, keys, b_ct_i, w_ct_i)
                    print(f"  [client {client_id}] Checkpoint refreshed.")
            else:
                H_c, rhs_c = build_lssvm_matrix(X_c_feat, y_c, gamma)
                print(
                    f"  [client {client_id}] H={H_c.shape}, cond={np.linalg.cond(H_c):.1f} ..."
                )
                t0 = time.perf_counter()
                b_ct_i, w_ct_i, _ = solv.solver(
                    cc,
                    keys,
                    H_c.tolist(),
                    rhs_c.tolist(),
                    X_c_feat,
                    y_c,
                    D_sqrt=D_SQRT,
                    D_inv=D_INV,
                    D_inv_backsub=D_INV_BACKSUB,
                    security=security,
                )
                print(
                    f"  [client {client_id}] FHE solve: {time.perf_counter() - t0:.1f}s"
                )
                _save_cts(ckpt_dir, cc, keys, b_ct_i, w_ct_i)
                print(f"  [client {client_id}] Checkpoint saved.")
            b_cts.append(b_ct_i)
            w_cts.append(w_ct_i)

            _report_depth(
                f"client {client_id} solve  (H={X_c_feat.shape[0]+1}×{X_c_feat.shape[0]+1})",
                depth,
                {"b_ct": b_ct_i, "w_ct": w_ct_i},
            )

        # FHE aggregation
        print(f"  Aggregating {k} encrypted models ...")
        t_agg = time.perf_counter()
        b_global, w_global = fhe_aggregate(cc, b_cts, w_cts)
        print(f"  Aggregation: {time.perf_counter() - t_agg:.3f}s")

        d = X_te_feat.shape[1]
        _report_depth(
            f"aggregation  (k={k} clients)",
            depth,
            {"b_global": b_global, "w_global": w_global},
        )

        # Encrypted inference with global model
        t_inf = time.perf_counter()
        scores_ct = solv.predict_cipher(cc, keys, b_global, w_global, X_te_feat, slots=slots)
        print(f"  Cipher predict: {time.perf_counter() - t_inf:.4f}s")
        scores_fed = np.array(solv.decrypt_vector(cc, keys, scores_ct, n_test))
        _report_depth(
            f"inference  (n_test={n_test})",
            depth,
            {"scores_ct": scores_ct},
        )
        preds_fed = np.sign(scores_fed)
        preds_fed[preds_fed == 0] = 1.0

        # Decrypt federated weights for error reporting
        w_fhe_fed = np.array(solv.decrypt_vector(cc, keys, w_global, d))

        # Plaintext federated reference
        preds_plain_fed, w_plain_fed, _ = plaintext_federated_reference(
            parts_feat, X_te_feat, gamma
        )

        # Single-client FHE baseline (matches lssvm_cipher.py, N_PER_CLASS=2).
        # Standalone client: preprocess with its OWN mean and evaluate on a test
        # set centered by that same mean, so train/test share one feature frame.
        X_s, y_s = _subsample_for_fhe(X_tr, y_tr, N_PER_CLASS_BASELINE, seed=42)
        X_s_feat, phi_mean_s = preprocess_features(X_s, feature_map)
        X_te_feat_s, _ = preprocess_features(X_te, feature_map, phi_mean=phi_mean_s)
        H_s, rhs_s = build_lssvm_matrix(X_s_feat, y_s, gamma)
        print(f"  Single-client baseline H={H_s.shape} ...")
        t_sc = time.perf_counter()
        b_ct_s, w_ct_s, _ = solv.solver(
            cc,
            keys,
            H_s.tolist(),
            rhs_s.tolist(),
            X_s_feat,
            y_s,
            D_sqrt=D_SQRT,
            D_inv=D_INV,
            D_inv_backsub=D_INV_BACKSUB,
            security=security,
        )
        print(f"  Single-client solve: {time.perf_counter() - t_sc:.1f}s")
        _report_depth(
            f"single-client solve  (H={H_s.shape[0]}×{H_s.shape[0]})",
            depth,
            {"b_ct": b_ct_s, "w_ct": w_ct_s},
        )
        scores_ct_s = solv.predict_cipher(cc, keys, b_ct_s, w_ct_s, X_te_feat_s, slots=slots)
        scores_s = np.array(solv.decrypt_vector(cc, keys, scores_ct_s, n_test))
        preds_single = np.sign(scores_s)
        preds_single[preds_single == 0] = 1.0

        _print_comparison_table(
            class_idx,
            name,
            y_te,
            preds_single,
            preds_fed,
            preds_plain_fed,
            preds_plain_full,
            w_fhe_fed,
            w_plain_fed,
            k,
        )

        # Serialize global model
        if serialize:
            out_dir = f"models/k={k}/class_{class_idx}"
            solv.save_global_checkpoint(
                cc,
                keys,
                b_global,
                w_global,
                out_dir,
                mode_str=mode_str,
                checkpoint_policy={"persist_public_key": True},
                security=security,
            )
            np.save(f"{out_dir}/phi_mean.npy", phi_mean)
            print(f"  Global model serialized to {out_dir}/  [{mode_str}]")

        # # Delete per-client checkpoints now that aggregation succeeded
        # for client_id in range(k):
        #     ckpt_dir = f"models/k={k}/class_{class_idx}/client_{client_id}"
        #     if os.path.exists(ckpt_dir):
        #         shutil.rmtree(ckpt_dir)
        # print(f"  Intermediate checkpoints deleted.")

        classifiers_fed.append({"class_idx": class_idx, "scores": scores_fed})
        classifiers_single.append({"class_idx": class_idx, "scores": scores_s})

    # OvR multiclass accuracy
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

    print(
        f"OvR Multiclass Accuracy (Federated FHE, k={k}): {_ovr_acc(classifiers_fed):.2f}%"
    )
    print(
        f"OvR Multiclass Accuracy (Single-client FHE):      {_ovr_acc(classifiers_single):.2f}%"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke-test" in args:
        solv = resolve_solver_module(parse_solver_name(args))
        smoke_test_fedavg(security=parse_security_level(args))
        sys.exit(0)

    k = 3
    serialize = "--no-serialize" not in args
    n_per_class = next(
        (int(a.split("=", 1)[1]) for a in args if a.startswith("--n-per-class=")),
        None,
    )
    numeric_args = [a for a in args if a.lstrip("-").isdigit()]
    if numeric_args:
        k = int(numeric_args[0])

    solver_name = parse_solver_name(args)
    security = parse_security_level(args)
    main(
        k=k,
        serialize=serialize,
        n_per_class=n_per_class,
        solver_name=solver_name,
        security=security,
    )
