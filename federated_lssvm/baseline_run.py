"""WP3 baseline: local plaintext LSSVM solve + encrypted FedAvg aggregation.

Pipeline:
  1. Load the selected dataset (--dataset=iris or breast_cancer), stratified split.
  2. Partition all training samples disjointly across k clients (same
     partitioning as federated_lssvm.train, for apples-to-apples comparison).
  3. Each client solves its local LSSVM in plaintext (no FHE) -> (w_i, b_i).
  4. Each client encrypts only (w_i, b_i) and "sends" it to the server.
  5. Server aggregates homomorphically (EvalAdd + EvalMult by 1/k, no
     decryption) via federated_lssvm.train.fhe_aggregate.
  6. Evaluate the encrypted global model on the test set via predict_cipher.
  7. Serialize to models/k={k}_baseline/class_{i}/.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import numpy as np
from config.parallel import bootstrap as _init_parallel

_init_parallel()

from openfhe import SerializeToFile, BINARY

import federated_lssvm.train as T
from federated_lssvm.solver_selection import (
    DEFAULT_SOLVER_NAME,
    DEFAULT_SECURITY_LEVEL,
    parse_solver_name,
    parse_security_level,
    parse_dataset_name,
    resolve_solver_module,
)
from lssvm.preprocessing import (
    build_lssvm_matrix,
    preprocess_features,
    recalibration_threshold,
    solve_client_plain,
)
from lssvm.plain import predict_lssvm
from lssvm.preprocessors import DEFAULT_DATASET, prepare_binary, raw_test_labels
from lssvm.solvers.utils import make_packed_plaintext, SPARSE_BOOTSTRAP_SLOTS
from config.metrics import weight_relative_error

solv = None

# fhe_aggregate EvalMult(1) + predict_cipher 2xEvalMult(2) + implicit ModDown(1)
# + decrypt margin(2). No H-matrix solve happens in this baseline (unlike
# train.py's context_depth), so this fixed budget is the whole depth story
# for security="notset".
_NOTSET_DEPTH = 6

_LOG_LINES: list[str] = []


def _log(msg: str = "") -> None:
    print(msg)
    _LOG_LINES.append(msg)


def _max_feat_dim(splits, class_kernels) -> int:
    """Largest feature dimension across all OvR classes, for rotation-key sizing."""
    sample_X = splits[0][0][:1]
    max_dim = sample_X.shape[1]
    for _, _, feature_map_fn, _ in class_kernels.values():
        d = feature_map_fn(sample_X).shape[1] if feature_map_fn else sample_X.shape[1]
        max_dim = max(max_dim, d)
    return max_dim


def _print_comparison_table(
    class_idx: int,
    name: str,
    y_te: np.ndarray,
    preds_baseline: np.ndarray,
    preds_fed_plain: np.ndarray,
    preds_full_plain: np.ndarray,
    w_baseline: np.ndarray,
    w_plain_fed: np.ndarray,
    k: int,
) -> None:
    def acc(p: np.ndarray) -> float:
        return float(np.mean(p == y_te) * 100)

    def precision(p: np.ndarray) -> float:
        tp = float(np.sum((p == 1.0) & (y_te == 1.0)))
        fp = float(np.sum((p == 1.0) & (y_te != 1.0)))
        return float(tp / (tp + fp) * 100) if tp + fp > 0 else 0.0

    def f1(p: np.ndarray) -> float:
        tp = float(np.sum((p == 1.0) & (y_te == 1.0)))
        fp = float(np.sum((p == 1.0) & (y_te != 1.0)))
        fn = float(np.sum((p != 1.0) & (y_te == 1.0)))
        if tp == 0:
            return 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(2 * prec * rec / (prec + rec) * 100) if prec + rec > 0 else 0.0

    _log(f"  --- Class {class_idx} ({name} vs rest) comparison (k={k}) ---")
    _log(f"  {'Approach':<40} | Accuracy  | Precision | F1")
    _log(f"  {'-'*40}-+----------+-----------+------")
    _log(
        f"  {'Baseline (plaintext solve + encrypted agg)':<40} | {acc(preds_baseline):6.2f}% | {precision(preds_baseline):8.2f}% | {f1(preds_baseline):5.2f}%"
    )
    _log(
        f"  {'Federated plaintext reference':<40} | {acc(preds_fed_plain):6.2f}% | {precision(preds_fed_plain):8.2f}% | {f1(preds_fed_plain):5.2f}%"
    )
    _log(
        f"  {'Full-data plaintext reference':<40} | {acc(preds_full_plain):6.2f}% | {precision(preds_full_plain):8.2f}% | {f1(preds_full_plain):5.2f}%"
    )
    w_err = weight_relative_error(w_baseline, w_plain_fed)
    # Wording matches config/report.py's _RE_W_ERR regex verbatim (kept unmodified,
    # shared with train.py's finalize output) so this stat isn't silently dropped from
    # the report. "FHE fed" here means the encrypted-aggregation result, not that the
    # per-client solve itself ran under FHE (it's plaintext in this baseline).
    _log(f"  FHE fed weights vs plaintext fed weights: {w_err:.4e}")
    _log()


def main(
    k: int = 3,
    solver_name: str | None = None,
    security: str = DEFAULT_SECURITY_LEVEL,
    dataset: str = DEFAULT_DATASET,
    serialize: bool = True,
) -> None:
    global solv
    solv = resolve_solver_module(solver_name or DEFAULT_SOLVER_NAME)
    _LOG_LINES.clear()

    T.configure_dataset(dataset)
    splits = prepare_binary(dataset)
    n_train = len(splits[0][0])
    n_test = len(splits[0][1])
    label = "OvR" if len(splits) > 1 else "binary"

    _log(f"=== WP3 Baseline: plaintext solve + encrypted aggregation ({dataset} {label}, k={k}) ===")
    _log(f"Dataset: {n_train} train / {n_test} test\n")

    max_feat_dim = _max_feat_dim(splits, T.CLASS_KERNELS)
    _log(f"Setting up crypto context (security={security}) ...")
    t_ctx = time.perf_counter()
    cc, keys = solv.setup_crypto_context(
        _NOTSET_DEPTH,
        matrix_size=1,
        n_test=n_test,
        feature_dim=max_feat_dim,
        N=None,
        security=security,
    )
    pack_slots = SPARSE_BOOTSTRAP_SLOTS if security == "128" else solv.get_slot_count(cc)
    predict_slots = SPARSE_BOOTSTRAP_SLOTS if security == "128" else None
    _log(f"Context ready in {time.perf_counter() - t_ctx:.1f}s  (slots={solv.get_slot_count(cc)})\n")

    classifiers = []
    for class_idx, (X_tr, X_te, y_tr, y_te, name) in enumerate(splits):
        kernel_name, feature_map, mode_str, gamma, X_tr_feat, phi_mean = T.class_setup(
            class_idx, X_tr, y_tr, verbose=False
        )
        _log(f"--- Class {class_idx} ({name} vs rest) ---")
        X_te_feat, _ = preprocess_features(X_te, feature_map, phi_mean=phi_mean)
        d = X_te_feat.shape[1]

        parts = T.partition_all(X_tr, y_tr, k)
        parts_feat = []
        b_cts, w_cts = [], []
        t_clients = time.perf_counter()
        for X_c, y_c in parts:
            X_c_feat, _ = preprocess_features(X_c, feature_map, phi_mean=phi_mean)
            parts_feat.append((X_c_feat, y_c))
            w_i, b_i = solve_client_plain(X_c_feat, y_c, gamma)
            w_ptxt = make_packed_plaintext(cc, list(w_i), pack_slots)
            b_ptxt = make_packed_plaintext(cc, [b_i], pack_slots)
            w_cts.append(cc.Encrypt(keys.publicKey, w_ptxt))
            b_cts.append(cc.Encrypt(keys.publicKey, b_ptxt))
        _log(f"  {k} clients solved (plaintext) + encrypted in {time.perf_counter() - t_clients:.4f}s")

        tmp_path = tempfile.mktemp(suffix=".bin")
        SerializeToFile(tmp_path, w_cts[0], BINARY)
        w_bytes = os.path.getsize(tmp_path)
        os.remove(tmp_path)
        SerializeToFile(tmp_path, b_cts[0], BINARY)
        b_bytes = os.path.getsize(tmp_path)
        os.remove(tmp_path)
        _log(f"  Per-client ciphertext size: w={w_bytes/1024:.1f} KiB, b={b_bytes/1024:.1f} KiB (traffic proxy)")

        t_agg = time.perf_counter()
        b_global, w_global = T.fhe_aggregate(cc, b_cts, w_cts)
        _log(f"  Aggregation: {time.perf_counter() - t_agg:.4f}s")

        w_baseline = np.array(solv.decrypt_vector(cc, keys, w_global, d))
        b_baseline = solv.decrypt_vector(cc, keys, b_global, 1)[0]
        train_scores = X_tr_feat @ w_baseline + b_baseline
        threshold = recalibration_threshold(train_scores, y_tr)

        t_inf = time.perf_counter()
        scores = np.array(
            solv.predict_cipher(cc, keys, b_global, w_global, X_te_feat, slots=predict_slots)
        ) - threshold
        _log(f"  Cipher predict: {time.perf_counter() - t_inf:.4f}s")
        preds_baseline = np.sign(scores)
        preds_baseline[preds_baseline == 0] = 1.0

        _, w_plain_fed, b_plain_fed = T.plaintext_federated_reference(
            parts_feat, X_te_feat, gamma
        )
        # Recalibrate this row too, for the same reason as the full-data row below:
        # an uncalibrated sign(score) would understate accuracy on imbalanced OvR
        # classes, and would make this row incomparable to the calibrated baseline
        # row even though both aggregate the same per-client models.
        train_scores_fed = X_tr_feat @ w_plain_fed + b_plain_fed
        thr_fed = recalibration_threshold(train_scores_fed, y_tr)
        scores_fed_plain = X_te_feat @ w_plain_fed + b_plain_fed - thr_fed
        preds_plain_fed = np.sign(scores_fed_plain)
        preds_plain_fed[preds_plain_fed == 0] = 1.0

        H_full, rhs_full = build_lssvm_matrix(X_tr_feat, y_tr, gamma)
        try:
            sol_full = np.linalg.solve(H_full, rhs_full)
        except np.linalg.LinAlgError:
            sol_full = np.linalg.lstsq(H_full, rhs_full, rcond=None)[0]
        alpha_full = sol_full[1:]
        # Recalibrate the full-data reference too (mirrors plain_run.py): this
        # solver's KKT variant is only zero-centred for balanced classes, so an
        # uncalibrated sign(score) understates accuracy on imbalanced OvR classes.
        _, train_scores_full = predict_lssvm(X_tr_feat, X_tr_feat, alpha_full, y_tr, sol_full[0])
        thr_full = recalibration_threshold(train_scores_full, y_tr)
        _, scores_full = predict_lssvm(X_te_feat, X_tr_feat, alpha_full, y_tr, sol_full[0])
        scores_full = scores_full - thr_full
        preds_full_plain = np.sign(scores_full)
        preds_full_plain[preds_full_plain == 0] = 1.0

        _print_comparison_table(
            class_idx, name, y_te, preds_baseline, preds_plain_fed, preds_full_plain,
            w_baseline, w_plain_fed, k,
        )

        if serialize:
            out_dir = f"models/k={k}_baseline/class_{class_idx}"
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
            np.save(f"{out_dir}/threshold.npy", threshold)
            _log(f"  Baseline model serialized to {out_dir}/  [{mode_str}]")

        classifiers.append(
            {"class_idx": class_idx, "scores": scores, "preds": preds_baseline, "y_te": y_te}
        )

    y_test_raw = raw_test_labels(dataset)
    if len(classifiers) == 1:
        c = classifiers[0]
        acc = float(np.mean(c["preds"] == c["y_te"]) * 100)
        _log(f"Binary Accuracy (Baseline, k={k}): {acc:.2f}%")
    else:
        score_matrix = np.column_stack([c["scores"] for c in classifiers])
        class_indices = np.array([c["class_idx"] for c in classifiers])
        ovr_preds = class_indices[score_matrix.argmax(axis=1)]
        ovr_acc = float(np.mean(ovr_preds == y_test_raw) * 100)
        _log(f"OvR Multiclass Accuracy (Baseline, k={k}): {ovr_acc:.2f}%")

    logs_dir = f"models/k={k}_baseline/logs"
    os.makedirs(logs_dir, exist_ok=True)
    with open(f"{logs_dir}/finalize.log", "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG_LINES) + "\n")

    from config.report import generate_report

    report_path = generate_report(k, dataset, logs_dir, f"models/k={k}_baseline/report.md")
    print(f"[baseline_run] appended run section to {report_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    k = 3
    numeric_args = [a for a in args if a.lstrip("-").isdigit()]
    if numeric_args:
        k = int(numeric_args[0])
    serialize = "--no-serialize" not in args
    solver_name = parse_solver_name(args)
    security = parse_security_level(args)
    dataset = parse_dataset_name(args)
    main(k=k, solver_name=solver_name, security=security, dataset=dataset, serialize=serialize)
