# WP3 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `federated_lssvm/baseline_run.py`, a new federated LSSVM entry point implementing WP3's "local plaintext training + encrypted aggregation" baseline — each client solves its LSSVM in plaintext, encrypts only the resulting `(w_i, b_i)`, and the server aggregates homomorphically — plus the small supporting changes (`infer.py` model-root support, a shared `solve_client_plain` helper) that let it plug into the existing federated pipeline.

**Architecture:** Reuse existing plaintext solve math, `fhe_aggregate`, and the chosen solver's `predict_cipher`/`save_global_checkpoint` unchanged; the only new FHE code is manually encrypting each client's plaintext `(w_i, b_i)` before aggregation. Output lands in a separate `models/k={k}_baseline/` tree so it never collides with `train.py`'s `models/k={k}/` artifacts, and `config/report.py`'s existing log-scraping reporter is reused by making `baseline_run.py`'s console output regex-compatible with it.

**Tech Stack:** Python, OpenFHE (CKKS), numpy, pytest. No new dependencies.

## Global Constraints

- Follow `CLAUDE.md`: import paths are `lssvm.*` / `federated_lssvm.*` / `config.*`, no flat prefixes; no algorithm changes bundled into structural commits; package-per-concern layout.
- Every client vector/scalar encryption **must** go through `lssvm.solvers.utils.make_packed_plaintext(cc, vals, slots)` with an explicit `slots` value — never a bare `cc.MakeCKKSPackedPlaintext` call. Omitting explicit `slots` metadata causes `EvalMult` against a sparse-bootstrapped ciphertext to silently return near-zero garbage (no exception) at `security="128"`.
- `--dataset`/`--solver`/`--security` CLI flags must reuse `federated_lssvm.solver_selection`'s existing parsing (`parse_dataset_name`, `parse_solver_name`, `parse_security_level`), matching `train.py`/`plain_run.py`/`infer.py` conventions exactly (default solver `qr_row`, default security `"128"`, default dataset `"iris"`).
- No per-client checkpoint/resume machinery — client plaintext solves are µs-scale, so a failed run is just re-run from scratch.
- Spec: `docs/superpowers/specs/2026-07-20-wp3-baseline-design.md`.

---

### Task 1: Extract `solve_client_plain` and reuse it in `plaintext_federated_reference`

**Files:**
- Modify: `lssvm/preprocessing.py` (add function after `recalibration_threshold`, currently ending at line 236)
- Modify: `federated_lssvm/train.py:372-401` (`plaintext_federated_reference`), `federated_lssvm/train.py:45-55` (import block)
- Test: `lssvm/test_preprocessing.py` (new)

**Interfaces:**
- Produces: `lssvm.preprocessing.solve_client_plain(X_c: np.ndarray, y_c: np.ndarray, gamma: float) -> tuple[np.ndarray, float]` — returns `(w, b)`, the primal weight vector and bias from solving one client's LSSVM in plaintext. Consumed by `federated_lssvm/train.py`'s `plaintext_federated_reference` (this task) and by `federated_lssvm/baseline_run.py` (Task 3).

- [ ] **Step 1: Write the failing tests**

Create `lssvm/test_preprocessing.py`:

```python
import numpy as np
import pytest

from lssvm.preprocessing import build_lssvm_matrix, solve_client_plain


def test_solve_client_plain_matches_manual_solve():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    y = np.where(X[:, 0] > 0, 1.0, -1.0)
    gamma = 1.0

    w, b = solve_client_plain(X, y, gamma)

    H, rhs = build_lssvm_matrix(X, y, gamma)
    sol = np.linalg.solve(H, rhs)
    expected_b = sol[0]
    expected_w = X.T @ (sol[1:] * y)

    assert b == pytest.approx(expected_b)
    np.testing.assert_allclose(w, expected_w)


def test_solve_client_plain_separates_linearly_separable_data():
    rng = np.random.default_rng(1)
    X_pos = rng.normal(loc=[3, 3], size=(10, 2))
    X_neg = rng.normal(loc=[-3, -3], size=(10, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * 10 + [-1.0] * 10)

    w, b = solve_client_plain(X, y, gamma=1.0)
    preds = np.sign(X @ w + b)
    assert np.mean(preds == y) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest lssvm/test_preprocessing.py -v`
Expected: FAIL with `ImportError: cannot import name 'solve_client_plain'`

- [ ] **Step 3: Add `solve_client_plain` to `lssvm/preprocessing.py`**

Insert immediately after `recalibration_threshold` (after line 236, before `def prepare_iris_binary`):

```python
def solve_client_plain(X_c: np.ndarray, y_c: np.ndarray, gamma: float) -> tuple[np.ndarray, float]:
    """Solve one client's LSSVM in plaintext, returning primal (w, b).

    Shared by the plaintext federated reference and the WP3 baseline (which
    encrypts only the resulting (w, b) instead of the training data/matrix).
    """
    H, rhs = build_lssvm_matrix(X_c, y_c, gamma)
    try:
        sol = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(H, rhs, rcond=None)[0]
    b = float(sol[0])
    alpha = sol[1:]
    w = X_c.T @ (alpha * y_c)
    return w, b
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest lssvm/test_preprocessing.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Refactor `plaintext_federated_reference` to reuse it**

In `federated_lssvm/train.py`, add `solve_client_plain` to the import block at line 45-55:

```python
from lssvm.preprocessing import (
    build_lssvm_matrix,
    linear_kernel,
    polynomial_kernel,
    homogeneous_poly_kernel,
    poly_feature_map,
    homogeneous_poly_feature_map,
    preprocess_features,
    gcv_gamma,
    recalibration_threshold,
    solve_client_plain,
)
```

Replace the body of `plaintext_federated_reference` (lines 372-401) with:

```python
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
        w_i, b_i = solve_client_plain(X_c, y_c, gamma)
        b_sum += b_i
        w_sum = w_i if w_sum is None else w_sum + w_i

    k = len(partitions_feat)
    w_avg = w_sum / k
    b_avg = b_sum / k
    scores = X_te_feat @ w_avg + b_avg
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    return preds, w_avg, b_avg
```

- [ ] **Step 6: Run the full existing test suite to confirm no regression**

Run: `pytest -q`
Expected: PASS, same pass count as before this change (this refactor is behavior-preserving — `plaintext_federated_reference` computes byte-identical `w_avg`/`b_avg` for the same inputs, since it's the same arithmetic just extracted).

- [ ] **Step 7: Commit**

```bash
git add lssvm/preprocessing.py lssvm/test_preprocessing.py federated_lssvm/train.py
git commit -m "refactor: extract solve_client_plain, shared by plaintext ref and WP3 baseline"
```

---

### Task 2: Add `--model-root` support to `federated_lssvm/infer.py`

**Files:**
- Modify: `federated_lssvm/infer.py`
- Test: `federated_lssvm/test_infer_model_root.py` (new)

**Interfaces:**
- Produces: `federated_lssvm.infer._model_dir(k: int, class_idx: int, model_root: str | None = None) -> str`. `main()` gains an additive `model_root: str | None = None` parameter (default preserves today's `models/k={k}` path exactly).

- [ ] **Step 1: Write the failing test**

Create `federated_lssvm/test_infer_model_root.py`:

```python
from federated_lssvm.infer import _model_dir


def test_model_dir_defaults_to_k_folder():
    assert _model_dir(20, 2) == "models/k=20/class_2"


def test_model_dir_respects_model_root_override():
    assert _model_dir(20, 2, model_root="models/k=20_baseline") == "models/k=20_baseline/class_2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest federated_lssvm/test_infer_model_root.py -v`
Expected: FAIL with `ImportError: cannot import name '_model_dir'`

- [ ] **Step 3: Add `_model_dir` and wire it into `main()`**

In `federated_lssvm/infer.py`, insert before `def main(` (currently line 61):

```python
def _model_dir(k: int, class_idx: int, model_root: str | None = None) -> str:
    root = model_root if model_root is not None else f"models/k={k}"
    return f"{root}/class_{class_idx}"
```

Change the `main()` signature (line 61) from:

```python
def main(k: int = 20, solver_name: str | None = None, dataset: str = DEFAULT_DATASET) -> None:
```

to:

```python
def main(
    k: int = 20,
    solver_name: str | None = None,
    dataset: str = DEFAULT_DATASET,
    model_root: str | None = None,
) -> None:
```

Replace line 79 (`model_dir = f"models/k={k}/class_{class_idx}"`) with:

```python
        model_dir = _model_dir(k, class_idx, model_root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest federated_lssvm/test_infer_model_root.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the `--model-root=` CLI flag**

Replace the `if __name__ == "__main__":` block (lines 130-137) with:

```python
if __name__ == "__main__":
    k = 20
    args = [a for a in sys.argv[1:] if a.lstrip("-").isdigit()]
    if args:
        k = int(args[0])
    solver_name = parse_solver_name(sys.argv[1:])
    dataset = parse_dataset_name(sys.argv[1:])
    model_root = next(
        (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model-root=")),
        None,
    )
    main(k=k, solver_name=solver_name, dataset=dataset, model_root=model_root)
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add federated_lssvm/infer.py federated_lssvm/test_infer_model_root.py
git commit -m "feat: add --model-root override to federated_lssvm.infer"
```

---

### Task 3: Create `federated_lssvm/baseline_run.py` — core plaintext-solve + encrypted-aggregation pipeline

**Files:**
- Create: `federated_lssvm/baseline_run.py`
- Test: `federated_lssvm/test_baseline_run.py` (new)

**Interfaces:**
- Consumes: `federated_lssvm.train.configure_dataset(dataset)`, `.CLASS_KERNELS` (module global, populated by `configure_dataset`), `.class_setup(class_idx, X_tr, y_tr, verbose=False) -> (kernel_name, feature_map, mode_str, gamma, X_tr_feat, phi_mean)`, `.partition_all(X, y, k) -> list[(X_c, y_c)]`, `.fhe_aggregate(cc, b_cts, w_cts) -> (b_global, w_global)`, `.plaintext_federated_reference(parts_feat, X_te_feat, gamma) -> (preds, w_avg, b_avg)`; `lssvm.preprocessing.{preprocess_features, recalibration_threshold, solve_client_plain, build_lssvm_matrix}`; `lssvm.plain.predict_lssvm`; `lssvm.preprocessors.{DEFAULT_DATASET, prepare_binary, raw_test_labels}`; `lssvm.solvers.utils.{make_packed_plaintext, SPARSE_BOOTSTRAP_SLOTS}`; `federated_lssvm.solver_selection.{DEFAULT_SOLVER_NAME, DEFAULT_SECURITY_LEVEL, parse_solver_name, parse_security_level, parse_dataset_name, resolve_solver_module}`; `config.metrics.weight_relative_error`; a solver module's `setup_crypto_context`, `get_slot_count`, `decrypt_vector`, `predict_cipher`.
- Produces: `federated_lssvm.baseline_run._max_feat_dim(splits, class_kernels) -> int`; `federated_lssvm.baseline_run.main(k=3, solver_name=None, security=DEFAULT_SECURITY_LEVEL, dataset=DEFAULT_DATASET, serialize=True) -> None`. Task 4 modifies `main()` to add serialization; Task 5 modifies it to add report generation and adds the `__main__` CLI block — both are additive edits to this task's `main()`, not rewrites.

- [ ] **Step 1: Write the failing test for `_max_feat_dim`**

Create `federated_lssvm/test_baseline_run.py`:

```python
import federated_lssvm.train as T
from lssvm.preprocessors import prepare_binary


def test_max_feat_dim_iris_matches_poly_expansion():
    T.configure_dataset("iris")
    splits = prepare_binary("iris")
    from federated_lssvm.baseline_run import _max_feat_dim

    d = _max_feat_dim(splits, T.CLASS_KERNELS)
    # class 0 is linear (4 raw features), classes 1-2 are degree-2 poly on 4
    # features -> 4 + C(4,2) + 4 = 14 explicit terms (poly_feature_map's actual
    # output width); the max across classes must be >= the raw feature count.
    assert d >= 4


def test_max_feat_dim_breast_cancer_is_linear_30_features():
    T.configure_dataset("breast_cancer")
    splits = prepare_binary("breast_cancer")
    from federated_lssvm.baseline_run import _max_feat_dim

    d = _max_feat_dim(splits, T.CLASS_KERNELS)
    assert d == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest federated_lssvm/test_baseline_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'federated_lssvm.baseline_run'`

- [ ] **Step 3: Create `federated_lssvm/baseline_run.py`**

```python
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
    _log(f"  Baseline weights vs plaintext fed weights: {w_err:.4e}")
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

        preds_plain_fed, w_plain_fed, _ = T.plaintext_federated_reference(
            parts_feat, X_te_feat, gamma
        )

        H_full, rhs_full = build_lssvm_matrix(X_tr_feat, y_tr, gamma)
        try:
            sol_full = np.linalg.solve(H_full, rhs_full)
        except np.linalg.LinAlgError:
            sol_full = np.linalg.lstsq(H_full, rhs_full, rcond=None)[0]
        alpha_full = sol_full[1:]
        preds_full_plain, _ = predict_lssvm(X_te_feat, X_tr_feat, alpha_full, y_tr, sol_full[0])

        _print_comparison_table(
            class_idx, name, y_te, preds_baseline, preds_plain_fed, preds_full_plain,
            w_baseline, w_plain_fed, k,
        )

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
```

Note: `w_cts`/`b_cts`/`kernel_name` are intentionally computed even though
`kernel_name` isn't used yet — Task 4 needs `mode_str` (already captured)
for serialization; `kernel_name` is dropped from the unused-variable list by
prefixing with `_` if your linter complains, but leave it as-is per this
codebase's existing style (`train.py`'s own loop does the same).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest federated_lssvm/test_baseline_run.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: PASS, no regressions (new file isn't imported by anything else yet).

- [ ] **Step 6: Commit**

```bash
git add federated_lssvm/baseline_run.py federated_lssvm/test_baseline_run.py
git commit -m "feat: add WP3 baseline_run.py core plaintext-solve + encrypted-aggregation pipeline"
```

---

### Task 4: Serialize the baseline model and measure per-client ciphertext size

**Files:**
- Modify: `federated_lssvm/baseline_run.py` (`main()`, inside the per-class loop)

**Interfaces:**
- Consumes: `solv.save_global_checkpoint(cc, keys, b_ct, w_ct, out_dir, mode_str, checkpoint_policy, security)` (existing solver contract, validated by `solver_selection.validate_solver_hooks`).
- Produces: `models/k={k}_baseline/class_{i}/` artifacts (`cryptocontext.bin`, `secret_key.bin`, `bias.bin`, `weights.bin`, `mode.txt`, `checkpoint.json`, plus `phi_mean.npy` and `threshold.npy`) — same shape `train.py` produces under `models/k={k}/`, loadable by `infer.py --model-root=models/k={k}_baseline`.

- [ ] **Step 1: Add the ciphertext-size measurement right after the client encryption loop**

In `federated_lssvm/baseline_run.py`, inside `main()`'s per-class loop, immediately after the line
`_log(f"  {k} clients solved (plaintext) + encrypted in {time.perf_counter() - t_clients:.4f}s")`
add:

```python
        tmp_path = tempfile.mktemp(suffix=".bin")
        SerializeToFile(tmp_path, w_cts[0], BINARY)
        w_bytes = os.path.getsize(tmp_path)
        os.remove(tmp_path)
        SerializeToFile(tmp_path, b_cts[0], BINARY)
        b_bytes = os.path.getsize(tmp_path)
        os.remove(tmp_path)
        _log(f"  Per-client ciphertext size: w={w_bytes/1024:.1f} KiB, b={b_bytes/1024:.1f} KiB (traffic proxy)")
```

- [ ] **Step 2: Add serialization after the comparison table**

Immediately after the `_print_comparison_table(...)` call (before `classifiers.append(...)`), add:

```python
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
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -q`
Expected: PASS, no regressions (these edits only add code inside `main()`, which no test calls yet).

- [ ] **Step 4: Commit**

```bash
git add federated_lssvm/baseline_run.py
git commit -m "feat: serialize WP3 baseline model + measure per-client ciphertext size"
```

---

### Task 5: Wire up `config/report.py` integration and the CLI entrypoint

**Files:**
- Modify: `federated_lssvm/baseline_run.py` (end of `main()`, plus `__main__` block)

**Interfaces:**
- Consumes: `config.report.generate_report(k, dataset, logs_dir, out_path, phase_seconds=None) -> str` (existing, unmodified).
- Produces: `models/k={k}_baseline/logs/finalize.log`, `models/k={k}_baseline/report.md`, `models/k={k}_baseline/metrics.csv`; CLI `python -m federated_lssvm.baseline_run [k] --dataset=... --solver=... --security=... [--no-serialize]`.

- [ ] **Step 1: Add log flush + report generation at the end of `main()`**

Immediately after the final `_log(...)` line that prints binary/OvR accuracy (the last statement in `main()`), add:

```python

    logs_dir = f"models/k={k}_baseline/logs"
    os.makedirs(logs_dir, exist_ok=True)
    with open(f"{logs_dir}/finalize.log", "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG_LINES) + "\n")

    from config.report import generate_report

    report_path = generate_report(k, dataset, logs_dir, f"models/k={k}_baseline/report.md")
    print(f"[baseline_run] appended run section to {report_path}")
```

- [ ] **Step 2: Add the CLI entrypoint**

At the end of `federated_lssvm/baseline_run.py`, add:

```python


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
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add federated_lssvm/baseline_run.py
git commit -m "feat: wire WP3 baseline_run into config.report and add CLI entrypoint"
```

---

### Task 6: End-to-end manual verification

**Files:** none (verification only, no code changes).

**Interfaces:** none.

- [ ] **Step 1: Run the baseline on iris**

Run: `python -m federated_lssvm.baseline_run 3 --dataset=iris`
Expected: Completes without error; prints a comparison table per class (setosa/versicolor/virginica) with a "Baseline (plaintext solve + encrypted agg)" row, an OvR multiclass accuracy line, and `models/k=3_baseline/class_{0,1,2}/` gets created with `cryptocontext.bin`, `bias.bin`, `weights.bin`, `phi_mean.npy`, `threshold.npy`. `models/k=3_baseline/report.md` and `metrics.csv` are created/appended.

- [ ] **Step 2: Run the baseline on breast_cancer with a small k**

Run: `python -m federated_lssvm.baseline_run 2 --dataset=breast_cancer`
Expected: Completes without error — confirming the design's claim that WP3 has no `k>=15` bootstrap-slot constraint (unlike `train.py`, which would reject `k=2` for this dataset via `assert_fits_bootstrap_slots`). Prints a binary accuracy line and serializes `models/k=2_baseline/class_0/`.

- [ ] **Step 3: Confirm `infer.py` can load the baseline model via `--model-root`**

Run: `python -m federated_lssvm.infer 3 --dataset=iris --model-root=models/k=3_baseline`
Expected: Loads each class's model from `models/k=3_baseline/class_{i}/` (not `models/k=3/class_{i}/`) and prints per-class + OvR accuracy that matches the numbers `baseline_run.py` printed in Step 1 (same recalibration threshold, same `predict_cipher` scoring path).

- [ ] **Step 4: Confirm `config/report.py` output is well-formed**

Run: `cat models/k=3_baseline/report.md` and `cat models/k=3_baseline/metrics.csv`
Expected: `report.md` has a `## Run <timestamp> — dataset=iris, k=3` section with a results table per class matching what was printed to console; `metrics.csv` has one row per (class, approach) with `accuracy_pct`/`precision_pct`/`f1_pct` populated.

- [ ] **Step 5: Confirm no regressions in the full suite one more time**

Run: `pytest -q`
Expected: PASS.

---

## Plan Self-Review Notes

- **Spec coverage:** Protocol (Task 3), crypto/slot-packing detail (Task 3 Step 3, uses `make_packed_plaintext` throughout), reporting/serialization (Tasks 4-5), `infer.py` compatibility (Task 2), no-checkpointing (Global Constraints + Task 3's absence of any checkpoint code), no bootstrap-slot client-count constraint (verified explicitly in Task 6 Step 2), testing plan (Tasks 1-3's unit tests + Task 6's manual verification) — all spec sections have a corresponding task.
- **Deferred by spec, not by this plan:** the full A/B/C table (needs WP1) and WP2's grid — explicitly out of scope, no task references them as if they existed.
- **Type/signature consistency checked:** `solve_client_plain(X_c, y_c, gamma) -> (w, b)` used identically in Task 1's refactor and Task 3's `baseline_run.py`; `_model_dir(k, class_idx, model_root=None)` signature matches its two call sites (default CLI path and `baseline_run` verification in Task 6); `main()`'s `serialize` parameter introduced in Task 3's signature is first *used* in Task 4 and read by the CLI in Task 5 — consistent across all three.
