"""Plaintext validation of the Dirichlet non-IID partitioner (per-OvR binary skew).

Project convention: the plaintext path is the correctness oracle; the ciphertext
test (test_partition_dirichlet_cipher.py) checks the encrypted FedAvg against the
plaintext reference exercised here.
"""

import numpy as np
import pytest

from federated_lssvm.train import partition_all, plaintext_federated_reference
from lssvm.preprocessors import prepare_binary

# Severe-skew clients are tiny/degenerate; their solves provoke a spurious
# OpenBLAS "encountered in matmul" RuntimeWarning (verified benign -- weights stay
# finite). Filter it so the oracle's output is pristine.
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")


def _indices_of(partitions):
    """Recover the original row indices from an identity-feature partition."""
    return [np.sort(Xc[:, 0].astype(int)) for Xc, _ in partitions]


def _identity_binary_problem(n_pos, n_neg):
    """X where row i is [i]; y is +1 for the first n_pos rows, -1 for the rest."""
    n = n_pos + n_neg
    X = np.arange(n, dtype=float).reshape(n, 1)
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
    return X, y


def _pos_fraction_spread(partitions):
    """Std of the per-client positive fraction (0 when perfectly balanced)."""
    fracs = [float(np.mean(yc == 1.0)) if len(yc) else 0.0 for _, yc in partitions]
    return float(np.std(fracs))


# ── structural properties ──────────────────────────────────────────────────

def test_dirichlet_partition_disjoint_and_complete():
    X, y = _identity_binary_problem(40, 40)
    parts = partition_all(X, y, k=5, partition="dirichlet", alpha=0.5)

    all_idx = np.concatenate(_indices_of(parts))
    assert np.array_equal(np.sort(all_idx), np.arange(80))  # complete, no dupes


def test_dirichlet_no_empty_client_even_at_low_alpha():
    X, y = _identity_binary_problem(30, 30)
    parts = partition_all(X, y, k=8, partition="dirichlet", alpha=0.05)
    assert all(len(yc) >= 1 for _, yc in parts)


def test_dirichlet_deterministic_for_same_seed():
    X, y = _identity_binary_problem(30, 30)
    a = partition_all(X, y, k=6, partition="dirichlet", alpha=0.3, base_seed=7)
    b = partition_all(X, y, k=6, partition="dirichlet", alpha=0.3, base_seed=7)
    assert [i.tolist() for i in _indices_of(a)] == [i.tolist() for i in _indices_of(b)]


# ── skew behavior ──────────────────────────────────────────────────────────

def test_low_alpha_more_skewed_than_high_alpha():
    X, y = _identity_binary_problem(50, 50)
    spread_hi = _pos_fraction_spread(
        partition_all(X, y, k=6, partition="dirichlet", alpha=100.0)
    )
    spread_lo = _pos_fraction_spread(
        partition_all(X, y, k=6, partition="dirichlet", alpha=0.1)
    )
    assert spread_lo > spread_hi


def test_low_alpha_produces_zero_positive_client_and_logs(capsys):
    # Few positives spread over many clients under strong skew => some client
    # ends up with none. This degenerate all-negative client is the pathology
    # under study; it must be allowed and counted, not crash.
    X, y = _identity_binary_problem(6, 60)
    parts = partition_all(X, y, k=10, partition="dirichlet", alpha=0.05)

    zero_pos = [i for i, (_, yc) in enumerate(parts) if not np.any(yc == 1.0)]
    assert zero_pos, "expected at least one client with no positive examples"

    out = capsys.readouterr().out
    assert "no positive examples" in out


# ── IID regression (unchanged behavior) ────────────────────────────────────

def test_iid_default_is_class_stratified_balanced():
    X, y = _identity_binary_problem(40, 40)
    parts = partition_all(X, y, k=4)  # default partition="iid"
    # every client keeps the global 50/50 ratio
    for _, yc in parts:
        assert float(np.mean(yc == 1.0)) == pytest.approx(0.5, abs=0.1)


# ── plaintext FedAvg oracle (real iris class-0, setosa vs rest) ─────────────

# Use a real OvR sub-problem, not synthetic blobs: this LSSVM formulation
# degenerates to w=0 on perfectly balanced/symmetric data, whereas real OvR
# problems are imbalanced. setosa-vs-rest is linearly separable, so near-IID
# FedAvg reaches 100% and skew degrades it -- the reference the ciphertext test
# is checked against. gamma=10 matches the light regularization the solver uses.
_ORACLE_GAMMA = 10.0


def _iris_class0():
    X_tr, X_te, y_tr, y_te, _ = prepare_binary("iris")[0]
    return X_tr, X_te, y_tr, y_te


def _fed_test_accuracy(X_tr, y_tr, X_te, y_te, k, alpha, base_seed):
    parts = partition_all(X_tr, y_tr, k=k, partition="dirichlet", alpha=alpha,
                          base_seed=base_seed)
    preds, w, b = plaintext_federated_reference(parts, X_te, gamma=_ORACLE_GAMMA)
    assert np.all(np.isfinite(w)) and np.isfinite(b)  # valid model, no NaN
    return float(np.mean(preds == y_te))


def test_plaintext_fedavg_valid_and_degrades_with_alpha():
    X_tr, X_te, y_tr, y_te = _iris_class0()
    # average over seeds to keep the degradation signal stable
    acc_hi = np.mean([_fed_test_accuracy(X_tr, y_tr, X_te, y_te, k=4, alpha=100.0,
                                         base_seed=s) for s in range(6)])
    acc_lo = np.mean([_fed_test_accuracy(X_tr, y_tr, X_te, y_te, k=4, alpha=0.05,
                                         base_seed=s) for s in range(6)])

    assert acc_hi > 0.9             # near-IID FedAvg classifies separable setosa well
    assert acc_lo < acc_hi          # severe non-IID measurably degrades accuracy


def test_plaintext_fedavg_breast_cancer_primary_dataset():
    # breast_cancer is the revision's PRIMARY dataset (binary, 30-dim, 455 pts).
    # Being binary, per-OvR skew is the fully faithful non-IID case. Use k=20
    # (the FHE-recommended client count) and confirm a valid model that degrades
    # from near-IID to severe skew, with zero-positive clients tolerated.
    X_tr, X_te, y_tr, y_te, _ = prepare_binary("breast_cancer")[0]

    def acc(alpha, seed):
        parts = partition_all(X_tr, y_tr, k=20, partition="dirichlet", alpha=alpha,
                              base_seed=seed)
        preds, w, b = plaintext_federated_reference(parts, X_te, _ORACLE_GAMMA)
        assert np.all(np.isfinite(w)) and np.isfinite(b)  # no NaN even under skew
        return float(np.mean(preds == y_te))

    acc_hi = np.mean([acc(100.0, s) for s in range(6)])
    acc_lo = np.mean([acc(0.05, s) for s in range(6)])
    assert acc_hi > 0.7             # near-IID FedAvg is well above chance
    assert acc_lo < acc_hi          # severe non-IID degrades toward majority-class
