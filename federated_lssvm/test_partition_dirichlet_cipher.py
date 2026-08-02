"""Ciphertext validation of the Dirichlet partitioner against the plaintext oracle.

Runs a non-IID client split through the REAL CKKS FedAvg aggregation
(train.fhe_aggregate) and asserts the decrypted global model matches the plaintext
reference (test_partition_dirichlet.py). Uses a small security="notset" context so
the encrypted path runs without the slow 128-bit bootstrap.
"""

import numpy as np
import pytest

pytest.importorskip("openfhe", reason="requires the OpenFHE C++/Python build")

import federated_lssvm.train as T
from federated_lssvm.baseline_run import _NOTSET_DEPTH
from federated_lssvm.solver_selection import resolve_solver_module, DEFAULT_SOLVER_NAME
from lssvm.preprocessing import solve_client_plain
from lssvm.preprocessors import prepare_binary
from lssvm.solvers.utils import make_packed_plaintext

# Tiny/degenerate clients under strong skew provoke a spurious OpenBLAS
# "encountered in matmul" RuntimeWarning (OMP-threaded matmul on small arrays);
# verified benign -- np.seterr(all="raise") does not trip and every solve returns
# finite weights matching the plaintext oracle to 1e-2.
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

_GAMMA = 10.0


@pytest.fixture(scope="module", params=["iris", "breast_cancer"])
def cipher_ctx(request):
    """One notset CKKS context per dataset (setup is the slow part).

    Covers both the iris OvR sub-problem and the 30-dim breast_cancer primary
    dataset, so the encrypted FedAvg is validated at both feature widths.
    """
    solv = resolve_solver_module(DEFAULT_SOLVER_NAME)
    X_tr, X_te, y_tr, y_te, _ = prepare_binary(request.param)[0]
    d = X_tr.shape[1]
    cc, keys = solv.setup_crypto_context(
        _NOTSET_DEPTH, matrix_size=1, n_test=len(X_te), feature_dim=d,
        N=None, security="notset",
    )
    return solv, cc, keys, (X_tr, X_te, y_tr, y_te), d


def _encrypted_fedavg(solv, cc, keys, parts, d):
    """Encrypt each client's plaintext (w,b), aggregate in CKKS, decrypt global."""
    slots = solv.get_slot_count(cc)
    b_cts, w_cts = [], []
    for X_c, y_c in parts:
        w_i, b_i = solve_client_plain(X_c, y_c, _GAMMA)
        w_cts.append(cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, list(w_i), slots)))
        b_cts.append(cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, [b_i], slots)))
    b_global, w_global = T.fhe_aggregate(cc, b_cts, w_cts)
    w_dec = np.array(solv.decrypt_vector(cc, keys, w_global, d))
    b_dec = solv.decrypt_vector(cc, keys, b_global, 1)[0]
    return w_dec, b_dec


def test_dirichlet_fedavg_ciphertext_matches_plaintext_oracle(cipher_ctx):
    solv, cc, keys, (X_tr, X_te, y_tr, y_te), d = cipher_ctx
    parts = T.partition_all(X_tr, y_tr, k=4, partition="dirichlet", alpha=0.5, base_seed=1)

    # plaintext oracle
    preds_p, w_p, b_p = T.plaintext_federated_reference(parts, X_te, _GAMMA)
    # ciphertext
    w_c, b_c = _encrypted_fedavg(solv, cc, keys, parts, d)

    assert np.allclose(w_c, w_p, atol=1e-2), f"w mismatch: {w_c} vs {w_p}"
    assert abs(b_c - b_p) < 1e-2
    # same decision on the test set
    preds_c = np.sign(X_te @ w_c + b_c)
    preds_c[preds_c == 0] = 1.0
    assert np.array_equal(preds_c, preds_p)


def test_zero_positive_client_ciphertext_no_garbage(cipher_ctx):
    solv, cc, keys, (X_tr, X_te, y_tr, y_te), d = cipher_ctx
    # Force the non-IID pathology: client 0 holds ONLY negatives (no positives),
    # the rest hold everything else. The degenerate all-negative local model must
    # aggregate cleanly under encryption -- no NaN / garbage.
    neg = np.where(y_tr == -1.0)[0]
    pos = np.where(y_tr == 1.0)[0]
    c0 = neg[:5]
    c1 = np.concatenate([neg[5:], pos])
    parts = [(X_tr[c0], y_tr[c0]), (X_tr[c1], y_tr[c1])]
    assert not np.any(parts[0][1] == 1.0)  # client 0 truly has no positives

    preds_p, w_p, b_p = T.plaintext_federated_reference(parts, X_te, _GAMMA)
    w_c, b_c = _encrypted_fedavg(solv, cc, keys, parts, d)

    assert np.all(np.isfinite(w_c)) and np.isfinite(b_c)
    assert np.allclose(w_c, w_p, atol=1e-2)
    assert abs(b_c - b_p) < 1e-2
