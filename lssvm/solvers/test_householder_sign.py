"""Regression tests for Householder sign selection (catastrophic cancellation fix).

Fixed-sign (+1) reflections blow up when a pivot column has x0 ~ -||x||: v.v collapses,
the he_inv Chebyshev domain [vtv/2, 2*vtv] becomes tiny, and the encrypted value falls
outside it -> polynomial diverges -> NaN (notset) or undecodable noise (128).
Killed 8/40 k=40 class-0 clients (2026-07-16). Fix: choose s_k = sign(x0) per step in
plaintext (H is plaintext at the client) so v0 = s*(|x0| + norm) never cancels.
"""

import numpy as np
import pytest

from lssvm.solvers.utils import simulate_norms


def _cancellation_matrix():
    # First pivot column ~ [-1, 1e-3, 1e-3]: x0 ~ -||x|| -> sign=+1 gives vtv ~ 1e-6
    return [
        [-1.0, 0.3, 0.2],
        [1e-3, 0.9, 0.1],
        [1e-3, 0.2, 0.8],
    ]


def test_simulate_norms_returns_per_step_sign():
    info = simulate_norms(_cancellation_matrix())
    assert all(len(step) == 3 for step in info), "expected (norm_sq, vtv, sign) triples"
    signs = [s for _, _, s in info]
    assert all(s in (-1.0, 1.0) for s in signs)
    assert signs[0] == -1.0, "x0 < 0 must select sign=-1 to avoid cancellation"


def test_simulate_norms_avoids_cancellation():
    info = simulate_norms(_cancellation_matrix())
    for k, step in enumerate(info):
        norm_sq, vtv = step[0], step[1]
        # with s = sign(x0): v.v = 2*norm*(|x0| + norm) >= 2*norm^2
        assert vtv >= 1.99 * norm_sq, (
            f"step {k}: vtv={vtv:.3g} < 2*norm_sq={2 * norm_sq:.3g} (cancellation)"
        )


@pytest.fixture(scope="module")
def k40_class0_setup():
    """Deterministic k=40 class-0 pipeline pieces + a shared notset context."""
    import federated_lssvm.train as T
    from federated_lssvm.solver_selection import resolve_solver_module
    from lssvm.preprocessing import prepare_iris_binary, preprocess_features

    solv = resolve_solver_module("qr_row")
    T.solv = solv
    splits = prepare_iris_binary()
    parts, max_client_n, max_feat_dim = T.compute_problem_dims(splits, 40, None)
    X_tr, X_te, y_tr, _, _ = splits[0]
    _, feature_map, _, gamma, _, phi_mean = T.class_setup(0, X_tr, y_tr, verbose=False)
    depth = T.context_depth(max_client_n, "notset")
    cc, keys = solv.setup_crypto_context(
        depth, matrix_size=max_client_n, n_test=len(X_te),
        feature_dim=max_feat_dim, N=T.N_OVERRIDE, security="notset",
    )

    def solve_client(cid):
        X_c, y_c = parts[0][cid]
        X_f, _ = preprocess_features(X_c, feature_map, phi_mean=phi_mean)
        H, rhs = T.build_lssvm_matrix(X_f, y_c, gamma)
        sol = np.linalg.solve(H, rhs)
        w_ref = X_f.T @ (sol[1:] * y_c)
        b_ct, w_ct, _ = solv.solver(
            cc, keys, H.tolist(), rhs.tolist(), X_f, y_c,
            D_sqrt=T.D_SQRT, D_inv=T.D_INV, D_inv_backsub=T.D_INV_BACKSUB,
            security="notset",
        )
        w = np.array(solv.decrypt_vector(cc, keys, w_ct, X_f.shape[1]))
        return w, w_ref

    return solve_client


def test_cancellation_client_35_solves(k40_class0_setup):
    """Client 35 (min plaintext vtv=0.001 under sign=+1) previously returned NaN."""
    w, w_ref = k40_class0_setup(35)
    assert np.all(np.isfinite(w)), f"solve produced non-finite weights: {w}"
    rel = np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref)
    assert rel < 0.35, f"w_rel_err={rel:.3f} vs plaintext reference {w_ref}"


def test_control_client_1_still_correct(k40_class0_setup):
    """Client 1 solved correctly before the fix (10.6% rel err) and must stay correct."""
    w, w_ref = k40_class0_setup(1)
    rel = np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref)
    assert rel < 0.2, f"w_rel_err={rel:.3f} vs plaintext reference {w_ref}"
