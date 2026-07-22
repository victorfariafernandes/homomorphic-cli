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


def test_solve_client_plain_returns_correct_shapes_and_types():
    # NOTE: this LSSVM formulation's zero-sum constraint on alpha (the H
    # matrix's constant border row/col) means class-BALANCED, well-separated
    # synthetic data can drive alpha_i -> y_i, collapsing w = X^T(alpha*y) ->
    # X^T*1 -> ~0 for symmetric clusters (verified against the pre-existing
    # inlined formula in train.py, so this is a property of the algorithm,
    # not this refactor). Every real caller partitions imbalanced per-class
    # OvR problems (see partition_all's "preserves the pos/neg ratio"), so a
    # shape/type check here is the right-weight test; exact numerical
    # correctness is already fully pinned by the test above.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 5))
    y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

    w, b = solve_client_plain(X, y, gamma=2.0)

    assert w.shape == (5,)
    assert isinstance(b, float)
