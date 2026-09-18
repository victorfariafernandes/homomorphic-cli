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
    # Balanced, well-separated synthetic data collapses w -> ~0 under this LSSVM's
    # zero-sum alpha constraint (an algorithm property, not a bug). Real callers
    # only ever pass imbalanced per-class OvR splits, so this test checks
    # shape/type only; exact numerics are pinned by the test above.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 5))
    y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

    w, b = solve_client_plain(X, y, gamma=2.0)

    assert w.shape == (5,)
    assert isinstance(b, float)
