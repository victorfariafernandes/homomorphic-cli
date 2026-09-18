"""Primal-augmented reformulation: equivalence to the dual solver + builder shape.

The primal system solved via Householder QR on ``build_primal_augmented``'s
augmented data matrix must recover byte-identical ``(b, w)`` to the existing dual
solver (``build_lssvm_matrix`` + recovery ``w = Phi^T(alpha * y)``) -- this is the
executable equivalence proof for the SPD reformulation.
"""

import numpy as np
import pytest

from lssvm.preprocessing import build_lssvm_matrix, build_primal_augmented


def _dual_solution(Phi, y, gamma):
    """Solve the dual system and recover primal (b, w), matching the FHE dual path."""
    H, rhs = build_lssvm_matrix(Phi, y, gamma)
    sol = np.linalg.solve(H, rhs)
    b = sol[0]
    alpha = sol[1:]
    w = Phi.T @ (alpha * y)
    return np.concatenate([[b], w])


def _primal_solution(Phi, y, gamma):
    """Solve the primal least-squares problem via the augmented matrix."""
    M_aug, target = build_primal_augmented(Phi, y, gamma)
    x, *_ = np.linalg.lstsq(M_aug, target, rcond=None)
    return x


def test_build_primal_augmented_shape_and_blocks():
    rng = np.random.default_rng(0)
    n, d = 7, 4
    Phi = rng.normal(size=(n, d))
    y = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    gamma = 1.1

    M_aug, target = build_primal_augmented(Phi, y, gamma)

    assert M_aug.shape == (n + d, d + 1)
    assert target.shape == (n + d,)
    # first column is [y; 0], first d+1... check blocks
    np.testing.assert_allclose(M_aug[:n, 0], y)
    np.testing.assert_allclose(M_aug[:n, 1:], Phi)
    np.testing.assert_allclose(M_aug[n:, 1:], np.sqrt(1.0 / gamma) * np.eye(d))
    np.testing.assert_allclose(M_aug[n:, 0], np.zeros(d))
    np.testing.assert_allclose(target[:n], np.ones(n))
    np.testing.assert_allclose(target[n:], np.zeros(d))


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("gamma", [1.1, 3.9, 0.5])
def test_primal_matches_dual_solution(seed, gamma):
    rng = np.random.default_rng(seed)
    n, d = 20, 6
    Phi = rng.normal(size=(n, d))
    y = np.where(rng.normal(size=n) > 0, 1.0, -1.0)

    ref = _dual_solution(Phi, y, gamma)
    got = _primal_solution(Phi, y, gamma)

    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_primal_matches_dual_imbalanced_small():
    # Tiny imbalanced client -- the regime where the DUAL Householder QR pivots
    # collapsed; the primal reformulation must still match exactly.
    rng = np.random.default_rng(3)
    Phi = rng.normal(size=(3, 4))
    y = np.array([1.0, 1.0, -1.0])
    gamma = 1.1

    ref = _dual_solution(Phi, y, gamma)
    got = _primal_solution(Phi, y, gamma)

    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_build_primal_augmented_rejects_nonpositive_gamma():
    Phi = np.ones((3, 2))
    y = np.array([1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        build_primal_augmented(Phi, y, 0.0)
