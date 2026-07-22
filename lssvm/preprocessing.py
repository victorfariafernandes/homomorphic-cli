"""LSSVM preprocessing: build the symmetric block matrix for Least Squares SVM.

Given dataset X, labels y, and regularisation parameter gamma, assembles:

    H = [0       1_N^T          ]   (N+1 x N+1)
        [1_N   Omega + (1/g)*I_N]

    rhs = [0, y_1, ..., y_N]^T

where Omega_ij = y_i * K(x_i, x_j) * y_j.

Kernel options
--------------
linear_kernel            : K(x,y) = x·y                    (default)
polynomial_kernel        : K(x,y) = (x·y + c)^degree
homogeneous_poly_kernel  : K(x,y) = (x·y)^degree

For kernels with explicit finite feature maps (linear, polynomial,
homogeneous_poly) the corresponding feature map functions allow full
primal-weight computation — no training data needed at inference.
"""

from __future__ import annotations

import numpy as np
from math import factorial, sqrt


# ── Kernel functions ──────────────────────────────────────────────


def linear_kernel(X: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
    """K[i,j] = x_i · x_j"""
    if X2 is None:
        X2 = X
    return X @ X2.T


def polynomial_kernel(
    X: np.ndarray, X2: np.ndarray = None, degree: int = 2, c: float = 1.0
) -> np.ndarray:
    """K[i,j] = (x_i · x_j + c)^degree"""
    if X2 is None:
        X2 = X
    return (X @ X2.T + c) ** degree


def homogeneous_poly_kernel(
    X: np.ndarray, X2: np.ndarray = None, degree: int = 2
) -> np.ndarray:
    """K[i,j] = (x_i · x_j)^degree"""
    if X2 is None:
        X2 = X
    return (X @ X2.T) ** degree


# ── Feature maps (explicit φ such that K(x,y) = φ(x)·φ(y)) ───────


def poly_feature_map(X: np.ndarray, degree: int = 2, c: float = 1.0) -> np.ndarray:
    """Explicit feature map for polynomial kernel (x·y + c)^degree.

    Each monomial x^α with |α|=m is scaled by
        sqrt(degree! * c^(degree-m) / ((degree-m)! * Π αᵢ!))
    so that φ(x)·φ(y) == (x·y + c)^degree exactly (multinomial theorem).
    Output shape: (N, C(d+degree, degree)).
    """
    from sklearn.preprocessing import PolynomialFeatures

    pf = PolynomialFeatures(degree=degree, include_bias=True)
    phi = pf.fit_transform(X).copy()
    deg_fact = factorial(degree)
    for i, powers in enumerate(pf.powers_):
        m = int(powers.sum())
        k = degree - m          # how many times c appears in this term
        denom = factorial(k)
        for p in powers:
            denom *= factorial(int(p))
        phi[:, i] *= sqrt(deg_fact * (c ** k) / denom)
    return phi


def homogeneous_poly_feature_map(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Explicit feature map for homogeneous polynomial kernel (x·y)^degree.

    Only degree-exactly monomials are kept. Each x^α is scaled by
        sqrt(degree! / Π αᵢ!)
    so that φ(x)·φ(y) == (x·y)^degree exactly (multinomial theorem).
    Output shape: (N, C(d+degree-1, degree)).
    """
    from sklearn.preprocessing import PolynomialFeatures

    pf = PolynomialFeatures(degree=degree, include_bias=False)
    phi_all = pf.fit_transform(X)
    mask = pf.powers_.sum(axis=1) == degree
    phi = phi_all[:, mask].copy()
    deg_fact = factorial(degree)
    for i, powers in enumerate(pf.powers_[mask]):
        denom = 1
        for p in powers:
            denom *= factorial(int(p))
        phi[:, i] *= sqrt(deg_fact / denom)
    return phi


# ── Feature centering and normalization ──────────────────────────


def center_features(
    Phi: np.ndarray,
    phi_mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Center a feature matrix by subtracting the training mean.

    Training mode (phi_mean=None): compute mean from Phi, return (Phi_c, phi_mean).
    Inference mode (phi_mean given): subtract provided mean, return (Phi_c, phi_mean).
    phi_mean MUST be fit on training data only — never refit on test data.
    """
    if phi_mean is None:
        phi_mean = Phi.mean(axis=0)
    return Phi - phi_mean, phi_mean

def normalize_features(Phi: np.ndarray) -> np.ndarray:
    """Normalize each row to unit L2 norm: φ_n(x) = φ(x) / ‖φ(x)‖.

    Rows with norm < 1e-12 are left unchanged to avoid division by zero.
    """
    norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Phi / norms


def preprocess_features(
    X: np.ndarray,
    feature_map,
    phi_mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class feature pipeline — identical in plaintext and FHE paths.

    This is the single source of truth for how features enter the KKT system,
    so that the plaintext reference, the FHE solver, and inference all operate
    on byte-for-byte the same feature frame.

    Linear kernel (``feature_map is None``): the already-standardized features
    are used unchanged. Centering is a no-op and unit-L2 normalization is
    deliberately *skipped* — normalizing the raw 4-D features destroys the
    linear separability of setosa (drops 100% → 86.7%). ``phi_mean`` is
    returned as a zero vector so that inference (which subtracts ``phi_mean``)
    stays a no-op and serialized models remain self-describing.

    Non-linear kernels (explicit ``feature_map``): map → center by the training
    mean → normalize each row to unit L2 norm. Centering/normalization bounds
    the KKT entries and narrows the Chebyshev approximation intervals the FHE
    solver relies on, and lets versicolor/virginica reach ~90% instead of
    collapsing to the degenerate all-positive classifier.

    Modes
    -----
    Training  : pass ``phi_mean=None`` to fit the mean on ``X``.
    Inference : pass the stored ``phi_mean`` from training.

    Returns ``(X_processed, phi_mean)``.
    """
    X = np.asarray(X, dtype=float)
    if feature_map is None:
        if phi_mean is None:
            phi_mean = np.zeros(X.shape[1])
        return X, phi_mean
    Phi = feature_map(X)
    Phi, phi_mean = center_features(Phi, phi_mean=phi_mean)
    Phi = normalize_features(Phi)
    return Phi, phi_mean


# ── Matrix assembly ───────────────────────────────────────────────


def build_omega(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Omega_ij = y_i * K_ij * y_j"""
    y_col = y.reshape(-1, 1)
    return y_col * K * y_col.T


def build_lssvm_matrix(
    X: np.ndarray,
    y: np.ndarray,
    gamma: float,
    kernel=linear_kernel,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (N+1)x(N+1) block matrix H and the rhs vector.

    Parameters
    ----------
    X : (N, d) feature matrix
    y : (N,)   labels in {-1, +1}
    gamma : regularisation parameter (> 0)

    Returns
    -------
    H   : (N+1, N+1) symmetric block matrix
    rhs : (N+1,)     right-hand side vector [0, y_1, ..., y_N]
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    N = len(y)
    K = kernel(X)
    Omega = build_omega(K, y)

    H = np.zeros((N + 1, N + 1))
    H[0, 1:] = 1.0
    H[1:, 0] = 1.0
    H[1:, 1:] = Omega + (1.0 / gamma) * np.eye(N)

    rhs = np.zeros(N + 1)
    rhs[1:] = y

    return H, rhs


def build_primal_augmented(
    Phi: np.ndarray,
    y: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the augmented data matrix M_aug and target t for the primal solve.

    This is the SPD-equivalent reformulation of ``build_lssvm_matrix``'s dual
    system. Householder QR of M_aug solves ``min ||M_aug @ x - t||^2``, whose
    solution ``x = [b; w]`` is byte-identical (to ~1e-13) to the dual solver's
    recovered ``(b, w = Phi^T(alpha * y))`` -- same LSSVM problem, but the QR runs
    on an SPD normal-equations system so every pivot is bounded below by
    ``sqrt(lambda_min)`` and a fixed reflection sign is safe (no per-step data
    simulation needed). See docs/superpowers/plans for the derivation.

    The primal normal equations are ``H @ [b; w] = [sum(y); s]`` with
    ``H = [[N, p^T], [p, G + I/gamma]]``, ``G = Phi^T Phi``, ``p = Phi^T y``,
    ``s = Phi^T 1``, ``N = n``. Since ``N = y^T y`` and ``p = Phi^T y``, this
    factors as ``H = M^T M + diag(0, 1/gamma, ...)`` with ``M = [y | Phi]``, so
    stacking the ridge as ``d`` virtual rows gives M_aug below and the QR of the
    tall matrix produces R = the Cholesky factor of H.

    Parameters
    ----------
    Phi : (n, d) feature matrix (already mapped/centered/normalized per class)
    y : (n,) labels in {-1, +1}
    gamma : regularisation parameter (> 0)

    Returns
    -------
    M_aug : (n + d, d + 1) augmented data matrix ``[[y, Phi], [0, sqrt(1/gamma) I]]``
    target : (n + d,) least-squares target ``[1, ..., 1, 0, ..., 0]`` (n ones, d zeros)
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = Phi.shape

    M_aug = np.zeros((n + d, d + 1))
    M_aug[:n, 0] = y
    M_aug[:n, 1:] = Phi
    M_aug[n:, 1:] = np.sqrt(1.0 / gamma) * np.eye(d)

    target = np.zeros(n + d)
    target[:n] = 1.0

    return M_aug, target


def recalibration_threshold(train_scores: np.ndarray, y_train: np.ndarray) -> float:
    """Class-mean-midpoint decision threshold for LSSVM decision scores.

    This solver's KKT variant (constant border / label rhs) yields the correct
    separating direction but a bias that is only centred when the two classes are
    balanced. For imbalanced problems (including every One-vs-Rest sub-problem,
    which is 1-vs-rest and thus imbalanced) the accuracy-optimal threshold is the
    midpoint of the per-class mean training scores rather than 0. Subtracting this
    threshold from the decision score recovers the well-calibrated (Suykens-style)
    decision boundary. Returns 0.0 if either class is absent.
    """
    pos = train_scores[y_train == 1.0]
    neg = train_scores[y_train == -1.0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(0.5 * (pos.mean() + neg.mean()))


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


def prepare_iris_binary(
    class_idx: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load Iris, split, scale, and return OvR binary sub-problems.

    Parameters
    ----------
    class_idx : If given, return only that class-vs-rest problem.
                If None, return all three OvR problems.
    test_size : Fraction of data held out for testing.
    random_state : Seed for reproducible splits.

    Returns
    -------
    List of (X_train, X_test, y_train, y_test, class_name) tuples.
    y values are +1 (target class) or -1 (rest).

    Back-compat wrapper: delegates to ``lssvm.preprocessors.iris.prepare_binary``.
    New code should prefer ``lssvm.preprocessors.prepare_binary(dataset, ...)``.
    """
    from lssvm.preprocessors import iris as iris_preparer

    return iris_preparer.prepare_binary(
        class_idx=class_idx, test_size=test_size, random_state=random_state
    )


def prepare_dataset(
    X: np.ndarray, y: np.ndarray, gamma: float
) -> list[tuple[list, list, dict]]:
    """Build LSSVM matrices for binary or multi-class (OvR) data.

    Parameters
    ----------
    X : (N, d) feature matrix (already scaled)
    y : (N,)   integer class labels

    Returns
    -------
    List of (H_list, rhs_list, meta) where H_list/rhs_list are plain
    Python lists (compatible with FHE functions) and meta is a dict
    with 'y_binary' and 'class_label'.
    """
    classes = np.unique(y)

    if len(classes) == 2:
        y_binary = np.where(y == classes[1], 1.0, -1.0)
        H, rhs = build_lssvm_matrix(X, y_binary, gamma)
        meta = {"y_binary": y_binary, "class_label": f"{classes[1]} vs {classes[0]}"}
        return [(H.tolist(), rhs.tolist(), meta)]

    problems = []
    for c in classes:
        y_binary = np.where(y == c, 1.0, -1.0)
        H, rhs = build_lssvm_matrix(X, y_binary, gamma)
        meta = {"y_binary": y_binary, "class_label": f"class {c} vs rest"}
        problems.append((H.tolist(), rhs.tolist(), meta))
    return problems


def gcv_gamma(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    feature_map,
    bounds: tuple[float, float] = (1e-6, 1e3),
) -> tuple[float, float]:
    """GCV-optimal gamma for LSSVM/KRR. Returns (best_gamma, gcv_value).

    One eigendecomposition of K=ΦΦᵀ; GCV(γ) evaluated in O(r) per query.
    Bias handled by centering labels (ỹ = y − ȳ).
    Only call for non-linear kernels — GCV finds suboptimal γ for linearly separable data.
    """
    from scipy.optimize import minimize_scalar

    Phi = feature_map(X_tr) if feature_map else X_tr
    Phi, _ = center_features(Phi)
    Phi = normalize_features(Phi)

    y_c = y_tr - y_tr.mean()

    K = Phi @ Phi.T
    eigvals, eigvecs = np.linalg.eigh(K)
    mask = eigvals > 1e-10 * eigvals[-1]
    lam = eigvals[mask]
    U   = eigvecs[:, mask]

    u_tilde = U.T @ y_c
    y_perp_sq = float(y_c @ y_c - u_tilde @ u_tilde)
    N = len(y_tr)

    def gcv(gamma: float) -> float:
        gl  = gamma * lam
        inv = 1.0 / (gl + 1.0)
        rss = float(np.dot(u_tilde ** 2, inv ** 2)) + y_perp_sq
        trH_frac = float(np.dot(gl, inv)) / N
        denom = (1.0 - trH_frac) ** 2
        return (rss / N) / denom if denom > 1e-12 else 1e12

    result = minimize_scalar(gcv, bounds=bounds, method="bounded")
    return float(result.x), float(result.fun)


if __name__ == "__main__":
    splits = prepare_iris_binary()
    for X_tr, X_te, y_tr, y_te, name in splits:
        H, rhs = build_lssvm_matrix(X_tr, y_tr, gamma=1.0)
        sym = np.allclose(H, H.T)
        cond = np.linalg.cond(H)
        print(
            f"{name} vs rest:  H shape {H.shape},  "
            f"symmetric: {sym},  cond: {cond:.1f}"
        )
