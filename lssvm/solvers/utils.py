"""Shared FHE utilities for Householder QR solvers."""

from __future__ import annotations

import math
import random

# Slot width used for every plaintext/ciphertext touched by the bootstrapping
# (security="128") path. Must be >= the largest rotation index ever used
# against a bootstrapped ciphertext (n_test=30 for Iris, feature_dim<=15,
# matrix_size<=7) -- validated empirically (see plan doc): EvalRotate wraps
# around modulo the ciphertext's bootstrap num_slots with no error, so this
# must stay >= max(n_test, feature_dim, matrix_size) for every caller.
SPARSE_BOOTSTRAP_SLOTS = 32


def make_packed_plaintext(cc, vals: list, slots: int):
    """Build a CKKS plaintext with explicit slot-count metadata.

    Required whenever the plaintext will be multiplied against a
    sparse-bootstrapped ciphertext: EvalMult silently returns near-zero
    garbage (no exception) if the plaintext's `slots` metadata doesn't match
    the ciphertext's bootstrap num_slots. Passing `slots` explicitly here
    (rather than relying on the default full-ring-width encoding) is the
    validated fix.
    """
    padded = list(vals) + [0.0] * (slots - len(vals))
    return cc.MakeCKKSPackedPlaintext(padded, 1, 0, None, slots)


def bootstrap_all(cc, cts: list, min_level: int = 0) -> list:
    """Bootstrap every ciphertext in a list, returning a new list of refreshed ciphertexts.

    min_level: skip ciphertexts below this level. EvalBootstrap silently returns the input
    unchanged (a no-op, after spending the full bootstrap compute time) whenever the
    ciphertext isn't depleted past ~bootdepth levels — callers pass their bootdepth here so
    those provably-pointless calls are skipped entirely.
    """
    return [cc.EvalBootstrap(ct) if ct.GetLevel() >= min_level else ct for ct in cts]


def write_security_marker(out_dir: str, security: str) -> None:
    """Persist which security path (security="128"|"notset") produced a checkpoint.

    Needed at load time so predict_cipher is called with the matching `slots` value
    (SPARSE_BOOTSTRAP_SLOTS for security="128", full ring width otherwise).
    """
    with open(f"{out_dir}/security.txt", "w", encoding="utf-8") as f:
        f.write(security)


def read_security_marker(out_dir: str) -> str:
    """Read the security marker written by write_security_marker.

    Defaults to "notset" for checkpoints created before this marker existed.
    """
    import os

    path = f"{out_dir}/security.txt"
    if not os.path.exists(path):
        return "notset"
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def slots_for_security(cc, security: str) -> int:
    """Resolve the packing width to use for a given security mode."""
    return SPARSE_BOOTSTRAP_SLOTS if security == "128" else cc.GetRingDimension() // 2


def _resolve_slots(cc, slots: int | None) -> int:
    return slots if slots is not None else cc.GetRingDimension() // 2


def encrypt_row(cc, keys, row: list, slots: int | None = None):
    """Encrypt a single matrix row, zero-padded to fill the slot count."""
    slots = _resolve_slots(cc, slots)
    return cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, row, slots))

def encrypt_matrix_rows(cc, keys, A: list, slots: int | None = None) -> list:
    """Encrypt matrix A as a list of row ciphertexts."""
    return [encrypt_row(cc, keys, row, slots=slots) for row in A]


def encrypt_matrix_cols(cc, keys, A: list, slots: int | None = None) -> list:
    """Encrypt m x n matrix A as n column ciphertexts: R_cts[j] = [A[0,j], ..., A[m-1,j], 0, ...]."""
    m, n = len(A), len(A[0])
    slots = _resolve_slots(cc, slots)
    R_cts = []
    for j in range(n):
        col = [A[i][j] for i in range(m)]
        R_cts.append(cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, col, slots)))
    return R_cts



def encrypt_identity_cols(cc, keys, m: int, slots: int | None = None) -> list:
    """Encrypt the m x m identity matrix as a list of column ciphertexts."""
    slots = _resolve_slots(cc, slots)
    cols  = []
    for j in range(m):
        col = [1.0 if i == j else 0.0 for i in range(m)]
        cols.append(cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, col, slots)))
    return cols


def decrypt_vector(cc, keys, ct, length: int) -> list:
    """Decrypt a ciphertext and return the first `length` real values."""
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(length)
    return [v.real for v in pt.GetCKKSPackedValue()]


def decrypt_matrix_rows(cc, keys, R_cts: list, n: int) -> list:
    """Decrypt row-packed ciphertexts back to an m x n matrix."""
    return [decrypt_vector(cc, keys, ct, n) for ct in R_cts]


def decrypt_matrix_cols(cc, keys, Q_cols: list, m: int) -> list:
    """Decrypt column-packed ciphertexts back to an m x m matrix."""
    cols = [decrypt_vector(cc, keys, ct, m) for ct in Q_cols]
    return [[cols[j][i] for j in range(m)] for i in range(m)]


def _cheby_depth(degree: int) -> int:
    """Multiplicative depth for EvalChebyshevFunction: 2 * ceil(log2(degree + 1)) (BSGS estimate)."""
    return 2 * math.ceil(math.log2(degree + 1))


def depth_for_size(m: int, n: int, D_sqrt: int = 64, D_inv: int = 64,
                   D_inv_backsub: int = None, safety_factor: float = 1.15,
                   depth_override: int = None) -> int:
    """Estimated multiplicative depth for m x n FHE Householder QR + encrypted-pivot back-substitution.

    D_inv:          Chebyshev degree for 1/t in QR steps.
    D_inv_backsub:  Chebyshev degree for 1/t in back-sub pivot inversion (defaults to D_inv).
    safety_factor:  multiplicative buffer applied to calibrated core estimate.
    depth_override: if set, bypass estimation and return this depth (floored at 30).
    """
    if depth_override is not None:
        return max(30, int(depth_override))

    if D_inv_backsub is None:
        D_inv_backsub = D_inv

    steps = min(m, n)
    # Calibrated for FLEXIBLEAUTO: ct*pt masks often do not consume full levels.
    qr_depth = steps * (_cheby_depth(D_sqrt) + _cheby_depth(D_inv) + 1)
    backsub_depth = n * (_cheby_depth(D_inv_backsub) + 1)
    base_overhead = 8

    estimate = math.ceil((qr_depth + backsub_depth + base_overhead) * safety_factor)
    return max(30, estimate)


def he_sqrt(cc, ct, a: float, b: float, degree: int = 16):
    """Chebyshev sqrt on [a, b]. Clamps to a to avoid sqrt(negative). Depth: ~2*ceil(log2(degree))."""
    return cc.EvalChebyshevFunction(lambda t: math.sqrt(max(t, a)), ct, a, b, degree)


def he_inv(cc, ct, a: float, b: float, degree: int = 16):
    """Chebyshev 1/t on [a, b]. Domain must not cross zero (both positive or both negative)."""
    return cc.EvalChebyshevFunction(lambda t: 1.0 / t, ct, a, b, degree)


def safe_rotate(cc, ct, k: int):
    """EvalRotate left by k, no-op when k == 0 (OpenFHE has no automorphism key for index 0)."""
    return cc.EvalRotate(ct, k) if k != 0 else ct


def replicate_slot_0(cc, ct, active_slots: int):
    """Broadcast slot 0 to the first active_slots via rotation-doubling tree.

    Depth cost: 0 (additions/rotations only). Requires other slots in ct to be zero.
    """
    result = ct
    step = 1
    while step < active_slots:
        result = cc.EvalAdd(result, cc.EvalRotate(result, -step))
        step *= 2
    return result


def sum_slots(cc, ct, n: int):
    """Tree-based sum of slots 0..n-1 into slot 0. Depth cost: 0 (additions only). Caller must mask with e_0 after."""
    result = ct
    step = 1
    while step < n:
        result = cc.EvalAdd(result, cc.EvalRotate(result, step))
        step *= 2
    return result

def simulate_norms(A: list) -> list:
    """Plaintext Householder QR returning [(norm_sq_k, vtv_k, sign_k), ...] per step.

    sign_k = sign(x0) is selected here in plaintext (A is client-known) and baked into
    the FHE circuit as a constant: v0 = x0 + sign*norm = sign*(|x0| + norm), so
    v.v = 2*norm*(|x0| + norm) >= 2*norm^2 and never suffers catastrophic cancellation.
    A fixed sign=+1 collapses v.v toward 0 whenever x0 ~ -||x||, which shrinks the
    he_inv Chebyshev domain [vtv/margin, vtv*margin] below the approximation/bootstrap
    noise floor — the encrypted value then escapes the domain and the solve blows up.
    """
    m, n = len(A), len(A[0])
    r = [list(map(float, row)) for row in A]
    info = []

    for k in range(min(m, n)):
        x = [r[i][k] for i in range(k, m)]
        norm_sq = sum(xi * xi for xi in x)
        norm_x = math.sqrt(norm_sq)
        if norm_x < 1e-15:
            info.append((1.0, 1.0, 1.0))
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        v0 = x[0] + sign * norm_x
        vtv = 2.0 * sign * norm_x * v0
        if vtv < 1e-15:
            info.append((norm_sq, 1.0, sign))
            continue
        info.append((norm_sq, vtv, sign))

        v = [v0] + x[1:]
        norm_v = math.sqrt(sum(vi * vi for vi in v))
        if norm_v < 1e-15:
            continue
        v = [vi / norm_v for vi in v]
        for j in range(k, n):
            dot = sum(v[i] * r[k + i][j] for i in range(len(v)))
            for i in range(len(v)):
                r[k + i][j] -= 2.0 * v[i] * dot

    return info

def simulate_diag_values(A: list) -> list:
    """Plaintext QR simulation returning the diagonal values R[i][i] for scalar inverse in back-sub."""
    from lssvm.qr_householder import householder_qr

    _, R = householder_qr(A)
    return [R[i][i] for i in range(len(R[0]))]


def _simulate_qr_signed(A: list) -> list:
    """Plaintext Householder QR with sign=sign(x0) (matching FHE solver). Returns the R matrix."""
    m, n = len(A), len(A[0])
    r = [list(map(float, row)) for row in A]

    for k in range(min(m, n)):
        x = [r[i][k] for i in range(k, m)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x < 1e-15:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        v = [x[0] + sign * norm_x] + x[1:]
        norm_v = math.sqrt(sum(vi * vi for vi in v))
        if norm_v < 1e-15:
            continue
        v = [vi / norm_v for vi in v]
        for j in range(k, n):
            dot = sum(v[i] * r[k + i][j] for i in range(len(v)))
            for i in range(len(v)):
                r[k + i][j] -= 2.0 * v[i] * dot

    return r


def simulate_diag_bounds(A: list, diag_bounds: list = None, safety_margin: float = 0.1,
                         eps_floor: float = 1e-15) -> list:
    """Compute per-diagonal signed bounds (lo_i, hi_i) for R[i][i] from signed QR simulation.

    Uses the same sign=sign(x0) Householder convention as the FHE solver so that
    diagonal signs match the encrypted values exactly.

    Bounds preserve the sign of the diagonal. For a negative diagonal d:
        lo = d * (1 + margin)  (more negative), hi = d * (1 - margin)  (less negative).
    For a positive diagonal d:
        lo = d * (1 - margin), hi = d * (1 + margin).
    In both cases lo < hi and the interval does not cross zero.

    Args:
        A: input matrix (m x n).
        diag_bounds: optional precomputed bounds. If provided, returned as-is.
        safety_margin: fraction to expand the interval around the diagonal (default 0.1 = ±10%).
        eps_floor: minimum absolute value to enforce to avoid near-zero domains.

    Returns:
        List of tuples [(lo_0, hi_0), ...] with lo_i < hi_i, preserving sign.
    """
    if diag_bounds is not None:
        return diag_bounds

    R = _simulate_qr_signed(A)
    n = len(R[0])
    bounds = []

    for i in range(n):
        d = R[i][i]
        ad = abs(d)
        if ad < eps_floor:
            bounds.append((eps_floor, eps_floor * (1.0 + safety_margin)))
            continue

        if d > 0:
            lo = max(ad * (1.0 - safety_margin), eps_floor)
            hi = ad * (1.0 + safety_margin)
        else:
            lo = -ad * (1.0 + safety_margin)
            hi = -max(ad * (1.0 - safety_margin), eps_floor)
        bounds.append((lo, hi))

    return bounds


def random_matrix(m: int, n: int, seed: int = 42) -> list:
    """Random m x n matrix with entries in [0.01, 10]."""
    rng = random.Random(seed)
    return [[rng.uniform(0.01, 10.0) for _ in range(n)] for _ in range(m)]


def he_matmul_T_vec(cc, Q_cols: list, rhs: list, m: int, n: int, slots: int | None = None):
    """Compute c = Q^T @ rhs homomorphically. Q_cols encrypted, rhs plaintext.

    Returns a single ciphertext with c[j] in slot j (j = 0..n-1).
    Depth cost: +2 levels beyond Q_cols depth.
    """
    slots = _resolve_slots(cc, slots)
    rhs_ptxt = make_packed_plaintext(cc, rhs, slots)
    e0_ptxt = make_packed_plaintext(cc, [1.0], slots)

    c_ct = None
    for j in range(n):
        prod = cc.EvalMult(Q_cols[j], rhs_ptxt)
        dot = sum_slots(cc, prod, m)
        dot = cc.EvalMult(dot, e0_ptxt)
        if j != 0:
            dot = safe_rotate(cc, dot, -j)
        c_ct = dot if c_ct is None else cc.EvalAdd(c_ct, dot)

    return c_ct


def he_back_substitute(cc, keys, R_cts: list, c_ct, n: int,
                       diag_bounds: list, D_inv: int = 64,
                       slots: int | None = None, bootstrap: bool = False,
                       bootstrap_min_level: int = 0):
    """Solve upper-triangular Rx = c homomorphically with encrypted pivot inversion.

    R_cts:      row-packed encrypted R (m ciphertexts).
    c_ct:       encrypted c vector with c[j] in slot j.
    n:          number of unknowns (columns of R).
    diag_bounds: list of tuples [(lo_i, hi_i), ...] for encrypted diagonal bounds.
    D_inv:      Chebyshev degree for encrypted reciprocal approximation.
    bootstrap:  if True, refresh x_ct and R_cts[i] each row (security="128" path).
    bootstrap_min_level: skip bootstraps below this level (see bootstrap_all).

    Returns ciphertext with x[j] in slot j. Depth cost: ~2*n + n*2*ceil(log2(D_inv+1)) levels.
    """
    if diag_bounds is None or len(diag_bounds) < n:
        raise ValueError("diag_bounds must provide one (lo, hi) pair per unknown for encrypted pivot inversion")

    slots = _resolve_slots(cc, slots)
    e0_ptxt = make_packed_plaintext(cc, [1.0], slots)

    # Start with x = 0 (encrypted)
    x_ct = cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, [], slots))

    for i in range(n - 1, -1, -1):
        if bootstrap and R_cts[i].GetLevel() >= bootstrap_min_level:
            R_cts[i] = cc.EvalBootstrap(R_cts[i])

        # 1. Inner product: sum_{j>i} R[i][j] * x[j]
        if i < n - 1:
            # R_cts[i] has row i in slots 0..n-1; x_ct has solution so far
            products = cc.EvalMult(R_cts[i], x_ct)          # depth +1 (ct*ct)
            inner = sum_slots(cc, products, n)               # depth +0
            inner = cc.EvalMult(inner, e0_ptxt)              # depth +1 (ct*pt)
        else:
            inner = None

        # 2. Extract c[i] to slot 0
        c_i = cc.EvalMult(safe_rotate(cc, c_ct, i), e0_ptxt)

        # 3. Numerator: c[i] - inner
        if inner is not None:
            numerator = cc.EvalSub(c_i, inner)
        else:
            numerator = c_i

        # 4. Extract encrypted pivot R[i][i] and compute reciprocal
        # Extract R[i][i] to slot 0: rotate row i by i, then mask with e0
        diag_i_ct = cc.EvalMult(safe_rotate(cc, R_cts[i], i), e0_ptxt)

        # Guard: check that the diagonal bound is not too small
        lo_i, hi_i = diag_bounds[i]
        if abs(hi_i) < 1e-15 and abs(lo_i) < 1e-15:
            raise ValueError(
                f"Diagonal bound too small at position {i}: [{lo_i:.2e}, {hi_i:.2e}]. "
                "Cannot compute encrypted reciprocal. Check QR stability or increase bound margin."
            )

        # Compute encrypted reciprocal: 1/R[i][i] using Chebyshev approximation
        # Domain [lo_i, hi_i] may be negative (sign=+1 Householder produces negative diagonals)
        inv_diag_ct = he_inv(cc, diag_i_ct, lo_i, hi_i, D_inv)

        # Multiply numerator by encrypted reciprocal
        x_i = cc.EvalMult(numerator, inv_diag_ct)

        # 5. Place x[i] into slot i and accumulate
        if i != 0:
            x_i = safe_rotate(cc, x_i, -i)
        x_ct = cc.EvalAdd(x_ct, x_i)

        if bootstrap and x_ct.GetLevel() >= bootstrap_min_level:
            x_ct = cc.EvalBootstrap(x_ct)

    return x_ct


def he_primal_weights(cc, x_ct, X_train, y_train, slots: int | None = None):
    """Compute primal weight vector w = Σᵢ αᵢ·yᵢ·x_train_i inside FHE.

    x_ct:    encrypted [b, α₀, ..., α_{n-1}] with b in slot 0, αᵢ in slot i+1.
    X_train: plaintext training data (n_train, d).
    y_train: plaintext labels (n_train,), values ±1.

    Returns w_ct: encrypted d-dimensional weight vector in slots 0..d-1.
    Depth cost: +2 levels (extract αᵢ + multiply by plaintext row).
    """
    slots = _resolve_slots(cc, slots)
    n_train = len(X_train)
    d = len(X_train[0])

    e0_ptxt = make_packed_plaintext(cc, [1.0], slots)

    w_ct = None
    for i in range(n_train):
        # Extract αᵢ from slot i+1 to slot 0, then broadcast to slots 0..d-1
        alpha_i_ct = cc.EvalMult(safe_rotate(cc, x_ct, i + 1), e0_ptxt)  # depth +1
        alpha_i_bc = replicate_slot_0(cc, alpha_i_ct, d)                  # depth +0

        # Plaintext scaled training row: yᵢ · x_train_i
        sv = [float(y_train[i] * X_train[i][k]) for k in range(d)]
        sv_ptxt = make_packed_plaintext(cc, sv, slots)

        term = cc.EvalMult(alpha_i_bc, sv_ptxt)                           # depth +1
        w_ct = term if w_ct is None else cc.EvalAdd(w_ct, term)

    return w_ct
