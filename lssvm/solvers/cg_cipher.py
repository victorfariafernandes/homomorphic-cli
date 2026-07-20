"""Conjugate Gradient solver over CKKS (OpenFHE-Python).

Iterative SPD solver. No sqrt, no triangular solve. Each iter:
matvec + 2 inner products + 2 scalar inverses (Chebyshev) + axpy.

H encrypted as row ciphertexts. Vectors packed in slots 0..n-1.
Per-iter scalar bounds derived from plaintext CG simulation.
"""

from __future__ import annotations

import gc
import json
import math
import os

from openfhe import *

from .utils import (
    encrypt_matrix_rows,
    he_inv,
    safe_rotate,
    sum_slots,
    replicate_slot_0,
    he_primal_weights,
    write_security_marker,
    read_security_marker,
)


# ── Crypto context ────────────────────────────────────────────────────────────


def _rotation_indices(matrix_size, n_test=None, feature_dim=None):
    matrix_size = max(1, int(matrix_size or 1))
    n_test = max(1, int(n_test or matrix_size))
    neg_shift_limit = max(matrix_size, n_test)

    pos = list(range(1, matrix_size))
    neg = [-i for i in range(1, neg_shift_limit)]

    pos_pow2, step = [], 1
    while step < matrix_size:
        pos_pow2.append(step)
        step *= 2
    neg_pow2, step = [], 1
    while step < matrix_size:
        neg_pow2.append(-step)
        step *= 2

    feat_pow2 = []
    if feature_dim is not None and feature_dim > matrix_size:
        step = 1
        while step < feature_dim:
            feat_pow2.append(step)
            step *= 2

    return sorted(set(pos + neg + pos_pow2 + neg_pow2 + feat_pow2))


def _reject_bootstrap_security(security):
    if security == "128":
        raise ValueError(
            "cg_cipher does not support the security='128' bootstrapping path; "
            "use --solver=qr_row (or qr_col) or --security=notset"
        )


def setup_crypto_context(
    mult_depth, N=None, matrix_size=None, n_test=None, feature_dim=None,
    security="notset",
):
    _reject_bootstrap_security(security)
    if N is None:
        total_mod_bits = 60 + mult_depth * 50
        N_min = 2 * total_mod_bits
        N = 1
        while N < N_min:
            N <<= 1
        N = max(N, 1024)

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    params.SetRingDim(N)
    params.SetBatchSize(N // 2)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    rot = _rotation_indices(matrix_size, n_test, feature_dim=feature_dim)
    cc.EvalRotateKeyGen(keys.secretKey, rot)
    return cc, keys


# ── Plaintext CG sim for Chebyshev bounds ─────────────────────────────────────


def _simulate_cg(H, rhs, n_iter, margin=2.0):
    """Run plaintext CG, record (pAp, rr_old) per iter for he_inv domains."""
    n = len(H)
    x = [0.0] * n
    r = list(map(float, rhs))
    p = list(r)
    rr = sum(v * v for v in r)
    pAp_vals, rr_vals = [], []

    for _ in range(n_iter):
        Ap = [sum(H[i][j] * p[j] for j in range(n)) for i in range(n)]
        pAp = sum(p[i] * Ap[i] for i in range(n))
        pAp_vals.append(pAp)
        rr_vals.append(rr)
        if abs(pAp) < 1e-18 or abs(rr) < 1e-18:
            # pad with last-known values
            while len(pAp_vals) < n_iter:
                pAp_vals.append(pAp_vals[-1])
                rr_vals.append(rr_vals[-1])
            break
        alpha = rr / pAp
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * ai for ri, ai in zip(r, Ap)]
        rr_new = sum(v * v for v in r)
        beta = rr_new / rr
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rr = rr_new

    def _bounds(v, m):
        if v > 0:
            return (v / m, v * m)
        return (v * m, v / m)

    return [_bounds(v, margin) for v in pAp_vals], [_bounds(v, margin) for v in rr_vals]


# ── FHE primitives ────────────────────────────────────────────────────────────


def _matvec_row(cc, H_rows, p_ct, n, e0_ptxt):
    """y_ct = H @ p, where H_rows[i] holds row i in slots 0..n-1, p_ct in slots 0..n-1.
    Result: slots 0..n-1. Depth: +2 (ct*ct + ct*pt mask).
    """
    y_ct = None
    for i in range(n):
        prod = cc.EvalMult(H_rows[i], p_ct)
        s = sum_slots(cc, prod, n)
        s = cc.EvalMult(s, e0_ptxt)
        if i != 0:
            s = safe_rotate(cc, s, -i)
        y_ct = s if y_ct is None else cc.EvalAdd(y_ct, s)
    return y_ct


def _inner_product(cc, x_ct, y_ct, n, e0_ptxt):
    """Scalar in slot 0. Depth: +2."""
    prod = cc.EvalMult(x_ct, y_ct)
    s = sum_slots(cc, prod, n)
    return cc.EvalMult(s, e0_ptxt)


def _scalar_div(cc, num_ct, den_ct, lo, hi, D_inv, e0_ptxt):
    """num/den, both scalars in slot 0. Depth: he_inv + 1."""
    inv_den = he_inv(cc, den_ct, lo, hi, D_inv)
    inv_den = cc.EvalMult(inv_den, e0_ptxt)
    return cc.EvalMult(num_ct, inv_den)


def _broadcast_axpy(cc, x_ct, alpha_ct, p_ct, n, sign=+1):
    """x ± alpha*p where alpha in slot 0. Broadcast alpha then mul. Depth: +1 (mul)."""
    alpha_bc = replicate_slot_0(cc, alpha_ct, n)
    term = cc.EvalMult(alpha_bc, p_ct)
    return cc.EvalAdd(x_ct, term) if sign > 0 else cc.EvalSub(x_ct, term)


# ── Solver ────────────────────────────────────────────────────────────────────


def jacobi_precondition(H, rhs, eps=1e-12):
    """Symmetric Jacobi: H' = D^-1/2 H D^-1/2, rhs' = D^-1/2 rhs.
    Returns (H_pre, rhs_pre, d_inv_sqrt) where x = d_inv_sqrt * y_pre.
    Diag entries below eps replaced with 1.0 (LSSVM has H[0][0]=0).
    """
    n = len(H)
    diag = [abs(H[i][i]) if abs(H[i][i]) > eps else 1.0 for i in range(n)]
    d_inv_sqrt = [1.0 / math.sqrt(d) for d in diag]
    H_pre = [
        [H[i][j] * d_inv_sqrt[i] * d_inv_sqrt[j] for j in range(n)] for i in range(n)
    ]
    rhs_pre = [rhs[i] * d_inv_sqrt[i] for i in range(n)]
    return H_pre, rhs_pre, d_inv_sqrt


def cg_solve_fhe(cc, keys, H, rhs, n_iter, D_inv=4, margin=4.0):
    """Encrypted CG. Returns x_ct with solution in slots 0..n-1."""
    n = len(H)
    slots = cc.GetRingDimension() // 2

    e0 = [0.0] * slots
    e0[0] = 1.0
    e0_ptxt = cc.MakeCKKSPackedPlaintext(e0)

    H_rows = encrypt_matrix_rows(cc, keys, H)
    rhs_padded = list(rhs) + [0.0] * (slots - len(rhs))
    rhs_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(rhs_padded))

    # x = 0; r = rhs - H@0 = rhs; p = r
    x_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.0] * slots))
    r_ct = rhs_ct
    p_ct = rhs_ct

    # initial rr — encrypted scalar
    rr_ct = _inner_product(cc, r_ct, r_ct, n, e0_ptxt)

    pAp_bounds, rr_bounds = _simulate_cg(H, rhs, n_iter, margin=margin)

    for k in range(n_iter):
        Ap_ct = _matvec_row(cc, H_rows, p_ct, n, e0_ptxt)
        pAp_ct = _inner_product(cc, p_ct, Ap_ct, n, e0_ptxt)

        lo, hi = pAp_bounds[k]
        alpha_ct = _scalar_div(cc, rr_ct, pAp_ct, lo, hi, D_inv, e0_ptxt)

        x_ct = _broadcast_axpy(cc, x_ct, alpha_ct, p_ct, n, sign=+1)
        r_ct = _broadcast_axpy(cc, r_ct, alpha_ct, Ap_ct, n, sign=-1)

        rr_new_ct = _inner_product(cc, r_ct, r_ct, n, e0_ptxt)

        if k < n_iter - 1:
            lo_r, hi_r = rr_bounds[k]
            beta_ct = _scalar_div(cc, rr_new_ct, rr_ct, lo_r, hi_r, D_inv, e0_ptxt)
            beta_bc = replicate_slot_0(cc, beta_ct, n)
            p_ct = cc.EvalAdd(r_ct, cc.EvalMult(beta_bc, p_ct))

        rr_ct = rr_new_ct
        gc.collect()

    return x_ct


# ── Required interface ────────────────────────────────────────────────────────


N_ITER_DEFAULT = int(os.environ.get("CG_N_ITER", "6"))
D_INV_DEFAULT = int(os.environ.get("CG_D_INV", "8"))
PRECOND_DEFAULT = bool(int(os.environ.get("CG_PRECOND", "1")))
MARGIN_DEFAULT = float(os.environ.get("CG_MARGIN", "4.0"))
CHECKPOINT_SCHEMA_VERSION = 1


def checkpoint_capabilities():
    return {
        "solver": "cg_cipher",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "public_key_optional": True,
        "supported_policies": {"persist_public_key": True},
    }


def _checkpoint_metadata_path(out_dir):
    return os.path.join(out_dir, "checkpoint.json")


def _write_checkpoint_metadata(
    out_dir, mode_str, checkpoint_policy, fmt, persist_public_key
):
    metadata = {
        "solver": checkpoint_capabilities()["solver"],
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mode": mode_str,
        "format": str(fmt),
        "checkpoint_policy": dict(checkpoint_policy or {}),
        "artifacts": [
            "cryptocontext.bin",
            "secret_key.bin",
            "bias.bin",
            "weights.bin",
            "mode.txt",
        ],
    }
    if persist_public_key:
        metadata["artifacts"].insert(1, "public_key.bin")

    with open(_checkpoint_metadata_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def save_global_checkpoint(
    cc,
    keys,
    b_ct,
    w_ct,
    out_dir,
    mode_str,
    checkpoint_policy,
    fmt=BINARY,
    security="notset",
):
    _reject_bootstrap_security(security)
    policy = dict(checkpoint_policy or {})
    persist_public_key = bool(policy.get("persist_public_key", False))

    os.makedirs(out_dir, exist_ok=True)
    assert SerializeToFile(f"{out_dir}/cryptocontext.bin", cc, fmt)
    assert SerializeToFile(f"{out_dir}/secret_key.bin", keys.secretKey, fmt)
    assert SerializeToFile(f"{out_dir}/bias.bin", b_ct, fmt)
    assert SerializeToFile(f"{out_dir}/weights.bin", w_ct, fmt)

    public_key_path = f"{out_dir}/public_key.bin"
    if persist_public_key and getattr(keys, "publicKey", None) is not None:
        assert SerializeToFile(public_key_path, keys.publicKey, fmt)
    elif os.path.exists(public_key_path):
        os.remove(public_key_path)

    with open(f"{out_dir}/mode.txt", "w", encoding="utf-8") as f:
        f.write(mode_str)
    write_security_marker(out_dir, security)

    _write_checkpoint_metadata(out_dir, mode_str, policy, fmt, persist_public_key)


def _load_checkpoint_metadata(out_dir):
    metadata_path = _checkpoint_metadata_path(out_dir)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def load_global_checkpoint(out_dir, d, n_test=None, checkpoint_policy=None, fmt=BINARY):
    import os as _os

    metadata = _load_checkpoint_metadata(out_dir)
    if metadata:
        solver_name = metadata.get("solver")
        if solver_name not in (None, checkpoint_capabilities()["solver"]):
            expected = checkpoint_capabilities()["solver"]
            raise ValueError(
                f"Checkpoint solver mismatch: expected {expected}, got {solver_name}"
            )
        schema_version = int(metadata.get("schema_version", CHECKPOINT_SCHEMA_VERSION))
        if schema_version > CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema version {schema_version} in {out_dir}"
            )

    cc, ok = DeserializeCryptoContext(f"{out_dir}/cryptocontext.bin", fmt)
    assert ok

    pk = None
    public_key_path = f"{out_dir}/public_key.bin"
    if _os.path.exists(public_key_path):
        pk, ok = DeserializePublicKey(public_key_path, fmt)
        assert ok

    sk, ok = DeserializePrivateKey(f"{out_dir}/secret_key.bin", fmt)
    assert ok
    b_ct, ok = DeserializeCiphertext(f"{out_dir}/bias.bin", fmt)
    assert ok
    w_ct, ok = DeserializeCiphertext(f"{out_dir}/weights.bin", fmt)
    assert ok

    with open(f"{out_dir}/mode.txt", encoding="utf-8") as f:
        mode_str = f.read().strip()
    security = read_security_marker(out_dir)

    cc.EvalMultKeyGen(sk)
    cc.EvalRotateKeyGen(sk, _rotation_indices(d, n_test))

    class _K:
        pass

    keys = _K()
    keys.publicKey = pk
    keys.secretKey = sk
    return cc, keys, b_ct, w_ct, mode_str, security


def serialize_model(
    cc, keys, b_ct, w_ct, out_dir, mode_str="primal:linear", fmt=BINARY,
    security="notset",
):
    save_global_checkpoint(
        cc,
        keys,
        b_ct,
        w_ct,
        out_dir,
        mode_str,
        checkpoint_policy={"persist_public_key": True},
        fmt=fmt,
        security=security,
    )


def load_model(out_dir, d, n_test=None, fmt=BINARY):
    return load_global_checkpoint(out_dir, d=d, n_test=n_test, fmt=fmt)


def decrypt_vector(cc, keys, ct, length: int) -> list:
    """Decrypt and extract first length values (compatible with lssvm/cipher.py)."""
    from .utils import decrypt_vector as _decrypt_vector

    return _decrypt_vector(cc, keys, ct, length)


def _cg_with_optional_precond(cc, keys, A, b, n_iter, D_inv, margin, n):
    """Solve A x = b via encrypted CG, with optional Jacobi precond.
    Returns x_ct in slots 0..n-1.
    """
    slots = cc.GetRingDimension() // 2
    if PRECOND_DEFAULT:
        A_pre, b_pre, d_inv_sqrt = jacobi_precondition(A, b)
        y_ct = cg_solve_fhe(
            cc,
            keys,
            A_pre,
            b_pre,
            n_iter=n_iter,
            D_inv=D_inv,
            margin=margin,
        )
        scale = list(d_inv_sqrt) + [0.0] * (slots - n)
        scale_ptxt = cc.MakeCKKSPackedPlaintext(scale)
        return cc.EvalMult(y_ct, scale_ptxt)
    return cg_solve_fhe(
        cc,
        keys,
        A,
        b,
        n_iter=n_iter,
        D_inv=D_inv,
        margin=margin,
    )


def he_primal_weights_from_alpha(cc, alpha_ct, X_train, y_train):
    """w = Σᵢ αᵢ yᵢ x_i with α in slots 0..n_train-1 (no leading b)."""
    slots = cc.GetRingDimension() // 2
    n_train = len(X_train)
    d = len(X_train[0])
    e0_ptxt = cc.MakeCKKSPackedPlaintext([1.0] + [0.0] * (slots - 1))
    w_ct = None
    for i in range(n_train):
        a_i = cc.EvalMult(safe_rotate(cc, alpha_ct, i), e0_ptxt)
        a_bc = replicate_slot_0(cc, a_i, d)
        sv = [float(y_train[i] * X_train[i][k]) for k in range(d)]
        sv = sv + [0.0] * (slots - d)
        sv_ptxt = cc.MakeCKKSPackedPlaintext(sv)
        term = cc.EvalMult(a_bc, sv_ptxt)
        w_ct = term if w_ct is None else cc.EvalAdd(w_ct, term)
    return w_ct


def solver(cc, keys, H, rhs, X_train, y_train, D_sqrt=64, D_inv=64, D_inv_backsub=64,
           security="notset"):
    """Solve LSSVM saddle-point system via Schur reduction + CG.

    LSSVM H = [[0, 1ᵀ], [1, K_aug]] is symmetric INDEFINITE. CG fails directly.
    Schur trick: K_aug is SPD → solve twice with CG:
        u = K_aug^-1 y_alpha
        v = K_aug^-1 ones
        b = (1ᵀu) / (1ᵀv)
        α = u - b·v
    Falls back to direct CG when H[0][0] != 0.
    Returns (b_ct, w_ct, n).
    """
    _reject_bootstrap_security(security)
    slots = cc.GetRingDimension() // 2
    n = len(H)
    e0_ptxt = cc.MakeCKKSPackedPlaintext([1.0] + [0.0] * (slots - 1))

    if abs(H[0][0]) < 1e-12:
        # Schur reduction
        n_a = n - 1
        K_aug = [row[1:] for row in H[1:]]
        y_alpha = list(rhs[1:])
        ones = [1.0] * n_a

        u_ct = _cg_with_optional_precond(
            cc,
            keys,
            K_aug,
            y_alpha,
            N_ITER_DEFAULT,
            D_INV_DEFAULT,
            MARGIN_DEFAULT,
            n_a,
        )
        v_ct = _cg_with_optional_precond(
            cc,
            keys,
            K_aug,
            ones,
            N_ITER_DEFAULT,
            D_INV_DEFAULT,
            MARGIN_DEFAULT,
            n_a,
        )

        # Plaintext sums for he_inv bounds
        import numpy as _np

        K_np = _np.array(K_aug)
        v_pt = _np.linalg.solve(K_np, _np.array(ones))
        sum_v_val = float(v_pt.sum())
        m = MARGIN_DEFAULT
        if sum_v_val > 0:
            lo, hi = sum_v_val / m, sum_v_val * m
        else:
            lo, hi = sum_v_val * m, sum_v_val / m

        # 1ᵀu, 1ᵀv → scalars in slot 0
        sum_u_ct = cc.EvalMult(sum_slots(cc, u_ct, n_a), e0_ptxt)
        sum_v_ct = cc.EvalMult(sum_slots(cc, v_ct, n_a), e0_ptxt)

        # b = sum_u / sum_v
        inv_sum_v = he_inv(cc, sum_v_ct, lo, hi, D_INV_DEFAULT)
        inv_sum_v = cc.EvalMult(inv_sum_v, e0_ptxt)
        b_ct = cc.EvalMult(sum_u_ct, inv_sum_v)

        # α = u - b·v  (broadcast b across n_a slots)
        b_bc = replicate_slot_0(cc, b_ct, n_a)
        alpha_ct = cc.EvalSub(u_ct, cc.EvalMult(b_bc, v_ct))

        w_ct = he_primal_weights_from_alpha(cc, alpha_ct, X_train, y_train)
        return b_ct, w_ct, n

    # Fallback: direct CG (assumes H is SPD)
    x_ct = _cg_with_optional_precond(
        cc,
        keys,
        H,
        rhs,
        N_ITER_DEFAULT,
        D_INV_DEFAULT,
        MARGIN_DEFAULT,
        n,
    )
    b_ct = cc.EvalMult(x_ct, e0_ptxt)
    w_ct = he_primal_weights(cc, x_ct, X_train, y_train)
    return b_ct, w_ct, n


def get_slot_count(cc):
    return cc.GetRingDimension() // 2


def predict_cipher(cc, keys, b_ct, w_ct, X_test, slots=None):
    """Score test samples; returns a plaintext list of decision scores (length
    n_test). Delegates to utils.score_test_batched for consistency with the QR
    solvers (batched, and slots-metadata-safe for security="128")."""
    from .utils import score_test_batched

    slots = slots if slots is not None else cc.GetRingDimension() // 2
    _, d = X_test.shape
    return score_test_batched(cc, keys, b_ct, w_ct, X_test, d, slots)
