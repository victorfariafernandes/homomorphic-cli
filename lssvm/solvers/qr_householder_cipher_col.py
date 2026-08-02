"""Column-Packed Householder QR over CKKS (OpenFHE-Python).

R packed as n column ciphertexts (each packing m values), Q as m column ciphertexts.
For 150x4: only 4 R ciphertexts instead of 150 row ciphertexts.
Sign assumed +1, zero decryptions inside the Householder loop.
Chebyshev domains derived per-step from plaintext simulation.
"""

from __future__ import annotations

import gc
from typing import Tuple

from openfhe import *

from lssvm.qr_householder import matmul, fro_norm, sub
from .utils import (
    encrypt_matrix_cols,
    encrypt_identity_cols,
    safe_rotate,
    sum_slots,
    he_sqrt,
    he_inv,
    replicate_slot_0,
    simulate_norms,
    simulate_diag_bounds,
    he_matmul_T_vec,
    he_back_substitute,
    he_primal_weights,
    make_packed_plaintext,
    bootstrap_all,
    SPARSE_BOOTSTRAP_SLOTS,
    write_security_marker,
    read_security_marker,
    slots_for_security,
)

# Bootstrap CoeffToSlot/SlotToCoeff level budget. Must stay small enough for the
# sparse 32-slot packing (log2(32)=5 rotation stages): (4,4) over-factorizes the
# homomorphic FFT and OpenFHE silently returns garbage (no exception). (3,2) is
# validated at N=131072/HEStd_128_classic: max_rel_err 8.8e-3, usable=25, bootdepth=19.
_BOOTSTRAP_LEVEL_BUDGET = (3, 2)
_BOOTSTRAP_USABLE_DEPTH = 25
# EvalBootstrap silently returns the input unchanged (after full compute cost) when the
# ciphertext isn't depleted past ~bootdepth levels; skip those calls entirely.
_BOOTSTRAP_SKIP_BELOW_LEVEL = FHECKKSRNS.GetBootstrapDepth(
    _BOOTSTRAP_LEVEL_BUDGET, SecretKeyDist.UNIFORM_TERNARY
)


def _rotation_indices(
    matrix_size: int, n_test: int = None, feature_dim: int = None
) -> list:
    """Return the minimal rotation indices for matrix size, test size, and feature dimension.

    feature_dim: maximum feature dimension after any kernel feature map — ensures sum_slots
                 has rotation keys for powers-of-2 up to feature_dim.
    """
    matrix_size = 1 if matrix_size is None else max(1, int(matrix_size))
    n_test = matrix_size if n_test is None else max(1, int(n_test))
    neg_shift_limit = max(matrix_size, n_test)

    pos_shifts = list(range(1, matrix_size))
    neg_shifts = [-i for i in range(1, neg_shift_limit)]

    # Powers-of-2 for matrix operations
    pos_pow2, step = [], 1
    while step < matrix_size:
        pos_pow2.append(step)
        step *= 2

    neg_pow2, step = [], 1
    while step < matrix_size:
        neg_pow2.append(-step)
        step *= 2

    # Additional powers-of-2 for sum_slots over feature_dim (predict_primal_cipher)
    feat_pow2 = []
    if feature_dim is not None and feature_dim > matrix_size:
        step = 1
        while step < feature_dim:
            feat_pow2.append(step)
            step *= 2

    return sorted(set(pos_shifts + neg_shifts + pos_pow2 + neg_pow2 + feat_pow2))


def setup_crypto_context(
    mult_depth: int,
    N: int = None,
    matrix_size: int = None,
    n_test: int = None,
    feature_dim: int = None,
    security: str = "128",
) -> Tuple:
    """CKKS context with targeted rotation keys for the active matrix and test sizes.

    feature_dim: maximum feature dimension after any kernel feature map.  Pass the
                 largest d across all OvR classifiers so sum_slots has the needed keys.
    security:    "128" (default) -- standardized HEStd_128_classic security via periodic
                 bootstrapping (N auto-selects to 131072 at the validated depth budget).
                 "notset" -- legacy fast/insecure path (HEStd_NotSet, no bootstrapping),
                 kept only for quick local iteration. Ring dimension N is auto-scaled
                 from mult_depth in that path if not provided.
    """
    if security == "128":
        skd = SecretKeyDist.UNIFORM_TERNARY
        # Ignore the caller's mult_depth: it is the NO-bootstrap circuit's full depth
        # estimate (e.g. 96-161+ for a 7x7 solve). The bootstrapping path refreshes
        # ciphertexts every round, so only the validated per-round budget is needed --
        # using the caller's value would explode N (and key sizes) for nothing.
        usable = _BOOTSTRAP_USABLE_DEPTH
        bootdepth = FHECKKSRNS.GetBootstrapDepth(_BOOTSTRAP_LEVEL_BUDGET, skd)
        depth = usable + bootdepth

        params = CCParamsCKKSRNS()
        params.SetSecretKeyDist(skd)
        params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        params.SetScalingModSize(50)
        params.SetFirstModSize(60)
        params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
        params.SetMultiplicativeDepth(depth)

        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.KEYSWITCH)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)
        cc.Enable(PKESchemeFeature.FHE)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalBootstrapSetup(_BOOTSTRAP_LEVEL_BUDGET, [0, 0], SPARSE_BOOTSTRAP_SLOTS)
        cc.EvalBootstrapKeyGen(keys.secretKey, SPARSE_BOOTSTRAP_SLOTS)

        rot_indices = _rotation_indices(matrix_size, n_test, feature_dim=feature_dim)
        cc.EvalRotateKeyGen(keys.secretKey, rot_indices)

        return cc, keys

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

    rot_indices = _rotation_indices(matrix_size, n_test, feature_dim=feature_dim)
    cc.EvalRotateKeyGen(keys.secretKey, rot_indices)

    return cc, keys


def householder_step_fhe_col(
    cc,
    R_cts: list,
    Q_cols: list,
    k: int,
    m: int,
    n: int,
    norm_sq_lo: float,
    norm_sq_hi: float,
    vtv_lo: float,
    vtv_hi: float,
    D_sqrt: int = 16,
    D_inv: int = 16,
    slots: int = None,
    bootstrap: bool = False,
    sign: float = 1.0,
):
    """One Householder reflection at pivot k with column-packed R (zero decryptions).

    sign: reflector sign for this step, chosen in plaintext by simulate_norms as
    sign(x0) so v0 = x0 + sign*norm never cancels (see qr_householder_cipher_row).

    Updates R_cts and Q_cols in-place. If bootstrap=True, every ciphertext in R_cts and
    Q_cols is refreshed via EvalBootstrap after the step (security="128" path).
    """
    slots = slots if slots is not None else cc.GetRingDimension() // 2
    length = m - k

    mask_k_vec = [1.0 if k <= i < m else 0.0 for i in range(min(m, slots))]
    mask_k = make_packed_plaintext(cc, mask_k_vec, slots)

    ptxt_e0 = make_packed_plaintext(cc, [1.0], slots)

    x_ct = cc.EvalMult(R_cts[k], mask_k)

    x_sq = cc.EvalMult(x_ct, x_ct)
    norm_sq_raw = sum_slots(cc, x_sq, m)
    norm_sq_slot0 = cc.EvalMult(norm_sq_raw, ptxt_e0)

    norm_ct = he_sqrt(cc, norm_sq_slot0, norm_sq_lo, norm_sq_hi, D_sqrt)
    norm_ct = cc.EvalMult(norm_ct, ptxt_e0)

    x0_slot0 = cc.EvalMult(safe_rotate(cc, x_ct, k), ptxt_e0)
    # v0 = x0 + sign*norm; v.v = 2*sign*norm*v0 (positive by construction of sign)
    v0_ct = (
        cc.EvalAdd(x0_slot0, norm_ct)
        if sign >= 0
        else cc.EvalSub(x0_slot0, norm_ct)
    )

    vtv_ct = cc.EvalMult(cc.EvalMult(norm_ct, 2.0 * sign), v0_ct)

    two_over_vtv_ct = he_inv(cc, vtv_ct, vtv_lo, vtv_hi, D_inv)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, 2.0)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, ptxt_e0)

    norm_at_k = safe_rotate(cc, norm_ct, -k)
    v_ct = cc.EvalAdd(x_ct, norm_at_k)

    active_slots = max(1, min(m, slots))

    tau_bc = replicate_slot_0(cc, two_over_vtv_ct, active_slots)
    tau_v_ct = cc.EvalMult(tau_bc, v_ct)

    for j in range(n):
        dot_ct = cc.EvalMult(v_ct, R_cts[j])
        w_j_raw = sum_slots(cc, dot_ct, m)
        w_j_slot0 = cc.EvalMult(w_j_raw, ptxt_e0)
        w_j_bc = replicate_slot_0(cc, w_j_slot0, active_slots)
        R_cts[j] = cc.EvalSub(R_cts[j], cc.EvalMult(tau_v_ct, w_j_bc))

    v_bc = []
    for j in range(length):
        vj_slot0 = cc.EvalMult(safe_rotate(cc, v_ct, k + j), ptxt_e0)
        v_bc.append(replicate_slot_0(cc, vj_slot0, active_slots))

    d_ct = cc.EvalMult(v_bc[0], Q_cols[k])
    for j in range(1, length):
        d_ct = cc.EvalAdd(d_ct, cc.EvalMult(v_bc[j], Q_cols[k + j]))

    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), d_ct)
        Q_cols[k + i] = cc.EvalSub(Q_cols[k + i], update)

    del v_bc, tau_bc, tau_v_ct, v_ct, d_ct
    gc.collect()

    if bootstrap:
        R_cts[:] = bootstrap_all(cc, R_cts, min_level=_BOOTSTRAP_SKIP_BELOW_LEVEL)
        Q_cols[:] = bootstrap_all(cc, Q_cols, min_level=_BOOTSTRAP_SKIP_BELOW_LEVEL)


def _qr(
    cc,
    keys,
    A: list,
    D_sqrt: int = 64,
    D_inv: int = 64,
    diag_bounds: list = None,
    slots: int = None,
    bootstrap: bool = False,
) -> Tuple[list, list, list]:
    """Column-packed Householder QR (internal). Returns (Q_cols, R_cts, diag_bounds)."""
    m, n = len(A), len(A[0])

    R_cts = encrypt_matrix_cols(cc, keys, A, slots=slots)
    Q_cols = encrypt_identity_cols(cc, keys, m, slots=slots)

    step_norms = simulate_norms(A)
    diag_bounds = simulate_diag_bounds(A, diag_bounds=diag_bounds)
    margin = 2.0

    steps = min(m, n)
    for k in range(steps):
        ns, vt, s = step_norms[k]
        householder_step_fhe_col(
            cc,
            R_cts,
            Q_cols,
            k,
            m,
            n,
            norm_sq_lo=ns / margin,
            norm_sq_hi=ns * margin,
            vtv_lo=vt / margin,
            vtv_hi=vt * margin,
            D_sqrt=D_sqrt,
            D_inv=D_inv,
            slots=slots,
            bootstrap=bootstrap,
            sign=s,
        )

    return Q_cols, R_cts, diag_bounds


def solver(
    cc,
    keys,
    H: list,
    rhs: list,
    X_train,
    y_train,
    D_sqrt: int = 64,
    D_inv: int = 64,
    D_inv_backsub: int = 64,
    security: str = "128",
) -> Tuple:
    """Solve H @ x = rhs fully in FHE using column-packed QR, then compute primal weights.

    security: "128" bootstraps every ciphertext each step at SPARSE_BOOTSTRAP_SLOTS width
              (must match the crypto context's setup_crypto_context(security="128") call).
              "notset" preserves the original full-width, no-bootstrap behavior.

    Returns (b_ct, w_ct, n):
      b_ct — encrypted bias scalar in slot 0.
      w_ct — encrypted primal weight vector in slots 0..d-1.
      n    — solution size (n_train + 1).
    """
    m, n = len(H), len(H[0])
    bootstrap = security == "128"
    slots = SPARSE_BOOTSTRAP_SLOTS if bootstrap else cc.GetRingDimension() // 2

    Q_cols, R_cts, diag_bounds = _qr(
        cc, keys, H, D_sqrt=D_sqrt, D_inv=D_inv, slots=slots, bootstrap=bootstrap
    )
    c_ct = he_matmul_T_vec(cc, Q_cols, rhs, m, n, slots=slots)
    x_ct = he_back_substitute(
        cc, keys, R_cts, c_ct, n, diag_bounds=diag_bounds, D_inv=D_inv_backsub,
        slots=slots, bootstrap=bootstrap,
        bootstrap_min_level=_BOOTSTRAP_SKIP_BELOW_LEVEL,
    )

    e0_ptxt = make_packed_plaintext(cc, [1.0], slots)
    b_ct = cc.EvalMult(x_ct, e0_ptxt)
    w_ct = he_primal_weights(cc, x_ct, X_train, y_train, slots=slots)

    return b_ct, w_ct, n


def serialize_model(
    cc, keys, b_ct, w_ct, out_dir: str, mode_str: str = "primal:linear", fmt=BINARY,
    security: str = "128",
) -> None:
    """Serialize crypto context, public/secret keys, bias, primal weights, and mode to out_dir.

    Eval keys (mult + rotation) are not serialized — regenerate them on load with
    cc.EvalMultKeyGen(keys.secretKey) and cc.EvalRotateKeyGen(keys.secretKey, indices).
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    assert SerializeToFile(
        f"{out_dir}/cryptocontext.bin", cc, fmt
    ), "Failed to serialize crypto context"
    assert SerializeToFile(
        f"{out_dir}/public_key.bin", keys.publicKey, fmt
    ), "Failed to serialize public key"
    assert SerializeToFile(
        f"{out_dir}/secret_key.bin", keys.secretKey, fmt
    ), "Failed to serialize secret key"
    assert SerializeToFile(
        f"{out_dir}/bias.bin", b_ct, fmt
    ), "Failed to serialize bias ciphertext"
    assert SerializeToFile(
        f"{out_dir}/weights.bin", w_ct, fmt
    ), "Failed to serialize weight ciphertext"
    with open(f"{out_dir}/mode.txt", "w") as f:
        f.write(mode_str)
    write_security_marker(out_dir, security)


def load_model(out_dir: str, d: int, n_test: int = None, fmt=BINARY):
    """Load a serialized model from out_dir and regenerate eval keys in memory.

    d:       feature dimension (used to size rotation keys).
    n_test:  number of test samples (used to size negative rotation keys).

    Returns (cc, keys, b_ct, w_ct, mode_str, security).
    """
    cc, ok = DeserializeCryptoContext(f"{out_dir}/cryptocontext.bin", fmt)
    assert ok, f"Failed to deserialize crypto context from {out_dir}"

    pk, ok = DeserializePublicKey(f"{out_dir}/public_key.bin", fmt)
    assert ok, f"Failed to deserialize public key from {out_dir}"

    sk, ok = DeserializePrivateKey(f"{out_dir}/secret_key.bin", fmt)
    assert ok, f"Failed to deserialize secret key from {out_dir}"

    b_ct, ok = DeserializeCiphertext(f"{out_dir}/bias.bin", fmt)
    assert ok, f"Failed to deserialize bias from {out_dir}"

    w_ct, ok = DeserializeCiphertext(f"{out_dir}/weights.bin", fmt)
    assert ok, f"Failed to deserialize weights from {out_dir}"

    with open(f"{out_dir}/mode.txt") as f:
        mode_str = f.read().strip()
    security = read_security_marker(out_dir)

    cc.EvalMultKeyGen(sk)
    rot_indices = _rotation_indices(d, n_test)
    cc.EvalRotateKeyGen(sk, rot_indices)

    class _Keys:
        pass

    keys = _Keys()
    keys.publicKey = pk
    keys.secretKey = sk

    return cc, keys, b_ct, w_ct, mode_str, security


def transpose(a: list) -> list:
    m, n = len(a), len(a[0])
    return [[a[i][j] for i in range(m)] for j in range(n)]


def verify_fhe(A: list, Q: list, R: list, tol: float = 1e-4) -> bool:
    """Verify ||A - QR||_F / ||A||_F, ||Q^T Q - I||_F, and max |R[i,j]| for i > j against tol."""
    m, n = len(A), len(A[0])
    norm_A = fro_norm(A)

    rel_recon = fro_norm(sub(A, matmul(Q, R))) / norm_A if norm_A > 0 else 0.0
    print(f"  ||A - QR||_F / ||A||_F = {rel_recon:.2e}  (target < {tol:.0e})")

    I = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    ortho_err = fro_norm(sub(matmul(transpose(Q), Q), I))
    print(f"  ||Q^T Q - I||_F       = {ortho_err:.2e}  (target < {tol:.0e})")

    max_lower = max(
        (abs(R[i][j]) for i in range(m) for j in range(min(i, n))),
        default=0.0,
    )
    print(f"  max |R[i,j]| (i > j)  = {max_lower:.2e}  (target < {tol:.0e})")

    ok = rel_recon < tol and ortho_err < tol and max_lower < tol
    print(f"  PASS: {ok}\n")
    return ok


# ─── Compatibility exports (used by lssvm/cipher.py) ───

CHECKPOINT_SCHEMA_VERSION = 1


def checkpoint_capabilities():
    return {
        "solver": "qr_householder_cipher_col",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "public_key_optional": True,
        "supported_policies": {"persist_public_key": True},
    }


def _checkpoint_metadata_path(out_dir: str):
    import os as _os

    return _os.path.join(out_dir, "checkpoint.json")


def _write_checkpoint_metadata(
    out_dir: str,
    mode_str: str,
    checkpoint_policy,
    fmt,
    persist_public_key: bool,
) -> None:
    import json as _json

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
        _json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def save_global_checkpoint(
    cc,
    keys,
    b_ct,
    w_ct,
    out_dir: str,
    mode_str: str,
    checkpoint_policy,
    fmt=BINARY,
    security: str = "128",
) -> None:
    import os as _os

    policy = dict(checkpoint_policy or {})
    persist_public_key = bool(policy.get("persist_public_key", False))

    _os.makedirs(out_dir, exist_ok=True)
    assert SerializeToFile(
        f"{out_dir}/cryptocontext.bin", cc, fmt
    ), "Failed to serialize crypto context"
    assert SerializeToFile(
        f"{out_dir}/secret_key.bin", keys.secretKey, fmt
    ), "Failed to serialize secret key"
    assert SerializeToFile(
        f"{out_dir}/bias.bin", b_ct, fmt
    ), "Failed to serialize bias ciphertext"
    assert SerializeToFile(
        f"{out_dir}/weights.bin", w_ct, fmt
    ), "Failed to serialize weight ciphertext"

    public_key_path = f"{out_dir}/public_key.bin"
    if persist_public_key and getattr(keys, "publicKey", None) is not None:
        assert SerializeToFile(
            public_key_path, keys.publicKey, fmt
        ), "Failed to serialize public key"
    elif _os.path.exists(public_key_path):
        _os.remove(public_key_path)

    with open(f"{out_dir}/mode.txt", "w", encoding="utf-8") as f:
        f.write(mode_str)
    write_security_marker(out_dir, security)

    _write_checkpoint_metadata(out_dir, mode_str, policy, fmt, persist_public_key)


def _load_checkpoint_metadata(out_dir: str):
    import json as _json
    import os as _os

    metadata_path = _checkpoint_metadata_path(out_dir)
    if not _os.path.exists(metadata_path):
        return None
    with open(metadata_path, encoding="utf-8") as f:
        return _json.load(f)


def load_global_checkpoint(
    out_dir: str,
    d: int,
    n_test: int = None,
    checkpoint_policy=None,
    fmt=BINARY,
):
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
    assert ok, f"Failed to deserialize crypto context from {out_dir}"

    pk = None
    public_key_path = f"{out_dir}/public_key.bin"
    if _os.path.exists(public_key_path):
        pk, ok = DeserializePublicKey(public_key_path, fmt)
        assert ok, f"Failed to deserialize public key from {out_dir}"

    sk, ok = DeserializePrivateKey(f"{out_dir}/secret_key.bin", fmt)
    assert ok, f"Failed to deserialize secret key from {out_dir}"

    b_ct, ok = DeserializeCiphertext(f"{out_dir}/bias.bin", fmt)
    assert ok, f"Failed to deserialize bias from {out_dir}"

    w_ct, ok = DeserializeCiphertext(f"{out_dir}/weights.bin", fmt)
    assert ok, f"Failed to deserialize weights from {out_dir}"

    with open(f"{out_dir}/mode.txt", encoding="utf-8") as f:
        mode_str = f.read().strip()
    security = read_security_marker(out_dir)

    cc.EvalMultKeyGen(sk)
    rot_indices = _rotation_indices(d, n_test)
    cc.EvalRotateKeyGen(sk, rot_indices)

    class _Keys:
        pass

    keys = _Keys()
    keys.publicKey = pk
    keys.secretKey = sk

    return cc, keys, b_ct, w_ct, mode_str, security


def decrypt_vector(cc, keys, ct, length: int) -> list:
    """Decrypt and extract first length values (compatible with lssvm/cipher.py)."""
    from .utils import decrypt_vector as _decrypt_vector

    return _decrypt_vector(cc, keys, ct, length)


def get_slot_count(cc) -> int:
    """Get slot count (compatible with lssvm/cipher.py)."""
    return cc.GetRingDimension() // 2


def predict_cipher(cc, keys, b_ct, w_ct, X_test, slots: int = None):
    """Score test samples with encrypted primal weights; returns a plaintext list
    of decision scores (length n_test).

    Batched via utils.score_test_batched into chunks of `slots`, so n_test greater
    than the sparse bootstrap width stays correct at security="128" (an un-batched
    rotation by -j up to -(n_test-1) would wrap modulo num_slots and collide).

    slots: must be SPARSE_BOOTSTRAP_SLOTS when b_ct/w_ct came from a security="128"
           solve (bootstrapped ciphertexts). Defaults to full ring width for
           legacy/"notset" models.
    """
    from .utils import score_test_batched

    slots = slots if slots is not None else cc.GetRingDimension() // 2
    _, d = X_test.shape
    return score_test_batched(cc, keys, b_ct, w_ct, X_test, d, slots)
