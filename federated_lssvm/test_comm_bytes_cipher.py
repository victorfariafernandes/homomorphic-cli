"""Ciphertext validation of measure_comm_bytes against a real serialized payload.

Serializes a genuine CKKS-encrypted (weights, bias) pair into the on-disk layout a
federated client produces, then checks the walker reports the real byte sizes.
Uses security="notset" so no 128-bit bootstrap is needed. (Absolute sizes here are
notset ring-dimension sizes, not the 128-bit numbers reported from cloud runs; this
test validates the measurement mechanism, not the headline figure.)
"""

import os

import pytest

openfhe = pytest.importorskip("openfhe", reason="requires the OpenFHE C++/Python build")
SerializeToFile, BINARY = openfhe.SerializeToFile, openfhe.BINARY

from config.report import measure_comm_bytes
from federated_lssvm.baseline_run import _NOTSET_DEPTH
from federated_lssvm.solver_selection import resolve_solver_module, DEFAULT_SOLVER_NAME
from lssvm.solvers.utils import make_packed_plaintext


def test_measure_comm_bytes_reads_real_ciphertext_sizes(tmp_path):
    solv = resolve_solver_module(DEFAULT_SOLVER_NAME)
    cc, keys = solv.setup_crypto_context(
        _NOTSET_DEPTH, matrix_size=1, n_test=4, feature_dim=4, N=None, security="notset",
    )
    slots = solv.get_slot_count(cc)
    w_ct = cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, [0.1, 0.2, 0.3, 0.4], slots))
    b_ct = cc.Encrypt(keys.publicKey, make_packed_plaintext(cc, [0.5], slots))

    client = tmp_path / "k=1" / "class_0" / "client_0"
    os.makedirs(client, exist_ok=True)
    SerializeToFile(str(client / "weights.bin"), w_ct, BINARY)
    SerializeToFile(str(client / "bias.bin"), b_ct, BINARY)

    r = measure_comm_bytes(str(tmp_path / "k=1"))

    assert r["n_uploads"] == 1
    assert r["rounds"] == 1
    # a real CKKS ciphertext is far from empty; total = weights + bias on disk
    expected = os.path.getsize(client / "weights.bin") + os.path.getsize(client / "bias.bin")
    assert r["total_bytes"] == expected
    assert r["per_client_bytes"] == expected
    assert r["total_bytes"] > 1000
