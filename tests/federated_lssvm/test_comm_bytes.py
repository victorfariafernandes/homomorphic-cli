"""Mechanism test for the communication-cost walker (plaintext, fake payloads).

Validates measure_comm_bytes' accounting on a controlled directory tree; the
ciphertext test (test_comm_bytes_cipher.py) checks it against a real serialized
CKKS payload.
"""

import os

from config.report import measure_comm_bytes


def _write(path, nbytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * nbytes)


def test_measure_comm_bytes_sums_client_payloads_and_excludes_baseline(tmp_path):
    root = tmp_path / "k=4"
    # two federated clients, each uploading weights (400 B) + bias (100 B)
    for i in range(2):
        d = root / "class_0" / f"client_{i}"
        _write(str(d / "weights.bin"), 400)
        _write(str(d / "bias.bin"), 100)
    # the single-client baseline is NOT federated traffic and must be excluded
    bd = root / "class_0" / "baseline"
    _write(str(bd / "weights.bin"), 400)
    _write(str(bd / "bias.bin"), 100)

    r = measure_comm_bytes(str(root))

    assert r["n_uploads"] == 2
    assert r["total_bytes"] == 1000
    assert r["per_client_bytes"] == 500
    assert r["rounds"] == 1


def test_measure_comm_bytes_empty_root(tmp_path):
    r = measure_comm_bytes(str(tmp_path))
    assert r["n_uploads"] == 0
    assert r["total_bytes"] == 0
    assert r["per_client_bytes"] == 0
    assert r["rounds"] == 1
