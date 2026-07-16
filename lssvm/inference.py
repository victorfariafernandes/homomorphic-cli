"""FHE Inference Engine for LSSVM.

Loads serialized model (crypto context + keys + encrypted primal weights) and
classifies a single input sample across all OvR binary classifiers without
requiring any training data at inference time.

Usage:
    python fhe_inference.py                          # uses default SAMPLE
    python fhe_inference.py "5.1,3.5,1.4,0.2"       # setosa
    python fhe_inference.py "6.3,3.3,6.0,2.5"       # virginica
"""

from __future__ import annotations

import sys
import numpy as np

from lssvm.solvers import qr_householder_cipher_row as solver_mod
from lssvm.solvers.utils import decrypt_vector, sum_slots, make_packed_plaintext, SPARSE_BOOTSTRAP_SLOTS
from lssvm.preprocessing import (
    prepare_iris_binary,
    poly_feature_map,
    homogeneous_poly_feature_map,
    center_features,
    normalize_features,
)

# ── configuration ──────────────────────────────────────────────────
# Override with: python fhe_inference.py "5.1,3.5,1.4,0.2"
SAMPLE = [5.1, 3.5, 1.4, 0.2]  # [sepal_length, sepal_width, petal_length, petal_width]
N_PER_CLASS = 2  # must match the value used during training
MODELS_DIR = "models"


def _apply_feature_map(X: np.ndarray, mode_str: str) -> np.ndarray:
    """Apply the feature map encoded in mode_str to X.

    mode_str formats:
        "primal:linear"
        "primal:poly:degree=2:c=1.0"
        "primal:homo_poly:degree=2"
    """
    parts = mode_str.split(":")
    kernel_type = parts[1]

    if kernel_type == "linear":
        return X

    if kernel_type == "poly":
        params = {kv.split("=")[0]: kv.split("=")[1] for kv in parts[2:]}
        degree = int(params.get("degree", 2))
        c = float(params.get("c", 1.0))
        return poly_feature_map(X, degree=degree, c=c)

    if kernel_type == "homo_poly":
        params = {kv.split("=")[0]: kv.split("=")[1] for kv in parts[2:]}
        degree = int(params.get("degree", 2))
        return homogeneous_poly_feature_map(X, degree=degree)

    raise ValueError(f"Unknown kernel type in mode string: '{mode_str}'")


def run_inference(
    model_dir: str,
    X_sample: np.ndarray,
) -> tuple[float, bool]:
    """Load model from model_dir and classify a single sample.

    X_sample: shape (1, d) — the raw input feature vector (before any feature map).

    Returns (score, is_positive):
      score       — raw decision score (positive = belongs to this class)
      is_positive — True if the sample is predicted as this class
    """
    # Peek at mode.txt to determine feature map before loading (rotation keys need mapped d)
    with open(f"{model_dir}/mode.txt") as f:
        mode_str = f.read().strip()

    # Apply feature map first so we know the mapped dimension for rotation key generation
    X_mapped = _apply_feature_map(X_sample, mode_str)
    phi_mean = np.load(f"{model_dir}/phi_mean.npy")
    # Match preprocess_features: linear → raw; non-linear → center + unit-L2 norm.
    if mode_str.split(":")[1] != "linear":
        X_mapped, _ = center_features(X_mapped, phi_mean=phi_mean)
        X_mapped = normalize_features(X_mapped)
    d = X_mapped.shape[1]

    cc, keys, b_ct, w_ct, _, security = solver_mod.load_model(model_dir, d=d, n_test=1)

    slots = SPARSE_BOOTSTRAP_SLOTS if security == "128" else cc.GetRingDimension() // 2
    e0_ptxt = make_packed_plaintext(cc, [1.0], slots)

    xj = list(X_mapped[0])
    xj_ptxt = make_packed_plaintext(cc, xj, slots)

    dot = cc.EvalMult(w_ct, xj_ptxt)
    score_ct = sum_slots(cc, dot, d)
    score_ct = cc.EvalAdd(score_ct, b_ct)
    score_ct = cc.EvalMult(score_ct, e0_ptxt)

    score = decrypt_vector(cc, keys, score_ct, 1)[0]
    return score, score > 0


if __name__ == "__main__":
    # Parse optional CLI sample
    if len(sys.argv) > 1:
        try:
            SAMPLE = [float(x) for x in sys.argv[1].split(",")]
        except ValueError:
            print(
                f"Invalid sample format. Expected comma-separated floats, got: {sys.argv[1]}"
            )
            sys.exit(1)

    X_sample = np.array(SAMPLE, dtype=float).reshape(1, -1)
    print(f"Sample: {SAMPLE}\n")

    # Get class names from the data splits
    splits = prepare_iris_binary()
    class_names = [name for _, _, _, _, name in splits]

    scores = []
    for class_idx, name in enumerate(class_names):
        model_dir = f"{MODELS_DIR}/class_{class_idx}"
        score, is_positive = run_inference(model_dir, X_sample)
        label = f"IS {name}" if is_positive else f"IS NOT {name}"
        print(f"  [Class {class_idx} - {name:<12}]  {label:<20}  (score={score:+.4f})")
        scores.append(score)

    predicted_idx = int(np.argmax(scores))
    print(f"\n→ Predicted class: {class_names[predicted_idx]}")
