# lwe

## Purpose
Least-Squares SVM training + inference over CKKS-encrypted data using OpenFHE. Includes federated variant for multi-party training without sharing plaintext.

## Layout
- `lssvm/` — plaintext + ciphertext LSSVM, preprocessing, solvers
- `federated_lssvm/` — multi-party training + inference
- `config/` — run script, metrics, shared init helpers (parallel)
- `requirements.txt`, `pytest.ini`, `activate_env.sh` — dev tooling

## Entry points
- `bash config/run.sh [module] [args]` — top-level threaded runner
- `python -m federated_lssvm.train` — federated training
- `python -m federated_lssvm.infer` — federated inference
- `python -m lssvm.cipher` — single-client encrypted LSSVM
- `python -m lssvm.inference` — encrypted inference
- `python -m lssvm.plain` — plaintext reference

## Key modules
- `lssvm/plain.py` — reference plaintext LSSVM
- `lssvm/cipher.py` — encrypted LSSVM over CKKS
- `lssvm/preprocessing.py` — feature scaling, kernel prep
- `lssvm/qr_householder.py` — plaintext QR-Householder reference
- `lssvm/inference.py` — encrypted inference engine
- `lssvm/solvers/cg_cipher.py` — Conjugate Gradient solver, encrypted LHS/RHS
- `lssvm/solvers/qr_householder_cipher_{col,row}.py` — Householder QR variants trading multiplicative depth vs slot packing
- Supported federated checkpoint solvers: `cg`, `qr_row`, `qr_col`
- `lssvm/solvers/utils.py` — rotation/masking helpers shared across solvers
- `federated_lssvm/train.py` — multi-party training driver (FedAvg over CKKS)
- `federated_lssvm/infer.py` — federated inference
- `config/parallel.py` — OpenMP/thread bootstrap
- `config/metrics.py` — accuracy + timing collection

## Dev setup
```bash
source activate_env.sh
pip install -r requirements.txt
pytest
```
Tested on Python 3.11. `federated_lssvm/` tests that exercise the encrypted path
require the OpenFHE C++/Python build (see below) and are skipped automatically
when `openfhe` isn't importable — `pytest lssvm` alone runs on plain
`numpy`/`scipy`/`scikit-learn`, no OpenFHE build needed.

## Building OpenFHE locally
The encrypted path needs OpenFHE's C++ library and the `openfhe-python`
bindings built from source (there is no working `openfhe` package on PyPI).

```bash
sudo apt install build-essential cmake git libssl-dev libomp-dev autoconf python3-dev python3-venv

# 1. OpenFHE C++ core
git clone --depth 1 --branch v1.5.1 https://github.com/openfheorg/openfhe-development.git /tmp/openfhe
cmake -S /tmp/openfhe -B /tmp/openfhe/build -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_NATIVEOPT=ON -DWITH_OPENMP=ON
cmake --build /tmp/openfhe/build -j"$(nproc)"
sudo cmake --install /tmp/openfhe/build
sudo ldconfig

# 2. Python bindings (built into the venv from activate_env.sh)
pip install pybind11
git clone --depth 1 https://github.com/openfheorg/openfhe-python.git /tmp/openfhe-python
cmake -S /tmp/openfhe-python -B /tmp/openfhe-python/build \
  -DCMAKE_PREFIX_PATH="/usr/local;$(python -m pybind11 --cmakedir)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(which python)"
cmake --build /tmp/openfhe-python/build -j"$(nproc)"
cp /tmp/openfhe-python/build/openfhe*.so "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
python -m config.omp_smoke   # PASS = OpenFHE is running parallel, FAIL = serial build
```
`WITH_OPENMP=ON` is required — without it OpenFHE runs single-threaded and every
FHE worker rides one core. `LD_LIBRARY_PATH` must be set in every shell that
imports `openfhe`.

## Reproducing results
Quick sanity check, no OpenFHE build needed (seconds):
```bash
pytest lssvm
```

Full experiment suite (needs the OpenFHE build above):
```bash
bash config/run_campaign.sh
```
Sweeps iris + breast_cancer across IID and Dirichlet non-IID partitions
(`alpha` in `{0.5, 0.05}`) at 128-bit security. Per-run metrics land in
`campaign_results/<config>/report.md`, aggregated into
`campaign_results/all_metrics.csv` and `campaign_results/SUMMARY.md`. Sizes
itself to the machine's core/RAM budget; expect on the order of hours on a
modest multi-core machine. Worker logs live at `models/k=<K>/logs/worker_*.log`
while a config runs, then move to `campaign_results/<config>/logs/` when done.

Fast pipeline smoke test, minutes, insecure `notset` crypto params — validates
the pipeline shape, not the reported accuracy/security numbers:
```bash
SECURITY=notset ALPHAS="0.5" bash config/run_campaign.sh
```

All train/test splits use a fixed seed (`random_state=42`), so plaintext
metrics reproduce exactly; encrypted-path numbers reproduce to CKKS's
approximate-arithmetic tolerance.

## Conventions
- package-per-concern, no flat scripts at root except `activate_env.sh`
- `git mv` for moves, preserve history
- no algorithm changes in structural commits
- import paths: `lssvm.*`, `federated_lssvm.*`, `config.*` (no flat `lssvm_*`/`fhe_*` prefixes)
- Federated global checkpoints persist `secret_key.bin`; `public_key.bin` is optional.

## License / citation
MIT — see `LICENSE`. If you use this code, please cite it via `CITATION.cff`.