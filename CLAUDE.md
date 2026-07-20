# lwe — Privacy-Preserving LSSVM via OpenFHE

## Purpose
Least-Squares SVM training + inference over CKKS-encrypted data using OpenFHE. Includes federated variant for multi-party training without sharing plaintext.

## Layout
- `lssvm/` — plaintext + ciphertext LSSVM, preprocessing, solvers
- `federated_lssvm/` — multi-party training + inference
- `config/` — run script, metrics, shared init helpers (parallel)
- `infra/ansible/` — node provisioning, OpenFHE build
- `infra/terraform/` — cloud resource provisioning
- `requirements.txt`, `pytest.ini`, `activate_env.sh` — dev tooling

## Entry points
- `bash config/run.sh [module] [args]` — top-level threaded runner
- `python -m federated_lssvm.train` — federated training
- `python -m federated_lssvm.infer` — federated inference
- `python -m lssvm.cipher` — single-client encrypted LSSVM
- `python -m lssvm.inference` — encrypted inference
- `python -m lssvm.plain` — plaintext reference

## Dataset selection
- `--dataset=iris` (default) or `--dataset=breast_cancer` on train/infer/plain_run
  (parsed by `federated_lssvm.solver_selection.parse_dataset_name`; datasets live
  in the `lssvm/preprocessors/` package, one folder each).
- 128-bit FHE packs a fixed 32-slot bootstrap width, so per-client matrices must
  be ≤ 32. Breast cancer (455 train pts) therefore needs many clients:
  **use `k>=15` (recommend `k=20`)**, e.g.
  `bash config/run.sh federated_lssvm.train 20 --dataset=breast_cancer`.
  A guard (`assert_fits_bootstrap_slots`) fails fast with the minimum k if too small.
  `predict_cipher` batches the test set into ≤32 chunks so n_test can exceed 32.

## Key modules
- `lssvm/plain.py` — reference plaintext LSSVM
- `lssvm/cipher.py` — encrypted LSSVM over CKKS
- `lssvm/preprocessing.py` — feature scaling, kernel prep
- `lssvm/qr_householder.py` — plaintext QR-Householder reference
- `lssvm/inference.py` — encrypted inference engine
- `lssvm/solvers/cg_cipher.py` — Conjugate Gradient solver, encrypted LHS/RHS
- `lssvm/solvers/qr_householder_cipher_{col,row}.py` — two QR-Householder variants trading multiplicative depth vs slot packing
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

## Deploy
```bash
cd infra/terraform && terraform apply
ansible-playbook -i ../ansible/inventory.oci.ini ../ansible/site.yml \
  --extra-vars "repo_root=$PWD/../.."
```

Local-only smoke (no remote): use `infra/ansible/inventory.local.ini`.

## Conventions
- package-per-concern, no flat scripts at root except `activate_env.sh`
- `git mv` for moves, preserve history
- no algorithm changes in structural commits
- import paths: `lssvm.*`, `federated_lssvm.*`, `config.*` (no flat `lssvm_*`/`fhe_*` prefixes)
