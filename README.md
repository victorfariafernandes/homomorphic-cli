# lwe

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
require the OpenFHE C++/Python build (see Deploy below) and are skipped
automatically when `openfhe` isn't importable — `pytest lssvm` alone runs on
plain `numpy`/`scipy`/`scikit-learn`, no OpenFHE build needed.

## Reproducing results
Quick sanity check, no OpenFHE build needed (seconds):
```bash
pytest lssvm
```

Full experiment suite (needs OpenFHE — build via `infra/ansible/site.yml`,
locally with `inventory.local.ini` or on a cloud node per Deploy below):
```bash
bash config/run_campaign.sh
```
Sweeps iris + breast_cancer across IID and Dirichlet non-IID partitions
(`alpha` in `{0.5, 0.05}`) at 128-bit security. Per-run metrics land in
`campaign_results/<config>/report.md`, aggregated into
`campaign_results/all_metrics.csv` and `campaign_results/SUMMARY.md`. Expect
on the order of hours on a multi-core instance (see `infra/terraform` for the
reference cloud shape).

Fast pipeline smoke test, minutes, insecure `notset` crypto params — validates
the pipeline shape, not the reported accuracy/security numbers:
```bash
SECURITY=notset ALPHAS="0.5" bash config/run_campaign.sh
```

All train/test splits use a fixed seed (`random_state=42`), so plaintext
metrics reproduce exactly; encrypted-path numbers reproduce to CKKS's
approximate-arithmetic tolerance.

## Deploy
```bash
cd infra/terraform && terraform apply
ansible-playbook -i ../ansible/inventory.oci.ini ../ansible/site.yml \
	--extra-vars "repo_root=$PWD/../.."
```

Local-only smoke (no remote): use `infra/ansible/inventory.local.ini`.

## Monitoring the campaign
On the cloud node the `lssvm-campaign` systemd unit runs `config/run_campaign.sh`
(pytest gate + the iris/breast_cancer × {iid, dirichlet} sweep). Set `IP`/`KEY` to the
instance, then:

```bash
IP=<public-ip> ; KEY=~/.ssh/<key>

# campaign progress (per-config OK/FAIL, pytest, sizing)
ssh -i $KEY ubuntu@$IP 'tail -f /opt/lssvm/app/campaign_results/campaign.log'

# ALL worker logs, live + completed, one header per worker
ssh -i $KEY ubuntu@$IP '
  find /opt/lssvm/app/models /opt/lssvm/app/campaign_results -name "worker_*.log" 2>/dev/null \
    | sort | xargs -r tail -n +1 -v'

# follow the CURRENTLY-running workers (re-run at each new config / k-dir)
ssh -i $KEY ubuntu@$IP 'tail -n +1 -f /opt/lssvm/app/models/k=*/logs/worker_*.log'
```

Worker logs live at `models/k=<K>/logs/worker_*.log` while a config runs, then move to
`campaign_results/<config>/logs/` when it finishes. Final artifacts (`all_metrics.csv`,
`SUMMARY.md`, `campaign_results.tar.gz`) land in `campaign_results/`. To confirm OpenMP is
actually engaging on the node: `python -m config.omp_smoke` (PASS = parallel, FAIL = serial).

## Conventions
- package-per-concern, no flat scripts at root except `activate_env.sh`
- `git mv` for moves, preserve history
- no algorithm changes in structural commits
- import paths: `lssvm.*`, `federated_lssvm.*`, `config.*` (no flat `lssvm_*`/`fhe_*` prefixes)
- Federated global checkpoints persist `secret_key.bin`; `public_key.bin` is optional.

## License / citation
MIT — see `LICENSE`. If you use this code, please cite it via `CITATION.cff`.