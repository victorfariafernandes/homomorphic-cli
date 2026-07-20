# WP3 — Local-Plaintext-Training + Encrypted-Aggregation Baseline

## Context

This is the first of three work packages from the paper resubmission plan
(`relatorio-plano-ressubmissao-lssvm-ckks`), responding to reviewer point N9
("no comparison against the obvious alternative: solve locally in plaintext,
encrypt only the parameters"). It's scoped independently of WP1 (primal +
encrypted Gram aggregation) and WP2 (secure-parameter scale study), which
will get their own specs later; WP2 depends on WP1's primal system, WP3 does
not depend on either.

## Goal

Add a new federated LSSVM entry point, `federated_lssvm/baseline_run.py`,
that implements the "baseline C" protocol from the plan:

1. Each client solves its local LSSVM **in plaintext** (µs-scale — no FHE
   training at all).
2. Each client encrypts only its resulting `(w_i, b_i)` and sends the
   ciphertexts to the server.
3. The server homomorphically aggregates (`EvalAdd` + one `EvalMult` by the
   plaintext scalar `1/k` — no decryption) and returns the encrypted global
   model.

This gives an "honest-but-curious server sees only ciphertexts" baseline at a
fraction of the cost of `federated_lssvm/train.py`'s full approach (A), where
every client's LSSVM solve itself runs under FHE.

## Non-goals

- WP1's Gram-aggregation protocol (separate spec).
- WP2's experimental grid across security levels / d' (separate spec).
- The full WP3.2 three-way A/B/C comparison table — B (WP1) doesn't exist
  yet. This spec only produces A-vs-C-comparable artifacts; the three-way
  table is assembled once WP1 lands.
- Real network transport — "tráfego" is measured as static serialized
  ciphertext size, not an actual client/server round trip.

## Key architectural consequence: no bootstrap-slot client-count constraint

`train.py` requires `k≥15` for breast_cancer because each client's local FHE
solve needs an `(n_client+1)×(n_client+1)` matrix that must fit
`SPARSE_BOOTSTRAP_SLOTS` (32) at `security="128"`. In this baseline, no
client ever performs an FHE solve — the only thing that goes into an
FHE-sized slot is `w_i` (length `d'`, the feature dimension, not the sample
count) and the scalar `b_i`. So arbitrarily small `k` works for any dataset;
the constraint that drives `CLAUDE.md`'s "use k>=15 for breast_cancer"
guidance does not apply to `baseline_run.py`.

## Protocol detail: crypto context and slot packing

Evaluation stays **fully encrypted end-to-end**: the aggregated
`(w_global, b_global)` is fed directly into the chosen solver module's
existing `predict_cipher`, the same function `train.py` and `infer.py` use.
This means the crypto context must be the same bootstrap-capable
`security="128"` context those solvers already build via
`setup_crypto_context` (reused unmodified via
`federated_lssvm.solver_selection.resolve_solver_module`) — not a cheap
depth-1 context, since `predict_cipher` needs the full rotation-key /
bootstrap machinery.

Each client's plaintext `(w_i, b_i)` must be packed with
`lssvm.solvers.utils.make_packed_plaintext(cc, vals, slots=SPARSE_BOOTSTRAP_SLOTS)`
before `cc.Encrypt(keys.publicKey, ...)`. **This is load-bearing**: per the
documented OpenFHE sharp edge, if the plaintext's `slots` metadata doesn't
match the ciphertext's bootstrap `num_slots`, `EvalMult` (used inside
`fhe_aggregate` and inside `predict_cipher`) silently returns near-zero
garbage — no exception is raised. Every encryption of a client vector/scalar
in this module must go through `make_packed_plaintext`, never a bare
`cc.MakeCKKSPackedPlaintext` call.

Key management mirrors the rest of the codebase's federation simulation:
a single process holds `keys.publicKey`/`keys.secretKey` throughout (used to
"encrypt as each client" and "decrypt for reporting/recalibration"), exactly
as `train.py` and `infer.py` already do. This is a simulated federation, not
real multi-party key separation.

## Module layout

### `federated_lssvm/baseline_run.py` (new)

CLI: `python -m federated_lssvm.baseline_run [k] --dataset=iris|breast_cancer --solver=... --security=...`
— same flag conventions as `train.py`/`plain_run.py`/`infer.py`
(`federated_lssvm.solver_selection.parse_dataset_name` /
`parse_solver_name` / `parse_security_level`, defaults `qr_row` + `128`).

Per class (OvR sub-problem):

1. `class_setup` / `partition_all` / `preprocess_features` — reused
   unmodified from `train.py` / `lssvm.preprocessing` to build the same
   per-client feature partitions as the full-FHE path, so results are
   comparable apples-to-apples.
2. For each client: solve `build_lssvm_matrix` + `np.linalg.solve` (the same
   math already in `train.py`'s `plaintext_federated_reference`, extracted
   per-client here instead of averaged in plaintext), producing `(w_i, b_i)`.
3. Encrypt `(w_i, b_i)` with `make_packed_plaintext` + `cc.Encrypt`.
4. Aggregate via `federated_lssvm.train.fhe_aggregate` (imported, unchanged).
5. Decrypt the aggregate once (`solv.decrypt_vector` / manual bias decrypt)
   to compute `recalibration_threshold` (mirrors the `train.py` fix already
   applied) — this decrypt is for recalibration/reporting only; the actual
   test-set scoring still goes through `predict_cipher` on the ciphertext.
6. Score the test set via `solv.predict_cipher(cc, keys, b_global, w_global, X_te_feat, slots=...)`, subtract the threshold.
7. Print a comparison table: **Baseline (plaintext-solve + encrypted-agg)**
   vs. **Federated plaintext reference** vs. **Full-data plaintext
   reference** — same row format as `train.py`'s `_print_comparison_table`
   (regex-compatible with `config/report.py`, see below), but without the
   "single-client FHE" row (not applicable — no client ever runs FHE here).
8. Serialize per class to `models/k={k}_baseline/class_{i}/` via the
   solver's existing `save_global_checkpoint`, plus `phi_mean.npy` and
   `threshold.npy` — same artifact shape `train.py` produces, so `infer.py`
   can load it (see below).
9. Measure one representative client's serialized ciphertext size
   (`SerializeToFile` to a temp path, `os.path.getsize`) as the WP3.2
   "tráfego" proxy; include it in the printed report.

After all classes: write the accumulated console output to
`models/k={k}_baseline/logs/finalize.log`, then call
`config.report.generate_report(k, dataset, logs_dir="models/k={k}_baseline/logs", out_path="models/k={k}_baseline/report.md")`
directly (in-process — no shell wrapper needed; there's no parallel worker
orchestration to launch since client solves are trivial-cost).

### `federated_lssvm/infer.py` (small additive change)

Add an optional `model_root: str | None = None` parameter to `main()`
(default `None` → falls back to today's `f"models/k={k}"`, fully backward
compatible) and a `--model-root=` CLI flag, so it can later evaluate
`models/k={k}_baseline/` identically to full-FHE models.

### `config/report.py` (no change)

Its regex parser (`_RE_CLASS_HDR`, `_RE_METRIC_ROW`, `_RE_W_ERR`,
`_RE_MULTICLASS`) already matches `train.py`'s comparison-table format
generically enough to consume `baseline_run.py`'s output as long as the new
module's print statements follow the same shape. `parse_worker_log` is a
no-op when no `worker_*.log` files exist (this run has none), so
`generate_report` degrades gracefully to just the finalize-log content.

## Testing

- Pure-logic unit tests (no crypto context needed) in
  `federated_lssvm/test_baseline_run.py`: per-client plaintext solve
  produces the same `(w_i, b_i)` as `plaintext_federated_reference`'s
  internal per-client step for a fixed seed; CLI arg parsing reuses
  `solver_selection` (already tested there).
- Manual/integration verification: run
  `python -m federated_lssvm.baseline_run 3 --dataset=iris` and
  `... 3 --dataset=breast_cancer` end-to-end (small k, since the bootstrap-slot
  client-count constraint doesn't apply), confirm the printed table, the
  serialized `models/k=3_baseline/` artifacts, and that
  `config/report.py`'s output (`report.md`/`metrics.csv`) is well-formed.
- Confirm `infer.py --model-root=models/k=3_baseline` (once wired) loads and
  scores the baseline model correctly.

## Open items deferred to later specs

- The full A/B/C comparison table (needs WP1).
- WP2's secure-parameter grid reporting.
