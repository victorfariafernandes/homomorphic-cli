# R1 Revision Campaign — Design Spec

**Date:** 2026-07-29
**Deadline:** 4 days (fixed, journal/conference revision)
**Compute:** Cloud (OCI via `infra/terraform` + `infra/ansible`)
**Strategy:** Portfolio A — Breadth rebuttal (address 3 of 4 reviewer asks empirically, defend the 4th analytically)

## Context

Reviewer R1 and the meta-review raise four points about the FHE federated LSSVM paper:

1. Evaluate on larger / higher-dimensional datasets (central criticism: "only Iris").
2. Experimental comparison with other secure-FL / iterative-homomorphic methods.
3. Scalability experiments: runtime, memory, communication cost vs. #clients and data dimension.
4. Non-IID data distributions across clients.

## Governing constraint (drives all scoping)

The validated 128-bit FHE config packs a fixed sparse-bootstrap width
`SPARSE_BOOTSTRAP_SLOTS = 32` (`lssvm/solvers/utils.py`). It must stay
`>= max(n_test, feature_dim, matrix_size)`. Consequences:

- `feature_dim <= 31`
- `points-per-client <= ~30` (matrix is `(N_local+1) x (N_local+1)`)

**We do NOT touch the slot width.** Raising it risks the documented
"(4,4)-budget × sparse-32 garbage" failure with no recovery time in a 4-day window.
breast_cancer (30 features) already sits at this ceiling — a point worth stating in
the paper as a genuine finding, not a limitation to hide.

## Scope decisions

- **breast_cancer is the workhorse** for the "larger / higher-dimensional" answer
  (item 1). It already exists in the codebase (30-dim, 455 train pts vs. Iris's
  4-dim, 150 pts — 7.5× dimensionality, 3× size). No new dataset code required.
- **digits (64-dim) + PCA→30 is deferred to a subsequent paper.** Stated as
  future work with the honest note that native feature_dim is slot-capped at 31,
  so higher-dimensional inputs require a variance-preserving projection. This
  demonstrates awareness of the scaling path and strengthens the rebuttal.
- **Baseline reproduction (item 2) is defended analytically, not reimplemented.**
  Reproducing literature HE-FL baselines in 4 days is infeasible and high-variance.

## Workstreams

### WS1 — Larger / higher-dimensional result (item 1) — reporting

- No new dataset code. Run breast_cancer at `k=20` (already supported; guard
  `assert_fits_bootstrap_slots` enforces the minimum k).
- Report as the larger + higher-dimensional evidence alongside Iris.
- Write the digits + train-only-PCA→30 future-work paragraph, including the
  slot-cap rationale and expected retained variance.

**Deliverable:** clean breast_cancer k=20 run + metrics; future-work paragraph.

### WS2 — Non-IID partitioner (item 4) — primary new code

- Add a Dirichlet(α) label-skew partitioner alongside the current IID disjoint
  split in `federated_lssvm/train.py`, selected via new flags
  `--partition=iid|dirichlet` and `--alpha=<float>`.
- Partitioner assigns each client a class mixture drawn from Dirichlet(α); IID is
  the α→∞ control (current behavior, relabeled).
- Sweep **α ∈ {0.1 (severe), 0.5 (mild)}** vs. IID on **breast_cancer** (binary:
  per-client malignant/benign imbalance) **and iris** (3-class: richer skew curve).
- **Edge case:** under severe skew a client may hold zero positive examples for an
  OvR sub-problem (degenerate all-reject local model). Design decision: such a
  client still contributes its all-negative weight vector to FedAvg (this *is* the
  non-IID pathology under study); the occurrence is logged and counted. No crash,
  no silent skip.
- The partitioner is a self-contained unit: input `(y_train_raw, k, alpha, rng)`,
  output a list of `k` disjoint index arrays. Testable in plaintext without FHE.

**Deliverable:** partitioner + flags + unit tests; accuracy-vs-α runs on both datasets.

### WS3 — Scalability instrumentation (item 3) — harvest

- **Runtime:** already logged via `config/report.py` / `metrics.csv` — harvest.
- **Memory:** capture peak RSS per worker via `resource.getrusage(RUSAGE_SELF)`
  at worker exit, emit to the worker log; add a parse rule + CSV field in
  `config/report.py`.
- **Communication cost:** measure serialized encrypted-weight ciphertext bytes
  from `models/k=K/` on disk; total upload = `k × per-client bytes`. Headline
  framing: **one-shot FedAvg = 1 communication round vs. iterative HE methods'
  T rounds.**
- **Scaling axes:** vs. **k** (client count) and vs. **feature dim**
  (iris=4, breast_cancer=30), harvested from runs already being executed for
  WS1/WS2. No dedicated campaign.

**Deliverable:** RSS + comm-byte instrumentation + parser; scaling curves
(vs. k, vs. dim) and a communication-cost table.

### WS4 — Analytical baseline comparison (item 2) — defend

- Comparison table: rows = {this work, ~3 cited HE-FL / iterative-HE methods};
  columns = {HE scheme, security model, communication rounds (1 vs. T),
  communication complexity, multiplicative depth, non-IID handling,
  accuracy-gap-vs-plaintext}.
- Our row populated from WS3 measurements; other rows from cited literature.
- Framing: the one-shot, non-iterative design is the differentiator.

**Deliverable:** comparison table + supporting prose for the rebuttal.

## Deliverables summary

1. **Code:** Dirichlet partitioner + `--partition`/`--alpha` flags; peak-RSS +
   comm-byte instrumentation; unit tests for partitioner and instrumentation parse.
2. **Runs:** breast_cancer k=20 (IID + α∈{0.1,0.5}); iris α-sweep; iris k-sweep for
   scaling; all metrics harvested into `metrics.csv` / `report.md`.
3. **Analysis artifacts:** scaling plots (vs. k, vs. dim), accuracy-vs-α plot,
   communication-cost table.
4. **Rebuttal text:** analytical baseline table (WS4), point-by-point R1 response,
   digits + PCA future-work note (WS1).

## Testing

- Partitioner: unit tests for (a) disjointness + full coverage of indices,
  (b) IID reproduces balanced class proportions, (c) low-α produces measurable
  skew, (d) the zero-positive-client edge case is detected and counted.
- Instrumentation: parser tests that a synthetic worker log with an RSS line and a
  comm-byte value round-trips into the expected CSV fields.
- No changes to the validated FHE solver path — existing solver tests must stay green.

## Schedule / risk

- Risk is **low**: the only new algorithmic code is the plaintext partitioner;
  everything else is instrumentation + runs on an already-validated config.
- **End-of-day-1 sanity gate:** partitioner unit tests green AND one non-IID
  breast_cancer run completes end-to-end. If it slips, drop the iris α-sweep first,
  then reduce α-sweep to a single value.

## Explicitly out of scope

- Raising `SPARSE_BOOTSTRAP_SLOTS` or any change to the validated 128-bit config.
- New dataset ingestion beyond the existing iris / breast_cancer.
- Reimplementing or reproducing any literature baseline (item 2 is analytical only).
- The digits + PCA experiment (deferred to a subsequent paper).
