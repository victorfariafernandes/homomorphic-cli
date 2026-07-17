"""Parallel FHE worker: compute a shard of the per-client (and baseline) checkpoints.

The 40x3 client solves are embarrassingly parallel until FedAvg aggregation, and
train.py already resumes from per-client checkpoints — so parallelism is just: several
worker processes, each solving a disjoint shard of the task list into the shared
checkpoint layout, followed by a normal `python -m federated_lssvm.train k ...` pass
that finds every checkpoint present and only aggregates/evaluates.

Usage:
  python -m federated_lssvm.worker --k=40 --prepare-context [--security=128] [--threads=4]
      Create the shared crypto context + keys once, serialized to models/k=40/class_0/.

  python -m federated_lssvm.worker --k=40 --shard=3/16 [--security=128] [--threads=4]
      Load the shared context and solve tasks[3::16] of the deterministic task list
      [(class 0..2, client 0..k-1)] + [(class 0..2, baseline)]. Idempotent: tasks whose
      checkpoint already exists (and decrypts finite) are skipped.

Orchestrated by config/run_parallel.sh.
"""

from __future__ import annotations

import sys
from config.parallel import bootstrap as _init_parallel

_init_parallel()

import os
import time

from federated_lssvm.solver_selection import (
    DEFAULT_SOLVER_NAME,
    parse_solver_name,
    parse_security_level,
    resolve_solver_module,
)
import federated_lssvm.train as T
from lssvm.preprocessing import prepare_iris_binary, preprocess_features


def _parse_args(args: list[str]):
    k = 3
    shard = None
    prepare_context = False
    n_per_class = None
    for a in args:
        if a.startswith("--k="):
            k = int(a.split("=", 1)[1])
        elif a.startswith("--shard="):
            idx, total = a.split("=", 1)[1].split("/")
            shard = (int(idx), int(total))
            if not (0 <= shard[0] < shard[1]):
                raise SystemExit(f"--shard index must be in [0, {shard[1]}): got {shard[0]}")
        elif a == "--prepare-context":
            prepare_context = True
        elif a.startswith("--n-per-class="):
            n_per_class = int(a.split("=", 1)[1])
    if not prepare_context and shard is None:
        raise SystemExit("worker requires --shard=I/W or --prepare-context")
    return k, shard, prepare_context, n_per_class


def task_list(n_classes: int, k: int) -> list[tuple[int, int | str]]:
    """Deterministic task list: every (class, client) pair plus one baseline per class."""
    tasks: list[tuple[int, int | str]] = [
        (c, i) for c in range(n_classes) for i in range(k)
    ]
    tasks += [(c, "baseline") for c in range(n_classes)]
    return tasks


def main() -> None:
    args = sys.argv[1:]
    k, shard, prepare_context, n_per_class = _parse_args(args)
    solver_name = parse_solver_name(args)
    security = parse_security_level(args)
    T.solv = resolve_solver_module(solver_name or DEFAULT_SOLVER_NAME)

    splits = prepare_iris_binary()
    n_test = len(splits[0][1])
    all_partitions, max_client_n, max_feat_dim = T.compute_problem_dims(
        splits, k, n_per_class
    )
    depth = T.context_depth(max_client_n, security)
    context_dir = f"models/k={k}/class_0"

    if prepare_context:
        if T._class_context_exists(context_dir, depth) and (
            T.read_security_marker(context_dir) == security
        ):
            print(f"[prepare-context] Reusing existing context at {context_dir}")
            return
        T.assert_safe_to_create_context(k)
        print(f"[prepare-context] Creating shared context (depth={depth}, security={security}) ...")
        t0 = time.perf_counter()
        cc, keys = T.solv.setup_crypto_context(
            depth,
            matrix_size=max_client_n,
            n_test=n_test,
            feature_dim=max_feat_dim,
            N=T.N_OVERRIDE,
            security=security,
        )
        T._save_class_context(context_dir, cc, keys, depth, security=security)
        print(
            f"[prepare-context] Context ready and serialized to {context_dir} "
            f"in {time.perf_counter() - t0:.1f}s  (N={cc.GetRingDimension()})"
        )
        return

    shard_idx, num_shards = shard
    tag = f"worker {shard_idx}/{num_shards}"

    if not T._class_context_exists(context_dir, depth):
        raise SystemExit(
            f"[{tag}] No shared context at {context_dir} — run --prepare-context first"
        )
    marker = T.read_security_marker(context_dir)
    if marker != security:
        raise SystemExit(
            f"[{tag}] Context at {context_dir} has security={marker!r} but this run "
            f"requested security={security!r} — re-run --prepare-context (or move "
            f"models/k={k} aside) before sharding"
        )

    print(f"[{tag}] Loading shared context from {context_dir} ...")
    t0 = time.perf_counter()
    cc, keys = T._load_class_context(context_dir, max_client_n, n_test, max_feat_dim)
    print(f"[{tag}] Context loaded in {time.perf_counter() - t0:.1f}s")

    # Deterministic task list, interleaved assignment for class balance across workers.
    tasks = task_list(len(splits), k)
    mine = tasks[shard_idx::num_shards]
    print(f"[{tag}] {len(mine)}/{len(tasks)} tasks: {mine}")

    # Per-class setup is deterministic (fixed seeds/GCV) — cache per class.
    setups = {}
    t_start = time.perf_counter()
    done = skipped = 0
    for c, i in mine:
        if c not in setups:
            setups[c] = T.class_setup(c, splits[c][0], splits[c][2], verbose=False)
        _, feature_map, _, gamma, _, phi_mean = setups[c]
        class_dir = f"models/k={k}/class_{c}"

        if i == "baseline":
            X_s_feat, y_s, _ = T.baseline_features(
                splits[c][0], splits[c][2], splits[c][1], feature_map
            )
            ckpt_dir = f"{class_dir}/baseline"
            already = T._cts_exist(ckpt_dir)
            T.solve_client_checkpointed(
                cc, keys, ckpt_dir, X_s_feat, y_s, gamma, security,
                label=f"{tag} class {c} baseline",
            )
        else:
            X_c, y_c = all_partitions[c][i]
            X_c_feat, _ = preprocess_features(X_c, feature_map, phi_mean=phi_mean)
            ckpt_dir = f"{class_dir}/client_{i}"
            already = T._cts_exist(ckpt_dir)
            T.solve_client_checkpointed(
                cc, keys, ckpt_dir, X_c_feat, y_c, gamma, security,
                label=f"{tag} class {c} client {i}",
            )
        if already:
            skipped += 1
        else:
            done += 1
        elapsed = time.perf_counter() - t_start
        print(
            f"[{tag}] progress: {done + skipped}/{len(mine)} "
            f"(solved={done}, resumed={skipped})  elapsed={elapsed / 60:.1f}min",
            flush=True,
        )

    print(f"[{tag}] DONE: {done} solved, {skipped} resumed, "
          f"{(time.perf_counter() - t_start) / 60:.1f}min total")


if __name__ == "__main__":
    main()
