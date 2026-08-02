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
import resource
import time
import traceback

from config.parallel import init_threads

from federated_lssvm.solver_selection import (
    DEFAULT_SOLVER_NAME,
    parse_dataset_name,
    parse_solver_name,
    parse_security_level,
    parse_partition_name,
    parse_alpha,
    parse_models_root,
    resolve_solver_module,
)
import federated_lssvm.train as T
from lssvm.preprocessing import preprocess_features
from lssvm.preprocessors import prepare_binary


def _parse_args(args: list[str]):
    k = 3
    shard = None
    prepare_context = False
    n_per_class = None
    workers = None
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
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
            if workers < 1:
                raise SystemExit(f"--workers must be >= 1: got {workers}")
        elif a.startswith("--n-per-class="):
            n_per_class = int(a.split("=", 1)[1])
    modes = sum(x is not None for x in (shard, workers)) + int(prepare_context)
    if modes != 1:
        raise SystemExit(
            "worker requires exactly one of --shard=I/W, --workers=W, --prepare-context"
        )
    return k, shard, prepare_context, n_per_class, workers


def task_list(n_classes: int, k: int) -> list[tuple[int, int | str]]:
    """Deterministic task list: every (class, client) pair plus one baseline per class."""
    tasks: list[tuple[int, int | str]] = [
        (c, i) for c in range(n_classes) for i in range(k)
    ]
    tasks += [(c, "baseline") for c in range(n_classes)]
    return tasks


def _load_context_or_die(
    context_dir, depth, security, k, max_client_n, n_test, max_feat_dim, tag,
    models_root="models",
):
    """Verify the shared context exists with a matching security marker, then load it.

    Shared by the single-process `--shard` path and the `--workers` fork pool (which
    loads once in the parent before forking).
    """
    if not T._class_context_exists(context_dir, depth):
        raise SystemExit(
            f"[{tag}] No shared context at {context_dir} — run --prepare-context first"
        )
    marker = T.read_security_marker(context_dir)
    if marker != security:
        raise SystemExit(
            f"[{tag}] Context at {context_dir} has security={marker!r} but this run "
            f"requested security={security!r} — re-run --prepare-context (or move "
            f"{models_root}/k={k} aside) before sharding"
        )
    print(f"[{tag}] Loading shared context from {context_dir} ...")
    t0 = time.perf_counter()
    cc, keys = T._load_class_context(context_dir, max_client_n, n_test, max_feat_dim)
    print(f"[{tag}] Context loaded in {time.perf_counter() - t0:.1f}s")
    return cc, keys


def _peak_rss_mib() -> float:
    """Peak resident set size of this process in MiB.

    ru_maxrss units differ by platform: bytes on macOS, KiB on Linux (the cloud
    target). Normalize both to MiB so the memory-scaling numbers are comparable.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def run_shard(
    cc, keys, splits, all_partitions, k, shard_idx, num_shards, security,
    models_root="models",
) -> None:
    """Solve this shard's slice of the deterministic task list into the shared
    checkpoint layout. Idempotent: tasks with a finite existing checkpoint are skipped.

    Pure worker body shared by the single-process `--shard` path and the `--workers`
    fork pool — no algorithm here, just task assignment + checkpointed solves.
    """
    tag = f"worker {shard_idx}/{num_shards}"

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
        class_dir = f"{models_root}/k={k}/class_{c}"

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
    print(f"[{tag}] PEAK RSS: {_peak_rss_mib():.1f} MiB", flush=True)


def run_fork_pool(
    cc, keys, splits, all_partitions, k, workers, security, threads, log_dir,
    models_root="models",
) -> int:
    """Fork W children that each run one shard against the SAME already-loaded context.

    The parent loads the ~big CKKS context once; os.fork() gives each child a
    copy-on-write view, so the read-only eval keys stay shared in physical RAM instead
    of being deserialized W times. Returns a process exit code (0 = all shards ok).

    Fork safety: the context load has already finished (OMP idle) before the first
    fork, and nothing runs an OMP region between forks; each child gets a fresh libgomp
    pool and sets its own thread count. Children pin to disjoint core blocks on Linux.
    """
    ncpu = os.cpu_count() or 1
    can_pin = hasattr(os, "sched_setaffinity") and workers * threads <= ncpu
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Flush before forking so children don't inherit (and later re-emit) the parent's
    # buffered stdio into their redirected logs.
    sys.stdout.flush()
    sys.stderr.flush()

    child_of: dict[int, int] = {}
    for i in range(workers):
        pid = os.fork()
        if pid == 0:  # ── child ──
            try:
                if can_pin:
                    os.sched_setaffinity(0, set(range(i * threads, (i + 1) * threads)))
                init_threads(threads)
                if log_dir:
                    logf = open(os.path.join(log_dir, f"worker_{i}.log"), "w", buffering=1)
                    os.dup2(logf.fileno(), sys.stdout.fileno())
                    os.dup2(logf.fileno(), sys.stderr.fileno())
                    # dup2 redirects the fd, but Python's TextIO stays block-buffered
                    # to a file — line-buffer so `tail -f worker_N.log` shows progress
                    # live (matches the separate-process monitoring workflow).
                    sys.stdout.reconfigure(line_buffering=True)
                    sys.stderr.reconfigure(line_buffering=True)
                run_shard(
                    cc, keys, splits, all_partitions, k, i, workers, security,
                    models_root=models_root,
                )
            except BaseException:
                traceback.print_exc()
                sys.stderr.flush()
                os._exit(1)
            sys.stdout.flush()
            os._exit(0)
        child_of[pid] = i

    pins = "pinned" if can_pin else "unpinned"
    print(f"[pool] {workers} workers x {threads} threads ({pins}); "
          f"logs: {log_dir or 'inherited stdout'}", flush=True)

    failed: list[int] = []
    for _ in range(workers):
        pid, status = os.waitpid(-1, 0)
        i = child_of[pid]
        ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        if not ok:
            failed.append(i)
    if failed:
        print(f"[pool] ERROR: workers failed: {sorted(failed)} — see {log_dir}/worker_N.log",
              file=sys.stderr, flush=True)
        return 1
    print(f"[pool] all {workers} workers finished", flush=True)
    return 0


def main() -> None:
    args = sys.argv[1:]
    k, shard, prepare_context, n_per_class, workers = _parse_args(args)
    solver_name = parse_solver_name(args)
    security = parse_security_level(args)
    dataset = parse_dataset_name(args)
    partition = parse_partition_name(args)
    alpha = parse_alpha(args)
    models_root = parse_models_root(args)
    T.solv = resolve_solver_module(solver_name or DEFAULT_SOLVER_NAME)

    # Must match train.py: configure the same kernel map, load the same splits, AND
    # use the same partition/alpha, or the worker's context sizing / checkpoints
    # won't line up with aggregation.
    T.configure_dataset(dataset)
    splits = prepare_binary(dataset)
    n_test = len(splits[0][1])
    all_partitions, max_client_n, max_feat_dim = T.compute_problem_dims(
        splits, k, n_per_class, partition=partition, alpha=alpha
    )
    T.assert_fits_bootstrap_slots(security, splits, max_client_n, max_feat_dim)
    depth = T.context_depth(max_client_n, security)
    context_dir = f"{models_root}/k={k}/class_0"

    if prepare_context:
        if T._class_context_exists(context_dir, depth) and (
            T.read_security_marker(context_dir) == security
        ):
            print(f"[prepare-context] Reusing existing context at {context_dir}")
            return
        T.assert_safe_to_create_context(k, models_root)
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

    if workers is not None:
        # Fork pool: load the context once (single-threaded), then fork W children
        # that share it copy-on-write. Loading single-threaded is required: fork()
        # doesn't duplicate OpenMP worker threads, so a pool created before the fork
        # leaves children with stale thread state that deadlocks on first use. Each
        # child spins up its own fresh pool at `threads` inside run_fork_pool.
        threads = init_threads()  # requested per-worker threads (from --threads)
        init_threads(1)
        cc, keys = _load_context_or_die(
            context_dir, depth, security, k, max_client_n, n_test, max_feat_dim,
            tag=f"pool {workers}w", models_root=models_root,
        )
        rc = run_fork_pool(
            cc, keys, splits, all_partitions, k, workers, security, threads,
            log_dir=f"{models_root}/k={k}/logs", models_root=models_root,
        )
        raise SystemExit(rc)

    shard_idx, num_shards = shard
    tag = f"worker {shard_idx}/{num_shards}"
    cc, keys = _load_context_or_die(
        context_dir, depth, security, k, max_client_n, n_test, max_feat_dim, tag,
        models_root=models_root,
    )
    run_shard(
        cc, keys, splits, all_partitions, k, shard_idx, num_shards, security,
        models_root=models_root,
    )


if __name__ == "__main__":
    main()
