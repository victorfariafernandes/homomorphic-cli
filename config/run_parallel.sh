#!/usr/bin/env bash
# Parallel federated FHE experiment: prepare shared context -> W parallel workers over
# the 3*k client solves + 3 baselines -> single finalize pass (aggregate + evaluate).
#
# Usage:
#   config/run_parallel.sh <k> <workers> <threads_per_worker> [extra args]
#   e.g. config/run_parallel.sh 40 16 4 --solver=qr_row --security=128
#        config/run_parallel.sh 3 3 2 --solver=qr_row --security=notset   # local test
#        config/run_parallel.sh 40 5 2 --dataset=breast_cancer --models-root=models_bc
#            # --models-root avoids colliding with an existing models/k=40/ from a
#            # different dataset/partition run at the same k (default: "models").
#
# Preemption/interruption-safe: rerunning the same command resumes from checkpoints.
set -euo pipefail

K=${1:?usage: run_parallel.sh <k> <workers> <threads_per_worker> [extra args]}
W=${2:?usage: run_parallel.sh <k> <workers> <threads_per_worker> [extra args]}
THREADS=${3:?usage: run_parallel.sh <k> <workers> <threads_per_worker> [extra args]}
shift 3
EXTRA=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
# NOT forcing OMP_WAIT_POLICY=passive here (measured 2026-07-30 on the cloud ARM
# box): passive sets libgomp's spin count to 0 (GCC docs), so every idle-to-active
# transition between the many short OMP regions in one Householder step pays a full
# park+wake round trip -- measured ~0.87 cores/worker vs ~1.35 with the policy left
# unset (libgomp default spin count 300000), a real +55%. It was originally set to
# guard the fork-pool deadlock, but that is actually prevented by loading the shared
# context single-threaded before fork() (see worker.py run_fork_pool) -- wait policy
# was never load-bearing for that fix. Left overridable for anyone who wants to test
# ACTIVE (30B spin, GCC default) or PASSIVE explicitly; measured no further gain from
# ACTIVE over unset on this workload.
if [ -n "${OMP_WAIT_POLICY:-}" ]; then
    export OMP_WAIT_POLICY
fi

# Cloud node layout (infra/ansible/site.yml): venv + native libs.
if [ -f /opt/lssvm/venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /opt/lssvm/venv/bin/activate
    export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
TOTAL=$((W * THREADS))
if [ "$TOTAL" -gt "$NCPU" ]; then
    echo "WARNING: $W workers x $THREADS threads = $TOTAL > $NCPU cores (oversubscribed)" >&2
fi

# Models root (default matches federated_lssvm.solver_selection.DEFAULT_MODELS_ROOT)
# — lets concurrent/sequential runs with the same k avoid colliding under models/k=N/.
MODELS_ROOT="models"
for a in "${EXTRA[@]}"; do
    case "$a" in --models-root=*) MODELS_ROOT="${a#--models-root=}" ;; esac
done

LOGDIR="${MODELS_ROOT}/k=${K}/logs"
mkdir -p "$LOGDIR"

# Dataset name for the run report (default matches federated_lssvm defaults).
DATASET="iris"
for a in "${EXTRA[@]}"; do
    case "$a" in --dataset=*) DATASET="${a#--dataset=}" ;; esac
done

T0=$SECONDS
# Prepare and finalize run alone on the box — give them every core, not a worker's share.
echo "[1/4] Preparing shared crypto context (k=$K) ..."
python3 -u -m federated_lssvm.worker --k="$K" --prepare-context --threads="$NCPU" \
    "${EXTRA[@]}" 2>&1 | tee "$LOGDIR/prepare.log"
T_PREPARE=$((SECONDS - T0))

echo "[2/4] Fork pool: load context once, fork $W workers x $THREADS threads ..."
# One process loads the ~big CKKS context and os.fork()s W children that share it
# copy-on-write — instead of W processes each deserializing their own ~21 GB copy.
# Core pinning + per-worker logs (worker_N.log) are handled inside the coordinator
# (federated_lssvm.worker run_fork_pool), so no per-worker OMP_PLACES export here.
FAIL=0
python3 -u -m federated_lssvm.worker --k="$K" --workers="$W" --threads="$THREADS" \
    "${EXTRA[@]}" 2>&1 | tee "$LOGDIR/pool.log" || FAIL=1
T_WORKERS=$((SECONDS - T0 - T_PREPARE))
if [ "$FAIL" -ne 0 ]; then
    echo "ERROR: fork pool failed — see $LOGDIR/worker_*.log and $LOGDIR/pool.log" >&2
    grep -l -i -E "traceback|error" "$LOGDIR"/worker_*.log >&2 || true
    exit 1
fi

echo "[3/4] Finalize: aggregate + evaluate ..."
python3 -u -m federated_lssvm.train "$K" --threads="$NCPU" \
    "${EXTRA[@]}" 2>&1 | tee "$LOGDIR/finalize.log"
T_FINALIZE=$((SECONDS - T0 - T_PREPARE - T_WORKERS))

echo "[4/4] Appending run report ..."
python3 -m config.report --k="$K" --dataset="$DATASET" --logs="$LOGDIR" \
    --out="${MODELS_ROOT}/k=${K}/report.md" --models-root="$MODELS_ROOT" \
    --prepare-s="$T_PREPARE" --workers-s="$T_WORKERS" --finalize-s="$T_FINALIZE"
