#!/usr/bin/env bash
# Parallel federated FHE experiment: prepare shared context -> W parallel workers over
# the 3*k client solves + 3 baselines -> single finalize pass (aggregate + evaluate).
#
# Usage:
#   config/run_parallel.sh <k> <workers> <threads_per_worker> [extra args]
#   e.g. config/run_parallel.sh 40 16 4 --solver=qr_row --security=128
#        config/run_parallel.sh 3 3 2 --solver=qr_row --security=notset   # local test
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

LOGDIR="models/k=${K}/logs"
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

echo "[2/4] Launching $W workers x $THREADS threads ..."
PIDS=()
for i in $(seq 0 $((W - 1))); do
    # Pin each worker to its own disjoint core block. Without this, every worker
    # independently computes the SAME OMP spread placement (config/parallel.py sets
    # OMP_PROC_BIND=spread), stacking all threads on ~4 cores. setdefault in
    # init_threads keeps the values exported here.
    if [ "$TOTAL" -le "$NCPU" ]; then
        export OMP_PROC_BIND=close OMP_PLACES="{$((i * THREADS)):$THREADS}"
    else
        export OMP_PROC_BIND=false
    fi
    python3 -u -m federated_lssvm.worker --k="$K" --shard="$i/$W" --threads="$THREADS" \
        "${EXTRA[@]}" > "$LOGDIR/worker_$i.log" 2>&1 &
    PIDS+=($!)
done
unset OMP_PROC_BIND OMP_PLACES
echo "  worker PIDs: ${PIDS[*]}  (logs: $LOGDIR/worker_N.log)"

FAIL=0
for p in "${PIDS[@]}"; do
    wait "$p" || FAIL=1
done
T_WORKERS=$((SECONDS - T0 - T_PREPARE))
if [ "$FAIL" -ne 0 ]; then
    echo "ERROR: one or more workers failed — see $LOGDIR/worker_*.log" >&2
    grep -l -i -E "traceback|error" "$LOGDIR"/worker_*.log >&2 || true
    exit 1
fi

echo "[3/4] Finalize: aggregate + evaluate ..."
python3 -u -m federated_lssvm.train "$K" --threads="$NCPU" \
    "${EXTRA[@]}" 2>&1 | tee "$LOGDIR/finalize.log"
T_FINALIZE=$((SECONDS - T0 - T_PREPARE - T_WORKERS))

echo "[4/4] Appending run report ..."
python3 -m config.report --k="$K" --dataset="$DATASET" --logs="$LOGDIR" \
    --out="models/k=${K}/report.md" \
    --prepare-s="$T_PREPARE" --workers-s="$T_WORKERS" --finalize-s="$T_FINALIZE"
