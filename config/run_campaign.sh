#!/usr/bin/env bash
# Full metrics campaign: pytest gate + the FHE run matrix over both datasets and
# all partitions, collecting every run's metrics into files on this machine.
#
# Runs are STRICTLY SEQUENTIAL (one run_parallel.sh at a time), but each run is
# given the whole box: the fork pool is sized so W*THREADS ~= nproc. The COW
# context-sharing optimization is preserved unchanged (run_parallel.sh loads the
# CKKS context once and forks W workers that share it copy-on-write).
#
# Everything is tunable via environment variables (see defaults below), e.g.
#   SECURITY=notset THREADS=2 ALPHAS="0.5" bash config/run_campaign.sh   # fast smoke
#
# Resume-safe at config granularity: a config whose results dir already has a
# report.md is skipped, so a re-run continues where it left off.
set -uo pipefail

# ── Tunables ────────────────────────────────────────────────────────────────
SECURITY="${SECURITY:-128}"          # 128 (real) | notset (fast smoke)
NPC="${NPC:-15}"                      # --n-per-class cap: matrix <= 2*NPC+1 = 31 (<=32 slots)
DATASETS="${DATASETS:-iris breast_cancer}"
K_IRIS="${K_IRIS:-5}"
K_BC="${K_BC:-20}"
# Peak RAM per 128-bit worker at this crypto footprint (N=131072, depth~44) was
# measured at ~33 GB (an OOM). Budget 40 GB/worker for headroom; workers are capped
# so W*PER_WORKER_GB fits RAM. This auto-scales: resize the instance to more RAM and
# more workers run. Override W/THREADS explicitly to bypass.
PER_WORKER_GB="${PER_WORKER_GB:-40}"
# shellcheck disable=SC2206
ALPHAS=(${ALPHAS:-0.5 0.05})         # Dirichlet concentrations for the non-IID sweep
RESULTS_DIR_NAME="${RESULTS_DIR:-campaign_results}"
RUN_PARALLEL="${RUN_PARALLEL:-config/run_parallel.sh}"   # overridable for testing

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR_NAME"
mkdir -p "$RESULTS_DIR"

NCPU="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
MEM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"; [ -z "$MEM_GB" ] && MEM_GB=8
# Workers are bounded by BOTH memory and cores; give each worker the leftover cores.
W_MEM=$(( MEM_GB / PER_WORKER_GB )); (( W_MEM < 1 )) && W_MEM=1
W="${W:-$(( W_MEM < NCPU ? W_MEM : NCPU ))}"
THREADS="${THREADS:-$(( NCPU / W ))}"; (( THREADS < 1 )) && THREADS=1

# ── Environment (pytest + aggregation need the venv; run_parallel.sh self-activates) ──
if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/venv/bin/activate"
fi
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[campaign $(date -u +%H:%M:%S)] $*"; }

log "start: security=$SECURITY npc=$NPC workers=$W threads=$THREADS ncpu=$NCPU mem=${MEM_GB}G budget=${PER_WORKER_GB}G/worker"
log "datasets: iris(k=$K_IRIS) breast_cancer(k=$K_BC)  alphas: ${ALPHAS[*]}"

# ── Step 0: fast plaintext test gate (recorded, non-fatal, time-bounded) ────
# ONLY the fast plaintext logic tests run here — they validate the campaign's own
# new code (partitioner, flags, report/comm instrumentation) in seconds. The real
# FHE/solver tests (test_householder_sign, *_cipher) are deliberately EXCLUDED:
# pytest inherits uncontrolled OMP threads, so on a many-core box every EvalRotate
# oversubscribes all cores and those tests crawl for many minutes — they must never
# block the actual metrics runs (which the campaign runs anyway, with controlled
# threading). Threads pinned to 1 (tiny inputs) + a hard timeout as a backstop.
GATE_TESTS="federated_lssvm/test_partition_dirichlet.py federated_lssvm/test_solver_selection.py federated_lssvm/test_run_report.py federated_lssvm/test_comm_bytes.py"
command -v timeout >/dev/null && TO="timeout 300" || TO=""
log "pytest (fast plaintext gate) ..."
if env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       $TO pytest -q $GATE_TESTS >"$RESULTS_DIR/pytest.txt" 2>&1; then
    PYTEST_STATUS="passed"
else
    PYTEST_STATUS="FAILED/timeout(rc=$?)"
fi
log "pytest: $PYTEST_STATUS (see $RESULTS_DIR_NAME/pytest.txt)"

# ── Step 1: the FHE run matrix (datasets x {iid, dirichlet alphas}) ─────────
# One run at a time; each writes models/k=$K, which we move to a per-config dir.
run_one() {
    local name="$1" dataset="$2" k="$3"; shift 3
    local extra=("$@")            # partition flags (empty for iid)
    local dest="$RESULTS_DIR/$name"

    if [ -f "$dest/report.md" ]; then
        log "skip $name (already done)"
        return 0
    fi

    local disp="(iid)"
    [ "${#extra[@]}" -gt 0 ] && disp="${extra[*]}"
    log "RUN $name: dataset=$dataset k=$k security=$SECURITY $disp"
    rm -rf "$dest" "models/k=$k"          # clean stale checkpoints (per-client, not per-partition)
    mkdir -p "$dest"

    if bash "$RUN_PARALLEL" "$k" "$W" "$THREADS" \
            --dataset="$dataset" --security="$SECURITY" --n-per-class="$NPC" \
            "${extra[@]+"${extra[@]}"}" >"$dest/run.log" 2>&1; then
        # flatten models/k=$k INTO $dest so files land at $dest/{report.md,metrics.csv,…}
        mv "models/k=$k"/* "$dest/" && rm -rf "models/k=$k"
        log "OK   $name"
    else
        touch "$dest/FAILED"
        log "FAIL $name (see $RESULTS_DIR_NAME/$name/run.log) — continuing"
    fi
}

for dataset in $DATASETS; do
    case "$dataset" in
        iris)          k="$K_IRIS"; short="iris" ;;
        breast_cancer) k="$K_BC";   short="bc" ;;
        *) log "unknown dataset '$dataset' — skipping"; continue ;;
    esac
    run_one "${short}_iid" "$dataset" "$k"
    for a in "${ALPHAS[@]}"; do
        run_one "${short}_dir_a${a}" "$dataset" "$k" --partition=dirichlet --alpha="$a"
    done
done

# ── Step 2: aggregate all metrics into files ────────────────────────────────
log "aggregating ..."

ALL_CSV="$RESULTS_DIR/all_metrics.csv"
echo "config,partition,alpha,timestamp,dataset,k,class,approach,accuracy_pct,precision_pct,f1_pct,w_err,total_s" >"$ALL_CSV"
COMM="$RESULTS_DIR/comm_memory.txt"
: >"$COMM"

for dir in "$RESULTS_DIR"/*/; do
    name="$(basename "$dir")"
    # derive partition/alpha from the config name (iid, or dir_a<alpha>)
    part="iid"; alpha=""
    case "$name" in *_dir_a*) part="dirichlet"; alpha="${name##*_a}";; esac

    csv="$dir/metrics.csv"
    [ -f "$csv" ] || csv="$(find "$dir" -name metrics.csv -print -quit 2>/dev/null)"
    if [ -n "${csv:-}" ] && [ -f "$csv" ]; then
        awk -v c="$name" -v p="$part" -v a="$alpha" 'NR>1 {print c","p","a","$0}' "$csv" >>"$ALL_CSV"
    fi

    rep="$dir/report.md"
    [ -f "$rep" ] || rep="$(find "$dir" -name report.md -print -quit 2>/dev/null)"
    if [ -n "${rep:-}" ] && [ -f "$rep" ]; then
        {
            echo "== $name =="
            grep -iE "per-client uplink|rounds=|peak RSS|\*\*Total:" "$rep" || echo "(no comm/mem lines)"
            echo
        } >>"$COMM"
    fi
done

# ── SUMMARY.md ──────────────────────────────────────────────────────────────
SUMMARY="$RESULTS_DIR/SUMMARY.md"
{
    echo "# Campaign summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- security=$SECURITY  n-per-class=$NPC  workers=$W  threads=$THREADS  ncpu=$NCPU"
    echo "- pytest: **$PYTEST_STATUS** (\`pytest.txt\`)"
    echo
    echo "| config | status | report |"
    echo "|---|---|---|"
    for dir in "$RESULTS_DIR"/*/; do
        name="$(basename "$dir")"
        if [ -f "$dir/FAILED" ]; then st="FAILED"
        elif [ -f "$dir/report.md" ] || find "$dir" -name report.md -print -quit | grep -q .; then st="ok"
        else st="—"; fi
        echo "| $name | $st | \`$name/report.md\` |"
    done
    echo
    echo "Combined metrics: \`all_metrics.csv\`  |  comm/memory: \`comm_memory.txt\`"
    echo "Per-worker peak RSS is in each config's \`report.md\` (**Per worker** table)."
} >"$SUMMARY"

# ── tarball for scp ─────────────────────────────────────────────────────────
tar czf "$RESULTS_DIR/campaign_results.tar.gz" -C "$REPO_ROOT" \
    --exclude="$RESULTS_DIR_NAME/campaign_results.tar.gz" "$RESULTS_DIR_NAME" 2>/dev/null || true

log "done. results in $RESULTS_DIR (SUMMARY.md, all_metrics.csv, campaign_results.tar.gz)"
