"""Append-only run report for the federated FHE pipeline.

Parses the logs written by config/run_parallel.sh (prepare/worker/finalize) and
federated_lssvm.train, then appends a per-run section to models/k=K/report.md and
per-metric rows to models/k=K/metrics.csv. Nothing is ever overwritten: every
invocation adds a new "## Run <timestamp>" section and new CSV rows, so metric
history accumulates across runs.

Usage (invoked by run_parallel.sh step [4/4], or manually):
  python3 -m config.report --k=40 --dataset=iris --logs=models/k=40/logs \
      [--out=models/k=40/report.md] [--prepare-s=13 --workers-s=1260 --finalize-s=900]
"""

from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime, timezone

_RE_CTX_LOAD = re.compile(r"\[worker (\d+)/\d+\] Context loaded in ([\d.]+)s")
_RE_TASK_SOLVE = re.compile(
    r"\[worker \d+/\d+ class (\d+) (?:client (\S+)|(baseline))\] FHE solve: ([\d.]+)s"
)
_RE_TASK_RESUME = re.compile(
    r"\[worker \d+/\d+ class (\d+) (?:client (\S+)|(baseline))\] Resuming from checkpoint"
)
_RE_DONE = re.compile(r"\[worker \d+/\d+\] DONE: (\d+) solved, (\d+) resumed, ([\d.]+)min total")

_RE_CLASS_HDR = re.compile(r"--- Class (\d+) \((\S+) vs rest\)")
_RE_FIN_SOLVE = re.compile(r"\[(?:client (\S+)|(baseline))\] FHE solve: ([\d.]+)s")
_RE_FIN_RESUME = re.compile(r"\[(?:client (\S+)|(baseline))\] Resuming from checkpoint")
_RE_METRIC_ROW = re.compile(
    r"^\s{2,}(\S.*?)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*$"
)
_RE_W_ERR = re.compile(r"FHE fed weights vs plaintext fed weights: ([\d.eE+-]+)")
_RE_MULTICLASS = re.compile(r"Multiclass Accuracy \((.+?)\):\s+([\d.]+)%")


def parse_worker_log(text: str) -> dict:
    """Extract context-load time, per-task solve times (None = resumed), DONE totals."""
    w: dict = {"context_load_s": None, "tasks": {}, "solved": 0, "resumed": 0, "total_min": None}
    for line in text.splitlines():
        m = _RE_CTX_LOAD.search(line)
        if m:
            w["context_load_s"] = float(m.group(2))
            continue
        m = _RE_TASK_SOLVE.search(line)
        if m:
            cls, client = int(m.group(1)), (m.group(2) or m.group(3))
            w["tasks"][(cls, client)] = float(m.group(4))
            continue
        m = _RE_TASK_RESUME.search(line)
        if m:
            cls, client = int(m.group(1)), (m.group(2) or m.group(3))
            w["tasks"].setdefault((cls, client), None)
            continue
        m = _RE_DONE.search(line)
        if m:
            w["solved"], w["resumed"] = int(m.group(1)), int(m.group(2))
            w["total_min"] = float(m.group(3))
    return w


def parse_finalize(text: str) -> dict:
    """Extract per-class metric tables, w_err, multiclass accuracies, per-client times."""
    out: dict = {"classes": [], "multiclass": {}, "client_times": {}}
    current_class: int | None = None
    current_name: str | None = None
    cls_entry: dict | None = None
    for line in text.splitlines():
        m = _RE_CLASS_HDR.search(line)
        if m:
            current_class, current_name = int(m.group(1)), m.group(2)
            if "comparison" in line:
                cls_entry = {"idx": current_class, "name": current_name, "rows": [], "w_err": None}
                out["classes"].append(cls_entry)
            continue
        if current_class is not None:
            m = _RE_FIN_SOLVE.search(line)
            if m:
                client = m.group(1) or m.group(2)
                out["client_times"][(current_class, client)] = float(m.group(3))
                continue
            m = _RE_FIN_RESUME.search(line)
            if m:
                client = m.group(1) or m.group(2)
                out["client_times"].setdefault((current_class, client), None)
                continue
        if cls_entry is not None:
            m = _RE_METRIC_ROW.match(line)
            if m and m.group(1) != "Approach":
                cls_entry["rows"].append(
                    (m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)))
                )
                continue
            m = _RE_W_ERR.search(line)
            if m:
                cls_entry["w_err"] = float(m.group(1))
                continue
        m = _RE_MULTICLASS.search(line)
        if m:
            out["multiclass"][m.group(1)] = float(m.group(2))
    return out


def _fmt_min(seconds: float | None) -> str:
    return f"{seconds / 60:.1f} min" if seconds is not None else "n/a"


def generate_report(
    k: int,
    dataset: str,
    logs_dir: str,
    out_path: str,
    phase_seconds: dict | None = None,
) -> str:
    """Append one run section to out_path (+ rows to metrics.csv next to it)."""
    phase_seconds = phase_seconds or {}
    workers = {}
    for fn in sorted(os.listdir(logs_dir)):
        m = re.match(r"worker_(\d+)\.log$", fn)
        if m:
            with open(os.path.join(logs_dir, fn), encoding="utf-8") as f:
                workers[int(m.group(1))] = parse_worker_log(f.read())
    finalize = {"classes": [], "multiclass": {}, "client_times": {}}
    fin_path = os.path.join(logs_dir, "finalize.log")
    if os.path.exists(fin_path):
        with open(fin_path, encoding="utf-8") as f:
            finalize = parse_finalize(f.read())

    # Merge per-client times: worker solves take precedence; finalize fills serial solves.
    client_times: dict = {}
    client_worker: dict = {}
    for wid, w in workers.items():
        for key, secs in w["tasks"].items():
            if secs is not None or key not in client_times:
                client_times[key] = secs
                client_worker[key] = wid
    for key, secs in finalize["client_times"].items():
        if secs is not None or key not in client_times:
            client_times.setdefault(key, secs)
            if secs is not None:
                client_times[key] = secs
                client_worker.setdefault(key, "finalize")

    solve_sum = sum(s for s in client_times.values() if s)
    total_s = sum(v for v in phase_seconds.values() if v) or None
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []
    if not os.path.exists(out_path):
        lines.append("# FHE LS-SVM run report\n")
    lines.append(f"\n## Run {ts} — dataset={dataset}, k={k}\n")

    lines.append("### Results\n")
    for cls in finalize["classes"]:
        lines.append(f"**Class {cls['idx']} ({cls['name']} vs rest)**\n")
        lines.append("| Approach | Accuracy | Precision | F1 |")
        lines.append("|---|---|---|---|")
        for approach, acc, prec, f1 in cls["rows"]:
            lines.append(f"| {approach} | {acc:.2f}% | {prec:.2f}% | {f1:.2f}% |")
        if cls["w_err"] is not None:
            lines.append(f"\nFHE vs plaintext federated weights rel. error: {cls['w_err']:.4e}\n")
    for label, acc in finalize["multiclass"].items():
        lines.append(f"- Multiclass accuracy ({label}): {acc:.2f}%")

    lines.append("\n### Time spent\n")
    lines.append(f"- **Total: {_fmt_min(total_s)}**")
    for phase in ("prepare", "workers", "finalize"):
        if phase in phase_seconds:
            lines.append(f"  - {phase}: {_fmt_min(phase_seconds[phase])}")
    lines.append(f"- Sum of FHE solve times: {_fmt_min(solve_sum)}")

    if workers:
        lines.append("\n**Per worker**\n")
        lines.append("| worker | context load (s) | solved | resumed | total (min) |")
        lines.append("|---|---|---|---|---|")
        for wid in sorted(workers):
            w = workers[wid]
            lines.append(
                f"| worker {wid} | {w['context_load_s'] if w['context_load_s'] is not None else 'n/a'} "
                f"| {w['solved']} | {w['resumed']} | {w['total_min'] if w['total_min'] is not None else 'n/a'} |"
            )

    if client_times:
        lines.append("\n**Per client**\n")
        lines.append("| class | client | solve time (s) | worker |")
        lines.append("|---|---|---|---|")
        for (cls, client) in sorted(client_times, key=lambda t: (t[0], t[1] == "baseline", str(t[1]).zfill(4))):
            secs = client_times[(cls, client)]
            who = client_worker.get((cls, client), "")
            lines.append(
                f"| {cls} | {client} | {f'{secs:.1f}' if secs is not None else 'resumed'} | {who} |"
            )
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Append machine-readable metric history next to the report.
    csv_path = os.path.join(os.path.dirname(out_path) or ".", "metrics.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(
                ["timestamp", "dataset", "k", "class", "approach",
                 "accuracy_pct", "precision_pct", "f1_pct", "w_err", "total_s"]
            )
        for cls in finalize["classes"]:
            for approach, acc, prec, f1 in cls["rows"]:
                wr.writerow(
                    [ts, dataset, k, cls["idx"], approach, acc, prec, f1,
                     cls["w_err"] if cls["w_err"] is not None else "", total_s or ""]
                )
    return out_path


def main() -> None:
    args = sys.argv[1:]

    def _get(name: str, default=None):
        for a in args:
            if a.startswith(f"--{name}="):
                return a.split("=", 1)[1]
        return default

    k = int(_get("k", "3"))
    dataset = _get("dataset", "iris")
    logs_dir = _get("logs", f"models/k={k}/logs")
    out_path = _get("out", f"models/k={k}/report.md")
    phases = {}
    for phase in ("prepare", "workers", "finalize"):
        v = _get(f"{phase}-s")
        if v is not None:
            phases[phase] = float(v)
    path = generate_report(k, dataset, logs_dir, out_path, phase_seconds=phases)
    print(f"[report] appended run section to {path}")


if __name__ == "__main__":
    main()
