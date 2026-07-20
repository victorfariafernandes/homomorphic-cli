"""Tests for config.report — markdown run report generated from run logs."""

import textwrap

import pytest

from config.report import generate_report, parse_finalize, parse_worker_log

WORKER_LOG = textwrap.dedent("""\
    [parallel] OMP_NUM_THREADS=4
    [worker 1/4] Loading shared context from models/k=40/class_0 ...
    [worker 1/4] Context loaded in 152.3s
    [worker 1/4] 3/123 tasks: [(0, 1), (1, 9), (2, 'baseline')]
      [worker 1/4 class 0 client 1] Resuming from checkpoint models/k=40/class_0/client_1
    [worker 1/4] progress: 1/3 (solved=0, resumed=1)  elapsed=0.1min
      [worker 1/4 class 1 client 9] H=(4, 4), cond=8.3 ...
      [worker 1/4 class 1 client 9] FHE solve: 551.2s
      [worker 1/4 class 1 client 9] Checkpoint saved.
    [worker 1/4] progress: 2/3 (solved=1, resumed=1)  elapsed=9.3min
      [worker 1/4 class 2 baseline] H=(5, 5), cond=20.3 ...
      [worker 1/4 class 2 baseline] FHE solve: 700.0s
      [worker 1/4 class 2 baseline] Checkpoint saved.
    [worker 1/4] DONE: 2 solved, 1 resumed, 21.0min total
""")

FINALIZE_LOG = textwrap.dedent("""\
    === Federated FHE LS-SVM (Iris OvR, k=40 clients) ===
    --- Class 0 (setosa vs rest) [kernel=linear] ---
      [client 0] H=(4, 4), cond=15.8 ...
      [client 0] FHE solve: 397.6s
      [client 0] Checkpoint saved.
      [client 1] Resuming from checkpoint models/k=40/class_0/client_1
      Aggregating 40 encrypted models ...
      Aggregation: 20.101s
      Cipher predict: 21.5958s
      [baseline] FHE solve: 537.7s
      --- Class 0 (setosa vs rest) comparison (k=40) ---
      Approach                                 | Accuracy  | Precision | F1
      -----------------------------------------+----------+-----------+------
      Single-client FHE (N_per_class=2, seed=42) |  96.67% |   100.00% | 94.74%
      Federated FHE  (40 clients avg.)         | 100.00% |   100.00% | 100.00%
      Federated plaintext reference            | 100.00% |   100.00% | 100.00%
      Full-data plaintext reference            | 100.00% |   100.00% | 100.00%
      FHE fed weights vs plaintext fed weights: 1.2219e-01

    OvR Multiclass Accuracy (Federated FHE, k=40): 100.00%
    OvR Multiclass Accuracy (Single-client FHE):      43.33%
""")


def test_parse_worker_log():
    w = parse_worker_log(WORKER_LOG)
    assert w["context_load_s"] == pytest.approx(152.3)
    assert w["solved"] == 2 and w["resumed"] == 1
    assert w["total_min"] == pytest.approx(21.0)
    assert (0, "1") in w["tasks"] and w["tasks"][(0, "1")] is None  # resumed
    assert w["tasks"][(1, "9")] == pytest.approx(551.2)
    assert w["tasks"][(2, "baseline")] == pytest.approx(700.0)


def test_parse_finalize_metrics():
    f = parse_finalize(FINALIZE_LOG)
    c0 = f["classes"][0]
    assert c0["name"] == "setosa"
    fed = dict((r[0], r[1:]) for r in c0["rows"])["Federated FHE  (40 clients avg.)"]
    assert fed == (pytest.approx(100.0), pytest.approx(100.0), pytest.approx(100.0))
    assert c0["w_err"] == pytest.approx(1.2219e-01)
    assert f["multiclass"]["Federated FHE, k=40"] == pytest.approx(100.0)
    assert f["multiclass"]["Single-client FHE"] == pytest.approx(43.33)
    # serial-pass solve/resume times also captured, keyed by class
    assert f["client_times"][(0, "0")] == pytest.approx(397.6)
    assert f["client_times"][(0, "1")] is None
    assert f["client_times"][(0, "baseline")] == pytest.approx(537.7)


def test_generate_report(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "worker_1.log").write_text(WORKER_LOG)
    (logs / "finalize.log").write_text(FINALIZE_LOG)
    out = tmp_path / "report.md"
    generate_report(
        k=40, dataset="iris", logs_dir=str(logs), out_path=str(out),
        phase_seconds={"prepare": 13.0, "workers": 1260.0, "finalize": 900.0},
    )
    text = out.read_text()
    assert "# FHE LS-SVM run report" in text
    assert "iris" in text and "k=40" in text
    # metrics
    assert "100.00" in text and "43.33" in text and "94.74" in text
    # time spent: total + phases
    assert "36.2 min" in text  # (13+1260+900)/60 total
    # per worker
    assert "worker 1" in text and "21.0" in text and "152.3" in text
    # per client
    assert "551.2" in text and "700.0" in text and "resumed" in text


def test_report_appends_never_replaces(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "worker_1.log").write_text(WORKER_LOG)
    (logs / "finalize.log").write_text(FINALIZE_LOG)
    out = tmp_path / "report.md"
    for _ in range(2):
        generate_report(
            k=40, dataset="iris", logs_dir=str(logs), out_path=str(out),
            phase_seconds={"prepare": 1.0, "workers": 2.0, "finalize": 3.0},
        )
    text = out.read_text()
    assert text.count("## Run ") == 2, "each invocation must append a new run section"
    csv_text = (tmp_path / "metrics.csv").read_text()
    lines = [l for l in csv_text.strip().splitlines() if l]
    assert lines[0].startswith("timestamp,")  # header written once
    assert len(lines) == 1 + 2 * 4  # 4 approach-rows per class per run, 1 class, 2 runs
