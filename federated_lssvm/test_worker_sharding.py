"""Sharding invariants for the parallel worker: disjoint, complete, deterministic."""

import pytest

import federated_lssvm.train as T
from federated_lssvm.worker import _parse_args, task_list


@pytest.mark.parametrize("n_classes,k", [(3, 3), (3, 40), (1, 1), (2, 7)])
def test_task_list_shape(n_classes, k):
    tasks = task_list(n_classes, k)
    assert len(tasks) == n_classes * k + n_classes
    assert len(set(tasks)) == len(tasks)
    assert tasks == task_list(n_classes, k)  # deterministic
    for c in range(n_classes):
        assert (c, "baseline") in tasks
        assert all((c, i) in tasks for i in range(k))


@pytest.mark.parametrize("num_shards", [1, 3, 16, 200])
def test_shards_are_disjoint_and_complete(num_shards):
    tasks = task_list(3, 40)
    shards = [tasks[i::num_shards] for i in range(num_shards)]
    combined = [t for shard in shards for t in shard]
    assert sorted(map(str, combined)) == sorted(map(str, tasks))
    assert len(combined) == len(tasks)
    # Interleaved assignment keeps worker loads within one task of each other
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_parse_args_shard():
    k, shard, prepare, n_per_class, workers = _parse_args(["--k=40", "--shard=3/16"])
    assert (k, shard, prepare, n_per_class, workers) == (40, (3, 16), False, None, None)


def test_parse_args_prepare_context():
    k, shard, prepare, n_per_class, workers = _parse_args(
        ["--k=3", "--prepare-context", "--n-per-class=2"]
    )
    assert (k, shard, prepare, n_per_class, workers) == (3, None, True, 2, None)


def test_parse_args_workers():
    k, shard, prepare, n_per_class, workers = _parse_args(["--k=40", "--workers=12"])
    assert (k, shard, prepare, n_per_class, workers) == (40, None, False, None, 12)


def test_parse_args_ignores_solver_and_security():
    k, shard, *_ = _parse_args(["--k=3", "--shard=0/3", "--solver=qr_row", "--security=notset"])
    assert (k, shard) == (3, (0, 3))


def test_parse_args_tolerates_partition_and_alpha():
    # partition/alpha are resolved in worker.main() via solver_selection (like
    # solver/security/dataset), not by _parse_args -- passing them must not break it.
    k, shard, *_ = _parse_args(["--k=3", "--shard=0/3", "--partition=dirichlet", "--alpha=0.5"])
    assert (k, shard) == (3, (0, 3))


@pytest.mark.parametrize("bad", ["--shard=16/16", "--shard=-1/16"])
def test_parse_args_rejects_out_of_range_shard(bad):
    with pytest.raises(SystemExit):
        _parse_args(["--k=40", bad])


def test_parse_args_rejects_zero_workers():
    with pytest.raises(SystemExit):
        _parse_args(["--k=40", "--workers=0"])


def test_parse_args_requires_a_mode():
    with pytest.raises(SystemExit):
        _parse_args(["--k=40"])


@pytest.mark.parametrize("combo", [
    ["--shard=0/3", "--workers=3"],
    ["--shard=0/3", "--prepare-context"],
    ["--workers=3", "--prepare-context"],
])
def test_parse_args_rejects_multiple_modes(combo):
    with pytest.raises(SystemExit):
        _parse_args(["--k=40", *combo])


@pytest.mark.parametrize("workers", [1, 3, 12, 16])
def test_fork_pool_shards_match_separate_processes(workers):
    """The --workers=W fork pool runs run_shard(shard_idx=i, num_shards=W) per child,
    so its per-child task slices must union to the full list with no overlap/gaps —
    identical coverage to W separate --shard=i/W processes."""
    tasks = task_list(3, 40)
    per_child = [tasks[i::workers] for i in range(workers)]
    combined = [t for child in per_child for t in child]
    assert sorted(map(str, combined)) == sorted(map(str, tasks))
    assert len(combined) == len(tasks)
    assert max(len(c) for c in per_child) - min(len(c) for c in per_child) <= 1


# ── context marker: must survive save_global_checkpoint's checkpoint.json ──


def test_marker_is_not_checkpoint_json(tmp_path):
    assert T._class_checkpoint_marker(str(tmp_path)) != f"{tmp_path}/checkpoint.json"


def test_read_depth_from_context_marker(tmp_path):
    (tmp_path / "context_marker.txt").write_text("schema_version=1\ndepth=124\n")
    assert T._read_checkpoint_depth(str(tmp_path)) == 124


def test_read_depth_legacy_checkpoint_json(tmp_path):
    (tmp_path / "checkpoint.json").write_text("schema_version=1\ndepth=44\n")
    assert T._read_checkpoint_depth(str(tmp_path)) == 44


def test_solver_json_does_not_mask_marker(tmp_path):
    # save_global_checkpoint format: real JSON, no depth= line
    (tmp_path / "checkpoint.json").write_text('{\n  "schema_version": 1\n}\n')
    assert T._read_checkpoint_depth(str(tmp_path)) is None
    (tmp_path / "context_marker.txt").write_text("schema_version=1\ndepth=124\n")
    assert T._read_checkpoint_depth(str(tmp_path)) == 124


def test_refuses_fresh_context_over_existing_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "models" / "k=5" / "class_0" / "client_3"
    ckpt.mkdir(parents=True)
    (ckpt / "weights.bin").write_bytes(b"x")
    with pytest.raises(SystemExit, match="Refusing to create"):
        T.assert_safe_to_create_context(5)
    T.assert_safe_to_create_context(6)  # clean k: no error
