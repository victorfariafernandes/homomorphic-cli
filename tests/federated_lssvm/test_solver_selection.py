"""Parsing of the --partition / --alpha flags (non-IID partitioner selection)."""

import pytest

from federated_lssvm.solver_selection import parse_partition_name, parse_alpha


# ── parse_partition_name ───────────────────────────────────────────────────

def test_partition_defaults_to_iid():
    assert parse_partition_name([]) == "iid"


@pytest.mark.parametrize("args", [["--partition", "dirichlet"], ["--partition=dirichlet"]])
def test_partition_accepts_both_flag_forms(args):
    assert parse_partition_name(args) == "dirichlet"


def test_partition_env_fallback(monkeypatch):
    monkeypatch.setenv("LSSVM_PARTITION", "dirichlet")
    assert parse_partition_name([], env_var="LSSVM_PARTITION") == "dirichlet"


def test_partition_rejects_unsupported():
    with pytest.raises(ValueError):
        parse_partition_name(["--partition=shards"])


# ── parse_alpha ────────────────────────────────────────────────────────────

def test_alpha_defaults_to_none():
    assert parse_alpha([]) is None


@pytest.mark.parametrize("args", [["--alpha", "0.5"], ["--alpha=0.5"]])
def test_alpha_accepts_both_flag_forms(args):
    assert parse_alpha(args) == 0.5


def test_alpha_rejects_non_numeric():
    with pytest.raises(ValueError):
        parse_alpha(["--alpha=abc"])
