"""Shared solver selection helpers for federated LSSVM entry points."""

from __future__ import annotations

import importlib
import os

from lssvm.preprocessors import DEFAULT_DATASET, SUPPORTED_DATASETS

SUPPORTED_SOLVER_MODULES = {
    "cg": "lssvm.solvers.cg_cipher",
    "cg_cipher": "lssvm.solvers.cg_cipher",
    "qr_row": "lssvm.solvers.qr_householder_cipher_row",
    "qr_householder_cipher_row": "lssvm.solvers.qr_householder_cipher_row",
    "qr_col": "lssvm.solvers.qr_householder_cipher_col",
    "qr_householder_cipher_col": "lssvm.solvers.qr_householder_cipher_col",
}

UNSUPPORTED_SOLVERS = {"qr_cell", "qr_householder_cipher_cell"}

DEFAULT_SOLVER_NAME = "qr_row"
SOLVER_ENV_VAR = "LSSVM_SOLVER"

SUPPORTED_SECURITY_LEVELS = {"128", "notset"}
DEFAULT_SECURITY_LEVEL = "128"
SECURITY_ENV_VAR = "LSSVM_SECURITY"

DATASET_ENV_VAR = "LSSVM_DATASET"

SUPPORTED_PARTITIONS = {"iid", "dirichlet"}
DEFAULT_PARTITION = "iid"
PARTITION_ENV_VAR = "LSSVM_PARTITION"
ALPHA_ENV_VAR = "LSSVM_ALPHA"

DEFAULT_MODELS_ROOT = "models"
MODELS_ROOT_ENV_VAR = "LSSVM_MODELS_ROOT"


def _extract_flag(args: list[str], flag: str) -> str | None:
    """Return the value of `--flag value` or `--flag=value`, or None if absent."""
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 >= len(args):
                raise ValueError(f"{flag} requires a value")
            return args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def parse_solver_name(args: list[str], env_var: str = SOLVER_ENV_VAR) -> str:
    """Resolve a solver name from argv-style args and an optional environment override."""
    arg_solver = None
    for index, arg in enumerate(args):
        if arg == "--solver":
            if index + 1 >= len(args):
                raise ValueError("--solver requires a value")
            arg_solver = args[index + 1]
            break
        if arg.startswith("--solver="):
            arg_solver = arg.split("=", 1)[1]
            break

    env_solver = os.environ.get(env_var)
    solver_name = (arg_solver or env_solver or DEFAULT_SOLVER_NAME).strip()
    if not solver_name:
        raise ValueError("solver name cannot be empty")
    return solver_name


def parse_security_level(args: list[str], env_var: str = SECURITY_ENV_VAR) -> str:
    """Resolve a security level ("128" or "notset") from argv-style args and env override."""
    arg_security = None
    for index, arg in enumerate(args):
        if arg == "--security":
            if index + 1 >= len(args):
                raise ValueError("--security requires a value")
            arg_security = args[index + 1]
            break
        if arg.startswith("--security="):
            arg_security = arg.split("=", 1)[1]
            break

    env_security = os.environ.get(env_var)
    security = (arg_security or env_security or DEFAULT_SECURITY_LEVEL).strip()
    if security not in SUPPORTED_SECURITY_LEVELS:
        supported = ", ".join(sorted(SUPPORTED_SECURITY_LEVELS))
        raise ValueError(f"Unsupported security level '{security}'. Supported: {supported}")
    return security


def parse_dataset_name(args: list[str], env_var: str = DATASET_ENV_VAR) -> str:
    """Resolve a dataset name from argv-style args and an optional env override."""
    arg_dataset = None
    for index, arg in enumerate(args):
        if arg == "--dataset":
            if index + 1 >= len(args):
                raise ValueError("--dataset requires a value")
            arg_dataset = args[index + 1]
            break
        if arg.startswith("--dataset="):
            arg_dataset = arg.split("=", 1)[1]
            break

    env_dataset = os.environ.get(env_var)
    dataset = (arg_dataset or env_dataset or DEFAULT_DATASET).strip()
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {supported}")
    return dataset


def parse_partition_name(args: list[str], env_var: str = PARTITION_ENV_VAR) -> str:
    """Resolve the client partitioning scheme ("iid" or "dirichlet")."""
    arg_partition = _extract_flag(args, "--partition")
    env_partition = os.environ.get(env_var)
    partition = (arg_partition or env_partition or DEFAULT_PARTITION).strip()
    if partition not in SUPPORTED_PARTITIONS:
        supported = ", ".join(sorted(SUPPORTED_PARTITIONS))
        raise ValueError(f"Unsupported partition '{partition}'. Supported: {supported}")
    return partition


def parse_alpha(args: list[str], env_var: str = ALPHA_ENV_VAR) -> float | None:
    """Resolve the Dirichlet concentration alpha; None when unset (IID)."""
    arg_alpha = _extract_flag(args, "--alpha")
    raw = arg_alpha if arg_alpha is not None else os.environ.get(env_var)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"--alpha must be a number, got '{raw}'")


def parse_models_root(args: list[str], env_var: str = MODELS_ROOT_ENV_VAR) -> str:
    """Resolve the output root directory for serialized models (default "models").

    Lets concurrent/sequential runs with the same k (e.g. different datasets or
    partition schemes) write to separate trees instead of colliding under models/k=N/.
    """
    arg_root = _extract_flag(args, "--models-root")
    env_root = os.environ.get(env_var)
    root = (arg_root or env_root or DEFAULT_MODELS_ROOT).strip()
    if not root:
        raise ValueError("models root cannot be empty")
    return root.rstrip("/")


def resolve_solver_module(solver_name: str):
    """Import a supported solver module and verify its checkpoint hooks."""
    normalized = solver_name.strip()
    if normalized in UNSUPPORTED_SOLVERS:
        raise ValueError(
            "qr_householder_cipher_cell has been removed; choose cg, qr_row, or qr_col"
        )

    module_name = SUPPORTED_SOLVER_MODULES.get(
        normalized,
        normalized if normalized.startswith("lssvm.solvers.") else None,
    )
    if module_name is None:
        supported = ", ".join(sorted({"cg", "qr_row", "qr_col"}))
        raise ValueError(
            f"Unsupported solver '{solver_name}'. Supported solvers: {supported}"
        )

    module = importlib.import_module(module_name)
    validate_solver_hooks(module)
    return module


def validate_solver_hooks(module) -> None:
    """Fail fast if a solver does not expose the federated checkpoint contract."""
    required_hooks = (
        "save_global_checkpoint",
        "load_global_checkpoint",
        "checkpoint_capabilities",
    )
    missing = [hook for hook in required_hooks if not hasattr(module, hook)]
    if missing:
        raise AttributeError(
            f"Solver module '{module.__name__}' is missing required checkpoint hooks: "
            f"{', '.join(missing)}"
        )

    capabilities = module.checkpoint_capabilities()
    if not isinstance(capabilities, dict):
        raise TypeError(
            f"Solver module '{module.__name__}' returned non-dict checkpoint capabilities"
        )
    if "schema_version" not in capabilities:
        raise ValueError(
            f"Solver module '{module.__name__}' checkpoint capabilities must include 'schema_version'"
        )
