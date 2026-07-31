"""Thread-count bootstrap for FHE entry scripts.

`init_threads()` sets OpenMP env vars before openfhe / numpy import so the C++
runtime picks them up. Resolution order: explicit arg > FHE_THREADS env >
OMP_NUM_THREADS env > os.cpu_count().
"""

from __future__ import annotations

import ctypes
import os
import sys


DEFAULT_THREADS = int(os.environ.get("FHE_DEFAULT_THREADS", "6"))

# OpenMP runtime candidates across platforms/toolchains (libgomp on Linux, LLVM
# libomp on macOS/clang). Used to set the thread count at runtime — needed by the
# fork pool, which loads the context single-threaded then bumps each child's pool.
_OMP_LIBS = (
    "libgomp.so.1",
    "libomp.so",
    "libomp.dylib",
    "/opt/homebrew/opt/libomp/lib/libomp.dylib",
    "/usr/local/opt/libomp/lib/libomp.dylib",
)


def set_omp_threads(n: int) -> bool:
    """Call omp_set_num_threads(n) via whichever OMP runtime is loadable.

    Returns True if the runtime API was reached (so the change takes effect even
    after OpenMP has middle-initialized, which env vars can no longer influence).
    """
    for name in _OMP_LIBS:
        try:
            lib = ctypes.CDLL(name)
            lib.omp_set_num_threads.argtypes = [ctypes.c_int]
            lib.omp_set_num_threads(int(n))
            return True
        except (OSError, AttributeError):
            continue
    return False


def _parse_threads_arg(argv: list[str]) -> tuple[int | None, list[str]]:
    """Extract --threads=N from argv. Returns (n_or_None, remaining_argv)."""
    n: int | None = None
    rest: list[str] = []
    for a in argv:
        if a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
        elif a == "--threads":
            raise SystemExit("--threads requires =N form, e.g. --threads=4")
        else:
            rest.append(a)
    return n, rest


def init_threads(n: int | None = None) -> int:
    # Honor the documented resolution order. FHE_THREADS is set by a previous
    # init_threads() call, so a second bootstrap in the same process (worker.py
    # then its train.py import) keeps the --threads value instead of resetting
    # to the default.
    if n is None:
        for var in ("FHE_THREADS", "OMP_NUM_THREADS"):
            val = os.environ.get(var)
            if val and val.isdigit():
                n = int(val)
                break
    if n is None:
        n = DEFAULT_THREADS
    n = max(1, n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["FHE_THREADS"] = str(n)
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    set_omp_threads(n)
    return n


def bootstrap() -> int:
    """Parse --threads= from sys.argv (mutating it) and call init_threads."""
    n, rest = _parse_threads_arg(sys.argv[1:])
    sys.argv[1:] = rest
    resolved = init_threads(n)
    print(f"[parallel] OMP_NUM_THREADS={resolved}")
    return resolved
