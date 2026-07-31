"""OpenMP parallelism smoke test for the FHE bootstrapping path.

Runs a small CKKS bootstrapping workload and measures how many cores it actually
uses (CPU-seconds / wall-seconds). If OpenFHE was built WITHOUT OpenMP it runs
single-threaded (avg_cores ~1 regardless of --threads) — the 2026-07-29 cloud
regression. If OpenMP is linked, bootstrapping parallelizes to ~threads*0.8.

Runs at a small ring by default (fits ~24 GB) since bootstrapping parallelism is
ring-independent (verified: N=65536 and N=131072 both scale ~0.8/thread on x86).

Usage:
  python -m config.omp_smoke                 # 4 threads, ring 65536
  OMP=8 RING=131072 python -m config.omp_smoke
Exit code 0 = OpenMP engaged (PASS), 1 = serial build (FAIL).
"""
import os
import sys
import time
import resource

from config.parallel import init_threads

N = int(os.environ.get("OMP", "4"))
RING = int(os.environ.get("RING", "65536"))
NBOOT = int(os.environ.get("NBOOT", "8"))
SLOTS = 32
PASS_THRESHOLD = 1.5   # avg_cores below this at >=2 threads => OpenMP not engaging

init_threads(N)

from openfhe import (  # noqa: E402
    CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature, ScalingTechnique,
    SecretKeyDist, SecurityLevel, FHECKKSRNS,
)
from lssvm.solvers.qr_householder_cipher_row import (  # noqa: E402
    _BOOTSTRAP_LEVEL_BUDGET, _BOOTSTRAP_USABLE_DEPTH,
)

skd = SecretKeyDist.UNIFORM_TERNARY
depth = _BOOTSTRAP_USABLE_DEPTH + FHECKKSRNS.GetBootstrapDepth(_BOOTSTRAP_LEVEL_BUDGET, skd)

p = CCParamsCKKSRNS()
p.SetSecretKeyDist(skd)
p.SetSecurityLevel(SecurityLevel.HEStd_NotSet)   # force a small, RAM-friendly ring
p.SetRingDim(RING)
p.SetScalingModSize(50)
p.SetFirstModSize(60)
p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
p.SetMultiplicativeDepth(depth)
cc = GenCryptoContext(p)
for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE,
          PKESchemeFeature.ADVANCEDSHE, PKESchemeFeature.FHE):
    cc.Enable(f)
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalBootstrapSetup(_BOOTSTRAP_LEVEL_BUDGET, [0, 0], SLOTS)
cc.EvalBootstrapKeyGen(keys.secretKey, SLOTS)
print(f"[omp-smoke] N={cc.GetRingDimension()} depth={depth} threads={N}", flush=True)

ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.5] * SLOTS, 1, 0, None, SLOTS))

r0 = resource.getrusage(resource.RUSAGE_SELF); t0 = time.perf_counter()
for _ in range(NBOOT):
    ct = cc.EvalBootstrap(ct)
wall = time.perf_counter() - t0
r1 = resource.getrusage(resource.RUSAGE_SELF)
cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
avg_cores = cpu / wall if wall > 0 else 0.0

print(f"[omp-smoke] {NBOOT} bootstraps: wall={wall:.1f}s cpu={cpu:.1f}s "
      f"avg_cores={avg_cores:.2f} util={100*avg_cores/N:.0f}% of {N} threads", flush=True)

ok = N < 2 or avg_cores >= PASS_THRESHOLD
print(f"[omp-smoke] {'PASS — OpenMP engaged' if ok else 'FAIL — OpenFHE is running SERIAL (built without OpenMP)'}",
      flush=True)
sys.exit(0 if ok else 1)
