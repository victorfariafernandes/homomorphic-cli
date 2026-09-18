"""analytic_primal_bounds: data-independent Chebyshev domains for the primal QR.

Guards that the returned floors sit below the empirical minima observed in the
exhaustive plaintext sweep (so a future accidental tightening is caught), that
domains never cross zero, and that the diagonal-inverse interval is
sign-preserving negative (fixed sign +1 makes R's diagonal negative).
"""

import math

import pytest

from lssvm.solvers.utils import analytic_primal_bounds

# Empirical minima observed across iris + breast_cancer, k in {2..40}, 6 seeds
# (see the plan doc). The analytic floors must stay strictly below these so the
# Chebyshev domains contain every real value.
EMPIRICAL_MIN_NORM_SQ = 0.237
EMPIRICAL_MIN_VTV = 0.273
EMPIRICAL_MIN_ABS_DIAG = 0.487
# Observed maxima (domains must contain these).
EMPIRICAL_MAX_NORM_SQ = 228.0
EMPIRICAL_MAX_VTV = 553.7
EMPIRICAL_MAX_ABS_DIAG = 15.1


@pytest.mark.parametrize(
    "n,gamma",
    [(6, 4.2), (40, 1.1), (227, 1.1), (24, 1.27), (60, 3.9)],
)
def test_floors_below_proven_lambda_min(n, gamma):
    # Proven property (Cauchy interlacing): R[k][k]^2 >= lambda_min(H) PER CONFIG.
    # So the floor (lambda_min/margin) must be compared against this config's own
    # lambda_min, not the cross-config global minimum from a different gamma.
    lam_min = 1.0 / (2.0 * gamma + 2.0 * math.sqrt(gamma / n) + 1.0 / n)
    b = analytic_primal_bounds(n, d=15, gamma=gamma)
    assert b["sqrt"][0] < lam_min
    assert b["inv_vtv"][0] < lam_min
    assert abs(b["inv_diag"][1]) < math.sqrt(lam_min)  # hi is closest to zero


def test_worst_case_config_floor_below_global_minima():
    # The global minima (0.237 norm_sq / 0.273 vtv / 0.487 |diag|) came from a
    # high-gamma config; its analytic floor must sit below them.
    b = analytic_primal_bounds(n=60, d=15, gamma=4.2)
    assert b["sqrt"][0] < EMPIRICAL_MIN_NORM_SQ
    assert b["inv_vtv"][0] < EMPIRICAL_MIN_VTV
    assert abs(b["inv_diag"][1]) < EMPIRICAL_MIN_ABS_DIAG


def test_upper_bounds_contain_empirical_maxima():
    # The largest single client (breast_cancer k=2, n~227) drives the max norm_sq.
    b = analytic_primal_bounds(n=227, d=30, gamma=1.1)
    assert b["sqrt"][1] >= EMPIRICAL_MAX_NORM_SQ
    assert b["inv_vtv"][1] >= EMPIRICAL_MAX_VTV
    assert abs(b["inv_diag"][0]) >= EMPIRICAL_MAX_ABS_DIAG  # lo is most negative


@pytest.mark.parametrize("n,gamma", [(6, 4.2), (40, 1.1), (227, 1.1)])
def test_domains_never_cross_zero(n, gamma):
    b = analytic_primal_bounds(n, d=15, gamma=gamma)
    # sqrt and inv_vtv are strictly positive intervals
    assert 0.0 < b["sqrt"][0] < b["sqrt"][1]
    assert 0.0 < b["inv_vtv"][0] < b["inv_vtv"][1]
    # inv_diag is strictly negative and ordered lo < hi < 0
    assert b["inv_diag"][0] < b["inv_diag"][1] < 0.0


def test_lambda_min_floor_formula():
    # norm_sq lower bound = lambda_min / margin, with the proven lambda_min bound.
    n, gamma, margin = 40, 1.1, 1.1
    b = analytic_primal_bounds(n, d=4, gamma=gamma, margin=margin)
    lam_min = 1.0 / (2.0 * gamma + 2.0 * math.sqrt(gamma / n) + 1.0 / n)
    assert b["sqrt"][0] == pytest.approx(lam_min / margin)
    # diag interval endpoints are the sqrt of the norm_sq interval (negated).
    assert b["inv_diag"][1] == pytest.approx(-math.sqrt(lam_min / margin))


def test_sign_is_fixed_positive():
    b = analytic_primal_bounds(n=10, d=4, gamma=1.1)
    assert b["sign"] == 1.0


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        analytic_primal_bounds(n=10, d=4, gamma=0.0)
    with pytest.raises(ValueError):
        analytic_primal_bounds(n=0, d=4, gamma=1.1)
