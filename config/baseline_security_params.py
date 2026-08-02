"""Reported crypto parameters for the HEAAN-based baseline paper (R1 item 2,
analytical comparison).

Values are exactly as reported in the paper's Section IV-B ("Experimental
Settings") — nothing here is inferred beyond the depth derivation noted below.
The paper does not report M', h, P, q0, or sigma, and does not state a target
security level lambda; that gap is itself part of the R1 rebuttal framing.

CITATION: TODO — fill in once confirmed (title/authors/venue).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineHeaanParams:
    citation: str
    log_qL: int          # top-level ciphertext modulus bit-length (qL = p^L * q0)
    log_p: int            # scaling-factor exponent used at each rescale
    col_subciphertexts: int   # column-wise packing: n+1 sub-ciphertexts (n=100 samples)
    submatrix_blocks: int      # submatrix packing block count, s
    gd_iterations: int         # gradient-descent iterations (both SVM and logistic regression)

    @property
    def depth_naive(self) -> int:
        """log_qL / log_p, if qL is read as L rescale-levels with no separate q0 budget."""
        return self.log_qL // self.log_p

    def depth_with_first_modulus(self, first_mod_bits: int = 60) -> int:
        """(log_qL - first_mod_bits) / log_p, if qL = q0 (first_mod_bits) * p^L.

        first_mod_bits=60 matches this project's own FirstModSize convention
        (see lssvm/solvers/qr_householder_cipher_row.py) — used only as a
        plausible assumption since the paper doesn't state q0 separately.
        """
        return (self.log_qL - first_mod_bits) // self.log_p


# As reported, Section IV-B: log(qL)=1200, log(p)=25, column-wise packing = 101
# sub-ciphertexts (n+1 for their 100-sample datasets), submatrix packing s=16,
# 10 GD iterations for both SVM and logistic regression.
BASELINE = BaselineHeaanParams(
    citation="TODO: paper title/authors/venue",
    log_qL=1200,
    log_p=25,
    col_subciphertexts=101,
    submatrix_blocks=16,
    gd_iterations=10,
)
