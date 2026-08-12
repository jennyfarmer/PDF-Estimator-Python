"""
pdf_estimator
=============
Python wrapper for the PDF-Estimator C++ library by Farmer & Jacobs (2019).

Quickstart
----------
>>> import numpy as np
>>> from pdf_estimator import estimate_pdf
>>> result = estimate_pdf(np.random.randn(500))
>>> print(result)
"""

import math
from dataclasses import dataclass
#from .estimator import DensityEstimator

import numpy as np

try:
    from . import _pdf_estimator as _ext
except ImportError as e:
    raise ImportError("The _pdf_estimator C extension is not built. Run:  uv sync") from e


@dataclass(frozen=True)
class PDFResult:
    """All outputs from an EstimatePDF run.

    Attributes
    ----------
    failed : int
        Non-zero if no valid solution was found.
    y : ndarray
        x-axis values.
    pdf : ndarray
        Estimated probability density function.
    cdf : ndarray
        Estimated cumulative distribution function.
    sqr : ndarray
        Scaled Quantile Residual (diagnostic).
    lagrange : ndarray
        Lagrange multiplier coefficients.
    score : float
        Final score value (threshold).
    confidence : float
        SURD confidence percentage achieved.
    SURD : ndarray
        Sample Uniform Random Data.
    """

    y: np.ndarray
    pdf: np.ndarray
    pdfSizes: np.ndarray
    cdf: np.ndarray
    sqr: np.ndarray
    lagrange: np.ndarray
    confidence: float
    SURD: np.ndarray
    marginals: np.ndarray
    marginalsY: np.ndarray
    marginalSizes: np.ndarray
    failed: int

    def __repr__(self) -> str:
        status = "OK" if not self.failed else f"FAILED (code {self.failed})"
        return (
            f"PDFResult(status={status}, "
            f"n_lagrange={len(self.lagrange)}, "
            f"confidence={self.confidence:.1f}%, "
        )


def estimate_pdf(
    data,
    *,
    surd_target: int = 40,
    surd_min: int = 5,
    surd_max: int = 100,
    lagrange_min: int = 1,
    lagrange_max: int = 200,
    low_bound: float | None = None,
    high_bound: float | None = None,
    debug: bool = False,
) -> PDFResult:
    """Estimate the probability density function of a 1-D sample.

    Parameters
    ----------
    data : array-like
        Input sample of i.i.d. continuous random variables.
    surd_target : int
        Target SURD confidence threshold (default 40).
    surd_min : int
        Minimum acceptable SURD (default 5).
    surd_max : int
        Maximum SURD (default 100).
    lagrange_min : int
        Minimum Lagrange multipliers (default 1).
    lagrange_max : int
        Maximum Lagrange multipliers (default 200).
    low_bound : float | None
        Lower bound for estimation domain. Auto-calculated if None.
    high_bound : float | None
        Upper bound for estimation domain. Auto-calculated if None.
    integration_points : int
        Number of integration points; 0 = auto (default).
    debug : bool
        Print verbose C++ progress to stdout (default False).


    Returns
    -------
    PDFResult
    """

    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("data must be non-empty.")

    arr_cpp = np.ascontiguousarray(arr.reshape(1, -1))

    raw = _ext.estimate_pdf(
        arr_cpp,
        surd_target=surd_target,
        surd_min=surd_min,
        surd_max=surd_max,
        lagrange_min=lagrange_min,
        lagrange_max=lagrange_max,
        low_bound=low_bound if low_bound is not None else math.nan,
        high_bound=high_bound if high_bound is not None else math.nan,
        debug=debug,
    )

    return PDFResult(
        failed=int(raw["failed"]),
        y=np.asarray(raw["y"]),
        pdf=np.asarray(raw["pdf"]),
        pdfSizes=np.asarray(raw["pdfSizes"]),
        cdf=np.asarray(raw["cdf"]),
        sqr=np.asarray(raw["sqr"]),
        lagrange=np.asarray(raw["lagrange"]),
        confidence=float(raw["confidence"]),
        SURD=np.asarray(raw["SURD"]),
        marginalsY=np.asarray(raw["marginalsY"]),
        marginals=np.asarray(raw["marginals"]),
        marginalSizes=np.asarray(raw["marginalSizes"]),
    )


from .estimator import DensityEstimator
__all__ = ["estimate_pdf", "PDFResult"]
