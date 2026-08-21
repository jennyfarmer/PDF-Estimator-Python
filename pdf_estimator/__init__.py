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

from dataclasses import dataclass

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

    y : ndarray
        Axes grid values for the estimated probability density function, flattened to a single dimension.
    pdf : ndarray
        Estimated joint probability density function, flattened to a single dimension.
    pdfSizes : ndarray
        Number of grid points along each feature axis.
        Provides the shape of the flattened joint PDF and is useful for consistency
        checks and reconstruction of the multidimensional grid.
    marginalsY : ndarray
        Axes grid values for each of the one-dimensional marginal densities, flattened to a single dimension.
    marginals : ndarray
        Estimated one-dimensional marginal probability density function for each dimension,
        flattened to a single dimension.
    marginalSizes : ndarray
        Number of grid points along each marginal axis.
        Provides the shape of the flattened marginals and is useful for consistency
        checks and reconstruction of the multidimensional grid.
    """

    y: np.ndarray
    pdf: np.ndarray
    pdfSizes: np.ndarray
    marginals: np.ndarray
    marginalsY: np.ndarray
    marginalSizes: np.ndarray

    def __repr__(self) -> str:
        return (
            f"PDFResult("
            f"n_features={len(self.pdfSizes)}, "
            f"shape={tuple(self.pdfSizes)}"
            f")"
        )



def estimate_pdf(
    data,
) -> PDFResult:
    """Estimate the probability density function of a multivariate data sample.

    Parameters
    ----------
    data : array-like
        Input sample of i.i.d. continuous random variables.

    Returns
    -------
    PDFResult
    """

    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("data must be non-empty.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("data must be a 1D or 2D array.")

    n_samples, n_features = arr.shape
    if n_features > n_samples:
        raise ValueError(
            "Expected data with shape (n_samples, n_features). "
            f"Received {arr.shape}. The number of samples must "
            "exceed the number of features; check whether the "
            "input array needs to be transposed."
        )

    arr_cpp = arr.T

    pdf_size = int(
        0.75 * n_samples ** (1.0 / n_features)
    )

    if pdf_size > 100:
        pdf_size = 100
    if pdf_size < 32:
        pdf_size = 32

    raw = _ext.estimate_pdf(
        arr_cpp,
        pdf_size,
    )

    return PDFResult(
        y=np.asarray(raw["y"]),
        pdf=np.asarray(raw["pdf"]),
        pdfSizes=np.asarray(raw["pdfSizes"]),
        marginalsY=np.asarray(raw["marginalsY"]),
        marginals=np.asarray(raw["marginals"]),
        marginalSizes=np.asarray(raw["marginalSizes"]),
    )


from .estimator import DensityEstimator
__all__ = ["estimate_pdf", "PDFResult"]
