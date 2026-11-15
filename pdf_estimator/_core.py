"""Core PDF estimation: the C++ extension wrapper and its result type.

This module exists separately from ``__init__`` so that ``estimator.py`` can
import ``estimate_pdf`` without creating a circular import through the package
root.
"""

from dataclasses import dataclass

import numpy as np

try:
    from . import _pdf_estimator as _ext
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The pdf_estimator C extension (_pdf_estimator) could not be imported. "
        "If you installed from a wheel, this indicates a broken installation: "
        "try `pip install --force-reinstall --no-cache-dir pdf-estimator`. "
        "If you are working from a source checkout, build it first with "
        "`pip install -e .` (or `uv sync`). Note that importing from inside the "
        "source tree shadows the installed package, which also causes this error."
    ) from e


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
    pdf_size=None
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

    if n_features == 1:
        if pdf_size is not None:
            raise ValueError("pdf_size is only supported for multivariate data.")
        pdf_size = 0  # Unused
    elif pdf_size is None:
        pdf_size = int(
            (n_samples / 100) ** (1.0 / (n_features - 1))
        )
        pdf_size = max(5, min(100, pdf_size))
    else:
        pdf_size = int(pdf_size)
        if pdf_size < 5:
            raise ValueError("pdf_size must be at least 5")

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

