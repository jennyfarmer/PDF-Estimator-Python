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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from ._core import PDFResult, estimate_pdf
from .estimator import DensityEstimator

try:
    __version__ = _version("pdf-estimator")
except PackageNotFoundError:  # pragma: no cover - source tree, not installed
    __version__ = "0+unknown"

__all__ = ["DensityEstimator", "PDFResult", "__version__", "estimate_pdf"]
