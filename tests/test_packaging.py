"""Tests for packaging-level guarantees: exports, version, optional deps."""

import importlib.util
import sys
from importlib.metadata import version

import numpy as np
import pytest

import pdf_estimator


def test_public_exports_are_importable():
    for name in pdf_estimator.__all__:
        assert hasattr(pdf_estimator, name), f"{name} missing from pdf_estimator"


def test_density_estimator_is_exported():
    """DensityEstimator is what the README and examples use."""
    assert "DensityEstimator" in pdf_estimator.__all__


def test_version_matches_installed_metadata():
    assert pdf_estimator.__version__ == version("pdf-estimator")


def test_importing_package_does_not_import_matplotlib():
    """matplotlib is an optional dependency, so it must not be imported eagerly."""
    assert not [m for m in sys.modules if m.startswith("matplotlib")], (
        "importing pdf_estimator pulled in matplotlib"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is not None,
    reason="matplotlib is installed, so the missing-dependency path cannot be exercised",
)
def test_plot_without_matplotlib_raises_helpful_error():
    est = pdf_estimator.DensityEstimator()
    est.fit(np.random.default_rng(0).standard_normal(200))
    with pytest.raises(ModuleNotFoundError, match=r"pdf-estimator\[plot\]"):
        est.plot(show=False)
