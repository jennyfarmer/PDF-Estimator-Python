import numpy as np
import pytest

from pdf_estimator import PDFResult, estimate_pdf


class TestSmoke:
    def test_import(self):
        from pdf_estimator import PDFResult, estimate_pdf  # noqa: F401

    def test_returns_pdfresult(self, gaussian_sample):
        assert isinstance(estimate_pdf(gaussian_sample), PDFResult)

    def test_multivariate_returns_pdfresult(self, gaussian_sample_3D):
        assert isinstance(estimate_pdf(gaussian_sample_3D), PDFResult)



class TestOutputContract:
    def test_all_fields_present(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        for field in (
            "y",
            "pdf",
            "pdfSizes",
            "marginals",
            "marginalsY",
            "marginalSizes",
        ):
            assert hasattr(r, field)

    def test_arrays_are_numpy(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        for arr in (r.y, r.pdf):
            assert isinstance(arr, np.ndarray)

    def test_y_pdf_same_length(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        assert r.y.shape == r.pdf.shape

    def test_no_nans_or_infs(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        for arr in (r.y, r.pdf):
            assert np.all(np.isfinite(arr))

    def test_multivariate_arrays_are_numpy(self, gaussian_sample_3D):
        r = estimate_pdf(gaussian_sample_3D)
        for arr in (r.y, r.pdf, r.marginalsY, r.marginals):
            assert isinstance(arr, np.ndarray)

    def test_multivariate_no_nans_or_infs(self, gaussian_sample_3D):
        r = estimate_pdf(gaussian_sample_3D)
        for arr in (r.y, r.pdf, r.marginalsY, r.marginals):
            assert np.all(np.isfinite(arr))

    def test_2d_gaussian_shape(self, gaussian_sample_2D):
        r = estimate_pdf(gaussian_sample_2D)
        assert r.pdfSizes.size == 2
        assert r.pdf.size == np.prod(r.pdfSizes)

    def test_multivariate_marginals_shape(self, gaussian_sample_3D):
        r = estimate_pdf(gaussian_sample_3D)
        assert r.marginalsY.shape == r.marginals.shape

    def test_multivariate_marginals_sizes(self, gaussian_sample_3D):
        r = estimate_pdf(gaussian_sample_3D)
        assert np.sum(r.marginalSizes) == r.marginals.size

    def test_multivariate_y_size(self, gaussian_sample_3D):
        r = estimate_pdf(gaussian_sample_3D)
        assert np.sum(r.pdfSizes) == r.y.size

    def test_multivariate_pdf_size(self, gaussian_sample_3D):
        r = estimate_pdf(gaussian_sample_3D)
        assert np.prod(r.pdfSizes) == r.pdf.size

    def test_rejects_transposed_input(self):
        rng = np.random.default_rng(42)
        bad_data = rng.standard_normal((3, 500))
        with pytest.raises(ValueError):
            estimate_pdf(bad_data)

class TestMathProperties:
    def test_pdf_non_negative(self, gaussian_sample):
        assert np.all(estimate_pdf(gaussian_sample).pdf >= 0)

    def test_pdf_integrates_to_one(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        assert abs(np.trapezoid(r.pdf, r.y) - 1.0) < 0.05

    def test_y_is_sorted(self, gaussian_sample):
        assert np.all(np.diff(estimate_pdf(gaussian_sample).y) > 0)

    def test_multivariate_pdf_non_negative(self, gaussian_sample_3D):
        assert np.all(estimate_pdf(gaussian_sample_3D).pdf >= 0)


class TestStatisticalAccuracy:
    def test_gaussian_mode_near_zero(self):
        RNG = np.random.default_rng(42)
        r = estimate_pdf(RNG.standard_normal(1000))
        assert abs(r.y[np.argmax(r.pdf)]) < 0.5



class TestParameters:
    def test_empty_data_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            estimate_pdf(np.array([]))


