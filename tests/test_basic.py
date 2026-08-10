import numpy as np
import pytest
from scipy import stats

from pdf_estimator import PDFResult, estimate_pdf

RNG = np.random.default_rng(42)


class TestSmoke:
    def test_import(self):
        from pdf_estimator import PDFResult, estimate_pdf  # noqa: F401

    def test_returns_pdfreuslt(self, gaussian_sample):
        assert isinstance(estimate_pdf(gaussian_sample), PDFResult)

    def test_succeeds_on_gaussian(self, gaussian_sample):
        assert estimate_pdf(gaussian_sample).failed == 0

    def test_succeeds_on_uniform(self, uniform_sample):
        assert estimate_pdf(uniform_sample).failed == 0

    def test_succeeds_on_bimodal(self, bimodal_sample):
        assert estimate_pdf(bimodal_sample).failed == 0


class TestOutputContract:
    def test_all_fields_present(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        for field in (
            "failed",
            "y",
            "pdf",
            "cdf",
            "sqr",
            "lagrange",
            "score",
            "confidence",
            "SURD",
        ):
            assert hasattr(r, field)

    def test_arrays_are_numpy(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        for arr in (r.y, r.pdf, r.cdf, r.sqr, r.lagrange, r.SURD):
            assert isinstance(arr, np.ndarray)

    def test_y_pdf_cdf_same_length(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        assert r.y.shape == r.pdf.shape == r.cdf.shape

    def test_no_nans_or_infs(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        for arr in (r.y, r.pdf, r.cdf):
            assert np.all(np.isfinite(arr))

    def test_score_and_confidence_are_scalar_floats(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        assert isinstance(r.score, float)
        assert isinstance(r.confidence, float)

    def test_confidence_in_valid_range(self, gaussian_sample):
        assert 0.0 <= estimate_pdf(gaussian_sample).confidence <= 100.0

    def test_at_least_one_lagrange_multiplier(self, gaussian_sample):
        assert len(estimate_pdf(gaussian_sample).lagrange) >= 1


class TestMathProperties:
    def test_pdf_non_negative(self, gaussian_sample):
        assert np.all(estimate_pdf(gaussian_sample).pdf >= 0)

    def test_pdf_integrates_to_one(self, gaussian_sample):
        r = estimate_pdf(gaussian_sample)
        assert abs(np.trapezoid(r.pdf, r.y) - 1.0) < 0.05

    def test_cdf_is_monotone_increasing(self, gaussian_sample):
        assert np.all(np.diff(estimate_pdf(gaussian_sample).cdf) >= -1e-9)

    def test_cdf_starts_near_zero(self, gaussian_sample):
        assert estimate_pdf(gaussian_sample).cdf[0] < 0.05

    def test_cdf_ends_near_one(self, gaussian_sample):
        assert estimate_pdf(gaussian_sample).cdf[-1] > 0.95

    def test_y_is_sorted(self, gaussian_sample):
        assert np.all(np.diff(estimate_pdf(gaussian_sample).y) > 0)


class TestStatisticalAccuracy:
    def test_gaussian_mode_near_zero(self):
        r = estimate_pdf(RNG.standard_normal(1000))
        assert abs(r.y[np.argmax(r.pdf)]) < 0.5

    def test_uniform_pdf_roughly_flat(self):
        r = estimate_pdf(RNG.uniform(0, 1, 1000), low_bound=0.0, high_bound=1.0)
        interior = r.pdf[(r.y > 0.1) & (r.y < 0.9)]
        assert interior.std() / interior.mean() < 0.3

    def test_ks_test_passes_gaussian(self):
        data = RNG.standard_normal(500)
        r = estimate_pdf(data)
        estimated_cdf = np.interp(np.sort(data), r.y, r.cdf)
        assert np.max(np.abs(estimated_cdf - stats.norm.cdf(np.sort(data)))) < 0.1


class TestParameters:
    def test_score_types(self, gaussian_sample):
        for score_type in ("QZ", "LL", "AD"):
            assert estimate_pdf(gaussian_sample, score_type=score_type).failed == 0

    def test_invalid_score_type_raises(self, gaussian_sample):
        with pytest.raises(ValueError, match="score_type"):
            estimate_pdf(gaussian_sample, score_type="INVALID")

    def test_explicit_bounds(self):
        r = estimate_pdf(RNG.standard_normal(500), low_bound=-5.0, high_bound=5.0)
        assert r.failed == 0
        assert float(r.y.min()) >= -5.0 - 1e-9
        assert float(r.y.max()) <= 5.0 + 1e-9

    def test_lagrange_limits_respected(self, gaussian_sample):
        assert len(estimate_pdf(gaussian_sample, lagrange_min=2, lagrange_max=5).lagrange) <= 5

    def test_empty_data_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            estimate_pdf(np.array([]))

    def test_list_input_accepted(self):
        assert estimate_pdf(list(RNG.standard_normal(200))).failed == 0

    def test_debug_mode_does_not_crash(self, gaussian_sample):
        assert estimate_pdf(gaussian_sample, debug=True).failed == 0

    def test_fuzz_mode_does_not_crash(self, gaussian_sample):
        assert estimate_pdf(gaussian_sample, fuzz=True).failed == 0
