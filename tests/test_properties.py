import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pdf_estimator import estimate_pdf

pytestmark = pytest.mark.slow

valid_sample = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=50, max_value=300),
    elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
).filter(lambda x: np.unique(x).size > 5)

slow_settings = settings(
    max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
)


@given(data=valid_sample)
@slow_settings
def test_property_pdf_non_negative(data):
    r = estimate_pdf(data)
    if not r.failed:
        assert np.all(r.pdf >= -1e-10)


@given(data=valid_sample)
@slow_settings
def test_property_pdf_integrates_to_one(data):
    r = estimate_pdf(data)
    if not r.failed:
        assert abs(np.trapezoid(r.pdf, r.y) - 1.0) < 0.1


@given(data=valid_sample)
@slow_settings
def test_property_cdf_monotone(data):
    r = estimate_pdf(data)
    if not r.failed:
        assert np.all(np.diff(r.cdf) >= -1e-9)


@given(data=valid_sample)
@slow_settings
def test_property_output_always_finite(data):
    r = estimate_pdf(data)
    if not r.failed:
        for arr in (r.y, r.pdf, r.cdf):
            assert np.all(np.isfinite(arr))


@given(data=valid_sample)
@slow_settings
def test_property_confidence_bounded(data):
    r = estimate_pdf(data)
    if not r.failed:
        assert 0.0 <= r.confidence <= 100.0
