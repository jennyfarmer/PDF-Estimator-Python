import numpy as np
import pytest

RNG = np.random.default_rng(42)


@pytest.fixture
def gaussian_sample():
    return RNG.standard_normal(500)


@pytest.fixture
def uniform_sample():
    return RNG.uniform(0, 1, 500)


@pytest.fixture
def bimodal_sample():
    a = RNG.normal(-2, 0.5, 250)
    b = RNG.normal(2, 0.5, 250)
    return np.concatenate([a, b])
