import numpy as np
import pytest


@pytest.fixture
def gaussian_sample():
    RNG = np.random.default_rng(42)
    return RNG.standard_normal(500)

@pytest.fixture
def gaussian_sample_2D():
    RNG = np.random.default_rng(42)
    return RNG.standard_normal((300, 2))

@pytest.fixture
def gaussian_sample_3D():
    RNG = np.random.default_rng(42)
    return RNG.standard_normal((200, 3))

@pytest.fixture
def uniform_sample():
    RNG = np.random.default_rng(42)
    return RNG.uniform(0, 1, 500)


@pytest.fixture
def bimodal_sample():
    RNG = np.random.default_rng(42)
    a = RNG.normal(-2, 0.5, 250)
    b = RNG.normal(2, 0.5, 250)
    return np.concatenate([a, b])
