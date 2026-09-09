import numpy as np
from scipy.stats import multivariate_normal

from .base import Distribution


class CorrelatedGaussian3D(Distribution):
    def __init__(self, correlation: float = 0.8):
        if not -1.0 < correlation < 1.0:
            raise ValueError(
                "correlation must be strictly between -1 and 1."
            )

        self.mean = np.zeros(3)

        self.covariance = np.array([
            [1.0, correlation, 0.0],
            [correlation, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    @property
    def dimension(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "CorrelatedGaussian3D"

    @property
    def description(self) -> str:
        return "Three-dimensional correlated Gaussian"

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.multivariate_normal(
            mean=self.mean,
            cov=self.covariance,
            size=n_samples,
        )

    def pdf(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)

        if (
            points.ndim != 2
            or points.shape[1] != self.dimension
        ):
            raise ValueError(
                "points must have shape (n_points, 3)."
            )

        return multivariate_normal.pdf(
            points,
            mean=self.mean,
            cov=self.covariance,
        )