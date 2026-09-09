import numpy as np
from scipy.stats import multivariate_normal

from .base import Distribution


class GaussianMixture4D(Distribution):
    def __init__(self):
        self.weights = np.array([0.6, 0.4])

        self.means = np.array([
            [-1.2, -0.8,  0.7, -0.5],
            [ 1.3,  1.0, -0.9,  1.1],
        ])

        self.covariances = np.array([
            [
                [1.00,  0.75,  0.10, -0.20],
                [0.75,  1.00,  0.25,  0.00],
                [0.10,  0.25,  0.70,  0.40],
                [-0.20, 0.00,  0.40,  0.85],
            ],
            [
                [0.70, -0.45,  0.00,  0.20],
                [-0.45, 0.90, -0.35,  0.10],
                [0.00, -0.35,  1.00, -0.60],
                [0.20,  0.10, -0.60,  0.95],
            ],
        ])

    @property
    def dimension(self) -> int:
        return 4

    @property
    def name(self) -> str:
        return "GaussianMixture4D"

    @property
    def name(self) -> str:
        return "Four-dimensional correlated Gaussian mixture"

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        components = rng.choice(
            len(self.weights),
            size=n_samples,
            p=self.weights,
        )

        samples = np.empty(
            (n_samples, self.dimension)
        )

        for component_index in range(len(self.weights)):
            mask = components == component_index

            samples[mask] = rng.multivariate_normal(
                mean=self.means[component_index],
                cov=self.covariances[component_index],
                size=np.count_nonzero(mask),
            )

        return samples

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
                "points must have shape (n_points, 4)"
            )

        density = np.zeros(points.shape[0])

        for weight, mean, covariance in zip(
            self.weights,
            self.means,
            self.covariances,
        ):
            density += (
                weight
                * multivariate_normal.pdf(
                    points,
                    mean=mean,
                    cov=covariance,
                )
            )

        return density