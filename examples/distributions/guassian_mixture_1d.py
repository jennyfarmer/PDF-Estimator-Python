import numpy as np
from scipy.stats import norm
from .base import Distribution

class GaussianMixture1D(Distribution):
    def __init__(
        self,
        weight: float = 0.65,
        mean_1: float = -1.2,
        std_1: float = 0.45,
        mean_2: float = 1.4,
        std_2: float = 0.75,
    ):
        self.weight = weight
        self.mean_1 = mean_1
        self.std_1 = std_1
        self.mean_2 = mean_2
        self.std_2 = std_2

    @property
    def dimension(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "One-dimensional Gaussian mixture"

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        use_component_1 = (
            rng.random(n_samples) < self.weight
        )

        samples = np.empty(n_samples)

        samples[use_component_1] = rng.normal(
            loc=self.mean_1,
            scale=self.std_1,
            size=np.count_nonzero(use_component_1),
        )

        samples[~use_component_1] = rng.normal(
            loc=self.mean_2,
            scale=self.std_2,
            size=np.count_nonzero(~use_component_1),
        )

        return samples.reshape(-1, 1)

    def pdf(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)

        x = points.reshape(-1)

        return (
            self.weight
            * norm.pdf(
                x,
                loc=self.mean_1,
                scale=self.std_1,
            )
            + (1.0 - self.weight)
            * norm.pdf(
                x,
                loc=self.mean_2,
                scale=self.std_2,
            )
        )