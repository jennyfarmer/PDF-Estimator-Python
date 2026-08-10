import numpy as np
from .base import Distribution

class BananaDistributionSharp(Distribution):
    def __init__(
        self,
        bend: float = 0.8,
        transverse_scale: float = 0.35,
    ):
        self.bend = bend
        self.transverse_scale = transverse_scale

    @property
    def dimension(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "Two-dimensional banana distribution"

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        z = rng.standard_normal((n_samples, 2))

        samples = np.empty_like(z)
        samples[:, 0] = z[:, 0]
        samples[:, 1] = (
            self.bend * (z[:, 0] ** 2 - 1.0)
            + self.transverse_scale * z[:, 1]
        )

        return samples

    def pdf(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "points must have shape (n_points, 2)"
            )

        x0 = points[:, 0]
        x1 = points[:, 1]

        z0 = x0
        z1 = (
            x1 - self.bend * (x0**2 - 1.0)
        ) / self.transverse_scale

        return (
            np.exp(-0.5 * (z0**2 + z1**2))
            / (
                2.0
                * np.pi
                * self.transverse_scale
            )
        )