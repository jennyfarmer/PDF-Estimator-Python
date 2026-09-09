import numpy as np
from .base import Distribution

class Banana2D(Distribution):
    def __init__(
        self,
        bend: float = 0.45,
    ):
        self.bend = bend

    @property
    def dimension(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "Banana2D"

    @property
    def description(self) -> str:
        return "Two-dimensional banana distribution"

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        z = rng.standard_normal(
            size=(n_samples, 2)
        )

        samples = np.empty_like(z)

        samples[:, 0] = z[:, 0]
        samples[:, 1] = (
            z[:, 1]
            + self.bend * (z[:, 0] ** 2 - 1.0)
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

        x = points[:, 0]
        y = points[:, 1]

        z_1 = x
        z_2 = (
            y
            - self.bend * (x**2 - 1.0)
        )

        return (
            np.exp(
                -0.5 * (z_1**2 + z_2**2)
            )
            / (2.0 * np.pi)
        )