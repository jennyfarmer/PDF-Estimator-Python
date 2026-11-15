import math

import numpy as np

from .base import Distribution


class Banana(Distribution):
    def __init__(
        self,
        d = 2,
        bend: float = 0.8,
        transverse_scale: float = 0.35,
    ):
        self.d = d
        self.bend = bend
        self.transverse_scale = transverse_scale
        self.true_analytic_max = (2 * np.pi) ** (-d / 2) / transverse_scale ** (d - 1)

    @property
    def dimension(self) -> int:
        return self.d

    @property
    def name(self) -> str:
        return ("Banana" + str(self.d) + "D_bend" + str(self.bend) + "_scale" + str(self.transverse_scale))

    @property
    def description(self) -> str:
        return "Multi-dimensional banana distribution"

    @property
    def parameters(self):
        return {
            "bend": self.bend,
            "scale": self.transverse_scale,
            "true_analytic_max": self.true_analytic_max,
        }

    def sample(
            self,
            n_samples: int,
            rng: np.random.Generator,
    ) -> np.ndarray:

        z = rng.standard_normal((n_samples, self.d))
        samples = z.copy()

        for j in range(1, self.d):
            samples[:, j] = (
                    self.bend * (z[:, 0] ** 2 - 1.0)
                    + self.transverse_scale * z[:, j]
            )

        return samples



    def pdf(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)

        norm = 1 / (self.transverse_scale ** (self.d - 1) * (math.sqrt(2 * np.pi) ** self.d))

        x0 = points[:, 0]

        sum_square = 0
        for i in range(1, self.d):
            z = (
                points[:, i] - self.bend * (x0 ** 2 - 1.0)
                ) / self.transverse_scale

            sum_square += z ** 2

        exponent = np.exp(-0.5 * (x0 ** 2 + sum_square))

        return (
            norm * exponent
        )









