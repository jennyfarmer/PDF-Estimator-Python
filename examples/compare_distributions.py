from importlib.metadata import distribution

import matplotlib.pyplot as plt
import numpy as np
from pyexpat import features
from sklearn.neighbors import KernelDensity
from pdf_estimator import DensityEstimator

from distributions.guassian_mixture_1d import GaussianMixture1D
from distributions.banana_2d import BananaDistribution
from distributions.sharp_banana import BananaDistributionSharp
from distributions.gaussian_mixture_4d import GaussianMixture4D
from distributions.guassian_3d import CorrelatedGaussian3D

#distribution = GaussianMixture1D()
#distribution = GaussianMixture4D()
#distribution = BananaDistribution()
#distribution = BananaDistributionSharp()
#distribution = CorrelatedGaussian3D()
distribution = GaussianMixture4D()

RANDOM_SEED = 2984
N_SAMPLES = 20_000
KDE_BANDWIDTH = 0.20

def integrated_absolute_error(
    estimate: np.ndarray,
    reference: np.ndarray,
    axes: list[np.ndarray],
) -> float:
    error = np.abs(estimate - reference)
    return integrate_grid(error, axes)


def integrated_squared_error(
    estimate: np.ndarray,
    reference: np.ndarray,
    axes: list[np.ndarray],
) -> float:
    error = (estimate - reference) ** 2
    return integrate_grid(error, axes)


def integrate_grid(
    density: np.ndarray,
    axes: list[np.ndarray],
) -> float:
    """
    Integrate a density defined on a rectangular, possibly nonuniform grid.
    """
    result = np.asarray(density, dtype=float)

    for axis_index in reversed(range(len(axes))):
        result = np.trapezoid(
            result,
            x=axes[axis_index],
            axis=axis_index,
        )

    return float(result)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    samples = distribution.sample(
        n_samples=N_SAMPLES,
        rng=rng,
    )

    print(distribution.name)
    print(distribution.dimension)
    print(samples.shape)
    sample_matrix = samples#.reshape(-1, 1)

    # ------------------------------------------------------------
    # 1. Run DensityEstimator first.
    #    Its adaptive grid becomes the common evaluation grid.
    # ------------------------------------------------------------
    adaptive_estimate = DensityEstimator()
    adaptive_estimate.fit(sample_matrix)
    adaptive_estimate.pairplot()#features=[0,1])

    adaptive_estimate.evaluate_joint_pdf()
    grid_coords = adaptive_estimate.expand_axes()

    # Analytical density
    pdf_true = distribution.pdf(
        grid_coords
    ).reshape(
        adaptive_estimate.pdf_grid.shape
    )

    true_density = DensityEstimator().set_pdf(
        pdf_grid=pdf_true,
        axes=adaptive_estimate.axes,
    )

    true_density.pairplot_integrate()



    # ------------------------------------------------------------
    # 4. Basic numerical checks.
    # ------------------------------------------------------------
    print(f"Number of samples:       {N_SAMPLES}")
    print(f"Adaptive grid points:    {len(grid_coords)}")

    print("Error relative to analytical density")
    print(
        "DensityEstimator IAE:",
        integrated_absolute_error(
            adaptive_estimate.pdf_grid,
            pdf_true,
            adaptive_estimate.axes,
        ),
    )

    print(
        "DensityEstimator ISE:",
        integrated_squared_error(
            adaptive_estimate.pdf_grid,
            pdf_true,
            adaptive_estimate.axes,
        ),
    )





    # KDE
    kde_model = KernelDensity(
        kernel="gaussian",
        bandwidth=KDE_BANDWIDTH,
    )

    kde_model.fit(samples)

    pdf_kde = np.exp(
        kde_model.score_samples(grid_coords)
    ).reshape(
        adaptive_estimate.pdf_grid.shape
    )

    kde_density = DensityEstimator().set_pdf(
        pdf_grid=pdf_kde,
        axes=adaptive_estimate.axes,
    )

    print(
        "KDE IAE:",
        integrated_absolute_error(
            pdf_kde,
            pdf_true,
            adaptive_estimate.axes,
        ),
    )


    print(
        "KDE ISE:",
        integrated_squared_error(
            pdf_kde,
            pdf_true,
            adaptive_estimate.axes,
        ),
    )


    # ------------------------------------------------------------
    # 5. Plot all three PDFs through DensityEstimator.plot().
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    true_density.plot(
        ax=ax,
        label="Analytical density",
        show=False,
        linewidth=2.5,
    )

    adaptive_estimate.plot(
        ax=ax,
        label="DensityEstimator",
        show=False,
        linewidth=2.0,
    )

    kde_density.plot(
        ax=ax,
        label="Gaussian KDE",
        show=False,
        linewidth=2.0,
        linestyle="--",
    )

    ax.set_title(
        "One-dimensional Gaussian mixture"
    )
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()