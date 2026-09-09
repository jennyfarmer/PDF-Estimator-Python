import evaluate_densities

from distributions.guassian_mixture_1d import GaussianMixture1D
from distributions.banana_2d import Banana2D
from distributions.sharp_banana import SharpBanana2D
from distributions.guassian_3d import CorrelatedGaussian3D
from distributions.gaussian_mixture_4d import GaussianMixture4D

DISTRIBUTIONS = {
    0: GaussianMixture1D,
    1: Banana2D,
    2: SharpBanana2D,
    3: CorrelatedGaussian3D,
    4: GaussianMixture4D,
}

# Currently not implemented
# KDE_BANDWIDTH = 0.20

def compare(
    distribution_id,
    n_samples,
    seed,
    write_file=False,
    evaluate_kde=False,
    plot_pdfs=False,
    verbose=False,
):
    distribution_class = DISTRIBUTIONS[distribution_id]
    distribution = distribution_class()

    filename = distribution.name + "_N" + str(n_samples) + "_seed" + str(seed) + ".csv"

    evaluate_densities.evaluate(distribution=distribution,
                                n_samples=n_samples,
                                seed=seed,
                                filename=filename,
                                write_file=write_file,
                                evaluate_kde=evaluate_kde,
                                plot_pdfs=plot_pdfs,
                                verbose=verbose,
                                )
