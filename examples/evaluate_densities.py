
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import KernelDensity

from pdf_estimator import DensityEstimator


@dataclass
class ErrorResults:
    mse: float
    mae: float
    max_error: float



def write_results(filename, results, append=False):
    """
    Write summary test results to a CSV file.

    Parameters
    ----------
    filename : str or Path
        Output filename.
    results : dict
        Dictionary of estimator results.
    """
    filename = Path(filename)
    mode = "a" if append else "w"

    with filename.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not append:
            writer.writeheader()
        writer.writerow(results)

def calculate_errors(true, estimated):
    error = estimated - true
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    results = ErrorResults(
        mse=mse,
        mae=mae,
        max_error=max_error
    )
    return results


def evaluate(
    distribution,
    n_samples,
    seed,
    filename,
    write_file=False,
    evaluate_kde=False,
    plot_pdfs=False,
    verbose=False,
):
    rng = np.random.default_rng(seed)

    samples = distribution.sample(
        n_samples=n_samples,
        rng=rng,
    )

    if verbose:
        print(distribution.name)
        print(samples.shape)


    #PDFe
    pdfe = DensityEstimator()
    pdfe.fit(samples)
    pdfe_start = time.perf_counter()
    pdfe.evaluate_joint_pdf()
    pdfe_elapsed = time.perf_counter() - pdfe_start
    grid_coords = pdfe.expand_axes()
    if plot_pdfs:
        if distribution.dimension <= 2:
            pdfe.plot()
        else:
            pdfe.pairplot()


    # True density
    true_model = distribution.pdf(
        grid_coords
    ).reshape(
        pdfe.pdf_grid.shape
    )

    pdf_true = DensityEstimator().set_pdf(
        pdf_grid=true_model,
        axes=pdfe.axes,
    )
    if plot_pdfs:
        if distribution.dimension <= 2:
            pdf_true._joint_pdf_evaluated = True
            pdf_true.plot()
        else:
            pdf_true.pairplot_integrate()


    pdfe_results = calculate_errors(pdf_true.pdf_grid, pdfe.pdf_grid)
    if verbose:
        print("PDFe")
        print("mse:", pdfe_results.mse)
        print("mae:", pdfe_results.mae)
        print("max_error:", pdfe_results.max_error)
        print("time:", pdfe_elapsed)
        print()

    if write_file:
        results = {
            "distribution": distribution.name,
            "sample_size": n_samples,
            "seed": seed,
            "estimator": "PDFE",
            "eval_time": pdfe_elapsed,
            "mse": pdfe_results.mse,
            "mae": pdfe_results.mae,
            "max_error": pdfe_results.max_error,
        }

        write_results(filename, results, append=False)

    #KDE
    if evaluate_kde:
        kde_model = KernelDensity(
            kernel="gaussian",
# Bandwidth currently not implemented, default used
#        bandwidth=KDE_BANDWIDTH,
        )

        kde_model.fit(samples)

        start = time.perf_counter()
        scores = np.exp(
            kde_model.score_samples(grid_coords)
        ).reshape(
            pdfe.pdf_grid.shape
        )
        kde_elapsed = time.perf_counter() - start

        pdf_kde = DensityEstimator().set_pdf(
            pdf_grid=scores,
            axes=pdfe.axes,
        )

        kde_results = calculate_errors(pdf_true.pdf_grid, pdf_kde.pdf_grid)

        if verbose:
            print("KDE")
            print("mse:", kde_results.mse)
            print("mae:", kde_results.mae)
            print("max_error:", kde_results.max_error)
            print("time:", kde_elapsed)
            print()

        if write_file:
            results = {
                "distribution": distribution.name,
                "sample_size": n_samples,
                "seed": seed,
                "estimator": "KDE",
                "eval_time": kde_elapsed,
                "mse": kde_results.mse,
                "mae": kde_results.mae,
                "max_error": kde_results.max_error,
            }

            write_results(filename, results, append=True)
