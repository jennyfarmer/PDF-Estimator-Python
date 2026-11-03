#from pdf_estimator import estimate_pdf
from . import estimate_pdf
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

class DensityEstimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._fitted = False
        self._joint_pdf_evaluated = False

    def fit(self, X):
        """
        Fit the density estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Sample data. A one-dimensional array is interpreted as
            samples from a single feature.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features)."
            )

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        self.n_samples, self.d = X.shape
        self.samples = X.copy()
        self.d = X.shape[1]

        self._result = None
        self.axes = None
        self.pdf_grid = None
        self.marginal_axes = None
        self.marginal_pdfs = None

        self._fitted = True
        self._joint_pdf_evaluated = False
        return self

    def evaluate_joint_pdf(self):

        if not self._fitted:
            raise RuntimeError(
                "Call fit(X) before evaluating the joint PDF."
            )

        if self._joint_pdf_evaluated:
            return self

        X_cpp = np.ascontiguousarray(self.samples)

        self._result = estimate_pdf(
            X_cpp,
            **self.kwargs,
        )

        self.n_eval_points = len(self._result.pdf)

        self.n = int(len(self._result.y) / self.d)

        # unpack grid
        self.axes = [
            self._result.y[i*self.n:(i+1)*self.n]
            for i in range(self.d)
        ]
        self.pdf_grid = self._result.pdf.reshape([self.n]*self.d)
        self.marginal_axes = []
        self.marginal_pdfs = []
        offset = 0
        for size in self._result.marginalSizes:
            self.marginal_axes.append(
                self._result.marginalsY[offset:offset + size]
            )
            self.marginal_pdfs.append(
                self._result.marginals[offset:offset + size]
            )
            offset += size

        self._joint_pdf_evaluated = True
        return self


    def pdf(self):
        return self.pdf_grid

    @staticmethod
    def _resolve_smooth_sigma(
            smooth_sigma,
            grid_shape,
            fraction=0.02,
    ):
        if smooth_sigma is not None:
            return float(smooth_sigma)

        return fraction * min(grid_shape)


    def plot(
            self,
            smooth_sigma=None,
            ax=None,
            label=None,
            show=True,
            norm=None,
            levels=10,
            cmap="viridis"
    ):
        """
        Plot the estimated density.

        Parameters
        ----------
        smooth_sigma : float, default=0.0
            Standard deviation used for optional Gaussian smoothing.
            Smoothing is applied in grid-index units.

        ax : matplotlib.axes.Axes, optional
            Existing axes on which to draw. If omitted, a new figure and
            axes are created.

        label : str, optional
            Label passed to Matplotlib.

        show : bool, default=True
            Whether to call plt.show().

        **plot_kwargs
            Additional keyword arguments passed to ax.plot() for 1D data
            or ax.contourf() for 2D data.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        if not self._fitted:
            raise RuntimeError(
                "Call fit(X) before plotting."
            )

        if not self._joint_pdf_evaluated:
            self.evaluate_joint_pdf()

        if ax is None:
            _, ax = plt.subplots()

        smooth_sigma = self._resolve_smooth_sigma(
            smooth_sigma,
            self.pdf_grid.shape,
        )

        if self.d == 1:
            x = np.asarray(self.axes[0])
            density = np.asarray(self.pdf_grid).squeeze()

            if smooth_sigma > 0:
                density = gaussian_filter1d(
                    density,
                    sigma=smooth_sigma,
                )

            ax.plot(
                x,
                density,
                label=label,
            )

            ax.set_xlabel("x")
            ax.set_ylabel("Probability density")


        elif self.d == 2:
            x0_grid, x1_grid = np.meshgrid(
                self.axes[0],
                self.axes[1],
                indexing="ij",
            )

            Zij = self.pdf_grid
            if smooth_sigma > 0:
                Z_smooth = gaussian_filter(
                    Zij,
                    sigma=smooth_sigma,
                    mode="nearest",
                )
                Zij = Z_smooth

            ax.contourf(
                x0_grid,
                x1_grid,
                Zij,
                norm = norm,
                levels = levels, cmap = cmap
            )

            ax.set_xlabel("x0")
            ax.set_ylabel("x1")

        else:
            raise NotImplementedError(
                "Use pairplot() for d >= 3."
            )

        if show:
            plt.show()

        return ax


    def pairplot(self,
                 smooth_sigma=None,
                 features=None,
                 show=True,
                 norm=None,
                 levels=10,
                 cmap="viridis"):
        if features is None:
            features = list(range(self.d))
        else:
            features = list(features)

        n_plotted = len(features)

        if n_plotted < 2:
            raise ValueError("features must contain at least two variables.")

        for f in features:
            if f < 0 or f >= self.d:
                raise IndexError(
                    f"Feature {f} outside valid range."
                )

        if len(set(features)) != n_plotted:
            raise ValueError("Duplicate feature indices.")

        requested_smooth_sigma = smooth_sigma

        pair_estimates = {}
        for row, feature_i in enumerate(features):
            for feature_j in features[row + 1:]:
                pair_samples = self.samples[
                    :,
                    [feature_i, feature_j],
                ]

                pair_estimates[(feature_i, feature_j)] = (
                    DensityEstimator()
                    .fit(pair_samples)
                )
                pair_estimates[(feature_i, feature_j)].evaluate_joint_pdf()

        # --- Plot ---
        fig, axs = plt.subplots(n_plotted, n_plotted, figsize=(3 * n_plotted, 3 * n_plotted))
        for i, feature_i in enumerate(features):
            for j, feature_j in enumerate(features):
                ax = axs[i, j]

                if feature_i == feature_j:
                    # --- 1D marginal ---
                    if feature_i == features[-1]:
                        x = pair_estimates[features[i - 1], feature_i].marginal_axes[1]
                        y = pair_estimates[features[i - 1], feature_i].marginal_pdfs[1]
                    else:
                        x = pair_estimates[feature_i, features[i + 1]].marginal_axes[0]
                        y = pair_estimates[feature_i, features[i + 1]].marginal_pdfs[0]

                    panel_sigma = self._resolve_smooth_sigma(
                        requested_smooth_sigma,
                        y.shape,
                    )

                    # --- Optional smoothing ---
                    if panel_sigma > 0:
                        y = gaussian_filter1d(y, sigma=panel_sigma)

                    ax.plot(x, y, color="black")
                # --- 2D marginal ---
                else:
                    if i < j:
                        pair = pair_estimates[feature_i, feature_j]
                        x = pair.axes[1]
                        y = pair.axes[0]
                        Zij = pair.pdf_grid
                    else:
                        pair = pair_estimates[feature_j, feature_i]
                        x = pair.axes[0]
                        y = pair.axes[1]
                        Zij = pair.pdf_grid.T

                    panel_sigma = self._resolve_smooth_sigma(
                        requested_smooth_sigma,
                        pair.pdf_grid.shape,
                    )
                    if panel_sigma > 0:
                        Z_smooth = gaussian_filter(
                            Zij,
                            sigma=panel_sigma,
                            mode="nearest",
                        )
                        Zij = Z_smooth


                    ax.contourf(x, y, Zij, norm=norm, levels=levels, cmap=cmap)

                # Labels
                if i == n_plotted - 1:
                    ax.set_xlabel(f"x{feature_j}")
                if j == 0:
                    ax.set_ylabel(f"x{feature_i}")

        plt.tight_layout()
        if show:
            plt.show()

        return fig, axs

    def pairplot_integrate(self, levels=10, cmap="viridis", smooth_sigma=None):

        d = self.d
        axes = self.axes
        Z_full = self.pdf_grid

        smooth_sigma = self._resolve_smooth_sigma(
            smooth_sigma,
            self.pdf_grid.shape,
        )

            # --- Integration helper ---
        def integrate_out(Z, axes_vals, keep_axes):
            result = Z
            axes_local = list(axes_vals)

            for ax in sorted(set(range(Z.ndim)) - set(keep_axes), reverse=True):
                result = np.trapezoid(result, x=axes_local[ax], axis=ax)
                axes_local.pop(ax)

            return result

        # --- Plot ---
        fig, axs = plt.subplots(d, d, figsize=(3 * d, 3 * d))

        for i in range(d):
            for j in range(d):
                ax = axs[i, j]

                if i == j:
                    # --- 1D marginal ---

                    has_marginals = (
                            hasattr(self, "marginal_axes") and
                            hasattr(self, "marginal_pdfs") and
                            len(self.marginal_axes) == d and
                            len(self.marginal_pdfs) == d
                    )

                    if has_marginals:
                        # --- Use precomputed marginals ---
                        x = self.marginal_axes[i]
                        y = self.marginal_pdfs[i]

                    else:
                        # --- Compute marginal from full PDF ---
                        Zi = integrate_out(Z_full, axes.copy(), [i])
                        x = axes[i]
                        y = Zi

                    # --- Optional smoothing ---
                    if smooth_sigma > 0:
                        y = gaussian_filter1d(y, sigma=smooth_sigma)
                    ax.plot(x, y, color="black")
                else:
                    # --- 2D marginal ---
                    Zij = integrate_out(Z_full, axes.copy(), [i, j])
                    if smooth_sigma > 0:
                        Zij = gaussian_filter(Zij, sigma=smooth_sigma)
                    if i > j:
                        Zij = Zij.T

                    x = axes[j]
                    y = axes[i]
                    X, Y = np.meshgrid(x, y)
                    ax.contourf(X, Y, Zij, levels=levels, cmap=cmap)
                    #                    ax.contour(X, Y, Zij, levels=levels, colors='k', linewidths=0.5)

                    # Labels
                    if i == d - 1:
                        ax.set_xlabel(f"x{j}")
                    if j == 0:
                       ax.set_ylabel(f"x{i}")

        plt.tight_layout()
        plt.show()

    def set_pdf(self, pdf_grid, axes):
        """
        Store a precomputed PDF on a supplied grid.

        This permits analytical densities and results from other estimators
        to use the DensityEstimator plotting methods.
        """
        axes = [
            np.asarray(axis, dtype=float)
            for axis in axes
        ]
        pdf_grid = np.asarray(pdf_grid, dtype=float)

        expected_shape = tuple(len(axis) for axis in axes)

        if pdf_grid.shape != expected_shape:
            raise ValueError(
                f"pdf_grid has shape {pdf_grid.shape}, "
                f"but axes require shape {expected_shape}."
            )

        self.pdf_grid = pdf_grid
        self.axes = axes
        self.d = len(axes)

        self.marginal_axes = []
        self.marginal_pdfs = []

        self._fitted = True
        return self


    def expand_axes(self):
        coordinate_grids = np.meshgrid(
            *self.axes,
            indexing="ij",
        )

        return np.column_stack(
            [
                grid.ravel()
                for grid in coordinate_grids
            ]
        )