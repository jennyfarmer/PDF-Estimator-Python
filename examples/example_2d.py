import numpy as np

from pdf_estimator import DensityEstimator

# Generate 2D sample data
n_samples = 20_000
random_seed = 1234
rng = np.random.default_rng(random_seed)
z = rng.standard_normal((n_samples, 2))

samples = np.empty_like(z)
samples[:, 0] = z[:, 0]
samples[:, 1] = (
    0.8 * (z[:, 0] ** 2 - 1.0)
    + 0.35 * z[:, 1]
)
# Fit estimator
est = DensityEstimator()
est.fit(samples)

# Plot
est.plot()