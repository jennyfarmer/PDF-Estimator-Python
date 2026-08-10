import numpy as np
from pdf_estimator import DensityEstimator

n_samples = 20_000
random_seed = 1234
rng = np.random.default_rng(random_seed)

samples = np.empty((n_samples, 4))

weights = np.array([0.6, 0.4])

means = np.array([
    [-1.2, -0.8,  0.7, -0.5],
    [ 1.3,  1.0, -0.9,  1.1],
])

covariances = np.array([
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


components = rng.choice(
    len(weights),
    size=n_samples,
    p=weights,
)

for component_index in range(len(weights)):
    mask = components == component_index
    samples[mask] = rng.multivariate_normal(
        mean=means[component_index],
        cov=covariances[component_index],
        size=np.count_nonzero(mask),
    )


est = DensityEstimator()
est.fit(samples)
est.pairplot()

est.pairplot(features = [0, 1, 3])