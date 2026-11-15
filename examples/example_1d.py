import numpy as np

from pdf_estimator import DensityEstimator

n_samples = 20_000
random_seed = 1234
rng = np.random.default_rng(random_seed)
components = rng.choice(
    2,
    size=n_samples,
    p=[0.65, 0.35],
)

samples = np.empty(n_samples)
samples[components == 0] = rng.normal(
    loc=-1.2,
    scale=0.45,
    size=np.count_nonzero(components == 0),
)
samples[components == 1] = rng.normal(
    loc=1.4,
    scale=0.75,
    size=np.count_nonzero(components == 1),
)

est = DensityEstimator().fit(samples)
est.plot()