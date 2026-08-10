# pdf-estimator

`pdf-estimator` is a Python interface to a C++ implementation of a
nonparametric probability-density estimator based on maximum entropy
and conditional probabilities.

The package estimates adaptive univariate and multivariate probability
densities directly from sample data and provides built-in visualization
for one-dimensional, two-dimensional, and higher-dimensional datasets.

## Features

- Nonparametric joint probability-density estimation
- Adaptive evaluation grids
- Automatic high-resolution univariate marginals
- Filled-contour plots for two-dimensional estimates
- Directly estimated pairplots for multivariate data
- Selection of variable subsets for high-dimensional pairplots
- Optional display smoothing and Matplotlib customization
- C++ backend with a Python-oriented interface

## Installation

```bash
pip install pdf_estimator
```
## API Overview

| Method or attribute        | Purpose                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `fit(X)`                   | Validate and store samples with shape `(n_samples, n_features)` |
| `evaluate_joint_pdf()`     | Compute and cache the full joint density                        |
| `plot()`                   | Plot a one- or two-dimensional density                          |
| `pairplot()`               | Plot directly estimated univariate and bivariate densities      |
| `pairplot(features=[...])` | Plot only a selected subset of variables                        |
| `pdf`                      | Joint probability-density array                                 |
| `axes`                     | Adaptive coordinate axes for the joint density                  |
| `marginal_pdfs`            | High-resolution univariate marginal densities                   |
| `marginal_axes`            | Coordinate axes for the marginal densities                      |




## Examples
### 1-Dimensional Gaussian mixture distribution
```python
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

```

### 2-Dimensional banana distribution

```python

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
```


### 4-Dimensional Gaussian mixture distribution

```python

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

```

## Citation

Farmer, J. & Jacobs, D. (2019). High throughput nonparametric probability density estimation. _PLOS ONE_. https://doi.org/10.1371/journal.pone.0196937


## Development

```bash
uv sync --extra dev   # install dev dependencies
uv run pytest tests/test_basic.py   # fast tests
uv run pytest                       # all tests (slow — includes property-based)
uv run ruff check .                 # lint
uv run ruff format .                # format
```

## License

This project is licensed under the GNU General Public License v3.0 (or later).
You are free to redistribute and modify this software under the terms of the GPL.
This program is distributed **without any warranty**, including without implied warranties of merchantability or fitness for a particular purpose.

See the LICENSE file for full details: <http://www.gnu.org/licenses/>
