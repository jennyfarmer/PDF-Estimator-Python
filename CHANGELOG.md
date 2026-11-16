# Changelog

All notable changes to this project are documented here.

This project follows [Semantic Versioning](https://semver.org/). While the
major version is `0`, the public API may still change between minor releases.

## 0.1.0 — unreleased

First release published to PyPI.

### Added

- Binary wheels for Linux (manylinux and musllinux, x86_64 and aarch64),
  macOS (arm64 and x86_64) and Windows (AMD64), on CPython 3.11 through 3.15.
- `pdf_estimator.__version__`, read from installed package metadata.
- An optional `plot` extra for matplotlib-based plotting.
- `DensityEstimator` is now part of the public API (`__all__`).

### Changed

- `requires-python` is now `>=3.11` (was `>=3.12`).
- matplotlib is no longer a required dependency. It is imported lazily by the
  plotting methods, so `import pdf_estimator` no longer pulls it in. Install
  `pdf-estimator[plot]` to use `plot`, `pairplot` or `pairplot_integrate`.
- `numpy>=2.0` is now required explicitly; the library uses `np.trapezoid`.
- `LICENSE.txt` now contains the full GPL-3.0 text rather than a short notice.

### Fixed

- `DensityEstimator` was importable but missing from `__all__`.
- A circular import between `__init__.py` and `estimator.py` has been removed
  by moving the core wrapper into `pdf_estimator/_core.py`.
- The error raised when the C extension is missing no longer gives
  `uv`-specific advice to users who installed from a wheel.
- Removed `tests/test_properties.py`, which referenced `PDFResult.failed`,
  `.cdf` and `.confidence` — attributes that do not exist on the current API,
  so every test in the file raised `AttributeError`.
