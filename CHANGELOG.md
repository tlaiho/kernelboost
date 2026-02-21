# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-02-21

### Changed
- **Breaking**: Renamed API parameters for sklearn compatibility:
  `rounds` to `n_estimators`, `max_tree_depth` to `max_depth`, `early_stopping_rounds` to `n_iter_no_change`
- Changed default `max_sample` and `min_sample` parameters to more sensible values 

### Added
- Pilot estimation method for precision search bounds (`precision_method='pilot-cv'`). Pilot estimated
bounds controlled by `pilot_factor` parameter passed on from KernelBooster and KernelTree to KernelEstimator.

## [0.1.0] - 2026-02-10

### Added
- Initial public release
- KernelBooster for 1d targets with support for MSE, entropy, and quantile objectives
- MulticlassBooster for multiclass classification
- GPU acceleration via CuPy
- Nadaraya-Watson kernel regression with LOO-CV bandwidth optimization
- Feature selection (random and smart selectors)
- Early stopping with validation loss monitoring
- Uncertainty quantification via prediction intervals and conditional variance prediction
