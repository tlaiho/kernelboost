# Changelog

All notable changes to this project will be documented in this file.

## [0.3.1] - 2026-03-11

### Added
- `eval_set` parameter for `predict_variance()`. When provided, variance trees are fitted on evaluation set residuals instead of training residuals.
- inverse_transform method for `RankTransformer`.

### Changed
- Code formatting in `booster.py`.
- Code formatting in `feature_selection.py`.

## [0.3.0] - 2026-03-07

### Added
- Constant-leaf tree mode (`tree_type='constant'`) with vectorized MSE reduction splitting.
- MI-based relevance scoring in SmartSelector using histogram mutual information.
- C implementation of MI computation (`mi_bins.c`) with OpenMP parallelization.
- Temperature scheduling in SmartSelector via `temperature_max` parameter.
- `constant_tree_frequency` parameter in SmartSelector to control constant-leaf round frequency.

### Changed
- **Breaking**: `feature_list` renamed to `feature_tree_tuple`. Now accepts `(feature_indices, tree_type)` tuples.
- Feature selectors now return `(features, tree_type)` tuples from `get_features()`.
- `feature_importances_` now correctly accumulates all rounds instead of losing duplicates.

## [0.2.1] - 2026-03-01

### Changed
- Replaced built-in variance estimation with Fan & Yao (1998) double kernel method. `predict_variance()` now fits dedicated KernelTrees on squared residuals instead of reusing mean-optimized kernel weights. Default aggregation changed from `'mean'` to `'max'`.
- Small bug fixes. 

### Removed
- Removed `_predict_with_variance()` from KernelEstimator, KernelTree, CompiledTree, and Backend.
- Removed `cuda_predict_with_variance()` and `cpu_predict_with_variance()` backend functions.
- Removed `predict_with_variance` C function from `kernels.c`.

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
