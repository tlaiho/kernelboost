# KernelBoost

**Gradient boosting with kernel-based local constant estimators**

![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![NumPy](https://img.shields.io/badge/NumPy-array%20backend-blue)
![C](https://img.shields.io/badge/C-language-blue)
![GPU](https://img.shields.io/badge/GPU-CUDA%20C%2FCuPy-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.2.0-blue)

KernelBoost is a gradient boosting algorithm that uses Nadaraya-Watson (local constant) kernel estimators as base learners instead of decision trees. It has:

- Support for regression, classification and quantile regression tasks.
- sklearn style API (`fit`, `predict`).
- CPU (via C) and GPU (via CuPy/CUDA) backends.

## Installation

```bash
# Basic installation
pip install kernelboost

# With GPU support (requires CUDA)
pip install cupy-cuda12x  # for CUDA 12
```

> **Dependencies**: NumPy only. CuPy optional for GPU acceleration.

## Quick Start

```python
from kernelboost import KernelBooster, MulticlassBooster
from kernelboost.objectives import MSEObjective, EntropyObjective

# Regression
booster = KernelBooster(objective=MSEObjective()).fit(X_train, y_train)
predictions = booster.predict(X_test)

# Binary classification
booster = KernelBooster(objective=EntropyObjective()).fit(X_train, y_train)
logits = booster.predict(X_test)
probabilities = booster.predict_proba(X_test)

# Multiclass classification (fits one booster per class)
booster = MulticlassBooster().fit(X_train, y_train)
class_labels = booster.predict(X_test)
```

### How it works

KernelBoost uses gradient boosting with kernel-based local constant estimators instead of decision trees. Each boosting round fits a KernelTree that partitions the data into regions, then applies Nadaraya-Watson kernel regression at each leaf to predict pseudo-residuals. Unlike tree-based boosters where splits implicitly select features, KernelBoost selects features explicitly at the boosting stage before tree construction.

### What it delivers

With [suitable preprocessing](#data-preprocessing), KernelBoost can match popular gradient boosters like XGBoost and LightGBM on prediction accuracy while outperforming traditional kernel methods (KernelRidge, SVR, Gaussian Processes). Training time is comparable to other kernel methods. See [Benchmarks](#benchmarks) for detailed comparisons.

### Architecture

There are three main components to KernelBoost: KernelBooster class that does the boosting, KernelTree class that does the splitting and KernelEstimator class that implements the local constant estimation. As kernel methods are computationally expensive, the guiding principle has been computational efficiency.  

After calling fit, KernelBooster starts a training loop which is mostly identical to the algorithm described in Friedman (2001). The main difference is that KernelTree does not choose features through its splits but is instead given them by the booster class. Default feature selection is random with increasing kernel sizes in terms of number of features. Random feature selection naturally creates randomness to training results, which can be mitigated with a lower learning rate and more boosting iterations. Similarly to Friedman (2001), KernelBooster can fit several different objective functions, which are passed in as an Objective class. 

KernelTree splits numerical data by density and categorical data by MSE. The idea here is that the kernel bandwidth should largely depend on how dense the data is. For numerical data, KernelTree splits until number of observations is below the 'max_sample' parameter. Besides finding regions which would be well served by the same bandwidth, this has the benefit of speeding up computation significantly in calculating the kernel matrices for the kernel estimator. For example, with ten splits we go from computing a (n, n) matrix to computing ten (n/10, n/10) matrices with n²/10 operations instead of n² (assuming equal splits). This saves a whopping 90% of compute.

The actual estimation is handled by KernelEstimator. It optimizes a scalar precision (inverse bandwidth) for the local constant estimator using leave-one-out cross validation and random search between given bounds. It has both Gaussian and (isotropic) Laplace kernels with default being the Laplace kernel. KernelEstimator also has uncertainty quantification methods for quantile and conditional variance prediction, but these are at this moment still experimental as they use a "naive" single kernel method whose precision is optimized for mean prediction.

### Notable features 

Beyond the core boosting algorithm, KernelBoost includes a few features worth highlighting:

#### Smart Feature Selection

While the default feature selection is random (RandomSelector), the package includes an mRMR style probabilistic algorithm (SmartSelector) based on correlations between features and pseudo-residuals and performance in previous boosting rounds.

```python
from kernelboost.feature_selection import SmartSelector

selector = SmartSelector(
    redundancy_penalty=0.4,
    relevance_alpha=0.7,
    recency_penalty=0.3,
)

booster = KernelBooster(
    objective=MSEObjective(),
    feature_selector=selector,
)
```

#### Early Stopping

Training stops automatically if evaluation loss doesn't improve for consecutive rounds (controlled by n_iter_no_change parameter).

```python
booster.fit(X_train, y_train, eval_set=(X_val, y_val))
```

#### RhoOptimizer

RhoOptimizer performs post-hoc optimization of step sizes, often improving predictions at minimal additional cost. It can also back out optimal regularization parameters (L1 penalty and learning rate) — useful when unsure what level of regularization to use.

```python
from kernelboost.rho_optimizer import RhoOptimizer

opt = RhoOptimizer(booster, lambda_reg=1.0)
opt.fit(X_val, y_val)
opt.update_booster()

# Back out optimal hyperparameters
lambda1, learning_rate = opt.find_hyperparameters()
```

#### Uncertainty Quantification (Experimental)

KernelBooster has both prediction intervals and conditional variance prediction based on kernel estimation. These come for "free" on top of training and require no extra data. Still work in progress.

```python
# Prediction intervals (90% by default)
lower, upper = booster.predict_intervals(X, alpha=0.1)

# Conditional variance estimates
variance = booster.predict_variance(X)
```

Both interval coverage and conditional variance have a tendency to be underestimated, but this depends on the data and how well boosting has converged. No special tuning required: settings that optimize MSE have also given reasonable uncertainty estimates in testing. See [benchmarks](#uncertainty-quantification-california-housing) for a comparison with Gaussian Processes.

#### Data Preprocessing

Scaling data is a good idea for kernel estimation methods. The package includes a simple RankTransformer that often works well (used for all benchmarks). 

```python
from kernelboost.utilities import RankTransformer

scaler = RankTransformer(pct=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Like other kernel methods, KernelBoost works best with continuous, smooth features. For datasets with many categorical features, tree-based methods are often better suited—they handle splits on categories naturally.

## API Reference

| Class | Purpose |
|-------|---------|
| `KernelBooster` | Main booster for regression/binary classification |
| `MulticlassBooster` | One-vs-rest multiclass wrapper |
| `MSEObjective` | Mean squared error (regression) |
| `EntropyObjective` | Cross-entropy (binary classification) |
| `QuantileObjective` | Pinball loss (quantile regression) |
| `SmartSelector` | mRMR-style feature selection |
| `RandomSelector` | Random feature selection |
| `RhoOptimizer` | Post-hoc step size optimization |
| `RankTransformer` | Rank-based feature scaling |

## Main Parameters

### KernelBooster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objective` | Required | Loss function: `MSEObjective()`, `EntropyObjective()`, `QuantileObjective()` |
| `n_estimators` | auto | Boosting iterations (auto = n_features * 15) |
| `max_features` | auto | Max features per estimator (auto = min(10, n_features)) |
| `min_features` | 1 | Min features per estimator |
| `subsample_share` | 0.5 | Training sample share per round |
| `learning_rate` | 0.5 | Step size shrinkage factor |
| `lambda1` | 0.0 | L1 regularization |
| `n_iter_no_change` | 20 | Rounds without improvement before early stopping |
| `verbose` | 0 | Verbosity level |
| `use_gpu` | False | Enable GPU acceleration |

### KernelTree (exposed via KernelBooster)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 3 | Maximum tree depth |
| `max_sample` | 5000 | Maximum samples per leaf (triggers splits) |
| `min_sample` | 500 | Minimum samples for kernel fitting |
| `overlap_epsilon` | 0.05 | Fraction of feature range to expand data beyond split boundaries |

### KernelEstimator (exposed via KernelBooster / KernelTree)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_type` | 'laplace' | Kernel function: 'laplace' or 'gaussian' |
| `precision_method` | 'pilot-cv' | Bandwidth optimization: 'pilot-cv', 'search', or 'silverman' |
| `search_rounds` | 20 | Precision optimization iterations |
| `bounds` | (0.1, 35.0) | Precision search bounds |

## Benchmarks

Results have inherent randomness due to feature selection and subsampling. Scripts available in `benchmarks/`.

### Regression (California Housing)
```text
=================================================================
Model                       MSE        MAE         R²       Time
-----------------------------------------------------------------
KernelBoost              0.2053     0.2985     0.8452      11.0s
sklearn HGBR             0.2247     0.3146     0.8306       0.1s
XGBoost                  0.2155     0.3050     0.8376       0.1s
LightGBM                 0.2097     0.3047     0.8419       0.1s
=================================================================
```

### Binary Classification (Breast Cancer)
```text
=================================================================
Model                  Accuracy    AUC-ROC         F1       Time
-----------------------------------------------------------------
KernelBoost              0.9825     0.9984     0.9861       1.6s
sklearn HGBC             0.9649     0.9944     0.9722       0.1s
XGBoost                  0.9561     0.9938     0.9650       0.0s
LightGBM                 0.9649     0.9925     0.9722       0.0s
=================================================================
```

### Comparison with Kernel Methods (California Housing)
```text
=================================================================
Kernel Methods Benchmark (n_train=10000)
=================================================================
Model                       MSE        MAE         R²       Time
-----------------------------------------------------------------
KernelBoost              0.2091     0.3054     0.8430       6.5s
KernelRidge              0.4233     0.4835     0.6822       1.7s
SVR                      0.3136     0.3780     0.7646       3.5s
GP (n=5000)              0.3297     0.4061     0.7524      67.7s
=================================================================
```

### Uncertainty Quantification (California Housing)

Prediction intervals and conditional variance estimates compared to Gaussian Process (sklearn) regression:
```text
=================================================================
Uncertainty Quantification (90% intervals, alpha=0.1)
=================================================================
Model                  Coverage    Width    Var Corr   Var Ratio
-----------------------------------------------------------------
KernelBoost              88.1%    1.235      0.206       1.621
GP (n=5000)              90.9%    1.863      0.157       1.026
=================================================================
```

Var Corr is the correlation between predicted variance and squared errors.
Var Ratio is the ratio between mean of squared_errors and predicted variance. 

### CPU/GPU training time comparison (California Housing)

```text
=================================================================
GPU vs CPU Training Time (California Housing, n=10000)
=================================================================
Backend                                                  Time
-----------------------------------------------------------------
CPU (C/OpenMP)                                          38.6s
GPU (CuPy/CUDA)                                          4.6s
=================================================================
GPU speedup: 8.3x
```

All benchmarks run on Ubuntu 22.04 with Ryzen 7700 and RTX 3090.

## References

- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
- Fan, J., & Yao, Q. (1998). Efficient estimation of conditional variance functions in stochastic regression. Biometrika, 85(3), 645–660.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232.
- Hansen, B. E. (2004). Nonparametric Conditional Density Estimation. Working paper, University of Wisconsin.
- Nadaraya, E. A. (1964). On Estimating Regression. Theory of Probability and Its Applications, 9(1), 141-142.
- Watson, G. S. (1964). Smooth Regression Analysis. Sankhyā: The Indian Journal of Statistics, Series A, 26(4), 359-372.

## About

KernelBoost is a hobby project exploring alternatives to tree-based gradient boosting. Currently v0.2.0. Pre-compiled binaries included for Linux and Windows. Contributions and feedback welcome.

## License

MIT License
