# kernelboost

**Gradient boosting with kernel-based local constant estimators**

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-array%20backend-blue)
![C](https://img.shields.io/badge/C-language-blue)
![GPU](https://img.shields.io/badge/GPU-CUDA%20C%2FCuPy-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.4.0-blue)

kernelboost is a gradient boosting algorithm that uses Nadaraya-Watson (local constant) kernel estimators as base learners instead of decision trees. It has:

- Support for regression, classification and quantile regression tasks.
- sklearn style API (`fit`, `predict`).
- CPU (via C) and GPU (via CuPy/CUDA) backends.

## Installation

```bash
# Basic installation
pip install kernelboost

# With GPU support (requires CUDA)
pip install cupy-cuda12x  # for CUDA 12; use cupy-cuda11x for CUDA 11
```

> **Dependencies**: NumPy. CuPy optional for GPU acceleration.

## Quick Start

```python
from kernelboost import KernelBooster, MulticlassBooster
from kernelboost.objectives import MSEObjective, EntropyObjective

# Regression (use_gpu=True enables the CuPy/CUDA backend)
booster = KernelBooster(objective=MSEObjective(), use_gpu=True).fit(X_train, y_train)
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

kernelboost uses gradient boosting with kernel-based local constant estimators instead of decision trees. Each boosting round fits a KernelTree that partitions the data into regions, then applies Nadaraya-Watson kernel regression at each leaf to predict pseudo-residuals. Unlike tree-based boosters where splits implicitly select features, kernelboost selects features explicitly at the boosting stage before tree construction.

### What it delivers

With [suitable preprocessing](#data-preprocessing), kernelboost can match popular gradient boosters like XGBoost and LightGBM on prediction accuracy while outperforming traditional kernel methods (KernelRidge, SVR, Gaussian Processes). Training time is much slower than tree based methods but much faster than Gaussian processes. See [Benchmarks](#benchmarks) for detailed comparisons.

### Architecture

There are three main components to kernelboost: KernelBooster class that does the boosting, KernelTree class that does the splitting and KernelEstimator class that implements the local constant estimation. As kernel methods are computationally expensive, the guiding principle has been computational efficiency.  

After calling fit, KernelBooster starts a training loop which is mostly identical to the algorithm described in Friedman (2001). The main difference is that KernelTree does not choose features through its splits but is instead given them by the booster class. Default feature selection is probabilistic, based on joint mutual information (JMISelector), with kernel sizes increasing in terms of number of features. Probabilistic feature selection naturally creates randomness to training results, which can be mitigated with a lower learning rate and more boosting iterations. Similarly to Friedman (2001), KernelBooster can fit several different objective functions, which are passed in as an Objective class. 

KernelTree splits numerical data by density and categorical data by MSE. It can also fit pure decision trees with mean values at leaves. The idea here is that the kernel bandwidth should largely depend on how dense the data is. For numerical data, KernelTree splits until number of observations is below the 'max_sample' parameter. Besides finding regions which would be well served by the same bandwidth, this has the benefit of speeding up computation significantly in calculating the kernel matrices for the kernel estimator. For example, with ten regions we go from computing a (n, n) matrix to computing ten (n/10, n/10) matrices with n²/10 operations instead of n² (assuming equal splits). This saves a nice 90% of compute.

The actual estimation is handled by KernelEstimator. It optimizes a scalar precision (inverse bandwidth) for the local constant estimator using leave-one-out cross validation and random search between given bounds. It has both Gaussian and (isotropic) Laplace kernels with default being the Laplace kernel. KernelEstimator also has uncertainty quantification methods for quantile and conditional variance prediction (Fan & Yao 1998).

### Notable features 

Beyond the core boosting algorithm, a few features worth highlighting:

#### Smart Feature Selection

The default selector (JMISelector) scores candidate features by joint mutual information (JMI; Yang & Moody 1999) between feature pairs and the current pseudo-residuals, which measures relevance, redundancy and synergy with a single metric (Brown et al. 2012). Selection is probabilistic, and blends the JMI score with per-feature gain history and a recency penalty. Purely random selection is available through RandomSelector.

```python
from kernelboost.feature_selection import JMISelector

selector = JMISelector(
    relevance_alpha=0.9,
    temperature=0.2,
)

booster = KernelBooster(
    objective=MSEObjective(),
    feature_selector=selector,
)
```

#### Early Stopping

Training stops automatically if evaluation loss doesn't improve for `n_iter_no_change` consecutive rounds.

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

KernelBooster has conditional quantile prediction (Hall, Wolff & Yao 1999) with a prediction interval wrapper, and conditional variance prediction (Fan & Yao 1998), all based on kernel estimation. Quantile and interval prediction require held-out data (eval_set); variance prediction also works on training data alone. Still work in progress.

```python
# Conditional quantiles and intervals (eval_set required)
quantiles = booster.predict_quantiles(X, taus=(0.05, 0.5, 0.95), eval_set=(X_val, y_val))
lower, upper = booster.predict_intervals(X, alpha=0.1, eval_set=(X_val, y_val))

# Conditional variance estimates
variance = booster.predict_variance(X)
```

Both methods fit dedicated trees on model residuals. Quantile trees are fit on eval_set residuals: fitting them on training residuals collapses to unconditional quantiles, as boosting leaves little conditional mean signal in the residuals. Variance trees fit on training data are corrected with leave-one-out residuals by default (overfit_correction argument). The correction removes own-observation effect, but estimation bias at the booster stage is still present, which is why the variance estimate tends to overestimate the true (aleatoric) variance. See [benchmarks](#uncertainty-quantification-california-housing) for a comparison with Gaussian Processes.

#### Data Preprocessing

Scaling data is a good idea for kernel estimation methods. The package includes a simple RankTransformer that often works well (used for all benchmarks). 

```python
from kernelboost.utilities import RankTransformer

scaler = RankTransformer(pct=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Like other kernel methods, kernelboost works best with continuous, smooth features. For datasets with many categorical features, tree-based methods are often better suited—they handle splits on categories naturally.

## API Reference

| Class | Purpose |
|-------|---------|
| `KernelBooster` | Main booster for regression/binary classification |
| `MulticlassBooster` | One-vs-rest multiclass wrapper |
| `KernelTree` | Data partitioning with kernel estimators at leaves |
| `KernelEstimator` | Nadaraya-Watson local constant estimator |
| `MSEObjective` | Mean squared error (regression) |
| `EntropyObjective` | Cross-entropy (binary classification) |
| `QuantileObjective` | Pinball loss (quantile regression) |
| `JMISelector` | Joint-MI feature selection (default) |
| `RandomSelector` | Random feature selection |
| `RhoOptimizer` | Post-hoc step size optimization |
| `RankTransformer` | Rank-based feature scaling |

Besides `fit`, `predict`, `predict_proba` and `score`, `KernelBooster` exposes `staged_predict` and `staged_predict_proba`, which yield predictions after every boosting round. Unlike `predict`, these sweep all fitted rounds by default, so they can be used to inspect the learning curve past the early-stopping cutoff.

```python
for round_idx, preds in enumerate(booster.staged_predict(X_test)):
    print(round_idx, mean_squared_error(y_test, preds))
```

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
| `precision_method` | 'pilot-cv' | Bandwidth optimization: 'pilot-cv' (pilot bounds, then LOO-CV), 'pilot-aicc' (pilot bounds, then AICc), 'search' (LOO-CV over `bounds`), or 'silverman' (rule of thumb, no CV) |
| `pilot_factor` | 3.0 | Width of the pilot search range: `[p/factor, p*factor]` around the pilot precision (pilot methods only) |
| `search_rounds` | 10 | Precision optimization iterations |
| `bounds` | (0.1, 35.0) | Precision search bounds |

## Benchmarks

Results have inherent randomness due to feature selection and subsampling. Scripts available in `benchmarks/`.

### Regression (California Housing)

*kernelboost on GPU, n_train=16000, n_estimators=200.*

```text
=================================================================
Model                       MSE        MAE         R²       Time
-----------------------------------------------------------------
kernelboost              0.1816     0.2794     0.8631      16.5s
sklearn HGBR             0.2024     0.2941     0.8474       0.3s
XGBoost                  0.2080     0.2962     0.8432       0.2s
LightGBM                 0.1972     0.2894     0.8513       0.1s
=================================================================
```

### Binary Classification (Breast Cancer)

*kernelboost on GPU, n_train=455 (80/20 split), n_estimators=100.*

```text
=================================================================
Model                  Accuracy    AUC-ROC         F1       Time
-----------------------------------------------------------------
kernelboost              0.9825     0.9971     0.9861       1.6s
sklearn HGBC             0.9649     0.9948     0.9722       0.1s
XGBoost                  0.9561     0.9941     0.9650       0.0s
LightGBM                 0.9737     0.9921     0.9790       0.0s
=================================================================
```

### Comparison with Kernel Methods (California Housing)
```text
=================================================================
Kernel Methods Benchmark (n_train=10000)
=================================================================
Model                       MSE        MAE         R²       Time
-----------------------------------------------------------------
kernelboost              0.1996     0.2948     0.8508      11.5s
KernelRidge              0.4277     0.4840     0.6803       1.6s
SVR                      0.3103     0.3766     0.7681       3.5s
GP (n=5000)              0.3358     0.4080     0.7490      31.9s
=================================================================
```

### Uncertainty Quantification (California Housing)

Prediction intervals and conditional variance estimates compared to Gaussian Process (sklearn) regression:
```text
=================================================================
Uncertainty Quantification (90% intervals, alpha=0.1)
=================================================================
Model                  Coverage    Width   Var Corr   Var Ratio
-----------------------------------------------------------------
kernelboost              90.3%    1.233      0.241       0.812
GP (n=5000)              90.8%    1.865      0.196       1.042
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
CPU (C/OpenMP)                                          56.7s
GPU (CuPy/CUDA)                                          8.6s
=================================================================
GPU speedup: 6.6x
```

All benchmarks run on Ubuntu 22.04 with Ryzen 7700 and RTX 3090.

## References

- Brown, G., Pocock, A., Zhao, M.-J., & Luján, M. (2012). Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection. Journal of Machine Learning Research, 13, 27-66.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
- Fan, J., & Yao, Q. (1998). Efficient estimation of conditional variance functions in stochastic regression. Biometrika, 85(3), 645–660.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232.
- Hall, P., Wolff, R. C. L., & Yao, Q. (1999). Methods for Estimating a Conditional Distribution Function. Journal of the American Statistical Association, 94(445), 154-163.
- Hansen, B. E. (2004). Nonparametric Conditional Density Estimation. Working paper, University of Wisconsin.
- Nadaraya, E. A. (1964). On Estimating Regression. Theory of Probability and Its Applications, 9(1), 141-142.
- Watson, G. S. (1964). Smooth Regression Analysis. Sankhyā: The Indian Journal of Statistics, Series A, 26(4), 359-372.
- Yang, H. H., & Moody, J. (1999). Data Visualization and Feature Selection: New Algorithms for Nongaussian Data. Advances in Neural Information Processing Systems 12 (NIPS), 687-693.

## About

kernelboost is a hobby project exploring alternatives to tree-based gradient boosting. Pre-compiled binaries included for Linux and Windows. Contributions and feedback welcome.

## License

MIT License
