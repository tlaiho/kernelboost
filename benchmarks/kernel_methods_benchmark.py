"""Kernel methods benchmark on California Housing dataset.

Compares KernelBoost against sklearn kernel-based methods:
KernelRidge, SVR, and GaussianProcessRegressor.

Usage:
    python benchmarks/kernel_methods_benchmark.py [--no-gpu] [--n-train N]
"""

import argparse
import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from kernelboost import KernelBooster
from kernelboost.feature_selection import SmartSelector
from kernelboost.objectives import MSEObjective
from kernelboost.utilities import RankTransformer


def evaluate(model, X, y):
    """Return (mse, mae, r2, predict_time) for a fitted model."""
    t0 = time.time()
    y_pred = np.asarray(model.predict(X)).ravel()
    pred_time = time.time() - t0
    return mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred), r2_score(y, y_pred), pred_time


def evaluate_intervals(y_true, lower, upper):
    """Evaluate prediction interval quality."""
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    avg_width = np.mean(upper - lower)
    return coverage, avg_width


def evaluate_variance(y_true, y_pred, variance):
    """Evaluate variance calibration via correlation and ratio with squared errors."""
    squared_errors = (y_true - y_pred) ** 2
    correlation = np.corrcoef(variance.ravel(), squared_errors.ravel())[0, 1]
    ratio = np.mean(squared_errors) / np.mean(variance) 
    return correlation, ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel methods benchmark")
    parser.add_argument("--no-gpu", action="store_true", help="CPU only for KernelBoost")
    parser.add_argument("--n-train", type=int, default=10000, help="Training set size (default: 10000)")
    parser.add_argument("--n-train-gp", type=int, default=5000, help="Training set size for GP (default: 5000)")
    args = parser.parse_args()

    # Configuration 
    use_gpu = not args.no_gpu
    n_train = args.n_train
    n_train_gp = args.n_train_gp
    n_val = 1000

    # Load and split data 
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    n_test = X.shape[0] - n_train - n_val
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=n_test)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=n_val)
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Preprocess 
    scaler = RankTransformer(pct=True)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # GP subset 
    gp_idx = np.random.RandomState().choice(len(X_train), size=n_train_gp, replace=False)
    X_train_gp = X_train[gp_idx]
    y_train_gp = y_train[gp_idx]

    results = []

    print("\nTraining KernelBoost...")
    selector = SmartSelector(
        redundancy_penalty=0.4,
        relevance_alpha=0.7,
        recency_penalty=0.3,
        recency_decay=0.7,
        temperature=0.35,
        weight_decay=0.95,
        feature_groups=[[6, 7]],
    )

    kb = KernelBooster(
        objective=MSEObjective(),
        feature_selector=selector,
        max_sample=4000,
        min_sample=750,
        rounds=250,
        subsample_share=0.8,
        lambda1=0.0002,
        learning_rate=0.8,
        min_features=1,
        max_features=5,
        overlap_epsilon=0.05,
        early_stopping_rounds=25,
        bounds=(0.10, 18.0),
        use_gpu=use_gpu,
        verbose=0,
    )

    t0 = time.time()
    kb.fit(X_train, y_train, eval_set=(X_val, y_val))
    kb_time = time.time() - t0
    mse, mae, r2, _ = evaluate(kb, X_test, y_test)
    results.append(("KernelBoost", mse, mae, r2, kb_time))

    print("Training KernelRidge...")
    kr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)

    t0 = time.time()
    kr.fit(X_train, y_train)
    kr_time = time.time() - t0
    mse, mae, r2, _ = evaluate(kr, X_test, y_test)
    results.append(("KernelRidge", mse, mae, r2, kr_time))

    print("Training SVR...")
    svr = SVR(kernel='rbf', C=10.0, gamma='scale', cache_size=1000)

    t0 = time.time()
    svr.fit(X_train, y_train)
    svr_time = time.time() - t0
    mse, mae, r2, _ = evaluate(svr, X_test, y_test)
    results.append(("SVR", mse, mae, r2, svr_time))

    print(f"Training GaussianProcessRegressor (n={n_train_gp})...")
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)

    t0 = time.time()
    gp.fit(X_train_gp, y_train_gp)
    gp_time = time.time() - t0
    mse, mae, r2, _ = evaluate(gp, X_test, y_test)
    results.append((f"GP (n={n_train_gp})", mse, mae, r2, gp_time))

    # Summary
    print("\n" + "=" * 65)
    print(f"Kernel Methods Benchmark (n_train={n_train})")
    print("=" * 65)
    print(f"{'Model':<20} {'MSE':>10} {'MAE':>10} {'R²':>10} {'Time':>10}")
    print("-" * 65)
    for name, mse, mae, r2, t in results:
        print(f"{name:<20} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f} {t:>9.1f}s")
    print("=" * 65)

    # Uncertainty Quantification Comparison 
    print("\n" + "=" * 65)
    print("Uncertainty Quantification (90% intervals, alpha=0.1)")
    print("=" * 65)

    alpha = 0.1
    z = 1.645  

    # KernelBoost uncertainty
    kb_pred = kb.predict(X_test).ravel()
    kb_lower, kb_upper = kb.predict_intervals(X_test, alpha=alpha)
    kb_variance = kb.predict_variance(X_test)
    kb_coverage, kb_width = evaluate_intervals(y_test, kb_lower, kb_upper)
    kb_var_corr, kb_var_ratio = evaluate_variance(y_test, kb_pred, kb_variance)

    # GP uncertainty
    gp_pred, gp_std = gp.predict(X_test, return_std=True)
    gp_lower = gp_pred - z * gp_std
    gp_upper = gp_pred + z * gp_std
    gp_variance = gp_std ** 2
    gp_coverage, gp_width = evaluate_intervals(y_test, gp_lower, gp_upper)
    gp_var_corr, gp_var_ratio = evaluate_variance(y_test, gp_pred, gp_variance)

    print(f"{'Model':<20} {'Coverage':>10} {'Width':>8} {'Var Corr':>10} {'Var Ratio':>11}")
    print("-" * 65)
    print(f"{'KernelBoost':<20} {kb_coverage:>9.1%} {kb_width:>8.3f} {kb_var_corr:>10.3f} {kb_var_ratio:>11.3f}")
    print(f"{f'GP (n={n_train_gp})':<20} {gp_coverage:>9.1%} {gp_width:>8.3f} {gp_var_corr:>10.3f} {gp_var_ratio:>11.3f}")
    print("=" * 65)
