"""CPU vs GPU training time benchmark.

Compares training time between CPU (C/OpenMP) and GPU (CuPy/CUDA) backends
on the California Housing dataset.

Usage:
    python benchmarks/cpu_gpu_benchmark.py
"""

import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from kernelboost import KernelBooster
from kernelboost.feature_selection import SmartSelector
from kernelboost.objectives import MSEObjective
from kernelboost.utilities import RankTransformer


if __name__ == "__main__":
    # --- Configuration ---
    n_samples = 10000
    n_val = 1500
    n_test = 1500
    n_train = n_samples - n_val - n_test

    # --- Load and subsample data ---
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    # Subsample to n_samples
    rng = np.random.default_rng()
    idx = rng.choice(len(X), size=n_samples, replace=False)
    X, y = X[idx], y[idx]
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # --- Split data ---
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=n_test)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=n_val)
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # --- Preprocess ---
    scaler = RankTransformer(pct=True)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    selector = SmartSelector(
        redundancy_penalty=0.4,
        relevance_alpha=0.7,
        recency_penalty=0.3,
        recency_decay=0.7,
        temperature=0.35,
        weight_decay=0.95,
        feature_groups=[[6, 7]],
    )

    common_params = dict(
        objective=MSEObjective(),
        feature_selector=selector,
        max_sample=4000,
        min_sample=750,
        n_estimators=250,
        subsample_share=0.8,
        lambda1=0.0002,
        learning_rate=0.8,
        min_features=1,
        max_features=5,
        overlap_epsilon=0.05,
        n_iter_no_change=30,
        verbose=0,
    )

    results = []

    print("\nRunning CPU benchmark...")
    kb_cpu = KernelBooster(**common_params, use_gpu=False)
    t0 = time.time()
    kb_cpu.fit(X_train, y_train, eval_set=(X_val, y_val))
    cpu_time = time.time() - t0
    results.append(("CPU (C/OpenMP)", cpu_time))

    print("Running GPU benchmark...")
    kb_gpu = KernelBooster(**common_params, use_gpu=True)
    t0 = time.time()
    kb_gpu.fit(X_train, y_train, eval_set=(X_val, y_val))
    gpu_time = time.time() - t0
    results.append(("GPU (CuPy/CUDA)", gpu_time))

    speedup = cpu_time / gpu_time
    print("\n" + "=" * 65)
    print(f"GPU vs CPU Training Time (California Housing, n={n_samples})")
    print("=" * 65)
    print(f"{'Backend':<50} {'Time':>10}")
    print("-" * 65)
    for name, t in results:
        print(f"{name:<50} {t:>9.1f}s")
    print("=" * 65)
    print(f"GPU speedup: {speedup:.1f}x")
